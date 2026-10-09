"""The stage runner: the box-side entry point for one rlvr_lean stage. Spec §1 (topology and job-runner
integration).

Runs on the interpreter that starts it with the STANDARD LIBRARY ONLY, and installs nothing into it. It:
  1. installs a pinned `uv` into the persistent store (checksum-verified),
  2. builds the project's environments there from the repository's `uv.lock` (`gpu`, `quantize`),
  3. runs each step of the requested stage as its OWN process in the right environment, so GPU memory is
     released between steps, with a GPU guard in between (used memory back to the task's baseline),
  4. writes a TensorBoard heartbeat to `<out>/tb/` every minute (a job runner that stops a silent task, ours
     after 90 minutes, sees that the stage is alive), and samples card memory and the step's process-tree
     memory so peaks can be reported,
  5. writes `<out>/<stage>.json` (the completion artifact) and per-step JSON beside it.

Every step is idempotent: it skips work whose outputs already exist in the store, so a rerun resumes.

    python -m rlvr_lean.runner.entry --stage m2 --kimina-api-key KEY --out OUT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

from rlvr_lean.runner.heartbeat import ScalarEventWriter

UV_VERSION = "0.12.15"
UV_ARCHIVE = "uv-x86_64-unknown-linux-gnu.tar.gz"
REPO_ROOT = Path(__file__).resolve().parents[3]          # the task's repo directory (cwd for every child)
PROJECT_DIR = REPO_ROOT                                  # the `uv` project: pyproject.toml and uv.lock
PACKAGE_DIR = REPO_ROOT / "src" / "rlvr_lean"            # the package: its data fixtures
STORE = Path(os.environ.get("RLVR_LEAN_STORE", "/root/rlvr_lean_store"))

# (environment, step) in order; ("guard", None) waits for the GPU to be idle again.
STAGES: dict[str, list[tuple[str, str | None]]] = {
    "m2": [
        ("sync", "gpu"), ("sync", "quantize"),
        ("gpu", "download"),
        ("guard", None), ("quantize", "quantize_fp8"),
        ("guard", None), ("gpu", "quantize_nf4"),
        ("gpu", "fix_tokenizers"),
        ("guard", None), ("gpu", "sample_smoke"),
        ("guard", None), ("gpu", "qlora_step"),
        ("guard", None), ("gpu", "adapter_check"),
        ("guard", None),
    ],
    # Phase A (spec §2–§8). The first six steps are Milestone 2's and are no-ops once their outputs exist;
    # they stay so a fresh box can run this stage on its own. The profile picks the sizes.
    "phase_a": [
        ("sync", "gpu"), ("sync", "quantize"),
        ("gpu", "download"),
        ("guard", None), ("quantize", "quantize_fp8"),
        ("guard", None), ("gpu", "quantize_nf4"),
        ("gpu", "fix_tokenizers"),
        ("gpu", "prepare_data"),
        ("guard", None), ("gpu", "generate_conjectures"),
        ("guard", None), ("gpu", "sample_proofs"),
        ("guard", None), ("gpu", "score_selection"),
        ("guard", None), ("gpu", "train_adapter"),
        ("guard", None), ("gpu", "evaluate_sampling"),
        ("guard", None), ("gpu", "evaluate_loss"),
        ("guard", None), ("gpu", "report"),
    ],
}
# Phase B (spec §13) runs the same steps over several arms (`--arms`): everything up to scoring is a no-op on
# Phase A's store, and the reused arm's training and evaluation are skipped by their done markers.
STAGES["phase_b"] = list(STAGES["phase_a"])
# Diagnostics (the diagnosis of the held-out loss's saturation and of the learning-progress score): one read-only
# step each against a finished run's store (`--profile full`), which already holds the models, the data and the trained adapters.
STAGES["diagnose_tokens"] = [("sync", "gpu"), ("guard", None), ("gpu", "diagnose_tokens"), ("guard", None)]
STAGES["diagnose_gradients"] = [("sync", "gpu"), ("guard", None), ("gpu", "diagnose_gradients"),
                                ("guard", None), ("gpu", "diagnose_gradients_second"), ("guard", None)]
STAGES["diagnose_native_round"] = [("sync", "gpu"), ("guard", None), ("gpu", "diagnose_native_round"), ("guard", None)]

# The ladder's first rung and what goes with it, as ONE task (spec §13b):
#   part 1  §13b: one pass on the 50% band in the native format, the fresh-sample probes (base and adapter), the
#           standard evaluation, the census of false conjectures, the report with its branch;
#   part 2  the escalation §13a's scout earned: `native_same_picks` seeds 1 and 2, and the read over three seeds;
#   part 3  last and optional: the base model's reach on the workbook problems it never proved.
# A step may carry options: its own `arms` and `seeds` (the task's arguments otherwise), a `label` for its
# result file when a step runs twice, and its `part`. A failure ends its own part and the parts that need it
# (`needs`); every finished step's results are already in the store and stay there.
_LADDER = {"arms": "ladder_half_native", "seeds": "0", "part": 1}
_SCOUT_SEEDS = {"arms": "native_same_picks", "seeds": "0,1,2", "part": 2}
_REACH = {"part": 3, "needs": (1, 2)}
STAGES["ladder_13b"] = [
    ("sync", "gpu"), ("gpu", "fix_tokenizers"),
    ("guard", None, _LADDER), ("gpu", "train_adapter", {**_LADDER, "label": "train_adapter_ladder"}),
    ("guard", None, _LADDER), ("gpu", "ladder_probes", _LADDER),
    ("guard", None, _LADDER), ("gpu", "evaluate_sampling", {**_LADDER, "label": "evaluate_sampling_ladder"}),
    ("guard", None, _LADDER), ("gpu", "evaluate_loss", {**_LADDER, "label": "evaluate_loss_ladder"}),
    ("guard", None, _LADDER), ("gpu", "negation_census", _LADDER),
    ("guard", None, _LADDER), ("gpu", "report", {**_LADDER, "label": "report_ladder"}),
    ("guard", None, _SCOUT_SEEDS), ("gpu", "train_adapter", {**_SCOUT_SEEDS, "label": "train_adapter_scout"}),
    ("guard", None, _SCOUT_SEEDS), ("gpu", "evaluate_sampling", {**_SCOUT_SEEDS, "label": "evaluate_sampling_scout"}),
    ("guard", None, _SCOUT_SEEDS), ("gpu", "evaluate_loss", {**_SCOUT_SEEDS, "label": "evaluate_loss_scout"}),
    ("guard", None, _SCOUT_SEEDS), ("gpu", "report", {**_SCOUT_SEEDS, "label": "report_scout"}),
    ("guard", None, _REACH), ("gpu", "base_reach", _REACH),
    ("guard", None, _REACH),
]


# The ladder loop's L0, the GPU half (docs/spec/ladder-loop.spec.md; `gpu/ladder_loop.py`): the base model's
# episodes on the held-out set H and on the base-map sample of the pool, both read from the package's data
# directory (they travel with the code), at the ladder loop's own Lean pin. No parts: a failure, or a
# soundness alarm (a step's exit code 3), ends the stage; every block that finished is in the box's store.
_LADDER_L0B = [
    ("sync", "gpu"), ("gpu", "fix_tokenizers"),
    ("gpu", "ladder_prepare"),
    ("guard", None), ("gpu", "ladder_episodes_heldout"),
    ("guard", None), ("gpu", "ladder_episodes_base_map"),
    ("guard", None), ("gpu", "ladder_l0_report"),
]
STAGES["ladder_l0b"] = list(_LADDER_L0B)
# The same steps on the ten-statement fixture, in a run directory of its own: the ONE small task that goes to a
# terminal state before the real one is queued. It is the first stage run at Lean v4.27 (the OEIS Open spec, O2a 9g).
_LADDER_SMOKE = {"environment": {"RLVR_LEAN_LADDER_DATA": str(PACKAGE_DIR / "data" / "ladder_l0_fixture"),
                                 "RLVR_LEAN_LADDER_RUN": "ladder_l0_smoke"}}
STAGES["ladder_l0b_smoke"] = [(environment, step, _LADDER_SMOKE) for environment, step in _LADDER_L0B]

# The ladder loop's L1 (spec ladder-loop, "A round" and "L1's read"; `gpu/ladder_round.py`): ONE round, two arms that
# differ only in who chooses the round's problems (the challenger, or a uniform draw of the same candidates), on one
# box, one after the other. Shared first: the data check, the statement embeddings, both arms' proposals, the base's
# fresh episodes on G and on the rungs. Then each arm: its round (n episodes of the base), one training pass, the
# trained model's measurements. One seed per task (`--seeds 0`): its run directory is the seed's. No parts: a
# failure, or a soundness alarm, ends the stage; every finished step and block is in the box's store and a rerun
# resumes there.
_LADDER_L1 = [
    ("sync", "gpu"), ("gpu", "fix_tokenizers"),
    ("gpu", "ladder_l1_prepare"),
    ("guard", None), ("gpu", "ladder_l1_embed"),
    ("guard", None), ("gpu", "ladder_l1_propose"),
    ("guard", None), ("gpu", "ladder_l1_base"),
]
for _arm in ("challenger", "random"):
    _LADDER_L1 += [("guard", None), ("gpu", f"ladder_l1_round_{_arm}"),
                   ("guard", None), ("gpu", f"ladder_l1_train_{_arm}"),
                   ("guard", None), ("gpu", f"ladder_l1_measure_{_arm}")]
_LADDER_L1 += [("guard", None), ("gpu", "ladder_l1_report")]
STAGES["ladder_l1"] = list(_LADDER_L1)
# The same steps on the fixtures (L0's ten statements, L1's twelve candidates and made-up base results), in a run
# directory of its own: the ONE small task that goes to a terminal state before the real one is queued. It is the
# first run of the embedding step, of a native-format adapter served beside the base, and of the report.
_LADDER_L1_SMOKE = {"environment": {"RLVR_LEAN_LADDER_DATA": str(PACKAGE_DIR / "data" / "ladder_l0_fixture"),
                                    "RLVR_LEAN_LADDER_ROUND_DATA": str(PACKAGE_DIR / "data" / "ladder_l1_fixture"),
                                    "RLVR_LEAN_LADDER_ROUND_RUN": "ladder_l1_smoke"}}
STAGES["ladder_l1_smoke"] = [(environment, step, _LADDER_L1_SMOKE) for environment, step in _LADDER_L1]

# The ladder loop's L1b, the dose curve (spec ladder-loop, "L1b: the dose curve"; `gpu/ladder_dose.py`): one seed's
# challenger arm again, from the base, on the training proofs L1 stored, for three passes, with the loss read over
# time and the held-out rungs measured at five checkpoints. It READS the L1 run directory of its seed in the box's
# store and writes a run directory of its own. One seed per task (`--seeds 0`). No parts: a failure, or a soundness
# alarm, ends the stage; a rerun resumes at the first step (and, inside a measurement, the first block) not done.
_LADDER_L1B = [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_l1b_prepare"), ("guard", None), ("gpu", "ladder_l1b_train")]
for _checkpoint in ("p050", "p100", "p150", "p200", "p300"):
    _LADDER_L1B += [("guard", None), ("gpu", f"ladder_l1b_measure_{_checkpoint}")]
_LADDER_L1B += [("guard", None), ("gpu", "ladder_l1b_report")]
STAGES["ladder_l1b"] = list(_LADDER_L1B)
# The same steps on what the L1 SMOKE run left in the box's store (`ladder_l1_smoke`: a dozen training proofs, five
# rung problems), in a run directory of its own.
_LADDER_L1B_SMOKE = {"environment": {"RLVR_LEAN_LADDER_DOSE_SOURCE": "ladder_l1_smoke", "RLVR_LEAN_LADDER_DOSE_RUN": "ladder_l1b_smoke"}}
STAGES["ladder_l1b_smoke"] = [(environment, step, _LADDER_L1B_SMOKE) for environment, step in _LADDER_L1B]

# The ladder loop's L2 (spec ladder-loop, "L2: three rounds"; `gpu/ladder_l2.py`): three rounds of the challenger arm at
# one seed, then the equal-compute control. Round r starts from M(r - 1) and trains M(r) from the base on every round's
# training set so far; M(0) is the base. A round's 1,000 proposals are made in four batches inside ONE process (the
# challenger is refitted after each batch's episodes), then one training pass, then M(r) on the held-out rungs and on G.
# It READS the L1 run directory of its seed in the box's store (G, the rungs, the base's fresh results) and writes a run
# directory of its own; the statement embeddings are stored once per box, beside the runs, and another seed's task
# reuses them. One seed per task (`--seeds 0`). No parts: a failure, or a soundness alarm, ends the stage; a rerun resumes
# at the first step, batch and block not done. A round whose below-band interval lies entirely below zero stops the
# loop: the steps after it run nothing and the report says so.
_LADDER_L2 = [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_l2_prepare"), ("guard", None), ("gpu", "ladder_l2_embed")]
for _round in (1, 2, 3):
    _LADDER_L2 += [("guard", None), ("gpu", f"ladder_l2_round_{_round}"),
                   ("guard", None), ("gpu", f"ladder_l2_train_{_round}"),
                   ("guard", None), ("gpu", f"ladder_l2_measure_{_round}")]
# After the control, the LAST model's own extra attempts on G (added 2026-10-05, after seeds 0 and 1): the control's
# number of episodes with the control's sampling seed, so M(3) and the base are compared at equal attempts. A run that
# finished before this step existed gets it alone when its task is queued again: every step before it returns what is
# stored, and the report is built again with the new part.
_LADDER_L2 += [("guard", None), ("gpu", "ladder_l2_control"), ("guard", None), ("gpu", "ladder_l2_control_trained"),
               ("guard", None), ("gpu", "ladder_l2_report")]
STAGES["ladder_l2"] = list(_LADDER_L2)
# The same steps on the fixtures (L0's ten statements; L1's twelve candidates as the whole pool), reading what the L1
# SMOKE run left in the box's store (`ladder_l1_smoke`), in a run directory of its own. Tiny sizes: three rounds of four
# problems, in two batches of two, use the twelve candidates up.
_LADDER_L2_SMOKE = {"environment": {"RLVR_LEAN_LADDER_DATA": str(PACKAGE_DIR / "data" / "ladder_l0_fixture"),
                                    "RLVR_LEAN_LADDER_L2_DATA": str(PACKAGE_DIR / "data" / "ladder_l1_fixture"),
                                    "RLVR_LEAN_LADDER_L2_SOURCE": "ladder_l1_smoke", "RLVR_LEAN_LADDER_L2_RUN": "ladder_l2_smoke",
                                    "RLVR_LEAN_LADDER_L2_PROBLEMS": "4", "RLVR_LEAN_LADDER_L2_BATCHES": "2"}}
STAGES["ladder_l2_smoke"] = [(environment, step, _LADDER_L2_SMOKE) for environment, step in _LADDER_L2]
# L2t, the lower target (spec ladder-loop, "L2t: the lower target"): the SAME steps as an ARM of the stage. The arm's
# one changed setting is in the config (`ladder_loop.l2_arms.t010`: the challenger's target rate, 0.10 for 0.25); the
# run directory is the arm's own (`ladder_l2_t010_seed<N>`); the candidates, batches, seeds, sampling seeds and the
# held-out rungs (L1's stored groups) are the ones `ladder_l2` has, and the stored statement embeddings are reused.
_LADDER_L2_T010 = {"environment": {"RLVR_LEAN_LADDER_L2_ARM": "t010"}}
STAGES["ladder_l2_t010"] = [(environment, step, _LADDER_L2_T010) for environment, step in _LADDER_L2]

# The ladder loop's L3a, the repair check, no training (spec ladder-loop, "L3a: the repair check, no training";
# `gpu/ladder_l3a.py`): the base model's blind first attempts on G and on the three rungs, then every one that failed
# continued in three arms from that same attempt, each to at most four more: blind (whole proofs), resume with the state
# (the arm's latest failed proof cut at its first error, Lean's proof state there as a comment), resume without the
# state. ONE sampling step, one engine: its loops alternate generation and Lean checks. It READS the L1 run directory of
# its seed in the box's store (G and the rungs with their negations and groups) and writes a run directory of its own.
# One seed per task (`--seeds 0`: it is the sampling seed). No parts: a failure, or a soundness alarm, ends the stage;
# a rerun resumes at the first loop that is not done.
_LADDER_L3A = [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_l3a_prepare"),
               ("guard", None), ("gpu", "ladder_l3a_attempts"), ("guard", None), ("gpu", "ladder_l3a_report")]
STAGES["ladder_l3a"] = list(_LADDER_L3A)
# The same steps on what the L1 SMOKE run left in the box's store (`ladder_l1_smoke`: the fixture's one goal problem and
# five rung problems), in a run directory of its own.
_LADDER_L3A_SMOKE = {"environment": {"RLVR_LEAN_LADDER_L3A_SOURCE": "ladder_l1_smoke", "RLVR_LEAN_LADDER_L3A_RUN": "ladder_l3a_smoke"}}
STAGES["ladder_l3a_smoke"] = [(environment, step, _LADDER_L3A_SMOKE) for environment, step in _LADDER_L3A]

# The ladder loop's L3a2, one repair step after each fresh failure, then start over (spec ladder-loop, "L3a2: one repair
# step after each fresh failure, then start over (no training)"; `gpu/ladder_l3a2.py`, on L3a's loops): the base model's
# blind first attempts on G and on the three rungs, then every one that failed continued in two arms from that same
# attempt, each to at most four more: blind (whole proofs), and alternate (attempts 2 and 4 are one repair step each, from
# the fresh attempt just before; attempts 3 and 5 are whole proofs). In both arms a proof Lean already rejected in the
# same episode is not sent to Lean again. ONE sampling step, one engine. It READS the L1 run directory of its seed in the
# box's store and writes a run directory of its own. One seed per task (`--seeds 0`: it is the sampling seed, and no
# draw is one of L3a's). No parts: a failure, or a soundness alarm, ends the stage; a rerun resumes at the first loop
# that is not done.
_LADDER_L3A2 = [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_l3a2_prepare"),
                ("guard", None), ("gpu", "ladder_l3a2_attempts"), ("guard", None), ("gpu", "ladder_l3a2_report")]
STAGES["ladder_l3a2"] = list(_LADDER_L3A2)
# The same steps on what the L1 SMOKE run left in the box's store (`ladder_l1_smoke`), in a run directory of its own,
# with the episodes L3a's smoke run has (`ladder_loop.repair`: 4 on the fixture's goal problem, 2 on each of its five rung
# problems) and not the 12 of the real run.
_LADDER_L3A2_SMOKE = {"environment": {"RLVR_LEAN_LADDER_L3A2_SOURCE": "ladder_l1_smoke", "RLVR_LEAN_LADDER_L3A2_RUN": "ladder_l3a2_smoke",
                                      "RLVR_LEAN_LADDER_L3A2_EPISODES": "repair"}}
STAGES["ladder_l3a2_smoke"] = [(environment, step, _LADDER_L3A2_SMOKE) for environment, step in _LADDER_L3A2]

# The ladder loop's L3c, an episode that keeps what verified (spec ladder-loop, "L3c: an episode that keeps what verified (no
# training)"; `gpu/ladder_l3c.py`, with L3a's run, first attempts and sampling): the base model's first generations on G and
# on the three rungs, then every one that failed continued in two arms from that same generation, each to at most seven
# more: blind (whole proofs), and accumulate (the episode holds a pool of verified lemmas harvested from its failed proofs;
# a generation continues from the pool and Lean's state after it when the pool has changed, and is a whole proof otherwise;
# the kept closing steps are tried again after the pool whenever it has grown). In both arms a proof Lean already rejected
# in the same episode is not sent to Lean again. ONE sampling step, one engine. It READS the L1 run directory of its seed in
# the box's store and writes a run directory of its own. One seed per task (`--seeds 0`: it is the sampling seed, and no
# generation is one of L3a's or L3a2's). No parts: a failure, or a soundness alarm, ends the stage; a rerun resumes at the
# first loop that is not done.
_LADDER_L3C = [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_l3c_prepare"),
               ("guard", None), ("gpu", "ladder_l3c_attempts"), ("guard", None), ("gpu", "ladder_l3c_report")]
STAGES["ladder_l3c"] = list(_LADDER_L3C)
# The PILOT: the same steps on the real problems, the first 40 of G and the first 40 of the below-band rung at 2 episodes
# each (`ladder_loop.accumulate.pilot_problems`, `pilot_episodes`), in a run directory of its own
# (`ladder_l3c_pilot_seed<N>`) and with a sampling seed of its own. It is the run that reaches a continuing generation
# before the real one is queued: the fixture's problems are too easy for the smoke run to. Not read as a result.
_LADDER_L3C_PILOT = {"environment": {"RLVR_LEAN_LADDER_L3C_PILOT": "1"}}
STAGES["ladder_l3c_pilot"] = [(environment, step, _LADDER_L3C_PILOT) for environment, step in _LADDER_L3C]
# The same steps on what the L1 SMOKE run left in the box's store (`ladder_l1_smoke`: the fixture's one goal problem and
# five rung problems), in a run directory of its own, with the stage's own episodes (6 on a hard problem, 2 on the others).
_LADDER_L3C_SMOKE = {"environment": {"RLVR_LEAN_LADDER_L3C_SOURCE": "ladder_l1_smoke", "RLVR_LEAN_LADDER_L3C_RUN": "ladder_l3c_smoke"}}
STAGES["ladder_l3c_smoke"] = [(environment, step, _LADDER_L3C_SMOKE) for environment, step in _LADDER_L3C]

# The ceiling: a labelled diagnostic (spec ladder-loop, "The ceiling: a labelled diagnostic"; `gpu/ladder_ceiling.py`). AN
# EXCEPTION to the rule that published proofs are certificates and not training text: ONE training from the base, one pass
# over the published proofs of `data/ladder_ceiling/training.jsonl` in the file's order (the round's recipe otherwise),
# an adapter saved after the smaller dose and at the end, and each measured in one-shot attempts: 8 episodes on the three
# held-out rungs and 93 attempts on every goal problem, with the sampling seeds L2 used for its models. It READS, in the
# box's store, the L1 run directory of its seed (G, the rungs, the base's fresh results) and L2's two (the base's control
# attempts on G; the three-round model at target 1/10), and writes a run directory of its own. The report step deletes the
# adapters once a report that can be read is written: no model trained this way is kept. (A report that is not to be read,
# Lean having answered too little of a set, keeps them and fails: queued again, the task measures that set again from
# them.) One seed per task (`--seeds 0`). No parts: a failure, or a soundness alarm, ends the stage; a rerun resumes at
# the first step (and, inside a measurement, the first block) not done, and a run whose report is written is not trained
# again.
_LADDER_CEILING = [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_ceiling_prepare"), ("guard", None), ("gpu", "ladder_ceiling_train")]
for _checkpoint in ("small", "full"):
    _LADDER_CEILING += [("guard", None), ("gpu", f"ladder_ceiling_measure_{_checkpoint}")]
_LADDER_CEILING += [("guard", None), ("gpu", "ladder_ceiling_report")]
STAGES["ladder_ceiling"] = list(_LADDER_CEILING)
# The same steps on what the L1 SMOKE run left in the box's store (`ladder_l1_smoke`: the fixture's one goal problem and five
# rung problems), in a run directory of its own, with a 24-row fixture in the training file's place (`data/ladder_ceiling_fixture`:
# published proofs, not distributed with this copy) and its two checkpoints
# after 12 and 24 rows (the first is saved after the second optimizer step, 16 rows seen). No L2 smoke run holds the stored
# attempts the real run reads, so none are read: the goal problem gets its first sampling only.
_LADDER_CEILING_SMOKE = {"environment": {"RLVR_LEAN_LADDER_CEILING_SOURCE": "ladder_l1_smoke", "RLVR_LEAN_LADDER_CEILING_RUN": "ladder_ceiling_smoke",
                                         "RLVR_LEAN_LADDER_CEILING_STORED": "none",
                                         "RLVR_LEAN_LADDER_CEILING_TRAINING": str(PACKAGE_DIR / "data" / "ladder_ceiling_fixture" / "training.jsonl"),
                                         "RLVR_LEAN_LADDER_CEILING_CHECKPOINTS": "12,24"}}
STAGES["ladder_ceiling_smoke"] = [(environment, step, _LADDER_CEILING_SMOKE) for environment, step in _LADDER_CEILING]

# L3d Step 1: do assembled proofs teach? (spec ladder-loop, "L3d: train on what the episode reaches", Step 1; `gpu/ladder_l3d1.py`).
# The ceiling stage's shape with two trainings in the place of two checkpoints. TWO models from the base, one pass each, the
# round's recipe: `without` (the training examples the three rounds at target 1/10 stored at this seed, in a seeded order)
# and `with` (the same rows in the same relative order, with the assembled proofs of `data/ladder_l3d/harvest_h0.jsonl` at
# seeded places among them). Each is measured as the ceiling's models are: 8 episodes on the three held-out rungs and 93
# attempts on every goal problem, with the sampling seeds L2 used for its models. It READS, in the box's store, the L1 run
# directory of its seed and L2's two (the base's control attempts on G; the run at target 1/10: its three-round model and its
# three rounds' training examples), and writes a run directory of its own. Both adapters are KEPT: they are the loop's own
# models. One seed per task (`--seeds 0`). No parts: a failure, or a soundness alarm, ends the stage; a rerun resumes at the
# first step (and, inside a measurement, the first block) not done; a report that is not to be read fails its step, and the
# task queued again measures the set Lean did not answer again from the kept adapter.
_LADDER_L3D1 = [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_l3d1_prepare")]
for _model in ("without", "with"):
    _LADDER_L3D1 += [("guard", None), ("gpu", f"ladder_l3d1_train_{_model}")]
for _model in ("without", "with"):
    _LADDER_L3D1 += [("guard", None), ("gpu", f"ladder_l3d1_measure_{_model}")]
_LADDER_L3D1 += [("guard", None), ("gpu", "ladder_l3d1_report")]
STAGES["ladder_l3d1"] = list(_LADDER_L3D1)
# The same steps on what the L1 SMOKE run left in the box's store (`ladder_l1_smoke`: the fixture's one goal problem and five
# rung problems), in a run directory of its own. No L2 smoke run holds what the real run reads, so none is read: `without`'s
# rows are the L1 smoke run's OWN stored training examples (`training_examples_challenger.jsonl`), the goal problem gets its
# first sampling only and no three-round model stands beside. H0 is the 8-row fixture (`data/ladder_l3d_fixture`: assembled
# proofs of pool problems from the round replay of 2026-10-08, the model's own), with the least number of rows lowered to 8.
_LADDER_L3D1_SMOKE = {"environment": {"RLVR_LEAN_LADDER_L3D1_SOURCE": "ladder_l1_smoke", "RLVR_LEAN_LADDER_L3D1_RUN": "ladder_l3d1_smoke",
                                      "RLVR_LEAN_LADDER_L3D1_STORED": "none",
                                      "RLVR_LEAN_LADDER_L3D1_H0": str(PACKAGE_DIR / "data" / "ladder_l3d_fixture" / "harvest_h0.jsonl"),
                                      "RLVR_LEAN_LADDER_L3D1_MINIMUM": "8"}}
STAGES["ladder_l3d1_smoke"] = [(environment, step, _LADDER_L3D1_SMOKE) for environment, step in _LADDER_L3D1]

# L3d Step 2: the loop with assembly in the round, and a twin trained without the assembled proofs (spec ladder-loop, "L3d:
# train on what the episode reaches", Step 2; `gpu/ladder_l3d2.py` on the L2 stage's arm `t010_assembly`, `gpu/ladder_assembly.py`).
# The arm: the target rate 1/10, SIX rounds, and after each batch's attempts the Lean-only assembly over every problem none
# of them resolved (a problem it resolves counts as k = 1 for the challenger, and its minimised proof is its training example);
# M(r) is trained from the base after each round on the rounds' own proofs and the harvest H0's. No model is measured between
# the rounds. After round six: the TWIN, trained from the base on the rounds' one-shot proofs alone in the same order; then
# `with` (M(6)) and `without` (the twin) are each measured as the ceiling's models are: 8 episodes on the three held-out rungs
# and 93 attempts on every goal problem. It READS, in the box's store, the L1 run directory of its seed and L2's two (the base's
# control attempts on G; the three-round model at target 1/10), and writes the arm's run directory. Every adapter is KEPT. One
# seed per task (`--seeds 0`). No parts: a failure, or a soundness alarm, ends the stage; a rerun resumes at the first step,
# batch and block not done.
def _l3d2_steps(rounds: int) -> list:
    steps = [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_l2_prepare"), ("guard", None), ("gpu", "ladder_l3d2_prepare"),
             ("guard", None), ("gpu", "ladder_l2_embed")]
    for number in range(1, rounds + 1):
        steps += [("guard", None), ("gpu", f"ladder_l2_round_{number}"), ("guard", None), ("gpu", f"ladder_l2_train_{number}")]
    steps += [("guard", None), ("gpu", "ladder_l3d2_train_without")]
    for model in ("with", "without"):
        steps += [("guard", None), ("gpu", f"ladder_l3d2_measure_{model}")]
    return steps + [("guard", None), ("gpu", "ladder_l3d2_report")]


_LADDER_L3D2 = {"environment": {"RLVR_LEAN_LADDER_L2_ARM": "t010_assembly"}}
STAGES["ladder_l3d2"] = [(environment, step, _LADDER_L3D2) for environment, step in _l3d2_steps(6)]
# The same steps as the L2 smoke run does them (the fixtures; what the L1 SMOKE run left in the box's store), with TWO rounds of
# four problems in two batches of two, assembly on, the 8-row fixture as H0 and the least number of assembled proofs lowered
# to 8. No L2 smoke run holds what the real run reads beside its models, so none is read: the goal problem gets its first
# sampling only and no three-round model stands beside. `with` is M(2).
_LADDER_L3D2_SMOKE = {"environment": {**_LADDER_L2_SMOKE["environment"], "RLVR_LEAN_LADDER_L2_RUN": "ladder_l3d2_smoke",
                                      "RLVR_LEAN_LADDER_L2_ARM": "t010_assembly", "RLVR_LEAN_LADDER_L2_ROUNDS": "2",
                                      "RLVR_LEAN_LADDER_L3D2_STORED": "none", "RLVR_LEAN_LADDER_L3D2_MINIMUM": "8",
                                      "RLVR_LEAN_LADDER_L3D2_H0": str(PACKAGE_DIR / "data" / "ladder_l3d_fixture" / "harvest_h0.jsonl")}}
STAGES["ladder_l3d2_smoke"] = [(environment, step, _LADDER_L3D2_SMOKE) for environment, step in _l3d2_steps(2)]

# L4, the loop from a model PRETRAINED ON PUBLISHED PROOFS (spec ladder-loop, "L4: the loop from a model pretrained on published
# proofs"; `gpu/ladder_l4.py`). TWO stages, two tasks. Everything they produce is labelled pretrained on published proofs.
#
# `ladder_l4_pretrain`: the ceiling stage's shape with one model. From the base, the round's recipe, ONE pass over the published
# proofs of `data/ladder_l4/pretraining.jsonl` (the `pretrain` half of the pool) in the file's order; the adapter `pre` is KEPT
# (`ladder_l4_pretrain_seed<N>/adapters/pre`); it is measured as the ceiling's models are (8 episodes on the three rungs, 93
# attempts on every goal problem); then `pre`'s OWN MAP is made (the base map's problems attempted again by `pre`, 8 attempts
# each with the stored map's sides and sampling seed: what the arm's challenger starts from); and its report stores the goal
# set drawn again (G') and the two checks the arm stands on. It READS, in the box's store, the L1 run directory of its seed
# and L2's two, and the ceiling's run when its report is there. A training that did not finish starts again from the base: the
# pass is one schedule and one optimizer, kept only as the adapter.
_LADDER_L4_PRETRAIN = [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_l4_pretrain_prepare"), ("guard", None), ("gpu", "ladder_l4_pretrain_train"),
                       ("guard", None), ("gpu", "ladder_l4_pretrain_measure"), ("guard", None), ("gpu", "ladder_l4_pretrain_map"),
                       ("guard", None), ("gpu", "ladder_l4_pretrain_report")]
STAGES["ladder_l4_pretrain"] = list(_LADDER_L4_PRETRAIN)
# The same steps on what the L1 SMOKE run left in the box's store (`ladder_l1_smoke`), in a run directory of its own, with the
# 12-row fixture in the pretraining file's place (`data/ladder_l4_fixture`: published proofs of the `pretrain` half, not
# distributed with this copy) and the two checks' minimums at zero (the fixture's world has one goal problem). No L2 run is read. The data are
# the L2 smoke run's fixtures, so that the map is of the fixture's four base-map problems: the ones the arm's smoke run reads.
_LADDER_L4_PRETRAIN_SMOKE = {"environment": {"RLVR_LEAN_LADDER_DATA": _LADDER_L2_SMOKE["environment"]["RLVR_LEAN_LADDER_DATA"],
                                             "RLVR_LEAN_LADDER_L2_DATA": _LADDER_L2_SMOKE["environment"]["RLVR_LEAN_LADDER_L2_DATA"],
                                             "RLVR_LEAN_LADDER_L4_SOURCE": "ladder_l1_smoke", "RLVR_LEAN_LADDER_L4_PRETRAIN_RUN": "ladder_l4_pretrain_smoke",
                                             "RLVR_LEAN_LADDER_L4_STORED": "none", "RLVR_LEAN_LADDER_L4_MINIMUMS": "0,0",
                                             "RLVR_LEAN_LADDER_L4_FILE": str(PACKAGE_DIR / "data" / "ladder_l4_fixture" / "pretraining.jsonl")}}
STAGES["ladder_l4_pretrain_smoke"] = [(environment, step, _LADDER_L4_PRETRAIN_SMOKE) for environment, step in _LADDER_L4_PRETRAIN]


# `ladder_l4`: L3d Step 2's steps again as the arm `t010_assembly_pre` (six rounds at the target rate 1/10 with assembly after
# each batch, the twin, the two measurements), with a start adapter: round 1 is ATTEMPTED by `pre` and every model of the arm
# (M(1) to M(6), and the twin) is trained FROM `pre`; the candidates are the `loop` half of the pool; no H0. Two steps of its
# own: `ladder_l4_prepare`, before anything is sampled (it refuses a pretraining run whose report is not written or whose two
# checks failed, and a missing adapter), and `ladder_l4_report`. It READS, in the box's store, the ONE pretraining run (that of
# the seed `ladder_loop.l4.pretraining_seed`, whatever the task's own seed: the pretraining is not repeated for another seed of
# the arm, and `--seeds 1` and `--seeds 2` read seed 0's `pre`, map and G'), the L1 run and L2's two of the task's seed, and the
# base arm's run of that seed when it is there; it writes the arm's run directory (`ladder_l2_t010_assembly_pre_seed<N>`).
# Every adapter is KEPT. Resumes as Step 2 does.
def _l4_steps(rounds: int) -> list:
    steps = _l3d2_steps(rounds)
    after = steps.index(("gpu", "ladder_l2_prepare")) + 1
    return steps[:after] + [("guard", None), ("gpu", "ladder_l4_prepare")] + steps[after:-1] + [("gpu", "ladder_l4_report")]


_LADDER_L4 = {"environment": {"RLVR_LEAN_LADDER_L2_ARM": "t010_assembly_pre"}}
STAGES["ladder_l4"] = [(environment, step, _LADDER_L4) for environment, step in _l4_steps(6)]
# The same steps in the L2 smoke run's world, reading the pretraining SMOKE run (`ladder_l4_pretrain_smoke`: its adapter, its
# report): TWO rounds of two problems in two batches of one (five of the fixture's twelve candidates are of the `loop` half).
# No L2 smoke run holds what the real run reads beside its models, and no base arm's run is read: the goal problem gets its
# first sampling only, so the primary has no fresh attempts and is not read.
_LADDER_L4_SMOKE = {"environment": {**_LADDER_L2_SMOKE["environment"], "RLVR_LEAN_LADDER_L2_RUN": "ladder_l4_smoke", "RLVR_LEAN_LADDER_L2_ARM": "t010_assembly_pre",
                                    "RLVR_LEAN_LADDER_L2_ROUNDS": "2", "RLVR_LEAN_LADDER_L2_PROBLEMS": "2", "RLVR_LEAN_LADDER_L3D2_STORED": "none",
                                    "RLVR_LEAN_LADDER_L4_PRETRAIN_RUN": "ladder_l4_pretrain_smoke"}}
STAGES["ladder_l4_smoke"] = [(environment, step, _LADDER_L4_SMOKE) for environment, step in _l4_steps(2)]

# L4r: is the pretrained model capped by the size of its adapter? (spec ladder-loop, "L4r: is the pretrained model capped by the
# size of its adapter?"; `gpu/ladder_l4_rank.py`, on the pretraining stage's own functions). A labelled check of the pretraining,
# labelled pretrained on published proofs as L4 is. `pre`'s pass AGAIN, from the base, over the same file in the same order with
# the same seed, with ONE change: the adapter's rank and alpha (`ladder_loop.l4.rank_check`: 64 and 128 where `pre` has 16 and
# 32). The model is `pre_r64`, in a run directory of its own (`ladder_l4_pretrain_r64_seed<N>`); it is measured as `pre` was
# (8 episodes on the three rungs, 93 attempts on every goal problem, `pre`'s sampling seeds), with the model server's largest
# adapter rank raised to the check's for that step alone, and reported against `pre`'s stored rows. No map is made, and no step
# deletes the adapter. It READS, in the box's store, what the pretraining read (the L1 run of its seed and L2's two) and the ONE
# pretraining's run (`ladder_l4_pretrain_seed<pretraining_seed>`: its report, its prepare and measure steps' markers, its
# training record, `pre`'s rows on its three sets). The stage names its check (`RLVR_LEAN_LADDER_L4_RANK_CHECK`); no other stage
# sets the variable, and the pretraining stage does not read it. A training that runs out of GPU memory fails its step and says
# to queue the fallback: nothing smaller is tried inside the task.
_LADDER_L4_RANK = [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_l4_rank_prepare"), ("guard", None), ("gpu", "ladder_l4_rank_train"),
                   ("guard", None), ("gpu", "ladder_l4_rank_measure"), ("guard", None), ("gpu", "ladder_l4_rank_report")]
_LADDER_L4_RANK_CHECK = {"environment": {"RLVR_LEAN_LADDER_L4_RANK_CHECK": "rank"}}
STAGES["ladder_l4_rank"] = [(environment, step, _LADDER_L4_RANK_CHECK) for environment, step in _LADDER_L4_RANK]
# THE FALLBACK, run ONLY when the box could not hold the rank above (that run is then VOID): the same steps at the setting's
# smaller rank and alpha (32 and 64), the model `pre_r32`, the run directory `ladder_l4_pretrain_r32_seed<N>`, the same read.
_LADDER_L4_RANK_FALLBACK = {"environment": {"RLVR_LEAN_LADDER_L4_RANK_CHECK": "fallback"}}
STAGES["ladder_l4_rank_fallback"] = [(environment, step, _LADDER_L4_RANK_FALLBACK) for environment, step in _LADDER_L4_RANK]
# The same steps AT THE REAL RANK in the pretraining smoke run's world: what the L1 SMOKE run left in the box's store
# (`ladder_l1_smoke`), the 12-row fixture in the pretraining file's place, no L2 run read, and the pretraining SMOKE run
# (`ladder_l4_pretrain_smoke`) as `pre`'s run. A rank-64 adapter is trained, saved, read back, served by the model server and
# measured once on the box before the real task is queued.
_LADDER_L4_RANK_SMOKE = {"environment": {"RLVR_LEAN_LADDER_DATA": _LADDER_L4_PRETRAIN_SMOKE["environment"]["RLVR_LEAN_LADDER_DATA"],
                                         "RLVR_LEAN_LADDER_L4_SOURCE": "ladder_l1_smoke", "RLVR_LEAN_LADDER_L4_STORED": "none",
                                         "RLVR_LEAN_LADDER_L4_FILE": _LADDER_L4_PRETRAIN_SMOKE["environment"]["RLVR_LEAN_LADDER_L4_FILE"],
                                         "RLVR_LEAN_LADDER_L4_PRETRAIN_RUN": "ladder_l4_pretrain_smoke", "RLVR_LEAN_LADDER_L4_RANK_RUN": "ladder_l4_rank_smoke",
                                         "RLVR_LEAN_LADDER_L4_RANK_CHECK": "rank"}}
STAGES["ladder_l4_rank_smoke"] = [(environment, step, _LADDER_L4_RANK_SMOKE) for environment, step in _LADDER_L4_RANK]

# L4t: what should a round train on? Three one-change checks on the rounds already made (spec ladder-loop, "L4t: what should a round
# train on? Three one-change checks on the rounds already made"; `gpu/ladder_l4_rows.py`). Labelled pretrained on published proofs,
# as everything built on `pre` is. NO NEW ROUND. It READS, in the box's store and never writing there, the arm's run of its seed
# (`ladder_l2_t010_assembly_pre_seed<N>`: the rounds' training examples and each batch's picks, the twin's rows, the adapter of its
# last model `with`, the per-problem rows of `with` and `without`) and the ONE pretraining's (`ladder_l4_pretrain_seed<pretraining_seed>`:
# the adapter `pre`, its sampling seeds, its per-problem rows, G', the rows it was trained on); it writes `ladder_l4_rows_seed<N>`.
# Two training sets are built from the stored rounds (`rehearse`: the twin's rows and as many rows of the pretraining file;
# `reward_rows`: each row kept with the probability of its problem's reward); each is trained FROM `pre`, one pass, the arm's recipe,
# then measured as `with` was (8 episodes on the three rungs, 93 attempts on every goal problem, `pre`'s sampling seeds) before the
# next is trained. Then `with` and `pre` are each sampled again on G alone at `ladder_loop.l4.rows.temperature_hot`, with sampling
# seeds of their own: the ONLY two steps of any stage that do not sample at the config's temperature. Every adapter is KEPT. One
# seed per task (`--seeds 0`). No parts: a failure, or a soundness alarm, ends the stage; every step is a step of its own and a
# rerun resumes at the first one (and, inside a measurement, the first block) not done.
_LADDER_L4_ROWS = [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_l4_rows_prepare")]
for _model in ("rehearse", "reward_rows"):
    _LADDER_L4_ROWS += [("guard", None), ("gpu", f"ladder_l4_rows_train_{_model}"), ("guard", None), ("gpu", f"ladder_l4_rows_measure_{_model}")]
for _model in ("with", "pre"):
    _LADDER_L4_ROWS += [("guard", None), ("gpu", f"ladder_l4_rows_measure_hot_{_model}")]
_LADDER_L4_ROWS += [("guard", None), ("gpu", "ladder_l4_rows_report")]
STAGES["ladder_l4_rows"] = list(_LADDER_L4_ROWS)
# The same steps in the L4 smoke runs' world: the arm's SMOKE run (`ladder_l4_smoke`: two rounds of two problems) as the arm's run,
# the pretraining SMOKE run (`ladder_l4_pretrain_smoke`) as `pre`'s, the 12-row fixture in the pretraining file's place and the
# fixtures' held-out set, in a run directory of its own. The fixture's problems are solved by every attempt or by none, so the
# rule may keep no row: `reward_rows` then holds at least two (`RLVR_LEAN_LADDER_L4_ROWS_MINIMUM`), marked as not kept by the rule.
_LADDER_L4_ROWS_SMOKE = {"environment": {"RLVR_LEAN_LADDER_DATA": _LADDER_L4_PRETRAIN_SMOKE["environment"]["RLVR_LEAN_LADDER_DATA"],
                                         "RLVR_LEAN_LADDER_L4_FILE": _LADDER_L4_PRETRAIN_SMOKE["environment"]["RLVR_LEAN_LADDER_L4_FILE"],
                                         "RLVR_LEAN_LADDER_L4_PRETRAIN_RUN": "ladder_l4_pretrain_smoke", "RLVR_LEAN_LADDER_L4_ROWS_ARM_RUN": "ladder_l4_smoke",
                                         "RLVR_LEAN_LADDER_L4_ROWS_RUN": "ladder_l4_rows_smoke", "RLVR_LEAN_LADDER_L4_ROWS_MINIMUM": "2"}}
STAGES["ladder_l4_rows_smoke"] = [(environment, step, _LADDER_L4_ROWS_SMOKE) for environment, step in _LADDER_L4_ROWS]
# L4t FROM `pre_r64` (spec ladder-loop, "L4r's first branch was taken: what L4t runs"): the same stage with another start than the
# setting's, which the stage names (`RLVR_LEAN_LADDER_L4_ROWS_START`). The start model is the check of the adapter's rank's
# (`ladder_l4_pretrain_r64_seed<pretraining_seed>/adapters/pre_r64`: only read, with its report, its markers and its rows), every
# model is at ITS rank (64, alpha 128: read from what that run recorded, never the config's 16), and the model server of this
# stage's measure steps alone is started with that rank as its largest. The rows are still the arm's rounds, made from `pre`. A
# THIRD model is trained and measured FIRST, `old_rule` (the twin's rows in the twin's order: what `without` is, at this rank):
# it stands in `without`'s place in every read and is read by itself against `pre_r64`, which is the cap's question. Then
# `reward_rows`, then `rehearse` (the order of `ladder_loop.l4.rows.models_from_another_start`), then the two hot measurements,
# of `old_rule` and of the start model (`..._hot_start`: a step's name is registered without a config), then the report. It
# writes `ladder_l4_rows_pre_r64_seed<N>`.
_LADDER_L4_ROWS_R64 = [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_l4_rows_prepare")]
for _model in ("old_rule", "reward_rows", "rehearse"):
    _LADDER_L4_ROWS_R64 += [("guard", None), ("gpu", f"ladder_l4_rows_train_{_model}"), ("guard", None), ("gpu", f"ladder_l4_rows_measure_{_model}")]
for _model in ("old_rule", "start"):
    _LADDER_L4_ROWS_R64 += [("guard", None), ("gpu", f"ladder_l4_rows_measure_hot_{_model}")]
_LADDER_L4_ROWS_R64 += [("guard", None), ("gpu", "ladder_l4_rows_report")]
_LADDER_L4_ROWS_FROM_R64 = {"environment": {"RLVR_LEAN_LADDER_L4_ROWS_START": "pre_r64"}}
STAGES["ladder_l4_rows_r64"] = [(environment, step, _LADDER_L4_ROWS_FROM_R64) for environment, step in _LADDER_L4_ROWS_R64]
# The same steps in the L4 smoke runs' world: the smoke stage above with the SMOKE run of the check of the adapter's rank
# (`ladder_l4_rank_smoke`: its adapter `pre_r64`, rank 64, and its rows on the fixture) as the start model's run, in a run directory
# of its own. That smoke run's own report reads INCONCLUSIVE (twelve rows have no loss to compare), which a real run from it would
# be refused for: `RLVR_LEAN_LADDER_L4_ROWS_START_CHECKS=smoke` (this stage alone) goes on, and the prepare step records it.
_LADDER_L4_ROWS_R64_SMOKE = {"environment": {**_LADDER_L4_ROWS_SMOKE["environment"], "RLVR_LEAN_LADDER_L4_ROWS_START": "pre_r64",
                                             "RLVR_LEAN_LADDER_L4_RANK_RUN": "ladder_l4_rank_smoke", "RLVR_LEAN_LADDER_L4_ROWS_RUN": "ladder_l4_rows_r64_smoke",
                                             "RLVR_LEAN_LADDER_L4_ROWS_START_CHECKS": "smoke"}}
STAGES["ladder_l4_rows_r64_smoke"] = [(environment, step, _LADDER_L4_ROWS_R64_SMOKE) for environment, step in _LADDER_L4_ROWS_R64]

# L4b: the arm again, from the larger pretrained model (spec ladder-loop, "L4b: the arm again, from the larger pretrained model";
# `gpu/ladder_l4b.py` on the L2 stage's arm `t010_assembly_pre_r64`). Labelled pretrained on published proofs. L4's arm again with
# what L4r and L4t name: it STARTS FROM `pre_r64` (the check of the adapter's rank's model: only read, in
# `ladder_l4_pretrain_r64_seed<pretraining_seed>`), every model of the arm and its twin is trained from it AT ITS RANK (64, alpha
# 128, read from what that run recorded), every step that samples starts the model server for that rank, and the TRAINING RULE is
# the one the STAGE names (`RLVR_LEAN_LADDER_L2_RULE`): `ladder_l4b_old`, `ladder_l4b_reward_rows`, `ladder_l4b_rehearse`. L4t names
# the rule by the spec's own words; whoever queues picks the stage. THE MAP COMES FIRST: `ladder_l4b_map` makes `pre_r64`'s own map
# (the base map's problems, 8 attempts each, the stored map's seed and sides) in a run directory of its own
# (`ladder_l4_map_pre_r64_seed<pretraining_seed>`), so that a failed arm keeps the map and every rule and seed of the arm reads the
# one map. Then L4's steps, with `ladder_l4b_prepare` in the place of `ladder_l4_prepare` (it makes G' for `pre_r64` and reads L4's
# first two checks on it) and `ladder_l4b_report` in the place of `ladder_l4_report` (L4's read, the breadth beside the primary).
# It READS, in the box's store, that check's run, the L1 run and L2's two of the task's seed, and, when they are there, the base
# arm's run and the rank-16 arm's; it writes the map's run and the arm's (`ladder_l2_t010_assembly_pre_r64_seed<N>`, with the rule
# in its name when it is not `old`). Every adapter is KEPT. Resumes as Step 2 does.
def _l4b_steps(rounds: int) -> list:
    renamed = {"ladder_l4_prepare": "ladder_l4b_prepare", "ladder_l4_report": "ladder_l4b_report"}
    steps = [(environment, renamed.get(step, step)) for environment, step in _l4_steps(rounds)]
    after = steps.index(("gpu", "fix_tokenizers")) + 1
    return steps[:after] + [("gpu", "ladder_l4b_map"), ("guard", None)] + steps[after:]


_LADDER_L4B_RULES = ("old", "reward_rows", "rehearse")
# The same steps in the L2 smoke run's world, from the SMOKE run of the check of the adapter's rank (`ladder_l4_rank_smoke`: its
# adapter `pre_r64`, rank 64, and its rows on the fixture): TWO rounds of two problems, the map of the fixture's four base-map
# problems in a run directory of its own (`ladder_l4b_map_smoke`), the 12-row fixture in the pretraining file's place (the rule
# `rehearse` draws among its rows), the two checks' minimums at zero. That smoke run's own report reads INCONCLUSIVE, which a real
# run from it would be refused for: `RLVR_LEAN_LADDER_L2_START_CHECKS=smoke` (these stages alone) goes on, and the prepare steps
# record it. The fixture's problems are solved by every attempt or by none, so the rule `reward_rows` may keep no row: a model's
# set then holds at least two (`RLVR_LEAN_LADDER_L2_RULE_MINIMUM`), each marked as not kept by the rule.
for _rule in _LADDER_L4B_RULES:
    _LADDER_L4B = {"environment": {"RLVR_LEAN_LADDER_L2_ARM": "t010_assembly_pre_r64", "RLVR_LEAN_LADDER_L2_RULE": _rule}}
    STAGES[f"ladder_l4b_{_rule}"] = [(environment, step, _LADDER_L4B) for environment, step in _l4b_steps(6)]
    _LADDER_L4B_SMOKE = {"environment": {**_LADDER_L2_SMOKE["environment"], "RLVR_LEAN_LADDER_L2_RUN": f"ladder_l4b_{_rule}_smoke",
                                         "RLVR_LEAN_LADDER_L2_ARM": "t010_assembly_pre_r64", "RLVR_LEAN_LADDER_L2_RULE": _rule,
                                         "RLVR_LEAN_LADDER_L2_ROUNDS": "2", "RLVR_LEAN_LADDER_L2_PROBLEMS": "2", "RLVR_LEAN_LADDER_L3D2_STORED": "none",
                                         "RLVR_LEAN_LADDER_L4_PRETRAIN_RUN": "ladder_l4_pretrain_smoke", "RLVR_LEAN_LADDER_L4_RANK_RUN": "ladder_l4_rank_smoke",
                                         "RLVR_LEAN_LADDER_L4_MAP_RUN": "ladder_l4b_map_smoke", "RLVR_LEAN_LADDER_L4_MINIMUMS": "0,0",
                                         "RLVR_LEAN_LADDER_L4_FILE": _LADDER_L4_PRETRAIN_SMOKE["environment"]["RLVR_LEAN_LADDER_L4_FILE"],
                                         "RLVR_LEAN_LADDER_L2_START_CHECKS": "smoke", "RLVR_LEAN_LADDER_L2_RULE_MINIMUM": "2"}}
    STAGES[f"ladder_l4b_{_rule}_smoke"] = [(environment, step, _LADDER_L4B_SMOKE) for environment, step in _l4b_steps(2)]
SOUNDNESS_ALARM_EXIT = 3       # `gpu.__main__`'s code for a proof of both sides of one statement (it cannot be imported here)


def step_fields(entry_step: tuple) -> tuple[str, str | None, dict]:
    """(environment, step, options) of one stage entry; the options are empty for a plain (environment, step)."""
    return entry_step[0], entry_step[1], (dict(entry_step[2]) if len(entry_step) > 2 else {})


def part_is_blocked(options: dict, failed_parts: set[int]) -> bool:
    """A step is skipped when its own part has failed, or a part it needs has."""
    return options.get("part", 0) in failed_parts or any(part in failed_parts for part in options.get("needs", ()))


def log(message: str) -> None:
    print(f"[rlvr_lean.entry {time.strftime('%H:%M:%S')}] {message}", flush=True)


def gpu_memory_used_mib() -> int | None:
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=20, check=True).stdout
        return int(output.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _proportional_set_kib(proc_entry: Path, status_fields: dict[str, str]) -> int:
    """Proportional set size: each shared page is split between the processes that map it, so a tree's total
    is real memory. Plain RSS counts shared pages once per process; vLLM's processes share most of theirs,
    and summed RSS read 33 GB on a 30 GB box (Milestone 3 smoke run)."""
    try:
        for line in (proc_entry / "smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1])
    except OSError:
        pass
    return int(status_fields.get("VmRSS", "0 kB").split()[0]) if "VmRSS" in status_fields else 0


def process_tree_rss_mib(root_pid: int) -> int:
    """Memory of `root_pid` and all its descendants (vLLM runs its engine in a child process), as PSS."""
    parents: dict[int, int] = {}
    fields_by_pid: dict[int, dict[str, str]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
        except OSError:
            continue
        fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
        pid = int(entry.name)
        parents[pid] = int(fields.get("PPid", "0").strip() or 0)
        fields_by_pid[pid] = fields
    total, frontier = 0, [root_pid]
    while frontier:                      # only the tree's own processes are measured
        pid = frontier.pop()
        if pid in fields_by_pid:
            total += _proportional_set_kib(Path("/proc") / str(pid), fields_by_pid[pid])
        frontier.extend(child for child, parent in parents.items() if parent == pid)
    return total // 1024


class Monitor:
    """Background sampler: heartbeat events every 60 s, memory samples every 5 s."""

    def __init__(self, out: Path) -> None:
        self.writer = ScalarEventWriter(out / "tb")
        self.started = time.monotonic()
        self.child_pid: int | None = None
        self.peak_gpu_mib = 0
        self.peak_tree_rss_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def reset_peaks(self) -> None:
        self.peak_gpu_mib = 0
        self.peak_tree_rss_mib = 0

    def _run(self) -> None:
        last_heartbeat = 0.0
        while not self._stop.is_set():
            used = gpu_memory_used_mib()
            if used is not None:
                self.peak_gpu_mib = max(self.peak_gpu_mib, used)
            if self.child_pid is not None:
                self.peak_tree_rss_mib = max(self.peak_tree_rss_mib, process_tree_rss_mib(self.child_pid))
            now = time.monotonic()
            if now - last_heartbeat >= 60:
                last_heartbeat = now
                self.writer.scalars(int(now - self.started), {
                    "heartbeat/elapsed_minutes": (now - self.started) / 60,
                    "gpu/used_mib": float(used or 0),
                    "memory/step_tree_rss_mib": float(self.peak_tree_rss_mib),
                })
            self._stop.wait(5)


def ensure_uv() -> Path:
    uv = STORE / "bin" / f"uv-{UV_VERSION}" / "uv"
    if uv.exists():
        return uv
    base = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/"
    archive = STORE / "downloads" / UV_ARCHIVE
    archive.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(base + UV_ARCHIVE, archive)
    expected = urllib.request.urlopen(base + UV_ARCHIVE + ".sha256", timeout=60).read().decode().split()[0]
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"uv archive checksum mismatch: {actual} != {expected}")
    with tarfile.open(archive) as bundle:
        member = next(m for m in bundle.getmembers() if m.name.endswith("/uv"))
        member.name = "uv"
        bundle.extract(member, uv.parent, filter="data")
    uv.chmod(0o755)
    return uv


def environment_python(group: str) -> Path:
    return STORE / "envs" / group / "bin" / "python"


PROFILE = {"value": "smoke", "seeds": "", "arms": ""}      # set from --profile / --seeds / --arms in main(); read by every child step


def child_environment(group: str | None, kimina_api_key: str | None, options: dict | None = None) -> dict[str, str]:
    options = options or {}
    env = dict(os.environ)
    env.update({
        "RLVR_LEAN_PROFILE": PROFILE["value"],
        "RLVR_LEAN_TRAINING_SEEDS": options.get("seeds", PROFILE["seeds"]),      # a step's own, else the task's
        "RLVR_LEAN_ARMS": options.get("arms", PROFILE["arms"]),
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "RLVR_LEAN_STORE": str(STORE),
        "HF_HOME": str(STORE / "hf"),
        "UV_CACHE_DIR": str(STORE / "uv-cache"),
        "UV_PYTHON_INSTALL_DIR": str(STORE / "python"),
        # A job runner may pin every task to one thread (ours does); GPU steps need their own (vLLM tokenizers, data loading).
        "OMP_NUM_THREADS": os.environ.get("RLVR_LEAN_THREADS", "8"),
        "MKL_NUM_THREADS": os.environ.get("RLVR_LEAN_THREADS", "8"),
        "TOKENIZERS_PARALLELISM": "false",
    })
    if group is not None:
        env["UV_PROJECT_ENVIRONMENT"] = str(STORE / "envs" / group)
    if kimina_api_key:
        env["RLVR_LEAN_KIMINA_API_KEY"] = kimina_api_key
    env.update(options.get("environment", {}))          # a stage's own variables for its steps (the smoke run's data)
    return env


def run_child(command: list[str], env: dict[str, str], monitor: Monitor) -> int:
    log("run: " + " ".join(command[:6]) + (" ..." if len(command) > 6 else ""))
    process = subprocess.Popen(command, cwd=REPO_ROOT, env=env)     # cwd inside the repo: a leaked process can be found by it
    monitor.child_pid = process.pid
    try:
        return process.wait()
    finally:
        monitor.child_pid = None


GUARD_RELEASE_MIB = 6000          # a step that held a model gives back at least this much (the engine holds over 11 GiB)


def wait_for_idle_gpu(baseline_mib: int | None, margin_mib: int, timeout_seconds: int = 120,
                      step_peak_mib: int | None = None, release_mib: int = GUARD_RELEASE_MIB) -> dict:
    """The GPU guard: the step before it must have given its memory back. Either card-wide used memory is back
    at the baseline plus a margin, or it has fallen at least `release_mib` below that step's own peak. The
    second rule is for the card's OTHER users: the desktop session may hold more than it did when the task
    started (ladder_l0b_r1 failed here with both attempt steps done: 485 MiB of desktop growth against a 300
    MiB margin), and that is not this task's to wait for. A model that lingers leaves the card near the peak
    and fails both rules, as before."""
    if baseline_mib is None:
        return {"ok": True, "detail": "nvidia-smi unavailable; guard skipped"}

    def passed(used: int | None) -> str | None:
        if used is None:
            return None
        if used <= baseline_mib + margin_mib:
            return "back at the baseline"
        if step_peak_mib is not None and step_peak_mib - used >= release_mib:
            return "the step's memory was released; the card's other users hold more than at the baseline"
        return None

    deadline = time.monotonic() + timeout_seconds
    used = gpu_memory_used_mib()
    while used is not None and passed(used) is None and time.monotonic() < deadline:
        time.sleep(3)
        used = gpu_memory_used_mib()
    why = passed(used)
    result = {"ok": why is not None, "used_mib": used, "baseline_mib": baseline_mib, "margin_mib": margin_mib,
              "step_peak_mib": step_peak_mib}
    if why:
        result["passed_because"] = why
    return result


def run_stage(stage: str, out: Path, kimina_api_key: str | None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "steps").mkdir(exist_ok=True)
    STORE.mkdir(parents=True, exist_ok=True)
    monitor = Monitor(out)
    monitor.start()
    baseline = gpu_memory_used_mib()
    previous_peak: int | None = None
    report: dict = {"stage": stage, "box": platform.node(), "gpu_baseline_mib": baseline, "steps": []}
    log(f"stage {stage}; store {STORE}; GPU memory in use at start: {baseline} MiB")
    try:
        uv = ensure_uv()
        failed_parts: set[int] = set()
        for entry_step in STAGES[stage]:
            environment, step, options = step_fields(entry_step)
            label = options.get("label", step)
            if part_is_blocked(options, failed_parts):
                report["steps"].append({"step": label or "gpu_guard", "skipped": f"part {options.get('part', 0)}: an earlier step it needs failed"})
                continue
            started = time.monotonic()
            monitor.reset_peaks()
            if environment == "guard":
                result = {"step": "gpu_guard", **wait_for_idle_gpu(baseline, 300, step_peak_mib=previous_peak)}
                returncode = 0 if result["ok"] else 1
                if result["ok"] and result.get("passed_because", "").startswith("the step's memory was released"):
                    baseline = result["used_mib"]       # what the card's other users hold now: the next guard's baseline
            elif environment == "sync":
                command = [str(uv), "sync", "--project", str(PROJECT_DIR), "--group", step, "--no-dev",
                           "--frozen", "--python", "3.12"]
                returncode = run_child(command, child_environment(step, None), monitor)
                result = {"step": f"sync_{step}"}
            else:
                step_out = out / "steps" / f"{label}.json"
                command = [str(environment_python(environment)), "-m", "rlvr_lean.gpu", step, "--out", str(step_out)]
                env = child_environment(environment, kimina_api_key, options)
                env["RLVR_LEAN_STEP_DIR"] = str(out / "steps")      # side files (samples) land in the output directory a job runner collects
                env["RLVR_LEAN_TB_DIR"] = str(out / "tb")           # one TensorBoard run per arm and seed (spec §13)
                returncode = run_child(command, env, monitor)
                result = {"step": label}
                if step_out.exists():
                    result["result"] = json.loads(step_out.read_text())
            result.update({"returncode": returncode, "seconds": round(time.monotonic() - started, 1),
                           "peak_gpu_used_mib": monitor.peak_gpu_mib, "peak_step_rss_mib": monitor.peak_tree_rss_mib})
            if environment not in ("guard", "sync"):
                previous_peak = monitor.peak_gpu_mib or None       # what the next guard holds this step to
            report["steps"].append(result)
            (out / f"{stage}.partial.json").write_text(json.dumps(report, indent=2))
            log(f"{result['step']}: returncode {returncode} in {result['seconds']} s")
            if returncode != 0:
                report.setdefault("failed_step", result["step"])
                report.setdefault("failed_steps", []).append(result["step"])
                if returncode == SOUNDNESS_ALARM_EXIT and environment not in ("guard", "sync"):
                    report["soundness_alarm"] = result["step"]      # not an ordinary failure: nothing may be read or rerun
                if "part" not in options:           # a stage without parts, or the setup before them: stop here
                    break
                failed_parts.add(options["part"])
    finally:
        monitor.stop()
    report["passed"] = "failed_step" not in report
    return report


def store_ca_certificate(encoded: str, store: Path) -> Path:
    """Write the Lean pool authority's certificate into the store and return its path. The OEIS Open spec, O2a item
    9b: it reaches a task the way the key does (an argument at queue time, never committed, never printed)
    and is public. Steps need a FILE (the TLS library reads one), named by its content so two pools' never
    collide. A value that is not base64 of a PEM certificate is refused without being echoed."""
    import base64
    import binascii

    try:
        pem = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("--kimina-ca-certificate must be base64 on one line (`base64 -w0 ca.crt`)") from None
    if b"-----BEGIN CERTIFICATE-----" not in pem:
        raise ValueError("--kimina-ca-certificate does not decode to a PEM certificate")
    target = store / "lean-ca" / f"{hashlib.sha256(pem).hexdigest()[:16]}.crt"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_bytes(pem)
    temporary.replace(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--profile", default="smoke", choices=["smoke", "full"], help="sizes, from config/experiment.yaml")
    parser.add_argument("--seeds", default="", help="training seeds, e.g. 0 or 0,1,2; default: the config's list")
    parser.add_argument("--arms", default="", help="selections to train and evaluate, e.g. learning_progress_cosine,random; default: Phase A's")
    parser.add_argument("--kimina-api-key", default=None, help="handed to steps as an environment variable")
    parser.add_argument("--kimina-ca-certificate", default=None, metavar="BASE64",
                        help="the Lean pool authority's certificate (its PEM file, base64 on one line: `base64 -w0 ca.crt`); "
                             "passed at queue time beside the key, written into the store, never printed")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--config", default=None, help="accepted and ignored (a job runner may append one)")
    parser.add_argument("--print-run-identity", action="store_true")
    arguments = parser.parse_args()
    if arguments.print_run_identity:
        # What a job runner may record as this run's identity; the API key is never part of it.
        print(json.dumps({"config": {"tool": "rlvr_lean.runner.entry", "stage": arguments.stage, "profile": arguments.profile,
                                     "training_seeds": arguments.seeds, "arms": arguments.arms, "seed": None,
                                     "run": {"out_dir": str(arguments.out) if arguments.out else None}}}))
        return 0
    if arguments.out is None:
        parser.error("--out is required")
    if arguments.seeds and not all(part.isdigit() for part in arguments.seeds.split(",")):
        parser.error("--seeds must look like 0 or 0,1,2")
    # Format only: the names are checked by the pipeline (`domain.training.arms`), which this stdlib-only
    # shim cannot import without numpy (the package `__init__`s pull in the selection context).
    if arguments.arms and not all(part.replace("_", "").isalpha() for part in arguments.arms.split(",")):
        parser.error("--arms must look like learning_progress_cosine,random")
    PROFILE["value"], PROFILE["seeds"], PROFILE["arms"] = arguments.profile, arguments.seeds, arguments.arms
    if arguments.kimina_ca_certificate:
        try:
            # Every step inherits this process's environment (`child_environment`).
            os.environ["RLVR_LEAN_KIMINA_CA_FILE"] = str(store_ca_certificate(arguments.kimina_ca_certificate, STORE))
        except ValueError as error:
            parser.error(str(error))
    report = run_stage(arguments.stage, arguments.out, arguments.kimina_api_key)
    (arguments.out / f"{arguments.stage}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], "failed_step": report.get("failed_step")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
