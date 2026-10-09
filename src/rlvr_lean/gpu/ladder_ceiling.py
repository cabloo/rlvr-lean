"""The ceiling: a labelled diagnostic. Spec: docs/spec/ladder-loop.spec.md, "The ceiling: a labelled diagnostic".
Each step is its own process (`rlvr_lean.runner.entry`, stage `ladder_ceiling`).

AN EXCEPTION, NOT A CHANGE OF RULE. Published proofs are certificates and not training text. This trains ONCE on them,
from the base, as a diagnostic: shown longer proofs of problems it cannot solve, does this model learn to write such
proofs in one shot? No model trained this way is used in a round or kept (the report step deletes the adapters), the
training file is not published, and every stored file, report heading and printed line says "ceiling".

  ladder_ceiling_prepare          reads, ON THE BOX and never writing there: L1's run directory of this seed (G and the
                                  rungs with their negations and groups, the base's fresh results and attempts on them)
                                  and L2's two (the base's control attempts on G; the three-round model of the arm
                                  `ceiling.loop_arm`). Checks the training file that travels with the code and
                                  REFUSES it when a row is a held-out problem (or the base map's), when it has fewer
                                  rows than the last checkpoint, or when a row would be cut as a training example.
                                  Fixes the schedule and the sampling seeds. No GPU, no Lean.
  ladder_ceiling_train            ONE pass from the base over the file's rows IN THE FILE'S ORDER, the round's recipe
                                  otherwise (`ladder_dose._train_with_readings`, the round's own example builder, the
                                  native format); every row's loss, read before its step's update; an adapter saved
                                  after each checkpoint's rows
  ladder_ceiling_measure_<name>   one checkpoint, served beside the base: 8 episodes on every problem of the three
                                  rungs and the attempts on every goal problem, with the sampling seeds L2 used for
                                  its models, so that it pairs by problem with the stored attempts
  ladder_ceiling_report           the read fixed before the run (`reporting/ladder_ceiling.py`); then, once a report
                                  THAT CAN BE READ is written, the adapters are deleted and the report says so. A
                                  report that is not to be read (Lean gave no verdict on too much of a set) keeps
                                  them and fails the step: queued again, the task measures that set again from the
                                  kept adapters, with no new training

The run directory is its own (`ladder_ceiling_seed<seed>`): nothing here can mark a step of another run done. A rerun
resumes at the first step, and inside a measurement the first block, not done; a run whose report is written is not
trained again. A verified proof on the side a certificate contradicts stops the step with the soundness alarm's exit
code, as everywhere.

Three parts of this stage are functions another stage calls (L3d Step 1, `gpu/ladder_l3d1.py`, measures its two models
as the ceiling's are): `read_stored_runs` (what L1's and L2's runs stored, refused unless it pairs), `one_pass` (one
pass from the base over examples in the order given) and `measure_model` (a model on the rungs and on G, a set Lean
did not answer measured again). Called by this stage they do what they always did.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from rlvr_lean.data.heldout_proof_lines import length_group
from rlvr_lean.domain.ladder_round.ceiling import (
    CEILING,
    CHECKPOINTS,
    LABEL,
    check_examples_fit,
    checkpoint_steps,
    doses,
    the_training_took,
    training_example,
    training_rows,
)
from rlvr_lean.domain.ladder_round.dose import run_dose
from rlvr_lean.domain.ladder_round.read import GOAL, group_ids
from rlvr_lean.domain.problem_pool.episodes import RUNGS, VERIFIED
from rlvr_lean.domain.repair.accumulate import proof_line_count
from rlvr_lean.domain.training.target_format import NATIVE
from rlvr_lean.gpu import ladder_dose, ladder_l2, ladder_loop, pipeline
from rlvr_lean.gpu.ladder_dose import _episodes, _rows, _runs
from rlvr_lean.gpu.ladder_round import BASE, Engines, _attempts, _results, _stand_in, _write_json, add_problems, distinct_attempts, training_seed
from rlvr_lean.gpu.model_utils import encode_pair, training_text
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.reporting.ladder_ceiling import lean_did_not_answer

CEILING_RUN_VARIABLE = "RLVR_LEAN_LADDER_CEILING_RUN"                    # another run directory than `ladder_ceiling_seed<seed>` (a smoke run)
CEILING_SOURCE_VARIABLE = "RLVR_LEAN_LADDER_CEILING_SOURCE"              # another L1 run directory to read than `ladder_l1_seed<seed>`
CEILING_STORED_VARIABLE = "RLVR_LEAN_LADDER_CEILING_STORED"              # `none`: L2's stored attempts are not read (a smoke run: no L2 smoke run holds them)
CEILING_TRAINING_VARIABLE = "RLVR_LEAN_LADDER_CEILING_TRAINING"          # another training file than the package's (the fixture)
CEILING_CHECKPOINTS_VARIABLE = "RLVR_LEAN_LADDER_CEILING_CHECKPOINTS"    # other rows for the two checkpoints than `ceiling.checkpoints`, as `12,24`
PACKAGE_TRAINING = Path(__file__).resolve().parents[1] / "data" / "ladder_ceiling" / "training.jsonl"
PREPARE, TRAIN, REPORT = "ladder_ceiling_prepare", "ladder_ceiling_train", "ladder_ceiling_report"
REPORT_FILE, LOSS_FILE, STORED_MODELS_FILE = "report_ladder_ceiling.json", "ceiling_loss.json", "ceiling_stored_models.json"
TRAINING_ROWS_FILE, GROUPS_FILE = "ceiling_training_rows.jsonl", "ceiling_heldout_groups.jsonl"
ADAPTERS = "ceiling_adapters"                           # under the run directory, one directory a checkpoint; deleted by the report step
LOOP = "loop"                                           # the three-round model that stands beside the ceiling's: a stored run's, never trained here
RUNG_PART, REACH, MORE = "rungs", "reach", "more"       # the three rungs; G's first sampling (L1's seed for G) and its second (L2's control's seed)
LAST_ROUND = f"m{ladder_l2.ROUNDS[-1]}"                 # how an L2 run names its three-round model's sets
# What is read where. L1's run of the seed: the base on the rungs and on G's first sampling. L2's run of the seed: the
# base's second sampling of G (its control). The arm's L2 run: its three-round model on all three.
L1_SETS = {RUNG_PART: f"rungs_{BASE}", REACH: f"reach_{BASE}"}
STORED_SETS = {BASE: {MORE: ladder_l2.CONTROL}, LOOP: {RUNG_PART: f"rungs_{LAST_ROUND}", REACH: f"reach_{LAST_ROUND}", MORE: ladder_l2.CONTROL_TRAINED}}
UNCUT = 10 ** 9                                         # a length no example reaches: `encode_pair` then cuts nothing
MAX_LEFTOVER_GB = pipeline.MAX_LEFTOVER_GB


# ------------------------------------------------------------------------------------------------- the run
def _store(config: dict) -> ArtifactStore:
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    run = os.environ.get(CEILING_RUN_VARIABLE) or f"ladder_ceiling_seed{training_seed(config)}"
    return ArtifactStore(_runs(config) / run, Path(mirror) if mirror else None)


def source_directory(config: dict) -> Path:
    """L1's run directory for this seed. It is only ever READ, with plain file reads: nothing is created in it."""
    return _runs(config) / (os.environ.get(CEILING_SOURCE_VARIABLE) or f"ladder_l1_seed{training_seed(config)}")


def stored_directories(config: dict, variable: str = CEILING_STORED_VARIABLE, setting: str = "ladder_loop.ceiling.loop_arm") -> dict[str, Path] | None:
    """L2's two run directories of this seed whose stored attempts stand beside the ceiling's: `base` (the L2 run:
    the base's control attempts on G) and `loop` (the run of the arm `ceiling.loop_arm`: its three-round model).
    None when the task says there are none to read (a smoke run). They are only ever READ. `variable` and
    `setting`: another stage's own (the environment variable that says `none`; where its arm is configured)."""
    asked = os.environ.get(variable)
    if asked:
        if asked != "none":
            raise ValueError(f"{variable} is {asked!r}: it is `none` (L2's stored attempts are not read) or it is not set")
        return None
    arm, seed = loop_arm(config, setting), training_seed(config)
    if arm not in (config["ladder_loop"].get("l2_arms") or {}):
        raise ValueError(f"{setting} is {arm!r} and ladder_loop.l2_arms has {sorted(config['ladder_loop'].get('l2_arms') or {})}: refused")
    return {BASE: _runs(config) / f"ladder_l2_seed{seed}", LOOP: _runs(config) / f"ladder_l2_{arm}_seed{seed}"}


def loop_arm(config: dict, setting: str = "ladder_loop.ceiling.loop_arm") -> str:
    """The L2 arm whose three-round model stands beside: the value of the dotted `setting`."""
    value = config
    for key in setting.split("."):
        value = value[key]
    return value


def training_file() -> Path:
    override = os.environ.get(CEILING_TRAINING_VARIABLE)
    return Path(override) if override else PACKAGE_TRAINING


def checkpoint_rows(config: dict) -> list[int]:
    override = os.environ.get(CEILING_CHECKPOINTS_VARIABLE)
    return [int(part) for part in override.split(",")] if override else list(config["ladder_loop"]["ceiling"]["checkpoints"])


def adapter_directory(store: ArtifactStore, name: str) -> Path:
    return store.root / ADAPTERS / name


def rung_set(name: str, label: str = CEILING) -> str:
    return f"{label}_{RUNG_PART}_{name}"


def goal_set(sampling: str, name: str, label: str = CEILING) -> str:
    return f"{label}_{sampling}_{name}"


def measure_marker(name: str) -> str:
    return f"ladder_ceiling_measure_{name}"


def stored_file(who: str, part: str, label: str = CEILING) -> str:
    return f"{label}_stored_{who}_{part}.jsonl"


def _say(message: str) -> None:
    print(f"{CEILING}: {message}", flush=True)


def _need(store: ArtifactStore, marker: str, what: str, stage: str = "ladder_ceiling") -> None:
    if not store.is_done(marker):
        raise RuntimeError(f"{what} needs the step {marker} of this run, which is not done: the stage `{stage}` runs the steps in order")


# --------------------------------------------------------------------------------------- what a run stored
def _missing(directory: Path, set_names) -> list[str]:
    """What `directory` lacks of the measurements `set_names` as a finished run stores them."""
    missing = [name for set_name in set_names for name in (f"episodes_{set_name}_problems.jsonl", f"episodes_{set_name}.done.json")
               if not (directory / name).exists()]
    return missing + [f"episodes_{set_name}_attempts_*.jsonl" for set_name in set_names
                      if not (directory.is_dir() and any(directory.glob(f"episodes_{set_name}_attempts_*.jsonl")))]


def _stored_attempts(directory: Path, set_name: str) -> list[dict]:
    return [row for path in sorted(directory.glob(f"episodes_{set_name}_attempts_*.jsonl")) for row in _rows(path)]


@dataclass(frozen=True)
class Reading:
    """Who reads the stored runs, for what is said when one is refused. The ceiling's own; another stage that reads
    them the same way gives its own."""
    reader: str = "The ceiling"                         # "<reader> reads the run directory ..."
    models: str = "the ceiling's checkpoints"           # "<models> would not pair with the stored results"
    not_read: str = "that set is not to be read, no report of this run could be read against it, and so its adapters could never be deleted"
    setting: str = "ladder_loop.ceiling.loop_arm"       # where the arm of the three-round model is configured


def _stored(directory: Path, set_name: str, problem_ids: list[str], seed: int, episodes: int | None, what: str,
            reading: Reading = Reading()) -> tuple[list[dict], dict]:
    """(the per-problem rows, what its step recorded) of one stored measurement, refused unless it was sampled
    with `seed` (and `episodes` episodes, when the number is this config's to say) on exactly `problem_ids`: the
    ceiling's checkpoints are sampled with them, and would not pair with it otherwise."""
    rows, recorded = _rows(directory / f"episodes_{set_name}_problems.jsonl"), json.loads((directory / f"episodes_{set_name}.done.json").read_text())
    if recorded["sampling_seed"] != seed or (episodes is not None and recorded["episodes_each"] != episodes):
        raise RuntimeError(f"{directory.name} measured {what} with sampling seed {recorded['sampling_seed']} and {recorded['episodes_each']} episodes; this config "
                           f"gives {seed}{'' if episodes is None else ' and ' + str(episodes)}: {reading.models} would not pair with the stored results")
    if sorted(row["problem_id"] for row in rows) != sorted(problem_ids):
        raise RuntimeError(f"{directory.name} measured {what} on other problems than L1's run holds for it: {reading.models} would not pair with it")
    if lean_did_not_answer(rows):
        raise RuntimeError(f"{directory.name} measured {what} with too many attempts without a verdict from Lean: {reading.not_read}. Nothing was written")
    return rows, recorded


def what_a_model_wrote(rung_attempts: list[dict], goal_attempts: list[dict]) -> dict:
    """Of one model's attempts on the rungs and on G: how many lines each proof Lean verified has (counted as the
    published proofs' are: `proof_line_count`), as a count by number of lines, and the share of distinct attempts."""
    def lines(attempts: list[dict]) -> dict[str, int]:
        counts = Counter(proof_line_count(attempt["completion"]) for attempt in attempts if attempt["status"] == VERIFIED)
        return {str(count): counts[count] for count in sorted(counts)}

    return {"verified_proof_lines": {RUNG_PART: lines(rung_attempts), GOAL: lines(goal_attempts)},
            "distinct_attempts": {RUNG_PART: distinct_attempts(rung_attempts), GOAL: distinct_attempts(goal_attempts)}}


# ------------------------------------------------------------------------------------------------ the file
def _file_rows(path: Path, what: str = "the ceiling's training file",
               how: str = "it is built on the dev machine with Lean checks and committed with the code (spec, 'The training file')") -> tuple[list[dict], str]:
    """(the rows of a file that travels with the code, in its order; its SHA-256). `what` and `how`: what the file
    is and how it is made, for the refusal when it is not there (the ceiling's training file by default)."""
    if not path.exists():
        raise RuntimeError(f"{what} {path} is not in this snapshot: {how}. Nothing was written")
    content = path.read_bytes()
    return [json.loads(line) for line in content.decode().splitlines() if line.strip()], hashlib.sha256(content).hexdigest()


def example_tokens(config: dict) -> tuple[Callable[[Mapping], int], str]:
    """(how many tokens a training example is, who counted): the example as the training step encodes it (the
    round's own builder: `training_text`, then `encode_pair` in the native format), before anything is cut. The
    stand-in has no tokenizer: four characters a token, as its own samples are counted."""
    if _stand_in():
        return (lambda example: 1 + sum(len(part) for part in training_text(example["theorem"], example["completion"])) // 4), "the stand-in: four characters a token"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(pipeline.NF4_DIR))
    return (lambda example: len(encode_pair(tokenizer, *training_text(example["theorem"], example["completion"]), UNCUT, NATIVE)[0])), "the model's tokenizer"


# ------------------------------------------------------------------------------- the stored runs, read whole
@dataclass
class StoredRuns:
    """What `read_stored_runs` read: L1's held-out groups and its problems on the rungs and on G; the sampling
    seeds and G's samplings; and by (who, part) the per-problem rows (`kept`) and what the step recorded
    (`recorded`), by who the attempts on the rungs and on G (`wrote`; G's also one list a sampling, `on_g`)."""
    groups: list[dict]
    rung_problems: list[dict]
    goal_problems: list[dict]
    seeds: dict
    samplings: list[dict]
    kept: dict = field(default_factory=dict)
    recorded: dict = field(default_factory=dict)
    wrote: dict = field(default_factory=dict)
    on_g: dict = field(default_factory=dict)


def read_stored_runs(config: dict, source: Path, stored: Mapping[str, Path] | None, reading: Reading = Reading()) -> StoredRuns:
    """What the runs before this one stored on the box, read and never written: L1's run of the seed (`source`: G
    and the rungs with their negations and groups, the base's fresh results and attempts on them) and, unless
    `stored` is None, L2's two (the base's control attempts on G; the three-round model of the arm). Refused, with
    the task to run, when one lacks a measurement, and when one was sampled otherwise than this config samples."""
    seed, measure = training_seed(config), config["ladder_loop"]["measure"]
    lacking = [name for name in ("problems.jsonl", "heldout_groups.jsonl") if not (source / name).exists()] + _missing(source, L1_SETS.values())
    if lacking:
        raise RuntimeError(
            f"{source} does not hold {lacking}. {reading.reader} reads the run directory the L1 task for seed {seed} wrote on this box (stage `ladder_l1` with "
            f"--seeds {seed}: `python -m rlvr_lean.runner.entry --stage ladder_l1 --seeds {seed}`; for the smoke run, stage `ladder_l1_smoke`). "
            "Run that task to its end first; nothing was written.")
    for who, directory in (stored or {}).items():
        lacking = _missing(directory, STORED_SETS[who].values())
        if lacking:
            stage = "ladder_l2" if who == BASE else f"ladder_l2_{loop_arm(config, reading.setting)}"
            raise RuntimeError(
                f"{directory} does not hold {lacking}. {reading.reader} reads the run directory the task of stage `{stage}` for seed {seed} wrote on this box "
                f"(`python -m rlvr_lean.runner.entry --stage {stage} --seeds {seed}`), {'the base' if who == BASE else 'its three-round model'}'s attempts on G among them. "
                "Run that task to its end first (its last sampling step is the one named here); nothing was written.")
    groups, problems = _rows(source / "heldout_groups.jsonl"), _rows(source / "problems.jsonl")
    rung_problems = [row for row in problems if row["set"] == L1_SETS[RUNG_PART]]
    goal_problems = [row for row in problems if row["set"] == L1_SETS[REACH]]
    rung_ids, goal_ids = [row["problem_id"] for row in rung_problems], [row["problem_id"] for row in goal_problems]
    seeds = ladder_l2.sampling_seeds(config)         # L1's for the rungs and for G; L2's for its control: what L2 sampled its models with
    seeds = {RUNG_PART: seeds[RUNG_PART], REACH: seeds[REACH], MORE: seeds[ladder_l2.CONTROL]}
    kept, recorded = {}, {}
    kept[BASE, RUNG_PART], recorded[BASE, RUNG_PART] = _stored(source, L1_SETS[RUNG_PART], rung_ids, seeds[RUNG_PART], measure["rung_episodes"], "the base on the rungs",
                                                               reading)
    kept[BASE, REACH], recorded[BASE, REACH] = _stored(source, L1_SETS[REACH], goal_ids, seeds[REACH], measure["reach_episodes"], "the base on G", reading)
    samplings = [{"name": REACH, "episodes": measure["reach_episodes"], "sampling_seed": seeds[REACH]}]
    on_g = {BASE: [_stored_attempts(source, L1_SETS[REACH])]}
    wrote = {BASE: {RUNG_PART: _stored_attempts(source, L1_SETS[RUNG_PART])}}
    if stored:
        # The second sampling of G is as large as the control L2 gave the base (derived there, not configured: 61).
        kept[BASE, MORE], recorded[BASE, MORE] = _stored(stored[BASE], STORED_SETS[BASE][MORE], goal_ids, seeds[MORE], None, "the base's control attempts on G", reading)
        more = recorded[BASE, MORE]["episodes_each"]
        samplings.append({"name": MORE, "episodes": more, "sampling_seed": seeds[MORE]})
        on_g[BASE].append(_stored_attempts(stored[BASE], STORED_SETS[BASE][MORE]))
        for part, ids, episodes in ((RUNG_PART, rung_ids, measure["rung_episodes"]), (REACH, goal_ids, measure["reach_episodes"]), (MORE, goal_ids, more)):
            kept[LOOP, part], recorded[LOOP, part] = _stored(stored[LOOP], STORED_SETS[LOOP][part], ids, seeds[part], episodes, f"its three-round model ({part})",
                                                             reading)
        wrote[LOOP] = {RUNG_PART: _stored_attempts(stored[LOOP], STORED_SETS[LOOP][RUNG_PART])}
        on_g[LOOP] = [_stored_attempts(stored[LOOP], STORED_SETS[LOOP][REACH]), _stored_attempts(stored[LOOP], STORED_SETS[LOOP][MORE])]
    for who, by_sampling in on_g.items():
        wrote[who][GOAL] = [attempt for attempts in by_sampling for attempt in attempts]
    return StoredRuns(groups, rung_problems, goal_problems, seeds, samplings, kept, recorded, wrote, on_g)


def write_stored_runs(store: ArtifactStore, read: StoredRuns, names: Sequence[str], label: str = CEILING, add: bool = False) -> None:
    """What every later step of the run reads from its OWN directory: the problems of each model's sets (the name
    the episode step reads; every set in it is this run's), the held-out groups, the stored per-problem rows."""
    def as_set(own: list[dict], set_name: str) -> list[dict]:
        return [{**row, "set": set_name} for row in own]

    own = [row for name in names for row in as_set(read.rung_problems, rung_set(name, label))]
    own += [row for name in names for sampling in read.samplings for row in as_set(read.goal_problems, goal_set(sampling["name"], name, label))]
    if add:         # a run directory that holds other sets already (an L2 arm's): these are put beside them
        add_problems(store, own)
    else:
        store.write_rows("problems.jsonl", own)
    store.write_rows(f"{label}_heldout_groups.jsonl", read.groups)
    for (who, part), own_rows in read.kept.items():
        store.write_rows(stored_file(who, part, label), own_rows)


def measured_sizes(config: dict, read: StoredRuns) -> dict:
    """What a prepare step records of the measurements: the sets' sizes, the seeds, the samplings of G."""
    measure = config["ladder_loop"]["measure"]
    return {"goal_set": len(read.goal_problems), "rungs": {name: len(group_ids(read.groups, name)) for name in RUNGS}, "rung_problems": len(read.rung_problems),
            "sampling_seeds": read.seeds, "rung_episodes": measure["rung_episodes"], "goal_samplings": read.samplings,
            "attempts_a_goal_problem": sum(sampling["episodes"] for sampling in read.samplings),
            "stored_measurements": {f"{who}_{part}": {key: entry.get(key) for key in ("set", "problems", "episodes_each", "sampling_seed", "attempts")}
                                    for (who, part), entry in read.recorded.items()},
            "l1_contradicted_side_setting": read.recorded[BASE, RUNG_PART].get("contradicted_side_setting"),
            "contradicted_side_setting": config["ladder_loop"]["episode"].get("contradicted_side", "all")}


# ------------------------------------------------------------------------------------------------- prepare
def ladder_ceiling_prepare(config: dict) -> dict:
    store = _store(config)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    seed, source, stored = training_seed(config), source_directory(config), stored_directories(config)
    # ---- what the runs before this one stored, on the box
    read = read_stored_runs(config, source, stored)
    groups, goal_problems, wrote = read.groups, read.goal_problems, read.wrote
    # ---- the training file, which travels with the code
    path = training_file()
    rows, sha256 = _file_rows(path)
    heldout, base_map, _ = ladder_loop.load_problems(ladder_loop.data_directory())
    schedule = checkpoint_steps(checkpoint_rows(config), config["training"]["effective_batch"])
    chosen = training_rows(rows, {row["problem_id"] for row in heldout} | {row["problem_id"] for row in groups}, {row["problem_id"] for row in base_map},
                           schedule[CHECKPOINTS[-1]]["rows"])
    count, counted_by = example_tokens(config)
    tokens = [count(training_example(row)) for row in chosen]
    check_examples_fit([row["problem_id"] for row in chosen], tokens, config["training"]["max_sequence_tokens"])
    write_stored_runs(store, read, CHECKPOINTS)         # every set of `problems.jsonl` is the ceiling's
    _write_json(store, STORED_MODELS_FILE, {"diagnostic": CEILING, **{who: what_a_model_wrote(attempts[RUNG_PART], attempts[GOAL]) for who, attempts in wrote.items()}})
    # The rows the run trains on, WITHOUT their statements and proofs: the published proofs stay in the training file.
    store.write_rows(TRAINING_ROWS_FILE, [{"row": position, "problem_id": row["problem_id"], "kind": row["kind"], "proof_lines": row["proof_lines"],
                                           "length_group": length_group(row["proof_lines"]), "tokens": tokens[position],
                                           "certificate_source": row.get("certificate_source")} for position, row in enumerate(chosen)])
    training, arm = config["training"], config["ladder_loop"]["ceiling"]["loop_arm"] if stored else None
    summary = {"diagnostic": CEILING, "label": LABEL, "seed": seed, "source_run": str(source),
               "stored_runs": {who: str(directory) for who, directory in stored.items()} if stored else None,
               "loop_arm": arm, "loop_target_rate": config["ladder_loop"]["l2_arms"][arm].get("target_rate") if arm else None, "stand_in_engine": _stand_in(),
               "training_file": str(path), "training_file_sha256": sha256, "training_file_rows": len(rows), "rows_trained_on": len(chosen),
               "rows_of_the_file_not_read": len(rows) - len(chosen), "order": "the file's: no row is moved",
               "effective_batch": training["effective_batch"], "steps": schedule[CHECKPOINTS[-1]]["step"], "checkpoints": schedule,
               "doses": doses(chosen, tokens, schedule, length_group),
               "max_sequence_tokens": training["max_sequence_tokens"], "longest_example_tokens": max(tokens), "tokens_counted_by": counted_by,
               "recipe": {"what": "the round's (`ladder_round._train`), from the base: one pass, the native format, the round's own example builder",
                          "learning_rate": training["learning_rate"], "warmup_steps": training["warmup_steps"], "effective_batch": training["effective_batch"],
                          "lora": config["lora"], "adapter_seed": seed},
               **measured_sizes(config, read)}
    store.mark_done(PREPARE, summary)
    _say(f"prepared, seed {seed}: {len(chosen)} rows of {path.name} in the file's order, checkpoints {schedule}; "
         f"{summary['attempts_a_goal_problem']} attempts on each of {len(goal_problems)} goal problems")
    return summary


# --------------------------------------------------------------------------------------------------- train
def one_pass(config: dict, examples: list[dict], steps: Mapping[str, int], seed: int, batch: int, adapters: Path, tensorboard_run: str,
             positions: bool = False, start: Path | None = None) -> tuple[dict, list[dict], list[float], list[int] | None]:
    """ONE pass from the base over `examples` IN THE ORDER GIVEN (nothing is shuffled), the round's recipe otherwise
    (`ladder_dose._train_with_readings`), an adapter saved under `adapters` after each step of `steps` (name ->
    optimizer step). Returns (what the training says of itself, every optimizer step's row, each example's mean
    loss per target token in the order trained and read BEFORE its step's update, and with `positions` the
    position of every example trained on, in the order trained: the training loop's own record). The stand-in
    trains nothing and makes the losses up. `start` (L4's arm): a stored adapter the pass goes on FROM, in the place
    of a fresh one on the base."""
    in_order = [list(range(len(examples)))]
    record = {"positions": True} if positions else {}      # asked for only by a caller that reads it: the ceiling's own call is what it was
    if start is not None:
        record["start"] = start
    if _stand_in():
        result = run_dose(len(examples), batch, 1, seed, [], steps, *ladder_dose._stand_in_calls(len(examples), [], []),
                          orders=in_order, per_example=True, **{key: value for key, value in record.items() if key != "start"})
        result["note"] = "no adapter was trained and every loss is made up: a pre-flight without a GPU"
    else:
        import torch

        pipeline._cap_torch_memory(config)              # the desktop's reserve stays free
        schedule = {"passes": 1, "reading_steps": [], "checkpoint_steps": steps}
        result = ladder_dose._train_with_readings(config, examples, [], [], schedule, seed, adapters, tensorboard_run,
                                                  orders=in_order, per_example=True, **record)
        gc.collect()
        torch.cuda.empty_cache()
        result["allocated_after_cleanup_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
    step_rows = result.pop("training_steps")
    result.pop("readings")
    row_losses = [loss for row in step_rows for loss in row.pop("example_losses")]
    trained = [position for row in step_rows for position in row.pop("example_positions")] if positions else None
    return result, step_rows, row_losses, trained


def ladder_ceiling_train(config: dict) -> dict:
    """The one training: from the base, over the rows the prepare step checked, in the file's order."""
    store = _store(config)
    if store.is_done(TRAIN):
        return store.done_summary(TRAIN)
    if store.is_done(REPORT):
        raise RuntimeError(f"{store.root} holds the ceiling's report: a run whose report is written is not trained again")
    _need(store, PREPARE, "the ceiling's training")
    prepared, seed = store.done_summary(PREPARE), training_seed(config)
    rows, sha256 = _file_rows(training_file())
    if sha256 != prepared["training_file_sha256"]:
        raise RuntimeError(f"the ceiling's training file has SHA-256 {sha256} and this run was prepared on {prepared['training_file_sha256']}: "
                           "a run trains on the file its prepare step checked")
    examples = [training_example(row) for row in rows[:prepared["rows_trained_on"]]]
    steps = {name: entry["step"] for name, entry in prepared["checkpoints"].items()}
    _say(f"training from the base on {len(examples)} published proofs in the file's order, {prepared['steps']} optimizer steps, adapters after steps {steps}")
    result, step_rows, row_losses, _ = one_pass(config, examples, steps, seed, prepared["effective_batch"], store.root / ADAPTERS, f"ladder_ceiling_seed{seed}")
    saved = [{"checkpoint": entry["checkpoint"], "step": entry["step"], "rows_seen": prepared["checkpoints"][entry["checkpoint"]]["rows_seen"]}
             for entry in result.pop("checkpoints")]
    batch = prepared["effective_batch"]
    _write_json(store, LOSS_FILE, {
        "diagnostic": CEILING, "seed": seed, "stand_in_engine": _stand_in(), "rows": len(row_losses), "steps": result["steps"], "effective_batch": batch,
        "what": "row_losses: each training row's mean loss per target token, in the file's order, read BEFORE the update of the optimizer step it was in. "
                "training_steps: every optimizer step by part of the target, read before its update, with the rows seen once it was made",
        "row_losses": row_losses, "checkpoints": saved,
        "training_steps": [{**row, "rows_seen": min(row["step"] * batch, len(row_losses))} for row in step_rows]})
    took = the_training_took(row_losses, config["ladder_loop"]["ceiling"]["loss_window_rows"])
    summary = {"diagnostic": CEILING, "seed": seed, "rows": len(examples), "passes": 1, "order": prepared["order"], "stand_in_engine": _stand_in(),
               "checkpoints": saved, "first_step_loss": step_rows[0]["mean_loss"], "last_step_loss": step_rows[-1]["mean_loss"],
               "mean_loss_over_the_first_rows": took["first"], "mean_loss_over_the_last_rows": took["last"], "rows_compared": took["rows_compared"], **result}
    store.mark_done(TRAIN, summary)         # the checkpoints are saved: keep them until the report is written
    _say(f"trained: {summary['steps']} steps over {len(examples)} rows; mean loss over the first {took['rows_compared']} rows {took['first']}, over the last {took['last']}")
    if summary.get("allocated_after_cleanup_gb", 0) > MAX_LEFTOVER_GB:
        raise RuntimeError(f"{summary['allocated_after_cleanup_gb']} GB is still allocated on the GPU after the ceiling's training was released")
    return summary


# ------------------------------------------------------------------------------------------------- measure
def forget_measurement(store: ArtifactStore, set_name: str) -> None:
    """Un-mark one set's episodes, so that the episode step samples it again (every block of it: its stored rows
    are written over as the blocks finish)."""
    for path in (store.path(f"episodes_{set_name}.done.json"), *store.root.glob(f"episodes_{set_name}_block_*.done.json")):
        path.unlink(missing_ok=True)


@dataclass(frozen=True)
class Model:
    """One trained model to measure as the ceiling's checkpoints are (`measure_model`)."""
    name: str                           # in its sets' names and its adapter's
    called: str                         # in a printed line: "the checkpoint small"
    whose: str                          # in a refusal: "the ceiling's checkpoint small"
    marker: str                         # the step this measurement is
    trained_by: str                     # the step that saved its adapter
    adapter: Path                       # its adapter's directory
    no_adapter: str                     # what is said when that directory is not there
    detail: str                         # in the line that says it is being measured: "2000 rows seen, step 250"
    summary: Mapping                    # what the step's summary says of the model, before what was measured
    stage: str = "ladder_ceiling"       # the stage, as `runner.entry` names it
    label: str = CEILING                # what its sets and printed lines are called
    also: Callable[[list[dict], list[list[dict]]], dict] | None = None      # more to say of what it wrote: (its attempts on the rungs, on G a sampling)
    rungs: bool = True                  # False (L4t's measurements at another temperature, and no other caller): G alone, no set of the rungs is sampled or named


def measure_model(config: dict, store: ArtifactStore, prepared: Mapping, one: Model) -> dict:
    """One model on the three held-out rungs and on G, with the sampling seeds the prepare step recorded (those L2
    used for its models): the same problems, the same random numbers and the same sides as the stored
    measurements of the base and of the three-round model, so it pairs by problem with them.

    A model that is measured is returned as stored, UNLESS Lean gave no verdict on too much of one of its sets
    (`lean_did_not_answer`: the report then cannot be read): that set, and no other, is measured again from the
    adapter."""
    name, label, marker = one.name, one.label, one.marker
    stored = store.done_summary(marker) if store.is_done(marker) else None
    sets = [*([rung_set(name, label)] if one.rungs else []), *(goal_set(sampling["name"], name, label) for sampling in prepared["goal_samplings"])]
    again = [set_name for set_name in sets if lean_did_not_answer(_results(store, set_name))] if stored is not None else []
    if stored is not None and not again:
        return stored
    _need(store, one.trained_by, f"the measurement of {one.whose}", one.stage)
    if _stand_in():
        adapter = None
    else:
        if not one.adapter.is_dir():
            raise RuntimeError(f"{one.adapter} is not there: {one.no_adapter}")
        from vllm.lora.request import LoRARequest

        adapter = LoRARequest(f"{one.stage}_{name}", 1, str(one.adapter))
    for set_name in again:
        print(f"{label}: {one.called}: Lean gave no verdict on too many attempts of {set_name}: it is measured again from the kept adapter", flush=True)
        forget_measurement(store, set_name)
    kit = Engines(enable_lora=True).kit(adapter)
    print(f"{label}: measuring {one.called} ({one.detail})", flush=True)
    summary = {**one.summary,
               **({RUNG_PART: _episodes(config, store, rung_set(name, label), prepared["rung_episodes"], prepared["sampling_seeds"][RUNG_PART], kit)} if one.rungs else {}),
               GOAL: {sampling["name"]: _episodes(config, store, goal_set(sampling["name"], name, label), sampling["episodes"], sampling["sampling_seed"], kit)
                      for sampling in prepared["goal_samplings"]}}
    on_rungs = _attempts(store, rung_set(name, label)) if one.rungs else []
    by_sampling = [_attempts(store, goal_set(sampling["name"], name, label)) for sampling in prepared["goal_samplings"]]
    summary.update(what_a_model_wrote(on_rungs, [attempt for attempts in by_sampling for attempt in attempts]))
    if one.also is not None:
        summary.update(one.also(on_rungs, by_sampling))
    summary["stand_in_engine"] = _stand_in()
    if stored is not None:
        summary["measured_again"] = [*stored.get("measured_again", []), *again]
    store.mark_done(marker, summary)
    return summary


def ladder_ceiling_measure(config: dict, name: str) -> dict:
    """One checkpoint on the three held-out rungs and on G (`measure_model`). A checkpoint that is measured is
    returned as stored, unless Lean gave no verdict on too much of one of its sets: the report then cannot be read
    and keeps the adapters for this, and that set, and no other, is measured again from the kept adapter."""
    store = _store(config)
    _need(store, PREPARE, f"the measurement of the ceiling's checkpoint {name}")
    prepared = store.done_summary(PREPARE)
    entry = prepared["checkpoints"][name]
    return measure_model(config, store, prepared, Model(
        name=name, called=f"the checkpoint {name}", whose=f"the ceiling's checkpoint {name}", marker=measure_marker(name), trained_by=TRAIN,
        adapter=adapter_directory(store, name),
        no_adapter="the ceiling's adapters are deleted once a report that can be read is written, and one trained again would not be the model this "
                   "run's other measurements came from. The measurement cannot be added to this run",
        detail=f"{entry['rows_seen']} rows seen, step {entry['step']}",
        summary={"diagnostic": CEILING, "checkpoint": name, "rows_seen": entry["rows_seen"], "step": entry["step"]}))


# -------------------------------------------------------------------------------------------------- report
def delete_adapters(store: ArtifactStore) -> dict:
    """No model trained this way is kept: every checkpoint's adapter directory is removed, and what was done is
    returned for the report. (The stand-in trains none, and a report built again finds them already gone.)"""
    deleted, absent = [], []
    for name in CHECKPOINTS:
        directory = adapter_directory(store, name)
        if directory.is_dir():
            shutil.rmtree(directory)
            deleted.append(name)
        else:
            absent.append(name)
    root = store.root / ADAPTERS
    if root.is_dir() and not any(root.iterdir()):
        root.rmdir()
    return {"kept": False, "what": "no model trained this way is kept: the ceiling's adapters are deleted once a report that can be read is written",
            "directory": str(root), "deleted_after_this_report_was_written": deleted, "not_there_to_delete": absent}


def keep_adapters(store: ArtifactStore, not_to_be_read: list[str]) -> dict:
    """A report that is not to be read deletes nothing: the adapters stay where they are, so that the set Lean did
    not answer can be measured again without a new training. What is kept, and why, is returned for the report."""
    there = [name for name in CHECKPOINTS if adapter_directory(store, name).is_dir()]
    return {"kept": True, "what": "the adapters were KEPT: this report is not to be read (Lean gave no verdict on too many attempts of "
                                  f"{', '.join(not_to_be_read)}). Queue the task again: a set of this run's own that Lean did not answer is measured again "
                                  "from the kept adapters, with no new training, and the first report that can be read deletes them",
            "directory": str(store.root / ADAPTERS), "kept_because_not_to_be_read": list(not_to_be_read), "there": there,
            "deleted_after_this_report_was_written": [], "not_there_to_delete": [name for name in CHECKPOINTS if name not in there]}


def ladder_ceiling_report(config: dict) -> dict:
    from rlvr_lean.gpu.ladder_l3c import proof_lengths
    from rlvr_lean.reporting.ladder_ceiling import build_ceiling_report

    store = _store(config)
    for marker in (PREPARE, TRAIN, *(measure_marker(name) for name in CHECKPOINTS)):     # every measurement's rows are stored, or nothing is written or deleted
        _need(store, marker, "the ceiling's report")
    prepared = store.done_summary(PREPARE)
    samplings, stored_models = prepared["goal_samplings"], json.loads(store.path(STORED_MODELS_FILE).read_text())
    base = {RUNG_PART: store.read_rows(stored_file(BASE, RUNG_PART)), GOAL: [store.read_rows(stored_file(BASE, sampling["name"])) for sampling in samplings],
            **stored_models[BASE]}
    models = {}
    for name in CHECKPOINTS:
        measured = store.done_summary(measure_marker(name))
        models[name] = {"rows": prepared["checkpoints"][name]["rows_seen"], RUNG_PART: _results(store, rung_set(name)),
                        GOAL: [_results(store, goal_set(sampling["name"], name)) for sampling in samplings],
                        "verified_proof_lines": measured["verified_proof_lines"], "distinct_attempts": measured["distinct_attempts"],
                        "stand_in_engine": measured["stand_in_engine"]}
    if prepared["stored_runs"]:
        models[LOOP] = {"rows": None, RUNG_PART: store.read_rows(stored_file(LOOP, RUNG_PART)),
                        GOAL: [store.read_rows(stored_file(LOOP, sampling["name"])) for sampling in samplings], **stored_models[LOOP]}
    # The length of a held-out problem's published proof is read HERE, by the report, and by nothing before it.
    report = build_ceiling_report(prepared, store.done_summary(TRAIN), json.loads(store.path(LOSS_FILE).read_text()), store.read_rows(GROUPS_FILE),
                                  proof_lengths(), base, models, config["ladder_loop"]["ceiling"], config["evaluation"])
    # A report built again (the task queued once more) finds the adapters gone: what the first readable report deleted stays on record.
    earlier = store.done_summary(REPORT).get("adapters") if store.is_done(REPORT) else None
    first = (earlier.get("deleted_when_the_report_was_first_written") or earlier.get("deleted_after_this_report_was_written")) if earlier else None
    if report["ok"]:
        report["adapters"] = {"kept": False, "what": "to be deleted once this report is written"}
        _write_json(store, REPORT_FILE, report)             # the report is written BEFORE anything is deleted
        report["adapters"] = done = delete_adapters(store)
        if first:
            done["deleted_when_the_report_was_first_written"] = first
        report["lines"].append(f"{CEILING}: no model trained this way is kept. Adapters deleted after this report was written: "
                               f"{', '.join(done['deleted_after_this_report_was_written']) or 'none'}; not there to delete: "
                               f"{', '.join(done['not_there_to_delete']) or 'none'}" + (f" (deleted when the report was first written: {', '.join(first)})" if first else ""))
    else:
        # NOT TO BE READ: nothing is deleted, and the step fails (`ok` is false), as every report here does then.
        report["adapters"] = kept = keep_adapters(store, report["not_to_be_read"])
        if first:
            kept["deleted_when_the_report_was_first_written"] = first
        report["lines"].append(f"{CEILING}: {kept['what']}. Adapters there: {', '.join(kept['there']) or 'none'}")
    _write_json(store, REPORT_FILE, report)
    # Everything a reader needs beside the report goes out again from here (a rerun is another task with an output
    # directory of its own): the per-problem results, the stored rows read beside them, the loss. Not the attempts
    # and not the per-block files.
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    for name in (LOSS_FILE, STORED_MODELS_FILE):
        store.mirror(name)
    for line in report["lines"]:
        print(line, flush=True)
    store.mark_done(REPORT, {"diagnostic": CEILING, "headline": report["headline"], "branch": report["branch"], "adapters": report["adapters"]})
    return report


def _for_checkpoint(name: str):
    def run(config: dict) -> dict:
        return ladder_ceiling_measure(config, name)
    return run


STEPS = {
    PREPARE: ladder_ceiling_prepare,
    TRAIN: ladder_ceiling_train,
    **{measure_marker(name): _for_checkpoint(name) for name in CHECKPOINTS},
    REPORT: ladder_ceiling_report,
}
