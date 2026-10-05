"""The ladder loop's L3a2: one repair step after each fresh failure, then start over (no training). Spec:
docs/spec/ladder-loop.spec.md, "L3a2: one repair step after each fresh failure, then start over (no training)".
Each step is its own process (`rlvr_lean.runner.entry`, stage `ladder_l3a2`).

  ladder_l3a2_prepare   reads L1's run directory for this seed ON THE BOX (never writes there): G and the three rungs,
                        with their exact negations and their groups. Fixes the sampling seed. No GPU, no Lean.
  ladder_l3a2_attempts  the base model, ONE process and one engine. Loop 0: the blind first attempts
                        (`repair_alternate.episodes_goal` on every problem of G, `episodes_below` on every problem of
                        the below-band rung, `episodes_rungs` on every problem of the other two), on the side the
                        certificate allows. Loops 1 to `episode.loops` - 1: every episode whose first attempt failed
                        goes on in TWO arms from that same attempt, each to its first verified attempt: blind (a whole
                        proof from the plain prompt at every loop) and alternate (loops 1 and 3 are ONE repair step
                        each, from the fresh attempt just before: its proof cut at its first error, Lean's proof state
                        there as a comment; loops 2 and 4 are whole proofs from the plain prompt)
  ladder_l3a2_report    the read fixed before the run, the two "can this run see a win" checks, the branch

THE LOOPS ARE L3a's (`gpu/ladder_l3a.py`: the run, the first attempts, one loop as a unit of work, the settling, the
trimmed loops and the reading without them, the stored files, the resume). This module is what L3a2 changes, as a
`Check` (`L3A2`): its two arms, which loops of the alternate arm are repair steps, its sizes and its sampling seed, and
KNOWN COPIES: in both arms a proof whose text equals one Lean has already rejected in the same episode is not sent to
Lean. It is a failed attempt with the status `known_copy`, its tokens are counted, and a repair step that would start
from it is a blind attempt (Lean was not asked, so there is no error to cut it by). The rules and the read are
`domain/repair/alternate.py`; what is done to one failed proof is L3a's `domain/repair/cut.py`.

RANDOM NUMBERS, as in L3a: an attempt's seed is a hash of the stage's sampling seed, the problem, the side, the episode
and the loop, and NOT of the arm, and attempts whose prompt and seed are the same are ONE generation. So at loops 2
and 4, where both arms write a whole proof from the plain prompt, an episode's two arms hold the SAME sample as long as
both are open; so do they at a repair step that had no state to give. The arms then differ by what the repair steps
wrote and by nothing else. Where one arm had already seen that proof rejected and the other had not, it is a known
copy in the first and a checked attempt in the second. The stage's sampling seed is `round.sampling_seed` + 100 x the
task's seed + `SEED_PLACE`, one place after L3a's: no first attempt and no later draw is one of L3a's.

The run directory is its own (`ladder_l3a2_seed<seed>`). A verified proof on the side a certificate contradicts stops
the step with the soundness alarm's exit code, as everywhere: that side is attempted (first attempts only) for the
audited share of problems.
"""

from __future__ import annotations

import os
from pathlib import Path

from rlvr_lean.domain.ladder_round.read import GOAL
from rlvr_lean.domain.problem_pool.episodes import RUNGS
from rlvr_lean.domain.repair.alternate import ALTERNATE, ARMS, REPAIR_ARMS, is_repair_step
from rlvr_lean.gpu import ladder_l3a
from rlvr_lean.gpu.ladder_l3a import Check
from rlvr_lean.gpu.ladder_round import training_seed
from rlvr_lean.infrastructure.artifact_store import ArtifactStore

L3A2_RUN_VARIABLE = "RLVR_LEAN_LADDER_L3A2_RUN"          # another run directory than `ladder_l3a2_seed<seed>` (a smoke run)
L3A2_SOURCE_VARIABLE = "RLVR_LEAN_LADDER_L3A2_SOURCE"    # another L1 run directory to read than `ladder_l1_seed<seed>`
L3A2_EPISODES_VARIABLE = "RLVR_LEAN_LADDER_L3A2_EPISODES"    # the config section the episodes are read from, when it is not
                                                         # this check's own (a smoke run: `repair`, the sizes L3a's smoke has)
OWN_EPISODES, L3A_EPISODES = "repair_alternate", "repair"
PREPARE, ATTEMPTS, REPORT = "ladder_l3a2_prepare", "ladder_l3a2_attempts", "ladder_l3a2_report"
# The stage's sampling seed is `round.sampling_seed` + 100 x the task's seed + this place: L3a's is 30
# (`ladder_l3a.SEED_PLACE`, where the other measurements' places are listed), and no measurement has 31.
SEED_PLACE = 31


def sizes(config: dict) -> dict:
    """What a run is made with: the episodes a problem of each group gets, the attempts of an episode, the caps,
    the seed. A group with no number of its own in the section (`episodes_goal`, `episodes_below`) has the rungs'."""
    ladder = config["ladder_loop"]
    section, episode = os.environ.get(L3A2_EPISODES_VARIABLE) or OWN_EPISODES, ladder["episode"]
    if section not in (OWN_EPISODES, L3A_EPISODES):
        raise ValueError(f"{L3A2_EPISODES_VARIABLE} names the section {section!r}: the episodes are read from ladder_loop.{OWN_EPISODES} or, for a smoke "
                         f"run, from ladder_loop.{L3A_EPISODES}")
    if episode["loops"] < 2:
        raise ValueError(f"ladder_loop.episode.loops is {episode['loops']}: an episode is a first attempt and at least one more")
    settings = ladder[section]
    return {"episodes": {group: settings.get(f"episodes_{group}", settings["episodes_rungs"]) for group in (GOAL, *RUNGS)}, "episodes_from": section,
            "loops": episode["loops"], "max_new_tokens": episode["max_new_tokens"], "lean_seconds": episode["lean_seconds"],
            "sampling_seed": ladder["round"]["sampling_seed"] + 100 * training_seed(config) + SEED_PLACE}


# L3a2: two arms; the alternate arm repairs at the odd loops only, each time from the fresh attempt just before, and
# always with the state; a proof Lean already rejected in its episode is not sent again, in either arm.
L3A2 = Check(name="ladder_l3a2", title="L3a2", run_variable=L3A2_RUN_VARIABLE, source_variable=L3A2_SOURCE_VARIABLE, sizes=sizes,
             episodes=lambda own, group: own["episodes"][group],
             arms=ARMS, repair_arms=REPAIR_ARMS, repairs=lambda arm, loop: arm == ALTERNATE and is_repair_step(loop), shows_state=lambda arm: True,
             known_copies=True)


def _store(config: dict) -> ArtifactStore:
    return ladder_l3a._store(config, L3A2)


def source_directory(config: dict) -> Path:
    """L1's run directory for this seed. It is only ever READ, with plain file reads: nothing is created in it."""
    return ladder_l3a.source_directory(config, L3A2)


def ladder_l3a2_prepare(config: dict) -> dict:
    return ladder_l3a.prepare_run(config, L3A2)


def ladder_l3a2_attempts(config: dict) -> dict:
    return ladder_l3a.run_attempts(config, L3A2)


def ladder_l3a2_report(config: dict) -> dict:
    from rlvr_lean.reporting.ladder_l3a2 import build_l3a2_report

    return ladder_l3a.write_report(config, L3A2, build_l3a2_report, "report_ladder_l3a2.json")


STEPS = {
    PREPARE: ladder_l3a2_prepare,
    ATTEMPTS: ladder_l3a2_attempts,
    REPORT: ladder_l3a2_report,
}
