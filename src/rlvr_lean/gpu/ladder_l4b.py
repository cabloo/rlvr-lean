"""L4b: the arm again, from the larger pretrained model. Spec: docs/spec/ladder-loop.spec.md, "L4b: the arm again, from
the larger pretrained model" and its "Made exact by the build". Everything it makes is labelled pretrained on published
proofs. Each step is its own process (`rlvr_lean.runner.entry`, the stages `ladder_l4b_<rule>`).

The stage is L4's arm again (`gpu/ladder_l2.py`, `gpu/ladder_assembly.py`, `gpu/ladder_l3d2.py`: six rounds with assembly,
the twin, the two measurements) for the arm `ladder_loop.l4.again.arm`, whose `start` is a model of the check of the
adapter's rank (`pre_r64`) and whose TRAINING RULE is one setting the stage names. What is the arm's own is in those
modules, under the arm's settings; the three steps here are what a start that is not `pre` needs beside:

  ladder_l4b_map       FIRST, and in a run directory of its own (`ladder_l4_map_<start>_seed<pretraining seed>`, one
                       for every seed and rule of the arm): the start model's OWN MAP, the base map's problems attempted
                       again by it as `pre`'s map was made (`ladder_l4.the_map_of`: the same sides, attempts and
                       sampling seed, the 2% rule, block-resumable), served under a model server started for ITS rank.
                       The run that made the start model is only read. A failed arm keeps the map
  ladder_l4b_prepare   (right after the arm's prepare step, before anything is sampled) REFUSES a start model's run
                       that cannot carry the arm, a missing adapter, map or stored row; MAKES G' for the start model by
                       the pretraining's rule and reads L4's first two checks on it; copies into the arm's run what its
                       report reads of that run, of the base arm's and of the rank-16 arm's when they are on the box
  ladder_l4b_report    the read fixed before the run (`reporting/ladder_l4b.py`: L4's, with the breadth beside the
                       primary on every branch)

Every adapter is kept. A rerun resumes at the first step, batch and block not done.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from rlvr_lean.domain.ladder_round.l4 import goal_set_again, pretraining_checks
from rlvr_lean.domain.ladder_round.l4b import REHEARSE
from rlvr_lean.domain.ladder_round.read import GOAL, group_ids
from rlvr_lean.gpu import ladder_assembly, ladder_ceiling, ladder_l2, ladder_l3d2, ladder_l4, ladder_l4_rank, ladder_l4_start
from rlvr_lean.gpu.ladder_ceiling import RUNG_PART, Reading, _need, stored_file
from rlvr_lean.gpu.ladder_dose import _runs
from rlvr_lean.gpu.ladder_l3d1 import WROTE
from rlvr_lean.gpu.ladder_l4 import AGAIN_FILE, L4, LABEL, _say
from rlvr_lean.gpu.ladder_round import _stand_in, _write_json, training_seed
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.reporting.ladder_ceiling import attempts_on_g

STAGE = "ladder_l4b"
SETTING = "ladder_loop.l4.again"
MAP, PREPARE, REPORT = ladder_l4_start.MAP_STAGE_STEP, f"{STAGE}_prepare", f"{STAGE}_report"
REPORT_FILE = "report_ladder_l4b.json"
RANK_16_FILE = "l4b_rank_16.json"                       # what is kept of the rank-16 arm's stored report, or why it was not read
NEVER_SOLVED_FILE = "l4b_never_solved.jsonl"            # GIVEN, not made here: one `problem_id` a row, the goal problems nothing stored has ever solved
PLAN = ("sampling_seeds", "rung_episodes", "goal_samplings")       # how a model is measured, as a prepare step records it
READER = ladder_l4_start.Reader(reads="L4b's arm starts from the model that task kept: it reads what", carry="L4b's arm")
READING = Reading(reader="L4b", models="the arm's models", not_read="that set is the start model's side of every comparison, and no report of this run could be read against it",
                  setting=ladder_l4.SETTING)


def the_settings(config: dict) -> dict:
    """THE one place that reads `ladder_loop.l4.again`: the arm the stage runs, and the rank-16 arm whose stored figures stand beside."""
    return dict(config["ladder_loop"]["l4"]["again"])


def _waived() -> bool:
    """A smoke run's task alone: it goes on from a start model whose own report's checks failed, and records it."""
    return os.environ.get(ladder_l2.L2_START_CHECKS_VARIABLE) == "smoke"


def the_start(config: dict):
    """The start model of the stage's arm (`ladder_l2.the_start_of`), refused unless this task names that arm and its
    start is a model of the check of the adapter's rank: the arm from `pre` is the stage `ladder_l4`'s."""
    wanted, named = the_settings(config)["arm"], ladder_l2.arm_name()
    if named != wanted:
        raise RuntimeError(f"the stages `{STAGE}_*` run the arm {wanted} of ladder_loop.{ladder_l2.ASSEMBLY_ARMS} and {ladder_l2.L2_ARM_VARIABLE} names {named!r}: "
                           "their steps are run by the stage, which sets it")
    start = ladder_l2.the_start_of(config)
    if start is None or start.check is None:
        raise RuntimeError(f"the arm {wanted} starts from {'the base' if start is None else '`' + start.name + '`'}: the steps of `{STAGE}_*` are for an arm that starts from "
                           f"a model of {ladder_l4_rank.SETTING} (the arm from `pre` is run by the stage `{ladder_l4.ARM_STAGE}`)")
    return start


def _arm_store(config: dict) -> ArtifactStore:
    the_start(config)
    return ladder_l3d2._store(config)


# ----------------------------------------------------------------------------------------------- the map
def ladder_l4b_map(config: dict) -> dict:
    """The start model's OWN MAP (spec L4b, "What runs", 1): the base map's 4,000 problems attempted again by it, 8
    attempts each, the stored map's sampling seed and sides, as `pre`'s map was made (`ladder_l4.the_map_of`, the
    pretraining's own step under this model's names). In a run directory of its own: the run that made the start model
    is only read, and every seed and rule of the arm reads this one map. BEFORE anything is sampled the run that made
    the start model is read and refused as the arm's prepare step refuses it, and the model server is started with the
    START ADAPTER's rank as its largest."""
    start = the_start(config)
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    store = ArtifactStore(start.map_directory, Path(mirror) if mirror else None)
    if store.is_done(start.map_step):
        return store.done_summary(start.map_step)
    read = ladder_l4_start.read_the_start_run(start, ladder_l2.pretraining_seed(config), waived=_waived(), reader=READER)
    recipe = ladder_l4_start.recipe_recorded(read["prepared"])
    one = ladder_l4.MapOf(name=start.name, step=start.map_step, episodes=f"{L4}_map_{start.name}", file=start.map_file, request=f"{STAGE}_{start.name}",
                          kept_by=f"the run that made it ({start.directory.name}: {start.task})")
    return ladder_l4.the_map_of(ladder_l4_start.config_from(config, {"start_recipe": recipe}, start, serving=True), store, one, start.adapter,
                                also={"start_run": str(start.directory), "start_adapter": str(start.adapter), "start_recipe": recipe,
                                      "read_by": "every seed and rule of the arm that starts from this model: the map is sampled with the base map's own seed, which no task's "
                                                 "seed moves", "checks_of_the_start_run_waived_for_a_smoke_run": read["checks_waived"]})


# ----------------------------------------------------------------------------------------------- prepare
def _rank_16(config: dict, seed: int) -> dict:
    """What is kept of the rank-16 arm's stored report, when its run of this seed is on the box; why it was not read
    otherwise. Never a reason to refuse."""
    from rlvr_lean.reporting.ladder_l4b import of_the_rank_16_arm

    arm = the_settings(config)["rank_16_arm"]
    directory = _runs(config) / f"ladder_l2_{arm}_seed{seed}"
    path = directory / ladder_l4.ARM_REPORT_FILE
    if os.environ.get(ladder_l3d2.STORED_VARIABLE) == "none":
        return {"read": False, "arm": arm, "why": "the stored runs are not read (a smoke run)"}
    try:
        stored = json.loads(path.read_text())
        if not stored.get("ok", True):
            return {"read": False, "arm": arm, "run": str(directory), "why": f"{path.name} of {directory.name} is not to be read"}
        return {"read": True, "arm": arm, "run": str(directory), **of_the_rank_16_arm(stored)}
    except (OSError, ValueError, KeyError, AttributeError) as error:
        return {"read": False, "arm": arm, "run": str(directory), "why": f"{type(error).__name__}: {path} could not be read"}


def ladder_l4b_prepare(config: dict) -> dict:
    """What the arm stands on, checked BEFORE anything is sampled, for a start that is not `pre`: the run that made
    the start model (its report, its markers, its recorded rank), its adapter, its own map and its stored rows on the
    sets it was measured on, held to this run's problems. G' is MADE here for the start model (the pretraining's rule,
    from its first sampling) and L4's first two checks are read on it: it solves the pretraining's minimum of goal
    problems and the checks of its own report pass; G' holds the minimum. The arm is not run on a model that fails one."""
    store = _arm_store(config)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    _need(store, ladder_l2.PREPARE, "L4b's own prepare step", STAGE)
    seed, of_the_pretraining, start = training_seed(config), ladder_l2.pretraining_seed(config), the_start(config)
    arm_prepared, rule = store.done_summary(ladder_l2.PREPARE), ladder_l2.rule_of(config)
    read = ladder_l4_start.read_the_start_run(start, of_the_pretraining, waived=_waived(), reader=READER)
    of_start, report = read["prepared"], read["report"]
    if not _stand_in() and not start.adapter.is_dir():
        raise RuntimeError(f"{start.adapter} is not there: the adapter `{start.name}` is what every model of the arm is trained from and what attempts round 1. Run "
                           f"{start.task} again; nothing was written.")
    lacking = [name for name in (start.map_file, f"{start.map_step}.done.json") if not (start.map_directory / name).exists()]
    if lacking:
        raise RuntimeError(f"{start.map_directory} does not hold {lacking}: the arm's challenger starts from `{start.name}`'s own map. Run {start.map_task}; nothing was written.")
    parts = ladder_l4.pre_parts(of_start)
    lacking = [name for part in parts for name in (f"episodes_{ladder_l4_start.start_set(part, start)}_problems.jsonl", f"episodes_{ladder_l4_start.start_set(part, start)}.done.json")
               if not (start.directory / name).exists()]
    if lacking:
        raise RuntimeError(f"{start.directory} does not hold {lacking}: `{start.name}`'s per-problem rows on a set it was measured on, or that set's own marker. The arm's "
                           f"report is read against them; {start.task} wrote them on this box. Run that task to its end first; nothing was written.")
    # ---- the start model's stored rows pair by problem with this run's held-out problems, as its own run sampled them
    plan = {key: of_start[key] for key in PLAN}
    ids = {RUNG_PART: [row["problem_id"] for row in store.read_rows("base_rungs.jsonl")], GOAL: group_ids(store.read_rows("heldout_groups.jsonl"), GOAL)}
    measured = [(RUNG_PART, RUNG_PART, plan["rung_episodes"], plan["sampling_seeds"][RUNG_PART]),
                *((sampling["name"], GOAL, sampling["episodes"], sampling["sampling_seed"]) for sampling in plan["goal_samplings"])]
    kept = {part: ladder_ceiling._stored(start.directory, ladder_l4_start.start_set(part, start), ids[of], sampling_seed, episodes, f"`{start.name}` ({part})", READING)[0]
            for part, of, episodes, sampling_seed in measured}
    # ---- G' is the START MODEL's, made here; L4's first two checks, read on it
    first = plan["goal_samplings"][0]
    again = goal_set_again(kept[first["name"]], ids[GOAL])
    minimum_solved, minimum_again = ladder_l4.minimums(config)
    checks = pretraining_checks(attempts_on_g([kept[sampling["name"]] for sampling in plan["goal_samplings"]]), again, ids[GOAL], minimum_solved, minimum_again, start.name)
    of_its_report = {name: bool(entry.get("passes")) for name, entry in (report.get("can_this_run_see_a_win") or {}).items() if isinstance(entry, dict)}
    took = checks["the_pretraining_took"]
    checks["the_pretraining_took"] = {**took, "what": f"{took['what']}, and the checks of the report of the run that made it pass", "checks_of_its_own_report": of_its_report,
                                      "its_own_report_reads": (report.get("branch") or {}).get("name"),
                                      "passes": took["passes"] and all(of_its_report.values()) and not report.get("inconclusive")}
    failed = [name for name, check in checks.items() if not check["passes"]]
    if failed and not read["checks_waived"]:
        raise RuntimeError(f"`{start.name}` ({start.directory.name}) cannot carry the arm: its checks failed ({', '.join(failed)}). The arm is not run on it; nothing was written.")
    if rule == REHEARSE:        # the rehearsal rows are rows the start model was pretrained on: the pretraining file must be that file, before anything is sampled
        ladder_l4_start.pretrained_on(start.directory, of_start["pretraining_file_sha256"], start.name)
    # ---- from here on files are written: G', the start model's rows and what it wrote, the base arm's rows and the rank-16 arm's figures when they are on the box
    store.write_rows(AGAIN_FILE, [{"problem_id": problem_id} for problem_id in again])
    for part in parts:
        store.write_rows(stored_file(start.name, part, L4), kept[part])
    _write_json(store, ladder_l4_start.wrote_file(start.name), {"label": LABEL, **{key: read["measured"][key] for key in WROTE}})
    base_arm, of_the_base_arm, copied = ladder_l4.copy_the_base_arms_rows(config, store, seed)
    rank_16 = _rank_16(config, seed)
    _write_json(store, RANK_16_FILE, {"stage": L4, "label": LABEL, **rank_16})
    summary = {"stage": L4, "label": LABEL, "check": "L4b", "seed": seed, "arm": ladder_l2.arm_name(), "rule": rule, "pretraining_run": str(start.directory), "start": start.name,
               "start_adapter": str(start.adapter), "start_recipe": arm_prepared["start_recipe"], "pretraining_rows": of_start["rows"],
               "pretraining_file_sha256": of_start["pretraining_file_sha256"], "goal_set_again": len(again), "the_two_checks_of_the_pretraining": checks,
               "map": arm_prepared.get("map"), "map_file": arm_prepared.get("map_file"), "pre_parts": parts, "base_arm": base_arm, "base_arm_run": copied,
               "stand_in_engine": _stand_in(),
               "goal_set_again_made_here": {"what": f"G' of `{start.name}`: the goal problems it does not solve in its FIRST sampling of G ({first['episodes']} attempts), by the "
                                                    "pretraining's rule (`l4.goal_set_again`) from its stored rows; the run that made it stored none",
                                            "problems": len(again), "solved_in_the_first_sampling": len(ids[GOAL]) - len(again), "file": AGAIN_FILE},
               "start_run_read": {"branch": (report.get("branch") or {}).get("name"), "headline": report.get("headline"), "checks_failed": read["checks_failed"],
                                  "checks_waived_for_a_smoke_run": read["checks_waived"], "sampling_seeds": plan["sampling_seeds"]},
               "rank_16": {key: rank_16.get(key) for key in ("read", "arm", "run", "why", "branch")}, "rank_16_file": RANK_16_FILE, "never_solved_file": NEVER_SOLVED_FILE}
    if seed != of_the_pretraining:      # a further seed of the arm says so, as L4's does
        summary["a_further_seed"] = {
            "what": f"a further seed of the arm: nothing is pretrained or mapped again. `{start.name}`, its own map, G' and its stored rows are the same at every seed of the "
                    "arm; its rows were sampled with its own run's sampling seeds, this seed's models are sampled with its own",
            "seed": seed, "pretraining_seed": of_the_pretraining, "sampling_seeds_of_pre": plan["sampling_seeds"], "base_arm_run_looked_for": str(of_the_base_arm)}
    store.mark_done(PREPARE, summary)
    _say(f"L4b: the arm {summary['arm']} is prepared, seed {seed}: from `{start.name}` at rank {summary['start_recipe']['rank']} ({of_start['rows']} published proofs), the "
         f"training rule `{rule}`; G' holds {len(again)} goal problems; the rank-16 arm's figures "
         f"{'are read from ' + str(rank_16.get('run')) if rank_16['read'] else 'are not there (' + str(rank_16.get('why')) + ')'}")
    return summary


# ------------------------------------------------------------------------------------------------ report
def ladder_l4b_report(config: dict) -> dict:
    from rlvr_lean.reporting.ladder_l4b import build_l4b_report

    store, rounds = _arm_store(config), ladder_l2.rounds_of(config)
    _need(store, PREPARE, "L4b's report", STAGE)
    own = store.done_summary(PREPARE)
    wrote = ladder_l4_start.wrote_file(own["start"])
    read, beside = ladder_l4.read_for_the_arms_report(config, store, prepare=PREPARE, stage=STAGE, start=own["start"], wrote=wrote)
    # The goal problems nothing stored has ever solved are GIVEN (a file of ids in the run directory), never computed here.
    never = [row["problem_id"] for row in store.read_rows(NEVER_SOLVED_FILE)] if store.path(NEVER_SOLVED_FILE).exists() else None
    report = build_l4b_report(read, own, beside["trained_rows"], beside["last_set"], json.loads(store.path(RANK_16_FILE).read_text()), never)
    report["adapters"] = {"kept": True, "what": f"every model of the arm is kept, and `{own['start']}` with them (in the directory of the run that made it)",
                          "directory": str(store.root / ladder_assembly.ADAPTERS), "start_adapter": own["start_adapter"],
                          "there": sorted(path.name for path in (store.root / ladder_assembly.ADAPTERS).glob("*") if path.is_dir())}
    ladder_l4.write_the_arms_report(store, report, rounds, marker=REPORT, file=REPORT_FILE, wrote=wrote, also=(RANK_16_FILE,))
    return report


STEPS = {MAP: ladder_l4b_map, PREPARE: ladder_l4b_prepare, REPORT: ladder_l4b_report}
