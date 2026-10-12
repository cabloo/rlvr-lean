"""L6b: a second round, from the best model. Does the improved model's own search yield proofs that add? Spec:
docs/spec/ladder-loop.spec.md, "L6b: a second round, from the best model" and its "Made exact by the build".
Labelled pretrained on published proofs, as L4's stages are.

B is L5f's stored averaged `published` of the seed: the best model on G's first sampling. ONE task, one seed.

  ladder_l6b_search_prepare       READS, each left file for file as it was: L5's finished search of the search's seed and
                                  every stored run its task read, by L5d's OWN first-step reading
                                  (`ladder_l5_draws.drawn_by_the_seed`); `second_full` in L4p's run as L5f reads it; L5f's
                                  finished run of the seed, as L5c reads it (`ladder_l5_cont.read_l5fs_run`: B and the four
                                  trainings it is the mean of); L5c's finished run of the seed (`read_l5cs_run`: its `again`,
                                  THE BAR); L5d's finished run of the seed, as L5f reads it (its stored `deep` set: the old
                                  own rows). SEARCH 2's problems are made by L6's OWN functions
                                  (`ladder_l6.solved_by_search_one`, `the_second_search`): L6's run is not read and M(1) is
                                  not asked for. It stores the search's problems, the old own rows, the next published rows
                                  of the seed's order (L5c's `again` set, HELD to it) and the four stored first samplings
  ladder_l6b_search               B, served at the start adapter's rank from L5f's run, makes `l6.search.attempts` attempts
                                  at each problem, by L5's own search body (`ladder_l5.search_deep`), L6's sampling place
  ladder_l6b_prepare              THE TWO SETS (`l6b.the_sets`, held by `hold_the_sets`): the new solves (one verified
                                  attempt of each problem B solved, drawn by the task's seed with `l5.solve_rows`), K, the
                                  sets and each in its four orders. FEWER NEW SOLVES THAN `l6.search.minimum`: no set is
                                  built, the twelve steps after it train and measure nothing, the search is the finding
  ladder_l6b_train_<model>_<k>    L4t's own training body FROM B (`start_from`, as L5c's `again` is trained)
  ladder_l6b_average_<model>      NO TRAINING: L5n's own function, the mean of the four
  ladder_l6b_measure_<model>      the goal set's first sampling alone (`ladder_ceiling.measure_model`, no rung)
  ladder_l6b_report               the checks first, then the search's yield, the primary and whose proofs

Nothing of any stored run is written. Every adapter made here is kept; B, its four trainings, `second_full` and L5c's
`again` are their own runs'. THE SMOKE-ONLY PATH: `RLVR_LEAN_LADDER_L6B_SMOKE_PROBLEMS` (the least number of problems the
search attempts), refused unless the run directory is one the task names and the map is the fixtures': L5's INCONCLUSIVE
smoke run is read all the same, and so is an INCONCLUSIVE report of L5c's smoke run (both RECORDED); the map's other
problems join the search; one new solve is enough; orders that are alike are trained; an averaged adapter that is one of
its models is kept.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from rlvr_lean.domain.ladder_round.dose import steps_per_pass
from rlvr_lean.domain.ladder_round.l4_rows import PRETRAINING, THE_RULE
from rlvr_lean.domain.ladder_round.l5 import DEEP, refuse_barred, solve_rows
from rlvr_lean.domain.ladder_round.l5_cont import fixed_seeds
from rlvr_lean.domain.ladder_round.l5_draws import MEAN_CONSTANT, refuse_problems_of, trained_of
from rlvr_lean.domain.ladder_round.l5_noise import ORDERS, numbered
from rlvr_lean.domain.ladder_round.l6 import PART_A, PART_B, enough_round_two_solves, second_search, second_search_table
from rlvr_lean.domain.ladder_round.l6b import (
    BAR,
    MODELS,
    NEW,
    NEW_OWN,
    OLD_OWN,
    START,
    TRAINED,
    hold_the_next,
    hold_the_setting,
    hold_the_sets,
    in_their_orders,
    of_l5c,
    published_next,
    the_dose,
    the_seed,
    the_sets,
    the_yield,
)
from rlvr_lean.gpu import (
    ladder_assembly,
    ladder_ceiling,
    ladder_l4,
    ladder_l4_rows,
    ladder_l4_start,
    ladder_l5,
    ladder_l5_cont,
    ladder_l5_draws,
    ladder_l5_full,
    ladder_l5_noise,
    ladder_l6,
    ladder_loop,
)
from rlvr_lean.gpu.ladder_ceiling import Model, Reading, _need, measure_model, stored_file
from rlvr_lean.gpu.ladder_dose import _rows, _runs
from rlvr_lean.gpu.ladder_l3d1 import what_else_a_model_wrote
from rlvr_lean.gpu.ladder_l4 import L4, LABEL, _say
from rlvr_lean.gpu.ladder_l4_rows import described, lines_counted, train_a_model
from rlvr_lean.gpu.ladder_l4_start import Start, config_from
from rlvr_lean.gpu.ladder_l5 import block_of
from rlvr_lean.gpu.ladder_l5_draws import the_set
from rlvr_lean.gpu.ladder_round import _attempts, _results, _stand_in, _write_json, training_seed
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.reporting.ladder_ceiling import verified_lines

L6B = "l6b"                                                 # what this stage's own files are called
STAGE = ENTRY_STAGE = "ladder_l6b"                          # what its steps begin with; the stage, as `runner.entry` names it; its run directory begins with it
RUN_VARIABLE = "RLVR_LEAN_LADDER_L6B_RUN"                   # another run directory than `ladder_l6b_<start>_seed<seed>` (a smoke run)
SMOKE_VARIABLE = "RLVR_LEAN_LADDER_L6B_SMOKE_PROBLEMS"      # a SMOKE run alone (refused elsewhere): the search attempts at least this many problems, and what else the module's note says
SETTING = "ladder_loop.l6b"
THE_STAGE = ladder_l4_rows.Stage(name=ENTRY_STAGE, steps=STAGE, files=L6B, reader="L6b", start_said_by=f"{SETTING}.start")
SEARCH_PREPARE, SEARCH, PREPARE, REPORT = f"{STAGE}_search_prepare", f"{STAGE}_search", THE_STAGE.prepare, f"{STAGE}_report"
SEARCH_SET = f"{L6B}_search"                                # the set the search's episodes are stored under
THE_SEARCH = ladder_l5.Search(marker=SEARCH, set_name=SEARCH_SET, stage=THE_STAGE, said="search")
REPORT_FILE = "report_ladder_l6b.json"
SEARCH_PROBLEMS_FILE = f"{L6B}_search_problems.jsonl"       # the search's problems, (a) then (b), each with its part, the map's k and why it was taken
NEW_SOLVES_FILE = f"{L6B}_new_solves.jsonl"                 # every new solve: its row (the model's own proof), its part, the place of its first success, whether a set holds it
OLD_SOLVES_FILE = f"{L6B}_old_solves.jsonl"                 # the own rows of L5d's `deep` set of the seed, in its stored order (the model's own proofs)
NEXT_FILE = f"{L6B}_published_next.jsonl"                   # the NEXT rows of the seed's order of the pretraining file, in the draw's order: by id and place, NEVER with text
THE_BEST = f"l5f_{START}"                                   # B, as this run names the copy of its stored first sampling (L5c's name for it)
THE_BAR = f"l5c_{BAR}"                                      # L5c's `again`, likewise
OF_L5C = ladder_l5_cont.THE_STAGE                           # the stage whose stored run of the seed holds the bar, as it names its own files and markers
OF_L5D = ladder_l5_full.OF_L5D                              # ... and the one whose stored `deep` set holds the old own rows
OF_THIS_STAGE = ladder_l5_draws.OfASeed(who="L6b", run_variable=RUN_VARIABLE, smoke_variable=SMOKE_VARIABLE)      # what L5d's first-step reading says of this stage when it refuses
READING = Reading(reader="L6b", models="L6b's models", not_read="that set stands on the other side of a comparison, and no report of this run could be read against it",
                  setting=ladder_l4.SETTING)
# What the sets' step carries on from the step that read the runs: what L4t's training body, the measurement and the report read of a prepare step.
CARRIED = ("stage", "label", "check", "seed", "start", "start_adapter", "start_run", "first_start", "second_start", "models", "effective_batch", "the_seeds", "the_start_model",
           "the_best", "l5cs_run", "l5ds_run", "round_one", "the_map", "map_run", "the_search", "pretraining_file", "pretraining_file_sha256", "recipe", "the_sampling",
           "goal_samplings", "goal_set", "goal_set_again", "attempts_a_goal_problem", "smoke", "contradicted_side_setting", "start_recipe", "start_run_read")


@dataclass(frozen=True)
class Round:
    """`ladder_loop.l6b`, as the stage's steps read it."""
    start: str                          # `published`: L5f's stored averaged `published` of the seed, B
    bar: str                            # `again`: L5c's stored model of the seed
    seeds: tuple[int, ...]              # the one seed: one task
    cont: ladder_l5_cont.Cont           # L5c's settings as ITS task reads them: its run of the seed holds the bar, and it finds L5f's run
    full: ladder_l5_full.Full           # L5f's settings as ITS task reads them: `second_full`, and L5d's run of the seed
    draws: ladder_l5_draws.Draws        # L5d's settings as ITS task reads them: the first start, the sampling, its attempts and the recipe
    deep: ladder_l5.Deep                # L5's settings as ITS task reads them: the search's seed
    cycle: ladder_l6.Cycle              # L6's settings as ITS task reads them: the search's attempts, its sampling place, the minimum of new solves


def the_settings(config: dict) -> Round:
    """THE one place that reads `ladder_loop.l6b`. Refused (ValueError): what `l6b.hold_the_setting` refuses (another
    key, another start or bar, no seed, a seed that is none of L5c's); L5c's and L6's own settings, when their readers
    refuse them."""
    settings = config["ladder_loop"]["l6b"]
    cont, cycle = ladder_l5_cont.the_settings(config), ladder_l6.the_settings(config)
    seeds = hold_the_setting(settings, fixed_seeds(cont.seeds, cont.more, cont.escalated))
    return Round(start=settings["start"], bar=settings["bar"], seeds=seeds, cont=cont, full=cont.full, draws=cont.draws, deep=cont.deep, cycle=cycle)


def run_name(round_: Round, seed: int) -> str:
    """The stage's run directory, its own: `ladder_l6b_published_seed<N>`."""
    return f"{ENTRY_STAGE}_{round_.start}_seed{seed}"


def own_run(config: dict, round_: Round) -> str:
    """This task's run directory: the seed's own, or the one the task names (a smoke run)."""
    return os.environ.get(RUN_VARIABLE) or run_name(round_, training_seed(config))


def _store(config: dict, round_: Round) -> ArtifactStore:
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    return ArtifactStore(_runs(config) / own_run(config, round_), Path(mirror) if mirror else None)


def smoke_problems() -> int:
    """A smoke run's least number of problems in the search; 0 for every other run: the rule and nothing else."""
    return max(0, int(os.environ.get(SMOKE_VARIABLE) or 0))


def the_best(config: dict, round_: Round) -> Start:
    """B AS A START MODEL: L5f's stored averaged `published` of the seed's run, as L5c hands it to `again`'s trainings
    (`ladder_l5_cont.the_bar`: `pre_r64`'s rank, alpha and check, with the directory and adapter of L5f's run). It makes
    the search, and both models are trained from it. Only read."""
    return ladder_l5_cont.the_bar(config, round_.cont)


def average_marker(model: str) -> str:
    """The step that makes a model's averaged adapter, and its marker."""
    return f"{STAGE}_average_{model}"


def the_reader(round_: Round, smoke: bool) -> ladder_l5.OfTheSearch:
    """Who reads L5's finished run here, for what L5's own reading says when it refuses it. A SMOKE run alone reads a run
    whose report is INCONCLUSIVE (one not to be read never is)."""
    return ladder_l5.OfTheSearch(
        who="L6b", why=f"L6b makes no first round: round 1 is L5's finished search of seed {round_.deep.seed_of_the_search}, whose problems, solves and held-aside set "
                       "its own search stands on",
        stands="no second round stands on such a run", models="L6b's models are trained on rows of a search that stands on that run's", inconclusive=smoke)


def task_of(stage: str, seed: int) -> str:
    """The task that writes a stage's run of a seed, for a refusal."""
    return f"the task of stage `{stage}` for seed {seed} (`python -m rlvr_lean.runner.entry --stage {stage} --seeds {seed}`; for the smoke run, stage `{stage}_smoke`)"


# ------------------------------------------------------------------------------------------- L5c's run of the seed
def l5c_directory(config: dict, round_: Round) -> Path:
    """L5c's run of this task's seed, by L5c's OWN naming (`ladder_l5_cont.own_run`: its seed's own directory; in a smoke
    run, the one L5c's run variable names). Only read."""
    return _runs(config) / ladder_l5_cont.own_run(config, round_.cont)


def read_l5c(config: dict, round_: Round) -> dict:
    """L5c's run of this task's seed, ONLY READ: what its report says of its own seed (`of_this_seed`) WHEN ITS RUN IS ON
    THE BOX AND ITS REPORT WRITTEN, NOT THERE otherwise, with why. Nothing is refused here."""
    directory = l5c_directory(config, round_)
    lacking = [name for name in (f"{ladder_l5_cont.REPORT}.done.json", ladder_l5_cont.REPORT_FILE) if not (directory / name).exists()]
    if lacking:
        return {"there": False, "run": str(directory), "why": f"{directory.name} does not hold {lacking}: L5c's run of this seed is not on the box, or not finished"}
    return {**(json.loads((directory / ladder_l5_cont.REPORT_FILE).read_text()).get("of_this_seed") or {}), "there": True, "run": str(directory)}


def l5cs_again(directory: Path) -> list[str] | None:
    """The ids of the set L5c's run in `directory` STORED as `again`, in its order; None where it holds none. Only read."""
    path = directory / OF_L5C.training_file(BAR)
    return [row["id"] for row in _rows(path)] if path.exists() else None


def read_l5cs_run(config: dict, round_: Round, seed: int, measurement: dict, sampling: dict, goal_ids: list[str], best: Start, smoke: bool) -> dict:
    """L5c's FINISHED run of this task's seed, ONLY READ. REFUSED (RuntimeError), naming the task to run: a run that is
    not there or not finished, of another seed, start or recipe, not to be read, INCONCLUSIVE (a SMOKE run reads such a
    report all the same, and that is recorded) or measured otherwise than this run (`l6b.of_l5c`); one without the
    markers of its first step, of `again`'s four trainings, of its averaging and of its measurement, its stored `again`
    set, or `again`'s rows on the first sampling; one whose `again` was not trained from B, the adapter this stage
    starts from; a stored first sampling that would not pair. Returns its directory, what its report says of its seed,
    its `again` set's ids as stored, `again`'s rows on the first sampling and what its measuring step wrote."""
    entry, task = read_l5c(config, round_), task_of(ladder_l5_cont.ENTRY_STAGE, seed)
    why = of_l5c(entry, seed, MEAN_CONSTANT, measurement, though_inconclusive=smoke)
    if why:
        raise RuntimeError(f"L5c's run of seed {seed} cannot carry L6b: {why}. Its stored `{BAR}` (from the same model, the next published rows of the same order, the same "
                           f"recipe) is the bar of the primary read. Run {task} to its end first; nothing was written.")
    directory, its_set = Path(entry["run"]), the_set(BAR, sampling["name"])
    members = [trained_of(BAR, order) for order in ORDERS]
    needed = [*(f"{marker}.done.json" for marker in (OF_L5C.prepare, *(OF_L5C.train_marker(name) for name in members), ladder_l5_cont.average_marker(BAR),
                                                      OF_L5C.measure_marker(BAR))),
              OF_L5C.training_file(BAR), f"episodes_{its_set}_problems.jsonl", f"episodes_{its_set}.done.json"]
    lacking = [name for name in needed if not (directory / name).exists()]
    if lacking:
        raise RuntimeError(f"{directory} does not hold {lacking}: the trainings of `{BAR}` L5c's run of seed {seed} made, its averaged adapter's record, its stored set or its "
                           f"rows on `{sampling['name']}`. Run {task} to its end first; nothing was written.")
    done = lambda marker: json.loads((directory / f"{marker}.done.json").read_text())      # noqa: E731
    trained_from = ((done(OF_L5C.prepare).get("the_models") or {}).get(BAR) or {}).get("from")
    if trained_from != str(best.adapter):
        raise RuntimeError(f"L5c's run {directory.name} trained `{BAR}` from {trained_from} and this stage starts from {best.adapter}: the bar is a model trained from the "
                           f"SAME model as `{NEW_OWN}` and `{OLD_OWN}`, or the primary reads two starts. Nothing was written.")
    try:
        rows, _ = ladder_ceiling._stored(directory, its_set, goal_ids, sampling["sampling_seed"], sampling["episodes"], f"`{BAR}` ({sampling['name']})", READING)
    except RuntimeError as error:
        raise RuntimeError(f"{error} (L5c's run of seed {seed}: its stored `{BAR}` is the bar, paired by problem with this run's models). Nothing was written.") from error
    return {"directory": directory, "entry": entry, "stored": l5cs_again(directory), "rows": rows, "trained_from": trained_from,
            "wrote": {"distinct_attempts": done(OF_L5C.measure_marker(BAR)).get("distinct_attempts")}}


def l5ds_old(directory: Path) -> list[dict] | None:
    """The OWN rows of the `deep` set L5d's run in `directory` stores, in its stored order (the start model's own proofs,
    with their text); None where it holds no such set. Only read."""
    path = directory / OF_L5D.training_file(DEEP)
    return [row for row in _rows(path) if row["origin"] != PRETRAINING] if path.exists() else None


# -------------------------------------------------------------------------------------------- search_prepare
def ladder_l6b_search_prepare(config: dict) -> dict:
    round_ = the_settings(config)
    store = _store(config, round_)
    if store.is_done(SEARCH_PREPARE):
        return store.done_summary(SEARCH_PREPARE)
    seed, cont, full, draws, deep, cycle, at_least = training_seed(config), round_.cont, round_.full, round_.draws, round_.deep, round_.cycle, smoke_problems()
    smoke = bool(at_least)
    try:
        the_seed(seed, round_.seeds)
    except ValueError as error:
        raise RuntimeError(f"{error} ({SETTING}). Nothing was written.") from error
    if smoke and not os.environ.get(RUN_VARIABLE):
        raise RuntimeError(f"{SMOKE_VARIABLE} is set ({at_least}): it is for a SMOKE run alone, in a run directory the task names ({RUN_VARIABLE}) and on the fixtures' "
                           "base map. A run in the seed's own directory reads a run of L5 whose checks passed and takes its problems by the rule and by nothing else. Nothing "
                           "was written.")
    # ---- L5's finished run and every stored run its task read, how a model is measured and trained, the rows the first start was pretrained on, L5d's sets at this seed: by
    # L5d's OWN first-step reading, with the averaged recipe's hold of the first start adapter's type (refused there, with nothing written)
    drawn = ladder_l5_draws.drawn_by_the_seed(config, draws, seed, smoke, the_reader(round_, smoke), True, OF_THIS_STAGE)
    found, read, first, the_map, directory, theirs, sampling, training, batch, recipe, sha256, pretraining, aside = (drawn[key] for key in (
        "found", "read", "start", "the_map", "directory", "theirs", "sampling", "training", "batch", "recipe", "sha256", "pretraining", "aside"))
    goal_ids = [row["problem_id"] for row in read["goal_problems"]]
    measurement = {"sampling": sampling["name"], "episodes": sampling["episodes"], "sampling_seed": sampling["sampling_seed"], "problems": len(goal_ids)}

    # ---- `second_full`, as L5f reads it; B in L5f's finished run of the seed, as L5c reads it; the bar in L5c's; the old own rows in L5d's (each refused here, nothing written)
    of_the_start = ladder_l5_full.read_second_full(config, full, read, sampling)
    if of_the_start["summary"]["recorded_by_l4p"]["rank"] != theirs["start_recipe"]["rank"] or of_the_start["summary"]["recorded_by_l4p"]["alpha"] != theirs["start_recipe"]["alpha"]:
        raise RuntimeError(f"L4p's run recorded rank {of_the_start['summary']['recorded_by_l4p']['rank']} and alpha {of_the_start['summary']['recorded_by_l4p']['alpha']} and L5's "
                           f"run {directory.name} trained its models at {theirs['start_recipe']}: every model here is at the rank L5f's are. Nothing was written.")
    try:
        of_l5f = ladder_l5_cont.read_l5fs_run(config, cont, seed, measurement, sampling["name"])
    except RuntimeError as error:
        raise RuntimeError(f"L6b reads L5f's run of seed {seed} as L5c does (its stored averaged `{START}` is B: it makes the search and both models are trained from it), "
                           f"and that reading refuses it: {error}") from error
    best = the_best(config, round_)
    of_l5c_ = read_l5cs_run(config, round_, seed, measurement, sampling, goal_ids, best, smoke)
    try:
        of_l5d = ladder_l5_full.read_l5ds_run(config, full, seed, measurement)
    except RuntimeError as error:
        raise RuntimeError(f"L6b reads L5d's run of seed {seed} as L5f does (its stored `{DEEP}` set holds the old own rows of `{OLD_OWN}`), and that reading refuses "
                           f"it: {error}") from error
    their_run, bars_run, old_run = of_l5f["directory"], of_l5c_["directory"], of_l5d["directory"]
    old_set = drawn["sets"][DEEP]
    if [row["id"] for row in old_set] != of_l5d["stored"][DEEP]:
        raise RuntimeError(f"L5d's rule for seed {seed} gives a `{DEEP}` set of {len(old_set)} rows and its run {old_run.name} stores {len(of_l5d['stored'][DEEP] or [])}, or "
                           f"not the same ids in the same order: the old own rows of `{OLD_OWN}` are the first of the ones that run trained. Nothing was written.")
    old = [row for row in old_set if row["origin"] != PRETRAINING]
    size = len(of_l5f["stored"][START] or [])
    try:
        next_rows = published_next(pretraining, seed, size)
        held = hold_the_next(next_rows, seed, of_l5c_["stored"])
    except ValueError as error:
        raise RuntimeError(f"{error} (seed {seed}; L5f's run {their_run.name}; L5c's run {bars_run.name}). Nothing was written.") from error

    # ---- the search's problems: L6's SEARCH 2 by L6's own functions (what search 1 left unsolved, and the candidates neither draw took), with L6's sampling seed
    solved = ladder_l6.solved_by_search_one(found)
    second = ladder_l6.the_second_search(config, cycle, found, read, solved, at_least)
    searched, partition, search_seed = second["problems"], second["partition"], second["sampling_seed"]
    by_part = {part: sum(row["part"] == part for row in searched) for part in (PART_A, PART_B)}
    second_start = ladder_l5_cont.the_start(config, cont)

    # ---- from here on files are written: the problems of every set this run samples, the goal set's groups, G', the held-aside problems, the four stored first samplings,
    # the search's problems, the old own rows, the next published rows
    by_id = {row["problem_id"]: row for row in the_map["problems"]}
    own = [{**row, "set": the_set(model, sampling["name"])} for model in MODELS for row in read["goal_problems"]]
    own += [{**by_id[row["problem_id"]], "set": SEARCH_SET} for row in searched]
    store.write_rows("problems.jsonl", own)
    store.write_rows(ladder_l4.GROUPS_FILE, read["groups"])
    store.write_rows(ladder_l4.AGAIN_FILE, [{"problem_id": problem_id} for problem_id in read["again"]])
    store.write_rows(ladder_l5.HELD_ASIDE_FILE, read["aside"])
    store.write_rows(stored_file(draws.start, sampling["name"], L4), read["kept"][draws.start, sampling["name"]])
    store.write_rows(stored_file(cont.start, sampling["name"], L4), of_the_start["rows"])
    store.write_rows(stored_file(THE_BEST, sampling["name"], L4), of_l5f["rows"])
    store.write_rows(stored_file(THE_BAR, sampling["name"], L4), of_l5c_["rows"])
    store.write_rows(SEARCH_PROBLEMS_FILE, searched)
    store.write_rows(OLD_SOLVES_FILE, old)
    store.write_rows(NEXT_FILE, next_rows)
    report, l5cs = found["report"], of_l5c_["entry"]
    summary = {
        "stage": L4, "label": LABEL, "check": "L6b", "seed": seed, "start": round_.start, "start_adapter": str(best.adapter), "start_run": str(their_run),
        "first_start": draws.start, "second_start": cont.start, "models": list(MODELS), "stand_in_engine": _stand_in(), "effective_batch": batch,
        "the_seeds": {"what": "one seed, one task: a further seed needs L5f's averaged model and L5c's bar at that seed, which no run holds", "seeds": list(round_.seeds)},
        "the_start_model": of_the_start["summary"],
        "the_best": {"what": f"B: L5f's stored averaged `{START}` of this seed's run, ONLY READ, as L5c reads it: the mean of its four stored trainings, at the rank, not "
                             f"`{cont.start}`'s. It makes the search, and `{NEW_OWN}` and `{OLD_OWN}` are trained from it. Held before anything was written",
                     "run": str(their_run), "task": ladder_l5_cont.l5f_task(seed), "adapter": str(best.adapter), "held": of_l5f["held"],
                     "rows_of_its_set": size,
                     "at_the_first_step": {key: of_l5f["entry"].get(key) for key in ("seed", "start", "recipe", "ok", "inconclusive", "failed_checks", "the_measurement")},
                     "wrote": {"distinct_attempts": of_l5f["measured"].get("distinct_attempts")}},
        "l5cs_run": {"what": f"L5c's FINISHED run of this seed, ONLY READ: its stored `{BAR}` first sampling is THE BAR of the primary read (from the same B, the next "
                             "published rows of the same order, the same recipe). Its adapters are not read. Held before anything was written",
                     "run": str(bars_run), "task": task_of(ladder_l5_cont.ENTRY_STAGE, seed), "trained_from": of_l5c_["trained_from"], "the_next_rows": held,
                     "at_the_first_step": {key: l5cs.get(key) for key in ("seed", "start", "recipe", "ok", "inconclusive", "failed_checks", "the_measurement")},
                     "read_though_inconclusive": bool(l5cs.get("inconclusive")), "wrote": of_l5c_["wrote"]},
        "l5ds_run": {"what": f"L5d's FINISHED run of this seed, ONLY READ, as L5f reads it: the own rows of its stored `{DEEP}` set, in its stored order, are the old own "
                             f"rows (`pre_r64`'s deep solves); `{OLD_OWN}` takes the first K", "run": str(old_run), "old_own_rows": len(old), "file": OLD_SOLVES_FILE},
        "round_one": {"what": f"ROUND 1 IS L5's, only read: the finished run of seed {deep.seed_of_the_search}, by L5d's own first-step reading. Its search's problems, the "
                              "problems it solved and its held-aside set are what the search here stands on. Nothing of it is made again, and nothing is written there. "
                              "L6's own run is NOT read: only its functions are called", "run": str(directory), "seed": deep.seed_of_the_search,
                      "headline": report.get("headline"), "inconclusive": bool(report.get("inconclusive")), "failed_checks": list(report.get("failed_checks") or []),
                      "attempted": len(read["drawn"]), "solved": len(solved), "sampling_seed": read["sampling"]},
        "the_map": theirs["the_map"], "map_run": str(the_map["directory"]),
        "the_search": {"what": f"B (L5f's averaged `{START}`), served at the start adapter's rank from L5f's run, makes {cycle.attempts} attempts at each problem of L6's "
                               "SEARCH 2, made by L6's own functions: (a) the problems search 1 attempted and left unsolved, (b) the candidates neither search 1's draw nor "
                               "the held-aside set took; the round's episode in every other respect, at L6's sampling place. The held-aside set is not searched",
                       "set": SEARCH_SET, "file": SEARCH_PROBLEMS_FILE, "model": round_.start, "adapter": str(best.adapter), "problems": len(searched), "by_part": by_part,
                       "partition": partition, "episodes_each": cycle.attempts, "sampling_seed": search_seed, "sampling_place": cycle.sampling_place,
                       "seeds_of_the_stored_samplings_it_is_none_of": second["used"], "block_problems": block_of(config, cycle.attempts),
                       "taken_by_the_rule": sum(row["taken_by"] == THE_RULE for row in searched), "taken_by_the_runs_minimum": sum(row["taken_by"] != THE_RULE for row in searched),
                       "problems_at_least": at_least, "seed_of_the_search": cycle.seed_of_the_search, "minimum_new_solves": 1 if smoke else cycle.minimum},
        "pretraining_file": str(ladder_l4.pretraining_file()), "pretraining_file_sha256": sha256,
        "recipe": {"what": "L4t's (`ladder_ceiling.one_pass`, as the arm's twin is trained): FROM B's ADAPTER, one pass, the native format, the recipe as it is (a constant "
                           "learning rate after the warm-up). It is held to the recipe L5's run trained its models with. Each set is trained four times, in the orders of "
                           "the order seeds; the model measured is the adapter whose every number is the mean of the four (L5n's own function)",
                   "name": MEAN_CONSTANT, **recipe, "adapter_seed": seed, "order_seeds": list(ORDERS), "warmup_steps": training["warmup_steps"]},
        "the_sampling": {"what": f"each model is measured on the goal set's FIRST sampling alone, as the run of `{draws.start}` recorded it and as L5f's and L5c's models "
                                 "were measured: its attempts a problem and its sampling seed, so that each pairs by problem with every stored model's first sampling. No "
                                 "rung, no second sampling, no held-aside set", **sampling, "sets": {model: the_set(model, sampling["name"]) for model in MODELS},
                         "stored_best": stored_file(THE_BEST, sampling["name"], L4), "stored_bar": stored_file(THE_BAR, sampling["name"], L4),
                         "stored_second_start": stored_file(cont.start, sampling["name"], L4), "stored_first_start": stored_file(draws.start, sampling["name"], L4)},
        "goal_samplings": [sampling], "goal_set": len(goal_ids), "goal_set_again": len(read["again"]), "attempts_a_goal_problem": sampling["episodes"],
        "smoke": {"what": f"{SMOKE_VARIABLE}: a smoke run alone. L5's smoke run reads INCONCLUSIVE (a training of two rows has no loss to compare), which a real run is "
                          "refused for, and is read all the same; so is a report of L5c's smoke run that reads INCONCLUSIVE; the search attempts at least this many "
                          "problems, the map's other problems that no search took and that are not held aside joining when the rule gives fewer (each marked as NOT the "
                          "rule's); one new solve is enough; orders of a set's handful of rows that are alike are trained all the same; and an averaged adapter that is "
                          "one of its models is kept all the same, and said", "problems_at_least": at_least,
                  "l5_read_though_inconclusive": bool(report.get("inconclusive")), "l5s_failed_checks": list(report.get("failed_checks") or []),
                  "l5c_read_though_inconclusive": bool(l5cs.get("inconclusive")), "l5cs_failed_checks": list(l5cs.get("failed_checks") or [])}
        if smoke else None,
        "contradicted_side_setting": config["ladder_loop"]["episode"].get("contradicted_side", "all"),
        "start_recipe": theirs["start_recipe"], "start_run_read": theirs["start_run_read"],
        "second_start_adapter": str(second_start.adapter)}
    store.mark_done(SEARCH_PREPARE, summary)
    _say(f"L6b's search prepared, seed {seed}, from B ({best.adapter}; L5f's run {their_run.name}), the bar in L5c's run {bars_run.name}, round 1 in {directory.name} "
         f"({len(solved):,} of {len(read['drawn']):,} problems solved; {len(aside):,} held aside): the search attempts {len(searched):,} problems, {by_part[PART_A]:,} search 1 "
         f"left unsolved and {by_part[PART_B]:,} of the {partition['candidates']:,} candidates in neither draw"
         + (f" ({summary['the_search']['taken_by_the_runs_minimum']} of them NOT by the rule: a smoke run)" if smoke else "")
         + f", {cycle.attempts} attempts each, sampling seed {search_seed}; {len(old):,} old own rows; sets of {size:,} rows")
    return summary


# -------------------------------------------------------------------------------------------- the search
def ladder_l6b_search(config: dict) -> dict:
    """THE SEARCH, by the deep search's own body (`ladder_l5.search_deep`): B, from its adapter in L5f's run."""
    round_ = the_settings(config)
    store = _store(config, round_)
    if store.is_done(SEARCH):
        return store.done_summary(SEARCH)
    _need(store, SEARCH_PREPARE, "the search", ENTRY_STAGE)
    prepared, cycle, best = store.done_summary(SEARCH_PREPARE), round_.cycle, the_best(config, round_)
    if prepared["the_search"]["adapter"] != str(best.adapter):
        raise RuntimeError(f"this run was prepared to search with {prepared['the_search']['adapter']} and B is now {best.adapter}: a search is made as it was prepared. "
                           "Nothing was sampled")
    return ladder_l5.search_deep(config, store, prepared, best, prepared["the_search"], cycle.attempts, ladder_l6.search_seed(config, cycle), round_.start, best.adapter,
                                 THE_SEARCH)


# ------------------------------------------------------------------------------------------ the two sets
def the_new_solves(store: ArtifactStore, seed: int) -> list[dict]:
    """B's NEW SOLVES, one row a problem its search solved: ONE verified attempt of it, drawn by the task's seed with
    L5's own function (`l5.solve_rows`, as L6 draws a round-2 row), of origin `new`, with its part of the search, the
    map's k, why it was taken and the place of its first success among B's own attempts (from 1). Refused (ValueError)
    by that function when the stored search does not agree with itself."""
    of_the_search = {row["problem_id"]: row for row in store.read_rows(SEARCH_PROBLEMS_FILE)}
    return [{**row, "origin": NEW, "part": of_the_search[row["problem_id"]]["part"], "map_k": of_the_search[row["problem_id"]]["map_k"],
             "taken_by": of_the_search[row["problem_id"]]["taken_by"]}
            for row in solve_rows([row for row in store.read_rows("problems.jsonl") if row["set"] == SEARCH_SET], _attempts(store, SEARCH_SET), _results(store, SEARCH_SET), seed,
                                  DEEP)]


def ladder_l6b_prepare(config: dict) -> dict:
    round_ = the_settings(config)
    store = _store(config, round_)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    for step in (SEARCH_PREPARE, SEARCH):
        _need(store, step, "L6b's two training sets", ENTRY_STAGE)
    read, seed = store.done_summary(SEARCH_PREPARE), training_seed(config)
    of_the_search, of_the_map = read["the_search"], read["the_map"]
    smoke, batch = bool(read.get("smoke")), config["training"]["effective_batch"]
    searched, old, next_rows = store.read_rows(SEARCH_PROBLEMS_FILE), store.read_rows(OLD_SOLVES_FILE), store.read_rows(NEXT_FILE)
    aside, grouped = [row["problem_id"] for row in store.read_rows(ladder_l5.HELD_ASIDE_FILE)], [row["problem_id"] for row in store.read_rows(ladder_l4.GROUPS_FILE)]
    try:
        new = the_new_solves(store, seed)
    except ValueError as error:
        raise RuntimeError(f"{error} (the search of this run). Nothing was written.") from error
    results = _results(store, SEARCH_SET)
    table = second_search_table(searched, results, new, of_the_search["episodes_each"], of_the_map["episodes_each"])
    enough = enough_round_two_solves(len(new), of_the_search["minimum_new_solves"])
    lines = {"new_solves": verified_lines(lines_counted(new)), "old_own_rows": verified_lines(lines_counted(old))}
    found = the_yield(table, new, enough, lines)
    carried = {key: read[key] for key in CARRIED}
    if not enough["passes"]:            # nothing is trained: the later steps say so and return, and the search is the finding
        summary = {**carried, "stand_in_engine": _stand_in(), "enough_new_solves": enough, "new_solves": len(new), "old_own_rows": len(old), "K": 0, "trained": [],
                   "averaged": [], "training_sets": {}, "the_search_found": found,
                   "trainings": {"run": False, "why": f"the search found {len(new)} new solves and {enough['minimum']} are asked: nothing is trained. No set is built and "
                                                      "no model is trained or measured: the search is the finding"}}
        store.write_rows(NEW_SOLVES_FILE, [{**row, "in_the_set": False} for row in new])
        store.mark_done(PREPARE, summary)
        _say(f"L6b: the search found {len(new)} new solves of {len(searched):,} problems, FEWER THAN THE {enough['minimum']} asked: no set is built and the eight trainings "
             "are not run. The search is the finding")
        return summary

    # ---- K, the two sets and each in its four orders, HELD before anything is written
    heldout = {row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[0]} | set(grouped)
    base_map = {row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[1]}
    goal = {row["problem_id"] for row in store.read_rows("problems.jsonl") if row["set"] != SEARCH_SET}
    try:
        dose = the_dose(len(next_rows), len(old), len(new))
        sets = {model: numbered(own) for model, own in the_sets(new, old, next_rows, seed, dose).items()}
        held = hold_the_sets(sets, next_rows, seed, dose)
        counted = {model: refuse_barred(sets[model], heldout, config["ladder_loop"]["l4"]["half_seed"], base_map, f"the training set of `{model}`") for model in MODELS}
        beside = {model: refuse_problems_of(sets[model], goal, set(aside), f"the training set of `{model}`") for model in MODELS}
        in_orders, coincide = in_their_orders(sets, smoke)
    except ValueError as error:
        raise RuntimeError(f"{error} (seed {seed}; the search of this run; L5d's run {Path(read['l5ds_run']['run']).name}). Nothing was written.") from error
    trained, steps = list(TRAINED), steps_per_pass(len(next_rows), batch)
    in_the_set = {row["problem_id"] for row in sets[NEW_OWN] if row["origin"] == NEW}
    factors = ladder_l5_noise.learning_rate_factors(read["recipe"]["warmup_steps"], steps)

    # ---- from here on files are written: every new solve, the two sets and each in its four orders
    store.write_rows(NEW_SOLVES_FILE, [{**row, "in_the_set": row["problem_id"] in in_the_set} for row in new])
    for name, own in (*sets.items(), *in_orders.items()):
        store.write_rows(THE_STAGE.training_file(name), own)
    said = {NEW_OWN: f"K = {dose} of B's new solves (" + ("every one" if len(new) == dose else f"a seeded draw of the {len(new)}") + ") and the first "
                     f"{len(next_rows) - dose:,} of the next rows of the seed's order of the pretraining file",
            OLD_OWN: f"the first K = {dose} own rows of L5d's `{DEEP}` set of this seed (`{read['first_start']}`'s deep solves), in their stored order, and THE SAME "
                     f"{len(next_rows) - dose:,} published rows"}
    summary = {
        **carried, "stand_in_engine": _stand_in(), "trained": trained, "averaged": list(MODELS), "enough_new_solves": enough, "new_solves": len(new),
        "old_own_rows": len(old), "K": dose, "rows_a_set": len(next_rows), "published_rows_a_set": len(next_rows) - dose, "the_search_found": found,
        "the_dose": {"what": "K, the rows of the model's own in each set: the smaller of half a set, the old own rows L5d's set holds and the new solves (the section's "
                             "\"the smaller of 400 and N\")", "half_a_set": len(next_rows) // 2, "old_own_rows": len(old), "new_solves": len(new), "K": dose,
                     "new_solves_drawn_by_the_seed": len(new) > dose},
        "the_sets": {model: {"what": said[model], "file": THE_STAGE.training_file(model), **described(sets[model], batch), "barred": counted[model],
                             "problems_of_g_or_of_the_held_aside_set": {key: beside[model][key] for key in ("problems_of_the_goal_set", "problems_of_the_held_aside_set")}}
                     for model in MODELS},
        "the_sets_held": held,
        "training_sets": {name: {"what": f"the set of `{name.rpartition('_')[0]}` in the content-hash order of ORDER SEED {name.rpartition('_')[2]}",
                                 "file": THE_STAGE.training_file(name), "model": name.rpartition("_")[0], "order_seed": int(name.rpartition("_")[2]),
                                 **described(in_orders[name], batch)} for name in trained},
        "the_averaged": {model: {"what": "trained by nothing: every number of its adapter is the mean of its four trainings' numbers at that place",
                                 "of": [trained_of(model, order) for order in ORDERS]} for model in MODELS},
        "the_orders": {"what": "each set in the content-hash order of each ORDER SEED (`assembly.training_order`: a hash of the order seed and each row's id), as L5n orders "
                               "rows. An order seed that is the task's own seed gives the seed's own order: it is trained all the same, and named here",
                       "order_seeds": list(ORDERS), **{model: coincide[model] for model in MODELS}},
        "optimizer_steps": steps, "learning_rate_factor": {key: factors[key] for key in ("what", "steps", "warmup_steps", ladder_l5_draws.CONSTANT)},
        "trainings": {"run": True}}
    store.mark_done(PREPARE, summary)
    _say(f"L6b: the search found {len(new):,} new solves of {len(searched):,} problems ({table[PART_A]['solved']:,} of the {table[PART_A]['problems']:,} search 1 left "
         f"unsolved, {table[PART_B]['solved']:,} of the {table[PART_B]['problems']:,} in neither draw). K = {dose}: `{NEW_OWN}` {said[NEW_OWN]}; `{OLD_OWN}` {said[OLD_OWN]}. "
         f"{len(trained)} trainings of {steps} optimizer steps from B, {len(MODELS)} averaged adapters, {len(MODELS)} measurements on `{read['the_sampling']['name']}` alone"
         + ("; A SMOKE RUN" if smoke else ""))
    return summary


def _of_this_run(store: ArtifactStore, what: str) -> dict:
    """The sets step's summary, for a later step of THIS run."""
    _need(store, PREPARE, what, ENTRY_STAGE)
    return store.done_summary(PREPARE)


# --------------------------------------------------------------------------------------------------- train
def ladder_l6b_train(config: dict, name: str) -> dict:
    """One training (`<model>_<k>`), by L4t's own training body (`ladder_l4_rows.train_a_model`) under this stage's
    names, HANDED B AS ITS START (`start_from`, as L5c's `again` is trained): one pass over the model's set in the order
    of order seed k, the recipe as it is, each step's learning rate recorded and held to it."""
    round_ = the_settings(config)
    store, marker = _store(config, round_), THE_STAGE.train_marker(name)
    if store.is_done(marker):
        return store.done_summary(marker)
    not_run = ladder_l5._not_run(store, f"the training of `{name}`", THE_STAGE)
    if not_run:
        return not_run
    prepared, best = _of_this_run(store, f"the training of `{name}`"), the_best(config, round_)
    schedule = ladder_l4_rows.Schedule(decay=False, order=f"a content hash of the ORDER SEED {prepared['training_sets'][name]['order_seed']} and each row's id "
                                                          "(`assembly.training_order`): the set's own rows, in one of the orders its averaged adapter is the mean of")
    return train_a_model(config, store, name, best.name, run_name(round_, training_seed(config)), THE_STAGE, schedule=schedule, start_from=best)


# ------------------------------------------------------------------------------------------------- average
def ladder_l6b_average(config: dict, model: str) -> dict:
    """NO TRAINING: the adapter `model`, whose every number is the mean of its four trainings' numbers at its place, BY
    L5n's OWN FUNCTION (`ladder_l5_noise.the_mean_adapter`): saved under `adapters/<model>`, read back, and held to be
    that mean, none of them and at the start adapter's rank. Refused there, with the step not marked done."""
    round_ = the_settings(config)
    store, marker = _store(config, round_), average_marker(model)
    if store.is_done(marker):
        return store.done_summary(marker)
    not_run = ladder_l5._not_run(store, f"the averaged adapter `{model}`", THE_STAGE)
    if not_run:
        return not_run
    prepared = _of_this_run(store, f"the averaged adapter `{model}`")
    own = list(prepared["the_averaged"][model]["of"])
    summary = {"stage": L4, "label": LABEL, "check": "L6b", "model": model, "recipe": MEAN_CONSTANT, "of": own, "adapter": str(store.root / ladder_assembly.ADAPTERS / model),
               "stand_in_engine": _stand_in()}
    return ladder_l5_noise.the_mean_adapter(store, the_best(config, round_), model, own, marker, summary, bool(prepared.get("smoke")), THE_STAGE)


# ------------------------------------------------------------------------------------------------- measure
def ladder_l6b_measure(config: dict, model: str) -> dict:
    """One of the two models on the goal set's FIRST sampling alone, as L5f and L5c measure (`ladder_ceiling.measure_model`
    with no rung): the attempts and the sampling seed the first start's run recorded for that sampling, at the config's
    own temperature, from the averaged adapter this run keeps. The model server is started with the start adapter's
    rank as its largest (`config_from`)."""
    round_ = the_settings(config)
    store = _store(config, round_)
    not_run = ladder_l5._not_run(store, f"the measurement of `{model}`", THE_STAGE)
    if not_run:
        return not_run
    prepared = _of_this_run(store, f"the measurement of `{model}`")
    sampling, adapter = prepared["the_sampling"], store.root / ladder_assembly.ADAPTERS / model
    return measure_model(config_from(config, prepared, the_best(config, round_), serving=True), store, {"goal_samplings": prepared["goal_samplings"]}, Model(
        name=model, called=f"L6b's model `{model}`", whose=f"L6b's model `{model}`", marker=THE_STAGE.measure_marker(model), trained_by=average_marker(model), adapter=adapter,
        no_adapter=f"the adapter `{model}` is kept by this stage, and one made again would not be the model its other figures stand beside. The measurement cannot be made",
        detail=f"G's first sampling alone, {sampling['episodes']} attempts a problem, sampling seed {sampling['sampling_seed']}",
        summary={"stage": L4, "label": LABEL, "check": "L6b", "model": model, "adapter": str(adapter)}, stage=ENTRY_STAGE, label=L4, also=what_else_a_model_wrote, rungs=False))


# -------------------------------------------------------------------------------------------------- report
def the_search_again(config: dict, round_: Round, prepared: dict, aside: list[dict]) -> dict:
    """L6's SEARCH 2 MADE AGAIN for the report's check, by L6's own rule (`l6.second_search`) on the stored runs as they
    are NOW: the map's rows, search 1's problems and the ones it solved (round 1's run), the held-aside set as this run
    copied it. Nothing is refused: a rule that refuses gives its words."""
    directory, of_the_search = Path(prepared["round_one"]["run"]), prepared["the_search"]
    try:
        map_rows = _rows(Path(prepared["map_run"]) / f"episodes_{prepared['the_map']['set']}_problems.jsonl")
        drawn = _rows(directory / ladder_l5.SEARCH_PROBLEMS_FILE)
        solved = [row["problem_id"] for row in _rows(directory / f"episodes_{ladder_l5.SEARCH_SET}_problems.jsonl") if row["resolved"] > 0]
        again = second_search(map_rows, drawn, aside, solved, of_the_search["seed_of_the_search"], of_the_search["problems_at_least"])["problems"]
    except (ValueError, OSError, KeyError) as error:
        return {"problems": None, "solved_by_search_one": [], "refused": str(error)}
    return {"problems": again, "solved_by_search_one": solved, "refused": None}


def ladder_l6b_report(config: dict) -> dict:
    from rlvr_lean.gpu.ladder_l3c import proof_lengths
    from rlvr_lean.reporting.ladder_l6b import build_l6b_report

    round_ = the_settings(config)
    store = _store(config, round_)
    prepared = _of_this_run(store, "L6b's report")
    _need(store, SEARCH, "L6b's report", ENTRY_STAGE)
    ran = bool(prepared["trainings"]["run"])
    names, models, sampling, seed = list(prepared["trained"]), (MODELS if ran else ()), prepared["the_sampling"], prepared["seed"]
    for step in (step for model in models for step in (*(THE_STAGE.train_marker(name) for name in prepared["the_averaged"][model]["of"]), average_marker(model),
                                                       THE_STAGE.measure_marker(model))):
        _need(store, step, "L6b's report", ENTRY_STAGE)
    sets = {name: store.read_rows(THE_STAGE.training_file(name)) for name in (*models, *names)}
    trains, trained = {}, {}
    for name in names:                  # what a model WAS trained on is the training loop's own record of row ids, read back to its stored set's rows (never their text)
        trains[name] = store.done_summary(THE_STAGE.train_marker(name))
        by_id = {row["id"]: row for row in sets[name]}
        trained[name] = [{key: by_id[row_id].get(key) for key in ("id", "problem_id", "origin", "row_of_the_file")}
                         for row_id in json.loads(store.path(THE_STAGE.loss_file(name)).read_text())["rows_trained"]]
    averaged = {model: store.done_summary(average_marker(model)) for model in models}
    measured = {model: store.done_summary(THE_STAGE.measure_marker(model)) for model in models}
    rows = {model: _results(store, the_set(model, sampling["name"])) for model in models}
    first = ladder_l5_full.the_first_start(config, round_.full)
    # THE SETS ARE MADE AGAIN AND HELD by the report: against the rows the first start was pretrained on (the pretraining file, held to its hash), the new solves made again from
    # the stored search, and L5d's and L5c's runs AS THEY ARE NOW; the searched set against L6's own rule on the stored runs.
    pretraining = ladder_l4_start.pretrained_on(first.directory, prepared["pretraining_file_sha256"], first.name)[0]
    base_map = [row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[1]]
    heldout = [row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[0]]
    ids_of = lambda name: [row["problem_id"] for row in store.read_rows(name)]      # noqa: E731
    try:
        new_again, new_refused = the_new_solves(store, seed), None
    except ValueError as error:
        new_again, new_refused = [], str(error)
    aside_rows = store.read_rows(ladder_l5.HELD_ASIDE_FILE)
    now = {"l5f": ladder_l5_cont.read_l5f(config, round_.cont), "l5c": read_l5c(config, round_), "l5cs_again": l5cs_again(l5c_directory(config, round_)),
           "old": l5ds_old(Path(prepared["l5ds_run"]["run"])), "new": new_again, "new_refused": new_refused}
    of_the_search = {"summary": store.done_summary(SEARCH), "problems": store.read_rows(SEARCH_PROBLEMS_FILE), "results": _results(store, SEARCH_SET),
                     "again": the_search_again(config, round_, prepared, aside_rows), "solves": store.read_rows(NEW_SOLVES_FILE), "heldout": heldout}
    stored = {"best": store.read_rows(sampling["stored_best"]), "bar": store.read_rows(sampling["stored_bar"]),
              "second_start": store.read_rows(sampling["stored_second_start"]), "first_start": store.read_rows(sampling["stored_first_start"])}
    # The length of a held-out problem's published proof is read HERE, by the report, and by nothing before it.
    report = build_l6b_report(prepared, of_the_search, trains, trained, averaged, measured, rows, sets, pretraining, now, stored, store.read_rows(ladder_l4.GROUPS_FILE),
                              proof_lengths(), ids_of(ladder_l4.AGAIN_FILE), [row["problem_id"] for row in aside_rows], config["ladder_loop"]["l4"], base_map,
                              {"sampling": round_.draws.sampling, "attempts": round_.draws.attempts})
    directory = store.root / ladder_assembly.ADAPTERS
    report["adapters"] = {"kept": True, "what": (f"the adapters of {', '.join(f'`{name}`' for name in (*names, *models))} ({LABEL}) are KEPT: no step of this stage deletes one. "
                                                 if ran else "no model was trained. ") + f"B and its four trainings are L5f's run's, `{round_.cont.start}` L4p's, `{BAR}` "
                                                "L5c's: they were only read",
                          "directory": str(directory), "there": sorted(path.name for path in directory.glob("*") if path.is_dir()) if directory.is_dir() else []}
    _write_json(store, REPORT_FILE, report)
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    for name in (THE_STAGE.loss_file(name) for name in names):
        store.mirror(name)
    for line in report["lines"]:
        print(line, flush=True)
    store.mark_done(REPORT, {"stage": L4, "label": LABEL, "check": "L6b", "seed": seed, "recipe": MEAN_CONSTANT, "headline": report["headline"],
                             "branches": report["branches"], "measured": list(models), "ok": report["ok"]})
    return report


def _for(step, name: str):
    def run(config: dict) -> dict:
        return step(config, name)
    return run


# In the section's order: the first step; the search; the two sets; then model by model its four trainings, its averaged adapter and its measurement; then the report.
STEPS = {SEARCH_PREPARE: ladder_l6b_search_prepare, SEARCH: ladder_l6b_search, PREPARE: ladder_l6b_prepare}
for _model in MODELS:
    for _order in ORDERS:
        STEPS[THE_STAGE.train_marker(trained_of(_model, _order))] = _for(ladder_l6b_train, trained_of(_model, _order))
    STEPS[average_marker(_model)] = _for(ladder_l6b_average, _model)
    STEPS[THE_STAGE.measure_marker(_model)] = _for(ladder_l6b_measure, _model)
STEPS[REPORT] = ladder_l6b_report
