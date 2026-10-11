"""L6: a second round. Does the cycle compound? Spec: docs/spec/ladder-loop.spec.md, "L6: a second round. Does the
cycle compound?" and its "Made exact by the build". Everything it makes is labelled pretrained on published proofs. Each
step is its own process (`rlvr_lean.runner.entry`, stage `ladder_l6_cycle`). One seed.

ROUND 1 IS L5's, READ AND NEVER MADE AGAIN: the finished run of the seed that made L5's deep search
(`ladder_l5.search_directory`): its search (the problems it attempted and the ones it solved), its held-aside set, the start
model's measurement on it, and M(1), its `deep` model, whose adapter that run keeps. With it, only read, every stored run
that task read (the start model's own map, L4t2's finished run, the start model's run), by L5's OWN first-step reading
(`ladder_l5.read_the_search`, `read_for_the_search`: its refusals are a further seed's of L5), and, WHEN IT IS ON THE BOX,
L4p's run for `second_full` (its stored rows on G and its adapter; NOT THERE otherwise, and nothing is refused for it).
The run directory is its own (`ladder_l6_cycle_<start>_seed<N>`).

  ladder_l6_search_prepare      REFUSES, naming the task to run, before anything is written: a round 1 that is missing,
                                unfinished, not to be read, INCONCLUSIVE or made otherwise than this task would make it; one
                                that trained no M(1), or whose M(1) was measured otherwise; a sampling seed a stored sampling
                                used; a task of another seed than `ladder_loop.l6.search_seed` (a further seed of L6 is not
                                built). HOLDS the partition and stores SEARCH 2's problems: (a) what search 1 left unsolved,
                                (b) the candidates neither draw took. No GPU, no Lean
  ladder_l6_search              SEARCH 2 (`ladder_l5.search_deep`, the deep search's own body): M(1), served from round 1's
                                run at the start adapter's rank, makes `search.attempts` attempts at each problem with a
                                sampling seed of its own. A search Lean did not answer is not kept
  ladder_l6_prepare             a problem search 2 solved is a ROUND-2 solve. FEWER THAN `search.minimum`: no set is built,
                                the later steps train and measure nothing, and the report says the search is the finding.
                                Otherwise BUILDS and stores THE FOUR SETS of 2N rows (`domain/ladder_round/l6.py`), N every
                                deep solve of both rounds, each row ONE verified attempt drawn by the task's seed as L5 draws
  ladder_l6_train_<model>       L4t's OWN two step bodies (`ladder_l4_rows.train_a_model`, `measure_a_model`) under this
  ladder_l6_measure_<model>     stage's names, for `published2`, `one_round`, `two_rounds`, `long_ordinary` in turn: FROM THE
                                START ADAPTER at its rank, one pass; then the model as L5's were measured
  ladder_l6_held_aside_<model>  L5's OWN body (`ladder_l5.on_the_held_aside_set`): the model's fresh attempts at every
                                held-aside problem, with ROUND 1's sampling seed for that set, so that it pairs by problem
                                with every model round 1 measured there
  ladder_l6_held_aside_second_full   one more measurement of a STORED model, the same way, from its adapter in L4p's run;
                                NOT THERE (said, nothing marked, nothing refused) when that run or adapter is not on the box
  ladder_l6_report              the read fixed before the run (`reporting/ladder_l6.py`)

Every step is resumable and a step of its own. NO STEP DELETES AN ADAPTER, and no step writes into a run it reads.

A PUBLISHED ROW is named by its id, its problem and its place in the file, never with its text: the training step reads
the text from the pretraining file (`data/ladder_l4/pretraining.jsonl`, not distributed), held to the SHA-256 the start
model's run recorded. A deep row (of either round) and an ordinary solve's row are a model's OWN proofs, verified by Lean,
of base-map problems, and are stored with their text as a round's training examples are.

A SMOKE RUN ALONE (`RLVR_LEAN_LADDER_L6_SMOKE_PROBLEMS`, refused unless the run directory is one the task names and the
map and the data are the fixtures'): round 1's smoke run reads INCONCLUSIVE (a training of two rows has no loss to
compare) and is read all the same; search 2 attempts at least that many problems, the map's other problems joining when
the rule gives fewer; one round-2 solve is enough; `long_ordinary` takes its ordinary solves as loosely. Every problem and
row so taken is marked as NOT the rule's, and the first step records what it went on from.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rlvr_lean.domain.ladder_round.l4_dose import SECOND_FULL
from rlvr_lean.domain.ladder_round.l4_rows import PRETRAINING, THE_MINIMUM, THE_RULE
from rlvr_lean.domain.ladder_round.l4_rows2 import refuse_text
from rlvr_lean.domain.ladder_round.l5 import DEEP, SHALLOW, refuse_barred, refuse_overlap, solve_rows
from rlvr_lean.domain.ladder_round.l6 import (
    LONG_ORDINARY,
    MODELS,
    ONE_ROUND,
    ORDINARY,
    PART_A,
    PART_B,
    PUBLISHED2,
    ROUND_1,
    ROUND_2,
    TWO_ROUNDS,
    enough_round_two_solves,
    line_bins,
    matched_in_length,
    second_search,
    second_search_table,
    the_sets,
)
from rlvr_lean.domain.ladder_round.read import GOAL
from rlvr_lean.domain.problem_pool.episodes import VERIFIED
from rlvr_lean.gpu import ladder_assembly, ladder_ceiling, ladder_l4, ladder_l4_dose, ladder_l4_rows, ladder_l4_start, ladder_l5, ladder_loop
from rlvr_lean.gpu.ladder_ceiling import RUNG_PART, Reading, _need, goal_set, rung_set, stored_file
from rlvr_lean.gpu.ladder_dose import _rows, _runs
from rlvr_lean.gpu.ladder_l3d1 import WROTE
from rlvr_lean.gpu.ladder_l4 import L4, LABEL, _say
from rlvr_lean.gpu.ladder_l4_rows import described, lines_counted, measure_a_model, read_the_trained, train_a_model
from rlvr_lean.gpu.ladder_l4_start import Start
from rlvr_lean.gpu.ladder_l5 import HELD_ASIDE_PART, block_of, held_aside_set
from rlvr_lean.gpu.ladder_round import _attempts, _results, _stand_in, _write_json, training_seed
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.reporting.ladder_ceiling import verified_lines

L6 = "l6"                                                   # what this stage's own files are called
STAGE = "ladder_l6"                                         # what its steps begin with
ENTRY_STAGE = f"{STAGE}_cycle"                              # the stage, as `runner.entry` names it; its run directory begins with it
RUN_VARIABLE = "RLVR_LEAN_LADDER_L6_RUN"                    # another run directory than `ladder_l6_cycle_<start>_seed<seed>` (a smoke run)
L4P_RUN_VARIABLE = "RLVR_LEAN_LADDER_L6_L4P_RUN"            # another run directory of L4p than `ladder_l4_dose_<start>_seed<seed>` (a smoke run: L4p's smoke run)
SMOKE_VARIABLE = "RLVR_LEAN_LADDER_L6_SMOKE_PROBLEMS"       # a SMOKE run alone (refused elsewhere): search 2 attempts at least this many problems, and what else the module's note says
SETTING = "ladder_loop.l6"
THE_STAGE = ladder_l4_rows.Stage(name=ENTRY_STAGE, steps=STAGE, files=L6, reader="L6", start_said_by=f"{SETTING}.start")
SEARCH_PREPARE, SEARCH, PREPARE, REPORT = f"{STAGE}_search_prepare", f"{STAGE}_search", THE_STAGE.prepare, f"{STAGE}_report"
SEARCH_SET = f"{L6}_search"                                 # the set search 2's episodes are stored under
THE_SEARCH = ladder_l5.Search(marker=SEARCH, set_name=SEARCH_SET, stage=THE_STAGE, said="second search")
HELD_ASIDE = f"{STAGE}_held_aside"                          # what the steps that measure the held-aside set begin with: one a trained model, then the stored model's
M1 = DEEP                                                   # M(1), the model round 1 trained on its deep solves: L5's `deep`, by the name its run keeps it under
STORED = SECOND_FULL                                        # the ONE stored model measured on the held-aside set beside: L4p's whole second pass, the best pretraining-only model
REPORT_FILE = "report_ladder_l6.json"
STORED_MODELS_FILE = f"{L6}_stored_models.json"             # what the start model wrote, from its own run's marker
SEARCH_PROBLEMS_FILE = f"{L6}_search_problems.jsonl"        # search 2's problems, (a) then (b), each with its part, the map's k and why it was taken
HELD_ASIDE_FILE = f"{L6}_held_aside_problems.jsonl"         # the held-aside problems, as round 1's run stored them
SOLVES_FILE = f"{L6}_deep_solves.jsonl"                     # every deep solve of both rounds: its row, its round, the place of its first success
ORDINARY_FILE = f"{L6}_ordinary_solves.jsonl"               # the ordinary solves `long_ordinary` is trained on, each with the lines of its row, its bin and why it was taken (no text)
NEVER_SOLVED_FILE = f"{L6}_never_solved.jsonl"              # GIVEN, not made here: one `problem_id` a row, the goal problems nothing stored has ever solved
SPREAD_FILE = f"{L6}_l5_spread.json"                        # GIVEN, not made here: the read over L5's seeds, as `python -m rlvr_lean.tools.ladder_l5_seeds --out` writes it
READING = Reading(reader="L6", models="L6's models", not_read="that set stands on the other side of a comparison, and no report of this run could be read against it",
                  setting=ladder_l4.SETTING)
# What the four sets' step carries on from the step that read the runs: what L4t's two bodies, L5's held-aside body and the report read of a prepare step.
CARRIED = ("stage", "label", "check", "seed", "start", "start_adapter", "start_run", "l4t2_run", "map_run", "models", "effective_batch", "rows_the_start_was_pretrained_on",
           "pretraining_file", "pretraining_file_sha256", "recipe", "goal_set", "goal_set_again", "rung_problems", *ladder_l5.PLAN, "attempts_a_goal_problem", "parts",
           "stored_models", "never_solved_file", "spread_file", "contradicted_side_setting", "start_recipe", "start_run_read", "the_map", "round_one", "the_search",
           "the_held_aside", "the_stored_model", "smoke")


@dataclass(frozen=True)
class Cycle:
    """`ladder_loop.l6`, as the stage's steps read it."""
    start: str                          # the ONE start adapter: every model here is trained from it at its rank (round 1's start)
    models: tuple[str, ...]             # the four trained models, in the order they are trained and measured
    attempts: int                       # the attempts M(1) makes at each problem of search 2
    sampling_place: int                 # ... its sampling seed is `round.sampling_seed` + 100 x the seed of the search's task + this place, which no other sampling has
    minimum: int                        # with fewer round-2 solves no set is built and nothing is trained
    least: int                          # an ordinary solve: a map problem the start model solved in at least this many of its map attempts
    edges: tuple[int, ...]              # the lower edges of the bins of a proof's lines that `long_ordinary` is matched to `two_rounds` by
    seed_of_the_search: int             # the seed whose task makes search 2; a task of another seed would be a further seed, which is not built
    deep: ladder_l5.Deep                # L5's settings as ITS task reads them: round 1 is the finished run of its search's seed


def the_settings(config: dict) -> Cycle:
    """THE one place that reads `ladder_loop.l6`. Refused (ValueError): models that are not the stage's (its steps are
    registered for them: change both together); a start that is not L5's (round 1 is L5's run from that start); a size
    that is not a whole number of at least 1; line bins that do not begin at 1 and rise; a sampling place that is one
    of L5's two; a seed of the search that is no seed; L5's own settings, when its own reader refuses them."""
    settings = config["ladder_loop"]["l6"]
    start, models, search, ordinary = settings["start"], tuple(settings["models"]), settings["search"], settings["long_ordinary"]
    if models != MODELS:
        raise ValueError(f"{SETTING}.models is {list(models)} and the stage trains and measures {list(MODELS)}: change both together")
    of_the_search = settings["search_seed"]
    if isinstance(of_the_search, bool) or not isinstance(of_the_search, int) or of_the_search < 0:
        raise ValueError(f"{SETTING}.search_seed is {of_the_search!r}: it is the seed of the ONE task that makes search 2, a whole number of at least 0")
    deep = ladder_l5.the_settings(config)
    if start != deep.start:
        raise ValueError(f"{SETTING}.start is `{start}` and {ladder_l5.SETTING}.start is `{deep.start}`: round 1 is L5's finished run from that start, and every model here "
                         "is trained from it")
    sizes = {"search.attempts": search["attempts"], "search.sampling_place": search["sampling_place"], "search.minimum": search["minimum"],
             "long_ordinary.solved_at_least": ordinary["solved_at_least"]}
    wrong = [f"{SETTING}.{key} is {value!r}" for key, value in sizes.items() if isinstance(value, bool) or not isinstance(value, int) or value < 1]
    if wrong or search["sampling_place"] in (deep.sampling_place, deep.held_aside_place):
        raise ValueError("; ".join(wrong or [f"{SETTING}.search.sampling_place is {search['sampling_place']} and L5's two are {deep.sampling_place} and {deep.held_aside_place}"])
                         + ": each is a whole number of at least 1, and search 2's sampling seed is neither the first search's nor the held-aside set's")
    edges = tuple(ordinary["line_bins"])
    line_bins(edges)                    # refused unless they begin at 1 and rise
    return Cycle(start=start, models=models, attempts=search["attempts"], sampling_place=search["sampling_place"], minimum=search["minimum"], least=ordinary["solved_at_least"],
                 edges=edges, seed_of_the_search=of_the_search, deep=deep)


def run_name(cycle: Cycle, seed: int) -> str:
    """The stage's run directory, with the start adapter's name in it: `ladder_l6_cycle_pre_r64_seed<N>`."""
    return f"{ENTRY_STAGE}_{cycle.start}_seed{seed}"


def _store(config: dict, cycle: Cycle) -> ArtifactStore:
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    return ArtifactStore(_runs(config) / (os.environ.get(RUN_VARIABLE) or run_name(cycle, training_seed(config))), Path(mirror) if mirror else None)


def round_one_directory(config: dict, cycle: Cycle) -> Path:
    """Round 1's run: the run of the task that made L5's deep search (`ladder_l5.search_directory`, or the run the task
    names for L5: a smoke run). Only read."""
    return ladder_l5.search_directory(config, cycle.deep)


def l4p_directory(config: dict, cycle: Cycle) -> Path:
    """L4p's run from this stage's start, by L4p's own name for it (or the run the task names: a smoke run), of the seed
    of L5's search, as every stored run read here is. Only read, and only when it is there."""
    return _runs(config) / (os.environ.get(L4P_RUN_VARIABLE) or ladder_l4_dose.run_name(ladder_l4_dose.the_settings(config), cycle.deep.seed_of_the_search))


def the_start(config: dict, cycle: Cycle) -> Start:
    """The start model the setting names, and where it lives (`ladder_l4_start.the_start`)."""
    return ladder_l4_start.the_start(config, cycle.start, THE_STAGE.start_said_by)


def search_seed(config: dict, cycle: Cycle) -> int:
    """Search 2's own sampling seed: `round.sampling_seed` + 100 x the seed of the search's task +
    `search.sampling_place`, as L5's two are derived, at a place neither has. It is of the seed the SETTING names, so
    that a further seed could later read this search and sample nothing of it again."""
    return config["ladder_loop"]["round"]["sampling_seed"] + 100 * cycle.seed_of_the_search + cycle.sampling_place


def held_aside_marker(who: str) -> str:
    """The step that measures the held-aside set for one model (a trained model, or the stored one), and its marker."""
    return f"{HELD_ASIDE}_{who}"


def smoke_problems() -> int:
    """A smoke run's least number of problems in search 2; 0 for every other run: the rule and nothing else."""
    return max(0, int(os.environ.get(SMOKE_VARIABLE) or 0))


def round_one_reader(cycle: Cycle, smoke: bool) -> ladder_l5.OfTheSearch:
    """Who reads L5's finished run here, for what L5's own reading says when it refuses it: L6, whose round 1 it is. A
    SMOKE run alone reads a round 1 whose report is INCONCLUSIVE (one not to be read never is)."""
    return ladder_l5.OfTheSearch(
        who="L6", why=f"L6 makes no first round of its own: round 1 is L5's finished run of seed {cycle.deep.seed_of_the_search}, whose search, held-aside set and `{M1}` model "
                      "(M(1), which makes search 2) are read from that run",
        stands="no second round stands on such a run", models="L6's models are read beside that run's, on its search and its held-aside set", inconclusive=smoke)


def read_round_one(found: dict, read: dict, cycle: Cycle) -> dict:
    """What L6 needs of round 1's run BESIDE what a further seed of L5 reads of it (`ladder_l5.read_the_search` and
    `read_for_the_search` have read and held that): M(1). REFUSED (RuntimeError), naming the task to run: a round 1
    that trained no model (its search found too few deep solves); one whose `deep` model was not trained, measured, or
    measured on G otherwise than the start model was; one whose adapter is not there; one whose stored search does not
    count the deep solves its own step counted, or holds no verified attempt for a problem it counts as solved (held
    HERE, before search 2 is made: a round-1 row is one of those attempts). Returns M(1)'s adapter, its stored rows on
    G (a list a sampling) and the problems search 1 solved."""
    directory, task, prepared, plan = found["directory"], ladder_l5.search_task(cycle.deep), found["prepared"], read["plan"]
    if not (prepared.get("trainings") or {}).get("run"):
        raise RuntimeError(f"round 1's run {directory.name} trained no model ({(prepared.get('trainings') or {}).get('why')}): there is no M(1) to make search 2. Nothing "
                           "was written.")
    of_l5 = ladder_l5.THE_STAGE
    lacking = [f"{marker}.done.json" for marker in (of_l5.train_marker(M1), of_l5.measure_marker(M1)) if not (directory / f"{marker}.done.json").exists()]
    sets = [goal_set(sampling["name"], M1, L4) for sampling in plan["goal_samplings"]]
    lacking += ladder_ceiling._missing(directory, sets)
    if lacking:
        raise RuntimeError(f"{directory} does not hold {lacking}: what round 1 recorded of `{M1}`, M(1), or its stored rows on G. Run {task} to its end first; nothing was "
                           "written.")
    adapter = directory / ladder_assembly.ADAPTERS / M1
    if not ladder_ceiling._stand_in() and not adapter.is_dir():
        raise RuntimeError(f"{adapter} is not there: the adapter `{M1}` of round 1 is M(1), which makes search 2. Queue {task} again (its training step trains it again); "
                           "nothing was written.")
    goal_ids = [row["problem_id"] for row in read["goal_problems"]]
    on_g = [ladder_ceiling._stored(directory, set_name, goal_ids, sampling["sampling_seed"], sampling["episodes"], f"`{M1}` ({sampling['name']})", READING)[0]
            for set_name, sampling in zip(sets, plan["goal_samplings"])]
    solved = [row["problem_id"] for row in _rows(directory / f"episodes_{ladder_l5.SEARCH_SET}_problems.jsonl") if row["resolved"] > 0]
    verified = {attempt["problem_id"] for attempt in ladder_ceiling._stored_attempts(directory, ladder_l5.SEARCH_SET) if attempt["status"] == VERIFIED}
    if len(solved) != prepared["deep_solves"] or verified != set(solved):       # held HERE, before search 2 is made: a round-1 row is one of these stored attempts
        raise RuntimeError(f"round 1's stored search ({directory.name}) counts {len(solved)} problems as solved, its stored attempts hold a verified one for {len(verified)}, "
                           f"and its own step counted {prepared['deep_solves']} deep solves: the run does not agree with itself. Nothing was written.")
    return {"adapter": adapter, "on_g": on_g, "solved": solved}


def read_the_stored_model(config: dict, cycle: Cycle, read: dict) -> dict:
    """`second_full`, the best pretraining-only model stored, WHEN L4p's run is on the box: its rows on G (a list a
    sampling), held to the problems, the sampling seeds and the attempts the start model was measured with, and where
    its adapter is. NOTHING IS REFUSED FOR IT: a run that is not there, is of another start, lacks the model's markers
    or its rows, or holds rows that would not pair is NOT THERE, with why."""
    directory, of_l4p, plan = l4p_directory(config, cycle), ladder_l4_dose.THE_STAGE, read["plan"]
    adapter = directory / ladder_assembly.ADAPTERS / STORED
    found = {"model": STORED, "run": str(directory), "adapter": str(adapter)}
    sets = [goal_set(sampling["name"], STORED, L4) for sampling in plan["goal_samplings"]]
    names = [f"{marker}.done.json" for marker in (of_l4p.prepare, of_l4p.train_marker(STORED), of_l4p.measure_marker(STORED))]
    lacking = [name for name in names if not (directory / name).exists()] + ladder_ceiling._missing(directory, sets)
    if lacking:
        return {**found, "there": False, "why": f"{directory.name} does not hold {lacking}"}
    prepared = json.loads((directory / names[0]).read_text())
    if prepared.get("start") != cycle.start:
        return {**found, "there": False, "why": f"{directory.name} recorded the start {prepared.get('start')!r}, and this stage's is `{cycle.start}`"}
    goal_ids = [row["problem_id"] for row in read["goal_problems"]]
    try:
        on_g = [ladder_ceiling._stored(directory, set_name, goal_ids, sampling["sampling_seed"], sampling["episodes"], f"`{STORED}` ({sampling['name']})", READING)[0]
                for set_name, sampling in zip(sets, plan["goal_samplings"])]
    except RuntimeError as error:
        return {**found, "there": False, "why": str(error)}
    return {**found, "there": True, "on_g": on_g}


# ------------------------------------------------------------------------------------------ search_prepare
def ladder_l6_search_prepare(config: dict) -> dict:
    cycle = the_settings(config)
    store = _store(config, cycle)
    if store.is_done(SEARCH_PREPARE):
        return store.done_summary(SEARCH_PREPARE)
    seed, deep, at_least = training_seed(config), cycle.deep, smoke_problems()
    if seed != cycle.seed_of_the_search:
        raise RuntimeError(f"this task's seed is {seed} and {SETTING}.search_seed is {cycle.seed_of_the_search}: the task of that seed makes search 2, and a further seed of "
                           "L6 (the four models again, on the same two searches) is not built. Nothing was written.")
    if at_least and not os.environ.get(RUN_VARIABLE):
        raise RuntimeError(f"{SMOKE_VARIABLE} is set ({at_least}): it is for a SMOKE run alone, in a run directory the task names ({RUN_VARIABLE}) and on the fixtures' "
                           "base map. A run in the seed's own directory reads a round 1 whose checks passed and takes its problems by the rule and by nothing else. Nothing "
                           "was written.")
    # ---- round 1, and every stored run its task read: by L5's OWN reading, as a further seed of L5 reads them (refused there, with nothing written)
    reader = round_one_reader(cycle, bool(at_least))
    found = ladder_l5.read_the_search(config, deep, reader)
    # A smoke run reads a SMOKE run of L5: its two looser draws are made again as that run recorded them (L5's own smoke-only variable is L5's smoke stage's alone).
    read = ladder_l5.read_for_the_search(config, deep, found, reader, at_least=((found["searched"].get("smoke") or {}).get("problems_at_least") or 0) if at_least else 0)
    start, the_map, drawn, aside, plan = (read[key] for key in ("start", "the_map", "drawn", "aside", "plan"))
    if at_least and not the_map["fixture"]:
        raise RuntimeError(f"{SMOKE_VARIABLE} is set ({at_least}): it is for a SMOKE run alone, in a run directory the task names ({RUN_VARIABLE}) and on the fixtures' "
                           "base map. A run on the pool's problems reads a round 1 whose checks passed and takes its problems by the rule and by nothing else. Nothing was "
                           "written.")
    of_round_one = read_round_one(found, read, cycle)
    of_the_stored = read_the_stored_model(config, cycle, read)

    # ---- search 2's problems: what search 1 left unsolved, and the candidates neither draw took; HELD to partition the candidates with round 1's solves and the held-aside set
    try:
        second = second_search(the_map["rows"], drawn, aside, of_round_one["solved"], cycle.seed_of_the_search, at_least)
    except ValueError as error:
        raise RuntimeError(f"{error} (round 1's run {found['directory'].name}; the map {the_map['directory'].name}). Nothing was written.") from error
    searched = second["problems"]
    heldout = {row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[0]} | {row["problem_id"] for row in read["groups"]}
    barred = sorted(row["problem_id"] for row in searched if row["problem_id"] in heldout or row["problem_id"] not in the_map["base_map"])
    if not searched or barred:
        raise RuntimeError((f"{len(barred)} problems of search 2 are held out or are not base-map problems (first: {barred[0]})" if barred else
                            "search 2 has no problem to attempt: search 1 solved every problem it attempted, and its draw and the held-aside set took every candidate")
                           + f" (round 1's run {found['directory'].name}). Nothing was written.")
    sampling = search_seed(config, cycle)
    used = {**read["used"], "L5's deep search": read["sampling"], "L5's held-aside set": read["aside_sampling"]}
    taken = sorted(name for name, value in used.items() if value == sampling)
    if taken:
        raise RuntimeError(f"search 2's sampling seed is {sampling} ({SETTING}.search.sampling_place {cycle.sampling_place}) and {taken} already sampled with it: it has a "
                           "sampling seed of its own. Nothing was written.")

    # ---- from here on files are written: the problems of every set this run samples, what its report reads of the stored models, search 2's problems
    def as_set(own: list[dict], set_name: str) -> list[dict]:
        return [{**row, "set": set_name} for row in own]

    by_id = {row["problem_id"]: row for row in the_map["problems"]}
    rung_problems, goal_problems, again, kept = (read[key] for key in ("rung_problems", "goal_problems", "again", "kept"))
    measured_aside = (*cycle.models, STORED)                # the models THIS run measures on the held-aside set, in the order it does: its four, then the stored one
    own = [row for model in cycle.models for row in as_set(rung_problems, rung_set(model, L4))]
    own += [row for model in cycle.models for sampling_ in plan["goal_samplings"] for row in as_set(goal_problems, goal_set(sampling_["name"], model, L4))]
    own += as_set([by_id[row["problem_id"]] for row in searched], SEARCH_SET)
    own += [row for name in measured_aside for row in as_set([by_id[row["problem_id"]] for row in aside], held_aside_set(name, L6))]
    store.write_rows("problems.jsonl", own)
    store.write_rows(ladder_l4.GROUPS_FILE, read["groups"])
    store.write_rows(ladder_l4.AGAIN_FILE, [{"problem_id": problem_id} for problem_id in again])
    for (who, part), own_rows in kept.items():
        store.write_rows(stored_file(who, part, L4), own_rows)
    _write_json(store, STORED_MODELS_FILE, {"stage": L4, "label": LABEL, cycle.start: {key: read["of_the_start_run"]["measured"][key] for key in WROTE}})
    store.write_rows(SEARCH_PROBLEMS_FILE, searched)
    store.write_rows(HELD_ASIDE_FILE, aside)
    store.write_rows(stored_file(cycle.start, HELD_ASIDE_PART, L6), read["start_on_the_set"])      # the start model on the held-aside set, as round 1's run stored it
    stored_on_g = {M1: of_round_one["on_g"], **({STORED: of_the_stored["on_g"]} if of_the_stored["there"] else {})}
    for who, on_g in stored_on_g.items():                   # M(1)'s rows on G, and the stored model's when its run is there: what `two_rounds` is read against, beside
        for sampling_, own_rows in zip(plan["goal_samplings"], on_g):
            store.write_rows(stored_file(who, sampling_["name"], L6), own_rows)
    theirs, report, of_theirs = found["searched"], found["report"], found["prepared"]
    partition, batch = second["partition"], config["training"]["effective_batch"]
    by_part = Counter(row["part"] for row in searched)
    summary = {
        "stage": L4, "label": LABEL, "check": "L6", "seed": seed, "start": cycle.start, "start_adapter": str(start.adapter), "start_run": str(start.directory),
        "l4t2_run": str(read["l4t2"]), "map_run": str(the_map["directory"]), "models": list(cycle.models), "stand_in_engine": _stand_in(), "effective_batch": batch,
        "rows_the_start_was_pretrained_on": len(read["pretraining"]), "the_map": theirs["the_map"],
        "round_one": {"what": f"ROUND 1 IS L5's, only read: the finished run of seed {deep.seed_of_the_search}. Its deep search (`{cycle.start}`, {theirs['the_search']['episodes_each']} "
                              "more attempts at each problem it drew), the held-aside set, the start model's measurement on it, and M(1): its model "
                              f"`{M1}`, trained on its deep solves, whose adapter that run keeps. Nothing of it is made again, and nothing is written there",
                      "run": str(found["directory"]), "seed": deep.seed_of_the_search, "model": M1, "adapter": str(of_round_one["adapter"]),
                      "headline": report.get("headline"), "branches": report.get("branches"), "inconclusive": bool(report.get("inconclusive")),
                      "failed_checks": list(report.get("failed_checks") or []), "attempted": len(drawn), "solved": len(of_round_one["solved"]),
                      "deep_rows_in_its_own_sets": of_theirs["rows_of_each_kind"], "sampling_seed": read["sampling"],
                      "read_from_it": [ladder_l5.SEARCH_PROBLEMS_FILE, ladder_l5.HELD_ASIDE_FILE, "problems.jsonl", f"episodes_{ladder_l5.SEARCH_SET}_problems.jsonl",
                                       f"episodes_{ladder_l5.SEARCH_SET}_attempts_*.jsonl", f"episodes_{held_aside_set(cycle.start)}_problems.jsonl",
                                       *(f"episodes_{goal_set(sampling_['name'], M1, L4)}_problems.jsonl" for sampling_ in plan["goal_samplings"]), f"{ladder_assembly.ADAPTERS}/{M1}"]},
        "the_search": {"what": f"SEARCH 2: `{M1}` of round 1 (M(1)), served at the start adapter's rank, makes {cycle.attempts} attempts at each of the hard problems that are "
                               "neither round 1's solves nor held aside: (a) the problems search 1 attempted and left unsolved, (b) the candidates (map problems "
                               f"`{cycle.start}` failed in every map attempt) that neither search 1's draw nor the held-aside set took; the round's episode in every other "
                               "respect", "set": SEARCH_SET, "file": SEARCH_PROBLEMS_FILE, "model": M1, "problems": len(searched),
                       "by_part": {PART_A: by_part[PART_A], PART_B: by_part[PART_B]}, "partition": partition, "episodes_each": cycle.attempts, "sampling_seed": sampling,
                       "sampling_place": cycle.sampling_place, "seeds_of_the_stored_samplings_it_is_none_of": used, "block_problems": block_of(config, cycle.attempts),
                       "taken_by_the_rule": sum(row["taken_by"] == THE_RULE for row in searched), "taken_by_the_runs_minimum": sum(row["taken_by"] != THE_RULE for row in searched),
                       "minimum_round_two_solves": 1 if at_least else cycle.minimum},
        "the_held_aside": {**{key: theirs["the_held_aside"][key] for key in ("candidates_not_searched", "problems", "episodes_each", "sampling_seed", "sampling_place",
                                                                              "block_problems", "taken_by_the_rule", "taken_by_the_runs_minimum")},
                           "what": f"HA, the held-aside set, is ROUND 1's: problems `{cycle.start}` failed in every map attempt that no search took and no model trains on. "
                                   f"Round 1's run measured `{cycle.start}` there (its rows are copied here); this run's four models and, when its adapter is on the box, "
                                   f"`{STORED}` each make the same {theirs['the_held_aside']['episodes_each']} fresh attempts at every one, with THAT run's sampling seed",
                           "file": HELD_ASIDE_FILE, "measured": list(measured_aside), "sets": {name: held_aside_set(name, L6) for name in measured_aside},
                           "of_the_start": {"file": stored_file(cycle.start, HELD_ASIDE_PART, L6), "set": held_aside_set(cycle.start), "run": str(found["directory"]),
                                            "successes": sum(row["resolved"] for row in read["start_on_the_set"])}},
        "the_stored_model": {"what": f"`{STORED}` (L4p: a whole second pass over the pretraining file, the best pretraining-only model stored), ONLY READ and only when it is "
                                     "on the box: its stored rows on G, read beside, and its adapter, from which it is measured on the held-aside set. NOT THERE otherwise, "
                                     "and nothing is refused for it", **{key: value for key, value in of_the_stored.items() if key != "on_g"},
                             "sets_on_g": [stored_file(STORED, sampling_["name"], L6) for sampling_ in plan["goal_samplings"]] if of_the_stored["there"] else []},
        "smoke": {"what": f"{SMOKE_VARIABLE}: a smoke run alone. Round 1's smoke run reads INCONCLUSIVE (a training of two rows has no loss to compare), which a real run is "
                          "refused for, and is read all the same; search 2 attempts at least this many problems, the map's other problems that no search took and that are "
                          "not held aside joining when the rule gives fewer; one round-2 solve is enough; `long_ordinary` may take its ordinary solves as loosely. Every "
                          "problem and row so taken is marked as NOT the rule's", "problems_at_least": at_least,
                  "round_one_read_though_inconclusive": bool(report.get("inconclusive")), "round_ones_failed_checks": list(report.get("failed_checks") or [])}
        if at_least else None,
        "pretraining_file": str(ladder_l4.pretraining_file()), "pretraining_file_sha256": read["sha256"], "recipe": {**theirs["recipe"], "adapter_seed": seed},
        "goal_set": len(goal_problems), "goal_set_again": len(again), "rung_problems": len(rung_problems), **plan,
        "attempts_a_goal_problem": sum(sampling_["episodes"] for sampling_ in plan["goal_samplings"]), "parts": read["parts"],
        "stored_models": {cycle.start: {"run": str(start.directory)}, M1: {"run": str(found["directory"])},
                          **({STORED: {"run": of_the_stored["run"]}} if of_the_stored["there"] else {})},
        "never_solved_file": NEVER_SOLVED_FILE, "spread_file": SPREAD_FILE, "contradicted_side_setting": config["ladder_loop"]["episode"].get("contradicted_side", "all"),
        "start_recipe": theirs["start_recipe"], "start_run_read": theirs["start_run_read"]}
    store.mark_done(SEARCH_PREPARE, summary)
    _say(f"L6's second search prepared, seed {seed}, from round 1 ({found['directory'].name}: {len(of_round_one['solved']):,} of {len(drawn):,} problems solved; "
         f"{len(aside):,} held aside): M(1) is `{M1}` of that run; search 2 attempts {len(searched):,} problems, {by_part[PART_A]:,} search 1 left unsolved and "
         f"{by_part[PART_B]:,} of the {partition['candidates']:,} candidates in neither draw"
         + (f" ({summary['the_search']['taken_by_the_runs_minimum']} of them NOT by the rule: a smoke run)" if at_least else "")
         + f", {cycle.attempts} attempts each, sampling seed {sampling}; `{STORED}` " + ("is on the box" if of_the_stored["there"] else f"is NOT THERE ({of_the_stored['why']})"))
    return summary


# -------------------------------------------------------------------------------------------- the search
def ladder_l6_search(config: dict) -> dict:
    """SEARCH 2, by the deep search's own body (`ladder_l5.search_deep`): M(1), from its adapter in round 1's run."""
    cycle = the_settings(config)
    store = _store(config, cycle)
    if store.is_done(SEARCH):
        return store.done_summary(SEARCH)
    _need(store, SEARCH_PREPARE, "search 2", ENTRY_STAGE)
    prepared = store.done_summary(SEARCH_PREPARE)
    of_round_one = prepared["round_one"]
    return ladder_l5.search_deep(config, store, prepared, the_start(config, cycle), prepared["the_search"], cycle.attempts, search_seed(config, cycle), of_round_one["model"],
                                 Path(of_round_one["adapter"]), THE_SEARCH)


# ------------------------------------------------------------------------------------------ the four sets
def build_the_sets(round_1: list[dict], round_2: list[dict], ordinary: list[dict], pretraining: list[dict], seed: int, loosely: bool = False) -> dict:
    """The four training sets (`l6.the_sets`), each row numbered by its place in ITS set. Refused (ValueError) as the
    rule refuses, and when a published row carries text (`l4_rows2.refuse_text`)."""
    numbered = lambda rows: [{"row": position, **{key: value for key, value in row.items() if key != "row"}} for position, row in enumerate(rows)]      # noqa: E731
    sets = {model: numbered(own) for model, own in the_sets(round_1, round_2, ordinary, pretraining, seed, loosely).items()}
    for model, own in sets.items():
        refuse_text(own, f"the training set of `{model}`")
    return sets


def ladder_l6_prepare(config: dict) -> dict:
    cycle = the_settings(config)
    store = _store(config, cycle)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    for step in (SEARCH_PREPARE, SEARCH):
        _need(store, step, "L6's four training sets", ENTRY_STAGE)
    read, seed, start = store.done_summary(SEARCH_PREPARE), training_seed(config), the_start(config, cycle)
    of_the_map, of_the_search, of_round_one = read["the_map"], read["the_search"], read["round_one"]
    smoke, batch = bool(read.get("smoke")), config["training"]["effective_batch"]

    # ---- the deep solves of both rounds: one verified attempt of every problem a search solved, drawn by THIS task's seed with L5's own function
    searched, aside = store.read_rows(SEARCH_PROBLEMS_FILE), [row["problem_id"] for row in store.read_rows(HELD_ASIDE_FILE)]
    of_search_2, first_run = {row["problem_id"]: row for row in searched}, Path(of_round_one["run"])
    of_search_1 = {row["problem_id"]: row for row in _rows(first_run / ladder_l5.SEARCH_PROBLEMS_FILE)}
    try:
        round_1 = [{**row, "origin": ROUND_1, "round": 1, "map_k": of_search_1[row["problem_id"]]["map_k"], "taken_by": of_search_1[row["problem_id"]]["taken_by"]}
                   for row in solve_rows([row for row in _rows(first_run / "problems.jsonl") if row["set"] == ladder_l5.SEARCH_SET],
                                         ladder_ceiling._stored_attempts(first_run, ladder_l5.SEARCH_SET), _rows(first_run / f"episodes_{ladder_l5.SEARCH_SET}_problems.jsonl"),
                                         seed, DEEP, of_the_map["episodes_each"])]
        results = _results(store, SEARCH_SET)
        round_2 = [{**row, "origin": ROUND_2, "round": 2, "part": of_search_2[row["problem_id"]]["part"], "map_k": of_search_2[row["problem_id"]]["map_k"],
                    "taken_by": of_search_2[row["problem_id"]]["taken_by"]}
                   for row in solve_rows([row for row in store.read_rows("problems.jsonl") if row["set"] == SEARCH_SET], _attempts(store, SEARCH_SET), results, seed, DEEP)]
    except ValueError as error:
        raise RuntimeError(f"{error} (round 1's search in {first_run.name}, or search 2). Nothing was written.") from error
    if len(round_1) != of_round_one["solved"]:
        raise RuntimeError(f"round 1's stored search ({first_run.name}) now gives {len(round_1)} deep solves and this run was prepared on {of_round_one['solved']}. Nothing "
                           "was written.")
    table = second_search_table(searched, results, round_2, of_the_search["episodes_each"], of_the_map["episodes_each"])
    enough = enough_round_two_solves(len(round_2), of_the_search["minimum_round_two_solves"])
    solves = [{**row, "in_the_sets": enough["passes"]} for row in (*round_1, *round_2)]
    carried = {key: read[key] for key in CARRIED}
    lines_of_the_solves = {"round_1": verified_lines(lines_counted(round_1)), "round_2": verified_lines(lines_counted(round_2))}
    found = {**table, "lines_of_the_deep_rows": lines_of_the_solves}
    count = len(round_1) + len(round_2)
    if not enough["passes"]:            # nothing is trained: the later steps say so and return, and the search is the finding
        summary = {**carried, "stand_in_engine": _stand_in(), "enough_round_two_solves": enough, "round_one_solves": len(round_1), "round_two_solves": len(round_2),
                   "deep_solves": 0, "rows_a_set": 0, "training_sets": {}, "the_search_found": found,
                   "trainings": {"run": False, "why": f"search 2 found {len(round_2)} round-2 solves and {enough['minimum']} are asked: nothing is trained. No set is built "
                                                      "and no model is trained or measured: the search is the finding"}}
        store.write_rows(SOLVES_FILE, solves)
        store.mark_done(PREPARE, summary)
        _say(f"L6: search 2 found {len(round_2)} round-2 solves of {len(searched):,} problems, FEWER THAN THE {enough['minimum']} asked: no set is built and the four "
             "trainings are not run. The search is the finding")
        return summary

    # ---- the ordinary solves: map problems the start model solved, one verified map attempt each drawn by the seed, MATCHED IN LENGTH to the deep rows bin by bin
    directory = Path(read["map_run"])
    map_rows = _rows(directory / f"episodes_{of_the_map['set']}_problems.jsonl")
    if {str(k): times for k, times in sorted(Counter(row["resolved"] for row in map_rows).items())} != of_the_map["problems_by_k"]:
        raise RuntimeError(f"the map {directory.name} is not the map round 1's search was prepared on (its problems by k are no longer {of_the_map['problems_by_k']}). "
                           "Nothing was written.")
    try:
        attempted = {*of_search_1, *of_search_2}             # the problems either search attempted: by the rule none of them is an ordinary solve (each was failed in every map attempt)
        map_problems = {row["problem_id"]: row for row in _rows(directory / "problems.jsonl") if row["set"] == of_the_map["set"]}
        by_the_rule = [row["problem_id"] for row in map_rows if row["resolved"] >= cycle.least and row["problem_id"] not in attempted and row["problem_id"] not in aside]
        beside = [row["problem_id"] for row in map_rows if row["resolved"] > 0 and row["problem_id"] in attempted and row["problem_id"] not in aside] if smoke else []
        k_of = {row["problem_id"]: row["resolved"] for row in map_rows}
        offered = [{**row, "origin": ORDINARY, "map_k": k_of[row["problem_id"]]}
                   for row in solve_rows([map_problems[key] for key in (*by_the_rule, *beside)], ladder_ceiling._stored_attempts(directory, of_the_map["set"]), map_rows, seed, SHALLOW)]
        matched, bins = matched_in_length([*round_1, *round_2], offered, cycle.edges, seed, loosely=smoke)
        ordinary = [{**row, "taken_by": THE_MINIMUM if row["problem_id"] in beside else row["taken_by"]} for row in matched]
        pretraining, sha256 = ladder_l4_start.pretrained_on(start.directory, read["pretraining_file_sha256"], start.name)
        sets = build_the_sets(round_1, round_2, ordinary, pretraining, seed, loosely=smoke)
        refuse_overlap(aside, sorted(attempted), sets)          # THE HELD-ASIDE SET IS THE TEST: neither search attempted one of its problems, and no set holds a row of one
        heldout = {row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[0]} | {row["problem_id"] for row in store.read_rows(ladder_l4.GROUPS_FILE)}
        base_map, half_seed = {row["problem_id"] for row in map_rows}, config["ladder_loop"]["l4"]["half_seed"]
        counted = {model: refuse_barred(sets[model], heldout, half_seed, base_map, f"the training set of `{model}`") for model in cycle.models}
    except ValueError as error:
        raise RuntimeError(f"{error} (the map {directory.name}; `{start.name}`'s {start.directory.name}). Nothing was written.") from error

    # ---- from here on files are written
    store.write_rows(SOLVES_FILE, solves)
    store.write_rows(ORDINARY_FILE, [{key: row[key] for key in ("problem_id", "id", "map_k", "lines", "bin", "taken_by")} for row in ordinary])
    for model in cycle.models:
        store.write_rows(THE_STAGE.training_file(model), sets[model])
    order = (f"the hash order L4t's rehearsal draw uses (a content hash of the task's seed and each row's id among the rows `{cycle.start}` was pretrained on); trained in the "
             "arm's content-hash order. A published row's text stays in the pretraining file")
    short = [entry for entry in bins if entry["shortfall"]]
    lines_of = {"round_1_rows": lines_of_the_solves["round_1"], "round_2_rows": lines_of_the_solves["round_2"], "ordinary_rows": verified_lines(lines_counted(ordinary)),
                "the_published_rows_beside_them": verified_lines(lines_counted([row for row in sets[TWO_ROUNDS] if row["origin"] == PRETRAINING])),
                "published2_rows": verified_lines(lines_counted(sets[PUBLISHED2]))}
    not_the_rules = {TWO_ROUNDS: sum(row["taken_by"] != THE_RULE for row in (*round_1, *round_2)), LONG_ORDINARY: sum(row["taken_by"] != THE_RULE for row in ordinary)}
    said = {PUBLISHED2: f"the first {2 * count:,} rows of {order}, and nothing else: the bar at equal steps",
            ONE_ROUND: f"round 1's {len(round_1):,} deep solves (one verified attempt each, drawn by the seed) and the first {2 * count - len(round_1):,} rows of {order}",
            TWO_ROUNDS: f"the {count:,} deep solves of both rounds ({len(round_1):,} and {len(round_2):,}; one verified attempt each, drawn by the seed; no cap) and the first "
                        f"{count:,} rows of {order}: M(2)",
            LONG_ORDINARY: f"{count:,} ORDINARY solves (map problems `{cycle.start}` solved in at least {cycle.least} of its {of_the_map['episodes_each']} map attempts, one "
                           f"verified map attempt each, drawn by the seed to have the line counts of `{TWO_ROUNDS}`' own {count:,} rows bin by bin) and the SAME first "
                           f"{count:,} rows of {order}"}
    summary = {
        **carried, "stand_in_engine": _stand_in(), "enough_round_two_solves": enough, "round_one_solves": len(round_1), "round_two_solves": len(round_2), "deep_solves": count,
        "rows_a_set": 2 * count, "the_search_found": found, "trainings": {"run": True},
        "matched_in_length": {"what": f"the ordinary solves of `{LONG_ORDINARY}` against the lines of `{TWO_ROUNDS}`' own rows, bin by bin (the lines of a proof, counted as a "
                                      "model's verified proofs are): how many rows of a bin are wanted, how many ordinary solves the map offers there, what longer bins "
                                      "asked of it beside, what was taken, and its shortfall, which is asked of the next SHORTER bin",
                              "line_bins": list(cycle.edges), "ordinary_solves_the_map_offers": len(by_the_rule), "bins": bins, "bins_short": [entry["lines"] for entry in short],
                              "rows_short": sum(entry["shortfall"] for entry in short), "rows_of_another_bin_than_wanted": sum(entry["asked_by_longer_bins"] for entry in bins)},
        "lines_of_the_rows": {"what": "the lines of the proofs of the rows trained on (counted as a model's verified proofs are): each round's deep rows, the ordinary solves' "
                                      f"rows, the N published rows that stand beside either, and `{PUBLISHED2}`'s 2N", **lines_of},
        "rows_not_taken_by_the_rule": {"what": "a smoke run alone: rows of problems the rule did not give (zero in every other run)", **not_the_rules},
        "training_sets": {model: {"what": said[model], "file": THE_STAGE.training_file(model), **described(sets[model], batch), "barred": counted[model]} for model in cycle.models},
        "deep_solves_file": SOLVES_FILE, "ordinary_solves_file": ORDINARY_FILE, "pretraining_file_sha256": sha256}
    store.mark_done(PREPARE, summary)
    of = summary["training_sets"]
    _say(f"L6 prepared, seed {seed}, from `{cycle.start}`: search 2 solved {len(round_2):,} of {len(searched):,} problems; N = {count:,} ({len(round_1):,} of round 1); "
         + "; ".join(f"`{model}` {of[model]['rows']:,} rows ({of[model]['rows_by_origin']}), {of[model]['optimizer_steps']:,} optimizer steps" for model in cycle.models)
         + "; ordinary solves matched in length: " + ("every bin filled" if not short else "SHORT in " + ", ".join(f"{entry['lines']} ({entry['shortfall']})" for entry in short)
                                                    + ", each filled from the next shorter bin")
         + (f"; A SMOKE RUN: {not_the_rules[TWO_ROUNDS]} deep rows and {not_the_rules[LONG_ORDINARY]} ordinary rows are NOT the rule's" if smoke else ""))
    return summary


# ----------------------------------------------------------------------------------------- train and measure
def ladder_l6_train(config: dict, model: str) -> dict:
    """One of the four models, by L4t's own training body (`ladder_l4_rows.train_a_model`) under this stage's names."""
    cycle = the_settings(config)
    store = _store(config, cycle)
    return ladder_l5._not_run(store, f"the training of `{model}`", THE_STAGE) or train_a_model(config, store, model, cycle.start, run_name(cycle, training_seed(config)), THE_STAGE)


def ladder_l6_measure(config: dict, model: str) -> dict:
    """One of the four models, by L4t's own measuring body (`ladder_l4_rows.measure_a_model`) under this stage's names."""
    cycle = the_settings(config)
    store = _store(config, cycle)
    return ladder_l5._not_run(store, f"the measurement of `{model}`", THE_STAGE) or measure_a_model(config, store, model, cycle.start, THE_STAGE)


# ------------------------------------------------------------------------------------- the held-aside set
def ladder_l6_held_aside(config: dict, who: str) -> dict:
    """ONE model's fresh attempts at the held-aside problems, by L5's own body (`ladder_l5.on_the_held_aside_set`): as
    many as round 1's models made, with ROUND 1's sampling seed for that set, so the model pairs by problem with every
    model measured there. `who`: one of the four trained models (the adapter this run trained and kept), or the STORED
    model, `second_full` (its adapter in L4p's run, only read). THE STORED MODEL IS NOT THERE when L4p's run was not on
    the box at the first step or its adapter is not there now: the step says so, marks nothing, and nothing is refused."""
    cycle = the_settings(config)
    store, deep = _store(config, cycle), cycle.deep
    stored = who == STORED

    def not_there(prepared: dict) -> dict | None:
        of_the_stored = prepared["the_stored_model"]
        if of_the_stored["there"] and Path(of_the_stored["adapter"]).is_dir():
            return None
        why = of_the_stored.get("why") or f"{of_the_stored['adapter']} is not there"
        _say(f"L6: `{STORED}` is NOT THERE ({why}): it is not measured on the held-aside set, and nothing is refused for it")
        return {"stage": L4, "label": LABEL, "check": "L6", "run": False, "not_there": True, "model": STORED, "why": why}

    return ladder_l5.on_the_held_aside_set(config, store, cycle.start, who, held_aside_marker(who), deep.held_aside_attempts, ladder_l5.held_aside_seed(config, deep),
                                           trained_by=None if stored else THE_STAGE.train_marker(who),
                                           directory=l4p_directory(config, cycle) / ladder_assembly.ADAPTERS / STORED if stored else None,
                                           not_made=not_there if stored else None, one=THE_STAGE)


# -------------------------------------------------------------------------------------------------- report
def ladder_l6_report(config: dict) -> dict:
    from rlvr_lean.gpu.ladder_l3c import proof_lengths
    from rlvr_lean.reporting.ladder_l6 import build_l6_report

    cycle = the_settings(config)
    store = _store(config, cycle)
    _need(store, PREPARE, "L6's report", ENTRY_STAGE)
    prepared = store.done_summary(PREPARE)
    measured = cycle.models if prepared["trainings"]["run"] else ()
    for step in (of(model) for model in measured for of in (THE_STAGE.train_marker, THE_STAGE.measure_marker, held_aside_marker)):
        _need(store, step, "L6's report", ENTRY_STAGE)
    start, samplings = cycle.start, prepared["goal_samplings"]
    # The stored model on the held-aside set: read when ITS step measured it, NOT THERE otherwise (that step marks nothing then, and the report asks for no marker of it).
    there = bool(measured) and store.is_done(held_aside_marker(STORED))
    on_the_held_aside = {name: _results(store, held_aside_set(name, L6)) for name in (*measured, *((STORED,) if there else ()))}
    if measured:        # the start model's rows there are round 1's own run's, as the first step copied them
        on_the_held_aside = {start: store.read_rows(stored_file(start, HELD_ASIDE_PART, L6)), **on_the_held_aside}
    of_the_stored = json.loads(store.path(STORED_MODELS_FILE).read_text())
    stored = {start: {RUNG_PART: store.read_rows(stored_file(start, RUNG_PART, L4)), GOAL: [store.read_rows(stored_file(start, sampling["name"], L4)) for sampling in samplings],
                      **{key: of_the_stored[start][key] for key in WROTE}}}
    beside = {who: [store.read_rows(stored_file(who, sampling["name"], L6)) for sampling in samplings] for who in prepared["stored_models"] if who != start}
    models, trains, losses, trained = read_the_trained(store, prepared, measured, THE_STAGE)
    # GIVEN, never computed here: the goal problems nothing stored has ever solved (a file of ids), and the read over L5's seeds (the spread between trainings of one kind).
    never = [row["problem_id"] for row in store.read_rows(NEVER_SOLVED_FILE)] if store.path(NEVER_SOLVED_FILE).exists() else None
    spread = json.loads(store.path(SPREAD_FILE).read_text()) if store.path(SPREAD_FILE).exists() else None
    base_map = [row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[1]]
    # The length of a held-out problem's published proof is read HERE, by the report, and by nothing before it.
    report = build_l6_report(prepared, trains, losses, trained, store.read_rows(ladder_l4.GROUPS_FILE), proof_lengths(), stored, models,
                             [row["problem_id"] for row in store.read_rows(ladder_l4.AGAIN_FILE)], config["ladder_loop"]["l4"], config["evaluation"], base_map,
                             on_the_held_aside, beside, never, spread)
    directory = store.root / ladder_assembly.ADAPTERS
    report["adapters"] = {"kept": True, "what": (f"the adapters of {' and '.join(f'`{model}`' for model in measured)} ({LABEL}) are KEPT: no step of this stage deletes one. "
                                                 if measured else "no model was trained. ") + f"`{start}`, `{M1}` of round 1 and `{STORED}` are their own runs' and were only read",
                          "directory": str(directory), "there": sorted(path.name for path in directory.glob("*") if path.is_dir()) if directory.is_dir() else []}
    _write_json(store, REPORT_FILE, report)
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    for name in (STORED_MODELS_FILE, *(THE_STAGE.loss_file(model) for model in measured)):
        store.mirror(name)
    for line in report["lines"]:
        print(line, flush=True)
    store.mark_done(REPORT, {"stage": L4, "label": LABEL, "check": "L6", "headline": report["headline"], "branches": report["branches"], "measured": list(measured),
                             "ok": report["ok"]})
    return report


def _for(step, name: str):
    def run(config: dict) -> dict:
        return step(config, name)
    return run


# In the order the stage runs them: round 1 read, search 2, the four sets; each model trained, measured and measured on the held-aside set before the next is
# trained; then the stored model on the held-aside set; then the report.
STEPS = {SEARCH_PREPARE: ladder_l6_search_prepare, SEARCH: ladder_l6_search, PREPARE: ladder_l6_prepare}
for _model in MODELS:
    STEPS[THE_STAGE.train_marker(_model)] = _for(ladder_l6_train, _model)
    STEPS[THE_STAGE.measure_marker(_model)] = _for(ladder_l6_measure, _model)
    STEPS[held_aside_marker(_model)] = _for(ladder_l6_held_aside, _model)
STEPS[held_aside_marker(STORED)] = _for(ladder_l6_held_aside, STORED)
STEPS[REPORT] = ladder_l6_report
