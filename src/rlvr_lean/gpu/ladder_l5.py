"""L5: do DEEP solves teach what ordinary solves do not? One step-matched check. Spec:
docs/spec/ladder-loop.spec.md, "L5: do DEEP solves teach what ordinary solves do not? One step-matched check" and its
"Made exact by the build". Everything it makes is labelled pretrained on published proofs. Each step is its own process
(`rlvr_lean.runner.entry`, stage `ladder_l5_deep`). A scout: one seed, NO ROUND, no assembly, no challenger.

It runs from ONE start, a model of the check of the adapter's rank (`ladder_loop.l5.start`: `pre_r64`), every model at
ITS rank. Three runs are only READ, on the box:

  the start model's own map   `ladder_l4_map_<start>_seed<pretraining seed>`, made by `ladder_l4b_map`: each base-map
                              problem's k of the map's attempts, its problems with their exact negations as the map
                              sampled them, and its stored attempts with their texts (an ordinary solve's row is one)
  L4t2's from that start      `ladder_l4_rows2_<start>_seed<N>`, FINISHED, as L4p reads it: the problems of the measured
                              sets, the held-out groups, G', how a model is measured, the rank it trained at
  the start model's           the run of the check that made it: the adapter, the rank and alpha it recorded, its
                              sampling seeds, its per-problem rows, and the rows it was pretrained on

The run directory is its own (`ladder_l5_deep_<start>_seed<N>`).

  ladder_l5_search_prepare    REFUSES, naming the task to run, a run that lacks what is read, a map that is not the base
                              map's problems at the stored map's attempts and seed, L4t2's run of another seed or start,
                              stored rows that would not pair, a G' or a pretraining file that is not L4t2's, and a
                              sampling seed another stored sampling used. DRAWS the deep search's problems: the map
                              problems the start model failed in EVERY map attempt, a seeded draw; and THE HELD-ASIDE
                              SET, a seeded draw of its own among the candidates that draw did NOT take. No GPU, no Lean
  ladder_l5_search            THE DEEP SEARCH: the start model, served at its rank, makes `deep.attempts` more attempts at
                              each drawn problem with a sampling seed of the search's own, by `ladder_loop.run_episodes` as
                              a round's episodes run (the prompt, the limits, one side by the audit rule, Lean's verdict,
                              the soundness alarm; block-resumable). A search Lean did not answer is not kept
  ladder_l5_prepare           BUILDS and stores the three training sets (`domain/ladder_round/l5.py`): a problem the search
                              solved is a DEEP solve, its row ONE of its verified attempts drawn as a round draws one.
                              FEWER DEEP SOLVES THAN `deep.minimum`: no set is built, the three trainings are NOT run (their
                              steps say so and train nothing), and the report says that the search is the finding
  ladder_l5_held_aside_start  THE HELD-ASIDE SET, measured by FOUR models with ONE sampling seed of its own: the start
  ladder_l5_held_aside_<model>  model itself (its step is called `start`; its 8 stored map attempts CHOSE the set, so it
                              is measured again there), and each trained model right after its own measure step:
                              `deep.held_aside_attempts` fresh attempts at each held-aside problem, by
                              `ladder_loop.run_episodes` as the search runs. A set Lean did not answer is measured again,
                              and no other, when the task is queued again
  ladder_l5_train_<model>     L4t's OWN two step bodies (`ladder_l4_rows.train_a_model`, `measure_a_model`) under this
  ladder_l5_measure_<model>   stage's names, for `published`, then `shallow`, then `deep`: FROM THE START ADAPTER at its
                              rank, one pass, the arm's recipe; then the model as L4t's were measured
  ladder_l5_report            the read fixed before the run (`reporting/ladder_l5.py`): the primary ON THE HELD-ASIDE SET

Every step is resumable and a step of its own. NO STEP DELETES AN ADAPTER.

A FURTHER SEED (spec "L5, further seeds"): the deep search, the held-aside set and the start model's measurement on it are
made ONCE, by the task of the seed `ladder_loop.l5.search_seed` names. A task of ANOTHER seed runs the same steps and
writes a run of its own (`ladder_l5_deep_<start>_seed<N>`), and of those three things it samples NOTHING again:

  ladder_l5_search_prepare    READS, and never writes, the search's own FINISHED run (`read_the_search`) and every stored
                              run that task read (L4t2's of THAT seed). REFUSED, naming the task to run, before anything is
                              written: that run missing, unfinished, with a report not to be read or INCONCLUSIVE, or made
                              with another start model, rank, search or held-aside set than this task would make
  ladder_l5_search            runs NO search, and `ladder_l5_held_aside_start` measures NOTHING: each says so, marks nothing
  ladder_l5_prepare           by THE TASK'S seed, from the search's stored attempts: which deep solves the cap keeps and
                              which verified attempt of each; which ordinary solves and which attempt; the published rows
                              (the first D and the first 2D of the hash order of that seed); each set's training order
  the three models            trained, measured (the rungs and G with the sampling seeds the start model's run recorded;
                              the held-aside set with the SEARCH's seed's sampling seed) and reported as the search's own
                              seed's are: a problem's attempts differ between the seeds by the model alone

At the search's own seed every step, summary and file is what it was. THE READ OVER THE SEEDS is
`reporting.ladder_l5.over_seeds`, run by `python -m rlvr_lean.tools.ladder_l5_seeds` on the pulled task directories.

TWO STAGES CALL FOUR OF ITS FUNCTIONS (spec "L6: a second round. Does the cycle compound?"; `gpu/ladder_l6.py`, whose
round 1 is this stage's finished run): what the first step reads, holds and draws before it writes (`read_for_the_search`,
with `read_the_search` and `held_to_the_search` under the words an `OfTheSearch` gives), the deep search's body
(`search_deep`, under the names a `Search` gives), one model's measurement on the held-aside set
(`on_the_held_aside_set`) and what a step returns when nothing was trained (`_not_run`). Every default is this stage's:
change a body and both stages change; this stage writes what it wrote. L5n (`gpu/ladder_l5_noise.py`) and L5d
(`gpu/ladder_l5_draws.py`) call the first of them too, and L5d the two functions the three sets step draws by
(`deep_solves_by_the_seed`, `sets_by_the_seed`): a seed of L5d builds its sets as a further seed of this stage does.

A PUBLISHED ROW is named by its id, its problem and its place in the file, never with its text: the training step reads
the text from the pretraining file (`data/ladder_l4/pretraining.jsonl`, not distributed), held to the SHA-256 the start
model's run recorded. A deep row and an ordinary solve's row are the START MODEL'S OWN proofs, verified by Lean, of
base-map problems, and are stored with their text as a round's training examples are.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from rlvr_lean.domain.ladder_round.l4 import check_start_map, goal_set_again
from rlvr_lean.domain.ladder_round.l4_rows import PRETRAINING, THE_RULE
from rlvr_lean.domain.ladder_round.l4_rows2 import REHEARSE_ONLY, refuse_text
from rlvr_lean.domain.ladder_round.l5 import (
    DEEP,
    MODELS,
    PUBLISHED,
    SHALLOW,
    capped,
    enough_deep_solves,
    failed_every_attempt,
    held_aside,
    ordinary_draw,
    refuse_barred,
    refuse_overlap,
    search_problems,
    search_table,
    solve_rows,
    the_sets,
)
from rlvr_lean.domain.ladder_round.read import GOAL
from rlvr_lean.gpu import ladder_assembly, ladder_ceiling, ladder_l2, ladder_l4, ladder_l4_rows, ladder_l4_rows2, ladder_l4_start, ladder_loop
from rlvr_lean.gpu.ladder_ceiling import RUNG_PART, Reading, _need, forget_measurement, goal_set, rung_set, stored_file
from rlvr_lean.gpu.ladder_dose import _episodes, _rows, _runs
from rlvr_lean.gpu.ladder_l3d1 import WROTE
from rlvr_lean.gpu.ladder_l4 import L4, LABEL, _say
from rlvr_lean.gpu.ladder_l4_rows import described, lines_counted, measure_a_model, read_the_trained, train_a_model
from rlvr_lean.gpu.ladder_l4_start import Start, config_from, start_set
from rlvr_lean.gpu.ladder_round import Engines, _attempts, _results, _stand_in, _write_json, training_seed
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.reporting.ladder_ceiling import lean_did_not_answer, verified_lines

L5 = "l5"                                                   # what this stage's own files are called
STAGE = "ladder_l5"                                         # what its steps begin with
ENTRY_STAGE = f"{STAGE}_deep"                               # the stage, as `runner.entry` names it; its run directory begins with it
RUN_VARIABLE = "RLVR_LEAN_LADDER_L5_RUN"                    # another run directory than `ladder_l5_deep_<start>_seed<seed>` (a smoke run)
L4T2_RUN_VARIABLE = "RLVR_LEAN_LADDER_L5_L4T2_RUN"          # another run directory of L4t2 than `ladder_l4_rows2_<start>_seed<seed>` (a smoke run: L4t2's smoke run)
SMOKE_VARIABLE = "RLVR_LEAN_LADDER_L5_SMOKE_PROBLEMS"       # a SMOKE run alone (refused elsewhere): the search attempts at least this many problems, by a looser rule when the rule gives fewer
SETTING = "ladder_loop.l5"
THE_STAGE = ladder_l4_rows.Stage(name=ENTRY_STAGE, steps=STAGE, files=L5, reader="L5", start_said_by=f"{SETTING}.start")
SEARCH_PREPARE, SEARCH, PREPARE, REPORT = f"{STAGE}_search_prepare", f"{STAGE}_search", THE_STAGE.prepare, f"{STAGE}_report"
SEARCH_SET = f"{L5}_deep_search"                            # the set the deep search's episodes are stored under
HELD_ASIDE = f"{STAGE}_held_aside"                          # what the steps that measure the held-aside set begin with: `..._start`, then one a trained model
THE_START = "start"                                         # the start model in the name of ITS held-aside step (a step's name is registered without a config)
HELD_ASIDE_FILE = f"{L5}_held_aside_problems.jsonl"         # the held-aside problems, in the draw's order, each with the map's k and why it was taken
HELD_ASIDE_PART = "held_aside"                              # A FURTHER SEED: the start model's rows on the held-aside set, in the name of the file they are copied to
REPORT_FILE = "report_ladder_l5.json"
STORED_MODELS_FILE = f"{L5}_stored_models.json"             # what the start model wrote, from its own run's marker
SEARCH_PROBLEMS_FILE = f"{L5}_search_problems.jsonl"        # the problems the deep search attempts, in the draw's order, each with the map's k and why it was taken
SOLVES_FILE = f"{L5}_deep_solves.jsonl"                     # every deep solve: its row, the place of its first success, whether a set holds it
ORDINARY_FILE = f"{L5}_ordinary_solves.jsonl"               # the ordinary solves `shallow` is trained on, in the draw's order, each with the map's k and why it was taken
NEVER_SOLVED_FILE = f"{L5}_never_solved.jsonl"              # GIVEN, not made here: one `problem_id` a row, the goal problems nothing stored has ever solved
PLAN = ladder_l4_rows2.PLAN                                 # how a model is measured, as a prepare step records it
READER = ladder_l4_start.Reader(reads="L5 makes no round and no pretraining of its own: it reads what", carry="L5")
READING = Reading(reader="L5", models="L5's models", not_read="that set stands on the other side of a comparison, and no report of this run could be read against it",
                  setting=ladder_l4.SETTING)
# What the three sets step carries on from the step that read the runs: what L4t's two bodies and the report read of a prepare step.
CARRIED = ("stage", "label", "check", "seed", "start", "start_adapter", "start_run", "l4t2_run", "map_run", "models", "effective_batch", "rows_the_start_was_pretrained_on",
           "pretraining_file", "pretraining_file_sha256", "recipe", "goal_set", "goal_set_again", "rung_problems", *PLAN, "attempts_a_goal_problem", "parts", "stored_models",
           "never_solved_file", "contradicted_side_setting", "start_recipe", "start_run_read", "the_map", "the_search", "the_held_aside", "smoke")
FURTHER = "a_further_seed"                                  # ... and, of a further seed alone, what its first step recorded of being one (no run of the search's own seed has the key)
# The deep search's table (`l5.search_table`) without its words: what a further seed finds again in the stored search, held to what the search's own task counted.
SEARCH_TABLE = ("problems", "taken_by_the_rule", "taken_by_the_runs_minimum", "solved", "share_solved", "solved_by_first_success", "episodes_each", "attempts_before_each",
                "episodes", "attempts", "successes", "per_1000", "without_an_answer")


@dataclass(frozen=True)
class Deep:
    """`ladder_loop.l5`, as the stage's steps read it."""
    start: str                          # the ONE start adapter: it makes the deep search, and every model here is trained from it at its rank
    models: tuple[str, ...]             # the three trained models, in the order they are trained and measured
    problems: int                       # the deep search's seeded draw among the map problems the start model failed every time
    attempts: int                       # ... and the attempts it makes at each (beside the map's)
    sampling_place: int                 # ... its sampling seed is `round.sampling_seed` + 100 x the task's seed + this place, which no other sampling has
    held_aside: int                     # the held-aside set: a seeded draw of this many of the candidates the search's draw did not take
    held_aside_attempts: int            # ... the fresh attempts each of the four models makes at each of its problems
    held_aside_place: int               # ... the place of ITS sampling seed, one seed for the four models, and no other sampling's
    cap: int                            # D is the deep solves, at most this many
    minimum: int                        # with fewer deep solves the three sets are not built and nothing is trained
    least: int                          # an ordinary solve: a map problem the start model solved in `least` to `most` of its map attempts
    most: int
    l4t2: ladder_l4_rows2.Rows2         # L4t2's settings as ITS task reads them: its finished run is read for how a model is measured
    seed_of_the_search: int             # the seed whose task makes the deep search, the held-aside set and the start model's measurement on it; a task of another seed READS that run


def the_settings(config: dict) -> Deep:
    """THE one place that reads `ladder_loop.l5`. Refused (ValueError): models that are not the stage's (its steps are
    registered for them: change both together); a start that is not the one L4t2 ran from (its run is read for the
    measured sets and G'); a size that is not a whole number of at least 1; a minimum above the cap; `least` above
    `most`; one sampling place for the search and for the held-aside set; a seed of the search that is no seed."""
    settings = config["ladder_loop"]["l5"]
    start, models, deep, shallow = settings["start"], tuple(settings["models"]), settings["deep"], settings["shallow"]
    if models != MODELS:
        raise ValueError(f"{SETTING}.models is {list(models)} and the stage trains and measures {list(MODELS)}: change both together")
    of_the_search = settings["search_seed"]
    if isinstance(of_the_search, bool) or not isinstance(of_the_search, int) or of_the_search < 0:
        raise ValueError(f"{SETTING}.search_seed is {of_the_search!r}: it is the seed of the ONE task that makes the deep search, a whole number of at least 0 (a task of "
                         "another seed reads that task's run)")
    l4t2 = ladder_l4_rows2.the_settings(config)
    if start != l4t2.start:
        raise ValueError(f"{SETTING}.start is `{start}` and {ladder_l4_rows2.SETTING}.start is `{l4t2.start}`: L4t2's finished run from this start is read for the "
                         "measured sets, G' and how a model is measured")
    sizes = {"deep.problems": deep["problems"], "deep.attempts": deep["attempts"], "deep.sampling_place": deep["sampling_place"], "deep.held_aside": deep["held_aside"],
             "deep.held_aside_attempts": deep["held_aside_attempts"], "deep.held_aside_sampling_place": deep["held_aside_sampling_place"], "deep.cap": deep["cap"],
             "deep.minimum": deep["minimum"], "shallow.solved_at_least": shallow["solved_at_least"], "shallow.solved_at_most": shallow["solved_at_most"]}
    wrong = [f"{SETTING}.{key} is {value!r}" for key, value in sizes.items() if isinstance(value, bool) or not isinstance(value, int) or value < 1]
    if wrong or deep["minimum"] > deep["cap"] or shallow["solved_at_least"] > shallow["solved_at_most"] or deep["sampling_place"] == deep["held_aside_sampling_place"]:
        raise ValueError("; ".join(wrong or [f"{SETTING}.deep.minimum is {deep['minimum']} and deep.cap {deep['cap']}; shallow.solved_at_least is {shallow['solved_at_least']} and "
                                             f"solved_at_most {shallow['solved_at_most']}; deep.sampling_place is {deep['sampling_place']} and deep.held_aside_sampling_place "
                                             f"{deep['held_aside_sampling_place']}"]) + ": each is a whole number of at least 1, the minimum at most the cap, the least at most "
                         "the most, and the held-aside set's sampling seed is not the search's")
    return Deep(start=start, models=models, problems=deep["problems"], attempts=deep["attempts"], sampling_place=deep["sampling_place"], held_aside=deep["held_aside"],
                held_aside_attempts=deep["held_aside_attempts"], held_aside_place=deep["held_aside_sampling_place"], cap=deep["cap"], minimum=deep["minimum"],
                least=shallow["solved_at_least"], most=shallow["solved_at_most"], l4t2=l4t2, seed_of_the_search=of_the_search)


def run_name(deep: Deep, seed: int) -> str:
    """The stage's run directory, with the start adapter's name in it: `ladder_l5_deep_pre_r64_seed<N>`."""
    return f"{ENTRY_STAGE}_{deep.start}_seed{seed}"


def a_further_seed(config: dict, deep: Deep) -> bool:
    """Whether this task is A FURTHER SEED (spec "L5, further seeds"): its seed is not the one whose task makes the deep
    search (`ladder_loop.l5.search_seed`). It runs no search and does not measure the start model on the held-aside
    set: it reads the run of the task that did. A task of the search's own seed is that task, and nothing here moves."""
    return training_seed(config) != deep.seed_of_the_search


def own_run(config: dict, deep: Deep, seed: int | None = None) -> str:
    """This task's run directory: the seed's own, or the one the task names (a smoke run). A FURTHER SEED of a run the
    task names writes beside it, `<the named run>_seed<N>`: the named run is the search's own, which it only reads.
    `seed` (L5n, which reads the runs of several seeds of this stage; None, every caller here: the task's own): the
    run directory of the task of THAT seed."""
    named = os.environ.get(RUN_VARIABLE)
    if seed is None:
        seed, further = training_seed(config), a_further_seed(config, deep)
    else:
        further = seed != deep.seed_of_the_search
    if not named:
        return run_name(deep, seed)
    return f"{named}_seed{seed}" if further else named


def search_directory(config: dict, deep: Deep) -> Path:
    """The run whose task makes the deep search, the held-aside set and the start model's measurement on it: this
    task's own at the search's seed; at A FURTHER SEED the run of that seed (or the run the task names), ONLY READ."""
    return _runs(config) / (os.environ.get(RUN_VARIABLE) or run_name(deep, deep.seed_of_the_search))


def _store(config: dict, deep: Deep) -> ArtifactStore:
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    return ArtifactStore(_runs(config) / own_run(config, deep), Path(mirror) if mirror else None)


def l4t2_directory(config: dict, deep: Deep) -> Path:
    """L4t2's run from this stage's start, by L4t2's own name for it (or the run the task names: a smoke run). Only read.
    It is the run of THE SEARCH's seed, at every seed of this stage: a further seed reads what the search's task read."""
    return _runs(config) / (os.environ.get(L4T2_RUN_VARIABLE) or ladder_l4_rows2.run_name(deep.l4t2, deep.seed_of_the_search))


def the_start(config: dict, deep: Deep) -> Start:
    """The start model the setting names, where it lives and where its own map is (`ladder_l4_start.the_start`). A name
    that is no model of the check of the adapter's rank is refused (RuntimeError)."""
    return ladder_l4_start.the_start(config, deep.start, THE_STAGE.start_said_by)


def search_seed(config: dict, deep: Deep) -> int:
    """The deep search's own sampling seed: `round.sampling_seed` + 100 x the seed of the search's task +
    `deep.sampling_place`, as every measurement of the loop derives its own, at a place no other has (0 to 3, 11 to 16,
    20, 30 to 33; L4t's hot samplings are 40 and 41 at seed 0). The seed of the search's task is the task's own, unless
    this task is a further seed, which makes no search."""
    return config["ladder_loop"]["round"]["sampling_seed"] + 100 * deep.seed_of_the_search + deep.sampling_place


def held_aside_seed(config: dict, deep: Deep) -> int:
    """The held-aside set's own sampling seed, ONE for the four models that are measured on it (so they pair by problem
    on the same random numbers): `round.sampling_seed` + 100 x the seed of the search's task +
    `deep.held_aside_sampling_place`. Not the search's, and no stored sampling's. A FURTHER SEED's three models are
    measured there with THIS seed too, the one the start model was measured with: its attempts are not made again."""
    return config["ladder_loop"]["round"]["sampling_seed"] + 100 * deep.seed_of_the_search + deep.held_aside_place


def block_of(config: dict, attempts: int) -> int:
    """The problems a block holds when each gets `attempts` attempts: as many as make a block about as many ATTEMPTS
    as a round's block (`episode.block_problems` problems at `round.solvers` attempts each: 128 x 8). A block is what a
    kill loses and what waits on Lean together: 18 problems for the search's 56 attempts, 64 for the held-aside 16."""
    ladder = config["ladder_loop"]
    return max(1, ladder["episode"]["block_problems"] * ladder["round"]["solvers"] // attempts)


def episodes_config(config: dict, attempts: int) -> dict:
    """The config as an episode step of this stage is handed it (the deep search, a measurement of the held-aside set):
    a block holds `block_of` problems. Nothing else moves, and `config` itself is not changed."""
    ladder = config["ladder_loop"]
    return {**config, "ladder_loop": {**ladder, "episode": {**ladder["episode"], "block_problems": block_of(config, attempts)}}}


def held_aside_name(deep: Deep, who: str) -> str:
    """The model a held-aside step measures, by its own name: the start model's step is called `start`."""
    return deep.start if who == THE_START else who


def held_aside_set(name: str, files: str = L5) -> str:
    """The set a model's attempts at the held-aside problems are stored under (`files`: what the stage's own files are
    called; this stage's, unless L6 measures its models there)."""
    return f"{files}_held_aside_{name}"


def held_aside_marker(who: str) -> str:
    """The step that measures the held-aside set for one model (`start`, or a trained model), and its marker."""
    return f"{HELD_ASIDE}_{who}"


def smoke_problems() -> int:
    """A smoke run's least number of problems in the deep search; 0 for every other run: the rule and nothing else."""
    return max(0, int(os.environ.get(SMOKE_VARIABLE) or 0))


def l4t2_task(seed: int) -> str:
    """The task that writes L4t2's run, for a refusal."""
    stage = ladder_l4_rows2.ENTRY_STAGE
    return f"the task of stage `{stage}` for seed {seed} (`python -m rlvr_lean.runner.entry --stage {stage} --seeds {seed}`; for the smoke run, stage `{stage}_smoke`)"


def l4t2_set(part: str, model: str) -> str:
    """The set a model of L4t2 was measured under, in L4t2's own run directory."""
    return rung_set(model, L4) if part == RUNG_PART else goal_set(part, model, L4)


def read_the_map(config: dict, start: Start) -> dict:
    """The start model's OWN MAP, only read: what its step recorded, its per-problem rows and its problems as it
    sampled them (each with its exact negation). REFUSED (RuntimeError), naming what to run: a map run that lacks its
    marker, its rows, its problems or its stored attempts; a map of another model; one made with another sampling seed
    or number of attempts than the stored map's, or not of exactly the base map's problems (`l4.check_start_map`, as an
    arm's prepare step holds it); rows that do not agree with the map's own file; a problem whose statement is not the
    package's."""
    directory, settings = start.map_directory, config["ladder_loop"]["base_map"]
    marker = directory / f"{start.map_step}.done.json"
    lacking = [name for name in (marker.name, start.map_file, "problems.jsonl") if not (directory / name).exists()]
    recorded = json.loads(marker.read_text()) if marker.exists() else {}
    map_set = recorded.get("set")
    lacking += ladder_ceiling._missing(directory, [map_set]) if map_set else []
    if lacking:
        raise RuntimeError(f"{directory} does not hold {lacking}: the map of `{start.name}`, its problems or its stored attempts. The deep search draws its problems from "
                           f"that map, and an ordinary solve's row is one of its attempts. Run {start.map_task}; nothing was written.")
    base_map, data = ladder_loop.load_problems(ladder_loop.data_directory())[1:]
    ids = [row["problem_id"] for row in base_map]
    rows, of_the_file = _rows(directory / f"episodes_{map_set}_problems.jsonl"), _rows(directory / start.map_file)
    problems = [row for row in _rows(directory / "problems.jsonl") if row["set"] == map_set]
    try:
        check_start_map(of_the_file, recorded, ids, settings["episodes"], settings["sampling_seed"])
        check_start_map(rows, recorded, ids, settings["episodes"], settings["sampling_seed"])
    except ValueError as error:
        raise RuntimeError(f"{directory.name} is not the base map of `{start.name}`: {error}. Nothing was written.") from error
    statements = {row["problem_id"]: row["statement"] for row in base_map}
    if recorded.get("model") != start.name or {row["problem_id"]: row["resolved"] for row in rows} != {row["problem_id"]: row["resolved"] for row in of_the_file} \
            or {row["problem_id"]: row["statement"] for row in problems} != statements:
        raise RuntimeError(f"{directory.name} recorded the model {recorded.get('model')!r}; `{start.name}` is asked. Or its per-problem rows do not agree with its own file "
                           f"{start.map_file}, or its problems are not the package's base map with their statements. Nothing was written.")
    return {"directory": directory, "recorded": recorded, "set": map_set, "rows": rows, "problems": problems, "base_map": ids,
            "fixture": bool(data.get("fixture")) and bool(recorded.get("fixture"))}


# --------------------------------------------------------------------------- a further seed: the search's own run
def search_task(deep: Deep) -> str:
    """The task that makes the deep search, for a further seed's refusal."""
    seed = deep.seed_of_the_search
    return f"the task of stage `{ENTRY_STAGE}` for seed {seed} (`python -m rlvr_lean.runner.entry --stage {ENTRY_STAGE} --seeds {seed}`; for the smoke run, stage `{ENTRY_STAGE}_smoke`)"


@dataclass(frozen=True)
class OfTheSearch:
    """Who reads the deep search's FINISHED run, for what is said when it is refused. Every default is A FURTHER SEED's of
    this stage; L6 (`gpu/ladder_l6.py`), whose round 1 that run is, gives its own."""
    who: str = "a further seed"                             # "(<who>: it reads the runs the search's task of seed N read)"
    why: str | None = None                                  # why that run is read; None: a further seed's own words, which name the two seeds
    stands: str = "no further seed stands beside such a run"        # ... of a report that is not to be read, or INCONCLUSIVE
    models: str = "A further seed's models are read beside that run's, on its search and its held-aside set"        # ... of a run made otherwise
    inconclusive: bool = False                              # True (L6's SMOKE run alone): a report that reads INCONCLUSIVE is read all the same; one NOT TO BE READ never is


def read_the_search(config: dict, deep: Deep, reader: OfTheSearch = OfTheSearch()) -> dict:
    """A FURTHER SEED: the run of the task that made the deep search, ONLY READ: what its steps recorded, its report,
    the problems it searched and the ones it held aside. REFUSED (RuntimeError), naming the task to run: a run that is
    missing or UNFINISHED (a step's marker, its report, a file or a stored sampling is not there); a report that is not
    to be read or that reads INCONCLUSIVE (a search that found too few deep solves is one: nothing was trained beside
    it); a run that recorded another seed, start or models than the search's own. `reader`: who reads it (L6 does too)."""
    directory, task, seed = search_directory(config, deep), search_task(deep), training_seed(config)
    said = reader.why or (f"This task's seed is {seed} and {SETTING}.search_seed is {deep.seed_of_the_search}: a further seed makes no search of its own, and reads the search, the "
                          "held-aside set and the start model's measurement on it from that run")
    steps = (SEARCH_PREPARE, SEARCH, PREPARE, held_aside_marker(THE_START), REPORT)
    lacking = [name for name in (*(f"{step}.done.json" for step in steps), REPORT_FILE, "problems.jsonl", SEARCH_PROBLEMS_FILE, HELD_ASIDE_FILE) if not (directory / name).exists()]
    lacking += ladder_ceiling._missing(directory, [SEARCH_SET, held_aside_set(deep.start)])
    if lacking:
        raise RuntimeError(f"{directory} does not hold {lacking}: the FINISHED run of the deep search. {said}. Run {task} to its end first; nothing was written.")
    of = lambda step: json.loads((directory / f"{step}.done.json").read_text())      # noqa: E731
    searched, prepared, report = of(SEARCH_PREPARE), of(PREPARE), json.loads((directory / REPORT_FILE).read_text())
    if not report.get("ok", True) or (report.get("inconclusive") and not reader.inconclusive):
        raise RuntimeError(f"the report of the deep search's run {directory.name} is " + ("not to be read (Lean gave no verdict on too much of a set)" if not report.get("ok", True)
                           else f"INCONCLUSIVE ({', '.join(report.get('failed_checks') or [])})") + f": {reader.stands}. {said}. Queue {task} again, "
                           "or repair that run; nothing was written.")
    if (searched.get("seed"), searched.get("start"), searched.get("models")) != (deep.seed_of_the_search, deep.start, list(deep.models)):
        raise RuntimeError(f"the run {directory.name} recorded seed {searched.get('seed')}, the start {searched.get('start')!r} and the models {searched.get('models')}; "
                           f"{SETTING} names the search's seed {deep.seed_of_the_search}, the start {deep.start!r} and {list(deep.models)}. {said}. Nothing was written.")
    return {"directory": directory, "searched": searched, "prepared": prepared, "report": report, "drawn": _rows(directory / SEARCH_PROBLEMS_FILE),
            "aside": _rows(directory / HELD_ASIDE_FILE)}


def held_to_the_search(found: dict, now: Mapping, reader: OfTheSearch = OfTheSearch()) -> None:
    """A FURTHER SEED's three models stand beside the ones the search's own task trained and measured. REFUSED
    (RuntimeError) unless that run was made as this task would make it. `found`: `read_the_search`'s. `now`: what this
    task reads and would do, by the names the refusal gives: the start adapter's rank and alpha, how a model is
    measured, the pretraining file, G', the map, the search (its problems, its attempts at each, its sampling seed),
    the held-aside set (the same three), the cap and the minimum of deep solves. `reader`: who reads that run."""
    searched = found["searched"]
    of_the_search, of_the_set, recipe = searched["the_search"], searched["the_held_aside"], searched.get("start_recipe") or {}
    then = {"the start adapter's rank and alpha": [recipe.get("rank"), recipe.get("alpha")], "how a model is measured": {key: searched.get(key) for key in PLAN},
            "the pretraining file's SHA-256": searched.get("pretraining_file_sha256"), "the size of G'": searched.get("goal_set_again"),
            "the map's problems by k": searched["the_map"]["problems_by_k"], "the problems searched": found["drawn"],
            "the search's attempts a problem and its sampling seed": [of_the_search["episodes_each"], of_the_search["sampling_seed"]],
            "the held-aside problems": found["aside"],
            "the held-aside set's attempts a problem and its sampling seed": [of_the_set["episodes_each"], of_the_set["sampling_seed"]],
            "the cap and the minimum of deep solves": [of_the_search["cap"], of_the_search["minimum_deep_solves"]]}
    if set(now) != set(then):
        raise ValueError(f"what a further seed holds the search's run to is {sorted(then)}; it was given {sorted(now)}")
    short = lambda value: f"{len(value):,} problems" if isinstance(value, list) and value and isinstance(value[0], dict) else repr(value)      # noqa: E731
    differs = [f"{key} (that run: {short(then[key])}; this task: {short(value)})" for key, value in now.items() if then[key] != value]
    if differs:
        raise RuntimeError(f"the deep search's run {found['directory'].name} was made with another start model, rank, search or held-aside set than this task would make: "
                           + "; ".join(differs) + f". {reader.models}. Nothing was written.")


def counted_by_the_search(found: Mapping) -> dict:
    """What the search's own task counted on ITS search, as a task that draws from that search by another seed holds its
    own count to it: the deep solves, the rows of each kind (D), and the search's table without its words. `found`:
    `read_the_search`'s."""
    theirs = found["prepared"]
    return {"deep_solves": theirs["deep_solves"], "rows_of_each_kind": theirs["rows_of_each_kind"],
            "the_search_found": {key: value for key, value in theirs["the_search_found"].items() if key in SEARCH_TABLE}}


def the_stored_search(searched: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """The deep search of a FINISHED run, only read: (its problems as it sampled them, its per-problem rows, its
    attempts with their texts)."""
    return ([row for row in _rows(searched / "problems.jsonl") if row["set"] == SEARCH_SET], _rows(searched / f"episodes_{SEARCH_SET}_problems.jsonl"),
            ladder_ceiling._stored_attempts(searched, SEARCH_SET))


def _read_not_made(further: Mapping, step: str) -> dict:
    """What the deep search's step and the start model's held-aside step return AT A FURTHER SEED, with nothing marked
    done and nothing sampled: what each would make is the search's own run's, only read. `further`: what the first
    step recorded of being a further seed."""
    why = further["not_made_again"][step]
    _say(f"L5, a further seed: the step {step} makes NOTHING: {why}")
    return {"stage": L4, "label": LABEL, "check": "L5", "run": False, "why": why, "search_seed": further["search_seed"], "search_run": further["search_run"]}


# ------------------------------------------------------------------------------------------ search_prepare
# What `read_for_the_search` hands the step that writes, by name: what was read of the three runs, the two draws, the two sampling seeds, and (a further seed) the start
# model's rows on the held-aside set as the search's own run stored them.
READ = ("start", "l4t2", "of_the_start_run", "of_start", "recorded", "plan", "parts", "the_map", "at_least", "drawn", "aside", "sampling", "aside_sampling", "used", "rung_problems",
        "goal_problems", "kept", "again", "pretraining", "sha256", "groups", "by_k", "start_on_the_set")


def read_for_the_search(config: dict, deep: Deep, found: dict | None = None, reader: OfTheSearch = OfTheSearch(), at_least: int | None = None) -> dict:
    """What the stage's first step READS, HOLDS and DRAWS, with NOTHING WRITTEN: the three stored runs (the start
    model's own map, L4t2's finished run, the start model's), the deep search's problems and the held-aside set drawn
    from the map by the seed of the search's task, the two sampling seeds. REFUSED (RuntimeError) as the module's note
    says of `ladder_l5_search_prepare`. `found` (`read_the_search`'s: A FURTHER SEED, and L6): the search's own FINISHED
    run, HELD to what is read and drawn here (`held_to_the_search`), its two stored samplings to the draws, the sizes
    and the seeds. `at_least`: a SMOKE run's least number of problems in each draw; None (every caller of this stage):
    what this stage's own smoke-only variable says (`smoke_problems`), held here to a run the task names and the
    fixtures' map; L6's smoke run gives what the search's own run recorded, and holds ITS OWN smoke-only variable so.
    Returns what `READ` names. TWO STAGES CALL IT (this one's first step; L6's, whose round 1 the search's run is):
    change it and both change."""
    seed, of_the_pretraining = training_seed(config), ladder_l2.pretraining_seed(config)
    start, l4t2 = the_start(config, deep), l4t2_directory(config, deep)
    task = l4t2_task(deep.seed_of_the_search)

    def need(directory: Path, names, what: str, whose: str) -> None:
        lacking = [name for name in names if not (directory / name).exists()]
        if lacking:
            raise RuntimeError(f"{directory} does not hold {lacking}: {what}. {READER.reads} {whose} wrote on this box. Run that task to its end first; nothing was written.")

    # ---- L4t2's run, as L4p reads it: FINISHED, of this seed and this start, with a report that can be read
    need(l4t2, (f"{ladder_l4_rows2.PREPARE}.done.json", f"{ladder_l4_rows2.REPORT}.done.json", ladder_l4_rows2.REPORT_FILE, "problems.jsonl", ladder_l4.GROUPS_FILE,
                ladder_l4.AGAIN_FILE), "what L4t2's steps recorded, its report, its problems, its held-out groups or its G'", task)
    of_l4t2 = json.loads((l4t2 / f"{ladder_l4_rows2.PREPARE}.done.json").read_text())
    if of_l4t2.get("seed") != deep.seed_of_the_search or of_l4t2.get("start") != deep.start:
        raise RuntimeError(f"L4t2's run {l4t2.name} recorded seed {of_l4t2.get('seed')} and the start {of_l4t2.get('start')!r}; this task's seed is {seed}"
                           + (f" ({reader.who}: it reads the runs the search's task of seed {deep.seed_of_the_search} read)" if found is not None else "") + f" and {SETTING} "
                           f"names the start {deep.start!r}: the measured sets and G' read here are that run's. Nothing was written.")
    if not json.loads((l4t2 / ladder_l4_rows2.REPORT_FILE).read_text()).get("ok", True):
        raise RuntimeError(f"the report of L4t2's run {l4t2.name} is not to be read (Lean gave no verdict on too much of a set). Queue {task} again, which measures that "
                           "set again; nothing was written.")

    # ---- the start model's run: what L4t2 read of it, read again; a smoke run goes on from failed checks only where L4t2's own run recorded that IT did
    waived = bool((of_l4t2.get("start_run_read") or {}).get("checks_waived_as_l4ts_run_recorded"))
    of_the_start_run = ladder_l4_start.read_the_start_run(start, of_the_pretraining, waived=waived, reader=READER)
    of_start = of_the_start_run["prepared"]
    recorded, of_l4t2s = ladder_l4_start.recipe_recorded(of_start), of_l4t2.get("start_recipe") or {}
    if (recorded["rank"], recorded["alpha"]) != (of_l4t2s.get("rank"), of_l4t2s.get("alpha")):
        raise RuntimeError(f"the run {start.directory.name} recorded an adapter of rank {recorded['rank']} and alpha {recorded['alpha']} for `{start.name}`, and L4t2's run "
                           f"{l4t2.name} trained its models at rank {of_l4t2s.get('rank')} and alpha {of_l4t2s.get('alpha')}: every model here is at the start adapter's "
                           "rank, as L4t2's were. Nothing was written.")
    plan, of_l4t2s_plan = {key: of_start[key] for key in PLAN}, {key: of_l4t2.get(key) for key in PLAN}
    if plan != of_l4t2s_plan:
        raise RuntimeError(f"the run {start.directory.name} recorded how `{start.name}` was measured ({plan}), and L4t2's run {l4t2.name} measured its models otherwise "
                           f"({of_l4t2s_plan}): {READING.models} would not pair with both. Nothing was written.")
    parts = ladder_l4.pre_parts(of_start)
    need(start.directory, [ladder_l4.ROWS_FILE, *(name for part in parts for name in (f"episodes_{start_set(part, start)}_problems.jsonl", f"episodes_{start_set(part, start)}.done.json"))],
         f"the rows `{start.name}` was pretrained on (a published row trained on here is one of them), its per-problem rows on a set it was measured on, or that set's own marker",
         start.task)
    if not ladder_ceiling._stand_in() and not start.adapter.is_dir():
        raise RuntimeError(f"{start.adapter} is not there: the adapter `{deep.start}` makes the deep search, and every model here is trained from it. Run {start.task} again; "
                           "nothing was written.")

    # ---- the start model's own map, and the deep search's problems: the map problems it failed in EVERY map attempt, a seeded draw
    the_map = read_the_map(config, start)
    own_variable = at_least is None                 # the least number is this stage's own smoke-only variable's (held here), or its caller's (which holds its own)
    at_least = smoke_problems() if own_variable else at_least
    if at_least and own_variable and not (os.environ.get(RUN_VARIABLE) and the_map["fixture"]):
        raise RuntimeError(f"{SMOKE_VARIABLE} is set ({at_least}): it is for a SMOKE run alone, in a run directory the task names ({RUN_VARIABLE}) and on the fixtures' "
                           "base map. A run in the seed's own directory, or on the pool's problems, takes its problems by the rule and by nothing else. Nothing was written.")
    try:        # BOTH DRAWS ARE BY THE SEED OF THE SEARCH'S TASK (this task's own, unless it is a further seed: then they are made again only to hold the stored ones to them)
        drawn = search_problems(the_map["rows"], deep.problems, deep.seed_of_the_search, at_least)
        aside = held_aside(the_map["rows"], {row["problem_id"] for row in drawn}, deep.held_aside, deep.seed_of_the_search, at_least)
        refuse_overlap([row["problem_id"] for row in aside], [row["problem_id"] for row in drawn], {})
    except ValueError as error:
        raise RuntimeError(f"{error} (the map {the_map['directory'].name}). Nothing was written.") from error
    sampling, aside_sampling = search_seed(config, deep), held_aside_seed(config, deep)
    of_the_map = the_map["recorded"]
    used = {**{f"`{deep.start}`'s {name}": value for name, value in plan["sampling_seeds"].items()},
            **{f"`{deep.start}`'s sampling `{sampling_['name']}` of G": sampling_["sampling_seed"] for sampling_ in plan["goal_samplings"]},
            f"the map of `{deep.start}`": of_the_map["sampling_seed"], "the base's placing episodes (ladder_loop.goal.sampling_seed)": config["ladder_loop"]["goal"]["sampling_seed"],
            **{f"L4t's hot sampling {index}": value for index, value in enumerate(ladder_l4_rows.the_settings(config, deep.start).hot_seeds, start=1)}}
    for what, own, place in (("the deep search's", sampling, f"sampling_place {deep.sampling_place}"),
                             ("the held-aside set's", aside_sampling, f"held_aside_sampling_place {deep.held_aside_place}")):
        taken = sorted(name for name, value in used.items() if value == own)
        if taken:
            raise RuntimeError(f"{what} sampling seed is {own} ({SETTING}.deep.{place}) and {taken} already sampled with it: it has a sampling seed of its own. Nothing "
                               "was written.")

    # ---- the stored rows every read here stands on: the start model's, held to L4t2's problems, seeds and attempts; G' made again and held to L4t2's
    problems = _rows(l4t2 / "problems.jsonl")
    rung_problems = [row for row in problems if row["set"] == l4t2_set(RUNG_PART, REHEARSE_ONLY)]
    goal_problems = [row for row in problems if row["set"] == l4t2_set(plan["goal_samplings"][0]["name"], REHEARSE_ONLY)]
    ids = {RUNG_PART: [row["problem_id"] for row in rung_problems], GOAL: [row["problem_id"] for row in goal_problems]}
    measured = [(RUNG_PART, RUNG_PART, plan["rung_episodes"], plan["sampling_seeds"][RUNG_PART]),
                *((sampling_["name"], GOAL, sampling_["episodes"], sampling_["sampling_seed"]) for sampling_ in plan["goal_samplings"])]
    kept = {(deep.start, part): ladder_ceiling._stored(start.directory, start_set(part, start), ids[of], sampling_seed, episodes, f"`{deep.start}` ({part})", READING)[0]
            for part, of, episodes, sampling_seed in measured}
    again = goal_set_again(kept[deep.start, plan["goal_samplings"][0]["name"]], ids[GOAL])
    if again != [row["problem_id"] for row in _rows(l4t2 / ladder_l4.AGAIN_FILE)]:
        raise RuntimeError(f"G' made from the rows of `{deep.start}` ({start.directory.name}: {len(again)} goal problems it does not solve in its first sampling) is not the "
                           f"G' L4t2's run {l4t2.name} stored: the two runs are not of one start model's rows. Nothing was written.")
    pretraining, sha256 = ladder_l4_start.pretrained_on(start.directory, of_start["pretraining_file_sha256"], start.name)      # refused unless the pretraining file is the one it was trained on
    if sha256 != of_l4t2.get("pretraining_file_sha256"):
        raise RuntimeError(f"L4t2's run {l4t2.name} was prepared on a pretraining file of SHA-256 {of_l4t2.get('pretraining_file_sha256')} and `{start.name}` was trained on "
                           f"{sha256}: the published rows here are along the order L4t2's were drawn in. Nothing was written.")
    groups = _rows(l4t2 / ladder_l4.GROUPS_FILE)
    heldout = {row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[0]} | {row["problem_id"] for row in groups}
    barred = sorted(row["problem_id"] for row in (*drawn, *aside) if row["problem_id"] in heldout or row["problem_id"] not in the_map["base_map"])
    if barred:
        raise RuntimeError(f"{len(barred)} problems of the deep search or of the held-aside set are held out or are not base-map problems (first: {barred[0]}). Nothing "
                           "was written.")
    by_k = Counter(row["resolved"] for row in the_map["rows"])
    start_on_the_set = None
    if found is not None:
        # ---- A FURTHER SEED: the search's own run is held to what this task reads now and would do; its two stored samplings to the draws, the sizes and the seeds
        held_to_the_search(found, {
            "the start adapter's rank and alpha": [recorded["rank"], recorded["alpha"]], "how a model is measured": plan, "the pretraining file's SHA-256": sha256,
            "the size of G'": len(again), "the map's problems by k": {str(k): by_k[k] for k in sorted(by_k)}, "the problems searched": drawn,
            "the search's attempts a problem and its sampling seed": [deep.attempts, sampling], "the held-aside problems": aside,
            "the held-aside set's attempts a problem and its sampling seed": [deep.held_aside_attempts, aside_sampling],
            "the cap and the minimum of deep solves": [deep.cap, 1 if at_least else deep.minimum]}, reader)
        ids_of = lambda rows_: [row["problem_id"] for row in rows_]      # noqa: E731
        ladder_ceiling._stored(found["directory"], SEARCH_SET, ids_of(drawn), sampling, deep.attempts, f"the deep search of `{deep.start}`", READING)
        start_on_the_set = ladder_ceiling._stored(found["directory"], held_aside_set(deep.start), ids_of(aside), aside_sampling, deep.held_aside_attempts,
                                                  f"`{deep.start}` on the held-aside set", READING)[0]
    return dict(zip(READ, (start, l4t2, of_the_start_run, of_start, recorded, plan, parts, the_map, at_least, drawn, aside, sampling, aside_sampling, used, rung_problems, goal_problems,
                           kept, again, pretraining, sha256, groups, by_k, start_on_the_set), strict=True))


def ladder_l5_search_prepare(config: dict) -> dict:
    deep = the_settings(config)
    store = _store(config, deep)
    if store.is_done(SEARCH_PREPARE):
        return store.done_summary(SEARCH_PREPARE)
    seed = training_seed(config)
    found = read_the_search(config, deep) if a_further_seed(config, deep) else None        # A FURTHER SEED: the search's own FINISHED run, only read, before anything else is
    (start, l4t2, of_the_start_run, of_start, recorded, plan, parts, the_map, at_least, drawn, aside, sampling, aside_sampling, used, rung_problems, goal_problems, kept, again,
     pretraining, sha256, groups, by_k, start_on_the_set) = read_for_the_search(config, deep, found).values()
    of_the_map = the_map["recorded"]
    by_id = {row["problem_id"]: row for row in the_map["problems"]}
    search_rows = [{**by_id[row["problem_id"]], "set": SEARCH_SET} for row in drawn]
    measured_aside = (deep.start, *deep.models)             # the FOUR models that are measured on the held-aside set, in the order they are
    sampled_here = deep.models if found is not None else measured_aside        # ... and the ones THIS run samples there: a further seed does not measure the start model again
    aside_rows = [{**by_id[row["problem_id"]], "set": held_aside_set(name)} for name in sampled_here for row in aside]

    # ---- from here on files are written: the problems of every set this run samples, what its report reads of the start model, the search's problems
    def as_set(own: list[dict], set_name: str) -> list[dict]:
        return [{**row, "set": set_name} for row in own]

    own = [row for model in deep.models for row in as_set(rung_problems, rung_set(model, L4))]
    own += [row for model in deep.models for sampling_ in plan["goal_samplings"] for row in as_set(goal_problems, goal_set(sampling_["name"], model, L4))]
    store.write_rows("problems.jsonl", own + ([] if found is not None else search_rows) + aside_rows)      # every set of it is this run's (a further seed samples no search)
    store.write_rows(ladder_l4.GROUPS_FILE, groups)
    store.write_rows(ladder_l4.AGAIN_FILE, [{"problem_id": problem_id} for problem_id in again])
    for (who, part), own_rows in kept.items():
        store.write_rows(stored_file(who, part, L4), own_rows)
    _write_json(store, STORED_MODELS_FILE, {"stage": L4, "label": LABEL, deep.start: {key: of_the_start_run["measured"][key] for key in WROTE}})
    store.write_rows(SEARCH_PROBLEMS_FILE, drawn)
    store.write_rows(HELD_ASIDE_FILE, aside)
    if found is not None:       # the start model's rows on the held-aside set, as the search's own run stored them: what this run's report reads in the place of a measurement of its own
        store.write_rows(stored_file(deep.start, HELD_ASIDE_PART, L5), start_on_the_set)
    batch, candidates = config["training"]["effective_batch"], failed_every_attempt(the_map["rows"])
    ordinary = sum(count for k, count in by_k.items() if deep.least <= k <= deep.most)
    summary = {
        "stage": L4, "label": LABEL, "check": "L5", "seed": seed, "start": deep.start, "start_adapter": str(start.adapter), "start_run": str(start.directory),
        "l4t2_run": str(l4t2), "map_run": str(the_map["directory"]), "models": list(deep.models), "stand_in_engine": _stand_in(), "effective_batch": batch,
        "rows_the_start_was_pretrained_on": len(pretraining),
        "the_map": {"what": f"the own map of `{deep.start}`, only read: the base map's problems, each attempted {of_the_map['episodes_each']} times by it. The deep search "
                            "draws among the problems it failed every time; an ordinary solve is one it solved in a few of them", "run": str(the_map["directory"]),
                    "set": the_map["set"], "problems": len(the_map["rows"]), "episodes_each": of_the_map["episodes_each"], "sampling_seed": of_the_map["sampling_seed"],
                    "problems_by_k": {str(k): by_k[k] for k in sorted(by_k)}, "failed_in_every_attempt": len(candidates),
                    "ordinary_solves": {"solved_at_least": deep.least, "solved_at_most": deep.most, "problems": ordinary}, "fixture": the_map["fixture"]},
        "the_search": {"what": f"a seeded draw (a content hash of the task's seed and the problem's id) of {deep.problems:,} of the {len(candidates):,} map problems "
                               f"`{deep.start}` failed in every map attempt; it makes {deep.attempts} more attempts at each, the round's episode in every other respect",
                       "set": SEARCH_SET, "file": SEARCH_PROBLEMS_FILE, "candidates": len(candidates), "problems": len(drawn), "episodes_each": deep.attempts,
                       "attempts_with_the_maps": of_the_map["episodes_each"] + deep.attempts, "sampling_seed": sampling, "sampling_place": deep.sampling_place,
                       "seeds_of_the_stored_samplings_it_is_none_of": used, "block_problems": block_of(config, deep.attempts),
                       "taken_by_the_rule": sum(row["taken_by"] == THE_RULE for row in drawn), "taken_by_the_runs_minimum": sum(row["taken_by"] != THE_RULE for row in drawn),
                       "cap": deep.cap, "minimum_deep_solves": 1 if at_least else deep.minimum},
        "the_held_aside": {"what": f"HA, the held-aside set: a seeded draw of its own (a content hash of the task's seed and the problem's id, under a label of its own) of "
                                   f"{deep.held_aside:,} of the {len(candidates) - sum(row['taken_by'] == THE_RULE for row in drawn):,} candidates the search's draw did NOT take: "
                                   f"problems `{deep.start}` failed in every map attempt, of the same pool and kind as the ones searched, that no model here trains on. "
                                   f"`{deep.start}` and the three trained models each make {deep.held_aside_attempts} fresh attempts at every one, with ONE sampling seed",
                           "file": HELD_ASIDE_FILE, "candidates_not_searched": len(candidates) - sum(row["taken_by"] == THE_RULE for row in drawn), "problems": len(aside),
                           "episodes_each": deep.held_aside_attempts, "sampling_seed": aside_sampling, "sampling_place": deep.held_aside_place,
                           "block_problems": block_of(config, deep.held_aside_attempts), "measured": list(measured_aside),
                           "sets": {name: held_aside_set(name) for name in measured_aside},
                           "taken_by_the_rule": sum(row["taken_by"] == THE_RULE for row in aside), "taken_by_the_runs_minimum": sum(row["taken_by"] != THE_RULE for row in aside)},
        "smoke": {"what": f"{SMOKE_VARIABLE}: a smoke run alone. A fixture's map may hold no problem its model failed every time: the search then attempts at least this "
                          "many and the held-aside set holds at least this many, the map's other problems joining by the same hash order, `shallow` may take its ordinary "
                          "solves as loosely, and one deep solve is enough. Every problem and row so taken is marked as NOT the rule's", "problems_at_least": at_least}
        if at_least else None,
        "pretraining_file": str(ladder_l4.pretraining_file()), "pretraining_file_sha256": sha256,
        "recipe": {"what": "L4t's (`ladder_ceiling.one_pass`, as the arm's twin is trained): FROM THE START ADAPTER, one pass, the native format, the content-hash order",
                   "learning_rate": config["training"]["learning_rate"], "warmup_steps": config["training"]["warmup_steps"], "effective_batch": batch,
                   "lora": config_from(config, {"start_recipe": recorded}, start)["lora"], "adapter_seed": seed},
        "goal_set": len(goal_problems), "goal_set_again": len(again), "rung_problems": len(rung_problems), **plan,
        "attempts_a_goal_problem": sum(sampling_["episodes"] for sampling_ in plan["goal_samplings"]), "parts": parts,
        "stored_models": {deep.start: {"run": str(start.directory), "sets": {part: start_set(part, start) for part in parts}}},
        "never_solved_file": NEVER_SOLVED_FILE, "contradicted_side_setting": config["ladder_loop"]["episode"].get("contradicted_side", "all"),
        "start_recipe": {"what": f"the adapter of `{deep.start}`, as the run that made it recorded it and as L4t2's run trained its models: the deep search is served at this "
                                 f"rank, EVERY training here attaches this rank and alpha (not the config's own, {config['lora']['rank']} and {config['lora']['alpha']}), the "
                                 "adapter each saves is read back and held to them, and the model server of this stage's measure steps is started with this rank as its largest",
                         **recorded, "check": of_start["rank_check"].get("check"), "stage": of_start["rank_check"].get("stage"),
                         "adapter_saved": of_the_start_run["trained"].get("adapter_saved"), "rank_of_the_config": config["lora"]["rank"], "alpha_of_the_config": config["lora"]["alpha"]},
        "start_run_read": {"branch": (of_the_start_run["report"].get("branch") or {}).get("name"), "headline": of_the_start_run["report"].get("headline"),
                           "pretraining_file_sha256": of_start["pretraining_file_sha256"], "rows": of_start["rows"],
                           "checks_failed": of_the_start_run["checks_failed"], "checks_waived_as_l4t2s_run_recorded": of_the_start_run["checks_waived"]}}
    if found is not None:       # a further seed says so, and what it read of the search's own run; at the search's own seed the summary is what it always was
        theirs = found["prepared"]
        read_once = f"it is that of the run of seed {deep.seed_of_the_search} ({found['directory'].name}), only read, and is not made a second time"
        summary[FURTHER] = {
            "what": f"A FURTHER SEED (spec, L5, further seeds). The deep search, the held-aside set and the measurement of `{deep.start}` on it are those of the run of seed "
                    f"{deep.seed_of_the_search}, ONLY READ, with the stored runs that task read: nothing of them is sampled again, and this run's three models are measured "
                    "with the sampling seeds that run's were, so a problem's attempts differ between the seeds by the model alone. BY THIS TASK'S SEED: which of the search's "
                    "deep solves the cap keeps and which verified attempt of each; which ordinary solves and which attempt; the published rows (the first D and the first 2D of "
                    "the hash order of this seed); each set's training order; the three trainings",
            "seed": seed, "search_seed": deep.seed_of_the_search, "search_run": str(found["directory"]),
            "read_from_it": [SEARCH_PROBLEMS_FILE, HELD_ASIDE_FILE, "problems.jsonl", f"episodes_{SEARCH_SET}_problems.jsonl", f"episodes_{SEARCH_SET}_attempts_*.jsonl",
                             f"episodes_{held_aside_set(deep.start)}_problems.jsonl"],
            "the_start_on_the_held_aside_set": {"file": stored_file(deep.start, HELD_ASIDE_PART, L5), "set": held_aside_set(deep.start), "problems": len(start_on_the_set),
                                                "episodes_each": deep.held_aside_attempts, "sampling_seed": aside_sampling,
                                                "successes": sum(row["resolved"] for row in start_on_the_set)},
            "of_the_search_run": {"headline": found["report"].get("headline"), "branches": found["report"].get("branches"), **counted_by_the_search(found)},
            "not_made_again": {SEARCH: f"the deep search: {read_once}", held_aside_marker(THE_START): f"the measurement of `{deep.start}` on the held-aside set: {read_once}"}}
    store.mark_done(SEARCH_PREPARE, summary)
    if found is not None:
        _say(f"L5 prepared to be read as A FURTHER SEED, seed {seed}, from `{deep.start}`: the deep search ({len(drawn):,} problems, {deep.attempts} attempts each, "
             f"{theirs['deep_solves']:,} solved), the held-aside set ({len(aside):,} problems) and the measurement of `{deep.start}` on "
             f"it are those of the run of seed {deep.seed_of_the_search} ({found['directory'].name}), only read; NO search is run and `{deep.start}` is not measured there again; "
             f"this run's models are measured there with sampling seed {aside_sampling}; rank {recorded['rank']}; G' of `{deep.start}` holds {len(again)} goal problems")
        return summary
    _say(f"L5's deep search prepared, seed {seed}, from `{deep.start}`: {len(candidates):,} of its map's {len(the_map['rows']):,} problems failed in every one of "
         f"{of_the_map['episodes_each']} attempts ({ordinary:,} solved in {deep.least} to {deep.most}); {len(drawn):,} drawn"
         + (f" ({summary['the_search']['taken_by_the_runs_minimum']} of them NOT by the rule: a smoke run)" if at_least else "")
         + f", {deep.attempts} more attempts each, sampling seed {sampling}; HELD ASIDE: {len(aside):,} of the candidates the draw did not take, {deep.held_aside_attempts} "
         f"attempts each by four models, sampling seed {aside_sampling}; rank {recorded['rank']}; G' of `{deep.start}` holds {len(again)} goal problems")
    return summary


# -------------------------------------------------------------------------------------------- the search
@dataclass(frozen=True)
class Search:
    """A deep search, as its body (`search_deep`) names what is a stage's own. Every default is this stage's; L6's
    second search (`gpu/ladder_l6.py`) gives its own."""
    marker: str = SEARCH                        # the step, and its marker
    set_name: str = SEARCH_SET                  # the set its episodes are stored under
    stage: ladder_l4_rows.Stage = THE_STAGE     # the stage: its name (in the name a model server gives the adapter) and who speaks
    said: str = "deep search"                   # "<reader>'s <said>", in a printed line


def search_deep(config: dict, store: ArtifactStore, prepared: Mapping, start: Start, of_the_search: Mapping, attempts: int, sampling_seed: int, model: str, adapter: Path,
                one: Search = Search()) -> dict:
    """THE BODY OF A DEEP SEARCH, which two stages call (change it and both change): `model`, served from `adapter` (its
    own run's, only read) at the start adapter's rank, makes `attempts` attempts at each problem of the set, by
    `ladder_loop.run_episodes` exactly as a round's episodes run, in blocks of `block_of` problems. `prepared`: the
    summary the start adapter's recipe is read from. `of_the_search`: what the prepare step recorded of the search,
    which `attempts` and `sampling_seed` (the config's, now) are held to. A search on which Lean gave no verdict for too
    many attempts is NOT KEPT: the task queued again samples it again."""
    blocks = block_of(config, attempts)
    if (of_the_search["episodes_each"], of_the_search["sampling_seed"], of_the_search["block_problems"]) != (attempts, sampling_seed, blocks):
        raise RuntimeError(f"this run was prepared for a search of {of_the_search['episodes_each']} attempts a problem with sampling seed {of_the_search['sampling_seed']} in "
                           f"blocks of {of_the_search['block_problems']} problems, and the config now gives {attempts}, {sampling_seed} and {blocks}: a "
                           "search is made as it was prepared. Nothing was sampled")
    if _stand_in():
        request = None
    else:
        if not adapter.is_dir():
            raise RuntimeError(f"{adapter} is not there: the adapter `{model}` is its own run's, only read here, and the deep search is its. Nothing was sampled")
        from vllm.lora.request import LoRARequest

        request = LoRARequest(f"{one.stage.name}_{model}", 1, str(adapter))
    serving = episodes_config(config_from(config, prepared, start, serving=True), attempts)
    _say(f"{one.stage.reader}'s {one.said}: `{model}` makes {attempts} attempts at each of {of_the_search['problems']:,} problems, sampling seed {of_the_search['sampling_seed']}")
    sampled = _episodes(serving, store, one.set_name, attempts, of_the_search["sampling_seed"], Engines(enable_lora=True).kit(request))
    results = _results(store, one.set_name)
    if lean_did_not_answer(results):        # the deep solves are counted on what Lean answered: the set is sampled again by a rerun
        forget_measurement(store, one.set_name)
        raise RuntimeError(f"Lean gave no verdict on too many attempts of {one.set_name}: the search is not kept. Queue the task again: this step samples it again")
    summary = {"stage": L4, "label": LABEL, "check": one.stage.reader, "model": model, "adapter": str(adapter), "set": one.set_name, "problems": len(results),
               "episodes_each": attempts, "sampling_seed": of_the_search["sampling_seed"], "block_problems": blocks,
               "resolved_at_least_once": sum(row["resolved"] > 0 for row in results), "attempts": sampled.get("attempts"), "statuses": sampled.get("statuses"),
               "contradicted_side": sampled.get("contradicted_side"), "generated_tokens": sampled.get("generated_tokens"), "generation_seconds": sampled.get("generation_seconds"),
               "tokens_per_second": sampled.get("tokens_per_second"), "pipeline": sampled.get("pipeline"), "stand_in_engine": _stand_in()}
    store.mark_done(one.marker, summary)
    _say(f"{one.stage.reader}'s {one.said} is stored: {summary['resolved_at_least_once']:,} of {len(results):,} problems solved in {attempts} more attempts")
    return summary


def ladder_l5_search(config: dict) -> dict:
    """THE DEEP SEARCH (`search_deep`): the start model, served at its rank, makes `deep.attempts` attempts at each drawn
    problem, with the search's own sampling seed, by `ladder_loop.run_episodes` exactly as a round's episodes run. A
    search on which Lean gave no verdict for too many attempts is not kept: the task queued again samples it again."""
    deep = the_settings(config)
    store = _store(config, deep)
    if store.is_done(SEARCH):
        return store.done_summary(SEARCH)
    _need(store, SEARCH_PREPARE, "the deep search", ENTRY_STAGE)
    prepared, start = store.done_summary(SEARCH_PREPARE), the_start(config, deep)
    if prepared.get(FURTHER):      # A FURTHER SEED RUNS NO SEARCH: the one search is the run's it read, and nothing is sampled or marked here
        return _read_not_made(prepared[FURTHER], SEARCH)
    return search_deep(config, store, prepared, start, prepared["the_search"], deep.attempts, search_seed(config, deep), start.name, start.adapter)


# ----------------------------------------------------------------------------------------- the three sets
def build_the_sets(solves: list[dict], ordinary: list[dict], pretraining: list[dict], seed: int) -> dict:
    """The three training sets (`l5.the_sets`), each row numbered by its place in ITS set. `solves`: the deep rows a set
    holds (under the cap). `ordinary`: as many rows of ordinary solves. `pretraining`: the rows the start model was
    pretrained on, in the file's order, without their text. Refused (ValueError) as the rule refuses, and when a
    published row carries text (`l4_rows2.refuse_text`)."""
    numbered = lambda rows: [{"row": position, **{key: value for key, value in row.items() if key != "row"}} for position, row in enumerate(rows)]      # noqa: E731
    sets = {model: numbered(own) for model, own in the_sets(solves, ordinary, pretraining, seed).items()}
    for model, own in sets.items():
        refuse_text(own, f"the training set of `{model}`")
    return sets


def deep_solves_by_the_seed(deep: Deep, seed: int, drawn: Sequence[Mapping], searched: Path, problems: list[dict], results: list[dict], attempts: list[dict],
                            of_the_map: Mapping, of_the_search: Mapping, theirs: Mapping | None = None) -> dict:
    """THE DEEP SOLVES BY A SEED, with NOTHING WRITTEN: one verified attempt of every problem the search solved, drawn by
    `seed` as a round draws one; the search's table; whether there are enough; and which of them the cap keeps, a draw
    by `seed`. `drawn`: the search's problems, in the draw's order. `searched`: the run the search is of; `problems`,
    `results`, `attempts`: its problems as it sampled them, its per-problem rows and its attempts. `of_the_map`,
    `of_the_search`: what the first step recorded of the map (`episodes_each`) and of the search (`episodes_each`,
    `minimum_deep_solves`). `theirs` (a task that draws from ANOTHER task's search: a further seed of this stage, and a
    seed of L5d; `counted_by_the_search`): what the search's own task counted, which the count made here is HELD to.
    Refused (RuntimeError) as the three sets step refuses. TWO STAGES CALL IT (this one's three sets step; L5d's first
    step, `gpu/ladder_l5_draws.py`): change it and both change."""
    taken = {row["problem_id"]: row for row in drawn}
    try:
        solves = [{**row, "map_k": taken[row["problem_id"]]["map_k"], "taken_by": taken[row["problem_id"]]["taken_by"]}
                  for row in solve_rows(problems, attempts, results, seed, DEEP, of_the_map["episodes_each"])]
    except ValueError as error:
        raise RuntimeError(f"{error} (the deep search of {searched.name}). Nothing was written.") from error
    table = search_table(drawn, results, solves, of_the_map["episodes_each"], of_the_search["episodes_each"])
    enough = enough_deep_solves(len(solves), of_the_search["minimum_deep_solves"])
    in_the_set = {row["problem_id"] for row in capped(solves, deep.cap, seed)}
    if theirs is not None:      # the search found again in the stored rows is the one its own task counted, and D is its D: the three models of each seed are trained on as many rows
        if {key: table[key] for key in SEARCH_TABLE} != theirs["the_search_found"] or (len(solves), len(in_the_set) if enough["passes"] else 0) != (theirs["deep_solves"], theirs["rows_of_each_kind"]):
            raise RuntimeError(f"the deep search read from {searched.name} is not the one its own task built its sets on: this task counts {len(solves)} deep solves and D = "
                               f"{len(in_the_set) if enough['passes'] else 0}, that run {theirs['deep_solves']} and {theirs['rows_of_each_kind']}, or the search's table differs. "
                               "Nothing was written.")
    return {"solves": solves, "table": table, "enough": enough, "in_the_set": in_the_set}


def sets_by_the_seed(config: dict, deep: Deep, start: Start, seed: int, kept: list[dict], drawn: Sequence[Mapping], aside: Sequence[str], read: Mapping,
                     groups: Sequence[Mapping]) -> dict:
    """THE THREE SETS BY A SEED, with NOTHING WRITTEN: as many ordinary solves as there are deep rows (`kept`: the deep
    solves the cap keeps) and one verified map attempt of each, both drawn by `seed`; the published rows of that seed's
    hash order; each set in its order (`build_the_sets`); no set holding a held-aside problem, a searched one or a
    barred row. `drawn`: the search's problems; `aside`: the held-aside problems' ids. `read`: what the first step
    recorded (`map_run`; `the_map`: its `set`, `episodes_each` and `problems_by_k`; `pretraining_file_sha256`; `smoke`).
    `groups`: the held-out groups. Returns the ordinary solves as drawn, their rows, the three sets, what was counted
    of barred rows in each, and the pretraining file's SHA-256. Refused (RuntimeError) as the three sets step refuses.
    TWO STAGES CALL IT, as `deep_solves_by_the_seed`."""
    of_the_map, smoke, taken = read["the_map"], bool(read.get("smoke")), {row["problem_id"]: row for row in drawn}
    directory = Path(read["map_run"])
    map_rows = _rows(directory / f"episodes_{of_the_map['set']}_problems.jsonl")
    if {str(k): count for k, count in sorted(Counter(row["resolved"] for row in map_rows).items())} != of_the_map["problems_by_k"]:
        raise RuntimeError(f"the map {directory.name} is not the map this run's search was prepared on (its problems by k are no longer {of_the_map['problems_by_k']}). "
                           "Nothing was written.")
    try:
        ordinary = ordinary_draw(map_rows, deep.least, deep.most, len(kept), seed, not_among={*taken, *aside}, loosely=smoke)
        why = {row["problem_id"]: row for row in ordinary}
        map_problems = {row["problem_id"]: row for row in _rows(directory / "problems.jsonl") if row["set"] == of_the_map["set"]}
        of_ordinary = [{**row, "map_k": why[row["problem_id"]]["map_k"], "taken_by": why[row["problem_id"]]["taken_by"]}
                       for row in solve_rows([map_problems[row["problem_id"]] for row in ordinary], ladder_ceiling._stored_attempts(directory, of_the_map["set"]), map_rows,
                                             seed, SHALLOW)]
        pretraining, sha256 = ladder_l4_start.pretrained_on(start.directory, read["pretraining_file_sha256"], start.name)
        sets = build_the_sets(kept, of_ordinary, pretraining, seed)
        refuse_overlap(aside, list(taken), sets)                # THE HELD-ASIDE SET IS THE TEST: none of its problems was searched, and no set holds a row of one
        heldout = {row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[0]} | {row["problem_id"] for row in groups}
        base_map, half_seed = {row["problem_id"] for row in map_rows}, config["ladder_loop"]["l4"]["half_seed"]
        counted = {model: refuse_barred(sets[model], heldout, half_seed, base_map, f"the training set of `{model}`") for model in deep.models}
    except ValueError as error:
        raise RuntimeError(f"{error} (the map {directory.name}; `{start.name}`'s {start.directory.name}). Nothing was written.") from error
    return {"ordinary": ordinary, "of_ordinary": of_ordinary, "sets": sets, "counted": counted, "sha256": sha256, "pretraining": pretraining}


def ladder_l5_prepare(config: dict) -> dict:
    deep = the_settings(config)
    store = _store(config, deep)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    _need(store, SEARCH_PREPARE, "L5's three training sets", ENTRY_STAGE)
    read, seed, start = store.done_summary(SEARCH_PREPARE), training_seed(config), the_start(config, deep)
    further = read.get(FURTHER)                 # A FURTHER SEED: the search is the run's it read; its own step made none
    if not further:
        _need(store, SEARCH, "L5's three training sets", ENTRY_STAGE)
    of_the_map, of_the_search = read["the_map"], read["the_search"]
    smoke, batch = bool(read.get("smoke")), config["training"]["effective_batch"]

    # ---- the deep solves: one verified attempt of every problem the search solved, drawn by the seed as a round draws one
    drawn, aside = store.read_rows(SEARCH_PROBLEMS_FILE), [row["problem_id"] for row in store.read_rows(HELD_ASIDE_FILE)]
    if further:     # the search's problems as it sampled them, its per-problem rows and its attempts: the search's own run's, only read. THE DRAWS BELOW ARE BY THIS TASK'S SEED
        searched = Path(further["search_run"])
        problems, results, attempts = the_stored_search(searched)
    else:
        searched = store.root
        problems = [row for row in store.read_rows("problems.jsonl") if row["set"] == SEARCH_SET]
        results, attempts = _results(store, SEARCH_SET), _attempts(store, SEARCH_SET)
    solves, table, enough, in_the_set = deep_solves_by_the_seed(deep, seed, drawn, searched, problems, results, attempts, of_the_map, of_the_search,
                                                                further["of_the_search_run"] if further else None).values()
    store.write_rows(SOLVES_FILE, [{**row, "in_the_set": enough["passes"] and row["problem_id"] in in_the_set} for row in solves])
    carried = {**{key: read[key] for key in CARRIED}, **({FURTHER: further} if further else {})}
    of_the_deep_rows = verified_lines(lines_counted(solves))
    if not enough["passes"]:        # the three sets would be too small to read: none is built, nothing is trained, and the search is the finding
        summary = {**carried, "stand_in_engine": _stand_in(), "enough_deep_solves": enough, "deep_solves": len(solves), "rows_of_each_kind": 0, "training_sets": {},
                   "the_search_found": {**table, "lines_of_the_deep_rows": of_the_deep_rows},
                   "trainings": {"run": False, "why": f"the deep search found {len(solves)} deep solves and {enough['minimum']} are asked: the three sets would be too small "
                                                      "to read. No set is built and no model is trained or measured: the search is the finding"}}
        store.mark_done(PREPARE, summary)
        _say(f"L5: the deep search found {len(solves)} deep solves of {len(drawn):,} problems, FEWER THAN THE {enough['minimum']} asked: no set is built and the three "
             "trainings are not run. The search is the finding")
        return summary

    # ---- the ordinary solves: as many map problems the start model solved in a few of its map attempts, one verified map attempt each
    kept = [row for row in solves if row["problem_id"] in in_the_set]
    ordinary, of_ordinary, sets, counted, sha256, _ = sets_by_the_seed(config, deep, start, seed, kept, drawn, aside, read, store.read_rows(ladder_l4.GROUPS_FILE)).values()

    # ---- from here on files are written
    store.write_rows(ORDINARY_FILE, ordinary)
    for model in deep.models:
        store.write_rows(THE_STAGE.training_file(model), sets[model])
    count = len(kept)
    order = (f"the hash order L4t's rehearsal draw uses (a content hash of the task's seed and each row's id among the rows `{deep.start}` was pretrained on); trained in the "
             "arm's content-hash order. A published row's text stays in the pretraining file")
    lines_of = {"deep_rows": verified_lines(lines_counted(kept)), "shallow_rows": verified_lines(lines_counted(of_ordinary)),
                "the_published_rows_beside_them": verified_lines(lines_counted([row for row in sets[DEEP] if row["origin"] == PRETRAINING])),
                "published_rows": verified_lines(lines_counted(sets[PUBLISHED]))}
    not_the_rules = {DEEP: sum(row["taken_by"] != THE_RULE for row in kept), SHALLOW: sum(row["taken_by"] != THE_RULE for row in of_ordinary)}
    summary = {
        **carried, "stand_in_engine": _stand_in(), "enough_deep_solves": enough, "deep_solves": len(solves), "rows_of_each_kind": count,
        "the_search_found": {**table, "lines_of_the_deep_rows": of_the_deep_rows, "deep_rows_in_the_sets": count, "cap": deep.cap, "left_out_by_the_cap": len(solves) - count},
        "trainings": {"run": True},
        "lines_of_the_rows": {"what": "the lines of the proofs of the rows trained on (counted as a model's verified proofs are): the deep rows, the ordinary solves' rows, the "
                                      "D published rows that stand beside either, and `published`'s 2D", **lines_of},
        "rows_not_taken_by_the_rule": {"what": "a smoke run alone: rows of problems the rule did not give (zero in every other run)", **not_the_rules},
        "training_sets": {
            PUBLISHED: {"what": f"the first {2 * count:,} rows of {order}, and nothing else", "file": THE_STAGE.training_file(PUBLISHED), **described(sets[PUBLISHED], batch),
                        "barred": counted[PUBLISHED]},
            SHALLOW: {"what": f"{count:,} rows of ORDINARY solves (map problems `{deep.start}` solved in {deep.least} to {deep.most} of its {of_the_map['episodes_each']} map "
                              f"attempts: a seeded draw, one verified map attempt each, drawn by the seed) and the first {count:,} rows of {order}",
                      "file": THE_STAGE.training_file(SHALLOW), **described(sets[SHALLOW], batch), "ordinary_solves_file": ORDINARY_FILE, "barred": counted[SHALLOW]},
            DEEP: {"what": f"the {count:,} deep rows (one verified attempt of each problem the deep search solved, drawn by the seed as a round draws one"
                           + (f"; a seeded draw of {count:,} of the {len(solves):,} deep solves, the cap" if len(solves) > count else "")
                           + f") and the SAME first {count:,} rows of {order}", "file": THE_STAGE.training_file(DEEP), **described(sets[DEEP], batch),
                   "deep_solves_file": SOLVES_FILE, "barred": counted[DEEP]}},
        "pretraining_file_sha256": sha256}
    store.mark_done(PREPARE, summary)
    of = summary["training_sets"]
    _say(f"L5 prepared, seed {seed}, from `{deep.start}`: the deep search solved {len(solves):,} of {len(drawn):,} problems; D = {count:,}; "
         + "; ".join(f"`{model}` {of[model]['rows']:,} rows ({of[model]['rows_by_origin']}), {of[model]['optimizer_steps']:,} optimizer steps" for model in deep.models)
         + (f"; A SMOKE RUN: {not_the_rules[DEEP]} deep rows and {not_the_rules[SHALLOW]} ordinary rows are NOT the rule's" if smoke else ""))
    return summary


# ----------------------------------------------------------------------------------------- train and measure
def _not_run(store: ArtifactStore, what: str, one: ladder_l4_rows.Stage = THE_STAGE) -> dict | None:
    """What a training or measuring step returns, with nothing marked done, when the deep search found too few deep
    solves for the three sets (`ladder_l5_prepare` built none): nothing is trained or measured. None otherwise. `one`:
    the stage whose step it is (this one's; L6's, whose sets its own prepare step builds or does not)."""
    _need(store, one.prepare, what, one.name)
    trainings = store.done_summary(one.prepare)["trainings"]
    if trainings["run"]:
        return None
    _say(f"{one.reader}: {what} is NOT RUN: {trainings['why']}")
    return {"stage": L4, "label": LABEL, "check": one.reader, "run": False, "why": trainings["why"]}


def ladder_l5_train(config: dict, model: str) -> dict:
    """One of the three models, by L4t's own training body (`ladder_l4_rows.train_a_model`) under this stage's names."""
    deep = the_settings(config)
    store = _store(config, deep)
    return _not_run(store, f"the training of `{model}`") or train_a_model(config, store, model, deep.start, run_name(deep, training_seed(config)), THE_STAGE)


def ladder_l5_measure(config: dict, model: str) -> dict:
    """One of the three models, by L4t's own measuring body (`ladder_l4_rows.measure_a_model`) under this stage's names."""
    deep = the_settings(config)
    store = _store(config, deep)
    return _not_run(store, f"the measurement of `{model}`") or measure_a_model(config, store, model, deep.start, THE_STAGE)


# ------------------------------------------------------------------------------------- the held-aside set
def ladder_l5_held_aside(config: dict, who: str) -> dict:
    """ONE model's fresh attempts at the held-aside problems: `deep.held_aside_attempts` at each, with the set's ONE
    sampling seed (the four models pair by problem on the same random numbers), by `ladder_loop.run_episodes` as the
    deep search runs (the prompt, the limits, one side by the audit rule, Lean's verdict, the soundness alarm;
    block-resumable). `who`: `start` (the start model, from its stored adapter, only read) or a trained model (the
    adapter this run trained and kept). Served at the start adapter's rank. A model that is measured is returned as
    stored, UNLESS Lean gave no verdict on too much of its set (the report then cannot be read): that set is measured
    again from the adapter. With too few deep solves nothing was trained, and nothing is measured here either. AT A
    FURTHER SEED the start model is NOT measured again (its step says so and marks nothing: its rows are the search's
    own run's, copied by the first step); the three trained models are, with the sampling seed that run measured it with."""
    deep = the_settings(config)
    store, marker, name = _store(config, deep), held_aside_marker(who), held_aside_name(deep, who)
    # A FURTHER SEED DOES NOT MEASURE THE START MODEL AGAIN: nothing is sampled or marked here
    return on_the_held_aside_set(config, store, deep.start, name, marker, deep.held_aside_attempts, held_aside_seed(config, deep),
                                 trained_by=None if who == THE_START else THE_STAGE.train_marker(name),
                                 not_made=lambda prepared: _read_not_made(prepared[FURTHER], marker) if who == THE_START and prepared.get(FURTHER) else None)


def on_the_held_aside_set(config: dict, store: ArtifactStore, start_model: str, name: str, marker: str, attempts: int, sampling_seed: int, trained_by: str | None = None,
                          directory: Path | None = None, not_made: Callable[[Mapping], dict | None] | None = None, one: ladder_l4_rows.Stage = THE_STAGE) -> dict:
    """THE BODY OF ONE MODEL'S MEASUREMENT ON THE HELD-ASIDE SET, which two stages call (change it and both change):
    `attempts` fresh attempts at each held-aside problem with the set's ONE sampling seed, by `ladder_loop.run_episodes`
    in blocks of `block_of` problems, served at the start adapter's rank; `attempts` and `sampling_seed` (the config's,
    now) are held to what the run was prepared for. `name`: the model; `marker`: its step. `trained_by`: the step of
    THIS run that trained it and keeps its adapter (`adapters/<name>`), or None for a model that is its own run's, only
    read: the start model's adapter, or the one at `directory` (L6's stored model). `not_made`: given the prepare step's
    summary, what the step returns when it makes nothing (a further seed's start model). `one`: the stage whose step it
    is. A model that is measured is returned as stored, UNLESS Lean gave no verdict on too much of its set: that set is
    measured again from the adapter. With nothing trained (`_not_run`) nothing is measured."""
    set_name = held_aside_set(name, one.files)
    stored = store.done_summary(marker) if store.is_done(marker) else None
    again = stored is not None and lean_did_not_answer(_results(store, set_name))
    if stored is not None and not again:
        return stored
    not_run = _not_run(store, f"the measurement of `{name}` on the held-aside set", one)
    if not_run:
        return not_run
    prepared, start = store.done_summary(one.prepare), ladder_l4_start.the_start(config, start_model, one.start_said_by)
    elsewhere = not_made(prepared) if not_made is not None else None
    if elsewhere:
        return elsewhere
    of_the_set = prepared["the_held_aside"]
    blocks = block_of(config, attempts)
    if (of_the_set["episodes_each"], of_the_set["sampling_seed"], of_the_set["block_problems"]) != (attempts, sampling_seed, blocks):
        raise RuntimeError(f"this run was prepared for {of_the_set['episodes_each']} attempts a held-aside problem with sampling seed {of_the_set['sampling_seed']} in blocks of "
                           f"{of_the_set['block_problems']} problems, and the config now gives {attempts}, {sampling_seed} and {blocks}: the "
                           "four models are measured there alike. Nothing was sampled")
    trained_here = trained_by is not None
    if trained_here:
        _need(store, trained_by, f"the measurement of `{name}` on the held-aside set", one.name)
    directory = store.root / ladder_assembly.ADAPTERS / name if trained_here else directory or start.adapter
    if _stand_in():
        adapter = None
    else:
        if not directory.is_dir():
            raise RuntimeError(f"{directory} is not there: the adapter `{name}` is " + ("kept by this stage" if trained_here else "its own run's, only read here")
                               + ". The measurement cannot be made")
        from vllm.lora.request import LoRARequest

        adapter = LoRARequest(f"{one.name}_{name}", 1, str(directory))
    if again:
        print(f"{L4}: `{name}`: Lean gave no verdict on too many attempts of {set_name}: it is measured again from the adapter", flush=True)
        forget_measurement(store, set_name)
    serving = episodes_config(config_from(config, prepared, start, serving=True), attempts)
    _say(f"{one.reader}: `{name}` makes {attempts} attempts at each of the {of_the_set['problems']:,} held-aside problems, sampling seed {of_the_set['sampling_seed']}")
    sampled = _episodes(serving, store, set_name, attempts, of_the_set["sampling_seed"], Engines(enable_lora=True).kit(adapter))
    results = _results(store, set_name)
    summary = {"stage": L4, "label": LABEL, "check": one.reader, "model": name, "adapter": str(directory), "set": set_name, "problems": len(results),
               "episodes_each": attempts, "sampling_seed": of_the_set["sampling_seed"], "block_problems": blocks,
               "resolved_at_least_once": sum(row["resolved"] > 0 for row in results), "successes": sum(row["resolved"] for row in results),
               "attempts": sampled.get("attempts"), "statuses": sampled.get("statuses"), "contradicted_side": sampled.get("contradicted_side"),
               "generated_tokens": sampled.get("generated_tokens"), "generation_seconds": sampled.get("generation_seconds"), "tokens_per_second": sampled.get("tokens_per_second"),
               "pipeline": sampled.get("pipeline"), "stand_in_engine": _stand_in(), **({"measured_again": [*stored.get("measured_again", []), set_name]} if stored is not None else {})}
    store.mark_done(marker, summary)
    return summary


# -------------------------------------------------------------------------------------------------- report
def ladder_l5_report(config: dict) -> dict:
    from rlvr_lean.gpu.ladder_l3c import proof_lengths
    from rlvr_lean.reporting.ladder_l5 import build_l5_report

    deep = the_settings(config)
    store = _store(config, deep)
    _need(store, PREPARE, "L5's report", ENTRY_STAGE)
    prepared = store.done_summary(PREPARE)
    measured, further = deep.models if prepared["trainings"]["run"] else (), prepared.get(FURTHER)
    for step in (*((held_aside_marker(THE_START),) if measured and not further else ()),
                 *(of(model) for model in measured for of in (THE_STAGE.train_marker, THE_STAGE.measure_marker, held_aside_marker))):
        _need(store, step, "L5's report", ENTRY_STAGE)
    on_the_held_aside = {name: _results(store, held_aside_set(name)) for name in ((deep.start, *measured) if measured and not further else measured)}
    if measured and further:        # A FURTHER SEED: the start model's rows there are the search's own run's, as the first step copied them
        on_the_held_aside = {deep.start: store.read_rows(stored_file(deep.start, HELD_ASIDE_PART, L5)), **on_the_held_aside}
    samplings, of_the_stored = prepared["goal_samplings"], json.loads(store.path(STORED_MODELS_FILE).read_text())
    stored = {deep.start: {RUNG_PART: store.read_rows(stored_file(deep.start, RUNG_PART, L4)),
                           GOAL: [store.read_rows(stored_file(deep.start, sampling["name"], L4)) for sampling in samplings], **{key: of_the_stored[deep.start][key] for key in WROTE}}}
    models, trains, losses, trained = read_the_trained(store, prepared, measured, THE_STAGE)
    # The goal problems nothing stored has ever solved are GIVEN (a file of ids in the run directory), never computed here.
    never = [row["problem_id"] for row in store.read_rows(NEVER_SOLVED_FILE)] if store.path(NEVER_SOLVED_FILE).exists() else None
    base_map = [row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[1]]
    # The length of a held-out problem's published proof is read HERE, by the report, and by nothing before it.
    report = build_l5_report(prepared, trains, losses, trained, store.read_rows(ladder_l4.GROUPS_FILE), proof_lengths(), stored, models,
                             [row["problem_id"] for row in store.read_rows(ladder_l4.AGAIN_FILE)], config["ladder_loop"]["l4"], config["evaluation"], base_map,
                             on_the_held_aside, never)
    directory = store.root / ladder_assembly.ADAPTERS
    report["adapters"] = {"kept": True, "what": (f"the adapters of {' and '.join(f'`{model}`' for model in measured)} ({LABEL}) are KEPT: no step of this stage deletes one. "
                                                 if measured else "no model was trained. ") + f"`{deep.start}` is its own run's and was only read",
                          "directory": str(directory), "there": sorted(path.name for path in directory.glob("*") if path.is_dir()) if directory.is_dir() else []}
    _write_json(store, REPORT_FILE, report)
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    for name in (STORED_MODELS_FILE, *(THE_STAGE.loss_file(model) for model in measured)):
        store.mirror(name)
    for line in report["lines"]:
        print(line, flush=True)
    store.mark_done(REPORT, {"stage": L4, "label": LABEL, "check": "L5", "headline": report["headline"], "branches": report["branches"], "measured": list(measured),
                             "ok": report["ok"]})
    return report


def _for(step, name: str):
    def run(config: dict) -> dict:
        return step(config, name)
    return run


# In the order the stage runs them: the search, the three sets, the start model on the held-aside set, then each model trained, measured and
# measured on the held-aside set before the next is trained.
STEPS = {SEARCH_PREPARE: ladder_l5_search_prepare, SEARCH: ladder_l5_search, PREPARE: ladder_l5_prepare, held_aside_marker(THE_START): _for(ladder_l5_held_aside, THE_START)}
for _model in MODELS:
    STEPS[THE_STAGE.train_marker(_model)] = _for(ladder_l5_train, _model)
    STEPS[THE_STAGE.measure_marker(_model)] = _for(ladder_l5_measure, _model)
    STEPS[held_aside_marker(_model)] = _for(ladder_l5_held_aside, _model)
STEPS[REPORT] = ladder_l5_report
