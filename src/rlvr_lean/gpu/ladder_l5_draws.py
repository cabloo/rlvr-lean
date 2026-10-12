"""L5d: the three kinds again, over fresh draws, measured on the goal set's first sampling. Spec:
docs/spec/ladder-loop.spec.md, "L5d: the three kinds again, over fresh draws, measured on the goal set's first
sampling" and its "Made exact by the build". Everything it makes is labelled pretrained on published proofs. Each step is
its own process (`rlvr_lean.runner.entry`, stage `ladder_l5_draws`). NO ROUND AND NO SEARCH: a seed here is A FURTHER SEED
OF L5 with the measurement cut to the one sampling L5n measures.

L5's FINISHED SEARCH IS READ AND NEVER SAMPLED AGAIN (`ladder_l5.search_directory`), with every stored run that task
read, by L5's OWN first-step reading (`ladder_l5.read_the_search`, `read_for_the_search`: a further seed's refusals), and
THE THREE SETS ARE BUILT BY L5's OWN RULE FOR THE TASK'S SEED, by the two functions L5's three sets step draws by
(`ladder_l5.deep_solves_by_the_seed`, `sets_by_the_seed`). The run directory is its own
(`ladder_l5_draws_<start>_seed<N>`).

THE RECIPE IS A SETTING (`ladder_loop.l5_draws.recipe`), HELD TO L5n's REPORT by the first step (`l5_draws.the_recipe`):
`constant` (the recipe as it is, one training a model, exactly as L5 trains that seed's set), `decay` (one training with
L5n's decay), `mean_constant` (each set trained in the orders of order seeds 1 to 4 with the recipe as it is; the model
measured is the adapter whose every number is the mean of the four, by L5n's own function). A run keeps its recipe.

THE STAGE'S STEPS ARE THOSE OF THE LARGEST RECIPE (the list in `runner.entry` is fixed): a step the run's recipe does not
use returns `run: false` with the reason, and marks nothing.

  ladder_l5_draws_prepare           REFUSES, naming what to do, before anything is written: a task of a seed of L5's
                                    own, of a further seed without explicit approval, or of no seed of the setting; a
                                    recipe that is not the one L5n's report asks for; L5's run missing, unfinished, not to
                                    be read, INCONCLUSIVE or made otherwise; a recipe or a first sampling that is not that
                                    run's; a set with a problem of G or of the held-aside set. STORES the three sets (a
                                    published row by id and place, never with text) and, under the averaged recipe, each
                                    set in its four orders. No GPU, no Lean
  ladder_l5_draws_train_<model>     `constant`, `decay`: L4t's OWN training body (`ladder_l4_rows.train_a_model`) under
                                    this stage's names, from the start adapter at its rank, one pass
  ladder_l5_draws_train_<model>_<k> `mean_constant`: the same body, the set in the order of order seed k
  ladder_l5_draws_average_<model>   `mean_constant`, NO TRAINING: the adapter that is the mean of its four
                                    (`ladder_l5_noise.the_mean_adapter`), read back and held as L5n holds its own
  ladder_l5_draws_measure_<model>   the model on the goal set's FIRST sampling alone, as L5n measures: the attempts and the
                                    sampling seed the start model's run recorded. No rung, no second sampling, no
                                    held-aside set
  ladder_l5_draws_report            this seed's checks and figures, and THE READ OVER THE FRESH SEEDS from the runs that
                                    are on the box and finished (`reporting/ladder_l5_draws.py`): with a seed of the fixed
                                    number not there, NOT READ, naming it

Every step is resumable and a step of its own. NO STEP DELETES AN ADAPTER, and no step writes into a run it reads.

A SMOKE RUN ALONE (`RLVR_LEAN_LADDER_L5_DRAWS_SMOKE`, refused unless the run directory is one the task names and the map
is the fixtures'): L5's smoke run reads INCONCLUSIVE and is read all the same, its two looser draws made again as that
run recorded them; one deep solve is enough; orders of its two rows that are alike are trained all the same; and an
averaged adapter that is one of its models is kept. A run the task names carries a recipe that is not `constant` in its
directory's name (`<named>_<recipe>`), so that the smoke run of one recipe does not stand in another's way.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from rlvr_lean.domain.ladder_round.dose import steps_per_pass
from rlvr_lean.domain.ladder_round.l5 import DEEP, MODELS, PUBLISHED, SHALLOW
from rlvr_lean.domain.ladder_round.l5_draws import (
    CONSTANT,
    DECAY,
    MEAN_CONSTANT,
    OF_L5,
    RECIPES,
    T_CRITICAL,
    fixed_seeds,
    four_orders,
    its_trainings,
    refuse_problems_of,
    the_recipe,
    the_seed,
    trained_models,
    trained_of,
)
from rlvr_lean.domain.ladder_round.l5_noise import ORDERS, numbered
from rlvr_lean.gpu import ladder_assembly, ladder_ceiling, ladder_l4, ladder_l4_rows, ladder_l4_start, ladder_l5, ladder_l5_noise, ladder_loop
from rlvr_lean.gpu.ladder_ceiling import Model, Reading, _need, goal_set, measure_model, stored_file
from rlvr_lean.gpu.ladder_dose import _runs
from rlvr_lean.gpu.ladder_l3d1 import what_else_a_model_wrote
from rlvr_lean.gpu.ladder_l4 import L4, LABEL, _say
from rlvr_lean.gpu.ladder_l4_rows import described, train_a_model
from rlvr_lean.gpu.ladder_l4_start import Start, config_from
from rlvr_lean.gpu.ladder_round import _results, _stand_in, _write_json, training_seed
from rlvr_lean.infrastructure.adapter_files import ADAPTER_FILE, MEAN_TYPES, tensors_of
from rlvr_lean.infrastructure.artifact_store import ArtifactStore

L5D = "l5_draws"                                            # what this stage's own files are called
STAGE = ENTRY_STAGE = "ladder_l5_draws"                     # what its steps begin with; the stage, as `runner.entry` names it; its run directory begins with it
RUN_VARIABLE = "RLVR_LEAN_LADDER_L5_DRAWS_RUN"              # another run directory than `ladder_l5_draws_<start>_seed<seed>` (a smoke run)
SMOKE_VARIABLE = "RLVR_LEAN_LADDER_L5_DRAWS_SMOKE"          # a SMOKE run alone (refused elsewhere): L5's INCONCLUSIVE smoke run is read, and what else the module's note says
SETTING = "ladder_loop.l5_draws"
THE_STAGE = ladder_l4_rows.Stage(name=ENTRY_STAGE, steps=STAGE, files=L5D, reader="L5d", start_said_by=f"{ladder_l5.SETTING}.start")
PREPARE, REPORT = THE_STAGE.prepare, f"{STAGE}_report"
REPORT_FILE = "report_ladder_l5_draws.json"
READING = Reading(reader="L5d", models="L5d's models", not_read="that set stands beside this run's, and no report of this run could be read against it", setting=ladder_l4.SETTING)


@dataclass(frozen=True)
class Draws:
    """`ladder_loop.l5_draws`, as the stage's steps read it."""
    start: str                          # L5's start: every model here is trained from it at its rank
    seeds: tuple[int, ...]              # the fresh seeds: one task a seed
    more: tuple[int, ...]               # the further fresh seeds, which run WITH EXPLICIT APPROVAL
    approved: bool                      # ... recorded here once it has come: the reads are then over both lists
    sampling: str                       # the sampling of the goal set every model here is measured on, by the name the start model's run gave it: its FIRST, as L5n's
    attempts: int                       # ... and its attempts a problem
    recipe: str                         # how a model is trained: held to L5n's report by the first step
    of_l5: tuple[int, ...]              # L5's own seeds, which found the sign: no task here is of one, and they stand beside the fresh ones as exploratory
    noise: ladder_l5_noise.Noise        # L5n's settings as ITS task reads them: its report names the recipe, and a model here is measured as its are
    deep: ladder_l5.Deep                # L5's settings as ITS task reads them: the sets are drawn from the finished search of its search's seed


def the_settings(config: dict) -> Draws:
    """THE one place that reads `ladder_loop.l5_draws`. Refused (ValueError): a recipe that is none of the three; seeds
    that are not distinct whole numbers, in the two lists together, or that hold a seed of L5's own; an approval that
    is not true or false; a number of seeds (the first list's, and both lists') the interval over seeds is not given
    for; a sampling that is not L5n's; L5n's and L5's own settings, when their readers refuse them."""
    settings = config["ladder_loop"]["l5_draws"]
    noise = ladder_l5_noise.the_settings(config)
    seeds, more, approved, recipe = tuple(settings["seeds"]), tuple(settings["more_seeds"]), settings["more_seeds_approved"], settings["recipe"]
    if recipe not in RECIPES:
        raise ValueError(f"{SETTING}.recipe is {recipe!r}: one of {list(RECIPES)} (the one L5n's report names)")
    both = (*seeds, *more)
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in both) or len(set(both)) != len(both) or set(both) & set(noise.seeds):
        raise ValueError(f"{SETTING}.seeds is {list(seeds)} and more_seeds {list(more)}: distinct whole numbers of at least 0, none of them a seed of L5's own "
                         f"({list(noise.seeds)}: {ladder_l5_noise.SETTING}.published_seeds), which found the sign")
    if not isinstance(approved, bool):
        raise ValueError(f"{SETTING}.more_seeds_approved is {approved!r}: true once the further seeds are approved, false until then")
    if len(seeds) - 1 not in T_CRITICAL or len(both) - 1 not in T_CRITICAL:
        raise ValueError(f"{SETTING}.seeds holds {len(seeds)} seeds and, with more_seeds, {len(both)}: the interval over seeds is given for "
                         f"{sorted(count + 1 for count in T_CRITICAL)} seeds")
    attempts, sampling = settings["attempts"], settings["sampling"]
    if (sampling, attempts) != (noise.sampling, noise.attempts):
        raise ValueError(f"{SETTING} names the sampling `{sampling}` of {attempts!r} attempts a problem and {ladder_l5_noise.SETTING} `{noise.sampling}` of {noise.attempts}: a "
                         "model here is measured as L5n measures")
    return Draws(start=noise.start, seeds=seeds, more=more, approved=approved, sampling=sampling, attempts=attempts, recipe=recipe, of_l5=noise.seeds, noise=noise, deep=noise.deep)


def run_name(draws: Draws, seed: int) -> str:
    """The stage's run directory, with the start adapter's name in it: `ladder_l5_draws_pre_r64_seed<N>`."""
    return f"{ENTRY_STAGE}_{draws.start}_seed{seed}"


def own_run(config: dict, draws: Draws) -> str:
    """This task's run directory: the seed's own, or the one the task names (a smoke run), which carries a recipe that
    is not the recipe as it is in its name."""
    named = os.environ.get(RUN_VARIABLE)
    if not named:
        return run_name(draws, training_seed(config))
    return named if draws.recipe == CONSTANT else f"{named}_{draws.recipe}"


def _store(config: dict, draws: Draws) -> ArtifactStore:
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    return ArtifactStore(_runs(config) / own_run(config, draws), Path(mirror) if mirror else None)


def the_start(config: dict, draws: Draws) -> Start:
    """The start model, and where it lives (`ladder_l4_start.the_start`): L5's."""
    return ladder_l4_start.the_start(config, draws.start, THE_STAGE.start_said_by)


def smoke_run() -> bool:
    """Whether the task says it is a smoke run; every other run: the rule, and nothing else."""
    return bool(os.environ.get(SMOKE_VARIABLE))


def average_marker(model: str) -> str:
    """The step that makes a model's averaged adapter, and its marker."""
    return f"{STAGE}_average_{model}"


def the_set(name: str, sampling: str) -> str:
    """The set a model's attempts at the goal problems are stored under, as every stage names a model's sampling of G:
    L5's and L5n's name for it, so that the stored first samplings pair by problem."""
    return goal_set(sampling, name, L4)


def the_reader(draws: Draws, smoke: bool) -> ladder_l5.OfTheSearch:
    """Who reads L5's finished run here, for what L5's own reading says when it refuses it. A SMOKE run alone reads a run
    whose report is INCONCLUSIVE (one not to be read never is)."""
    return ladder_l5.OfTheSearch(
        who="L5d", why=f"L5d runs no search: a seed of it is a further seed of L5, and draws its three sets from the finished search of seed {draws.deep.seed_of_the_search}",
        stands="no model here stands beside such a run", models="L5d's models are trained on rows drawn from that run's search and stand beside its models", inconclusive=smoke)


def l5n_task(draws: Draws) -> str:
    """The task that writes L5n's report, for a refusal."""
    stage, seed = ladder_l5_noise.ENTRY_STAGE, draws.deep.seed_of_the_search
    return f"the task of stage `{stage}` for seed {seed} (`python -m rlvr_lean.runner.entry --stage {stage} --seeds {seed}`)"


def read_l5n(config: dict, draws: Draws) -> dict:
    """What L5n's run says of the recipe, ONLY READ: its report's "what follows", when its run is on the box and its
    report written. NOTHING IS REFUSED HERE: a run that is not there is said to be not there, with why. The run is the
    real one's at every task (a smoke run names no other): a smoke run trains with the recipe the setting names."""
    directory = _runs(config) / ladder_l5_noise.run_name(draws.noise, draws.deep.seed_of_the_search)
    lacking = [name for name in (f"{ladder_l5_noise.REPORT}.done.json", ladder_l5_noise.REPORT_FILE) if not (directory / name).exists()]
    if lacking:
        return {"there": False, "run": str(directory), "why": f"{directory.name} does not hold {lacking}"}
    report = json.loads((directory / ladder_l5_noise.REPORT_FILE).read_text())
    return {"there": True, "run": str(directory), "ok": bool(report.get("ok", True)), "inconclusive": bool(report.get("inconclusive")),
            "what_follows": (report.get("branches") or {}).get("what_follows"), "headline": report.get("headline")}


# ------------------------------------------------------------------------------------------------- prepare
@dataclass(frozen=True)
class OfASeed:
    """Who draws from L5's finished search by a seed, as a refusal names what is its own. Every default is L5d's; L5s
    (`gpu/ladder_l5_stock.py`), whose sets hold a seed's rows, gives its own."""
    who: str = "L5d"                            # "... and <who> has nothing to train"
    run_variable: str = RUN_VARIABLE            # the variable that names a smoke run's directory
    smoke_variable: str = SMOKE_VARIABLE        # ... and its smoke-only variable, refused on the pool's problems


DRAWN = ("found", "read", "start", "the_map", "at_least", "directory", "theirs", "sampling", "training", "batch", "recipe", "of_this", "solves", "table", "enough", "kept", "ordinary",
         "of_ordinary", "sets", "counted", "sha256", "pretraining", "aside")


def drawn_by_the_seed(config: dict, draws: Draws, seed: int, smoke: bool, reader: ladder_l5.OfTheSearch, averages: bool, one: OfASeed = OfASeed()) -> dict:
    """WHAT A SEED READS, HOLDS AND DRAWS, with NOTHING WRITTEN: L5's finished run and every stored run its task read, by
    L5's own reading; how a model here is measured and trained, held to those runs; the deep solves by `seed`, and the
    three sets by L5's own rule for it. `smoke`: a SMOKE run. `reader`: who reads L5's run. `averages`: the averaged
    recipe, whose start adapter's floating type is held here. Returns what `DRAWN` names. REFUSED (RuntimeError) as the
    module's note says of the first step. TWO STAGES CALL IT (this one's first step; L5s's, `gpu/ladder_l5_stock.py`):
    change it and both change."""
    deep = draws.deep
    # ---- L5's finished run, and every stored run its task read: by L5's OWN reading, as a further seed of L5 reads them (refused there, with nothing written)
    found = ladder_l5.read_the_search(config, deep, reader)
    # A smoke run reads a SMOKE run of L5: its two looser draws are made again as that run recorded them (L5's own smoke-only variable is L5's smoke stage's alone).
    read = ladder_l5.read_for_the_search(config, deep, found, reader, at_least=((found["searched"].get("smoke") or {}).get("problems_at_least") or 0) if smoke else 0)
    start, the_map, plan, recorded, at_least = (read[key] for key in ("start", "the_map", "plan", "recorded", "at_least"))
    if smoke and not the_map["fixture"]:
        raise RuntimeError(f"{one.smoke_variable} is set: it is for a SMOKE run alone, in a run directory the task names ({one.run_variable}) and on the fixtures' base map. A run on "
                           "the pool's problems reads a run of L5 whose checks passed. Nothing was written.")
    directory, theirs = found["directory"], found["searched"]

    # ---- how a model here is measured, and how it is trained: the start model's FIRST sampling of G as its run recorded it; the recipe that trained L5's models
    sampling = plan["goal_samplings"][0]
    if (sampling["name"], sampling["episodes"]) != (draws.sampling, draws.attempts):
        raise RuntimeError(f"{SETTING} names the sampling `{draws.sampling}` of {draws.attempts} attempts a problem, and the run of `{draws.start}` ({start.directory.name}) "
                           f"recorded its first sampling of the goal set as `{sampling['name']}` of {sampling['episodes']}: every model here is measured on that sampling "
                           "alone, with its recorded sampling seed. Nothing was written.")
    training, batch = config["training"], config["training"]["effective_batch"]
    recipe = {"learning_rate": training["learning_rate"], "warmup_steps": training["warmup_steps"], "effective_batch": batch,
              "lora": config_from(config, {"start_recipe": recorded}, start)["lora"]}
    then = {key: (theirs.get("recipe") or {}).get(key) for key in recipe}
    if then != recipe:
        raise RuntimeError(f"L5's run {directory.name} trained its models with {then} and this task would train with {recipe}: a seed here is a further seed of L5, and "
                           f"`{CONSTANT}` is the recipe that run's models were trained with. Nothing was written.")
    if averages and not ladder_ceiling._stand_in():     # the mean of adapters is written in the adapters' own floating type: held HERE, on the start adapter, and not after the trainings
        other = sorted({entry["dtype"] for entry in tensors_of(start.adapter / ADAPTER_FILE)[0].values()} - set(MEAN_TYPES))
        if other:
            raise RuntimeError(f"the start adapter {start.adapter} holds tensors of type {other}: the averaged adapters are written in {list(MEAN_TYPES)} "
                               "(`infrastructure.adapter_files.mean_of`), and an adapter trained from it has its types. Nothing was written.")

    # ---- THE THREE SETS, by L5's own rule for THIS TASK's seed: the two functions L5's three sets step draws by, on what a further seed of L5 would have recorded
    by_k, aside = read["by_k"], [row["problem_id"] for row in read["aside"]]
    of_this = {"map_run": str(the_map["directory"]), "pretraining_file_sha256": read["sha256"], "smoke": bool(at_least),
               "the_map": {"set": the_map["set"], "episodes_each": the_map["recorded"]["episodes_each"], "problems_by_k": {str(k): by_k[k] for k in sorted(by_k)}},
               "the_search": {"episodes_each": deep.attempts, "minimum_deep_solves": 1 if at_least else deep.minimum}}
    solves, table, enough, in_the_set = ladder_l5.deep_solves_by_the_seed(deep, seed, read["drawn"], directory, *ladder_l5.the_stored_search(directory), of_this["the_map"],
                                                                          of_this["the_search"], ladder_l5.counted_by_the_search(found)).values()
    if not enough["passes"]:
        raise RuntimeError(f"the deep search of {directory.name} found {len(solves)} deep solves and {enough['minimum']} are asked: no set is built on it, and {one.who} has "
                           "nothing to train. Nothing was written.")
    kept = [row for row in solves if row["problem_id"] in in_the_set]
    ordinary, of_ordinary, sets, counted, sha256, pretraining = ladder_l5.sets_by_the_seed(config, deep, start, seed, kept, read["drawn"], aside, of_this, read["groups"]).values()
    return dict(zip(DRAWN, (found, read, start, the_map, at_least, directory, theirs, sampling, training, batch, recipe, of_this, solves, table, enough, kept, ordinary, of_ordinary,
                            sets, counted, sha256, pretraining, aside), strict=True))


def ladder_l5_draws_prepare(config: dict) -> dict:
    draws = the_settings(config)
    store = _store(config, draws)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    seed, deep, smoke = training_seed(config), draws.deep, smoke_run()
    try:
        the_seed(seed, draws.seeds, draws.more, draws.approved, draws.of_l5)
    except ValueError as error:
        raise RuntimeError(f"{error} ({SETTING}). Nothing was written.") from error
    if smoke and not os.environ.get(RUN_VARIABLE):
        raise RuntimeError(f"{SMOKE_VARIABLE} is set: it is for a SMOKE run alone, in a run directory the task names ({RUN_VARIABLE}) and on the fixtures' base map. A run in "
                           "the seed's own directory reads a run of L5 whose checks passed. Nothing was written.")
    # ---- THE RECIPE: the setting's, held to what L5n's report names (the real run's, at a smoke run too)
    of_l5n = read_l5n(config, draws)
    try:
        of_the_recipe = the_recipe(draws.recipe, of_l5n)
    except ValueError as error:
        raise RuntimeError(f"{error} ({SETTING}.recipe; L5n's run is written by {l5n_task(draws)}). Nothing was written.") from error
    # ---- L5's finished run and every stored run its task read, how a model is measured and trained, and THE THREE SETS by L5's own rule for THIS TASK's seed
    averages = draws.recipe == MEAN_CONSTANT
    drawn = drawn_by_the_seed(config, draws, seed, smoke, the_reader(draws, smoke), averages)
    found, read, start, at_least, directory, theirs, sampling, training, batch, recipe, solves, table, enough, kept, ordinary, sets, counted, sha256, aside = (drawn[key] for key in (
        "found", "read", "start", "at_least", "directory", "theirs", "sampling", "training", "batch", "recipe", "solves", "table", "enough", "kept", "ordinary", "sets", "counted",
        "sha256", "aside"))
    goal_ids = [row["problem_id"] for row in read["goal_problems"]]
    try:
        beside = {model: refuse_problems_of(sets[model], set(goal_ids), set(aside), f"the training set of `{model}`") for model in MODELS}
        in_orders, coincide = {}, {}
        for model in MODELS if averages else ():        # THE AVERAGED RECIPE: each set in the content-hash order of each ORDER SEED, as L5n orders rows
            by_order, coincide[model] = four_orders(sets[model], smoke)
            in_orders.update({trained_of(model, order): numbered(by_order[order]) for order in ORDERS})
    except ValueError as error:
        raise RuntimeError(f"{error} (seed {seed}; L5's run {directory.name}). Nothing was written.") from error
    trained = list(trained_models(draws.recipe))
    count, steps = len(kept), steps_per_pass(2 * len(kept), batch)

    # ---- from here on files are written: the goal problems of each model's set, the goal set's groups, G', the held-aside problems, the start model's first sampling, the sets
    store.write_rows("problems.jsonl", [{**row, "set": the_set(model, sampling["name"])} for model in MODELS for row in read["goal_problems"]])
    store.write_rows(ladder_l4.GROUPS_FILE, read["groups"])
    store.write_rows(ladder_l4.AGAIN_FILE, [{"problem_id": problem_id} for problem_id in read["again"]])
    store.write_rows(ladder_l5.HELD_ASIDE_FILE, read["aside"])
    store.write_rows(stored_file(draws.start, sampling["name"], L4), read["kept"][draws.start, sampling["name"]])
    for model in MODELS:
        store.write_rows(THE_STAGE.training_file(model), sets[model])
    for name, own in in_orders.items():
        store.write_rows(THE_STAGE.training_file(name), own)
    is_trained = {CONSTANT: "the recipe as it is (five warm-up steps, then a constant learning rate), one training a model, in the order L5 itself trains that seed's set",
                  DECAY: "ONE change: after the warm-up the learning rate falls linearly to zero at the end of the pass (L5n's `decay`); one training a model, in the seed's order",
                  MEAN_CONSTANT: f"the recipe as it is, each set trained {len(ORDERS)} times, in the content-hash order of the order seeds {list(ORDERS)} of its own rows; the model "
                                 "measured is the adapter whose every number is the mean of the four's (L5n's own function)"}[draws.recipe]
    report = found["report"]
    # The learning-rate factor a training's recorded rates are held to, as L5n records it: the one of this run's recipe (the recipe as it is records no rate: L5's own call).
    factors = ladder_l5_noise.learning_rate_factors(training["warmup_steps"], steps)
    factor = {key: factors[key] for key in ("what", "steps", "warmup_steps", DECAY if draws.recipe == DECAY else CONSTANT)}
    summary = {
        "stage": L4, "label": LABEL, "check": "L5d", "seed": seed, "start": draws.start, "start_adapter": str(start.adapter), "start_run": str(start.directory),
        "models": list(MODELS), "trained": trained, "averaged": list(MODELS) if averages else [], "stand_in_engine": _stand_in(), "effective_batch": batch,
        "the_recipe": {"what": f"how a model is trained is a setting ({SETTING}.recipe), held to what L5n's report names for what follows: {is_trained}", **of_the_recipe,
                       "of_l5n": of_l5n},
        "the_seeds": {"what": "the fresh seeds, one task each; the further ones run with explicit approval, recorded in the setting; L5's own seeds found the sign and are "
                              "no seeds of this stage", "seeds": list(draws.seeds), "more_seeds": list(draws.more), "more_seeds_approved": draws.approved,
                      "fixed": list(fixed_seeds(draws.seeds, draws.more, draws.approved)), "of_l5": list(draws.of_l5)},
        "the_search": {"what": f"L5's finished search of seed {deep.seed_of_the_search}, ONLY READ: nothing of it is sampled again. BY THIS TASK'S SEED, as a further seed of L5 "
                               "draws them: which of its deep solves the cap keeps and which verified attempt of each; which ordinary solves and which attempt; the published "
                               "rows (the first D and the first 2D of the hash order of this seed); each set's order", "run": str(directory),
                       "seed": deep.seed_of_the_search, "problems": len(read["drawn"]), "deep_solves": len(solves), "rows_of_each_kind": count, "cap": deep.cap,
                       "enough_deep_solves": enough, "found": table, "headline": report.get("headline"), "inconclusive": bool(report.get("inconclusive")),
                       "ordinary_solves_drawn": len(ordinary), "held_aside_problems": len(aside), "held_aside_file": ladder_l5.HELD_ASIDE_FILE},
        "the_sets": {model: {"what": {PUBLISHED: f"the first {2 * count:,} rows of the hash order of this seed among the rows `{draws.start}` was pretrained on, and nothing else",
                                      SHALLOW: f"{count:,} ordinary solves (one verified map attempt each, drawn by the seed) and the first {count:,} rows of that order",
                                      DEEP: f"{count:,} deep rows (one verified attempt of each deep solve the cap keeps, drawn by the seed) and the SAME first {count:,} rows"}[model],
                             "file": THE_STAGE.training_file(model), **described(sets[model], batch), "barred": counted[model],
                             "problems_of_g_or_of_the_held_aside_set": {key: beside[model][key] for key in ("problems_of_the_goal_set", "problems_of_the_held_aside_set")}}
                     for model in MODELS},
        "training_sets": {name: ({"what": f"the set of `{name}`, as L5's rule builds it for this seed, in the seed's own content-hash order", "file": THE_STAGE.training_file(name),
                                  "model": name, **described(sets[name], batch)} if name in sets else
                                 {"what": f"the set of `{name.rpartition('_')[0]}` in the content-hash order of ORDER SEED {name.rpartition('_')[2]}",
                                  "file": THE_STAGE.training_file(name), "model": name.rpartition("_")[0], "order_seed": int(name.rpartition("_")[2]),
                                  **described(in_orders[name], batch)}) for name in trained},
        "the_averaged": {model: {"what": "trained by nothing: every number of its adapter is the mean of its four trainings' numbers at that place",
                                 "of": list(its_trainings(draws.recipe, model))} for model in MODELS} if averages else {},
        "the_orders": {"what": "each set in the content-hash order of each ORDER SEED (`assembly.training_order`: a hash of the order seed and each row's id), as L5n orders "
                               "rows. An order seed that is the task's own seed gives the seed's own order: it is trained all the same, and named here",
                       "order_seeds": list(ORDERS), **{model: coincide[model] for model in MODELS}} if averages else None,
        "pretraining_file": str(ladder_l4.pretraining_file()), "pretraining_file_sha256": sha256,
        "recipe": {"what": "L4t's (`ladder_ceiling.one_pass`, as the arm's twin is trained): FROM THE START ADAPTER, one pass, the native format. It is held to the recipe "
                           f"L5's run trained its models with. This run: {is_trained}", **recipe, "adapter_seed": seed, "optimizer_steps": steps,
                   **({} if draws.recipe == CONSTANT else {"learning_rate_factor": factor})},
        "the_sampling": {"what": f"every model is measured on the goal set's FIRST sampling alone, as the run of `{draws.start}` recorded it: its attempts a problem and its "
                                 "sampling seed, so that each pairs by problem with every stored model's first sampling. No rung, no second sampling, no held-aside set",
                         **sampling, "sets": {model: the_set(model, sampling["name"]) for model in MODELS},
                         "stored_start": stored_file(draws.start, sampling["name"], L4)},
        "goal_samplings": [sampling], "goal_set": len(goal_ids), "goal_set_again": len(read["again"]), "attempts_a_goal_problem": sampling["episodes"],
        "smoke": {"what": f"{SMOKE_VARIABLE}: a smoke run alone. L5's smoke run reads INCONCLUSIVE (a training of two rows has no loss to compare), which a real run is refused "
                          "for, and is read all the same; its two looser draws are made again as that run recorded them, and one deep solve is enough; orders of a set's "
                          "handful of rows that are alike are trained all the same; and an averaged adapter that is one of its models is kept all the same, and said",
                  "problems_at_least": at_least, "l5_read_though_inconclusive": bool(report.get("inconclusive")), "l5s_failed_checks": list(report.get("failed_checks") or [])}
        if smoke else None,
        "contradicted_side_setting": config["ladder_loop"]["episode"].get("contradicted_side", "all"),
        "start_recipe": theirs["start_recipe"], "start_run_read": theirs["start_run_read"]}
    store.mark_done(PREPARE, summary)
    _say(f"L5d prepared, seed {seed}, from `{draws.start}`, the recipe `{draws.recipe}` ({of_the_recipe['found']}): the three sets of {2 * count:,} rows by L5's own rule for "
         f"this seed, from the search of {directory.name} ({len(solves):,} deep solves, D = {count:,}); {len(trained)} trainings of {steps:,} optimizer steps"
         + (f", then {len(MODELS)} averaged adapters" if averages else "") + f"; each of the {len(MODELS)} models measured on `{sampling['name']}` alone: {sampling['episodes']} "
         f"attempts at each of {len(goal_ids)} goal problems, sampling seed {sampling['sampling_seed']}" + ("; A SMOKE RUN" if smoke else ""))
    return summary


def _of_this_run(store: ArtifactStore, draws: Draws, what: str) -> dict:
    """The prepare step's summary, for a later step of THIS run. Refused (RuntimeError) when the setting now names
    another recipe than the run was prepared with: A RUN KEEPS ITS RECIPE."""
    _need(store, PREPARE, what, ENTRY_STAGE)
    prepared = store.done_summary(PREPARE)
    if prepared["the_recipe"]["recipe"] != draws.recipe:
        raise RuntimeError(f"this run ({store.root.name}) was prepared with the recipe `{prepared['the_recipe']['recipe']}` and {SETTING}.recipe now names `{draws.recipe}`: a "
                           f"run keeps its recipe, and seeds of two recipes are not read together. Name the recipe it was prepared with, or move the run away. {what} is not made")
    return prepared


def _not_of_the_recipe(prepared: dict, step: str, what: str) -> dict:
    """What a step the run's recipe does not use returns, with nothing marked done and nothing made."""
    why = f"the recipe of this run is `{prepared['the_recipe']['recipe']}`: {what}"
    _say(f"L5d: the step {step} makes NOTHING: {why}")
    return {"stage": L4, "label": LABEL, "check": "L5d", "run": False, "why": why}


# --------------------------------------------------------------------------------------------------- train
def ladder_l5_draws_train(config: dict, name: str) -> dict:
    """One training, by L4t's own training body (`ladder_l4_rows.train_a_model`) under this stage's names: from the
    start adapter at its rank, one pass over its stored set. `constant`: EXACTLY the call L5 trains a seed's model by
    (no schedule; nothing more is recorded). `decay`: ONE change, the learning rate falling linearly to zero after the
    warm-up, each step's rate recorded and held to that schedule. `mean_constant` (`<model>_<k>`): the recipe as it is
    on the set in the order of order seed k, each step's rate recorded and held to it, as L5n's `constant_<k>`."""
    draws = the_settings(config)
    store, marker = _store(config, draws), THE_STAGE.train_marker(name)
    if store.is_done(marker):
        return store.done_summary(marker)
    prepared = _of_this_run(store, draws, f"the training of `{name}`")
    if name not in prepared["trained"]:
        return _not_of_the_recipe(prepared, marker, "a model is trained " + ("in four orders, and never once in the seed's own" if prepared["averaged"] else
                                                                           "once, in the seed's own order, and never in the order of an order seed"))
    schedule = None
    if draws.recipe == DECAY:
        schedule = ladder_l4_rows.Schedule(decay=True, order="a content hash of the task's seed and each row's id (`assembly.training_order`)")
    elif draws.recipe == MEAN_CONSTANT:
        schedule = ladder_l4_rows.Schedule(decay=False, order=f"a content hash of the ORDER SEED {prepared['training_sets'][name]['order_seed']} and each row's id "
                                                              "(`assembly.training_order`): the set's own rows, in one of the four orders its averaged adapter is the mean of")
    return train_a_model(config, store, name, draws.start, run_name(draws, training_seed(config)), THE_STAGE, schedule=schedule)


# ------------------------------------------------------------------------------------------------- average
def ladder_l5_draws_average(config: dict, model: str) -> dict:
    """`mean_constant`, NO TRAINING: the adapter `model`, whose every number is the mean of its four trainings' numbers
    at its place, BY L5n's OWN FUNCTION (`ladder_l5_noise.the_mean_adapter`): saved under `adapters/<model>` with their
    adapter config, read back, and held to be that mean, none of them and at the start adapter's rank, as L5n holds its
    own. Refused there, with the step not marked done."""
    draws = the_settings(config)
    store, marker = _store(config, draws), average_marker(model)
    if store.is_done(marker):
        return store.done_summary(marker)
    prepared = _of_this_run(store, draws, f"the averaged adapter `{model}`")
    if model not in prepared["averaged"]:
        return _not_of_the_recipe(prepared, marker, "no adapter is averaged, and each model measured is its one training's")
    of = list(prepared["the_averaged"][model]["of"])
    summary = {"stage": L4, "label": LABEL, "check": "L5d", "model": model, "recipe": draws.recipe, "of": of, "adapter": str(store.root / ladder_assembly.ADAPTERS / model),
               "stand_in_engine": _stand_in()}
    return ladder_l5_noise.the_mean_adapter(store, the_start(config, draws), model, of, marker, summary, bool(prepared.get("smoke")), THE_STAGE)


# ------------------------------------------------------------------------------------------------- measure
def ladder_l5_draws_measure(config: dict, model: str) -> dict:
    """One of the three models, trained or averaged, on the goal set's FIRST sampling alone, as L5n measures
    (`ladder_ceiling.measure_model` with no rung): the attempts and the sampling seed the start model's run recorded
    for that sampling, at the config's own temperature, from the adapter this run keeps. The model server is started
    with the start adapter's rank as its largest (`config_from`)."""
    draws = the_settings(config)
    store = _store(config, draws)
    prepared = _of_this_run(store, draws, f"the measurement of `{model}`")
    averaged = model in prepared["averaged"]
    adapter, sampling = store.root / ladder_assembly.ADAPTERS / model, prepared["the_sampling"]
    return measure_model(config_from(config, prepared, the_start(config, draws), serving=True), store, {"goal_samplings": prepared["goal_samplings"]}, Model(
        name=model, called=f"L5d's model `{model}`", whose=f"L5d's model `{model}`", marker=THE_STAGE.measure_marker(model),
        trained_by=average_marker(model) if averaged else THE_STAGE.train_marker(model), adapter=adapter,
        no_adapter=f"the adapter `{model}` is kept by this stage, and one made again would not be the model its other figures stand beside. The measurement cannot be made",
        detail=f"G's first sampling alone, {sampling['episodes']} attempts a problem, sampling seed {sampling['sampling_seed']}",
        summary={"stage": L4, "label": LABEL, "check": "L5d", "model": model, "adapter": str(adapter)}, stage=ENTRY_STAGE, label=L4, also=what_else_a_model_wrote, rungs=False))


# -------------------------------------------------------------------------------------------------- report
def read_the_seeds(config: dict, draws: Draws, own: int) -> dict[int, dict]:
    """The OTHER fresh seeds of the setting, as the read over the seeds takes them: what each one's report says of its
    own seed (`of_this_seed`) WHEN ITS RUN IS ON THE BOX AND ITS REPORT WRITTEN, NOT THERE otherwise, with why. NOTHING
    IS REFUSED FOR THEM. A run in a directory the task names (a smoke run) reads no other seed."""
    found = {}
    for seed in (*draws.seeds, *draws.more):
        if seed == own:
            continue
        if os.environ.get(RUN_VARIABLE):
            found[seed] = {"there": False, "why": f"this run is in a directory the task names ({RUN_VARIABLE}: a smoke run), and no other seed's run is read from it"}
            continue
        directory = _runs(config) / run_name(draws, seed)
        lacking = [name for name in (f"{REPORT}.done.json", REPORT_FILE) if not (directory / name).exists()]
        entry = {} if lacking else json.loads((directory / REPORT_FILE).read_text()).get("of_this_seed") or {}
        if lacking or entry.get("seed") != seed:
            found[seed] = {"there": False, "run": str(directory), "why": f"{directory.name} does not hold {lacking}" if lacking else
                           f"the report of {directory.name} is not a report of L5d at seed {seed}"}
            continue
        found[seed] = {**entry, "there": True, "run": str(directory)}
    return found


def read_l5s_seeds(config: dict, draws: Draws, prepared: dict, goal_ids: list[str]) -> dict[int, dict]:
    """L5's three models at each of L5's own seeds, on the sampling every model here is measured on: each seed's
    per-problem rows WHEN ITS RUN IS ON THE BOX AND FINISHED (its report written and to be read, its three models
    measured, its stored rows pairing with this run's problems, sampling seed and attempts), NOT THERE otherwise, with
    why. NOTHING IS REFUSED FOR THEM. Read only when this run's recipe is the one L5 trained with: under another they do
    not stand beside these. `goal_ids`: the goal problems, as this run's copy of the start model's first sampling has
    them."""
    if prepared["the_recipe"]["recipe"] != OF_L5:
        return {}
    deep, sampling, of_l5, found = draws.deep, prepared["the_sampling"], ladder_l5.THE_STAGE, {}
    for seed in draws.of_l5:
        directory, sets = _runs(config) / ladder_l5.own_run(config, deep, seed), {model: the_set(model, sampling["name"]) for model in MODELS}
        names = (f"{ladder_l5.REPORT}.done.json", ladder_l5.REPORT_FILE, *(f"{of_l5.measure_marker(model)}.done.json" for model in MODELS),
                 *(name for set_name in sets.values() for name in (f"episodes_{set_name}_problems.jsonl", f"episodes_{set_name}.done.json")))
        lacking = [name for name in names if not (directory / name).exists()]
        if lacking:
            found[seed] = {"there": False, "run": str(directory), "why": f"{directory.name} does not hold {lacking}"}
            continue
        report = json.loads((directory / ladder_l5.REPORT_FILE).read_text())
        try:
            if not report.get("ok", True):
                raise RuntimeError(f"the report of {directory.name} is not to be read")
            rows = {model: ladder_ceiling._stored(directory, set_name, goal_ids, sampling["sampling_seed"], sampling["episodes"], f"`{model}` of L5's seed {seed}", READING)[0]
                    for model, set_name in sets.items()}
        except RuntimeError as error:
            found[seed] = {"there": False, "run": str(directory), "why": str(error)}
            continue
        found[seed] = {"there": True, "run": str(directory), "rows": rows, "inconclusive": bool(report.get("inconclusive"))}
    return found


def ladder_l5_draws_report(config: dict) -> dict:
    from rlvr_lean.gpu.ladder_l3c import proof_lengths
    from rlvr_lean.reporting.ladder_l5_draws import build_l5_draws_report

    draws = the_settings(config)
    store = _store(config, draws)
    prepared = _of_this_run(store, draws, "L5d's report")
    trained_models_, averaged_models, sampling = prepared["trained"], prepared["averaged"], prepared["the_sampling"]
    for step in (*(THE_STAGE.train_marker(name) for name in trained_models_), *(average_marker(model) for model in averaged_models),
                 *(THE_STAGE.measure_marker(model) for model in prepared["models"])):
        _need(store, step, "L5d's report", ENTRY_STAGE)
    sets = {name: store.read_rows(THE_STAGE.training_file(name)) for name in dict.fromkeys((*prepared["models"], *trained_models_))}
    trains, trained = {}, {}
    for name in trained_models_:        # what a model WAS trained on is the training loop's own record of row ids, read back to its stored set's rows (never their text)
        trains[name] = store.done_summary(THE_STAGE.train_marker(name))
        by_id = {row["id"]: row for row in sets[name]}
        trained[name] = [{key: by_id[row_id].get(key) for key in ("id", "problem_id", "origin", "row_of_the_file")}
                         for row_id in json.loads(store.path(THE_STAGE.loss_file(name)).read_text())["rows_trained"]]
    averaged = {model: store.done_summary(average_marker(model)) for model in averaged_models}
    measured = {model: store.done_summary(THE_STAGE.measure_marker(model)) for model in prepared["models"]}
    rows = {model: _results(store, the_set(model, sampling["name"])) for model in prepared["models"]}
    start = the_start(config, draws)
    # THE SETS ARE MADE AGAIN AND HELD by the report: against the rows the start model was pretrained on, read as the first step read them (the pretraining file, held to its hash).
    pretraining = ladder_l4_start.pretrained_on(start.directory, prepared["pretraining_file_sha256"], start.name)[0]
    base_map = [row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[1]]
    ids_of = lambda name: [row["problem_id"] for row in store.read_rows(name)]      # noqa: E731
    start_rows = store.read_rows(sampling["stored_start"])
    # The length of a held-out problem's published proof is read HERE, by the report, and by nothing before it.
    report = build_l5_draws_report(prepared, trains, trained, averaged, measured, rows, sets, pretraining, start_rows, store.read_rows(ladder_l4.GROUPS_FILE), proof_lengths(),
                                   ids_of(ladder_l4.AGAIN_FILE), ids_of(ladder_l5.HELD_ASIDE_FILE), config["ladder_loop"]["l4"], base_map,
                                   {"sampling": draws.sampling, "attempts": draws.attempts}, read_the_seeds(config, draws, prepared["seed"]), fixed_seeds(draws.seeds, draws.more, draws.approved),
                                   read_l5s_seeds(config, draws, prepared, [row["problem_id"] for row in start_rows]))
    directory = store.root / ladder_assembly.ADAPTERS
    report["adapters"] = {"kept": True, "what": f"the adapters of {', '.join(f'`{name}`' for name in dict.fromkeys((*trained_models_, *prepared['models'])))} ({LABEL}) are KEPT: "
                                                f"no step of this stage deletes one. `{draws.start}` is its own run's and was only read",
                          "directory": str(directory), "there": sorted(path.name for path in directory.glob("*") if path.is_dir()) if directory.is_dir() else []}
    _write_json(store, REPORT_FILE, report)
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    for name in (THE_STAGE.loss_file(name) for name in trained_models_):
        store.mirror(name)
    for line in report["lines"]:
        print(line, flush=True)
    store.mark_done(REPORT, {"stage": L4, "label": LABEL, "check": "L5d", "seed": prepared["seed"], "recipe": draws.recipe, "headline": report["headline"],
                             "branches": report["branches"], "measured": list(prepared["models"]), "ok": report["ok"]})
    return report


def _for(step, name: str):
    def run(config: dict) -> dict:
        return step(config, name)
    return run


# In the order the stage runs them, THE STEPS OF THE LARGEST RECIPE: the three sets built; then model by model its one training (`constant`, `decay`) and its four
# (`mean_constant`), its averaged adapter (`mean_constant`) and its measurement, before the next model; then the report. A step the run's recipe does not use makes nothing.
STEPS = {PREPARE: ladder_l5_draws_prepare}
for _model in MODELS:
    for _name in (_model, *(trained_of(_model, _order) for _order in ORDERS)):
        STEPS[THE_STAGE.train_marker(_name)] = _for(ladder_l5_draws_train, _name)
    STEPS[average_marker(_model)] = _for(ladder_l5_draws_average, _model)
    STEPS[THE_STAGE.measure_marker(_model)] = _for(ladder_l5_draws_measure, _model)
STEPS[REPORT] = ladder_l5_draws_report
