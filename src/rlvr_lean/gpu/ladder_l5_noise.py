"""L5n: what is one training worth? The spread between trainings, and a recipe that lowers it. Spec:
docs/spec/ladder-loop.spec.md, "L5n: what is one training worth? The spread between trainings, and a recipe that
lowers it" and its "Made exact by the build". Everything it makes is labelled pretrained on published proofs. Each step
is its own process (`rlvr_lean.runner.entry`, stage `ladder_l5_noise`). NO ROUND, NO SEARCH, NO DRAW OF ROWS.

THE ROWS ARE L5's, READ AND NEVER DRAWN AGAIN: the set L5's own task of its search's seed trained `published` on, as that
FINISHED run stored it (`ladder_l5.search_directory`), with every stored run that task read, by L5's OWN first-step
reading (`ladder_l5.read_the_search`, `read_for_the_search`: its refusals are a further seed's of L5). The run directory
is its own (`ladder_l5_noise_<start>_seed<N>`).

  ladder_l5_noise_prepare           REFUSES, naming the task to run, before anything is written: L5's run missing,
                                    unfinished, not to be read, INCONCLUSIVE or made otherwise than this task would make it;
                                    one that trained no `published`, or whose stored set is not what its rule draws or what
                                    its training recorded; a recipe or a sampling that is not that run's; an order that is
                                    the one that run trained in; a task of another seed. STORES the fixed rows, by id and
                                    place and never with text, and each model's set: the rows in the content-hash order of
                                    its ORDER SEED. No GPU, no Lean
  ladder_l5_noise_train_<model>     for each order seed in turn `constant_<k>`, then `decay_<k>`: L4t's OWN training body
                                    (`ladder_l4_rows.train_a_model`) under this stage's names, FROM THE START ADAPTER at its
                                    rank, one pass. `constant`: the recipe as it is. `decay`: ONE change, the learning rate
                                    falling linearly to zero after the warm-up. Every training records each step's learning
                                    rate, held to its schedule before the step is marked done
  ladder_l5_noise_measure_<model>   the model on the goal set's FIRST sampling alone (`ladder_ceiling.measure_model`, as
                                    L4t's measurements at another temperature sample G alone): the attempts and the
                                    sampling seed the start model's run recorded for that sampling. No rung, no second
                                    sampling, no held-aside set
  ladder_l5_noise_average_<recipe>  NO TRAINING: `mean_<recipe>`, the adapter whose every number is the mean of that
                                    recipe's adapters' (`infrastructure.adapter_files.mean_of`), saved with their adapter
                                    config; read back, held to be that mean and none of them, and to the start adapter's rank
  ladder_l5_noise_measure_mean_<recipe>   the averaged model, measured as every other
  ladder_l5_noise_report            the read fixed before the run (`reporting/ladder_l5_noise.py`); beside it L5's
                                    `published` at each of its seeds, read from those runs WHEN THEY ARE ON THE BOX AND
                                    FINISHED (NOT THERE otherwise, and nothing is refused for them)

Every step is resumable and a step of its own. NO STEP DELETES AN ADAPTER, and no step writes into a run it reads.

A PUBLISHED ROW is named by its id, its problem and its place in the file, never with its text: the training step reads
the text from the pretraining file (`data/ladder_l4/pretraining.jsonl`, not distributed), held to the SHA-256 the start
model's run recorded.

A SMOKE RUN ALONE (`RLVR_LEAN_LADDER_L5_NOISE_SMOKE_ORDERS`, refused unless the run directory is one the task names and
the map is the fixtures'): L5's smoke run reads INCONCLUSIVE (a training of two rows has no loss to compare) and is read
all the same; only that many of the setting's orders are trained; an order that is the one L5 trained its handful of
rows in, or another's, is trained all the same; and an averaged adapter that IS one of its models (trainings of one
optimizer step on the same rows are one training) is kept and served all the same. The first step records what it went
on from, and the averaging step what it kept.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from rlvr_lean.domain.ladder_round.dose import decayed, steps_per_pass, warmed
from rlvr_lean.domain.ladder_round.l5 import PUBLISHED, refuse_barred
from rlvr_lean.domain.ladder_round.l5_noise import CONSTANT, DECAY, ORDERS, RECIPES, mean_of, model_of, models_of, numbered, parts_of, the_orders, the_rows
from rlvr_lean.domain.ladder_round.read import GOAL
from rlvr_lean.gpu import ladder_assembly, ladder_ceiling, ladder_l4, ladder_l4_rank, ladder_l4_rows, ladder_l4_start, ladder_l5, ladder_loop
from rlvr_lean.gpu.ladder_ceiling import Model, Reading, _need, goal_set, measure_model, what_a_model_wrote
from rlvr_lean.gpu.ladder_dose import _rows, _runs
from rlvr_lean.gpu.ladder_l3d1 import what_else_a_model_wrote
from rlvr_lean.gpu.ladder_l4 import L4, LABEL, _say
from rlvr_lean.gpu.ladder_l4_rows import described, train_a_model
from rlvr_lean.gpu.ladder_l4_start import Start, config_from
from rlvr_lean.gpu.ladder_round import _results, _stand_in, _write_json, training_seed
from rlvr_lean.infrastructure.adapter_files import ADAPTER_FILE, MEAN_TYPES, compared, is_the_mean, tensors_of
from rlvr_lean.infrastructure.adapter_files import mean_of as mean_of_adapters
from rlvr_lean.infrastructure.artifact_store import ArtifactStore

L5N = "l5_noise"                                            # what this stage's own files are called
STAGE = ENTRY_STAGE = "ladder_l5_noise"                     # what its steps begin with; the stage, as `runner.entry` names it; its run directory begins with it
RUN_VARIABLE = "RLVR_LEAN_LADDER_L5_NOISE_RUN"              # another run directory than `ladder_l5_noise_<start>_seed<seed>` (a smoke run)
SMOKE_VARIABLE = "RLVR_LEAN_LADDER_L5_NOISE_SMOKE_ORDERS"   # a SMOKE run alone (refused elsewhere): only this many of the setting's orders are trained, and what else the module's note says
SETTING = "ladder_loop.l5_noise"
THE_STAGE = ladder_l4_rows.Stage(name=ENTRY_STAGE, steps=STAGE, files=L5N, reader="L5n", start_said_by=f"{ladder_l5.SETTING}.start")
PREPARE, REPORT = THE_STAGE.prepare, f"{STAGE}_report"
REPORT_FILE = "report_ladder_l5_noise.json"
ROWS_FILE = f"{L5N}_rows.jsonl"                             # THE FIXED ROWS, by id and place (never with text), in the order L5's own task trained them in
READING = Reading(reader="L5n", models="L5n's models", not_read="that set stands beside this run's, and no report of this run could be read against it", setting=ladder_l4.SETTING)


@dataclass(frozen=True)
class Noise:
    """`ladder_loop.l5_noise`, as the stage's steps read it."""
    start: str                          # L5's start: the fixed rows are the rows a model of L5 was trained on from it, and every model here is trained from it at its rank
    orders: tuple[int, ...]             # the order seeds: the fixed rows in the content-hash order of each, trained once with each recipe
    recipes: tuple[str, ...]            # the two recipes, in the order an order is trained with them
    sampling: str                       # the sampling of the goal set every model here is measured on, by the name the start model's run gave it: its FIRST
    attempts: int                       # ... and its attempts a problem, as that run recorded them
    seeds: tuple[int, ...]              # L5's seeds whose `published` stands beside (rows and order both drawn): read from their runs when they are on the box
    deep: ladder_l5.Deep                # L5's settings as ITS task reads them: the fixed rows are of the finished run of its search's seed


def the_settings(config: dict) -> Noise:
    """THE one place that reads `ladder_loop.l5_noise`. Refused (ValueError): orders or recipes that are not the stage's
    (its steps are registered for them: change both together); an order seed that is the seed L5's own task trained in;
    attempts that are no whole number of at least 1; a sampling without a name; seeds of L5 that are not distinct whole
    numbers, or that leave out the seed whose task trained the fixed rows; L5's own settings, when its reader refuses
    them."""
    settings = config["ladder_loop"]["l5_noise"]
    orders, recipes, seeds = tuple(settings["orders"]), tuple(settings["recipes"]), tuple(settings["published_seeds"])
    if orders != ORDERS or recipes != RECIPES:
        raise ValueError(f"{SETTING}.orders is {list(orders)} and recipes {list(recipes)}; the stage trains and measures {list(ORDERS)} with {list(RECIPES)}: change both together")
    deep = ladder_l5.the_settings(config)
    if deep.seed_of_the_search in orders:
        raise ValueError(f"{SETTING}.orders is {list(orders)} and {ladder_l5.SETTING}.search_seed is {deep.seed_of_the_search}: that order is the one L5's own task trained "
                         "the fixed rows in, and each order here is another draw of the order")
    attempts, sampling = settings["attempts"], settings["sampling"]
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1 or not isinstance(sampling, str) or not sampling:
        raise ValueError(f"{SETTING}.attempts is {attempts!r} and sampling {sampling!r}: a whole number of at least 1, and the name of the start model's first sampling of the "
                         "goal set")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds) or deep.seed_of_the_search not in seeds:
        raise ValueError(f"{SETTING}.published_seeds is {list(seeds)}: distinct whole numbers of at least 0, the seed whose task trained the fixed rows "
                         f"({ladder_l5.SETTING}.search_seed, {deep.seed_of_the_search}) among them")
    return Noise(start=deep.start, orders=orders, recipes=recipes, sampling=sampling, attempts=attempts, seeds=seeds, deep=deep)


def run_name(noise: Noise, seed: int) -> str:
    """The stage's run directory, with the start adapter's name in it: `ladder_l5_noise_pre_r64_seed<N>`."""
    return f"{ENTRY_STAGE}_{noise.start}_seed{seed}"


def _store(config: dict, noise: Noise) -> ArtifactStore:
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    return ArtifactStore(_runs(config) / (os.environ.get(RUN_VARIABLE) or run_name(noise, training_seed(config))), Path(mirror) if mirror else None)


def the_start(config: dict, noise: Noise) -> Start:
    """The start model, and where it lives (`ladder_l4_start.the_start`): L5's."""
    return ladder_l4_start.the_start(config, noise.start, THE_STAGE.start_said_by)


def smoke_orders() -> int:
    """A smoke run's number of orders; 0 for every other run: the setting's, and nothing else."""
    return max(0, int(os.environ.get(SMOKE_VARIABLE) or 0))


def average_marker(recipe: str) -> str:
    """The step that makes a recipe's averaged adapter, and its marker."""
    return f"{STAGE}_average_{recipe}"


def stored_published_file(seed: int) -> str:
    """L5's `published` of one seed of L5 on the sampling every model here is measured on, as this run's own directory holds it."""
    return f"{L5N}_stored_{PUBLISHED}_seed{seed}.jsonl"


def the_set(name: str, sampling: str) -> str:
    """The set a model's attempts at the goal problems are stored under, as every stage names a model's sampling of G."""
    return goal_set(sampling, name, L4)


def the_reader(noise: Noise, smoke: bool) -> ladder_l5.OfTheSearch:
    """Who reads L5's finished run here, for what L5's own reading says when it refuses it. A SMOKE run alone reads a run
    whose report is INCONCLUSIVE (one not to be read never is)."""
    return ladder_l5.OfTheSearch(
        who="L5n", why=f"L5n draws no row: its fixed rows are the set L5's finished run of seed {noise.deep.seed_of_the_search} trained `{PUBLISHED}` on, read from that run",
        stands="no model here stands beside such a run", models="L5n's models are trained on that run's rows and stand beside its model", inconclusive=smoke)


def learning_rate_factors(warmup_steps: int, steps: int) -> dict:
    """What the prepare step records of each recipe's learning-rate factor: in words, and at a handful of steps (the
    first, the warm-up's last, the one after it, the middle, the last but one, the last)."""
    at = sorted({step for step in (1, warmup_steps, warmup_steps + 1, max(1, steps // 2), steps - 1, steps) if 1 <= step <= steps})
    return {"what": "the learning rate of an optimizer step is the config's times this factor (`domain.ladder_round.dose`). `constant`, the recipe as it is: step / warm-up "
                    "steps over the warm-up, then one. `decay`: the same over the warm-up, then (steps + 1 - step) / (steps + 1 - warm-up steps): one at the warm-up's last "
                    "step, above zero at the last step, zero at the step after it. Each training's recorded rates are held to this",
            "steps": steps, "warmup_steps": warmup_steps,
            CONSTANT: {str(step): warmed(step, warmup_steps) for step in at}, DECAY: {str(step): decayed(step, warmup_steps, steps) for step in at}}


def read_the_published(found: dict, read: dict, noise: Noise) -> dict:
    """What L5n needs of L5's run BESIDE what a further seed of L5 reads of it (`ladder_l5.read_the_search` and
    `read_for_the_search` have read and held that): its model `published`. REFUSED (RuntimeError), naming the task to
    run: a run that trained no model; one without the markers of that model's training and measuring, its stored set,
    its training's own record or its stored rows on the sampling read here; a stored set that is not as many rows as
    that run's own steps counted; stored rows that would not pair. Returns its stored set, the ids its training loop
    recorded, what its training said, and its per-problem rows and attempts on that sampling."""
    directory, task, prepared, of_l5 = found["directory"], ladder_l5.search_task(noise.deep), found["prepared"], ladder_l5.THE_STAGE
    if not (prepared.get("trainings") or {}).get("run"):
        raise RuntimeError(f"L5's run {directory.name} trained no model ({(prepared.get('trainings') or {}).get('why')}): there is no set of `{PUBLISHED}` to train on again. "
                           "Nothing was written.")
    sampling = read["plan"]["goal_samplings"][0]
    set_name = the_set(PUBLISHED, sampling["name"])
    names = (f"{of_l5.train_marker(PUBLISHED)}.done.json", f"{of_l5.measure_marker(PUBLISHED)}.done.json", of_l5.training_file(PUBLISHED), of_l5.loss_file(PUBLISHED))
    lacking = [name for name in names if not (directory / name).exists()] + ladder_ceiling._missing(directory, [set_name])
    if lacking:
        raise RuntimeError(f"{directory} does not hold {lacking}: what L5's run recorded of `{PUBLISHED}`, the set it was trained on, its training's own record of the rows, "
                           f"or its stored rows on the goal set's first sampling. Run {task} to its end first; nothing was written.")
    stored, trained = _rows(directory / of_l5.training_file(PUBLISHED)), json.loads((directory / f"{of_l5.train_marker(PUBLISHED)}.done.json").read_text())
    counted = 2 * prepared["rows_of_each_kind"]
    if not len(stored) == trained.get("rows") == counted:
        raise RuntimeError(f"L5's run {directory.name} stored {len(stored)} rows for `{PUBLISHED}`, its training recorded {trained.get('rows')} and its own step counted "
                           f"{counted} (twice its rows of each kind): the run does not agree with itself. Nothing was written.")
    goal_ids = [row["problem_id"] for row in read["goal_problems"]]
    on_g = ladder_ceiling._stored(directory, set_name, goal_ids, sampling["sampling_seed"], sampling["episodes"], f"`{PUBLISHED}` ({sampling['name']})", READING)[0]
    return {"set": stored, "trained_ids": json.loads((directory / of_l5.loss_file(PUBLISHED)).read_text())["rows_trained"], "trained": trained, "on_g": on_g,
            "distinct": what_a_model_wrote([], ladder_ceiling._stored_attempts(directory, set_name))["distinct_attempts"][GOAL]["mean_share_distinct"]}


# ------------------------------------------------------------------------------------------------- prepare
def ladder_l5_noise_prepare(config: dict) -> dict:
    noise = the_settings(config)
    store = _store(config, noise)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    seed, deep, smoke = training_seed(config), noise.deep, smoke_orders()
    if seed != deep.seed_of_the_search:
        raise RuntimeError(f"this task's seed is {seed} and {ladder_l5.SETTING}.search_seed is {deep.seed_of_the_search}: the fixed rows are that seed's, the orders are the "
                           f"setting's ({SETTING}.orders) and nothing here is drawn by the task's seed, so a task of another seed would train the same models again. Nothing "
                           "was written.")
    if smoke and (not os.environ.get(RUN_VARIABLE) or not 2 <= smoke <= len(noise.orders)):
        raise RuntimeError(f"{SMOKE_VARIABLE} is set ({smoke}): it is for a SMOKE run alone, in a run directory the task names ({RUN_VARIABLE}) and on the fixtures' base "
                           f"map, and it names 2 to {len(noise.orders)} of the setting's orders. A run in the seed's own directory reads a run of L5 whose checks passed and "
                           "trains every order. Nothing was written.")
    # ---- L5's finished run, and every stored run its task read: by L5's OWN reading, as a further seed of L5 reads them (refused there, with nothing written)
    reader = the_reader(noise, bool(smoke))
    found = ladder_l5.read_the_search(config, deep, reader)
    # A smoke run reads a SMOKE run of L5: its two looser draws are made again as that run recorded them (L5's own smoke-only variable is L5's smoke stage's alone).
    read = ladder_l5.read_for_the_search(config, deep, found, reader, at_least=((found["searched"].get("smoke") or {}).get("problems_at_least") or 0) if smoke else 0)
    start, the_map, plan, recorded = (read[key] for key in ("start", "the_map", "plan", "recorded"))
    if smoke and not the_map["fixture"]:
        raise RuntimeError(f"{SMOKE_VARIABLE} is set ({smoke}): it is for a SMOKE run alone, in a run directory the task names ({RUN_VARIABLE}) and on the fixtures' base "
                           "map. A run on the pool's problems reads a run of L5 whose checks passed and trains every order. Nothing was written.")
    of_l5, directory, theirs = read_the_published(found, read, noise), found["directory"], found["searched"]

    # ---- how a model here is measured, and how it is trained: the start model's FIRST sampling of G as its run recorded it; the recipe that trained L5's `published`
    sampling = plan["goal_samplings"][0]
    if (sampling["name"], sampling["episodes"]) != (noise.sampling, noise.attempts):
        raise RuntimeError(f"{SETTING} names the sampling `{noise.sampling}` of {noise.attempts} attempts a problem, and the run of `{noise.start}` ({start.directory.name}) "
                           f"recorded its first sampling of the goal set as `{sampling['name']}` of {sampling['episodes']}: every model here is measured on that sampling "
                           "alone, with its recorded sampling seed. Nothing was written.")
    training, batch = config["training"], config["training"]["effective_batch"]
    recipe = {"learning_rate": training["learning_rate"], "warmup_steps": training["warmup_steps"], "effective_batch": batch,
              "lora": config_from(config, {"start_recipe": recorded}, start)["lora"]}
    then = {key: (theirs.get("recipe") or {}).get(key) for key in recipe}
    if then != recipe:
        raise RuntimeError(f"L5's run {directory.name} trained its models with {then} and this task would train with {recipe}: `{CONSTANT}` is the recipe as it is, the one "
                           f"that trained that run's `{PUBLISHED}`. Nothing was written.")
    if not ladder_ceiling._stand_in():      # the mean of adapters is written in the adapters' own floating type: held HERE, on the start adapter, and not after the eight trainings
        other = sorted({entry["dtype"] for entry in tensors_of(start.adapter / ADAPTER_FILE)[0].values()} - set(MEAN_TYPES))
        if other:
            raise RuntimeError(f"the start adapter {start.adapter} holds tensors of type {other}: the averaged adapters are written in {list(MEAN_TYPES)} "
                               "(`infrastructure.adapter_files.mean_of`), and an adapter trained from it has its types. Nothing was written.")

    # ---- THE FIXED ROWS, held to the file, to L5's rule and to what that training recorded; and each order seed's order of them
    orders = noise.orders[:smoke] if smoke else noise.orders
    try:
        rows = the_rows(of_l5["set"], of_l5["trained_ids"], read["pretraining"], deep.seed_of_the_search)
        by_order, coincide = the_orders(rows, orders, loosely=bool(smoke))
        heldout = {row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[0]} | {row["problem_id"] for row in read["groups"]}
        counted = refuse_barred(rows, heldout, config["ladder_loop"]["l4"]["half_seed"], set(the_map["base_map"]), "the fixed rows")
    except ValueError as error:
        raise RuntimeError(f"{error} (L5's run {directory.name}). Nothing was written.") from error
    models, trained, averaged = models_of(orders, noise.recipes), [model_of(recipe_, order) for order in orders for recipe_ in noise.recipes], [mean_of(recipe_) for recipe_ in noise.recipes]
    sets = {model_of(recipe_, order): numbered(by_order[order]) for order in orders for recipe_ in noise.recipes}
    steps = steps_per_pass(len(rows), batch)

    # ---- from here on files are written: the problems of every set this run samples, the fixed rows, each model's set, L5's own `published` on the sampling read here
    store.write_rows("problems.jsonl", [{**row, "set": the_set(model, sampling["name"])} for model in models for row in read["goal_problems"]])
    store.write_rows(ROWS_FILE, numbered(rows))
    for model, own in sets.items():
        store.write_rows(THE_STAGE.training_file(model), own)
    store.write_rows(stored_published_file(deep.seed_of_the_search), of_l5["on_g"])
    report = found["report"]
    summary = {
        "stage": L4, "label": LABEL, "check": "L5n", "seed": seed, "start": noise.start, "start_adapter": str(start.adapter), "start_run": str(start.directory),
        "orders": list(orders), "recipes": list(noise.recipes), "models": list(models), "trained": trained, "averaged": averaged, "stand_in_engine": _stand_in(),
        "effective_batch": batch,
        "the_rows": {"what": f"THE FIXED ROWS: the set L5's own task of seed {deep.seed_of_the_search} trained `{PUBLISHED}` on, as its finished run stored it, each row by its "
                             "id and its place in the pretraining file and never with its text: held to that file, to the draw L5's rule makes for that seed and to what that "
                             "training's loop recorded. No row is drawn here", "model": PUBLISHED, "run": str(directory), "seed": deep.seed_of_the_search, "file": ROWS_FILE,
                     "read_from": [ladder_l5.THE_STAGE.training_file(PUBLISHED), ladder_l5.THE_STAGE.loss_file(PUBLISHED)], **described(rows, batch), "barred": counted,
                     "its_training": {key: of_l5["trained"].get(key) for key in ("rows", "steps", "adapter", "first_step_loss", "last_step_loss")}},
        "the_orders": {"what": "the fixed rows in the content-hash order of each ORDER SEED (`assembly.training_order`: a hash of the order seed and each row's id). No order "
                               "is the order L5's own task trained the rows in, and no two are one", "order_seeds": list(orders),
                       "the_order_trained_before": coincide["the_order_trained_before"], "alike": coincide["alike"]},
        "training_sets": {model_of(recipe_, order): {"what": f"the fixed rows in the order of order seed {order}, trained " + ("with the recipe as it is" if recipe_ == CONSTANT else
                                                             "with ONE change: after the warm-up the learning rate falls linearly to zero at the end of the pass"),
                                                     "file": THE_STAGE.training_file(model_of(recipe_, order)), "order_seed": order, "recipe": recipe_,
                                                     **described(sets[model_of(recipe_, order)], batch)} for order in orders for recipe_ in noise.recipes},
        "the_averaged": {mean_of(recipe_): {"what": "trained by nothing: every number of its adapter is the mean of its models' numbers at that place",
                                            "of": [model_of(recipe_, order) for order in orders]} for recipe_ in noise.recipes},
        "pretraining_file": str(ladder_l4.pretraining_file()), "pretraining_file_sha256": read["sha256"],
        "recipe": {"what": "L4t's (`ladder_ceiling.one_pass`, as the arm's twin is trained): FROM THE START ADAPTER, one pass, the native format. It is held to the recipe "
                           f"L5's run trained its models with. `{CONSTANT}` is that recipe; `{DECAY}` changes the learning rate after the warm-up and nothing else",
                   **recipe, "adapter_seed": seed, "learning_rate_factor": learning_rate_factors(training["warmup_steps"], steps)},
        "the_sampling": {"what": f"every model is measured on the goal set's FIRST sampling alone, as the run of `{noise.start}` recorded it: its attempts a problem and its "
                                 "sampling seed, so that each pairs by problem with every stored model's first sampling. No rung, no second sampling, no held-aside set",
                         **sampling, "sets": {model: the_set(model, sampling["name"]) for model in models}},
        "goal_samplings": [sampling], "goal_set": len(read["goal_problems"]), "attempts_a_goal_problem": sampling["episodes"],
        "the_published_of_l5": {"what": f"L5's own `{PUBLISHED}` of seed {deep.seed_of_the_search} on that sampling, from its stored rows: the same rows in that seed's order, "
                                        "the recipe as it is. One more draw of `constant`, in no standard deviation", "seed": deep.seed_of_the_search, "run": str(directory),
                                "set": the_set(PUBLISHED, sampling["name"]), "file": stored_published_file(deep.seed_of_the_search),
                                "share_of_distinct_attempts": of_l5["distinct"], "inconclusive": bool(report.get("inconclusive")), "seeds_read_by_the_report": list(noise.seeds)},
        "smoke": {"what": f"{SMOKE_VARIABLE}: a smoke run alone. L5's smoke run reads INCONCLUSIVE (a training of two rows has no loss to compare), which a real run is refused "
                          "for, and is read all the same; only this many of the setting's orders are trained; an order that is the one L5 trained its handful of rows in, or "
                          "another's, is trained all the same; and an averaged adapter that is one of its models (two trainings of one optimizer step on the same rows are "
                          "one training) is kept all the same, and said", "orders": smoke, "orders_of_the_setting": list(noise.orders),
                  "l5_read_though_inconclusive": bool(report.get("inconclusive")), "l5s_failed_checks": list(report.get("failed_checks") or []),
                  "the_order_trained_before": coincide["the_order_trained_before"], "alike": coincide["alike"]} if smoke else None,
        "contradicted_side_setting": config["ladder_loop"]["episode"].get("contradicted_side", "all"),
        "start_recipe": theirs["start_recipe"], "start_run_read": theirs["start_run_read"]}
    store.mark_done(PREPARE, summary)
    _say(f"L5n prepared, seed {seed}, from `{noise.start}`: {len(rows):,} fixed rows (L5's `{PUBLISHED}` of seed {deep.seed_of_the_search}, {directory.name}), {steps:,} "
         f"optimizer steps a training; the order seeds {list(orders)}, each trained as {' and as '.join(f'`{recipe_}`' for recipe_ in noise.recipes)}; then "
         + " and ".join(f"`{name}`" for name in averaged) + f"; each of the {len(models)} models measured on `{sampling['name']}` alone: {sampling['episodes']} attempts at each of "
         f"{len(read['goal_problems'])} goal problems, sampling seed {sampling['sampling_seed']}" + (f"; A SMOKE RUN of {smoke} orders" if smoke else ""))
    return summary


def _of_this_run(store: ArtifactStore, name: str, what: str) -> dict:
    """The prepare step's summary, for a step of a model or a recipe of THIS run. Refused (RuntimeError) for one of an
    order this run does not train (a smoke run trains fewer than the stage's steps are registered for)."""
    _need(store, PREPARE, what, ENTRY_STAGE)
    prepared = store.done_summary(PREPARE)
    if name not in (*prepared["models"], *prepared["recipes"]):
        raise RuntimeError(f"`{name}` is none of this run's models ({prepared['models']}): {what} is not a step of this run. Nothing was written")
    return prepared


# --------------------------------------------------------------------------------------------------- train
def ladder_l5_noise_train(config: dict, model: str) -> dict:
    """One trained model, by L4t's own training body (`ladder_l4_rows.train_a_model`) under this stage's names: from the
    start adapter at its rank, one pass over the fixed rows in ITS order seed's order. `constant`: the recipe as it is.
    `decay`: ONE change, the learning rate falling linearly to zero after the warm-up. Each step's learning rate is
    recorded and held to that schedule before the step is marked done."""
    noise = the_settings(config)
    store = _store(config, noise)
    if store.is_done(THE_STAGE.train_marker(model)):
        return store.done_summary(THE_STAGE.train_marker(model))
    _of_this_run(store, model, f"the training of `{model}`")
    recipe, order = parts_of(model)
    schedule = ladder_l4_rows.Schedule(decay=recipe == DECAY, order=f"a content hash of the ORDER SEED {order} and each row's id (`assembly.training_order`): the fixed rows, in "
                                                                    "another order than L5's own task trained them in")
    return train_a_model(config, store, model, noise.start, run_name(noise, training_seed(config)), THE_STAGE, schedule=schedule)


# ------------------------------------------------------------------------------------------------- average
def _adapter_config(directory: Path) -> dict:
    """An adapter's config as the library saved it, with every list in it sorted (a set of module names is saved in no
    fixed order): what two adapters of one recipe must agree on."""
    recorded = json.loads((directory / ladder_l4_rank.ADAPTER_CONFIG).read_text())
    return {key: sorted(value, key=repr) if isinstance(value, list) else value for key, value in recorded.items()}


def ladder_l5_noise_average(config: dict, recipe: str) -> dict:
    """NO TRAINING: `mean_<recipe>`, the adapter whose every number is the arithmetic mean of the numbers of that
    recipe's trained adapters at its place (`adapter_files.mean_of`), saved under `adapters/mean_<recipe>` with their
    adapter config. Refused (RuntimeError), with the step not marked done: a model of the recipe that is not trained,
    or whose adapter is not there; adapters that do not hold the same tensors, or were saved with another config; a
    file that, READ BACK, is not that mean, is one of its models, or is not at the start adapter's rank. The stand-in
    trains no adapter, and none is averaged."""
    noise = the_settings(config)
    store, marker, model = _store(config, noise), average_marker(recipe), mean_of(recipe)
    if store.is_done(marker):
        return store.done_summary(marker)
    prepared = _of_this_run(store, recipe, f"the averaged adapter `{model}`")
    of = list(prepared["the_averaged"][model]["of"])
    summary = {"stage": L4, "label": LABEL, "check": "L5n", "model": model, "recipe": recipe, "of": of, "adapter": str(store.root / ladder_assembly.ADAPTERS / model),
               "stand_in_engine": _stand_in()}
    return the_mean_adapter(store, the_start(config, noise), model, of, marker, summary, bool(prepared.get("smoke")))


def the_mean_adapter(store: ArtifactStore, start: Start, model: str, of: list[str], marker: str, summary: dict, smoke: bool, one: ladder_l4_rows.Stage = THE_STAGE) -> dict:
    """THE BODY OF AN AVERAGING STEP, which two stages call (this one's; L5d's, `gpu/ladder_l5_draws.py`, whose averaged
    recipe measures such an adapter: change it and both change): the adapter `model`, kept under `adapters/<model>`, is
    the mean of the adapters of the trained models `of`, and is refused as `ladder_l5_noise_average` says. `marker`: the
    step. `summary`: what the step's summary begins with. `smoke`: a SMOKE run, in which an averaged adapter that is one
    of its models is kept. `one`: the stage whose step it is."""
    for name in of:
        _need(store, one.train_marker(name), f"the averaged adapter `{model}`", one.name)
    target = store.root / ladder_assembly.ADAPTERS / model
    if ladder_ceiling._stand_in():
        summary.update({"averaged": False, "note": "no adapter was trained, and none was averaged: a pre-flight without a GPU"})
        store.mark_done(marker, summary)
        return summary
    directories = [store.root / ladder_assembly.ADAPTERS / name for name in of]
    lacking = [str(directory) for directory in directories if not (directory / ADAPTER_FILE).exists() or not (directory / ladder_l4_rank.ADAPTER_CONFIG).exists()]
    if lacking:
        raise RuntimeError(f"{lacking} are not there: the adapters `{model}` is the mean of are kept by this stage. Nothing was averaged")
    configs = [_adapter_config(directory) for directory in directories]
    if any(own != configs[0] for own in configs[1:]):
        raise RuntimeError(f"the adapters of {of} were not saved with one adapter config: `{model}` is saved with the config they share. Nothing was averaged")
    files = [directory / ADAPTER_FILE for directory in directories]
    _say(f"{one.reader}: averaging the adapters of {', '.join(f'`{name}`' for name in of)} into `{model}`: no training")
    target.mkdir(parents=True, exist_ok=True)
    try:
        made = mean_of_adapters(files, target / ADAPTER_FILE)
        shutil.copyfile(directories[0] / ladder_l4_rank.ADAPTER_CONFIG, target / ladder_l4_rank.ADAPTER_CONFIG)
        held = is_the_mean(target / ADAPTER_FILE, files)
    except ValueError as error:
        raise RuntimeError(f"`{model}`: {error}. Nothing of this model is measured, and the step is not marked done") from error
    saved = ladder_l4_rank.adapter_read_back(target)
    # A SMOKE RUN ALONE: its trainings are one optimizer step on one batch of the same rows, whatever their order, so its models may be ONE adapter, and their mean then is each of them.
    one_of_them = smoke and held["is_the_mean"] and not held["is_none_of_them"]
    if not (held["passes"] or one_of_them) or not ladder_l4_rank.is_the_checks(saved, start.check):
        raise RuntimeError(f"the adapter `{model}` saved at {target}, read back, has {held['elements_not_the_mean']:,} of {held['elements']:,} numbers that are not the mean of "
                           f"its models', differs from each of them in {held['elements_not_those_of_each_adapter']} numbers, and has rank {saved['rank']} and alpha "
                           f"{saved['alpha']} with matrices of rank {saved['ranks']} (the start adapter `{start.name}`: {start.check.rank} and {start.check.alpha}): it is the "
                           "mean of its models, none of them, and at the start adapter's rank, or nothing of it is measured. The step is not marked done")
    summary.update({"averaged": True, "made": made, "read_back": held, "one_of_its_models_in_a_smoke_run": one_of_them, "adapter_saved": saved,
                    "against_the_start_adapter": compared(target / ADAPTER_FILE, start.adapter / ADAPTER_FILE)})
    store.mark_done(marker, summary)        # the adapter is saved, and kept
    _say(f"{one.reader}: `{model}` is saved: {made['elements']:,} numbers in {made['tensors']} tensors, each the mean of {made['adapters']} adapters'; read back, "
         f"{held['elements_not_the_mean']} are not that mean")
    return summary


# ------------------------------------------------------------------------------------------------- measure
def ladder_l5_noise_measure(config: dict, model: str) -> dict:
    """One model, trained or averaged, on the goal set's FIRST sampling alone (`ladder_ceiling.measure_model` with no
    rung, as L4t's measurements at another temperature sample G alone), with the attempts and the sampling seed the
    start model's run recorded for that sampling, at the config's own temperature, from the adapter this run keeps. The
    model server is started with the start adapter's rank as its largest (`config_from`)."""
    noise = the_settings(config)
    store = _store(config, noise)
    prepared = _of_this_run(store, model, f"the measurement of `{model}`")
    averaged = model in prepared["averaged"]
    adapter, sampling = store.root / ladder_assembly.ADAPTERS / model, prepared["the_sampling"]
    return measure_model(config_from(config, prepared, the_start(config, noise), serving=True), store, {"goal_samplings": prepared["goal_samplings"]}, Model(
        name=model, called=f"L5n's model `{model}`", whose=f"L5n's model `{model}`", marker=THE_STAGE.measure_marker(model),
        trained_by=average_marker(model.removeprefix("mean_")) if averaged else THE_STAGE.train_marker(model), adapter=adapter,
        no_adapter=f"the adapter `{model}` is kept by this stage, and one made again would not be the model its other figures stand beside. The measurement cannot be made",
        detail=f"G's first sampling alone, {sampling['episodes']} attempts a problem, sampling seed {sampling['sampling_seed']}",
        summary={"stage": L4, "label": LABEL, "check": "L5n", "model": model, "adapter": str(adapter)}, stage=ENTRY_STAGE, label=L4, also=what_else_a_model_wrote, rungs=False))


# -------------------------------------------------------------------------------------------------- report
def read_the_seeds(config: dict, noise: Noise, store: ArtifactStore, prepared: dict) -> dict[int, dict]:
    """L5's `published` at each seed of L5 the setting names, on the sampling every model here is measured on: each seed's
    per-problem rows WHEN ITS RUN IS ON THE BOX AND FINISHED (its report written and to be read, its `published` trained
    and measured, its stored rows pairing with this run's problems, sampling seed and attempts), NOT THERE otherwise,
    with why. NOTHING IS REFUSED FOR THEM. The seed whose task trained the fixed rows is this run's own copy (the first
    step held and copied it). A seed that is there is copied into this run."""
    deep, sampling, of_l5 = noise.deep, prepared["the_sampling"], ladder_l5.THE_STAGE
    own = prepared["the_published_of_l5"]
    rows_of_own = store.read_rows(own["file"])
    goal_ids, set_name = [row["problem_id"] for row in rows_of_own], the_set(PUBLISHED, sampling["name"])
    found = {}
    for seed in noise.seeds:
        if seed == deep.seed_of_the_search:
            found[seed] = {"there": True, "run": own["run"], "rows": rows_of_own, "inconclusive": own["inconclusive"], "share_of_distinct_attempts": own["share_of_distinct_attempts"]}
            continue
        directory = _runs(config) / ladder_l5.own_run(config, deep, seed)
        names = (f"{ladder_l5.REPORT}.done.json", ladder_l5.REPORT_FILE, f"{of_l5.train_marker(PUBLISHED)}.done.json", f"{of_l5.measure_marker(PUBLISHED)}.done.json",
                 f"episodes_{set_name}_problems.jsonl", f"episodes_{set_name}.done.json")
        lacking = [name for name in names if not (directory / name).exists()]
        if lacking:
            found[seed] = {"there": False, "run": str(directory), "why": f"{directory.name} does not hold {lacking}"}
            continue
        report = json.loads((directory / ladder_l5.REPORT_FILE).read_text())
        try:
            if not report.get("ok", True):
                raise RuntimeError(f"the report of {directory.name} is not to be read")
            rows = ladder_ceiling._stored(directory, set_name, goal_ids, sampling["sampling_seed"], sampling["episodes"], f"`{PUBLISHED}` of seed {seed}", READING)[0]
        except RuntimeError as error:
            found[seed] = {"there": False, "run": str(directory), "why": str(error)}
            continue
        store.write_rows(stored_published_file(seed), rows)
        found[seed] = {"there": True, "run": str(directory), "rows": rows, "inconclusive": bool(report.get("inconclusive")),
                       "share_of_distinct_attempts": what_a_model_wrote([], ladder_ceiling._stored_attempts(directory, set_name))["distinct_attempts"][GOAL]["mean_share_distinct"]}
    return found


def ladder_l5_noise_report(config: dict) -> dict:
    from rlvr_lean.reporting.ladder_l5_noise import build_l5_noise_report

    noise = the_settings(config)
    store = _store(config, noise)
    _need(store, PREPARE, "L5n's report", ENTRY_STAGE)
    prepared = store.done_summary(PREPARE)
    trained_models, averaged_models, sampling = prepared["trained"], prepared["averaged"], prepared["the_sampling"]
    for step in (*(of(model) for model in trained_models for of in (THE_STAGE.train_marker, THE_STAGE.measure_marker)),
                 *(step for model in averaged_models for step in (average_marker(model.removeprefix("mean_")), THE_STAGE.measure_marker(model)))):
        _need(store, step, "L5n's report", ENTRY_STAGE)
    fixed = store.read_rows(ROWS_FILE)
    trains, trained = {}, {}
    for model in trained_models:        # what a model WAS trained on is the training loop's own record of row ids, read back to its stored set's rows (never their text)
        trains[model] = store.done_summary(THE_STAGE.train_marker(model))
        by_id = {row["id"]: row for row in store.read_rows(THE_STAGE.training_file(model))}
        trained[model] = [{key: by_id[row_id].get(key) for key in ("id", "problem_id", "origin", "row_of_the_file")}
                          for row_id in json.loads(store.path(THE_STAGE.loss_file(model)).read_text())["rows_trained"]]
    averaged = {model: store.done_summary(average_marker(model.removeprefix("mean_"))) for model in averaged_models}
    measured = {model: store.done_summary(THE_STAGE.measure_marker(model)) for model in prepared["models"]}
    rows = {model: _results(store, the_set(model, sampling["name"])) for model in prepared["models"]}
    report = build_l5_noise_report(prepared, trains, trained, averaged, measured, rows, fixed, read_the_seeds(config, noise, store, prepared))
    directory = store.root / ladder_assembly.ADAPTERS
    report["adapters"] = {"kept": True, "what": f"the adapters of {', '.join(f'`{model}`' for model in prepared['models'])} ({LABEL}) are KEPT: no step of this stage deletes "
                                                f"one. `{noise.start}` is its own run's and was only read",
                          "directory": str(directory), "there": sorted(path.name for path in directory.glob("*") if path.is_dir()) if directory.is_dir() else []}
    _write_json(store, REPORT_FILE, report)
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    for name in (THE_STAGE.loss_file(model) for model in trained_models):
        store.mirror(name)
    for line in report["lines"]:
        print(line, flush=True)
    store.mark_done(REPORT, {"stage": L4, "label": LABEL, "check": "L5n", "headline": report["headline"], "branches": report["branches"], "measured": list(prepared["models"]),
                             "ok": report["ok"]})
    return report


def _for(step, name: str):
    def run(config: dict) -> dict:
        return step(config, name)
    return run


# In the order the stage runs them: the fixed rows read; each order trained and measured with the recipe as it is, then with the decay, before the next order; then
# each recipe's averaged adapter made and measured; then the report.
STEPS = {PREPARE: ladder_l5_noise_prepare}
for _order in ORDERS:
    for _recipe in RECIPES:
        STEPS[THE_STAGE.train_marker(model_of(_recipe, _order))] = _for(ladder_l5_noise_train, model_of(_recipe, _order))
        STEPS[THE_STAGE.measure_marker(model_of(_recipe, _order))] = _for(ladder_l5_noise_measure, model_of(_recipe, _order))
for _recipe in RECIPES:
    STEPS[average_marker(_recipe)] = _for(ladder_l5_noise_average, _recipe)
    STEPS[THE_STAGE.measure_marker(mean_of(_recipe))] = _for(ladder_l5_noise_measure, mean_of(_recipe))
STEPS[REPORT] = ladder_l5_noise_report
