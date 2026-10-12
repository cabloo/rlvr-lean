"""L5s: the dose. Every own solve in stock that the model finds less than half the time, against the same number of
published proofs. Spec: docs/spec/ladder-loop.spec.md, "L5s: the dose. Every own solve in stock that the model
finds less than half the time, against the same number of published proofs" and its "Made exact by the build".
Everything it makes is labelled pretrained on published proofs. Each step is its own process (`rlvr_lean.runner.entry`,
stage `ladder_l5_stock`). NO ROUND, NO SEARCH, NO SAMPLING BUT THE MEASUREMENTS: one task at the scout's seed.

L5's FINISHED SEARCH IS READ AS L5d READS IT, BY L5d's OWN FIRST-STEP READING (`ladder_l5_draws.drawn_by_the_seed`: L5's
run and every stored run its task read, how a model is measured and trained, the deep solves by the task's seed and
L5d's three sets at that seed), and nothing of it is sampled again. THE TWO SETS ARE BUILT WITH L5's OWN FUNCTIONS FOR
THE TASK'S SEED (`ladder_l5.ordinary_by_the_seed` with every ordinary solve the rule gives; `l5_stock.the_sets`), and
HELD to hold that seed's rows (`l5_stock.holds_the_seeds_rows`). The run directory is its own
(`ladder_l5_stock_<start>_seed<N>`).

THE RECIPE IS L5d's `mean_constant` AND NO OTHER (the settings reader refuses another): each set trained from the start
adapter at its rank, one pass, in the orders of order seeds 1 to 4 with the recipe as it is; the model measured is the
adapter whose every number is the mean of the four, by L5n's own function.

  ladder_l5_stock_prepare           REFUSES, naming what to do, before anything is written: a task of a seed the setting
                                    does not list, or of a further seed while the escalation is not recorded; L5's run
                                    missing, unfinished, not to be read, INCONCLUSIVE or made otherwise; a recipe or a
                                    first sampling that is not that run's; a set with a held-out problem, a problem of G
                                    or of the held-aside set, or a published row with text; sets that do not hold the
                                    rows of L5d's three at this seed; two order seeds that give one order. STORES the two
                                    sets (a published row by id and place, never with text), each in its four orders,
                                    and the ordinary solves as drawn. No GPU, no Lean
  ladder_l5_stock_train_<model>_<k> L4t's OWN training body (`ladder_l4_rows.train_a_model`) under this stage's names:
                                    the set in the order of order seed k, each step's learning rate recorded and held
  ladder_l5_stock_average_<model>   NO TRAINING: the adapter that is the mean of its four
                                    (`ladder_l5_noise.the_mean_adapter`), read back and held as L5n holds its own
  ladder_l5_stock_measure_<model>   the model on the goal set's FIRST sampling alone, as L5d measures: the attempts and the
                                    sampling seed the start model's run recorded
  ladder_l5_stock_report            the checks first, then the reads (`reporting/ladder_l5_stock.py`). THE DOSE is read
                                    against `deep` of L5d's run of this seed, from that run WHEN IT IS ON THE BOX AND
                                    FINISHED: NOT READ, and said, otherwise

`stock` is trained four times, averaged and measured first, then `published_stock`, then the report. Every step is
resumable and a step of its own. NO STEP DELETES AN ADAPTER, and no step writes into a run it reads.

A SMOKE RUN ALONE (`RLVR_LEAN_LADDER_L5_STOCK_SMOKE`, refused unless the run directory is one the task names and the map
is the fixtures'): L5's smoke run reads INCONCLUSIVE and is read all the same, its two looser draws made again as that
run recorded them; one deep solve is enough; the fixtures' map leaves the rule no ordinary solve, so as many as there are
deep solves are taken by the run's minimum, as L5d's smoke run takes its own; orders of its four rows that are alike are
trained all the same; and an averaged adapter that is one of its models is kept. The dose is read against L5d's SMOKE
run, which L5d's own function finds by L5d's run variable (the smoke stage names it).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from rlvr_lean.domain.ladder_round.dose import steps_per_pass
from rlvr_lean.domain.ladder_round.l4_rows import THE_RULE
from rlvr_lean.domain.ladder_round.l4_rows2 import refuse_text
from rlvr_lean.domain.ladder_round.l5 import DEEP, PUBLISHED, SHALLOW, refuse_barred, refuse_overlap
from rlvr_lean.domain.ladder_round.l5_draws import CONSTANT, MEAN_CONSTANT, four_orders, refuse_problems_of, trained_of
from rlvr_lean.domain.ladder_round.l5_noise import ORDERS, numbered
from rlvr_lean.domain.ladder_round.l5_stock import MODELS, PUBLISHED_STOCK, STOCK, fixed_seeds, holds_the_seeds_rows, ordinary_of_the_rule, the_seed, the_sets
from rlvr_lean.gpu import ladder_assembly, ladder_l4, ladder_l4_rows, ladder_l4_start, ladder_l5, ladder_l5_draws, ladder_l5_noise, ladder_loop
from rlvr_lean.gpu.ladder_ceiling import Model, _need, measure_model, stored_file
from rlvr_lean.gpu.ladder_dose import _rows, _runs
from rlvr_lean.gpu.ladder_l3d1 import what_else_a_model_wrote
from rlvr_lean.gpu.ladder_l4 import L4, LABEL, _say
from rlvr_lean.gpu.ladder_l4_rows import described, train_a_model
from rlvr_lean.gpu.ladder_l4_start import Start, config_from
from rlvr_lean.gpu.ladder_l5_draws import the_set
from rlvr_lean.gpu.ladder_round import _results, _stand_in, _write_json, training_seed
from rlvr_lean.infrastructure.artifact_store import ArtifactStore

L5S = "l5_stock"                                            # what this stage's own files are called
STAGE = ENTRY_STAGE = "ladder_l5_stock"                     # what its steps begin with; the stage, as `runner.entry` names it; its run directory begins with it
RUN_VARIABLE = "RLVR_LEAN_LADDER_L5_STOCK_RUN"              # another run directory than `ladder_l5_stock_<start>_seed<seed>` (a smoke run)
SMOKE_VARIABLE = "RLVR_LEAN_LADDER_L5_STOCK_SMOKE"          # a SMOKE run alone (refused elsewhere): L5's INCONCLUSIVE smoke run is read, and what else the module's note says
SETTING = "ladder_loop.l5_stock"
THE_STAGE = ladder_l4_rows.Stage(name=ENTRY_STAGE, steps=STAGE, files=L5S, reader="L5s", start_said_by=f"{ladder_l5.SETTING}.start")
PREPARE, REPORT = THE_STAGE.prepare, f"{STAGE}_report"
REPORT_FILE = "report_ladder_l5_stock.json"
ORDINARY_FILE = f"{L5S}_ordinary_solves.jsonl"              # every ordinary solve `stock` is trained on, in the draw's order, each with the map's k and why it was taken
OF_THIS_STAGE = ladder_l5_draws.OfASeed(who="L5s", run_variable=RUN_VARIABLE, smoke_variable=SMOKE_VARIABLE)      # what L5d's first-step reading says of this stage when it refuses


@dataclass(frozen=True)
class Stock:
    """`ladder_loop.l5_stock`, as the stage's steps read it."""
    start: str                          # L5's start: every model here is trained from it at its rank
    seeds: tuple[int, ...]              # the scout's seed: one task
    more: tuple[int, ...]               # the further seeds, which run once the scout's read names the escalation
    escalated: bool                     # ... recorded here once it has: a task of a further seed is refused until then
    draws: ladder_l5_draws.Draws        # L5d's settings as ITS task reads them: the sampling, its attempts and the recipe are L5d's, and a seed here is one of L5d's
    deep: ladder_l5.Deep                # L5's settings as ITS task reads them: the search's seed, and what an ordinary solve is


def the_settings(config: dict) -> Stock:
    """THE one place that reads `ladder_loop.l5_stock`. Refused (ValueError): seeds that are not distinct whole numbers,
    in the two lists together, or that hold a seed that is none of L5d's (each stands beside L5d's own run of its seed);
    no seed; an escalation that is not true or false; a recipe of L5d's that is not the averaged one (this stage trains
    in four orders and measures the mean, as L5d runs it); L5d's own settings, when their reader refuses them."""
    settings = config["ladder_loop"]["l5_stock"]
    draws = ladder_l5_draws.the_settings(config)
    seeds, more, escalated = tuple(settings["seeds"]), tuple(settings["more_seeds"]), settings["escalated"]
    both, of_l5d = (*seeds, *more), (*draws.seeds, *draws.more)
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in both) or len(set(both)) != len(both) or set(both) - set(of_l5d):
        raise ValueError(f"{SETTING}.seeds is {list(seeds)} and more_seeds {list(more)}: distinct whole numbers of at least 0, at least one in seeds, each a seed of L5d's "
                         f"({list(of_l5d)}: {ladder_l5_draws.SETTING}), whose own run of that seed its two models stand beside")
    if not isinstance(escalated, bool):
        raise ValueError(f"{SETTING}.escalated is {escalated!r}: true once the scout's read names the escalation, false until then")
    if draws.recipe != MEAN_CONSTANT:
        raise ValueError(f"{ladder_l5_draws.SETTING}.recipe is `{draws.recipe}`: this stage trains each set in four orders and measures the averaged adapter "
                         f"(`{MEAN_CONSTANT}`), as L5d runs it, and its models stand beside L5d's of that recipe")
    return Stock(start=draws.start, seeds=seeds, more=more, escalated=escalated, draws=draws, deep=draws.deep)


def run_name(stock: Stock, seed: int) -> str:
    """The stage's run directory, with the start adapter's name in it: `ladder_l5_stock_pre_r64_seed<N>`."""
    return f"{ENTRY_STAGE}_{stock.start}_seed{seed}"


def own_run(config: dict, stock: Stock) -> str:
    """This task's run directory: the seed's own, or the one the task names (a smoke run)."""
    return os.environ.get(RUN_VARIABLE) or run_name(stock, training_seed(config))


def _store(config: dict, stock: Stock) -> ArtifactStore:
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    return ArtifactStore(_runs(config) / own_run(config, stock), Path(mirror) if mirror else None)


def the_start(config: dict, stock: Stock) -> Start:
    """The start model, and where it lives (`ladder_l4_start.the_start`): L5's."""
    return ladder_l4_start.the_start(config, stock.start, THE_STAGE.start_said_by)


def smoke_run() -> bool:
    """Whether the task says it is a smoke run; every other run: the rule, and nothing else."""
    return bool(os.environ.get(SMOKE_VARIABLE))


def average_marker(model: str) -> str:
    """The step that makes a model's averaged adapter, and its marker."""
    return f"{STAGE}_average_{model}"


def the_reader(stock: Stock, smoke: bool) -> ladder_l5.OfTheSearch:
    """Who reads L5's finished run here, for what L5's own reading says when it refuses it. A SMOKE run alone reads a run
    whose report is INCONCLUSIVE (one not to be read never is)."""
    return ladder_l5.OfTheSearch(
        who="L5s", why=f"L5s runs no search: its sets are every own solve in stock, from the finished search of seed {stock.deep.seed_of_the_search} and the start model's map",
        stands="no model here stands beside such a run", models="L5s's models are trained on rows drawn from that run's search and stand beside L5d's models", inconclusive=smoke)


def l5d_run(config: dict, stock: Stock) -> Path:
    """L5d's run of this task's seed, by L5d's OWN naming (`ladder_l5_draws.own_run`: its seed's own directory; in a smoke
    run, the one L5d's run variable names)."""
    return _runs(config) / ladder_l5_draws.own_run(config, stock.draws)


def stock_by_the_seed(config: dict, deep: ladder_l5.Deep, seed: int, smoke: bool, drawn: dict) -> dict:
    """THE TWO SETS BY THIS STAGE's RULE FOR A SEED, with NOTHING WRITTEN, from what L5d's first-step reading drew
    (`drawn`: `ladder_l5_draws.drawn_by_the_seed`): EVERY ordinary solve L5's setting gives, one stored verified map
    attempt of each drawn by the seed; `stock` and `published_stock` (`l5_stock.the_sets`), numbered; and each held to
    carry no text, no row of a held-aside or barred problem and no problem of G. Refused (ValueError) as those
    functions refuse. Returns the ordinary solves the rule names, the ones drawn (as stored, and as rows), the two
    sets, and what was counted of each. TWO STAGES CALL IT (this one's first step; L5c's, `gpu/ladder_l5_cont.py`,
    whose `more` is `published_stock` by this rule): change it and both change."""
    read, the_map, of_this, solves, pretraining, aside = (drawn[key] for key in ("read", "the_map", "of_this", "solves", "pretraining", "aside"))
    searched, goal_ids = [row["problem_id"] for row in read["drawn"]], [row["problem_id"] for row in read["goal_problems"]]
    of_the_rule = ordinary_of_the_rule(the_map["rows"], deep.least, deep.most, {*searched, *aside})
    # A SMOKE RUN ALONE: the fixtures' model solves every map problem in every map attempt, so the rule gives none: as many as there are deep solves are taken, the others by
    # the run's minimum (L5's own looser draw, which marks them), as L5d's smoke run takes its own.
    count = max(len(of_the_rule), len(solves)) if smoke else len(of_the_rule)
    map_rows, ordinary, of_ordinary = ladder_l5.ordinary_by_the_seed(deep, seed, count, read["drawn"], aside, of_this)
    sets = {model: numbered(own) for model, own in the_sets(solves, of_ordinary, pretraining, seed).items()}
    for model in MODELS:
        refuse_text(sets[model], f"the training set of `{model}`")
    refuse_overlap(aside, searched, sets)                   # THE HELD-ASIDE SET WAS L5's TEST: none of its problems was searched, and no set holds a row of one
    heldout = {row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[0]} | {row["problem_id"] for row in read["groups"]}
    counted = {model: refuse_barred(sets[model], heldout, config["ladder_loop"]["l4"]["half_seed"], {row["problem_id"] for row in map_rows}, f"the training set of `{model}`")
               for model in MODELS}
    beside = {model: refuse_problems_of(sets[model], set(goal_ids), set(aside), f"the training set of `{model}`") for model in MODELS}
    return {"of_the_rule": of_the_rule, "ordinary": ordinary, "of_ordinary": of_ordinary, "sets": sets, "counted": counted, "beside": beside}


# ------------------------------------------------------------------------------------------------- prepare
def ladder_l5_stock_prepare(config: dict) -> dict:
    stock = the_settings(config)
    store = _store(config, stock)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    seed, deep, draws, smoke = training_seed(config), stock.deep, stock.draws, smoke_run()
    try:
        the_seed(seed, stock.seeds, stock.more, stock.escalated)
    except ValueError as error:
        raise RuntimeError(f"{error} ({SETTING}). Nothing was written.") from error
    if smoke and not os.environ.get(RUN_VARIABLE):
        raise RuntimeError(f"{SMOKE_VARIABLE} is set: it is for a SMOKE run alone, in a run directory the task names ({RUN_VARIABLE}) and on the fixtures' base map. A run in "
                           "the seed's own directory reads a run of L5 whose checks passed. Nothing was written.")
    # ---- L5's finished run and every stored run its task read, how a model is measured and trained, the deep solves by THIS TASK's seed and L5d's three sets at it: by L5d's
    # OWN first-step reading, with the averaged recipe's hold of the start adapter's type (refused there, with nothing written)
    drawn = ladder_l5_draws.drawn_by_the_seed(config, draws, seed, smoke, the_reader(stock, smoke), True, OF_THIS_STAGE)
    found, read, start, the_map, at_least, directory, theirs, sampling, training, batch, recipe, of_this, solves, table, enough, kept, of_l5d, sha256, pretraining, aside = (
        drawn[key] for key in ("found", "read", "start", "the_map", "at_least", "directory", "theirs", "sampling", "training", "batch", "recipe", "of_this", "solves", "table",
                               "enough", "kept", "sets", "sha256", "pretraining", "aside"))

    # ---- THE TWO SETS, with L5's own functions for THIS TASK's seed: EVERY deep solve (no cap), EVERY ordinary solve L5's setting gives (no draw of which), and the published
    # rows of the seed's hash order; held to hold the rows of L5d's three sets at this seed
    goal_ids = [row["problem_id"] for row in read["goal_problems"]]
    try:
        of_the_rule, ordinary, of_ordinary, sets, counted, beside = stock_by_the_seed(config, deep, seed, smoke, drawn).values()
        held = holds_the_seeds_rows(sets, of_l5d)
        in_orders, coincide = {}, {}
        for model in MODELS:                                    # each set in the content-hash order of each ORDER SEED, as L5n orders rows and L5d trains
            by_order, coincide[model] = four_orders(sets[model], smoke)
            in_orders.update({trained_of(model, order): numbered(by_order[order]) for order in ORDERS})
    except ValueError as error:
        raise RuntimeError(f"{error} (seed {seed}; L5's run {directory.name}). Nothing was written.") from error
    trained, own_rows = list(in_orders), len(solves) + len(of_ordinary)
    steps, by_the_minimum = steps_per_pass(2 * own_rows, batch), sum(row["taken_by"] != THE_RULE for row in ordinary)

    # ---- from here on files are written: the goal problems of each model's set, the goal set's groups, G', the held-aside problems, the start model's first sampling, the sets
    store.write_rows("problems.jsonl", [{**row, "set": the_set(model, sampling["name"])} for model in MODELS for row in read["goal_problems"]])
    store.write_rows(ladder_l4.GROUPS_FILE, read["groups"])
    store.write_rows(ladder_l4.AGAIN_FILE, [{"problem_id": problem_id} for problem_id in read["again"]])
    store.write_rows(ladder_l5.HELD_ASIDE_FILE, read["aside"])
    store.write_rows(stored_file(stock.start, sampling["name"], L4), read["kept"][stock.start, sampling["name"]])
    store.write_rows(ORDINARY_FILE, ordinary)
    for name, own in (*sets.items(), *in_orders.items()):
        store.write_rows(THE_STAGE.training_file(name), own)
    is_trained = (f"the recipe as it is, each set trained {len(ORDERS)} times, in the content-hash order of the order seeds {list(ORDERS)} of its own rows; the model measured is "
                  "the adapter whose every number is the mean of the four's (L5n's own function): L5d's `mean_constant`")
    report = found["report"]
    factors = ladder_l5_noise.learning_rate_factors(training["warmup_steps"], steps)        # the learning-rate factor a training's recorded rates are held to, as L5n records it
    summary = {
        "stage": L4, "label": LABEL, "check": "L5s", "seed": seed, "start": stock.start, "start_adapter": str(start.adapter), "start_run": str(start.directory),
        "models": list(MODELS), "trained": trained, "averaged": list(MODELS), "stand_in_engine": _stand_in(), "effective_batch": batch,
        "the_seeds": {"what": "the scout's seed, one task; the further ones run once its read names the escalation, recorded in the setting. Each is a seed of L5d's, whose own "
                              "run of that seed the two models stand beside", "seeds": list(stock.seeds), "more_seeds": list(stock.more), "escalated": stock.escalated,
                      "fixed": list(fixed_seeds(stock.seeds, stock.more, stock.escalated))},
        "the_search": {"what": f"L5's finished search of seed {deep.seed_of_the_search}, ONLY READ, as L5d reads it: nothing of it is sampled again. BY THIS TASK'S SEED, as L5 "
                               "draws them: which verified attempt of each deep solve and of each ordinary solve; the published rows; each set's order",
                       "run": str(directory), "seed": deep.seed_of_the_search, "problems": len(read["drawn"]), "deep_solves": len(solves), "rows_of_each_kind": len(kept),
                       "cap": deep.cap, "enough_deep_solves": enough, "found": table, "headline": report.get("headline"), "inconclusive": bool(report.get("inconclusive")),
                       "held_aside_problems": len(aside), "held_aside_file": ladder_l5.HELD_ASIDE_FILE},
        "the_map": {"what": "the start model's own map, ONLY READ: an ordinary solve is one of its problems solved in so many of its attempts (L5's own setting), and its row "
                            "one stored verified map attempt", "run": str(the_map["directory"]), "set": the_map["set"], "solved_at_least": deep.least, "solved_at_most": deep.most},
        "the_stock": {"what": f"`{STOCK}` holds EVERY deep solve of the search (no cap) and EVERY ordinary solve the rule gives (no draw of which), one verified attempt of each "
                              "drawn by the seed as L5 draws it, and as many published rows; the counts are the stored runs', recorded here and never numbers of the code",
                      "deep_solves": len(solves), "ordinary_solves": len(of_ordinary), "ordinary_solves_of_the_rule": len(of_the_rule),
                      "ordinary_solves_taken_by_the_runs_minimum": by_the_minimum, "own_rows": own_rows, "rows": {model: len(sets[model]) for model in MODELS},
                      "solved_at_least": deep.least, "solved_at_most": deep.most, "ordinary_file": ORDINARY_FILE},
        "holds_the_seeds_rows": held,
        "the_sets": {model: {"what": {STOCK: f"{len(solves):,} deep rows, {len(of_ordinary):,} ordinary solves and the first {own_rows:,} rows of the hash order of this seed "
                                             f"among the rows `{stock.start}` was pretrained on",
                                      PUBLISHED_STOCK: f"the first {2 * own_rows:,} rows of that order, and nothing else: the bar at equal steps"}[model],
                             "file": THE_STAGE.training_file(model), **described(sets[model], batch), "barred": counted[model],
                             "problems_of_g_or_of_the_held_aside_set": {key: beside[model][key] for key in ("problems_of_the_goal_set", "problems_of_the_held_aside_set")}}
                     for model in MODELS},
        "training_sets": {name: {"what": f"the set of `{name.rpartition('_')[0]}` in the content-hash order of ORDER SEED {name.rpartition('_')[2]}",
                                 "file": THE_STAGE.training_file(name), "model": name.rpartition("_")[0], "order_seed": int(name.rpartition("_")[2]),
                                 **described(in_orders[name], batch)} for name in trained},
        "the_averaged": {model: {"what": "trained by nothing: every number of its adapter is the mean of its four trainings' numbers at that place",
                                 "of": [trained_of(model, order) for order in ORDERS]} for model in MODELS},
        "the_orders": {"what": "each set in the content-hash order of each ORDER SEED (`assembly.training_order`: a hash of the order seed and each row's id), as L5n orders "
                               "rows. An order seed that is the task's own seed gives the seed's own order: it is trained all the same, and named here",
                       "order_seeds": list(ORDERS), **{model: coincide[model] for model in MODELS}},
        "pretraining_file": str(ladder_l4.pretraining_file()), "pretraining_file_sha256": sha256,
        "recipe": {"what": "L4t's (`ladder_ceiling.one_pass`, as the arm's twin is trained): FROM THE START ADAPTER, one pass, the native format. It is held to the recipe "
                           f"L5's run trained its models with. This run: {is_trained}", "name": MEAN_CONSTANT, **recipe, "adapter_seed": seed, "optimizer_steps": steps,
                   "learning_rate_factor": {key: factors[key] for key in ("what", "steps", "warmup_steps", CONSTANT)}},
        "the_sampling": {"what": f"every model is measured on the goal set's FIRST sampling alone, as the run of `{stock.start}` recorded it: its attempts a problem and its "
                                 "sampling seed, so that each pairs by problem with every stored model's first sampling. No rung, no second sampling, no held-aside set",
                         **sampling, "sets": {model: the_set(model, sampling["name"]) for model in MODELS},
                         "stored_start": stored_file(stock.start, sampling["name"], L4)},
        "the_start_wrote": {"what": f"what `{stock.start}`'s own run recorded of what it wrote, for the overfitting reads",
                            "distinct_attempts": (read["of_the_start_run"].get("measured") or {}).get("distinct_attempts")},
        "goal_samplings": [sampling], "goal_set": len(goal_ids), "goal_set_again": len(read["again"]), "attempts_a_goal_problem": sampling["episodes"],
        "smoke": {"what": f"{SMOKE_VARIABLE}: a smoke run alone. L5's smoke run reads INCONCLUSIVE (a training of two rows has no loss to compare), which a real run is refused "
                          "for, and is read all the same; its two looser draws are made again as that run recorded them, and one deep solve is enough; where the rule gives "
                          "fewer ordinary solves than there are deep solves, the others are taken by the run's minimum and marked; orders of a set's handful of rows that are "
                          "alike are trained all the same; and an averaged adapter that is one of its models is kept all the same, and said",
                  "problems_at_least": at_least, "l5_read_though_inconclusive": bool(report.get("inconclusive")), "l5s_failed_checks": list(report.get("failed_checks") or []),
                  "ordinary_solves_taken_by_the_runs_minimum": by_the_minimum}
        if smoke else None,
        "contradicted_side_setting": config["ladder_loop"]["episode"].get("contradicted_side", "all"),
        "start_recipe": theirs["start_recipe"], "start_run_read": theirs["start_run_read"]}
    store.mark_done(PREPARE, summary)
    _say(f"L5s prepared, seed {seed}, from `{stock.start}`: `{STOCK}` is {len(solves):,} deep solves, {len(of_ordinary):,} ordinary solves and the first {own_rows:,} published "
         f"rows of the seed's order ({2 * own_rows:,} rows), `{PUBLISHED_STOCK}` the first {2 * own_rows:,}; from the search of {directory.name}; it holds the rows of L5d's "
         f"`{DEEP}`, `{SHALLOW}` and `{PUBLISHED}` at this seed; {len(trained)} trainings of {steps:,} optimizer steps, then {len(MODELS)} averaged adapters, each measured on "
         f"`{sampling['name']}` alone: {sampling['episodes']} attempts at each of {len(goal_ids)} goal problems, sampling seed {sampling['sampling_seed']}"
         + ("; A SMOKE RUN" if smoke else ""))
    return summary


def _of_this_run(store: ArtifactStore, what: str) -> dict:
    """The prepare step's summary, for a later step of THIS run."""
    _need(store, PREPARE, what, ENTRY_STAGE)
    return store.done_summary(PREPARE)


# --------------------------------------------------------------------------------------------------- train
def ladder_l5_stock_train(config: dict, name: str) -> dict:
    """One training (`<model>_<k>`), by L4t's own training body (`ladder_l4_rows.train_a_model`) under this stage's
    names: from the start adapter at its rank, one pass over the model's set in the order of order seed k, the recipe as
    it is, each step's learning rate recorded and held to it, as L5d trains under `mean_constant`."""
    stock = the_settings(config)
    store, marker = _store(config, stock), THE_STAGE.train_marker(name)
    if store.is_done(marker):
        return store.done_summary(marker)
    prepared = _of_this_run(store, f"the training of `{name}`")
    schedule = ladder_l4_rows.Schedule(decay=False, order=f"a content hash of the ORDER SEED {prepared['training_sets'][name]['order_seed']} and each row's id "
                                                          "(`assembly.training_order`): the set's own rows, in one of the four orders its averaged adapter is the mean of")
    return train_a_model(config, store, name, stock.start, run_name(stock, training_seed(config)), THE_STAGE, schedule=schedule)


# ------------------------------------------------------------------------------------------------- average
def ladder_l5_stock_average(config: dict, model: str) -> dict:
    """NO TRAINING: the adapter `model`, whose every number is the mean of its four trainings' numbers at its place, BY
    L5n's OWN FUNCTION (`ladder_l5_noise.the_mean_adapter`): saved under `adapters/<model>`, read back, and held to be
    that mean, none of them and at the start adapter's rank, as L5n holds its own. Refused there, with the step not
    marked done."""
    stock = the_settings(config)
    store, marker = _store(config, stock), average_marker(model)
    if store.is_done(marker):
        return store.done_summary(marker)
    prepared = _of_this_run(store, f"the averaged adapter `{model}`")
    of = list(prepared["the_averaged"][model]["of"])
    summary = {"stage": L4, "label": LABEL, "check": "L5s", "model": model, "recipe": MEAN_CONSTANT, "of": of, "adapter": str(store.root / ladder_assembly.ADAPTERS / model),
               "stand_in_engine": _stand_in()}
    return ladder_l5_noise.the_mean_adapter(store, the_start(config, stock), model, of, marker, summary, bool(prepared.get("smoke")), THE_STAGE)


# ------------------------------------------------------------------------------------------------- measure
def ladder_l5_stock_measure(config: dict, model: str) -> dict:
    """One of the two averaged models on the goal set's FIRST sampling alone, as L5d measures (`ladder_ceiling.measure_model`
    with no rung): the attempts and the sampling seed the start model's run recorded for that sampling, at the config's
    own temperature, from the adapter this run keeps. The model server is started with the start adapter's rank as its
    largest (`config_from`)."""
    stock = the_settings(config)
    store = _store(config, stock)
    prepared = _of_this_run(store, f"the measurement of `{model}`")
    adapter, sampling = store.root / ladder_assembly.ADAPTERS / model, prepared["the_sampling"]
    return measure_model(config_from(config, prepared, the_start(config, stock), serving=True), store, {"goal_samplings": prepared["goal_samplings"]}, Model(
        name=model, called=f"L5s's model `{model}`", whose=f"L5s's model `{model}`", marker=THE_STAGE.measure_marker(model), trained_by=average_marker(model), adapter=adapter,
        no_adapter=f"the adapter `{model}` is kept by this stage, and one made again would not be the model its other figures stand beside. The measurement cannot be made",
        detail=f"G's first sampling alone, {sampling['episodes']} attempts a problem, sampling seed {sampling['sampling_seed']}",
        summary={"stage": L4, "label": LABEL, "check": "L5s", "model": model, "adapter": str(adapter)}, stage=ENTRY_STAGE, label=L4, also=what_else_a_model_wrote, rungs=False))


# -------------------------------------------------------------------------------------------------- report
def read_l5d(config: dict, stock: Stock) -> dict:
    """L5d's run of this task's seed, ONLY READ: what its report says of its own seed (`of_this_seed`) WHEN ITS RUN IS ON
    THE BOX AND ITS REPORT WRITTEN, NOT THERE otherwise, with why. NOTHING IS REFUSED FOR IT: the dose is then NOT READ,
    and the task queued again builds the report again once that run is there."""
    directory = l5d_run(config, stock)
    lacking = [name for name in (f"{ladder_l5_draws.REPORT}.done.json", ladder_l5_draws.REPORT_FILE) if not (directory / name).exists()]
    if lacking:
        return {"there": False, "run": str(directory), "why": f"{directory.name} does not hold {lacking}: L5d's run of this seed is not on the box, or not finished"}
    return {**(json.loads((directory / ladder_l5_draws.REPORT_FILE).read_text()).get("of_this_seed") or {}), "there": True, "run": str(directory)}


def the_rule_again(store: ArtifactStore, stock: Stock, prepared: dict) -> tuple[list[str], list[str]]:
    """THE RULE MADE AGAIN, for the report's first check, from the runs the first step read and not from the stored sets:
    the problems L5's search solved (its stored per-problem rows), and the ordinary solves L5's setting gives on the
    start model's map, none of them a searched or a held-aside problem. A SMOKE run alone adds the ordinary solves its
    first step recorded as taken by the run's minimum."""
    searched, of_the_map = Path(prepared["the_search"]["run"]), prepared["the_map"]
    deep_solves = [row["problem_id"] for row in _rows(searched / f"episodes_{ladder_l5.SEARCH_SET}_problems.jsonl") if row["resolved"] > 0]
    left_out = {row["problem_id"] for name in (ladder_l5.SEARCH_PROBLEMS_FILE, ladder_l5.HELD_ASIDE_FILE) for row in _rows(searched / name)}
    ordinary = ordinary_of_the_rule(_rows(Path(of_the_map["run"]) / f"episodes_{of_the_map['set']}_problems.jsonl"), stock.deep.least, stock.deep.most, left_out)
    if prepared.get("smoke"):
        ordinary += [row["problem_id"] for row in store.read_rows(ORDINARY_FILE) if row["taken_by"] != THE_RULE]
    return deep_solves, ordinary


def ladder_l5_stock_report(config: dict) -> dict:
    from rlvr_lean.gpu.ladder_l3c import proof_lengths
    from rlvr_lean.reporting.ladder_l5_stock import build_l5_stock_report

    stock = the_settings(config)
    store = _store(config, stock)
    prepared = _of_this_run(store, "L5s's report")
    names, sampling = prepared["trained"], prepared["the_sampling"]
    for step in (step for model in MODELS for step in (*(THE_STAGE.train_marker(name) for name in prepared["the_averaged"][model]["of"]), average_marker(model),
                                                       THE_STAGE.measure_marker(model))):
        _need(store, step, "L5s's report", ENTRY_STAGE)
    sets = {name: store.read_rows(THE_STAGE.training_file(name)) for name in (*MODELS, *names)}
    trains, trained = {}, {}
    for name in names:                  # what a model WAS trained on is the training loop's own record of row ids, read back to its stored set's rows (never their text)
        trains[name] = store.done_summary(THE_STAGE.train_marker(name))
        by_id = {row["id"]: row for row in sets[name]}
        trained[name] = [{key: by_id[row_id].get(key) for key in ("id", "problem_id", "origin", "row_of_the_file")}
                         for row_id in json.loads(store.path(THE_STAGE.loss_file(name)).read_text())["rows_trained"]]
    averaged = {model: store.done_summary(average_marker(model)) for model in MODELS}
    measured = {model: store.done_summary(THE_STAGE.measure_marker(model)) for model in MODELS}
    rows = {model: _results(store, the_set(model, sampling["name"])) for model in MODELS}
    start = the_start(config, stock)
    # THE SETS ARE MADE AGAIN AND HELD by the report: against the rows the start model was pretrained on, read as the first step read them (the pretraining file, held to its hash).
    pretraining = ladder_l4_start.pretrained_on(start.directory, prepared["pretraining_file_sha256"], start.name)[0]
    base_map = [row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[1]]
    ids_of = lambda name: [row["problem_id"] for row in store.read_rows(name)]      # noqa: E731
    # The length of a held-out problem's published proof is read HERE, by the report, and by nothing before it.
    report = build_l5_stock_report(prepared, trains, trained, averaged, measured, rows, sets, pretraining, store.read_rows(sampling["stored_start"]),
                                   store.read_rows(ladder_l4.GROUPS_FILE), proof_lengths(), ids_of(ladder_l4.AGAIN_FILE), ids_of(ladder_l5.HELD_ASIDE_FILE),
                                   config["ladder_loop"]["l4"], base_map, {"sampling": stock.draws.sampling, "attempts": stock.draws.attempts},
                                   *the_rule_again(store, stock, prepared), read_l5d(config, stock))
    directory = store.root / ladder_assembly.ADAPTERS
    report["adapters"] = {"kept": True, "what": f"the adapters of {', '.join(f'`{name}`' for name in (*names, *MODELS))} ({LABEL}) are KEPT: no step of this stage deletes "
                                                f"one. `{stock.start}` is its own run's and was only read",
                          "directory": str(directory), "there": sorted(path.name for path in directory.glob("*") if path.is_dir()) if directory.is_dir() else []}
    _write_json(store, REPORT_FILE, report)
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    for name in (THE_STAGE.loss_file(name) for name in names):
        store.mirror(name)
    for line in report["lines"]:
        print(line, flush=True)
    store.mark_done(REPORT, {"stage": L4, "label": LABEL, "check": "L5s", "seed": prepared["seed"], "recipe": MEAN_CONSTANT, "headline": report["headline"],
                             "branches": report["branches"], "measured": list(MODELS), "ok": report["ok"]})
    return report


def _for(step, name: str):
    def run(config: dict) -> dict:
        return step(config, name)
    return run


# In the order the stage runs them: the two sets built; then `stock` trained in its four orders, averaged and measured, then `published_stock` the same; then the report.
STEPS = {PREPARE: ladder_l5_stock_prepare}
for _model in MODELS:
    for _order in ORDERS:
        STEPS[THE_STAGE.train_marker(trained_of(_model, _order))] = _for(ladder_l5_stock_train, trained_of(_model, _order))
    STEPS[average_marker(_model)] = _for(ladder_l5_stock_average, _model)
    STEPS[THE_STAGE.measure_marker(_model)] = _for(ladder_l5_stock_measure, _model)
STEPS[REPORT] = ladder_l5_stock_report
