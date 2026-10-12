"""L5f: the three kinds again, from the twice-pretrained model. Spec: docs/spec/ladder-loop.spec.md, "L5f: the three
kinds again, from the twice-pretrained model. Do the model's own proofs still add after a whole second pass over
published ones?" and its "Made exact by the build". Everything it makes is labelled pretrained on published proofs. Each
step is its own process (`rlvr_lean.runner.entry`, stage `ladder_l5_full`). NO ROUND, NO SEARCH, NO SAMPLING BUT THE
MEASUREMENTS: one task at the scout's seed.

IT IS L5d's STAGE WITH ONE CHANGE, THE START ADAPTER: `second_full`, the adapter L4p's run keeps (a whole second pass
over the published rows from `pre_r64`), ONLY READ, found by the function that already finds it
(`ladder_l6.l4p_directory`) and read by L6's own reading of it (`ladder_l6.read_the_stored_model`). Every model is at
`pre_r64`'s rank and alpha, as L4p's run recorded them. The run directory is its own, with the start's name in it
(`ladder_l5_draws_second_full_seed<N>`). L5d from `pre_r64` writes what it wrote.

THE THREE SETS ARE L5d's FOR THE SEED, by L5d's OWN FIRST-STEP READING (`ladder_l5_draws.drawn_by_the_seed`: L5's
finished run and every stored run its task read, how a model is measured and trained, the three sets by L5's rule for the
task's seed), and are HELD to be, id for id and order for order (each of the four orders too), the sets L5d's finished
run of that seed stored and trained (`l5_full.hold_to_theirs`).

  ladder_l5_full_prepare            REFUSES, naming what to run, before anything is written: a task of a seed the
                                    setting does not list, or of a further seed while the escalation is not recorded;
                                    L5's run missing, unfinished, not to be read, INCONCLUSIVE or made otherwise; L4p's
                                    run without `second_full`, its markers, its adapter or its stored rows on G, or with
                                    rows that would not pair; an adapter of `second_full` that is not at `pre_r64`'s rank
                                    and alpha, or that IS `pre_r64`'s; L5d's run of the seed missing, unfinished,
                                    INCONCLUSIVE, not to be read, of another recipe or measured otherwise; sets that are
                                    not that run's. L4p's OWN LOSS CHECK OF `second_full` IS RECORDED AND DECIDES
                                    NOTHING. STORES the three sets and each in its four orders (a published row by id
                                    and place, never with text) and both start models' first samplings. No GPU, no Lean
  ladder_l5_full_train_<model>_<k>  L4t's OWN training body (`ladder_l4_rows.train_a_model`, handed the start model as
                                    this stage found it): from `second_full`'s adapter, one pass, the set in the order of
                                    order seed k, each step's learning rate recorded and held
  ladder_l5_full_average_<model>    NO TRAINING: the adapter that is the mean of its four
                                    (`ladder_l5_noise.the_mean_adapter`), read back and held as L5n holds its own
  ladder_l5_full_measure_<model>    the model on the goal set's FIRST sampling alone, as L5d measures, served at the
                                    start adapter's rank
  ladder_l5_full_report             the checks first, then the reads (`reporting/ladder_l5_full.py`)

The steps stand in L5d's order: model by model its four trainings, its averaged adapter and its measurement, then the
report. Every step is resumable and a step of its own. NO STEP DELETES AN ADAPTER, and no step writes into a run it
reads.

A SMOKE RUN ALONE (`RLVR_LEAN_LADDER_L5_FULL_SMOKE`, refused unless the run directory is one the task names and the map
is the fixtures'): L5's smoke run reads INCONCLUSIVE and is read all the same, its two looser draws made again as that
run recorded them; one deep solve is enough; orders of its two rows that are alike are trained all the same; and an
averaged adapter that is one of its models is kept. L4p's smoke run is named by L6's variable for it and L5d's by L5d's
own run variable (the smoke stage names both).
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path

from rlvr_lean.domain.ladder_round.dose import steps_per_pass
from rlvr_lean.domain.ladder_round.l4 import the_adapter_differs
from rlvr_lean.domain.ladder_round.l5 import DEEP, MODELS, PUBLISHED, SHALLOW
from rlvr_lean.domain.ladder_round.l5_draws import CONSTANT, MEAN_CONSTANT, four_orders, refuse_problems_of, trained_of
from rlvr_lean.domain.ladder_round.l5_full import START, fixed_seeds, hold_to_theirs, the_seed
from rlvr_lean.domain.ladder_round.l5_noise import ORDERS, numbered
from rlvr_lean.domain.ladder_round.l5_stock import of_l5d
from rlvr_lean.gpu import (
    ladder_assembly,
    ladder_ceiling,
    ladder_l4,
    ladder_l4_dose,
    ladder_l4_rank,
    ladder_l4_rows,
    ladder_l4_start,
    ladder_l5,
    ladder_l5_draws,
    ladder_l5_noise,
    ladder_l5_stock,
    ladder_l6,
    ladder_loop,
)
from rlvr_lean.gpu.ladder_ceiling import Model, _need, measure_model, stored_file
from rlvr_lean.gpu.ladder_dose import _rows, _runs
from rlvr_lean.gpu.ladder_l3d1 import what_else_a_model_wrote
from rlvr_lean.gpu.ladder_l4 import L4, LABEL, _say
from rlvr_lean.gpu.ladder_l4_rows import described, train_a_model
from rlvr_lean.gpu.ladder_l4_start import Start, config_from
from rlvr_lean.gpu.ladder_l5_draws import the_set
from rlvr_lean.gpu.ladder_round import _results, _stand_in, _write_json, training_seed
from rlvr_lean.infrastructure.adapter_files import ADAPTER_FILE, MEAN_TYPES, compared, tensors_of
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.reporting.ladder_l4 import adapter_against_the_start

L5F = "l5_full"                                             # what this stage's own files are called
STAGE = ENTRY_STAGE = "ladder_l5_full"                      # what its steps begin with; the stage, as `runner.entry` names it
RUN_VARIABLE = "RLVR_LEAN_LADDER_L5_FULL_RUN"               # another run directory than `ladder_l5_draws_<start>_seed<seed>` (a smoke run)
SMOKE_VARIABLE = "RLVR_LEAN_LADDER_L5_FULL_SMOKE"           # a SMOKE run alone (refused elsewhere): L5's INCONCLUSIVE smoke run is read, and what else the module's note says
SETTING = "ladder_loop.l5_full"
THE_STAGE = ladder_l4_rows.Stage(name=ENTRY_STAGE, steps=STAGE, files=L5F, reader="L5f", start_said_by=f"{SETTING}.start")
PREPARE, REPORT = THE_STAGE.prepare, f"{STAGE}_report"
REPORT_FILE = "report_ladder_l5_full.json"
OF_L5D, OF_L4P = ladder_l5_draws.THE_STAGE, ladder_l4_dose.THE_STAGE       # the two stages whose stored runs this one stands on, as they name their own files and markers
OF_THIS_STAGE = ladder_l5_draws.OfASeed(who="L5f", run_variable=RUN_VARIABLE, smoke_variable=SMOKE_VARIABLE)      # what L5d's first-step reading says of this stage when it refuses
TRAINED = tuple(trained_of(model, order) for model in MODELS for order in ORDERS)       # the twelve trainings, in the order they are made


@dataclass(frozen=True)
class Full:
    """`ladder_loop.l5_full`, as the stage's steps read it."""
    start: str                          # `second_full`: every model here is trained from its adapter, at `pre_r64`'s rank
    seeds: tuple[int, ...]              # the scout's seed: one task
    more: tuple[int, ...]               # the further seeds, which run once the scout's read names the escalation
    escalated: bool                     # ... recorded here once it has: a task of a further seed is refused until then
    draws: ladder_l5_draws.Draws        # L5d's settings as ITS task reads them: the first start (`pre_r64`), the sampling, its attempts and the recipe are L5d's, and a seed here is one of L5d's
    deep: ladder_l5.Deep                # L5's settings as ITS task reads them: the search's seed


def the_settings(config: dict) -> Full:
    """THE one place that reads `ladder_loop.l5_full`. Refused (ValueError): a start that is not `second_full` (the one
    stored model this stage is specified from); seeds that are not distinct whole numbers, in the two lists together, or
    that hold a seed that is none of L5d's (the sets are held to L5d's own run of the seed); no seed; an escalation that
    is not true or false; a recipe of L5d's that is not the averaged one; L5d's own settings, when their reader refuses
    them."""
    settings = config["ladder_loop"]["l5_full"]
    draws = ladder_l5_draws.the_settings(config)
    start, seeds, more, escalated = settings["start"], tuple(settings["seeds"]), tuple(settings["more_seeds"]), settings["escalated"]
    if start != START:
        raise ValueError(f"{SETTING}.start is {start!r}: this stage is specified from `{START}` alone (L4p's whole second pass over the published rows from `{draws.start}`, "
                         "the adapter that run keeps)")
    both, of_theirs = (*seeds, *more), (*draws.seeds, *draws.more)
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in both) or len(set(both)) != len(both) or set(both) - set(of_theirs):
        raise ValueError(f"{SETTING}.seeds is {list(seeds)} and more_seeds {list(more)}: distinct whole numbers of at least 0, at least one in seeds, each a seed of L5d's "
                         f"({list(of_theirs)}: {ladder_l5_draws.SETTING}), whose own run of that seed stored the sets this stage trains")
    if not isinstance(escalated, bool):
        raise ValueError(f"{SETTING}.escalated is {escalated!r}: true once the scout's read names the escalation, false until then")
    if draws.recipe != MEAN_CONSTANT:
        raise ValueError(f"{ladder_l5_draws.SETTING}.recipe is `{draws.recipe}`: this stage trains each set in four orders and measures the averaged adapter "
                         f"(`{MEAN_CONSTANT}`), as L5d runs it, and its sets are held to L5d's run of that recipe")
    return Full(start=start, seeds=seeds, more=more, escalated=escalated, draws=draws, deep=draws.deep)


def run_name(full: Full, seed: int) -> str:
    """The stage's run directory: L5d's, with THIS start adapter's name in it: `ladder_l5_draws_second_full_seed<N>`."""
    return f"{ladder_l5_draws.ENTRY_STAGE}_{full.start}_seed{seed}"


def own_run(config: dict, full: Full) -> str:
    """This task's run directory: the seed's own, or the one the task names (a smoke run)."""
    return os.environ.get(RUN_VARIABLE) or run_name(full, training_seed(config))


def _store(config: dict, full: Full) -> ArtifactStore:
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    return ArtifactStore(_runs(config) / own_run(config, full), Path(mirror) if mirror else None)


def the_first_start(config: dict, full: Full) -> Start:
    """`pre_r64`, L5's start and L5d's: the model `second_full` was trained from. Its check gives the rank and alpha
    every model here is at, and its run holds the rows both were pretrained on (`ladder_l4_start.the_start`)."""
    return ladder_l4_start.the_start(config, full.draws.start, ladder_l5_draws.THE_STAGE.start_said_by)


def l4p_directory(config: dict) -> Path:
    """L4p's run, where `second_full` lives: by L6's own function for it (`ladder_l6.l4p_directory`, on L6's own
    settings: L4p's own name for its run of the seed of L5's search, or the run L6's variable names: a smoke run)."""
    return ladder_l6.l4p_directory(config, ladder_l6.the_settings(config))


def the_start(config: dict, full: Full) -> Start:
    """THE START MODEL OF THIS STAGE, as L4t's training body, L5n's averaging body and the measurements take one:
    `second_full`, its adapter in L4p's run (no path is stated here: `l4p_directory`), held at the rank and alpha of the
    check that made `pre_r64` (`the_first_start`), which L4p's run recorded for it. Only read."""
    directory = l4p_directory(config)
    return dataclasses.replace(the_first_start(config, full), name=full.start, directory=directory, adapter=directory / ladder_assembly.ADAPTERS / full.start,
                               task=l4p_task(full), prepare=OF_L4P.prepare, measure=OF_L4P.measure_marker(full.start))


def l4p_task(full: Full) -> str:
    """The task that writes L4p's run, for a refusal."""
    stage, seed = ladder_l4_dose.ENTRY_STAGE, full.deep.seed_of_the_search
    return f"the task of stage `{stage}` for seed {seed} (`python -m rlvr_lean.runner.entry --stage {stage} --seeds {seed}`; for the smoke run, stage `{stage}_smoke`)"


def l5d_task(seed: int) -> str:
    """The task that writes L5d's run of a seed, for a refusal."""
    stage = ladder_l5_draws.ENTRY_STAGE
    return f"the task of stage `{stage}` for seed {seed} (`python -m rlvr_lean.runner.entry --stage {stage} --seeds {seed}`; for the smoke run, stage `{stage}_smoke`)"


def smoke_run() -> bool:
    """Whether the task says it is a smoke run; every other run: the rule, and nothing else."""
    return bool(os.environ.get(SMOKE_VARIABLE))


def average_marker(model: str) -> str:
    """The step that makes a model's averaged adapter, and its marker."""
    return f"{STAGE}_average_{model}"


def the_reader(full: Full, smoke: bool) -> ladder_l5.OfTheSearch:
    """Who reads L5's finished run here, for what L5's own reading says when it refuses it. A SMOKE run alone reads a run
    whose report is INCONCLUSIVE (one not to be read never is)."""
    return ladder_l5.OfTheSearch(
        who="L5f", why=f"L5f runs no search: its three sets are L5d's for the seed, drawn from the finished search of seed {full.deep.seed_of_the_search}",
        stands="no model here stands beside such a run", models="L5f's models are trained on rows drawn from that run's search and stand beside L5d's models", inconclusive=smoke)


# ----------------------------------------------------------------------------------- the two runs it stands on
def read_second_full(config: dict, full: Full, read: dict, sampling: dict) -> dict:
    """`second_full`, ONLY READ, with nothing written: L6's own reading of the stored model (`ladder_l6.read_the_stored_model`:
    L4p's run, its markers of the model, its rows on G held to the problems, sampling seeds and attempts the start
    model was measured with), which says NOT THERE and is REFUSED here (RuntimeError), naming L4p's task: every model of
    this stage is trained from that adapter. Then HELD, each a refusal: L4p's run recorded `pre_r64`'s rank and alpha
    for it; the adapter it saved was read back at them, and was NOT `pre_r64`'s (the training's own bit-for-bit
    comparison); and, on a GPU box, the adapter is THERE now, reads back at that rank, holds numbers that are not
    `pre_r64`'s and only tensors of a type the mean is written in. L4p's OWN CHECKS OF THE MODEL (its report reads it
    INCONCLUSIVE for a loss that rose over a second pass) ARE RECORDED AND DECIDE NOTHING, as in L6. Returns what the
    prepare step stores of it and its rows on the first sampling."""
    start, first, task = the_start(config, full), the_first_start(config, full), l4p_task(full)
    found = ladder_l6.read_the_stored_model(config, ladder_l6.the_settings(config), read)
    if not found["there"]:
        raise RuntimeError(f"`{full.start}` is NOT THERE ({found['why']}): it is the model every model of this stage is trained FROM, kept by L4p's run "
                           f"({start.directory}). Run {task} to its end first; nothing was written.")
    directory, adapter, check = start.directory, start.adapter, first.check
    prepared, trained, measured = (json.loads((directory / f"{marker}.done.json").read_text())
                                   for marker in (OF_L4P.prepare, OF_L4P.train_marker(full.start), OF_L4P.measure_marker(full.start)))
    recorded, saved = prepared.get("start_recipe") or {}, trained.get("adapter_saved")
    differs = the_adapter_differs(adapter_against_the_start(trained))
    if (recorded.get("rank"), recorded.get("alpha")) != (check.rank, check.alpha) or (saved is not None and not ladder_l4_rank.is_the_checks(saved, check)):
        raise RuntimeError(f"L4p's run {directory.name} recorded rank {recorded.get('rank')} and alpha {recorded.get('alpha')} for its models"
                           + (f" and saved `{full.start}` at rank {saved.get('rank')} and alpha {saved.get('alpha')} with matrices of rank {saved.get('ranks')}" if saved else "")
                           + f"; `{first.name}` is rank {check.rank} and alpha {check.alpha}, and every model here is trained and served at that rank. Nothing was written.")
    if differs["compared"] and not differs["passes"]:
        raise RuntimeError(f"L4p's run {directory.name} recorded that no number of the adapter `{full.start}` differs from `{first.name}`'s: it is not a second pass, and "
                           f"this stage would be L5d's again. Queue {task} again; nothing was written.")
    now = None
    if not ladder_ceiling._stand_in():     # on a GPU box (the trainings' own test of it): the files themselves, as every later step reads them
        if not (adapter / ADAPTER_FILE).exists() or not (adapter / ladder_l4_rank.ADAPTER_CONFIG).exists():
            raise RuntimeError(f"{adapter} is not there: the adapter `{full.start}` is what every model here is trained from, and L4p's run keeps it. Queue {task} again (its "
                               "training step trains it again); nothing was written.")
        read_back, against = ladder_l4_rank.adapter_read_back(adapter), compared(adapter / ADAPTER_FILE, first.adapter / ADAPTER_FILE)
        other = sorted({entry["dtype"] for entry in tensors_of(adapter / ADAPTER_FILE)[0].values()} - set(MEAN_TYPES))
        if not ladder_l4_rank.is_the_checks(read_back, check) or not against["elements_changed"] or other:
            raise RuntimeError(f"the adapter `{full.start}` at {adapter}, read back, has rank {read_back['rank']} and alpha {read_back['alpha']} with matrices of rank "
                               f"{read_back['ranks']} (`{first.name}`: {check.rank} and {check.alpha}); {against['elements_changed']:,} of its {against['elements']:,} numbers "
                               f"are not `{first.name}`'s; tensors of a type the averaged adapters are not written in: {other}. It is at `{first.name}`'s rank, is not "
                               f"`{first.name}`'s, and is of {list(MEAN_TYPES)}, or nothing is trained from it. Nothing was written.")
        now = {"there": True, "read_back": read_back, "against_the_first_start": against}
    report_file = directory / ladder_l4_dose.REPORT_FILE
    of_the_model = ((json.loads(report_file.read_text()).get("models") or {}).get(full.start) or {}) if report_file.exists() else {}
    its_checks = of_the_model.get("can_this_model_be_read") or {}
    failed = [name for name, entry in its_checks.items() if isinstance(entry, dict) and entry.get("passes") is False]
    rows = found["on_g"][[of["name"] for of in read["plan"]["goal_samplings"]].index(sampling["name"])]        # the ONE sampling every model here is measured on
    return {"rows": rows,
            "summary": {
                "model": full.start, "run": str(directory), "adapter": str(adapter), "trained_from": first.name, "task": task,
                "what": f"`{full.start}`, the adapter L4p's run keeps (a whole second pass over the published rows from `{first.name}`), ONLY READ: every model here is trained "
                        "from it. Held before anything was written: it is there (on a GPU box), at the first start's rank and alpha, and not the first start's adapter",
                "recorded_by_l4p": {"rank": recorded.get("rank"), "alpha": recorded.get("alpha"), "adapter_saved": saved, "against_the_first_start": differs},
                "read_now": now, "rank": check.rank, "alpha": check.alpha,
                # It passes on the files themselves (a GPU box), or on what L4p's run recorded when it saved it: read back at the rank, and not the first start's.
                "passes": now is not None or (differs["passes"] and saved is not None),
                "first_sampling": {"sampling": sampling["name"], "episodes": sampling["episodes"], "sampling_seed": sampling["sampling_seed"], "problems": len(rows)},
                "distinct_attempts": measured.get("distinct_attempts"),
                "l4ps_own_checks": {"what": f"L4p's own checks of `{full.start}`, as ITS report read them: PRINTED, AND DECIDING NOTHING HERE (a second pass over rows the model "
                                            "has trained on has no falling loss to show; L6 treats it so). Nothing is refused for them",
                                    "report_there": report_file.exists(), "failed": failed, "inconclusive_by_its_own_report": bool(failed),
                                    "mean_loss_over_the_first_rows": trained.get("mean_loss_over_the_first_rows"),
                                    "mean_loss_over_the_last_rows": trained.get("mean_loss_over_the_last_rows"), "rows_compared": trained.get("rows_compared")}}}


def l5ds_sets(directory: Path) -> tuple[dict, dict]:
    """(by model and by trained model, the ids of the set L5d's run in `directory` STORED, in its order; by trained
    model, the ids its TRAINING LOOP recorded), each None where that run holds no such file. Only read; nothing is
    refused here."""
    def ids(name: str) -> list[str] | None:
        path = directory / OF_L5D.training_file(name)
        return [row["id"] for row in _rows(path)] if path.exists() else None

    def trained(name: str) -> list[str] | None:
        path = directory / OF_L5D.loss_file(name)
        return json.loads(path.read_text()).get("rows_trained") if path.exists() else None

    return {name: ids(name) for name in (*MODELS, *TRAINED)}, {name: trained(name) for name in TRAINED}


def read_l5ds_run(config: dict, full: Full, seed: int, measurement: dict) -> dict:
    """L5d's FINISHED run of this task's seed, ONLY READ, by L5d's OWN naming (`ladder_l5_stock.read_l5d`, which finds it
    by `ladder_l5_draws.own_run`) and L5s's own reading of what its report says of its seed (`l5_stock.of_l5d`).
    REFUSED (RuntimeError), naming the task to run: a run that is not there or not finished, of another seed or another
    recipe than the averaged one, INCONCLUSIVE, not to be read, or measured otherwise than this run; one that does not
    hold its three sets, each in its four orders, and its twelve trainings' own records. Returns its directory, what
    its report hands the reads, and the ids of its sets as stored and as trained."""
    of_theirs = of_l5d(ladder_l5_stock.read_l5d(config, full), seed, MEAN_CONSTANT, measurement)
    if not of_theirs["read"]:
        raise RuntimeError(f"L5d's run of seed {seed} cannot carry L5f: {of_theirs['why']}. The three sets of this stage are HELD to be the ones that run trained, each in "
                           f"its four orders. Run {l5d_task(seed)} to its end first; nothing was written.")
    directory = Path(of_theirs["run"])
    stored, recorded = l5ds_sets(directory)
    lacking = [OF_L5D.training_file(name) for name, own in stored.items() if own is None] + [OF_L5D.loss_file(name) for name, own in recorded.items() if own is None]
    if lacking:
        raise RuntimeError(f"{directory} does not hold {lacking}: the sets L5d's run of seed {seed} stored, or what its trainings recorded. Run {l5d_task(seed)} to its end "
                           "first; nothing was written.")
    return {"directory": directory, "of_theirs": of_theirs, "stored": stored, "recorded": recorded}


# ------------------------------------------------------------------------------------------------- prepare
def ladder_l5_full_prepare(config: dict) -> dict:
    full = the_settings(config)
    store = _store(config, full)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    seed, deep, draws, smoke = training_seed(config), full.deep, full.draws, smoke_run()
    try:
        the_seed(seed, full.seeds, full.more, full.escalated)
    except ValueError as error:
        raise RuntimeError(f"{error} ({SETTING}). Nothing was written.") from error
    if smoke and not os.environ.get(RUN_VARIABLE):
        raise RuntimeError(f"{SMOKE_VARIABLE} is set: it is for a SMOKE run alone, in a run directory the task names ({RUN_VARIABLE}) and on the fixtures' base map. A run in "
                           "the seed's own directory reads a run of L5 whose checks passed. Nothing was written.")
    # ---- L5's finished run and every stored run its task read, how a model is measured and trained, and L5d's THREE SETS at this task's seed: by L5d's OWN first-step
    # reading, with the averaged recipe's hold of the first start adapter's type (refused there, with nothing written)
    drawn = ladder_l5_draws.drawn_by_the_seed(config, draws, seed, smoke, the_reader(full, smoke), True, OF_THIS_STAGE)
    found, read, first, at_least, directory, theirs, sampling, training, batch, recipe, solves, table, enough, kept, ordinary, sets, counted, sha256, aside = (drawn[key] for key in (
        "found", "read", "start", "at_least", "directory", "theirs", "sampling", "training", "batch", "recipe", "solves", "table", "enough", "kept", "ordinary", "sets", "counted",
        "sha256", "aside"))
    goal_ids = [row["problem_id"] for row in read["goal_problems"]]
    measurement = {"sampling": sampling["name"], "episodes": sampling["episodes"], "sampling_seed": sampling["sampling_seed"], "problems": len(goal_ids)}

    # ---- THE ONE CHANGE: `second_full`, where L4p's run keeps it; and L5d's finished run of this seed, whose sets these are (both refused here, with nothing written)
    of_the_start = read_second_full(config, full, read, sampling)
    if of_the_start["summary"]["recorded_by_l4p"]["rank"] != theirs["start_recipe"]["rank"] or of_the_start["summary"]["recorded_by_l4p"]["alpha"] != theirs["start_recipe"]["alpha"]:
        raise RuntimeError(f"L4p's run recorded rank {of_the_start['summary']['recorded_by_l4p']['rank']} and alpha {of_the_start['summary']['recorded_by_l4p']['alpha']} and L5's "
                           f"run {directory.name} trained its models at {theirs['start_recipe']}: every model here is at the rank L5d's are. Nothing was written.")
    of_l5ds = read_l5ds_run(config, full, seed, measurement)
    start = the_start(config, full)
    try:
        beside = {model: refuse_problems_of(sets[model], set(goal_ids), set(aside), f"the training set of `{model}`") for model in MODELS}
        in_orders, coincide = {}, {}
        for model in MODELS:                                    # each set in the content-hash order of each ORDER SEED, as L5n orders rows and L5d trains
            by_order, coincide[model] = four_orders(sets[model], smoke)
            in_orders.update({trained_of(model, order): numbered(by_order[order]) for order in ORDERS})
        held = hold_to_theirs({**{model: sets[model] for model in MODELS}, **in_orders}, of_l5ds["stored"], of_l5ds["recorded"])
    except ValueError as error:
        raise RuntimeError(f"{error} (seed {seed}; L5's run {directory.name}; L5d's run {of_l5ds['directory'].name}). Nothing was written.") from error
    trained, count = list(in_orders), len(kept)
    steps = steps_per_pass(2 * count, batch)

    # ---- from here on files are written: the goal problems of each model's set, the goal set's groups, G', the held-aside problems, both start models' first samplings, the sets
    store.write_rows("problems.jsonl", [{**row, "set": the_set(model, sampling["name"])} for model in MODELS for row in read["goal_problems"]])
    store.write_rows(ladder_l4.GROUPS_FILE, read["groups"])
    store.write_rows(ladder_l4.AGAIN_FILE, [{"problem_id": problem_id} for problem_id in read["again"]])
    store.write_rows(ladder_l5.HELD_ASIDE_FILE, read["aside"])
    store.write_rows(stored_file(draws.start, sampling["name"], L4), read["kept"][draws.start, sampling["name"]])
    store.write_rows(stored_file(full.start, sampling["name"], L4), of_the_start["rows"])
    for name, own in (*((model, sets[model]) for model in MODELS), *in_orders.items()):
        store.write_rows(THE_STAGE.training_file(name), own)
    is_trained = (f"the recipe as it is, each set trained {len(ORDERS)} times, in the content-hash order of the order seeds {list(ORDERS)} of its own rows; the model measured is "
                  "the adapter whose every number is the mean of the four's (L5n's own function): L5d's `mean_constant`")
    report = found["report"]
    factors = ladder_l5_noise.learning_rate_factors(training["warmup_steps"], steps)        # the learning-rate factor a training's recorded rates are held to, as L5n records it
    summary = {
        "stage": L4, "label": LABEL, "check": "L5f", "seed": seed, "start": full.start, "start_adapter": str(start.adapter), "start_run": str(start.directory),
        "first_start": draws.start, "first_start_adapter": str(first.adapter), "first_start_run": str(first.directory),
        "models": list(MODELS), "trained": trained, "averaged": list(MODELS), "stand_in_engine": _stand_in(), "effective_batch": batch,
        "the_seeds": {"what": "the scout's seed, one task; the further ones run once its read names the escalation, recorded in the setting. Each is a seed of L5d's, whose own "
                              "run of that seed stored the sets this stage trains", "seeds": list(full.seeds), "more_seeds": list(full.more), "escalated": full.escalated,
                      "fixed": list(fixed_seeds(full.seeds, full.more, full.escalated))},
        "the_start_model": of_the_start["summary"],
        "the_search": {"what": f"L5's finished search of seed {deep.seed_of_the_search}, ONLY READ, as L5d reads it: nothing of it is sampled again. BY THIS TASK'S SEED, as a "
                               "further seed of L5 draws them: which deep solves the cap keeps and which verified attempt of each; which ordinary solves and which attempt; the "
                               "published rows; each set's order", "run": str(directory), "seed": deep.seed_of_the_search, "problems": len(read["drawn"]),
                       "deep_solves": len(solves), "rows_of_each_kind": count, "cap": deep.cap, "enough_deep_solves": enough, "found": table, "headline": report.get("headline"),
                       "inconclusive": bool(report.get("inconclusive")), "ordinary_solves_drawn": len(ordinary), "held_aside_problems": len(aside),
                       "held_aside_file": ladder_l5.HELD_ASIDE_FILE},
        "l5ds_run": {"what": "L5d's FINISHED run of this seed (from the first start), ONLY READ: its three sets, each in its four orders, are the sets this run trains, HELD id for "
                             "id and order for order before anything was written; what its report said of its seed then is kept here, and the report step reads it again",
                     "run": str(of_l5ds["directory"]), "task": l5d_task(seed), "held": held, "at_the_first_step": of_l5ds["of_theirs"]},
        "the_sets": {model: {"what": {PUBLISHED: f"the first {2 * count:,} rows of the hash order of this seed among the rows `{draws.start}` was pretrained on, and nothing else",
                                      SHALLOW: f"{count:,} ordinary solves (one verified map attempt each, drawn by the seed) and the first {count:,} rows of that order",
                                      DEEP: f"{count:,} deep rows (one verified attempt of each deep solve the cap keeps, drawn by the seed) and the SAME first {count:,} rows"}[model],
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
        "recipe": {"what": f"L4t's (`ladder_ceiling.one_pass`, as the arm's twin is trained): FROM THE START ADAPTER (`{full.start}`), one pass, the native format. It is held "
                           f"to the recipe L5's run trained its models with. This run: {is_trained}", "name": MEAN_CONSTANT, **recipe, "adapter_seed": seed,
                   "optimizer_steps": steps, "learning_rate_factor": {key: factors[key] for key in ("what", "steps", "warmup_steps", CONSTANT)}},
        "the_sampling": {"what": f"every model is measured on the goal set's FIRST sampling alone, as the run of `{draws.start}` recorded it and as `{full.start}` was measured: "
                                 "its attempts a problem and its sampling seed, so that each pairs by problem with every stored model's first sampling. No rung, no second "
                                 "sampling, no held-aside set", **sampling, "sets": {model: the_set(model, sampling["name"]) for model in MODELS},
                         "stored_start": stored_file(full.start, sampling["name"], L4), "stored_first_start": stored_file(draws.start, sampling["name"], L4)},
        "the_first_start_wrote": {"what": f"what `{draws.start}`'s own run recorded of what it wrote, for the overfitting reads",
                                  "distinct_attempts": (read["of_the_start_run"].get("measured") or {}).get("distinct_attempts")},
        "goal_samplings": [sampling], "goal_set": len(goal_ids), "goal_set_again": len(read["again"]), "attempts_a_goal_problem": sampling["episodes"],
        "smoke": {"what": f"{SMOKE_VARIABLE}: a smoke run alone. L5's smoke run reads INCONCLUSIVE (a training of two rows has no loss to compare), which a real run is refused "
                          "for, and is read all the same; its two looser draws are made again as that run recorded them, and one deep solve is enough; orders of a set's "
                          "handful of rows that are alike are trained all the same; and an averaged adapter that is one of its models is kept all the same, and said",
                  "problems_at_least": at_least, "l5_read_though_inconclusive": bool(report.get("inconclusive")), "l5s_failed_checks": list(report.get("failed_checks") or [])}
        if smoke else None,
        "contradicted_side_setting": config["ladder_loop"]["episode"].get("contradicted_side", "all"),
        "start_recipe": theirs["start_recipe"], "start_run_read": theirs["start_run_read"]}
    store.mark_done(PREPARE, summary)
    own_checks = of_the_start["summary"]["l4ps_own_checks"]
    _say(f"L5f prepared, seed {seed}, from `{full.start}` ({start.adapter}; trained from `{draws.start}`, rank {theirs['start_recipe']['rank']}): L5d's three sets of "
         f"{2 * count:,} rows for this seed, held to the ones {of_l5ds['directory'].name} stored and trained, each in its four orders; {len(trained)} trainings of {steps:,} "
         f"optimizer steps, then {len(MODELS)} averaged adapters, each measured on `{sampling['name']}` alone: {sampling['episodes']} attempts at each of {len(goal_ids)} goal "
         f"problems, sampling seed {sampling['sampling_seed']}. L4p's own checks of `{full.start}` (PRINTED, deciding nothing): "
         + (f"failed {own_checks['failed']}" if own_checks["failed"] else "none failed" if own_checks["report_there"] else "its report is not there")
         + ("; A SMOKE RUN" if smoke else ""))
    return summary


def _of_this_run(store: ArtifactStore, what: str) -> dict:
    """The prepare step's summary, for a later step of THIS run."""
    _need(store, PREPARE, what, ENTRY_STAGE)
    return store.done_summary(PREPARE)


# --------------------------------------------------------------------------------------------------- train
def ladder_l5_full_train(config: dict, name: str) -> dict:
    """One training (`<model>_<k>`), by L4t's own training body (`ladder_l4_rows.train_a_model`) under this stage's
    names, HANDED THE START MODEL AS THIS STAGE FOUND IT (`the_start`: `second_full`'s adapter in L4p's run, at
    `pre_r64`'s rank and alpha): one pass over the model's set in the order of order seed k, the recipe as it is, each
    step's learning rate recorded and held to it, as L5d trains under `mean_constant`."""
    full = the_settings(config)
    store, marker = _store(config, full), THE_STAGE.train_marker(name)
    if store.is_done(marker):
        return store.done_summary(marker)
    prepared = _of_this_run(store, f"the training of `{name}`")
    schedule = ladder_l4_rows.Schedule(decay=False, order=f"a content hash of the ORDER SEED {prepared['training_sets'][name]['order_seed']} and each row's id "
                                                          "(`assembly.training_order`): the set's own rows, in one of the four orders its averaged adapter is the mean of")
    return train_a_model(config, store, name, full.start, run_name(full, training_seed(config)), THE_STAGE, schedule=schedule, start_from=the_start(config, full))


# ------------------------------------------------------------------------------------------------- average
def ladder_l5_full_average(config: dict, model: str) -> dict:
    """NO TRAINING: the adapter `model`, whose every number is the mean of its four trainings' numbers at its place, BY
    L5n's OWN FUNCTION (`ladder_l5_noise.the_mean_adapter`): saved under `adapters/<model>`, read back, and held to be
    that mean, none of them and at the start adapter's rank, as L5n holds its own; what of it is not the start adapter's
    is counted against `second_full`'s. Refused there, with the step not marked done."""
    full = the_settings(config)
    store, marker = _store(config, full), average_marker(model)
    if store.is_done(marker):
        return store.done_summary(marker)
    prepared = _of_this_run(store, f"the averaged adapter `{model}`")
    of = list(prepared["the_averaged"][model]["of"])
    summary = {"stage": L4, "label": LABEL, "check": "L5f", "model": model, "recipe": MEAN_CONSTANT, "of": of, "adapter": str(store.root / ladder_assembly.ADAPTERS / model),
               "stand_in_engine": _stand_in()}
    return ladder_l5_noise.the_mean_adapter(store, the_start(config, full), model, of, marker, summary, bool(prepared.get("smoke")), THE_STAGE)


# ------------------------------------------------------------------------------------------------- measure
def ladder_l5_full_measure(config: dict, model: str) -> dict:
    """One of the three averaged models on the goal set's FIRST sampling alone, as L5d measures (`ladder_ceiling.measure_model`
    with no rung): the attempts and the sampling seed the first start's run recorded for that sampling (the one
    `second_full` was measured with), at the config's own temperature, from the adapter this run keeps. The model server
    is started with the start adapter's rank as its largest (`config_from`)."""
    full = the_settings(config)
    store = _store(config, full)
    prepared = _of_this_run(store, f"the measurement of `{model}`")
    adapter, sampling = store.root / ladder_assembly.ADAPTERS / model, prepared["the_sampling"]
    return measure_model(config_from(config, prepared, the_start(config, full), serving=True), store, {"goal_samplings": prepared["goal_samplings"]}, Model(
        name=model, called=f"L5f's model `{model}`", whose=f"L5f's model `{model}`", marker=THE_STAGE.measure_marker(model), trained_by=average_marker(model), adapter=adapter,
        no_adapter=f"the adapter `{model}` is kept by this stage, and one made again would not be the model its other figures stand beside. The measurement cannot be made",
        detail=f"G's first sampling alone, {sampling['episodes']} attempts a problem, sampling seed {sampling['sampling_seed']}",
        summary={"stage": L4, "label": LABEL, "check": "L5f", "model": model, "adapter": str(adapter)}, stage=ENTRY_STAGE, label=L4, also=what_else_a_model_wrote, rungs=False))


# -------------------------------------------------------------------------------------------------- report
def ladder_l5_full_report(config: dict) -> dict:
    from rlvr_lean.gpu.ladder_l3c import proof_lengths
    from rlvr_lean.reporting.ladder_l5_full import build_l5_full_report

    full = the_settings(config)
    store = _store(config, full)
    prepared = _of_this_run(store, "L5f's report")
    names, sampling = prepared["trained"], prepared["the_sampling"]
    for step in (step for model in MODELS for step in (*(THE_STAGE.train_marker(name) for name in prepared["the_averaged"][model]["of"]), average_marker(model),
                                                       THE_STAGE.measure_marker(model))):
        _need(store, step, "L5f's report", ENTRY_STAGE)
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
    first = the_first_start(config, full)
    # THE SETS ARE MADE AGAIN AND HELD by the report: against the rows the first start was pretrained on, read as the first step read them (the pretraining file, held to its hash).
    pretraining = ladder_l4_start.pretrained_on(first.directory, prepared["pretraining_file_sha256"], first.name)[0]
    base_map = [row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[1]]
    ids_of = lambda name: [row["problem_id"] for row in store.read_rows(name)]      # noqa: E731
    # L5d's run of the seed, read AGAIN by the report: its sets as stored and as trained (the check), and what its report says of its seed now (the level)
    l5d = ladder_l5_stock.read_l5d(config, full)
    # The length of a held-out problem's published proof is read HERE, by the report, and by nothing before it.
    report = build_l5_full_report(prepared, trains, trained, averaged, measured, rows, sets, pretraining, store.read_rows(sampling["stored_start"]),
                                  store.read_rows(sampling["stored_first_start"]), store.read_rows(ladder_l4.GROUPS_FILE), proof_lengths(), ids_of(ladder_l4.AGAIN_FILE),
                                  ids_of(ladder_l5.HELD_ASIDE_FILE), config["ladder_loop"]["l4"], base_map, {"sampling": full.draws.sampling, "attempts": full.draws.attempts},
                                  l5ds_sets(Path(l5d["run"])), l5d)
    directory = store.root / ladder_assembly.ADAPTERS
    report["adapters"] = {"kept": True, "what": f"the adapters of {', '.join(f'`{name}`' for name in (*names, *MODELS))} ({LABEL}) are KEPT: no step of this stage deletes "
                                                f"one. `{full.start}` is L4p's run's and was only read",
                          "directory": str(directory), "there": sorted(path.name for path in directory.glob("*") if path.is_dir()) if directory.is_dir() else []}
    _write_json(store, REPORT_FILE, report)
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    for name in (THE_STAGE.loss_file(name) for name in names):
        store.mirror(name)
    for line in report["lines"]:
        print(line, flush=True)
    store.mark_done(REPORT, {"stage": L4, "label": LABEL, "check": "L5f", "seed": prepared["seed"], "recipe": MEAN_CONSTANT, "headline": report["headline"],
                             "branches": report["branches"], "measured": list(MODELS), "ok": report["ok"]})
    return report


def _for(step, name: str):
    def run(config: dict) -> dict:
        return step(config, name)
    return run


# In L5d's order: the sets held and stored; then model by model its four trainings, its averaged adapter and its measurement, before the next model; then the report.
STEPS = {PREPARE: ladder_l5_full_prepare}
for _model in MODELS:
    for _order in ORDERS:
        STEPS[THE_STAGE.train_marker(trained_of(_model, _order))] = _for(ladder_l5_full_train, trained_of(_model, _order))
    STEPS[average_marker(_model)] = _for(ladder_l5_full_average, _model)
    STEPS[THE_STAGE.measure_marker(_model)] = _for(ladder_l5_full_measure, _model)
STEPS[REPORT] = ladder_l5_full_report
