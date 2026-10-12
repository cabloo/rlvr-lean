"""L5c: the continuation. What lifted the published kind by 34 per 1,000 from the twice-pretrained model, and does more
of it go higher? Spec: docs/spec/ladder-loop.spec.md, "L5c: the continuation" and its "Made exact by the build".
Everything it makes is labelled pretrained on published proofs. Each step is its own process (`rlvr_lean.runner.entry`,
stage `ladder_l5_cont`). NO ROUND, NO SEARCH, NO SAMPLING BUT THE MEASUREMENTS, AND NO ROW THAT IS ONE OF THE MODEL'S
OWN PROOFS: one task at the scout's seed, four models, each measured on the goal set's first sampling alone and read
against L5f's STORED `published` of that seed (the bar).

L5f's FINISHED RUN OF THE SEED IS ONLY READ (`ladder_l5_full.own_run`: its seed's own directory; in a smoke run the one
L5f's run variable names): its four trainings of its `published` set, its averaged `published`, that set as stored
and its rows on the first sampling. `second_full` is read as L5f reads it (`ladder_l5_full.read_second_full`,
`the_start`). L5's finished search and every stored run its task read are read by L5d's own first-step reading
(`ladder_l5_draws.drawn_by_the_seed`). The run directory is its own (`ladder_l5_cont_<start>_seed<N>`).

  single    L5f's stored `published_1`, measured: no training
  wide      the mean of EIGHT adapters: L5f's stored `published_1` to `published_4` and four trainings here of the same
            rows from `second_full` in the orders of order seeds 5 to 8
  again     from L5f's stored averaged `published` (handed through the training body's `start_from`, as L5f hands
            `second_full`): the NEXT rows of the seed's order of the pretraining file, in four orders, averaged
  more      from `second_full`: L5s's `published_stock` by its rule for the seed (`ladder_l5_stock.stock_by_the_seed`:
            the rule is called, L5s's stored run is not read), in four orders, averaged

  ladder_l5_cont_prepare            REFUSES, naming what to do, before anything is written: a task of a seed the setting
                                    does not list, or of a further seed while the escalation is not recorded; what
                                    L5d's first-step reading and L5f's reading of `second_full` refuse; L5f's run of
                                    the seed missing, unfinished, INCONCLUSIVE, not to be read, of another seed, start or
                                    recipe, measured otherwise, or without its four `published` trainings, its averaged
                                    one, its stored sets and its rows on the first sampling; adapters of it that its own
                                    steps did not record at the rank, as not the start adapter's, or as the mean (on a
                                    GPU box, the files themselves); sets that are not L5f's stored `published` and L5s's
                                    rule, id for id; a row with text, of a held-out problem, of G or of the held-aside
                                    set; two order seeds that give one order. STORES the three sets, each in its four
                                    orders, and the three stored first samplings. No GPU, no Lean
  ladder_l5_cont_measure_single     L5f's stored `published_1` on the goal set's first sampling alone
  ladder_l5_cont_train_<model>_<k>  L4t's OWN training body (`ladder_l4_rows.train_a_model`) under this stage's names,
                                    handed its start model: the set in the order of order seed k
  ladder_l5_cont_average_<model>    NO TRAINING: the mean adapter (`ladder_l5_noise.the_mean_adapter`; `wide`'s with
                                    L5f's four, only read, before its own four)
  ladder_l5_cont_measure_<model>    the model on the goal set's FIRST sampling alone, as L5f measures
  ladder_l5_cont_report             the checks first, then the reads (`reporting/ladder_l5_cont.py`)

Twenty-one steps: prepare; `single` measured; then `wide`, `again` and `more`, each trained four times, averaged and
measured; the report. Every step is resumable and a step of its own. NO STEP DELETES AN ADAPTER, and no step writes into
a run it reads.

A SMOKE RUN ALONE (`RLVR_LEAN_LADDER_L5_CONT_SMOKE`, refused unless the run directory is one the task names and the map
is the fixtures'): L5's smoke run reads INCONCLUSIVE and is read all the same; one deep solve is enough; orders of a
set's handful of rows that are alike are trained all the same; and an averaged adapter that is one of its models is
kept. It reads L5f's SMOKE run, which L5f's own function finds by L5f's run variable (the smoke stage names it).
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path

from rlvr_lean.domain.ladder_round.dose import steps_per_pass
from rlvr_lean.domain.ladder_round.l4 import the_adapter_differs
from rlvr_lean.domain.ladder_round.l4_rows import THE_RULE
from rlvr_lean.domain.ladder_round.l5 import refuse_barred
from rlvr_lean.domain.ladder_round.l5_cont import (
    AGAIN,
    AVERAGED,
    BAR,
    MODELS,
    MORE,
    OF_THEIRS,
    ORDERS_OF,
    SINGLE,
    START,
    THE_SINGLE,
    TRAINED,
    WIDE,
    fixed_seeds,
    hold_the_members,
    hold_the_sets,
    in_their_orders,
    of_l5f,
    the_seed,
    the_sets,
    theirs_are_held,
)
from rlvr_lean.domain.ladder_round.l5_draws import CONSTANT, MEAN_CONSTANT, refuse_problems_of, trained_of
from rlvr_lean.domain.ladder_round.l5_noise import numbered
from rlvr_lean.domain.ladder_round.l5_stock import PUBLISHED_STOCK
from rlvr_lean.gpu import (
    ladder_assembly,
    ladder_ceiling,
    ladder_l4,
    ladder_l4_rank,
    ladder_l4_rows,
    ladder_l4_start,
    ladder_l5,
    ladder_l5_draws,
    ladder_l5_full,
    ladder_l5_noise,
    ladder_l5_stock,
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

L5C = "l5_cont"                                             # what this stage's own files are called
STAGE = ENTRY_STAGE = "ladder_l5_cont"                      # what its steps begin with; the stage, as `runner.entry` names it
RUN_VARIABLE = "RLVR_LEAN_LADDER_L5_CONT_RUN"               # another run directory than `ladder_l5_cont_<start>_seed<seed>` (a smoke run)
SMOKE_VARIABLE = "RLVR_LEAN_LADDER_L5_CONT_SMOKE"           # a SMOKE run alone (refused elsewhere): L5's INCONCLUSIVE smoke run is read, and what else the module's note says
SETTING = "ladder_loop.l5_cont"
THE_STAGE = ladder_l4_rows.Stage(name=ENTRY_STAGE, steps=STAGE, files=L5C, reader="L5c", start_said_by=f"{SETTING}.start")
PREPARE, REPORT = THE_STAGE.prepare, f"{STAGE}_report"
REPORT_FILE = "report_ladder_l5_cont.json"
OF_L5F = ladder_l5_full.THE_STAGE                           # the stage whose stored run of the seed this one stands on, as it names its own files and markers
OF_THIS_STAGE = ladder_l5_draws.OfASeed(who="L5c", run_variable=RUN_VARIABLE, smoke_variable=SMOKE_VARIABLE)      # what L5d's first-step reading says of this stage when it refuses
THE_BAR = f"l5f_{BAR}"                                      # L5f's `published`, as this run names the copy of its stored first sampling


@dataclass(frozen=True)
class Cont:
    """`ladder_loop.l5_cont`, as the stage's steps read it."""
    start: str                          # `second_full`: `wide` and `more` are trained from its adapter; `again` from L5f's averaged `published`, itself trained from it
    seeds: tuple[int, ...]              # the scout's seed: one task
    more: tuple[int, ...]               # the further seeds, which run once the scout's read names the escalation
    escalated: bool                     # ... recorded here once it has: a task of a further seed is refused until then
    full: ladder_l5_full.Full           # L5f's settings as ITS task reads them: a seed here is one of L5f's, whose run of that seed holds the bar
    draws: ladder_l5_draws.Draws        # L5d's settings as ITS task reads them: the first start, the sampling, its attempts and the recipe
    deep: ladder_l5.Deep                # L5's settings as ITS task reads them: the search's seed, and what an ordinary solve is


def the_settings(config: dict) -> Cont:
    """THE one place that reads `ladder_loop.l5_cont`. Refused (ValueError): a start that is not `second_full`; seeds
    that are not distinct whole numbers, in the two lists together, or that hold a seed that is none of L5f's (the bar
    is L5f's own run of the seed); no seed; an escalation that is not true or false; L5f's own settings, when their
    reader refuses them."""
    settings = config["ladder_loop"]["l5_cont"]
    full = ladder_l5_full.the_settings(config)
    start, seeds, more, escalated = settings["start"], tuple(settings["seeds"]), tuple(settings["more_seeds"]), settings["escalated"]
    if start != START or start != full.start:
        raise ValueError(f"{SETTING}.start is {start!r}: this stage is specified from `{START}` alone, the start of L5f's models, whose `{BAR}` of the seed is the bar")
    both, of_theirs = (*seeds, *more), (*full.seeds, *full.more)
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in both) or len(set(both)) != len(both) or set(both) - set(of_theirs):
        raise ValueError(f"{SETTING}.seeds is {list(seeds)} and more_seeds {list(more)}: distinct whole numbers of at least 0, at least one in seeds, each a seed of L5f's "
                         f"({list(of_theirs)}: {ladder_l5_full.SETTING}), whose own run of that seed holds the bar and the adapters this stage reads")
    if not isinstance(escalated, bool):
        raise ValueError(f"{SETTING}.escalated is {escalated!r}: true once the scout's read names the escalation, false until then")
    return Cont(start=start, seeds=seeds, more=more, escalated=escalated, full=full, draws=full.draws, deep=full.deep)


def run_name(cont: Cont, seed: int) -> str:
    """The stage's run directory, its own: `ladder_l5_cont_second_full_seed<N>`."""
    return f"{ENTRY_STAGE}_{cont.start}_seed{seed}"


def own_run(config: dict, cont: Cont) -> str:
    """This task's run directory: the seed's own, or the one the task names (a smoke run)."""
    return os.environ.get(RUN_VARIABLE) or run_name(cont, training_seed(config))


def _store(config: dict, cont: Cont) -> ArtifactStore:
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    return ArtifactStore(_runs(config) / own_run(config, cont), Path(mirror) if mirror else None)


def l5f_directory(config: dict, cont: Cont) -> Path:
    """L5f's run of this task's seed, by L5f's OWN naming (`ladder_l5_full.own_run`: its seed's own directory; in a smoke
    run, the one L5f's run variable names). Only read."""
    return _runs(config) / ladder_l5_full.own_run(config, cont.full)


def l5f_task(seed: int) -> str:
    """The task that writes L5f's run of a seed, for a refusal."""
    stage = ladder_l5_full.ENTRY_STAGE
    return f"the task of stage `{stage}` for seed {seed} (`python -m rlvr_lean.runner.entry --stage {stage} --seeds {seed}`; for the smoke run, stage `{stage}_smoke`)"


def l5fs_adapters(directory: Path) -> dict[str, Path]:
    """L5f's five stored adapters of its `published` kind, where its run keeps them: its four trainings, then the mean."""
    return {name: directory / ladder_assembly.ADAPTERS / name for name in (*OF_THEIRS, BAR)}


def the_start(config: dict, cont: Cont) -> Start:
    """`second_full`, as L5f finds it (`ladder_l5_full.the_start`): the start of `wide` and of `more`, and the adapter
    every model here is served beside (its rank and alpha are `pre_r64`'s)."""
    return ladder_l5_full.the_start(config, cont.full)


def the_bar(config: dict, cont: Cont) -> Start:
    """L5f's stored averaged `published` of the seed's run AS A START MODEL: what `again` is trained from, as L4t's
    training body and L5n's averaging body take one. It is `pre_r64`'s `Start` (its rank, alpha and check) under the
    bar's name, with the directory and adapter of L5f's run. Only read."""
    directory = l5f_directory(config, cont)
    return dataclasses.replace(ladder_l5_full.the_first_start(config, cont.full), name=BAR, directory=directory, adapter=l5fs_adapters(directory)[BAR],
                               task=l5f_task(training_seed(config)), prepare=OF_L5F.prepare, measure=OF_L5F.measure_marker(BAR))


def start_of(config: dict, cont: Cont, model: str) -> Start:
    """The start model of one of the three averaged models: L5f's averaged `published` for `again`, `second_full` otherwise."""
    return the_bar(config, cont) if model == AGAIN else the_start(config, cont)


def smoke_run() -> bool:
    """Whether the task says it is a smoke run; every other run: the rule, and nothing else."""
    return bool(os.environ.get(SMOKE_VARIABLE))


def average_marker(model: str) -> str:
    """The step that makes a model's averaged adapter, and its marker."""
    return f"{STAGE}_average_{model}"


def the_reader(cont: Cont, smoke: bool) -> ladder_l5.OfTheSearch:
    """Who reads L5's finished run here, for what L5's own reading says when it refuses it. A SMOKE run alone reads a run
    whose report is INCONCLUSIVE (one not to be read never is)."""
    return ladder_l5.OfTheSearch(
        who="L5c", why=f"L5c runs no search: its rows are published rows of the seed's order, counted by the finished search of seed {cont.deep.seed_of_the_search}",
        stands="no model here stands beside such a run", models="L5c's models stand beside L5f's, whose sets were drawn from that run's search", inconclusive=smoke)


# ------------------------------------------------------------------------------------------- L5f's run of the seed
def read_l5f(config: dict, cont: Cont) -> dict:
    """L5f's run of this task's seed, ONLY READ: what its report says of its own seed (`of_this_seed`) WHEN ITS RUN IS ON
    THE BOX AND ITS REPORT WRITTEN, NOT THERE otherwise, with why. Nothing is refused here."""
    directory = l5f_directory(config, cont)
    lacking = [name for name in (f"{ladder_l5_full.REPORT}.done.json", ladder_l5_full.REPORT_FILE) if not (directory / name).exists()]
    if lacking:
        return {"there": False, "run": str(directory), "why": f"{directory.name} does not hold {lacking}: L5f's run of this seed is not on the box, or not finished"}
    return {**(json.loads((directory / ladder_l5_full.REPORT_FILE).read_text()).get("of_this_seed") or {}), "there": True, "run": str(directory)}


def l5fs_sets(directory: Path) -> tuple[dict, dict]:
    """(the ids of the set L5f's run in `directory` STORED as `published` and as each of its four orders, in its order;
    by trained model, the ids its TRAINING LOOP recorded), each None where that run holds no such file. Only read."""
    def ids(name: str) -> list[str] | None:
        path = directory / OF_L5F.training_file(name)
        return [row["id"] for row in _rows(path)] if path.exists() else None

    def trained(name: str) -> list[str] | None:
        path = directory / OF_L5F.loss_file(name)
        return json.loads(path.read_text()).get("rows_trained") if path.exists() else None

    return {name: ids(name) for name in (BAR, *OF_THEIRS)}, {name: trained(name) for name in OF_THEIRS}


def read_l5fs_run(config: dict, cont: Cont, seed: int, measurement: dict, sampling: str) -> dict:
    """L5f's FINISHED run of this task's seed, ONLY READ. REFUSED (RuntimeError), naming the task to run: a run that is
    not there or not finished, of another seed, start or recipe, INCONCLUSIVE, not to be read, or measured otherwise
    than this run (`l5_cont.of_l5f`); one without the markers of its prepare step, of its four trainings of `published`,
    of that model's averaging and of its measurement, its stored `published` set, each of its four orders, the four
    trainings' own records, or its rows on the first sampling; adapters that its own steps did not record as read back
    at the rank, not the start adapter's, and (the averaged one) the mean of the four (`l5_cont.theirs_are_held`); and,
    on a GPU box, an adapter of the five that is not there, reads back at another rank, is number for number the start
    adapter's, or is of a type the mean is not written in. Returns its directory, what its report says of its seed, its
    sets as stored and as trained, the check of its adapters, its `published` rows on the first sampling and what its
    measuring step recorded."""
    entry, task = read_l5f(config, cont), l5f_task(seed)
    why = of_l5f(entry, seed, MEAN_CONSTANT, measurement)
    if why:
        raise RuntimeError(f"L5f's run of seed {seed} cannot carry L5c: {why}. Its stored `{BAR}` is the bar of every read here, the start of `{AGAIN}`, and with its four "
                           f"trainings half of `{WIDE}`. Run {task} to its end first; nothing was written.")
    directory, first = Path(entry["run"]), ladder_l5_full.the_first_start(config, cont.full)
    markers = {name: OF_L5F.train_marker(name) for name in OF_THEIRS}
    rows_file = f"episodes_{the_set(BAR, sampling)}_problems.jsonl"
    needed = [*(f"{marker}.done.json" for marker in (OF_L5F.prepare, *markers.values(), ladder_l5_full.average_marker(BAR), OF_L5F.measure_marker(BAR))),
              *(OF_L5F.training_file(name) for name in (BAR, *OF_THEIRS)), *(OF_L5F.loss_file(name) for name in OF_THEIRS), rows_file]
    lacking = [name for name in needed if not (directory / name).exists()]
    if lacking:
        raise RuntimeError(f"{directory} does not hold {lacking}: the trainings of `{BAR}` L5f's run of seed {seed} made, its averaged adapter's record, its stored sets or its "
                           f"rows on `{sampling}`. Run {task} to its end first; nothing was written.")
    done = lambda marker: json.loads((directory / f"{marker}.done.json").read_text())      # noqa: E731
    trainings = {name: {"adapter": own.get("adapter"), "adapter_saved": own.get("adapter_saved"), "differs": the_adapter_differs(adapter_against_the_start(own))}
                 for name, own in ((name, done(marker)) for name, marker in markers.items())}
    now = None
    if not ladder_ceiling._stand_in():     # on a GPU box (the trainings' own test of it): the files themselves, as every later step reads them
        now, start = {}, the_start(config, cont)
        for name, adapter in l5fs_adapters(directory).items():
            if not (adapter / ADAPTER_FILE).exists() or not (adapter / ladder_l4_rank.ADAPTER_CONFIG).exists():
                raise RuntimeError(f"{adapter} is not there: the adapter `{name}` of L5f's run of seed {seed} is read by this stage, and that run keeps it. Queue {task} "
                                   "again (its step makes it again); nothing was written.")
            read_back, against = ladder_l4_rank.adapter_read_back(adapter), compared(adapter / ADAPTER_FILE, start.adapter / ADAPTER_FILE)
            other = sorted({own["dtype"] for own in tensors_of(adapter / ADAPTER_FILE)[0].values()} - set(MEAN_TYPES))
            now[name] = {"read_back": read_back, "against_the_start_adapter": against, "other_types": other,
                         "passes": ladder_l4_rank.is_the_checks(read_back, first.check) and against["elements_changed"] > 0 and not other}
    held = theirs_are_held(trainings, done(ladder_l5_full.average_marker(BAR)), first.check.rank, first.check.alpha, now)
    if not held["passes"]:
        failing = sorted(name for name, own in (now or {}).items() if not own["passes"])
        raise RuntimeError(f"the adapters of `{BAR}` in L5f's run {directory.name} are not held: its own steps recorded {held['trainings']} and, of the averaged one, "
                           f"{held['the_averaged']}" + (f"; read now, {failing} are not at rank {first.check.rank} and alpha {first.check.alpha}, are the start adapter's "
                                                        f"number for number, or are of a type the mean is not written in ({list(MEAN_TYPES)})" if failing else "")
                           + f". Each of the four is read back at `{first.name}`'s rank and is not `{cont.start}`'s, and the averaged one is their mean, or nothing here stands "
                           f"on them. Queue {task} again; nothing was written.")
    stored, recorded = l5fs_sets(directory)
    return {"directory": directory, "entry": entry, "stored": stored, "recorded": recorded, "held": held, "rows": _rows(directory / rows_file),
            "measured": done(OF_L5F.measure_marker(BAR))}


# ------------------------------------------------------------------------------------------------- prepare
def ladder_l5_cont_prepare(config: dict) -> dict:
    cont = the_settings(config)
    store = _store(config, cont)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    seed, deep, draws, full, smoke = training_seed(config), cont.deep, cont.draws, cont.full, smoke_run()
    try:
        the_seed(seed, cont.seeds, cont.more, cont.escalated)
    except ValueError as error:
        raise RuntimeError(f"{error} ({SETTING}). Nothing was written.") from error
    if smoke and not os.environ.get(RUN_VARIABLE):
        raise RuntimeError(f"{SMOKE_VARIABLE} is set: it is for a SMOKE run alone, in a run directory the task names ({RUN_VARIABLE}) and on the fixtures' base map. A run in "
                           "the seed's own directory reads a run of L5 whose checks passed. Nothing was written.")
    # ---- L5's finished run and every stored run its task read, how a model is measured and trained, the rows the first start was pretrained on: by L5d's OWN first-step
    # reading, with the averaged recipe's hold of the first start adapter's type (refused there, with nothing written)
    drawn = ladder_l5_draws.drawn_by_the_seed(config, draws, seed, smoke, the_reader(cont, smoke), True, OF_THIS_STAGE)
    found, read, first, the_map, at_least, directory, theirs, sampling, training, batch, recipe, solves, sha256, pretraining, aside = (drawn[key] for key in (
        "found", "read", "start", "the_map", "at_least", "directory", "theirs", "sampling", "training", "batch", "recipe", "solves", "sha256", "pretraining", "aside"))
    goal_ids = [row["problem_id"] for row in read["goal_problems"]]
    measurement = {"sampling": sampling["name"], "episodes": sampling["episodes"], "sampling_seed": sampling["sampling_seed"], "problems": len(goal_ids)}

    # ---- `second_full`, where L4p's run keeps it, as L5f reads it; and L5f's finished run of this seed, whose `published` is the bar (both refused here, with nothing written)
    of_the_start = ladder_l5_full.read_second_full(config, full, read, sampling)
    if of_the_start["summary"]["recorded_by_l4p"]["rank"] != theirs["start_recipe"]["rank"] or of_the_start["summary"]["recorded_by_l4p"]["alpha"] != theirs["start_recipe"]["alpha"]:
        raise RuntimeError(f"L4p's run recorded rank {of_the_start['summary']['recorded_by_l4p']['rank']} and alpha {of_the_start['summary']['recorded_by_l4p']['alpha']} and L5's "
                           f"run {directory.name} trained its models at {theirs['start_recipe']}: every model here is at the rank L5f's are. Nothing was written.")
    of_theirs = read_l5fs_run(config, cont, seed, measurement, sampling["name"])
    start, bar, their_run = the_start(config, cont), the_bar(config, cont), of_theirs["directory"]
    try:
        # THE THREE SETS: `more` by L5s's OWN RULE for this seed (called; its stored run is not read), `wide` and `again` the first and the next rows of the seed's order
        of_l5s = ladder_l5_stock.stock_by_the_seed(config, deep, seed, smoke, drawn)
        of_the_stock = of_l5s["sets"][PUBLISHED_STOCK]
        sets = {model: numbered(own) for model, own in the_sets(pretraining, seed, len(of_theirs["stored"][BAR] or []), len(of_the_stock)).items()}
        held = hold_the_sets(sets, of_theirs["stored"][BAR], [row["id"] for row in of_the_stock])
        members = hold_the_members(sets[WIDE], of_theirs["stored"], of_theirs["recorded"])
        heldout = {row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[0]} | {row["problem_id"] for row in read["groups"]}
        counted = {model: refuse_barred(sets[model], heldout, config["ladder_loop"]["l4"]["half_seed"], set(), f"the training set of `{model}`") for model in AVERAGED}
        beside = {model: refuse_problems_of(sets[model], set(goal_ids), set(aside), f"the training set of `{model}`") for model in AVERAGED}
        in_orders, coincide = in_their_orders(sets, smoke)      # each set in the content-hash order of each of ITS order seeds; `wide`'s four beside L5f's four
    except ValueError as error:
        raise RuntimeError(f"{error} (seed {seed}; L5's run {directory.name}; L5f's run {their_run.name}). Nothing was written.") from error
    trained = list(TRAINED)
    own_solves, by_the_minimum = len(solves) + len(of_l5s["of_ordinary"]), sum(row["taken_by"] != THE_RULE for row in of_l5s["ordinary"])
    steps = {model: steps_per_pass(len(sets[model]), batch) for model in AVERAGED}

    # ---- from here on files are written: the goal problems of each model's set, the goal set's groups, G', the held-aside problems, the three stored first samplings, the sets
    store.write_rows("problems.jsonl", [{**row, "set": the_set(model, sampling["name"])} for model in MODELS for row in read["goal_problems"]])
    store.write_rows(ladder_l4.GROUPS_FILE, read["groups"])
    store.write_rows(ladder_l4.AGAIN_FILE, [{"problem_id": problem_id} for problem_id in read["again"]])
    store.write_rows(ladder_l5.HELD_ASIDE_FILE, read["aside"])
    store.write_rows(stored_file(draws.start, sampling["name"], L4), read["kept"][draws.start, sampling["name"]])
    store.write_rows(stored_file(cont.start, sampling["name"], L4), of_the_start["rows"])
    store.write_rows(stored_file(THE_BAR, sampling["name"], L4), of_theirs["rows"])
    for name, own in (*((model, sets[model]) for model in AVERAGED), *in_orders.items()):
        store.write_rows(THE_STAGE.training_file(name), own)
    report = found["report"]
    factors = {model: ladder_l5_noise.learning_rate_factors(training["warmup_steps"], steps[model]) for model in AVERAGED}      # what a training's recorded rates are held to
    adapters = {name: str(path) for name, path in l5fs_adapters(their_run).items()}
    summary = {
        "stage": L4, "label": LABEL, "check": "L5c", "seed": seed, "start": cont.start, "start_adapter": str(start.adapter), "start_run": str(start.directory),
        "first_start": draws.start, "first_start_adapter": str(first.adapter), "first_start_run": str(first.directory),
        "models": list(MODELS), "trained": trained, "averaged": list(AVERAGED), "stand_in_engine": _stand_in(), "effective_batch": batch,
        "the_seeds": {"what": "the scout's seed, one task; the further ones run once its read names the escalation, recorded in the setting. Each is a seed of L5f's, whose own "
                              "run of that seed holds the bar", "seeds": list(cont.seeds), "more_seeds": list(cont.more), "escalated": cont.escalated,
                      "fixed": list(fixed_seeds(cont.seeds, cont.more, cont.escalated))},
        "the_start_model": of_the_start["summary"],
        "the_search": {"what": f"L5's finished search of seed {deep.seed_of_the_search}, ONLY READ, as L5d reads it: nothing of it is sampled again, and NO ROW HERE IS ONE OF ITS "
                               f"SOLVES. It counts the rows of `{MORE}` (L5s's rule: twice the own solves in stock)", "run": str(directory), "seed": deep.seed_of_the_search,
                       "problems": len(read["drawn"]), "deep_solves": len(solves), "headline": report.get("headline"), "inconclusive": bool(report.get("inconclusive")),
                       "held_aside_problems": len(aside), "held_aside_file": ladder_l5.HELD_ASIDE_FILE},
        "the_map": {"what": "the first start's own map, ONLY READ: L5s's rule counts its ordinary solves (a problem solved in so many of its attempts, L5's own setting)",
                    "run": str(the_map["directory"]), "set": the_map["set"], "solved_at_least": deep.least, "solved_at_most": deep.most},
        "the_more": {"what": f"`{MORE}` is L5s's `{PUBLISHED_STOCK}` by its own rule for this seed, CALLED and held id for id (L5s's stored run is not read): the first 2N rows "
                             "of the seed's order, N the own solves in stock (every deep solve of the search and every ordinary solve the rule gives)",
                     "deep_solves": len(solves), "ordinary_solves": len(of_l5s["of_ordinary"]), "ordinary_solves_of_the_rule": len(of_l5s["of_the_rule"]),
                     "ordinary_solves_taken_by_the_runs_minimum": by_the_minimum, "own_solves": own_solves, "rows": len(sets[MORE])},
        "l5fs_run": {"what": f"L5f's FINISHED run of this seed, ONLY READ: its stored `{BAR}` first sampling is THE BAR of every read; `{THE_SINGLE}` is `{SINGLE}`; its four "
                             f"trainings of `{BAR}` are half of `{WIDE}`; its averaged `{BAR}` is the start of `{AGAIN}`. Held before anything was written",
                     "run": str(their_run), "task": l5f_task(seed), "adapters": adapters, "held": of_theirs["held"], "the_sets": held, "the_members": members,
                     "at_the_first_step": {key: of_theirs["entry"].get(key) for key in ("seed", "start", "recipe", "ok", "inconclusive", "failed_checks", "the_measurement", "the_stack")},
                     "the_bar_wrote": {"distinct_attempts": of_theirs["measured"].get("distinct_attempts")}},
        "the_models": {SINGLE: {"what": f"L5f's stored `{THE_SINGLE}` (the first of the four trainings its `{BAR}` is the mean of), ONLY READ and measured: no training",
                                "adapter": adapters[THE_SINGLE]},
                       WIDE: {"what": f"the mean of EIGHT adapters: L5f's stored {', '.join(f'`{name}`' for name in OF_THEIRS)} (only read) and four trainings here of the same "
                                      f"{len(sets[WIDE]):,} rows from `{cont.start}`, in the orders of order seeds {list(ORDERS_OF[WIDE])}", "from": str(start.adapter)},
                       AGAIN: {"what": f"from L5f's stored averaged `{BAR}` (not from `{cont.start}`): the NEXT {len(sets[AGAIN]):,} rows of the seed's order, in the orders of "
                                       f"order seeds {list(ORDERS_OF[AGAIN])}, averaged: a second averaged step", "from": str(bar.adapter)},
                       MORE: {"what": f"from `{cont.start}`: the first {len(sets[MORE]):,} rows of the seed's order (L5s's `{PUBLISHED_STOCK}`), in the orders of order seeds "
                                      f"{list(ORDERS_OF[MORE])}, averaged: more rows in one step", "from": str(start.adapter)}},
        "the_sets": {model: {"what": {WIDE: f"the first {len(sets[WIDE]):,} rows of the hash order of this seed among the rows `{draws.start}` was pretrained on: L5f's `{BAR}` set",
                                      AGAIN: f"the NEXT {len(sets[AGAIN]):,} rows of that order (rows {len(sets[WIDE]) + 1:,} to {2 * len(sets[WIDE]):,})",
                                      MORE: f"the first {len(sets[MORE]):,} rows of that order: L5s's `{PUBLISHED_STOCK}` by its rule"}[model],
                             "file": THE_STAGE.training_file(model), **described(sets[model], batch), "barred": counted[model],
                             "problems_of_g_or_of_the_held_aside_set": {key: beside[model][key] for key in ("problems_of_the_goal_set", "problems_of_the_held_aside_set")}}
                     for model in AVERAGED},
        "training_sets": {name: {"what": f"the set of `{name.rpartition('_')[0]}` in the content-hash order of ORDER SEED {name.rpartition('_')[2]}",
                                 "file": THE_STAGE.training_file(name), "model": name.rpartition("_")[0], "order_seed": int(name.rpartition("_")[2]),
                                 **described(in_orders[name], batch)} for name in trained},
        "the_averaged": {model: {"what": "trained by nothing: every number of its adapter is the mean of its members' numbers at that place",
                                 "kept_by_l5f": list(OF_THEIRS) if model == WIDE else [], "of": [trained_of(model, order) for order in ORDERS_OF[model]]} for model in AVERAGED},
        "the_orders": {"what": "each set in the content-hash order of each ORDER SEED (`assembly.training_order`: a hash of the order seed and each row's id), as L5n orders "
                               f"rows; `{WIDE}`'s four are held beside L5f's four of the same rows. An order seed that is the task's own seed gives the seed's own order: it is "
                               "trained all the same, and named here", "order_seeds": {model: list(ORDERS_OF[model]) for model in AVERAGED},
                       **{model: coincide[model] for model in AVERAGED}},
        "pretraining_file": str(ladder_l4.pretraining_file()), "pretraining_file_sha256": sha256,
        "recipe": {"what": "L4t's (`ladder_ceiling.one_pass`, as the arm's twin is trained): FROM THE MODEL'S START ADAPTER, one pass, the native format, the recipe as it is "
                           "(a constant learning rate after the warm-up). It is held to the recipe L5's run trained its models with. Each set is trained four times, in the "
                           "orders of its order seeds; the model measured is the adapter whose every number is the mean of its members' (L5n's own function)",
                   "name": MEAN_CONSTANT, **recipe, "adapter_seed": seed, "optimizer_steps": steps,
                   "learning_rate_factor": {model: {key: factors[model][key] for key in ("what", "steps", "warmup_steps", CONSTANT)} for model in AVERAGED}},
        "the_sampling": {"what": f"every model is measured on the goal set's FIRST sampling alone, as the run of `{draws.start}` recorded it and as `{cont.start}` and L5f's "
                                 "models were measured: its attempts a problem and its sampling seed, so that each pairs by problem with every stored model's first sampling. "
                                 "No rung, no second sampling, no held-aside set", **sampling, "sets": {model: the_set(model, sampling["name"]) for model in MODELS},
                         "stored_start": stored_file(cont.start, sampling["name"], L4), "stored_first_start": stored_file(draws.start, sampling["name"], L4),
                         "stored_bar": stored_file(THE_BAR, sampling["name"], L4)},
        "goal_samplings": [sampling], "goal_set": len(goal_ids), "goal_set_again": len(read["again"]), "attempts_a_goal_problem": sampling["episodes"],
        "smoke": {"what": f"{SMOKE_VARIABLE}: a smoke run alone. L5's smoke run reads INCONCLUSIVE (a training of two rows has no loss to compare), which a real run is refused "
                          "for, and is read all the same; its two looser draws are made again as that run recorded them, and one deep solve is enough; orders of a set's "
                          "handful of rows that are alike are trained all the same; and an averaged adapter that is one of its models is kept all the same, and said",
                  "problems_at_least": at_least, "l5_read_though_inconclusive": bool(report.get("inconclusive")), "l5s_failed_checks": list(report.get("failed_checks") or [])}
        if smoke else None,
        "contradicted_side_setting": config["ladder_loop"]["episode"].get("contradicted_side", "all"),
        "start_recipe": theirs["start_recipe"], "start_run_read": theirs["start_run_read"]}
    store.mark_done(PREPARE, summary)
    _say(f"L5c prepared, seed {seed}, from `{cont.start}` ({start.adapter}) and from L5f's run {their_run.name}: `{SINGLE}` is its `{THE_SINGLE}`; `{WIDE}` its four "
         f"trainings and four more of the same {len(sets[WIDE]):,} rows (order seeds {list(ORDERS_OF[WIDE])}); `{AGAIN}` the next {len(sets[AGAIN]):,} rows from its averaged "
         f"`{BAR}`; `{MORE}` the first {len(sets[MORE]):,} rows from `{cont.start}`. {len(trained)} trainings of {[steps[name.rpartition('_')[0]] for name in trained]} optimizer "
         f"steps, {len(AVERAGED)} averaged adapters, {len(MODELS)} measurements on `{sampling['name']}` alone: {sampling['episodes']} attempts at each of {len(goal_ids)} goal "
         f"problems, sampling seed {sampling['sampling_seed']}" + ("; A SMOKE RUN" if smoke else ""))
    return summary


def _of_this_run(store: ArtifactStore, what: str) -> dict:
    """The prepare step's summary, for a later step of THIS run."""
    _need(store, PREPARE, what, ENTRY_STAGE)
    return store.done_summary(PREPARE)


# --------------------------------------------------------------------------------------------------- train
def ladder_l5_cont_train(config: dict, name: str) -> dict:
    """One training (`<model>_<k>`), by L4t's own training body (`ladder_l4_rows.train_a_model`) under this stage's
    names, HANDED ITS MODEL'S START (`start_of`: L5f's stored averaged `published` for `again`, `second_full` for the
    two others): one pass over the model's set in the order of order seed k, the recipe as it is, each step's learning
    rate recorded and held to it."""
    cont = the_settings(config)
    store, marker = _store(config, cont), THE_STAGE.train_marker(name)
    if store.is_done(marker):
        return store.done_summary(marker)
    prepared = _of_this_run(store, f"the training of `{name}`")
    start = start_of(config, cont, prepared["training_sets"][name]["model"])
    schedule = ladder_l4_rows.Schedule(decay=False, order=f"a content hash of the ORDER SEED {prepared['training_sets'][name]['order_seed']} and each row's id "
                                                          "(`assembly.training_order`): the set's own rows, in one of the orders its averaged adapter is the mean of")
    return train_a_model(config, store, name, start.name, run_name(cont, training_seed(config)), THE_STAGE, schedule=schedule, start_from=start)


# ------------------------------------------------------------------------------------------------- average
def ladder_l5_cont_average(config: dict, model: str) -> dict:
    """NO TRAINING: the adapter `model`, whose every number is the mean of its members' numbers at its place, BY L5n's
    OWN FUNCTION (`ladder_l5_noise.the_mean_adapter`): saved under `adapters/<model>`, read back, and held to be that
    mean, none of them and at the start adapter's rank. `wide`'s members are L5f's four stored trainings (`kept`: only
    read, where that run keeps them) and then this run's four. Refused there, with the step not marked done."""
    cont = the_settings(config)
    store, marker = _store(config, cont), average_marker(model)
    if store.is_done(marker):
        return store.done_summary(marker)
    prepared = _of_this_run(store, f"the averaged adapter `{model}`")
    own, of_l5f_ = list(prepared["the_averaged"][model]["of"]), list(prepared["the_averaged"][model]["kept_by_l5f"])
    theirs = l5fs_adapters(l5f_directory(config, cont))
    summary = {"stage": L4, "label": LABEL, "check": "L5c", "model": model, "recipe": MEAN_CONSTANT, "of": [*of_l5f_, *own], "kept_by_l5f": of_l5f_,
               "adapter": str(store.root / ladder_assembly.ADAPTERS / model), "stand_in_engine": _stand_in()}
    return ladder_l5_noise.the_mean_adapter(store, start_of(config, cont, model), model, own, marker, summary, bool(prepared.get("smoke")), THE_STAGE,
                                            kept={name: theirs[name] for name in of_l5f_} or None)


# ------------------------------------------------------------------------------------------------- measure
def ladder_l5_cont_measure(config: dict, model: str) -> dict:
    """One of the four models on the goal set's FIRST sampling alone, as L5f measures (`ladder_ceiling.measure_model` with
    no rung): the attempts and the sampling seed the first start's run recorded for that sampling, at the config's own
    temperature. `single` from the adapter L5f's run keeps (`published_1`, only read: it needs no step of this run but
    the first); each of the three others from the averaged adapter this run keeps. The model server is started with the
    start adapter's rank as its largest (`config_from`)."""
    cont = the_settings(config)
    store = _store(config, cont)
    prepared = _of_this_run(store, f"the measurement of `{model}`")
    sampling = prepared["the_sampling"]
    adapter = l5fs_adapters(l5f_directory(config, cont))[THE_SINGLE] if model == SINGLE else store.root / ladder_assembly.ADAPTERS / model
    whose = (f"it is L5f's stored `{THE_SINGLE}`, which that run keeps: queue {l5f_task(prepared['seed'])} again" if model == SINGLE else
             f"the adapter `{model}` is kept by this stage, and one made again would not be the model its other figures stand beside")
    return measure_model(config_from(config, prepared, the_start(config, cont), serving=True), store, {"goal_samplings": prepared["goal_samplings"]}, Model(
        name=model, called=f"L5c's model `{model}`", whose=f"L5c's model `{model}`", marker=THE_STAGE.measure_marker(model),
        trained_by=PREPARE if model == SINGLE else average_marker(model), adapter=adapter, no_adapter=f"{whose}. The measurement cannot be made",
        detail=f"G's first sampling alone, {sampling['episodes']} attempts a problem, sampling seed {sampling['sampling_seed']}",
        summary={"stage": L4, "label": LABEL, "check": "L5c", "model": model, "adapter": str(adapter)}, stage=ENTRY_STAGE, label=L4, also=what_else_a_model_wrote, rungs=False))


# -------------------------------------------------------------------------------------------------- report
def ladder_l5_cont_report(config: dict) -> dict:
    from rlvr_lean.gpu.ladder_l3c import proof_lengths
    from rlvr_lean.reporting.ladder_l5_cont import build_l5_cont_report

    cont = the_settings(config)
    store = _store(config, cont)
    prepared = _of_this_run(store, "L5c's report")
    names, sampling = prepared["trained"], prepared["the_sampling"]
    for step in (THE_STAGE.measure_marker(SINGLE), *(step for model in AVERAGED for step in (
            *(THE_STAGE.train_marker(name) for name in prepared["the_averaged"][model]["of"]), average_marker(model), THE_STAGE.measure_marker(model)))):
        _need(store, step, "L5c's report", ENTRY_STAGE)
    sets = {name: store.read_rows(THE_STAGE.training_file(name)) for name in (*AVERAGED, *names)}
    trains, trained = {}, {}
    for name in names:                  # what a model WAS trained on is the training loop's own record of row ids, read back to its stored set's rows (never their text)
        trains[name] = store.done_summary(THE_STAGE.train_marker(name))
        by_id = {row["id"]: row for row in sets[name]}
        trained[name] = [{key: by_id[row_id].get(key) for key in ("id", "problem_id", "origin", "row_of_the_file")}
                         for row_id in json.loads(store.path(THE_STAGE.loss_file(name)).read_text())["rows_trained"]]
    averaged = {model: store.done_summary(average_marker(model)) for model in AVERAGED}
    measured = {model: store.done_summary(THE_STAGE.measure_marker(model)) for model in MODELS}
    rows = {model: _results(store, the_set(model, sampling["name"])) for model in MODELS}
    first = ladder_l5_full.the_first_start(config, cont.full)
    # THE SETS ARE MADE AGAIN AND HELD by the report: against the rows the first start was pretrained on, read as the first step read them (the pretraining file, held to its hash),
    # against L5f's run AS IT IS NOW, and against L5s's rule counted again from the stored search and map (a SMOKE run's solves taken by the run's minimum are the first step's).
    pretraining = ladder_l4_start.pretrained_on(first.directory, prepared["pretraining_file_sha256"], first.name)[0]
    base_map = [row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[1]]
    ids_of = lambda name: [row["problem_id"] for row in store.read_rows(name)]      # noqa: E731
    l5f_now = read_l5f(config, cont)
    deep_solves, ordinary = ladder_l5_stock.the_rule_again(store, cont, {**prepared, "smoke": None})
    more_by_the_rule = 2 * (len(deep_solves) + len(ordinary) + prepared["the_more"]["ordinary_solves_taken_by_the_runs_minimum"])
    # The length of a held-out problem's published proof is read HERE, by the report, and by nothing before it.
    report = build_l5_cont_report(prepared, trains, trained, averaged, measured, rows, sets, pretraining, store.read_rows(sampling["stored_start"]),
                                  store.read_rows(sampling["stored_first_start"]), store.read_rows(sampling["stored_bar"]), store.read_rows(ladder_l4.GROUPS_FILE),
                                  proof_lengths(), ids_of(ladder_l4.AGAIN_FILE), ids_of(ladder_l5.HELD_ASIDE_FILE), config["ladder_loop"]["l4"], base_map,
                                  {"sampling": cont.draws.sampling, "attempts": cont.draws.attempts}, l5fs_sets(Path(l5f_now["run"])), l5f_now, more_by_the_rule)
    directory = store.root / ladder_assembly.ADAPTERS
    report["adapters"] = {"kept": True, "what": f"the adapters of {', '.join(f'`{name}`' for name in (*names, *AVERAGED))} ({LABEL}) are KEPT: no step of this stage deletes "
                                                f"one. `{cont.start}` is L4p's run's, `{BAR}` and its four trainings L5f's: they were only read",
                          "directory": str(directory), "there": sorted(path.name for path in directory.glob("*") if path.is_dir()) if directory.is_dir() else []}
    _write_json(store, REPORT_FILE, report)
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    for name in (THE_STAGE.loss_file(name) for name in names):
        store.mirror(name)
    for line in report["lines"]:
        print(line, flush=True)
    store.mark_done(REPORT, {"stage": L4, "label": LABEL, "check": "L5c", "seed": prepared["seed"], "recipe": MEAN_CONSTANT, "headline": report["headline"],
                             "branches": report["branches"], "measured": list(MODELS), "ok": report["ok"]})
    return report


def _for(step, name: str):
    def run(config: dict) -> dict:
        return step(config, name)
    return run


# In the section's order: the first step; `single` measured (no training); then model by model its four trainings, its averaged adapter and its measurement; then the report.
STEPS = {PREPARE: ladder_l5_cont_prepare, THE_STAGE.measure_marker(SINGLE): _for(ladder_l5_cont_measure, SINGLE)}
for _model in AVERAGED:
    for _order in ORDERS_OF[_model]:
        STEPS[THE_STAGE.train_marker(trained_of(_model, _order))] = _for(ladder_l5_cont_train, trained_of(_model, _order))
    STEPS[average_marker(_model)] = _for(ladder_l5_cont_average, _model)
    STEPS[THE_STAGE.measure_marker(_model)] = _for(ladder_l5_cont_measure, _model)
STEPS[REPORT] = ladder_l5_cont_report
