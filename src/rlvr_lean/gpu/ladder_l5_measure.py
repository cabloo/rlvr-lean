"""L5m: the best model, measured in full. Does the gain hold on the held-aside set and on fresh attempts? Spec:
docs/spec/ladder-loop.spec.md, "L5m: the best model, measured in full" and its "Made exact by the build".
Everything it makes is labelled pretrained on published proofs. Each step is its own process (`rlvr_lean.runner.entry`,
stage `ladder_l5_measure`). NO TRAINING, NO ROUND, NO SEARCH: one task, ONE stored averaged adapter (THE CANDIDATE, named
in `ladder_loop.l5_measure`: the stage that made it, its seed, its name there) measured IN FULL as L5's models are,
against `second_full` (THE BAR), the model it was trained from.

ONLY READ, each left file for file as it was: the candidate's adapter, in the run of the stage that made it (L5f's
`ladder_l5_draws_second_full_seed<N>` by `ladder_l5_full.own_run`, or L5c's `ladder_l5_cont_second_full_seed<N>` by
`ladder_l5_cont.own_run`), with that run's report, its record of the mean, its stored first sampling and the rows its
trainings recorded; `second_full` in L4p's run, as L5f reads it (`ladder_l5_full.read_second_full`, `the_start`), with
its stored attempts on the rungs and on G's two samplings; L5's finished search (the held-aside set) and every stored
run its task read, by L5's own first-step reading (`ladder_l5.read_the_search`, `read_for_the_search`). The run
directory is its own (`ladder_l5_measure_<stage>_<model>_seed<N>`). The steps are named `ladder_l5m_...` and the files
`l5m_...`: `gpu/ladder_l5.py` already has steps named `ladder_l5_measure_<model>`.

  ladder_l5m_prepare                  REFUSES, naming what to do, before anything is written: a task of another seed
                                      than the candidate's; what L5's first-step reading and L5f's reading of
                                      `second_full` refuse; the bar's stored rungs or samplings of G not there or
                                      sampled otherwise; the candidate's run missing, unfinished, INCONCLUSIVE, not to
                                      be read, of another seed, start or recipe, or measured otherwise; one without the
                                      markers, the stored sets or the loss records of the trainings its averaged
                                      adapter is the mean of; a record of the mean that does not hold (not the mean,
                                      another rank, another adapter than the one that stage measured); on a GPU box, the
                                      adapter's files missing, at another rank, or number for number `second_full`'s; a
                                      row the candidate was trained on that is held out, of G or of the held-aside set.
                                      STORES the problems of every set this run samples, the bar's and `pre_r64`'s
                                      stored rows, the candidate's stored first sampling, and the rows the candidate
                                      was trained on (by id and place, never with text). No GPU, no Lean
  ladder_l5m_measure_candidate        the candidate on the three rungs and on G's TWO samplings, by the body L5's
                                      models are measured with (`ladder_ceiling.measure_model`), with the sampling seeds
                                      `pre_r64`'s run recorded: its first sampling is MEASURED AGAIN, not copied
  ladder_l5m_held_aside_candidate     the candidate on the held-aside set, by L5's own body (`on_the_held_aside_set`)
  ladder_l5m_held_aside_second_full   the bar there, from the adapter L4p's run keeps, UNLESS a stored measurement of it
                                      with the same setting was found by the first step (L6's run): then the step makes
                                      nothing, marks nothing, and says so
  ladder_l5m_report                   the checks first, then the reads (`reporting/ladder_l5_measure.py`)

Five steps. Every step is resumable and a step of its own. NO STEP WRITES INTO A RUN IT READS, and no adapter is made.

A SMOKE RUN ALONE (`RLVR_LEAN_LADDER_L5_MEASURE_SMOKE`, refused unless the run directory is one the task names and the
map is the fixtures'): L5's smoke run reads INCONCLUSIVE and is read all the same, and a candidate whose averaged
adapter is one of its members (a smoke run's, recorded so by the stage that made it) is measured all the same. It
reads the candidate from the SMOKE run of the stage that made it, which that stage's own function finds by its run
variable (the smoke stage names both). THE SMOKE WORLD HOLDS ONE SAMPLING OF G (the smoke `pre_r64`'s run reads L1's
smoke run alone): a smoke run goes on with that one, records it (`smoke.one_sampling_of_g`), and its report makes no
read on fresh attempts; a real run is refused for it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rlvr_lean.domain.ladder_round.l5 import refuse_barred
from rlvr_lean.domain.ladder_round.l5_draws import MEAN_CONSTANT, refuse_problems_of
from rlvr_lean.domain.ladder_round.l5_measure import BAR, CANDIDATE, L5C, L5F, MODELS_OF, SAID, is_the_one_measured, of_its_run, the_candidate, the_seed
from rlvr_lean.gpu import (
    ladder_assembly,
    ladder_ceiling,
    ladder_l4,
    ladder_l4_rank,
    ladder_l4_rows,
    ladder_l5,
    ladder_l5_cont,
    ladder_l5_draws,
    ladder_l5_full,
    ladder_l6,
    ladder_loop,
)
from rlvr_lean.gpu.ladder_ceiling import RUNG_PART, Model, Reading, _need, goal_set, measure_model, rung_set, stored_file
from rlvr_lean.gpu.ladder_dose import _rows, _runs
from rlvr_lean.gpu.ladder_l3d1 import what_else_a_model_wrote
from rlvr_lean.gpu.ladder_l4 import L4, LABEL, _say
from rlvr_lean.gpu.ladder_l4_start import config_from
from rlvr_lean.gpu.ladder_round import _results, _stand_in, _write_json, training_seed
from rlvr_lean.infrastructure.adapter_files import ADAPTER_FILE, compared
from rlvr_lean.infrastructure.artifact_store import ArtifactStore

L5M = "l5m"                                                 # what this stage's own files are called
STAGE = "ladder_l5m"                                        # what its steps begin with (`ladder_l5_measure_<model>` is a step of L5's own)
ENTRY_STAGE = "ladder_l5_measure"                           # the stage, as `runner.entry` names it; its run directory begins with it
RUN_VARIABLE = "RLVR_LEAN_LADDER_L5_MEASURE_RUN"            # another run directory than `ladder_l5_measure_<stage>_<model>_seed<seed>` (a smoke run)
SMOKE_VARIABLE = "RLVR_LEAN_LADDER_L5_MEASURE_SMOKE"        # a SMOKE run alone (refused elsewhere): L5's INCONCLUSIVE smoke run is read, and what else the module's note says
SETTING = "ladder_loop.l5_measure"
THE_STAGE = ladder_l4_rows.Stage(name=ENTRY_STAGE, steps=STAGE, files=L5M, reader="L5m", start_said_by=f"{SETTING}.bar")
PREPARE, REPORT = THE_STAGE.prepare, f"{STAGE}_report"
MEASURE = THE_STAGE.measure_marker(CANDIDATE)               # the candidate on the rungs and on G's two samplings
HELD_ASIDE = f"{STAGE}_held_aside"                          # what the two steps that measure the held-aside set begin with: the candidate's, then the bar's
REPORT_FILE = "report_ladder_l5_measure.json"
TRAINED_FILE = f"{L5M}_trained_rows.jsonl"                  # the rows the candidate's trainings recorded, each with its training: by id and place, NEVER with text
OF_L4P = ladder_l5_full.OF_L4P                              # the stage whose stored run keeps the bar, as it names its own files and markers
WHO = {CANDIDATE: CANDIDATE, BAR: BAR}                      # the two models measured on the held-aside set, by the name in their step and their set
READING = Reading(reader="L5m", models="L5m's two models", not_read="that set stands on the other side of a comparison, and no report of this run could be read against it",
                  setting=SETTING)
ROW_KEYS = ("id", "problem_id", "origin", "row_of_the_file")        # what is kept of a row a training recorded: never its text


@dataclass(frozen=True)
class Measure:
    """`ladder_loop.l5_measure`, as the stage's steps read it."""
    of: str                             # the stage that made the candidate: `l5_full` or `l5_cont`
    seed: int                           # its seed, and this task's
    model: str                          # the candidate's name in that stage: one of its averaged adapters
    bar: str                            # `second_full`: the model the candidate was trained from
    full: ladder_l5_full.Full           # L5f's settings as ITS task reads them
    cont: ladder_l5_cont.Cont           # L5c's settings as ITS task reads them
    draws: ladder_l5_draws.Draws        # L5d's: the first start (`pre_r64`)
    deep: ladder_l5.Deep                # L5's: the search's seed, the held-aside set's attempts and its sampling seed


@dataclass(frozen=True)
class Maker:
    """The stage that made the candidate, as THIS stage reads its run of the candidate's seed: only read."""
    said: str                           # who it is, in a printed line and a refusal: "L5f"
    stage: ladder_l4_rows.Stage         # how it names its own markers and files
    directory: Path                     # its run of the seed, by its OWN naming (in a smoke run, the one its run variable names)
    report: str                         # its report step's marker
    report_file: str                    # ... and its report
    average: Callable[[str], str]       # its step that averages a model's trainings, and its marker
    task: str                           # the task that writes that run, for a refusal


def the_settings(config: dict) -> Measure:
    """THE one place that reads `ladder_loop.l5_measure`. Refused (ValueError): a candidate that is not an averaged
    adapter of `l5_full` or of `l5_cont` at a seed of that stage (`l5_measure.the_candidate`); a bar that is not
    `second_full`, the start of both stages' models; L5f's and L5c's own settings, when their readers refuse them."""
    settings = config["ladder_loop"]["l5_measure"]
    full, cont = ladder_l5_full.the_settings(config), ladder_l5_cont.the_settings(config)
    candidate, bar = settings["candidate"], settings["bar"]
    of, seed, model = candidate["of"], candidate["seed"], candidate["model"]
    try:
        the_candidate(of, seed, model, {L5F: (*full.seeds, *full.more), L5C: (*cont.seeds, *cont.more)})
    except ValueError as error:
        raise ValueError(f"{SETTING}.candidate: {error}") from error
    if bar != BAR or bar != full.start:
        raise ValueError(f"{SETTING}.bar is {bar!r}: this stage is specified against `{BAR}` alone, the model the candidate was trained from (the start of L5f's and of "
                         "L5c's models)")
    return Measure(of=of, seed=seed, model=model, bar=bar, full=full, cont=cont, draws=full.draws, deep=full.deep)


def run_name(measure: Measure, seed: int) -> str:
    """The stage's run directory, its own, with the candidate in its name: `ladder_l5_measure_l5_full_published_seed<N>`."""
    return f"{ENTRY_STAGE}_{measure.of}_{measure.model}_seed{seed}"


def own_run(config: dict, measure: Measure) -> str:
    """This task's run directory: the candidate's own, or the one the task names (a smoke run)."""
    return os.environ.get(RUN_VARIABLE) or run_name(measure, training_seed(config))


def _store(config: dict, measure: Measure) -> ArtifactStore:
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    return ArtifactStore(_runs(config) / own_run(config, measure), Path(mirror) if mirror else None)


def smoke_run() -> bool:
    """Whether the task says it is a smoke run; every other run: the rule, and nothing else."""
    return bool(os.environ.get(SMOKE_VARIABLE))


def held_aside_marker(who: str) -> str:
    """The step that measures the held-aside set for one of the two models, and its marker."""
    return f"{HELD_ASIDE}_{WHO[who]}"


def task_of(stage: str, seed: int) -> str:
    """The task that writes a stage's run of a seed, for a refusal."""
    return f"the task of stage `{stage}` for seed {seed} (`python -m rlvr_lean.runner.entry --stage {stage} --seeds {seed}`; for the smoke run, stage `{stage}_smoke`)"


def the_maker(config: dict, measure: Measure) -> Maker:
    """The stage that made the candidate, and where its run of the candidate's seed is, BY THAT STAGE's OWN NAMING (no
    run name is stated in this module): L5f's (`ladder_l5_full.own_run`) or L5c's (`ladder_l5_cont.own_run`)."""
    if measure.of == L5F:
        return Maker(said=SAID[L5F], stage=ladder_l5_full.THE_STAGE, directory=_runs(config) / ladder_l5_full.own_run(config, measure.full), report=ladder_l5_full.REPORT,
                     report_file=ladder_l5_full.REPORT_FILE, average=ladder_l5_full.average_marker, task=task_of(ladder_l5_full.ENTRY_STAGE, measure.seed))
    return Maker(said=SAID[L5C], stage=ladder_l5_cont.THE_STAGE, directory=_runs(config) / ladder_l5_cont.own_run(config, measure.cont), report=ladder_l5_cont.REPORT,
                 report_file=ladder_l5_cont.REPORT_FILE, average=ladder_l5_cont.average_marker, task=task_of(ladder_l5_cont.ENTRY_STAGE, measure.seed))


def the_reader(measure: Measure, smoke: bool) -> ladder_l5.OfTheSearch:
    """Who reads L5's finished run here, for what L5's own reading says when it refuses it. A SMOKE run alone reads a run
    whose report is INCONCLUSIVE (one not to be read never is)."""
    return ladder_l5.OfTheSearch(
        who="L5m", why=f"L5m runs no search and trains nothing: it measures two stored models on the held-aside set of the finished search of seed "
                       f"{measure.deep.seed_of_the_search}, with the sampling seed that run's models were measured with there",
        stands="no model here is measured on the held-aside set of such a run", models="L5m's two models are measured on that run's held-aside set", inconclusive=smoke)


def its_adapter(maker: Maker, measure: Measure) -> Path:
    """The candidate's adapter, where the run that made it keeps it. Only read."""
    return maker.directory / ladder_assembly.ADAPTERS / measure.model


# ------------------------------------------------------------------------------------ the candidate's own run
def _done(directory: Path, marker: str) -> dict:
    """What a step of a stored run recorded, or nothing where it holds no such marker. Only read."""
    path = directory / f"{marker}.done.json"
    return json.loads(path.read_text()) if path.exists() else {}


def read_its_report(maker: Maker) -> dict:
    """The run that made the candidate, ONLY READ: what its report says of its own seed (`of_this_seed`) WHEN THE RUN IS
    ON THE BOX AND ITS REPORT WRITTEN, NOT THERE otherwise, with why. Nothing is refused here."""
    directory = maker.directory
    lacking = [name for name in (f"{maker.report}.done.json", maker.report_file) if not (directory / name).exists()]
    if lacking:
        return {"there": False, "run": str(directory), "why": f"{directory.name} does not hold {lacking}: {maker.said}'s run of this seed is not on the box, or not finished"}
    return {**(json.loads((directory / maker.report_file).read_text()).get("of_this_seed") or {}), "there": True, "run": str(directory)}


def its_members(config: dict, measure: Measure, maker: Maker, averaging: dict) -> list[tuple[str, ladder_l4_rows.Stage, Path]]:
    """The trainings the candidate's averaged adapter is the mean of, by that stage's OWN averaging record (`of`), each
    with the stage that names its files and the run that keeps them: the candidate's own run, or, for the members L5c's
    `wide` reads from L5f's run (`kept_by_l5f`), L5f's run of the seed."""
    of_l5f = set(averaging.get("kept_by_l5f") or [])
    theirs = (ladder_l5_full.THE_STAGE, ladder_l5_cont.l5f_directory(config, measure.cont)) if of_l5f else None
    return [(name, *(theirs if name in of_l5f else (maker.stage, maker.directory))) for name in averaging.get("of") or []]


def rows_it_was_trained_on(members: list[tuple[str, ladder_l4_rows.Stage, Path]]) -> list[dict]:
    """The rows the candidate's trainings RECORDED (each training loop's own record of row ids, read back to that
    training's stored set: by id and place, never a row's text), each with its training. A training whose record or
    set is not there gives no row."""
    rows = []
    for name, stage, directory in members:
        loss, training = directory / stage.loss_file(name), directory / stage.training_file(name)
        if not loss.exists() or not training.exists():
            continue
        by_id = {row["id"]: row for row in _rows(training)}
        rows += [{"training": name, **{key: (by_id.get(row_id) or {"id": row_id}).get(key) for key in ROW_KEYS}} for row_id in json.loads(loss.read_text()).get("rows_trained") or []]
    return rows


def read_the_candidate(config: dict, measure: Measure, measurement: dict, goal_ids: list[str], sampling: dict, smoke: bool) -> dict:
    """THE CANDIDATE's OWN RUN, ONLY READ. REFUSED (RuntimeError), naming the task to run: a run that is not there or not
    finished, of another seed, start or recipe, INCONCLUSIVE, not to be read, or measured otherwise than this run
    measures G's first sampling (`l5_measure.of_its_run`); one without the markers of its first step, of the candidate's
    averaging and of its measurement, or the candidate's rows on the first sampling; one without the marker, the stored
    set or the loss record of a training the averaged adapter is the mean of; a record of the mean that does not hold
    (`l5_measure.is_the_one_measured`); a stored first sampling that would not pair; and, ON A GPU BOX, an adapter that
    is not there, reads back at another rank, or is number for number `second_full`'s. Returns its directory, what its
    report says of its seed, the record of the mean, what was held of the adapter, the candidate's rows on the first
    sampling with what its set's own step recorded, what its measuring step wrote, and the rows its trainings recorded."""
    maker, seed, model = the_maker(config, measure), measure.seed, measure.model
    entry, directory, said = read_its_report(maker), maker.directory, maker.said
    why = of_its_run(entry, seed, measure.bar, MEAN_CONSTANT, measurement)
    if why:
        raise RuntimeError(f"{said}'s run of seed {seed} cannot carry L5m: {why}. Its `{model}` is the candidate: the adapter this stage measures, with its stored first "
                           f"sampling and the record of what it was trained on. Run {maker.task} to its end first; nothing was written.")
    first, bar = ladder_l5_full.the_first_start(config, measure.full), ladder_l5_full.the_start(config, measure.full)
    its_set = goal_set(sampling["name"], model, L4)
    needed = [*(f"{marker}.done.json" for marker in (maker.stage.prepare, maker.average(model), maker.stage.measure_marker(model))),
              f"episodes_{its_set}_problems.jsonl", f"episodes_{its_set}.done.json"]
    lacking = [name for name in needed if not (directory / name).exists()]
    if lacking:
        raise RuntimeError(f"{directory} does not hold {lacking}: the record of the averaged adapter `{model}` {said}'s run of seed {seed} made, of its measurement, or its "
                           f"rows on `{sampling['name']}`. Run {maker.task} to its end first; nothing was written.")
    averaging, measuring = _done(directory, maker.average(model)), _done(directory, maker.stage.measure_marker(model))
    members = its_members(config, measure, maker, averaging)
    lacking = [str(own / name) for member, stage, own in members
               for name in (f"{stage.train_marker(member)}.done.json", stage.training_file(member), stage.loss_file(member)) if not (own / name).exists()]
    if lacking or not members:
        raise RuntimeError(f"{said}'s run of seed {seed} recorded `{model}` as the mean of {[member for member, _, _ in members]}, and {lacking or 'no training'} is not "
                           f"there: the rows the candidate was trained on are read from each training's own record. Run {maker.task} to its end first; nothing was written.")
    trained = rows_it_was_trained_on(members)
    unknown = sorted({row["training"] for row in trained if row["problem_id"] is None})
    if unknown or not trained:
        raise RuntimeError(f"the trainings {unknown or [member for member, _, _ in members]} of {said}'s run of seed {seed} recorded "
                           + ("rows their stored sets do not hold" if unknown else "no row") + f": what `{model}` was trained on cannot be read from that run's own record. "
                           f"Queue {maker.task} again; nothing was written.")
    adapter = its_adapter(maker, measure)
    record = is_the_one_measured(averaging, measuring, str(adapter), first.check.rank, first.check.alpha, smoke=smoke)
    if not record["passes"]:
        raise RuntimeError(f"the candidate `{model}` of {said}'s run {directory.name} is not held by that run's own record: the averaged adapter is {record['averaged_as']} "
                           f"(the mean of its members: {record['is_the_mean']}; at rank {first.check.rank} and alpha {first.check.alpha}: {record['at_its_rank']}), its "
                           f"measurement was sampled from {record['measured_there_from']}, and this stage would serve {adapter}: they are one adapter, the mean of its members "
                           f"at `{first.name}`'s rank, or what is measured here is not the model that stage measured. Queue {maker.task} again; nothing was written.")
    against = averaging.get("against_the_start_adapter") or {}
    recorded = {"adapter_saved": averaging.get("adapter_saved"), "numbers": against.get("elements"), "numbers_not_its_start_adapters": against.get("elements_changed")}
    now = None
    if not ladder_ceiling._stand_in():     # on a GPU box (the trainings' own test of it): the files themselves, as every later step reads them
        if not (adapter / ADAPTER_FILE).exists() or not (adapter / ladder_l4_rank.ADAPTER_CONFIG).exists():
            raise RuntimeError(f"{adapter} is not there: the adapter `{model}` of {said}'s run of seed {seed} is the candidate, and that run keeps it. Queue {maker.task} again "
                               "(its step makes it again); nothing was written.")
        read_back, of_the_bar = ladder_l4_rank.adapter_read_back(adapter), compared(adapter / ADAPTER_FILE, bar.adapter / ADAPTER_FILE)
        now = {"there": True, "read_back": read_back, "against_the_bar": of_the_bar,
               "passes": ladder_l4_rank.is_the_checks(read_back, first.check) and of_the_bar["elements_changed"] > 0}
        if not now["passes"]:
            raise RuntimeError(f"the candidate's adapter at {adapter}, read back, has rank {read_back['rank']} and alpha {read_back['alpha']} with matrices of rank "
                               f"{read_back['ranks']} (`{first.name}`: {first.check.rank} and {first.check.alpha}), and {of_the_bar['elements_changed']:,} of its "
                               f"{of_the_bar['elements']:,} numbers are not `{measure.bar}`'s: it is at `{first.name}`'s rank and is not the bar's adapter, or nothing is "
                               "read of it against the bar. Nothing was written.")
    try:
        rows, of_the_set = ladder_ceiling._stored(directory, its_set, goal_ids, sampling["sampling_seed"], sampling["episodes"],
                                                  f"`{model}` ({sampling['name']})", READING)
    except RuntimeError as error:
        raise RuntimeError(f"{error} ({said}'s run of seed {seed}: the candidate's stored first sampling is what its measurement here is read beside). Nothing was "
                           "written.") from error
    return {"maker": maker, "directory": directory, "entry": entry, "adapter": adapter, "record": record, "members": members,
            "held": {"adapter": str(adapter), "recorded": recorded, "read_now": now}, "rows": rows,
            "of_the_set": {"sampling_seed": of_the_set["sampling_seed"], "episodes_each": of_the_set["episodes_each"]},
            "wrote": {"distinct_attempts": measuring.get("distinct_attempts")}, "trained": trained}


# ----------------------------------------------------------------------------------------------------- the bar
def read_the_bar(config: dict, measure: Measure, read: dict) -> dict:
    """`second_full`, ONLY READ: L5f's own reading of it (`ladder_l5_full.read_second_full`: REFUSED there when it is not
    there, at another rank, `pre_r64`'s number for number, or, on a GPU box, without its files); then its stored
    attempts on the three rungs and on G's two samplings in L4p's run, each HELD (`ladder_ceiling._stored`: REFUSED,
    RuntimeError) to the problems, the attempts a problem and the sampling seeds `pre_r64`'s run recorded, so that the
    candidate pairs with them by problem. Returns what the first step stores of it, its rows by part with what each
    set's own step recorded, and what its measuring step wrote."""
    plan, full = read["plan"], measure.full
    of_the_start = ladder_l5_full.read_second_full(config, full, read, plan["goal_samplings"][0])
    start = ladder_l5_full.the_start(config, full)
    directory, task = start.directory, ladder_l5_full.l4p_task(full)
    ids = {RUNG_PART: [row["problem_id"] for row in read["rung_problems"]], "goal": [row["problem_id"] for row in read["goal_problems"]]}
    measured = [(RUNG_PART, RUNG_PART, rung_set(measure.bar, L4), plan["rung_episodes"], plan["sampling_seeds"][RUNG_PART]),
                *((sampling["name"], "goal", goal_set(sampling["name"], measure.bar, L4), sampling["episodes"], sampling["sampling_seed"]) for sampling in plan["goal_samplings"])]
    lacking = ladder_ceiling._missing(directory, [set_name for _, _, set_name, _, _ in measured])
    if lacking:
        raise RuntimeError(f"{directory} does not hold {lacking}: the attempts of `{measure.bar}` on the rungs and on G's samplings, which L4p's run stored and this stage "
                           f"reads the candidate against. Run {task} to its end first; nothing was written.")
    rows, recorded = {}, {}
    for part, of, set_name, episodes, sampling_seed in measured:
        try:
            rows[part], of_the_set = ladder_ceiling._stored(directory, set_name, ids[of], sampling_seed, episodes, f"`{measure.bar}` ({part})", READING)
        except RuntimeError as error:
            raise RuntimeError(f"{error} (L4p's run: the bar's stored attempts are the other side of every read on the rungs and on G). Nothing was written.") from error
        recorded[part] = {"run": str(directory), "set": set_name, "sampling_seed": of_the_set["sampling_seed"], "episodes_each": of_the_set["episodes_each"]}
    wrote = _done(directory, OF_L4P.measure_marker(measure.bar))
    return {"start": start, "summary": of_the_start["summary"], "rows": rows, "recorded": recorded, "wrote": {"distinct_attempts": wrote.get("distinct_attempts")}}


def read_its_held_aside(config: dict, adapter: Path, aside_ids: list[str], sampling_seed: int, attempts: int) -> dict:
    """A STORED measurement of the bar on the held-aside set WITH THE SAME SETTING, when one is on the box: L6's run (its
    step `ladder_l6_held_aside_second_full`, by L6's own names), sampled from the same adapter, on the same problems,
    with the same attempts a problem and the same sampling seed, and answered by Lean. NOT THERE otherwise, with why:
    NOTHING IS REFUSED HERE, and the bar is then measured by this run's own step."""
    cycle = ladder_l6.the_settings(config)
    directory = _runs(config) / (os.environ.get(ladder_l6.RUN_VARIABLE) or ladder_l6.run_name(cycle, cycle.seed_of_the_search))
    set_name, marker = ladder_l5.held_aside_set(ladder_l6.STORED, ladder_l6.L6), ladder_l6.held_aside_marker(ladder_l6.STORED)
    found = {"run": str(directory), "set": set_name, "step": marker}
    lacking = [name for name in (f"{marker}.done.json",) if not (directory / name).exists()] + ladder_ceiling._missing(directory, [set_name])
    if lacking:
        return {**found, "there": False, "why": f"{directory.name} does not hold {lacking}"}
    served = _done(directory, marker).get("adapter")
    if served != str(adapter):
        return {**found, "there": False, "why": f"{directory.name} sampled that set from {served}, and the bar's adapter is {adapter}"}
    try:
        rows, recorded = ladder_ceiling._stored(directory, set_name, aside_ids, sampling_seed, attempts, f"`{BAR}` on the held-aside set", READING)
    except RuntimeError as error:
        return {**found, "there": False, "why": str(error)}
    return {**found, "there": True, "rows": rows, "sampling_seed": recorded["sampling_seed"], "episodes_each": recorded["episodes_each"]}


# ------------------------------------------------------------------------------------------------- prepare
def ladder_l5m_prepare(config: dict) -> dict:
    measure = the_settings(config)
    store = _store(config, measure)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    seed, deep, smoke = training_seed(config), measure.deep, smoke_run()
    try:
        the_seed(seed, measure.seed)
    except ValueError as error:
        raise RuntimeError(f"{error} ({SETTING}.candidate). Nothing was written.") from error
    if smoke and not os.environ.get(RUN_VARIABLE):
        raise RuntimeError(f"{SMOKE_VARIABLE} is set: it is for a SMOKE run alone, in a run directory the task names ({RUN_VARIABLE}) and on the fixtures' base map. A run in "
                           "the candidate's own directory reads a run of L5 whose checks passed. Nothing was written.")
    # ---- L5's finished run (the held-aside set) and every stored run its task read, by L5's OWN first-step reading, as a further seed of L5 reads them (refused there)
    reader = the_reader(measure, smoke)
    found = ladder_l5.read_the_search(config, deep, reader)
    # A smoke run reads a SMOKE run of L5: its two looser draws are made again as that run recorded them (L5's own smoke-only variable is L5's smoke stage's alone).
    read = ladder_l5.read_for_the_search(config, deep, found, reader, at_least=((found["searched"].get("smoke") or {}).get("problems_at_least") or 0) if smoke else 0)
    first, the_map, plan, recorded, at_least = (read[key] for key in ("start", "the_map", "plan", "recorded", "at_least"))
    if smoke and not the_map["fixture"]:
        raise RuntimeError(f"{SMOKE_VARIABLE} is set: it is for a SMOKE run alone, in a run directory the task names ({RUN_VARIABLE}) and on the fixtures' base map. A run on "
                           "the pool's problems reads a run of L5 whose checks passed. Nothing was written.")
    directory, samplings = found["directory"], plan["goal_samplings"]
    # THE SMOKE WORLD HAS ONE SAMPLING OF G: the smoke `pre_r64`'s run reads L1's smoke run alone (no stored run of L2 beside it), so it recorded G's first sampling and no
    # second, and every smoke run after it measured its models on that one. A SMOKE run goes on with it, as L5's own smoke run does, and makes NO read on fresh attempts,
    # which is RECORDED and said by the report. A real run is refused, as it was.
    one_sampling = smoke and len(samplings) == 1
    if len(samplings) != 2 and not one_sampling:
        raise RuntimeError(f"the run of `{first.name}` ({first.directory.name}) recorded {len(samplings)} samplings of the goal set ({[own['name'] for own in samplings]}): this "
                           "stage reads its first (the one the night's models were compared on) and its second (fresh attempts). Nothing was written.")
    sampling, fresh = samplings[0], None if one_sampling else samplings[1]
    goal_ids, rung_ids = [row["problem_id"] for row in read["goal_problems"]], [row["problem_id"] for row in read["rung_problems"]]
    aside, aside_ids = read["aside"], [row["problem_id"] for row in read["aside"]]
    measurement = {"sampling": sampling["name"], "episodes": sampling["episodes"], "sampling_seed": sampling["sampling_seed"], "problems": len(goal_ids)}

    # ---- the bar, where L4p's run keeps it, with its stored rungs and samplings of G; and the candidate, in the run of the stage that made it (both refused here)
    of_the_bar = read_the_bar(config, measure, read)
    if (of_the_bar["summary"]["recorded_by_l4p"]["rank"], of_the_bar["summary"]["recorded_by_l4p"]["alpha"]) != (recorded["rank"], recorded["alpha"]):
        raise RuntimeError(f"L4p's run recorded rank {of_the_bar['summary']['recorded_by_l4p']['rank']} and alpha {of_the_bar['summary']['recorded_by_l4p']['alpha']} and the run "
                           f"of `{first.name}` {recorded}: both models here are served at the first start's rank. Nothing was written.")
    of_it = read_the_candidate(config, measure, measurement, goal_ids, sampling, smoke)
    maker, bar = of_it["maker"], of_the_bar["start"]
    try:        # no row the candidate was trained on is held out, of G or of the held-aside set: that stage's OWN record of its trainings, read before anything is sampled
        heldout = {row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[0]} | {row["problem_id"] for row in read["groups"]}
        barred = refuse_barred(of_it["trained"], heldout, config["ladder_loop"]["l4"]["half_seed"], set(the_map["base_map"]), f"the rows `{measure.model}` was trained on")
        beside = refuse_problems_of(of_it["trained"], set(goal_ids), set(aside_ids), f"the rows `{measure.model}` was trained on")
    except ValueError as error:
        raise RuntimeError(f"{error} ({maker.said}'s run {of_it['directory'].name}). Nothing was written.") from error
    attempts, aside_sampling = deep.held_aside_attempts, read["aside_sampling"]
    stored = read_its_held_aside(config, bar.adapter, aside_ids, aside_sampling, attempts)
    sampled = [CANDIDATE] if stored["there"] else [CANDIDATE, BAR]        # the models THIS run samples on the held-aside set

    # ---- from here on files are written: the problems of every set this run samples, the held-out groups, G', the held-aside problems, the stored rows, the trained rows
    def as_set(own: list[dict], set_name: str) -> list[dict]:
        return [{**row, "set": set_name} for row in own]

    by_id = {row["problem_id"]: row for row in the_map["problems"]}
    problems = as_set(read["rung_problems"], rung_set(CANDIDATE, L4))
    problems += [row for own in samplings for row in as_set(read["goal_problems"], goal_set(own["name"], CANDIDATE, L4))]
    problems += [row for who in sampled for row in as_set([by_id[problem_id] for problem_id in aside_ids], ladder_l5.held_aside_set(who, L5M))]
    store.write_rows("problems.jsonl", problems)
    store.write_rows(ladder_l4.GROUPS_FILE, read["groups"])
    store.write_rows(ladder_l4.AGAIN_FILE, [{"problem_id": problem_id} for problem_id in read["again"]])
    store.write_rows(ladder_l5.HELD_ASIDE_FILE, aside)
    for (who, part), own_rows in read["kept"].items():
        store.write_rows(stored_file(who, part, L4), own_rows)
    for part, own_rows in of_the_bar["rows"].items():
        store.write_rows(stored_file(measure.bar, part, L4), own_rows)
    store.write_rows(stored_file(CANDIDATE, sampling["name"], L4), of_it["rows"])
    if stored["there"]:
        store.write_rows(stored_file(measure.bar, ladder_l5.HELD_ASIDE_PART, L5M), stored["rows"])
    store.write_rows(TRAINED_FILE, of_it["trained"])
    report, parts = found["report"], [RUNG_PART, *(own["name"] for own in samplings)]
    block = ladder_l5.block_of(config, attempts)
    held_aside_sets = {who: ladder_l5.held_aside_set(who, L5M) for who in sampled}
    the_stored_sets = {      # what each stored set's OWN step recorded (its sampling seed, its attempts a problem), as the first step held it: the report holds them again
        "what": "the stored sets this run reads, each with the run that holds it and what its own step recorded: held to the setting before anything was written",
        "the_bar": {part: {"file": stored_file(measure.bar, part, L4), **of_the_bar["recorded"][part]} for part in parts},
        "the_candidates_first_sampling": {"file": stored_file(CANDIDATE, sampling["name"], L4), "run": str(of_it["directory"]),
                                          "set": goal_set(sampling["name"], measure.model, L4), **of_it["of_the_set"]},
        "the_bars_held_aside": {"file": stored_file(measure.bar, ladder_l5.HELD_ASIDE_PART, L5M), "run": stored["run"], "set": stored["set"],
                                "sampling_seed": stored["sampling_seed"], "episodes_each": stored["episodes_each"]} if stored["there"] else None}
    summary = {
        "stage": L4, "label": LABEL, "check": "L5m", "seed": seed, "bar": measure.bar, "bar_adapter": str(bar.adapter), "bar_run": str(bar.directory),
        "first_start": first.name, "first_start_adapter": str(first.adapter), "first_start_run": str(first.directory), "stand_in_engine": _stand_in(),
        "the_candidate": {
            "what": f"THE CANDIDATE: the stored averaged adapter `{measure.model}` of {maker.said}'s run of seed {measure.seed}, ONLY READ: no step of this stage trains, "
                    "averages or writes an adapter. Held before anything was written: its run is finished and to be read, its own record says the adapter is the mean of "
                    "its members at the rank and the one that stage measured, and (on a GPU box) the files are there, at the rank, and not the bar's",
            "of": measure.of, "made_by": maker.said, "seed": measure.seed, "model": measure.model, "run": str(of_it["directory"]), "task": maker.task,
            "adapter": str(of_it["adapter"]), "the_mean_of": [name for name, _, _ in of_it["members"]],
            "kept_by_another_run": {name: str(own) for name, _, own in of_it["members"] if own != of_it["directory"]},
            "held": of_it["held"], "the_record_of_the_mean": of_it["record"],
            "at_the_first_step": {key: of_it["entry"].get(key) for key in ("seed", "start", "recipe", "ok", "inconclusive", "failed_checks", "the_measurement")},
            "wrote": of_it["wrote"],
            "trained_rows": {"what": "the rows its trainings recorded (each training loop's own record of row ids, read back to that training's stored set): by id and "
                                     "place, never with text. Refused before anything was written when one is held out, of G or of the held-aside set",
                             "file": TRAINED_FILE, "rows": len(of_it["trained"]), "trainings": len(of_it["members"]), "barred": barred,
                             "problems_of_g_or_of_the_held_aside_set": {key: beside[key] for key in ("problems_of_the_goal_set", "problems_of_the_held_aside_set")}}},
        "the_bar": {"what": f"THE BAR: `{measure.bar}`, the model the candidate was trained from, ONLY READ. Its attempts on the rungs and on G's two samplings are the ones "
                            "L4p's run stored (held to the setting before anything was written); its adapter is served for the held-aside set alone, and only when no "
                            "stored measurement of it with the same setting is on the box",
                    "summary": of_the_bar["summary"], "wrote": of_the_bar["wrote"]},
        "the_search": {"what": f"L5's finished search of seed {deep.seed_of_the_search}, ONLY READ: its held-aside set is the primary's, with the sampling seed its own models "
                               "were measured with there. Nothing of it is sampled again", "run": str(directory), "seed": deep.seed_of_the_search,
                       "headline": report.get("headline"), "inconclusive": bool(report.get("inconclusive")), "held_aside_problems": len(aside_ids),
                       "held_aside_file": ladder_l5.HELD_ASIDE_FILE},
        "the_measurement": {"what": "the candidate is measured as L5's models are: 8 episodes on each rung problem, G's two samplings, the held-aside set; each with the "
                                    f"sampling seed the stored models were measured with (`{first.name}`'s run recorded the rungs' and G's; the search's run the held-aside "
                                    "set's), so that it pairs by problem with the bar. ITS FIRST SAMPLING IS MEASURED AGAIN HERE, not copied",
                            "first_sampling": measurement,
                            "fresh_sampling": {"sampling": fresh["name"], "episodes": fresh["episodes"], "sampling_seed": fresh["sampling_seed"]} if fresh else None,
                            "sets": {RUNG_PART: rung_set(CANDIDATE, L4), **{own["name"]: goal_set(own["name"], CANDIDATE, L4) for own in samplings},
                                     ladder_l5.HELD_ASIDE_PART: ladder_l5.held_aside_set(CANDIDATE, L5M)},
                            "stored": {**{part: stored_file(measure.bar, part, L4) for part in parts}, "first_start": {part: stored_file(first.name, part, L4) for part in parts},
                                       "the_candidates_first_sampling": stored_file(CANDIDATE, sampling["name"], L4)}},
        "the_held_aside": {"what": "the held-aside set L5's search drew: the two models make as many fresh attempts at each of its problems as L5's four made, with that set's "
                                   "ONE sampling seed, so that they pair by problem", "problems": len(aside_ids), "episodes_each": attempts, "sampling_seed": aside_sampling,
                           "block_problems": block, "file": ladder_l5.HELD_ASIDE_FILE, "sets": held_aside_sets, "sampled_here": sampled},
        "the_bars_held_aside": {
            "what": f"`{measure.bar}` on the held-aside set: " + (
                f"READ FROM A STORED RUN ({stored['run']}, its step `{stored['step']}`), sampled from the same adapter with the same problems, attempts and sampling seed: "
                "the bar's step here makes nothing" if stored["there"] else
                f"MEASURED HERE from the adapter L4p's run keeps: no stored measurement of it with the same setting is on the box ({stored['why']})"),
            "read_from_a_stored_run": stored["there"], "looked_in": stored["run"], "step_there": stored["step"], "set_there": stored["set"], "why_not": stored.get("why"),
            "file": stored_file(measure.bar, ladder_l5.HELD_ASIDE_PART, L5M) if stored["there"] else None,
            "set": None if stored["there"] else ladder_l5.held_aside_set(BAR, L5M)},
        "the_stored_sets": the_stored_sets,
        "trainings": {"run": True, "why": None, "what": "nothing is trained here: this is what L5's held-aside body asks of a first step before it samples"},
        **{key: plan[key] for key in ladder_l5.PLAN},
        "goal_set": len(goal_ids), "goal_set_again": len(read["again"]), "rung_problems": len(rung_ids),
        "smoke": {"what": f"{SMOKE_VARIABLE}: a smoke run alone. L5's smoke run reads INCONCLUSIVE (a training of two rows has no loss to compare), which a real run is refused "
                          "for, and is read all the same; its two looser draws are made again as that run recorded them; and a candidate whose averaged adapter is one of its "
                          "members (a smoke run's trainings are one optimizer step on the same rows) is measured all the same, and said. Where the smoke runs "
                          f"recorded ONE sampling of G (the smoke `{first.name}`'s run reads L1's smoke run alone), the candidate is measured on that one and NO read is made "
                          "on fresh attempts; a real run is refused for it",
                  "problems_at_least": at_least, "l5_read_though_inconclusive": bool(report.get("inconclusive")), "l5s_failed_checks": list(report.get("failed_checks") or []),
                  "one_sampling_of_g": one_sampling}
        if smoke else None,
        "contradicted_side_setting": config["ladder_loop"]["episode"].get("contradicted_side", "all"),
        "start_recipe": found["searched"]["start_recipe"], "start_run_read": found["searched"].get("start_run_read")}
    store.mark_done(PREPARE, summary)
    _say(f"L5m prepared, seed {seed}: THE CANDIDATE is `{measure.model}` of {maker.said}'s run {of_it['directory'].name} ({of_it['adapter']}), the mean of "
         f"{len(of_it['members'])} trainings on {len(of_it['trained']):,} recorded rows; THE BAR is `{measure.bar}` ({bar.adapter}). The candidate is measured on "
         f"{len(rung_ids)} rung problems ({plan['rung_episodes']} episodes, sampling seed {plan['sampling_seeds'][RUNG_PART]}), on {len(goal_ids)} goal problems "
         + (f"twice (`{sampling['name']}` {sampling['episodes']} attempts, seed {sampling['sampling_seed']}; `{fresh['name']}` {fresh['episodes']}, seed {fresh['sampling_seed']})"
            if fresh else f"ONCE (`{sampling['name']}` {sampling['episodes']} attempts, seed {sampling['sampling_seed']}: the smoke runs recorded one sampling of G, and no "
                          "read is made on fresh attempts)") + " and on "
         f"the {len(aside_ids)} held-aside problems ({attempts} attempts, seed {aside_sampling}); the bar on the held-aside set "
         + (f"is READ from {Path(stored['run']).name}" if stored["there"] else "is MEASURED HERE") + ("; A SMOKE RUN" if smoke else ""))
    return summary


def _of_this_run(store: ArtifactStore, measure: Measure, what: str) -> dict:
    """The prepare step's summary, for a later step of THIS run: a run keeps its candidate (RuntimeError when the
    setting names another one now, which only a run directory the task names can meet)."""
    _need(store, PREPARE, what, ENTRY_STAGE)
    prepared = store.done_summary(PREPARE)
    then, now = tuple(prepared["the_candidate"][key] for key in ("of", "seed", "model")), (measure.of, measure.seed, measure.model)
    if then != now:
        raise RuntimeError(f"this run ({store.root.name}) was prepared for the candidate {then} and {SETTING}.candidate now names {now}: a run keeps its candidate. Move the "
                           "run away, or name another run directory, to measure another one. Nothing was sampled")
    return prepared


# ------------------------------------------------------------------------------------------------- measure
def ladder_l5m_measure(config: dict) -> dict:
    """THE CANDIDATE on the three held-out rungs and on G's TWO samplings, by the body L5's models are measured with
    (`ladder_ceiling.measure_model`, with the rungs): the attempts and the sampling seeds the first start's run recorded,
    at the config's own temperature, from the adapter the run that made it keeps (only read: it needs no step of this
    run but the first). The model server is started with the start adapter's rank as its largest (`config_from`)."""
    measure = the_settings(config)
    store = _store(config, measure)
    prepared = _of_this_run(store, measure, "the measurement of the candidate")
    of_it, adapter = prepared["the_candidate"], Path(prepared["the_candidate"]["adapter"])
    first, fresh = prepared["the_measurement"]["first_sampling"], prepared["the_measurement"]["fresh_sampling"]
    return measure_model(config_from(config, prepared, ladder_l5_full.the_start(config, measure.full), serving=True), store, prepared, Model(
        name=CANDIDATE, called=f"L5m's candidate (`{of_it['model']}` of {of_it['made_by']})", whose="L5m's candidate", marker=MEASURE, trained_by=PREPARE, adapter=adapter,
        no_adapter=f"it is the adapter `{of_it['model']}` of {of_it['made_by']}'s run, which that run keeps: queue {of_it['task']} again. The measurement cannot be made",
        detail=f"the rungs, {prepared['rung_episodes']} episodes a problem; G's `{first['sampling']}`, {first['episodes']} attempts, sampling seed {first['sampling_seed']}; "
               + (f"G's `{fresh['sampling']}`, {fresh['episodes']} attempts, sampling seed {fresh['sampling_seed']}" if fresh else
                  "NO second sampling of G (a smoke run: the smoke runs recorded one)"),
        summary={"stage": L4, "label": LABEL, "check": "L5m", "model": CANDIDATE, "candidate": {key: of_it[key] for key in ("of", "seed", "model")}, "adapter": str(adapter)},
        stage=ENTRY_STAGE, label=L4, also=what_else_a_model_wrote))


def ladder_l5m_held_aside(config: dict, who: str) -> dict:
    """One of the two models on the held-aside set, BY L5's OWN BODY (`ladder_l5.on_the_held_aside_set`) under this
    stage's names: as many fresh attempts at each held-aside problem as L5's four models made, with that set's ONE
    sampling seed, served at the first start's rank. The candidate from the adapter the run that made it keeps; the bar
    from the one L4p's run keeps, UNLESS the first step found a stored measurement of it with the same setting: the
    step then makes nothing and marks nothing, and says where the rows are read from."""
    measure = the_settings(config)
    store = _store(config, measure)
    prepared = _of_this_run(store, measure, f"the measurement of `{WHO[who]}` on the held-aside set")
    adapter = Path(prepared["the_candidate"]["adapter"]) if who == CANDIDATE else Path(prepared["bar_adapter"])

    def read_elsewhere(of_this_run: dict) -> dict | None:
        of_the_bar = of_this_run["the_bars_held_aside"]
        if who != BAR or not of_the_bar["read_from_a_stored_run"]:
            return None
        why = (f"`{BAR}` on the held-aside set is READ from {of_the_bar['looked_in']} (its step `{of_the_bar['step_there']}`: the same adapter, problems, attempts and sampling "
               f"seed), copied by the first step as {of_the_bar['file']}: nothing is sampled here")
        _say(f"L5m: {why}")
        return {"stage": L4, "label": LABEL, "check": "L5m", "run": False, "why": why}

    return ladder_l5.on_the_held_aside_set(config, store, measure.draws.start, WHO[who], held_aside_marker(who), measure.deep.held_aside_attempts,
                                           ladder_l5.held_aside_seed(config, measure.deep), trained_by=None, directory=adapter, not_made=read_elsewhere, one=THE_STAGE)


# -------------------------------------------------------------------------------------------------- report
def _recorded(store: ArtifactStore, set_name: str) -> dict:
    """What a set THIS run sampled recorded of itself (its own step's marker): its sampling seed and its attempts."""
    path = store.path(f"episodes_{set_name}.done.json")
    own = json.loads(path.read_text()) if path.exists() else {}
    return {"sampling_seed": own.get("sampling_seed"), "episodes_each": own.get("episodes_each")}


def ladder_l5m_report(config: dict) -> dict:
    from rlvr_lean.gpu.ladder_l3c import proof_lengths
    from rlvr_lean.reporting.ladder_l5_measure import build_l5_measure_report

    measure = the_settings(config)
    store = _store(config, measure)
    prepared = _of_this_run(store, measure, "L5m's report")
    of_the_bars, sets, of_the_set = prepared["the_bars_held_aside"], prepared["the_measurement"]["sets"], prepared["the_held_aside"]
    steps = [MEASURE, held_aside_marker(CANDIDATE), *([] if of_the_bars["read_from_a_stored_run"] else [held_aside_marker(BAR)])]
    for step in steps:
        _need(store, step, "L5m's report", ENTRY_STAGE)
    samplings = [own["name"] for own in prepared["goal_samplings"]]
    parts = [RUNG_PART, *samplings]
    aside = ladder_l5.HELD_ASIDE_PART
    candidate = {**{part: _results(store, sets[part]) for part in parts}, aside: _results(store, sets[aside])}
    bar = {**{part: store.read_rows(stored_file(measure.bar, part, L4)) for part in parts},
           aside: store.read_rows(of_the_bars["file"]) if of_the_bars["read_from_a_stored_run"] else _results(store, of_the_bars["set"])}
    first_start = {part: store.read_rows(stored_file(prepared["first_start"], part, L4)) for part in parts}
    recorded = {**{part: _recorded(store, sets[part]) for part in parts}, f"{aside}_{CANDIDATE}": _recorded(store, sets[aside]),
                **({} if of_the_bars["read_from_a_stored_run"] else {f"{aside}_{BAR}": _recorded(store, of_the_bars["set"])})}
    on_the_set = {CANDIDATE: store.done_summary(held_aside_marker(CANDIDATE)),
                  BAR: None if of_the_bars["read_from_a_stored_run"] else store.done_summary(held_aside_marker(BAR))}
    # THE CANDIDATE's RUN IS READ AGAIN, AS IT IS NOW: what its report says of its seed, and its own record of the mean and of its measurement.
    maker = the_maker(config, measure)
    now = {"entry": read_its_report(maker), "averaging": _done(maker.directory, maker.average(measure.model)),
           "measuring": _done(maker.directory, maker.stage.measure_marker(measure.model))}
    heldout = sorted({row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[0]} | {row["problem_id"] for row in store.read_rows(ladder_l4.GROUPS_FILE)})
    base_map = [row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[1]]
    ids_of = lambda name: [row["problem_id"] for row in store.read_rows(name)]      # noqa: E731
    of_this_run = store.read_rows("problems.jsonl")         # the problems of each set, as the first step wrote them: what "the setting's problems" are held to
    problems = {part: [row["problem_id"] for row in of_this_run if row["set"] == sets[part]] for part in (*parts, aside)}
    # The length of a held-out problem's published proof is read HERE, by the report, and by nothing before it.
    report = build_l5_measure_report(prepared, store.done_summary(MEASURE), on_the_set, candidate, bar, first_start,
                                     store.read_rows(prepared["the_measurement"]["stored"]["the_candidates_first_sampling"]), recorded, problems,
                                     store.read_rows(ladder_l4.GROUPS_FILE), proof_lengths(), ids_of(ladder_l4.AGAIN_FILE), store.read_rows(TRAINED_FILE), heldout,
                                     config["ladder_loop"]["l4"]["half_seed"], base_map, config["evaluation"], now,
                                     {"attempts": measure.deep.held_aside_attempts, "sampling_seed": ladder_l5.held_aside_seed(config, measure.deep),
                                      "problems": of_the_set["problems"]})
    report["adapters"] = {"kept": True, "made_here": [], "what": f"NO ADAPTER IS MADE OR DELETED HERE ({LABEL}): the candidate's is its own run's "
                                                                 f"({prepared['the_candidate']['adapter']}) and `{measure.bar}`'s L4p's ({prepared['bar_adapter']}); both were "
                                                                 "only read"}
    _write_json(store, REPORT_FILE, report)
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    for line in report["lines"]:
        print(line, flush=True)
    store.mark_done(REPORT, {"stage": L4, "label": LABEL, "check": "L5m", "seed": prepared["seed"], "candidate": {key: prepared["the_candidate"][key] for key in ("of", "seed", "model")},
                             "headline": report["headline"], "branches": report["branches"], "measured": [CANDIDATE, *([] if of_the_bars["read_from_a_stored_run"] else [BAR])],
                             "ok": report["ok"]})
    return report


def _for(step, name: str):
    def run(config: dict) -> dict:
        return step(config, name)
    return run


assert set(MODELS_OF) == {L5F, L5C}     # the two stages `the_maker` knows are the ones a candidate may be of
# In the section's order: the first step; the candidate in full (the rungs and G's two samplings, then the held-aside set); the bar on the held-aside set; the report.
STEPS = {PREPARE: ladder_l5m_prepare, MEASURE: ladder_l5m_measure, held_aside_marker(CANDIDATE): _for(ladder_l5m_held_aside, CANDIDATE),
         held_aside_marker(BAR): _for(ladder_l5m_held_aside, BAR), REPORT: ladder_l5m_report}
