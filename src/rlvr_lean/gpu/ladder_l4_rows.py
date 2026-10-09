"""L4t: what should a round train on? Three one-change checks on the rounds already made. Spec:
docs/spec/ladder-loop.spec.md, "L4t: what should a round train on? Three one-change checks on the rounds already
made" and its "Made exact by the build". Everything it makes is labelled pretrained on published proofs, as everything
built on `pre` is. Each step is its own process (`rlvr_lean.runner.entry`, stage `ladder_l4_rows`).

NO NEW ROUND. Two runs are only READ, on the box: the arm's (`ladder_l2_<arm_run>_seed<N>`: its rounds' training
examples, each batch's picks with their k of n, the twin's rows, the adapter `with`, and the per-problem rows of `with`
and `without`) and `pre`'s (`ladder_l2.pretraining_run`: the start adapter, its sampling seeds, its per-problem rows, G',
the rows it was trained on). The run directory is its own (`ladder_l4_rows_seed<N>`; with the start adapter's name in
it when the start is not `pre`).

  ladder_l4_rows_prepare               REFUSES, naming the task to run, a run that lacks what is read, an arm's run of
                                       another seed, arm or start than the setting's, stored rows that would not pair by
                                       problem, and a training set with a barred problem. BUILDS and stores the two
                                       training sets (`domain/ladder_round/l4_rows.py`). No GPU, no Lean
  ladder_l4_rows_train_rehearse        FROM THE START ADAPTER, one pass, the arm's recipe (`ladder_ceiling.one_pass`, as
  ladder_l4_rows_train_reward_rows     the arm's twin is trained): the adapter kept under `adapters/<model>`, every row's
                                       loss, the training loop's own record of the rows, and the adapter it saved compared
                                       with the start adapter's
  ladder_l4_rows_measure_rehearse      the model as `with` was measured (`ladder_ceiling.measure_model`): 8 episodes on the
  ladder_l4_rows_measure_reward_rows   three rungs, G's samplings, with `pre`'s sampling seeds as its run recorded them
  ladder_l4_rows_measure_hot_with      NO TRAINING: `with` (the arm's last model) and `pre`, each sampled again on G
  ladder_l4_rows_measure_hot_pre       alone at `temperature_hot`, with sampling seeds of their own. `hot_config` is the
                                       ONE place a temperature is put into a config, and only these two steps call it
  ladder_l4_rows_report                the read fixed before the run (`reporting/ladder_l4_rows.py`): each trained model
                                       by itself (its checks, the primary, the branch), then the table of `hot`

Every step is resumable and a step of its own: a failure of a later one keeps the earlier ones. A report in which Lean
gave no verdict on too much of a set fails its step, and the task queued again measures that set, and no other, again
from the kept adapter. NO STEP DELETES AN ADAPTER.

THE REHEARSAL ROWS ARE PUBLISHED PROOFS. Their text stays in the pretraining file (`data/ladder_l4/pretraining.jsonl`,
not distributed): the stored training set names each by its id, its problem and its place in the file, and the training
step reads the text again from the file, held to the SHA-256 `pre`'s prepare step recorded.

ANOTHER START THAN `pre` (spec, "L4r's first branch was taken: what L4t runs"). The start is a setting that names a
pretrained model and where it lives (`the_start`): `pre`, in the pretraining's run, or a model of the check of the
adapter's rank (`pre_r64`), in that check's run (`ladder_l4_rank`'s own functions give its directory and its adapter).
A stage names another start than the setting's by `RLVR_LEAN_LADDER_L4_ROWS_START` (`ladder_l4_rows_r64`). From such a
start the run directory carries its name (`ladder_l4_rows_pre_r64_seed<N>`) and:

  the rank       every training attaches an adapter of the START ADAPTER's rank and alpha, as the check's run recorded
                 them (never `config["lora"]`), the adapter each saves is read back and held to them before its step is
                 marked done, and the model server of this stage's measure steps is started with that rank as its
                 largest (`config_from`, on `ladder_l4_rank.config_of`). From `pre` the config is handed on as it is
  the rows       are still the arm's, made from `pre` (`rows_made_by`): the arm's run must have started from `pre`
  old_rule       a THIRD trained model, FIRST: the twin's rows exactly, in the twin's order, from the start adapter. It
                 stands in `without`'s place in every read, and the start model in `pre`'s; its rows are measured here
  G'             is the START MODEL's, made here by the pretraining's rule (`l4.goal_set_again`) from its stored first
                 sampling, and stored in this run
  hot            `old_rule` and the start model (`ladder_l4_rows_measure_hot_old_rule`, `..._hot_start`)
  beside         the stored `without` and `pre` (rank 16), read when their rows are on the box and pair with this run's
                 problems and seeds; said to be not there otherwise, and nothing is refused for them
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from rlvr_lean.domain.ladder_round.assembly import ATTEMPT, training_set
from rlvr_lean.domain.ladder_round.ceiling import the_training_took
from rlvr_lean.domain.ladder_round.l3d import WITH, WITHOUT, tenth
from rlvr_lean.domain.ladder_round.l4 import PRE, goal_set_again
from rlvr_lean.domain.ladder_round.l4_rows import (
    HOT,
    MODELS,
    OLD_RULE,
    REHEARSE,
    REWARD_ROWS,
    kept_rows,
    models_of,
    reference_of,
    refuse_barred,
    rehearsal_rows,
    rehearse_set,
    reward_draws,
    with_k,
)
from rlvr_lean.domain.ladder_round.read import GOAL
from rlvr_lean.domain.repair.accumulate import proof_line_count
from rlvr_lean.gpu import ladder_assembly, ladder_ceiling, ladder_l2, ladder_l3d2, ladder_l4, ladder_l4_start, ladder_loop, pipeline
from rlvr_lean.gpu.ladder_ceiling import RUNG_PART, Model, Reading, _need, goal_set, measure_model, one_pass, rung_set, stored_file
from rlvr_lean.gpu.ladder_dose import _rows, _runs
from rlvr_lean.gpu.ladder_l3d1 import WROTE, what_else_a_model_wrote
from rlvr_lean.gpu.ladder_l4 import L4, LABEL, _say
from rlvr_lean.gpu.ladder_l4_start import Start, config_from, start_set, the_adapter_saved
from rlvr_lean.gpu.ladder_round import _results, _stand_in, _write_json, training_seed
from rlvr_lean.infrastructure.artifact_store import ArtifactStore

L4T = "l4_rows"                                         # what this stage's own files are called, beside L4's (`l4_...`)
STAGE = "ladder_l4_rows"
RUN_VARIABLE = "RLVR_LEAN_LADDER_L4_ROWS_RUN"           # another run directory than `ladder_l4_rows_seed<seed>` (a smoke run)
ARM_RUN_VARIABLE = "RLVR_LEAN_LADDER_L4_ROWS_ARM_RUN"   # another run directory of the arm than `ladder_l2_<arm_run>_seed<seed>` (a smoke run: the arm's smoke run)
MINIMUM_VARIABLE = "RLVR_LEAN_LADDER_L4_ROWS_MINIMUM"   # a smoke run: `reward_rows` holds at least this many rows (the rule keeps none of a fixture's rows)
START_VARIABLE = "RLVR_LEAN_LADDER_L4_ROWS_START"       # another start than the setting's: the stage `ladder_l4_rows_r64` names `pre_r64`
START_CHECKS_VARIABLE = "RLVR_LEAN_LADDER_L4_ROWS_START_CHECKS"     # `smoke` (a smoke run alone): the run that made the start model may have failed ITS checks (twelve rows have no loss to compare)
THE_START = "start"                                     # the start model in the name of ITS hot step, from another start than `pre` (its sets carry its own name)
SETTING = "ladder_loop.l4.rows"
PREPARE, REPORT = f"{STAGE}_prepare", f"{STAGE}_report"
REPORT_FILE = "report_ladder_l4_rows.json"
STORED_MODELS_FILE = f"{L4T}_stored_models.json"        # what `pre`, `with` and `without` wrote, from their own runs' markers
DRAWS_FILE = f"{L4T}_reward_draws.jsonl"                # every row of the rounds with its k, its reward and its draw, kept or not (no proof text)
NEVER_SOLVED_FILE = f"{L4T}_never_solved.jsonl"         # GIVEN, not made here: one `problem_id` a row, the goal problems nothing stored has ever solved
STORED = (PRE, WITH, WITHOUT)                           # the three stored models every read here is against
HOT_OF = (WITH, PRE)                                    # the two models sampled again at the other temperature, in the order they are
READING = Reading(reader="L4t", models="L4t's models", not_read="that set stands on the other side of a comparison, and no report of this run could be read against it",
                  setting=ladder_l4.SETTING)
MAX_LEFTOVER_GB = pipeline.MAX_LEFTOVER_GB


@dataclass(frozen=True)
class Rows:
    """`ladder_loop.l4.rows`, as the stage's steps read it."""
    start: str                          # the start adapter every model here is trained from: `pre`, or the one the stage names (`pre_r64`)
    arm: str                            # the arm whose stored rounds are read
    temperature: float                  # the two hot measurements' sampling temperature
    hot_seeds: tuple[int, ...]          # their sampling seeds, one for each sampling of G
    models: tuple[str, ...]             # the trained models, in the order they are trained and measured: two from `pre`, three from another start


def the_settings(config: dict) -> Rows:
    """THE one place that reads `ladder_loop.l4.rows`. The start is the setting's, or the one the stage names
    (`RLVR_LEAN_LADDER_L4_ROWS_START`); the models are `models` from `pre` and `models_from_another_start` from any
    other. Refused (ValueError): models that are not the stage's for that start (its steps are registered for them:
    change both together); hot sampling seeds that are not two distinct whole numbers; a hot temperature that is not a
    positive number other than the config's own."""
    settings = config["ladder_loop"]["l4"]["rows"]
    start = os.environ.get(START_VARIABLE) or settings["start"]
    named = "models" if start == PRE else "models_from_another_start"
    models, seeds, temperature = tuple(settings[named]), tuple(settings["sampling_seeds_hot"]), settings["temperature_hot"]
    if models != models_of(start):
        raise ValueError(f"{SETTING}.{named} is {list(models)} and the stage trains and measures {list(models_of(start))}: change both together")
    if len(seeds) != 2 or len(set(seeds)) != 2 or not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
        raise ValueError(f"{SETTING}.sampling_seeds_hot is {list(seeds)}: two distinct whole numbers, one for each sampling of G")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature <= 0 or temperature == config["sampling"]["temperature"]:
        raise ValueError(f"{SETTING}.temperature_hot is {temperature!r}: a positive number other than sampling.temperature ({config['sampling']['temperature']}), "
                         "which every other measurement keeps")
    return Rows(start=start, arm=settings["arm_run"], temperature=float(temperature), hot_seeds=seeds, models=models)


def hot_config(config: dict, rows: Rows) -> dict:
    """The config as the two hot measurements hand it on, and THE one place a temperature is put into one: the sampling
    temperature is `temperature_hot`. Nothing else moves, and `config` itself is not changed: every step that does not
    call this samples at the config's own."""
    return {**config, "sampling": {**config["sampling"], "temperature": rows.temperature}}


def run_name(rows: Rows, seed: int) -> str:
    """The stage's run directory: `ladder_l4_rows_seed<N>` from `pre`, and with the start adapter's name in it from any
    other (a later run from `pre_r64` is another directory)."""
    return f"{STAGE}_seed{seed}" if rows.start == PRE else f"{STAGE}_{rows.start}_seed{seed}"


def _store(config: dict, rows: Rows) -> ArtifactStore:
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    return ArtifactStore(_runs(config) / (os.environ.get(RUN_VARIABLE) or run_name(rows, training_seed(config))), Path(mirror) if mirror else None)


def arm_directory(config: dict, rows: Rows) -> Path:
    """The arm's run, whose stored rounds are read (or the run the task names: a smoke run). Only read."""
    return _runs(config) / (os.environ.get(ARM_RUN_VARIABLE) or f"ladder_l2_{rows.arm}_seed{training_seed(config)}")


def pre_directory(config: dict) -> Path:
    """`pre`'s run: the ONE pretraining's (`ladder_l2.pretraining_run`). Only read."""
    return _runs(config) / ladder_l2.pretraining_run(config)


def the_start(config: dict, rows: Rows) -> Start:
    """The start model the settings name, and where it lives (`ladder_l4_start.the_start`: `pre` in the ONE pretraining's run,
    a model of the check of the adapter's rank in that check's run). A name that is neither is refused (RuntimeError)."""
    return ladder_l4_start.the_start(config, rows.start, f"{SETTING}.start (or {START_VARIABLE})")


def start_adapter(config: dict, rows: Rows) -> Path:
    """The adapter every model here is trained FROM: the start model's, in the run that made it."""
    return the_start(config, rows).adapter


def hot_of(rows: Rows) -> tuple[tuple[str, str], ...]:
    """The two models sampled again at the other temperature, in the order they are, each as (what its step calls it,
    its name): `with` and `pre` from `pre`; from another start `old_rule` and the start model, whose step calls it
    `start` (a step's name is registered without a config)."""
    return ((WITH, WITH), (PRE, PRE)) if rows.start == PRE else ((OLD_RULE, OLD_RULE), (THE_START, rows.start))


def steps_of(start: str) -> list[str]:
    """The steps of a run from `start`, in the order its stage runs them: each model trained, then measured, before the
    next is trained; then the two hot measurements; then the report."""
    hot = (WITH, PRE) if start == PRE else (OLD_RULE, THE_START)
    return [PREPARE, *(step(model) for model in models_of(start) for step in (train_marker, measure_marker)), *(measure_marker(hot_name(who)) for who in hot), REPORT]


def _of_this_start(rows: Rows, step: str) -> None:
    if step not in steps_of(rows.start):
        raise RuntimeError(f"the step {step} is not one of a run from `{rows.start}` ({SETTING}.start, or {START_VARIABLE}): its steps are {steps_of(rows.start)}. "
                           "Nothing was written")


def minimum_rows() -> int:
    """A smoke run's least number of rows in `reward_rows`; 0 for every other run: the rule and nothing else."""
    return max(0, int(os.environ.get(MINIMUM_VARIABLE) or 0))


def training_file(model: str) -> str:
    return f"{L4T}_training_{model}.jsonl"


def loss_file(model: str) -> str:
    return f"{L4T}_loss_{model}.json"


def train_marker(model: str) -> str:
    return f"{STAGE}_train_{model}"


def measure_marker(model: str) -> str:
    return f"{STAGE}_measure_{model}"


def hot_name(who: str) -> str:
    """A model sampled at the other temperature, as its sets and its step name it: never the name its sets at the
    config's temperature have."""
    return f"{HOT}_{who}"


def arm_set(part: str, model: str) -> str:
    """The set a model of the arm was measured under, in the arm's own run directory."""
    return f"{ladder_l3d2.L3D2}_{part}_{model}"


def lines_counted(rows: list[dict]) -> dict[str, int]:
    """A training set's rows by the number of lines of their proofs (as a model's verified proofs are counted)."""
    counts: dict[int, int] = {}
    for row in rows:
        counts[row["lines"]] = counts.get(row["lines"], 0) + 1
    return {str(lines): counts[lines] for lines in sorted(counts)}


def described(rows: list[dict], batch: int) -> dict:
    """What is recorded of a training set: its rows by origin and by k, the lines of their proofs, its optimizer steps."""
    from rlvr_lean.reporting.ladder_ceiling import verified_lines

    def count_by(key) -> dict:
        found = [key(row) for row in rows if key(row) is not None]      # a rehearsal row has no k: it is counted by its origin
        return {str(name): found.count(name) for name in sorted(set(found))}

    return {"rows": len(rows), "rows_by_origin": count_by(lambda row: row["origin"]), "rows_by_k": count_by(lambda row: row.get("k")),
            "lines": verified_lines(lines_counted(rows)), "rows_by_lines": lines_counted(rows), "optimizer_steps": -(-len(rows) // batch)}


# ------------------------------------------------------------------------------------------------- prepare
def build_the_sets(by_round: dict, picks: list[dict], twin: list[dict], pretraining: list[dict], target_rate: float, seed: int, at_least: int = 0) -> dict:
    """The two training sets and every row's draw, from what the arm's run and `pre`'s stored. `by_round`: the rounds'
    training examples. `picks`: every batch's per-problem results. `twin`: the rows of the twin's file, in its order.
    `pretraining`: the rows `pre` was trained on, in the file's order (`problem_id`, `lines`, and what else is kept of
    one; no text). Refused (ValueError) when the stored files do not agree: the twin's rows are not the one-shot rows
    of the last model's set, a row has no k, or the rule keeps no row. Every row gets its `lines`. With them, `old_rule`
    (a run from another start than `pre` trains it; from `pre` it is made and not written): the twin's rows exactly, in
    the twin's own order, each with its k."""
    last = max(by_round)
    of_the_rounds = with_k([{**row, "lines": proof_line_count(row["completion"])} for row in training_set(by_round, last)], picks)
    one_shot = {row["id"]: row for row in of_the_rounds if row["origin"] == ATTEMPT}
    if sorted(row["id"] for row in twin) != sorted(one_shot):
        raise ValueError(f"the twin's file holds {len(twin)} rows and the rounds' training examples {len(one_shot)} one-shot rows, and they are not the same rows: "
                         "the stored run does not agree with itself. Refused")
    draws = reward_draws(of_the_rounds, target_rate, seed)
    kept = kept_rows(draws, seed, at_least)
    if not kept:
        raise ValueError(f"the rule keeps none of the rounds' {len(draws)} rows (each is kept when its draw is under its problem's reward): there is nothing to train "
                         f"`{REWARD_ROWS}` on. Refused")
    of_the_twin = [one_shot[row["id"]] for row in twin]         # the twin's rows, in its order, each with its k and its lines
    rehearse = rehearse_set(of_the_twin, rehearsal_rows(pretraining, len(of_the_twin), seed), seed)
    numbered = lambda rows: [{"row": position, **{key: value for key, value in row.items() if key != "row"}} for position, row in enumerate(rows)]      # noqa: E731
    return {REHEARSE: numbered(rehearse), REWARD_ROWS: numbered(kept), OLD_RULE: numbered(of_the_twin),
            "draws": [{key: row[key] for key in ("id", "problem_id", "side", "origin", "round", "batch", "k", "n", "lines", "reward", "draw", "kept")} for row in draws]}


def _lacking(directory: Path, names) -> list[str]:
    return [name for name in names if not (directory / name).exists()]


def read_rank_16(arm: Path, pre: Path, goal_ids: list[str], plan: dict) -> tuple[dict, dict, dict]:
    """For a run from another start than `pre`: what stands BESIDE `old_rule`'s own read, the stored `without` against
    `pre` (the old rule at rank 16), from the arm's and `pre`'s runs WHERE THEY ARE ON THE BOX and pair by problem with
    this run (G's samplings: the same problems, seeds and attempts, and Lean having answered). Returns (their rows by
    (who, sampling), what each wrote, what is recorded of the read). Nothing is refused for them: rows that are not
    there, or would not pair, are said to be not there, with the reason."""
    kept = {}
    try:
        for sampling in plan["goal_samplings"]:
            for who, directory, set_name in ((PRE, pre, ladder_l4.pre_set(sampling["name"])), (WITHOUT, arm, arm_set(sampling["name"], WITHOUT))):
                kept[who, sampling["name"]] = ladder_ceiling._stored(directory, set_name, goal_ids, sampling["sampling_seed"], sampling["episodes"],
                                                                     f"`{who}` ({sampling['name']})", READING)[0]
        markers = {PRE: pre / f"{ladder_l4.MEASURE}.done.json", WITHOUT: arm / f"{ladder_l3d2.measure_marker(WITHOUT)}.done.json"}
        wrote = {who: {key: json.loads(path.read_text())[key] for key in WROTE} for who, path in markers.items()}
    except (RuntimeError, OSError, KeyError, ValueError) as error:
        return {}, {}, {"read": False, "why": f"{type(error).__name__}: {error}"}
    return kept, wrote, {"read": True, "what": "the stored `without` and `pre` on G (the old rule at rank 16): they stand beside `old_rule`'s own read and decide nothing",
                         "runs": {PRE: str(pre), WITHOUT: str(arm)}, "sets": {who: {name: stored_file(who, name, L4) for (other, name) in kept if other == who} for who in (PRE, WITHOUT)}}


def ladder_l4_rows_prepare(config: dict) -> dict:
    rows = the_settings(config)
    store = _store(config, rows)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    seed, of_the_pretraining = training_seed(config), ladder_l2.pretraining_seed(config)
    arm, pre = arm_directory(config, rows), pre_directory(config)
    start = the_start(config, rows)
    from_pre = start.check is None                  # from another start, the start model is the check of the adapter's rank's, and `old_rule` stands in `without`'s place
    arm_task = (f"the task of stage `{ladder_l4.ARM_STAGE}` for seed {seed} (`python -m rlvr_lean.runner.entry --stage {ladder_l4.ARM_STAGE} --seeds {seed}`; for the smoke run, "
                f"stage `{ladder_l4.ARM_STAGE}_smoke`)")
    pre_task = (f"the task of stage `{ladder_l4.PRETRAIN_STAGE}` for seed {of_the_pretraining} (`python -m rlvr_lean.runner.entry --stage {ladder_l4.PRETRAIN_STAGE} --seeds {of_the_pretraining}`; "
                f"for the smoke run, stage `{ladder_l4.PRETRAIN_STAGE}_smoke`)")

    def need(directory: Path, names, what: str, task: str) -> None:
        lacking = _lacking(directory, names)
        if lacking:
            raise RuntimeError(f"{directory} does not hold {lacking}: {what}. L4t makes no round and no pretraining: it reads what {task} wrote on this box. Run that "
                               "task to its end first; nothing was written.")

    # ---- the arm's run: what its steps recorded (the two measured models' markers only from `pre`, where their rows stand on the other side)
    need(arm, (f"{ladder_l2.PREPARE}.done.json", f"{ladder_l3d2.PREPARE}.done.json", f"{ladder_l3d2.TRAIN_WITHOUT}.done.json",
               *(f"{ladder_l3d2.measure_marker(model)}.done.json" for model in ((WITH, WITHOUT) if from_pre else ())), "problems.jsonl", ladder_l3d2.GROUPS_FILE,
               ladder_l3d2.TRAINING_WITHOUT_FILE), "what the arm's steps recorded, its problems and held-out groups, the twin's rows", arm_task)
    of_the_arm = json.loads((arm / f"{ladder_l2.PREPARE}.done.json").read_text())
    # THE ROWS ARE THE ARM'S, MADE FROM `pre`, whatever this run starts from: from `pre` that is also the start the setting names.
    if of_the_arm.get("seed") != seed or of_the_arm.get("arm") != rows.arm or of_the_arm.get("start") != PRE:
        raise RuntimeError(f"the arm's run {arm.name} recorded seed {of_the_arm.get('seed')}, the arm {of_the_arm.get('arm')!r} and the start {of_the_arm.get('start')!r}; "
                           + (f"this task's seed is {seed} and {SETTING} names the arm {rows.arm!r} and the start {rows.start!r}: every model here is trained from the "
                              "adapter the arm's own models were, on the rounds that arm made. Nothing was written." if from_pre else
                              f"this task's seed is {seed} and {SETTING} names the arm {rows.arm!r}, whose rounds were made from `{PRE}`: from `{rows.start}` the rows "
                              f"are still the rounds that arm made from `{PRE}`. Nothing was written."))
    recorded_start = Path(of_the_arm.get("start_adapter") or "")
    if (recorded_start.parent.parent.name, recorded_start.name) != (pre.name, PRE):
        raise RuntimeError(f"the arm's run {arm.name} was trained from {recorded_start} and "
                           + (f"this task's start adapter is {start.adapter}" if from_pre else f"the rounds read here are those made from {pre / ladder_assembly.ADAPTERS / PRE}")
                           + ": not the same adapter. Nothing was written.")
    rounds, batches, last = list(of_the_arm["rounds"]), of_the_arm["batches"], of_the_arm["rounds"][-1]
    batch_sets = [ladder_l2.batch_set(number, batch) for number in rounds for batch in range(1, batches + 1)]
    need(arm, (*(f"training_examples_r{number}.jsonl" for number in rounds), *(f"episodes_{name}_problems.jsonl" for name in batch_sets),
               *(f"{ladder_assembly.assembly_marker(number, batch)}.done.json" for number in rounds for batch in range(1, batches + 1))),
         "a round's training examples, a batch's picks with their k of n, or the marker of a batch's assembly", arm_task)
    # ---- `pre`'s run: its record, its report, G', the rows it was trained on (from another start: its record and those rows, among which the rehearsal is drawn)
    if from_pre:
        need(pre, (f"{ladder_l4.PREPARE}.done.json", f"{ladder_l4.MEASURE}.done.json", ladder_l4.REPORT_FILE, ladder_l4.AGAIN_FILE, ladder_l4.ROWS_FILE),
             "what `pre`'s steps recorded, its report, G' or the rows it was trained on", pre_task)
    else:
        need(pre, (f"{ladder_l4.PREPARE}.done.json", ladder_l4.ROWS_FILE), "what `pre`'s prepare step recorded, or the rows it was trained on (the rehearsal is drawn among them)",
             pre_task)
    of_pre = json.loads((pre / f"{ladder_l4.PREPARE}.done.json").read_text())
    if of_pre.get("seed") != of_the_pretraining:
        raise RuntimeError(f"the pretraining run {pre.name} was made at seed {of_pre.get('seed')} and ladder_loop.l4.pretraining_seed is {of_the_pretraining}: every model "
                           "here starts from the ONE pretraining of that seed. Nothing was written.")
    if from_pre:
        report = json.loads((pre / ladder_l4.REPORT_FILE).read_text())
        if not report.get("ok", True):
            raise RuntimeError(f"the report of the pretraining run {pre.name} is not to be read (Lean gave no verdict on too much of a set): `pre`'s rows cannot stand on "
                               f"the other side of a comparison. Queue {pre_task} again, which measures that set again; nothing was written.")
        of_start, of_the_start_run = of_pre, None
    else:
        of_the_start_run = ladder_l4_start.read_the_start_run(start, of_the_pretraining, of_pre, waived=os.environ.get(START_CHECKS_VARIABLE) == "smoke")
        of_start = of_the_start_run["prepared"]
    parts = ladder_l4.pre_parts(of_start)
    need(start.directory, [name for part in parts for name in (f"episodes_{start_set(part, start)}_problems.jsonl", f"episodes_{start_set(part, start)}.done.json")],
         f"`{start.name}`'s per-problem rows on a set it was measured on, or that set's own marker", start.task)
    if from_pre:
        need(arm, [name for part in parts for model in (WITH, WITHOUT) for name in (f"episodes_{arm_set(part, model)}_problems.jsonl", f"episodes_{arm_set(part, model)}.done.json")],
             "the per-problem rows of `with` or of `without` on a set `pre` was measured on, or that set's own marker", arm_task)
    # ---- the adapters read here (the stand-in trains and serves none): the start model's, and from `pre` the arm's last model's, which is sampled again
    adapters = {rows.start: start.adapter, **({WITH: arm / ladder_assembly.ADAPTERS / f"m{last}"} if from_pre else {})}
    if not ladder_ceiling._stand_in():
        for name, directory, task in ((rows.start, adapters[rows.start], start.task), *(((WITH, adapters[WITH], arm_task),) if from_pre else ())):
            if not directory.is_dir():
                raise RuntimeError(f"{directory} is not there: the adapter `{name}` is what " + ("every model here is trained from" if name == rows.start else
                                   "the arm's last model is, and it is sampled again here") + f". Run {task} again; nothing was written.")

    # ---- how every model here is measured: as the start model was (`pre`'s sampling seeds); from `pre`, `with` and `without` must have been measured so too
    plan = {key: of_start[key] for key in ("sampling_seeds", "rung_episodes", "goal_samplings")}
    problems = _rows(arm / "problems.jsonl")
    rung_problems = [row for row in problems if row["set"] == arm_set(RUNG_PART, WITH)]
    goal_problems = [row for row in problems if row["set"] == arm_set(plan["goal_samplings"][0]["name"], WITH)]
    ids = {RUNG_PART: [row["problem_id"] for row in rung_problems], GOAL: [row["problem_id"] for row in goal_problems]}
    measured = [(RUNG_PART, RUNG_PART, plan["rung_episodes"], plan["sampling_seeds"][RUNG_PART]),
                *((sampling["name"], GOAL, sampling["episodes"], sampling["sampling_seed"]) for sampling in plan["goal_samplings"])]
    kept = {}
    for part, of, episodes, sampling_seed in measured:
        kept[rows.start, part] = ladder_ceiling._stored(start.directory, start_set(part, start), ids[of], sampling_seed, episodes, f"`{rows.start}` ({part})", READING)[0]
        for model in ((WITH, WITHOUT) if from_pre else ()):
            kept[model, part] = ladder_ceiling._stored(arm, arm_set(part, model), ids[of], sampling_seed, episodes, f"`{model}` ({part})", READING)[0]
    if from_pre:
        again = [row["problem_id"] for row in _rows(pre / ladder_l4.AGAIN_FILE)]
        stray = sorted(set(again) - set(ids[GOAL]))
        if stray:
            raise RuntimeError(f"{len(stray)} problems of G' ({pre.name}) are not goal problems of the arm's run (first: {stray[0]}): the two runs are not of one held-out set. "
                               "Nothing was written.")
    else:       # G' is the START MODEL's, and its own run stored none: made here by the pretraining's rule, from its first sampling of G
        again = goal_set_again(kept[rows.start, plan["goal_samplings"][0]["name"]], ids[GOAL])
    # ---- the hot measurements' sampling seeds are their own
    used = {**{f"`{rows.start}`'s {name}": value for name, value in plan["sampling_seeds"].items()}, **{f"the arm's {name}": value for name, value in of_the_arm["sampling_seeds"].items()}}
    taken = sorted(name for name, value in used.items() if value in rows.hot_seeds)
    if taken:
        raise RuntimeError(f"{SETTING}.sampling_seeds_hot is {list(rows.hot_seeds)} and {taken} already sampled with one of them: the hot measurements have sampling "
                           "seeds of their own. Nothing was written.")
    hot = [{"name": sampling["name"], "episodes": sampling["episodes"], "sampling_seed": hot_seed} for sampling, hot_seed in zip(plan["goal_samplings"], rows.hot_seeds)]

    # ---- the rows: the rounds', the twin's, and the pretraining file's (its text is read, and stays in the file)
    by_round = {number: _rows(arm / f"training_examples_r{number}.jsonl") for number in rounds}
    picks = [row for name in batch_sets for row in _rows(arm / f"episodes_{name}_problems.jsonl")]
    twin = _rows(arm / ladder_l3d2.TRAINING_WITHOUT_FILE)
    pretraining, sha256 = ladder_l4_start.pretrained_on(pre, of_pre["pretraining_file_sha256"])      # refused unless the pretraining file is the one `pre` was trained on
    batch, target_rate = config["training"]["effective_batch"], of_the_arm["target_rate"]
    try:
        sets = build_the_sets(by_round, picks, twin, pretraining, target_rate, seed, minimum_rows())
        groups = _rows(arm / ladder_l3d2.GROUPS_FILE)
        heldout = {row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[0]} | {row["problem_id"] for row in groups}
        half_seed = config["ladder_loop"]["l4"]["half_seed"]
        counted = {model: refuse_barred(sets[model], heldout, half_seed, f"the training set of `{model}`") for model in rows.models}
    except ValueError as error:
        raise RuntimeError(f"{error} (the arm's run {arm.name}; `pre`'s {pre.name}). Nothing was written.") from error
    rank_16_rows, rank_16_wrote, rank_16 = ({}, {}, None) if from_pre else read_rank_16(arm, pre, ids[GOAL], plan)

    # ---- from here on files are written: the problems of every set this run samples, what its report reads of the stored models, the training sets
    def as_set(own: list[dict], set_name: str) -> list[dict]:
        return [{**row, "set": set_name} for row in own]

    sampled_again = [name for _, name in hot_of(rows)]
    own = [row for model in rows.models for row in as_set(rung_problems, rung_set(model, L4))]
    own += [row for model in rows.models for sampling in plan["goal_samplings"] for row in as_set(goal_problems, goal_set(sampling["name"], model, L4))]
    own += [row for who in sampled_again for sampling in hot for row in as_set(goal_problems, goal_set(sampling["name"], hot_name(who), L4))]
    store.write_rows("problems.jsonl", own)             # every set of it is this run's
    store.write_rows(ladder_l4.GROUPS_FILE, groups)
    store.write_rows(ladder_l4.AGAIN_FILE, [{"problem_id": problem_id} for problem_id in again])
    for (who, part), own_rows in {**kept, **rank_16_rows}.items():
        store.write_rows(stored_file(who, part, L4), own_rows)
    if from_pre:
        wrote = {PRE: json.loads((pre / f"{ladder_l4.MEASURE}.done.json").read_text()),
                 **{model: json.loads((arm / f"{ladder_l3d2.measure_marker(model)}.done.json").read_text()) for model in (WITH, WITHOUT)}}
    else:
        wrote = {rows.start: of_the_start_run["measured"], **rank_16_wrote}
    _write_json(store, STORED_MODELS_FILE, {"stage": L4, "label": LABEL, **{who: {key: entry[key] for key in WROTE} for who, entry in wrote.items()}})
    store.write_rows(DRAWS_FILE, sets["draws"])
    for model in rows.models:
        store.write_rows(training_file(model), sets[model])
    by_rule = sum(row["kept"] for row in sets["draws"])
    # The adapter every training here attaches: the config's own from `pre`; from another start the START ADAPTER's rank and alpha, as its run recorded them.
    recorded = None if from_pre else {"rank": of_start["recipe"]["lora"]["rank"], "alpha": of_start["recipe"]["lora"]["alpha"]}
    recipe_of_the_start = config_from(config, {"start_recipe": recorded}, start)["lora"]
    described_sets = {
        REHEARSE: {"what": "the twin's one-shot rows and as many rows of the pretraining file (a draw by a content hash of the task's seed and each row's id among the "
                           "rows `pre` was trained on, no row twice), in ONE order, the arm's content hash. The rehearsal rows are published proofs of the `pretrain` "
                           "half: their text stays in the pretraining file", "file": training_file(REHEARSE), **described(sets[REHEARSE], batch),
                   "twin_rows": len(twin), "rehearsal_rows": len(twin), "rows_pre_was_trained_on": len(pretraining), "barred": counted.get(REHEARSE)},
        REWARD_ROWS: {"what": "each row of the rounds (one-shot and assembled) kept when a number in [0, 1) made from a content hash of the task's seed and the row's "
                              "id is under r(k), the challenger's reward at the arm's target rate for its problem's k of n (an assembled row: k = 1); no quota and no "
                              "screen; in the arm's content-hash order", "file": training_file(REWARD_ROWS), **described(sets[REWARD_ROWS], batch),
                      "rows_of_the_rounds": len(sets["draws"]), "kept_by_the_rule": by_rule, "kept_by_the_runs_minimum": len(sets[REWARD_ROWS]) - by_rule,
                      "minimum_of_a_smoke_run": minimum_rows(), "draws_file": DRAWS_FILE, "barred": counted.get(REWARD_ROWS)},
        OLD_RULE: {"what": "the twin's one-shot rows exactly, in the twin's own order (the arm's content hash with the other rows left out): what `without` is, trained "
                           "from this run's start adapter. The old rule at the start adapter's rank", "file": training_file(OLD_RULE), **described(sets[OLD_RULE], batch),
                   "twin_rows": len(twin), "twin_file": ladder_l3d2.TRAINING_WITHOUT_FILE, "barred": counted.get(OLD_RULE)}}
    summary = {
        "stage": L4, "label": LABEL, "check": "L4t", "seed": seed, "start": rows.start, "start_adapter": str(adapters[rows.start]), "arm": rows.arm, "arm_run": str(arm),
        "pretraining_run": str(pre), "with_adapter": str(adapters[WITH]) if from_pre else None, "rounds": rounds, "batches": batches, "solvers": of_the_arm.get("solvers"),
        "target_rate": target_rate, "models": list(rows.models), "stand_in_engine": _stand_in(), "effective_batch": batch,
        "training_sets": {model: described_sets[model] for model in rows.models},
        "pretraining_file": str(ladder_l4.pretraining_file()), "pretraining_file_sha256": sha256,
        "recipe": {"what": "the arm's (`ladder_ceiling.one_pass`, as its twin is trained): FROM THE START ADAPTER, one pass, the native format, the content-hash order",
                   "learning_rate": config["training"]["learning_rate"], "warmup_steps": config["training"]["warmup_steps"], "effective_batch": batch,
                   "lora": recipe_of_the_start, "adapter_seed": seed},
        HOT: {"what": f"NO TRAINING: `{sampled_again[0]}` and `{sampled_again[1]}` each sampled again on G alone at temperature {rows.temperature:g} (every other measurement: "
                      f"{float(config['sampling']['temperature'])!r}), with sampling seeds of their own", "temperature": rows.temperature,
              "temperature_of_every_other_measurement": config["sampling"]["temperature"], "models": sampled_again, "samplings": hot,
              "attempts_a_goal_problem": sum(sampling["episodes"] for sampling in hot)},
        "goal_set": len(goal_problems), "goal_set_again": len(again), "rung_problems": len(rung_problems), **plan,
        "attempts_a_goal_problem": sum(sampling["episodes"] for sampling in plan["goal_samplings"]), "parts": parts,
        "stored_models": {who: {"run": str(pre if who == PRE else arm), "sets": {part: ladder_l4.pre_set(part) if who == PRE else arm_set(part, who) for part in parts}}
                          for who in STORED} if from_pre else {rows.start: {"run": str(start.directory), "sets": {part: start_set(part, start) for part in parts}}},
        "never_solved_file": NEVER_SOLVED_FILE, "contradicted_side_setting": config["ladder_loop"]["episode"].get("contradicted_side", "all")}
    if not from_pre:        # what a run from another start than `pre` records beside: from `pre` the summary is what it always was
        of_the_check = of_start["rank_check"]
        summary.update({
            "reference": reference_of(rows.start),
            "start_run": str(start.directory),
            "start_recipe": {"what": f"the adapter of `{rows.start}`, as the run that made it recorded it: EVERY training here attaches this rank and alpha (not the config's "
                                     f"own, {config['lora']['rank']} and {config['lora']['alpha']}), the adapter each saves is read back and held to them, and the model "
                                     "server of this stage's measure steps is started with this rank as its largest",
                             **recorded, "check": of_the_check.get("check"), "stage": of_the_check.get("stage"),
                             "trained_parameters": of_the_check.get("trained_parameters"), "adapter_saved": of_the_start_run["trained"].get("adapter_saved"),
                             "rank_of_the_config": config["lora"]["rank"], "alpha_of_the_config": config["lora"]["alpha"]},
            "start_run_read": {"branch": (of_the_start_run["report"].get("branch") or {}).get("name"), "headline": of_the_start_run["report"].get("headline"),
                               "pretraining_file_sha256": of_start["pretraining_file_sha256"], "rows": of_start["rows"],
                               "checks_failed": of_the_start_run["checks_failed"], "checks_waived_for_a_smoke_run": of_the_start_run["checks_waived"]},
            "rows_made_by": {"what": f"the rows are the rounds of the arm {rows.arm}, attempted and trained from `{PRE}` (rank {of_pre['recipe']['lora']['rank']}): each "
                                     f"proof was written by a model built on `{PRE}`, and each row's k is that model's. This is the training rule on given rows at the "
                                     f"start adapter's rank, not the rounds `{rows.start}` would have made", "start": of_the_arm.get("start"),
                             "start_adapter": str(recorded_start), "rank": of_pre["recipe"]["lora"]["rank"], "alpha": of_pre["recipe"]["lora"]["alpha"]},
            "goal_set_again_made_here": {"what": f"G' of `{rows.start}`: the goal problems it does not solve in its FIRST sampling of G ({plan['goal_samplings'][0]['episodes']} "
                                                 "attempts), by the pretraining's rule (`l4.goal_set_again`) from its stored rows; the run that made it stored none",
                                         "problems": len(again), "solved_in_the_first_sampling": len(goal_problems) - len(again), "file": ladder_l4.AGAIN_FILE},
            "rank_16": rank_16})
    store.mark_done(PREPARE, summary)
    of = summary["training_sets"]
    _say(f"L4t prepared, seed {seed}, from `{rows.start}` on the rounds of {arm.name}: "
         + ("" if from_pre else f"`{OLD_RULE}` {of[OLD_RULE]['rows']:,} rows (the twin's, in its order), {of[OLD_RULE]['optimizer_steps']:,} optimizer steps; ")
         + f"`{REHEARSE}` {of[REHEARSE]['rows']:,} rows ({len(twin):,} of the twin and as many "
         f"of the pretraining file), {of[REHEARSE]['optimizer_steps']:,} optimizer steps; `{REWARD_ROWS}` {of[REWARD_ROWS]['rows']:,} of the rounds' "
         f"{len(sets['draws']):,} rows, {of[REWARD_ROWS]['optimizer_steps']:,} optimizer steps; `{HOT}` at temperature {rows.temperature:g}"
         + ("" if from_pre else f"; every model at rank {summary['start_recipe']['rank']}; G' of `{rows.start}` holds {len(again)} goal problems"))
    return summary


# --------------------------------------------------------------------------------------------------- train
def ladder_l4_rows_train(config: dict, model: str) -> dict:
    """One trained model: from the start adapter, one pass over its stored training set in its order, the arm's recipe
    (`one_pass(start=...)`, as the arm's twin is trained). The adapter is kept under `adapters/<model>`. From another
    start than `pre` the adapter attached has the START ADAPTER's rank and alpha (`config_from`), and the one saved is
    read back and held to them before the step is marked done (`the_adapter_saved`)."""
    rows = the_settings(config)
    store, marker = _store(config, rows), train_marker(model)
    _of_this_start(rows, marker)
    if store.is_done(marker):
        return store.done_summary(marker)
    _need(store, PREPARE, f"the training of `{model}`", STAGE)
    prepared, seed, batch = store.done_summary(PREPARE), training_seed(config), config["training"]["effective_batch"]
    of_the_start = the_start(config, rows)
    start = of_the_start.adapter
    if not ladder_ceiling._stand_in() and not start.is_dir():
        raise RuntimeError(f"{start} is not there: the adapter `{rows.start}` is what `{model}` is trained from. Nothing was trained")
    set_rows = store.read_rows(training_file(model))
    examples, steps = ladder_l4_start.examples_of(set_rows, prepared["pretraining_file_sha256"]), -(-len(set_rows) // batch)
    _say(f"L4t: training `{model}` from the stored adapter `{rows.start}` on {len(set_rows):,} rows in "
         + ("the twin's order" if model == OLD_RULE else "the content-hash order") + f", {steps:,} optimizer steps")
    result, step_rows, row_losses, positions = one_pass(config_from(config, prepared, of_the_start), examples, {model: steps}, seed, batch, store.root / ladder_assembly.ADAPTERS,
                                                        f"{run_name(rows, seed)}_{model}", positions=True, start=start)
    result.pop("checkpoints")
    _write_json(store, loss_file(model), {
        "stage": L4, "label": LABEL, "model": model, "seed": seed, "stand_in_engine": _stand_in(), "rows": len(row_losses), "steps": result["steps"], "effective_batch": batch,
        "what": "row_losses: each training row's mean loss per target token, in the order trained, read BEFORE the update of the optimizer step it was in. "
                "rows_trained: the id of the row each of those losses is of, from the positions the training loop recorded step by step",
        "row_losses": row_losses, "rows_trained": [set_rows[position]["id"] for position in positions],
        "training_steps": [{**row, "rows_seen": min(row["step"] * batch, len(row_losses))} for row in step_rows]})
    took = the_training_took(row_losses, tenth(len(row_losses), config["ladder_loop"]["l4"]["loss_share_of_rows"]))
    summary = {"stage": L4, "label": LABEL, "model": model, "seed": seed, "trained_from": f"the stored adapter {start}", "rows": len(set_rows),
               "rows_by_origin": prepared["training_sets"][model]["rows_by_origin"], "passes": 1,
               "order": "the twin's own (the content hash of the task's seed and each row's id, the other rows left out)" if model == OLD_RULE else
                        "a content hash of the task's seed and each row's id (`assembly.training_order`)", "adapter": str(store.root / ladder_assembly.ADAPTERS / model),
               "first_step_loss": step_rows[0]["mean_loss"], "last_step_loss": step_rows[-1]["mean_loss"], "mean_loss_over_the_first_rows": took["first"],
               "mean_loss_over_the_last_rows": took["last"], "rows_compared": took["rows_compared"], "stand_in_engine": _stand_in(), **result}
    summary.update(the_adapter_saved(store.root / ladder_assembly.ADAPTERS / model, model, of_the_start))      # refused, and the step not marked done, at another rank than the start adapter's
    store.mark_done(marker, summary)        # the adapter is saved, and kept
    if summary.get("allocated_after_cleanup_gb", 0) > MAX_LEFTOVER_GB:
        raise RuntimeError(f"{summary['allocated_after_cleanup_gb']} GB is still allocated on the GPU after the training of `{model}` was released")
    return summary


# ------------------------------------------------------------------------------------------------- measure
def ladder_l4_rows_measure(config: dict, model: str) -> dict:
    """One trained model as `with` was measured (`measure_model`): the rungs and G's samplings, with the sampling
    seeds `pre`'s run recorded, at the config's own temperature. From another start than `pre` the model server is
    started with the start adapter's rank as its largest (`config_from`: this stage's measure steps alone)."""
    rows = the_settings(config)
    store = _store(config, rows)
    _of_this_start(rows, measure_marker(model))
    _need(store, PREPARE, f"the measurement of L4t's model `{model}`", STAGE)
    prepared = store.done_summary(PREPARE)
    return measure_model(config_from(config, prepared, the_start(config, rows), serving=True), store, prepared, Model(
        name=model, called=f"L4t's model `{model}`", whose=f"L4t's model `{model}`", marker=measure_marker(model), trained_by=train_marker(model),
        adapter=store.root / ladder_assembly.ADAPTERS / model,
        no_adapter=f"the adapter `{model}` is kept by this stage, and one trained again would not be the model its other measurements came from. The measurement cannot be made",
        detail=f"{prepared['training_sets'][model]['rows']:,} rows, from `{prepared['start']}`", summary={"stage": L4, "label": LABEL, "model": model},
        stage=STAGE, label=L4, also=what_else_a_model_wrote))


def ladder_l4_rows_measure_hot(config: dict, who: str) -> dict:
    """`with` or `pre` sampled AGAIN on G alone at `temperature_hot` (`hot_config`: this step's config and no other's),
    with the hot sampling seeds, from its stored adapter. Nothing is trained; no rung is sampled. Its sets carry
    `hot` in their names, and the summary says the temperature. From another start than `pre` the two are `old_rule`
    (the adapter this run trained and kept) and the start model (`who` is then `start`: its sets carry its own name)."""
    rows = the_settings(config)
    store, marker = _store(config, rows), measure_marker(hot_name(who))
    _of_this_start(rows, marker)
    who = dict(hot_of(rows))[who]               # the model: its own name, whatever its step calls it
    _need(store, PREPARE, f"the measurement of `{who}` at temperature {rows.temperature:g}", STAGE)
    prepared = store.done_summary(PREPARE)
    hot = prepared[HOT]
    kept_here = who == OLD_RULE                 # trained by this run; the others are their own runs' adapters, only read
    adapter = store.root / ladder_assembly.ADAPTERS / who if kept_here else Path(prepared["with_adapter"] if who == WITH else prepared["start_adapter"])
    config = config_from(config, prepared, the_start(config, rows), serving=True)       # from `pre`: the config itself
    return measure_model(hot_config(config, rows), store, {"goal_samplings": hot["samplings"]}, Model(
        name=hot_name(who), called=f"`{who}` at temperature {rows.temperature:g}", whose=f"`{who}`, sampled again at temperature {rows.temperature:g}",
        marker=marker, trained_by=train_marker(who) if kept_here else PREPARE, adapter=adapter,
        no_adapter=(f"the adapter `{who}` is kept by this stage, and one trained again would not be the model its other measurements came from. The measurement cannot be made"
                    if kept_here else f"the adapter of `{who}` is its own run's, only read here. The measurement cannot be made"),
        detail=f"temperature {rows.temperature:g}, G alone, {hot['attempts_a_goal_problem']} attempts a problem",
        summary={"stage": L4, "label": LABEL, "model": who, "temperature": rows.temperature, "adapter": str(adapter)},
        stage=STAGE, label=L4, also=what_else_a_model_wrote, rungs=False))


# -------------------------------------------------------------------------------------------------- report
def ladder_l4_rows_report(config: dict) -> dict:
    from rlvr_lean.gpu.ladder_l3c import proof_lengths
    from rlvr_lean.reporting.ladder_l4_rows import build_rows_report

    rows = the_settings(config)
    store = _store(config, rows)
    sampled_again = hot_of(rows)
    for marker in (PREPARE, *(step(model) for model in rows.models for step in (train_marker, measure_marker)), *(measure_marker(hot_name(step)) for step, _ in sampled_again)):
        _need(store, marker, "L4t's report", STAGE)
    prepared = store.done_summary(PREPARE)
    from_pre = rows.start == PRE
    samplings, of_the_stored = prepared["goal_samplings"], json.loads(store.path(STORED_MODELS_FILE).read_text())
    stored = {who: {RUNG_PART: store.read_rows(stored_file(who, RUNG_PART, L4)), GOAL: [store.read_rows(stored_file(who, sampling["name"], L4)) for sampling in samplings],
                    **{key: of_the_stored[who][key] for key in WROTE}} for who in (STORED if from_pre else (rows.start,))}
    # From another start: the stored `without` and `pre` (rank 16) stand beside `old_rule`'s own read, when the prepare step found their rows on the box.
    rank_16 = None
    if not from_pre and (prepared.get("rank_16") or {}).get("read"):
        rank_16 = {who: {GOAL: [store.read_rows(stored_file(who, sampling["name"], L4)) for sampling in samplings], **{key: of_the_stored[who][key] for key in WROTE}}
                   for who in (PRE, WITHOUT)}
    models, trains, losses, trained = {}, {}, {}, {}
    for model in rows.models:
        measured = store.done_summary(measure_marker(model))
        models[model] = {**ladder_l4._read(store, prepared, L4, model, measured), "stand_in_engine": measured["stand_in_engine"]}
        trains[model] = store.done_summary(train_marker(model))
        stored_loss = json.loads(store.path(loss_file(model)).read_text())
        losses[model] = stored_loss["row_losses"]
        # What a model WAS trained on is the training loop's own record of row ids, read back to the stored set's rows (never their text).
        by_id = {row["id"]: row for row in store.read_rows(training_file(model))}
        trained[model] = [{key: by_id[row_id].get(key) for key in ("id", "problem_id", "origin", "k", "lines")} for row_id in stored_loss["rows_trained"]]
    hot = {}
    for step, who in sampled_again:
        measured = store.done_summary(measure_marker(hot_name(step)))
        hot[who] = {GOAL: [_results(store, goal_set(sampling["name"], hot_name(who), L4)) for sampling in prepared[HOT]["samplings"]],
                    **{key: measured[key] for key in WROTE}, "stand_in_engine": measured["stand_in_engine"]}
    # The goal problems nothing stored has ever solved are GIVEN (a file of ids in the run directory), never computed here.
    never = [row["problem_id"] for row in store.read_rows(NEVER_SOLVED_FILE)] if store.path(NEVER_SOLVED_FILE).exists() else None
    # The length of a held-out problem's published proof is read HERE, by the report, and by nothing before it.
    report = build_rows_report(prepared, trains, losses, trained, store.read_rows(ladder_l4.GROUPS_FILE), proof_lengths(), stored, models, hot,
                               [row["problem_id"] for row in store.read_rows(ladder_l4.AGAIN_FILE)], config["ladder_loop"]["l4"], config["evaluation"], never, rank_16)
    directory = store.root / ladder_assembly.ADAPTERS
    kept, read_only = " and ".join(f"`{model}`" for model in sorted(rows.models)), " and ".join(f"`{who}`" for who in ((PRE, WITH) if from_pre else (rows.start,)))
    report["adapters"] = {"kept": True, "what": f"the adapters of {kept} ({LABEL}) are KEPT: the one whose rule is taken is the next arm's "
                                                f"reference, and no step of this stage deletes one. {read_only} " + ("are their own runs' and were" if from_pre else "is its own run's and was")
                                                + " only read",
                          "directory": str(directory), "there": sorted(path.name for path in directory.glob("*") if path.is_dir()) if directory.is_dir() else []}
    _write_json(store, REPORT_FILE, report)
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    for name in (STORED_MODELS_FILE, *(loss_file(model) for model in rows.models)):
        store.mirror(name)
    for line in report["lines"]:
        print(line, flush=True)
    store.mark_done(REPORT, {"stage": L4, "label": LABEL, "check": "L4t", "headline": report["headline"], "branches": report["branches"], "ok": report["ok"]})
    return report


def _for(step, name: str):
    def run(config: dict) -> dict:
        return step(config, name)
    return run


# From `pre`, in the order the stage runs them: each model trained, then measured, before the next is trained. Then what a run from
# another start has beside (`steps_of` gives each start its own steps in its own order): `old_rule`, its hot measurement, the start model's.
STEPS = {PREPARE: ladder_l4_rows_prepare}
for _model in MODELS:
    STEPS[train_marker(_model)] = _for(ladder_l4_rows_train, _model)
    STEPS[measure_marker(_model)] = _for(ladder_l4_rows_measure, _model)
for _who in HOT_OF:
    STEPS[measure_marker(hot_name(_who))] = _for(ladder_l4_rows_measure_hot, _who)
STEPS[REPORT] = ladder_l4_rows_report
STEPS[train_marker(OLD_RULE)] = _for(ladder_l4_rows_train, OLD_RULE)
STEPS[measure_marker(OLD_RULE)] = _for(ladder_l4_rows_measure, OLD_RULE)
for _who in (OLD_RULE, THE_START):
    STEPS[measure_marker(hot_name(_who))] = _for(ladder_l4_rows_measure_hot, _who)
