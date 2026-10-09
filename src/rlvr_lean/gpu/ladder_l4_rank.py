"""L4r: is the pretrained model capped by the size of its adapter? Spec: docs/spec/ladder-loop.spec.md, "L4r: is the
pretrained model capped by the size of its adapter?" and its "Made exact by the build". A labelled check of L4's
pretraining: everything it makes is labelled pretrained on published proofs, as L4's is; it trains on L4's pretraining
file and on nothing else, and that file is not distributed. Each step is its own process (`rlvr_lean.runner.entry`).

`pre` is one adapter of rank 16. The check is `pre`'s pass AGAIN, from the base, with ONE change: the adapter's rank and
alpha (`ladder_loop.l4.rank_check`; alpha / rank stays `pre`'s). Its model is `pre_r<rank>`, in a run directory of its
own (`ladder_l4_pretrain_r<rank>_seed<N>`); it is measured as `pre` was, with `pre`'s sampling seeds, and reported
against `pre`'s stored rows. `pre`'s run is only read. No map is made.

  ladder_l4_rank_prepare   what the pretraining's prepare step does (the same stored runs, the same file, the same
                           refusals: `ladder_l4.read_for_a_pretraining`), and BEFORE anything is written: REFUSES a
                           task whose seed is not `pre`'s; a pretraining run (`ladder_l2.pretraining_run`: the ONE
                           pretraining's) without its report, its prepare and measure steps' markers, its training
                           record or a measured set's rows and marker, or whose report is not to be read; a run that is
                           not `pre`'s pass with the ONE change (`l4_rank.the_one_change`); and stored rows of `pre` it
                           would not pair with by problem. Copies what its report reads of `pre`. Records the rank,
                           the alpha and the adapter's number of trained numbers. No GPU, no Lean
  ladder_l4_rank_train     `ladder_l4.train_a_pretraining` with the check's rank and alpha: from the base, one pass,
                           the file's order, the task's seed; the adapter `pre_r<rank>`, KEPT. What it SAVED is read
                           back from its files and held to the check before the step is marked done. A training that
                           runs out of GPU memory FAILS THE STEP and says which stage to queue in its place: no smaller
                           rank is tried inside the task
  ladder_l4_rank_measure   `ladder_l4.measure_a_pretrained`: `pre_r<rank>` as `pre` was measured, the model server's
                           largest adapter rank raised to the check's FOR THIS STEP ALONE
  ladder_l4_rank_report    the read fixed before the run (`reporting/ladder_l4_rank.py`)

TWO STAGES run these steps, and name which check by `RLVR_LEAN_LADDER_L4_RANK_CHECK`: `ladder_l4_rank` (`rank`: the
setting's rank) and `ladder_l4_rank_fallback` (`fallback`: the smaller one, run ONLY when the box could not hold the
first, whose run is then VOID). `a_check` is the ONE place that reads the setting (`the_check`: the stage's own, by its
variable); `config_of` is the ONE place that puts a rank into a config, and only these steps call it, and L4t's when it
starts from this check's model (`gpu/ladder_l4_rows.py`: `check_of`, `run_directory`, `adapter_directory`,
`adapter_read_back`): the pretraining stage, and every other stage, reads the config's own rank and the model server's
own limit.

NO STEP DELETES THE ADAPTER. The spec keeps `pre_r<rank>` until the read: it is the arm's starting model if the first
branch is taken, and it is deleted otherwise, by hand.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

from rlvr_lean.domain.ladder_round.l4 import PRE
from rlvr_lean.domain.ladder_round.l4_rank import CHECKS, FALLBACK, RANK, adapter_numbers, model_name, rank_and_alpha, the_one_change
from rlvr_lean.domain.ladder_round.read import GOAL
from rlvr_lean.gpu import ladder_assembly, ladder_ceiling, ladder_l2, ladder_l4
from rlvr_lean.gpu.ladder_ceiling import RUNG_PART, Reading, _need, stored_file
from rlvr_lean.gpu.ladder_dose import _runs
from rlvr_lean.gpu.ladder_l4 import L4, LABEL, _say
from rlvr_lean.gpu.ladder_round import _write_json, training_seed
from rlvr_lean.infrastructure.adapter_files import ADAPTER_FILE, tensors_of
from rlvr_lean.infrastructure.artifact_store import ArtifactStore

L4R = "l4_rank"                                         # what this check's own files are called, beside the pretraining's (`l4_...`)
CHECK_VARIABLE = "RLVR_LEAN_LADDER_L4_RANK_CHECK"       # which check of `ladder_loop.l4.rank_check` the stage runs: `rank` or `fallback`
RUN_VARIABLE = "RLVR_LEAN_LADDER_L4_RANK_RUN"           # another run directory than `ladder_l4_pretrain_r<rank>_seed<seed>` (a smoke run)
STAGE, FALLBACK_STAGE = "ladder_l4_rank", "ladder_l4_rank_fallback"
STAGE_OF = {RANK: STAGE, FALLBACK: FALLBACK_STAGE}
PREPARE, TRAIN, MEASURE, REPORT = (f"{STAGE}_{name}" for name in ("prepare", "train", "measure", "report"))
REPORT_FILE = "report_ladder_l4_rank.json"
PRE_LOSS_FILE = f"{L4R}_loss_of_pre.json"               # `pre`'s rows' losses, copied from its run: the report puts them beside the check's
NEVER_SOLVED_FILE = f"{L4R}_never_solved.jsonl"         # GIVEN, not made here: one `problem_id` a row, the goal problems nothing stored has ever solved
ADAPTER_CONFIG = "adapter_config.json"                  # what PEFT writes beside an adapter's weights: its rank (`r`) and `lora_alpha`
SETTING = "ladder_loop.l4.rank_check"
PLAN = ("sampling_seeds", "rung_episodes", "goal_samplings")       # how a model is measured, as a prepare step records it
READING = Reading(reader="L4r", models="the check's model", not_read="that set is `pre`'s side of every comparison, and no report of this run could be read against it",
                  setting=ladder_l4.SETTING)


@dataclass(frozen=True)
class Check:
    """One check of `ladder_loop.l4.rank_check`, as the stage's steps read it."""
    which: str                          # `rank` or `fallback`
    rank: int
    alpha: int
    name: str                           # the model, its adapter's directory and its sets: `pre_r64`
    stage: str                          # the stage that runs it, as `runner.entry` names it
    in_the_place_of: int | None         # the fallback: the rank the box could not hold
    then: tuple[int, int] | None        # the check `rank`: the (rank, alpha) to run in its place when the box cannot hold it


def a_check(config: dict, which: str) -> Check:
    """THE one place that reads `ladder_loop.l4.rank_check`: its check `which` (`rank` or `fallback`)."""
    settings = config["ladder_loop"]["l4"]["rank_check"]
    rank, alpha = rank_and_alpha(settings, which)
    return Check(which=which, rank=rank, alpha=alpha, name=model_name(rank), stage=STAGE_OF[which],
                 in_the_place_of=rank_and_alpha(settings, RANK)[0] if which == FALLBACK else None,
                 then=rank_and_alpha(settings, FALLBACK) if which == RANK else None)


def the_check(config: dict) -> Check:
    """The check the stage names (`RLVR_LEAN_LADDER_L4_RANK_CHECK`), as the setting has it (`a_check`). A task that names
    none, or one the setting does not have, is refused: these steps are run by their stages, which set it."""
    which = os.environ.get(CHECK_VARIABLE)
    if which not in CHECKS:
        raise RuntimeError(f"{CHECK_VARIABLE} is {which!r}: it names the check of {SETTING} a task runs, one of {list(CHECKS)}. The steps of L4r are run by the stages "
                           f"`{STAGE}` and `{FALLBACK_STAGE}`, which set it")
    return a_check(config, which)


def check_of(config: dict, model: str) -> Check | None:
    """The check whose MODEL is `model` (`pre_r64`), for a stage that starts from that model (L4t) and so names it and
    not a check; None when the setting has no check of that model. It reads no variable: the stage that asks is not one
    of the check's own."""
    return next((check for check in (a_check(config, which) for which in CHECKS) if check.name == model), None)


def config_of(config: dict, check: Check) -> dict:
    """The config as the check's steps hand it on, and THE one place a rank is put into one: the adapter's rank and
    alpha are the check's (what the training attaches and what the prepare step records as its recipe), and the model
    server's largest adapter rank is the check's rank (what the measurement serves the adapter under). Nothing else
    moves, and `config` itself is not changed: a stage that does not call this reads the config's own."""
    return {**config, "lora": {**config["lora"], "rank": check.rank, "alpha": check.alpha}, "vllm": {**config["vllm"], "max_lora_rank": check.rank}}


def run_name(check: Check, seed: int) -> str:
    return f"{ladder_l4.PRETRAIN_STAGE}_r{check.rank}_seed{seed}"


def run_directory(config: dict, check: Check, seed: int) -> Path:
    """The check's run directory at `seed` (or the one the task names: a smoke run). Nothing is created: another stage
    that starts from the check's model reads it by this."""
    return _runs(config) / (os.environ.get(RUN_VARIABLE) or run_name(check, seed))


def adapter_directory(config: dict, check: Check, seed: int) -> Path:
    """Where the check's model is kept: `adapters/pre_r<rank>` of its run directory."""
    return run_directory(config, check, seed) / ladder_assembly.ADAPTERS / check.name


def _store(config: dict, check: Check) -> ArtifactStore:
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    return ArtifactStore(run_directory(config, check, training_seed(config)), Path(mirror) if mirror else None)


def pre_directory(config: dict) -> Path:
    """`pre`'s run: the ONE pretraining's (`ladder_l2.pretraining_run`: the seed `ladder_loop.l4.pretraining_seed`, or the
    run the task names). Only read."""
    return _runs(config) / ladder_l2.pretraining_run(config)


def _as_a_pretraining(check: Check) -> ladder_l4.Pretraining:
    """The check's model under the names the pretraining's steps take."""
    return ladder_l4.Pretraining(
        name=check.name, stage=check.stage, prepare=PREPARE, train=TRAIN, measure=MEASURE, run=f"{ladder_l4.PRETRAIN_STAGE}_r{check.rank}",
        what="the check's training", called=f"the check's model `{check.name}`", whose=f"L4r's model `{check.name}`",
        no_adapter=f"the adapter `{check.name}` is kept until the read, and one trained again would not be the model its other measurements came from. "
                   "The measurement cannot be made")


# ------------------------------------------------------------------------------------------------- prepare
def _shapes(path: Path) -> dict:
    """The header of an adapter's weights file (`adapter_files.tensors_of`: by tensor name, its shape). Refused
    (ValueError) for a file too short to hold the header it announces: it is not an adapter's file."""
    with open(path, "rb") as file:
        announced = file.read(8)
    if len(announced) < 8 or struct.unpack("<Q", announced)[0] > path.stat().st_size - 8:
        raise ValueError(f"{path} is not an adapter's weights file")
    return tensors_of(path)[0]


def _numbers(directory: Path, check: Check) -> dict:
    """The adapter's number of trained numbers, from the shapes of `pre`'s stored adapter (the same modules, at the
    check's rank): `l4_rank.adapter_numbers` on the header of its file. Not counted, with the reason, when that file is
    not on this box or is not an adapter's (the stand-in trains none): the training step counts the adapter it saves."""
    path = directory / ladder_assembly.ADAPTERS / PRE / ADAPTER_FILE
    try:
        counted = adapter_numbers(_shapes(path), check.rank)
    except (OSError, ValueError, KeyError) as error:
        return {"trained_parameters": None, "trained_parameters_of_pre": None, "counted_from": None,
                "not_counted": f"{path} could not be read as an adapter's weights file ({type(error).__name__}): the training step counts the adapter it saves"}
    return {"trained_parameters": counted["numbers_at_rank"], "trained_parameters_of_pre": counted["numbers"], "modules": counted["modules"],
            "counted_from": f"the shapes of `pre`'s stored adapter ({path}): rank x (inputs + outputs), summed over its {counted['modules']} modules"}


def ladder_l4_rank_prepare(config: dict) -> dict:
    check = the_check(config)
    store = _store(config, check)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    seed, of_the_pretraining, directory = training_seed(config), ladder_l2.pretraining_seed(config), pre_directory(config)
    task = (f"the task of stage `{ladder_l4.PRETRAIN_STAGE}` for seed {of_the_pretraining} (`python -m rlvr_lean.runner.entry --stage {ladder_l4.PRETRAIN_STAGE} --seeds {of_the_pretraining}`; "
            f"for the smoke run, stage `{ladder_l4.PRETRAIN_STAGE}_smoke`)")
    if seed != of_the_pretraining:
        raise RuntimeError(f"this task's seed is {seed} and `pre` is the pretraining of seed {of_the_pretraining} (ladder_loop.l4.pretraining_seed): the check is `pre`'s "
                           f"pass again, the SAME seed, with one change. Queue it with --seeds {of_the_pretraining}; nothing was written.")
    markers = (ladder_l4.REPORT_FILE, f"{ladder_l4.PREPARE}.done.json", f"{ladder_l4.MEASURE}.done.json", ladder_l4.LOSS_FILE)
    lacking = [name for name in markers if not (directory / name).exists()]
    if lacking:
        raise RuntimeError(f"{directory} does not hold {lacking}. L4r is read against `pre`'s stored rows and its training record, which {task} wrote on this box. "
                           "Run that task to its end first; nothing was written.")
    of_pre, report = json.loads((directory / f"{ladder_l4.PREPARE}.done.json").read_text()), json.loads((directory / ladder_l4.REPORT_FILE).read_text())
    parts = ladder_l4.pre_parts(of_pre)
    lacking = [name for part in parts for name in (ladder_l4.pre_rows_file(part), f"episodes_{ladder_l4.pre_set(part)}.done.json") if not (directory / name).exists()]
    if lacking:
        raise RuntimeError(f"{directory} does not hold {lacking}: `pre`'s per-problem rows on a set it was measured on, or that set's own marker. L4r is read against "
                           f"them; {task} wrote them on this box. Run that task to its end first; nothing was written.")
    if of_pre.get("seed") != of_the_pretraining:        # a directory that is not the run it is named for
        raise RuntimeError(f"the pretraining run {directory.name} was made at seed {of_pre.get('seed')} and ladder_loop.l4.pretraining_seed is {of_the_pretraining}: "
                           "the check is read against the ONE pretraining of that seed. Nothing was written.")
    if not report.get("ok", True):
        raise RuntimeError(f"the report of the pretraining run {directory.name} is not to be read (Lean gave no verdict on too much of a set): `pre`'s rows cannot "
                           f"stand on the other side of the check. Queue {task} again, which measures that set again; nothing was written.")
    # ---- what the pretraining's own prepare step reads and refuses, with the check's rank and alpha in the recipe
    found = ladder_l4.read_for_a_pretraining(config_of(config, check))
    try:
        change = the_one_change(of_pre, found.head)
    except ValueError as error:
        raise RuntimeError(f"{error}. Nothing was written.") from error
    # ---- `pre`'s stored rows pair by problem with what this run will measure: the same sets' sizes, seeds and problems
    plan = {key: found.sizes[key] for key in PLAN}
    if {key: of_pre.get(key) for key in PLAN} != plan:
        raise RuntimeError(f"this run would measure `{check.name}` with {plan} and {directory.name} recorded {({key: of_pre.get(key) for key in PLAN})} for `pre`: the two "
                           "models would not pair by problem. Nothing was written.")
    ids = {RUNG_PART: [row["problem_id"] for row in found.read.rung_problems], GOAL: [row["problem_id"] for row in found.read.goal_problems]}
    measured = [(RUNG_PART, RUNG_PART, plan["rung_episodes"]), *((sampling["name"], GOAL, sampling["episodes"]) for sampling in plan["goal_samplings"])]
    for part, of, episodes in measured:
        ladder_ceiling._stored(directory, ladder_l4.pre_set(part), ids[of], plan["sampling_seeds"][part], episodes, f"`pre` ({part})", READING)
    losses = json.loads((directory / ladder_l4.LOSS_FILE).read_text())
    if len(losses["row_losses"]) != found.head["rows"]:
        raise RuntimeError(f"{directory.name} stored the losses of {len(losses['row_losses'])} rows and this run trains on {found.head['rows']}: `pre`'s training "
                           "record is not of this file. Nothing was written.")
    # ---- from here on files are written: what the report reads of `pre`, then what the pretraining's prepare step writes
    ladder_l4.copy_what_pre_stored(store, directory, of_pre)
    _write_json(store, PRE_LOSS_FILE, {"stage": L4, "label": LABEL, "model": PRE, "run": str(directory), "rows": len(losses["row_losses"]), "what": losses["what"],
                                       "row_losses": losses["row_losses"]})
    own = {"rank_check": {
        "what": "L4r: `pre`'s pass again with ONE change, the adapter's rank and alpha; measured as `pre` was and read against `pre`'s stored rows",
        "check": check.which, "stage": check.stage, "model": check.name, "rank": check.rank, "alpha": check.alpha, "in_the_place_of_rank": check.in_the_place_of,
        "max_lora_rank": check.rank, "max_lora_rank_of_every_other_stage": config["vllm"]["max_lora_rank"], **_numbers(directory, check),
        "the_one_change": change,
        PRE: {"run": str(directory), "seed": of_pre["seed"], "rows": of_pre["rows"], "parts": parts, "report_checks_pass": report.get("checks_pass"),
              "rank": of_pre["recipe"]["lora"]["rank"], "alpha": of_pre["recipe"]["lora"]["alpha"]},
        "never_solved_file": NEVER_SOLVED_FILE}}
    summary = ladder_l4.write_prepared(store, found, _as_a_pretraining(check), own)
    counted = summary["rank_check"]["trained_parameters"]
    _say(f"L4r, the check `{check.which}`: `{check.name}`, rank {check.rank} and alpha {check.alpha} where `pre` has {change['rank'][PRE]} and {change['alpha'][PRE]}"
         + (f"; {counted:,} trained numbers against `pre`'s {summary['rank_check']['trained_parameters_of_pre']:,}" if counted else "")
         + f"; read against {directory.name}")
    return summary


# --------------------------------------------------------------------------------------------------- train
def out_of_memory(error: BaseException) -> bool:
    """Whether a training died for want of GPU memory: PyTorch's own error (by its name: torch is not imported
    here), or a CUDA error that says so."""
    return type(error).__name__ == "OutOfMemoryError" or "out of memory" in str(error).lower()


def the_adapter_saved(adapter: Path, check: Check) -> dict:
    """What the training SAVED, read back from its two files and HELD TO THE CHECK before the step is marked done: the
    rank and alpha PEFT recorded (`adapter_config.json`: what the model server will serve it with) and the rank of
    every matrix in the weights' file, with how many numbers it holds. A rank or an alpha that is not the check's
    means the ONE change did not reach the training: refused, and nothing of such a run is measured. Beside it, the
    most GPU memory the training's allocator held (the cap is on that figure, not on what was allocated). The stand-in
    trains nothing and saves nothing."""
    if ladder_ceiling._stand_in():
        return {"adapter_saved": None}
    import torch

    saved = adapter_read_back(adapter)
    if not is_the_checks(saved, check):
        raise RuntimeError(f"the adapter saved at {adapter} has rank {saved['rank']} and alpha {saved['alpha']} in its {ADAPTER_CONFIG} and matrices of rank "
                           f"{saved['ranks']}; the check `{check.which}` is rank {check.rank} and alpha {check.alpha}. The ONE change did not reach the training: "
                           "nothing of this run is measured, and the step is not marked done")
    return {"adapter_saved": saved, "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2)}


def adapter_read_back(adapter: Path) -> dict:
    """An adapter as its two files have it: the rank and alpha PEFT recorded (`adapter_config.json`), the rank of every
    matrix in the weights' file and how many numbers it holds (`l4_rank.adapter_numbers`)."""
    recorded = json.loads((adapter / ADAPTER_CONFIG).read_text())
    return {"rank": recorded.get("r"), "alpha": recorded.get("lora_alpha"), **adapter_numbers(_shapes(adapter / ADAPTER_FILE))}


def is_the_checks(saved: dict, check: Check) -> bool:
    """Whether an adapter read back (`adapter_read_back`) has the check's rank and alpha, in its config and in every matrix."""
    return saved["rank"] == check.rank and saved["alpha"] == check.alpha and saved["ranks"] == [check.rank]


def ladder_l4_rank_train(config: dict) -> dict:
    """The check's training: `pre`'s (from the base, one pass, the file's order, the task's seed), with the check's
    rank and alpha. AN OUT-OF-MEMORY FAILURE FAILS THE STEP: by the spec that run is VOID and the smaller rank is run
    in its place, as a task of its own, so that the note can say which ran. Nothing is tried again inside this one."""
    check = the_check(config)
    store, seed = _store(config, check), training_seed(config)
    try:
        return ladder_l4.train_a_pretraining(config_of(config, check), store, _as_a_pretraining(check), saved=lambda adapter: the_adapter_saved(adapter, check))
    except Exception as error:      # noqa: BLE001 - only an out-of-memory failure is said otherwise; every other error is raised as it is
        if not out_of_memory(error):
            raise
        said = str(error).splitlines()[0][:300] if str(error) else type(error).__name__
        again = ("(If the memory was another process's and not this training's, queue this task again instead: a training that did not finish starts again from the "
                 "base.) No adapter was saved")
        if check.then is None:
            raise RuntimeError(f"the box cannot hold an adapter of rank {check.rank} at the sequence limit either (the fallback of {SETTING}, run in the place of rank "
                               f"{check.in_the_place_of}): the training ran out of GPU memory ({said}). This run is VOID too, no smaller rank is configured, and the check "
                               f"cannot be made on this box as the spec has it. {again}") from error
        rank, alpha = check.then
        raise RuntimeError(f"the box cannot hold an adapter of rank {check.rank} at the sequence limit: the training ran out of GPU memory ({said}). By the spec this run is "
                           f"VOID, nothing of it is read, and rank {rank} with alpha {alpha} is run in its place under the same read. NO fallback is made inside this task: "
                           f"queue stage `{FALLBACK_STAGE}` (`python -m rlvr_lean.runner.entry --stage {FALLBACK_STAGE} --seeds {seed}`: the model `{model_name(rank)}`, the run directory "
                           f"{ladder_l4.PRETRAIN_STAGE}_r{rank}_seed{seed}), and say in the note which ran. {again}") from error


# ------------------------------------------------------------------------------------------------- measure
def ladder_l4_rank_measure(config: dict) -> dict:
    """`pre_r<rank>` as `pre` was measured: the rungs and G's samplings with the sampling seeds the prepare step held
    to `pre`'s, the adapter served beside the base by a model server started with the check's rank as its largest
    (`config_of`: this step's config alone). A set Lean did not answer is sampled again when the task is queued again,
    from the kept adapter (`measure_model`)."""
    check = the_check(config)
    return ladder_l4.measure_a_pretrained(config_of(config, check), _store(config, check), _as_a_pretraining(check))


# -------------------------------------------------------------------------------------------------- report
def ladder_l4_rank_report(config: dict) -> dict:
    from rlvr_lean.gpu.ladder_l3c import proof_lengths
    from rlvr_lean.reporting.ladder_l4_rank import build_rank_report

    check = the_check(config)
    store = _store(config, check)
    for marker in (PREPARE, TRAIN, MEASURE):
        _need(store, marker, "the check's report", check.stage)
    prepared, measured = store.done_summary(PREPARE), store.done_summary(MEASURE)
    samplings = prepared["goal_samplings"]
    pre = {RUNG_PART: store.read_rows(stored_file(PRE, RUNG_PART, L4)), GOAL: [store.read_rows(stored_file(PRE, sampling["name"], L4)) for sampling in samplings],
           **json.loads(store.path(ladder_l4.PRE_FILE).read_text())}
    model = {**ladder_l4._read(store, prepared, L4, check.name, measured), "stand_in_engine": measured["stand_in_engine"]}
    losses = {PRE: json.loads(store.path(PRE_LOSS_FILE).read_text())["row_losses"], check.name: json.loads(store.path(ladder_l4.LOSS_FILE).read_text())["row_losses"]}
    # The goal problems nothing stored has ever solved are GIVEN (a file of ids in the run directory), never computed here.
    never = [row["problem_id"] for row in store.read_rows(NEVER_SOLVED_FILE)] if store.path(NEVER_SOLVED_FILE).exists() else None
    # The length of a held-out problem's published proof is read HERE, by the report, and by nothing before it.
    report = build_rank_report(prepared, store.done_summary(TRAIN), losses, store.read_rows(ladder_l4.GROUPS_FILE), proof_lengths(), pre, model,
                               config["ladder_loop"]["l4"], config["evaluation"], never)
    adapter = store.root / ladder_assembly.ADAPTERS / check.name
    report["adapter"] = {"kept": True, "what": f"`{check.name}` ({LABEL}) is KEPT until the read: it is the arm's starting model if the first branch is taken, and it is "
                                               "deleted otherwise. No step of this stage deletes it", "directory": str(adapter), "there": adapter.is_dir()}
    _write_json(store, REPORT_FILE, report)
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    for name in (ladder_l4.STORED_MODELS_FILE, ladder_l4.LOSS_FILE, ladder_l4.PRE_FILE, PRE_LOSS_FILE):
        store.mirror(name)
    for line in report["lines"]:
        print(line, flush=True)
    store.mark_done(REPORT, {"stage": L4, "label": LABEL, "check": check.which, "model": check.name, "headline": report["headline"], "branch": report["branch"],
                             "ok": report["ok"]})
    return report


STEPS = {PREPARE: ladder_l4_rank_prepare, TRAIN: ladder_l4_rank_train, MEASURE: ladder_l4_rank_measure, REPORT: ladder_l4_rank_report}
