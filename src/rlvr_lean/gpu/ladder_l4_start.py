"""The model a stage STARTS FROM: a pretrained model, kept by the run that made it. Spec:
docs/spec/ladder-loop.spec.md, "L4r's first branch was taken: what L4t runs" (its "From `pre_r64`" build note) and
"L4b: the arm again, from the larger pretrained model". Everything built on such a model is labelled pretrained on
published proofs.

TWO stages start from a stored pretrained model, and both ask HERE where it lives and what it is: L4t (`gpu/ladder_l4_rows.py`:
three models trained on the rounds already made) and an arm of the loop (`gpu/ladder_l2.py` with `gpu/ladder_l4b.py`: the
arm's `start`). No path, rank or alpha of a start model is stated in either.

  the_start            `pre`: the ONE pretraining's run and its adapter. Any other name is a model of the check of the
                       adapter's rank (`pre_r64`): that check's run and adapter, by `ladder_l4_rank`'s own functions. With
                       it, where the start model's OWN MAP of the base map's problems is (`pre`'s: beside `pre`, made by
                       the pretraining stage; another's: a run directory of its own, made by `ladder_l4b_map`)
  read_the_start_run   what the run that made another start than `pre` recorded, REFUSED when it cannot carry a stage
  config_from          the config as a step hands it on: from `pre` the config ITSELF; from another start the START
                       ADAPTER's rank and alpha for a training (what its run recorded, never the config's own), and for a
                       step that samples the model server's largest adapter rank too (`ladder_l4_rank.config_of`, which
                       stays the ONE place a rank is put into a config)
  the_adapter_saved    an adapter a training saved, read back and held to the start adapter's rank and alpha
  pretrained_on        the rows a start model was pretrained on, without their text, and the file held to the SHA-256
                       its run recorded; `examples_of` reads a rehearsal row's text from that file and nowhere else
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.assembly import EXAMPLE_FIELDS
from rlvr_lean.domain.ladder_round.ceiling import training_example
from rlvr_lean.domain.ladder_round.l4 import MAP_FILE, MAP_STEP, PRE
from rlvr_lean.domain.ladder_round.l4_rows import PRETRAINING
from rlvr_lean.domain.repair.accumulate import proof_line_count
from rlvr_lean.gpu import ladder_assembly, ladder_ceiling, ladder_l2, ladder_l4, ladder_l4_rank
from rlvr_lean.gpu.ladder_ceiling import RUNG_PART, _file_rows, goal_set, rung_set
from rlvr_lean.gpu.ladder_dose import _rows, _runs
from rlvr_lean.gpu.ladder_l4 import L4

MAP_RUN_VARIABLE = "RLVR_LEAN_LADDER_L4_MAP_RUN"        # another run directory for the map of a start that is not `pre` than `ladder_l4_map_<start>_seed<seed>` (a smoke run)
MAP_STAGE_STEP = "ladder_l4b_map"                       # the step that makes the map of a start that is not `pre`, and its marker


@dataclass(frozen=True)
class Start:
    """The model every model of a run is trained FROM, and where it lives. Only read."""
    name: str                                   # `pre`, or a model of the check of the adapter's rank (`pre_r64`)
    directory: Path                             # the run that made it
    adapter: Path                               # its adapter
    check: ladder_l4_rank.Check | None          # the check that made it; None for `pre`, whose rank is the config's own
    task: str                                   # the task that writes that run, for a refusal
    prepare: str = ladder_l4.PREPARE            # the markers of that run's prepare and measure steps
    measure: str = ladder_l4.MEASURE
    map_directory: Path | None = None           # where ITS OWN MAP of the base map's problems is: the run that made it
    map_file: str = MAP_FILE                    # ... the map's rows, in the shape the challenger reads the base's
    map_step: str = MAP_STEP                    # ... and the step that made it, by its marker
    map_task: str = ""                          # what to run when the map is not there, for a refusal


@dataclass(frozen=True)
class Reader:
    """Who reads the run that made a start model, for what is said when it is refused."""
    reads: str = "L4t makes no round and no pretraining: it reads what"        # "<reads> <the task> wrote on this box"
    carry: str = "L4t"                                                         # "the run ... cannot carry <carry>"


def map_run(config: dict, name: str) -> Path:
    """The run directory of the map of a start that is not `pre`: one for every seed of an arm, at the pretraining's
    seed (the map is sampled with the base map's own seed, which no task's seed moves), or the one the task names."""
    return _runs(config) / (os.environ.get(MAP_RUN_VARIABLE) or f"ladder_l4_map_{name}_seed{ladder_l2.pretraining_seed(config)}")


def map_file(name: str) -> str:
    """The map of a start model, as a file of its run: `l4_map_pre.jsonl` is `pre`'s."""
    return MAP_FILE if name == PRE else f"{L4}_map_{name}.jsonl"


def the_start(config: dict, name: str, said_by: str = "the arm's `start`") -> Start:
    """The start model `name`. `pre`: the ONE pretraining's run and its adapter, its map beside it. Any other name is a
    model of the check of the adapter's rank: that check's run of the pretraining's seed and its adapter, by
    `ladder_l4_rank`'s own functions (no path is stated twice), its map in a run of its own. A name that is neither is
    refused (RuntimeError); `said_by`: the setting that named it."""
    of_the_pretraining = ladder_l2.pretraining_seed(config)
    if name == PRE:
        directory = _runs(config) / ladder_l2.pretraining_run(config)
        task = (f"the task of stage `{ladder_l4.PRETRAIN_STAGE}` for seed {of_the_pretraining} (`python -m rlvr_lean.runner.entry --stage {ladder_l4.PRETRAIN_STAGE} --seeds {of_the_pretraining}`; "
                f"for the smoke run, stage `{ladder_l4.PRETRAIN_STAGE}_smoke`)")
        return Start(PRE, directory, directory / ladder_assembly.ADAPTERS / PRE, None, task, map_directory=directory,
                     map_task=(f"the task of stage `{ladder_l4.PRETRAIN_STAGE}` for seed {of_the_pretraining} to its end first "
                               f"(`python -m rlvr_lean.runner.entry --stage {ladder_l4.PRETRAIN_STAGE} --seeds {of_the_pretraining}`; for the smoke run, stage `{ladder_l4.PRETRAIN_STAGE}_smoke`)"))
    check = ladder_l4_rank.check_of(config, name)
    if check is None:
        raise RuntimeError(f"{said_by} is {name!r}: it names `{PRE}` or a model of {ladder_l4_rank.SETTING} (the check of the "
                           "adapter's rank made it, and its run is where it lives). Nothing was written.")
    return Start(check.name, ladder_l4_rank.run_directory(config, check, of_the_pretraining), ladder_l4_rank.adapter_directory(config, check, of_the_pretraining), check,
                 f"the task of stage `{check.stage}` for seed {of_the_pretraining} (`python -m rlvr_lean.runner.entry --stage {check.stage} --seeds {of_the_pretraining}`; for the smoke run, stage `{ladder_l4_rank.STAGE}_smoke`)",
                 prepare=ladder_l4_rank.PREPARE, measure=ladder_l4_rank.MEASURE, map_directory=map_run(config, check.name), map_file=map_file(check.name),
                 map_step=MAP_STAGE_STEP,
                 map_task=(f"the step `{MAP_STAGE_STEP}` to its end first: it is the first step of the arm's own stage (`ladder_l4b_*`), which makes `{check.name}`'s own map "
                           f"of the base map's problems before the arm's prepare step reads it"))


def is_known(config: dict, name: str) -> bool:
    """Whether `name` is a start model: `pre`, or a model of the check of the adapter's rank."""
    return name == PRE or ladder_l4_rank.check_of(config, name) is not None


def start_set(part: str, start: Start) -> str:
    """The set the start model's episodes on one part are stored under, in its own run directory."""
    return ladder_l4.pre_set(part) if start.check is None else rung_set(start.name, L4) if part == RUNG_PART else goal_set(part, start.name, L4)


def wrote_file(name: str) -> str:
    """What a start model wrote (from its measure step's marker), as another run's own directory holds it: `l4_stored_pre.json` is `pre`'s."""
    return ladder_l4.PRE_FILE if name == PRE else f"{L4}_stored_{name}.json"


def recipe_recorded(prepared: Mapping) -> dict:
    """The rank and alpha of a start model's adapter, as the prepare step of the run that made it recorded them."""
    lora = (prepared.get("recipe") or {}).get("lora") or {}
    return {"rank": lora.get("rank"), "alpha": lora.get("alpha")}


def config_from(config: dict, prepared: Mapping, start: Start, serving: bool = False) -> dict:
    """The config as a step of a stage that starts from `start` hands it on. FROM `pre`: the config itself, nothing moved.
    From another start: a training attaches an adapter of the START ADAPTER's rank and alpha, as the check's run recorded
    them and the stage's own prepare step stored them (`start_recipe`; never `config["lora"]`), and a step that samples
    (`serving`) is also handed a model server whose largest adapter rank is that rank (`ladder_l4_rank.config_of`: the ONE
    place a rank is put into a config). Refused (RuntimeError) when the setting's check is no longer what the prepare
    step recorded."""
    if start.check is None:
        return config
    recorded = prepared["start_recipe"]
    if (start.check.rank, start.check.alpha) != (recorded["rank"], recorded["alpha"]):
        raise RuntimeError(f"this run was prepared from `{start.name}` at rank {recorded['rank']} and alpha {recorded['alpha']} (what the run that made it recorded), and "
                           f"{ladder_l4_rank.SETTING} now gives it rank {start.check.rank} and alpha {start.check.alpha}: a model here is trained and served at the start "
                           "adapter's own rank. Nothing was trained or sampled")
    of_the_start = ladder_l4_rank.config_of(config, start.check)
    return of_the_start if serving else {**config, "lora": of_the_start["lora"]}


def read_the_start_run(start: Start, of_the_pretraining: int, of_pre: Mapping | None = None, waived: bool = False, reader: Reader = Reader()) -> dict:
    """What a run from another start than `pre` reads of the run that MADE the start model (the check of the adapter's
    rank), with nothing written. REFUSED (RuntimeError), naming the task to run: a run without its report or the
    markers of its prepare, train and measure steps; a report that is not to be read, or whose checks failed; a run of
    another seed or model; an adapter whose recorded rank and alpha are not the check's; and, with `of_pre` (what
    `pre`'s prepare step recorded: a stage whose rehearsal rows are drawn among `pre`'s), a model that was not trained
    on the file `pre` was. Returns what its prepare, train and measure steps recorded, its report, and the checks it
    failed: none, unless `waived` (a SMOKE run's task alone names it), which goes on from a run whose checks failed
    and says so. A report that is not to be read is refused for every run."""
    directory, check = start.directory, start.check
    steps = {"prepared": ladder_l4_rank.PREPARE, "trained": ladder_l4_rank.TRAIN, "measured": ladder_l4_rank.MEASURE}
    lacking = [name for name in (ladder_l4_rank.REPORT_FILE, *(f"{marker}.done.json" for marker in steps.values())) if not (directory / name).exists()]
    if lacking:
        raise RuntimeError(f"{directory} does not hold {lacking}: the report of the check that made `{start.name}`, or what its prepare, train or measure step recorded. "
                           f"{reader.reads} {start.task} wrote on this box. Run that task to its end first; nothing was written.")
    read = {name: json.loads((directory / f"{marker}.done.json").read_text()) for name, marker in steps.items()}
    report = json.loads((directory / ladder_l4_rank.REPORT_FILE).read_text())
    failed = [name for name, entry in (report.get("can_this_run_see_a_win") or {}).items() if isinstance(entry, dict) and not entry.get("passes")]
    waived = waived and bool(failed or report.get("inconclusive"))
    if not report.get("ok", True) or ((failed or report.get("inconclusive")) and not waived):
        raise RuntimeError(f"the run {directory.name} cannot carry {reader.carry}: "
                           + ("its report is not to be read (Lean gave no verdict on too much of a set)" if not report.get("ok", True) else
                              f"its checks failed ({', '.join(failed) or 'INCONCLUSIVE'})")
                           + f". `{start.name}`'s rows stand on the other side of every comparison here. Queue {start.task} again; nothing was written.")
    of_start, own = read["prepared"], read["prepared"].get("rank_check") or {}
    lora = recipe_recorded(of_start)
    if of_start.get("seed") != of_the_pretraining or own.get("model") != start.name:
        raise RuntimeError(f"the run {directory.name} recorded seed {of_start.get('seed')} and the model {own.get('model')!r}; `{start.name}` is the model the check of "
                           f"the adapter's rank made at seed {of_the_pretraining} (ladder_loop.l4.pretraining_seed). Nothing was written.")
    saved = read["trained"].get("adapter_saved")        # what that training read back from the files it saved (the stand-in saves none)
    another_saved = saved is not None and not ladder_l4_rank.is_the_checks(saved, check)
    if (lora["rank"], lora["alpha"]) != (check.rank, check.alpha) or another_saved:
        raise RuntimeError(f"the run {directory.name} recorded an adapter of rank {lora['rank']} and alpha {lora['alpha']} for `{start.name}`"
                           + (f" and saved one of rank {saved.get('rank')} and alpha {saved.get('alpha')} with matrices of rank {saved.get('ranks')}" if another_saved else "")
                           + f"; {ladder_l4_rank.SETTING} gives that model rank {check.rank} and alpha {check.alpha}. Every model here is trained and served at the "
                           "START ADAPTER's rank, which is the one its own run recorded. Nothing was written.")
    if of_pre is not None and (of_start.get("pretraining_file_sha256") != of_pre["pretraining_file_sha256"] or of_start.get("rows") != of_pre["rows"]):
        raise RuntimeError(f"the run {directory.name} trained `{start.name}` on a file of SHA-256 {of_start.get('pretraining_file_sha256')} ({of_start.get('rows')} rows) "
                           f"and `pre` was trained on {of_pre['pretraining_file_sha256']} ({of_pre['rows']} rows): the rehearsal rows are drawn among the rows `pre` was "
                           f"trained on, and they must be rows `{start.name}` was trained on too. Nothing was written.")
    return {**read, "report": report, "checks_failed": failed, "checks_waived": waived}


def the_adapter_saved(adapter: Path, model: str, start: Start) -> dict:
    """From another start than `pre`: what a training SAVED, read back from its two files and HELD TO THE START
    ADAPTER's rank and alpha before its step is marked done, as the check that made the start adapter reads its own
    (`ladder_l4_rank.adapter_read_back`); with it, the most GPU memory the training's allocator held. A model at
    another rank is refused (RuntimeError): the start adapter did not carry over, and nothing of it is measured. From
    `pre`, and for the stand-in (which saves nothing), nothing is read and nothing is added to the summary."""
    if start.check is None or ladder_ceiling._stand_in():
        return {}
    import torch

    saved = ladder_l4_rank.adapter_read_back(adapter)
    if not ladder_l4_rank.is_the_checks(saved, start.check):
        raise RuntimeError(f"the adapter `{model}` saved at {adapter} has rank {saved['rank']} and alpha {saved['alpha']} in its {ladder_l4_rank.ADAPTER_CONFIG} and "
                           f"matrices of rank {saved['ranks']}; the start adapter `{start.name}` is rank {start.check.rank} and alpha {start.check.alpha}, and every model "
                           "here is at the start adapter's rank. Nothing of this model is measured, and the step is not marked done")
    return {"adapter_saved": saved, "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2)}


# ------------------------------------------------------------------- the rows a start model was pretrained on
def pretrained_on(directory: Path, recorded_sha256: str, name: str = PRE) -> tuple[list[dict], str]:
    """(the rows the model `name` was pretrained on, in the file's order, WITHOUT their statements and proofs; the
    file's SHA-256). `directory`: the run that made it, which stored the rows' ids (`l4_pretraining_rows.jsonl`).
    `recorded_sha256`: what that run's prepare step recorded of the file. The pretraining file is read and stays where
    it is. Refused (RuntimeError) when the file is not the one that run recorded, or its rows are not the stored ones
    in their order: a rehearsal row is a row the start model was pretrained on."""
    file_rows, sha256 = _file_rows(ladder_l4.pretraining_file(), ladder_l4.WHAT, ladder_l4.HOW)
    if sha256 != recorded_sha256:
        raise RuntimeError(f"{ladder_l4.WHAT} has SHA-256 {sha256} and `{name}` was trained on {recorded_sha256} ({directory.name}): the rehearsal rows are rows "
                           f"`{name}` was trained on. Nothing was written.")
    trained_on = [row["problem_id"] for row in _rows(directory / ladder_l4.ROWS_FILE)]
    if trained_on != [row["problem_id"] for row in file_rows]:
        raise RuntimeError(f"{directory.name} stored {len(trained_on)} rows `{name}` was trained on and {ladder_l4.WHAT} holds {len(file_rows)}, or they are not the same rows in "
                           "the same order. Nothing was written.")
    return [{"problem_id": row["problem_id"], "kind": row["kind"], "proof_lines": row["proof_lines"], "lines": proof_line_count(row["proof"])} for row in file_rows], sha256


def examples_of(set_rows: Sequence[Mapping], sha256: str) -> list[dict]:
    """A stored training set as the round's training examples, in its order. A rehearsal row's text is read from the
    package's pretraining file, held to the SHA-256 the stage's prepare step recorded: a run trains on the file it
    prepared. A set with no rehearsal row reads no file."""
    if not any(row["origin"] == PRETRAINING for row in set_rows):
        return [{field: row[field] for field in EXAMPLE_FIELDS} for row in set_rows]
    file_rows, found = _file_rows(ladder_l4.pretraining_file(), ladder_l4.WHAT, ladder_l4.HOW)
    if found != sha256:
        raise RuntimeError(f"{ladder_l4.WHAT} has SHA-256 {found} and this run was prepared on {sha256}: a run trains on the file its "
                           "prepare step checked")
    by_problem = {row["problem_id"]: row for row in file_rows}
    return [training_example(by_problem[row["problem_id"]]) if row["origin"] == PRETRAINING else {field: row[field] for field in EXAMPLE_FIELDS} for row in set_rows]
