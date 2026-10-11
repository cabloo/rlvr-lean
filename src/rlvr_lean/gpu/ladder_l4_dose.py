"""L4p: how much more pretraining? The dose of a second pass. Spec: docs/spec/ladder-loop.spec.md, "L4p: how much
more pretraining? The dose of a second pass" and its "Made exact by the build". Everything it makes is labelled
pretrained on published proofs. Each step is its own process (`rlvr_lean.runner.entry`, stage `ladder_l4_dose_r64`).

A LABELLED CHECK OF THE PRETRAINING: NO ROUND, AND NO ROW OF A ROUND. It runs from ONE start, a model of the check of the
adapter's rank (`ladder_loop.l4.dose.start`: `pre_r64`, which must be the start L4t2 ran from), every model at ITS rank.
Two runs are only READ, on the box:

  L4t2's from that start      `ladder_l4_rows2_<start>_seed<N>`, FINISHED: the stored training sets of its two models
                              (`rehearse_only`, `reward_rehearse`), each one's per-problem rows on the three sets it was
                              measured on and what it wrote, the problems of those sets, the held-out groups, G', and
                              what its own report read of each model
  the start model's           the run of the check that made it: the adapter, the rank and alpha it recorded, its
                              sampling seeds, its per-problem rows, and the rows it was pretrained on

L4t's run is NOT read: what this stage needs of it (the problems, the groups, G') L4t2's run stored after holding it to
L4t's, and G' is made again here from the start model's own rows. The run directory is its own
(`ladder_l4_dose_<start>_seed<N>`).

  ladder_l4_dose_prepare               REFUSES, naming the task to run, a run that lacks what is read, L4t2's run of
                                       another seed or start, a report of it that is not to be read, stored rows that
                                       would not pair by problem, a G' or a pretraining file that is not L4t2's, and a set
                                       that is not along the one order. BUILDS and stores the three training sets
                                       (`domain/ladder_round/l4_dose.py`). No GPU, no Lean
  ladder_l4_dose_train_matched_1262    L4t's OWN two step bodies (`ladder_l4_rows.train_a_model`, `measure_a_model`),
  ladder_l4_dose_measure_matched_1262  under this stage's names: FROM THE START ADAPTER at its rank, one pass, the arm's
  ladder_l4_dose_train_second_half     recipe, the adapter kept under `adapters/<model>` and read back; then the model as
  ladder_l4_dose_measure_second_half   L4t's were measured (8 episodes on the three rungs, G's samplings, the sampling
  ladder_l4_dose_train_second_full     seeds the start model's run recorded). No measurement at another temperature.
  ladder_l4_dose_measure_second_full   `matched_1262` FIRST: its read is the one that is waited for
  ladder_l4_dose_report_matched        the report with `matched_1262` ALONE measured (the stage
                                       `ladder_l4_dose_r64_matched`: a job of its own, in the SAME run directory): the
                                       rounds' hard rows beside the same published rows; the dose's primary NOT READ
  ladder_l4_dose_report                the read fixed before the run (`reporting/ladder_l4_dose.py`): each trained
                                       model's four checks, the rounds' hard rows, the primary and its branch, the
                                       overfitting reads, the dose. It needs all three models; a model the other job
                                       already trained and measured is found done

Every step is resumable and a step of its own: a failure of a later one keeps the earlier ones. A report in which Lean
gave no verdict on too much of a set fails its step, and the task queued again measures that set, and no other, again
from the kept adapter. NO STEP DELETES AN ADAPTER.

EVERY ROW IS A PUBLISHED PROOF. Its text stays in the pretraining file (`data/ladder_l4/pretraining.jsonl`, not
distributed): a stored training set names each by its id, its problem and its place in the file, and the training step
reads the text again from the file, held to the SHA-256 the start model's run recorded (which must be the one L4t2's
run was prepared on).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from rlvr_lean.domain.ladder_round.l4 import goal_set_again
from rlvr_lean.domain.ladder_round.l4_dose import MATCHED, MODELS, SECOND_FULL, SECOND_HALF, STORED, dose_sets
from rlvr_lean.domain.ladder_round.l4_rows import refuse_barred
from rlvr_lean.domain.ladder_round.l4_rows2 import REHEARSE_ONLY, REWARD_REHEARSE, refuse_text
from rlvr_lean.domain.ladder_round.l4b import rehearsal_of
from rlvr_lean.domain.ladder_round.read import GOAL
from rlvr_lean.gpu import ladder_assembly, ladder_ceiling, ladder_l2, ladder_l4, ladder_l4_rows, ladder_l4_rows2, ladder_l4_start, ladder_loop
from rlvr_lean.gpu.ladder_ceiling import RUNG_PART, Reading, _need, goal_set, rung_set, stored_file
from rlvr_lean.gpu.ladder_dose import _rows, _runs
from rlvr_lean.gpu.ladder_l3d1 import WROTE
from rlvr_lean.gpu.ladder_l4 import L4, LABEL, _say
from rlvr_lean.gpu.ladder_l4_rows import described, measure_a_model, read_the_trained, train_a_model
from rlvr_lean.gpu.ladder_l4_start import Start, config_from, start_set
from rlvr_lean.gpu.ladder_round import _stand_in, _write_json, training_seed
from rlvr_lean.infrastructure.artifact_store import ArtifactStore

L4P = "l4_dose"                                             # what this stage's own files are called, beside L4t's (`l4_rows_...`) and L4t2's (`l4_rows2_...`)
STAGE = "ladder_l4_dose"                                    # what its steps and its run directory begin with
ENTRY_STAGE = f"{STAGE}_r64"                                # the stage, as `runner.entry` names it: it is specified from the rank-64 model alone
RUN_VARIABLE = "RLVR_LEAN_LADDER_L4_DOSE_RUN"               # another run directory than `ladder_l4_dose_<start>_seed<seed>` (a smoke run)
L4T2_RUN_VARIABLE = "RLVR_LEAN_LADDER_L4_DOSE_L4T2_RUN"     # another run directory of L4t2 than `ladder_l4_rows2_<start>_seed<seed>` (a smoke run: L4t2's smoke run)
SETTING = "ladder_loop.l4.dose"
THE_STAGE = ladder_l4_rows.Stage(name=ENTRY_STAGE, steps=STAGE, files=L4P, reader="L4p", start_said_by=f"{SETTING}.start")
PREPARE, REPORT = THE_STAGE.prepare, f"{STAGE}_report"
REPORT_MATCHED = f"{STAGE}_report_matched"                  # the report with `matched_1262` alone measured: the last step of the stage `ladder_l4_dose_r64_matched`
MATCHED_STAGE = f"{ENTRY_STAGE}_matched"                    # ... that stage: prepare, `matched_1262` trained and measured, that report; the same run directory
REPORT_FILE = "report_ladder_l4_dose.json"
STORED_MODELS_FILE = f"{L4P}_stored_models.json"            # what the start model and L4t2's two models wrote, from their own runs' markers
NEVER_SOLVED_FILE = f"{L4P}_never_solved.jsonl"             # GIVEN, not made here: one `problem_id` a row, the goal problems nothing stored has ever solved
PLAN = ladder_l4_rows2.PLAN                                 # how a model is measured, as a prepare step records it
READER = ladder_l4_start.Reader(reads="L4p makes no round and no pretraining of its own: it reads what", carry="L4p")
READING = Reading(reader="L4p", models="L4p's models", not_read="that set stands on the other side of a comparison, and no report of this run could be read against it",
                  setting=ladder_l4.SETTING)


@dataclass(frozen=True)
class Dose:
    """`ladder_loop.l4.dose`, as the stage's steps read it."""
    start: str                          # the ONE start adapter: every model here is trained from it, at its rank
    models: tuple[str, ...]             # the three trained models, in the order they are trained and measured
    l4t2: ladder_l4_rows2.Rows2         # L4t2's settings as ITS task reads them: its finished run is what this stage reads


def the_settings(config: dict) -> Dose:
    """THE one place that reads `ladder_loop.l4.dose`. Refused (ValueError): models that are not the stage's (its steps
    are registered for them: change both together); a start that is not the one L4t2 ran from (`rehearse_only` is the
    smallest dose, and the dose is of ONE start model); L4t2's own settings, when its own reader refuses them."""
    settings = config["ladder_loop"]["l4"]["dose"]
    start, models = settings["start"], tuple(settings["models"])
    if models != MODELS:
        raise ValueError(f"{SETTING}.models is {list(models)} and the stage trains and measures {list(MODELS)}: change both together")
    l4t2 = ladder_l4_rows2.the_settings(config)
    if start != l4t2.start:
        raise ValueError(f"{SETTING}.start is `{start}` and {ladder_l4_rows2.SETTING}.start is `{l4t2.start}`: `{REHEARSE_ONLY}`, the smallest dose, is L4t2's model, "
                         "trained from that start, and the dose is of ONE start model")
    return Dose(start=start, models=models, l4t2=l4t2)


def run_name(dose: Dose, seed: int) -> str:
    """The stage's run directory, with the start adapter's name in it as L4t2's has: `ladder_l4_dose_pre_r64_seed<N>`."""
    return f"{STAGE}_{dose.start}_seed{seed}"


def _store(config: dict, dose: Dose) -> ArtifactStore:
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    return ArtifactStore(_runs(config) / (os.environ.get(RUN_VARIABLE) or run_name(dose, training_seed(config))), Path(mirror) if mirror else None)


def l4t2_directory(config: dict, dose: Dose) -> Path:
    """L4t2's run from this stage's start, by L4t2's own name for it (or the run the task names: a smoke run). Only
    read: nothing is created."""
    return _runs(config) / (os.environ.get(L4T2_RUN_VARIABLE) or ladder_l4_rows2.run_name(dose.l4t2, training_seed(config)))


def the_start(config: dict, dose: Dose) -> Start:
    """The start model the setting names, and where it lives (`ladder_l4_start.the_start`: a model of the check of the
    adapter's rank, in that check's run). A name that is no such model is refused (RuntimeError)."""
    return ladder_l4_start.the_start(config, dose.start, THE_STAGE.start_said_by)


def l4t2_task(seed: int) -> str:
    """The task that writes L4t2's run, for a refusal."""
    stage = ladder_l4_rows2.ENTRY_STAGE
    return f"the task of stage `{stage}` for seed {seed} (`python -m rlvr_lean.runner.entry --stage {stage} --seeds {seed}`; for the smoke run, stage `{stage}_smoke`)"


def l4t2_set(part: str, model: str) -> str:
    """The set a model of L4t2 was measured under, in L4t2's own run directory."""
    return rung_set(model, L4) if part == RUNG_PART else goal_set(part, model, L4)


# ------------------------------------------------------------------------------------------------- prepare
def build_the_sets(rehearse_only: list[dict], reward_rehearse: list[dict], pretraining: list[dict], seed: int) -> dict:
    """The three training sets, from the two sets L4t2's run stored (`rehearse_only`, `reward_rehearse`: each file's rows
    in its order) and the rows the start model was pretrained on (`pretraining`: in the file's order, without their
    text). Each row is numbered by its place in ITS set. Refused (ValueError) as the rule refuses
    (`l4_dose.dose_sets`), and when a row carries a published proof's text (`l4_rows2.refuse_text`)."""
    numbered = lambda rows: [{"row": position, **{key: value for key, value in row.items() if key != "row"}} for position, row in enumerate(rows)]      # noqa: E731
    sets = {model: numbered(own) for model, own in dose_sets(rehearse_only, reward_rehearse, pretraining, seed).items()}
    for model, own in sets.items():
        refuse_text(own, f"the training set of `{model}`")
    return sets


def ladder_l4_dose_prepare(config: dict) -> dict:
    dose = the_settings(config)
    store = _store(config, dose)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    seed, of_the_pretraining = training_seed(config), ladder_l2.pretraining_seed(config)
    start, l4t2, of_l4t2_stage = the_start(config, dose), l4t2_directory(config, dose), ladder_l4_rows2.THE_STAGE
    task = l4t2_task(seed)

    def need(directory: Path, names, what: str, whose: str) -> None:
        lacking = [name for name in names if not (directory / name).exists()]
        if lacking:
            raise RuntimeError(f"{directory} does not hold {lacking}: {what}. {READER.reads} {whose} wrote on this box. Run that task to its end first; nothing was written.")

    # ---- L4t2's run: FINISHED, of this seed and this start, with a report that can be read
    need(l4t2, (f"{ladder_l4_rows2.PREPARE}.done.json", f"{ladder_l4_rows2.REPORT}.done.json", ladder_l4_rows2.REPORT_FILE, "problems.jsonl", ladder_l4.GROUPS_FILE,
                ladder_l4.AGAIN_FILE, *(name for model in STORED for name in (f"{of_l4t2_stage.measure_marker(model)}.done.json", of_l4t2_stage.training_file(model)))),
         "what L4t2's steps recorded, its report, its problems, its held-out groups, its G', what one of its two models wrote, or one of its two stored training sets", task)
    of_l4t2 = json.loads((l4t2 / f"{ladder_l4_rows2.PREPARE}.done.json").read_text())
    if of_l4t2.get("seed") != seed or of_l4t2.get("start") != dose.start or not set(STORED) <= set(of_l4t2.get("models") or ()):
        raise RuntimeError(f"L4t2's run {l4t2.name} recorded seed {of_l4t2.get('seed')}, the start {of_l4t2.get('start')!r} and the models {of_l4t2.get('models')}; this "
                           f"task's seed is {seed}, {SETTING} names the start {dose.start!r}, and {list(STORED)} are the two models read from it: the sets and the rows read "
                           "here are that run's. Nothing was written.")
    report = json.loads((l4t2 / ladder_l4_rows2.REPORT_FILE).read_text())
    if not report.get("ok", True):
        raise RuntimeError(f"the report of L4t2's run {l4t2.name} is not to be read (Lean gave no verdict on too much of a set): its models' rows cannot stand in a "
                           f"comparison. Queue {task} again, which measures that set again; nothing was written.")
    # What L4t2's OWN report read of each of its two models: when it read one as INCONCLUSIVE (a check of its own failed) nothing is read against it here either.
    failed_in_l4t2 = {model: [check for check, entry in ((report.get("models") or {}).get(model, {}).get("can_this_model_be_read") or {}).items()
                              if isinstance(entry, dict) and not entry.get("passes")] for model in STORED}

    # ---- the start model's run: what L4t2 read of it, read again; a smoke run goes on from failed checks only where L4t2's own run recorded that IT did
    waived = bool((of_l4t2.get("start_run_read") or {}).get("checks_waived_as_l4ts_run_recorded"))
    of_the_start_run = ladder_l4_start.read_the_start_run(start, of_the_pretraining, waived=waived, reader=READER)
    of_start = of_the_start_run["prepared"]
    recorded, of_l4t2s = ladder_l4_start.recipe_recorded(of_start), of_l4t2.get("start_recipe") or {}
    if (recorded["rank"], recorded["alpha"]) != (of_l4t2s.get("rank"), of_l4t2s.get("alpha")):
        raise RuntimeError(f"the run {start.directory.name} recorded an adapter of rank {recorded['rank']} and alpha {recorded['alpha']} for `{start.name}`, and L4t2's run "
                           f"{l4t2.name} trained its models at rank {of_l4t2s.get('rank')} and alpha {of_l4t2s.get('alpha')}: every model here is at the rank L4t2's "
                           "were. Nothing was written.")
    plan, of_l4t2s_plan = {key: of_start[key] for key in PLAN}, {key: of_l4t2.get(key) for key in PLAN}
    if plan != of_l4t2s_plan:
        raise RuntimeError(f"the run {start.directory.name} recorded how `{start.name}` was measured ({plan}), and L4t2's run {l4t2.name} measured its models otherwise "
                           f"({of_l4t2s_plan}): {READING.models} would not pair with both. Nothing was written.")
    parts = ladder_l4.pre_parts(of_start)
    need(start.directory, [ladder_l4.ROWS_FILE, *(name for part in parts for name in (f"episodes_{start_set(part, start)}_problems.jsonl", f"episodes_{start_set(part, start)}.done.json"))],
         f"the rows `{start.name}` was pretrained on (every row trained on here is one of them), its per-problem rows on a set it was measured on, or that set's own marker",
         start.task)
    need(l4t2, [name for part in parts for model in STORED for name in (f"episodes_{l4t2_set(part, model)}_problems.jsonl", f"episodes_{l4t2_set(part, model)}.done.json")],
         "the per-problem rows of one of L4t2's two models on a set it was measured on, or that set's own marker", task)
    if not ladder_ceiling._stand_in() and not start.adapter.is_dir():
        raise RuntimeError(f"{start.adapter} is not there: the adapter `{dose.start}` is what every model here is trained from. Run {start.task} again; nothing was written.")

    # ---- the stored rows every read here stands on: the start model's and L4t2's two models', held to the same problems, seeds and attempts
    problems = _rows(l4t2 / "problems.jsonl")
    rung_problems = [row for row in problems if row["set"] == l4t2_set(RUNG_PART, REHEARSE_ONLY)]
    goal_problems = [row for row in problems if row["set"] == l4t2_set(plan["goal_samplings"][0]["name"], REHEARSE_ONLY)]
    ids = {RUNG_PART: [row["problem_id"] for row in rung_problems], GOAL: [row["problem_id"] for row in goal_problems]}
    measured = [(RUNG_PART, RUNG_PART, plan["rung_episodes"], plan["sampling_seeds"][RUNG_PART]),
                *((sampling["name"], GOAL, sampling["episodes"], sampling["sampling_seed"]) for sampling in plan["goal_samplings"])]
    kept = {}
    for part, of, episodes, sampling_seed in measured:
        kept[dose.start, part] = ladder_ceiling._stored(start.directory, start_set(part, start), ids[of], sampling_seed, episodes, f"`{dose.start}` ({part})", READING)[0]
        for model in STORED:
            kept[model, part] = ladder_ceiling._stored(l4t2, l4t2_set(part, model), ids[of], sampling_seed, episodes, f"`{model}` ({part})", READING)[0]
    # G' is the START MODEL's, made by the pretraining's rule from its first sampling of G, as L4t and L4t2 made it: and it must be the one L4t2's run stored.
    again = goal_set_again(kept[dose.start, plan["goal_samplings"][0]["name"]], ids[GOAL])
    if again != [row["problem_id"] for row in _rows(l4t2 / ladder_l4.AGAIN_FILE)]:
        raise RuntimeError(f"G' made from the rows of `{dose.start}` ({start.directory.name}: {len(again)} goal problems it does not solve in its first sampling) is not the "
                           f"G' L4t2's run {l4t2.name} stored: the two runs are not of one start model's rows. Nothing was written.")

    # ---- the three sets: from the two sets L4t2's run stored and the rows the start model was pretrained on (the file's text is read, and stays in the file)
    pretraining, sha256 = ladder_l4_start.pretrained_on(start.directory, of_start["pretraining_file_sha256"], start.name)      # refused unless the pretraining file is the one it was trained on
    if sha256 != of_l4t2.get("pretraining_file_sha256"):
        raise RuntimeError(f"L4t2's run {l4t2.name} was prepared on a pretraining file of SHA-256 {of_l4t2.get('pretraining_file_sha256')} and `{start.name}` was trained on "
                           f"{sha256}: L4t2's sets name each published row by its place in the file it was prepared on. Nothing was written.")
    rehearse_only, reward_rehearse = (_rows(l4t2 / of_l4t2_stage.training_file(model)) for model in STORED)
    batch = config["training"]["effective_batch"]
    try:
        sets = build_the_sets(rehearse_only, reward_rehearse, pretraining, seed)
        groups = _rows(l4t2 / ladder_l4.GROUPS_FILE)
        heldout = {row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[0]} | {row["problem_id"] for row in groups}
        half_seed = config["ladder_loop"]["l4"]["half_seed"]
        counted = {model: refuse_barred(sets[model], heldout, half_seed, f"the training set of `{model}`") for model in dose.models}
    except ValueError as error:
        raise RuntimeError(f"{error} (L4t2's run {l4t2.name}; `{start.name}`'s {start.directory.name}). Nothing was written.") from error

    # ---- from here on files are written: the problems of every set this run samples, what its report reads of the stored models, the training sets
    def as_set(own: list[dict], set_name: str) -> list[dict]:
        return [{**row, "set": set_name} for row in own]

    own = [row for model in dose.models for row in as_set(rung_problems, rung_set(model, L4))]
    own += [row for model in dose.models for sampling in plan["goal_samplings"] for row in as_set(goal_problems, goal_set(sampling["name"], model, L4))]
    store.write_rows("problems.jsonl", own)             # every set of it is this run's
    store.write_rows(ladder_l4.GROUPS_FILE, groups)
    store.write_rows(ladder_l4.AGAIN_FILE, [{"problem_id": problem_id} for problem_id in again])
    for (who, part), own_rows in kept.items():
        store.write_rows(stored_file(who, part, L4), own_rows)
    wrote = {dose.start: of_the_start_run["measured"], **{model: json.loads((l4t2 / f"{of_l4t2_stage.measure_marker(model)}.done.json").read_text()) for model in STORED}}
    _write_json(store, STORED_MODELS_FILE, {"stage": L4, "label": LABEL, **{who: {key: entry[key] for key in WROTE} for who, entry in wrote.items()}})
    for model in dose.models:
        store.write_rows(THE_STAGE.training_file(model), sets[model])
    order = (f"the hash order L4t's rehearsal draw uses (a content hash of the task's seed and each row's id among the rows `{dose.start}` was pretrained on); trained in the "
             "arm's content-hash order. Every row is a published proof of the `pretrain` half: its text stays in the pretraining file")
    published = len(rehearsal_of(reward_rehearse))
    summary = {
        "stage": L4, "label": LABEL, "check": "L4p", "seed": seed, "start": dose.start, "start_adapter": str(start.adapter), "start_run": str(start.directory),
        "l4t2_run": str(l4t2), "models": list(dose.models), "stand_in_engine": _stand_in(), "effective_batch": batch,
        "rows_the_start_was_pretrained_on": len(pretraining),
        "rehearse_only": {"what": f"L4t2's model, a dose of its own: stored, read and never trained again. Its rows are the first of the hash order `{SECOND_HALF}`'s and "
                                  f"`{SECOND_FULL}`'s are taken along, held by id and by place in the file", "file": of_l4t2_stage.training_file(REHEARSE_ONLY),
                          "rows": len(rehearse_only), "optimizer_steps": -(-len(rehearse_only) // batch), "run": str(l4t2)},
        "reward_rehearse": {"what": f"L4t2's model, stored and read: the rounds' hard rows and as many published rows as it has one-shot rows. `{MATCHED}` is trained on those "
                                    "published rows and nothing else, held by id and by place in the file: its control", "file": of_l4t2_stage.training_file(REWARD_REHEARSE),
                            "rows": len(reward_rehearse), "published_rows": published, "rows_of_the_rounds": len(reward_rehearse) - published, "run": str(l4t2)},
        "training_sets": {
            MATCHED: {"what": f"the first {published:,} rows of {order}. They are EXACTLY the published rows of L4t2's `{REWARD_REHEARSE}` set, in that set's order with its "
                              "rounds' rows left out", "file": THE_STAGE.training_file(MATCHED), **described(sets[MATCHED], batch),
                      "published_rows_of_reward_rehearse": published, "barred": counted[MATCHED]},
            SECOND_HALF: {"what": f"the first half of {order}", "file": THE_STAGE.training_file(SECOND_HALF), **described(sets[SECOND_HALF], batch),
                          "rows_of_rehearse_only_they_begin_with": len(rehearse_only), "barred": counted[SECOND_HALF]},
            SECOND_FULL: {"what": f"every row of the pretraining file, the draw of every row along {order}", "file": THE_STAGE.training_file(SECOND_FULL),
                          **described(sets[SECOND_FULL], batch), "rows_of_second_half_they_begin_with": len(sets[SECOND_HALF]), "barred": counted[SECOND_FULL]}},
        "pretraining_file": str(ladder_l4.pretraining_file()), "pretraining_file_sha256": sha256,
        "recipe": {"what": "L4t's (`ladder_ceiling.one_pass`, as the arm's twin is trained): FROM THE START ADAPTER, one pass, the native format, the content-hash order",
                   "learning_rate": config["training"]["learning_rate"], "warmup_steps": config["training"]["warmup_steps"], "effective_batch": batch,
                   "lora": config_from(config, {"start_recipe": recorded}, start)["lora"], "adapter_seed": seed},
        "goal_set": len(goal_problems), "goal_set_again": len(again), "rung_problems": len(rung_problems), **plan,
        "attempts_a_goal_problem": sum(sampling["episodes"] for sampling in plan["goal_samplings"]), "parts": parts,
        "stored_models": {dose.start: {"run": str(start.directory), "sets": {part: start_set(part, start) for part in parts}},
                          **{model: {"run": str(l4t2), "sets": {part: l4t2_set(part, model) for part in parts}} for model in STORED}},
        "never_solved_file": NEVER_SOLVED_FILE, "contradicted_side_setting": config["ladder_loop"]["episode"].get("contradicted_side", "all"),
        "start_recipe": {"what": f"the adapter of `{dose.start}`, as the run that made it recorded it and as L4t2's run trained its models: EVERY training here attaches "
                                 f"this rank and alpha (not the config's own, {config['lora']['rank']} and {config['lora']['alpha']}), the adapter each saves is read back and "
                                 "held to them, and the model server of this stage's measure steps is started with this rank as its largest",
                         **recorded, "check": of_start["rank_check"].get("check"), "stage": of_start["rank_check"].get("stage"),
                         "adapter_saved": of_the_start_run["trained"].get("adapter_saved"), "rank_of_the_config": config["lora"]["rank"], "alpha_of_the_config": config["lora"]["alpha"]},
        "start_run_read": {"branch": (of_the_start_run["report"].get("branch") or {}).get("name"), "headline": of_the_start_run["report"].get("headline"),
                           "pretraining_file_sha256": of_start["pretraining_file_sha256"], "rows": of_start["rows"],
                           "checks_failed": of_the_start_run["checks_failed"], "checks_waived_as_l4t2s_run_recorded": of_the_start_run["checks_waived"]},
        "l4t2_read": {"what": f"L4t2's finished run from `{dose.start}`, only read: the stored training sets of its two models, their per-problem rows and what each wrote. "
                              "`failed_checks`: by model, the checks ITS report read as failed; such a model is INCONCLUSIVE there, and nothing is read against it here either",
                      "run": str(l4t2), "headline": report.get("headline"), "branches": report.get("branches"), "failed_checks": failed_in_l4t2}}
    store.mark_done(PREPARE, summary)
    of = summary["training_sets"]
    _say(f"L4p prepared, seed {seed}, from `{dose.start}` along the order of {l4t2.name}'s `{REHEARSE_ONLY}` ({len(rehearse_only):,} rows): `{MATCHED}` "
         f"{of[MATCHED]['rows']:,} rows (the published rows of `{REWARD_REHEARSE}`'s {len(reward_rehearse):,}), {of[MATCHED]['optimizer_steps']:,} optimizer steps; `{SECOND_HALF}` "
         f"{of[SECOND_HALF]['rows']:,} rows, {of[SECOND_HALF]['optimizer_steps']:,} optimizer steps; `{SECOND_FULL}` {of[SECOND_FULL]['rows']:,} rows (every row `{dose.start}` "
         f"was pretrained on), {of[SECOND_FULL]['optimizer_steps']:,} optimizer steps; every model at rank {recorded['rank']}; G' of `{dose.start}` holds {len(again)} goal problems")
    return summary


# ----------------------------------------------------------------------------------------- train and measure
def ladder_l4_dose_train(config: dict, model: str) -> dict:
    """One of the three models, by L4t's own training body (`ladder_l4_rows.train_a_model`) under this stage's names."""
    dose = the_settings(config)
    return train_a_model(config, _store(config, dose), model, dose.start, run_name(dose, training_seed(config)), THE_STAGE)


def ladder_l4_dose_measure(config: dict, model: str) -> dict:
    """One of the three models, by L4t's own measuring body (`ladder_l4_rows.measure_a_model`) under this stage's names."""
    dose = the_settings(config)
    return measure_a_model(config, _store(config, dose), model, dose.start, THE_STAGE)


# -------------------------------------------------------------------------------------------------- report
def write_the_report(config: dict, needed: tuple[str, ...], marker: str, stage: str) -> dict:
    """The report of the run, on EVERY model of the stage that is trained and measured by now. `needed`: the models this
    step asks for, each refused (RuntimeError) when its steps are not done: all three for the stage's own report,
    `matched_1262` alone for the report of the job that trains and measures it alone. A model that is not asked for and
    is done is read all the same (that job queued again after the whole stage ran writes the whole report); one that is
    not done is NOT MEASURED YET, and what stands on it is NOT READ. `marker`: the step, `stage`: the stage it is the
    last step of, for a refusal."""
    from rlvr_lean.gpu.ladder_l3c import proof_lengths
    from rlvr_lean.reporting.ladder_l4_dose import build_dose_report

    dose = the_settings(config)
    store = _store(config, dose)
    for step in (PREPARE, *(of(model) for model in needed for of in (THE_STAGE.train_marker, THE_STAGE.measure_marker))):
        _need(store, step, "L4p's report", stage)
    measured = tuple(model for model in dose.models if store.is_done(THE_STAGE.train_marker(model)) and store.is_done(THE_STAGE.measure_marker(model)))
    prepared = store.done_summary(PREPARE)
    samplings, of_the_stored = prepared["goal_samplings"], json.loads(store.path(STORED_MODELS_FILE).read_text())
    stored = {who: {RUNG_PART: store.read_rows(stored_file(who, RUNG_PART, L4)), GOAL: [store.read_rows(stored_file(who, sampling["name"], L4)) for sampling in samplings],
                    **{key: of_the_stored[who][key] for key in WROTE}} for who in (dose.start, *STORED)}
    models, trains, losses, trained = read_the_trained(store, prepared, measured, THE_STAGE)
    # The goal problems nothing stored has ever solved are GIVEN (a file of ids in the run directory), never computed here.
    never = [row["problem_id"] for row in store.read_rows(NEVER_SOLVED_FILE)] if store.path(NEVER_SOLVED_FILE).exists() else None
    # The length of a held-out problem's published proof is read HERE, by the report, and by nothing before it.
    report = build_dose_report(prepared, trains, losses, trained, store.read_rows(ladder_l4.GROUPS_FILE), proof_lengths(), stored, models,
                               [row["problem_id"] for row in store.read_rows(ladder_l4.AGAIN_FILE)], config["ladder_loop"]["l4"], config["evaluation"], never)
    directory = store.root / ladder_assembly.ADAPTERS
    report["adapters"] = {"kept": True, "what": f"the adapters of {' and '.join(f'`{model}`' for model in measured)} ({LABEL}) are KEPT: no step of this stage deletes "
                                                f"one. `{dose.start}` is its own run's and was only read; L4t2's two are its run's and were not read at all",
                          "directory": str(directory), "there": sorted(path.name for path in directory.glob("*") if path.is_dir()) if directory.is_dir() else []}
    _write_json(store, REPORT_FILE, report)
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    for name in (STORED_MODELS_FILE, *(THE_STAGE.loss_file(model) for model in measured)):
        store.mirror(name)
    for line in report["lines"]:
        print(line, flush=True)
    store.mark_done(marker, {"stage": L4, "label": LABEL, "check": "L4p", "headline": report["headline"], "branches": report["branches"], "measured": list(measured),
                             "ok": report["ok"]})
    return report


def ladder_l4_dose_report(config: dict) -> dict:
    """The stage's report: all three models trained and measured."""
    return write_the_report(config, MODELS, REPORT, ENTRY_STAGE)


def ladder_l4_dose_report_matched(config: dict) -> dict:
    """The report of the job that trains and measures `matched_1262` alone: the two other models are not asked for."""
    return write_the_report(config, (MATCHED,), REPORT_MATCHED, MATCHED_STAGE)


def _for(step, name: str):
    def run(config: dict) -> dict:
        return step(config, name)
    return run


# In the order the stage runs them: each model trained, then measured, before the next is trained.
STEPS = {PREPARE: ladder_l4_dose_prepare}
for _model in MODELS:
    STEPS[THE_STAGE.train_marker(_model)] = _for(ladder_l4_dose_train, _model)
    STEPS[THE_STAGE.measure_marker(_model)] = _for(ladder_l4_dose_measure, _model)
STEPS[REPORT] = ladder_l4_dose_report
STEPS[REPORT_MATCHED] = ladder_l4_dose_report_matched       # the last step of the stage `ladder_l4_dose_r64_matched`, after `matched_1262`'s two
