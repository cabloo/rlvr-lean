"""L4t2: two more one-change checks, on what L4t left open. Spec: docs/spec/ladder-loop.spec.md, "L4t2: two more
one-change checks, on what L4t left open" and its "Made exact by the build". Everything it makes is labelled pretrained on
published proofs. Each step is its own process (`rlvr_lean.runner.entry`, stage `ladder_l4_rows2_r64`).

NO NEW ROUND, AND NO NEW DRAW OF THE ROUNDS' ROWS. It runs from ONE start, a model of the check of the adapter's rank
(`ladder_loop.l4.rows2.start`: `pre_r64`), every model at ITS rank. Two runs are only READ, on the box:

  L4t's from that start       `ladder_l4_rows_<start>_seed<N>`, FINISHED: its two stored training sets (`rehearse` and
                              `reward_rows`), the per-problem rows of its three measured models (`old_rule`,
                              `reward_rows`, `rehearse`) with what each wrote, its problems, its held-out groups, its G',
                              and what its own report read of each model
  the start model's           the run of the check that made it: the adapter, the rank and alpha it recorded, its
                              sampling seeds, its per-problem rows, and the rows it was pretrained on

The run directory is its own (`ladder_l4_rows2_<start>_seed<N>`).

  ladder_l4_rows2_prepare                  REFUSES, naming the task to run, a run that lacks what is read, L4t's run of
                                           another seed or start, a report of it that is not to be read, stored rows
                                           that would not pair by problem, a G' or a pretraining file that is not L4t's,
                                           and a training set with a barred problem. BUILDS and stores the two training
                                           sets (`domain/ladder_round/l4_rows2.py`). No GPU, no Lean
  ladder_l4_rows2_train_rehearse_only      L4t's OWN two step bodies (`ladder_l4_rows.train_a_model`, `measure_a_model`),
  ladder_l4_rows2_measure_rehearse_only    under this stage's names: FROM THE START ADAPTER at its rank, one pass, the
  ladder_l4_rows2_train_reward_rehearse    arm's recipe, the adapter kept under `adapters/<model>` and read back; then
  ladder_l4_rows2_measure_reward_rehearse  the model as L4t's were measured (8 episodes on the three rungs, G's samplings,
                                           the sampling seeds the start model's run recorded). No measurement at
                                           another temperature
  ladder_l4_rows2_report                   the read fixed before the run (`reporting/ladder_l4_rows2.py`): each trained
                                           model's four checks, the three reads of the five models, the two questions,
                                           the arm's rule

Every step is resumable and a step of its own: a failure of a later one keeps the earlier ones. A report in which Lean
gave no verdict on too much of a set fails its step, and the task queued again measures that set, and no other, again
from the kept adapter. NO STEP DELETES AN ADAPTER.

THE REHEARSAL ROWS ARE PUBLISHED PROOFS. Their text stays in the pretraining file (`data/ladder_l4/pretraining.jsonl`,
not distributed): a stored training set names each by its id, its problem and its place in the file, and the training
step reads the text again from the file, held to the SHA-256 the start model's run recorded (which must be the one
L4t's run was prepared on).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from rlvr_lean.domain.ladder_round.assembly import ATTEMPT
from rlvr_lean.domain.ladder_round.l4 import PRE, goal_set_again
from rlvr_lean.domain.ladder_round.l4_rows import REHEARSE, REWARD_ROWS, refuse_barred
from rlvr_lean.domain.ladder_round.l4_rows2 import MODELS, REHEARSE_ONLY, REWARD_REHEARSE, STORED, refuse_text, rehearse_only_set, reward_rehearse_set
from rlvr_lean.domain.ladder_round.l4b import rehearsal_of
from rlvr_lean.domain.ladder_round.read import GOAL
from rlvr_lean.gpu import ladder_assembly, ladder_ceiling, ladder_l2, ladder_l4, ladder_l4_rank, ladder_l4_rows, ladder_l4_start, ladder_loop
from rlvr_lean.gpu.ladder_ceiling import RUNG_PART, Reading, _need, goal_set, rung_set, stored_file
from rlvr_lean.gpu.ladder_dose import _rows, _runs
from rlvr_lean.gpu.ladder_l3d1 import WROTE
from rlvr_lean.gpu.ladder_l4 import L4, LABEL, _say
from rlvr_lean.gpu.ladder_l4_rows import described, measure_a_model, read_the_trained, train_a_model
from rlvr_lean.gpu.ladder_l4_start import Start, config_from, start_set
from rlvr_lean.gpu.ladder_round import _stand_in, _write_json, training_seed
from rlvr_lean.infrastructure.artifact_store import ArtifactStore

L4T2 = "l4_rows2"                                           # what this stage's own files are called, beside L4t's (`l4_rows_...`)
STAGE = "ladder_l4_rows2"                                   # what its steps and its run directory begin with
ENTRY_STAGE = f"{STAGE}_r64"                                # the stage, as `runner.entry` names it: it is specified from the rank-64 model alone
RUN_VARIABLE = "RLVR_LEAN_LADDER_L4_ROWS2_RUN"              # another run directory than `ladder_l4_rows2_<start>_seed<seed>` (a smoke run)
L4T_RUN_VARIABLE = "RLVR_LEAN_LADDER_L4_ROWS2_L4T_RUN"      # another run directory of L4t than `ladder_l4_rows_<start>_seed<seed>` (a smoke run: L4t's smoke run from that start)
SETTING = "ladder_loop.l4.rows2"
THE_STAGE = ladder_l4_rows.Stage(name=ENTRY_STAGE, steps=STAGE, files=L4T2, reader="L4t2", start_said_by=f"{SETTING}.start")
PREPARE, REPORT = THE_STAGE.prepare, f"{STAGE}_report"
REPORT_FILE = "report_ladder_l4_rows2.json"
STORED_MODELS_FILE = f"{L4T2}_stored_models.json"           # what the start model and L4t's three wrote, from their own runs' markers
NEVER_SOLVED_FILE = f"{L4T2}_never_solved.jsonl"            # GIVEN, not made here: one `problem_id` a row, the goal problems nothing stored has ever solved
PLAN = ("sampling_seeds", "rung_episodes", "goal_samplings")       # how a model is measured, as a prepare step records it
READER = ladder_l4_start.Reader(reads="L4t2 makes no round and no pretraining: it reads what", carry="L4t2")
READING = Reading(reader="L4t2", models="L4t2's models", not_read="that set stands on the other side of a comparison, and no report of this run could be read against it",
                  setting=ladder_l4.SETTING)


@dataclass(frozen=True)
class Rows2:
    """`ladder_loop.l4.rows2`, as the stage's steps read it."""
    start: str                          # the ONE start adapter: every model here is trained from it, at its rank
    models: tuple[str, ...]             # the two trained models, in the order they are trained and measured
    l4t: ladder_l4_rows.Rows            # L4t's settings as ITS task from that start reads them: its finished run is what this stage reads


def the_settings(config: dict) -> Rows2:
    """THE one place that reads `ladder_loop.l4.rows2`. Refused (ValueError): models that are not the stage's (its
    steps are registered for them: change both together); the start `pre` (L4t from `pre` trains no `old_rule`, and
    the three reads are against a model of the check of the adapter's rank); L4t's own settings from that start, when
    its own reader refuses them."""
    settings = config["ladder_loop"]["l4"]["rows2"]
    start, models = settings["start"], tuple(settings["models"])
    if models != MODELS:
        raise ValueError(f"{SETTING}.models is {list(models)} and the stage trains and measures {list(MODELS)}: change both together")
    if start == PRE:
        raise ValueError(f"{SETTING}.start is `{PRE}`: L4t2 is specified from a model of {ladder_l4_rank.SETTING} alone (`pre_r64`). L4t from `{PRE}` trains no "
                         f"`{STORED[0]}`, and the three reads are against that model")
    return Rows2(start=start, models=models, l4t=ladder_l4_rows.the_settings(config, start))


def run_name(rows: Rows2, seed: int) -> str:
    """The stage's run directory, with the start adapter's name in it as L4t's has: `ladder_l4_rows2_pre_r64_seed<N>`."""
    return f"{STAGE}_{rows.start}_seed{seed}"


def _store(config: dict, rows: Rows2) -> ArtifactStore:
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    return ArtifactStore(_runs(config) / (os.environ.get(RUN_VARIABLE) or run_name(rows, training_seed(config))), Path(mirror) if mirror else None)


def l4t_directory(config: dict, rows: Rows2) -> Path:
    """L4t's run from this stage's start, by L4t's own name for it (or the run the task names: a smoke run). Only read:
    nothing is created."""
    return _runs(config) / (os.environ.get(L4T_RUN_VARIABLE) or ladder_l4_rows.run_name(rows.l4t, training_seed(config)))


def the_start(config: dict, rows: Rows2) -> Start:
    """The start model the setting names, and where it lives (`ladder_l4_start.the_start`: a model of the check of the
    adapter's rank, in that check's run). A name that is no such model is refused (RuntimeError)."""
    return ladder_l4_start.the_start(config, rows.start, THE_STAGE.start_said_by)


def l4t_task(start: Start, seed: int) -> str:
    """The task that writes L4t's run from `start`, for a refusal: its stage carries the start adapter's rank."""
    stage = f"{ladder_l4_rows.STAGE}_r{start.check.rank}"
    return f"the task of stage `{stage}` for seed {seed} (`python -m rlvr_lean.runner.entry --stage {stage} --seeds {seed}`; for the smoke run, stage `{stage}_smoke`)"


def l4t_set(part: str, model: str) -> str:
    """The set a model of L4t was measured under, in L4t's own run directory."""
    return rung_set(model, L4) if part == RUNG_PART else goal_set(part, model, L4)


# ------------------------------------------------------------------------------------------------- prepare
def build_the_sets(rehearse: list[dict], reward_rows: list[dict], pretraining: list[dict], seed: int) -> dict:
    """The two training sets, from the two L4t's run stored (`rehearse`, `reward_rows`: each file's rows in its order)
    and the rows the start model was pretrained on (`pretraining`: in the file's order, without their text). Each row
    is numbered by its place in ITS set; nothing else of a row moves. Refused (ValueError) as the two rules refuse, and
    when a rehearsal row carries a published proof's text (`l4_rows2.refuse_text`)."""
    numbered = lambda rows: [{"row": position, **{key: value for key, value in row.items() if key != "row"}} for position, row in enumerate(rows)]      # noqa: E731
    sets = {REHEARSE_ONLY: numbered(rehearse_only_set(rehearse)), REWARD_REHEARSE: numbered(reward_rehearse_set(reward_rows, rehearse, pretraining, seed))}
    for model, own in sets.items():
        refuse_text(own, f"the training set of `{model}`")
    return sets


def ladder_l4_rows2_prepare(config: dict) -> dict:
    rows = the_settings(config)
    store = _store(config, rows)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    seed, of_the_pretraining = training_seed(config), ladder_l2.pretraining_seed(config)
    start, l4t, of_l4t_stage = the_start(config, rows), l4t_directory(config, rows), ladder_l4_rows.THE_STAGE
    task = l4t_task(start, seed)

    def need(directory: Path, names, what: str, whose: str) -> None:
        lacking = [name for name in names if not (directory / name).exists()]
        if lacking:
            raise RuntimeError(f"{directory} does not hold {lacking}: {what}. {READER.reads} {whose} wrote on this box. Run that task to its end first; nothing was written.")

    # ---- L4t's run: FINISHED, of this seed and this start, with a report that can be read
    need(l4t, (f"{ladder_l4_rows.PREPARE}.done.json", f"{ladder_l4_rows.REPORT}.done.json", ladder_l4_rows.REPORT_FILE, "problems.jsonl", ladder_l4.GROUPS_FILE,
               ladder_l4.AGAIN_FILE, *(f"{of_l4t_stage.measure_marker(model)}.done.json" for model in STORED),
               *(of_l4t_stage.training_file(model) for model in (REHEARSE, REWARD_ROWS))),
         "what L4t's steps recorded, its report, its problems, its held-out groups, its G', what one of its three models wrote, or one of its two stored training sets", task)
    of_l4t = json.loads((l4t / f"{ladder_l4_rows.PREPARE}.done.json").read_text())
    if of_l4t.get("seed") != seed or of_l4t.get("start") != rows.start or tuple(of_l4t.get("models") or ()) != STORED:
        raise RuntimeError(f"L4t's run {l4t.name} recorded seed {of_l4t.get('seed')}, the start {of_l4t.get('start')!r} and the models {of_l4t.get('models')}; this task's "
                           f"seed is {seed}, {SETTING} names the start {rows.start!r}, and from it L4t trains and measures {list(STORED)}: the sets and the rows read here "
                           "are that run's. Nothing was written.")
    report = json.loads((l4t / ladder_l4_rows.REPORT_FILE).read_text())
    if not report.get("ok", True):
        raise RuntimeError(f"the report of L4t's run {l4t.name} is not to be read (Lean gave no verdict on too much of a set): its models' rows cannot stand in a "
                           f"comparison. Queue {task} again, which measures that set again; nothing was written.")
    # What L4t's OWN report read of each of its three models: one it read as INCONCLUSIVE (a check of its own failed) is not read here either.
    failed_in_l4t = {model: [check for check, entry in ((report.get("models") or {}).get(model, {}).get("can_this_model_be_read") or {}).items()
                             if isinstance(entry, dict) and not entry.get("passes")] for model in STORED}

    # ---- the start model's run: what L4t read of it, read again; a smoke run goes on from failed checks only where L4t's own run recorded that IT did
    waived = bool((of_l4t.get("start_run_read") or {}).get("checks_waived_for_a_smoke_run"))
    of_the_start_run = ladder_l4_start.read_the_start_run(start, of_the_pretraining, waived=waived, reader=READER)
    of_start = of_the_start_run["prepared"]
    recorded, of_l4ts = ladder_l4_start.recipe_recorded(of_start), of_l4t.get("start_recipe") or {}
    if (recorded["rank"], recorded["alpha"]) != (of_l4ts.get("rank"), of_l4ts.get("alpha")):
        raise RuntimeError(f"the run {start.directory.name} recorded an adapter of rank {recorded['rank']} and alpha {recorded['alpha']} for `{start.name}`, and L4t's run "
                           f"{l4t.name} trained its models at rank {of_l4ts.get('rank')} and alpha {of_l4ts.get('alpha')}: every model here is at the rank L4t's were. "
                           "Nothing was written.")
    plan, of_l4ts_plan = {key: of_start[key] for key in PLAN}, {key: of_l4t.get(key) for key in PLAN}
    if plan != of_l4ts_plan:
        raise RuntimeError(f"the run {start.directory.name} recorded how `{start.name}` was measured ({plan}), and L4t's run {l4t.name} measured its models otherwise "
                           f"({of_l4ts_plan}): {READING.models} would not pair with both. Nothing was written.")
    parts = ladder_l4.pre_parts(of_start)
    need(start.directory, [ladder_l4.ROWS_FILE, *(name for part in parts for name in (f"episodes_{start_set(part, start)}_problems.jsonl", f"episodes_{start_set(part, start)}.done.json"))],
         f"the rows `{start.name}` was pretrained on (the rehearsal rows are rows of them), its per-problem rows on a set it was measured on, or that set's own marker", start.task)
    need(l4t, [name for part in parts for model in STORED for name in (f"episodes_{l4t_set(part, model)}_problems.jsonl", f"episodes_{l4t_set(part, model)}.done.json")],
         "the per-problem rows of one of L4t's three models on a set it was measured on, or that set's own marker", task)
    if not ladder_ceiling._stand_in() and not start.adapter.is_dir():
        raise RuntimeError(f"{start.adapter} is not there: the adapter `{rows.start}` is what every model here is trained from. Run {start.task} again; nothing was written.")

    # ---- the stored rows every read here stands on: the start model's, and L4t's three models', held to the same problems, seeds and attempts
    problems = _rows(l4t / "problems.jsonl")
    rung_problems = [row for row in problems if row["set"] == l4t_set(RUNG_PART, STORED[0])]
    goal_problems = [row for row in problems if row["set"] == l4t_set(plan["goal_samplings"][0]["name"], STORED[0])]
    ids = {RUNG_PART: [row["problem_id"] for row in rung_problems], GOAL: [row["problem_id"] for row in goal_problems]}
    measured = [(RUNG_PART, RUNG_PART, plan["rung_episodes"], plan["sampling_seeds"][RUNG_PART]),
                *((sampling["name"], GOAL, sampling["episodes"], sampling["sampling_seed"]) for sampling in plan["goal_samplings"])]
    kept = {}
    for part, of, episodes, sampling_seed in measured:
        kept[rows.start, part] = ladder_ceiling._stored(start.directory, start_set(part, start), ids[of], sampling_seed, episodes, f"`{rows.start}` ({part})", READING)[0]
        for model in STORED:
            kept[model, part] = ladder_ceiling._stored(l4t, l4t_set(part, model), ids[of], sampling_seed, episodes, f"`{model}` ({part})", READING)[0]
    # G' is the START MODEL's, made by the pretraining's rule from its first sampling of G, as L4t made it: and it must be the one L4t's run stored.
    again = goal_set_again(kept[rows.start, plan["goal_samplings"][0]["name"]], ids[GOAL])
    if again != [row["problem_id"] for row in _rows(l4t / ladder_l4.AGAIN_FILE)]:
        raise RuntimeError(f"G' made from the rows of `{rows.start}` ({start.directory.name}: {len(again)} goal problems it does not solve in its first sampling) is not the "
                           f"G' L4t's run {l4t.name} stored: the two runs are not of one start model's rows. Nothing was written.")

    # ---- the two sets: from L4t's two stored sets and the rows the start model was pretrained on (the file's text is read, and stays in the file)
    pretraining, sha256 = ladder_l4_start.pretrained_on(start.directory, of_start["pretraining_file_sha256"], start.name)      # refused unless the pretraining file is the one it was trained on
    if sha256 != of_l4t.get("pretraining_file_sha256"):
        raise RuntimeError(f"L4t's run {l4t.name} was prepared on a pretraining file of SHA-256 {of_l4t.get('pretraining_file_sha256')} and `{start.name}` was trained on "
                           f"{sha256}: L4t's sets name each rehearsal row by its place in the file it was prepared on. Nothing was written.")
    rehearse, reward_rows = (_rows(l4t / of_l4t_stage.training_file(model)) for model in (REHEARSE, REWARD_ROWS))
    batch = config["training"]["effective_batch"]
    try:
        sets = build_the_sets(rehearse, reward_rows, pretraining, seed)
        groups = _rows(l4t / ladder_l4.GROUPS_FILE)
        heldout = {row["problem_id"] for row in ladder_loop.load_problems(ladder_loop.data_directory())[0]} | {row["problem_id"] for row in groups}
        half_seed = config["ladder_loop"]["l4"]["half_seed"]
        counted = {model: refuse_barred(sets[model], heldout, half_seed, f"the training set of `{model}`") for model in rows.models}
    except ValueError as error:
        raise RuntimeError(f"{error} (L4t's run {l4t.name}; `{start.name}`'s {start.directory.name}). Nothing was written.") from error

    # ---- from here on files are written: the problems of every set this run samples, what its report reads of the stored models, the training sets
    def as_set(own: list[dict], set_name: str) -> list[dict]:
        return [{**row, "set": set_name} for row in own]

    own = [row for model in rows.models for row in as_set(rung_problems, rung_set(model, L4))]
    own += [row for model in rows.models for sampling in plan["goal_samplings"] for row in as_set(goal_problems, goal_set(sampling["name"], model, L4))]
    store.write_rows("problems.jsonl", own)             # every set of it is this run's
    store.write_rows(ladder_l4.GROUPS_FILE, groups)
    store.write_rows(ladder_l4.AGAIN_FILE, [{"problem_id": problem_id} for problem_id in again])
    for (who, part), own_rows in kept.items():
        store.write_rows(stored_file(who, part, L4), own_rows)
    wrote = {rows.start: of_the_start_run["measured"], **{model: json.loads((l4t / f"{of_l4t_stage.measure_marker(model)}.done.json").read_text()) for model in STORED}}
    _write_json(store, STORED_MODELS_FILE, {"stage": L4, "label": LABEL, **{who: {key: entry[key] for key in WROTE} for who, entry in wrote.items()}})
    for model in rows.models:
        store.write_rows(THE_STAGE.training_file(model), sets[model])
    of_rehearse, one_shot = rehearsal_of(rehearse), sum(row["origin"] == ATTEMPT for row in reward_rows)
    summary = {
        "stage": L4, "label": LABEL, "check": "L4t2", "seed": seed, "start": rows.start, "start_adapter": str(start.adapter), "start_run": str(start.directory),
        "l4t_run": str(l4t), "models": list(rows.models), "stand_in_engine": _stand_in(), "effective_batch": batch,
        "training_sets": {
            REHEARSE_ONLY: {"what": f"the rehearsal rows of L4t's `{REHEARSE}` set and nothing else, in that set's order with the other rows left out: published proofs of the "
                                    f"`pretrain` half that `{rows.start}` was pretrained on, a second pass over them alone. Their text stays in the pretraining file",
                            "file": THE_STAGE.training_file(REHEARSE_ONLY), **described(sets[REHEARSE_ONLY], batch), "rehearsal_rows": len(sets[REHEARSE_ONLY]),
                            "rows_of_rehearse": len(rehearse), "from": of_l4t_stage.training_file(REHEARSE), "barred": counted[REHEARSE_ONLY]},
            REWARD_REHEARSE: {"what": f"L4t's `{REWARD_ROWS}` set and as many rehearsal rows as it has one-shot rows: the first of the hash order L4t's rehearsal draw uses "
                                      f"(a content hash of the task's seed and each row's id among the rows `{rows.start}` was pretrained on), so they are the first of "
                                      f"`{REHEARSE}`'s own; in ONE order, the arm's content hash. The rehearsal rows' text stays in the pretraining file",
                              "file": THE_STAGE.training_file(REWARD_REHEARSE), **described(sets[REWARD_REHEARSE], batch), "rows_of_reward_rows": len(reward_rows),
                              "one_shot_rows_of_reward_rows": one_shot, "rehearsal_rows": len(rehearsal_of(sets[REWARD_REHEARSE])), "rehearsal_rows_of_rehearse": len(of_rehearse),
                              "rows_the_start_was_pretrained_on": len(pretraining), "from": of_l4t_stage.training_file(REWARD_ROWS), "barred": counted[REWARD_REHEARSE]}},
        "pretraining_file": str(ladder_l4.pretraining_file()), "pretraining_file_sha256": sha256,
        "recipe": {"what": "L4t's (`ladder_ceiling.one_pass`, as the arm's twin is trained): FROM THE START ADAPTER, one pass, the native format, the content-hash order",
                   "learning_rate": config["training"]["learning_rate"], "warmup_steps": config["training"]["warmup_steps"], "effective_batch": batch,
                   "lora": config_from(config, {"start_recipe": recorded}, start)["lora"], "adapter_seed": seed},
        "goal_set": len(goal_problems), "goal_set_again": len(again), "rung_problems": len(rung_problems), **plan,
        "attempts_a_goal_problem": sum(sampling["episodes"] for sampling in plan["goal_samplings"]), "parts": parts,
        "stored_models": {rows.start: {"run": str(start.directory), "sets": {part: start_set(part, start) for part in parts}},
                          **{model: {"run": str(l4t), "sets": {part: l4t_set(part, model) for part in parts}} for model in STORED}},
        "never_solved_file": NEVER_SOLVED_FILE, "contradicted_side_setting": config["ladder_loop"]["episode"].get("contradicted_side", "all"),
        "start_recipe": {"what": f"the adapter of `{rows.start}`, as the run that made it recorded it and as L4t's run trained its models: EVERY training here attaches this "
                                 f"rank and alpha (not the config's own, {config['lora']['rank']} and {config['lora']['alpha']}), the adapter each saves is read back and held "
                                 "to them, and the model server of this stage's measure steps is started with this rank as its largest",
                         **recorded, "check": of_start["rank_check"].get("check"), "stage": of_start["rank_check"].get("stage"),
                         "adapter_saved": of_the_start_run["trained"].get("adapter_saved"), "rank_of_the_config": config["lora"]["rank"], "alpha_of_the_config": config["lora"]["alpha"]},
        "start_run_read": {"branch": (of_the_start_run["report"].get("branch") or {}).get("name"), "headline": of_the_start_run["report"].get("headline"),
                           "pretraining_file_sha256": of_start["pretraining_file_sha256"], "rows": of_start["rows"],
                           "checks_failed": of_the_start_run["checks_failed"], "checks_waived_as_l4ts_run_recorded": of_the_start_run["checks_waived"]},
        "l4t_read": {"what": f"L4t's finished run from `{rows.start}`, only read: its two stored training sets, the per-problem rows of its three measured models and what "
                             "each wrote. `failed_checks`: by model, the checks ITS report read as failed; such a model is INCONCLUSIVE there and is not read here either",
                     "run": str(l4t), "headline": report.get("headline"), "branches": report.get("branches"), "failed_checks": failed_in_l4t,
                     "rows_made_by": of_l4t.get("rows_made_by")}}
    store.mark_done(PREPARE, summary)
    of = summary["training_sets"]
    _say(f"L4t2 prepared, seed {seed}, from `{rows.start}` on the sets of {l4t.name}: `{REHEARSE_ONLY}` {of[REHEARSE_ONLY]['rows']:,} rows (the rehearsal rows of "
         f"`{REHEARSE}`'s {len(rehearse):,}), {of[REHEARSE_ONLY]['optimizer_steps']:,} optimizer steps; `{REWARD_REHEARSE}` {of[REWARD_REHEARSE]['rows']:,} rows "
         f"(`{REWARD_ROWS}`'s {len(reward_rows):,} and the first {of[REWARD_REHEARSE]['rehearsal_rows']:,} of `{REHEARSE}`'s {len(of_rehearse):,} rehearsal rows), "
         f"{of[REWARD_REHEARSE]['optimizer_steps']:,} optimizer steps; every model at rank {recorded['rank']}; G' of `{rows.start}` holds {len(again)} goal problems")
    return summary


# ----------------------------------------------------------------------------------------- train and measure
def ladder_l4_rows2_train(config: dict, model: str) -> dict:
    """One of the two models, by L4t's own training body (`ladder_l4_rows.train_a_model`) under this stage's names."""
    rows = the_settings(config)
    return train_a_model(config, _store(config, rows), model, rows.start, run_name(rows, training_seed(config)), THE_STAGE)


def ladder_l4_rows2_measure(config: dict, model: str) -> dict:
    """One of the two models, by L4t's own measuring body (`ladder_l4_rows.measure_a_model`) under this stage's names."""
    rows = the_settings(config)
    return measure_a_model(config, _store(config, rows), model, rows.start, THE_STAGE)


# -------------------------------------------------------------------------------------------------- report
def ladder_l4_rows2_report(config: dict) -> dict:
    from rlvr_lean.gpu.ladder_l3c import proof_lengths
    from rlvr_lean.reporting.ladder_l4_rows2 import build_rows2_report

    rows = the_settings(config)
    store = _store(config, rows)
    for marker in (PREPARE, *(step(model) for model in rows.models for step in (THE_STAGE.train_marker, THE_STAGE.measure_marker))):
        _need(store, marker, "L4t2's report", ENTRY_STAGE)
    prepared = store.done_summary(PREPARE)
    samplings, of_the_stored = prepared["goal_samplings"], json.loads(store.path(STORED_MODELS_FILE).read_text())
    stored = {who: {RUNG_PART: store.read_rows(stored_file(who, RUNG_PART, L4)), GOAL: [store.read_rows(stored_file(who, sampling["name"], L4)) for sampling in samplings],
                    **{key: of_the_stored[who][key] for key in WROTE}} for who in (rows.start, *STORED)}
    models, trains, losses, trained = read_the_trained(store, prepared, rows.models, THE_STAGE)
    # The goal problems nothing stored has ever solved are GIVEN (a file of ids in the run directory), never computed here.
    never = [row["problem_id"] for row in store.read_rows(NEVER_SOLVED_FILE)] if store.path(NEVER_SOLVED_FILE).exists() else None
    # The length of a held-out problem's published proof is read HERE, by the report, and by nothing before it.
    report = build_rows2_report(prepared, trains, losses, trained, store.read_rows(ladder_l4.GROUPS_FILE), proof_lengths(), stored, models,
                                [row["problem_id"] for row in store.read_rows(ladder_l4.AGAIN_FILE)], config["ladder_loop"]["l4"], config["evaluation"], never)
    directory = store.root / ladder_assembly.ADAPTERS
    report["adapters"] = {"kept": True, "what": f"the adapters of {' and '.join(f'`{model}`' for model in rows.models)} ({LABEL}) are KEPT: no step of this stage deletes "
                                                f"one. `{rows.start}` is its own run's and was only read; so were L4t's",
                          "directory": str(directory), "there": sorted(path.name for path in directory.glob("*") if path.is_dir()) if directory.is_dir() else []}
    _write_json(store, REPORT_FILE, report)
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    for name in (STORED_MODELS_FILE, *(THE_STAGE.loss_file(model) for model in rows.models)):
        store.mirror(name)
    for line in report["lines"]:
        print(line, flush=True)
    store.mark_done(REPORT, {"stage": L4, "label": LABEL, "check": "L4t2", "headline": report["headline"], "branches": report["branches"], "ok": report["ok"]})
    return report


def _for(step, name: str):
    def run(config: dict) -> dict:
        return step(config, name)
    return run


# In the order the stage runs them: each model trained, then measured, before the next is trained.
STEPS = {PREPARE: ladder_l4_rows2_prepare}
for _model in MODELS:
    STEPS[THE_STAGE.train_marker(_model)] = _for(ladder_l4_rows2_train, _model)
    STEPS[THE_STAGE.measure_marker(_model)] = _for(ladder_l4_rows2_measure, _model)
STEPS[REPORT] = ladder_l4_rows2_report
