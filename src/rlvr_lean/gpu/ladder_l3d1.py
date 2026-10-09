"""L3d Step 1: do assembled proofs teach? Spec: docs/spec/ladder-loop.spec.md, "L3d: train on what the episode
reaches" ("Step 1", "The read, fixed before the run", "Made exact by the build"). Each step is its own process
(`rlvr_lean.runner.entry`, stage `ladder_l3d1`).

TWO models are trained from the base in this run, one pass each, the round's recipe: `without` (the training examples
the three rounds at t = 1/10 stored at this seed, in a seeded order) and `with` (the same rows in the same relative
order, with the harvest H0's assembled proofs at seeded places among them). Both by the same code, the same adapter
seed and the same sampling seeds, so that what differs between them is H0 and not the run. Both are the loop's own
models (every proof they were trained on is the model's own, verified): the adapters are KEPT.

It is the ceiling stage's shape with two trainings in the place of two checkpoints, and its reading of the stored
runs, its one pass and its measurement are the ceiling's functions (`gpu/ladder_ceiling.py`), not copies of them.

  ladder_l3d1_prepare            reads, ON THE BOX and never writing there: L1's run directory of this seed (G and the
                                 rungs, the base's fresh results and attempts) and L2's two (the base's control
                                 attempts on G; the run of the arm `l3d.step_1.loop_arm`: its three-round model, and
                                 its `training_examples_r1` to `_r3`, which are `without`'s rows). Reads H0, which
                                 travels with the code (`data/ladder_l3d/harvest_h0.jsonl`), and REFUSES it with
                                 fewer than `minimum_h0` rows, a held-out or base-map problem, a problem the rounds
                                 trained on already, or a row the recipe would cut. Fixes both orders and stores them
                                 (`l3d1_training_<model>.jsonl`). No GPU, no Lean.
  ladder_l3d1_train_<model>      one pass from the base over that model's rows in the order the prepare step fixed
                                 (`ladder_ceiling.one_pass`); every row's loss, read before its step's update, and
                                 the training loop's own record of the rows each step was made on
  ladder_l3d1_measure_<model>    the model served beside the base: 8 episodes on every problem of the three rungs and
                                 the attempts on every goal problem, with the sampling seeds L2 used for its models
                                 (`ladder_ceiling.measure_model`), so that it pairs by problem with the stored attempts
  ladder_l3d1_report             the read fixed before the run (`reporting/ladder_l3d1.py`). The counts "with assembly"
                                 are not in it: `tools/ladder_goal_assembly.py` makes them afterwards, Lean only

The run directory is its own (`ladder_l3d1_seed<seed>`). A rerun resumes at the first step, and inside a measurement
the first block, not done. A report that is not to be read (Lean gave no verdict on too much of a set) fails its step:
queued again, the task measures that set, and no other, again from the kept adapter. A verified proof on the side a
certificate contradicts stops the step with the soundness alarm's exit code, as everywhere.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from rlvr_lean.domain.ladder_round import reliable
from rlvr_lean.domain.ladder_round.ceiling import the_training_took
from rlvr_lean.domain.ladder_round.l3d import (
    ARMS,
    EXAMPLE_FIELDS,
    H0,
    L3D1,
    WITH,
    WITHOUT,
    check_h0,
    h0_rows,
    opens_with_a_have,
    round_rows,
    seeded_order,
    tenth,
    with_h0,
)
from rlvr_lean.domain.ladder_round.read import GOAL
from rlvr_lean.gpu import ladder_ceiling, ladder_l2, ladder_loop, pipeline
from rlvr_lean.gpu.ladder_ceiling import (
    CHECKPOINTS,
    LOOP,
    RUNG_PART,
    Model,
    Reading,
    StoredRuns,
    _file_rows,
    _need,
    example_tokens,
    goal_set,
    loop_arm,
    measure_model,
    measured_sizes,
    one_pass,
    read_stored_runs,
    rung_set,
    stored_file,
    what_a_model_wrote,
    write_stored_runs,
)
from rlvr_lean.gpu.ladder_dose import _rows, _runs
from rlvr_lean.gpu.ladder_round import BASE, _results, _stand_in, _write_json, training_seed
from rlvr_lean.infrastructure.artifact_store import ArtifactStore

STAGE = "ladder_l3d1"
RUN_VARIABLE = "RLVR_LEAN_LADDER_L3D1_RUN"              # another run directory than `ladder_l3d1_seed<seed>` (a smoke run)
SOURCE_VARIABLE = "RLVR_LEAN_LADDER_L3D1_SOURCE"        # another L1 run directory to read than `ladder_l1_seed<seed>`
STORED_VARIABLE = "RLVR_LEAN_LADDER_L3D1_STORED"        # `none`: L2's runs are not read (a smoke run): `without` is the L1 run's own training examples
H0_VARIABLE = "RLVR_LEAN_LADDER_L3D1_H0"                # another harvest than the package's (the fixture)
MINIMUM_VARIABLE = "RLVR_LEAN_LADDER_L3D1_MINIMUM"      # another least number of H0's rows than `l3d.step_1.minimum_h0` (the fixture holds 8)
PACKAGE_H0 = Path(__file__).resolve().parents[1] / "data" / "ladder_l3d" / "harvest_h0.jsonl"
SETTING = "ladder_loop.l3d.step_1.loop_arm"
PREPARE, REPORT = f"{STAGE}_prepare", f"{STAGE}_report"
REPORT_FILE, STORED_MODELS_FILE, GROUPS_FILE = "report_ladder_l3d1.json", f"{L3D1}_stored_models.json", f"{L3D1}_heldout_groups.jsonl"
ADAPTERS = "adapters"                                   # under the run directory, one directory a model; kept
SMOKE_ROUNDS = "training_examples_challenger.jsonl"     # what an L1 run stores of its challenger arm: a smoke run's `without`
READING = Reading(reader="L3d Step 1", models="L3d Step 1's two models",
                  not_read="that set is not to be read, and no report of this run could be read against it", setting=SETTING)
MAX_LEFTOVER_GB = pipeline.MAX_LEFTOVER_GB


# ------------------------------------------------------------------------------------------------- the run
def _store(config: dict) -> ArtifactStore:
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    run = os.environ.get(RUN_VARIABLE) or f"{STAGE}_seed{training_seed(config)}"
    return ArtifactStore(_runs(config) / run, Path(mirror) if mirror else None)


def source_directory(config: dict) -> Path:
    """L1's run directory for this seed. It is only ever READ."""
    return _runs(config) / (os.environ.get(SOURCE_VARIABLE) or f"ladder_l1_seed{training_seed(config)}")


def stored_directories(config: dict) -> dict[str, Path] | None:
    """L2's two run directories of this seed (`ladder_ceiling.stored_directories`, with this stage's variable and
    arm), or None when the task says there are none to read (a smoke run). They are only ever READ."""
    return ladder_ceiling.stored_directories(config, STORED_VARIABLE, SETTING)


def h0_file() -> Path:
    override = os.environ.get(H0_VARIABLE)
    return Path(override) if override else PACKAGE_H0


def minimum_h0(config: dict) -> int:
    override = os.environ.get(MINIMUM_VARIABLE)
    return max(1, int(override)) if override else config["ladder_loop"]["l3d"]["step_1"]["minimum_h0"]


def adapter_directory(store: ArtifactStore, arm: str) -> Path:
    return store.root / ADAPTERS / arm


def train_marker(arm: str) -> str:
    return f"{STAGE}_train_{arm}"


def measure_marker(arm: str) -> str:
    return f"{STAGE}_measure_{arm}"


def training_file(arm: str) -> str:
    return f"{L3D1}_training_{arm}.jsonl"


def loss_file(arm: str) -> str:
    return f"{L3D1}_loss_{arm}.json"


def _say(message: str) -> None:
    print(f"{L3D1}: {message}", flush=True)


# --------------------------------------------------------------------------------------- what a model wrote
def what_else_a_model_wrote(rung_attempts: list[dict], goal_attempts: list[list[dict]]) -> dict:
    """Beside `what_a_model_wrote`, of one model's attempts on the rungs and on G (`goal_attempts`: one list a
    sampling): the share whose first step is a `have`, and each goal problem's episodes of 8 one-shot attempts in the
    order drawn, each sampling cut on its own (`domain/ladder_round/reliable.py`: resolved by attempts alone)."""
    on_g = [attempt for attempts in goal_attempts for attempt in attempts]
    problems = sorted({attempt["problem_id"] for attempt in on_g})
    return {"opens_with_a_have": {RUNG_PART: opens_with_a_have(rung_attempts), GOAL: opens_with_a_have(on_g)},
            "episodes_of_8": reliable.episodes_by_attempts([reliable.attempt_flags(attempts) for attempts in goal_attempts], problems)}


def _wrote(read: StoredRuns, who: str) -> dict:
    return {**what_a_model_wrote(read.wrote[who][RUNG_PART], read.wrote[who][GOAL]), **what_else_a_model_wrote(read.wrote[who][RUNG_PART], read.on_g[who])}


# ------------------------------------------------------------------------------------------------- prepare
def round_files(config: dict, source: Path, stored: dict[str, Path] | None) -> list[Path]:
    """Where `without`'s rows are stored on the box: the three rounds' training examples of the arm's L2 run of this
    seed, in the order of the rounds; for a smoke run (no L2 run is read) the L1 run's own challenger examples."""
    if stored is None:
        return [source / SMOKE_ROUNDS]
    return [stored[LOOP] / f"training_examples_r{number}.jsonl" for number in ladder_l2.ROUNDS]


def ladder_l3d1_prepare(config: dict) -> dict:
    store = _store(config)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    seed, source, stored, settings = training_seed(config), source_directory(config), stored_directories(config), config["ladder_loop"]["l3d"]["step_1"]
    # ---- what the runs before this one stored, on the box: as the ceiling reads it
    read = read_stored_runs(config, source, stored, READING)
    arm = loop_arm(config, SETTING) if stored else None
    files = round_files(config, source, stored)
    lacking = [path.name for path in files if not path.exists()]
    if lacking:
        task = f"stage `ladder_l2_{arm}` for seed {seed} (`python -m rlvr_lean.runner.entry --stage ladder_l2_{arm} --seeds {seed}`)" if stored else "stage `ladder_l1_smoke`"
        raise RuntimeError(f"{files[0].parent} does not hold {lacking}. L3d Step 1 trains `without` on the training examples the task of {task} stored on this "
                           "box. Run that task to its end first; nothing was written.")
    by_round = [_rows(path) for path in files]
    rounds = round_rows(by_round)
    # ---- H0, which travels with the code
    path = h0_file()
    harvest, sha256 = _file_rows(path, "the harvest H0", "it is built on the dev machine by `tools/ladder_harvest.py` (Lean only) and committed with the code "
                                                         "(spec, L3d, 'The harvest H0')")
    heldout, base_map, _ = ladder_loop.load_problems(ladder_loop.data_directory())
    minimum = minimum_h0(config)
    check_h0(harvest, {row["problem_id"] for row in heldout} | {row["problem_id"] for row in read.groups}, {row["problem_id"] for row in base_map},
             {row["problem_id"] for row in rounds}, minimum)
    # ---- the two orders, by the task's seed
    orders = {WITHOUT: seeded_order(rounds, seed)}
    orders[WITH] = with_h0(orders[WITHOUT], h0_rows(harvest), seed)
    count, counted_by = example_tokens(config)
    tokens = {row["id"]: count(row) for row in orders[WITH]}
    training = config["training"]
    too_long = [(row["id"], tokens[row["id"]]) for row in orders[WITH] if tokens[row["id"]] > training["max_sequence_tokens"]]
    if too_long:
        raise ValueError(f"{len(too_long)} training rows are longer than the {training['max_sequence_tokens']} tokens a training example is cut at (first: "
                         f"{too_long[0][0]}, {too_long[0][1]} tokens): they would be trained on without their end. Refused")
    batch = training["effective_batch"]
    trainings = {name: {"rows": len(rows), "h0_rows": sum(row["source"] == H0 for row in rows), "steps": -(-len(rows) // batch),
                        "tokens": sum(tokens[row["id"]] for row in rows)} for name, rows in orders.items()}
    write_stored_runs(store, read, ARMS, L3D1)          # every set of `problems.jsonl` is this run's
    _write_json(store, STORED_MODELS_FILE, {"stage": L3D1, **{who: _wrote(read, who) for who in read.wrote}})
    for name, rows in orders.items():                   # both orders, by id, with the texts: a training step reads its own run's file and no other
        store.write_rows(training_file(name), [{"row": position, **row, "tokens": tokens[row["id"]]} for position, row in enumerate(rows)])
    places = [position for position, row in enumerate(orders[WITH]) if row["source"] == H0]
    summary = {"stage": L3D1, "seed": seed, "source_run": str(source), "stored_runs": {who: str(directory) for who, directory in stored.items()} if stored else None,
               "loop_arm": arm, "loop_target_rate": config["ladder_loop"]["l2_arms"][arm].get("target_rate") if arm else None, "stand_in_engine": _stand_in(),
               "rounds_files": {str(file): len(rows) for file, rows in zip(files, by_round)}, "rounds_rows": len(rounds),
               "h0_file": str(path), "h0_file_sha256": sha256, "h0_rows": len(harvest), "minimum_h0": minimum,
               "h0_sides": {side: sum(row["side"] == side for row in harvest) for side in sorted({row["side"] for row in harvest})},
               "order": {"what": "`without`: the rounds' rows (round 1's, then 2's, then 3's) sorted by a content hash of the task's seed and each row's attempt id "
                                 "(`l3d.seeded_order`). `with`: the same rows in that order, none moved, with each row of H0 in a gap chosen by a content hash "
                                 "of the task's seed and the row's id (`l3d.with_h0`)", "seed": seed,
                         "h0_places_in_with": {"first": places[0], "last": places[-1], "in_the_first_half": sum(place < len(orders[WITH]) / 2 for place in places)}},
               "trainings": trainings, "effective_batch": batch, "max_sequence_tokens": training["max_sequence_tokens"],
               "longest_example_tokens": max(tokens.values()), "tokens_counted_by": counted_by,
               "recipe": {"what": "the round's (`ladder_round._train`), from the base: one pass, the native format, the round's own example builder; both models "
                                  "by the same code and the same adapter seed",
                          "learning_rate": training["learning_rate"], "warmup_steps": training["warmup_steps"], "effective_batch": batch,
                          "lora": config["lora"], "adapter_seed": seed},
               "settings": dict(settings), **measured_sizes(config, read)}
    store.mark_done(PREPARE, summary)
    _say(f"prepared, seed {seed}: `without` {trainings[WITHOUT]['rows']} rows ({trainings[WITHOUT]['steps']} steps), `with` {trainings[WITH]['rows']} rows "
         f"({trainings[WITH]['steps']} steps), {len(harvest)} of them H0's; {summary['attempts_a_goal_problem']} attempts on each of {len(read.goal_problems)} goal problems")
    return summary


# --------------------------------------------------------------------------------------------------- train
def ladder_l3d1_train(config: dict, arm: str) -> dict:
    """One of the two trainings: from the base, one pass over the rows the prepare step stored for it, in that order."""
    store, marker = _store(config), train_marker(arm)
    if store.is_done(marker):
        return store.done_summary(marker)
    _need(store, PREPARE, f"the training of `{arm}`", STAGE)
    prepared, seed = store.done_summary(PREPARE), training_seed(config)
    rows = store.read_rows(training_file(arm))
    examples = [{field: row[field] for field in EXAMPLE_FIELDS} for row in rows]
    planned, batch = prepared["trainings"][arm], prepared["effective_batch"]
    _say(f"training `{arm}` from the base on {len(examples)} rows ({planned['h0_rows']} of them H0's) in the prepared order, {planned['steps']} optimizer steps")
    result, step_rows, row_losses, positions = one_pass(config, examples, {arm: planned["steps"]}, seed, batch, store.root / ADAPTERS,
                                                        f"{STAGE}_{arm}_seed{seed}", positions=True)
    result.pop("checkpoints")
    trained = [rows[position]["id"] for position in positions]      # the training loop's own record: the rows each optimizer step was made on
    from_h0 = {row["id"] for row in rows if row["source"] == H0}
    _write_json(store, loss_file(arm), {
        "stage": L3D1, "model": arm, "seed": seed, "stand_in_engine": _stand_in(), "rows": len(row_losses), "steps": result["steps"], "effective_batch": batch,
        "what": "row_losses: each training row's mean loss per target token, in the order trained, read BEFORE the update of the optimizer step it was in. "
                "rows_trained: the id of the row each of those losses is of, from the positions the training loop recorded step by step. training_steps: "
                "every optimizer step by part of the target, read before its update, with the rows seen once it was made",
        "row_losses": row_losses, "rows_trained": trained,
        "training_steps": [{**row, "rows_seen": min(row["step"] * batch, len(row_losses))} for row in step_rows]})
    took = the_training_took(row_losses, tenth(len(row_losses), config["ladder_loop"]["l3d"]["step_1"]["loss_share_of_rows"]))
    summary = {"stage": L3D1, "model": arm, "seed": seed, "rows": len(examples), "h0_rows": planned["h0_rows"], "passes": 1, "stand_in_engine": _stand_in(),
               "h0_rows_trained": sum(key in from_h0 for key in trained), "first_step_loss": step_rows[0]["mean_loss"],
               "last_step_loss": step_rows[-1]["mean_loss"], "mean_loss_over_the_first_rows": took["first"], "mean_loss_over_the_last_rows": took["last"],
               "rows_compared": took["rows_compared"], **result}
    store.mark_done(marker, summary)        # the adapter is saved: keep it whatever follows
    _say(f"trained `{arm}`: {summary['steps']} steps over {len(examples)} rows; mean loss over the first {took['rows_compared']} rows {took['first']}, over the last {took['last']}")
    if summary.get("allocated_after_cleanup_gb", 0) > MAX_LEFTOVER_GB:
        raise RuntimeError(f"{summary['allocated_after_cleanup_gb']} GB is still allocated on the GPU after the training of `{arm}` was released")
    return summary


# ------------------------------------------------------------------------------------------------- measure
def ladder_l3d1_measure(config: dict, arm: str) -> dict:
    """One of the two models on the three held-out rungs and on G, as the ceiling's are (`measure_model`)."""
    store = _store(config)
    _need(store, PREPARE, f"the measurement of L3d Step 1's model `{arm}`", STAGE)
    prepared = store.done_summary(PREPARE)
    planned = prepared["trainings"][arm]
    return measure_model(config, store, prepared, Model(
        name=arm, called=f"the model `{arm}`", whose=f"L3d Step 1's model `{arm}`", marker=measure_marker(arm), trained_by=train_marker(arm),
        adapter=adapter_directory(store, arm),
        no_adapter="this run keeps its adapters, and one trained again would not be the model its other measurements came from. The measurement cannot be made",
        detail=f"{planned['rows']} rows, {planned['h0_rows']} of them H0's",
        summary={"stage": L3D1, "model": arm, "rows": planned["rows"], "h0_rows": planned["h0_rows"], "step": planned["steps"]},
        stage=STAGE, label=L3D1, also=what_else_a_model_wrote))


# -------------------------------------------------------------------------------------------------- report
WROTE = ("verified_proof_lines", "distinct_attempts", "opens_with_a_have", "episodes_of_8")       # what is said of what a model wrote


def ceiling_doses(config: dict) -> dict | None:
    """The ceiling's two doses, for the note beside an interval that holds zero: from the report the ceiling's run
    of this seed wrote ON THIS BOX (`ladder_ceiling_seed<seed>`; read, never written), each model's primary and the
    rows it had been trained on. None when there is no such report, or it is inconclusive or not to be read."""
    path = _runs(config) / f"ladder_ceiling_seed{training_seed(config)}" / ladder_ceiling.REPORT_FILE
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    if not report.get("ok") or report.get("inconclusive") or not report.get("secondary"):
        return None
    by_model, rows = report["secondary"]["by_length_group"]["models"], report["training"]["checkpoints"]
    doses = {name: {"rows": rows[name]["rows_seen"], **{key: by_model[name]["4_or_more"][key] for key in ("mean", "low", "high")}}
             for name in CHECKPOINTS if name in by_model and by_model[name]["4_or_more"].get("mean") is not None}
    return doses or None


def ladder_l3d1_report(config: dict) -> dict:
    from rlvr_lean.gpu.ladder_l3c import proof_lengths
    from rlvr_lean.reporting.ladder_l3d1 import build_l3d1_report

    store = _store(config)
    for marker in (PREPARE, *(train_marker(arm) for arm in ARMS), *(measure_marker(arm) for arm in ARMS)):
        _need(store, marker, "L3d Step 1's report", STAGE)
    prepared = store.done_summary(PREPARE)
    samplings, stored_models = prepared["goal_samplings"], json.loads(store.path(STORED_MODELS_FILE).read_text())

    def stored(who: str) -> dict:
        return {RUNG_PART: store.read_rows(stored_file(who, RUNG_PART, L3D1)),
                GOAL: [store.read_rows(stored_file(who, sampling["name"], L3D1)) for sampling in samplings], **stored_models[who]}

    models = {}
    for arm in ARMS:
        measured = store.done_summary(measure_marker(arm))
        models[arm] = {"rows": prepared["trainings"][arm]["rows"], RUNG_PART: _results(store, rung_set(arm, L3D1)),
                       GOAL: [_results(store, goal_set(sampling["name"], arm, L3D1)) for sampling in samplings],
                       **{key: measured[key] for key in WROTE}, "stand_in_engine": measured["stand_in_engine"]}
    if prepared["stored_runs"]:
        models[LOOP] = {"rows": None, **stored(LOOP)}
    # The length of a held-out problem's published proof is read HERE, by the report, and by nothing before it.
    report = build_l3d1_report(prepared, {arm: store.done_summary(train_marker(arm)) for arm in ARMS},
                               {arm: json.loads(store.path(loss_file(arm)).read_text()) for arm in ARMS},
                               {arm: [{"id": row["id"], "source": row["source"]} for row in store.read_rows(training_file(arm))] for arm in ARMS},
                               store.read_rows(GROUPS_FILE), proof_lengths(), stored(BASE), models, config["ladder_loop"]["l3d"]["step_1"], config["evaluation"],
                               ceiling_doses(config) if prepared["stored_runs"] else None)
    report["adapters"] = {"kept": True, "what": "both models are the loop's own (trained on the model's own verified proofs): their adapters are kept",
                          "directory": str(store.root / ADAPTERS), "there": [arm for arm in ARMS if adapter_directory(store, arm).is_dir()]}
    _write_json(store, REPORT_FILE, report)
    # Everything a reader needs beside the report goes out again from here: the per-problem results, the stored rows read
    # beside them, both orders, the losses. Not the attempts and not the per-block files.
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    for name in (STORED_MODELS_FILE, *(loss_file(arm) for arm in ARMS)):
        store.mirror(name)
    for line in report["lines"]:
        print(line, flush=True)
    store.mark_done(REPORT, {"stage": L3D1, "headline": report["headline"], "branch": report["branch"], "ok": report["ok"]})
    return report


def _for(step, arm: str):
    def run(config: dict) -> dict:
        return step(config, arm)
    return run


STEPS = {
    PREPARE: ladder_l3d1_prepare,
    **{train_marker(arm): _for(ladder_l3d1_train, arm) for arm in ARMS},
    **{measure_marker(arm): _for(ladder_l3d1_measure, arm) for arm in ARMS},
    REPORT: ladder_l3d1_report,
}
