"""L3d Step 2: the loop with assembly in the round, and a twin trained without the assembled proofs. Spec:
docs/spec/ladder-loop.spec.md, "L3d: train on what the episode reaches", "Step 2, made exact before it is built"
and "Made exact by the build (Step 2)". Each step is its own process (`rlvr_lean.runner.entry`, stage `ladder_l3d2`).

The stage is the L2 stage's arm with assembly (`gpu/ladder_l2.py`, `gpu/ladder_assembly.py`; the arm named by
`ladder_loop.l3d.step_2.arm`, set for every step by `RLVR_LEAN_LADDER_L2_ARM`), and after its last round the steps
here, in the ARM'S run directory (`ladder_l2_t010_assembly_seed<N>`):

  ladder_l2_prepare, ladder_l2_embed, ladder_l2_round_<r>, ladder_l2_train_<r>    the arm: six rounds with assembly
                                 after each batch; M(r) from the base after each. Rounds 4 to 6 are registered here
  ladder_l3d2_prepare            (right after the arm's own prepare step, before anything is sampled) reads, ON THE BOX
                                 and never writing there, what the ceiling reads: L1's run of the seed (the base's
                                 fresh results) and L2's two (the base's control attempts on G; the stored three-round
                                 model at t = 1/10), refused unless they pair. Adds the two models' sets to the run
  ladder_l3d2_train_without      the TWIN: from the base, one pass over the rows of origin `attempt` the last model was
                                 trained on, in ITS order with the other rows left out (`assembly.attempts_alone`);
                                 the same recipe and adapter seed. The last round's model is `with`
  ladder_l3d2_measure_<model>    `with` and `without` as the ceiling's models are (`ladder_ceiling.measure_model`): 8
                                 episodes on the three rungs, 93 attempts on every goal problem with L2's sampling seeds
  ladder_l3d2_report             the read fixed before the run (`reporting/ladder_l3d2.py`). The counts with assembly
                                 are made afterwards, Lean only (`tools/ladder_goal_assembly.py`)

Every adapter is KEPT: they are the loop's own models. A rerun resumes at the first step, batch and block not done; a
report that is not to be read fails its step, and the task queued again measures that set again from the kept adapter.
"""

from __future__ import annotations

import json
import os

from rlvr_lean.domain.ladder_round.assembly import EXAMPLE_FIELDS, FROM_ASSEMBLY, attempts_alone, by_origin
from rlvr_lean.domain.ladder_round.ceiling import the_training_took
from rlvr_lean.domain.ladder_round.l3d import ARMS, WITH, WITHOUT, tenth
from rlvr_lean.domain.ladder_round.l3d2 import L3D2
from rlvr_lean.domain.ladder_round.read import GOAL
from rlvr_lean.gpu import ladder_assembly, ladder_ceiling, ladder_l2, pipeline
from rlvr_lean.gpu.ladder_ceiling import (
    LOOP,
    RUNG_PART,
    Model,
    Reading,
    _need,
    goal_set,
    loop_arm,
    measure_model,
    measured_sizes,
    one_pass,
    read_stored_runs,
    rung_set,
    stored_file,
    write_stored_runs,
)
from rlvr_lean.gpu.ladder_l3d1 import WROTE, _wrote, what_else_a_model_wrote
from rlvr_lean.gpu.ladder_round import BASE, _results, _stand_in, _write_json, training_seed
from rlvr_lean.infrastructure.artifact_store import ArtifactStore

STAGE = "ladder_l3d2"
STORED_VARIABLE = "RLVR_LEAN_LADDER_L3D2_STORED"        # `none`: L2's stored runs are not read (a smoke run): no second sampling of G, no three-round model beside
MINIMUM_VARIABLE = "RLVR_LEAN_LADDER_L3D2_MINIMUM"      # another least number of assembled proofs than `l3d.step_2.minimum_assembled` (a smoke run)
SETTING = "ladder_loop.l3d.step_2.loop_arm"
PREPARE, TRAIN_WITHOUT, REPORT = f"{STAGE}_prepare", f"{STAGE}_train_without", f"{STAGE}_report"
REPORT_FILE, STORED_MODELS_FILE, GROUPS_FILE = "report_ladder_l3d2.json", f"{L3D2}_stored_models.json", f"{L3D2}_heldout_groups.jsonl"
TRAINING_WITHOUT_FILE, LOSS_WITHOUT_FILE = f"{L3D2}_training_without.jsonl", f"{L3D2}_loss_without.json"
READING = Reading(reader="L3d Step 2", models="L3d Step 2's two models",
                  not_read="that set is not to be read, and no report of this run could be read against it", setting=SETTING)
MAX_LEFTOVER_GB = pipeline.MAX_LEFTOVER_GB


# ------------------------------------------------------------------------------------------------- the run
def _arm(config: dict) -> dict:
    """The config as the arm's steps read it. The stage runs ONE arm (`l3d.step_2.arm`), and every one of its tasks
    names it: a step here that is run for another arm, or for none, is refused."""
    wanted, named = config["ladder_loop"]["l3d"]["step_2"]["arm"], ladder_l2.arm_name()
    also = (config["ladder_loop"].get("l4") or {}).get("arm")          # L4's arm is this stage's steps again, from a stored adapter (`gpu/ladder_l4.py`)
    if named not in (wanted, also) or named is None or ladder_l2.assembly_arm(config) is None:
        raise RuntimeError(f"the stage `{STAGE}` runs the arm {wanted} of ladder_loop.{ladder_l2.ASSEMBLY_ARMS} and {ladder_l2.L2_ARM_VARIABLE} names "
                           f"{named!r}: its steps are run by the stage, which sets it")
    return ladder_l2.arm_config(config)


def _store(config: dict) -> ArtifactStore:
    return ladder_l2._store(_arm(config))


def minimum_assembled(config: dict) -> int:
    override = os.environ.get(MINIMUM_VARIABLE)
    return max(1, int(override)) if override else config["ladder_loop"]["l3d"]["step_2"]["minimum_assembled"]


def last_round(config: dict) -> int:
    return ladder_l2.rounds_of(config)[-1]


def adapter_of(config: dict, store: ArtifactStore, arm: str):
    """`with` is the last round's model, where the arm saved it; `without` is the twin's own."""
    return ladder_l2.adapter_directory(store, last_round(config)) if arm == WITH else store.root / ladder_assembly.ADAPTERS / WITHOUT


def trained_by(config: dict, arm: str) -> str:
    return f"ladder_l2_train_{last_round(config)}" if arm == WITH else TRAIN_WITHOUT


def measure_marker(arm: str) -> str:
    return f"{STAGE}_measure_{arm}"


def _say(message: str) -> None:
    print(f"{L3D2}: {message}", flush=True)


# ------------------------------------------------------------------------------------------------- prepare
def ladder_l3d2_prepare(config: dict) -> dict:
    """What the two models are read against, checked BEFORE anything is sampled: the stored runs as the ceiling
    reads them. The two models' sets are put beside the arm's own in its `problems.jsonl`."""
    store = _store(config)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    _need(store, ladder_l2.PREPARE, "L3d Step 2's own prepare step", STAGE)        # the arm's prepare step makes the run's `problems.jsonl`: these sets are added to it
    seed, source, stored = training_seed(config), ladder_l2.source_directory(config), ladder_ceiling.stored_directories(config, STORED_VARIABLE, SETTING)
    read = read_stored_runs(config, source, stored, READING)
    write_stored_runs(store, read, ARMS, L3D2, add=True)
    _write_json(store, STORED_MODELS_FILE, {"stage": L3D2, **{who: _wrote(read, who) for who in read.wrote}})
    arm, settings = loop_arm(config, SETTING) if stored else None, config["ladder_loop"]["l3d"]["step_2"]
    summary = {"stage": L3D2, "seed": seed, "arm": ladder_l2.arm_name(), "rounds": list(ladder_l2.rounds_of(config)), "source_run": str(source),
               "stored_runs": {who: str(directory) for who, directory in stored.items()} if stored else None, "loop_arm": arm,
               "loop_target_rate": config["ladder_loop"]["l2_arms"][arm].get("target_rate") if arm else None, "stand_in_engine": _stand_in(),
               "minimum_assembled": minimum_assembled(config), "settings": dict(settings), **measured_sizes(config, read)}
    store.mark_done(PREPARE, summary)
    _say(f"prepared, seed {seed}: the arm {summary['arm']}, rounds {summary['rounds']}; {summary['attempts_a_goal_problem']} attempts on each of "
         f"{len(read.goal_problems)} goal problems for `with` and for `without`")
    return summary


# ------------------------------------------------------------------------------------------------ the twin
def ladder_l3d2_train_without(config: dict) -> dict:
    """The twin: from the base, by the same code, on the one-shot rows of the last model's training set, in that
    set's order with the other rows left out."""
    store = _store(config)
    if store.is_done(TRAIN_WITHOUT):
        return store.done_summary(TRAIN_WITHOUT)
    last, seed, batch = last_round(config), training_seed(config), config["training"]["effective_batch"]
    for needed in (PREPARE, f"ladder_l2_train_{last}"):
        _need(store, needed, "the training of `without`", STAGE)
    ordered = store.read_rows(ladder_assembly.training_set_file(last))
    rows = [{**row, "row": position, "row_of_with": row["row"]} for position, row in enumerate(attempts_alone(ordered))]
    if not rows:
        raise RuntimeError(f"no row of M({last})'s training set is a one-shot attempt's: there is nothing to train `without` on")
    store.write_rows(TRAINING_WITHOUT_FILE, rows)
    examples = [{field: row[field] for field in EXAMPLE_FIELDS} for row in rows]
    steps = -(-len(rows) // batch)
    left_out = by_origin([row for row in ordered if row["origin"] in FROM_ASSEMBLY])
    _say(f"training `without` from the base on {len(rows)} one-shot rows, M({last})'s order with its {len(ordered) - len(rows)} assembled rows left out, {steps} optimizer steps")
    start = ladder_l2.start_directory(config)           # an arm whose models start from a stored adapter: so does its twin
    result, step_rows, row_losses, positions = one_pass(config, examples, {WITHOUT: steps}, seed, batch, store.root / ladder_assembly.ADAPTERS,
                                                        f"{STAGE}_{ladder_l2.arm_name()}_{WITHOUT}_seed{seed}" if start is not None else f"{STAGE}_{WITHOUT}_seed{seed}",
                                                        positions=True, **({"start": start} if start is not None else {}))
    result.pop("checkpoints")
    _write_json(store, LOSS_WITHOUT_FILE, {
        "stage": L3D2, "model": WITHOUT, "seed": seed, "stand_in_engine": _stand_in(), "rows": len(row_losses), "steps": result["steps"], "effective_batch": batch,
        "what": "row_losses: each training row's mean loss per target token, in the order trained, read BEFORE the update of the optimizer step it was in. "
                "rows_trained: the id of the row each of those losses is of, from the positions the training loop recorded step by step",
        "row_losses": row_losses, "rows_trained": [rows[position]["id"] for position in positions],
        "training_steps": [{**row, "rows_seen": min(row["step"] * batch, len(row_losses))} for row in step_rows]})
    took = the_training_took(row_losses, tenth(len(row_losses), config["ladder_loop"]["l3d"]["step_2"]["loss_share_of_rows"]))
    summary = {"stage": L3D2, "model": WITHOUT, "seed": seed, "trained_from": "the base" if start is None else f"the stored adapter {start}",
               "rows": len(rows), "rows_of_with": len(ordered),
               "left_out": {origin: left_out[origin] for origin in FROM_ASSEMBLY}, "order": f"M({last})'s, with the assembled rows left out and no row moved",
               "first_step_loss": step_rows[0]["mean_loss"], "last_step_loss": step_rows[-1]["mean_loss"], "mean_loss_over_the_first_rows": took["first"],
               "mean_loss_over_the_last_rows": took["last"], "rows_compared": took["rows_compared"], "stand_in_engine": _stand_in(), **result}
    store.mark_done(TRAIN_WITHOUT, summary)         # the adapter is saved: keep it whatever follows
    if summary.get("allocated_after_cleanup_gb", 0) > MAX_LEFTOVER_GB:
        raise RuntimeError(f"{summary['allocated_after_cleanup_gb']} GB is still allocated on the GPU after the training of `without` was released")
    return summary


# ------------------------------------------------------------------------------------------------- measure
def ladder_l3d2_measure(config: dict, arm: str) -> dict:
    """`with` (the last round's model) or `without` (its twin) on the three held-out rungs and on G, as the
    ceiling's models are (`measure_model`)."""
    store = _store(config)
    _need(store, PREPARE, f"the measurement of L3d Step 2's model `{arm}`", STAGE)
    prepared, last = store.done_summary(PREPARE), last_round(config)
    return measure_model(config, store, prepared, Model(
        name=arm, called=f"the model `{arm}`", whose=f"L3d Step 2's model `{arm}`", marker=measure_marker(arm), trained_by=trained_by(config, arm),
        adapter=adapter_of(config, store, arm),
        no_adapter="this run keeps its adapters, and one trained again would not be the model its other measurements came from. The measurement cannot be made",
        detail=f"M({last}), the last round's model" if arm == WITH else f"M({last})'s twin, trained without the assembled proofs",
        summary={"stage": L3D2, "model": arm, "is": f"M({last})" if arm == WITH else f"the twin of M({last})"},
        stage=STAGE, label=L3D2, also=what_else_a_model_wrote))


# -------------------------------------------------------------------------------------------------- report
def measured_models(store: ArtifactStore, prepared: dict) -> tuple[dict, dict]:
    """(the base, from the stored rows the prepare step copied; the models a report reads: `with` and `without` as this
    run measured them, and the stored three-round model when it was read). Each: its rows on the rungs, one list of
    rows for each sampling of G, and what it wrote."""
    samplings, stored_models = prepared["goal_samplings"], json.loads(store.path(STORED_MODELS_FILE).read_text())

    def stored(who: str) -> dict:
        return {RUNG_PART: store.read_rows(stored_file(who, RUNG_PART, L3D2)),
                GOAL: [store.read_rows(stored_file(who, sampling["name"], L3D2)) for sampling in samplings], **stored_models[who]}

    models = {}
    for arm in ARMS:
        measured = store.done_summary(measure_marker(arm))
        models[arm] = {RUNG_PART: _results(store, rung_set(arm, L3D2)), GOAL: [_results(store, goal_set(sampling["name"], arm, L3D2)) for sampling in samplings],
                       **{key: measured[key] for key in WROTE}, "stand_in_engine": measured["stand_in_engine"]}
    if prepared["stored_runs"]:
        models[LOOP] = stored(LOOP)
    return stored(BASE), models


def rounds_read(store: ArtifactStore, rounds: tuple[int, ...], batches: int) -> dict:
    """By round of the arm: what its step recorded, every batch's per-problem results and its training examples."""
    return {number: {"summary": store.done_summary(f"ladder_l2_round_{number}"),
                     "results": [row for batch in range(1, batches + 1) for row in _results(store, ladder_l2.batch_set(number, batch))],
                     "examples": store.read_rows(f"training_examples_r{number}.jsonl")} for number in rounds}


def ladder_l3d2_report(config: dict) -> dict:
    from rlvr_lean.gpu.ladder_l3c import proof_lengths
    from rlvr_lean.reporting.ladder_l3d2 import build_l3d2_report

    store, rounds = _store(config), ladder_l2.rounds_of(config)
    last = rounds[-1]
    for marker in (PREPARE, f"ladder_l2_train_{last}", TRAIN_WITHOUT, *(measure_marker(arm) for arm in ARMS)):
        _need(store, marker, "L3d Step 2's report", STAGE)
    prepared, sizes = store.done_summary(PREPARE), ladder_l2._sizes(_arm(config), store)
    base, models = measured_models(store, prepared)
    by_round = rounds_read(store, rounds, sizes["batches"])
    identified = lambda rows: [{"id": row["id"], "origin": row["origin"]} for row in rows]      # noqa: E731
    report = build_l3d2_report(
        prepared, store.done_summary(ladder_l2.PREPARE), {WITH: store.done_summary(f"ladder_l2_train_{last}"), WITHOUT: store.done_summary(TRAIN_WITHOUT)},
        {WITH: json.loads(store.path(ladder_assembly.loss_file(last)).read_text()), WITHOUT: json.loads(store.path(LOSS_WITHOUT_FILE).read_text())},
        {WITH: identified(store.read_rows(ladder_assembly.training_set_file(last))), WITHOUT: identified(store.read_rows(TRAINING_WITHOUT_FILE))},
        by_round, store.read_rows(GROUPS_FILE), proof_lengths(), base, models, config["ladder_loop"]["l3d"]["step_2"],
        _arm(config)["ladder_loop"]["challenger"]["target_rate"], config["evaluation"])
    report["adapters"] = {"kept": True, "what": "every model here is the loop's own (trained on the model's own verified proofs): the adapters are kept",
                          "directory": str(store.root / ladder_assembly.ADAPTERS), "there": sorted(path.name for path in (store.root / ladder_assembly.ADAPTERS).glob("*") if path.is_dir())}
    _write_json(store, REPORT_FILE, report)
    # Everything a reader needs beside the report goes out again from here: the per-problem results, the proposals, the
    # training examples and sets, the assembled proofs, the stored rows read beside them, what each step recorded. Not
    # the attempts, the kept errors or the per-block files.
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    for name in sorted(path.name for path in store.root.glob("*.done.json")):
        if "_block_" not in name:
            store.mirror(name)
    for name in (STORED_MODELS_FILE, LOSS_WITHOUT_FILE, *(ladder_assembly.loss_file(number) for number in rounds)):
        store.mirror(name)
    for line in report["lines"]:
        print(line, flush=True)
    store.mark_done(REPORT, {"stage": L3D2, "headline": report["headline"], "branch": report["branch"], "ok": report["ok"]})
    return report


def _for(step, arm: str):
    def run(config: dict) -> dict:
        return step(config, arm)
    return run


# The arm's rounds past the stage's three are steps of THIS stage: `ladder_l2.STEPS` stays the three rounds'.
LATER_ROUNDS = tuple(range(ladder_l2.ROUNDS[-1] + 1, ladder_l2.MOST_ROUNDS + 1))
STEPS = {
    PREPARE: ladder_l3d2_prepare,
    **{f"ladder_l2_{name}_{number}": ladder_l2._for_round(step, number) for number in LATER_ROUNDS
       for name, step in (("round", ladder_l2.ladder_l2_round), ("train", ladder_l2.ladder_l2_train))},
    TRAIN_WITHOUT: ladder_l3d2_train_without,
    **{measure_marker(arm): _for(ladder_l3d2_measure, arm) for arm in (WITH, WITHOUT)},
    REPORT: ladder_l3d2_report,
}
