"""The ladder loop's L1b, the dose curve: the schedule, the held-out proofs, the training driver, the read and its
branch, and the stage end to end on what an L1 run stored. Spec: docs/spec/ladder-loop.spec.md, "L1b: the dose
curve". The engine, the model and Lean are stand-ins; nothing touches a GPU or the network."""

import json
import random
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_round import _run_stage, stage  # noqa: E402, F401 - the L1 stage on the fixtures, with stand-ins

from rlvr_lean.domain.ladder_round.dose import (  # noqa: E402
    BRANCHES,
    ESCALATE,
    ONE_PASS_STANDS,
    ONE_PASS_STANDS_LATER_WORSE,
    SERIES_KEYS,
    VOID,
    check_nothing_held_out_is_trained_on,
    checkpoint_name,
    checkpoint_steps,
    dose_read,
    epoch_orders,
    heldout_pairs,
    overfitting_by_loss,
    parts_row,
    reading_steps,
    run_dose,
    series,
    steps_per_pass,
    training_sample,
)
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import ladder_dose, ladder_round  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
DOSE = CONFIG["ladder_loop"]["dose"]


# ------------------------------------------------------------------------------------------------ schedule
def test_the_schedule_at_l1_seed_0s_sizes_is_the_specs():
    assert (DOSE["passes"], DOSE["checkpoints"], DOSE["reach_at"]) == (3, [0.5, 1, 1.5, 2, 3], [1, 2, 3])
    assert steps_per_pass(741, CONFIG["training"]["effective_batch"]) == 93          # L1 seed 0 trained 93 steps
    checkpoints = checkpoint_steps(741, 8, DOSE["checkpoints"])
    assert checkpoints == {"p050": 47, "p100": 93, "p150": 140, "p200": 186, "p300": 279}
    assert tuple(checkpoints) == ladder_dose.CHECKPOINTS and checkpoint_name(1.5) == "p150"
    readings = reading_steps(279, DOSE["reading_every_step_until"], DOSE["reading_every"], list(checkpoints.values()))
    assert readings[:14] == [*range(13), 18] and readings[-1] == 279 and len(readings) == 61
    assert set(checkpoints.values()) <= set(readings)                                  # every checkpoint step is a reading point
    assert all(step in readings for step in range(0, 280, 6)) and readings == sorted(set(readings))
    # A dozen proofs (the smoke run): two steps a pass, the five checkpoints at five different steps.
    assert checkpoint_steps(11, 8, DOSE["checkpoints"]) == {"p050": 1, "p100": 2, "p150": 3, "p200": 4, "p300": 6}
    assert reading_steps(6, 12, 6, [1, 2, 3, 4, 6]) == [0, 1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError):
        steps_per_pass(0, 8)


def test_the_first_pass_is_l1s_training_step_for_step():
    """`ladder_round._train` draws its order from ONE generator seeded with the round's seed, reshuffled each
    pass. The dose curve's order is the same, so its first pass feeds the same examples to the same steps."""
    generator, as_l1 = random.Random(0), []
    for _ in range(3):                      # the three lines of `_train`
        order = list(range(741))
        generator.shuffle(order)
        as_l1.append(order)
    assert epoch_orders(741, 3, 0) == as_l1 and epoch_orders(741, 1, 0)[0] == as_l1[0]
    assert as_l1[0] != as_l1[1] and sorted(as_l1[1]) == list(range(741)) and epoch_orders(741, 1, 1)[0] != as_l1[0]


# -------------------------------------------------------------------------------------------------- driver
def test_the_driver_reads_before_the_update_and_saves_and_reads_after_it():
    calls, updates = [], {"made": 0}

    def train_step(batch):
        calls.append(("step", tuple(batch), updates["made"]))
        losses = [[float(updates["made"]), 1.0, 0.5, 0.25] for _ in batch]      # what the model was BEFORE this update
        updates["made"] += 1
        return losses

    def read(step):
        calls.append(("read", step, updates["made"]))
        return {"training_sample": {"mean_loss": 1.0}, "heldout": {"mean_loss": 2.0}}

    def save(name, step):
        calls.append(("save", name, updates["made"]))

    result = run_dose(11, 8, 3, 0, [0, 1, 2, 3, 4, 5, 6], {"p050": 1, "p100": 2, "p150": 3, "p200": 4, "p300": 6}, train_step, read, save)
    assert result["steps"] == 6 and result["steps_per_pass"] == 2
    assert calls[0] == ("read", 0, 0)                                           # step 0 is the base: read before any update
    assert [call[0] for call in calls[1:4]] == ["step", "save", "read"] and calls[2] == ("save", "p050", 1) and calls[3] == ("read", 1, 1)
    steps = [call for call in calls if call[0] == "step"]
    assert [len(call[1]) for call in steps] == [8, 3, 8, 3, 8, 3]               # a pass is every example once
    assert all(sorted(steps[index][1] + steps[index + 1][1]) == list(range(11)) for index in (0, 2, 4))
    assert [call[1] for call in steps[:2]] == [tuple(epoch_orders(11, 3, 0)[0][:8]), tuple(epoch_orders(11, 3, 0)[0][8:])]
    # The loss recorded for step s is the model after s - 1 updates.
    assert [(row["step"], row["updates_before"], row["first_token_nats"]) for row in result["training_steps"]] == [(s, s - 1, float(s - 1)) for s in range(1, 7)]
    assert [row["pass"] for row in result["training_steps"]] == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert result["checkpoints"] == [{"checkpoint": name, "step": step, "pass": passes} for name, step, passes in
                                     (("p050", 1, 0.5), ("p100", 2, 1.0), ("p150", 3, 1.5), ("p200", 4, 2.0), ("p300", 6, 3.0))]
    assert [row["step"] for row in result["readings"]] == [0, 1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError, match="beyond the last step"):
        run_dose(11, 8, 1, 0, [0], {"p300": 6}, train_step, read, save)


def test_a_reading_is_reported_by_part_and_as_arrays_a_page_can_plot():
    row = parts_row([[12.0, 0.5, 0.3, 0.2, 0.1], [10.0, 0.4, 0.2, 0.1]])
    assert row["pairs"] == 2 and row["first_token_nats"] == 11.0 and row["newline_nats"] == pytest.approx(0.2) and row["fence_nats"] == pytest.approx(0.1)
    assert row["body_nats_per_token"] == pytest.approx((0.5 + 0.3 + 0.4) / 3) and set(SERIES_KEYS) <= set(row)
    readings = [{"step": 0, "pass": 0.0, "heldout": row, "training_sample": row}, {"step": 6, "pass": 0.5, "heldout": None, "training_sample": row}]
    assert series(readings, "heldout") == {"step": [0], "pass": [0.0], **{key: [row[key]] for key in SERIES_KEYS}}
    assert series(readings, "training_sample")["step"] == [0, 6]


# --------------------------------------------------------------------------------------------------- pairs
def _groups(per_rung=60):
    return [{"problem_id": f"{rung}{index}", "group": rung} for rung in ("below", "in", "above") for index in range(per_rung)] + \
           [{"problem_id": f"goal{index}", "group": "goal"} for index in range(5)] + [{"problem_id": "nowhere", "group": None}]


def _rung_problems(groups):
    return [{"problem_id": row["problem_id"], "statement": f"theorem {row['problem_id']} : P := by\n",
             "negation": f"theorem negation_of_{row['problem_id']} : ¬ (P) := by\n"} for row in groups if row["group"] in ("below", "in", "above")]


def _base_attempts(groups, verified_of=lambda problem_id: True):
    attempts = []
    for row in groups:
        for episode in range(4):
            side = "negation" if row["problem_id"].endswith("7") else "statement"
            status = "verified" if verified_of(row["problem_id"]) and episode != 0 else "lean_error"
            attempts.append({"attempt_id": f"{row['problem_id']}#rungs_base#{side}#{episode}", "problem_id": row["problem_id"], "side": side,
                             "status": status, "completion": f"  proof {episode} of {row['problem_id']}\n"})
    return attempts


def test_the_held_out_proofs_are_one_verified_proof_per_problem_fifty_per_rung_drawn_with_the_seed():
    groups = _groups()
    problems, attempts = _rung_problems(groups), _base_attempts(groups)
    pairs = heldout_pairs(groups, problems, attempts, seed=0, per_rung=50)
    assert [sum(pair["rung"] == rung for pair in pairs) for rung in ("below", "in", "above")] == [50, 50, 50]
    assert len({pair["problem_id"] for pair in pairs}) == 150                           # one proof per problem
    verified = {attempt["attempt_id"]: attempt for attempt in attempts if attempt["status"] == "verified"}
    assert all(pair["attempt_id"] in verified and pair["completion"] == verified[pair["attempt_id"]]["completion"] for pair in pairs)
    assert all(pair["problem_id"].startswith(pair["rung"]) for pair in pairs)           # never a goal problem, never one outside the rungs
    on_negation = [pair for pair in pairs if pair["side"] == "negation"]
    assert on_negation and all(pair["theorem"].startswith("theorem negation_of_") for pair in on_negation)
    assert heldout_pairs(groups, problems, attempts, seed=0, per_rung=50) == pairs      # the same seed, the same draw
    other = heldout_pairs(groups, problems, attempts, seed=1, per_rung=50)
    assert {pair["problem_id"] for pair in other} != {pair["problem_id"] for pair in pairs}
    assert len({pair["attempt_id"].rsplit("#", 1)[1] for pair in pairs}) > 1            # not always a problem's first proof
    # A rung with fewer proved problems than wanted gives all it has.
    short = heldout_pairs(groups, problems, _base_attempts(groups, lambda problem_id: not problem_id.startswith("below") or problem_id < "below2"), 0, 50)
    assert sum(pair["rung"] == "below" for pair in short) == 12 and sum(pair["rung"] == "in" for pair in short) == 50      # below0, below1, below10 to below19


def test_nothing_held_out_may_be_a_training_example():
    groups = _groups()
    pairs = heldout_pairs(groups, _rung_problems(groups), _base_attempts(groups), 0, 50)
    examples = [{"problem_id": f"pool{index}", "attempt_id": f"pool{index}#round#statement#0"} for index in range(300)]
    check_nothing_held_out_is_trained_on(examples, pairs, groups)
    with pytest.raises(ValueError, match="held-out problems"):
        check_nothing_held_out_is_trained_on([*examples, {"problem_id": "goal3", "attempt_id": "x"}], pairs, groups)
    sample = training_sample(examples, seed=0, size=150)
    assert len(sample) == 150 == len(set(sample)) and sample == sorted(sample) and training_sample(examples, 0, 150) == sample
    assert training_sample(examples, 1, 150) != sample and training_sample(examples[:20], 0, 150) == list(range(20))


# ---------------------------------------------------------------------------------------------------- read
def _reading(step, heldout, training):
    return {"step": step, "pass": step / 10, "heldout": {"body_nats_per_token": heldout}, "training_sample": {"body_nats_per_token": training}}


def test_the_loss_half_of_the_overfitting_rule():
    falling = [_reading(step, 0.5 - 0.02 * step, 0.5 - 0.03 * step) for step in range(8)]
    assert overfitting_by_loss(falling, 7)["by_loss"] is False                  # the held-out loss is at its lowest
    turned = falling + [_reading(8, 0.40, 0.25), _reading(9, 0.41, 0.22), _reading(10, 0.42, 0.20), _reading(11, 0.43, 0.18)]
    found = overfitting_by_loss(turned, 11)
    assert found["lowest_at_step"] == 7 and found["above_its_lowest_by"] == pytest.approx(0.07) and found["spread_of_the_last_four_readings"] == pytest.approx(0.03)
    assert found["training_sample_still_falls"] is True and found["by_loss"] is True
    # Above its lowest point by LESS than the spread of the last four readings: noise, not a turn.
    noisy = falling + [_reading(8, 0.37, 0.25), _reading(9, 0.36, 0.22), _reading(10, 0.39, 0.20), _reading(11, 0.37, 0.18)]
    assert overfitting_by_loss(noisy, 11)["by_loss"] is False
    # The held-out loss rose, and the training-sample loss did too: that is not overfitting by this rule.
    both_up = falling + [_reading(8, 0.45, 0.40), _reading(9, 0.46, 0.41), _reading(10, 0.47, 0.42), _reading(11, 0.48, 0.43)]
    assert overfitting_by_loss(both_up, 11)["by_loss"] is False
    assert overfitting_by_loss(turned, 7)["by_loss"] is False                   # read at an earlier checkpoint: only what was known then
    assert overfitting_by_loss([], 5) == {"readings": 0, "by_loss": False}


CHECKPOINTS = {"p050": {"step": 5, "pass": 0.5}, "p100": {"step": 10, "pass": 1}, "p150": {"step": 15, "pass": 1.5},
               "p200": {"step": 20, "pass": 2}, "p300": {"step": 30, "pass": 3}}
REFERENCE = {"below": {"mean": 0.0, "low": -0.05, "high": 0.05}, "in": {"mean": 0.125, "low": 0.05, "high": 0.2},
             "above": {"mean": 0.125, "low": 0.05, "high": 0.2}}


def _rows(groups, resolved_of):
    return [{"problem_id": row["problem_id"], "resolved": resolved_of(row), "episodes": 8} for row in groups if row["group"] in ("below", "in", "above")]


def _read(changes, readings=(), reference=REFERENCE):
    """Base: 1 of 8 below, 2 of 8 in, 6 of 8 above. One pass: +0, +1, +1 of 8 (what the reference reports).
    `changes[name][rung]` adds to that, in eighths, on every problem of the rung."""
    groups = _groups(per_rung=40)
    base = {"below": 1, "in": 2, "above": 6}
    one = {"below": 1, "in": 3, "above": 7}
    rungs = {name: _rows(groups, lambda row, name=name: one[row["group"]] + changes.get(name, {}).get(row["group"], 0)) for name in CHECKPOINTS}
    return dose_read(CHECKPOINTS, groups, rungs, _rows(groups, lambda row: base[row["group"]]), list(readings), reference, resamples=200, seed=0)


def test_no_checkpoint_differs_and_one_pass_stands():
    read = _read({})
    assert read["branch"]["name"] == ONE_PASS_STANDS and "half a pass minus one pass" in read["branch"]["reason"]
    assert read["void_check"]["reproduces_l1_at_one_pass"] is True and read["one_pass_checkpoint"] == "p100"
    assert read["did_l1_measure_at_the_peak"]["at_the_peak"] is True and read["are_we_overfitting"]["yes_at"] == []
    assert set(read["would_training_longer_help"]) == {"p200", "p300"} and set(read["would_training_longer_help"]["p200"]) == {"below", "in"}
    assert read["checkpoints"]["p100"]["minus_base"]["in"]["mean"] == 0.125 and read["checkpoints"]["p100"]["minus_one_pass"] is None
    assert read["checkpoints"]["p050"]["overfitting"]["asked"] is False     # the rule is asked of checkpoints later than one pass


def test_a_later_checkpoint_that_beats_one_pass_on_the_below_band_rung_escalates_its_passes():
    read = _read({"p200": {"below": 1}, "p300": {"below": 2}})
    assert read["branch"]["name"] == ESCALATE and read["branch"]["passes"] == 3 and "three seeds" in read["branch"]["reason"]
    assert read["did_l1_measure_at_the_peak"] == {"highest_pass_rate_on_below_and_in_together": "p300", "checkpoints_that_beat_one_pass": ["p200", "p300"],
                                                  "at_the_peak": False}
    assert read["would_training_longer_help"]["p300"]["below"]["mean"] == 0.25 and read["would_training_longer_help"]["p300"]["in"]["mean"] == 0.0
    # A gain on the in-band rung alone is reported and is not the branch: the primary is the below-band rung.
    assert _read({"p300": {"in": 1}})["branch"]["name"] == ONE_PASS_STANDS


def test_later_checkpoints_that_are_worse_leave_one_pass_standing_and_record_the_overfitting_point():
    turned = [_reading(step, 0.5 - 0.02 * step, 0.5 - 0.015 * step) for step in range(11)] + \
             [_reading(step, 0.30 + 0.02 * (step - 10), 0.35 - 0.01 * (step - 10)) for step in range(11, 31)]
    read = _read({"p200": {"in": -1}, "p300": {"in": -1, "above": -2}}, turned)
    assert read["branch"]["name"] == ONE_PASS_STANDS_LATER_WORSE and "p200, p300" in read["branch"]["reason"]
    assert read["are_we_overfitting"]["yes_at"] == ["p200", "p300"] and read["checkpoints"]["p300"]["overfitting"]["rungs_below"] == ["in", "above"]
    # p150: the loss has turned, and no rung is below its one-pass value: loss alone is not a verdict.
    assert read["checkpoints"]["p150"]["overfitting"]["loss"]["by_loss"] is True and read["checkpoints"]["p150"]["overfitting"]["yes"] is False
    assert read["are_we_overfitting"]["by_loss_alone_at"] == ["p150", "p200", "p300"]
    # A pass rate that fell with a loss that did not turn is not overfitting by the rule either; the branch still says worse.
    flat = _read({"p300": {"in": -1}})
    assert flat["are_we_overfitting"]["yes_at"] == [] and flat["branch"]["name"] == ONE_PASS_STANDS_LATER_WORSE


def test_a_one_pass_checkpoint_that_does_not_reproduce_l1_is_void():
    wrong = {**REFERENCE, "above": {"mean": 0.3, "low": 0.25, "high": 0.35}}      # L1 reported more than this run's one pass gives
    read = _read({"p300": {"below": 2}}, reference=wrong)
    assert read["branch"]["name"] == VOID and "above rung" in read["branch"]["reason"]
    assert read["void_check"]["by_rung"]["above"] == {"measured": 0.125, "l1_reported": 0.3, "l1_interval": [0.25, 0.35], "within": False}
    assert read["void_check"]["by_rung"]["in"]["within"] is True
    assert {VOID, ESCALATE, ONE_PASS_STANDS, ONE_PASS_STANDS_LATER_WORSE} == set(BRANCHES)


# ---------------------------------------------------------------------------- the stage, with stand-ins
def _run_dose_stage(config):
    summaries = {}
    for environment, step, _ in map(entry.step_fields, entry.STAGES["ladder_l1b"]):
        if environment == "gpu" and step in ladder_dose.STEPS:
            summaries[step] = ladder_dose.STEPS[step](config)
    return summaries


def _files(directory):
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir()) if path.is_file()}


def test_the_whole_stage_runs_on_what_an_l1_run_stored_and_writes_a_run_of_its_own(stage, monkeypatch):  # noqa: F811
    for name in (ladder_dose.DOSE_RUN_VARIABLE, ladder_dose.DOSE_SOURCE_VARIABLE):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="stage `ladder_l1` with --seeds 0"):            # nothing to read yet: refused, and it says what to run
        ladder_dose.ladder_l1b_prepare(stage.config)
    assert not ladder_dose.source_directory(stage.config).exists()                        # and the refusal created nothing
    _run_stage(stage.config)
    source = ladder_dose.source_directory(stage.config)
    assert source == ladder_round._store(stage.config).root
    before = _files(source)
    summaries = _run_dose_stage(stage.config)
    assert _files(source) == before                                                        # L1's run directory was only read
    store = ladder_dose._store(stage.config)
    assert store.root.name == "ladder_l1b_seed0" and store.root.parent == source.parent
    prepare = summaries["ladder_l1b_prepare"]
    examples = store.read_rows("training_examples.jsonl")
    assert examples == [json.loads(line) for line in (source / "training_examples_challenger.jsonl").read_text().splitlines()]
    assert prepare["training_examples"] == len(examples) and prepare["steps"] == 3 * prepare["steps_per_pass"]
    pairs = store.read_rows("heldout_pairs.jsonl")
    groups = {row["problem_id"]: row["group"] for row in store.read_rows("heldout_groups.jsonl")}
    assert pairs and all(groups[pair["problem_id"]] == pair["rung"] for pair in pairs) and len({pair["problem_id"] for pair in pairs}) == len(pairs)
    assert not {pair["problem_id"] for pair in pairs} & {example["problem_id"] for example in examples}
    assert prepare["sampling_seeds"] == {"rungs": ladder_round.sampling_seed(stage.config, "rungs_base"), "reach": ladder_round.sampling_seed(stage.config, "reach_base")}
    train = summaries["ladder_l1b_train"]
    assert train["steps"] == prepare["steps"] and [row["checkpoint"] for row in train["checkpoints"]] == list(ladder_dose.CHECKPOINTS)
    assert "made up" in train["note"] and train["stand_in_engine"] is True
    base_rungs = {row["problem_id"]: row for row in store.read_rows("base_rungs.jsonl")}
    for name in ladder_dose.CHECKPOINTS:
        rows = store.read_rows(f"episodes_rungs_{name}_problems.jsonl")
        assert {row["problem_id"] for row in rows} == set(base_rungs) and all(row["episodes"] == 8 for row in rows)
        measured = summaries[f"ladder_l1b_measure_{name}"]
        assert measured["rungs"]["sampling_seed"] == prepare["sampling_seeds"]["rungs"] and ("reach" in measured) == (name in ("p100", "p200", "p300"))
        assert measured["distinct_attempts_on_the_rungs"]["prompts"] >= 5
        # The same sides as L1's base measurement, problem by problem: each checkpoint pairs with it.
        assert {row["problem_id"]: row["sides"] for row in rows} == {key: row["sides"] for key, row in base_rungs.items()}
    assert summaries["ladder_l1b_measure_p300"]["reach"]["sampling_seed"] == prepare["sampling_seeds"]["reach"]
    report = summaries["ladder_l1b_report"]
    assert report["branch"]["name"] in BRANCHES and report["stand_in_engine"] is True and report["ok"] is True
    curves = report["loss_curves"]
    assert curves["training_step"]["step"] == list(range(1, prepare["steps"] + 1)) and curves["steps_per_pass"] == prepare["steps_per_pass"]
    for name in ("training_step", "training_sample", "heldout"):
        assert set(SERIES_KEYS) | {"step", "pass"} <= set(curves[name]) and len({len(values) for values in curves[name].values()}) == 1
    assert curves["heldout"]["step"] == prepare["reading_steps"] and set(curves["heldout_by_rung"]) <= {"below", "in", "above"}
    assert [entry_["checkpoint"] for entry_ in curves["checkpoints"]] == list(ladder_dose.CHECKPOINTS)
    assert set(report["pass_rate_curve"]) == {"below", "in", "above"} and report["pass_rate_curve"]["in"]["pass"] == [0.5, 1, 1.5, 2, 3]
    assert set(report["reach_on_g"]["by_checkpoint"]) == {"p100", "p200", "p300"} and set(report["distinct_attempts_on_the_rungs"]) == {"base", *ladder_dose.CHECKPOINTS}
    assert set(report["void_check"]["by_rung"]) == {"in", "above"} and set(report["checkpoints"]) == set(ladder_dose.CHECKPOINTS)
    assert json.loads(store.path("report_ladder_l1b.json").read_text())["headline"] == report["headline"]
    # A rerun returns what is stored: nothing is trained or sampled again.
    assert _run_dose_stage(stage.config)["ladder_l1b_train"] == train and not ladder_round._store(stage.config).is_done("ladder_l1b_train")


def test_a_finished_run_asked_for_reach_at_one_more_checkpoint_measures_that_and_nothing_else(stage, monkeypatch):  # noqa: F811
    for name in (ladder_dose.DOSE_RUN_VARIABLE, ladder_dose.DOSE_SOURCE_VARIABLE):
        monkeypatch.delenv(name, raising=False)
    _run_stage(stage.config)
    stage.config["ladder_loop"]["dose"]["reach_at"] = [1, 3]                    # the run as seed 0 was made
    first = _run_dose_stage(stage.config)
    store = ladder_dose._store(stage.config)
    assert "reach" not in first["ladder_l1b_measure_p200"] and set(first["ladder_l1b_report"]["reach_on_g"]["by_checkpoint"]) == {"p100", "p300"}
    assert not any(row["set"] == "reach_p200" for row in store.read_rows("problems.jsonl"))
    kept = {name: content for name, content in _files(store.root).items()
            if name not in ("problems.jsonl", "ladder_l1b_measure_p200.done.json", "report_ladder_l1b.json", "ladder_l1b_report.done.json")}
    stage.config["ladder_loop"]["dose"]["reach_at"] = [1, 2, 3]                 # the escalation asks for G at 2 passes too
    second = _run_dose_stage(stage.config)
    added = second["ladder_l1b_measure_p200"]
    assert added["reach"]["sampling_seed"] == first["ladder_l1b_prepare"]["sampling_seeds"]["reach"]
    assert added["rungs"] == first["ladder_l1b_measure_p200"]["rungs"]          # the rung results are the stored ones
    after = _files(store.root)
    assert all(after[name] == content for name, content in kept.items())      # nothing else was measured or written again
    assert {name for name in after if name not in kept and "reach_p200" not in name} == {
        "problems.jsonl", "ladder_l1b_measure_p200.done.json", "report_ladder_l1b.json", "ladder_l1b_report.done.json"}
    goal = [row["problem_id"] for row in store.read_rows("problems.jsonl") if row["set"] == "reach_p300"]
    assert [row["problem_id"] for row in store.read_rows("problems.jsonl") if row["set"] == "reach_p200"] == goal
    assert set(second["ladder_l1b_report"]["reach_on_g"]["by_checkpoint"]) == {"p100", "p200", "p300"}
    assert second["ladder_l1b_report"]["pass_rate_curve"] == first["ladder_l1b_report"]["pass_rate_curve"]
    assert _run_dose_stage(stage.config)["ladder_l1b_measure_p200"] == added   # and a third run adds nothing more


def test_a_measurement_that_would_not_pair_with_l1s_is_refused(stage, monkeypatch):  # noqa: F811
    for name in (ladder_dose.DOSE_RUN_VARIABLE, ladder_dose.DOSE_SOURCE_VARIABLE):
        monkeypatch.delenv(name, raising=False)
    _run_stage(stage.config)
    stage.config["ladder_loop"]["round"]["sampling_seed"] += 7                 # another sampling seed than L1 stored
    with pytest.raises(RuntimeError, match="would not pair with L1's stored results"):
        ladder_dose.ladder_l1b_prepare(stage.config)
    stage.config["ladder_loop"]["round"]["sampling_seed"] -= 7
    stage.config["ladder_loop"]["dose"]["checkpoints"] = [1, 2, 3]
    with pytest.raises(ValueError, match="change both together"):
        ladder_dose.ladder_l1b_prepare(stage.config)


def test_the_stage_is_registered_and_its_smoke_run_reads_the_l1_smoke_run():
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l1b"]]
    gpu_steps = [step for environment, step, _ in steps if environment == "gpu"]
    assert gpu_steps == ["fix_tokenizers", "ladder_l1b_prepare", "ladder_l1b_train", *(f"ladder_l1b_measure_{name}" for name in ladder_dose.CHECKPOINTS),
                         "ladder_l1b_report"]
    assert all(step in ladder_dose.STEPS for step in gpu_steps[1:]) and all(options == {} for _, _, options in steps)
    assert not set(ladder_dose.STEPS) & set(ladder_round.STEPS)
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l1b_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in steps]
    variables = entry.child_environment("gpu", "key", smoke[2][2])
    assert variables[ladder_dose.DOSE_SOURCE_VARIABLE] == "ladder_l1_smoke" and variables[ladder_dose.DOSE_RUN_VARIABLE] == "ladder_l1b_smoke"
    assert ladder_dose.DOSE_RUN_VARIABLE not in entry.child_environment("gpu", "key", steps[2][2])
