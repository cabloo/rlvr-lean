"""L3d Step 2, the stage end to end on what an L1 run and two L2 runs stored: the arm's six rounds with assembly, the
twin trained on the one-shot rows in the last model's order, the two measurements with L2's sampling seeds, the report,
the adapters kept, a failed check, the smoke stage, the registration. Spec: docs/spec/ladder-loop.spec.md, "L3d:
train on what the episode reaches", Step 2. The engine and Lean are scripted; nothing touches a GPU or the network. The
arm itself is `test_ladder_assembly.py`; the rules and the report on hand-made rows are `test_ladder_l3d2.py`."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_assembly import ARM_NAME, ASSEMBLES, arm  # noqa: E402, F401 - the arm on the fixtures, with the scripted solver and Lean
from test_ladder_l2_stage import ARM, _run_loop, loop  # noqa: E402, F401
from test_ladder_round import _run_stage, stage  # noqa: E402, F401

from rlvr_lean.domain.ladder_round.dose import run_dose  # noqa: E402
from rlvr_lean.domain.ladder_round.l3d2 import BRANCHES, INCONCLUSIVE, NOT_READ  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import __main__ as gpu_main  # noqa: E402
from rlvr_lean.gpu import ladder_assembly, ladder_ceiling, ladder_dose, ladder_l2, ladder_l3d1, ladder_l3d2, ladder_round, milestone2, pipeline  # noqa: E402
from rlvr_lean.gpu.ladder_ceiling import MORE, REACH  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
FIXTURE = PACKAGE / "data" / "ladder_l3d_fixture" / "harvest_h0.jsonl"
VARIABLES = (ladder_l3d2.STORED_VARIABLE, ladder_l3d2.MINIMUM_VARIABLE)
STEPS = {**ladder_l2.STEPS, **ladder_l3d2.STEPS}


def _file_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _files(directory):
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir()) if path.is_file()}


def _run(config, stage_name="ladder_l3d2", until=None):
    summaries = {}
    for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
        if environment == "gpu" and step in STEPS:
            summaries[step] = STEPS[step](config)
            if step == until:
                break
    return summaries


@pytest.fixture
def step_2(arm, monkeypatch):  # noqa: F811
    """The arm's world (six rounds of two problems, the scripted solver and Lean, an H0 of three rows) with what the
    stage reads beside its models: `stored()` runs L1, then L2 and L2's arm t010, with the stand-in engine. Three
    assembled proofs are asked for, not 150."""
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ladder_l3d2.MINIMUM_VARIABLE, "3")

    def stored():
        monkeypatch.delenv(ARM)
        _run_stage(arm.config)
        _run_loop(arm.config)
        monkeypatch.setenv(ARM, "t010")
        _run_loop(arm.config)
        monkeypatch.setenv(ARM, ARM_NAME)
        arm.with_lean()

    return SimpleNamespace(config=arm.config, stored=stored, store=arm.store, solver=arm.solver, lean=arm.lean)


# ------------------------------------------------------------------------------------------ the whole stage
def test_the_whole_stage_runs_the_arm_its_twin_the_two_measurements_and_the_report(step_2):
    config = step_2.config
    step_2.stored()
    source, stored = ladder_l2.source_directory(config), ladder_ceiling.stored_directories(config, ladder_l3d2.STORED_VARIABLE, ladder_l3d2.SETTING)
    assert [path.name for path in stored.values()] == ["ladder_l2_seed0", "ladder_l2_t010_seed0"]
    before = {path: _files(path) for path in (source, *stored.values())}
    summaries = _run(config)
    assert {path: _files(path) for path in before} == before                              # the three run directories were only read
    store = step_2.store()
    assert store.root.name == "ladder_l2_t010_assembly_seed0" and list(summaries)[:3] == ["ladder_l2_prepare", "ladder_l3d2_prepare", "ladder_l2_embed"]
    assert [step for step in summaries if "round" in step or "train" in step] == [
        *(f"ladder_l2_{name}_{number}" for number in range(1, 7) for name in ("round", "train")), "ladder_l3d2_train_without"]

    # ---- the stage's own prepare: what the two models are read against, and their sets beside the arm's
    prepare = summaries["ladder_l3d2_prepare"]
    of_l2 = ladder_l2.sampling_seeds(config)
    control = json.loads((stored["base"] / "episodes_control.done.json").read_text())
    assert (prepare["stage"], prepare["arm"], prepare["rounds"], prepare["loop_arm"], prepare["minimum_assembled"]) == ("l3d2", ARM_NAME, [1, 2, 3, 4, 5, 6], "t010", 3)
    assert prepare["sampling_seeds"] == {"rungs": of_l2["rungs"], "reach": of_l2["reach"], "more": of_l2["control"]}
    assert prepare["goal_samplings"] == [{"name": "reach", "episodes": 32, "sampling_seed": of_l2["reach"]},
                                         {"name": "more", "episodes": control["episodes_each"], "sampling_seed": of_l2["control"]}]
    sets = {row["set"] for row in store.read_rows("problems.jsonl")}
    assert sets == {f"l3d2_{part}_{model}" for part in ("rungs", REACH, MORE) for model in ("with", "without")} | {f"round_r{number}_b{batch}" for number in range(1, 7) for batch in (1, 2)}

    # ---- the twin: the one-shot rows of M(6)'s training set, in its order with the other rows left out; the same code and seed
    with_rows, without_rows = store.read_rows(ladder_assembly.training_set_file(6)), store.read_rows(ladder_l3d2.TRAINING_WITHOUT_FILE)
    assert [row["id"] for row in without_rows] == [row["id"] for row in with_rows if row["origin"] == "attempt"] and {row["origin"] for row in without_rows} == {"attempt"}
    assert [row["row"] for row in without_rows] == list(range(len(without_rows))) and [row["row_of_with"] for row in without_rows] == [
        row["row"] for row in with_rows if row["origin"] == "attempt"]
    assert sum(row["origin"] == "assembled" for row in with_rows) == len(ASSEMBLES) and sum(row["origin"] == "h0" for row in with_rows) == 2
    twin = summaries["ladder_l3d2_train_without"]
    assert (twin["rows"], twin["rows_of_with"], twin["left_out"]) == (len(without_rows), len(with_rows), {"assembled": 4, "h0": 2}) and "made up" in twin["note"]
    loss = json.loads(store.path(ladder_l3d2.LOSS_WITHOUT_FILE).read_text())
    assert loss["rows_trained"] == [row["id"] for row in without_rows] and len(loss["row_losses"]) == len(without_rows)

    # ---- the measurements: both models as the ceiling's are, pairing by problem with the stored base
    base = {part: {row["problem_id"] for row in store.read_rows(ladder_ceiling.stored_file("base", part, "l3d2"))} for part in ("rungs", REACH, MORE)}
    for model in ("with", "without"):
        measured = summaries[f"ladder_l3d2_measure_{model}"]
        assert (measured["stage"], measured["model"]) == ("l3d2", model) and measured["rungs"]["sampling_seed"] == of_l2["rungs"] and measured["rungs"]["episodes_each"] == 8
        assert (measured["goal"][REACH]["sampling_seed"], measured["goal"][REACH]["episodes_each"]) == (of_l2["reach"], 32)
        assert (measured["goal"][MORE]["sampling_seed"], measured["goal"][MORE]["episodes_each"]) == (of_l2["control"], control["episodes_each"])
        for part in ("rungs", REACH, MORE):
            assert {row["problem_id"] for row in store.read_rows(f"episodes_l3d2_{part}_{model}_problems.jsonl")} == base[part]
        assert set(measured["opens_with_a_have"]) == {"rungs", "goal"} and len(measured["episodes_of_8"]) == 1
    assert summaries["ladder_l3d2_measure_with"]["is"] == "M(6)" and summaries["ladder_l3d2_measure_without"]["is"] == "the twin of M(6)"

    # ---- the report: the four checks first, then the read
    report = summaries["ladder_l3d2_report"]
    assert report["stage"] == "l3d2" and report["branch"]["name"] in BRANCHES and report["ok"] is True and report["arm"] == ARM_NAME
    assert all(line.startswith("l3d2: ") for line in report["lines"]) and [line.split(": ", 1)[1][:7] for line in report["lines"][1:5]] == ["CHECK 1", "CHECK 2", "CHECK 3", "CHECK 4"]
    checks = report["can_this_run_see_a_win"]
    assert checks["with_was_trained_on_enough_assembled_proofs"] == {**checks["with_was_trained_on_enough_assembled_proofs"], "assembled_proofs": 6, "from_the_rounds": 4,
                                                                    "from_h0": 2, "minimum": 3, "passes": True}
    record = checks["every_assembled_row_is_in_withs_record_once_and_none_in_withouts"]
    assert (record["assembled_rows"], record["counted_once_in_with"], record["counted_in_without"], record["passes"]) == (6, 6, 0, True)
    assert report["inconclusive"] == (report["branch"]["name"] == INCONCLUSIVE) == (not all(check["passes"] for check in checks.values() if isinstance(check, dict)))
    read = report["measured_and_not_read"] if report["inconclusive"] else report
    table = read["secondary"]["by_round"]["rows"]
    assert [row["round"] for row in table] == [1, 2, 3, 4, 5, 6] and sum(row["only_assembly_resolved"] for row in table) == 4 and all(row["picks"] == 2 for row in table)
    assert sum(row["training_rows"]["attempt"] for row in table) == 6 and sum(row["resolved_by_an_attempt"] for row in table) == 6
    assert all(row["share_of_the_unresolved"] == (row["only_assembly_resolved"] / row["left_unresolved_by_the_attempts"] if row["left_unresolved_by_the_attempts"] else None)
               for row in table)
    assert set(read["secondary"]["goal_problems_solved_by_attempts_alone"]) == {"what", "base", "without", "with", "loop"}
    assert read["primary"]["mean"] is None and read["secondary"]["by_length_group"]["problems"]["not_known"] == 1       # the fixture's goal problem has no shipped length
    assert report["adapters"]["kept"] is True and json.loads(store.path(ladder_l3d2.REPORT_FILE).read_text()) == json.loads(json.dumps(report, default=str))
    # The stage's own prepare step, made again on a run that has its rounds, puts its sets BESIDE the arm's: no batch's problems are lost.
    store.path(f"{ladder_l3d2.PREPARE}.done.json").unlink()
    assert ladder_l3d2.ladder_l3d2_prepare(config) == prepare and {row["set"] for row in store.read_rows("problems.jsonl")} == sets
    # A rerun returns what is stored: nothing is sampled, checked or trained again.
    calls, sent = step_2.solver.calls, len(step_2.lean.sources)
    again = _run(config)
    assert (step_2.solver.calls, len(step_2.lean.sources)) == (calls, sent) and all(again[step] == summaries[step] for step in summaries if step != "ladder_l3d2_report")


def test_both_models_adapters_are_made_by_the_same_code_and_kept_and_a_twin_trained_on_other_rows_makes_the_run_inconclusive(step_2, monkeypatch):
    config, calls = step_2.config, []
    step_2.stored()

    def train_with_readings(config, examples, sample, pairs, schedule, seed, directory, tensorboard_run, orders=None, per_example=False, positions=False):
        calls.append({"examples": examples, "schedule": schedule, "seed": seed, "directory": directory, "tensorboard_run": tensorboard_run, "orders": orders})
        train_step, read, _ = ladder_dose._stand_in_calls(len(examples), [], [])

        def save(name, step_number):
            (directory / name).mkdir(parents=True)

        return {**run_dose(len(examples), config["training"]["effective_batch"], schedule["passes"], seed, schedule["reading_steps"], schedule["checkpoint_steps"],
                           train_step, read, save, orders=orders, per_example=per_example, positions=positions), "target_format": "native", "seconds": 1.0}

    monkeypatch.setattr(ladder_dose, "_train_with_readings", train_with_readings)
    monkeypatch.setattr(pipeline, "_cap_torch_memory", lambda config: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None, memory_allocated=lambda: 0)))
    with pytest.raises(RuntimeError, match="the training of `without` needs the step ladder_l3d2_prepare of this run, which is not done: the stage `ladder_l3d2`"):
        ladder_l3d2.ladder_l3d2_train_without(config)
    with monkeypatch.context() as patched:                                               # the trainings, and only they, take their GPU path
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        _run(config, until="ladder_l3d2_train_without")
    store = step_2.store()
    # Seven trainings by one function: M(1) to M(6), then the twin; each from the base with the task's seed, in the stored order, an adapter at its end.
    assert [call["tensorboard_run"] for call in calls] == [*(f"ladder_l2_{ARM_NAME}_m{number}_seed0" for number in range(1, 7)), "ladder_l3d2_without_seed0"]
    assert {call["seed"] for call in calls} == {0} and all(call["orders"] == [list(range(len(call["examples"])))] and call["directory"] == store.root / "adapters" for call in calls)
    assert [list(call["schedule"]["checkpoint_steps"]) for call in calls] == [*([f"m{number}"] for number in range(1, 7)), ["without"]]
    with_rows = store.read_rows(ladder_assembly.training_set_file(6))
    assert calls[5]["examples"] == [{key: row[key] for key in ("problem_id", "side", "theorem", "completion")} for row in with_rows]
    assert calls[6]["examples"] == [example for example, row in zip(calls[5]["examples"], with_rows) if row["origin"] == "attempt"]       # `with`'s order, rows left out
    assert sorted(path.name for path in (store.root / "adapters").iterdir()) == ["m1", "m2", "m3", "m4", "m5", "m6", "without"]
    assert ladder_l3d2.adapter_of(config, store, "with") == store.root / "adapters" / "m6" and ladder_l3d2.adapter_of(config, store, "without") == store.root / "adapters" / "without"
    # A measurement of a model whose adapter is not there is refused.
    (store.root / "adapters" / "without").rename(store.root / "adapters" / "elsewhere")
    with monkeypatch.context() as patched:
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        with pytest.raises(RuntimeError, match="this run keeps its adapters, and one trained again would not be the model"):
            ladder_l3d2.ladder_l3d2_measure(config, "without")
    (store.root / "adapters" / "elsewhere").rename(store.root / "adapters" / "without")
    with pytest.raises(RuntimeError, match="L3d Step 2's report needs the step ladder_l3d2_measure_with"):
        ladder_l3d2.ladder_l3d2_report(config)
    for model in ("with", "without"):
        ladder_l3d2.ladder_l3d2_measure(config, model)
    report = ladder_l3d2.ladder_l3d2_report(config)
    assert report["adapters"]["kept"] is True and report["adapters"]["there"] == ["m1", "m2", "m3", "m4", "m5", "m6", "without"]
    # A twin whose record holds an assembled row: the fourth check fails, and the report says INCONCLUSIVE and nothing else.
    loss = json.loads(store.path(ladder_l3d2.LOSS_WITHOUT_FILE).read_text())
    assembled = next(row["id"] for row in with_rows if row["origin"] == "assembled")
    store.path(ladder_l3d2.LOSS_WITHOUT_FILE).write_text(json.dumps({**loss, "rows_trained": [*loss["rows_trained"], assembled]}))
    broken = ladder_l3d2.ladder_l3d2_report(config)
    assert broken["branch"]["name"] == INCONCLUSIVE and "every_assembled_row_is_in_withs_record_once_and_none_in_withouts" in broken["branch"]["failed_checks"]
    assert "every_assembled_row_is_in_withs_record_once_and_none_in_withouts" not in report["branch"].get("failed_checks", [])       # it passed before the record was changed
    assert broken["primary"] is None and not any("PRIMARY" in line or "SECONDARY" in line for line in broken["lines"]) and broken["lines"][4].endswith("FAIL")
    assert broken["can_this_run_see_a_win"]["with_was_trained_on_enough_assembled_proofs"]["assembled_proofs"] == 5      # that row is no longer `with`'s alone


def test_a_twin_whose_training_loop_missed_rows_makes_the_run_inconclusive(step_2, monkeypatch):
    config = step_2.config
    step_2.stored()
    real = ladder_ceiling.one_pass

    def a_pass_that_misses_rows(config, examples, steps, seed, batch, adapters, tensorboard_run, positions=False):
        result, step_rows, row_losses, trained = real(config, examples, steps, seed, batch, adapters, tensorboard_run, positions=positions)
        return result, step_rows, row_losses, trained[:-1]                             # the loop's own record: the last row was not trained on

    monkeypatch.setattr(ladder_l3d2, "one_pass", a_pass_that_misses_rows)             # the twin's training; the arm's own are as they are
    report = _run(config)["ladder_l3d2_report"]
    record = report["can_this_run_see_a_win"]["every_assembled_row_is_in_withs_record_once_and_none_in_withouts"]
    assert record["passes"] is False and record["as_prepared"] == {"without": False, "with": True} and record["rows_trained"]["without"] == record["rows_prepared"]["without"] - 1
    assert report["branch"]["name"] == INCONCLUSIVE and report["primary"] is None


def test_the_stage_needs_its_arm_and_the_stored_runs_before_anything_is_sampled(step_2, monkeypatch):
    config = step_2.config
    monkeypatch.delenv(ARM)
    _run_stage(config)
    for named in (None, "t010"):
        if named:
            monkeypatch.setenv(ARM, named)
        with pytest.raises(RuntimeError, match=f"the stage `ladder_l3d2` runs the arm t010_assembly .* names {named!r}"):
            ladder_l3d2.ladder_l3d2_prepare(config)
    monkeypatch.setenv(ARM, ARM_NAME)
    with pytest.raises(RuntimeError, match="L3d Step 2's own prepare step needs the step ladder_l2_prepare of this run"):
        ladder_l3d2.ladder_l3d2_prepare(config)
    ladder_l2.ladder_l2_prepare(config)
    with pytest.raises(RuntimeError, match=r"L3d Step 2 reads the run directory the task of stage `ladder_l2` for seed 0 .*the base's attempts on G"):
        ladder_l3d2.ladder_l3d2_prepare(config)                                          # refused before the first round: nothing was sampled
    assert step_2.solver.calls == 0 and not step_2.store().is_done(ladder_l3d2.PREPARE)
    monkeypatch.setenv(ladder_l3d2.STORED_VARIABLE, "ladder_l2_smoke")
    with pytest.raises(ValueError, match=f"{ladder_l3d2.STORED_VARIABLE} is 'ladder_l2_smoke': it is `none`"):
        ladder_l3d2.ladder_l3d2_prepare(config)
    monkeypatch.delenv(ladder_l3d2.MINIMUM_VARIABLE)
    assert ladder_l3d2.minimum_assembled(config) == 150 == config["ladder_loop"]["l3d"]["step_2"]["minimum_assembled"]


# ------------------------------------------------------------------------------------------- the smoke stage
def test_the_smoke_stage_runs_two_rounds_on_the_l1_smoke_run_with_the_h0_fixture(arm, monkeypatch):  # noqa: F811
    config = arm.config
    monkeypatch.delenv(ARM)
    monkeypatch.setenv(ladder_round.ROUND_RUN_VARIABLE, "ladder_l1_smoke")
    _run_stage(config)                                                         # what the L1 smoke task leaves on the box
    monkeypatch.delenv(ladder_round.ROUND_RUN_VARIABLE)
    options = entry.step_fields(entry.STAGES["ladder_l3d2_smoke"][2])[2]
    assert all(entry.step_fields(step)[2] == options for step in entry.STAGES["ladder_l3d2_smoke"])
    of_l2_smoke = entry.step_fields(entry.STAGES["ladder_l2_smoke"][2])[2]["environment"]
    assert options["environment"] == {**of_l2_smoke, ladder_l2.L2_RUN_VARIABLE: "ladder_l3d2_smoke", ARM: ARM_NAME, ladder_l2.L2_ROUNDS_VARIABLE: "2",
                                      ladder_l3d2.STORED_VARIABLE: "none", ladder_l3d2.MINIMUM_VARIABLE: "8", ladder_assembly.H0_VARIABLE: str(FIXTURE)}
    for name, value in options["environment"].items():
        monkeypatch.setenv(name, value)
    config["ladder_loop"]["round"].update({"problems": 4000, "batches": 4})    # the smoke's sizes are the environment's, whatever the config's
    arm.with_lean()
    summaries = _run(config, stage_name="ladder_l3d2_smoke")
    store = ladder_l3d2._store(config)
    assert store.root.name == "ladder_l3d2_smoke" and not (store.root.parent / "ladder_l2_t010_assembly_seed0").exists()
    assert [step for step in summaries if "round" in step] == ["ladder_l2_round_1", "ladder_l2_round_2"] and summaries["ladder_l2_prepare"]["rounds"] == [1, 2]
    assert (summaries["ladder_l2_prepare"]["problems_a_round"], summaries["ladder_l2_prepare"]["batch_sizes"], summaries["ladder_l2_prepare"]["h0_rows"]) == (4, [2, 2], 8)
    prepare = summaries["ladder_l3d2_prepare"]
    assert prepare["stored_runs"] is None and prepare["loop_arm"] is None and prepare["attempts_a_goal_problem"] == 32 and prepare["minimum_assembled"] == 8
    with_rows = store.read_rows(ladder_assembly.training_set_file(2))
    assert sum(row["origin"] == "h0" for row in with_rows) == 8 and summaries["ladder_l3d2_measure_with"]["is"] == "M(2)"       # the fixture's proofs are of no fixture problem
    report = summaries["ladder_l3d2_report"]
    assert report["branch"]["name"] in (NOT_READ, INCONCLUSIVE) and "loop" not in report["models"] and report["rounds"] == [1, 2]
    assert report["can_this_run_see_a_win"]["with_was_trained_on_enough_assembled_proofs"]["from_h0"] == 8
    assert any("L2's stored runs were not read (a smoke run)" in line for line in report["lines"]) and store.is_done(ladder_l3d2.REPORT)


# -------------------------------------------------------------------------------------------- registration
def test_the_stage_is_registered_with_a_guard_before_every_gpu_step_and_every_step_names_the_arm():
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l3d2"]]
    gpu_steps = [step for environment, step, _ in steps if environment == "gpu"]
    assert gpu_steps == ["fix_tokenizers", "ladder_l2_prepare", "ladder_l3d2_prepare", "ladder_l2_embed",
                         *(f"ladder_l2_{name}_{number}" for number in range(1, 7) for name in ("round", "train")),
                         "ladder_l3d2_train_without", "ladder_l3d2_measure_with", "ladder_l3d2_measure_without", "ladder_l3d2_report"]
    assert [(environment, step) for environment, step, _ in steps][:2] == [("sync", "gpu"), ("gpu", "fix_tokenizers")]
    assert all(steps[index - 1][0] == "guard" for index, (environment, _, _) in enumerate(steps) if environment == "gpu" and index > 2)
    assert all(options == {"environment": {ARM: ARM_NAME}} for _, _, options in steps)
    assert all(step in STEPS for step in gpu_steps[1:]) and not set(ladder_l3d2.STEPS) & (set(ladder_l2.STEPS) | set(ladder_l3d1.STEPS) | set(ladder_ceiling.STEPS))
    assert list(ladder_l3d2.STEPS) == ["ladder_l3d2_prepare", *(f"ladder_l2_{name}_{number}" for number in (4, 5, 6) for name in ("round", "train")),
                                       "ladder_l3d2_train_without", "ladder_l3d2_measure_with", "ladder_l3d2_measure_without", "ladder_l3d2_report"]
    smoke = [step for environment, step, _ in map(entry.step_fields, entry.STAGES["ladder_l3d2_smoke"]) if environment == "gpu"]
    assert smoke == [step for step in gpu_steps if not any(step.endswith(f"_{number}") for number in (3, 4, 5, 6))] and len(_file_rows(FIXTURE)) == 8
    assert ladder_assembly.PACKAGE_H0 == PACKAGE / "data" / "ladder_l3d" / "harvest_h0.jsonl"
    for other in ("ladder_l2", "ladder_l2_t010", "ladder_l2_smoke", "ladder_l3d1", "ladder_ceiling"):       # no other stage's task names the arm or this stage's variables
        own = entry.child_environment("gpu", "key", entry.step_fields(entry.STAGES[other][2])[2])
        assert own.get(ARM) != ARM_NAME and not [name for name in (*VARIABLES, ladder_l2.L2_ROUNDS_VARIABLE, ladder_assembly.H0_VARIABLE) if name in own]
    if not ladder_assembly.PACKAGE_H0.exists():
        pytest.skip("needs the harvest H0 (src/rlvr_lean/data/ladder_l3d: derived data), which is not shipped")


def test_the_steps_run_through_the_gpu_entry_point(step_2, monkeypatch, tmp_path):
    step_2.stored()
    monkeypatch.setattr(milestone2, "load_config", lambda path: step_2.config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    for environment, name, _ in map(entry.step_fields, entry.STAGES["ladder_l3d2"]):
        if environment == "gpu" and name in STEPS:
            monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", name, "--out", str(tmp_path / f"{name}.json")])
            assert gpu_main.main() == 0 and json.loads((tmp_path / f"{name}.json").read_text())["ok"] is True
    assert step_2.store().is_done(ladder_l3d2.REPORT)
