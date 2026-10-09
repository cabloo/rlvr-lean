"""L3d Step 1, the stage end to end on what an L1 run and two L2 runs stored: the refusals of its prepare step, the two
orders, the two trainings in the prepared order with the training loop's own record of the rows, the measurements with
L2's sampling seeds, the report, the adapters kept, a set Lean did not answer measured again, the smoke stage, the
registration. Spec: docs/spec/ladder-loop.spec.md, "L3d: train on what the episode reaches" (Step 1). The engine,
the model and Lean are stand-ins; nothing touches a GPU or the network. The rules and the report on hand-made rows are
`test_ladder_l3d1.py`."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l2_stage import ARM, _run_loop, loop  # noqa: E402, F401 - the L2 stage on the fixtures, with stand-ins
from test_ladder_round import ScriptedLean, _run_stage, stage  # noqa: E402, F401 - the L1 stage on the fixtures, with stand-ins

from rlvr_lean.domain.ladder_round.dose import run_dose  # noqa: E402
from rlvr_lean.domain.ladder_round.l3d import ARMS, BRANCHES, INCONCLUSIVE, NOT_READ, ORDER_LABEL  # noqa: E402
from rlvr_lean.domain.problem_pool.selection import rank  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import ladder_ceiling, ladder_dose, ladder_l2, ladder_l3a, ladder_l3c, ladder_l3d1, ladder_loop, ladder_round, pipeline  # noqa: E402
from rlvr_lean.gpu.ladder_ceiling import MORE, REACH  # noqa: E402
from rlvr_lean.gpu.ladder_l3d1 import REPORT_FILE, loss_file, measure_marker, train_marker, training_file  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
FIXTURE = PACKAGE / "data" / "ladder_l3d_fixture" / "harvest_h0.jsonl"
VARIABLES = (ladder_l3d1.RUN_VARIABLE, ladder_l3d1.SOURCE_VARIABLE, ladder_l3d1.STORED_VARIABLE, ladder_l3d1.H0_VARIABLE, ladder_l3d1.MINIMUM_VARIABLE)


def _file_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _files(directory):
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir()) if path.is_file()}


def _run(config, stage_name="ladder_l3d1"):
    return {step: ladder_l3d1.STEPS[step](config) for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name])
            if environment == "gpu" and step in ladder_l3d1.STEPS}


@pytest.fixture
def step_1(loop, monkeypatch):  # noqa: F811
    """The fixtures' world with nothing run yet: the 8-row fixture in H0's place, 8 rows asked for. `stored()` runs
    what the stage reads: L1, then L2 and L2's arm t010."""
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ladder_l3d1.H0_VARIABLE, str(FIXTURE))
    monkeypatch.setenv(ladder_l3d1.MINIMUM_VARIABLE, "8")

    def stored():
        _run_stage(loop.config)
        _run_loop(loop.config)
        monkeypatch.setenv(ARM, "t010")
        _run_loop(loop.config)
        monkeypatch.delenv(ARM)

    return SimpleNamespace(config=loop.config, stored=stored, store=lambda: ladder_l3d1._store(loop.config))


def _h0(tmp_path, monkeypatch, change):
    """A copy of the fixture with its rows changed by `change(rows)`, in H0's place."""
    rows = _file_rows(FIXTURE)
    change(rows)
    path = tmp_path / "changed" / "harvest_h0.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    monkeypatch.setenv(ladder_l3d1.H0_VARIABLE, str(path))
    return path


# ------------------------------------------------------------------------------------------ the whole stage
def test_the_whole_stage_runs_on_what_l1_and_l2_stored_and_writes_a_run_of_its_own(step_1):
    config = step_1.config
    with pytest.raises(RuntimeError, match="L3d Step 1 reads the run directory the L1 task for seed 0 wrote on this box .stage `ladder_l1` with --seeds 0"):
        ladder_l3d1.ladder_l3d1_prepare(config)                                           # nothing to read yet: refused, and it says what to run
    _run_stage(config)
    with pytest.raises(RuntimeError, match=r"L3d Step 1 reads the run directory the task of stage `ladder_l2` for seed 0 .*the base's attempts on G"):
        ladder_l3d1.ladder_l3d1_prepare(config)
    _run_loop(config)
    with pytest.raises(RuntimeError, match=r"stage `ladder_l2_t010` for seed 0 .*--stage ladder_l2_t010 --seeds 0.*its three-round model's attempts on G"):
        ladder_l3d1.ladder_l3d1_prepare(config)
    assert not step_1.store().is_done(ladder_l3d1.PREPARE) and not step_1.store().path("problems.jsonl").exists()
    step_1.stored()
    source, stored = ladder_l3d1.source_directory(config), ladder_l3d1.stored_directories(config)
    assert source == ladder_round._store(config).root and [path.name for path in stored.values()] == ["ladder_l2_seed0", "ladder_l2_t010_seed0"]
    before = {path: _files(path) for path in (source, *stored.values())}
    sent = ScriptedLean.submitted
    summaries = _run(config)
    assert {path: _files(path) for path in before} == before                              # the three run directories were only read
    store = step_1.store()
    assert store.root.name == "ladder_l3d1_seed0" and store.root.parent == source.parent

    # ---- prepare: `without` is the three rounds' stored training examples in a seeded order; `with` the same rows, none moved, and H0 among them
    prepare, harvest = summaries["ladder_l3d1_prepare"], _file_rows(FIXTURE)
    rounds = [row for number in (1, 2, 3) for row in _file_rows(stored["loop"] / f"training_examples_r{number}.jsonl")]
    assert [Path(name).name for name in prepare["rounds_files"]] == [f"training_examples_r{number}.jsonl" for number in (1, 2, 3)]      # round 1's, then 2's, then 3's
    assert prepare["rounds_files"] == {str(stored["loop"] / f"training_examples_r{number}.jsonl"): len(_file_rows(stored["loop"] / f"training_examples_r{number}.jsonl"))
                                       for number in (1, 2, 3)} and prepare["rounds_rows"] == len(rounds) > 8
    assert (prepare["stage"], prepare["h0_rows"], prepare["minimum_h0"], prepare["loop_arm"], prepare["loop_target_rate"]) == ("l3d1", 8, 8, "t010", 0.1)
    without, with_h0 = store.read_rows(training_file("without")), store.read_rows(training_file("with"))
    assert [row["id"] for row in without] == sorted((row["attempt_id"] for row in rounds), key=lambda key: rank(0, ORDER_LABEL, key))      # by the task's seed
    assert [row["id"] for row in with_h0 if row["source"] == "rounds"] == [row["id"] for row in without]                                  # the same relative order
    assert sorted(row["id"] for row in with_h0 if row["source"] == "h0") == sorted(f"{row['problem_id']}#h0" for row in harvest)
    assert [row["row"] for row in with_h0] == list(range(len(rounds) + 8)) and {row["source"] for row in without} == {"rounds"}
    by_id = {row["attempt_id"]: row for row in rounds}
    assert all((row["theorem"], row["completion"], row["problem_id"], row["side"]) == tuple(by_id[row["id"]][key] for key in ("theorem", "completion", "problem_id", "side"))
               for row in without)
    assert {row["completion"] for row in with_h0 if row["source"] == "h0"} == {row["completion"] for row in harvest} and all(row["tokens"] > 0 for row in with_h0)
    batch = config["training"]["effective_batch"]
    assert prepare["trainings"]["without"] == {"rows": len(rounds), "h0_rows": 0, "steps": -(-len(rounds) // batch), "tokens": sum(row["tokens"] for row in without)}
    assert (prepare["trainings"]["with"]["rows"], prepare["trainings"]["with"]["h0_rows"]) == (len(rounds) + 8, 8) and prepare["order"]["seed"] == 0
    of_l2 = ladder_l2.sampling_seeds(config)
    assert prepare["sampling_seeds"] == {"rungs": of_l2["rungs"], "reach": of_l2["reach"], "more": of_l2["control"]}
    control = json.loads((stored["base"] / "episodes_control.done.json").read_text())
    assert prepare["goal_samplings"] == [{"name": "reach", "episodes": 32, "sampling_seed": of_l2["reach"]},
                                         {"name": "more", "episodes": control["episodes_each"], "sampling_seed": of_l2["control"]}]
    assert prepare["recipe"]["adapter_seed"] == 0 and prepare["recipe"]["lora"] == config["lora"] and prepare["h0_file"] == str(FIXTURE)

    # ---- train: one pass each in the prepared order (stand-in losses), and the loop's own record of the rows
    for arm, rows in (("without", without), ("with", with_h0)):
        train = summaries[train_marker(arm)]
        assert train["steps"] == prepare["trainings"][arm]["steps"] and train["rows"] == len(rows) and train["passes"] == 1 and "made up" in train["note"]
        assert train["h0_rows_trained"] == train["h0_rows"] == (8 if arm == "with" else 0) and train["stand_in_engine"] is True and train["model"] == arm
        loss = json.loads(store.path(loss_file(arm)).read_text())
        assert loss["rows_trained"] == [row["id"] for row in rows] and len(loss["row_losses"]) == len(rows) and loss["stage"] == "l3d1"
        assert all("example_losses" not in row and "example_positions" not in row for row in loss["training_steps"])
        assert [row["rows_seen"] for row in loss["training_steps"]][-1] == len(rows) and train["rows_compared"] == max(1, len(rows) // 10)

    # ---- measure: L2's sampling seeds, so each model pairs by problem with the stored attempts
    base = {part: {row["problem_id"]: row for row in store.read_rows(ladder_ceiling.stored_file("base", part, "l3d1"))} for part in ("rungs", REACH, MORE)}
    assert base["rungs"] == {row["problem_id"]: row for row in _file_rows(source / "episodes_rungs_base_problems.jsonl")}
    assert store.read_rows(ladder_ceiling.stored_file("loop", MORE, "l3d1")) == _file_rows(stored["loop"] / "episodes_control_m3_problems.jsonl")
    for arm in ARMS:
        measured = summaries[measure_marker(arm)]
        assert (measured["stage"], measured["model"], measured["rows"], measured["h0_rows"]) == ("l3d1", arm, prepare["trainings"][arm]["rows"], 8 if arm == "with" else 0)
        assert measured["rungs"]["sampling_seed"] == of_l2["rungs"] and measured["rungs"]["episodes_each"] == 8
        assert (measured["goal"][REACH]["sampling_seed"], measured["goal"][REACH]["episodes_each"]) == (of_l2["reach"], 32)
        assert (measured["goal"][MORE]["sampling_seed"], measured["goal"][MORE]["episodes_each"]) == (of_l2["control"], control["episodes_each"])
        for part, set_name in (("rungs", f"l3d1_rungs_{arm}"), (REACH, f"l3d1_reach_{arm}"), (MORE, f"l3d1_more_{arm}")):
            own = {row["problem_id"]: row for row in store.read_rows(f"episodes_{set_name}_problems.jsonl")}
            assert own == base[part], (arm, part)       # the stand-in engine has no adapter: the same seed gives the same samples as the stored measurement
        # What the model wrote: the ceiling's two reads, the share that opens with a `have`, and the goal problem's episodes of 8 (32 attempts: 4; the control's: its own).
        assert set(measured["verified_proof_lines"]) == {"rungs", "goal"} == set(measured["opens_with_a_have"]) and measured["opens_with_a_have"]["rungs"]["attempts"] > 0
        (episodes,) = measured["episodes_of_8"].values()
        assert len(episodes) == 32 // 8 + control["episodes_each"] // 8
    assert ScriptedLean.submitted > sent
    stored_models = json.loads(store.path(ladder_l3d1.STORED_MODELS_FILE).read_text())
    assert set(stored_models) == {"stage", "base", "loop"} and set(stored_models["loop"]) == {"verified_proof_lines", "distinct_attempts", "opens_with_a_have", "episodes_of_8"}

    # ---- every stored file and every set says l3d1 (the one name the shared episode step reads apart)
    names = [path.name for path in store.root.iterdir()]
    assert [name for name in names if "l3d1" not in name] == ["problems.jsonl"]
    assert {row["set"] for row in store.read_rows("problems.jsonl")} == {f"l3d1_{part}_{arm}" for part in ("rungs", REACH, MORE) for arm in ARMS}

    # ---- the report
    report = summaries["ladder_l3d1_report"]
    assert report["stage"] == "l3d1" and report["branch"]["name"] in BRANCHES and report["stand_in_engine"] is True and report["ok"] is True
    assert all(line.startswith("l3d1: ") for line in report["lines"]) and "L3d Step 1" in report["headline"]
    checks = report["can_this_run_see_a_win"]
    assert checks["with_was_trained_on_h0"]["passes"] is True and checks["with_was_trained_on_h0"]["counted_once_in_with"] == 8
    assert checks["h0_holds_enough_proofs"] == {"what": "H0 holds at least 8 proofs", "h0_rows": 8, "minimum": 8, "passes": True}
    assert report["inconclusive"] == (report["branch"]["name"] == INCONCLUSIVE) == (not all(check["passes"] for check in checks.values() if isinstance(check, dict)))
    assert report["models"] == {"base": report["models"]["base"], "without": "`without`", "with": "`with`", "loop": "the stored three-round model at t = 1/10"}
    measured_read = report["measured_and_not_read"] if report["inconclusive"] else report
    by_length = measured_read["secondary"]["by_length_group"]
    assert by_length["problems"]["all"] == 1 == by_length["problems"]["not_known"] and by_length["problems"]["4_or_more"] == 0       # the fixture's goal problem has no shipped length
    assert set(by_length["pairs"]) == {"with_minus_without", "noise_floor", "with_minus_base", "without_minus_base", "loop_minus_base"}
    assert by_length["pairs"]["with_minus_without"]["all"]["mean"] == 0.0 and by_length["pairs"]["with_minus_without"]["all"]["attempts_each"] == prepare["attempts_a_goal_problem"]
    assert measured_read["primary"]["mean"] is None and measured_read["noise_floor"]["mean"] is None
    alone = measured_read["secondary"]["goal_problems_solved_by_attempts_alone"]
    assert set(alone) == {"what", "base", "without", "with", "loop"} and alone["with"]["all"]["problems"] == 1 and alone["with"]["all"]["episodes"] == len(episodes)
    assert report["adapters"]["kept"] is True and report["adapters"]["there"] == [] and report["the_ceilings_doses"] is None       # the stand-in trains none; no ceiling run here
    assert json.loads(store.path(REPORT_FILE).read_text()) == json.loads(json.dumps(report, default=str))
    assert store.done_summary(ladder_l3d1.REPORT)["stage"] == "l3d1"
    # A rerun returns what is stored: nothing is trained or sampled again.
    sent = ScriptedLean.submitted
    again = _run(config)
    assert ScriptedLean.submitted == sent and all(again[step] == summaries[step] for step in summaries if step != "ladder_l3d1_report")
    assert not ladder_round._store(config).is_done(ladder_l3d1.PREPARE) and not ladder_l2._store(config).is_done(ladder_l3d1.PREPARE)
    assert not (store.root.parent / "ladder_ceiling_seed0").exists()                    # and nothing of the ceiling's was made


# ---------------------------------------------------------------------------------------------- the refusals
def test_the_prepare_step_refuses_an_h0_it_must_not_train_on_and_writes_nothing(step_1, monkeypatch, tmp_path):
    config = step_1.config
    step_1.stored()
    held = _file_rows(ladder_l3d1.source_directory(config) / "heldout_groups.jsonl")[0]["problem_id"]
    trained = _file_rows(ladder_l3d1.stored_directories(config)["loop"] / "training_examples_r2.jsonl")[0]["problem_id"]

    def refused(match, error=ValueError):
        with pytest.raises(error, match=match):
            ladder_l3d1.ladder_l3d1_prepare(config)
        assert not step_1.store().is_done(ladder_l3d1.PREPARE) and not step_1.store().path("problems.jsonl").exists()

    _h0(tmp_path, monkeypatch, lambda rows: rows[5].update(problem_id=held))
    refused(f"H0 holds 1 held-out problems .first: {held}")
    _h0(tmp_path, monkeypatch, lambda rows: rows[2].update(problem_id=trained))                # a problem in both: `with` would hold two proofs of it
    refused(f"H0 holds 1 problems the rounds' training examples hold already .first: {trained}")
    _h0(tmp_path, monkeypatch, lambda rows: rows[1].update(problem_id=rows[0]["problem_id"]))
    refused("1 problems stand twice in H0")
    _h0(tmp_path, monkeypatch, lambda rows: rows[4].update(assembled=False))
    refused("1 rows of H0 are not assembled proofs with their statement .first: row 4")
    _h0(tmp_path, monkeypatch, lambda rows: rows[6].pop("theorem"))
    refused("1 rows of H0 are not assembled proofs with their statement .first: row 6")
    _h0(tmp_path, monkeypatch, lambda rows: rows[3].update(completion="  nlinarith [sq_nonneg (a - b)]\n" * 400))      # 12,800 characters of proof
    refused(r"1 training rows are longer than the 2048 tokens a training example is cut at .first: lean_workbook_\w+#h0")
    _h0(tmp_path, monkeypatch, lambda rows: rows.pop())
    refused("H0 holds 7 proofs and Step 1 asks for at least 8")
    monkeypatch.setenv(ladder_l3d1.H0_VARIABLE, str(FIXTURE))
    monkeypatch.delenv(ladder_l3d1.MINIMUM_VARIABLE)                                           # the config's own least number: 100
    refused("H0 holds 8 proofs and Step 1 asks for at least 100: with fewer it is not run on it")
    monkeypatch.setenv(ladder_l3d1.MINIMUM_VARIABLE, "8")
    monkeypatch.setenv(ladder_l3d1.H0_VARIABLE, str(tmp_path / "nowhere" / "harvest_h0.jsonl"))
    refused(r"the harvest H0 .*is not in this snapshot: it is built on the dev machine by `tools/ladder_harvest.py`", RuntimeError)
    monkeypatch.setenv(ladder_l3d1.H0_VARIABLE, str(FIXTURE))
    monkeypatch.setenv(ladder_l3d1.STORED_VARIABLE, "ladder_l2_smoke")
    refused(f"{ladder_l3d1.STORED_VARIABLE} is 'ladder_l2_smoke': it is `none`")
    monkeypatch.delenv(ladder_l3d1.STORED_VARIABLE)
    if not (PACKAGE / "data" / "ladder_l0" / "heldout.jsonl").exists():
        pytest.skip("the rest needs the pool's data directory (src/rlvr_lean/data/ladder_l0: derived data), which is not shipped")
    # The snapshot's own held-out set and base map (the real ones here) bar their problems too.
    with monkeypatch.context() as patched:
        patched.delenv(ladder_loop.DATA_VARIABLE)
        real = PACKAGE / "data" / "ladder_l0"
        for name, what in (("heldout.jsonl", "held-out problems"), ("base_map.jsonl", "problems of the base map")):
            barred = json.loads((real / name).read_text().splitlines()[0])["problem_id"]
            _h0(tmp_path, monkeypatch, lambda rows, barred=barred: rows[7].update(problem_id=barred))
            refused(f"H0 holds 1 {what} .first: {barred}")
        # ... and so do the held-out groups of the run that is read, which the snapshot's set here does not hold.
        _h0(tmp_path, monkeypatch, lambda rows: rows[5].update(problem_id=held))
        refused(f"H0 holds 1 held-out problems .first: {held}")
    monkeypatch.setenv(ladder_l3d1.H0_VARIABLE, str(FIXTURE))
    # The three rounds' training examples are read on the box: one that is not there is refused with the task to run.
    gone = ladder_l3d1.stored_directories(config)["loop"] / "training_examples_r2.jsonl"
    kept = gone.read_bytes()
    gone.unlink()
    refused(r"does not hold \['training_examples_r2.jsonl'\]. L3d Step 1 trains `without` on the training examples the task of stage `ladder_l2_t010` for seed 0", RuntimeError)
    gone.write_bytes(kept)
    # What would not pair with the stored measurements is refused in this stage's own words; so is an arm that is not configured.
    config["ladder_loop"]["round"]["sampling_seed"] += 7
    refused("L3d Step 1's two models would not pair with the stored results", RuntimeError)
    config["ladder_loop"]["round"]["sampling_seed"] -= 7
    config["ladder_loop"]["l3d"]["step_1"]["loop_arm"] = "t005"
    refused("ladder_loop.l3d.step_1.loop_arm is 't005' and ladder_loop.l2_arms has")
    config["ladder_loop"]["l3d"]["step_1"]["loop_arm"] = "t010"
    stored = ladder_l3d1.stored_directories(config)["loop"] / "episodes_reach_m3_problems.jsonl"
    rows = _file_rows(stored)
    rows[0]["attempts_without_an_answer"] = 8
    stored.write_text("".join(json.dumps(row) + "\n" for row in rows))
    refused(r"with too many attempts without a verdict from Lean: that set is not to be read, and no report of this run could be read against it. Nothing was written",
            RuntimeError)


# ------------------------------------------------------------------------- the trainings, and the adapters
def _as_the_gpu_would(monkeypatch, calls):
    """The train steps' GPU path with the model taken out: `_train_with_readings` is replaced by the schedule alone
    (stand-in losses), which records what it was given and writes each checkpoint's adapter directory."""
    def train_with_readings(config, examples, sample, pairs, schedule, seed, directory, tensorboard_run, orders=None, per_example=False, positions=False):
        calls.append({"examples": examples, "schedule": schedule, "seed": seed, "directory": directory, "tensorboard_run": tensorboard_run, "orders": orders,
                      "per_example": per_example, "positions": positions, "sample": sample, "pairs": pairs, "saved": []})
        train_step, read, _ = ladder_dose._stand_in_calls(len(examples), [], [])

        def save(name, step_number):
            calls[-1]["saved"].append((name, step_number))
            (directory / name).mkdir(parents=True)
            (directory / name / "adapter_model.safetensors").write_text("an adapter")

        return {**run_dose(len(examples), config["training"]["effective_batch"], schedule["passes"], seed, schedule["reading_steps"], schedule["checkpoint_steps"],
                           train_step, read, save, orders=orders, per_example=per_example, positions=positions), "target_format": "native", "seconds": 1.0}

    capped = []
    monkeypatch.setattr(ladder_dose, "_train_with_readings", train_with_readings)
    monkeypatch.setattr(pipeline, "_cap_torch_memory", lambda config: capped.append(config["gpu"]["desktop_reserve_gb"]))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None, memory_allocated=lambda: 0)))
    return capped


def test_both_models_are_trained_by_the_same_code_on_their_prepared_rows_and_their_adapters_are_kept(step_1, monkeypatch):
    config, calls = step_1.config, []
    step_1.stored()
    with pytest.raises(RuntimeError, match="the training of `with` needs the step ladder_l3d1_prepare of this run, which is not done: the stage `ladder_l3d1`"):
        ladder_l3d1.ladder_l3d1_train(config, "with")
    prepare = ladder_l3d1.ladder_l3d1_prepare(config)
    capped = _as_the_gpu_would(monkeypatch, calls)
    monkeypatch.delenv(ladder_loop.STAND_IN_VARIABLE)                          # the train steps take their GPU path
    trains = {arm: ladder_l3d1.ladder_l3d1_train(config, arm) for arm in ARMS}
    store = step_1.store()
    for call, arm in zip(calls, ARMS):
        rows = store.read_rows(training_file(arm))
        # The rows as the round's training examples, in the prepared order: one pass, nothing shuffled, an adapter at the end.
        assert call["examples"] == [{key: row[key] for key in ("problem_id", "side", "theorem", "completion")} for row in rows]
        assert call["orders"] == [list(range(len(rows)))] and (call["per_example"], call["positions"]) == (True, True)
        assert call["schedule"] == {"passes": 1, "reading_steps": [], "checkpoint_steps": {arm: prepare["trainings"][arm]["steps"]}}
        assert (call["sample"], call["pairs"], call["seed"]) == ([], [], 0) and call["tensorboard_run"] == f"ladder_l3d1_{arm}_seed0"      # the same adapter seed for both
        assert call["saved"] == [(arm, prepare["trainings"][arm]["steps"])] and call["directory"] == store.root / "adapters"
        assert ladder_l3d1.adapter_directory(store, arm).is_dir() and "note" not in trains[arm] and trains[arm]["allocated_after_cleanup_gb"] == 0
        assert json.loads(store.path(loss_file(arm)).read_text())["rows_trained"] == [row["id"] for row in rows]
    assert len(calls) == 2 and capped == [3.0, 3.0]
    # A measurement of a model whose adapter is not there is refused (and no engine is loaded for it).
    adapter = ladder_l3d1.adapter_directory(store, "with")
    adapter.rename(adapter.with_name("elsewhere"))
    with pytest.raises(RuntimeError, match="this run keeps its adapters, and one trained again would not be the model"):
        ladder_l3d1.ladder_l3d1_measure(config, "with")
    adapter.with_name("elsewhere").rename(adapter)
    with pytest.raises(RuntimeError, match="L3d Step 1's report needs the step ladder_l3d1_measure_without"):
        ladder_l3d1.ladder_l3d1_report(config)
    monkeypatch.setenv(ladder_loop.STAND_IN_VARIABLE, "1")                     # the measurements with the stand-in engine
    for arm in ARMS:
        ladder_l3d1.ladder_l3d1_measure(config, arm)
    report = ladder_l3d1.ladder_l3d1_report(config)
    # The adapters are the loop's own models: the report keeps them, and says so. A rerun trains nothing again.
    assert report["adapters"] == {"kept": True, "what": report["adapters"]["what"], "directory": str(store.root / "adapters"), "there": ["without", "with"]}
    assert all(ladder_l3d1.adapter_directory(store, arm).is_dir() for arm in ARMS) and "their adapters are kept" in report["adapters"]["what"]
    monkeypatch.delenv(ladder_loop.STAND_IN_VARIABLE)
    assert _run(config)["ladder_l3d1_report"]["branch"] == report["branch"] and len(calls) == 2


def test_a_training_that_was_not_made_on_the_prepared_rows_makes_the_run_inconclusive(step_1, monkeypatch):
    config = step_1.config
    step_1.stored()
    ladder_l3d1.ladder_l3d1_prepare(config)
    real = ladder_ceiling.one_pass

    def a_pass_that_misses_rows(config, examples, steps, seed, batch, adapters, tensorboard_run, positions=False):
        result, step_rows, row_losses, trained = real(config, examples, steps, seed, batch, adapters, tensorboard_run, positions=positions)
        return result, step_rows, row_losses, [position for position in trained if position % 5]      # a loop that skipped every fifth row

    monkeypatch.setattr(ladder_l3d1, "one_pass", a_pass_that_misses_rows)
    summaries = _run(config)
    check = summaries["ladder_l3d1_report"]["can_this_run_see_a_win"]["with_was_trained_on_h0"]
    assert check["passes"] is False and check["as_prepared"] == {"without": False, "with": False} and check["counted_once_in_with"] < 8
    report = summaries["ladder_l3d1_report"]
    assert report["branch"]["name"] == INCONCLUSIVE and "with_was_trained_on_h0" in report["branch"]["failed_checks"] and report["primary"] is None
    assert any(line.startswith("l3d1: CHECK 4, `with` was trained on H0") and line.endswith("FAIL") for line in report["lines"])
    assert not any("PRIMARY" in line or "SECONDARY" in line for line in report["lines"]) and any(line.startswith("l3d1: INCONCLUSIVE.") for line in report["lines"])


def test_a_report_that_is_not_to_be_read_fails_its_step_and_the_set_is_measured_again_from_the_kept_adapter(step_1, monkeypatch, tmp_path):
    from rlvr_lean.gpu import __main__ as gpu_main
    from rlvr_lean.gpu import milestone2

    config, calls = step_1.config, []
    step_1.stored()
    ladder_l3d1.ladder_l3d1_prepare(config)
    _as_the_gpu_would(monkeypatch, calls)
    monkeypatch.delenv(ladder_loop.STAND_IN_VARIABLE)
    for arm in ARMS:
        ladder_l3d1.ladder_l3d1_train(config, arm)
    monkeypatch.setenv(ladder_loop.STAND_IN_VARIABLE, "1")
    store = step_1.store()
    first = {arm: ladder_l3d1.ladder_l3d1_measure(config, arm) for arm in ARMS}
    unanswered = "l3d1_more_with"                                              # the pool was in trouble while one set was measured
    name = f"episodes_{unanswered}_problems.jsonl"
    as_measured, rows = store.path(name).read_bytes(), store.read_rows(name)
    rows[0]["attempts_without_an_answer"] = 20
    store.write_rows(name, rows)
    monkeypatch.setattr(milestone2, "load_config", lambda path: config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", ladder_l3d1.REPORT, "--out", str(tmp_path / "report.json")])
    assert gpu_main.main() != 0                                                # exit non-zero, as every report that is not to be read
    report = json.loads(store.path(REPORT_FILE).read_text())
    assert report["ok"] is False and report["not_to_be_read"] == ["goal_with_2"] and "NOT TO BE READ" in report["lines"][-1]
    assert all(ladder_l3d1.adapter_directory(store, arm).is_dir() for arm in ARMS)
    # The task queued again: that set, and no other, is measured again; nothing is trained; the report can be read.
    sent = ScriptedLean.submitted
    again = _run(config)
    assert len(calls) == 2 and again[measure_marker("without")] == first["without"] and "measured_again" not in first["with"]
    assert again[measure_marker("with")]["measured_again"] == [unanswered]
    assert ScriptedLean.submitted - sent == first["with"]["goal"][MORE]["pipeline"]["sent_to_lean"]           # that one set's checks, and no other's
    assert store.path(name).read_bytes() == as_measured and again["ladder_l3d1_report"]["ok"] is True and again["ladder_l3d1_report"]["not_to_be_read"] == []
    assert all(ladder_l3d1.adapter_directory(store, arm).is_dir() for arm in ARMS)


def test_the_ceilings_two_doses_are_read_from_its_report_on_the_box_when_it_can_be_read(step_1):
    config = step_1.config
    assert ladder_l3d1.ceiling_doses(config) is None                           # no ceiling run on this box
    path = step_1.store().root.parent / "ladder_ceiling_seed0" / ladder_ceiling.REPORT_FILE      # where the ceiling's run of this seed writes its report
    path.parent.mkdir(parents=True)
    entry_of = lambda mean: {"model": "a model", "4_or_more": {"mean": mean, "low": mean - 0.001, "high": mean + 0.001, "problems": 230}}      # noqa: E731
    written = {"ok": True, "inconclusive": False, "training": {"checkpoints": {"small": {"rows": 2000, "step": 250, "rows_seen": 2000},
                                                                               "full": {"rows": 8000, "step": 1000, "rows_seen": 8000}}},
               "secondary": {"by_length_group": {"models": {"full": entry_of(0.004), "small": entry_of(0.002), "loop": entry_of(0.0006)}}}}
    path.write_text(json.dumps(written))
    assert ladder_l3d1.ceiling_doses(config) == {"small": {"rows": 2000, "mean": 0.002, "low": 0.001, "high": 0.003},
                                                 "full": {"rows": 8000, "mean": 0.004, "low": 0.003, "high": 0.005}}
    for unreadable in ({**written, "ok": False}, {**written, "inconclusive": True}, {**written, "secondary": None}):       # not to be read, or inconclusive: no dose is taken from it
        path.write_text(json.dumps(unreadable))
        assert ladder_l3d1.ceiling_doses(config) is None
    assert sorted(item.name for item in path.parent.iterdir()) == [ladder_ceiling.REPORT_FILE]               # read, never written


# ------------------------------------------------------------------------------------------- the smoke stage
def test_the_smoke_stage_runs_on_the_l1_smoke_run_alone_with_its_own_training_examples_and_the_fixture(stage, monkeypatch):  # noqa: F811
    if not (PACKAGE / "data" / "ladder_l0" / "summary.json").exists():
        pytest.skip("needs the pool's data directory (src/rlvr_lean/data/ladder_l0: derived data), which is not shipped")
    config = stage.config
    monkeypatch.setenv(ladder_round.ROUND_RUN_VARIABLE, "ladder_l1_smoke")
    _run_stage(config)                                                         # what the L1 smoke task leaves on the box
    monkeypatch.delenv(ladder_round.ROUND_RUN_VARIABLE)
    monkeypatch.delenv(ladder_loop.DATA_VARIABLE)                              # as on the box: the smoke stage names no L0 data, so the snapshot's own is read
    options = entry.step_fields(entry.STAGES["ladder_l3d1_smoke"][2])[2]
    variables = entry.child_environment("gpu", "key", options)
    assert {name: variables[name] for name in VARIABLES} == options["environment"] == {
        ladder_l3d1.SOURCE_VARIABLE: "ladder_l1_smoke", ladder_l3d1.RUN_VARIABLE: "ladder_l3d1_smoke", ladder_l3d1.STORED_VARIABLE: "none",
        ladder_l3d1.H0_VARIABLE: str(FIXTURE), ladder_l3d1.MINIMUM_VARIABLE: "8"}
    for name in VARIABLES:
        monkeypatch.setenv(name, variables[name])
    assert ladder_l3d1.stored_directories(config) is None and ladder_l3d1.source_directory(config).name == "ladder_l1_smoke"
    summaries = _run(config, stage_name="ladder_l3d1_smoke")
    store = ladder_l3d1._store(config)
    assert store.root.name == "ladder_l3d1_smoke" and not (store.root.parent / "ladder_l3d1_seed0").exists()
    prepare = summaries["ladder_l3d1_prepare"]
    own = _file_rows(ladder_l3d1.source_directory(config) / "training_examples_challenger.jsonl")              # the L1 smoke run's own stored training examples
    assert prepare["stored_runs"] is None and prepare["loop_arm"] is None and prepare["attempts_a_goal_problem"] == 32 and prepare["rounds_rows"] == len(own) > 0
    assert list(prepare["rounds_files"]) == [str(ladder_l3d1.source_directory(config) / "training_examples_challenger.jsonl")]
    assert sorted(row["id"] for row in store.read_rows(training_file("without"))) == sorted(row["attempt_id"] for row in own)
    assert (prepare["trainings"]["with"]["rows"], prepare["h0_rows"], prepare["minimum_h0"]) == (len(own) + 8, 8, 8)
    assert {row["set"] for row in store.read_rows("problems.jsonl")} == {f"l3d1_{part}_{arm}" for part in ("rungs", REACH) for arm in ARMS}
    report = summaries["ladder_l3d1_report"]
    assert report["branch"]["name"] in (NOT_READ, INCONCLUSIVE) and "loop" not in report["models"] and report["sizes"]["stored_runs"] is None
    assert any("L2's runs were not read (a smoke run)" in line for line in report["lines"]) and store.is_done(ladder_l3d1.REPORT)
    # Without the L1 smoke run's stored training examples the smoke stage refuses, naming the task to run.
    store.path(f"{ladder_l3d1.PREPARE}.done.json").unlink()
    (ladder_l3d1.source_directory(config) / "training_examples_challenger.jsonl").unlink()
    with pytest.raises(RuntimeError, match=r"does not hold \['training_examples_challenger.jsonl'\]. L3d Step 1 trains `without` on the training examples the task of stage `ladder_l1_smoke`"):
        ladder_l3d1.ladder_l3d1_prepare(config)


# -------------------------------------------------------------------------------------------- registration
def test_the_stage_is_registered_with_a_guard_before_every_gpu_step():
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l3d1"]]
    assert [(environment, step) for environment, step, _ in steps] == [
        ("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_l3d1_prepare"), ("guard", None), ("gpu", "ladder_l3d1_train_without"),
        ("guard", None), ("gpu", "ladder_l3d1_train_with"), ("guard", None), ("gpu", "ladder_l3d1_measure_without"), ("guard", None),
        ("gpu", "ladder_l3d1_measure_with"), ("guard", None), ("gpu", "ladder_l3d1_report")]
    gpu_steps = [step for environment, step, _ in steps if environment == "gpu"]
    assert gpu_steps[1:] == list(ladder_l3d1.STEPS) and all(options == {} for _, _, options in steps)
    assert [train_marker(arm) for arm in ARMS] == gpu_steps[2:4] and [measure_marker(arm) for arm in ARMS] == gpu_steps[4:6]
    others = (set(ladder_round.STEPS) | set(ladder_dose.STEPS) | set(ladder_l2.STEPS) | set(ladder_loop.STEPS) | set(ladder_l3a.STEPS) | set(ladder_l3c.STEPS)
              | set(ladder_ceiling.STEPS))
    assert not set(ladder_l3d1.STEPS) & others and all(name.startswith("ladder_l3d1_") for name in ladder_l3d1.STEPS)
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l3d1_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in steps]
    assert all(options == smoke[0][2] for _, _, options in smoke) and FIXTURE.exists() and len(_file_rows(FIXTURE)) == 8
    own = entry.child_environment("gpu", "key", steps[2][2])
    assert not [name for name in VARIABLES if name in own]
    assert ladder_l3d1.PACKAGE_H0 == PACKAGE / "data" / "ladder_l3d" / "harvest_h0.jsonl" and ladder_l3d1.h0_file() == ladder_l3d1.PACKAGE_H0
    for other in ("ladder_l1b_smoke", "ladder_l2_smoke", "ladder_l3c_smoke", "ladder_ceiling_smoke"):
        assert not [name for name in VARIABLES if name in entry.child_environment("gpu", "key", entry.step_fields(entry.STAGES[other][2])[2])]
    # The fixture is a smoke run's H0 in the harvest tool's row shape: assembled proofs of pool problems, the model's own.
    fixture = _file_rows(FIXTURE)
    assert all(row["assembled"] is True and row["side"] == "statement" and row["theorem"].startswith(f"theorem {row['problem_id']} ") and "fixture" in row for row in fixture)
    assert len({row["problem_id"] for row in fixture}) == 8


def test_the_steps_run_through_the_gpu_entry_point(step_1, monkeypatch, tmp_path):
    from rlvr_lean.gpu import __main__ as gpu_main
    from rlvr_lean.gpu import milestone2

    step_1.stored()
    monkeypatch.setattr(milestone2, "load_config", lambda path: step_1.config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    for name in ladder_l3d1.STEPS:
        monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", name, "--out", str(tmp_path / f"{name}.json")])
        assert gpu_main.main() == 0 and json.loads((tmp_path / f"{name}.json").read_text())["ok"] is True
    assert step_1.store().is_done(ladder_l3d1.REPORT)
