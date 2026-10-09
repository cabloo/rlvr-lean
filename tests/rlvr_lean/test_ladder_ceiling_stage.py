"""The ceiling, a labelled diagnostic: the stage end to end on what an L1 run and two L2 runs stored. The refusals of
its prepare step, the training in the file's order with its checkpoints by rows, the measurements with L2's sampling
seeds, the report, the adapters deleted only once the report is written, a finished run that is not trained again, the
smoke stage, the registration. Spec: docs/spec/ladder-loop.spec.md, "The ceiling: a labelled diagnostic". The
engine, the model and Lean are stand-ins; nothing touches a GPU or the network. The rules and the report on hand-made
rows are `test_ladder_ceiling.py`."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l2_stage import ARM, _run_loop, loop  # noqa: E402, F401 - the L2 stage on the fixtures, with stand-ins
from test_ladder_round import ScriptedLean, _run_stage, stage  # noqa: E402, F401 - the L1 stage on the fixtures, with stand-ins

from rlvr_lean.domain.ladder_round.ceiling import BRANCHES, CHECKPOINTS, INCONCLUSIVE, NOT_READ  # noqa: E402
from rlvr_lean.domain.ladder_round.dose import run_dose  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import ladder_ceiling, ladder_dose, ladder_l2, ladder_l3a, ladder_l3c, ladder_loop, ladder_round, pipeline  # noqa: E402
from rlvr_lean.gpu.ladder_ceiling import MORE, REACH, REPORT_FILE  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
FIXTURE = PACKAGE / "data" / "ladder_ceiling_fixture" / "training.jsonl"
VARIABLES = (ladder_ceiling.CEILING_RUN_VARIABLE, ladder_ceiling.CEILING_SOURCE_VARIABLE, ladder_ceiling.CEILING_STORED_VARIABLE,
             ladder_ceiling.CEILING_TRAINING_VARIABLE, ladder_ceiling.CEILING_CHECKPOINTS_VARIABLE)


def _file_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _files(directory):
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir()) if path.is_file()}


def _run_ceiling(config, until=None, stage_name="ladder_ceiling"):
    summaries = {}
    for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
        if environment == "gpu" and step in ladder_ceiling.STEPS:
            summaries[step] = ladder_ceiling.STEPS[step](config)
            if step == until:
                break
    return summaries


@pytest.fixture
def ceiling(loop, monkeypatch):  # noqa: F811
    """The fixtures' world with nothing run yet: the 24-row fixture in the training file's place and its two
    checkpoints after 12 and 24 rows. `stored()` runs what the ceiling reads: L1, then L2 and L2's arm t010."""
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ladder_ceiling.CEILING_TRAINING_VARIABLE, str(FIXTURE))
    monkeypatch.setenv(ladder_ceiling.CEILING_CHECKPOINTS_VARIABLE, "12,24")

    def stored():
        _run_stage(loop.config)
        _run_loop(loop.config)
        monkeypatch.setenv(ARM, "t010")
        _run_loop(loop.config)
        monkeypatch.delenv(ARM)

    return SimpleNamespace(config=loop.config, stored=stored, store=lambda: ladder_ceiling._store(loop.config))


def _training_file(tmp_path, monkeypatch, change):
    """A copy of the fixture with its rows changed by `change(rows)`, in the training file's place."""
    rows = _file_rows(FIXTURE)
    change(rows)
    path = tmp_path / "changed" / "training.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    monkeypatch.setenv(ladder_ceiling.CEILING_TRAINING_VARIABLE, str(path))
    return path


# ------------------------------------------------------------------------------------------ the whole stage
def test_the_whole_stage_runs_on_what_l1_and_l2_stored_and_writes_a_run_of_its_own(ceiling):
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    config = ceiling.config
    with pytest.raises(RuntimeError, match="stage `ladder_l1` with --seeds 0"):            # nothing to read yet: refused, and it says what to run
        ladder_ceiling.ladder_ceiling_prepare(config)
    assert not ladder_ceiling.source_directory(config).exists()                           # and the refusal created nothing
    _run_stage(config)
    with pytest.raises(RuntimeError, match=r"stage `ladder_l2` for seed 0 .*--stage ladder_l2 --seeds 0.*the base's attempts on G"):
        ladder_ceiling.ladder_ceiling_prepare(config)
    _run_loop(config)
    with pytest.raises(RuntimeError, match=r"stage `ladder_l2_t010` for seed 0 .*--stage ladder_l2_t010 --seeds 0.*its three-round model's attempts on G"):
        ladder_ceiling.ladder_ceiling_prepare(config)
    assert not ceiling.store().is_done(ladder_ceiling.PREPARE) and not ceiling.store().path("problems.jsonl").exists()
    ceiling.stored()
    source, stored = ladder_ceiling.source_directory(config), ladder_ceiling.stored_directories(config)
    assert source == ladder_round._store(config).root and [path.name for path in stored.values()] == ["ladder_l2_seed0", "ladder_l2_t010_seed0"]
    before = {path: _files(path) for path in (source, *stored.values())}
    sent = ScriptedLean.submitted
    summaries = _run_ceiling(config)
    assert {path: _files(path) for path in before} == before                              # the three run directories were only read
    store = ceiling.store()
    assert store.root.name == "ladder_ceiling_seed0" and store.root.parent == source.parent

    # ---- prepare: the file's rows in its order, the schedule by rows, L2's sampling seeds
    prepare, rows = summaries["ladder_ceiling_prepare"], _file_rows(FIXTURE)
    assert prepare["diagnostic"] == "ceiling" and prepare["rows_trained_on"] == 24 == prepare["training_file_rows"] and prepare["steps"] == 3
    assert prepare["checkpoints"] == {"small": {"rows": 12, "step": 2, "rows_seen": 16}, "full": {"rows": 24, "step": 3, "rows_seen": 24}}
    assert prepare["doses"]["small"]["rows"] == 16 and prepare["doses"]["small"]["by_kind"] == {
        kind: sum(row["kind"] == kind for row in rows[:16]) for kind in ("lean_workbook", "stp_conjecture")}
    assert sum(prepare["doses"]["full"]["by_proof_lines"].values()) == 24 and set(prepare["doses"]["full"]["by_proof_lines"]) == {"1", "2-3", "4-7", "8+"}
    of_l2 = ladder_l2.sampling_seeds(config)
    assert prepare["sampling_seeds"] == {"rungs": of_l2["rungs"], "reach": of_l2["reach"], "more": of_l2["control"]}
    control = json.loads((stored["base"] / "episodes_control.done.json").read_text())
    assert prepare["goal_samplings"] == [{"name": "reach", "episodes": 32, "sampling_seed": of_l2["reach"]},
                                         {"name": "more", "episodes": control["episodes_each"], "sampling_seed": of_l2["control"]}]
    assert prepare["attempts_a_goal_problem"] == 32 + control["episodes_each"] and (prepare["loop_arm"], prepare["loop_target_rate"]) == ("t010", 0.1)
    stored_rows = store.read_rows(ladder_ceiling.TRAINING_ROWS_FILE)
    assert [row["problem_id"] for row in stored_rows] == [row["problem_id"] for row in rows] and [row["row"] for row in stored_rows] == list(range(24))
    assert set(stored_rows[0]) == {"row", "problem_id", "kind", "proof_lines", "length_group", "tokens", "certificate_source"}       # no statement, no proof
    assert not any(row["proof"].strip() in content.decode() for row in rows for content in _files(store.root).values())               # no published proof is copied into the run

    # ---- train: one pass in the file's order (stand-in losses), a checkpoint after each number of rows
    train = summaries["ladder_ceiling_train"]
    assert train["steps"] == 3 and train["rows"] == 24 and train["passes"] == 1 and "made up" in train["note"] and train["stand_in_engine"] is True
    assert train["checkpoints"] == [{"checkpoint": "small", "step": 2, "rows_seen": 16}, {"checkpoint": "full", "step": 3, "rows_seen": 24}]
    loss = json.loads(store.path(ladder_ceiling.LOSS_FILE).read_text())
    assert loss["diagnostic"] == "ceiling" and len(loss["row_losses"]) == 24 and [row["rows_seen"] for row in loss["training_steps"]] == [8, 16, 24]
    assert all("example_losses" not in row for row in loss["training_steps"]) and train["rows_compared"] == 12

    # ---- measure: L2's sampling seeds, so every checkpoint pairs by problem with the stored attempts
    base = {part: {row["problem_id"]: row for row in store.read_rows(ladder_ceiling.stored_file("base", part))} for part in ("rungs", REACH, MORE)}
    assert base["rungs"] == {row["problem_id"]: row for row in _file_rows(source / "episodes_rungs_base_problems.jsonl")}
    assert base[MORE] == {row["problem_id"]: row for row in _file_rows(stored["base"] / "episodes_control_problems.jsonl")}
    assert store.read_rows(ladder_ceiling.stored_file("loop", MORE)) == _file_rows(stored["loop"] / "episodes_control_m3_problems.jsonl")
    for name in CHECKPOINTS:
        measured = summaries[f"ladder_ceiling_measure_{name}"]
        assert measured["rungs"]["sampling_seed"] == of_l2["rungs"] and measured["rungs"]["episodes_each"] == 8
        assert (measured["goal"][REACH]["sampling_seed"], measured["goal"][REACH]["episodes_each"]) == (of_l2["reach"], 32)
        assert (measured["goal"][MORE]["sampling_seed"], measured["goal"][MORE]["episodes_each"]) == (of_l2["control"], control["episodes_each"])
        for part, set_name in (("rungs", ladder_ceiling.rung_set(name)), (REACH, ladder_ceiling.goal_set(REACH, name)), (MORE, ladder_ceiling.goal_set(MORE, name))):
            own = {row["problem_id"]: row for row in store.read_rows(f"episodes_{set_name}_problems.jsonl")}
            # The same problems, sides and episodes as the stored measurement; and the stand-in engine has no adapter,
            # so with the same seed it gives the same samples: the same successes, problem by problem.
            assert own == base[part], (name, part)
        assert measured["distinct_attempts"]["rungs"]["prompts"] >= 5 and set(measured["verified_proof_lines"]) == {"rungs", "goal"}
    assert ScriptedLean.submitted > sent

    # ---- every stored file and every set says ceiling
    names = [path.name for path in store.root.iterdir()]
    assert [name for name in names if "ceiling" not in name] == ["problems.jsonl"]                 # the one name the shared episode step reads
    assert {row["set"].split("_")[0] for row in store.read_rows("problems.jsonl")} == {"ceiling"}
    assert {row["set"] for row in store.read_rows("problems.jsonl")} == {f"ceiling_{part}_{name}" for part in ("rungs", REACH, MORE) for name in CHECKPOINTS}

    # ---- the report
    report = summaries["ladder_ceiling_report"]
    assert report["diagnostic"] == "ceiling" and report["branch"]["name"] in BRANCHES and report["stand_in_engine"] is True and report["ok"] is True
    assert all(line.startswith("ceiling: ") for line in report["lines"]) and "labelled diagnostic" in report["headline"]
    assert report["inconclusive"] == (not report["can_this_run_see_a_win"]["all_pass"]) == (report["branch"]["name"] == INCONCLUSIVE)
    assert report["models"] == {"base": report["models"]["base"], "small": "the 16-proof model", "full": "the 24-proof model", "loop": "the three-round model at t = 1/10"}
    measured_read = report["measured_and_not_read"] if report["inconclusive"] else report
    by_length = measured_read["secondary"]["by_length_group"]
    assert by_length["problems"]["all"] == 1 == by_length["problems"]["not_known"] and by_length["problems"]["4_or_more"] == 0       # the fixture's goal problem has no shipped length
    assert by_length["models"]["full"]["all"]["mean"] == 0.0 and by_length["models"]["full"]["all"]["attempts_each"] == prepare["attempts_a_goal_problem"]
    assert by_length["models"]["loop"]["all"]["mean"] == 0.0 and measured_read["primary"]["mean"] is None
    assert report["adapters"]["kept"] is False and report["adapters"]["not_there_to_delete"] == ["small", "full"]                   # the stand-in trains none
    assert json.loads(store.path(REPORT_FILE).read_text()) == json.loads(json.dumps(report, default=str))
    assert store.done_summary(ladder_ceiling.REPORT)["diagnostic"] == "ceiling"
    # A rerun returns what is stored: nothing is trained or sampled again.
    sent = ScriptedLean.submitted
    again = _run_ceiling(config)
    assert ScriptedLean.submitted == sent and again["ladder_ceiling_train"] == train and again["ladder_ceiling_prepare"] == prepare
    assert not ladder_round._store(config).is_done(ladder_ceiling.TRAIN) and not ladder_l2._store(config).is_done(ladder_ceiling.PREPARE)


# ---------------------------------------------------------------------------------------------- the refusals
def test_the_prepare_step_refuses_a_file_it_must_not_train_on_and_writes_nothing(ceiling, monkeypatch, tmp_path):
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    config = ceiling.config
    ceiling.stored()
    held = _file_rows(ladder_ceiling.source_directory(config) / "heldout_groups.jsonl")[0]["problem_id"]

    def refused(match, error=ValueError):
        with pytest.raises(error, match=match):
            ladder_ceiling.ladder_ceiling_prepare(config)
        assert not ceiling.store().is_done(ladder_ceiling.PREPARE) and not ceiling.store().path("problems.jsonl").exists()

    _training_file(tmp_path, monkeypatch, lambda rows: rows[20].update(problem_id=held))       # a held-out problem of the run, past the first checkpoint
    refused(f"1 rows of the ceiling's training file are held-out problems .first: {held}")
    _training_file(tmp_path, monkeypatch, lambda rows: rows[3].update(proof="  nlinarith [sq_nonneg (a - b)]\n" * 400))       # 12,800 characters of proof
    refused(r"1 rows of the ceiling's training file are longer than the 2048 tokens a training example is cut at")
    _training_file(tmp_path, monkeypatch, lambda rows: rows.pop())                               # 23 rows, and the last checkpoint is after 24
    refused("has 23 rows and the last checkpoint is after 24")
    monkeypatch.setenv(ladder_ceiling.CEILING_TRAINING_VARIABLE, str(FIXTURE))
    monkeypatch.setenv(ladder_ceiling.CEILING_CHECKPOINTS_VARIABLE, "12,30")
    refused("has 24 rows and the last checkpoint is after 30")
    monkeypatch.setenv(ladder_ceiling.CEILING_CHECKPOINTS_VARIABLE, "24,12")
    refused("one increasing number of rows for each")
    monkeypatch.setenv(ladder_ceiling.CEILING_CHECKPOINTS_VARIABLE, "12,24")
    monkeypatch.setenv(ladder_ceiling.CEILING_TRAINING_VARIABLE, str(tmp_path / "nowhere" / "training.jsonl"))
    refused("is not in this snapshot", RuntimeError)
    monkeypatch.setenv(ladder_ceiling.CEILING_STORED_VARIABLE, "ladder_l2_smoke")
    refused("it is `none`")
    # The snapshot's own held-out set and base map (the real ones here) bar their problems too, whatever the run placed.
    monkeypatch.delenv(ladder_ceiling.CEILING_STORED_VARIABLE)
    monkeypatch.delenv(ladder_loop.DATA_VARIABLE)
    real = PACKAGE / "data" / "ladder_l0"
    for name, what in (("heldout.jsonl", "held-out problems"), ("base_map.jsonl", "problems of the base map")):
        barred = json.loads((real / name).read_text().splitlines()[0])["problem_id"]
        _training_file(tmp_path, monkeypatch, lambda rows, barred=barred: rows[23].update(problem_id=barred))
        refused(f"1 rows of the ceiling's training file are {what} .first: {barred}")


def test_a_measurement_that_would_not_pair_with_the_stored_ones_is_refused(ceiling, monkeypatch):
    config = ceiling.config
    ceiling.stored()
    config["ladder_loop"]["round"]["sampling_seed"] += 7                       # other sampling seeds than the runs stored
    with pytest.raises(RuntimeError, match="would not pair with the stored results"):
        ladder_ceiling.ladder_ceiling_prepare(config)
    config["ladder_loop"]["round"]["sampling_seed"] -= 7
    config["ladder_loop"]["measure"]["reach_episodes"] = 16                    # another number of attempts than the base's stored ones
    with pytest.raises(RuntimeError, match=r"ladder_l1_seed0 measured the base on G with sampling seed \d+ and 32 episodes"):
        ladder_ceiling.ladder_ceiling_prepare(config)
    config["ladder_loop"]["measure"]["reach_episodes"] = 32
    config["ladder_loop"]["ceiling"]["loop_arm"] = "t005"
    with pytest.raises(ValueError, match="ladder_loop.l2_arms has"):
        ladder_ceiling.ladder_ceiling_prepare(config)
    config["ladder_loop"]["ceiling"]["loop_arm"] = "t010"
    # A stored set Lean did not answer could never be read against: refused before anything is trained.
    stored = ladder_ceiling.stored_directories(config)["loop"] / "episodes_reach_m3_problems.jsonl"
    rows = _file_rows(stored)
    rows[0]["attempts_without_an_answer"] = 8                                  # of the 32 attempts on the one goal problem
    stored.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(RuntimeError, match=r"ladder_l2_t010_seed0 measured its three-round model \(reach\) with too many attempts without a verdict from Lean"):
        ladder_ceiling.ladder_ceiling_prepare(config)
    assert not ceiling.store().is_done(ladder_ceiling.PREPARE)


# ------------------------------------------------------------------------ the training, and the adapters' end
def _as_the_gpu_would(monkeypatch, calls):
    """The train step's GPU path with the model taken out: `_train_with_readings` is replaced by the schedule
    alone (stand-in losses), which records what it was given and writes each checkpoint's adapter directory."""
    def train_with_readings(config, examples, sample, pairs, schedule, seed, directory, tensorboard_run, orders=None, per_example=False):
        calls.append({"examples": examples, "sample": sample, "pairs": pairs, "schedule": schedule, "seed": seed, "directory": directory,
                      "tensorboard_run": tensorboard_run, "orders": orders, "per_example": per_example, "batches": [], "saved": []})
        train_step, read, _ = ladder_dose._stand_in_calls(len(examples), [], [])

        def step(batch):
            calls[-1]["batches"].append(list(batch))
            return train_step(batch)

        def save(name, step_number):
            calls[-1]["saved"].append((name, step_number, len(calls[-1]["batches"])))
            (directory / name).mkdir(parents=True)
            (directory / name / "adapter_model.safetensors").write_text("an adapter")

        return {**run_dose(len(examples), config["training"]["effective_batch"], schedule["passes"], seed, schedule["reading_steps"], schedule["checkpoint_steps"],
                           step, read, save, orders=orders, per_example=per_example), "target_format": "native", "seconds": 1.0}

    capped = []
    monkeypatch.setattr(ladder_dose, "_train_with_readings", train_with_readings)
    monkeypatch.setattr(pipeline, "_cap_torch_memory", lambda config: capped.append(config["gpu"]["desktop_reserve_gb"]))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None, memory_allocated=lambda: 0)))
    return capped


def test_the_training_sees_the_rows_in_the_files_order_with_the_rounds_recipe_and_saves_by_rows_seen(ceiling, monkeypatch):
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    config, calls = ceiling.config, []
    ceiling.stored()
    ladder_ceiling.ladder_ceiling_prepare(config)
    capped = _as_the_gpu_would(monkeypatch, calls)
    monkeypatch.delenv(ladder_loop.STAND_IN_VARIABLE)                          # the train step takes its GPU path
    train = ladder_ceiling.ladder_ceiling_train(config)
    monkeypatch.setenv(ladder_loop.STAND_IN_VARIABLE, "1")
    (call,), rows = calls, _file_rows(FIXTURE)
    # The rows as the round's training examples, in the file's order: the theorem is the statement, the completion the published proof.
    assert call["examples"] == [{"problem_id": row["problem_id"], "side": "statement", "theorem": row["statement"], "completion": row["proof"]} for row in rows]
    assert call["orders"] == [list(range(24))] and call["batches"] == [list(range(0, 8)), list(range(8, 16)), list(range(16, 24))]      # one pass, nothing shuffled
    assert call["schedule"] == {"passes": 1, "reading_steps": [], "checkpoint_steps": {"small": 2, "full": 3}} and call["per_example"] is True
    assert (call["sample"], call["pairs"], call["seed"]) == ([], [], 0) and call["tensorboard_run"] == "ladder_ceiling_seed0"
    assert call["saved"] == [("small", 2, 2), ("full", 3, 3)]                 # after the first step at or after 12 rows (16 seen), and at the end
    assert call["directory"] == ceiling.store().root / "ceiling_adapters" and all(ladder_ceiling.adapter_directory(ceiling.store(), name).is_dir() for name in CHECKPOINTS)
    assert capped == [3.0] and config["gpu"]["desktop_reserve_gb"] == 3.0      # 3 GiB of the card stay free
    assert train["checkpoints"] == [{"checkpoint": "small", "step": 2, "rows_seen": 16}, {"checkpoint": "full", "step": 3, "rows_seen": 24}]
    assert train["allocated_after_cleanup_gb"] == 0 and "note" not in train and train["target_format"] == "native"
    # The round's recipe, as the prepare step recorded it: the config's training settings, and no other.
    recipe = ceiling.store().done_summary(ladder_ceiling.PREPARE)["recipe"]
    assert (recipe["learning_rate"], recipe["warmup_steps"], recipe["effective_batch"], recipe["lora"]) == (1.0e-4, 5, 8, config["lora"])
    # A file that changed since the run was prepared is not trained on.
    ceiling.store().path(f"{ladder_ceiling.TRAIN}.done.json").unlink()
    changed = ceiling.store().root.parent / "changed.jsonl"
    changed.write_text(FIXTURE.read_text() + "\n")
    monkeypatch.setenv(ladder_ceiling.CEILING_TRAINING_VARIABLE, str(changed))
    with pytest.raises(RuntimeError, match="a run trains on the file its prepare step checked"):
        ladder_ceiling.ladder_ceiling_train(config)
    assert len(calls) == 1


def test_the_adapters_are_deleted_only_once_the_report_is_written_and_a_finished_run_is_not_trained_again(ceiling, monkeypatch):
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    from rlvr_lean.reporting import ladder_ceiling as reporting

    config, calls = ceiling.config, []
    ceiling.stored()
    ladder_ceiling.ladder_ceiling_prepare(config)
    _as_the_gpu_would(monkeypatch, calls)
    monkeypatch.delenv(ladder_loop.STAND_IN_VARIABLE)
    ladder_ceiling.ladder_ceiling_train(config)
    store = ceiling.store()
    adapters = [ladder_ceiling.adapter_directory(store, name) for name in CHECKPOINTS]
    with pytest.raises(RuntimeError, match="needs the step ladder_ceiling_measure_small"):          # a report before the measurements: nothing written, nothing deleted
        ladder_ceiling.ladder_ceiling_report(config)
    assert all(path.is_dir() for path in adapters) and not store.path(REPORT_FILE).exists()
    # A measurement of a checkpoint whose adapter is not there is refused (and no engine is loaded for it).
    adapters[0].rename(adapters[0].with_name("elsewhere"))
    with pytest.raises(RuntimeError, match="The measurement cannot be added to this run"):
        ladder_ceiling.ladder_ceiling_measure(config, "small")
    adapters[0].with_name("elsewhere").rename(adapters[0])
    monkeypatch.setenv(ladder_loop.STAND_IN_VARIABLE, "1")                     # the measurements with the stand-in engine
    for name in CHECKPOINTS:
        ladder_ceiling.ladder_ceiling_measure(config, name)
        assert all(path.is_dir() for path in adapters)
    # A report that cannot be built deletes nothing.
    with monkeypatch.context() as patched:
        patched.setattr(reporting, "build_ceiling_report", lambda *given: (_ for _ in ()).throw(KeyError("a report that fails")))
        with pytest.raises(KeyError):
            ladder_ceiling.ladder_ceiling_report(config)
    assert all(path.is_dir() for path in adapters) and not store.path(REPORT_FILE).exists() and not store.is_done(ladder_ceiling.REPORT)
    # The report is on disk, saying what is about to happen, BEFORE the first adapter goes.
    seen, delete = [], ladder_ceiling.delete_adapters

    def watched(given):
        written = json.loads(given.path(REPORT_FILE).read_text())
        seen.append((written["adapters"], written["branch"]["name"], [path.is_dir() for path in adapters]))
        return delete(given)

    monkeypatch.setattr(ladder_ceiling, "delete_adapters", watched)
    report = ladder_ceiling.ladder_ceiling_report(config)
    assert seen == [({"kept": False, "what": "to be deleted once this report is written"}, report["branch"]["name"], [True, True])]
    assert not any(path.exists() for path in adapters) and not (store.root / "ceiling_adapters").exists()
    on_disk = json.loads(store.path(REPORT_FILE).read_text())
    assert on_disk["adapters"]["deleted_after_this_report_was_written"] == ["small", "full"] and on_disk["adapters"]["not_there_to_delete"] == []
    assert on_disk["adapters"]["kept"] is False and "Adapters deleted after this report was written: small, full" in on_disk["lines"][-1]
    assert store.done_summary(ladder_ceiling.REPORT)["adapters"] == on_disk["adapters"]
    # The finished run again: nothing is trained, nothing sampled, and the report built again finds no adapter to delete.
    monkeypatch.delenv(ladder_loop.STAND_IN_VARIABLE)
    sent = ScriptedLean.submitted
    again = _run_ceiling(config)
    assert len(calls) == 1 and ScriptedLean.submitted == sent and again["ladder_ceiling_report"]["adapters"]["not_there_to_delete"] == ["small", "full"]
    assert again["ladder_ceiling_report"]["branch"] == report["branch"]
    # ... and what the first report deleted stays on record, however often the report is built again.
    assert again["ladder_ceiling_report"]["adapters"]["deleted_when_the_report_was_first_written"] == ["small", "full"]
    assert _run_ceiling(config)["ladder_ceiling_report"]["adapters"]["deleted_when_the_report_was_first_written"] == ["small", "full"]
    assert "(deleted when the report was first written: small, full)" in json.loads(store.path(REPORT_FILE).read_text())["lines"][-1]
    # Even with its training marker gone, a run whose report is written is not trained again.
    store.path(f"{ladder_ceiling.TRAIN}.done.json").unlink()
    with pytest.raises(RuntimeError, match="a run whose report is written is not trained again"):
        ladder_ceiling.ladder_ceiling_train(config)
    assert len(calls) == 1


def test_a_report_that_is_not_to_be_read_keeps_the_adapters_and_the_set_is_measured_again_without_a_new_training(ceiling, monkeypatch, tmp_path):
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    from rlvr_lean.gpu import __main__ as gpu_main
    from rlvr_lean.gpu import milestone2

    config, calls = ceiling.config, []
    ceiling.stored()
    ladder_ceiling.ladder_ceiling_prepare(config)
    _as_the_gpu_would(monkeypatch, calls)
    monkeypatch.delenv(ladder_loop.STAND_IN_VARIABLE)
    ladder_ceiling.ladder_ceiling_train(config)
    monkeypatch.setenv(ladder_loop.STAND_IN_VARIABLE, "1")                     # the measurements with the stand-in engine
    store = ceiling.store()
    adapters = [ladder_ceiling.adapter_directory(store, name) for name in CHECKPOINTS]
    first = {name: ladder_ceiling.ladder_ceiling_measure(config, name) for name in CHECKPOINTS}
    # The pool was in trouble while one set was measured: Lean gave no verdict on a fifth of its attempts.
    unanswered = ladder_ceiling.goal_set(MORE, "full")
    name = f"episodes_{unanswered}_problems.jsonl"
    as_measured, rows = store.path(name).read_bytes(), store.read_rows(name)
    rows[0]["attempts_without_an_answer"] = 20
    store.write_rows(name, rows)
    # ---- the report is written, says it is not to be read, keeps the adapters, and the step fails
    monkeypatch.setattr(milestone2, "load_config", lambda path: config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", ladder_ceiling.REPORT, "--out", str(tmp_path / "report.json")])
    assert gpu_main.main() != 0                                                # exit non-zero, as every report that is not to be read
    report = json.loads(store.path(REPORT_FILE).read_text())
    assert report["ok"] is False and report["not_to_be_read"] == ["goal_full_2"] and all(path.is_dir() for path in adapters)
    assert report["adapters"]["kept"] is True and report["adapters"]["there"] == ["small", "full"] and report["adapters"]["kept_because_not_to_be_read"] == ["goal_full_2"]
    assert report["adapters"]["deleted_after_this_report_was_written"] == [] and "the adapters were KEPT: this report is not to be read" in report["lines"][-1]
    assert "measured again from the kept adapters, with no new training" in report["lines"][-1] and "NOT TO BE READ" in report["lines"][-2]
    assert store.done_summary(ladder_ceiling.REPORT)["adapters"]["kept"] is True
    # The report again changes nothing: still not to be read, still kept.
    assert ladder_ceiling.ladder_ceiling_report(config)["adapters"]["kept"] is True and all(path.is_dir() for path in adapters)
    # ---- the task queued again: that set, and no other, is measured again; nothing is trained; the report can be read and deletes the adapters
    untouched = {key: content for key, content in _files(store.root).items() if unanswered not in key and "report" not in key and "measure_full" not in key}
    sent = ScriptedLean.submitted
    again = _run_ceiling(config)
    assert len(calls) == 1 and again["ladder_ceiling_train"]["steps"] == 3                                   # no new training
    assert again["ladder_ceiling_measure_small"] == first["small"] and "measured_again" not in first["full"]
    assert again["ladder_ceiling_measure_full"]["measured_again"] == [unanswered]
    assert ScriptedLean.submitted - sent == first["full"]["goal"][MORE]["pipeline"]["sent_to_lean"]           # that one set's checks, and no other's
    after = _files(store.root)
    assert all(after[key] == content for key, content in untouched.items()) and after[name] == as_measured  # the stand-in samples the same again
    assert again["ladder_ceiling_measure_full"]["goal"][REACH] == first["full"]["goal"][REACH] and again["ladder_ceiling_measure_full"]["rungs"] == first["full"]["rungs"]
    readable = again["ladder_ceiling_report"]
    assert readable["ok"] is True and readable["not_to_be_read"] == [] and not any(path.exists() for path in adapters)
    assert readable["adapters"]["kept"] is False and readable["adapters"]["deleted_after_this_report_was_written"] == ["small", "full"]
    # A third run measures nothing again and keeps the record of what the first readable report deleted.
    sent = ScriptedLean.submitted
    third = _run_ceiling(config)
    assert ScriptedLean.submitted == sent and third["ladder_ceiling_measure_full"] == again["ladder_ceiling_measure_full"]
    assert third["ladder_ceiling_report"]["adapters"]["deleted_when_the_report_was_first_written"] == ["small", "full"]


# ------------------------------------------------------------------------------------------- the smoke stage
def test_the_smoke_stage_runs_on_the_l1_smoke_run_alone_with_the_fixture_and_its_tiny_checkpoints(stage, monkeypatch):  # noqa: F811
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    config = stage.config
    monkeypatch.setenv(ladder_round.ROUND_RUN_VARIABLE, "ladder_l1_smoke")
    _run_stage(config)                                                         # what the L1 smoke task leaves on the box
    monkeypatch.delenv(ladder_round.ROUND_RUN_VARIABLE)
    monkeypatch.delenv(ladder_loop.DATA_VARIABLE)                              # as on the box: the smoke stage names no L0 data, so the snapshot's own is read
    options = entry.step_fields(entry.STAGES["ladder_ceiling_smoke"][2])[2]
    variables = entry.child_environment("gpu", "key", options)
    assert {name: variables[name] for name in VARIABLES} == options["environment"] == {
        ladder_ceiling.CEILING_SOURCE_VARIABLE: "ladder_l1_smoke", ladder_ceiling.CEILING_RUN_VARIABLE: "ladder_ceiling_smoke",
        ladder_ceiling.CEILING_STORED_VARIABLE: "none", ladder_ceiling.CEILING_TRAINING_VARIABLE: str(FIXTURE), ladder_ceiling.CEILING_CHECKPOINTS_VARIABLE: "12,24"}
    for name in VARIABLES:
        monkeypatch.setenv(name, variables[name])
    assert ladder_ceiling.stored_directories(config) is None and ladder_ceiling.source_directory(config).name == "ladder_l1_smoke"
    summaries = _run_ceiling(config, stage_name="ladder_ceiling_smoke")
    store = ladder_ceiling._store(config)
    assert store.root.name == "ladder_ceiling_smoke" and not (store.root.parent / "ladder_ceiling_seed0").exists()
    prepare = summaries["ladder_ceiling_prepare"]
    assert prepare["stored_runs"] is None and prepare["loop_arm"] is None and prepare["attempts_a_goal_problem"] == 32
    assert [sampling["name"] for sampling in prepare["goal_samplings"]] == [REACH] and prepare["checkpoints"]["small"] == {"rows": 12, "step": 2, "rows_seen": 16}
    assert {row["set"] for row in store.read_rows("problems.jsonl")} == {f"ceiling_{part}_{name}" for part in ("rungs", REACH) for name in CHECKPOINTS}
    report = summaries["ladder_ceiling_report"]
    assert report["branch"]["name"] in (NOT_READ, INCONCLUSIVE) and "loop" not in report["models"] and report["sizes"]["stored_runs"] is None
    assert any("L2's stored attempts were not read (a smoke run)" in line for line in report["lines"]) and store.is_done(ladder_ceiling.REPORT)


# -------------------------------------------------------------------------------------------- registration
def test_the_stage_is_registered_with_a_guard_before_every_gpu_step():
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_ceiling"]]
    assert [(environment, step) for environment, step, _ in steps] == [
        ("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_ceiling_prepare"), ("guard", None), ("gpu", "ladder_ceiling_train"),
        ("guard", None), ("gpu", "ladder_ceiling_measure_small"), ("guard", None), ("gpu", "ladder_ceiling_measure_full"), ("guard", None), ("gpu", "ladder_ceiling_report")]
    gpu_steps = [step for environment, step, _ in steps if environment == "gpu"]
    assert gpu_steps[1:] == list(ladder_ceiling.STEPS) and all(options == {} for _, _, options in steps)
    assert [f"ladder_ceiling_measure_{name}" for name in CHECKPOINTS] == gpu_steps[3:5]
    others = set(ladder_round.STEPS) | set(ladder_dose.STEPS) | set(ladder_l2.STEPS) | set(ladder_loop.STEPS) | set(ladder_l3a.STEPS) | set(ladder_l3c.STEPS)
    assert not set(ladder_ceiling.STEPS) & others and all("ceiling" in name for name in ladder_ceiling.STEPS)
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_ceiling_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in steps]
    assert all(options == smoke[0][2] for _, _, options in smoke)
    own = entry.child_environment("gpu", "key", steps[2][2])
    assert not [name for name in VARIABLES if name in own]
    assert ladder_ceiling.PACKAGE_TRAINING == PACKAGE / "data" / "ladder_ceiling" / "training.jsonl" and ladder_ceiling.training_file() == ladder_ceiling.PACKAGE_TRAINING
    for other in ("ladder_l1b_smoke", "ladder_l2_smoke", "ladder_l3c_smoke"):
        assert not [name for name in VARIABLES if name in entry.child_environment("gpu", "key", entry.step_fields(entry.STAGES[other][2])[2])]
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    assert len(_file_rows(FIXTURE)) == 24


def test_the_steps_run_through_the_gpu_entry_point(ceiling, monkeypatch, tmp_path):
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    from rlvr_lean.gpu import __main__ as gpu_main
    from rlvr_lean.gpu import milestone2

    ceiling.stored()
    monkeypatch.setattr(milestone2, "load_config", lambda path: ceiling.config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    for name in ladder_ceiling.STEPS:
        monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", name, "--out", str(tmp_path / f"{name}.json")])
        assert gpu_main.main() == 0 and json.loads((tmp_path / f"{name}.json").read_text())["ok"] is True
    assert ceiling.store().is_done(ladder_ceiling.REPORT)
