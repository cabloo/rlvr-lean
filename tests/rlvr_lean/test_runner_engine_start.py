"""The stage runner runs a GPU step again when its sampling engine did not start (src/rlvr_lean/runner/entry.py; the
first experiment's spec §7 item 8a).

`ladder_l5_noise_seed0_r1` (task e034e858, 2026-10-11) ended at its fourth engine start with three models of ten
measured: the engine's cache came to 1.87 GiB against the 1.88 it starts with, because about 0.3 GiB of the card was
taken by its other users in the 20 seconds of the start."""

import json
from pathlib import Path

import pytest

from rlvr_lean.runner import entry

DID_NOT_START = {"ok": False, "error": "RuntimeError('Engine core initialization failed. See root cause above. Failed core proc(s): {}')"}
RAN = {"ok": True}


class QuietMonitor:
    """The entry's monitor without its thread: a step's peak is what the scripted step says it was."""

    def __init__(self, out: Path) -> None:
        self.child_pid = None
        self.peak_gpu_mib = 0
        self.peak_tree_rss_mib = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def reset_peaks(self) -> None:
        self.peak_gpu_mib = 0
        self.peak_tree_rss_mib = 0


class World:
    """A stage of two GPU steps with a guard between them, each start of a step scripted: (return code, the result it
    writes or None for a step that leaves none, the card's peak while it ran)."""

    def __init__(self) -> None:
        self.script: dict[str, list[tuple]] = {}
        self.card: list[int] = [1600]            # card-wide used memory, one reading a look (the last one repeats)
        self.started: list[str] = []
        self.waits: list[float] = []

    def used(self) -> int:
        return self.card.pop(0) if len(self.card) > 1 else self.card[0]

    def run_child(self, command: list[str], env: dict, monitor: QuietMonitor) -> int:
        step, step_out = command[3], Path(command[5])
        self.started.append(step)
        returncode, result, peak = self.script[step].pop(0)
        monitor.peak_gpu_mib = peak
        if result is not None:
            step_out.write_text(json.dumps(result))
        return returncode


@pytest.fixture
def world(monkeypatch, tmp_path):
    world = World()
    monkeypatch.setitem(entry.STAGES, "stand_in", [("gpu", "first"), ("guard", None), ("gpu", "second")])
    monkeypatch.setattr(entry, "STORE", tmp_path / "store")
    monkeypatch.setattr(entry, "Monitor", QuietMonitor)
    monkeypatch.setattr(entry, "ensure_uv", lambda: Path("uv"))
    monkeypatch.setattr(entry, "gpu_memory_used_mib", world.used)
    monkeypatch.setattr(entry, "run_child", world.run_child)
    monkeypatch.setattr(entry.time, "sleep", world.waits.append)
    world.run = lambda: entry.run_stage("stand_in", tmp_path / "out", None)
    return world


def step_of(report: dict, name: str) -> dict:
    return next(step for step in report["steps"] if step["step"] == name)


def test_a_step_whose_engine_did_not_start_is_run_again_and_the_stage_goes_on(world):
    world.script = {"first": [(1, DID_NOT_START, 10900), (0, RAN, 14368)], "second": [(0, RAN, 14368)]}
    report = world.run()
    assert report["passed"] and "failed_step" not in report
    assert world.started == ["first", "first", "second"] and world.waits == [entry.ENGINE_START_WAIT_SECONDS]
    first = step_of(report, "first")
    assert first["returncode"] == 0 and first["result"] == RAN and first["peak_gpu_used_mib"] == 14368
    assert [start["returncode"] for start in first["engine_starts"]] == [1]
    assert entry.ENGINE_DID_NOT_START in first["engine_starts"][0]["error"]
    assert "engine_starts" not in step_of(report, "second")


def test_three_failed_starts_fail_the_stage_as_one_did(world):
    world.script = {"first": [(1, DID_NOT_START, 10900)] * 3, "second": [(0, RAN, 14368)]}
    report = world.run()
    assert not report["passed"] and report["failed_step"] == "first" and report["failed_steps"] == ["first"]
    assert world.started == ["first"] * entry.ENGINE_STARTS == ["first"] * 3
    assert world.waits == [entry.ENGINE_START_WAIT_SECONDS] * 2
    first = step_of(report, "first")
    assert first["returncode"] == 1 and first["result"] == DID_NOT_START and len(first["engine_starts"]) == 2
    assert [step["step"] for step in report["steps"]] == ["first"]          # the stage stops there: no guard, no second step


@pytest.mark.parametrize("returncode, result", [
    (1, {"ok": False, "error": "OutOfMemoryError('CUDA out of memory. Tried to allocate 562.00 MiB.')"}),      # a training at its cap
    (1, {"ok": False, "error": "ValueError('the run of L5 is not finished: queue rlvr_lean/ladder_l5_deep_seed0')"}),   # a refusal
    (entry.SOUNDNESS_ALARM_EXIT, {"ok": False, "error": "SoundnessAlarm('both sides verified')", "soundness_alarm": "both sides verified"}),
    (entry.SOUNDNESS_ALARM_EXIT, DID_NOT_START),        # the alarm's code is never run again, whatever the words beside it
    (-9, None),                                         # killed: it left no result
    (1, {"ok": False}),                                 # a result with no error
])
def test_no_other_failure_is_run_again(world, returncode, result):
    world.script = {"first": [(returncode, result, 14368)], "second": [(0, RAN, 14368)]}
    report = world.run()
    assert not report["passed"] and report["failed_step"] == "first"
    assert world.started == ["first"] and world.waits == []
    assert "engine_starts" not in step_of(report, "first") and "engine_start_guard" not in step_of(report, "first")


def test_a_stage_whose_engines_all_started_reports_what_it_reported(world):
    world.script = {"first": [(0, RAN, 14368)], "second": [(0, RAN, 14368)]}
    report = world.run()
    assert report["passed"] and world.started == ["first", "second"] and world.waits == []
    assert set(step_of(report, "first")) == {"step", "result", "returncode", "seconds", "peak_gpu_used_mib", "peak_step_rss_mib"}
    assert [step["step"] for step in report["steps"]] == ["first", "gpu_guard", "second"]


def test_a_failed_engine_that_keeps_the_card_is_not_started_again(world, monkeypatch):
    world.script = {"first": [(1, DID_NOT_START, 14368)], "second": [(0, RAN, 14368)]}
    asked = []

    def guard(baseline_mib, margin_mib, timeout_seconds=120, step_peak_mib=None, release_mib=entry.GUARD_RELEASE_MIB):
        asked.append((baseline_mib, margin_mib, step_peak_mib))
        return {"ok": False, "used_mib": 13400, "baseline_mib": baseline_mib, "margin_mib": margin_mib, "step_peak_mib": step_peak_mib}

    monkeypatch.setattr(entry, "wait_for_idle_gpu", guard)
    report = world.run()
    assert asked == [(1600, 300, 14368)]                # the guard the stage's own steps are held to, on the failed start's peak
    assert not report["passed"] and report["failed_step"] == "first" and world.started == ["first"] and world.waits == []
    first = step_of(report, "first")
    assert first["engine_start_guard"]["used_mib"] == 13400 and "engine_starts" not in first


def test_the_start_after_a_failed_one_has_a_result_of_its_own_or_none(world):
    """The failed start's result file is gone before the next start: a start that is killed leaves none, and is not read
    as a third engine that did not start."""
    world.script = {"first": [(1, DID_NOT_START, 10900), (-9, None, 14368), (0, RAN, 14368)], "second": [(0, RAN, 14368)]}
    report = world.run()
    assert world.started == ["first", "first"] and not report["passed"]
    first = step_of(report, "first")
    assert first["returncode"] == -9 and "result" not in first and len(first["engine_starts"]) == 1


def test_the_case_that_failed_the_run_the_desktop_took_memory_during_the_start(world):
    """Task e034e858: the card's other users held 1,641 MiB when the task began and about 300 MiB more after the failed
    start. The failed engine's memory is released, so the step is run again, and the next guard's baseline is what the
    card's other users hold now."""
    world.card[:] = [1641, 1948]
    world.script = {"first": [(1, DID_NOT_START, 10900), (0, RAN, 14368)], "second": [(0, RAN, 14368)]}
    report = world.run()
    assert report["passed"] and report["gpu_baseline_mib"] == 1641 and world.started == ["first", "first", "second"]
    guard = step_of(report, "gpu_guard")
    assert guard["ok"] and guard["baseline_mib"] == 1948 and guard["passed_because"] == "back at the baseline"


@pytest.mark.parametrize("returncode, result, expected", [
    (1, DID_NOT_START, True),
    (0, DID_NOT_START, False),
    (entry.SOUNDNESS_ALARM_EXIT, DID_NOT_START, False),
    (1, None, False),
    (1, {"ok": False, "error": None}, False),
    (1, {"ok": False, "error": "RuntimeError('CUDA error: out of memory')"}, False),
])
def test_what_counts_as_an_engine_that_did_not_start(returncode, result, expected):
    assert entry.engine_did_not_start(returncode, result) is expected


def test_the_baseline_moves_only_when_the_cards_other_users_hold_more():
    released = {"ok": True, "used_mib": 1948, "passed_because": "the step's memory was released; the card's other users hold more than at the baseline"}
    assert entry.baseline_after(released, 1641) == 1948
    assert entry.baseline_after({"ok": True, "used_mib": 1700, "passed_because": "back at the baseline"}, 1641) == 1641
    assert entry.baseline_after({"ok": False, "used_mib": 13400}, 1641) == 1641
    assert entry.baseline_after({"ok": True, "detail": "nvidia-smi unavailable; guard skipped"}, None) is None
