"""The stage runner's GPU guard (src/rlvr_lean/runner/entry.py): a step must have given its memory back before the
next one starts, and the card's OTHER users (a desktop session) are not this task's to wait for.

`ladder_l0b_r1` (2026-10-04) failed at its last guard with both attempt steps done: the desktop held 485 MiB
more than when the task started, against a margin of 300 MiB."""

import pytest

from rlvr_lean.runner import entry


@pytest.fixture
def card(monkeypatch):
    """A card whose used memory is read from a list, one reading per look (the last one repeats)."""
    readings: list[int | None] = []

    def used():
        return readings.pop(0) if len(readings) > 1 else readings[0]

    monkeypatch.setattr(entry, "gpu_memory_used_mib", used)
    monkeypatch.setattr(entry.time, "sleep", lambda seconds: None)
    return readings


def test_a_card_back_at_the_baseline_passes(card):
    card[:] = [1300]
    result = entry.wait_for_idle_gpu(1175, 300, timeout_seconds=0, step_peak_mib=14368)
    assert result["ok"] and result["passed_because"] == "back at the baseline"


def test_the_case_that_failed_the_first_run_passes_the_model_is_gone_and_the_desktop_holds_more(card):
    card[:] = [1660]
    result = entry.wait_for_idle_gpu(1175, 300, timeout_seconds=0, step_peak_mib=14368)
    assert result["ok"] and result["passed_because"].startswith("the step's memory was released")
    assert result["used_mib"] == 1660 and result["step_peak_mib"] == 14368


def test_a_model_that_lingers_still_fails(card):
    card[:] = [13400]
    result = entry.wait_for_idle_gpu(1175, 300, timeout_seconds=0, step_peak_mib=14368)
    assert not result["ok"] and "passed_because" not in result


def test_a_model_that_goes_while_the_guard_waits_passes(card):
    card[:] = [13400, 13400, 1660]
    result = entry.wait_for_idle_gpu(1175, 300, timeout_seconds=60, step_peak_mib=14368)
    assert result["ok"] and result["used_mib"] == 1660


def test_growth_after_a_step_that_held_no_model_is_not_excused(card):
    """Nothing says the memory is someone else's when the step before held none: the old rule alone applies."""
    card[:] = [1660]
    assert not entry.wait_for_idle_gpu(1175, 300, timeout_seconds=0, step_peak_mib=1200)["ok"]
    assert not entry.wait_for_idle_gpu(1175, 300, timeout_seconds=0)["ok"]


def test_a_box_without_nvidia_smi_skips_the_guard(card):
    card[:] = [None]
    assert entry.wait_for_idle_gpu(None, 300)["ok"]
