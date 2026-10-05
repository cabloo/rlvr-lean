"""The desktop's VRAM reserve (spec §1 item 8a): no stage may take the last `gpu.desktop_reserve_gb` of the
card. On 2026-10-03 the sampling engine at 85% of the card left 230 MiB and no window manager could
start."""

from pathlib import Path

import pytest
import yaml

from rlvr_lean.gpu.memory_budget import process_memory_fraction, sampling_memory_fraction, token_budget_batches, usable_mib

CARD_MIB = 15819          # the GPU box's usable memory (15.45 GiB, the figure vLLM reports)
CONFIG = yaml.safe_load((Path(__file__).resolve().parents[2] / "src" / "rlvr_lean" / "config" / "experiment.yaml").read_text())

# The sampling engine's own accounting on this card (vLLM's start-up report, task efbe391e): before any cache
# it spends 7.61 GiB on weights and non-PyTorch memory and reserves 1.42 GiB for peak activation; one
# 4,096-token sequence needs 1.88 GiB of cache, and the engine refuses to start without that much.
ENGINE_FIXED_GIB = 7.61 + 1.42
ONE_SEQUENCE_GIB_PER_4096_TOKENS = 1.88


def _cache_gib(reserve_gb: float) -> float:
    fraction = sampling_memory_fraction(CARD_MIB, reserve_gb, CONFIG["gpu"]["vllm_overhead_gb"], 0.85)
    return fraction * CARD_MIB / 1024 - ENGINE_FIXED_GIB


def test_the_accounting_reproduces_both_start_ups_the_engine_reported():
    """4.09 GiB of cache at 0.85 (started), 1.54 GiB at 0.685 (refused: task a7c41ab7)."""
    assert 0.85 * CARD_MIB / 1024 - ENGINE_FIXED_GIB == pytest.approx(4.09, abs=0.02)
    assert 0.685 * CARD_MIB / 1024 - ENGINE_FIXED_GIB == pytest.approx(1.54, abs=0.02)


def test_the_sampling_engines_whole_footprint_leaves_the_reserve_free():
    reserve, overhead = CONFIG["gpu"]["desktop_reserve_gb"], CONFIG["gpu"]["vllm_overhead_gb"]
    fraction = sampling_memory_fraction(CARD_MIB, reserve, overhead, CONFIG["vllm"]["gpu_memory_utilization"])
    footprint_mib = fraction * CARD_MIB + overhead * 1024
    assert footprint_mib <= CARD_MIB - reserve * 1024
    assert fraction == pytest.approx(0.741, abs=0.001)


def test_the_incident_setting_would_not_have_left_room():
    """0.85 of the card plus the engine's overhead plus the display's 1,135 MiB is what filled it."""
    assert 0.85 * CARD_MIB + 1024 + 1135 > CARD_MIB - 300


def test_the_configured_reserve_leaves_the_engine_one_full_sequence_with_margin():
    one_sequence = ONE_SEQUENCE_GIB_PER_4096_TOKENS * CONFIG["vllm"]["max_model_len"] / 4096
    assert _cache_gib(CONFIG["gpu"]["desktop_reserve_gb"]) > one_sequence + 0.25


def test_a_four_gib_reserve_cannot_start_the_engine_at_this_sequence_length():
    """Why the reserve is 3 and not 4: the first re-queue asked for 4 and the engine refused to start."""
    assert _cache_gib(4.0) < ONE_SEQUENCE_GIB_PER_4096_TOKENS


def test_the_ceiling_wins_on_a_card_with_room_to_spare_and_rounding_never_takes_from_the_reserve():
    assert sampling_memory_fraction(81559, 4.0, 1.0, 0.85) == 0.85                 # an 80 GB card
    for total in (8192, 12288, 15819, 16384, 24576):
        fraction = sampling_memory_fraction(total, 4.0, 1.0, 1.0)
        assert fraction * total + 1024 <= total - 4.0 * 1024


def test_a_torch_stages_whole_footprint_leaves_the_reserve_free():
    """The cap governs PyTorch's allocator; the CUDA context sits outside it and counts too."""
    reserve, overhead = CONFIG["gpu"]["desktop_reserve_gb"], CONFIG["gpu"]["torch_overhead_gb"]
    fraction = process_memory_fraction(CARD_MIB, reserve, overhead)
    assert fraction * CARD_MIB + overhead * 1024 <= CARD_MIB - reserve * 1024
    assert fraction == pytest.approx(0.773, abs=0.001)


def test_the_torch_cap_still_holds_the_largest_training_step_measured():
    """The heuristic arm's training peaked at 10.97e9 bytes allocated (Phase B seed 0), the largest so far."""
    fraction = process_memory_fraction(CARD_MIB, CONFIG["gpu"]["desktop_reserve_gb"], CONFIG["gpu"]["torch_overhead_gb"])
    assert fraction * CARD_MIB * 2**20 > 10.97e9


def test_usable_memory_is_used_plus_free_not_the_total_that_includes_the_drivers_share():
    """During the incident nvidia-smi showed 15,590 MiB used and 229 free; its `memory.total` says 16,303."""
    assert usable_mib("15590, 229\n") == CARD_MIB
    assert usable_mib("1143, 14676\n0, 24576\n") == CARD_MIB       # the first card only


def _need(count: int) -> int:
    return -(-(count + 1) // 16) * 16


def test_loss_prompts_are_grouped_to_fit_the_cache_in_order_and_none_is_dropped():
    """114 held-out prompts need about 25,000 tokens of cache at once; the engine had 5,264 and died
    (task 83efeb13). Each group must fit the budget, so nothing is evicted half done."""
    counts = [130 + (37 * index) % 420 for index in range(114)]
    budget = CONFIG["vllm"]["max_model_len"] // 2
    assert sum(_need(count) for count in counts) > 5264                      # one call would not fit
    groups = token_budget_batches(counts, budget)
    assert [index for start, stop in groups for index in range(start, stop)] == list(range(114))
    assert all(sum(_need(count) for count in counts[start:stop]) <= budget for start, stop in groups)
    assert len(groups) > 1


def test_a_sequence_larger_than_the_budget_is_scored_alone_and_an_empty_set_makes_no_group():
    assert token_budget_batches([100, 3000, 100], 2048) == [(0, 1), (1, 2), (2, 3)]
    assert token_budget_batches([2047], 2048) == [(0, 1)]                    # 2,048 with its generated token
    assert token_budget_batches([], 2048) == []
    assert token_budget_batches([15, 15, 15], 32) == [(0, 2), (2, 3)]        # each needs one 16-token block


@pytest.mark.parametrize("call", [
    lambda: token_budget_batches([10], 0),
    lambda: token_budget_batches([10], 2048, block_tokens=0),
    lambda: sampling_memory_fraction(0, 4.0, 1.0, 0.85),
    lambda: sampling_memory_fraction(4096, 4.0, 1.0, 0.85),        # nothing left after the reserve
    lambda: sampling_memory_fraction(15819, -1.0, 1.0, 0.85),
    lambda: sampling_memory_fraction(15819, 4.0, 1.0, 1.5),
    lambda: process_memory_fraction(15819, 16.0, 0.5),
    lambda: process_memory_fraction(15819, 15.2, 0.5),            # the overhead alone takes what is left
    lambda: process_memory_fraction(15819, 4.0, -0.5),
    lambda: process_memory_fraction(0, 4.0, 0.5),
])
def test_impossible_budgets_are_refused(call):
    with pytest.raises(ValueError):
        call()
