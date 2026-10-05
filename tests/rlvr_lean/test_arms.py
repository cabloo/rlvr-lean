"""Spec fixture 12 (§13 "Artifact names"): the reused arm keeps Phase A's names; every other arm is keyed."""

import pytest

from rlvr_lean.domain.training.arms import (
    LADDER_ARM,
    LEGACY_ARMS,
    NATIVE_ARMS,
    SCOUT_ARM,
    BaseLossNames,
    arm_names,
    base_loss_names,
    check_arm_list,
)
from rlvr_lean.domain.training.target_format import LEGACY, NATIVE

REUSED = "learning_progress_cosine"


def test_the_reused_arm_resolves_to_phase_a_names_exactly():
    names = arm_names(REUSED, 2, REUSED)
    assert (names.train_marker, names.sampling_marker, names.loss_marker) == (
        "train_adapter_seed2", "evaluate_sampling_seed2", "evaluate_loss_seed2")
    assert (names.eval_attempts_file, names.eval_verification_file) == ("eval_attempts_seed2.jsonl", "eval_verification_seed2.jsonl")
    assert (names.fp8_loss_file, names.nf4_loss_file) == ("heldout_loss_fp8_seed2.jsonl", "heldout_loss_nf4_seed2.jsonl")
    assert names.variant == "eval_adapter_s2"
    assert names.adapter_directory == "learning_progress_seed2"
    assert names.tensorboard_run == "learning_progress_cosine_seed2"


def test_other_arms_are_keyed_by_arm_and_seed():
    names = arm_names("half_pass_rate", 0, REUSED)
    assert names.train_marker == "train_adapter_half_pass_rate_seed0"
    assert names.eval_attempts_file == "eval_attempts_half_pass_rate_seed0.jsonl"
    assert names.nf4_loss_file == "heldout_loss_nf4_half_pass_rate_seed0.jsonl"
    assert names.variant == "eval_half_pass_rate_s0"
    assert names.adapter_directory == names.tensorboard_run == "half_pass_rate_seed0"


def test_keys_never_collide_across_arms_or_seeds():
    keys = [arm_names(arm, seed, REUSED).key for arm in ("learning_progress_cosine", "random", "difficulty_heuristic", "half_pass_rate")
            for seed in range(3)]
    assert len(set(keys)) == len(keys)


@pytest.mark.parametrize("arm,seed", [("lp", 0), ("random", -1)])
def test_unknown_arms_and_negative_seeds_are_refused(arm, seed):
    with pytest.raises(ValueError):
        arm_names(arm, seed, REUSED)


# --- spec §6 item 2a and §13a: a name fixes its pair format ------------------------------------------------

def test_every_phase_a_and_phase_b_arm_stays_in_the_legacy_format_and_trains_on_its_own_selection():
    for arm in LEGACY_ARMS:
        names = arm_names(arm, 0, REUSED)
        assert names.target_format == LEGACY and names.examples_arm == arm


def test_the_scout_arm_is_native_keyed_by_its_own_name_and_trains_on_phase_a_s_selection():
    names = arm_names(SCOUT_ARM, 0, REUSED)
    assert SCOUT_ARM == "native_same_picks"
    assert names.target_format == NATIVE and names.examples_arm == REUSED
    assert (names.train_marker, names.sampling_marker, names.loss_marker) == (
        "train_adapter_native_same_picks_seed0", "evaluate_sampling_native_same_picks_seed0", "evaluate_loss_native_same_picks_seed0")
    assert (names.eval_attempts_file, names.eval_verification_file) == (
        "eval_attempts_native_same_picks_seed0.jsonl", "eval_verification_native_same_picks_seed0.jsonl")
    assert (names.fp8_loss_file, names.nf4_loss_file) == (
        "heldout_loss_fp8_native_same_picks_seed0.jsonl", "heldout_loss_nf4_native_same_picks_seed0.jsonl")
    assert names.variant == "eval_native_same_picks_s0"
    assert names.adapter_directory == names.tensorboard_run == "native_same_picks_seed0"


def _every_name(names) -> set[str]:
    return {names.key, names.variant, names.adapter_directory, names.train_marker, names.sampling_marker, names.loss_marker,
            names.eval_attempts_file, names.eval_verification_file, names.fp8_loss_file, names.nf4_loss_file}


def test_no_native_name_is_a_legacy_name():
    legacy = set()
    for arm in LEGACY_ARMS:
        for seed in range(3):
            legacy |= _every_name(arm_names(arm, seed, REUSED))
    base = base_loss_names(LEGACY)
    legacy |= {base.nf4_marker, base.nf4_file, base.fp8_marker, base.fp8_file}
    native = set()
    for arm in NATIVE_ARMS:
        for seed in range(3):
            native |= _every_name(arm_names(arm, seed, REUSED))
    assert len(native) == len(NATIVE_ARMS) * 3 * 9                        # no two native runs share a name (key = adapter directory)
    native_base = base_loss_names(NATIVE)
    native |= {native_base.nf4_marker, native_base.nf4_file, native_base.fp8_marker, native_base.fp8_file}
    assert not legacy & native


def test_the_base_s_heldout_loss_keeps_phase_a_s_names_in_the_legacy_format_and_gets_its_own_in_the_native():
    assert base_loss_names(LEGACY) == BaseLossNames("evaluate_loss_base", "heldout_loss_nf4_base.jsonl", "evaluate_base", "heldout_loss_fp8_base.jsonl")
    assert base_loss_names(NATIVE) == BaseLossNames("evaluate_loss_native_base", "heldout_loss_nf4_native_base.jsonl",
                                                    "evaluate_fp8_loss_native_base", "heldout_loss_fp8_native_base.jsonl")
    with pytest.raises(ValueError):
        base_loss_names("nativ")


def test_a_native_arm_runs_on_its_own():
    check_arm_list([SCOUT_ARM])
    check_arm_list([LADDER_ARM])
    check_arm_list(["learning_progress_cosine", "random", "difficulty_heuristic", "half_pass_rate"])
    with pytest.raises(ValueError):
        check_arm_list([REUSED, SCOUT_ARM])
    with pytest.raises(ValueError):
        check_arm_list([SCOUT_ARM, LADDER_ARM])
    with pytest.raises(ValueError):
        check_arm_list(["random", "random"])
    with pytest.raises(ValueError):
        check_arm_list(["native_other_picks"])
