"""Spec §6 item 2a: a pair in the model's native format holds the sequence-start token between the prompt and
the proof, as context; the same encoding serves training, scoring and the held-out loss."""

import pytest

from rlvr_lean.domain.training.target_format import LEGACY, NATIVE, in_target_format, is_heldout_read_step, native_format
from rlvr_lean.gpu.model_utils import encode_pair, encode_with_proof_mask

BOS = 100


def test_native_format_puts_the_sequence_start_token_between_prompt_and_target():
    ids, mask = native_format([BOS, 7, 8, 9, 20, 21], [False, False, False, False, True, True], bos_token_id=BOS)
    assert ids == [BOS, 7, 8, 9, BOS, 20, 21]
    assert mask == [False, False, False, False, False, True, True]           # the inserted token is context, not a target
    assert [token for token, keep in zip(ids, mask) if keep] == [20, 21]


def test_native_format_refuses_a_pair_without_a_target_or_with_mismatched_lengths():
    with pytest.raises(ValueError):
        native_format([BOS, 7], [False, False], bos_token_id=BOS)
    with pytest.raises(ValueError):
        native_format([BOS, 7, 8], [False, True], bos_token_id=BOS)


def test_the_legacy_format_is_the_pair_unchanged_and_needs_no_token():
    assert in_target_format([BOS, 7, 20], [False, False, True], LEGACY, None) == ([BOS, 7, 20], [False, False, True])


def test_the_native_format_needs_the_token_and_an_unknown_format_is_refused():
    assert in_target_format([BOS, 7, 20], [False, False, True], NATIVE, BOS) == ([BOS, 7, BOS, 20], [False, False, False, True])
    with pytest.raises(ValueError):
        in_target_format([BOS, 7, 20], [False, False, True], NATIVE, None)
    with pytest.raises(ValueError):
        in_target_format([BOS, 7, 20], [False, False, True], "nativ", BOS)


class WordTokenizer:
    """One token per whitespace-separated word (its trailing space included), after a sequence-start token."""

    bos_token_id = BOS

    def __call__(self, text, return_offsets_mapping=False):
        ids, offsets, start = [BOS], [(0, 0)], 0
        for word in text.split(" "):
            end = start + len(word)
            if word:
                ids.append(1 + len(ids))
                offsets.append((start, end))
            start = end + 1
        return {"input_ids": ids, "offset_mapping": offsets}


def test_encode_pair_native_differs_from_legacy_by_exactly_the_inserted_context_token():
    tokenizer, prompt, target = WordTokenizer(), "theorem t := by ", "simp done"
    legacy_ids, legacy_mask = encode_pair(tokenizer, prompt, target, 2048, LEGACY)
    native_ids, native_mask = encode_pair(tokenizer, prompt, target, 2048, NATIVE)
    assert (legacy_ids, legacy_mask) == encode_with_proof_mask(tokenizer, prompt, target, 2048)
    boundary = legacy_mask.index(True)
    assert native_ids == legacy_ids[:boundary] + [BOS] + legacy_ids[boundary:]
    assert native_mask == legacy_mask[:boundary] + [False] + legacy_mask[boundary:]
    assert sum(native_mask) == sum(legacy_mask) == 2                          # the same target tokens, the same count
    assert [t for t, keep in zip(native_ids, native_mask) if keep] == [t for t, keep in zip(legacy_ids, legacy_mask) if keep]


def test_encode_pair_truncates_after_the_format_is_applied():
    ids, mask = encode_pair(WordTokenizer(), "theorem t := by ", "simp done", 7, NATIVE)      # 8 tokens in the native format
    assert len(ids) == len(mask) == 7 and ids[-2] == BOS and mask[-2:] == [False, True]


def test_the_heldout_loss_is_read_at_step_0_then_every_step_to_the_limit_then_every_sixth_and_last():
    read = [step for step in range(0, 55) if is_heldout_read_step(step, 54, every=6, every_step_until=12)]
    assert read == list(range(0, 13)) + [18, 24, 30, 36, 42, 48, 54]


def test_without_the_early_limit_the_schedule_is_phase_b_s_and_the_last_step_is_always_read():
    assert [step for step in range(0, 55) if is_heldout_read_step(step, 54, 6, 0)] == [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
    assert is_heldout_read_step(52, 52, 6, 0) and not is_heldout_read_step(51, 52, 6, 0)


def test_a_step_outside_the_run_and_a_zero_interval_are_refused():
    with pytest.raises(ValueError):
        is_heldout_read_step(55, 54, 6, 12)
    with pytest.raises(ValueError):
        is_heldout_read_step(3, 54, 0, 12)
