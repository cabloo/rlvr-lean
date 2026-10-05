"""The pure reads of `rlvr_lean.domain.diagnosis` (ledger: the held-out saturation and learning-progress score diagnosis)."""

import pytest

from rlvr_lean.domain.diagnosis import longest_other_proof, loss_by_part, native_format, per_proof_constant, split_target


def test_a_target_splits_into_first_body_newline_and_fence():
    assert split_target([1.0, 2.0, 3.0, 4.0, 5.0]) == {"first": [1.0], "body": [2.0, 3.0], "newline": [4.0], "fence": [5.0]}


def test_the_shortest_target_has_an_empty_body():
    assert split_target([1.0, 4.0, 5.0])["body"] == []
    with pytest.raises(ValueError):
        split_target([1.0, 2.0])


def test_the_parts_contributions_add_up_to_the_mean_per_token_loss():
    pairs = [[0.1, 0.2, 0.3, 0.0, 12.0], [0.1, 0.0, 12.0]]
    summary = loss_by_part(pairs)
    mean_loss = (sum(pairs[0]) / 5 + sum(pairs[1]) / 3) / 2
    assert summary["mean_loss"] == pytest.approx(mean_loss)
    assert sum(summary["contribution"].values()) == pytest.approx(mean_loss)
    assert summary["nats_per_proof"]["fence"] == pytest.approx(12.0)
    assert summary["body_per_token"] == pytest.approx(0.25)
    assert summary["mean_tokens"] == pytest.approx(4.0)


def test_a_fixed_cost_per_proof_dominates_the_mean_of_short_targets():
    # the same 12-nat fence and a free body: the mean per-token loss is all fence, and larger for the short pair
    summary = loss_by_part([[0.0, 0.0, 12.0], [0.0] * 10 + [0.0, 12.0]])
    assert summary["contribution"]["fence"] == pytest.approx((12 / 3 + 12 / 12) / 2)
    assert summary["contribution"]["body"] == 0.0


def test_per_proof_constant_recovers_a_planted_fixed_cost_and_rate():
    tokens = [4, 8, 16, 40, 100]
    losses = [12.0 / t + 0.15 for t in tokens]
    a, b = per_proof_constant(losses, tokens)
    assert a == pytest.approx(12.0) and b == pytest.approx(0.15)


def test_per_proof_constant_reads_zero_when_the_loss_is_a_pure_rate():
    a, b = per_proof_constant([0.3] * 4, [5, 10, 20, 40])
    assert a == pytest.approx(0.0, abs=1e-9) and b == pytest.approx(0.3)


def test_per_proof_constant_refuses_what_it_cannot_separate():
    with pytest.raises(ValueError):
        per_proof_constant([0.3, 0.4], [10, 10])
    with pytest.raises(ValueError):
        per_proof_constant([0.3], [10])
    with pytest.raises(ValueError):
        per_proof_constant([0.3, 0.4], [10, 0])


def test_native_format_puts_the_sequence_start_token_between_prompt_and_target():
    ids, mask = native_format([100, 7, 8, 9, 20, 21], [False, False, False, False, True, True], bos_token_id=100)
    assert ids == [100, 7, 8, 9, 100, 20, 21]
    assert mask == [False, False, False, False, False, True, True]           # the inserted token is context, not a target
    assert [token for token, keep in zip(ids, mask) if keep] == [20, 21]


def test_native_format_refuses_a_pair_without_a_target():
    with pytest.raises(ValueError):
        native_format([100, 7], [False, False], bos_token_id=100)


def test_longest_other_proof_is_the_longest_with_a_different_text():
    completions = {"s#0": "  simp\n", "s#1": "  simp\n", "s#2": "  intro x\n  simp\n", "s#3": "  intro y\n  simp\n"}
    attempts = [("s#0", 4), ("s#1", 4), ("s#3", 9), ("s#2", 9)]
    assert longest_other_proof(attempts, completions, "s#0") == "s#2"          # tie on length: the smaller id


def test_longest_other_proof_is_none_when_every_proof_has_the_canonical_text():
    completions = {"s#0": "  simp\n", "s#1": "  simp \n"}
    assert longest_other_proof([("s#0", 4), ("s#1", 5)], completions, "s#0") is None
