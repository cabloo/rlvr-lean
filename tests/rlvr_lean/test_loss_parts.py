"""Spec §6 item 2a: the held-out loss is stored and reported by part of the target, so a cost paid once per
proof cannot pass for learning."""

import pytest

from rlvr_lean.domain.evaluation.loss_parts import PARTS, loss_by_part, pair_loss_row, rows_have_parts, summarize_loss_rows


def test_a_pair_s_row_holds_its_mean_loss_its_token_count_and_each_part_s_nats():
    row = pair_loss_row("s1", [12.0, 0.1, 0.3, 0.2, 0.4])
    assert row == {"statement_id": "s1", "loss": pytest.approx(2.6), "target_tokens": 5, "first": 12.0,
                   "body": pytest.approx(0.4), "newline": 0.2, "fence": 0.4}


def test_a_target_of_three_tokens_has_an_empty_body_and_a_shorter_one_is_refused():
    assert pair_loss_row("s", [1.0, 2.0, 3.0])["body"] == 0
    with pytest.raises(ValueError):
        pair_loss_row("s", [1.0, 2.0])


def test_the_summary_of_stored_rows_equals_the_summary_of_the_per_position_losses():
    pairs = [[12.0, 0.1, 0.3, 0.2, 0.4], [11.0, 0.0, 0.5], [9.0] + [0.25] * 20 + [0.1, 0.0]]
    from_rows = summarize_loss_rows([pair_loss_row(str(i), losses) for i, losses in enumerate(pairs)])
    direct = loss_by_part(pairs)
    assert from_rows["mean_loss"] == pytest.approx(direct["mean_loss"])
    assert from_rows["body_per_token"] == pytest.approx(direct["body_per_token"]) == pytest.approx((0.4 + 5.0) / 22)
    for part in PARTS:
        assert from_rows["nats_per_proof"][part] == pytest.approx(direct["nats_per_proof"][part])
    assert sum(from_rows["contribution"].values()) == pytest.approx(from_rows["mean_loss"])
    assert from_rows["mean_tokens"] == pytest.approx((5 + 3 + 23) / 3)


def test_a_fixed_cost_on_the_first_token_shows_as_first_token_nats_not_as_body_loss():
    with_cost = summarize_loss_rows([pair_loss_row("a", [12.0, 0.2, 0.2, 0.0, 0.0]), pair_loss_row("b", [12.0] + [0.2] * 18 + [0.0, 0.0])])
    without = summarize_loss_rows([pair_loss_row("a", [0.0, 0.2, 0.2, 0.0, 0.0]), pair_loss_row("b", [0.0] + [0.2] * 18 + [0.0, 0.0])])
    assert with_cost["mean_loss"] - without["mean_loss"] == pytest.approx((12 / 5 + 12 / 21) / 2)     # the 1/T-weighted constant
    assert with_cost["body_per_token"] == pytest.approx(without["body_per_token"]) == pytest.approx(0.2)
    assert (with_cost["nats_per_proof"]["first"], without["nats_per_proof"]["first"]) == (12.0, 0.0)


def test_rows_stored_before_the_fix_hold_no_parts():
    assert rows_have_parts([pair_loss_row("a", [1.0, 2.0, 3.0])])
    assert not rows_have_parts([{"statement_id": "a", "loss": 0.9}])
    assert not rows_have_parts([])
    with pytest.raises(ValueError):
        summarize_loss_rows([])
