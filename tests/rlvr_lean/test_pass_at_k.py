"""Spec fixture 1 (the first experiment's spec): the unbiased pass@k estimator."""

import math

import pytest

from rlvr_lean.domain.evaluation import mean_pass_at_k, pass_at_k


@pytest.mark.parametrize("sample_count,correct_count,k,expected", [
    (32, 0, 32, 0.0),
    (32, 1, 32, 1.0),
    (32, 8, 1, 0.25),
    (4, 2, 2, 5 / 6),
])
def test_spec_fixture_values(sample_count, correct_count, k, expected):
    assert pass_at_k(sample_count, correct_count, k) == pytest.approx(expected, abs=1e-12)


def closed_form(sample_count, correct_count, k):
    # math.comb(a, b) is 0 when b > a, which is the "every draw contains a success" case.
    return 1 - math.comb(sample_count - correct_count, k) / math.comb(sample_count, k)


@pytest.mark.parametrize("sample_count", [*range(1, 21), 32, 64])
def test_equals_the_closed_form_on_a_grid(sample_count):
    for correct_count in range(sample_count + 1):
        for k in range(1, sample_count + 1):
            assert pass_at_k(sample_count, correct_count, k) == pytest.approx(
                closed_form(sample_count, correct_count, k), rel=1e-12, abs=1e-12)


def test_pass_at_1_is_the_verified_fraction():
    for correct_count in range(13):
        assert pass_at_k(12, correct_count, 1) == pytest.approx(correct_count / 12, abs=1e-15)


def test_large_counts_do_not_overflow():
    # C(2048, 1024) overflows a float; the product form does not.
    assert 0.0 < pass_at_k(2048, 1, 1024) <= 1.0
    assert pass_at_k(2048, 1, 1024) == pytest.approx(0.5)


@pytest.mark.parametrize("sample_count,correct_count,k", [(8, 9, 1), (8, -1, 1), (8, 2, 0), (8, 2, 9), (0, 0, 1)])
def test_invalid_counts_are_rejected(sample_count, correct_count, k):
    with pytest.raises(ValueError):
        pass_at_k(sample_count, correct_count, k)


def test_non_integer_counts_are_rejected():
    with pytest.raises(TypeError):
        pass_at_k(8.0, 2, 1)


def test_mean_over_problems_allows_different_sample_counts():
    assert mean_pass_at_k([(32, 0), (32, 8)], 1) == pytest.approx(0.125)
    assert mean_pass_at_k([(12, 3), (8, 8)], 1) == pytest.approx((0.25 + 1.0) / 2)
    with pytest.raises(ValueError):
        mean_pass_at_k([], 1)
    with pytest.raises(ValueError):
        mean_pass_at_k([(32, 1), (8, 1)], 32)   # k exceeds the second problem's samples
