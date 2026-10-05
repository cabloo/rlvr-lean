"""Spec fixture 2 and the decision labels of §8: the paired bootstrap over problems, and rank correlation."""

import math

import numpy as np
import pytest

from rlvr_lean.domain.evaluation import BootstrapInterval, interval_label, paired_bootstrap, phase_b_label, spearman


def test_identical_inputs_give_a_zero_width_interval_at_zero():
    trained = base = [0.25, 0.0, 1.0, 0.5, 0.75]
    interval = paired_bootstrap([t - b for t, b in zip(trained, base)])
    assert interval == BootstrapInterval(mean=0.0, low=0.0, high=0.0)
    assert interval_label(interval) == "no measurable change"


@pytest.mark.parametrize("difference", [0.1, -0.25, 1 / 3, 7.0])
def test_constant_difference_gives_exactly_d_d(difference):
    interval = paired_bootstrap([difference] * 37)
    assert interval == BootstrapInterval(mean=difference, low=difference, high=difference)


def test_fixed_seed_reproduces_the_interval_exactly():
    differences = np.random.default_rng(5).normal(0.02, 0.3, size=244).tolist()
    assert paired_bootstrap(differences, seed=11) == paired_bootstrap(differences, seed=11)
    assert paired_bootstrap(differences, seed=11) != paired_bootstrap(differences, seed=12)


def test_interval_matches_the_normal_approximation_for_a_simple_case():
    # 100 problems, half +1 and half 0: mean 0.5, standard error 0.05, so the 95% interval is ~0.5 ± 0.098.
    interval = paired_bootstrap([1.0, 0.0] * 50)
    assert interval.mean == pytest.approx(0.5)
    assert interval.low == pytest.approx(0.5 - 1.96 * 0.05, abs=0.02)
    assert interval.high == pytest.approx(0.5 + 1.96 * 0.05, abs=0.02)
    assert interval_label(interval) == "improved" and phase_b_label(interval) == "promising"


def test_resample_counts_that_are_not_a_whole_chunk():
    for resamples in (1, 999, 1001, 2500):
        interval = paired_bootstrap([0.0, 1.0, 0.5], resamples=resamples)
        assert 0.0 <= interval.low <= interval.high <= 1.0


@pytest.mark.parametrize("low,high,phase_a,phase_b", [
    (0.01, 0.2, "improved", "promising"),
    (-0.2, -0.01, "degraded", "negative"),
    (-0.1, 0.1, "no measurable change", "inconclusive"),
    (0.0, 0.1, "no measurable change", "inconclusive"),   # touching zero is not "entirely above"
    (-0.1, 0.0, "no measurable change", "inconclusive"),
])
def test_labels(low, high, phase_a, phase_b):
    interval = BootstrapInterval(mean=(low + high) / 2, low=low, high=high)
    assert interval_label(interval) == phase_a
    assert phase_b_label(interval) == phase_b


@pytest.mark.parametrize("arguments", [
    {"differences": []},
    {"differences": [0.1, math.nan]},
    {"differences": [0.1], "resamples": 0},
    {"differences": [0.1], "confidence": 1.0},
])
def test_invalid_bootstrap_inputs_are_rejected(arguments):
    with pytest.raises(ValueError):
        paired_bootstrap(**arguments)


def test_spearman_with_ties_and_degenerate_inputs():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # Average ranks: [1, 2.5, 2.5, 4] against [1, 2, 3, 4] gives 4.5 / sqrt(4.5 * 5) = 0.9486832980505138.
    assert spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486832980505138)
    assert math.isnan(spearman([1.0], [2.0]))
    assert math.isnan(spearman([], []))
    assert math.isnan(spearman([5, 5, 5], [1, 2, 3]))
    with pytest.raises(ValueError):
        spearman([1, 2], [1, 2, 3])


def test_spearman_is_invariant_to_monotone_transforms():
    values = np.random.default_rng(0).normal(size=50)
    other = values + np.random.default_rng(1).normal(scale=0.5, size=50)
    assert spearman(values, other) == pytest.approx(spearman(np.exp(values), other ** 3))
