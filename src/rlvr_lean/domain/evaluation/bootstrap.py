"""Paired bootstrap over problems, and the pre-registered labels read from its interval. Spec §8.

The input is one difference per problem (trained minus base, each the problem's pass@k under that
variant), so the pairing is already done: problems are resampled WITH replacement, the mean difference
is taken for each resample, and the percentile interval of those means is reported (10,000 resamples,
fixed seed, 95% by default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Resamples are drawn this many at a time, so memory stays bounded for large problem sets
# (10,000 resamples x 1,000 problems would otherwise be an 80 MB index array). Changing it changes
# which numbers a seed draws, so it is fixed.
_RESAMPLES_PER_CHUNK = 1_000


@dataclass(frozen=True)
class BootstrapInterval:
    mean: float   # the observed mean difference over problems
    low: float    # lower percentile bound of the resampled means
    high: float   # upper percentile bound of the resampled means


def _validated_differences(differences: Sequence[float]) -> np.ndarray:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"need a non-empty 1-D sequence of per-problem differences, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("per-problem differences must all be finite")
    return values


def _resampled_mean_deviations(deviations: np.ndarray, resamples: int, generator: np.random.Generator) -> np.ndarray:
    """Mean of each bootstrap resample of `deviations` (problems drawn with replacement)."""
    problem_count = deviations.size
    means = []
    for start in range(0, resamples, _RESAMPLES_PER_CHUNK):
        chunk = min(_RESAMPLES_PER_CHUNK, resamples - start)
        indices = generator.integers(0, problem_count, size=(chunk, problem_count))
        means.append(deviations[indices].mean(axis=1))
    return np.concatenate(means)


def paired_bootstrap(
    differences: Sequence[float],
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Percentile bootstrap interval for the mean of per-problem differences."""
    values = _validated_differences(differences)
    if resamples < 1:
        raise ValueError(f"resamples must be at least 1, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")

    # Means are taken of deviations from one observation and shifted back (the shifted-data form). A
    # constant input then has deviations of exactly zero, so it yields exactly [d, d] instead of d plus
    # rounding error, and large offsets do not cost precision in the sums.
    reference = values[0]
    deviations = values - reference
    generator = np.random.default_rng(seed)
    resampled_means = reference + _resampled_mean_deviations(deviations, resamples, generator)

    tail_probability = (1.0 - confidence) / 2.0
    low, high = np.quantile(resampled_means, [tail_probability, 1.0 - tail_probability])
    return BootstrapInterval(mean=float(reference + deviations.mean()), low=float(low), high=float(high))


def interval_label(interval: BootstrapInterval) -> str:
    """Phase A label (spec §8): the interval lies entirely above zero, entirely below, or contains it."""
    if interval.low > 0:
        return "improved"
    if interval.high < 0:
        return "degraded"
    return "no measurable change"


def phase_b_label(interval: BootstrapInterval) -> str:
    """Phase B label (spec §8): the same rule, named for the learning-progress-vs-heuristic comparison."""
    if interval.low > 0:
        return "promising"
    if interval.high < 0:
        return "negative"
    return "inconclusive"
