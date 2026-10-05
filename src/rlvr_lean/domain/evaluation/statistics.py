"""Rank correlation, for the score diagnostics of spec §5 (stability under a second `A` seed, the
cosine and band-only variants against the primary score, and the length-bias check)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks, with tied values all given the mean of the ranks they span."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts_tie_group = np.concatenate(([True], sorted_values[1:] != sorted_values[:-1]))
    tie_group_of_position = np.cumsum(starts_tie_group) - 1
    group_starts = np.flatnonzero(starts_tie_group)
    group_ends = np.concatenate((group_starts[1:], [values.size]))   # exclusive
    # Positions start..end-1 hold ranks start+1..end, whose mean is (start + 1 + end) / 2.
    group_average_rank = (group_starts + 1 + group_ends) / 2.0
    ranks = np.empty(values.size)
    ranks[order] = group_average_rank[tie_group_of_position]
    return ranks


def spearman(first: Sequence[float], second: Sequence[float]) -> float:
    """Spearman's rank correlation: the Pearson correlation of the two average-rank vectors.

    Returns nan when it is undefined: fewer than two items, or either side constant (zero rank variance).
    """
    first_values, second_values = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first_values.ndim != 1 or first_values.shape != second_values.shape:
        raise ValueError(f"need two 1-D sequences of equal length, got shapes {first_values.shape} and {second_values.shape}")
    if not (np.all(np.isfinite(first_values)) and np.all(np.isfinite(second_values))):
        raise ValueError("spearman needs finite values")
    if first_values.size < 2:
        return math.nan

    first_deviation = _average_ranks(first_values) - (first_values.size + 1) / 2.0
    second_deviation = _average_ranks(second_values) - (second_values.size + 1) / 2.0
    first_spread, second_spread = float(first_deviation @ first_deviation), float(second_deviation @ second_deviation)
    if first_spread == 0.0 or second_spread == 0.0:
        return math.nan
    return float(first_deviation @ second_deviation) / math.sqrt(first_spread * second_spread)
