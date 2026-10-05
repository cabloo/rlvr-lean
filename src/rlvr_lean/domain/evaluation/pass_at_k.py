"""pass@k by the unbiased estimator of Chen et al. (2021), §2.1. Spec §8, item 1.

For one problem with `n` samples drawn, of which `c` verified, pass@k estimates the probability that at
least one of `k` samples drawn WITHOUT replacement from those `n` is correct:

    pass@k = 1 - C(n - c, k) / C(n, k)

`C(n - c, k) / C(n, k)` is the probability that all `k` picks land among the `n - c` failures. Unlike
`1 - (1 - c/n)^k` (which samples with replacement), it is unbiased for the true pass@k of the model.
"""

from __future__ import annotations

import operator
from typing import Sequence

import numpy as np


def _validated_counts(sample_count: int, correct_count: int, k: int) -> tuple[int, int, int]:
    """The three counts as ints, or ValueError when they do not describe a valid draw."""
    sample_count, correct_count, k = operator.index(sample_count), operator.index(correct_count), operator.index(k)
    if not 0 <= correct_count <= sample_count:
        raise ValueError(f"need 0 <= correct_count <= sample_count, got c={correct_count}, n={sample_count}")
    if not 1 <= k <= sample_count:
        raise ValueError(f"need 1 <= k <= sample_count, got k={k}, n={sample_count}")
    return sample_count, correct_count, k


def pass_at_k(sample_count: int, correct_count: int, k: int) -> float:
    """Unbiased pass@k for one problem: `sample_count` samples drawn, `correct_count` of them verified.

    The ratio of binomials overflows for realistic n, so it is computed as a product (Chen et al.'s
    numerically stable form). Cancelling the factorials,

        C(n - c, k) / C(n, k) = [(n - c)! (n - k)!] / [n! (n - c - k)!]
                              = prod_{i = n - c + 1}^{n} (i - k) / i
                              = prod_{i = n - c + 1}^{n} (1 - k / i).

    When fewer than `k` samples failed (n - c < k), every draw of `k` contains a success: pass@k = 1.
    With c = 0 the product is empty (= 1) and pass@k = 0.
    """
    sample_count, correct_count, k = _validated_counts(sample_count, correct_count, k)
    failure_count = sample_count - correct_count
    if failure_count < k:
        return 1.0
    all_picks_fail = np.prod(1.0 - k / np.arange(failure_count + 1, sample_count + 1))
    return float(1.0 - all_picks_fail)


def mean_pass_at_k(counts: Sequence[tuple[int, int]], k: int) -> float:
    """Mean over problems of pass@k, each problem given as (sample_count, correct_count)."""
    if len(counts) == 0:
        raise ValueError("mean_pass_at_k needs at least one problem")
    return float(np.mean([pass_at_k(sample_count, correct_count, k) for sample_count, correct_count in counts]))
