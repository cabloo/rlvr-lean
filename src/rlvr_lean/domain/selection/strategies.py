"""The selection methods of spec §5 and §13, behind the `SelectionMethod` strategy interface of §7.

Every method returns exactly min(N, pool size) distinct conjectures, and the same list for the same pool
and seed regardless of the order the pool was given in.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from fractions import Fraction
from typing import ClassVar, Sequence

import numpy as np

from rlvr_lean.domain.selection.pool import PoolConjecture, in_difficulty_band


class SelectionMethod(abc.ABC):
    """Picks the training set: `training_set_size` conjectures from `pool`."""

    name: str

    @abc.abstractmethod
    def select(self, pool: Sequence[PoolConjecture], training_set_size: int, seed: int) -> list[PoolConjecture]:
        ...


def _selection_count(pool: Sequence[PoolConjecture], training_set_size: int) -> int:
    """How many to select, after checking the inputs every method relies on."""
    if training_set_size < 0:
        raise ValueError(f"training_set_size must be non-negative, got {training_set_size}")
    conjecture_ids = [conjecture.conjecture_id for conjecture in pool]
    if len(set(conjecture_ids)) != len(conjecture_ids):
        raise ValueError("the pool contains a conjecture id more than once")
    return min(training_set_size, len(pool))


def _sorted_by_id(pool: Sequence[PoolConjecture]) -> list[PoolConjecture]:
    """A canonical order, so seeded draws do not depend on the order the caller listed the pool in."""
    return sorted(pool, key=lambda conjecture: conjecture.conjecture_id)


@dataclass(frozen=True)
class RandomSelection(SelectionMethod):
    """Seeded shuffle of the pool; first N."""

    name: ClassVar[str] = "random"

    def select(self, pool: Sequence[PoolConjecture], training_set_size: int, seed: int) -> list[PoolConjecture]:
        count = _selection_count(pool, training_set_size)
        canonical_pool = _sorted_by_id(pool)
        permutation = np.random.default_rng(seed).permutation(len(canonical_pool))
        return [canonical_pool[index] for index in permutation[:count]]


@dataclass(frozen=True)
class DifficultyHeuristicSelection(SelectionMethod):
    """STP's "barely provable" rule: band members by ascending pass rate, seeded random tie-breaks.

    If the band holds fewer than N, the rest is filled upward by ascending pass rate (the same seeded
    tie-breaks). The pool holds no pass rate 0, so a fill-in is always above the band, never unsolved.
    """

    band_upper: float = 0.25
    name: ClassVar[str] = "difficulty_heuristic"

    def __post_init__(self) -> None:
        if not 0.0 < self.band_upper <= 1.0:
            raise ValueError(f"band_upper must lie in (0, 1], got {self.band_upper}")

    def select(self, pool: Sequence[PoolConjecture], training_set_size: int, seed: int) -> list[PoolConjecture]:
        count = _selection_count(pool, training_set_size)
        tie_break_rank = _seeded_tie_break_ranks(pool, seed)

        def ascending_pass_rate(conjecture: PoolConjecture) -> tuple[float, int]:
            return conjecture.pass_rate, tie_break_rank[conjecture.conjecture_id]

        band = [conjecture for conjecture in pool if in_difficulty_band(conjecture, self.band_upper)]
        band_by_pass_rate = sorted(band, key=ascending_pass_rate)
        if len(band_by_pass_rate) >= count:
            return band_by_pass_rate[:count]
        above_band = [conjecture for conjecture in pool if not in_difficulty_band(conjecture, self.band_upper)]
        fill_ins = sorted(above_band, key=ascending_pass_rate)[:count - len(band_by_pass_rate)]
        return band_by_pass_rate + fill_ins


@dataclass(frozen=True)
class HalfPassRateSelection(SelectionMethod):
    """The N conjectures whose base pass rate is closest to 0.5 (spec §13): the selection analogue of a
    challenger rewarded for problems the solver passes half the time. Ties break by the same seeded ranks
    as the heuristic, so a pass rate below and one above 0.5 at the same distance are ordered by seed."""

    target: float = 0.5
    name: ClassVar[str] = "half_pass_rate"

    def select(self, pool: Sequence[PoolConjecture], training_set_size: int, seed: int) -> list[PoolConjecture]:
        count = _selection_count(pool, training_set_size)
        tie_break_rank = _seeded_tie_break_ranks(pool, seed)
        target = Fraction(self.target)

        def distance(conjecture: PoolConjecture) -> Fraction:
            # Exact: in floats |5/12 - 0.5| and |7/12 - 0.5| differ in the last bit, so the mirror-image pass
            # rates would never tie and every one below 0.5 would precede its twin above.
            return abs(Fraction(conjecture.verified_count, conjecture.sample_count) - target)

        ranked = sorted(pool, key=lambda conjecture: (distance(conjecture), tie_break_rank[conjecture.conjecture_id]))
        return ranked[:count]


def _seeded_tie_break_ranks(pool: Sequence[PoolConjecture], seed: int) -> dict[str, int]:
    """A seeded random rank per conjecture id, drawn over the id-sorted pool. A permutation has no ties of
    its own, so (pass rate, rank) orders the pool totally."""
    canonical_pool = _sorted_by_id(pool)
    permutation = np.random.default_rng(seed).permutation(len(canonical_pool))
    return {conjecture.conjecture_id: int(rank) for conjecture, rank in zip(canonical_pool, permutation)}


@dataclass(frozen=True)
class LearningProgressSelection(SelectionMethod):
    """Descending learning-progress score (the dot product, or with `use_cosine` the cosine); first N.

    Ties go to the smaller conjecture id. The seed is unused: the ranking is fully determined by the scores.
    """

    use_cosine: bool = False

    @property
    def name(self) -> str:
        return "learning_progress_cosine" if self.use_cosine else "learning_progress"

    def _score(self, conjecture: PoolConjecture) -> float:
        score = conjecture.learning_progress_cosine if self.use_cosine else conjecture.learning_progress
        if score is None:
            raise ValueError(f"{conjecture.conjecture_id} has no {self.name} score; score the whole pool first")
        return score

    def select(self, pool: Sequence[PoolConjecture], training_set_size: int, seed: int) -> list[PoolConjecture]:
        count = _selection_count(pool, training_set_size)
        scores = {conjecture.conjecture_id: self._score(conjecture) for conjecture in pool}
        ranked = sorted(pool, key=lambda conjecture: (-scores[conjecture.conjecture_id], conjecture.conjecture_id))
        return ranked[:count]


def selection_overlap(first: Sequence[PoolConjecture], second: Sequence[PoolConjecture]) -> float:
    """Jaccard overlap of two selections' conjecture ids: |A ∩ B| / |A ∪ B|. Two empty selections are
    identical, so their overlap is 1."""
    first_ids = {conjecture.conjecture_id for conjecture in first}
    second_ids = {conjecture.conjecture_id for conjecture in second}
    union = first_ids | second_ids
    if not union:
        return 1.0
    return len(first_ids & second_ids) / len(union)
