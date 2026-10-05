"""The selection pool and the training-set size N. Spec §4 items 5-6 and §5.

Pool = the SOLVABLE conjectures (at least one verified attempt) outside the conjecture holdout. Each
carries its pass rate, its canonical proof, and (once scored) its learning-progress scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class PoolConjecture:
    conjecture_id: str
    sample_count: int             # proof attempts drawn for this conjecture (12 in the full run)
    verified_count: int           # of those, how many were `verified`
    canonical_proof_id: str       # attempt id of the shortest verified proof (choose_canonical_proof)
    canonical_proof_tokens: int   # its length in tokens
    learning_progress: float | None = None          # <g(c), g_mean>, the primary score
    learning_progress_cosine: float | None = None   # the same with both gradients normalised

    def __post_init__(self) -> None:
        if not self.conjecture_id:
            raise ValueError("conjecture_id must be non-empty")
        if not 1 <= self.verified_count <= self.sample_count:
            raise ValueError(
                f"{self.conjecture_id}: the pool holds solvable conjectures only, so need "
                f"1 <= verified_count <= sample_count, got {self.verified_count} of {self.sample_count}"
            )
        if not self.canonical_proof_id:
            raise ValueError(f"{self.conjecture_id}: canonical_proof_id must be non-empty")
        if self.canonical_proof_tokens < 1:
            raise ValueError(f"{self.conjecture_id}: a verified proof has at least one token, got {self.canonical_proof_tokens}")
        for score_name in ("learning_progress", "learning_progress_cosine"):
            score = getattr(self, score_name)
            if score is not None and not math.isfinite(score):
                raise ValueError(f"{self.conjecture_id}: {score_name} must be finite when present, got {score}")

    @property
    def pass_rate(self) -> float:
        """Verified attempts / samples drawn (spec §4 item 5)."""
        return self.verified_count / self.sample_count


@dataclass(frozen=True)
class TrainingSetSize:
    size: int                       # N, the number of examples every method trains on
    heuristic_fills_upward: bool    # True when the band was too small to bound N (spec §5)


def training_set_size(
    pool_size: int,
    band_size: int,
    maximum: int = 500,
    pool_fraction: float = 1 / 3,
    band_minimum: int = 150,
) -> TrainingSetSize:
    """N = min(maximum, floor(pool_size * pool_fraction), band_size), spec §5.

    The `pool_fraction` cap keeps N well below the pool size, or the three methods of Phase B would pick
    nearly the same set. If the band holds fewer than `band_minimum` conjectures it does not bound N; then
    N = min(maximum, floor(pool_size * pool_fraction)) and the heuristic may fill upward past the band
    (`heuristic_fills_upward`). The flag says the fill rule is in force; the heuristic adds fill-ins only
    if the band still holds fewer than N members. N may be 0 for a tiny pool.
    """
    if pool_size < 0 or not 0 <= band_size <= pool_size:
        raise ValueError(f"need 0 <= band_size <= pool_size, got band {band_size}, pool {pool_size}")
    if maximum < 0 or band_minimum < 0:
        raise ValueError(f"maximum and band_minimum must be non-negative, got {maximum} and {band_minimum}")
    if not 0.0 < pool_fraction <= 1.0:
        raise ValueError(f"pool_fraction must lie in (0, 1], got {pool_fraction}")

    # Floor of an exact rational: in floats, 0.29 * 100 is 28.999999999999996, which floors to 28.
    pool_share = math.floor(pool_size * Fraction(pool_fraction).limit_denominator(1_000_000))
    if band_size >= band_minimum:
        return TrainingSetSize(size=min(maximum, pool_share, band_size), heuristic_fills_upward=False)
    return TrainingSetSize(size=min(maximum, pool_share), heuristic_fills_upward=True)


def in_difficulty_band(conjecture: PoolConjecture, band_upper: float = 0.25) -> bool:
    """STP's "barely provable" band: pass rate in (0, band_upper], i.e. 1, 2 or 3 of 12 at the default."""
    return 0.0 < conjecture.pass_rate <= band_upper
