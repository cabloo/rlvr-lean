"""Selection context: the pool of solvable conjectures, the training-set size N, and the methods that pick
N of them (spec §5, §13)."""

from rlvr_lean.domain.selection.canonical_proof import choose_canonical_proof
from rlvr_lean.domain.selection.pool import PoolConjecture, TrainingSetSize, in_difficulty_band, training_set_size
from rlvr_lean.domain.selection.strategies import (
    DifficultyHeuristicSelection,
    HalfPassRateSelection,
    LearningProgressSelection,
    RandomSelection,
    SelectionMethod,
    selection_overlap,
)

__all__ = [
    "DifficultyHeuristicSelection",
    "HalfPassRateSelection",
    "LearningProgressSelection",
    "PoolConjecture",
    "RandomSelection",
    "SelectionMethod",
    "TrainingSetSize",
    "choose_canonical_proof",
    "in_difficulty_band",
    "selection_overlap",
    "training_set_size",
]
