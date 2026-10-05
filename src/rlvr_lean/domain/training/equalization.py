"""Training-set equalization. Spec §6 item 1: every adapter trains on exactly N examples, one canonical
proof per selected conjecture, so a conjecture chosen by two methods contributes the identical example
to both. The methods then differ only in WHICH conjectures they picked, never in how many examples or
which proof of a shared conjecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rlvr_lean.domain.selection.pool import PoolConjecture


@dataclass(frozen=True)
class TrainingExample:
    conjecture_id: str
    proof_attempt_id: str   # the conjecture's canonical proof


def build_training_examples(selected: Sequence[PoolConjecture]) -> list[TrainingExample]:
    """One example per selected conjecture, in selection order: its canonical proof."""
    seen_ids: set[str] = set()
    for conjecture in selected:
        if conjecture.conjecture_id in seen_ids:
            raise ValueError(f"{conjecture.conjecture_id} is selected twice; a training set has one example per conjecture")
        seen_ids.add(conjecture.conjecture_id)
    return [TrainingExample(conjecture.conjecture_id, conjecture.canonical_proof_id) for conjecture in selected]
