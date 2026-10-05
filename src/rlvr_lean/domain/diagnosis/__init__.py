"""Pure reads for diagnosing a per-token loss and a gradient score.

Ledger: the held-out saturation and learning-progress score diagnosis. The two reads that became part of the
pipeline live where they are used now: the loss by part of the target in `domain.evaluation.loss_parts`, and
the model's native pair format in `domain.training.target_format` (spec §6 item 2a). They are re-exported
here for the diagnostic steps.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from rlvr_lean.domain.evaluation.loss_parts import PARTS, loss_by_part, split_target
from rlvr_lean.domain.training.target_format import native_format

__all__ = ["PARTS", "longest_other_proof", "loss_by_part", "native_format", "per_proof_constant", "split_target"]


def per_proof_constant(mean_losses: Sequence[float], token_counts: Sequence[int]) -> tuple[float, float]:
    """Least-squares (a, b) of loss_i = a / T_i + b: `a` nats paid once per proof, `b` nats per token.

    A mean per-token loss whose fall is all in `a` moved a fixed number of positions, not the proofs' content.
    """
    if len(mean_losses) != len(token_counts) or len(mean_losses) < 2:
        raise ValueError("need the same number (at least two) of losses and token counts")
    if any(count <= 0 for count in token_counts):
        raise ValueError("token counts must be positive")
    inverse = 1.0 / np.asarray(token_counts, dtype=float)
    if np.ptp(inverse) == 0:
        raise ValueError("every pair has the same length: the two terms cannot be separated")
    design = np.stack([inverse, np.ones(len(inverse))], axis=1)
    (a, b), *_ = np.linalg.lstsq(design, np.asarray(mean_losses, dtype=float), rcond=None)
    return float(a), float(b)


def longest_other_proof(verified_attempts: Sequence[tuple[str, int]], completions: dict[str, str], canonical_id: str) -> str | None:
    """Attempt id of the LONGEST verified proof whose text differs from the canonical (shortest) one, or None
    when every verified attempt has the canonical text. Ties go to the smallest attempt id."""
    canonical_text = completions[canonical_id].strip()
    others = [(attempt_id, tokens) for attempt_id, tokens in verified_attempts if completions[attempt_id].strip() != canonical_text]
    if not others:
        return None
    longest_id, _ = min(others, key=lambda attempt: (-attempt[1], attempt[0]))
    return longest_id
