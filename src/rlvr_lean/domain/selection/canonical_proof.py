"""The canonical proof of a conjecture. Spec §4 item 6: its shortest verified proof by token count
(ties: smallest attempt id). It is the one proof used for scoring and for training under every method,
which is what makes a conjecture picked by two methods contribute the identical training example."""

from __future__ import annotations

from typing import Sequence


def choose_canonical_proof(verified_attempts: Sequence[tuple[str, int]]) -> str:
    """Attempt id of the shortest of a conjecture's VERIFIED attempts, given as (attempt_id, token_count)."""
    if len(verified_attempts) == 0:
        raise ValueError("a conjecture with no verified attempt has no canonical proof")
    for attempt_id, token_count in verified_attempts:
        if not attempt_id:
            raise ValueError("attempt ids must be non-empty")
        if token_count < 1:
            raise ValueError(f"attempt {attempt_id}: a verified proof has at least one token, got {token_count}")
    shortest_id, _ = min(verified_attempts, key=lambda attempt: (attempt[1], attempt[0]))
    return shortest_id
