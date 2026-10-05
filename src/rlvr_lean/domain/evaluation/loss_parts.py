"""The held-out loss, by part of the target. Spec §6 item 2a.

A training target is the proof followed by a newline and the closing code fence, so every target has the
same four parts: its first token (the indentation after `:= by`), the body, the newline before the fence,
and the fence. A MEAN per-token loss weights a cost paid once per proof by 1/T, so a fixed cost hides in it
as a 1/T term: in Phase A and B one position (the first token, 11.96 nats under the base) was 97% to 101% of
every adapter's held-out loss fall. Reporting the loss by part keeps a constant cost per proof from passing
for learning.
"""

from __future__ import annotations

from typing import Sequence

PARTS = ("first", "body", "newline", "fence")


def split_target(values: Sequence[float]) -> dict[str, list[float]]:
    """The per-position values of one target, by part. A target has at least three tokens (first, newline,
    fence); the body may be empty."""
    if len(values) < 3:
        raise ValueError(f"a target has at least three tokens (indentation, newline, fence), got {len(values)}")
    return {"first": [values[0]], "body": list(values[1:-2]), "newline": [values[-2]], "fence": [values[-1]]}


def pair_loss_row(statement_id: str, losses: Sequence[float]) -> dict:
    """One held-out pair's stored row: its mean per-token loss (`loss`, the figure Phase A and B stored), its
    target's token count, and each part's summed loss in nats."""
    parts = split_target(losses)
    return {"statement_id": statement_id, "loss": sum(losses) / len(losses), "target_tokens": len(losses),
            **{part: sum(parts[part]) for part in PARTS}}


def summarize_loss_rows(rows: Sequence[dict]) -> dict:
    """How a mean per-token loss over pairs divides between the four parts of a target.

    `mean_loss` is the measure the pipeline reports: the mean over pairs of each pair's mean per-token loss.
    `contribution[part]` is that part's share of it in loss units (the four add up to `mean_loss`).
    `nats_per_proof[part]` is the part's mean total cost per proof, and `body_per_token` the body's cost per
    body token pooled over all pairs: the two numbers a 1/T-weighted mean cannot tell apart.
    """
    if len(rows) == 0:
        raise ValueError("no pairs to summarise")
    pairs = len(rows)
    body_tokens = sum(row["target_tokens"] - 3 for row in rows)
    return {
        "pairs": pairs,
        "mean_tokens": sum(row["target_tokens"] for row in rows) / pairs,
        "mean_loss": sum(row["loss"] for row in rows) / pairs,
        "contribution": {part: sum(row[part] / row["target_tokens"] for row in rows) / pairs for part in PARTS},
        "nats_per_proof": {part: sum(row[part] for row in rows) / pairs for part in PARTS},
        "body_per_token": sum(row["body"] for row in rows) / body_tokens if body_tokens else 0.0,
    }


def loss_by_part(per_pair_losses: Sequence[Sequence[float]]) -> dict:
    """`summarize_loss_rows` of per-position losses, one sequence per pair."""
    if len(per_pair_losses) == 0:
        raise ValueError("no pairs to summarise")
    return summarize_loss_rows([pair_loss_row(str(index), losses) for index, losses in enumerate(per_pair_losses)])


def rows_have_parts(rows: Sequence[dict]) -> bool:
    """Phase A's and B's stored rows hold `loss` only; rows written since hold every part."""
    return len(rows) > 0 and all("target_tokens" in row and all(part in row for part in PARTS) for row in rows)
