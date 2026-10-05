"""The one-seed scout of spec §13a: does the pass@1 gain survive the format fix?

The branches were fixed before the run. `difference` is the workbook-holdout pass@1 of the native-format
adapter minus the base model's, paired over problems.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

ESCALATE = "ESCALATE"
TOKEN_HABIT = "TOKEN_HABIT"
VOID = "VOID"


@dataclass(frozen=True)
class ScoutBranch:
    name: str      # ESCALATE, TOKEN_HABIT or VOID
    reason: str    # which line of the pre-registered rule it met
    action: str    # what the rule says to do next


def scout_branch(difference: float, control_reproduced: bool, token_habit_below: float) -> ScoutBranch:
    """Spec §13a, in the order the rule is written. `token_habit_below` is the pre-registered -0.004: more
    than the largest seed-to-seed range any Phase B arm showed (0.0036)."""
    if token_habit_below >= 0:
        raise ValueError(f"the token-habit line lies below zero, got {token_habit_below}")
    if not control_reproduced:
        return ScoutBranch(VOID, "the legacy control does not reproduce", "fix and re-run one seed")
    if not math.isfinite(difference):
        raise ValueError(f"the difference must be finite, got {difference}")
    if difference >= 0:
        return ScoutBranch(ESCALATE, "difference >= 0: the gain survives or is undecided", "seeds 1 and 2; three seeds read it")
    if difference < token_habit_below:
        return ScoutBranch(TOKEN_HABIT, f"difference below {token_habit_below}: the legacy gain depended on the token habit",
                           "stop and diagnose what the round changes in the native format before any further training")
    return ScoutBranch(ESCALATE, f"difference between {token_habit_below} and 0: no measurable change at one seed",
                       "seeds 1 and 2, since Phase A's own claim was read at three")


def control_reproduces(measured: float, expected: float, decimals: int = 4) -> bool:
    """The stored control reads what the pre-registration quotes, to the decimals it quotes."""
    return round(measured, decimals) == round(expected, decimals)


def distinct_share(completions_by_statement: Mapping[str, Sequence[str]]) -> float:
    """Distinct proof texts as a share of attempts: per statement the number of different completions, summed
    over statements, over all attempts. 1.0 when no attempt repeats another on the same statement."""
    attempts = sum(len(completions) for completions in completions_by_statement.values())
    if attempts == 0:
        raise ValueError("no attempts")
    return sum(len(set(completions)) for completions in completions_by_statement.values()) / attempts


def share_starting_with(first_token_ids: Sequence[int | None], token_id: int) -> float:
    """The share of sampled attempts whose FIRST generated token is `token_id` (the sequence-start token)."""
    if len(first_token_ids) == 0:
        raise ValueError("no attempts")
    return sum(first == token_id for first in first_token_ids) / len(first_token_ids)
