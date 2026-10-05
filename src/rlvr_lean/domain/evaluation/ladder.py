"""The first rung of the 50% ladder (spec §13b): does training on the 50% band move a harder, untrained band up?

The sets, from the base model's STORED counts; the branches, fixed before the run; the census of false
conjectures. Everything here is arithmetic over counts and intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from rlvr_lean.domain.evaluation.bootstrap import BootstrapInterval

ESCALATE = "ESCALATE"
UNDETECTABLE = "UNDETECTABLE"
STOP = "STOP"
SECONDARY_ABOVE = "SECONDARY OUTSIDE LUCK (above)"
SECONDARY_BELOW = "SECONDARY OUTSIDE LUCK (below)"
VOID = "VOID"
BRANCHES = (ESCALATE, UNDETECTABLE, STOP, SECONDARY_ABOVE, SECONDARY_BELOW, VOID)
CENSUS_MARKER = "negation_census"      # the store marker of the census stage; its summary is what the report prints


def below_band_ids(verified_counts: Mapping[str, int], low: int, high: int) -> list[str]:
    """The statements the base proved between `low` and `high` times in its stored samples, sorted."""
    if not 1 <= low <= high:
        raise ValueError(f"need 1 <= low <= high, got {low} and {high}")
    return sorted(statement for statement, verified in verified_counts.items() if low <= verified <= high)


def never_proved_ids(verified_counts: Mapping[str, int]) -> list[str]:
    """The statements the base did not prove once in its stored samples, sorted."""
    return sorted(statement for statement, verified in verified_counts.items() if verified == 0)


def trained_on(candidates: Sequence[str], training_ids: Sequence[str]) -> list[str]:
    """The candidates that are in the training set. The primary's set must give an empty list."""
    training = set(training_ids)
    return sorted(candidate for candidate in candidates if candidate in training)


@dataclass(frozen=True)
class LadderBranch:
    name: str
    reason: str
    action: str


def ladder_branch(primary: BootstrapInterval, base_fresh_rate: float, void_range: tuple[float, float],
                  secondaries: Mapping[str, BootstrapInterval]) -> LadderBranch:
    """Spec §13b, in the order the rule is written. `primary` is adapter minus base on the below-band
    conjectures' pass rate; each secondary is adapter minus base on "proved at least once"; `base_fresh_rate`
    is the base's fresh mean pass rate on the below-band set, which must lie in `void_range` (what identical
    conditions predict once regression to the mean is allowed for).

    When the primary's interval contains zero and two secondaries disagree in direction, the one BELOW zero
    decides: stop and diagnose."""
    low, high = void_range
    if not low < high:
        raise ValueError(f"the VOID range must be an interval, got {void_range}")
    if not low <= base_fresh_rate <= high:
        return LadderBranch(VOID, f"the base's fresh mean pass rate on the below-band set is {base_fresh_rate:.4f}, outside [{low}, {high}]",
                            "fix and re-run: the fresh samples do not match the stored run's conditions")
    if primary.low > 0:
        return LadderBranch(ESCALATE, "the primary's interval lies entirely above zero: learning builds on this rung",
                            "seeds 1 and 2, and plan the second rung (the band re-measured with the new model)")
    if primary.high < 0:
        return LadderBranch(STOP, "the primary's interval lies entirely below zero: training on the band made harder conjectures worse",
                            "stop and diagnose")
    below = [name for name, interval in secondaries.items() if interval.high < 0]
    above = [name for name, interval in secondaries.items() if interval.low > 0]
    if below:
        return LadderBranch(SECONDARY_BELOW, f"the primary's interval contains zero; {', '.join(below)} lies entirely below zero", "stop and diagnose")
    if above:
        return LadderBranch(SECONDARY_ABOVE, f"the primary's interval contains zero; {', '.join(above)} lies entirely above zero",
                            f"ESCALATE to seeds 1 and 2 with {', '.join(above)} as the measure to confirm (exploratory: the primary did not decide)")
    return LadderBranch(UNDETECTABLE, "the primary's interval contains zero and the secondaries are inside the base's own luck",
                        "more below-band conjectures, from the 4,609 never sampled; not more seeds")


def census_summary(statements: Sequence[Mapping], verified_by_conjecture: Mapping[str, int]) -> dict:
    """The training-data census of spec §13b. `statements` holds one row per never-proved conjecture
    (`conjecture_id`, `built`, `compiles`, `exact`); `verified_by_conjecture` the number of verified proofs of
    its negation. A conjecture is DISPROVED (false as stated) when its negation has a verified proof. An
    inexact negation (the statement relies on a variable it never binds) is the stronger claim: a proof of it
    still disproves the conjecture, but its absence says less, so those are counted apart."""
    total = len(statements)
    if total == 0:
        raise ValueError("no never-proved conjecture to summarise")
    sampled = [row for row in statements if row["built"] and row["compiles"]]
    disproved = [row for row in sampled if verified_by_conjecture.get(row["conjecture_id"], 0) > 0]
    exact = [row for row in sampled if row["exact"]]
    return {"never_proved": total, "negation_not_built": sum(not row["built"] for row in statements),
            "negation_does_not_compile": sum(row["built"] and not row["compiles"] for row in statements),
            "sampled": len(sampled), "negation_exact": len(exact), "negation_inexact": len(sampled) - len(exact),
            "disproved": len(disproved), "disproved_among_exact": sum(row["exact"] for row in disproved),
            "disproved_share_of_never_proved": len(disproved) / total,
            "undetermined": total - len(disproved), "undetermined_share_of_never_proved": (total - len(disproved)) / total}
