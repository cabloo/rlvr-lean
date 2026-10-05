"""The version tax: what a prover loses because the checker moved to another Lean pin, measured on IDENTICAL
proof texts checked under both pins. Spec: the OEIS Open spec, O2 items 6-9 and O2a item 9e.

Because the texts are the same, the difference between the two pass rates is the tax and nothing else. That
only holds if the two pins judged the same attempts, so `build_version_tax` refuses anything else (fixture
3), and an attempt one pin gave NO ANSWER for (a server or transport failure: not a Lean verdict) is taken
out of both pins' counts and reported, never counted as a failure.

Pure: statuses and messages in, numbers out. Intervals are percentile bootstraps over STATEMENTS (an
attempt's fate is tied to its statement's, so attempts are not independent), paired where two pins are
compared.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

VERIFIED = "verified"
NO_ANSWER_STATUS = "server_error"          # `VerificationStatus.SERVER_ERROR`: the server could not answer

# The classes the later pin's failures are split into (item 9e).
UNKNOWN_IDENTIFIER, OTHER_ERROR, TIMEOUT = "unknown_identifier", "other_error", "timeout"
REJECTED_LEXICAL = "rejected_lexical"      # never sent to Lean, so the same under every pin
FAILURE_CLASSES = (UNKNOWN_IDENTIFIER, OTHER_ERROR, TIMEOUT, REJECTED_LEXICAL)

SMALL_TAX_RATIO = 0.8                      # pre-registered (O2): small if the later rate is within 20% (relative)

# A name the pin's Lean or Mathlib does not have: a renamed or removed lemma, constant or namespace, also
# when it is reached as a field (`h.le_div_iff`). Lean v4.27 words these "Unknown identifier `x`", "Unknown
# constant `X.y`" and "Invalid field `y`: The environment does not contain `X.y`"; v4.9 wrote the first two
# in lower case with straight quotes. Read off the stored attempts' real messages (the result file quotes
# some). NOT in this class: "unknown tactic", which is the parser's message for any text it cannot read as a
# tactic, and a syntax that was removed (`∑ i in s`): both are other errors.
_UNKNOWN_NAME = re.compile(r"\bunknown (?:identifier|constant|namespace)\b|the environment does not contain", re.IGNORECASE)


def names_something_unknown(error_message: str) -> bool:
    return _UNKNOWN_NAME.search(error_message) is not None


def is_echo_of_a_failed_declaration(error_message: str, theorem_name: str) -> bool:
    """True for "Unknown constant `<the attempt's own theorem>`": the trailing `#print axioms` not finding a
    theorem whose declaration failed above it. It is an echo of the real error, never a missing library name,
    so a caller drops it before classing the attempt."""
    return re.search(rf"\bunknown (?:identifier|constant)\s+([`']){re.escape(theorem_name)}\1",
                     error_message, re.IGNORECASE) is not None


def failure_class(status: str, error_messages: Sequence[str]) -> str | None:
    """The class of one attempt's outcome under a pin; None when it verified or got no answer.

    A Lean error is an unknown identifier if ANY of its error messages names something unknown: one missing
    lemma usually drags other errors behind it (unsolved goals, a failed `linarith`), and the missing name
    is the one the version explains."""
    if status in (VERIFIED, NO_ANSWER_STATUS):
        return None
    if status in (TIMEOUT, REJECTED_LEXICAL):
        return status
    if any(names_something_unknown(message) for message in error_messages):
        return UNKNOWN_IDENTIFIER
    return OTHER_ERROR                     # a Lean error, or a proof Lean accepted only with `sorry` or an axiom


def error_kind(error_message: str) -> str:
    """An error message's first line with every quoted name removed, so messages that differ only in the
    names they mention count as one kind (the report's finer breakdown of the three classes)."""
    first_line = error_message.strip().split("\n", 1)[0]
    return re.sub(r"`[^`]*`|'[^']*'|«[^»]*»", "_", first_line)[:80].rstrip(": ")


@dataclass(frozen=True)
class Outcome:
    """One attempt's outcome under one pin."""
    status: str
    error_messages: tuple[str, ...] = ()
    seconds: float | None = None

    @property
    def verified(self) -> bool:
        return self.status == VERIFIED

    @property
    def answered(self) -> bool:
        return self.status != NO_ANSWER_STATUS


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """The 95% Wilson score interval of a proportion (independent trials: statements, not attempts)."""
    if total <= 0:
        raise ValueError("a proportion needs at least one trial")
    if not 0 <= successes <= total:
        raise ValueError(f"need 0 <= successes <= total, got {successes} of {total}")
    share = successes / total
    centre = (share + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - half), min(1.0, centre + half)


def _percentiles(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def _figure(value: float | None, samples: np.ndarray) -> dict:
    low, high = _percentiles(samples)
    return {"value": value, "low": low, "high": high}


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _compare(drawn: np.ndarray, earlier: np.ndarray, later: np.ndarray, resamples: int, seed: int) -> dict:
    """Both pins' rates over one set of statements (per attempt, and per statement proved at least once), the
    later rate as a share of the earlier one, and their difference, with 95% intervals from resampling the
    STATEMENTS with replacement. One row per statement: attempts drawn, verified earlier, verified later."""
    statements = drawn.size
    if statements == 0:
        return {"statements": 0, "attempts": 0}
    generator = np.random.default_rng(seed)
    sums = {name: [] for name in ("drawn", "earlier", "later", "earlier_solved", "later_solved")}
    columns = {"drawn": drawn.astype(float), "earlier": earlier.astype(float), "later": later.astype(float),
               "earlier_solved": (earlier > 0).astype(float), "later_solved": (later > 0).astype(float)}
    for start in range(0, resamples, 500):                      # in chunks, so memory stays bounded
        indices = generator.integers(0, statements, size=(min(500, resamples - start), statements))
        for name, column in columns.items():
            sums[name].append(column[indices].sum(axis=1))
    resampled = {name: np.concatenate(parts) for name, parts in sums.items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        per_attempt = {
            "earlier": _figure(float(earlier.sum() / drawn.sum()), resampled["earlier"] / resampled["drawn"]),
            "later": _figure(float(later.sum() / drawn.sum()), resampled["later"] / resampled["drawn"]),
            "later_as_share_of_earlier": _figure(_ratio(float(later.sum()), float(earlier.sum())),
                                                 resampled["later"] / resampled["earlier"]),
            "difference": _figure(float((later.sum() - earlier.sum()) / drawn.sum()),
                                  (resampled["later"] - resampled["earlier"]) / resampled["drawn"]),
        }
        solved_earlier, solved_later = int((earlier > 0).sum()), int((later > 0).sum())
        per_statement = {
            "earlier": _figure(solved_earlier / statements, resampled["earlier_solved"] / statements),
            "later": _figure(solved_later / statements, resampled["later_solved"] / statements),
            "later_as_share_of_earlier": _figure(_ratio(solved_later, solved_earlier),
                                                 resampled["later_solved"] / resampled["earlier_solved"]),
            "difference": _figure((solved_later - solved_earlier) / statements,
                                  (resampled["later_solved"] - resampled["earlier_solved"]) / statements),
        }
    return {"statements": statements, "attempts": int(drawn.sum()),
            "verified_earlier": int(earlier.sum()), "verified_later": int(later.sum()),
            "solved_earlier": solved_earlier, "solved_later": solved_later,
            "solved_at_both": int(((earlier > 0) & (later > 0)).sum()),
            "per_attempt": per_attempt, "per_statement": per_statement}


def band_of(pass_rate: float, edges: Sequence[float]) -> str:
    """The base pass-rate band of a statement: `0`, then `(a, b]` between consecutive edges up to 1."""
    if pass_rate <= 0:
        return "0"
    lower = 0.0
    for upper in [*edges, 1.0]:
        if pass_rate <= upper:
            return f"({lower:g}, {upper:g}]"
        lower = upper
    return f"({lower:g}, 1]"


def band_names(edges: Sequence[float]) -> list[str]:
    bounds = [0.0, *edges, 1.0]
    return ["0"] + [f"({lower:g}, {upper:g}]" for lower, upper in zip(bounds, bounds[1:])]


def _classes(outcomes: Iterable[Outcome]) -> dict[str, int]:
    counts = {name: 0 for name in FAILURE_CLASSES}
    for outcome in outcomes:
        name = failure_class(outcome.status, outcome.error_messages)
        if name is not None:
            counts[name] += 1
    return counts


def _section(rows: list[tuple[str, str, Outcome, Outcome]], resamples: int, seed: int) -> dict:
    """The comparison and the later pin's failure classes over `rows` of (attempt, statement, earlier, later)."""
    statements = sorted({statement for _, statement, _, _ in rows})
    position = {statement: index for index, statement in enumerate(statements)}
    drawn, earlier, later = (np.zeros(len(statements), dtype=np.int64) for _ in range(3))
    for _, statement, before, after in rows:
        drawn[position[statement]] += 1
        earlier[position[statement]] += before.verified
        later[position[statement]] += after.verified
    section = _compare(drawn, earlier, later, resamples, seed)
    section["later_failures"] = _classes(after for _, _, _, after in rows if not after.verified)
    # The tax itself: attempts the earlier pin verified and the later one does not, by the later pin's class.
    section["lost"] = _classes(after for _, _, before, after in rows if before.verified and not after.verified)
    section["gained"] = sum(1 for _, _, before, after in rows if after.verified and not before.verified)
    return section


def build_version_tax(
    attempts: Sequence[tuple[str, str]],
    earlier: Mapping[str, Outcome],
    later: Mapping[str, Outcome],
    compiles_at_both: Iterable[str],
    group_of: Mapping[str, str] | None = None,
    base_pass_rate: Mapping[str, float] | None = None,
    band_edges: Sequence[float] = (0.25, 0.75),
    resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """The tax report over `attempts`, a list of (attempt id, statement id).

    `earlier` and `later` are the two pins' outcomes by attempt id and must cover EXACTLY the attempts given:
    the comparison is of identical texts, so a pin that judged another set makes the report fail (fixture 3).
    Compared are the attempts on statements in `compiles_at_both` that BOTH pins answered. `group_of` (a
    statement's group, e.g. conjecture or reward) and the base pass-rate bands give the breakdowns. A
    statement's band is `base_pass_rate` of it (the earlier run's own figure: verified over samples drawn);
    without that argument, its pass rate under the earlier pin over the attempts compared."""
    attempt_ids = [attempt_id for attempt_id, _ in attempts]
    if len(set(attempt_ids)) != len(attempt_ids):
        raise ValueError("an attempt id is listed twice")
    for name, outcomes in (("earlier", earlier), ("later", later)):
        missing, extra = set(attempt_ids) - set(outcomes), set(outcomes) - set(attempt_ids)
        if missing or extra:
            raise ValueError(f"the two pins must have judged the same attempts: the {name} pin lacks {len(missing)} "
                             f"of the {len(attempt_ids)} and has {len(extra)} others")
    both = set(compiles_at_both)
    on_both = [(attempt_id, statement) for attempt_id, statement in attempts if statement in both]
    rows = [(attempt_id, statement, earlier[attempt_id], later[attempt_id]) for attempt_id, statement in on_both
            if earlier[attempt_id].answered and later[attempt_id].answered]
    drawn_by_statement: dict[str, list[int]] = {}
    for _, statement, before, _ in rows:
        counts = drawn_by_statement.setdefault(statement, [0, 0])
        counts[0] += 1
        counts[1] += before.verified
    band = {statement: band_of(verified / drawn if base_pass_rate is None else base_pass_rate[statement], band_edges)
            for statement, (drawn, verified) in drawn_by_statement.items()}
    report = {
        "attempts": {"given": len(attempts), "on_statements_compiling_at_both_pins": len(on_both),
                     "no_answer_earlier": sum(not earlier[attempt_id].answered for attempt_id, _ in on_both),
                     "no_answer_later": sum(not later[attempt_id].answered for attempt_id, _ in on_both),
                     "compared": len(rows)},
        "overall": _section(rows, resamples, seed),
        "by_band": {name: _section([row for row in rows if band[row[1]] == name], resamples, seed)
                    for name in band_names(band_edges)},
    }
    if group_of is not None:
        groups = sorted({group_of[statement] for _, statement, _, _ in rows})
        report["by_group"] = {group: _section([row for row in rows if group_of[row[1]] == group], resamples, seed)
                              for group in groups}
    report["read"] = pre_registered_read(report["overall"])
    return report


def pre_registered_read(section: Mapping) -> dict:
    """O2's read, written before any result: the tax is SMALL if the later pin's pass rate is within 20%
    (relative) of the earlier pin's. Read on the per-attempt rate; the interval says whether the data decide."""
    figure = section.get("per_attempt", {}).get("later_as_share_of_earlier", {})
    share = figure.get("value")
    if share is None:
        return {"rule": f"small if later/earlier >= {SMALL_TAX_RATIO}", "later_as_share_of_earlier": None, "read": "undefined"}
    decided = figure["low"] is not None and (figure["low"] >= SMALL_TAX_RATIO or figure["high"] < SMALL_TAX_RATIO)
    return {"rule": f"small if later/earlier >= {SMALL_TAX_RATIO}", "later_as_share_of_earlier": share,
            "low": figure["low"], "high": figure["high"], "relative_loss": 1.0 - share,
            "read": "small" if share >= SMALL_TAX_RATIO else "not small",
            "interval_on_one_side_of_the_threshold": bool(decided)}
