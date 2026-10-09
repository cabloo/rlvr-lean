"""The ceiling: a labelled diagnostic. Spec: docs/spec/ladder-loop.spec.md, "The ceiling: a labelled diagnostic".
Pure: no model, no Lean, and no length of a HELD-OUT problem's published proof (the report reads those).

AN EXCEPTION, NOT A CHANGE OF RULE. Published proofs are certificates and not training text (the loop's rule; fixture
7; `training_set.py`). This trains ONCE on them, as a diagnostic: no model trained this way is kept or used in a round,
the training file is not published, and every result is labelled "ceiling".

  the file       its rows, checked (what is refused) and turned into the round's training examples, in the file's order
  the schedule   the optimizer step after which each checkpoint is saved, by ROWS seen
  the checks     the three "can this run see a win" checks, each with its number and whether it passes
  the branch     the one the numbers select, in the spec's words
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Collection, Mapping, Sequence

from rlvr_lean.domain.problem_pool.certificates import TRUE_SIDE
from rlvr_lean.domain.problem_pool.episodes import ABOVE_BAND, STATEMENT_SIDE

CEILING = "ceiling"                                 # what every stored file, report heading and printed line calls this
LABEL = "the ceiling, a labelled diagnostic: one training on published proofs, and no model trained this way is kept"
SMALL, FULL = "small", "full"                       # the smaller dose, and the end of the one pass
CHECKPOINTS = (SMALL, FULL)                         # the stage's measure steps; `ceiling.checkpoints` gives their rows
ROW_FIELDS = ("problem_id", "kind", "side", "statement", "proof", "proof_lines")

LEARNS = "THE MODEL CAN LEARN LONGER PROOFS FROM EXAMPLES"              # clear of zero and at least twice the loop's own gain
AS_THE_LOOP = "LONGER PROOFS HELP ABOUT AS THE LOOP'S OWN DO"           # clear of zero, not twice the loop's own
NOT_SHOWN = "NOT SHOWN TO BE LEARNABLE AT THIS SIZE"                    # the interval holds zero, the point not above the loop's own gain
TWO_MORE_SEEDS = "TWO MORE SEEDS BEFORE ANYTHING IS CONCLUDED"          # the interval holds zero, the point above the loop's own gain
FEWER = "FEWER SUCCESSES THAN THE BASE"                                 # the interval lies below zero: a case the read did not name
INCONCLUSIVE = "INCONCLUSIVE"                                           # a check failed: the run could not have seen a win
NOT_READ = "NOT READ"                                                   # the primary has no problem to be read on (a smoke run)
BRANCHES = (LEARNS, AS_THE_LOOP, NOT_SHOWN, TWO_MORE_SEEDS, FEWER, INCONCLUSIVE, NOT_READ)


# ------------------------------------------------------------------------------------------------ the file
def training_rows(rows: Sequence[Mapping], heldout: Collection[str], base_map: Collection[str], wanted: int,
                  what: str = "the ceiling's training file") -> list[Mapping]:
    """The rows the run trains on: the first `wanted` rows of the training file, in the file's order (`wanted` is the
    last checkpoint: the end of the one pass). Refused, with nothing returned (ValueError): a file with fewer rows; a
    row without one of its fields, with an empty statement or proof, or whose statement does not end at `:= by`; ANY
    row of the file whose problem is held out or is one of the base map's; a row on another side than `true` (its
    proof would be of the negation, and the statement is what is trained on); a problem that appears twice."""
    for position, row in enumerate(rows):
        missing = [name for name in ROW_FIELDS if name not in row]
        if missing:
            raise ValueError(f"row {position} of {what} has no {missing}: refused")
        if not row["statement"].rstrip().endswith(":= by") or not row["proof"].strip():
            raise ValueError(f"row {position} of {what} ({row['problem_id']}) is not a statement up to `:= by` and a proof: refused")
    for barred, barred_what in ((heldout, "held-out problems"), (base_map, "problems of the base map")):
        found = [row["problem_id"] for row in rows if row["problem_id"] in barred]
        if found:
            raise ValueError(f"{len(found)} rows of {what} are {barred_what} (first: {found[0]}): refused")
    other_side = [row["problem_id"] for row in rows if row["side"] != TRUE_SIDE]
    if other_side:
        raise ValueError(f"{len(other_side)} rows of {what} are not on the side `{TRUE_SIDE}` (first: {other_side[0]}): refused")
    repeated = sorted(problem_id for problem_id, count in Counter(row["problem_id"] for row in rows).items() if count > 1)
    if repeated:
        raise ValueError(f"{len(repeated)} problems appear twice in {what} (first: {repeated[0]}): refused")
    if len(rows) < wanted:
        raise ValueError(f"{what} has {len(rows)} rows and the last checkpoint is after {wanted}: refused")
    return list(rows[:wanted])


def training_example(row: Mapping) -> dict:
    """A row as the ROUND's training example (the shape `training_set.training_examples` gives, which the round's own
    builder encodes): the theorem is the statement, the completion its published proof."""
    return {"problem_id": row["problem_id"], "side": STATEMENT_SIDE, "theorem": row["statement"], "completion": row["proof"]}


def check_examples_fit(problem_ids: Sequence[str], tokens: Sequence[int], limit: int, what: str = "the ceiling's training file") -> None:
    """Every row must fit the length the round's recipe cuts an example at (`training.max_sequence_tokens`): a
    longer one would be trained on without its end. Refused (ValueError) otherwise."""
    too_long = [(problem_id, count) for problem_id, count in zip(problem_ids, tokens) if count > limit]
    if too_long:
        raise ValueError(f"{len(too_long)} rows of {what} are longer than the {limit} tokens a training example is cut at "
                         f"(first: {too_long[0][0]}, {too_long[0][1]} tokens): they would be trained on without their end. Refused")


# -------------------------------------------------------------------------------------------- the schedule
def checkpoint_steps(rows: Sequence[int], batch: int) -> dict[str, dict]:
    """Each checkpoint by ROWS seen: it is saved after the first optimizer step at or after its number of rows,
    and `rows_seen` is what the model had been trained on by then (the last checkpoint is the end of the pass: the
    training set is its number of rows). `rows` must be one increasing number for each of `CHECKPOINTS`."""
    rows = [int(count) for count in rows]
    if len(rows) != len(CHECKPOINTS) or rows[0] < 1 or any(later <= earlier for earlier, later in zip(rows, rows[1:])):
        raise ValueError(f"ladder_loop.ceiling.checkpoints is {list(rows)}: the stage measures {CHECKPOINTS}, one increasing number of rows for each")
    if batch < 1:
        raise ValueError(f"a batch of at least one is needed, got {batch}")
    total = rows[-1]
    return {name: {"rows": wanted, "step": math.ceil(wanted / batch), "rows_seen": min(math.ceil(wanted / batch) * batch, total)}
            for name, wanted in zip(CHECKPOINTS, rows)}


def doses(rows: Sequence[Mapping], tokens: Sequence[int], schedule: Mapping[str, Mapping], group_of_lines: Callable[[int], str]) -> dict[str, dict]:
    """What each checkpoint's model had been trained on: its rows by kind and by the length group of their OWN
    proofs (`proof_lines`: these are training rows, not held-out problems), and their tokens as training examples."""
    result = {}
    for name, entry in schedule.items():
        own, own_tokens = rows[:entry["rows_seen"]], list(tokens[:entry["rows_seen"]])
        result[name] = {"rows": len(own), "by_kind": dict(sorted(Counter(row["kind"] for row in own).items())),
                        "by_proof_lines": dict(Counter(group_of_lines(row["proof_lines"]) for row in own)),
                        "tokens": sum(own_tokens), "longest_tokens": max(own_tokens, default=None)}
    return result


# ---------------------------------------------------------------------------------------------- the checks
def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 5) if values else None


def the_training_took(row_losses: Sequence[float], window: int) -> dict:
    """Check 1: the mean training loss over the last `window` rows is below the mean over the first `window`. A
    row's loss is its mean loss per target token, read BEFORE the update of the step it was in. A run of fewer than
    twice the window compares its first half with its second."""
    compared = min(window, len(row_losses) // 2)
    first, last = _mean(row_losses[:compared]), _mean(row_losses[len(row_losses) - compared:] if compared else [])
    return {"what": f"the mean training loss over the last {compared} rows is below the mean over the first {compared}",
            "rows": len(row_losses), "rows_compared": compared, "first": first, "last": last,
            "passes": bool(compared and last < first)}


def still_writes_proofs(rung_rows: Sequence[Mapping], maximum: float) -> dict:
    """Check 2: the model still writes proofs. Its attempts on the rungs that got no answer (they reached the token
    cap and were never sent to Lean, or Lean gave no verdict on them) stay under `maximum` of what it attempted."""
    attempts = sum(row["episodes"] * row["sides"] for row in rung_rows)
    capped, no_verdict = sum(row["attempts_capped"] for row in rung_rows), sum(row["attempts_without_an_answer"] for row in rung_rows)
    share = round((capped + no_verdict) / attempts, 5) if attempts else None
    return {"what": f"the share of its attempts on the three rungs without an answer (the token cap reached, or no verdict from Lean) is under {maximum}",
            "attempts": attempts, "capped_at_the_token_limit": capped, "without_a_verdict_from_lean": no_verdict, "share": share, "maximum": maximum,
            "passes": bool(share is not None and share < maximum)}


def pass_rate(rows: Sequence[Mapping], problem_ids: Collection[str]) -> float | None:
    own = [row["resolved"] / row["episodes"] for row in rows if row["problem_id"] in problem_ids]
    return _mean(own)


def not_broken(rung_rows: Sequence[Mapping], base_rung_rows: Sequence[Mapping], above_band: Collection[str], minimum_share: float) -> dict:
    """Check 3: training on other provers' proofs has not broken it. Its pass rate on the above-band rung is at
    least `minimum_share` of the base's (the same problems, episodes and sampling seed)."""
    own, base = pass_rate(rung_rows, above_band), pass_rate(base_rung_rows, above_band)
    return {"what": f"its pass rate on the above-band rung is at least {minimum_share} of the base's", "rung": ABOVE_BAND, "problems": len(above_band),
            "pass_rate": own, "pass_rate_of_the_base": base, "minimum": round(minimum_share * base, 5) if base is not None else None,
            "passes": bool(own is not None and base is not None and own >= minimum_share * base)}


def can_this_run_see_a_win(row_losses: Sequence[float], rung_rows: Sequence[Mapping], base_rung_rows: Sequence[Mapping], above_band: Collection[str],
                           settings: Mapping) -> dict:
    """The three checks the spec reads before the branch, on the model the primary is read on. Any that fails
    makes the run INCONCLUSIVE: it could not have seen a win, and nothing else is said about the model."""
    checks = {"the_training_took": the_training_took(row_losses, settings["loss_window_rows"]),
              "the_model_still_writes_proofs": still_writes_proofs(rung_rows, settings["maximum_share_without_an_answer"]),
              "training_has_not_broken_it": not_broken(rung_rows, base_rung_rows, above_band, settings["minimum_share_of_the_bases_pass_rate_above_the_band"])}
    failed = [name for name, check in checks.items() if not check["passes"]]
    return {**checks, "failed": failed, "all_pass": not failed}


# ---------------------------------------------------------------------------------------------- the branch
def interval_text(change: Mapping | None) -> str:
    """A change with its 95% interval, signed and to five places (these are rates of a few per thousand)."""
    if not change or change.get("mean") is None:
        return "not measured"
    return f"{change['mean']:+.5f} [{change['low']:+.5f}, {change['high']:+.5f}]"


def ceiling_branch(checks: Mapping, primary: Mapping, own_gain: float, times: float) -> dict:
    """The branch the numbers select, in the spec's words. `primary`: the full model minus the base on the goal
    problems whose shortest published proof is 4 lines or more, successes per attempt, paired by problem, with its
    interval. `own_gain`: the loop's own gain there, ONE number fixed before the run (`ceiling.loops_own_gain`: the
    three-round model at t = 1/10 minus the base, the three seeds pooled). No seed's own three-round model decides.

    The checks first: a failed one is INCONCLUSIVE. Then, by the primary's interval: clear of zero above it (a point
    of at least `times` the loop's own gain, or under it); holding zero (the point above the loop's own gain: two
    more seeds before anything is concluded; else not shown); entirely below zero, which the read did not name."""
    if not checks["all_pass"]:
        failed = "; ".join(f"{name.replace('_', ' ')}: FAIL" for name in checks["failed"])
        return {"name": INCONCLUSIVE, "reason": f"a check that this run could have seen a win failed ({failed}). The run failed, and nothing is said about "
                                                "the model: fix what failed and run one seed again"}
    if primary.get("mean") is None:
        return {"name": NOT_READ, "reason": "the primary could not be computed: no goal problem with a published proof of 4 lines or more was measured on both sides"}
    bar = times * own_gain
    reference = f"the loop's own gain there is {own_gain:+.5f}, the three seeds pooled, fixed before the run"
    if primary["low"] > 0:
        if primary["mean"] >= bar:
            return {"name": LEARNS, "reason": f"the primary is {interval_text(primary)}, above zero with an interval clear of zero, and at least {times} times the "
                                              f"loop's own gain ({reference}; {times} times it is {bar:+.5f}): the model can learn longer proofs from "
                                              "examples; L3d is worth building, and the two doses say how the gain scales with proofs"}
        return {"name": AS_THE_LOOP, "reason": f"the primary is {interval_text(primary)}, above zero with an interval clear of zero but not {times} times the loop's "
                                               f"own ({reference}; {times} times it is {bar:+.5f}): longer proofs help about as the loop's own do; L3d's case "
                                               "rests on assembled proofs being better aimed than published ones, which this cannot tell"}
    if primary["high"] < 0:
        return {"name": FEWER, "reason": f"the primary is {interval_text(primary)}, an interval entirely below zero: the model trained on the published proofs solves "
                                         "fewer of these problems per attempt than the base. The read named no branch for this case; it is reported as that"}
    if primary["mean"] > own_gain:
        return {"name": TWO_MORE_SEEDS, "reason": f"the primary is {interval_text(primary)}: its interval holds zero and its point is above the loop's own gain "
                                                  f"({reference}). One seed is a diagnostic that sizes the next step: two more seeds are run before anything is "
                                                  "concluded"}
    return {"name": NOT_SHOWN, "reason": f"the primary is {interval_text(primary)}, an interval through zero with a point not above the loop's own gain ({reference}): "
                                         "for this model and recipe, one-shot writing of longer proofs is not shown to be learnable at this size; L3d as "
                                         "plain training is unlikely to pay, and reach stays in the episode (a larger pool, more generations, or training "
                                         "the continuing step)"}
