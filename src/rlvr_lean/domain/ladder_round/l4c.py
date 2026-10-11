"""L4c: the arm from `pre_r64` with a rule L4t2 names. Spec: docs/spec/ladder-loop.spec.md, "L4c: the arm from
`pre_r64` with a rule L4t2 names" and its "Made exact by the build". Pure.

It is L4b's arm with what "The arm's rule after L4t2" asks and nothing else. Two things are added to L4b's build:

  reward_rehearse   a FOURTH training rule of the arm. It is one of the arm's rules and lives where they do
                    (`l4b.rule_set`, on L4t's own draw and keep rule): the rows `reward_rows` keeps, and as many rows of
                    the pretraining file as that kept set has ONE-SHOT rows, the first of the hash order `rehearse`
                    draws from
  rehearsal_alone   THE ARM'S OWN SEPARATION, for the two rules with rehearsal rows (`l4b.WITH_REHEARSAL`): after the
                    last round ONE more model, trained from the start model on the rehearsal rows of the last model's
                    set ALONE, in that set's order with the other rows left out (`rehearsal_alone_set`), and measured as
                    the twin is. Under the two other rules the arm has no such model

Half of such an arm's training rows are published proofs the start model has seen once: a second pass over them alone
could give the primary's gain. So THE READ is L4b's, unchanged, and beside the primary, for a rule with rehearsal rows:

  the rounds, apart from the rehearsal   on G' over the fresh attempts, `with` minus `rehearsal_alone`, paired by problem
                    (`separation`). Above zero: THE ROUNDS ADD BESIDE THE REHEARSAL. Else, when `rehearsal_alone` itself
                    is above zero against the start model: THE GAIN IS NOT SHOWN TO BE THE ROUNDS' (a second pass over
                    published proofs gives it). Else: NOT SEPARATED AT THIS SIZE. NOT READ when a check of
                    `rehearsal_alone` itself fails, which changes nothing else. A primary that reads THE LOOP ADDS ON TOP
                    OF PRETRAINING is REPORTED with this line after it, and is called the loop's gain only when the first
                    of the three is read (`with_the_separation`)
  the long proofs   the goal problems of 4 lines or more, successes per attempt over all the attempts, `with` minus the
                    start model: an interval wholly below zero is written SHORTER beside the branch, as NARROWER is
                    (`long_proofs_beside`, `with_the_long_proofs`). It IS the read HOLDS THE LONG PROOFS of L4t2
                    (`l4_rows2.three_reads`), read on the arm's own last model: SHORTER is that read failing

A rehearsal row is a published proof: a set names it by its id and its place in the file, never with its text.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.ceiling import interval_text
from rlvr_lean.domain.ladder_round.l4 import ADDS, INCONCLUSIVE, NOT_READ
from rlvr_lean.domain.ladder_round.l4_rows2 import NOT_SEPARATED, refuse_text
from rlvr_lean.domain.ladder_round.l4b import rehearsal_of

REHEARSAL_ALONE = "rehearsal_alone"                                 # the one more model of an arm with rehearsal rows, and its adapter's name
ROUNDS_ADD = "THE ROUNDS ADD BESIDE THE REHEARSAL"                  # `with` minus `rehearsal_alone` on G' over the fresh attempts: the interval above zero
NOT_THE_ROUNDS = "THE GAIN IS NOT SHOWN TO BE THE ROUNDS'"          # ... else, `rehearsal_alone` itself above zero against the start model: a second pass gives it
SEPARATION = (ROUNDS_ADD, NOT_THE_ROUNDS, NOT_SEPARATED, NOT_READ)  # ... else NOT SEPARATED AT THIS SIZE (L4t2's own words); NOT READ: a check of `rehearsal_alone` failed
SHORTER = "SHORTER"                                                 # on the goal problems of 4 lines or more, `with` minus the start model: the interval wholly below zero
THE_LOOPS = "the gain is the loop's"                                # how an "adds" is said when the rounds add beside the rehearsal
NOT_THE_LOOPS = "the gain is NOT called the loop's"                 # ... and when anything else is read


# ------------------------------------------------------------------------------------------ the one more model
def rehearsal_alone_set(last_set: Sequence[Mapping]) -> list[Mapping]:
    """`rehearsal_alone`'s rows: the rehearsal rows of the last model's set and nothing else, in that set's order with
    the other rows left out (`l4b.rehearsal_of`). `last_set`: what the last model was trained on, in its order.
    Refused (ValueError): a set with no rehearsal row (there is nothing to train the model on), and a rehearsal row that
    carries a text field (`l4_rows2.refuse_text`)."""
    rows = rehearsal_of(last_set)
    if not rows:
        raise ValueError(f"the last model's set holds no rehearsal row among its {len(last_set)}: there is nothing to train `{REHEARSAL_ALONE}` on. Refused")
    refuse_text(rows, f"`{REHEARSAL_ALONE}`")
    return rows


# -------------------------------------------------------------------------- the rounds, apart from the rehearsal
def separation(difference: Mapping, of_alone: Mapping, start: str, failed_checks: Sequence[str] = ()) -> dict:
    """The arm's own separation, fixed before any run. `difference`: on G' over the fresh attempts, `with` minus
    `rehearsal_alone`, paired by problem, with its interval. `of_alone`: `rehearsal_alone`'s own read ADDS against the
    start model on the same attempts (`l4_rows2.three_reads`' entry: its interval, and `passes` when it is above zero).
    `failed_checks`: the checks `rehearsal_alone` ITSELF failed (its training ran; it still writes proofs; Lean answered).

    A check failed: NOT READ, and nothing else of the report changes. Nothing measured on both sides (a run with one
    sampling of G): NOT READ. The interval above zero (one that ends at zero is not above it): THE ROUNDS ADD BESIDE THE
    REHEARSAL. Else, `rehearsal_alone` itself above zero against the start model: THE GAIN IS NOT SHOWN TO BE THE ROUNDS'.
    Else: NOT SEPARATED AT THIS SIZE, with what this difference resolves (half the width of its interval)."""
    if failed_checks:
        return {"name": NOT_READ, "failed_checks": list(failed_checks), "the_rounds_add": None,
                "reason": f"`{REHEARSAL_ALONE}` could not have been read ({', '.join(failed_checks)}): the separation is not read, and nothing else of this report changes"}
    if difference.get("mean") is None:
        return {"name": NOT_READ, "failed_checks": [], "the_rounds_add": None,
                "reason": "G' holds no problem with fresh attempts on both sides: the difference has nothing to be read on"}
    read = interval_text(difference)
    said_alone = f"`{REHEARSAL_ALONE}` minus `{start}` on the same attempts: {interval_text(of_alone)}"
    if difference["low"] > 0:
        return {"name": ROUNDS_ADD, "failed_checks": [], "the_rounds_add": True, "rehearsal_alone_adds": of_alone.get("passes"),
                "reason": f"on G' over the fresh attempts `with` minus `{REHEARSAL_ALONE}` is above zero ({read}): the rounds add beside the rehearsal ({said_alone})"}
    if of_alone.get("passes"):
        return {"name": NOT_THE_ROUNDS, "failed_checks": [], "the_rounds_add": False, "rehearsal_alone_adds": True,
                "reason": f"on G' over the fresh attempts `with` minus `{REHEARSAL_ALONE}` is not above zero ({read}), and `{REHEARSAL_ALONE}` itself is above zero against "
                          f"`{start}` ({said_alone}): the gain is not shown to be the rounds'; a second pass over published proofs gives it"}
    resolves = round((difference["high"] - difference["low"]) / 2, 5)
    return {"name": NOT_SEPARATED, "failed_checks": [], "the_rounds_add": False, "rehearsal_alone_adds": False, "resolves": resolves,
            "reason": f"on G' over the fresh attempts `with` minus `{REHEARSAL_ALONE}` is not above zero ({read}), and `{REHEARSAL_ALONE}` itself is not above zero against "
                      f"`{start}` ({said_alone}): not separated at this size. This difference resolves {resolves:.5f} per attempt (half the width of its interval)"}


def with_the_separation(branch: Mapping, apart: Mapping) -> dict:
    """The branch with the separation read beside it. The branch's NAME is L4's and does not move. A primary that reads
    THE LOOP ADDS ON TOP OF PRETRAINING is REPORTED with the separation after it (`reported_as`), and `the_loops_gain`
    says whether it is called the loop's gain: only when THE ROUNDS ADD BESIDE THE REHEARSAL is read. Any other branch
    is said as it was (the separation stands beside it in its own line; there is no gain to call anyone's). An
    INCONCLUSIVE run says nothing else."""
    said = branch.get("reported_as", branch["name"])
    if branch["name"] == INCONCLUSIVE:
        return {**branch, "apart_from_the_rehearsal": None, "the_loops_gain": None, "reported_as": said}
    if branch["name"] != ADDS:
        return {**branch, "apart_from_the_rehearsal": apart["name"], "the_loops_gain": None, "reported_as": said}
    the_loops = apart["name"] == ROUNDS_ADD
    return {**branch, "apart_from_the_rehearsal": apart["name"], "the_loops_gain": the_loops, "reported_as": f"{said}; apart from the rehearsal: {apart['name']}",
            "reason": f"{branch['reason']}. APART FROM THE REHEARSAL, {apart['name']}: {apart['reason']}: {THE_LOOPS if the_loops else NOT_THE_LOOPS}"}


# ------------------------------------------------------------------------------------------- the long proofs
def long_proofs_beside(holds: Mapping, start: str) -> dict:
    """The long proofs, beside the primary. `holds`: the read HOLDS THE LONG PROOFS of `with` against the start model
    (`l4_rows2.three_reads`' entry: on the goal problems of 4 lines or more, successes per attempt over all the attempts,
    `with` minus the start model, with its interval; `passes` is False when the interval is wholly below zero, None when
    nothing was measured). SHORTER: that read fails. One that ends at zero is not wholly below it."""
    measured = {key: holds.get(key) for key in ("problems", "mean", "low", "high")}
    return {"what": f"the goal problems of 4 lines or more, successes per attempt over all the attempts, `with` minus `{start}`, paired by problem, 95% bootstrap over "
                    f"problems. An interval wholly below zero is written {SHORTER} beside the branch, as NARROWER is",
            **measured, "read": holds["passes"] is not None, "shorter": holds["passes"] is False,
            "interval": interval_text(measured) if measured["mean"] is not None else "not measured"}


def with_the_long_proofs(branch: Mapping, long: Mapping) -> dict:
    """The branch with the long proofs read beside it. The branch's NAME does not move; `reported_as` has SHORTER after
    it when `with` has fewer successes per attempt than the start model on the goal problems of 4 lines or more, the
    interval wholly below zero. An INCONCLUSIVE run says nothing else: what was measured is kept and not read."""
    said = branch.get("reported_as", branch["name"])
    if branch["name"] == INCONCLUSIVE:
        return {**branch, "shorter": None, "reported_as": said}
    if not long["shorter"]:
        return {**branch, "shorter": False, "reported_as": said}
    return {**branch, "shorter": True, "reported_as": f"{said}, {SHORTER}",
            "reason": f"{branch['reason']}. {SHORTER}: on the goal problems of 4 lines or more it has fewer successes per attempt than the start model, the interval "
                      f"wholly below zero ({long['interval']})"}
