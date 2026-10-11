"""L4p: how much more pretraining? The dose of a second pass. Spec: docs/spec/ladder-loop.spec.md, "L4p: how much
more pretraining? The dose of a second pass" and its "Made exact by the build". Pure.

A LABELLED CHECK OF THE PRETRAINING, as L4r was: no round, and no row of a round. L4t2's `rehearse_only` trained the
start model again on a seventh of the published proofs it had seen once and is the broadest and most reliable model
measured. Two explanations, which the dose tells apart: MORE TRAINING HELPS (the gain grows with the rows), or ANY SHORT
FURTHER TRAINING HELPS (the gain is flat in the rows). Three more models, each trained from the start model on rows of
the file it was pretrained on, along the ONE hash order L4t's rehearsal draw uses (`l4_rows.rehearsal_rows`):

  matched_1262  trained and measured FIRST: as many rows as L4t2's stored `reward_rehearse` set has published rows (1,262
                at seed 0), the first of that order: they ARE that set's published rows, and nothing else. The control
                L4t2 left `reward_rehearse` without, and the smallest dose
  second_half   the first half of that order (`rows_of`): `rehearse_only`'s rows are its first
  second_full   every row of the pretraining file, the draw of every row: `second_half`'s rows are its first

Every nesting is HELD, not assumed, and so is every row of the file standing once in `second_full` (`dose_sets`). Each
set is trained in the arm's content-hash order (`assembly.training_order`), as `rehearse_only` was. A row is a published
proof: a set names it by its id and its place in the file, never with its text.

  the rounds' rows  the question at the size of one pass: on G' over the fresh attempts, the stored
                    `reward_rehearse` minus `matched_1262` (`matched_branch`): above zero, THE ROUNDS' HARD ROWS ADD
                    BESIDE THE SAME PUBLISHED ROWS; holding zero, NOT SEPARATED AT THIS SIZE; below zero, THEY COST
  the primary       all of G, successes per attempt over all the attempts, `second_full` minus `rehearse_only`
                    (`dose_branch`): above zero, MORE OF THE SECOND PASS HELPS; holding zero, THE FIRST SEVENTH GAVE WHAT
                    A SECOND PASS GIVES; below zero, MORE COSTS
  overfitting       of each model against the start model, read on every branch (`overfitting`): NARROWER (L4b's own, by
                    its function), each rung whose interval is wholly below zero named, the share of distinct attempts

A model whose own checks failed is INCONCLUSIVE and none of its reads is made; what stands on it is not read either. A
model that is not measured yet (the job that trains and measures `matched_1262` alone writes a report of its own) is
NOT READ, and said as such.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.assembly import training_order
from rlvr_lean.domain.ladder_round.ceiling import interval_text
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ
from rlvr_lean.domain.ladder_round.l4_rows import PRETRAINING, SIGN_TEST_LEVEL, rehearsal_rows
from rlvr_lean.domain.ladder_round.l4_rows2 import NOT_SEPARATED, REHEARSE_ONLY, REWARD_REHEARSE, _placed
from rlvr_lean.domain.ladder_round.l4b import NARROWER, breadth_beside, rehearsal_of

MATCHED, SECOND_HALF, SECOND_FULL = "matched_1262", "second_half", "second_full"
MODELS = (MATCHED, SECOND_HALF, SECOND_FULL)        # the three trained models, in the order they are trained and measured
DOSES = (MATCHED, REHEARSE_ONLY, SECOND_HALF, SECOND_FULL)      # the dose along ONE order, the smallest first: the second is L4t2's, stored, read and never trained again
STORED = (REHEARSE_ONLY, REWARD_REHEARSE)           # L4t2's two models, read here from the rows its run stored: the dose's reference, and the model `matched_1262` is the control of

MORE_HELPS = "MORE OF THE SECOND PASS HELPS"                        # the primary's interval above zero
FIRST_SEVENTH = "THE FIRST SEVENTH GAVE WHAT A SECOND PASS GIVES"   # ... holding zero (one that ends at zero holds it)
MORE_COSTS = "MORE COSTS"                                           # ... below zero
BRANCHES = (MORE_HELPS, FIRST_SEVENTH, MORE_COSTS, INCONCLUSIVE, NOT_READ)
HARD_ROWS_ADD = "THE ROUNDS' HARD ROWS ADD BESIDE THE SAME PUBLISHED ROWS"      # `reward_rehearse` minus `matched_1262` on G': the interval above zero
THEY_COST = "THEY COST"                                                         # ... below zero (holding zero: `NOT_SEPARATED`, L4t2's own words)
MATCHED_BRANCHES = (HARD_ROWS_ADD, NOT_SEPARATED, THEY_COST, INCONCLUSIVE, NOT_READ)


def rows_of(model: str, rows: int) -> int:
    """How many rows of a pretraining file of `rows` rows a model's set holds: `second_full` every one, `second_half` half
    of them (rounded down: 12,433 of the file's 24,866). `matched_1262` is not asked here: it holds as many as L4t2's
    stored `reward_rehearse` set has published rows (`dose_sets`)."""
    if model not in (SECOND_HALF, SECOND_FULL):
        raise ValueError(f"{model!r} is not a dose by a share of the file: it is one of {[SECOND_HALF, SECOND_FULL]}")
    return rows if model == SECOND_FULL else rows // 2


# ------------------------------------------------------------------------------------------------- the sets
def dose_sets(rehearse_only: Sequence[Mapping], reward_rehearse: Sequence[Mapping], pretraining: Sequence[Mapping], seed: int) -> dict[str, list[Mapping]]:
    """The three training sets, each in the arm's content-hash order (`assembly.training_order`). `rehearse_only`,
    `reward_rehearse`: the two sets L4t2's run stored, each in its order. `pretraining`: the rows the start model was
    pretrained on, in the file's order, WITHOUT their text. `seed`: the task's, which is L4t2's and L4t's. Each row is a
    row of L4t's own rehearsal draw (`l4_rows.rehearsal_rows`, the same call with another count): its id, its problem,
    its place in the file.

    Refused (ValueError), because the dose would not be along ONE order: `rehearse_only` holds a row that is not a
    rehearsal row, more rows than `second_half`, rows that are not the first of the hash order (by id and by place in
    the file: another file or seed is refused), or rows that do not stand in the content-hash order; `second_half`'s
    rows are not the first of `second_full`'s; `second_full` does not hold every row of the file exactly once. And for
    `matched_1262`: `reward_rehearse` holds no published row, or more than `rehearse_only` has rows; the rows drawn are
    not the first of the order; they are not EXACTLY the published rows of `reward_rehearse`, by id and by place; or
    those do not stand in the content-hash order (the control is that set with its rounds' rows left out, none moved)."""
    full = rehearsal_rows(pretraining, rows_of(SECOND_FULL, len(pretraining)), seed)       # the draw of EVERY row: the whole hash order, the smallest hash first
    half = rehearsal_rows(pretraining, rows_of(SECOND_HALF, len(pretraining)), seed)
    places, ids = sorted(row["row_of_the_file"] for row in full), {row["id"] for row in full}
    if places != list(range(len(pretraining))) or len(ids) != len(pretraining):
        raise ValueError(f"`{SECOND_FULL}` holds {len(full)} rows ({len(ids)} ids, {len(set(places))} places of the file) and the start model was pretrained on "
                         f"{len(pretraining)}: it is every row of the pretraining file exactly once. Refused")
    if list(map(_placed, half)) != list(map(_placed, full[:len(half)])):
        raise ValueError(f"the {len(half)} rows drawn for `{SECOND_HALF}` are not the first {len(half)} of the hash order `{SECOND_FULL}`'s {len(full)} stand in. Refused")
    others = [row["id"] for row in rehearse_only if row["origin"] != PRETRAINING]
    if others or not rehearse_only:
        raise ValueError(f"L4t2's `{REHEARSE_ONLY}` set holds {len(rehearse_only)} rows and {len(others)} of them are not rehearsal rows"
                         + (f" (first: {others[0]})" if others else "") + ": the dose is of rows of the pretraining file and of no other. Refused")
    if len(rehearse_only) > len(half):
        raise ValueError(f"L4t2's `{REHEARSE_ONLY}` set holds {len(rehearse_only)} rows and `{SECOND_HALF}` {len(half)} (half of the {len(pretraining)} the start model was "
                         f"pretrained on): `{REHEARSE_ONLY}`'s rows are the first of `{SECOND_HALF}`'s, and a larger set cannot be. Refused")
    first = full[:len(rehearse_only)]
    if sorted(map(_placed, rehearse_only)) != sorted(map(_placed, first)):
        other = sorted(set(map(_placed, rehearse_only)) ^ set(map(_placed, first)))
        raise ValueError(f"L4t2's `{REHEARSE_ONLY}` set holds {len(rehearse_only)} rows, and they are not the first {len(first)} of the hash order among the "
                         f"{len(pretraining)} rows the start model was pretrained on (first that differs: {other[0][0]}): that set was not drawn from this file with this "
                         "seed, and the dose would not be along one order. Refused")
    if [row["id"] for row in rehearse_only] != [row["id"] for row in training_order(rehearse_only, seed)]:
        raise ValueError(f"L4t2's `{REHEARSE_ONLY}` set does not stand in the content-hash order of this seed: the models here are trained in that order, as "
                         f"`{REHEARSE_ONLY}` was. Refused")
    # ---- `matched_1262`: as many rows as `reward_rehearse` has published rows, the first of the order: EXACTLY that set's published rows
    published = rehearsal_of(reward_rehearse)
    if not published or len(published) > len(rehearse_only):
        raise ValueError(f"L4t2's `{REWARD_REHEARSE}` set holds {len(published)} published rows among its {len(reward_rehearse)} and `{REHEARSE_ONLY}` holds "
                         f"{len(rehearse_only)}: `{MATCHED}` is those published rows and nothing else, and the smallest dose. Refused")
    matched = rehearsal_rows(pretraining, len(published), seed)
    if list(map(_placed, matched)) != list(map(_placed, full[:len(published)])):
        raise ValueError(f"the {len(matched)} rows drawn for `{MATCHED}` are not the first {len(published)} of the hash order `{SECOND_FULL}`'s {len(full)} stand in. Refused")
    if sorted(map(_placed, published)) != sorted(map(_placed, matched)):
        other = sorted(set(map(_placed, published)) ^ set(map(_placed, matched)))
        raise ValueError(f"L4t2's `{REWARD_REHEARSE}` set holds {len(published)} published rows, and they are not the first {len(matched)} of the hash order among the "
                         f"{len(pretraining)} rows the start model was pretrained on (first that differs: {other[0][0]}): `{MATCHED}` would not be the control of that "
                         "set. Refused")
    if [row["id"] for row in published] != [row["id"] for row in training_order(published, seed)]:
        raise ValueError(f"the published rows of L4t2's `{REWARD_REHEARSE}` set do not stand in the content-hash order of this seed: `{MATCHED}` is that set with its "
                         "rounds' rows left out and none moved. Refused")
    return {MATCHED: training_order(matched, seed), SECOND_HALF: training_order(half, seed), SECOND_FULL: training_order(full, seed)}


# --------------------------------------------------------------------------------------------- the branches
def matched_branch(failed_checks: Sequence[str], difference: Mapping, of_reward_rehearse: Sequence[str] = ()) -> dict:
    """The question at the size of one pass, fixed before the run: do the model's own proofs of hard problems
    add anything that the same published rows do not? `difference`: on G' over the fresh attempts, the stored
    `reward_rehearse` minus `matched_1262`, paired by problem, with its interval. `failed_checks`: the checks
    `matched_1262` ITSELF failed: INCONCLUSIVE. `of_reward_rehearse`: the checks the run that measured
    `reward_rehearse` read as failed for it: NOT READ against it. Nothing measured on both sides (a run with one
    sampling of G has no fresh attempts): NOT READ.

    The interval above zero: THE ROUNDS' HARD ROWS ADD BESIDE THE SAME PUBLISHED ROWS. It holds zero (one that ends at
    zero holds it): NOT SEPARATED AT THIS SIZE, with what this difference resolves (half the width of its interval).
    Below zero: THEY COST. It decides no arm by itself."""
    if failed_checks:
        return {"name": INCONCLUSIVE, "failed_checks": list(failed_checks),
                "reason": f"`{MATCHED}` could not have been read ({', '.join(failed_checks)}): nothing is said about what the rounds' hard rows add beside the same "
                          "published rows"}
    if of_reward_rehearse:
        return {"name": NOT_READ, "failed_checks": [], "not_read_against": REWARD_REHEARSE,
                "reason": f"`{REWARD_REHEARSE}`, which stands on the other side, is {INCONCLUSIVE} in the report of the run that measured it "
                          f"({', '.join(of_reward_rehearse)}): the difference is not read"}
    if difference.get("mean") is None:
        return {"name": NOT_READ, "failed_checks": [], "reason": "G' holds no problem with fresh attempts on both sides: the difference has nothing to be read on"}
    read = interval_text(difference)
    if difference["low"] > 0:
        return {"name": HARD_ROWS_ADD, "failed_checks": [],
                "reason": f"on G' over the fresh attempts `{REWARD_REHEARSE}` minus `{MATCHED}` is above zero ({read}): the rounds' hard rows add beside the same "
                          "published rows. It decides no arm by itself: the arm with its own separation is the test of the cycle"}
    if difference["high"] < 0:
        return {"name": THEY_COST, "failed_checks": [],
                "reason": f"on G' over the fresh attempts `{REWARD_REHEARSE}` minus `{MATCHED}` is below zero ({read}): beside the same published rows the rounds' hard "
                          "rows cost. It decides no arm by itself: the arm with its own separation is the test of the cycle"}
    resolves = round((difference["high"] - difference["low"]) / 2, 5)
    return {"name": NOT_SEPARATED, "failed_checks": [], "resolves": resolves,
            "reason": f"on G' over the fresh attempts `{REWARD_REHEARSE}` minus `{MATCHED}` holds zero ({read}): not separated at this size. This difference resolves "
                      f"{resolves:.5f} per attempt (half the width of its interval). It decides no arm by itself: the arm with its own separation is the test of the cycle"}


def dose_branch(failed_checks: Sequence[str], primary: Mapping, start: str, of_rehearse_only: Sequence[str] = (), measured: bool = True) -> dict:
    """The primary's branch, fixed before the run. `primary`: all of G, successes per attempt over all the attempts,
    `second_full` minus `rehearse_only`, paired by problem, with its interval. `measured`: whether `second_full` is
    measured yet; in the report of the job that trains and measures `matched_1262` alone it is not, and the primary is
    NOT READ, said as such. `failed_checks`: the checks `second_full` ITSELF failed: INCONCLUSIVE, and nothing is said.
    `of_rehearse_only`: the checks the run that measured `rehearse_only` read as failed for it: the primary is then
    NOT READ against it. Nothing measured on both sides: NOT READ.

    The interval above zero: MORE OF THE SECOND PASS HELPS (and a third pass is specified next). It holds zero (one
    that ends at zero holds it): THE FIRST SEVENTH GAVE WHAT A SECOND PASS GIVES (the gain is not the amount of
    training; the recipe is one pass and a short second one), with what this difference resolves (half the width of its
    interval). Below zero: MORE COSTS."""
    if not measured:
        return {"name": NOT_READ, "failed_checks": [], "not_measured": SECOND_FULL,
                "reason": f"`{SECOND_FULL}` is not measured yet: this is the report of the job that trains and measures `{MATCHED}` alone, and the dose's primary is read "
                          "by the report of the whole stage"}
    if failed_checks:
        return {"name": INCONCLUSIVE, "failed_checks": list(failed_checks),
                "reason": f"`{SECOND_FULL}` could not have been read ({', '.join(failed_checks)}): nothing is said about how much more of a second pass helps"}
    if of_rehearse_only:
        return {"name": NOT_READ, "failed_checks": [], "not_read_against": REHEARSE_ONLY,
                "reason": f"`{REHEARSE_ONLY}`, which the primary is read against, is {INCONCLUSIVE} in the report of the run that measured it ({', '.join(of_rehearse_only)}): "
                          "the primary is not read; nothing is said about how much more of a second pass helps"}
    if primary.get("mean") is None:
        return {"name": NOT_READ, "failed_checks": [], "reason": "no goal problem was measured on both sides: the primary has nothing to be read on"}
    read = interval_text(primary)
    if primary["low"] > 0:
        return {"name": MORE_HELPS, "failed_checks": [],
                "reason": f"all of G, `{SECOND_FULL}` minus `{REHEARSE_ONLY}`: the interval is above zero ({read}): more of the second pass helps, and a third pass is "
                          "specified next"}
    if primary["high"] < 0:
        return {"name": MORE_COSTS, "failed_checks": [],
                "reason": f"all of G, `{SECOND_FULL}` minus `{REHEARSE_ONLY}`: the interval is below zero ({read}): more of the second pass costs. How each model stands "
                          f"against `{start}` is in the overfitting reads"}
    resolves = round((primary["high"] - primary["low"]) / 2, 5)
    return {"name": FIRST_SEVENTH, "failed_checks": [], "resolves": resolves,
            "reason": f"all of G, `{SECOND_FULL}` minus `{REHEARSE_ONLY}`: the interval holds zero ({read}): the first seventh gave what a second pass gives. The gain is "
                      f"not the amount of training; the recipe is one pass and a short second one. This difference resolves {resolves:.5f} per attempt (half the width of "
                      "its interval)"}


# -------------------------------------------------------------------------------------------- overfitting
def overfitting(solved: Mapping, rungs: Mapping[str, Mapping], distinct: float | None, distinct_of_the_start: float | None, start: str,
                failed_checks: Sequence[str] = (), level: float = SIGN_TEST_LEVEL) -> dict:
    """The overfitting reads of ONE model against the start model, read on every branch. `solved`: the goal problems
    solved at least once in all the attempts, the model against `start` (`gained`, `lost`, `sign_test_p`). `rungs`: by
    rung, the fresh pass rate, the model minus `start`, paired by problem, with its interval. `distinct`,
    `distinct_of_the_start`: each one's share of distinct attempts on G.

      NARROWER      lost more goal problems than gained with the sign test under `level`: L4b's own, by its function
                    (`l4b.breadth_beside`), which L4t2's read NOT NARROWER is the reverse of
      the rungs     a rung whose interval is WHOLLY below zero is named (one that ends at zero is not wholly below it)
      distinct      the two shares, said and not branched on

    `failed_checks`: the checks the model ITSELF failed: it is then INCONCLUSIVE, none of these is read, and what was
    measured is kept apart."""
    breadth = breadth_beside(solved, distinct, distinct_of_the_start, start, level)
    below = [rung for rung, change in rungs.items() if change.get("mean") is not None and change["high"] < 0]
    measured = {"narrower": breadth["narrower"], **{key: breadth[key] for key in ("problems", "gained", "lost", "sign_test_p", "counts", "smallest_decisive_split")},
                "the_three_rungs": dict(rungs), "rungs_wholly_below_zero": below,
                "share_of_distinct_attempts_on_g": {"of_the_model": distinct, f"of_{start}": distinct_of_the_start}}
    what = (f"against `{start}`: {NARROWER} when it lost more goal problems than it gained in all the attempts with the two-sided sign test under {level:g}; each of the "
            "three rungs whose interval (the fresh pass rate, the model minus the start model, 95% bootstrap over problems) is wholly below zero; the share of distinct "
            "attempts on G")
    if failed_checks:
        return {"what": what, "against": start, "failed_checks": list(failed_checks), "narrower": None, "rungs_wholly_below_zero": None,
                "said": f"NOT READ: this model is {INCONCLUSIVE} ({', '.join(failed_checks)}), and none of its reads is made", "measured_and_not_read": measured}
    return {"what": what, "against": start, "failed_checks": [], **measured,
            "said": (f"{NARROWER}" if breadth["narrower"] else f"not {NARROWER.lower()}") + f" ({breadth['counts']}); the three rungs: "
                    + "; ".join(f"{rung} {interval_text(change)}" for rung, change in rungs.items())
                    + (f": WHOLLY BELOW ZERO on {', '.join(below)}" if below else ": none wholly below zero")
                    + f"; the share of distinct attempts on G {distinct} against {distinct_of_the_start}"}
