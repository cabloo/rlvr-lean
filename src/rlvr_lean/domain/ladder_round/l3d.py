"""L3d Step 1: do assembled proofs teach? Spec: docs/spec/ladder-loop.spec.md, "L3d: train on what the episode
reaches" ("Step 1", "The read, fixed before the run", "Made exact by the build"). Pure: rows in, rows and verdicts out.

Two models are trained from the base in one run, one pass each, the round's recipe. `without`: the training examples
the three rounds at t = 1/10 stored, in a seeded order. `with`: the same rows in the same relative order, with the
harvest H0's assembled proofs at seeded places among them. What differs between them is H0 and not the run.

  the rows      `round_rows`, `h0_rows`, `check_h0` (what an H0 is refused for)
  the orders    `seeded_order` (`without`), `with_h0` (`with`): both by a content hash of the task's seed and the
                row's id, so a rerun gives the same order
  the checks    `both_took`, `both_write_proofs`, `trained_on_h0`, `can_this_run_see_a_win`
  the branch    `noise_band`, `could_have_been_seen`, `l3d1_branch`
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Collection, Mapping, Sequence

from rlvr_lean.domain.ladder_round.ceiling import still_writes_proofs, the_training_took
from rlvr_lean.domain.problem_pool.selection import rank

L3D1 = "l3d1"                                   # what every stored file, set and printed line of the stage is called
WITHOUT, WITH = "without", "with"
ARMS = (WITHOUT, WITH)                          # the two models, in the order they are trained and measured
ROUNDS, H0 = "rounds", "h0"                     # where a training row is from
ORDER_LABEL, PLACE_LABEL = "l3d1_order", "l3d1_place"
EXAMPLE_FIELDS = ("problem_id", "side", "theorem", "completion")
HAVE = re.compile(r"have\b")
# The branches (spec, "Branches"), and the cases the read needs beside them.
TEACH = "ASSEMBLED PROOFS TEACH"                                    # clear of zero above, and outside the noise floor
COST = "THE ASSEMBLED PROOFS COST ONE-SHOT ATTEMPTS"                # clear of zero below, and outside the noise floor
NOT_SHOWN = "NOT SHOWN AT H0'S SIZE"                                # the interval holds zero; a gain of the ceiling's size per proof could have been seen
UNDETECTABLE = "UNDETECTABLE AT THIS SIZE"                          # the interval holds zero; such a gain could not have been seen here
WITHIN_THE_NOISE = "WITHIN WHAT TWO TRAININGS ON THE SAME DATA DIFFER BY"      # clear of zero, inside the noise floor: no effect is read
INCONCLUSIVE = "INCONCLUSIVE"                                       # a check failed: the run could not have seen a win
NOT_READ = "NOT READ"                                               # the primary, or the noise floor, has nothing to be read on (a smoke run)
BRANCHES = (TEACH, COST, NOT_SHOWN, UNDETECTABLE, WITHIN_THE_NOISE, INCONCLUSIVE, NOT_READ)


# ------------------------------------------------------------------------------------------------ the rows
def _example(row: Mapping) -> dict:
    return {field: row[field] for field in EXAMPLE_FIELDS}


def round_rows(by_round: Sequence[Sequence[Mapping]]) -> list[dict]:
    """The training examples the rounds stored, concatenated in the order given (round 1's, then 2's, then 3's):
    each a training row whose id is its attempt's. Refused when an id, or a problem, stands twice: a round trains on
    one proof a problem."""
    rows = [{"id": row["attempt_id"], "source": ROUNDS, **_example(row)} for rows in by_round for row in rows]
    for what, counts in (("attempt", Counter(row["id"] for row in rows)), ("problem", Counter(row["problem_id"] for row in rows))):
        twice = sorted(key for key, count in counts.items() if count > 1)
        if twice:
            raise ValueError(f"{len(twice)} {what}s stand twice in the rounds' training examples (first: {twice[0]}): refused")
    return rows


def h0_id(problem_id: str) -> str:
    return f"{problem_id}#{H0}"


def h0_rows(harvest: Sequence[Mapping]) -> list[dict]:
    """The harvest's rows as training rows, in the file's order."""
    return [{"id": h0_id(row["problem_id"]), "source": H0, **_example(row)} for row in harvest]


def check_h0(harvest: Sequence[Mapping], heldout: Collection[str], base_map: Collection[str], trained_on: Collection[str], minimum: int) -> None:
    """What an H0 is refused for (ValueError), before anything is trained: a row that is not an assembled proof with
    its statement, a problem twice, a held-out problem (or one of the held-out groups), a base-map problem, a problem
    `without` trains on already, or fewer rows than `minimum` (with fewer, Step 1 is not run on it)."""
    lacking = [index for index, row in enumerate(harvest) if any(not row.get(field) for field in EXAMPLE_FIELDS) or row.get("assembled") is not True]
    if lacking:
        raise ValueError(f"{len(lacking)} rows of H0 are not assembled proofs with their statement (first: row {lacking[0]}; a row holds {list(EXAMPLE_FIELDS)} "
                         "and `assembled: true`). Refused")
    ids = [row["problem_id"] for row in harvest]
    twice = sorted(key for key, count in Counter(ids).items() if count > 1)
    if twice:
        raise ValueError(f"{len(twice)} problems stand twice in H0 (first: {twice[0]}): one proof a problem. Refused")
    for what, forbidden in (("held-out problems", heldout), ("problems of the base map", base_map),
                            ("problems the rounds' training examples hold already", trained_on)):
        found = sorted(set(ids) & set(forbidden))
        if found:
            raise ValueError(f"H0 holds {len(found)} {what} (first: {found[0]}). Refused")
    if len(ids) < minimum:
        raise ValueError(f"H0 holds {len(ids)} proofs and Step 1 asks for at least {minimum}: with fewer it is not run on it, and the harvest is enlarged "
                         "first. Refused")


# ---------------------------------------------------------------------------------------------- the orders
def seeded_order(rows: Sequence[Mapping], seed: int) -> list[Mapping]:
    """`without`'s order: the rows by a content hash of the seed and each row's id."""
    return sorted(rows, key=lambda row: rank(seed, ORDER_LABEL, row["id"]))


def with_h0(ordered: Sequence[Mapping], harvest: Sequence[Mapping], seed: int) -> list[Mapping]:
    """`with`'s order: `ordered` (`without`'s rows in `without`'s order, none moved) with every row of H0 put in at a
    seeded place. A place is one of the len(ordered) + 1 gaps (before the first row, between two, after the last),
    by a content hash of the seed and the row's id; the rows of H0 that fall in one gap stand there in the order of
    their hashes."""
    gaps: dict[int, list[Mapping]] = {}
    for row in sorted(harvest, key=lambda row: rank(seed, PLACE_LABEL, row["id"])):
        gaps.setdefault(int(rank(seed, PLACE_LABEL, row["id"]), 16) % (len(ordered) + 1), []).append(row)
    placed: list[Mapping] = []
    for gap in range(len(ordered) + 1):
        placed.extend(gaps.get(gap, ()))
        if gap < len(ordered):
            placed.append(ordered[gap])
    return placed


# ------------------------------------------------------------------------------------ how a model writes
def first_step(completion: str) -> str:
    """A proof's first step: its first line that is neither empty nor a comment (as a proof's lines are counted),
    without its indentation. Empty when it has none."""
    return next((line.strip() for line in completion.split("\n") if line.strip() and not line.strip().startswith("--")), "")


def opens_with_a_have(attempts: Sequence[Mapping]) -> dict:
    """Of a model's attempts (every one, whatever became of it), the share whose first step is a `have`."""
    count = sum(bool(HAVE.match(first_step(attempt.get("completion") or ""))) for attempt in attempts)
    return {"attempts": len(attempts), "with_a_have": count, "share": round(count / len(attempts), 5) if attempts else None}


# ---------------------------------------------------------------------------------------------- the checks
def tenth(rows: int, share: float) -> int:
    """How many rows `share` of a training's rows is: rounded down, and at least one."""
    return max(1, int(rows * share))


def both_took(row_losses: Mapping[str, Sequence[float]], share: float) -> dict:
    """Check 2: both trainings took. For each, the mean training loss over the last `share` of its rows (a tenth)
    is below the mean over the first (`the_training_took`: a row's loss is read before the update of its step)."""
    each = {arm: the_training_took(row_losses[arm], tenth(len(row_losses[arm]), share)) for arm in ARMS}
    return {"what": f"for each of the two trainings, the mean training loss over the last {share:g} of its rows is below the mean over the first {share:g}",
            **each, "passes": all(entry["passes"] for entry in each.values())}


def both_write_proofs(rung_rows: Mapping[str, Sequence[Mapping]], maximum: float) -> dict:
    """Check 3: both models still write proofs (`still_writes_proofs`: under `maximum` of a model's attempts on the
    three rungs reached the token cap or got no verdict from Lean)."""
    each = {arm: still_writes_proofs(rung_rows[arm], maximum) for arm in ARMS}
    return {"what": f"for each of the two models, the share of its attempts on the three rungs without an answer is under {maximum}",
            **each, "passes": all(entry["passes"] for entry in each.values())}


def trained_on_h0(orders: Mapping[str, Sequence[Mapping]], trained: Mapping[str, Sequence[str]]) -> dict:
    """Check 4: `with` was trained on H0. `orders`: each arm's rows as the prepare step fixed them (`id`, `source`).
    `trained`: each arm's row ids as its training step recorded them, one for each example an optimizer step was
    made on, in the order trained. Every row of H0 is counted once in `with`'s record and none in `without`'s, and
    each record is the prepared order and nothing else."""
    harvest = [row["id"] for row in orders[WITH] if row["source"] == H0]
    counts = {arm: Counter(trained[arm]) for arm in ARMS}
    as_prepared = {arm: list(trained[arm]) == [row["id"] for row in orders[arm]] for arm in ARMS}
    once = sum(counts[WITH][key] == 1 for key in harvest)
    in_without = sum(counts[WITHOUT][key] for key in harvest)
    return {"what": "every row of H0 is counted once in the record of the rows `with` was trained on, none is in `without`'s, and each record is the order "
                    "the prepare step fixed",
            "h0_rows": len(harvest), "counted_once_in_with": once, "counted_in_without": in_without,
            "rows_trained": {arm: len(trained[arm]) for arm in ARMS}, "rows_prepared": {arm: len(orders[arm]) for arm in ARMS}, "as_prepared": as_prepared,
            "passes": bool(harvest) and once == len(harvest) and in_without == 0 and all(as_prepared.values())}


def can_this_run_see_a_win(h0_count: int, minimum: int, row_losses: Mapping[str, Sequence[float]], rung_rows: Mapping[str, Sequence[Mapping]],
                           orders: Mapping[str, Sequence[Mapping]], trained: Mapping[str, Sequence[str]], settings: Mapping) -> dict:
    """The four checks the spec reads before the branch (any failing: INCONCLUSIVE)."""
    return {"h0_holds_enough_proofs": {"what": f"H0 holds at least {minimum} proofs", "h0_rows": h0_count, "minimum": minimum, "passes": h0_count >= minimum},
            "both_trainings_took": both_took(row_losses, settings["loss_share_of_rows"]),
            "both_models_still_write_proofs": both_write_proofs(rung_rows, settings["maximum_share_without_an_answer"]),
            "with_was_trained_on_h0": trained_on_h0(orders, trained)}


# ---------------------------------------------------------------------------------------------- the branch
def noise_band(noise_floor: Mapping | None) -> float | None:
    """What two trainings on the same data differ by, as ONE size: the end of the noise floor's interval farthest
    from zero. Which of the two trainings is subtracted from the other is arbitrary, so the floor has no sign: a
    primary is outside it only when its point is farther from zero than this."""
    if not noise_floor or noise_floor.get("mean") is None:
        return None
    return max(abs(noise_floor["low"]), abs(noise_floor["high"]))


def could_have_been_seen(primary: Mapping, band: float, h0_count: int, doses: Mapping[str, Mapping] | None) -> dict:
    """The note beside an interval that holds zero. `doses`: the ceiling's models, each `{rows, mean}` (its primary:
    the same quantity against the base, for `rows` published proofs trained on). A gain of the ceiling's size for
    each proof is `mean / rows`; H0's proofs would then give `h0_count` times it. It COULD HAVE BEEN SEEN here when
    that reaches what this run resolves: half the width of the primary's interval, and the noise band."""
    resolves = round(max((primary["high"] - primary["low"]) / 2, band), 5)
    if not doses:
        return {"what": "the ceiling's two doses were not read: whether a gain of the ceiling's size per proof could have been seen here is not said",
                "resolves": resolves, "doses": None, "seen": None}
    each = {}
    for name, dose in doses.items():
        per_proof = dose["mean"] / dose["rows"]
        each[name] = {"rows": dose["rows"], "gain": dose["mean"], "gain_per_proof": per_proof, "expected_here": round(per_proof * h0_count, 6),
                      "could_have_been_seen": per_proof * h0_count >= resolves}
    return {"what": "a gain of the size the ceiling shows for each proof (its primary over the proofs it was trained on, at each dose), times H0's proofs, "
                    "against what this run resolves (half the width of the primary's interval, or the noise band when that is larger)",
            "resolves": resolves, "h0_rows": h0_count, "doses": each, "seen": any(entry["could_have_been_seen"] for entry in each.values())}


def l3d1_branch(checks: Mapping, primary: Mapping, noise_floor: Mapping | None, h0_count: int, doses: Mapping[str, Mapping] | None = None) -> dict:
    """The branch, fixed before the run. `primary`: `with` minus `without` on the goal problems of 4 lines or more
    (`mean`, `low`, `high`, `problems`). `noise_floor`: `without` minus the stored three-round model, the same
    quantity. `doses`: the ceiling's two models, when its report was read (`could_have_been_seen`).

    A check failed: INCONCLUSIVE, and nothing else is said. The interval clear of zero and the point outside the
    noise floor: TEACH above zero, COST below. Clear of zero and inside the noise floor: no effect is read. The
    interval holds zero: NOT SHOWN AT H0'S SIZE, or UNDETECTABLE AT THIS SIZE when a gain of the ceiling's size per
    proof could not have been seen here."""
    failed = [name for name, check in checks.items() if not check["passes"]]
    if failed:
        return {"name": INCONCLUSIVE, "failed_checks": failed,
                "reason": f"this run could not have seen a win ({', '.join(failed)}): nothing is said about what assembled proofs teach"}
    if primary.get("mean") is None:
        return {"name": NOT_READ, "failed_checks": [], "reason": "no goal problem with a published proof of 4 lines or more was measured: the primary has nothing to be read on"}
    band = noise_band(noise_floor)
    if band is None:
        return {"name": NOT_READ, "failed_checks": [], "reason": "no stored three-round model was read: there is no noise floor, and no primary is read as an effect without one"}
    mean, low, high = primary["mean"], primary["low"], primary["high"]
    said = {"failed_checks": [], "noise_band": round(band, 5)}
    if low > 0 or high < 0:
        if abs(mean) <= band:
            return {"name": WITHIN_THE_NOISE, **said,
                    "reason": f"the interval is clear of zero and the point ({mean:+.5f}) is within what two trainings on the same data differ by here (up to "
                              f"{band:.5f} either way): it is not read as an effect"}
        if low > 0:
            return {"name": TEACH, **said,
                    "reason": f"the interval is clear of zero and the point ({mean:+.5f}) is outside the noise floor ({band:.5f}): assembled proofs teach. A "
                              "matched control is trained first (`without` and as many more ONE-SHOT proofs as H0 holds, from verified attempts stored in "
                              "other runs at pool problems outside the training set), to tell assembled proofs from more proofs; then Step 2 goes to three seeds"}
        return {"name": COST, **said,
                "reason": f"the interval lies below zero and the point ({mean:+.5f}) is outside the noise floor ({band:.5f}): the assembled proofs cost one-shot "
                          "attempts. Step 2 is not run as designed"}
    note = could_have_been_seen(primary, band, h0_count, doses)
    if note["seen"] is False:
        return {"name": UNDETECTABLE, **said, "could_have_been_seen": note,
                "reason": f"the interval holds zero, and a gain of the size the ceiling shows for each proof could not have been seen here (this run resolves "
                          f"{note['resolves']:.5f}): undetectable at this size. What follows (a larger harvest by generation, or none) is decided on the ceiling's read"}
    return {"name": NOT_SHOWN, **said, "could_have_been_seen": note,
            "reason": "the interval holds zero: not shown at H0's size"
                      + (f"; a gain of the size the ceiling shows for each proof could have been seen here (this run resolves {note['resolves']:.5f})"
                         if note["seen"] else "; the ceiling's two doses were not read, so whether such a gain could have been seen here is not said")}
