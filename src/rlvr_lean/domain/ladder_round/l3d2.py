"""L3d Step 2: the loop with assembly in the round, and a twin trained without the assembled proofs. Spec:
docs/spec/ladder-loop.spec.md, "L3d: train on what the episode reaches", "Step 2, made exact before it is built"
("The twin", "Primary", "Secondary", "Branches", "Can this run see a win") and "Made exact by the build (Step 2)".
Pure: rows in, verdicts out. The round's own rules are `assembly.py`; the checks shared with Step 1 are `l3d.py`'s.

`with` is the model of the arm's last round; `without` is trained from the base on the rounds' one-shot rows alone, in
`with`'s order with the other rows left out. What differs between them is the assembled proofs.

  the checks    `can_this_run_see_a_win`: `with` was trained on enough assembled proofs that `without` was not; both
                trainings took; both models still write proofs; every assembled row stands once in `with`'s record
                and none in `without`'s
  the branch    `the_note` (could a gain of the ceiling's size for each assembled proof have been seen), `l3d2_branch`
  by round      `round_table`: what the attempts resolved, what only assembly resolved, the picks' pass rate
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.assembly import ASSEMBLED, ATTEMPT, FROM_ASSEMBLY, H0, counted
from rlvr_lean.domain.ladder_round.l3d import ARMS, WITH, WITHOUT, both_took, both_write_proofs, trained_on_h0
from rlvr_lean.domain.problem_pool.episodes import NEGATION_SIDE, reward

L3D2 = "l3d2"                                   # what every file, set and printed line the stage adds is called
TEACH = "ASSEMBLED PROOFS TEACH"                                    # the interval clear of zero and above
NOT_SHOWN = "NOT SHOWN"                                             # holds zero; a gain of the ceiling's size for each proof could have been seen
UNDETECTABLE = "UNDETECTABLE AT THIS SIZE"                          # holds zero; such a gain could not have been seen here
COST = "THE ASSEMBLED PROOFS COST ONE-SHOT ATTEMPTS"                # the interval below zero
INCONCLUSIVE = "INCONCLUSIVE"                                       # a check failed: the run could not have seen a win
NOT_READ = "NOT READ"                                               # the primary has nothing to be read on (a smoke run)
BRANCHES = (TEACH, NOT_SHOWN, UNDETECTABLE, COST, INCONCLUSIVE, NOT_READ)


# ---------------------------------------------------------------------------------------------- the checks
def as_step_1_reads(orders: Mapping[str, Sequence[Mapping]]) -> dict[str, list[dict]]:
    """Each model's prepared rows (`id`, `origin`) as Step 1's fourth check reads rows: an assembled row (a
    round's or H0's) is what an H0 row was there."""
    return {arm: [{"id": row["id"], "source": "h0" if row["origin"] in FROM_ASSEMBLY else "rounds"} for row in orders[arm]] for arm in ARMS}


def can_this_run_see_a_win(orders: Mapping[str, Sequence[Mapping]], trained: Mapping[str, Sequence[str]], row_losses: Mapping[str, Sequence[float]],
                           rung_rows: Mapping[str, Sequence[Mapping]], settings: Mapping, minimum: int) -> dict:
    """The four checks the spec reads before the branch (any failing: INCONCLUSIVE). `orders`: each model's rows as
    they were prepared (`id`, `origin`), in order. `trained`: each model's row ids as its training loop recorded
    them. `minimum`: the least number of assembled proofs `with` must have been trained on and `without` not."""
    record = trained_on_h0(as_step_1_reads(orders), trained)
    assembled = [row["id"] for row in orders[WITH] if row["origin"] in FROM_ASSEMBLY]
    in_with, in_without = set(trained[WITH]), set(trained[WITHOUT])
    only_with = sum(key in in_with and key not in in_without for key in assembled)
    by_origin = {origin: sum(row["origin"] == origin and row["id"] in in_with and row["id"] not in in_without for row in orders[WITH]) for origin in FROM_ASSEMBLY}
    return {"with_was_trained_on_enough_assembled_proofs": {
                "what": f"`with` was trained on at least {minimum} assembled proofs (H0's and the rounds') that `without` was not, by the two training records",
                "assembled_proofs": only_with, "from_the_rounds": by_origin[ASSEMBLED], "from_h0": by_origin[H0], "minimum": minimum, "passes": only_with >= minimum},
            "both_trainings_took": both_took(row_losses, settings["loss_share_of_rows"]),
            "both_models_still_write_proofs": both_write_proofs(rung_rows, settings["maximum_share_without_an_answer"]),
            "every_assembled_row_is_in_withs_record_once_and_none_in_withouts": {
                "what": "every assembled row stands once in the record of the rows `with` was trained on, none stands in `without`'s, and each record is the "
                        "order that was prepared (`without`'s: `with`'s with the assembled rows left out)",
                "assembled_rows": record["h0_rows"], "counted_once_in_with": record["counted_once_in_with"], "counted_in_without": record["counted_in_without"],
                "rows_trained": record["rows_trained"], "rows_prepared": record["rows_prepared"], "as_prepared": record["as_prepared"], "passes": record["passes"]}}


# ---------------------------------------------------------------------------------------------- the branch
def the_note(primary: Mapping, assembled_proofs: int, gain_of_a_proof: float) -> dict:
    """Beside an interval that holds zero: the ceiling's figure for one published proof (`gain_of_a_proof`, per
    attempt on this quantity) times the assembled proofs `with` was trained on and `without` was not is the gain
    expected here if an assembled proof were worth a published one. It COULD HAVE BEEN SEEN when it reaches half
    the width of the primary's interval."""
    resolves, expected = (primary["high"] - primary["low"]) / 2, gain_of_a_proof * assembled_proofs
    return {"what": "the ceiling's gain for each published proof, times the assembled proofs `with` was trained on and `without` was not, against half the width "
                    "of the primary's interval", "gain_of_a_published_proof": gain_of_a_proof, "assembled_proofs": assembled_proofs,
            "expected_here": round(expected, 6), "resolves": round(resolves, 6), "seen": expected >= resolves}


def l3d2_branch(checks: Mapping, primary: Mapping, assembled_proofs: int, gain_of_a_proof: float) -> dict:
    """The branch, fixed before the run. `primary`: `with` minus `without` on the goal problems of 4 lines or more.
    A check failed: INCONCLUSIVE, and nothing else is said. The interval above zero: ASSEMBLED PROOFS TEACH. Below
    zero: they cost one-shot attempts. It holds zero (one that ends at zero holds it): NOT SHOWN when a gain of the
    ceiling's size for each assembled proof could have been seen here, UNDETECTABLE AT THIS SIZE when it could not."""
    failed = [name for name, check in checks.items() if not check["passes"]]
    if failed:
        return {"name": INCONCLUSIVE, "failed_checks": failed,
                "reason": f"this run could not have seen a win ({', '.join(failed)}): nothing is said about what assembled proofs teach"}
    if primary.get("mean") is None:
        return {"name": NOT_READ, "failed_checks": [], "reason": "no goal problem with a published proof of 4 lines or more was measured: the primary has nothing to be read on"}
    mean, low, high = primary["mean"], primary["low"], primary["high"]
    if low > 0:
        return {"name": TEACH, "failed_checks": [],
                "reason": f"the interval is clear of zero and above ({mean:+.5f} [{low:+.5f}, {high:+.5f}]): assembled proofs teach, at the rate the loop makes them. "
                          "Two more seeds, and the loop with assembly is the loop from here"}
    if high < 0:
        return {"name": COST, "failed_checks": [],
                "reason": f"the interval lies below zero ({mean:+.5f} [{low:+.5f}, {high:+.5f}]): the assembled proofs cost one-shot attempts"}
    note = the_note(primary, assembled_proofs, gain_of_a_proof)
    could = (f"a published proof of the ceiling's first 2,000 was worth about {gain_of_a_proof:+.6f} per attempt here; {assembled_proofs:,} assembled proofs at that "
             f"would give {note['expected_here']:+.6f}, and this run resolves {note['resolves']:.6f}")
    if note["seen"]:
        return {"name": NOT_SHOWN, "failed_checks": [], "could_have_been_seen": note,
                "reason": f"the interval holds zero, and a gain of the ceiling's size for each assembled proof could have been seen ({could}): not shown. Assembled "
                          "proofs teach less than published ones, each"}
    return {"name": UNDETECTABLE, "failed_checks": [], "could_have_been_seen": note,
            "reason": f"the interval holds zero, and a gain of the ceiling's size for each assembled proof could not have been seen ({could}): undetectable at this size"}


# ------------------------------------------------------------------------------------------------ by round
def round_table(rounds: Mapping[int, Mapping], target_rate: float) -> list[dict]:
    """One row a round: `rounds[number]` holds its `results` (every batch's per-problem rows, with
    `resolved_by_assembly`), its `examples` (the round's training examples, with `origin`) and its `assembly`
    counts. The problems an attempt resolved, the problems only assembly resolved and their share of what the
    attempts left unresolved (does the yield grow as the model is trained), the picks' mean pass rate (verified
    attempts over attempts: the solver's, with no assembled problem counted), the mean reward as the challenger
    reads k, and the share of the round's own training rows that are refutations."""
    table = []
    for number in sorted(rounds):
        results, examples = rounds[number]["results"], rounds[number]["examples"]
        own = [row for row in examples if row["origin"] != H0]
        unresolved, by_assembly = sum(row["resolved"] == 0 for row in results), sum(bool(row.get("resolved_by_assembly")) for row in results)
        rewards = [reward(row["resolved"], row["episodes"], target_rate) for row in counted(results)]
        share = lambda count, of: round(count / of, 5) if of else None      # noqa: E731
        table.append({"round": number, "picks": len(results), "resolved_by_an_attempt": sum(row["resolved"] > 0 for row in results),
                      "left_unresolved_by_the_attempts": unresolved, "only_assembly_resolved": by_assembly, "share_of_the_unresolved": share(by_assembly, unresolved),
                      "mean_pass_rate": share(sum(row["resolved"] / row["episodes"] for row in results), len(results)),
                      "mean_reward": share(sum(rewards), len(rewards)),
                      "training_rows": {ATTEMPT: sum(row["origin"] == ATTEMPT for row in own), ASSEMBLED: sum(row["origin"] == ASSEMBLED for row in own),
                                        H0: len(examples) - len(own)},
                      "share_of_refutations": share(sum(row["side"] == NEGATION_SIDE for row in own), len(own)),
                      "assembly_lean_checks": (rounds[number].get("assembly") or {}).get("lean_checks")})
    return table
