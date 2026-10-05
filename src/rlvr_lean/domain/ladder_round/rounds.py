"""L2: three rounds of the challenger arm, then the equal-compute control. Spec: docs/spec/ladder-loop.spec.md,
"L2: three rounds". Pure (numpy only): rows in, rows out.

NAMING, the one this code and the report use. Rounds are numbered 1, 2, 3, as L2's read numbers them ("from round 1
to round 3"). Round r starts from M(r - 1) and trains M(r); M(0) is the base model. So M(1) is the model after one
round and M(3) the model after three. ("A round" counts the same rounds from 0: its "round r starts from M(r)".)

  proposals    a round's `round.problems` proposals are made in `round.batches` EQUAL batches, each with its share of
               random places, from every candidate not proposed before, less H and the base map's problems
  the refit    after each batch's episodes the challenger is refitted on everything seen so far: the base map and
               every finished batch. The current round's results weigh 1, each earlier round's
               `challenger.recency_decay` per round. The base map is the BASE model's results and round 1 starts
               from the base, so the base map is as old as round 1's own results
  the control  the base gets the loop's attempt episodes on G, spread evenly over its problems
  the read     fixed before the run: each round's rungs minus the base, the climb, reach as gained against lost with
               a two-sided sign test, the control, the challenger's table by batch and by round, the branch
  added        (2026-10-05, after seeds 0 and 1) the last model gets the control's extra attempts too, with the
               control's sampling seed: the two models at the same number of attempts (`equal_attempts_read`)
"""

from __future__ import annotations

import math
from typing import Collection, Mapping, Sequence

import numpy as np

from rlvr_lean.domain.ladder_round.challenger import RANDOM_PLACE, SCORED, binomial_deviance, calibration_bins, propose, recency_weights
from rlvr_lean.domain.ladder_round.read import ESCALATE, STOP_AND_DIAGNOSE, VOID, Branch, group_ids, paired_change, stop_rule
from rlvr_lean.domain.problem_pool.certificates import FALSE_SIDE
from rlvr_lean.domain.problem_pool.episodes import ABOVE_BAND, BELOW_BAND, IN_BAND, NEGATION_SIDE, RUNGS, band, reward

FIRST_ROUND = 1
BASE_MAP_ROUND = FIRST_ROUND        # the base map is M(0)'s results, as round 1's own are: in round r it is r - 1 rounds old
BRANCHES = (ESCALATE, STOP_AND_DIAGNOSE, VOID)


def model_name(round_number: int) -> str:
    """`M(2)`: the model round 2 trained (and round 3 starts from). `M(0)` is the base."""
    return f"M({round_number})"


# ----------------------------------------------------------------------------------------------- proposals
def batch_sizes(problems: int, batches: int) -> list[int]:
    """A round's proposals in `batches` EQUAL batches. A number of problems the batches do not divide is refused."""
    if batches < 1 or problems < batches:
        raise ValueError(f"a round of {problems} problems cannot be proposed in {batches} batches")
    if problems % batches:
        raise ValueError(f"round.batches ({batches}) does not divide round.problems ({problems}): the batches must be equal")
    return [problems // batches] * batches


def propose_batch(candidate_ids: Sequence[str], predicted_rates: Sequence[float], scores: Sequence[float], proposed: Collection[str],
                  barred: Collection[str], places: int, random_share: float, seed: int) -> list[dict]:
    """One batch's proposals. The candidates it may choose from are every pool problem NOT YET PROPOSED (in any
    batch of any round), less H and the base map's problems (`barred`); among those it is `propose`: the best by
    expected reward, with `random_share` of the batch's places given to a seeded random draw of the others. Each
    row carries the pass rate the challenger predicted for the problem BEFORE its episodes: the calibration is
    read on these. Every proposed problem then gets its n episodes; nothing here sees a measured rate."""
    if not len(candidate_ids) == len(predicted_rates) == len(scores):
        raise ValueError("one predicted rate and one score per candidate is needed")
    taken, kept_out = set(proposed), set(barred)
    still_open = [index for index, problem_id in enumerate(candidate_ids) if problem_id not in taken and problem_id not in kept_out]
    chosen = propose([candidate_ids[index] for index in still_open], [scores[index] for index in still_open], places, random_share, seed)
    rate_of = {candidate_ids[index]: float(predicted_rates[index]) for index in still_open}
    return [{**row, "predicted_rate": round(rate_of[row["problem_id"]], 6)} for row in chosen]


# ----------------------------------------------------------------------------------------------- the refit
def refit_observations(base_map: Sequence[Mapping], finished: Sequence[Mapping], current_round: int, decay: float) -> dict:
    """Everything the challenger has seen when a batch of round `current_round` is proposed: the base map
    (`problem_id`, `resolved`, `episodes`) and the results of every FINISHED batch (the same, with the `round`
    whose model attempted them). `weights`: 1 for the current round's results, `decay` per round for an earlier
    round's, and the base map by its age (it is as old as round 1). No problem is observed twice."""
    if current_round < FIRST_ROUND:
        raise ValueError(f"rounds are numbered from {FIRST_ROUND}, got {current_round}")
    rows = [(row["problem_id"], row["resolved"], row["episodes"], BASE_MAP_ROUND) for row in base_map]
    for row in finished:
        if row["round"] < FIRST_ROUND:
            raise ValueError(f"the result of {row['problem_id']} names round {row['round']}: rounds are numbered from {FIRST_ROUND}")
        rows.append((row["problem_id"], row["resolved"], row["episodes"], row["round"]))
    problem_ids = [row[0] for row in rows]
    if len(set(problem_ids)) != len(problem_ids):
        raise ValueError("a problem is observed twice: no problem is proposed twice, and none of the base map is proposed")
    rounds = [row[3] for row in rows]
    return {"problem_ids": problem_ids, "resolved": np.array([row[1] for row in rows]), "episodes": np.array([row[2] for row in rows]),
            "rounds": rounds, "weights": recency_weights(rounds, current_round, decay)}


def weights_summary(observations: Mapping) -> dict:
    """What a refit was given, for the record: how many observations each round's model gave, and their weight."""
    summary: dict[str, dict] = {}
    for observed_in, weight in zip(observations["rounds"], observations["weights"]):
        entry = summary.setdefault(str(observed_in), {"observations": 0, "weight": round(float(weight), 6)})
        entry["observations"] += 1
    return summary


# --------------------------------------------------------------------------------------------- the control
def control_episodes(loop_attempt_episodes: int, goal_problems: int) -> int:
    """The equal-compute control: the base gets the loop's attempt episodes on G, the same number on every
    problem (what does not divide is not spent). Three rounds of 1,000 problems at 8 episodes are 24,000, which
    is 61 on each of G's 392 problems, on top of the 32 of its fresh reach measurement."""
    if loop_attempt_episodes < 0 or goal_problems < 0:
        raise ValueError(f"episodes and problems cannot be negative, got {loop_attempt_episodes} and {goal_problems}")
    return loop_attempt_episodes // goal_problems if goal_problems else 0


def summed_budget(first: Sequence[Mapping], more: Sequence[Mapping]) -> list[dict]:
    """One model's results over two samplings of the same problems, added: the base's 32 fresh episodes on G and
    the control's 61 more are its 93. Both must hold the same problems."""
    extra = {row["problem_id"]: row for row in more}
    if set(extra) != {row["problem_id"] for row in first} or len(extra) != len(more):
        raise ValueError("the two samplings do not hold the same problems")
    return [{"problem_id": row["problem_id"], "side": row.get("side"), "resolved": row["resolved"] + extra[row["problem_id"]]["resolved"],
             "episodes": row["episodes"] + extra[row["problem_id"]]["episodes"]} for row in first]


# ------------------------------------------------------------------------------------------------ the read
def sign_test(gained: int, lost: int) -> float | None:
    """The two-sided exact sign test: if a problem that changed were as likely to be gained as lost, the chance of
    a split at least this uneven. None when no problem changed."""
    if gained < 0 or lost < 0:
        raise ValueError(f"counts cannot be negative, got {gained} and {lost}")
    changed = gained + lost
    if changed == 0:
        return None
    tail = sum(math.comb(changed, count) for count in range(min(gained, lost) + 1)) / 2 ** changed
    return float(f"{min(1.0, 2 * tail):.6g}")       # six significant digits: a very small chance is not rounded to zero


def gained_lost(after: Sequence[Mapping], before: Sequence[Mapping]) -> dict:
    """Reach as gained against lost, problem by problem: the problems `after` resolves at least once and `before`
    does not (gained), against the reverse (lost), with a two-sided sign test. Both sides must hold the same
    problems; they need not have the same number of episodes (the control compares 32 with 93)."""
    after_ids, before_ids = {row["problem_id"] for row in after}, {row["problem_id"] for row in before}
    if after_ids != before_ids or len(after_ids) != len(after) or len(before_ids) != len(before):
        raise ValueError("the two sides of a reach comparison do not hold the same problems")
    by_after = {row["problem_id"] for row in after if row["resolved"] > 0}
    by_before = {row["problem_id"] for row in before if row["resolved"] > 0}
    gained, lost = sorted(by_after - by_before), sorted(by_before - by_after)
    return {"problems": len(after), "resolved_after": len(by_after), "resolved_before": len(by_before), "by_both": len(by_after & by_before),
            "gained": len(gained), "lost": len(lost), "sign_test_p": sign_test(len(gained), len(lost)),
            "episodes_after": after[0]["episodes"] if after else None, "episodes_before": before[0]["episodes"] if before else None,
            "resolved_after_known_false": sum(row.get("side") == FALSE_SIDE and row["resolved"] > 0 for row in after),
            "gained_problems": gained, "lost_problems": lost}


def control_read(last_reach: Sequence[Mapping], base_reach: Sequence[Mapping], control: Sequence[Mapping]) -> dict:
    """The equal-compute control's read: the problems the last model resolves in its reach episodes against the
    base in its fresh reach episodes PLUS the control's, as counts and as gained against lost."""
    in_all = summed_budget(base_reach, control)
    return {"what": "problems of G the last model resolves in its reach episodes, against the base in its fresh reach episodes plus the control's",
            "resolved_by_the_base_in_its_fresh_episodes": sum(row["resolved"] > 0 for row in base_reach),
            "resolved_by_the_base_in_the_control_episodes": sum(row["resolved"] > 0 for row in control),
            "control_episodes_each": control[0]["episodes"] if control else None,
            **gained_lost(last_reach, in_all)}


def equal_attempts_read(last_reach: Sequence[Mapping], last_more: Sequence[Mapping], base_reach: Sequence[Mapping], control: Sequence[Mapping]) -> dict:
    """The same extra attempts for the trained model (spec, "Added 2026-10-05"): the control charges the loop for
    its training attempts, the last model with its reach episodes against the base with those AND the control's.
    Here both have the same number on every problem of G: the last model's reach episodes plus its own extra
    ones (`last_more`, sampled with the control's seed), against the base's fresh reach episodes plus the
    control's. Problems resolved as counts and as gained against lost with the two-sided sign test, and each
    model's successful episodes and successes per attempt (an attempt is one episode). Refused when the two
    sides do not have the same number of attempts on every problem."""
    last_all, base_all = summed_budget(last_reach, last_more), summed_budget(base_reach, control)
    attempts_of = {row["problem_id"]: row["episodes"] for row in base_all}
    uneven = sorted(row["problem_id"] for row in last_all if attempts_of.get(row["problem_id"]) != row["episodes"])
    if uneven:
        raise ValueError(f"{len(uneven)} problems do not have the same number of attempts from both models (first: {uneven[0]}): "
                         "this read compares them at equal attempts")

    def tally(rows: Sequence[Mapping]) -> tuple[int, int, float | None]:
        successes, attempts = sum(row["resolved"] for row in rows), sum(row["episodes"] for row in rows)
        return successes, attempts, round(successes / attempts, 6) if attempts else None

    (last_successes, last_attempts, last_rate), (base_successes, base_attempts, base_rate) = tally(last_all), tally(base_all)
    return {"what": "problems of G each model resolves in the SAME number of attempts: the last model in its reach episodes plus the control's number "
                    "again, against the base in its fresh reach episodes plus the control's",
            "extra_episodes_each": control[0]["episodes"] if control else None,
            **gained_lost(last_all, base_all),
            "successful_episodes_of_the_last_model": last_successes, "successful_episodes_of_the_base": base_successes,
            "attempts_of_the_last_model": last_attempts, "attempts_of_the_base": base_attempts,
            "successes_per_attempt_of_the_last_model": last_rate, "successes_per_attempt_of_the_base": base_rate}


def picks_calibration(rates: Sequence[float], resolved: Sequence[int], episodes: Sequence[int]) -> dict:
    """How the pass rates the challenger predicted for its picks, BEFORE their episodes, read against what was
    measured: the observed rate by predicted-rate bin, and the deviance against the one-rate-for-all prediction."""
    rates = np.asarray(rates, dtype=np.float64)
    resolved, episodes = np.asarray(resolved, dtype=np.float64), np.asarray(episodes, dtype=np.float64)
    if len(rates) == 0:
        return {"problems": 0}
    observed = float(resolved.sum() / episodes.sum())
    return {"problems": len(rates), "mean_predicted_rate": round(float(rates.mean()), 4), "observed_rate": round(observed, 4),
            "bins": calibration_bins(rates, resolved, episodes),
            "deviance": round(binomial_deviance(rates, resolved, episodes), 5),
            "deviance_of_the_mean_rate": round(binomial_deviance(np.full(len(rates), observed), resolved, episodes), 5)}


def _picks(proposals: Sequence[Mapping], results: Mapping[str, Mapping], target_rate: float, floor: float) -> dict:
    if not proposals:
        return {"picks": 0}
    low, high = band(target_rate, floor)
    rows = [results[row["problem_id"]] for row in proposals]
    rates = [row["resolved"] / row["episodes"] for row in rows]
    share = lambda count: round(count / len(rows), 5)      # noqa: E731
    return {"picks": len(rows), "known_false": sum(row["side"] == FALSE_SIDE for row in rows),
            "share_known_false": share(sum(row["side"] == FALSE_SIDE for row in rows)),
            "mean_pass_rate": round(sum(rates) / len(rates), 5),
            "share_at_k_0": share(sum(row["resolved"] == 0 for row in rows)),
            "share_below_the_band": share(sum(0 < rate < low for rate in rates)),
            "share_in_the_band": share(sum(low <= rate <= high for rate in rates)),
            "share_above_the_band": share(sum(rate > high for rate in rates)),
            # Classes that do NOT move with the target rate (the band does): what two arms of the loop are compared on.
            "share_at_k_1_to_3": share(sum(1 <= row["resolved"] <= 3 for row in rows)),
            "share_at_k_4_or_more": share(sum(row["resolved"] >= 4 for row in rows)),
            "mean_reward": round(sum(reward(row["resolved"], row["episodes"], target_rate) for row in rows) / len(rows), 5),
            "mean_predicted_rate": round(sum(row["predicted_rate"] for row in proposals) / len(proposals), 5),
            "mean_expected_reward": round(sum(row["score"] for row in proposals) / len(proposals), 5),
            "calibration": picks_calibration([row["predicted_rate"] for row in proposals], [row["resolved"] for row in rows], [row["episodes"] for row in rows])}


def picks_row(proposals: Sequence[Mapping], results: Sequence[Mapping], examples: Sequence[Mapping], target_rate: float, floor: float) -> dict:
    """One row of the challenger's table, for one batch or one round. `proposals`: its picks (`how`, the
    `predicted_rate` and `score` at the time they were chosen); `results`: their k of n, with the side their
    certificate proves; `examples`: the training examples their verified proofs gave.

    Each quantity is given for the SCORED picks (the places the challenger chose by expected reward: the
    trajectory is read on these), for the random places, and for all of them: the share that are known false,
    the mean pass rate, the shares at k = 0, below, in and above the band, the shares at k = 1 to 3 and at k of
    4 or more (fixed classes: the band moves with the target rate), the mean reward, and the predictor's
    calibration. The training set: the share of its examples that come from problems above the band (k >= 4 of
    8 at t = 1/4) and from refutations (a proof of a negation)."""
    by_id = {row["problem_id"]: row for row in results}
    missing = [row["problem_id"] for row in proposals if row["problem_id"] not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} proposed problems have no result (first: {missing[0]}): every proposed problem gets its episodes")
    _, high = band(target_rate, floor)
    episodes = max((row["episodes"] for row in results), default=0)
    proposed = {row["problem_id"] for row in proposals}
    own = [example for example in examples if example["problem_id"] in proposed]
    above = sum(by_id[example["problem_id"]]["resolved"] / by_id[example["problem_id"]]["episodes"] > high for example in own)
    refutations = sum(example["side"] == NEGATION_SIDE for example in own)
    return {"scored": _picks([row for row in proposals if row["how"] == SCORED], by_id, target_rate, floor),
            "random_places": _picks([row for row in proposals if row["how"] == RANDOM_PLACE], by_id, target_rate, floor),
            "all": _picks(list(proposals), by_id, target_rate, floor),
            "training_set": {"examples": len(own), "from_problems_above_the_band": above, "from_refutations": refutations,
                             "share_from_problems_above_the_band": round(above / len(own), 5) if own else None,
                             "share_from_refutations": round(refutations / len(own), 5) if own else None,
                             "above_the_band_is_k_at_least": next((k for k in range(episodes + 1) if k / episodes > high), None) if episodes else None}}


def challenger_trajectory(by_round: Mapping[int, Mapping], target_rate: float) -> dict:
    """The challenger's trajectory (decisions 2 and 3 of the spec's L2 section), stated before the run: from the
    first round to the last, the known-false share of the SCORED picks falls and their mean pass rate moves toward
    the target. A last-round share at or above the first round's says the reward is not moving off the
    refutations: that is flagged for a decision before any further round (the key `needs_a_decision`)."""
    rounds = sorted(by_round)
    shares = {str(number): by_round[number]["scored"].get("share_known_false") for number in rounds}
    rates = {str(number): by_round[number]["scored"].get("mean_pass_rate") for number in rounds}
    result = {"known_false_share_of_the_scored_picks": shares, "mean_pass_rate_of_the_scored_picks": rates, "target_rate": target_rate,
              "compared": None, "the_known_false_share_falls": None, "the_mean_pass_rate_moves_toward_the_target": None, "needs_a_decision": None}
    if len(rounds) < 2 or None in (shares[str(rounds[0])], shares[str(rounds[-1])]):
        return result
    first, last = str(rounds[0]), str(rounds[-1])
    falls = shares[last] < shares[first]
    return {**result, "compared": [rounds[0], rounds[-1]], "the_known_false_share_falls": falls,
            "the_mean_pass_rate_moves_toward_the_target": abs(rates[last] - target_rate) < abs(rates[first] - target_rate),
            "needs_a_decision": not falls}


def l2_branch(rungs_minus_base: Mapping[int, Mapping[str, Mapping]], rounds: Sequence[int]) -> Branch:
    """The branch the numbers select at one seed, in the spec's order. `rungs_minus_base[r][rung]`: M(r) minus the
    base, paired by problem, for each round that was measured; `rounds`: the rounds the loop was to run.

    VOID first: round 1 with both the in-band and the above-band rung at or below zero did not train. Then the
    stop rule: an interval entirely below zero for ANY round's model minus the base on the below-band rung. Then
    the primary (the last model minus the base on that rung): at or above zero, or below zero with an interval
    that contains zero, escalates to three seeds. (A soundness alarm stops the stage before any read exists.)"""
    first, last = rounds[0], rounds[-1]
    if first not in rungs_minus_base:
        return Branch(VOID, f"round {first} was not measured: there is nothing to read")
    in_band, above = rungs_minus_base[first][IN_BAND], rungs_minus_base[first][ABOVE_BAND]
    if in_band.get("mean") is not None and above.get("mean") is not None and in_band["mean"] <= 0 and above["mean"] <= 0:
        return Branch(VOID, f"round {first} left both the in-band rung ({in_band['mean']}) and the above-band rung ({above['mean']}) at or below "
                            "zero: the round did not train. Fix and run again")
    for number in sorted(rungs_minus_base):
        below = rungs_minus_base[number][BELOW_BAND]
        if stop_rule(below):
            return Branch(STOP_AND_DIAGNOSE, f"{model_name(number)} minus the base on the below-band rung is {below['mean']} [{below['low']}, {below['high']}], "
                                             "entirely below zero: the loop stops for diagnosis. The run failed, and the idea is not judged until "
                                             "the verdict gates pass")
    primary = rungs_minus_base.get(last, {}).get(BELOW_BAND, {})
    if primary.get("mean") is None:
        return Branch(VOID, f"the primary could not be computed: {model_name(last)} was not measured on the below-band rung against the base")
    where = "at or above zero" if primary["mean"] >= 0 else "below zero with an interval that contains zero"
    return Branch(ESCALATE, f"the primary is {primary['mean']} [{primary['low']}, {primary['high']}], {where}: escalate to three seeds")


def rounds_read(groups: Sequence[Mapping], base_rungs: Sequence[Mapping], base_reach: Sequence[Mapping], rungs: Mapping[int, Sequence[Mapping]],
                reach: Mapping[int, Sequence[Mapping]], control: Sequence[Mapping] | None, rounds: Sequence[int], resamples: int, seed: int) -> dict:
    """L2's read, as the spec fixed it before the run. `rungs[r]` and `reach[r]`: M(r)'s fresh results on the three
    rungs and on G, for every round that was measured; `base_rungs` and `base_reach`: the base's, with the same
    sampling seeds; `control`: the base's extra episodes on G (None when the control did not run). Every change
    is paired by problem with a 95% bootstrap interval over problems."""
    ids = {name: group_ids(groups, name) for name in RUNGS}
    first, last = rounds[0], rounds[-1]
    minus_base = {number: {name: paired_change(rungs[number], base_rungs, ids[name], resamples, seed) for name in RUNGS} for number in sorted(rungs)}
    both = first in rungs and last in rungs and first != last
    climb = {name: paired_change(rungs[last], rungs[first], ids[name], resamples, seed) for name in RUNGS} if both else None
    both_reach = first in reach and last in reach and first != last
    branch = l2_branch(minus_base, rounds)
    return {
        "rounds_measured": sorted(rungs), "first_model": model_name(first), "last_model": model_name(last),
        "rungs_minus_base": minus_base,
        "primary": minus_base.get(last, {}).get(BELOW_BAND),
        "climb": climb,
        "reach_against_the_base_afresh": {number: gained_lost(reach[number], base_reach) for number in sorted(reach)},
        "reach_last_model_against_the_first": gained_lost(reach[last], reach[first]) if both_reach else None,
        "control": control_read(reach[last], base_reach, control) if control is not None and last in reach else None,
        "stop_rule_fires_at": [number for number in sorted(minus_base) if stop_rule(minus_base[number][BELOW_BAND])],
        "branch": {"name": branch.name, "reason": branch.reason},
    }
