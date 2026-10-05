"""L1's report: one round, two arms. Spec: docs/spec/ladder-loop.spec.md, "L1's read, fixed now". Pure: rows in,
a report out. It answers the read exactly and names the branch the numbers select, with the VOID conditions
stated by the report itself.

  primary     fresh pass rate on the held-out below-band rung, the challenger arm minus the base, paired by
              problem, 95% bootstrap interval over problems
  also        the same for the control arm, and challenger minus control; reach on G against the base's luck;
              the in-band rung; distinct attempts; each arm's mean reward and the share of its problems in the band
  gain by k   reported and not a branch: both arms' tables for the round's own problems and for the held-out rungs
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round import ARMS, CHALLENGER_ARM, RANDOM_ARM
from rlvr_lean.domain.ladder_round.read import (
    GOAL,
    arm_reward,
    gain_by_k,
    group_ids,
    l1_branch,
    paired_change,
    reach,
    rung_ids,
    stop_rule,
)
from rlvr_lean.domain.problem_pool.episodes import BELOW_BAND, IN_BAND, RUNGS, band

MAXIMUM_SHARE_WITHOUT_AN_ANSWER = 0.02      # above this share of attempts with no verdict a set's counts are not to be read


def _health(rows: Sequence[Mapping]) -> dict:
    attempts = sum(row["episodes"] * row["sides"] for row in rows)
    without = sum(row["attempts_without_an_answer"] for row in rows)
    return {"attempts": attempts, "capped_at_the_token_limit": sum(row["attempts_capped"] for row in rows),
            "timed_out_in_lean": sum(row["attempts_timed_out"] for row in rows), "without_an_answer": without,
            "share_without_an_answer": round(without / attempts, 5) if attempts else None}


def _negation_share(rows: Sequence[Mapping]) -> float | None:
    by_statement = sum(row["resolved_by_statement"] for row in rows)
    by_negation = sum(row["resolved_by_negation"] for row in rows)
    return round(by_negation / (by_statement + by_negation), 4) if by_statement + by_negation else None


def build_l1_report(groups: Sequence[Mapping], base: Mapping[str, Sequence[Mapping]], arms: Mapping[str, Mapping], settings: Mapping,
                    evaluation: Mapping, seed: int, context: Mapping) -> dict:
    """`groups`: where the base's placing episodes put each problem of H. `base`: its FRESH episodes
    (`reach`, `rungs`). `arms[arm]`: `round`, `reach`, `rungs`, `gain_trained`, `gain_base` result rows, plus
    `training`, `distinct_attempts` and `proposals`. `context`: what the steps recorded (fixture, stand-in, fit)."""
    target, floor = settings["challenger"]["target_rate"], settings["challenger"]["band_reward"]
    minimum = settings["goal"]["rung_minimum"]
    resamples, bootstrap_seed = evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"]
    low, high = band(target, floor)
    rungs = {name: group_ids(groups, name) for name in RUNGS}
    on_a_rung = rung_ids(groups)
    placed = [{"problem_id": row["problem_id"], "resolved": row["placing_resolved"], "episodes": row["placing_episodes"]}
              for row in groups if row["problem_id"] in set(on_a_rung)]

    report_arms, health = {}, {"reach_base": _health(base["reach"]), "rungs_base": _health(base["rungs"])}
    for arm in ARMS:
        data = arms[arm]
        by_rung = {name: paired_change(data["rungs"], base["rungs"], rungs[name], resamples, bootstrap_seed) for name in RUNGS}
        report_arms[arm] = {
            "round": {**arm_reward(data["round"], target, floor), "share_of_resolutions_that_were_negations": _negation_share(data["round"])},
            "training": data["training"],
            "rungs_trained_minus_base": by_rung,
            "reach_on_g": reach(data["reach"], base["reach"]),
            "gain_by_k": {"the_rounds_own_problems": gain_by_k(data["round"], data["gain_trained"], data["gain_base"], resamples, bootstrap_seed),
                          "held_out_rungs_by_the_bases_placing_count": gain_by_k(placed, data["rungs"], base["rungs"], resamples, bootstrap_seed)},
            "share_of_resolutions_that_were_negations_after_training": _negation_share([*data["reach"], *data["rungs"], *data["gain_trained"]]),
            "distinct_attempts": data["distinct_attempts"],
            "proposals_by_how": {how: sum(row["how"] == how for row in data["proposals"]) for how in sorted({row["how"] for row in data["proposals"]})},
        }
        for name in ("round", "reach", "rungs", "gain_trained", "gain_base"):
            health[f"{name}_{arm}"] = _health(data[name])

    primary = report_arms[CHALLENGER_ARM]["rungs_trained_minus_base"][BELOW_BAND]
    control = report_arms[RANDOM_ARM]["rungs_trained_minus_base"][BELOW_BAND]
    challenger_minus_control = {name: paired_change(arms[CHALLENGER_ARM]["rungs"], arms[RANDOM_ARM]["rungs"], rungs[name], resamples, bootstrap_seed)
                                for name in RUNGS}
    rewards = {arm: report_arms[arm]["round"]["mean_reward"] for arm in ARMS}
    branch = l1_branch(primary, rewards[CHALLENGER_ARM], rewards[RANDOM_ARM], len(rungs[BELOW_BAND]), minimum)
    not_to_be_read = sorted(name for name, entry in health.items()
                            if entry["share_without_an_answer"] is not None and entry["share_without_an_answer"] > MAXIMUM_SHARE_WITHOUT_AN_ANSWER)
    void_conditions = {
        "the_challengers_mean_reward_is_no_higher_than_the_random_draws": (
            None if None in rewards.values() else rewards[CHALLENGER_ARM] <= rewards[RANDOM_ARM]),
        "the_below_band_rung_has_fewer_problems_than_the_minimum": len(rungs[BELOW_BAND]) < minimum,
        "a_soundness_alarm": False,      # an alarm ends the stage at its step (exit code 3): no report is written after one
    }
    goal = group_ids(groups, GOAL)
    headline = (f"L1 seed {seed}: {branch.name}. Primary (below-band rung, challenger arm minus base, {primary['problems']} problems): "
                f"{primary['mean']} [{primary['low']}, {primary['high']}]; control arm {control['mean']} [{control['low']}, {control['high']}]; "
                f"reach on G ({len(goal)}): challenger {report_arms[CHALLENGER_ARM]['reach_on_g']['resolved_by_the_trained_model']}, "
                f"random {report_arms[RANDOM_ARM]['reach_on_g']['resolved_by_the_trained_model']}, base {report_arms[CHALLENGER_ARM]['reach_on_g']['resolved_by_the_base_fresh']}; "
                f"mean reward challenger {rewards[CHALLENGER_ARM]} against random {rewards[RANDOM_ARM]}"
                f"{'; NOT TO BE READ: too many attempts without an answer in ' + ', '.join(not_to_be_read) if not_to_be_read else ''}")
    return {
        "spec": "docs/spec/ladder-loop.spec.md, milestone L1", "headline": headline, "ok": not not_to_be_read, "seed": seed,
        "fixture": bool(context.get("fixture")), "stand_in_engine": bool(context.get("stand_in_engine")),
        "target_rate": target, "band_reward_floor": floor, "band": {"low": round(low, 4), "high": round(high, 4)},
        "branch": {"name": branch.name, "reason": branch.reason},
        "void_conditions": void_conditions,
        "stop_rule_fires": stop_rule(primary),
        "primary": {"what": "fresh pass rate on the held-out below-band rung, the challenger arm minus the base, paired by problem", **primary},
        "also": {
            "the_same_for_the_control_arm": control,
            "challenger_minus_control_by_rung": challenger_minus_control,
            "the_in_band_rung": {arm: report_arms[arm]["rungs_trained_minus_base"][IN_BAND] for arm in ARMS},
            "reach_on_g": {arm: report_arms[arm]["reach_on_g"] for arm in ARMS},
            "distinct_attempts": {arm: report_arms[arm]["distinct_attempts"] for arm in ARMS},
            "mean_reward_and_share_in_the_band": {arm: {key: report_arms[arm]["round"][key] for key in ("mean_reward", "share_in_the_band", "share_at_k_0", "share_at_k_n")}
                                                  for arm in ARMS},
        },
        "gain_by_k": {arm: report_arms[arm]["gain_by_k"] for arm in ARMS},
        "heldout": {"goal_set": len(goal), "rungs": {name: len(rungs[name]) for name in RUNGS}, "rung_minimum": minimum,
                    "in_neither": sum(row["group"] is None for row in groups)},
        "arms": report_arms, "challenger_fit": context.get("challenger_fit"), "attempts": health, "not_to_be_read": not_to_be_read,
        "not_measured": ["the standard evaluation of Phase A (its statements are not built at v4.27; see the stage's note)",
                         "reach on the problems of G with no near neighbour in the training sets (the spec does not define a near neighbour)"],
        "steps": context.get("steps", {}),
    }
