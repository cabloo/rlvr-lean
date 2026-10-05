"""L1's read, as the spec fixed it before any run. Spec: "A round" steps 5 and 6, "L1's read, fixed now",
fixtures 11 and 12.

Every quantity here compares FRESH episodes with FRESH episodes: the trained model's and, beside it, new episodes
of the model it is compared with. The episodes that PLACED a problem (the base's 32 on H, the round's own n) only
say which group a problem belongs to. A problem placed low by a noisy count reads higher the next time with no
training at all, and using the placing count as the "before" would report that as learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from rlvr_lean.domain.evaluation.bootstrap import BootstrapInterval, paired_bootstrap
from rlvr_lean.domain.problem_pool.episodes import RUNGS, band, reward, rung

GOAL = "goal"                               # G: a Lean Workbook problem of H the base resolved in none of its placing episodes
LEAN_WORKBOOK_PART = "lean_workbook"
ESCALATE = "ESCALATE"                       # primary at or above zero: three seeds, then L2
STOP_AND_DIAGNOSE = "STOP AND DIAGNOSE"     # the interval lies entirely below zero: the run failed; the idea is not judged
VOID = "VOID"                               # the run could not test what it was for: fix, re-run one seed
NOT_NAMED = "NOT NAMED BY THE SPEC"         # primary below zero with an interval that contains zero
BRANCHES = (ESCALATE, STOP_AND_DIAGNOSE, VOID, NOT_NAMED)


def heldout_groups(heldout: Sequence[Mapping], placing: Mapping[str, Mapping], target_rate: float, floor: float) -> list[dict]:
    """Where the base's placing episodes put each problem of H: in G (a Lean Workbook problem with no success),
    on a rung (at least one success, against the band computed from t), or in neither (an STP conjecture with
    no success). Spec: "Held-out sets", fixture 13."""
    groups = []
    for row in heldout:
        placed = placing[row["problem_id"]]
        k, n = placed["resolved"], placed["episodes"]
        if k == 0:
            group = GOAL if row["heldout_part"] == LEAN_WORKBOOK_PART else None
        else:
            group = rung(k, n, target_rate, floor)
        groups.append({"problem_id": row["problem_id"], "heldout_part": row["heldout_part"], "side": row["side"], "group": group,
                       "placing_resolved": k, "placing_episodes": n})
    return groups


def group_ids(groups: Sequence[Mapping], name: str) -> list[str]:
    return [row["problem_id"] for row in groups if row["group"] == name]


def rung_ids(groups: Sequence[Mapping]) -> list[str]:
    return [row["problem_id"] for row in groups if row["group"] in RUNGS]


def _interval(differences: Sequence[float], resamples: int, seed: int) -> dict:
    if not differences:
        return {"problems": 0, "mean": None, "low": None, "high": None}
    interval = paired_bootstrap(differences, resamples=resamples, seed=seed)
    return {"problems": len(differences), "mean": round(interval.mean, 5), "low": round(interval.low, 5), "high": round(interval.high, 5)}


def paired_change(after: Sequence[Mapping], before: Sequence[Mapping], problem_ids: Sequence[str], resamples: int, seed: int) -> dict:
    """The change in pass rate over `problem_ids`, `after` minus `before`, paired by problem, with its 95%
    bootstrap interval over problems. Both sides must hold every problem, with the same number of episodes."""
    after_rows = {row["problem_id"]: row for row in after}
    before_rows = {row["problem_id"]: row for row in before}
    differences = []
    for problem_id in problem_ids:
        if problem_id not in after_rows or problem_id not in before_rows:
            raise ValueError(f"{problem_id} was not attempted on both sides of the comparison")
        if after_rows[problem_id]["episodes"] != before_rows[problem_id]["episodes"]:
            raise ValueError(f"{problem_id} has {after_rows[problem_id]['episodes']} episodes on one side and "
                             f"{before_rows[problem_id]['episodes']} on the other: the comparison needs the same number")
        differences.append(after_rows[problem_id]["resolved"] / after_rows[problem_id]["episodes"]
                           - before_rows[problem_id]["resolved"] / before_rows[problem_id]["episodes"])
    result = _interval(differences, resamples, seed)
    if differences:
        result["mean_after"] = round(sum(after_rows[problem_id]["resolved"] / after_rows[problem_id]["episodes"] for problem_id in problem_ids) / len(problem_ids), 5)
        result["mean_before"] = round(sum(before_rows[problem_id]["resolved"] / before_rows[problem_id]["episodes"] for problem_id in problem_ids) / len(problem_ids), 5)
    return result


def gain_by_k(placed: Sequence[Mapping], after: Sequence[Mapping], before: Sequence[Mapping], resamples: int, seed: int) -> list[dict]:
    """The change in pass rate, the trained model minus the model the round started from, grouped by the k that
    PLACED each problem (its count in the round, or the base's count on a held-out rung). `after` and `before`
    are fresh episodes of the two models on the same problems; the placing episodes are used for the grouping
    and for nothing else."""
    groups: dict[int, list[str]] = {}
    for row in placed:
        groups.setdefault(row["resolved"], []).append(row["problem_id"])
    table = []
    for k in sorted(groups):
        entry = {"k": k, "of": next(row["episodes"] for row in placed if row["resolved"] == k),
                 **paired_change(after, before, groups[k], resamples, seed)}
        table.append(entry)
    return table


def stop_rule(interval: BootstrapInterval | Mapping) -> bool:
    """A round stops the loop when the interval for the below-band rung lies ENTIRELY below zero. An interval
    that contains zero does not stop it."""
    high = interval.high if isinstance(interval, BootstrapInterval) else interval["high"]
    return high is not None and high < 0


def arm_reward(round_rows: Sequence[Mapping], target_rate: float, floor: float) -> dict:
    """What an arm's choice of problems earned: the mean reward, where its problems landed against the band,
    and how hard the model the round started from found them."""
    if not round_rows:
        return {"problems": 0, "mean_reward": None}
    low, high = band(target_rate, floor)
    rewards = [reward(row["resolved"], row["episodes"], target_rate) for row in round_rows]
    rates = [row["resolved"] / row["episodes"] for row in round_rows]
    episodes = max(row["episodes"] for row in round_rows)
    counts = {str(k): sum(row["resolved"] == k for row in round_rows) for k in range(episodes + 1)}
    return {"problems": len(round_rows), "mean_reward": round(sum(rewards) / len(rewards), 5),
            "share_in_the_band": round(sum(low <= rate <= high for rate in rates) / len(rates), 5),
            "share_at_k_0": round(sum(row["resolved"] == 0 for row in round_rows) / len(round_rows), 5),
            "share_at_k_n": round(sum(row["resolved"] == row["episodes"] for row in round_rows) / len(round_rows), 5),
            "mean_pass_rate": round(sum(rates) / len(rates), 5), "k_histogram": counts,
            "with_a_training_proof": sum(row["resolved"] > 0 for row in round_rows)}


def reach(trained: Sequence[Mapping], base: Sequence[Mapping]) -> dict:
    """Problems of G resolved at least once: the trained model's count against the base's under the same fresh
    budget (its luck). G is the problems the base resolved in none of its placing episodes."""
    trained_ids = {row["problem_id"] for row in trained if row["resolved"] > 0}
    base_ids = {row["problem_id"] for row in base if row["resolved"] > 0}
    return {"problems": len(trained), "resolved_by_the_trained_model": len(trained_ids), "resolved_by_the_base_fresh": len(base_ids),
            "by_both": len(trained_ids & base_ids), "only_the_trained_model": len(trained_ids - base_ids), "only_the_base": len(base_ids - trained_ids),
            "resolved_by_the_trained_model_known_false": sum(row["side"] == "false" and row["resolved"] > 0 for row in trained),
            "episodes_each": trained[0]["episodes"] if trained else None}


@dataclass(frozen=True)
class Branch:
    name: str
    reason: str


def l1_branch(primary: Mapping, challenger_mean_reward: float | None, random_mean_reward: float | None,
              below_band_problems: int, rung_minimum: int) -> Branch:
    """The branch the numbers select, in the spec's order: the three VOID conditions first (a soundness alarm
    stops the stage before any report exists, so it is not seen here), then the primary."""
    if below_band_problems < rung_minimum:
        return Branch(VOID, f"the below-band rung holds {below_band_problems} problems and the read needs at least {rung_minimum}")
    if challenger_mean_reward is None or random_mean_reward is None:
        return Branch(VOID, "an arm's mean reward is missing: both arms must have run their round")
    if challenger_mean_reward <= random_mean_reward:
        return Branch(VOID, f"the challenger's mean reward ({challenger_mean_reward}) is no higher than the random draw's "
                            f"({random_mean_reward}): the arm did not test aiming")
    if primary.get("mean") is None:
        return Branch(VOID, "the primary could not be computed: the below-band rung was not measured on both sides")
    if stop_rule(primary):
        return Branch(STOP_AND_DIAGNOSE, f"the primary's interval [{primary['low']}, {primary['high']}] lies entirely below zero: the run "
                                         "failed, and the idea is not judged until the verdict gates pass")
    if primary["mean"] >= 0:
        return Branch(ESCALATE, f"the primary is {primary['mean']} [{primary['low']}, {primary['high']}], at or above zero: three seeds, then L2")
    return Branch(NOT_NAMED, f"the primary is {primary['mean']} [{primary['low']}, {primary['high']}]: below zero with an interval that "
                             "contains zero. The spec names 'at or above zero' and 'entirely below zero' and not this case")
