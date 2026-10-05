"""L2's report: three rounds of the challenger arm, then the equal-compute control. Spec:
docs/spec/ladder-loop.spec.md, "L2: three rounds", "The read, fixed before the run". Pure: rows in, a report out.
It answers the read exactly and names the branch the numbers select, with the VOID condition stated by the report.

Naming (`naming` in the report): rounds are 1, 2, 3; round r starts from M(r - 1) and trains M(r); M(0) is the base.

  primary       the below-band rung, the last model minus the base, paired by problem, 95% bootstrap over problems
  per round     each rung, M(r) minus the base; reach on G against the base afresh, as gained against lost with a
                two-sided sign test; what M(r) was trained on; distinct attempts
  the climb     each rung, the last model minus the ONE-ROUND model
  reach         the last model against the base afresh AND against the one-round model (a comparison of two models
                compares those two)
  the control   the last model in its reach episodes against the base in its fresh reach episodes plus the loop's
                attempt episodes, as counts and as gained against lost
  equal attempts   (added 2026-10-05; `control.equal_attempts`, present only when its step ran) the last model given
                the control's extra episodes too, against the base with the control's: the same number of attempts
                each; counts, gained against lost, successful episodes and successes per attempt of each
  challenger    its table by batch and by round, and the trajectory stated before the run
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.read import GOAL, group_ids
from rlvr_lean.domain.ladder_round.rounds import challenger_trajectory, equal_attempts_read, model_name, picks_row, rounds_read
from rlvr_lean.domain.problem_pool.episodes import ABOVE_BAND, IN_BAND, RUNGS, band

MAXIMUM_SHARE_WITHOUT_AN_ANSWER = 0.02      # above this share of attempts with no verdict a set's counts are not to be read
NAMING = "rounds are numbered 1, 2, 3; round r starts from M(r - 1) and trains M(r), from the base, on every round's training set so far; M(0) is the base"


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


def _interval(change: Mapping | None) -> str:
    return "not measured" if not change or change.get("mean") is None else f"{change['mean']} [{change['low']}, {change['high']}]"


def _split(counted: Mapping | None) -> str:
    return "not measured" if not counted else f"gained {counted['gained']}, lost {counted['lost']} (p = {counted['sign_test_p']})"


def build_l2_report(prepare: Mapping, groups: Sequence[Mapping], base_rungs: Sequence[Mapping], base_reach: Sequence[Mapping],
                    rounds: Mapping[int, Mapping], control: Mapping | None, planned: Sequence[int], settings: Mapping, evaluation: Mapping,
                    context: Mapping, control_trained: Mapping | None = None) -> dict:
    """`rounds[r]`, for every round whose attempts are done: `summary` (the round step's), `proposals`, `results` and
    `fits` by batch, `examples` (its training set), and, once they exist, `training`, `measure`, `rungs` and
    `reach` (M(r)'s). `control`: the control step's `summary` and its `results`, or None. `planned`: the rounds the
    loop was to run. `context`: the embedding step's summary and the round the stop rule fired after, if any.
    `control_trained`: the `summary` and `results` of the last model's own extra attempts on G, or None for a run
    that has not had that step: its report is then exactly what it was before the step existed."""
    target, floor = settings["challenger"]["target_rate"], settings["challenger"]["band_reward"]
    low, high = band(target, floor)
    read = rounds_read(groups, base_rungs, base_reach,
                       {number: entry["rungs"] for number, entry in rounds.items() if entry.get("rungs") is not None},
                       {number: entry["reach"] for number, entry in rounds.items() if entry.get("reach") is not None},
                       control["results"] if control and control.get("results") is not None else None,
                       planned, evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"])
    first, last = planned[0], planned[-1]

    # ---- the challenger's table, by batch and by round
    by_batch, by_round, health = [], {}, {"rungs_base": _health(base_rungs), "reach_base": _health(base_reach)}
    for number, entry in sorted(rounds.items()):
        for batch in sorted(entry["proposals"]):
            fit = entry["fits"][batch]
            by_batch.append({"round": number, "batch": batch, "attempted_by": model_name(number - 1),
                             **picks_row(entry["proposals"][batch], entry["results"][batch], entry["examples"], target, floor),
                             "refit": {"observations": fit["observations"], "by_the_round_that_gave_them": fit["observations_by_the_round_that_gave_them"],
                                       "ridge": fit["fit"]["ridge"], "dispersion": fit["fit"]["dispersion"],
                                       "candidates_not_yet_proposed": fit["candidates_not_yet_proposed"]}})
        proposals = [row for batch in sorted(entry["proposals"]) for row in entry["proposals"][batch]]
        results = [row for batch in sorted(entry["results"]) for row in entry["results"][batch]]
        by_round[number] = {"round": number, "attempted_by": model_name(number - 1), **picks_row(proposals, results, entry["examples"], target, floor),
                            "share_of_resolutions_that_were_negations": _negation_share(results)}
        health[f"round_{number}"] = _health(results)
        for name in ("rungs", "reach"):
            if entry.get(name) is not None:
                health[f"{name}_m{number}"] = _health(entry[name])
    if control and control.get("results") is not None:
        health["control"] = _health(control["results"])
    # ---- the same extra attempts for the last model (added 2026-10-05): present only when its step ran
    equal = None
    if control_trained and control_trained.get("results") is not None and control and control.get("results") is not None \
            and rounds.get(last, {}).get("reach") is not None:
        equal = {"what": f"the same extra attempts for the trained model: {model_name(last)} given the control's number of episodes on every problem of G, "
                         "with the control's sampling seed, so that both models have the same number of attempts",
                 "step": control_trained["summary"],
                 "read": equal_attempts_read(rounds[last]["reach"], control_trained["results"], base_reach, control["results"])}
        health[f"control_m{last}"] = _health(control_trained["results"])
    trajectory = challenger_trajectory(by_round, target)

    # ---- each round: what was trained, and what its model measured
    report_rounds = {}
    for number, entry in sorted(rounds.items()):
        measured = entry.get("measure")
        report_rounds[str(number)] = {
            "attempted_by": model_name(number - 1), "trained": model_name(number),
            "attempts": {key: entry["summary"][key] for key in ("problems", "attempt_episodes", "reward", "sampling_seed") if key in entry["summary"]},
            "training": entry.get("training"),
            "rungs_minus_base": read["rungs_minus_base"].get(number),
            "reach_on_g_against_the_base_afresh": read["reach_against_the_base_afresh"].get(number),
            "distinct_attempts_on_the_rungs": measured["distinct_attempts_on_the_rungs"] if measured else None,
            "share_of_resolutions_that_were_negations_after_training": _negation_share([*entry["rungs"], *entry["reach"]]) if measured else None,
            "stop_rule_fires": measured["stop_rule_fires"] if measured else None,
        }
    distinct = {"base": prepare.get("base_distinct_attempts_on_the_rungs"),
                **{model_name(number): entry["measure"]["distinct_attempts_on_the_rungs"] for number, entry in sorted(rounds.items()) if entry.get("measure")}}

    not_to_be_read = sorted(name for name, entry in health.items()
                            if entry["share_without_an_answer"] is not None and entry["share_without_an_answer"] > MAXIMUM_SHARE_WITHOUT_AN_ANSWER)
    one_round = read["rungs_minus_base"].get(first)
    void_conditions = {
        "round_1_left_both_the_in_band_and_the_above_band_rung_at_or_below_zero": (
            None if one_round is None or None in (one_round[IN_BAND]["mean"], one_round[ABOVE_BAND]["mean"])
            else one_round[IN_BAND]["mean"] <= 0 and one_round[ABOVE_BAND]["mean"] <= 0),
        "a_soundness_alarm": False,      # an alarm ends the stage at its step (exit code 3): no report is written after one
    }
    goal, rungs = group_ids(groups, GOAL), {name: len(group_ids(groups, name)) for name in RUNGS}
    primary, branch = read["primary"], read["branch"]
    against_base, against_one = read["reach_against_the_base_afresh"].get(last), read["reach_last_model_against_the_first"]
    controlled = read["control"]
    stopped = context.get("stopped_after_round")
    shares = trajectory["known_false_share_of_the_scored_picks"]
    below_by_round = ", ".join(model_name(number) + " " + _interval(read["rungs_minus_base"][number]["below"]) for number in read["rounds_measured"])
    climbed = ", ".join(name + " " + _interval(read["climb"][name]) for name in RUNGS) if read["climb"] else "not measured"
    at_equal = ""
    if equal:
        even = equal["read"]
        per_thousand = [None if rate is None else round(1000 * rate, 2) for rate in (even["successes_per_attempt_of_the_last_model"], even["successes_per_attempt_of_the_base"])]
        at_equal = (f"; at equal attempts ({even['episodes_after']} each): {model_name(last)} {even['resolved_after']} against the base {even['resolved_before']}, "
                    f"{_split(even)}, {per_thousand[0]} against {per_thousand[1]} successes per 1,000 attempts")
    headline = (
        f"L2 seed {prepare['seed']}: {branch['name']}. Primary (below-band rung, {model_name(last)} minus the base, {rungs['below']} problems): {_interval(primary)}; "
        f"by round: {below_by_round or 'none measured'}; climb ({model_name(last)} minus {model_name(first)}): {climbed}; "
        f"reach on G ({len(goal)}): {model_name(last)} against the base afresh "
        + (f"{against_base['resolved_after']} to {against_base['resolved_before']}, {_split(against_base)}" if against_base else "not measured")
        + f", against {model_name(first)} {_split(against_one)}; control: "
        + (f"{model_name(last)} {controlled['resolved_after']} in {controlled['episodes_after']} episodes against the base {controlled['resolved_before']} in "
           f"{controlled['episodes_before']}, {_split(controlled)}" if controlled else "not run")
        + at_equal
        + f"; known-false share of the scored picks by round: {', '.join(str(shares[key]) for key in sorted(shares)) or 'none'}"
        + (f"; the loop STOPPED after round {stopped}" if stopped else "")
        + (f"; NOT TO BE READ: too many attempts without an answer in {', '.join(not_to_be_read)}" if not_to_be_read else ""))
    return {
        "spec": "docs/spec/ladder-loop.spec.md, L2: three rounds", "headline": headline, "ok": not not_to_be_read, "seed": prepare["seed"],
        "naming": NAMING, "fixture": bool(prepare.get("fixture")),
        "stand_in_engine": bool(prepare.get("stand_in_engine") or any(entry["summary"].get("stand_in_engine") for entry in rounds.values())),
        "target_rate": target, "band_reward_floor": floor, "band": {"low": round(low, 4), "high": round(high, 4)},
        "branch": branch, "void_conditions": void_conditions,
        "stop_rule": {"what": "an interval entirely below zero for any round's model minus the base on the below-band rung stops the loop",
                      "fires_at_rounds": read["stop_rule_fires_at"], "the_loop_stopped_after_round": stopped},
        "rounds_planned": list(planned), "rounds_measured": read["rounds_measured"],
        "primary": {"what": f"fresh pass rate on the held-out below-band rung, {model_name(last)} minus the base, paired by problem", **(primary or {"mean": None})},
        "climb": {"what": f"each rung, {model_name(last)} minus {model_name(first)}, paired by problem: a ladder that climbs gains with rounds", "by_rung": read["climb"]},
        "reach_on_g": {"what": "problems resolved at least once in the reach episodes; gained = by the first-named model and not the second, lost = the reverse; "
                               "two-sided sign test",
                       "last_model_against_the_base_afresh": against_base,
                       "last_model_against_the_one_round_model": against_one,
                       "by_round_against_the_base_afresh": {str(number): counted for number, counted in read["reach_against_the_base_afresh"].items()}},
        "control": {"what": "the equal-compute control: the base given the loop's attempt episodes on G, on top of its fresh reach episodes",
                    "step": control["summary"] if control else None, "read": controlled, **({"equal_attempts": equal} if equal else {})},
        "rounds": report_rounds,
        "challenger": {"what": "by batch and by round, for the scored picks, the random places and all: the share known false, the mean pass rate, the shares "
                               "at k = 0, below, in and above the band, the mean reward, the calibration of what was predicted before the episodes; and the "
                               "share of the training set from problems above the band and from refutations",
                       "by_batch": by_batch, "by_round": {str(number): row for number, row in by_round.items()}, "trajectory": trajectory},
        "distinct_attempts_on_the_rungs": distinct,
        "heldout": {"goal_set": len(goal), "rungs": rungs, "in_neither": sum(row["group"] is None for row in groups)},
        "attempts": health, "not_to_be_read": not_to_be_read,
        "not_measured": ["the gain by k (the spec: not measured in L2; L1 measured it at three seeds and t stays 1/4)",
                         "how hard the base found the problems the challenger chooses in rounds 2 and 3 (it needs base episodes on them: L2 adds no sampling)",
                         "the standard evaluation of Phase A (its statements are not built at v4.27)",
                         "reach on the problems of G with no near neighbour in the training sets (the spec does not define a near neighbour)"],
        "sizes": {key: prepare[key] for key in ("problems_a_round", "batches", "batch_sizes", "solvers", "sampling_seeds", "rung_episodes", "reach_episodes",
                                                "candidates_short_by", "data") if key in prepare},
        "embedding": context.get("embed"),
    }
