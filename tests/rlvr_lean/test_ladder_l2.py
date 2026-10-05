"""The ladder loop's L2: three rounds of the challenger arm, then the equal-compute control. The RULES: batches, the
refit, the control, the read and its branch, the challenger's table, and the report built from hand-made rows. Spec:
docs/spec/ladder-loop.spec.md, "L2: three rounds"; fixtures 5, 6, 11. Pure:
no model, no Lean. The stage end to end is `test_ladder_l2_stage.py`.

Naming, as in the code and the report: rounds are 1, 2, 3; round r starts from M(r - 1) and trains M(r); M(0) is the
base."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from rlvr_lean.domain.ladder_round.challenger import RANDOM_PLACE, SCORED, fit_pass_rate_model
from rlvr_lean.domain.ladder_round.read import ESCALATE, STOP_AND_DIAGNOSE, VOID
from rlvr_lean.domain.ladder_round.rounds import (
    BRANCHES,
    batch_sizes,
    challenger_trajectory,
    control_episodes,
    control_read,
    equal_attempts_read,
    gained_lost,
    l2_branch,
    model_name,
    picks_calibration,
    picks_row,
    propose_batch,
    refit_observations,
    rounds_read,
    sign_test,
    summed_budget,
    weights_summary,
)
from rlvr_lean.domain.problem_pool.episodes import reward
from rlvr_lean.reporting.ladder_l2 import build_l2_report

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
SETTINGS = CONFIG["ladder_loop"]
T, FLOOR = SETTINGS["challenger"]["target_rate"], SETTINGS["challenger"]["band_reward"]
ROUNDS = (1, 2, 3)


# ----------------------------------------------------------------------------------------------- proposals
def test_a_rounds_proposals_are_made_in_equal_batches():
    assert (SETTINGS["round"]["problems"], SETTINGS["round"]["batches"], SETTINGS["round"]["rounds"]) == (1000, 4, 3)
    assert batch_sizes(1000, 4) == [250, 250, 250, 250] and batch_sizes(4, 2) == [2, 2] and batch_sizes(7, 1) == [7]
    with pytest.raises(ValueError, match="must be equal"):
        batch_sizes(1000, 3)
    for problems, batches in ((3, 4), (4, 0)):
        with pytest.raises(ValueError):
            batch_sizes(problems, batches)
    assert [model_name(number) for number in (0, *ROUNDS)] == ["M(0)", "M(1)", "M(2)", "M(3)"]


def test_no_problem_is_proposed_twice_across_batches_and_rounds_and_none_of_h_or_the_base_map_ever():
    """Three rounds of four batches, the challenger refitted before each (every score changes): 3,000 proposals,
    each problem once, and never one that is barred, however well it scores."""
    generator = np.random.default_rng(0)
    ids = [f"c{index}" for index in range(5000)]
    barred = {"c7", "c8", "c9", "c4000", "not_a_candidate"}             # H and the base map's problems
    proposed, sizes = set(), []
    for _ in ROUNDS:
        for places in batch_sizes(1000, 4):
            rates, scores = generator.random(len(ids)), generator.random(len(ids))
            for index in (7, 8, 9, 4000):
                scores[index] = 2.0                                     # the best score there is: still never proposed
            picks = propose_batch(ids, rates, scores, proposed, barred, places, SETTINGS["challenger"]["random_share"], seed=0)
            chosen = {row["problem_id"] for row in picks}
            assert len(picks) == len(chosen) == places and not chosen & proposed and not chosen & barred
            # The random share holds in EVERY batch: 25 of its 250 places, the other 225 the best that are still open.
            assert [sum(row["how"] == how for row in picks) for how in (SCORED, RANDOM_PLACE)] == [225, 25]
            still_open = [index for index in range(len(ids)) if ids[index] not in proposed and ids[index] not in barred]
            best = sorted(still_open, key=lambda index: -scores[index])[:225]
            assert {row["problem_id"] for row in picks if row["how"] == SCORED} == {ids[index] for index in best}
            assert all(row["predicted_rate"] == round(float(rates[int(row["problem_id"][1:])]), 6) for row in picks)      # what it predicted, before the episodes
            proposed |= chosen
            sizes.append(len(picks))
    assert sizes == [250] * 12 and len(proposed) == 3000


def test_a_batch_takes_what_is_left_when_the_candidates_run_out_and_the_same_seed_gives_the_same_batch():
    ids = [f"c{index}" for index in range(10)]
    scores = [index / 10 for index in range(10)]
    first = propose_batch(ids, scores, scores, set(), {"c9"}, 4, 0.25, seed=0)
    assert [row["problem_id"] for row in first if row["how"] == SCORED] == ["c8", "c7", "c6"] and sum(row["how"] == RANDOM_PLACE for row in first) == 1
    assert propose_batch(ids, scores, scores, set(), {"c9"}, 4, 0.25, seed=0) == first
    taken = {row["problem_id"] for row in first}
    second = propose_batch(ids, scores, scores, taken, {"c9"}, 4, 0.25, seed=0)
    assert not {row["problem_id"] for row in second} & taken and len(second) == 4
    rest = propose_batch(ids, scores, scores, taken | {row["problem_id"] for row in second}, {"c9"}, 4, 0.25, seed=0)
    assert len(rest) == 1 and propose_batch(ids, scores, scores, set(ids), {"c9"}, 4, 0.25, seed=0) == []
    with pytest.raises(ValueError):
        propose_batch(ids, scores[:5], scores, set(), set(), 4, 0.25, seed=0)


# ----------------------------------------------------------------------------------------------- the refit
def _batch(round_number, batch, count, resolved=2):
    return [{"problem_id": f"r{round_number}b{batch}_{index}", "resolved": resolved, "episodes": 8, "round": round_number, "batch": batch} for index in range(count)]


def test_the_refit_after_a_batch_sees_that_batch_at_full_weight_earlier_rounds_decayed_and_the_base_map_by_its_age():
    decay = SETTINGS["challenger"]["recency_decay"]
    assert decay == 0.5
    base_map = [{"problem_id": f"m{index}", "resolved": 1, "episodes": 8} for index in range(3)]
    # Round 1 starts from the base, whose results the base map is: it is current, and it is all there is.
    first = refit_observations(base_map, [], 1, decay)
    assert first["problem_ids"] == ["m0", "m1", "m2"] and list(first["weights"]) == [1.0, 1.0, 1.0]
    # After round 1's first batch: that batch is seen, at full weight, beside the base map.
    after_one = refit_observations(base_map, _batch(1, 1, 4), 1, decay)
    assert after_one["problem_ids"][3:] == ["r1b1_0", "r1b1_1", "r1b1_2", "r1b1_3"] and list(after_one["weights"]) == [1.0] * 7
    assert list(after_one["resolved"]) == [1, 1, 1, 2, 2, 2, 2] and list(after_one["episodes"]) == [8] * 7
    # Round 2, after its first batch: the base map and round 1 are one round old, the batch just finished is current.
    seen = _batch(1, 1, 2) + _batch(1, 2, 2) + _batch(2, 1, 2)
    second = refit_observations(base_map, seen, 2, decay)
    assert list(second["weights"]) == [0.5] * 3 + [0.5] * 4 + [1.0] * 2 and second["rounds"] == [1] * 7 + [2] * 2
    # Round 3: the base map and round 1 are two rounds old, round 2 one, round 3's own batches current.
    third = refit_observations(base_map, seen + _batch(2, 2, 2) + _batch(3, 1, 5), 3, decay)
    assert list(third["weights"]) == [0.25] * 7 + [0.5] * 4 + [1.0] * 5
    assert weights_summary(third) == {"1": {"observations": 7, "weight": 0.25}, "2": {"observations": 4, "weight": 0.5}, "3": {"observations": 5, "weight": 1.0}}
    with pytest.raises(ValueError, match="observed twice"):
        refit_observations(base_map, _batch(1, 1, 2) + _batch(1, 1, 2), 1, decay)
    with pytest.raises(ValueError, match="observed twice"):                 # a base-map problem is never proposed
        refit_observations(base_map, [{"problem_id": "m0", "resolved": 0, "episodes": 8, "round": 1}], 1, decay)
    with pytest.raises(ValueError, match="later"):                          # a batch of a round that has not begun
        refit_observations(base_map, _batch(3, 1, 2), 2, decay)
    with pytest.raises(ValueError):
        refit_observations(base_map, _batch(0, 1, 2), 1, decay)


def test_a_refit_follows_the_current_models_batch_against_what_an_earlier_model_said():
    """The point of the batches: once the current model's results on a batch are in, they outweigh an earlier
    model's on problems like them."""
    generator = np.random.default_rng(1)
    features = {f"p{index}": generator.normal(size=3) for index in range(240)}
    rate = lambda problem_id, sign: 1 / (1 + np.exp(-sign * 2.0 * features[problem_id][0]))      # noqa: E731
    base_map = [{"problem_id": f"p{index}", "resolved": int(generator.binomial(8, rate(f"p{index}", +1))), "episodes": 8} for index in range(120)]
    current = [{"problem_id": f"p{index}", "resolved": int(generator.binomial(8, rate(f"p{index}", -1))), "episodes": 8, "round": 3} for index in range(120, 240)]

    def coefficient(finished, current_round):
        seen = refit_observations(base_map, finished, current_round, 0.1)
        model, _ = fit_pass_rate_model(seen["problem_ids"], np.array([features[key] for key in seen["problem_ids"]]), seen["resolved"], seen["episodes"],
                                       names=["f0", "f1", "f2"], ridge_grid=[1.0], folds=5, dispersion_grid=[0.0], seed=0, weights=seen["weights"])
        return model.coefficients[0]

    assert coefficient([], 3) > 0                       # before the batch: what the base's results say
    assert coefficient(current, 3) < 0                  # after it: what the current model's say


# --------------------------------------------------------------------------------------------- the control
def _result(problem_id, resolved, episodes=8, side="true"):
    return {"problem_id": problem_id, "resolved": resolved, "episodes": episodes, "side": side, "sides": 1, "resolved_by_statement": resolved,
            "resolved_by_negation": 0, "attempts_capped": 0, "attempts_timed_out": 0, "attempts_without_an_answer": 0}


def test_the_control_gives_each_goal_problem_the_loops_attempt_episodes_spread_evenly():
    loop = SETTINGS["round"]["rounds"] * SETTINGS["round"]["problems"] * SETTINGS["round"]["solvers"]
    assert loop == 24000 and control_episodes(loop, 392) == 61 and 392 * 61 <= loop < 392 * 62      # the spec's 61 more on each of 392
    assert control_episodes(96, 1) == 96 and control_episodes(24000, 0) == 0 and control_episodes(0, 392) == 0
    with pytest.raises(ValueError):
        control_episodes(-1, 392)
    base = [_result("g1", 0, 32), _result("g2", 1, 32), _result("g3", 0, 32)]
    more = [_result("g2", 0, 61), _result("g1", 2, 61), _result("g3", 0, 61)]
    summed = summed_budget(base, more)
    assert [(row["problem_id"], row["resolved"], row["episodes"]) for row in summed] == [("g1", 2, 93), ("g2", 1, 93), ("g3", 0, 93)]
    with pytest.raises(ValueError, match="same problems"):
        summed_budget(base, more[:2])
    # The read: the last model in 32 episodes against the base in 32 + 61, as counts and as gained against lost.
    last = [_result("g1", 0, 32), _result("g2", 3, 32), _result("g3", 1, 32)]
    read = control_read(last, base, more)
    assert (read["episodes_after"], read["episodes_before"], read["control_episodes_each"]) == (32, 93, 61)
    assert (read["resolved_after"], read["resolved_before"], read["by_both"], read["gained"], read["lost"]) == (2, 2, 1, 1, 1)
    assert read["gained_problems"] == ["g3"] and read["lost_problems"] == ["g1"]
    assert read["resolved_by_the_base_in_its_fresh_episodes"] == 1 and read["resolved_by_the_base_in_the_control_episodes"] == 1


# ------------------------------------------------------------------------------------------------ the read
def test_the_sign_test_is_two_sided_and_exact():
    assert sign_test(0, 0) is None and sign_test(5, 5) == 1.0 and sign_test(1, 0) == 1.0
    assert sign_test(0, 6) == sign_test(6, 0) == pytest.approx(2 / 64)
    assert sign_test(3, 9) == pytest.approx(2 * sum((1, 12, 66, 220)) / 4096, abs=1e-6)
    assert sign_test(32, 63) == pytest.approx(0.002, abs=0.0002)        # L1b: 32 gained against 63 lost over one pass, p = 0.002
    assert sign_test(41, 44) > 0.8
    assert 0 < sign_test(355, 0) < 1e-100                               # a very small chance is not rounded to zero
    with pytest.raises(ValueError):
        sign_test(-1, 3)


def test_reach_is_gained_against_lost_problem_by_problem():
    after = [_result("a", 2, 32), _result("b", 0, 32), _result("c", 1, 32, side="false"), _result("d", 0, 32), _result("e", 5, 32)]
    before = [_result("a", 0, 32), _result("b", 3, 32), _result("c", 1, 32, side="false"), _result("d", 0, 32), _result("e", 0, 32)]
    counted = gained_lost(after, before)
    assert (counted["problems"], counted["resolved_after"], counted["resolved_before"], counted["by_both"]) == (5, 3, 2, 1)
    assert (counted["gained"], counted["lost"], counted["sign_test_p"]) == (2, 1, 1.0) and counted["resolved_after_known_false"] == 1
    assert counted["gained_problems"] == ["a", "e"] and counted["lost_problems"] == ["b"]
    # The count alone hides it: the same number resolved, and two problems changed hands.
    same = gained_lost([_result("a", 1, 32), _result("b", 0, 32)], [_result("a", 0, 32), _result("b", 1, 32)])
    assert same["resolved_after"] == same["resolved_before"] == 1 and (same["gained"], same["lost"]) == (1, 1)
    with pytest.raises(ValueError, match="same problems"):
        gained_lost(after, before[:4])
    assert gained_lost([], [])["sign_test_p"] is None


def _groups(per_rung=40, goal=30):
    return [{"problem_id": f"{rung}{index}", "group": rung} for rung in ("below", "in", "above") for index in range(per_rung)] + \
           [{"problem_id": f"goal{index}", "group": "goal"} for index in range(goal)] + [{"problem_id": "nowhere", "group": None}]


def _rungs(groups, resolved_of):
    return [_result(row["problem_id"], resolved_of[row["group"]]) for row in groups if row["group"] in ("below", "in", "above")]


def _reach(groups, resolved_ids):
    return [_result(row["problem_id"], 1 if row["problem_id"] in resolved_ids else 0, 32) for row in groups if row["group"] == "goal"]


def _read(by_round, reach_ids=None, control_ids=None, base_ids=frozenset({"goal0", "goal1"})):
    """Base: 1 of 8 below, 2 of 8 in, 6 of 8 above. `by_round[r]`: M(r)'s k of 8 on each rung."""
    groups = _groups()
    rungs = {number: _rungs(groups, by_round[number]) for number in by_round}
    reach = {number: _reach(groups, (reach_ids or {}).get(number, base_ids)) for number in by_round}
    control = None if control_ids is None else [_result(row["problem_id"], 2 if row["problem_id"] in control_ids else 0, 61) for row in groups if row["group"] == "goal"]
    return rounds_read(groups, _rungs(groups, {"below": 1, "in": 2, "above": 6}), _reach(groups, base_ids), rungs, reach, control, ROUNDS, resamples=200, seed=0)


def test_the_read_is_each_round_minus_the_base_the_climb_and_reach_against_both_comparators():
    read = _read({1: {"below": 1, "in": 3, "above": 7}, 2: {"below": 2, "in": 3, "above": 7}, 3: {"below": 3, "in": 4, "above": 7}},
                 reach_ids={1: {"goal0", "goal2"}, 2: {"goal0", "goal2", "goal3"}, 3: {"goal2", "goal3", "goal4", "goal5"}},
                 control_ids={"goal1", "goal2", "goal9"})
    assert read["rounds_measured"] == [1, 2, 3] and (read["first_model"], read["last_model"]) == ("M(1)", "M(3)")
    # Each round's rungs minus the base, paired by problem.
    assert [read["rungs_minus_base"][number]["below"]["mean"] for number in ROUNDS] == [0.0, 0.125, 0.25]
    assert read["rungs_minus_base"][3]["in"]["mean"] == 0.25 and read["rungs_minus_base"][1]["above"]["problems"] == 40
    assert read["primary"] == read["rungs_minus_base"][3]["below"] and read["primary"]["low"] == read["primary"]["high"] == 0.25
    # The climb: each rung, the last model minus the ONE-ROUND model.
    assert {rung: read["climb"][rung]["mean"] for rung in ("below", "in", "above")} == {"below": 0.25, "in": 0.125, "above": 0.0}
    # Reach on G, gained against lost: every round against the base afresh, and the last model against the one-round model.
    against_base = read["reach_against_the_base_afresh"]
    assert [(against_base[number]["gained"], against_base[number]["lost"]) for number in ROUNDS] == [(1, 1), (2, 1), (4, 2)]
    assert against_base[3]["sign_test_p"] == sign_test(4, 2) and against_base[3]["resolved_before"] == 2
    against_one = read["reach_last_model_against_the_first"]
    assert (against_one["gained"], against_one["lost"], against_one["resolved_before"]) == (3, 1, 2) and against_one["lost_problems"] == ["goal0"]
    # The control: M(3) in 32 episodes against the base in 32 + 61. The base resolves goal0, goal1 afresh and goal2, goal9 more.
    control = read["control"]
    assert (control["resolved_after"], control["resolved_before"], control["gained"], control["lost"]) == (4, 4, 3, 3)
    assert (control["episodes_after"], control["episodes_before"]) == (32, 93)
    assert read["stop_rule_fires_at"] == [] and read["branch"]["name"] == ESCALATE and "three seeds" in read["branch"]["reason"]


def test_a_loop_that_was_stopped_is_read_on_the_rounds_it_ran():
    read = _read({1: {"below": 1, "in": 3, "above": 7}, 2: {"below": 0, "in": 3, "above": 7}})
    assert read["rounds_measured"] == [1, 2] and read["climb"] is None and read["primary"] is None and read["control"] is None
    assert read["reach_last_model_against_the_first"] is None and set(read["reach_against_the_base_afresh"]) == {1, 2}
    assert read["stop_rule_fires_at"] == [2] and read["branch"]["name"] == STOP_AND_DIAGNOSE and "M(2)" in read["branch"]["reason"]


def _change(mean, low, high):
    return {"problems": 100, "mean": mean, "low": low, "high": high}


def _rounds(below, in_band=(0.04, 0.02, 0.06), above=(0.05, 0.03, 0.07)):
    """`below[i]`: round i + 1's (mean, low, high) on the below-band rung; the other rungs as one round measured them in L1."""
    return {number: {"below": _change(*below[number - 1]), "in": _change(*in_band), "above": _change(*above)} for number in range(1, len(below) + 1)}


def test_the_branch_follows_the_specs_read():
    fine = (0.01, -0.01, 0.03)
    # Primary (M(3) minus the base on the below-band rung) at or above zero: escalate to three seeds.
    assert l2_branch(_rounds([fine, fine, (0.02, -0.005, 0.045)]), ROUNDS).name == ESCALATE
    at_zero = l2_branch(_rounds([fine, fine, (0.0, -0.02, 0.02)]), ROUNDS)
    assert at_zero.name == ESCALATE and "at or above zero" in at_zero.reason
    # Below zero with an interval that CONTAINS zero: escalate too. The stop rule does not fire on it (fixture 11).
    contains = l2_branch(_rounds([fine, fine, (-0.01, -0.03, 0.001)]), ROUNDS)
    assert contains.name == ESCALATE and "contains zero" in contains.reason
    # An interval ENTIRELY below zero for ANY round's model stops the loop, whatever the last round reads.
    for stopped_at in ROUNDS:
        below = [fine, fine, fine]
        below[stopped_at - 1] = (-0.03, -0.05, -0.002)
        branch = l2_branch(_rounds(below), ROUNDS)
        assert branch.name == STOP_AND_DIAGNOSE and f"M({stopped_at})" in branch.reason and "not judged" in branch.reason
    # A loop that stopped after round 2 has no M(3): the stop is the branch.
    assert l2_branch(_rounds([fine, (-0.03, -0.05, -0.002)]), ROUNDS).name == STOP_AND_DIAGNOSE
    # VOID: round 1 with BOTH the in-band and the above-band rung at or below zero did not train.
    untrained = l2_branch(_rounds([fine, fine, fine], in_band=(0.0, -0.01, 0.01), above=(-0.01, -0.03, 0.01)), ROUNDS)
    assert untrained.name == VOID and "did not train" in untrained.reason
    assert l2_branch(_rounds([fine, fine, fine], in_band=(-0.01, -0.03, 0.01)), ROUNDS).name == ESCALATE          # one of the two is not both
    # VOID comes first: a round that did not train says nothing, even with an interval below zero.
    assert l2_branch(_rounds([(-0.03, -0.05, -0.002), fine, fine], in_band=(0.0, 0.0, 0.0), above=(0.0, 0.0, 0.0)), ROUNDS).name == VOID
    # Nothing measured, or a last round that was not measured with no stop to explain it.
    assert l2_branch({}, ROUNDS).name == VOID
    missing = l2_branch(_rounds([fine, fine]), ROUNDS)
    assert missing.name == VOID and "could not be computed" in missing.reason
    assert {ESCALATE, STOP_AND_DIAGNOSE, VOID} == set(BRANCHES)


# ------------------------------------------------------------------------------------ the challenger's table
def _pick(problem_id, how, predicted, score):
    return {"problem_id": problem_id, "how": how, "predicted_rate": predicted, "score": score}


def test_a_row_of_the_challengers_table_is_counted_from_its_picks_their_results_and_their_training_proofs():
    proposals = [_pick("a", SCORED, 0.30, 0.45), _pick("b", SCORED, 0.35, 0.46), _pick("c", SCORED, 0.40, 0.44), _pick("d", SCORED, 0.30, 0.45),
                 _pick("e", RANDOM_PLACE, 0.02, 0.05), _pick("f", RANDOM_PLACE, 0.90, 0.02)]
    results = [_result("a", 0), _result("b", 2, side="false"), _result("c", 5, side="false"), _result("d", 1), _result("e", 0), _result("f", 8),
               _result("another_batch", 3)]
    examples = [{"problem_id": "b", "side": "negation"}, {"problem_id": "c", "side": "negation"}, {"problem_id": "d", "side": "statement"},
                {"problem_id": "f", "side": "statement"}, {"problem_id": "another_batch", "side": "statement"}]
    row = picks_row(proposals, results, examples, T, FLOOR)
    scored = row["scored"]
    assert (scored["picks"], scored["known_false"], scored["share_known_false"]) == (4, 2, 0.5)
    assert scored["mean_pass_rate"] == pytest.approx(8 / 32) and scored["mean_predicted_rate"] == pytest.approx(0.3375)
    # k = 0, 2, 5 and 1 of 8: one at none, one below the band (k = 1), one in it (k = 2 or 3), one above (k >= 4).
    assert [scored[key] for key in ("share_at_k_0", "share_below_the_band", "share_in_the_band", "share_above_the_band")] == [0.25, 0.25, 0.25, 0.25]
    assert scored["mean_reward"] == pytest.approx((0 + reward(2, 8, T) + reward(5, 8, T) + reward(1, 8, T)) / 4, abs=1e-5)
    assert scored["mean_expected_reward"] == pytest.approx(0.45) and scored["calibration"]["problems"] == 4
    # Classes that do not move with the target rate: k = 0, k = 1 to 3 (here 2 and 1), k of 4 or more (here 5).
    assert (scored["share_at_k_0"], scored["share_at_k_1_to_3"], scored["share_at_k_4_or_more"]) == (0.25, 0.5, 0.25)
    assert (row["random_places"]["share_at_k_1_to_3"], row["random_places"]["share_at_k_4_or_more"], row["all"]["share_at_k_4_or_more"]) == (0.0, 0.5, round(2 / 6, 5))
    # At t = 1/10 the band is k = 1 of 8 alone: the shares against the band move, the reward moves, the fixed classes do not.
    lower = picks_row(proposals, results, examples, 0.10, FLOOR)["scored"]
    assert [lower[key] for key in ("share_at_k_0", "share_below_the_band", "share_in_the_band", "share_above_the_band")] == [0.25, 0.0, 0.25, 0.5]
    assert (lower["share_at_k_0"], lower["share_at_k_1_to_3"], lower["share_at_k_4_or_more"]) == (0.25, 0.5, 0.25) and lower["mean_reward"] != scored["mean_reward"]
    assert lower["mean_reward"] == pytest.approx((0 + reward(2, 8, 0.10) + reward(5, 8, 0.10) + reward(1, 8, 0.10)) / 4, abs=1e-5)
    assert row["random_places"]["picks"] == 2 and row["random_places"]["share_known_false"] == 0.0 and row["all"]["picks"] == 6
    assert row["all"]["share_known_false"] == pytest.approx(2 / 6, abs=1e-5)
    # The training set: four proofs from these picks; two from problems with k >= 4 (c and f), two refutations (b and c).
    assert row["training_set"] == {"examples": 4, "from_problems_above_the_band": 2, "from_refutations": 2, "share_from_problems_above_the_band": 0.5,
                                   "share_from_refutations": 0.5, "above_the_band_is_k_at_least": 4}
    with pytest.raises(ValueError, match="every proposed problem gets its episodes"):
        picks_row(proposals, results[1:], examples, T, FLOOR)
    empty = picks_row([], [], [], T, FLOOR)
    assert empty["scored"] == {"picks": 0} and empty["training_set"]["share_from_refutations"] is None


def test_the_predictors_calibration_is_read_on_what_it_predicted_before_the_episodes():
    table = picks_calibration([0.30, 0.30, 0.50, 0.02], [2, 4, 4, 0], [8, 8, 8, 8])
    assert table["problems"] == 4 and table["mean_predicted_rate"] == 0.28 and table["observed_rate"] == pytest.approx(10 / 32, abs=1e-4)
    assert [(row["problems"], row["observed_rate"]) for row in table["bins"]] == [(1, 0.0), (2, 0.375), (1, 0.5)]
    assert table["deviance"] < table["deviance_of_the_mean_rate"]          # it tells these problems apart better than one rate for all
    assert picks_calibration([], [], []) == {"problems": 0}


def test_the_trajectory_is_stated_before_the_run_and_a_share_that_does_not_fall_needs_a_decision():
    row = lambda share, rate: {"scored": {"share_known_false": share, "mean_pass_rate": rate}}      # noqa: E731
    falling = challenger_trajectory({1: row(0.42, 0.39), 2: row(0.30, 0.36), 3: row(0.11, 0.33)}, T)
    assert falling["compared"] == [1, 3] and falling["the_known_false_share_falls"] is True and falling["needs_a_decision"] is False
    assert falling["the_mean_pass_rate_moves_toward_the_target"] is True
    assert falling["known_false_share_of_the_scored_picks"] == {"1": 0.42, "2": 0.30, "3": 0.11}
    # Round 3's share AT round 1's is not a fall.
    level = challenger_trajectory({1: row(0.42, 0.39), 2: row(0.30, 0.36), 3: row(0.42, 0.45)}, T)
    assert level["the_known_false_share_falls"] is False and level["needs_a_decision"] is True and level["the_mean_pass_rate_moves_toward_the_target"] is False
    # One round only (a loop that stopped): nothing to compare.
    assert challenger_trajectory({1: row(0.42, 0.39)}, T)["compared"] is None
    assert challenger_trajectory({1: {"scored": {"picks": 0}}, 2: row(0.1, 0.3)}, T)["needs_a_decision"] is None


# ---------------------------------------------------------------------------------------------- the report
PREPARE = {"seed": 0, "fixture": False, "stand_in_engine": False, "base_distinct_attempts_on_the_rungs": {"prompts": 120, "mean_share_distinct": 0.95},
           "problems_a_round": 4, "batches": 2, "batch_sizes": [2, 2], "solvers": 8, "rung_episodes": 8, "reach_episodes": 32}
EVALUATION = {"bootstrap_resamples": 200, "bootstrap_seed": 0}


def _fit(observations, by_round):
    return {"observations": observations, "observations_by_the_round_that_gave_them": by_round, "fit": {"ridge": 10.0, "dispersion": 0.3},
            "candidates_not_yet_proposed": {"problems": 100, "known_false": 9}}


def _round(number, k_by_rung, reach_ids, false_picks, measured=True, stop=False, base_ids=frozenset({"goal0", "goal1"})):
    """Round `number` of two batches of two picks: one scored and one random place a batch; `false_picks` of its two
    scored picks are known false. Every pick resolved 2 of 8 but the last, which resolved none."""
    groups, names = _groups(), [f"r{number}p{index}" for index in range(4)]
    proposals = {batch: [_pick(names[2 * batch - 2], SCORED, 0.30, 0.45), _pick(names[2 * batch - 1], RANDOM_PLACE, 0.05, 0.10)] for batch in (1, 2)}
    results = {batch: [_result(names[2 * batch - 2], 2, side="false" if batch <= false_picks else "true"), _result(names[2 * batch - 1], 0 if batch == 2 else 2)] for batch in (1, 2)}
    examples = [{"problem_id": name, "side": "negation" if position // 2 < false_picks and position % 2 == 0 else "statement"} for position, name in enumerate(names[:3])]
    entry = {"summary": {"problems": 4, "attempt_episodes": 32, "reward": {"mean_reward": 0.75}, "sampling_seed": 1010 + number, "stand_in_engine": False},
             "proposals": proposals, "results": results, "examples": examples,
             "fits": {1: _fit(4 * number, {"1": {"observations": 4 * number, "weight": 1.0}}), 2: _fit(4 * number + 2, {"1": {"observations": 4 * number + 2, "weight": 1.0}})},
             "training": None, "measure": None, "rungs": None, "reach": None}
    if measured:
        entry.update({"training": {"model": f"M({number})", "examples": 3 * number, "rounds_trained_on": list(range(1, number + 1))},
                      "measure": {"distinct_attempts_on_the_rungs": {"prompts": 120, "mean_share_distinct": 0.9}, "stop_rule_fires": stop},
                      "rungs": _rungs(groups, k_by_rung), "reach": _reach(groups, reach_ids)})
    return entry


def _report(rounds, control_ids=None, stopped=None, base_rungs=None, trained_ids=None):
    groups = _groups()
    extra = lambda ids: [_result(row["problem_id"], 1 if row["problem_id"] in ids else 0, 61) for row in groups if row["group"] == "goal"]      # noqa: E731
    control = None if control_ids is None else {"summary": {"episodes_each": 61, "loop_attempt_episodes": 96}, "results": extra(control_ids)}
    trained = None if trained_ids is None else {"summary": {"episodes_each": 61, "set": "control_m3", "model": "M(3)"}, "results": extra(trained_ids)}
    arguments = (PREPARE, groups, base_rungs or _rungs(groups, {"below": 1, "in": 2, "above": 6}), _reach(groups, {"goal0", "goal1"}), rounds, control,
                 list(ROUNDS), SETTINGS, EVALUATION, {"embed": {"statements": 16}, "stopped_after_round": stopped})
    return build_l2_report(*arguments) if trained is None else build_l2_report(*arguments, trained)


def test_the_report_answers_the_read_fixed_before_the_run():
    rounds = {1: _round(1, {"below": 1, "in": 3, "above": 7}, {"goal0", "goal2"}, false_picks=2),
              2: _round(2, {"below": 2, "in": 3, "above": 7}, {"goal0", "goal2", "goal3"}, false_picks=1),
              3: _round(3, {"below": 3, "in": 4, "above": 7}, {"goal2", "goal3", "goal4", "goal5"}, false_picks=0)}
    report = _report(rounds, control_ids={"goal1", "goal2", "goal9"})
    assert report["ok"] is True and report["seed"] == 0 and report["rounds_planned"] == [1, 2, 3] == report["rounds_measured"] and "M(0) is the base" in report["naming"]
    assert report["branch"]["name"] == ESCALATE and report["stop_rule"]["fires_at_rounds"] == [] and report["stop_rule"]["the_loop_stopped_after_round"] is None
    assert report["void_conditions"] == {"round_1_left_both_the_in_band_and_the_above_band_rung_at_or_below_zero": False, "a_soundness_alarm": False}
    # Primary: the below-band rung, M(3) minus the base. Per round: each rung minus the base. The climb: M(3) minus M(1).
    assert report["primary"]["mean"] == 0.25 and "M(3) minus the base" in report["primary"]["what"]
    assert [report["rounds"][key]["rungs_minus_base"]["below"]["mean"] for key in ("1", "2", "3")] == [0.0, 0.125, 0.25]
    assert {rung: report["climb"]["by_rung"][rung]["mean"] for rung in ("below", "in", "above")} == {"below": 0.25, "in": 0.125, "above": 0.0}
    # Reach as gained against lost: M(3) against the base afresh AND against M(1); each round against the base.
    reach = report["reach_on_g"]
    assert (reach["last_model_against_the_base_afresh"]["gained"], reach["last_model_against_the_base_afresh"]["lost"]) == (4, 2)
    assert (reach["last_model_against_the_one_round_model"]["gained"], reach["last_model_against_the_one_round_model"]["lost"]) == (3, 1)
    assert reach["last_model_against_the_base_afresh"]["sign_test_p"] == sign_test(4, 2) and set(reach["by_round_against_the_base_afresh"]) == {"1", "2", "3"}
    assert report["rounds"]["2"]["reach_on_g_against_the_base_afresh"] == reach["by_round_against_the_base_afresh"]["2"]
    # The control: M(3) in 32 episodes against the base in 32 + 61.
    control = report["control"]["read"]
    assert (control["episodes_after"], control["episodes_before"], control["resolved_after"], control["resolved_before"]) == (32, 93, 4, 4)
    assert (control["gained"], control["lost"]) == (3, 3) and report["control"]["step"]["episodes_each"] == 61
    # The challenger's table by batch and by round, and the trajectory stated before the run.
    table = report["challenger"]
    assert [(row["round"], row["batch"], row["attempted_by"]) for row in table["by_batch"]] == [(1, 1, "M(0)"), (1, 2, "M(0)"), (2, 1, "M(1)"), (2, 2, "M(1)"), (3, 1, "M(2)"), (3, 2, "M(2)")]
    assert [row["scored"]["share_known_false"] for row in table["by_batch"]] == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert [table["by_round"][key]["scored"]["share_known_false"] for key in ("1", "2", "3")] == [1.0, 0.5, 0.0]
    assert table["by_round"]["1"]["all"]["picks"] == 4 and table["by_round"]["1"]["random_places"]["share_at_k_0"] == 0.5
    assert table["by_round"]["1"]["training_set"] == {"examples": 3, "from_problems_above_the_band": 0, "from_refutations": 2, "share_from_problems_above_the_band": 0.0,
                                                      "share_from_refutations": round(2 / 3, 5), "above_the_band_is_k_at_least": 4}
    assert table["by_batch"][1]["refit"]["observations"] == 6 and table["by_batch"][0]["scored"]["calibration"]["problems"] == 1
    assert table["trajectory"]["the_known_false_share_falls"] is True and table["trajectory"]["needs_a_decision"] is False
    assert report["rounds"]["3"]["training"]["rounds_trained_on"] == [1, 2, 3] and report["rounds"]["1"]["attempted_by"] == "M(0)" and report["rounds"]["1"]["trained"] == "M(1)"
    assert set(report["distinct_attempts_on_the_rungs"]) == {"base", "M(1)", "M(2)", "M(3)"} and report["heldout"]["goal_set"] == 30
    assert any("gain by k" in line for line in report["not_measured"]) and "gain_by_k" not in report
    headline = report["headline"]
    assert headline.startswith("L2 seed 0: ESCALATE. Primary (below-band rung, M(3) minus the base, 40 problems): 0.25 [0.25, 0.25]")
    assert "gained 4, lost 2" in headline and "against M(1) gained 3, lost 1" in headline and "against the base 4 in 93" in headline and "1.0, 0.5, 0.0" in headline
    # A run that has not had the trained model's extra attempts has no such part, and nothing says it should.
    assert "equal_attempts" not in report["control"] and "at equal attempts" not in headline and "control_m3" not in report["attempts"]


def test_at_equal_attempts_the_last_model_and_the_base_have_the_same_number_on_every_goal_problem():
    """Spec, "Added 2026-10-05": the last model gets the control's extra episodes too, so both have 32 + 61."""
    base = [_result("g1", 0, 32), _result("g2", 1, 32), _result("g3", 0, 32), _result("g4", 0, 32)]
    control = [_result("g1", 2, 61), _result("g2", 0, 61), _result("g3", 0, 61), _result("g4", 0, 61)]
    last = [_result("g1", 0, 32), _result("g2", 3, 32), _result("g3", 1, 32), _result("g4", 0, 32)]
    more = [_result("g4", 2, 61), _result("g1", 0, 61), _result("g2", 4, 61), _result("g3", 0, 61)]
    read = equal_attempts_read(last, more, base, control)
    assert (read["episodes_after"], read["episodes_before"], read["extra_episodes_each"]) == (93, 93, 61)
    # In 93 attempts the last model resolves g2, g3 and g4; the base g1 and g2.
    assert (read["problems"], read["resolved_after"], read["resolved_before"], read["by_both"], read["gained"], read["lost"]) == (4, 3, 2, 1, 2, 1)
    assert read["gained_problems"] == ["g3", "g4"] and read["lost_problems"] == ["g1"] and read["sign_test_p"] == sign_test(2, 1)
    # Successful episodes and successes per attempt of each model, over the same 4 x 93 attempts.
    assert (read["successful_episodes_of_the_last_model"], read["successful_episodes_of_the_base"]) == (10, 3)
    assert read["attempts_of_the_last_model"] == read["attempts_of_the_base"] == 4 * 93
    assert (read["successes_per_attempt_of_the_last_model"], read["successes_per_attempt_of_the_base"]) == (round(10 / 372, 6), round(3 / 372, 6))
    # It is another read than the control's, which gives the last model its 32 attempts only: g4 is not gained there.
    charged = control_read(last, base, control)
    assert (charged["episodes_after"], charged["episodes_before"], charged["gained"], charged["lost"]) == (32, 93, 1, 1)
    # Equal attempts are the point: another number on one side is refused, and so is another set of problems.
    with pytest.raises(ValueError, match="same number of attempts"):
        equal_attempts_read(last, [_result(row["problem_id"], 0, 60) for row in more], base, control)
    with pytest.raises(ValueError, match="same number of attempts"):
        equal_attempts_read(last, [*more[:3], _result("g3", 0, 32)], base, control)
    with pytest.raises(ValueError, match="same problems"):
        equal_attempts_read(last, more[:3], base, control)


def test_the_report_gains_the_equal_attempts_part_when_its_step_ran_and_changes_nothing_else():
    rounds = lambda: {1: _round(1, {"below": 1, "in": 3, "above": 7}, {"goal0", "goal2"}, false_picks=2),      # noqa: E731
                      2: _round(2, {"below": 2, "in": 3, "above": 7}, {"goal0", "goal2", "goal3"}, false_picks=1),
                      3: _round(3, {"below": 3, "in": 4, "above": 7}, {"goal2", "goal3", "goal4", "goal5"}, false_picks=0)}
    without = _report(rounds(), control_ids={"goal1", "goal2", "goal9"})
    report = _report(rounds(), control_ids={"goal1", "goal2", "goal9"}, trained_ids={"goal1", "goal7"})
    part = report["control"]["equal_attempts"]
    assert part["step"] == {"episodes_each": 61, "set": "control_m3", "model": "M(3)"} and "M(3)" in part["what"]
    read = part["read"]
    # M(3) in 32 + 61: goal2 to goal5 in its reach episodes, goal1 and goal7 in the extra ones. The base in 32 + 61:
    # goal0 and goal1 afresh, goal1, goal2 and goal9 in the control's.
    assert (read["episodes_after"], read["episodes_before"], read["resolved_after"], read["resolved_before"], read["by_both"]) == (93, 93, 6, 4, 2)
    assert (read["gained"], read["lost"], read["sign_test_p"]) == (4, 2, sign_test(4, 2))
    assert read["gained_problems"] == ["goal3", "goal4", "goal5", "goal7"] and read["lost_problems"] == ["goal0", "goal9"]
    assert (read["successful_episodes_of_the_last_model"], read["successful_episodes_of_the_base"]) == (6, 5)
    assert read["attempts_of_the_last_model"] == read["attempts_of_the_base"] == 30 * 93
    assert read["successes_per_attempt_of_the_last_model"] == round(6 / 2790, 6) and read["successes_per_attempt_of_the_base"] == round(5 / 2790, 6)
    clause = "; at equal attempts (93 each): M(3) 6 against the base 4, gained 4, lost 2 (p = 0.6875), 2.15 against 1.79 successes per 1,000 attempts"
    assert clause in report["headline"] and report["attempts"]["control_m3"]["attempts"] == 30 * 61
    # Nothing that was in the report changes: no existing key, not the branch, and the headline by that one clause.
    assert {key: value for key, value in report.items() if key not in ("headline", "control", "attempts")} == \
           {key: value for key, value in without.items() if key not in ("headline", "control", "attempts")}
    assert {key: value for key, value in report["control"].items() if key != "equal_attempts"} == without["control"]
    assert {key: value for key, value in report["attempts"].items() if key != "control_m3"} == without["attempts"]
    assert report["headline"].replace(clause, "") == without["headline"] and report["control"]["read"]["episodes_after"] == 32
    # The step without the control's results beside it, or a loop that stopped before the last round, has nothing to read.
    assert "equal_attempts" not in _report(rounds(), trained_ids={"goal1"})["control"]
    stopped = _report({1: _round(1, {"below": 0, "in": 3, "above": 7}, {"goal0"}, false_picks=1, stop=True)}, control_ids={"goal1"}, stopped=1, trained_ids={"goal1"})
    assert "equal_attempts" not in stopped["control"]
    # Too many of its attempts without an answer: its counts are not to be read, as for any set.
    unwell = _report(rounds(), control_ids={"goal1"}, trained_ids={"goal1"})
    assert unwell["ok"] is True
    trained = {"summary": {"episodes_each": 61}, "results": [{**_result(f"goal{index}", 0, 61), "attempts_without_an_answer": 5} for index in range(30)]}
    control = {"summary": {"episodes_each": 61}, "results": [_result(f"goal{index}", 0, 61) for index in range(30)]}
    groups = _groups()
    unwell = build_l2_report(PREPARE, groups, _rungs(groups, {"below": 1, "in": 2, "above": 6}), _reach(groups, {"goal0"}), rounds(), control, list(ROUNDS), SETTINGS,
                             EVALUATION, {"embed": None, "stopped_after_round": None}, trained)
    assert unwell["ok"] is False and unwell["not_to_be_read"] == ["control_m3"]


def test_the_report_of_an_arm_says_which_it_is_at_the_top_and_reads_every_reward_figure_at_its_target_rate():
    """Spec, "L2t: the lower target": the stage again with the challenger's target rate at 0.10. The report of a run
    with no arm is what it always was, with the fixed classes added to the challenger's table."""
    rounds = lambda: {number: _round(number, {"below": 2, "in": 3, "above": 7}, {"goal0", "goal2"}, false_picks=1) for number in ROUNDS}      # noqa: E731
    groups = _groups()
    arguments = lambda prepare, settings: (prepare, groups, _rungs(groups, {"below": 1, "in": 2, "above": 6}), _reach(groups, {"goal0", "goal1"}), rounds(), None,      # noqa: E731
                                           list(ROUNDS), settings, EVALUATION, {"embed": None, "stopped_after_round": None})
    plain = build_l2_report(*arguments(PREPARE, SETTINGS))
    lower = {**SETTINGS, "challenger": {**SETTINGS["challenger"], "target_rate": 0.10}}
    report = build_l2_report(*arguments({**PREPARE, "arm": "t010", "target_rate": 0.10}, lower))
    assert "arm" not in plain and plain["target_rate"] == 0.25 and plain["headline"].startswith("L2 seed 0: ")
    assert list(report)[:5] == ["spec", "headline", "ok", "seed", "arm"] and [key for key in report if key != "arm"] == list(plain)
    assert report["arm"]["name"] == "t010" and report["arm"]["target_rate"] == 0.10 == report["target_rate"] and "do not move" in report["arm"]["what"]
    assert report["headline"].startswith("L2 arm t010 (target rate 0.1) seed 0: ") and report["headline"].split("seed 0: ", 1)[1].split("; known-false")[0] == \
           plain["headline"].split("seed 0: ", 1)[1].split("; known-false")[0]
    # The band is the arm's (k = 1 of 8), and with it every share against the band and the mean reward...
    assert (report["band"]["low"], report["band"]["high"]) == (0.0485, 0.1749) and (plain["band"]["low"], plain["band"]["high"]) == pytest.approx((0.127, 0.409), abs=1e-3)
    ours, theirs = report["challenger"]["by_round"]["1"]["all"], plain["challenger"]["by_round"]["1"]["all"]
    assert ours["share_in_the_band"] == 0.0 and theirs["share_in_the_band"] == 0.75 and ours["share_above_the_band"] == 0.75 and ours["mean_reward"] < theirs["mean_reward"]
    assert report["challenger"]["trajectory"]["target_rate"] == 0.10
    # ... while the classes that do not move with t, the picks themselves and everything measured on the held-out sets are the same.
    fixed = ("picks", "known_false", "share_known_false", "mean_pass_rate", "share_at_k_0", "share_at_k_1_to_3", "share_at_k_4_or_more", "mean_predicted_rate")
    assert {key: ours[key] for key in fixed} == {key: theirs[key] for key in fixed} and (ours["share_at_k_0"], ours["share_at_k_1_to_3"], ours["share_at_k_4_or_more"]) == (0.25, 0.75, 0.0)
    for key in ("branch", "primary", "climb", "reach_on_g", "heldout", "rounds", "void_conditions", "stop_rule", "distinct_attempts_on_the_rungs"):
        assert report[key] == plain[key], key


def test_the_report_of_a_loop_that_stopped_or_lost_its_lean_answers_says_so():
    stopped = _report({1: _round(1, {"below": 0, "in": 3, "above": 7}, {"goal0"}, false_picks=1, stop=True)}, stopped=1)
    assert stopped["branch"]["name"] == STOP_AND_DIAGNOSE and stopped["rounds_measured"] == [1] and stopped["stop_rule"]["the_loop_stopped_after_round"] == 1
    assert stopped["primary"]["mean"] is None and stopped["climb"]["by_rung"] is None and stopped["control"] == {**stopped["control"], "step": None, "read": None}
    assert stopped["reach_on_g"]["last_model_against_the_base_afresh"] is None and "STOPPED after round 1" in stopped["headline"] and "control: not run" in stopped["headline"]
    # A round whose attempts are done and whose model is not yet measured is in the challenger's table and in no read.
    partial = _report({1: _round(1, {"below": 1, "in": 3, "above": 7}, {"goal0"}, false_picks=1), 2: _round(2, {}, set(), false_picks=0, measured=False)})
    assert partial["rounds_measured"] == [1] and set(partial["challenger"]["by_round"]) == {"1", "2"} and partial["rounds"]["2"]["rungs_minus_base"] is None
    assert partial["branch"]["name"] == VOID and "could not be computed" in partial["branch"]["reason"]
    # Round 1 with both upper rungs at or below zero: VOID, and the report states the condition.
    untrained = _report({number: _round(number, {"below": 2, "in": 2, "above": 6}, {"goal0"}, false_picks=1) for number in ROUNDS})
    assert untrained["branch"]["name"] == VOID and untrained["void_conditions"]["round_1_left_both_the_in_band_and_the_above_band_rung_at_or_below_zero"] is True
    # Too many attempts without an answer in a set: its counts are not to be read, and the headline says so.
    rounds = {number: _round(number, {"below": 2, "in": 3, "above": 7}, {"goal0"}, false_picks=1) for number in ROUNDS}
    rounds[2]["rungs"] = [{**row, "attempts_without_an_answer": 1} for row in rounds[2]["rungs"]]
    unwell = _report(rounds)
    assert unwell["ok"] is False and unwell["not_to_be_read"] == ["rungs_m2"] and "NOT TO BE READ" in unwell["headline"] and "rungs_m2" in unwell["headline"]
