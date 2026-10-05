"""The ladder loop's L1: the challenger, the round's training set, the read, the round's data, and the stage end to
end on the fixtures. Spec: docs/spec/ladder-loop.spec.md ("A round", "The challenger in this stage", "L1's
read, fixed now"; fixtures 2, 4 to 8, 10 to 13). The engine and Lean are stand-ins; nothing touches a GPU or the network."""

import contextlib
import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from rlvr_lean.data.ladder_round_export import (
    check_round_data,
    exported_candidate,
    facts_of,
    load_round_data,
    round_candidates,
    write_round_export,
)
from rlvr_lean.domain.ladder_round import ARMS, CHALLENGER_ARM, RANDOM_ARM
from rlvr_lean.domain.ladder_round.challenger import (
    FACT_FEATURES,
    RANDOM_DRAW,
    RANDOM_PLACE,
    SCORED,
    calibration,
    expected_reward,
    fact_features,
    features_of,
    fit_dispersion,
    fit_pass_rate_model,
    fit_projection,
    outcome_distribution,
    propose,
    random_draw,
    recency_weights,
)
from rlvr_lean.domain.ladder_round.read import (
    BRANCHES,
    ESCALATE,
    GOAL,
    NOT_NAMED,
    STOP_AND_DIAGNOSE,
    VOID,
    arm_reward,
    gain_by_k,
    group_ids,
    heldout_groups,
    l1_branch,
    paired_change,
    reach,
    stop_rule,
)
from rlvr_lean.domain.ladder_round.training_set import training_examples
from rlvr_lean.domain.problem_pool import SoundnessAlarm
from rlvr_lean.domain.problem_pool.episodes import ABOVE_BAND, BELOW_BAND, IN_BAND, NEGATION_SIDE, STATEMENT_SIDE, reward
from rlvr_lean.domain.verification import VerificationResult, VerificationStatus
from rlvr_lean.runner import entry
from rlvr_lean.gpu import ladder_loop, ladder_round, pipeline

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
SETTINGS = CONFIG["ladder_loop"]
T, FLOOR = SETTINGS["challenger"]["target_rate"], SETTINGS["challenger"]["band_reward"]
L0_FIXTURE, L1_FIXTURE = PACKAGE / "data" / "ladder_l0_fixture", PACKAGE / "data" / "ladder_l1_fixture"


# ------------------------------------------------------------------------------------- the expected reward
def test_k_around_a_rate_is_binomial_at_no_dispersion_and_wider_with_it():
    plain, wide = outcome_distribution(8, 0.25), outcome_distribution(8, 0.25, 0.5)
    assert plain.sum() == pytest.approx(1) and wide.sum() == pytest.approx(1)
    assert plain[2] == pytest.approx(28 * 0.25 ** 2 * 0.75 ** 6)
    mean = lambda distribution: sum(k * p for k, p in enumerate(distribution))      # noqa: E731
    assert mean(plain) == pytest.approx(2.0) and mean(wide) == pytest.approx(2.0)       # the mean rate is the predicted one
    assert wide[0] > plain[0] and wide[8] > plain[8]                                    # more problems at none and at all
    with pytest.raises(ValueError):
        outcome_distribution(8, 0.25, 1.0)


def test_a_challenger_that_maximises_expected_reward_aims_where_the_spec_says():
    """Spec, "Where the challenger aims": a true pass rate of 0.29 at t = 1/4, where 7% of proposals yield no proof."""
    rates = [index / 1000 for index in range(1, 1000)]
    best = max(rates, key=lambda rate: expected_reward(rate, 8, 0.25))
    assert best == pytest.approx(0.287, abs=0.01) and (1 - best) ** 8 == pytest.approx(0.07, abs=0.01)
    assert max(rates, key=lambda rate: expected_reward(rate, 8, 1 / 3)) == pytest.approx(0.358, abs=0.01)
    assert expected_reward(0.25, 8, 0.25) == pytest.approx(sum(outcome_distribution(8, 0.25)[k] * reward(k, 8, 0.25) for k in range(9)))
    # Problems of one predicted rate that differ among themselves earn less at the target: more of them land at 0 or n.
    assert expected_reward(0.29, 8, 0.25, dispersion=0.6) < expected_reward(0.29, 8, 0.25)


# ------------------------------------------------------------------------------------------- the predictor
def _synthetic(problems=400, seed=0, overdispersed=False):
    generator = np.random.default_rng(seed)
    features = generator.normal(size=(problems, 5))
    rates = 1 / (1 + np.exp(-(1.5 * features[:, 0] - 1.0)))
    if overdispersed:
        resolved = np.where(generator.random(problems) < rates, 8, 0)       # every problem is always or never resolved
    else:
        resolved = generator.binomial(8, rates)
    return [f"p{index}" for index in range(problems)], features, rates, resolved, np.full(problems, 8)


def _fit(ids, features, resolved, episodes, **changes):
    arguments = dict(names=[f"f{index}" for index in range(features.shape[1])], ridge_grid=[1.0, 10.0, 100.0], folds=5,
                     dispersion_grid=[0.0, 0.1, 0.3, 0.6, 0.9], seed=0)
    return fit_pass_rate_model(ids, features, resolved, episodes, **{**arguments, **changes})


def test_the_predictor_finds_a_planted_signal_and_says_how_it_was_chosen():
    ids, features, rates, resolved, episodes = _synthetic()
    model, chosen = _fit(ids, features, resolved, episodes)
    assert np.corrcoef(model.predict(features), rates)[0, 1] > 0.95
    assert chosen["ridge"] in (1.0, 10.0, 100.0) and min(chosen["deviance_by_ridge"].values()) < chosen["deviance_of_the_mean_rate"]
    assert chosen["dispersion"] <= 0.1                                      # the counts were binomial around the rate
    assert chosen["largest_coefficients"][0][0] == "f0" and len(chosen["out_of_fold_rates"]) == len(ids)
    again, _ = _fit(ids, features, resolved, episodes)
    assert np.allclose(again.predict(features), model.predict(features))    # a rerun is the same model


def test_problems_that_are_all_or_nothing_are_read_as_dispersed():
    ids, features, _, resolved, episodes = _synthetic(overdispersed=True)
    _, chosen = _fit(ids, features, resolved, episodes)
    assert chosen["dispersion"] >= 0.6
    best, likelihoods = fit_dispersion(np.full(50, 0.5), np.array([0, 8] * 25), np.full(50, 8), [0.0, 0.5, 0.9])
    assert best == 0.9 and likelihoods[0.9] > likelihoods[0.0]


def test_a_constant_feature_gets_no_weight_and_two_observations_are_the_least():
    ids, features, _, resolved, episodes = _synthetic(problems=60)
    features = np.hstack([features, np.ones((60, 1))])
    model, _ = _fit(ids, features, resolved, episodes)
    assert model.coefficients[-1] == pytest.approx(0, abs=1e-6) and np.all(np.isfinite(model.predict(features)))
    with pytest.raises(ValueError):
        _fit(ids[:1], features[:1], resolved[:1], episodes[:1])
    with pytest.raises(ValueError):
        _fit(ids, features, np.full(60, 9), episodes)                       # 9 of 8 is not k of n


def test_recent_rounds_weigh_most():
    assert list(recency_weights([0, 1, 2], 2, 0.5)) == [0.25, 0.5, 1.0]
    with pytest.raises(ValueError):
        recency_weights([3], 2, 0.5)
    ids, features, _, resolved, episodes = _synthetic(problems=200)
    old = np.array([0] * 100 + [5] * 100)                                   # the first hundred are five rounds old
    flipped = np.where(old == 0, 8 - resolved, resolved)                    # and say the opposite
    recent, _ = _fit(ids, features, flipped, episodes, weights=recency_weights(old, 5, 0.1))
    assert recent.coefficients[0] > 0                                       # the recent hundred decide the sign


def test_the_facts_the_challenger_reads_are_published_ones():
    rows = [{"kind": "stp_conjecture", "side": "true", "certificate_source": "stp", "published_proof_chars": 99, "stp_round": 25, "statement": "theorem a : 1 = 1 := by\n"},
            {"kind": "lean_workbook", "side": "false", "certificate_source": "internlm_rows", "published_proof_chars": None, "stp_round": None, "statement": "theorem b : 2 = 3 := by\n"}]
    features = dict(zip(FACT_FEATURES, fact_features(rows).T))
    assert list(features["is_stp_conjecture"]) == [1, 0] and list(features["is_known_false"]) == [0, 1]
    assert list(features["source_stp"]) == [1, 0] and list(features["source_internlm_rows"]) == [0, 1]
    assert list(features["has_stp_round"]) == [1, 0] and features["stp_round"][0] == 0.5 and features["log_published_proof_chars"][1] == 0
    projection = fit_projection(np.random.default_rng(0).normal(size=(10, 6)), 32)
    assert projection.basis.shape == (6, 6)                                 # never more directions than there are
    # Many statements or few, the leading directions are the same ones (up to sign) and are orthonormal.
    wide = np.random.default_rng(1).normal(size=(40, 6)) * np.array([9.0, 5.0, 3.0, 1.0, 0.5, 0.1])
    by_scatter, by_decomposition = fit_projection(wide, 3), fit_projection(wide[:5], 3)
    assert by_scatter.basis.shape == (3, 6) and by_decomposition.basis.shape == (3, 6)
    assert np.allclose(by_scatter.basis @ by_scatter.basis.T, np.eye(3), atol=1e-8)
    _, _, directions = np.linalg.svd(wide - wide.mean(axis=0), full_matrices=False)
    assert np.allclose(np.abs(by_scatter.basis @ directions[:3].T), np.eye(3), atol=1e-6)
    assert features_of(rows, np.zeros((2, 6)), projection).shape == (2, 6 + len(FACT_FEATURES))
    with pytest.raises(ValueError):
        features_of(rows, np.zeros((3, 6)), projection)


# ------------------------------------------------------------------------------------------------ choosing
def test_the_challenger_takes_the_best_and_gives_a_share_of_places_to_random_candidates():
    ids = [f"c{index}" for index in range(100)]
    scores = [index / 100 for index in range(100)]
    chosen = propose(ids, scores, 20, 0.1, seed=0)
    assert len(chosen) == 20 and len({row["problem_id"] for row in chosen}) == 20
    scored = [row["problem_id"] for row in chosen if row["how"] == SCORED]
    assert scored == [f"c{index}" for index in range(99, 81, -1)]            # the eighteen best, best first
    random_places = [row for row in chosen if row["how"] == RANDOM_PLACE]
    assert len(random_places) == 2 and all(int(row["problem_id"][1:]) <= 81 for row in random_places)
    assert propose(ids, scores, 20, 0.1, seed=0) == chosen and propose(ids, scores, 20, 0.1, seed=1) != chosen
    assert len(propose(ids[:5], scores[:5], 20, 0.1, seed=0)) == 5          # fewer candidates than places: all of them
    with pytest.raises(ValueError):
        propose(["a", "a"], [0.1, 0.2], 2, 0.1, seed=0)


def test_the_control_arm_is_a_uniform_seeded_draw_of_the_same_candidates():
    ids = [f"c{index}" for index in range(100)]
    high, low = random_draw(ids, 20, seed=0, scores=[1.0] * 100), random_draw(ids, 20, seed=0, scores=[0.0] * 100)
    assert [row["problem_id"] for row in high] == [row["problem_id"] for row in low]      # a score chooses nothing here
    assert all(row["how"] == RANDOM_DRAW for row in high) and len({row["problem_id"] for row in high}) == 20
    assert [row["problem_id"] for row in random_draw(ids, 20, seed=1)] != [row["problem_id"] for row in high]


def test_calibration_is_read_on_problems_the_model_was_not_fitted_on():
    ids, features, _, resolved, episodes = _synthetic()
    model, _ = _fit(ids[:300], features[:300], resolved[:300], episodes[:300])
    table = calibration(model.predict(features[300:]), resolved[300:], episodes[300:], T, FLOOR, model.dispersion)
    assert table["problems"] == 100 and table["deviance"] < table["deviance_of_the_mean_rate"]
    assert sum(row["problems"] for row in table["bins"]) == 100
    assert table["mean_reward_of_the_best_tenth_by_expected_reward"] > table["mean_reward_of_all"]
    assert calibration([], [], [], T, FLOOR, 0.0) == {"problems": 0}


# ------------------------------------------------------------------------------------- the training set
def _problem(problem_id, negation=True, side="true"):
    return {"problem_id": problem_id, "side": side, "statement": f"theorem {problem_id} : P := by\n",
            "negation": f"theorem negation_of_{problem_id} : ¬ (P) := by\n" if negation else None}


def _attempt(problem_id, side, episode, status, completion):
    return {"attempt_id": f"{problem_id}#round#{side}#{episode}", "problem_id": problem_id, "side": side, "episode": episode,
            "status": status, "completion": completion}


def test_a_round_trains_on_one_verified_proof_per_resolved_problem_on_the_side_that_was_proved():
    problems = [_problem("a"), _problem("b", side="false"), _problem("c")]
    attempts = [_attempt("a", STATEMENT_SIDE, index, "verified", f"  proof {'x' * index}\n") for index in range(6)]
    attempts += [_attempt("a", STATEMENT_SIDE, 6, "lean_error", "  wrong\n"),
                 _attempt("b", NEGATION_SIDE, 0, "verified", "  push_neg\n"), _attempt("b", STATEMENT_SIDE, 0, "lean_error", "  no\n"),
                 _attempt("c", STATEMENT_SIDE, 0, "lean_error", "  no\n"), _attempt("c", STATEMENT_SIDE, 1, "capped_tokens", "  long\n")]
    examples = training_examples(problems, attempts, seed=0)
    assert [example["problem_id"] for example in examples] == ["a", "b"]    # k = 0: no example
    by_id = {example["problem_id"]: example for example in examples}
    assert by_id["b"]["side"] == NEGATION_SIDE and by_id["b"]["theorem"] == problems[1]["negation"]
    assert by_id["a"]["theorem"] == problems[0]["statement"] and by_id["a"]["verified_attempts"] == 6
    verified = {attempt["completion"] for attempt in attempts if attempt["status"] == "verified"}
    assert all(example["completion"] in verified for example in examples)   # only what the solver wrote and Lean verified
    assert training_examples(problems, attempts, seed=0) == examples        # the same seed gives the same target
    chosen = {training_examples(problems, attempts, seed=seed)[0]["attempt_id"] for seed in range(40)}
    assert len(chosen) > 1 and chosen != {"a#round#statement#0"}            # the shortest proof has no special standing


def test_a_verified_negation_of_a_problem_without_one_is_refused():
    with pytest.raises(ValueError):
        training_examples([_problem("a", negation=False)], [_attempt("a", NEGATION_SIDE, 0, "verified", "  x\n")], seed=0)


# ------------------------------------------------------------------------------------------------ the read
def _result(problem_id, resolved, episodes=8, **extra):
    return {"problem_id": problem_id, "resolved": resolved, "episodes": episodes, "side": "true", "sides": 1, "resolved_by_statement": resolved,
            "resolved_by_negation": 0, "attempts_capped": 0, "attempts_timed_out": 0, "attempts_without_an_answer": 0, **extra}


def test_h_is_cut_into_the_goal_set_and_the_rungs_by_the_bases_placing_episodes():
    heldout = [{"problem_id": name, "heldout_part": part, "side": "true"} for name, part in
               (("w0", "lean_workbook"), ("s0", "stp_conjecture"), ("w3", "lean_workbook"), ("s8", "stp_conjecture"), ("w20", "lean_workbook"))]
    placing = {"w0": _result("w0", 0, 32), "s0": _result("s0", 0, 32), "w3": _result("w3", 3, 32), "s8": _result("s8", 8, 32), "w20": _result("w20", 20, 32)}
    groups = {row["problem_id"]: row["group"] for row in heldout_groups(heldout, placing, T, FLOOR)}
    # A rung holds only problems with a success; an STP conjecture with none is in no rung and not in G.
    assert groups == {"w0": GOAL, "s0": None, "w3": BELOW_BAND, "s8": IN_BAND, "w20": ABOVE_BAND}
    assert group_ids(heldout_groups(heldout, placing, T, FLOOR), GOAL) == ["w0"]


def test_a_change_is_paired_by_problem_and_needs_the_same_episodes_on_both_sides():
    after = [_result("a", 4), _result("b", 2), _result("c", 8)]
    before = [_result("a", 2), _result("b", 2), _result("c", 6)]
    change = paired_change(after, before, ["a", "b", "c"], resamples=500, seed=0)
    assert change["problems"] == 3 and change["mean"] == pytest.approx((0.25 + 0 + 0.25) / 3, abs=1e-5)
    assert change["low"] <= change["mean"] <= change["high"] and change["mean_before"] == pytest.approx(10 / 24, abs=1e-5)
    assert paired_change(after, before, [], 500, 0)["mean"] is None
    with pytest.raises(ValueError):
        paired_change(after, before, ["a", "missing"], 500, 0)
    with pytest.raises(ValueError):
        paired_change([_result("a", 4, 32)], [_result("a", 2, 8)], ["a"], 500, 0)


def test_the_gain_by_k_compares_fresh_with_fresh_and_groups_by_the_placing_k():
    """A problem placed at k = 1 by a noisy count reads higher next time with no training: the base's FRESH
    episodes take that out. The placing episodes only say which group a problem is in."""
    placed = [_result("a", 1), _result("b", 1), _result("c", 4)]
    base_fresh = [_result("a", 4), _result("b", 4), _result("c", 4)]        # the placing count was low by luck
    trained = [_result("a", 4), _result("b", 4), _result("c", 6)]
    table = {row["k"]: row for row in gain_by_k(placed, trained, base_fresh, resamples=200, seed=0)}
    assert table[1]["problems"] == 2 and table[1]["mean"] == 0.0             # not +0.375, which the placing count would give
    assert table[4]["problems"] == 1 and table[4]["mean"] == 0.25 and table[1]["of"] == 8


def test_the_stop_rule_fires_on_an_interval_entirely_below_zero_and_not_on_one_that_contains_it():
    assert stop_rule({"mean": -0.03, "low": -0.06, "high": -0.004}) is True
    assert stop_rule({"mean": -0.03, "low": -0.06, "high": 0.001}) is False
    assert stop_rule({"mean": 0.02, "low": 0.0, "high": 0.05}) is False
    assert stop_rule({"mean": None, "low": None, "high": None}) is False


def test_the_branch_follows_the_specs_read():
    primary = {"problems": 200, "mean": 0.02, "low": -0.01, "high": 0.05}
    assert l1_branch(primary, 0.5, 0.3, 200, 137).name == ESCALATE
    assert l1_branch({**primary, "mean": 0.0, "low": -0.02, "high": 0.02}, 0.5, 0.3, 200, 137).name == ESCALATE        # at zero
    assert l1_branch({**primary, "mean": -0.03, "low": -0.06, "high": -0.01}, 0.5, 0.3, 200, 137).name == STOP_AND_DIAGNOSE
    below = l1_branch({**primary, "mean": -0.01, "low": -0.04, "high": 0.02}, 0.5, 0.3, 200, 137)
    assert below.name == NOT_NAMED and "does not" not in below.reason and "not this case" in below.reason
    # VOID: the arm did not test aiming; the rung is too small; a measurement is missing.
    assert l1_branch(primary, 0.3, 0.3, 200, 137).name == VOID and "did not test aiming" in l1_branch(primary, 0.3, 0.3, 200, 137).reason
    assert l1_branch(primary, 0.5, 0.3, 136, 137).name == VOID
    assert l1_branch({"problems": 0, "mean": None, "low": None, "high": None}, 0.5, 0.3, 200, 137).name == VOID
    assert l1_branch(primary, None, 0.3, 200, 137).name == VOID
    assert {ESCALATE, STOP_AND_DIAGNOSE, VOID, NOT_NAMED} == set(BRANCHES)


def test_an_arms_reward_and_reach_are_counted_from_its_rows():
    rows = [_result("a", 0), _result("b", 2), _result("c", 3), _result("d", 8)]
    measured = arm_reward(rows, T, FLOOR)
    assert measured["mean_reward"] == pytest.approx((0 + reward(2, 8, T) + reward(3, 8, T) + 0) / 4, abs=1e-5)
    assert measured["share_in_the_band"] == 0.5 and measured["share_at_k_0"] == 0.25 and measured["share_at_k_n"] == 0.25
    assert measured["with_a_training_proof"] == 3 and measured["k_histogram"]["2"] == 1
    counted = reach([_result("g1", 2, 32), _result("g2", 0, 32), _result("g3", 1, 32)], [_result("g1", 1, 32), _result("g2", 0, 32), _result("g3", 0, 32)])
    assert counted["resolved_by_the_trained_model"] == 2 and counted["resolved_by_the_base_fresh"] == 1 and counted["only_the_trained_model"] == 1


# ------------------------------------------------------------------------------------- the round's data
def _l0_rows():
    read = lambda name: [json.loads(line) for line in (L0_FIXTURE / name).read_text().splitlines()]      # noqa: E731
    return read("heldout.jsonl"), read("base_map.jsonl"), json.loads((L0_FIXTURE / "summary.json").read_text())


def test_the_committed_round_fixture_is_what_its_summary_recorded_and_fits_the_l0_fixture():
    heldout, base_map, l0_summary = _l0_rows()
    candidates, facts, results, summary = load_round_data(L1_FIXTURE, l0_summary)
    assert summary["fixture"] is True and len(candidates) == 12
    assert check_round_data(heldout, base_map, candidates, facts, results, SETTINGS["goal"]["base_episodes"]) == {"candidates": 12, "heldout": 6, "base_map": 4}
    assert all("proof" not in key for row in candidates for key in row if key != "published_proof_chars")      # facts, never a proof's text


def test_round_data_that_breaks_the_held_out_rule_is_refused(tmp_path):
    heldout, base_map, l0_summary = _l0_rows()
    candidates, facts, results, _ = load_round_data(L1_FIXTURE, l0_summary)
    episodes = SETTINGS["goal"]["base_episodes"]
    in_h = [{**candidates[0], "problem_id": "fixture_h1"}, *candidates[1:]]
    with pytest.raises(ValueError, match="candidates are in H"):
        check_round_data(heldout, base_map, in_h, facts, results, episodes)
    with pytest.raises(ValueError, match="base-map sample"):
        check_round_data(heldout, base_map, [{**candidates[0], "problem_id": "fixture_p1"}, *candidates[1:]], facts, results, episodes)
    with pytest.raises(ValueError, match="do not cover H"):
        check_round_data(heldout, base_map, candidates, facts, results[1:], episodes)
    with pytest.raises(ValueError, match="another number of episodes"):
        check_round_data(heldout, base_map, candidates, facts, results, episodes + 1)
    with pytest.raises(ValueError, match="twice"):
        check_round_data(heldout, base_map, [candidates[0], candidates[0]], facts, results, episodes)
    # A data directory made for another L0 export, or changed after it was written, is refused.
    with pytest.raises(ValueError, match="another L0 data directory"):
        load_round_data(L1_FIXTURE, {"files": {"heldout.jsonl": {"sha256": "0" * 64}, "base_map.jsonl": {"sha256": "1" * 64}}})
    copy_of = tmp_path / "data"
    write_round_export(copy_of, candidates, facts, results, {"fixture": True, "l0_files": {name: entry["sha256"] for name, entry in l0_summary["files"].items()}})
    assert load_round_data(copy_of, l0_summary)[0] == candidates
    (copy_of / "candidates.jsonl").write_text((copy_of / "candidates.jsonl").read_text().replace("fixture_c1 ", "fixture_cX "))
    with pytest.raises(ValueError, match="SHA-256"):
        load_round_data(copy_of, l0_summary)


def test_the_candidates_are_a_seeded_draw_of_the_pool_without_what_is_barred():
    pool = [{"problem_id": f"p{index}", "kind": "lean_workbook", "side": "true", "statement": f"theorem p{index} : 1 = 1 := by\n", "rewritten": 0}
            for index in range(50)]
    barred = {"p0", "p1", "p2"}
    drawn = round_candidates(pool, barred, seed=7, wanted=20)
    assert len(drawn) == 20 and not barred & {row["problem_id"] for row in drawn}
    assert round_candidates(list(reversed(pool)), barred, seed=7, wanted=20) == drawn      # the order of the file does not matter
    assert round_candidates(pool, barred, seed=8, wanted=20) != drawn and len(round_candidates(pool, barred, 7, 100)) == 47
    # L2: the WHOLE pool (`wanted` None), in the same seeded order, so a smaller draw is its first rows.
    whole = round_candidates(pool, barred, seed=7, wanted=None)
    assert len(whole) == 47 and whole[:20] == drawn and not barred & {row["problem_id"] for row in whole}
    exported = exported_candidate(drawn[0], facts_of(drawn[0]["problem_id"], {drawn[0]["problem_id"]: "goedel"}, {drawn[0]["problem_id"]: 120}, {}))
    assert exported["certificate_source"] == "goedel" and exported["published_proof_chars"] == 120 and exported["stp_round"] is None


def test_the_export_tool_writes_what_the_round_reads_from_a_pool_store(tmp_path, monkeypatch):
    """`ladder_round export` on a small store in the pool tool's own formats: the candidates are pool problems
    outside H, the base map and what is set aside; each carries the length of the certificate that verified it."""
    from rlvr_lean.data.published import candidate_row
    from rlvr_lean.domain.problem_pool.certificates import Certificate
    from rlvr_lean.domain.problem_pool.selection import IN, Candidate, Verdict, pool_row, steps
    from rlvr_lean.tools import ladder_pool, ladder_round as tool

    heldout, base_map, l0_summary = _l0_rows()

    def candidate(problem_id, kind, side, statement, proofs):
        theorem = statement if side == "true" else statement.replace(f"theorem {problem_id}", f"theorem negation_of_{problem_id}")
        return Candidate(problem_id, kind, side, statement, statement, 0, problem_id,
                         tuple(Certificate(problem_id, side, "goedel" if side == "true" else "internlm_rows", theorem, proof) for proof in proofs))

    made = [candidate(row["problem_id"], row["kind"], row["side"], row["statement"], ["  wrong\n", "  norm_num\n"]) for row in base_map]
    made += [candidate(f"pool_{index}", "lean_workbook", "true", f"theorem pool_{index} : {index} = {index} := by\n", ["  rfl\n" + "  -- x\n" * index])
             for index in range(8)]
    made.append(candidate("aside_copy", "lean_workbook", "true", "theorem aside_copy : 1 = 1 := by\n", ["  rfl\n"]))
    pool_rows = []
    for item in made:
        checks = [check for check in steps(item) if check.kind == "certificate"]
        pool_rows.append(pool_row(item, Verdict(IN, certificate=item.certificates[-1], certificate_sha=checks[-1].sha, fingerprint=1)))
    store = tmp_path / "store"
    (store / "steps").mkdir(parents=True)
    write = lambda name, rows: (store / "steps" / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))      # noqa: E731
    write("pool.jsonl", [row for row in pool_rows if row["problem_id"] != "aside_copy"] + [next(row for row in pool_rows if row["problem_id"] == "aside_copy")])
    write("candidates.jsonl", [candidate_row(item) for item in made])
    write("set_aside.jsonl", [{"statement_id": "aside_copy"}])
    write("stp_index.jsonl", [])
    results = tmp_path / "l0b" / "steps"
    results.mkdir(parents=True)
    fixture_results = [json.loads(line) for line in (L1_FIXTURE / "base_results.jsonl").read_text().splitlines()]
    for set_name in ("heldout", "base_map"):
        (results / f"episodes_{set_name}_problems.jsonl").write_text(
            "".join(json.dumps({**row, "kind": "lean_workbook", "extra": "kept out"}) + "\n" for row in fixture_results if row["set"] == set_name))
    monkeypatch.setattr(ladder_pool, "local_files", lambda config, store, download: {})
    out = tmp_path / "ladder_l1"
    assert tool.export(SimpleNamespace(store=store, l0_results=results, l0_data=L0_FIXTURE, out=out), CONFIG) == 0
    candidates, facts, base_results, summary = load_round_data(out, l0_summary)
    assert sorted(row["problem_id"] for row in candidates) == [f"pool_{index}" for index in range(8)]      # not the base map, not what is set aside
    assert {row["problem_id"]: row["published_proof_chars"] for row in candidates}["pool_3"] == len("  rfl\n" + "  -- x\n" * 3)
    by_id = {row["problem_id"]: row for row in facts}
    assert by_id["fixture_p1"]["published_proof_chars"] == len("  norm_num\n") and by_id["fixture_p1"]["certificate_source"] == "goedel"
    assert by_id["fixture_p3"]["certificate_source"] == "internlm_rows" and by_id["fixture_p3"]["published_proof_chars"] == len("  norm_num\n")
    assert all("extra" not in row and "kind" not in row for row in base_results) and len(base_results) == 10
    assert summary["fixture"] is False and summary["counts"] == {"candidates": 8, "heldout": 6, "base_map": 4} and summary["problems_without_a_published_proof_length"] == 0
    assert all("proof" not in json.dumps({key: value for key, value in row.items() if key != "published_proof_chars"}) for row in candidates)
    # A store that is not the one L0's data came from is refused.
    write("pool.jsonl", [row for row in pool_rows if not row["problem_id"].startswith("fixture_p")])
    with pytest.raises(SystemExit, match="base-map problems are not in the pool file"):
        tool.export(SimpleNamespace(store=store, l0_results=results, l0_data=L0_FIXTURE, out=tmp_path / "other"), CONFIG)


def test_the_export_takes_the_configs_draw_a_number_or_the_whole_pool():
    from rlvr_lean.tools import ladder_round as tool

    assert tool.candidates_wanted(None, 20000) == 20000 and tool.candidates_wanted("500", 20000) == 500
    assert tool.candidates_wanted("all", 20000) is None                        # L2: every pool problem free to be a candidate
    for wrong in ("0", "-3", "most"):
        with pytest.raises(SystemExit, match="--candidates"):
            tool.candidates_wanted(wrong, 20000)


def test_the_committed_l2_data_is_the_whole_pool_without_h_and_the_base_map_and_holds_no_proof_text():
    """L2's candidates (spec, "Candidates: the whole pool"): every pool problem, less H and the base map, with
    statement and published FACTS only, tied by hash to the L0 data directory beside it."""
    if not all((PACKAGE / "data" / name / "summary.json").exists() for name in ("ladder_l0", "ladder_l1", "ladder_l2")):
        pytest.skip("the pool's data directories (src/rlvr_lean/data/ladder_l0, ladder_l1 and ladder_l2: derived from the published "
                    "datasets, about 30 MB) are not shipped. `python -m rlvr_lean.tools.ladder_pool export` rebuilds ladder_l0, and "
                    "`python -m rlvr_lean.tools.ladder_round export` the other two (ladder_l2 with `--candidates all`)")
    read = lambda name: [json.loads(line) for line in (PACKAGE / "data" / "ladder_l0" / name).read_text().splitlines()]      # noqa: E731
    heldout, base_map, l0_summary = read("heldout.jsonl"), read("base_map.jsonl"), json.loads((PACKAGE / "data" / "ladder_l0" / "summary.json").read_text())
    candidates, facts, results, summary = load_round_data(PACKAGE / "data" / "ladder_l2", l0_summary)
    counts = check_round_data(heldout, base_map, candidates, facts, results, SETTINGS["goal"]["base_episodes"])      # refuses a candidate of H or of the base map
    assert counts == {"candidates": summary["pool_problems_free_to_be_candidates"], "heldout": 2000, "base_map": 4000}
    assert summary["candidates_wanted"] == "all" and summary["fixture"] is False and summary["counts"] == counts
    assert summary["pool_problems"] == counts["candidates"] + counts["base_map"]       # nothing else of the pool is left out
    assert {key for row in candidates for key in row} == {"problem_id", "kind", "side", "statement", "rewritten", "certificate_source", "published_proof_chars", "stp_round"}
    assert all(row["published_proof_chars"] is not None for row in candidates)         # the LENGTH of the published proof, never its text
    # The base's placing results and the base map's facts are the ones L1 ran on.
    l1_summary = json.loads((PACKAGE / "data" / "ladder_l1" / "summary.json").read_text())
    assert all(summary["files"][name]["sha256"] == l1_summary["files"][name]["sha256"] for name in ("base_map_facts.jsonl", "base_results.jsonl"))


# ---------------------------------------------------------------------------- the stage, with stand-ins
KNOWN_FALSE = {"fixture_h4", "fixture_p3", "fixture_c7", "fixture_c8"}
NEVER = {"fixture_h3"}


class ScriptedLean:
    """A verification service by theorem name and proof text, in the shape of the fixture: a known-true statement
    is proved by a few tactics, a known-false one by its negation; `alarm` also proves the negation of a true one."""

    alarm, submitted = None, 0

    def __init__(self, settings=None):
        self.attempts = []

    def submit(self, attempts):
        self.attempts.extend(attempts)
        ScriptedLean.submitted += len(attempts)

    def results(self):
        found = {}
        for attempt in self.attempts:
            name = re.search(r"theorem (\S+)", attempt.statement).group(1)
            if name.startswith("negation_of_"):
                base = name[len("negation_of_"):]
                proved = (base in KNOWN_FALSE or base == ScriptedLean.alarm) and ("push_neg" in attempt.completion or "intro h" in attempt.completion)
            else:
                proved = name not in KNOWN_FALSE and name not in NEVER and any(tactic in attempt.completion for tactic in ("nlinarith", "linarith", "norm_num"))
            found[attempt.attempt_id] = VerificationResult(attempt.attempt_id, VerificationStatus.VERIFIED if proved else VerificationStatus.LEAN_ERROR,
                                                           verification_seconds=0.1, messages=() if proved else ("error: unsolved goals",))
        return found


@pytest.fixture
def stage(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "STORE", tmp_path / "store")
    monkeypatch.setenv(ladder_loop.DATA_VARIABLE, str(L0_FIXTURE))
    monkeypatch.setenv(ladder_round.ROUND_DATA_VARIABLE, str(L1_FIXTURE))
    monkeypatch.setenv(ladder_loop.STAND_IN_VARIABLE, "1")
    for name in (ladder_round.ROUND_RUN_VARIABLE, "RLVR_LEAN_STEP_DIR", "RLVR_LEAN_TRAINING_SEEDS", "RLVR_LEAN_TB_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ladder_loop, "_lean_settings", lambda config, lean_seconds=None: SimpleNamespace(pin=SimpleNamespace(name="the pin")))
    monkeypatch.setattr(ladder_round, "check_lean_sources", lambda settings, sources: {key: {"response": {"messages": []}} for key in sources})
    monkeypatch.setattr(ladder_loop, "lean_sessions", lambda config: contextlib.nullcontext(ScriptedLean))
    ScriptedLean.alarm, ScriptedLean.submitted = None, 0
    config = copy.deepcopy(CONFIG)
    config["evaluation"]["bootstrap_resamples"] = 200
    config["ladder_loop"]["episode"]["block_problems"] = 5
    return SimpleNamespace(config=config, store=lambda: ladder_round._store(config))


def _run_stage(config, until=None):
    summaries = {}
    for environment, step, _ in map(entry.step_fields, entry.STAGES["ladder_l1"]):
        if environment == "gpu" and step in ladder_round.STEPS:
            summaries[step] = ladder_round.STEPS[step](config)
            if step == until:
                break
    return summaries


def test_the_whole_stage_runs_on_the_fixtures_and_reads_void_for_a_rung_too_small(stage):
    summaries = _run_stage(stage.config)
    store = stage.store()
    assert store.root == pipeline.STORE / "runs-v4.27" / "ladder_l1_seed0"
    prepare = summaries["ladder_l1_prepare"]
    assert prepare["goal_set"] == 1 and prepare["rungs"] == {"below": 2, "in": 2, "above": 1} and prepare["rungs_below_the_minimum"] == ["below", "in", "above"]
    assert summaries["ladder_l1_embed"]["statements"] == 16 and summaries["ladder_l1_embed"]["stand_in_engine"] is True
    propose_summary = summaries["ladder_l1_propose"]
    assert propose_summary["problems"] == {CHALLENGER_ARM: 12, RANDOM_ARM: 12} and propose_summary["in_both_arms"] == 12
    assert propose_summary["challenger_fit"]["observations"] == 4
    for arm in ARMS:
        proposals = store.read_rows(f"proposals_{arm}.jsonl")
        results = {row["problem_id"]: row for row in store.read_rows(f"episodes_round_{arm}_problems.jsonl")}
        # Every proposed problem gets its n episodes: none is dropped or kept by a pass-rate estimate first.
        assert set(results) == {row["problem_id"] for row in proposals} and all(row["episodes"] == 8 for row in results.values())
        examples = store.read_rows(f"training_examples_{arm}.jsonl")
        attempts = [json.loads(line) for path in sorted(store.root.glob(f"episodes_round_{arm}_attempts_*.jsonl")) for line in path.read_text().splitlines()]
        verified = {(row["problem_id"], row["completion"]) for row in attempts if row["status"] == "verified"}
        assert examples and all((example["problem_id"], example["completion"]) in verified for example in examples)
        assert {example["problem_id"] for example in examples} == {key for key, row in results.items() if row["resolved"] > 0}
        assert all(example["side"] == NEGATION_SIDE for example in examples if example["problem_id"] in KNOWN_FALSE)
        assert summaries[f"ladder_l1_train_{arm}"]["examples"] == len(examples) and "no adapter was trained" in summaries[f"ladder_l1_train_{arm}"]["note"]
    hows = [row["how"] for row in store.read_rows(f"proposals_{CHALLENGER_ARM}.jsonl")]
    assert hows.count(SCORED) == 11 and hows.count(RANDOM_PLACE) == 1 and {row["how"] for row in store.read_rows(f"proposals_{RANDOM_ARM}.jsonl")} == {RANDOM_DRAW}
    report = summaries["ladder_l1_report"]
    assert report["fixture"] is True and report["stand_in_engine"] is True and report["ok"] is True
    assert report["branch"]["name"] == VOID and "2 problems" in report["branch"]["reason"]
    assert report["void_conditions"]["the_below_band_rung_has_fewer_problems_than_the_minimum"] is True
    assert report["primary"]["problems"] == 2 and report["heldout"] == {"goal_set": 1, "rungs": {"below": 2, "in": 2, "above": 1}, "rung_minimum": 137, "in_neither": 0}
    for arm in ARMS:
        own = report["gain_by_k"][arm]["the_rounds_own_problems"]
        assert sum(row["problems"] for row in own) == 12 and all(row["of"] == 8 for row in own)
        assert sum(row["problems"] for row in report["gain_by_k"][arm]["held_out_rungs_by_the_bases_placing_count"]) == 5
        assert report["also"]["reach_on_g"][arm]["problems"] == 1 and report["also"]["reach_on_g"][arm]["episodes_each"] == 32
        assert report["arms"][arm]["distinct_attempts"]["the_trained_model_on_the_rounds_problems"]["prompts"] >= 12
    assert json.loads(store.path("report_ladder_l1.json").read_text())["headline"] == report["headline"]
    # A rerun returns what is stored and samples nothing.
    sent = ScriptedLean.submitted
    again = _run_stage(stage.config)
    assert ScriptedLean.submitted == sent and again["ladder_l1_propose"] == summaries["ladder_l1_propose"]


def test_the_ruled_out_side_is_attempted_only_for_the_audited_problems(stage):
    """Decided 2026-10-04: until L3 the side a certificate rules out is not attempted, but for a seeded share."""
    assert SETTINGS["episode"]["contradicted_side"] == "audit" and SETTINGS["episode"]["skip_generating_contradicted_side"] is True
    _run_stage(stage.config, until="ladder_l1_round_challenger")
    store = stage.store()
    results = store.read_rows(f"episodes_round_{CHALLENGER_ARM}_problems.jsonl")
    assert {row["contradicted_side"] for row in results} <= {"not_generated", "checked"}
    attempts = [json.loads(line) for path in sorted(store.root.glob(f"episodes_round_{CHALLENGER_ARM}_attempts_*.jsonl")) for line in path.read_text().splitlines()]
    one_sided = {row["problem_id"] for row in results if row["contradicted_side"] == "not_generated"}
    assert one_sided and all(row["side"] == (NEGATION_SIDE if row["problem_id"] in KNOWN_FALSE else STATEMENT_SIDE)
                             for row in attempts if row["problem_id"] in one_sided)
    assert len(attempts) == sum(8 * row["sides"] for row in results)


def test_with_a_rung_large_enough_the_branch_is_read_from_the_primary(stage):
    stage.config["ladder_loop"]["goal"]["rung_minimum"] = 1
    report = _run_stage(stage.config)["ladder_l1_report"]
    assert report["branch"]["name"] in BRANCHES and report["void_conditions"]["the_below_band_rung_has_fewer_problems_than_the_minimum"] is False
    rewards = report["also"]["mean_reward_and_share_in_the_band"]
    # On the fixture both arms hold the same twelve problems, so the challenger's reward cannot be higher: VOID, and for that reason.
    assert rewards[CHALLENGER_ARM]["mean_reward"] == rewards[RANDOM_ARM]["mean_reward"]
    assert report["branch"]["name"] == VOID and "did not test aiming" in report["branch"]["reason"]


def test_a_proof_on_the_side_a_certificate_contradicts_stops_the_round(stage):
    stage.config["ladder_loop"]["episode"]["contradicted_side"] = "all"      # every problem's ruled-out side is checked
    ScriptedLean.alarm = "fixture_c1"
    for step in ("ladder_l1_prepare", "ladder_l1_embed", "ladder_l1_propose"):
        ladder_round.STEPS[step](stage.config)
    with pytest.raises(SoundnessAlarm, match="fixture_c1 is known true"):
        ladder_round.STEPS["ladder_l1_round_random"](stage.config)
    assert not stage.store().is_done("ladder_l1_round_random")
    assert list(stage.store().root.glob("episodes_round_random_attempts_*.jsonl"))       # the alarm leaves its evidence


def test_each_kind_of_measurement_has_a_seed_of_its_own_shared_by_the_arms_and_one_task_runs_one_seed(stage, monkeypatch):
    seed_of = lambda name: ladder_round.sampling_seed(stage.config, name)      # noqa: E731
    seeds = [seed_of(name) for name in ladder_round.SAMPLED_SETS]
    assert len(seeds) == 12 and len(set(seeds)) == 4 and min(seeds) == SETTINGS["round"]["sampling_seed"]
    # Both arms, and the base and the trained model, sample one kind of measurement with the same random numbers.
    assert seed_of("round_challenger") == seed_of("round_random")
    assert seed_of("reach_base") == seed_of("reach_challenger") == seed_of("reach_random")
    assert seed_of("gain_trained_challenger") == seed_of("gain_base_challenger") == seed_of("gain_base_random")
    assert seed_of("round_challenger") != seed_of("gain_base_challenger")       # the fresh re-attempt never reuses the round's samples
    assert SETTINGS["goal"]["sampling_seed"] not in seeds and SETTINGS["base_map"]["sampling_seed"] not in seeds     # never L0's placing seeds
    with pytest.raises(ValueError):
        seed_of("round_other_thing")
    monkeypatch.setenv("RLVR_LEAN_TRAINING_SEEDS", "2")
    assert ladder_round.training_seed(stage.config) == 2 and ladder_round._store(stage.config).root.name == "ladder_l1_seed2"
    assert not set(seeds) & {ladder_round.sampling_seed(stage.config, name) for name in ladder_round.SAMPLED_SETS}
    monkeypatch.setenv("RLVR_LEAN_TRAINING_SEEDS", "0,1")
    with pytest.raises(ValueError, match="ONE seed"):
        ladder_round.training_seed(stage.config)
    monkeypatch.setenv(ladder_round.ROUND_RUN_VARIABLE, "ladder_l1_smoke")
    monkeypatch.setenv("RLVR_LEAN_TRAINING_SEEDS", "0")
    assert ladder_round._store(stage.config).root.name == "ladder_l1_smoke"


def test_the_stage_is_registered_and_its_smoke_run_has_its_own_data_and_directory():
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l1"]]
    gpu_steps = [step for environment, step, _ in steps if environment == "gpu"]
    assert gpu_steps == ["fix_tokenizers", "ladder_l1_prepare", "ladder_l1_embed", "ladder_l1_propose", "ladder_l1_base",
                         "ladder_l1_round_challenger", "ladder_l1_train_challenger", "ladder_l1_measure_challenger",
                         "ladder_l1_round_random", "ladder_l1_train_random", "ladder_l1_measure_random", "ladder_l1_report"]
    assert all(step in ladder_round.STEPS for step in gpu_steps[1:]) and all(options == {} for _, _, options in steps)
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l1_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in steps]
    variables = entry.child_environment("gpu", "key", smoke[2][2])
    assert Path(variables[ladder_loop.DATA_VARIABLE]) == L0_FIXTURE and Path(variables[ladder_round.ROUND_DATA_VARIABLE]) == L1_FIXTURE
    assert variables[ladder_round.ROUND_RUN_VARIABLE] == "ladder_l1_smoke"
    assert ladder_round.ROUND_DATA_VARIABLE not in entry.child_environment("gpu", "key", steps[2][2])
