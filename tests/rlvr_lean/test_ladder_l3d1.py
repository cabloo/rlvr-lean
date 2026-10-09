"""L3d Step 1: the rows of its two trainings and their orders, what an H0 is refused for, the four checks, the branch
and the report on hand-made rows with a known answer. Spec: docs/spec/ladder-loop.spec.md, "L3d: train on what the
episode reaches" (Step 1; "The read, fixed before the run"; "Made exact by the build"). Pure: no model, no Lean. The
stage end to end is `test_ladder_l3d1_stage.py`."""

from pathlib import Path

import pytest
import yaml

from rlvr_lean.domain.ladder_round import l3d
from rlvr_lean.domain.ladder_round.dose import run_dose
from rlvr_lean.domain.ladder_round.l3d import (
    ARMS,
    BRANCHES,
    COST,
    INCONCLUSIVE,
    NOT_READ,
    NOT_SHOWN,
    TEACH,
    UNDETECTABLE,
    WITHIN_THE_NOISE,
    both_took,
    both_write_proofs,
    can_this_run_see_a_win,
    check_h0,
    could_have_been_seen,
    first_step,
    h0_rows,
    l3d1_branch,
    noise_band,
    opens_with_a_have,
    round_rows,
    seeded_order,
    tenth,
    trained_on_h0,
    with_h0,
)
from rlvr_lean.domain.problem_pool.selection import rank
from rlvr_lean.gpu.ladder_dose import _stand_in_calls
from rlvr_lean.reporting.ladder_l3d1 import build_l3d1_report, by_attempts_alone

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
SETTINGS = CONFIG["ladder_loop"]["l3d"]["step_1"]
EVALUATION = {"bootstrap_resamples": 400, "bootstrap_seed": 0}


def _example(problem_id, round_number=1, side="statement"):
    return {"problem_id": problem_id, "side": side, "attempt_id": f"{problem_id}#round_r{round_number}_b1#{side}#3", "theorem": f"theorem {problem_id} : P := by\n",
            "completion": "  step_one\n  done\n", "verified_attempts": 2, "round": round_number, "batch": 1}


def _harvested(problem_id, **changes):
    return {"problem_id": problem_id, "side": "statement", "theorem": f"theorem {problem_id} : Q := by\n", "completion": "  have h_g1 : A := by simp\n  linarith\n",
            "kind": "lean_workbook", "published_side": "true", "assembled": True, **changes}


ROUNDS = [[_example(f"p{index}", 1) for index in range(0, 14)], [_example(f"p{index}", 2) for index in range(14, 28)], [_example(f"p{index}", 3) for index in range(28, 40)]]
HARVEST = [_harvested(f"h{index}") for index in range(10)]


# ------------------------------------------------------------------------------------------------ the rows
def test_the_settings_are_the_specs():
    assert SETTINGS == {"loop_arm": "t010", "minimum_h0": 100, "loss_share_of_rows": 0.1, "maximum_share_without_an_answer": 0.05}
    assert CONFIG["ladder_loop"]["l2_arms"][SETTINGS["loop_arm"]] == {"target_rate": 0.10} and CONFIG["ladder_loop"]["ceiling"]["loop_arm"] == SETTINGS["loop_arm"]
    assert ARMS == ("without", "with") and l3d.L3D1 == "l3d1" and CONFIG["ladder_loop"]["measure"] == {"reach_episodes": 32, "rung_episodes": 8}


def test_withouts_rows_are_the_rounds_stored_training_examples_in_the_order_of_the_rounds():
    rows = round_rows(ROUNDS)
    assert [row["problem_id"] for row in rows] == [f"p{index}" for index in range(40)] and {row["source"] for row in rows} == {"rounds"}
    assert rows[0] == {"id": "p0#round_r1_b1#statement#3", "source": "rounds", "problem_id": "p0", "side": "statement", "theorem": "theorem p0 : P := by\n",
                       "completion": "  step_one\n  done\n"}
    with pytest.raises(ValueError, match="1 attempts stand twice in the rounds' training examples .first: p3#round_r1_b1#statement#3"):
        round_rows([ROUNDS[0], [ROUNDS[0][3]]])
    with pytest.raises(ValueError, match="1 problems stand twice in the rounds' training examples .first: p3"):       # one proof a problem, whatever the round
        round_rows([ROUNDS[0], [_example("p3", 2)]])
    harvest = h0_rows(HARVEST)
    assert [row["id"] for row in harvest] == [f"h{index}#h0" for index in range(10)] and {row["source"] for row in harvest} == {"h0"}
    assert set(harvest[0]) == {"id", "source", "problem_id", "side", "theorem", "completion"} and harvest[0]["completion"] == HARVEST[0]["completion"]


def test_an_h0_is_refused_for_what_must_not_be_trained_on_and_when_it_is_too_small():
    check_h0(HARVEST, {"held"}, {"mapped"}, {"p0", "p1"}, 10)                  # ten assembled proofs of other problems: taken
    for change, match in (
            (lambda rows: rows[2].update(problem_id="held"), "H0 holds 1 held-out problems .first: held"),
            (lambda rows: rows[2].update(problem_id="mapped"), "H0 holds 1 problems of the base map .first: mapped"),
            (lambda rows: rows[2].update(problem_id="p1"), "H0 holds 1 problems the rounds' training examples hold already .first: p1"),
            (lambda rows: rows[9].update(problem_id="h0"), "1 problems stand twice in H0 .first: h0.: one proof a problem"),
            (lambda rows: rows[4].update(assembled=False), "1 rows of H0 are not assembled proofs with their statement .first: row 4"),
            (lambda rows: rows[4].pop("assembled"), "1 rows of H0 are not assembled proofs"),
            (lambda rows: rows[7].update(completion=""), "1 rows of H0 are not assembled proofs with their statement .first: row 7"),
            (lambda rows: rows.pop(), "H0 holds 9 proofs and Step 1 asks for at least 10: with fewer it is not run on it, and the harvest is enlarged first")):
        rows = [dict(row) for row in HARVEST]
        change(rows)
        with pytest.raises(ValueError, match=match):
            check_h0(rows, {"held"}, {"mapped"}, {"p0", "p1"}, 10)


# ---------------------------------------------------------------------------------------------- the orders
def test_withouts_order_is_seeded_and_with_holds_the_same_rows_in_the_same_relative_order_with_h0_at_seeded_places():
    rows, harvest = round_rows(ROUNDS), h0_rows(HARVEST)
    without = seeded_order(rows, 0)
    assert [row["id"] for row in without] == sorted((row["id"] for row in rows), key=lambda key: rank(0, "l3d1_order", key))
    assert without != rows and seeded_order(list(reversed(rows)), 0) == without and seeded_order(rows, 1) != without      # by content, not by the order given; by the seed
    placed = with_h0(without, harvest, 0)
    assert len(placed) == 50 and [row for row in placed if row["source"] == "rounds"] == without                         # none of `without`'s rows is moved
    assert sorted(row["id"] for row in placed if row["source"] == "h0") == sorted(row["id"] for row in harvest)           # every row of H0, once
    # Each row of H0 stands in the gap its hash names: so many of `without`'s rows are before it.
    before = {row["id"]: sum(other["source"] == "rounds" for other in placed[:position]) for position, row in enumerate(placed) if row["source"] == "h0"}
    assert before == {row["id"]: int(rank(0, "l3d1_place", row["id"]), 16) % 41 for row in harvest}
    assert with_h0(without, list(reversed(harvest)), 0) == placed and with_h0(without, harvest, 1) != placed
    assert len({count for count in before.values()}) > 5                                                                  # spread among the rows, not in one block
    # Two rows of H0 in one gap stand in the order of their hashes; with no row to stand among they are all of it.
    alone = with_h0([], harvest, 0)
    assert [row["id"] for row in alone] == sorted((row["id"] for row in harvest), key=lambda key: rank(0, "l3d1_place", key)) and with_h0(without, [], 0) == without


# ------------------------------------------------------------------------------------ how a model writes
def test_a_proofs_first_step_is_its_first_line_that_is_neither_empty_nor_a_comment():
    assert first_step("\n  -- a plan\n  have h : A := by simp\n  linarith\n") == "have h : A := by simp" and first_step("  nlinarith [sq_nonneg x]\n") == "nlinarith [sq_nonneg x]"
    assert first_step("") == "" and first_step("  -- only a comment\n") == ""
    attempts = [{"completion": text} for text in ("  have h : A := by simp\n  done\n", "  have : B := by\n    simp\n", "  haveI := inst\n  simp\n", "  intro x\n  have h : A := h0\n",
                                                  "  -- first\n  have h := foo\n", "", None)]
    assert opens_with_a_have(attempts) == {"attempts": 7, "with_a_have": 3, "share": round(3 / 7, 5)}       # `haveI` is another word; a `have` further down does not count
    assert opens_with_a_have([]) == {"attempts": 0, "with_a_have": 0, "share": None}


# ---------------------------------------------------------------------------------------------- the checks
def _rung(problem_id, resolved, episodes=8, capped=0, no_answer=0, sides=1):
    return {"problem_id": problem_id, "kind": "lean_workbook", "side": "true", "episodes": episodes, "resolved": resolved, "sides": sides,
            "attempts_capped": capped, "attempts_timed_out": 0, "attempts_without_an_answer": no_answer, "attempts_not_checked": 0}


def test_both_trainings_took_when_the_last_tenth_of_each_ones_rows_cost_less_than_the_first_tenth():
    assert (tenth(1685, 0.1), tenth(1900, 0.1), tenth(12, 0.1), tenth(3, 0.1)) == (168, 190, 1, 1)
    falling, flat = [2.0 - index / 1000 for index in range(1685)], [1.0] * 1900
    took = both_took({"without": falling, "with": [2.0 - index / 1000 for index in range(1900)]}, 0.1)
    assert took["passes"] is True and took["without"]["rows_compared"] == 168 and took["with"]["rows_compared"] == 190
    assert took["without"]["first"] == pytest.approx(1.9165) and took["without"]["last"] == pytest.approx(0.3995)
    one_flat = both_took({"without": falling, "with": flat}, 0.1)
    assert one_flat["passes"] is False and one_flat["without"]["passes"] is True and one_flat["with"]["passes"] is False       # one of the two is enough to fail


def test_both_models_still_write_proofs_while_few_of_each_ones_attempts_get_no_answer():
    fine, capped = [_rung(f"r{index}", 4) for index in range(10)], [*[_rung(f"r{index}", 4) for index in range(9)], _rung("r9", 0, capped=4)]
    assert both_write_proofs({"without": fine, "with": fine}, 0.05)["passes"] is True
    one = both_write_proofs({"without": fine, "with": capped}, 0.05)                    # 4 of 80 is not under 5%
    assert one["passes"] is False and one["with"]["share"] == 0.05 and one["without"]["passes"] is True


def _orders():
    without = seeded_order(round_rows(ROUNDS), 0)
    return {"without": without, "with": with_h0(without, h0_rows(HARVEST), 0)}


def _trained(orders):
    return {arm: [row["id"] for row in rows] for arm, rows in orders.items()}


def test_with_was_trained_on_h0_when_every_one_of_its_rows_is_in_the_training_loops_record_once():
    orders = _orders()
    trained = _trained(orders)
    passed = trained_on_h0(orders, trained)
    assert passed["passes"] is True and (passed["h0_rows"], passed["counted_once_in_with"], passed["counted_in_without"]) == (10, 10, 0)
    assert passed["rows_trained"] == {"without": 40, "with": 50} == passed["rows_prepared"] and passed["as_prepared"] == {"without": True, "with": True}
    a_row = next(row["id"] for row in orders["with"] if row["source"] == "h0")
    missing = trained_on_h0(orders, {**trained, "with": [key for key in trained["with"] if key != a_row]})
    assert missing["passes"] is False and missing["counted_once_in_with"] == 9
    twice = trained_on_h0(orders, {**trained, "with": [*trained["with"], a_row]})
    assert twice["passes"] is False and twice["counted_once_in_with"] == 9                                              # twice is not once
    leaked = trained_on_h0(orders, {**trained, "without": [*trained["without"], a_row]})
    assert leaked["passes"] is False and leaked["counted_in_without"] == 1                                              # `without` must hold none
    # ... even were it prepared so: an order of `without` that held a row of H0, trained as prepared, fails on that row alone.
    wrong = {"without": [*orders["without"], next(row for row in orders["with"] if row["source"] == "h0")], "with": orders["with"]}
    prepared_so = trained_on_h0(wrong, _trained(wrong))
    assert prepared_so["as_prepared"] == {"without": True, "with": True} and prepared_so["counted_in_without"] == 1 and prepared_so["passes"] is False
    moved = trained_on_h0(orders, {**trained, "with": list(reversed(trained["with"]))})
    assert moved["passes"] is False and moved["counted_once_in_with"] == 10 and moved["as_prepared"] == {"without": True, "with": False}      # every row, in another order
    assert trained_on_h0({"without": orders["without"], "with": orders["without"]}, {"without": trained["without"], "with": trained["without"]})["passes"] is False   # no H0 at all
    # The record is the training loop's own: the positions `run_dose` gives are the rows each optimizer step was made on.
    train_step, read, save = _stand_in_calls(50, [], [])
    steps = run_dose(50, 8, 1, 0, [], {"with": 7}, train_step, read, save, orders=[list(range(50))], per_example=True, positions=True)["training_steps"]
    assert [row["example_positions"] for row in steps] == [list(range(start, min(start + 8, 50))) for start in range(0, 50, 8)]
    assert [len(row["example_losses"]) for row in steps] == [8, 8, 8, 8, 8, 8, 2]
    plain = run_dose(50, 8, 1, 0, [], {"with": 7}, *_stand_in_calls(50, [], []), orders=[list(range(50))], per_example=True)["training_steps"]
    assert all("example_positions" not in row for row in plain)                                                         # asked for by this stage only


def test_the_four_checks_are_read_together_and_any_failed_one_makes_the_run_inconclusive():
    orders, rungs = _orders(), [_rung("a1", 6), _rung("b1", 1)]
    losses = {arm: [2.0] * 25 + [1.0] * 25 for arm in ARMS}
    fine = can_this_run_see_a_win(10, 10, losses, {arm: rungs for arm in ARMS}, orders, _trained(orders), SETTINGS)
    assert list(fine) == ["h0_holds_enough_proofs", "both_trainings_took", "both_models_still_write_proofs", "with_was_trained_on_h0"]
    assert all(check["passes"] for check in fine.values())
    small = can_this_run_see_a_win(99, 100, losses, {arm: rungs for arm in ARMS}, orders, _trained(orders), SETTINGS)
    assert small["h0_holds_enough_proofs"] == {"what": "H0 holds at least 100 proofs", "h0_rows": 99, "minimum": 100, "passes": False}
    branch = l3d1_branch(small, _change(0.02, 0.01, 0.03), _change(0.0, -0.001, 0.001), 99)
    assert branch["name"] == INCONCLUSIVE and branch["failed_checks"] == ["h0_holds_enough_proofs"] and "nothing is said about what assembled proofs teach" in branch["reason"]


# ---------------------------------------------------------------------------------------------- the branch
PASSING = {"a": {"passes": True}}
CEILING = {"small": {"rows": 2000, "mean": 0.002}, "full": {"rows": 8000, "mean": 0.004}}     # +0.000001 and +0.0000005 for each proof trained on


def _change(mean, low, high):
    return {"problems": 230, "mean": mean, "low": low, "high": high}


def test_the_noise_floor_is_read_as_a_size_whichever_training_is_subtracted_from_the_other():
    assert noise_band(_change(-0.0005, -0.0012, 0.0002)) == 0.0012 == noise_band(_change(0.0005, -0.0002, 0.0012))      # the same floor, the two trainings the other way round
    assert noise_band(_change(0.0003, 0.0001, 0.0006)) == 0.0006 and noise_band(None) is None and noise_band({"mean": None, "low": None, "high": None}) is None


def test_the_branch_is_the_specs_and_no_primary_within_the_noise_floor_is_read_as_an_effect():
    floor = _change(-0.0005, -0.0012, 0.0002)                                  # two trainings on the same data differ by up to 0.0012 either way
    teach = l3d1_branch(PASSING, _change(0.0030, 0.0010, 0.0050), floor, 300)
    assert teach["name"] == TEACH and teach["noise_band"] == 0.0012 and "assembled proofs teach" in teach["reason"]
    assert "A matched control is trained first" in teach["reason"] and "ONE-SHOT proofs as H0 holds" in teach["reason"] and "then Step 2 goes to three seeds" in teach["reason"]
    # Clear of zero and INSIDE the floor: not an effect. So is +0.0004, which lies outside the floor's own interval [-0.0012, +0.0002] and inside its size.
    within = l3d1_branch(PASSING, _change(0.0012, 0.0002, 0.0022), floor, 300)
    assert within["name"] == WITHIN_THE_NOISE and "it is not read as an effect" in within["reason"] and "0.00120 either way" in within["reason"]
    assert l3d1_branch(PASSING, _change(0.0004, 0.0001, 0.0007), floor, 300)["name"] == WITHIN_THE_NOISE
    assert l3d1_branch(PASSING, _change(0.00121, 0.0002, 0.0022), floor, 300)["name"] == TEACH
    assert l3d1_branch(PASSING, _change(-0.0010, -0.0018, -0.0002), floor, 300)["name"] == WITHIN_THE_NOISE              # below zero, and within it too
    cost = l3d1_branch(PASSING, _change(-0.0030, -0.0050, -0.0010), floor, 300)
    assert cost["name"] == COST and "the assembled proofs cost one-shot attempts. Step 2 is not run as designed" in cost["reason"]
    # The interval holds zero: with the ceiling's two doses beside it, could a gain of that size for each proof have been seen here?
    seen = l3d1_branch(PASSING, _change(0.0001, -0.0002, 0.0004), _change(0.0, -0.0002, 0.0002), 400, CEILING)           # resolves 0.0003; 400 proofs would give 0.0004
    assert seen["name"] == NOT_SHOWN and seen["could_have_been_seen"]["seen"] is True and "not shown at H0's size" in seen["reason"]
    unseen = l3d1_branch(PASSING, _change(0.0001, -0.0002, 0.0004), _change(0.0, -0.0002, 0.0002), 250, CEILING)         # 250 proofs would give 0.00025 at most
    assert unseen["name"] == UNDETECTABLE and "undetectable at this size" in unseen["reason"] and "decided on the ceiling's read" in unseen["reason"]
    assert l3d1_branch(PASSING, _change(0.0001, -0.0002, 0.0004), floor, 400, CEILING)["name"] == UNDETECTABLE            # the floor is larger than the interval: 0.0012 is what is resolved
    unknown = l3d1_branch(PASSING, _change(0.0001, -0.0002, 0.0004), floor, 400)                                         # the ceiling's report was not read
    assert unknown["name"] == NOT_SHOWN and unknown["could_have_been_seen"]["seen"] is None and "is not said" in unknown["reason"]
    assert l3d1_branch(PASSING, _change(0.001, 0.0, 0.002), floor, 300)["name"] == NOT_SHOWN                              # an interval that ends at zero holds it
    assert l3d1_branch(PASSING, {"problems": 0, "mean": None, "low": None, "high": None}, floor, 300)["name"] == NOT_READ
    no_floor = l3d1_branch(PASSING, _change(0.0030, 0.0010, 0.0050), None, 300)
    assert no_floor["name"] == NOT_READ and "no primary is read as an effect without one" in no_floor["reason"]
    assert l3d1_branch({"a": {"passes": False}, "b": {"passes": True}}, _change(0.0030, 0.0010, 0.0050), floor, 300)["failed_checks"] == ["a"]
    assert {TEACH, COST, NOT_SHOWN, UNDETECTABLE, WITHIN_THE_NOISE, INCONCLUSIVE, NOT_READ} == set(BRANCHES)


def test_what_could_have_been_seen_is_the_ceilings_gain_for_each_proof_times_h0_against_what_this_run_resolves():
    note = could_have_been_seen(_change(0.0001, -0.0002, 0.0004), 0.0002, 400, CEILING)
    assert note["resolves"] == 0.0003 and note["h0_rows"] == 400 and note["seen"] is True
    assert note["doses"]["small"] == {"rows": 2000, "gain": 0.002, "gain_per_proof": 0.002 / 2000, "expected_here": 0.0004, "could_have_been_seen": True}
    assert note["doses"]["full"]["expected_here"] == 0.0002 and note["doses"]["full"]["could_have_been_seen"] is False      # either dose is enough
    assert could_have_been_seen(_change(0.0001, -0.0002, 0.0004), 0.0009, 400, CEILING)["resolves"] == 0.0009            # the noise band, when it is the larger
    nothing = could_have_been_seen(_change(0.0001, -0.0002, 0.0004), 0.0002, 400, {"full": {"rows": 8000, "mean": -0.001}})
    assert nothing["seen"] is False and nothing["doses"]["full"]["expected_here"] == -0.00005                             # a ceiling that shows no gain: none to be seen
    assert could_have_been_seen(_change(0.0001, -0.0002, 0.0004), 0.0002, 400, None) == {
        "what": "the ceiling's two doses were not read: whether a gain of the ceiling's size per proof could have been seen here is not said",
        "resolves": 0.0003, "doses": None, "seen": None}


# ---------------------------------------------------------------------------------- the report, hand-made rows
GROUPS = [{"problem_id": f"g{lines}{letter}", "group": "goal"} for lines in ("1", "2", "4", "8") for letter in "abc"] + \
         [{"problem_id": f"{rung}{index}", "group": rung} for rung in ("below", "in", "above") for index in (1, 2)] + [{"problem_id": "nowhere", "group": None}]
LENGTHS = {f"g{lines}{letter}": {"length_group": group} for lines, group in (("1", "1"), ("2", "2-3"), ("4", "4-7"), ("8", "8+")) for letter in "abc"}
GOAL_IDS = [row["problem_id"] for row in GROUPS if row["group"] == "goal"]
RUNG_IDS = [row["problem_id"] for row in GROUPS if row["group"] in ("below", "in", "above")]
# Successes by problem in (the 32 attempts, the 61 more), in GOAL_IDS' order: 1 line, 2-3 lines, 4-7 lines, 8 or more.
ON_G = {"base": ([2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0], [3, 3, 3, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
        "without": ([4, 4, 4, 2, 2, 2, 1, 0, 0, 0, 0, 0], [5, 5, 5, 3, 3, 3, 1, 1, 0, 0, 0, 0]),
        "with": ([4, 4, 4, 2, 2, 2, 3, 2, 2, 1, 1, 1], [5, 5, 5, 3, 3, 3, 4, 3, 2, 2, 1, 1]),
        "loop": ([4, 4, 4, 2, 2, 2, 1, 0, 0, 0, 0, 0], [5, 5, 5, 3, 3, 3, 1, 1, 0, 0, 0, 1])}
ON_THE_RUNGS = {"base": [1, 1, 2, 3, 6, 7], "without": [1, 2, 3, 3, 6, 6], "with": [1, 2, 3, 4, 6, 7], "loop": [2, 2, 3, 4, 7, 7]}
# Each goal problem's 11 episodes of 8 one-shot attempts, resolved or not: how many of them were.
EPISODES = {"base": [3, 3, 3, 1, 1, 1, 1, 1, 0, 0, 0, 0], "without": [6, 6, 6, 3, 3, 3, 2, 1, 0, 0, 0, 0], "with": [6, 6, 6, 3, 3, 3, 6, 5, 3, 3, 2, 1],
            "loop": [7, 6, 6, 3, 3, 3, 2, 1, 0, 0, 0, 1]}
PREPARE = {"seed": 0, "stand_in_engine": False, "attempts_a_goal_problem": 93, "rung_episodes": 8, "stored_runs": {"base": "ladder_l2_seed0", "loop": "ladder_l2_t010_seed0"},
           "loop_arm": "t010", "loop_target_rate": 0.1, "sampling_seeds": {"rungs": 1002, "reach": 1001, "more": 1020}, "stored_measurements": {},
           "goal_samplings": [{"name": "reach", "episodes": 32, "sampling_seed": 1001}, {"name": "more", "episodes": 61, "sampling_seed": 1020}],
           "contradicted_side_setting": "audit", "h0_rows": 10, "minimum_h0": 10, "h0_file": "harvest_h0.jsonl", "h0_file_sha256": "0" * 64, "h0_sides": {"statement": 10},
           "rounds_files": {"training_examples_r1.jsonl": 14, "training_examples_r2.jsonl": 14, "training_examples_r3.jsonl": 12}, "rounds_rows": 40,
           "trainings": {"without": {"rows": 40, "h0_rows": 0, "steps": 5, "tokens": 4000}, "with": {"rows": 50, "h0_rows": 10, "steps": 7, "tokens": 5200}},
           "recipe": {"what": "the round's"}, "order": {"what": "seeded", "seed": 0}, "longest_example_tokens": 300, "max_sequence_tokens": 2048}
TRAINS = {arm: {"stand_in_engine": False, "steps": PREPARE["trainings"][arm]["steps"], "model": arm} for arm in ARMS}


def _model(name, rows=None, rungs=None, on_g=None):
    first, more = on_g or ON_G[name]
    return {"rows": rows, "rungs": [_rung(problem_id, resolved) for problem_id, resolved in zip(RUNG_IDS, rungs or ON_THE_RUNGS[name])],
            "goal": [[_rung(problem_id, resolved, episodes=32) for problem_id, resolved in zip(GOAL_IDS, first)],
                     [_rung(problem_id, resolved, episodes=61) for problem_id, resolved in zip(GOAL_IDS, more)]],
            "verified_proof_lines": {"goal": {"1": 6, "2": 2, "5": 1, "9": 1}, "rungs": {"1": 20, "3": 4}},
            "distinct_attempts": {"goal": {"prompts": 12, "mean_share_distinct": 0.95}, "rungs": {"prompts": 6, "mean_share_distinct": 0.9}},
            "opens_with_a_have": {"goal": {"attempts": 1116, "with_a_have": 400 if name == "with" else 300, "share": round((400 if name == "with" else 300) / 1116, 5)},
                                  "rungs": {"attempts": 48, "with_a_have": 12, "share": 0.25}},
            "episodes_of_8": {problem_id: [True] * resolved + [False] * (11 - resolved) for problem_id, resolved in zip(GOAL_IDS, EPISODES[name])}}


def _losses(orders, trained=None):
    return {arm: {"row_losses": [1.5] * (len(rows) // 2) + [0.9] * (len(rows) - len(rows) // 2), "rows_trained": (trained or _trained(orders))[arm]}
            for arm, rows in orders.items()}


def _report(models=("without", "with", "loop"), prepare=PREPARE, losses=None, ceiling=None, **changed):
    orders = _orders()
    given = {name: changed.get(name) or _model(name, rows=PREPARE["trainings"].get(name, {}).get("rows")) for name in models}
    return build_l3d1_report(prepare, TRAINS, losses or _losses(orders), orders, GROUPS, LENGTHS, changed.get("base") or _model("base"), given, {**SETTINGS, "minimum_h0": 10},
                             EVALUATION, ceiling)


def test_the_primary_is_with_minus_without_on_the_problems_of_four_lines_or_more_and_the_noise_floor_stands_beside_it():
    report = _report()
    primary, floor = report["primary"], report["noise_floor"]
    # `with` 23 successes and `without` 3 on the six problems of 4 lines or more, in 558 attempts each; by problem (5, 4, 4, 3, 2, 2) of 93 more.
    assert (primary["problems"], primary["successes"], primary["successes_of_the_base"], primary["attempts_each"]) == (6, 23, 3, 558)
    assert primary["mean"] == pytest.approx(20 / 6 / 93, abs=1e-5) and primary["low"] > 0 and (primary["per_1000"], primary["per_1000_of_the_base"]) == (41.22, 5.38)
    # The noise floor: `without` 3 against the stored three-round model's 4 there: what two trainings on the same data differ by.
    assert (floor["successes"], floor["successes_of_the_base"]) == (3, 4) and floor["mean"] == pytest.approx(-1 / 6 / 93, abs=1e-5) and floor["low"] < 0 == floor["high"]
    assert floor["band"] == round(abs(floor["low"]), 5) and report["branch"]["name"] == TEACH and report["branch"]["noise_band"] == floor["band"]
    by_length = report["secondary"]["by_length_group"]
    assert by_length["problems"] == {"1": 3, "2-3": 3, "4-7": 3, "8+": 3, "4_or_more": 6, "all": 12}
    assert list(by_length["pairs"]) == ["with_minus_without", "noise_floor", "with_minus_base", "without_minus_base", "loop_minus_base"]
    assert by_length["pairs"]["with_minus_without"]["1"]["mean"] == 0.0 and by_length["pairs"]["with_minus_without"]["pair"] == "`with` minus `without`"
    assert by_length["pairs"]["noise_floor"]["pair"] == "`without` minus the stored three-round model at t = 1/10"
    assert by_length["pairs"]["without_minus_base"]["all"]["successes"] == 45 and by_length["pairs"]["without_minus_base"]["all"]["successes_of_the_base"] == 24
    rungs = report["secondary"]["the_three_rungs"]["pairs"]["with_minus_without"]
    assert rungs["in"]["mean"] == pytest.approx(1 / 16) and rungs["below"]["mean"] == 0.0 and rungs["above"]["mean"] == pytest.approx(1 / 16)
    solved = report["secondary"]["goal_problems_solved_with_against_without"]
    assert (solved["4_or_more"]["resolved_after"], solved["4_or_more"]["resolved_before"], solved["4_or_more"]["gained"], solved["4_or_more"]["lost"]) == (6, 2, 4, 0)
    assert report["ok"] is True and report["inconclusive"] is False and report["stand_in_engine"] is False and report["the_ceilings_doses"] is None


def test_goal_problems_are_counted_solved_at_least_once_and_reliably_by_attempts_alone_for_every_model():
    report = _report()
    alone = report["secondary"]["goal_problems_solved_by_attempts_alone"]
    assert set(alone) == {"what", "base", "without", "with", "loop"} and "No Lean here" in alone["what"]
    # Of 11 episodes a problem, half is 5.5: six resolved episodes are reliable, five are not.
    assert {name: (alone[name]["all"]["solved_at_least_once"], alone[name]["all"]["reliably"]) for name in ("base", "without", "with", "loop")} == {
        "base": (8, 0), "without": (8, 3), "with": (12, 4), "loop": (9, 3)}
    assert {name: (alone[name]["4_or_more"]["solved_at_least_once"], alone[name]["4_or_more"]["reliably"]) for name in ("base", "without", "with", "loop")} == {
        "base": (2, 0), "without": (2, 0), "with": (6, 1), "loop": (3, 0)}
    assert alone["with"]["all"]["episodes"] == 132 and alone["with"]["all"]["episodes_a_problem"] == [11] and alone["with"]["8+"]["in_a_quarter"] == 1
    assert by_attempts_alone({"m": {"episodes_of_8": {"a": [True, False], "b": []}}}, {"all": ["a", "b", "c"]})["m"]["all"]["reliably"] == 1


def test_the_report_prints_the_read_in_the_specs_order_and_every_line_says_l3d1():
    report = _report()
    lines = report["lines"]
    assert all(line.startswith("l3d1: ") for line in lines) and [line.split(": ", 1)[1].split(",")[0].split(".")[0].split(":")[0] for line in lines] == [
        "L3D STEP 1", "PRIMARY", "THE NOISE FLOOR", *["SECONDARY"] * 5, *["SECONDARY"] * 2, *["SECONDARY"] * 2, "SECONDARY", *["SECONDARY"] * 2, "SECONDARY",
        "CHECK 1", "CHECK 2", "CHECK 3", "CHECK 4", "BRANCH"]
    assert "`without` on the 40 training examples the three rounds stored, in a seeded order" in lines[0] and "H0's 10 assembled proofs at seeded places among them (50 rows)" in lines[0]
    assert "PRIMARY. The goal problems whose shortest published proof is 4 lines or more (6 problems), successes per attempt over 93 one-shot attempts a problem, `with` minus `without`" in lines[1]
    assert "41.22 against 5.38 per 1,000 (23 successes against 3 in 558 attempts each)" in lines[1]
    assert "THE NOISE FLOOR, the same quantity, `without` minus the stored three-round model at t = 1/10" in lines[2] and "(3 successes against 4 in 558 attempts each)" in lines[2]
    assert "either way, and no primary whose point is within that is read as an effect" in lines[2]
    text = "\n".join(lines)
    assert "BY ATTEMPTS ALONE, all of G (12): the base 8 / 0; `without` 8 / 3; `with` 12 / 4; the stored three-round model at t = 1/10 9 / 3. With assembly: not in this report" in text
    assert "BY ATTEMPTS ALONE, 4 lines or more (6): the base 2 / 0; `without` 2 / 0; `with` 6 / 1; the stored three-round model at t = 1/10 3 / 0" in text
    assert "goal problems solved at 93 attempts, `with` against `without`: all of G: 12 to 8, gained 4, lost 0" in text
    assert "the share of attempts whose first step is a `have` (on the rungs, on G): the base 0.25, 0.26882; `without` 0.25, 0.26882; `with` 0.25, 0.35842" in text
    assert "CHECK 1, H0 holds enough proofs: 10; at least 10 is asked: PASS" in text and "CHECK 2, both trainings took" in text and "`with` 0.9 against 1.5 over 5 rows: PASS" in text
    assert "CHECK 4, `with` was trained on H0: 10 of H0's 10 rows are counted once in the record of the 50 rows its training steps were made on, 0 in `without`'s 40; each record is the prepared order: yes: PASS" in text
    assert lines[-1].startswith("l3d1: BRANCH: ASSEMBLED PROOFS TEACH. the interval is clear of zero") and "L3d Step 1 (do assembled proofs teach?) seed 0: ASSEMBLED PROOFS TEACH." in report["headline"]
    assert report["models"] == {"base": report["models"]["base"], "without": "`without`", "with": "`with`", "loop": "the stored three-round model at t = 1/10"}
    assert "anything with assembly" in report["not_measured"][1] and report["training"]["h0_rows_in_with"] == 10


def test_a_primary_within_what_two_trainings_differ_by_is_not_read_as_an_effect_and_an_interval_through_zero_takes_the_ceilings_note():
    # `with` one success more than `without` on each problem of 4 lines or more; the stored three-round model far from `without` there.
    small = _model("with", rows=50, on_g=([4, 4, 4, 2, 2, 2, 2, 1, 1, 1, 1, 1], ON_G["without"][1]))
    far = _model("loop", on_g=([4, 4, 4, 2, 2, 2, 0, 0, 0, 3, 3, 3], ON_G["without"][1]))
    report = _report(**{"with": small, "loop": far})
    assert report["primary"]["mean"] == pytest.approx(1 / 93, abs=1e-5) and report["primary"]["low"] > 0 and report["noise_floor"]["band"] >= 0.01075
    assert report["branch"]["name"] == WITHIN_THE_NOISE and any("BRANCH: WITHIN WHAT TWO TRAININGS ON THE SAME DATA DIFFER BY" in line for line in report["lines"])
    # `with` as `without` but for one problem up and one down: the interval holds zero. The ceiling's doses say whether its gain could have been seen.
    mixed = _model("with", rows=50, on_g=([4, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0, 0], [5, 5, 5, 3, 3, 3, 1, 0, 0, 0, 0, 0]))
    through = _report(**{"with": mixed}, ceiling={"small": {"rows": 2000, "mean": 0.002}, "full": {"rows": 8000, "mean": 0.004}})
    assert through["primary"]["low"] < 0 < through["primary"]["high"] and through["branch"]["name"] == UNDETECTABLE          # ten proofs: far too few to see such a gain
    assert through["the_ceilings_doses"]["small"]["rows"] == 2000 and through["branch"]["could_have_been_seen"]["doses"]["small"]["expected_here"] == 0.00001
    assert any("beside it, the ceiling's two doses" in line and "after 2,000 proofs +0.00200, so +0.000010 expected here (could not have been seen)" in line for line in through["lines"])
    assert _report(**{"with": mixed})["branch"]["name"] == NOT_SHOWN


def test_a_failed_check_makes_the_report_inconclusive_and_it_says_nothing_else():
    orders = _orders()
    trained = _trained(orders)
    trained["with"] = [key for key in trained["with"] if not key.endswith("#h0")]      # the training loop's record holds no row of H0
    report = _report(losses=_losses(orders, trained))
    assert report["inconclusive"] is True and report["branch"]["name"] == INCONCLUSIVE and report["branch"]["failed_checks"] == ["with_was_trained_on_h0"]
    assert report["primary"] is None and report["noise_floor"] is None and report["secondary"] is None and report["measured_and_not_read"]["primary"]["successes"] == 23
    assert [line.split(": ", 1)[1].split(",")[0].split(".")[0] for line in report["lines"]] == ["L3D STEP 1", "CHECK 1", "CHECK 2", "CHECK 3", "CHECK 4", "INCONCLUSIVE"]
    assert "0 of H0's 10 rows are counted once" in report["lines"][4] and report["lines"][4].endswith("NO: FAIL") and "PRIMARY" not in report["headline"]
    flat = _report(losses={arm: {**entry, "row_losses": [1.0] * len(entry["row_losses"])} for arm, entry in _losses(orders).items()})
    assert flat["branch"]["failed_checks"] == ["both_trainings_took"] and flat["lines"][2].endswith("FAIL")
    capped = _report(without=_model("without", rows=40, rungs=None) | {"rungs": [_rung(problem_id, 0, capped=8) for problem_id in RUNG_IDS]})
    assert capped["branch"]["failed_checks"] == ["both_models_still_write_proofs"] and "`without` 1.0 of 48 (48 at the token cap, 0 without a verdict from Lean)" in capped["lines"][3]


def test_a_smoke_run_reads_no_noise_floor_and_a_set_lean_did_not_answer_is_not_to_be_read():
    prepare = {**PREPARE, "stored_runs": None, "loop_arm": None, "loop_target_rate": None, "attempts_a_goal_problem": 32,
               "goal_samplings": [{"name": "reach", "episodes": 32, "sampling_seed": 1001}]}
    first = lambda name: _model(name, rows=PREPARE["trainings"].get(name, {}).get("rows")) | {"goal": [_model(name)["goal"][0]]}      # noqa: E731
    smoke = build_l3d1_report(prepare, TRAINS, _losses(_orders()), _orders(), GROUPS, LENGTHS, first("base"), {arm: first(arm) for arm in ARMS},
                              {**SETTINGS, "minimum_h0": 10}, EVALUATION)
    assert smoke["noise_floor"] is None and smoke["branch"]["name"] == NOT_READ and "loop" not in smoke["models"] and smoke["primary"]["attempts_each"] == 6 * 32
    assert any("L2's runs were not read (a smoke run)" in line for line in smoke["lines"]) and any("THE NOISE FLOOR: no stored three-round model was read" in line for line in smoke["lines"])
    assert list(smoke["secondary"]["by_length_group"]["pairs"]) == ["with_minus_without", "with_minus_base", "without_minus_base"]
    unanswered = _model("with", rows=50)
    unanswered["goal"][1][0]["attempts_without_an_answer"] = 20                 # of the 61 attempts on one problem: over 2% of that set's 732
    report = _report(**{"with": unanswered})
    assert report["ok"] is False and report["not_to_be_read"] == ["goal_with_2"] and "NOT TO BE READ: too many attempts without a verdict from Lean in goal_with_2" in report["lines"][-1]
    assert report["attempts"]["goal_with_2"]["without_an_answer"] == 20 and "NOT TO BE READ" in report["headline"]
