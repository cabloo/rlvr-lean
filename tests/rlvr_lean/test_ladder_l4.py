"""L4, the loop from a model pretrained on published proofs: the rules of its read on hand-made rows with a known answer:
the goal set drawn again, the checks, the note, the branch. Spec: docs/spec/ladder-loop.spec.md, "L4: the loop from
a model pretrained on published proofs" ("The read, fixed before any run"). Pure: no model, no Lean. The halves and the
pretraining file are `test_ladder_l4_file.py`; the two reports are `test_ladder_l4_report.py`; the two stages end to end
are `test_ladder_l4_stage.py`."""

import math
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l3d1 import GOAL_IDS, RUNG_IDS, _model, _rung  # noqa: E402 - Step 1's hand-made world: 12 goal problems, 3 a length group

from rlvr_lean.domain.ladder_round.l4 import (  # noqa: E402
    ADDS,
    BRANCHES,
    COSTS,
    INCONCLUSIVE,
    LOOP,
    MAP_FIELDS,
    MAP_FILE,
    MAP_STEP,
    NOT_READ,
    NOT_SHOWN,
    PRE,
    PRETRAIN,
    START_MODEL,
    arm_checks,
    check_start_map,
    goal_set_again,
    half_of,
    l4_branch,
    loss_did_not_rise,
    map_rows,
    map_summary,
    pretraining_checks,
    the_adapter_differs,
    the_note,
    the_training_ran,
)

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
SETTINGS = CONFIG["ladder_loop"]["l4"]
# Successes by problem in (the 32 attempts, the 61 more), in GOAL_IDS' order: three problems of 1 line, of 2-3, of 4-7, of 8 or more.
ON_G = {"pre": ([6, 6, 6, 4, 4, 4, 2, 1, 0, 0, 0, 0], [9, 9, 9, 6, 6, 6, 3, 2, 1, 0, 0, 0]),
        "with": ([6, 6, 6, 4, 4, 4, 3, 2, 1, 1, 0, 0], [9, 9, 9, 6, 6, 6, 4, 3, 3, 2, 1, 0]),
        "without": ([6, 6, 6, 4, 4, 4, 2, 1, 0, 0, 0, 0], [9, 9, 9, 6, 6, 6, 3, 2, 2, 1, 0, 0])}
ON_THE_RUNGS = {"pre": [2, 3, 4, 5, 7, 8], "with": [2, 3, 4, 6, 7, 8], "without": [2, 3, 4, 5, 7, 7]}
EPISODES = {"pre": [8, 8, 8, 6, 6, 6, 4, 3, 1, 0, 0, 0], "with": [8, 8, 8, 6, 6, 6, 5, 4, 3, 3, 1, 0], "without": [8, 8, 8, 6, 6, 6, 4, 3, 2, 1, 0, 0]}
AGAIN = ["g4c", "g8a", "g8b", "g8c"]                # the goal problems `pre` does not solve in its 32 attempts
LOOP_HALF = [f"p{index}" for index in range(200) if half_of(f"p{index}", 0) == LOOP]       # problems a round of the arm may train on
OF_THE_OTHER_HALF = next(f"p{index}" for index in range(200) if half_of(f"p{index}", 0) == PRETRAIN)


def _made(name, on_g=None, rungs=None, rows=None):
    """`pre`, `with` or `without` as a report takes a model (the shape of Step 1's `_model`)."""
    first, more = on_g or ON_G[name]
    return {**_model("with"), "rows": rows, "rungs": [_rung(problem_id, resolved) for problem_id, resolved in zip(RUNG_IDS, rungs or ON_THE_RUNGS[name])],
            "goal": [[_rung(problem_id, resolved, episodes=32) for problem_id, resolved in zip(GOAL_IDS, first)],
                     [_rung(problem_id, resolved, episodes=61) for problem_id, resolved in zip(GOAL_IDS, more)]],
            "episodes_of_8": {problem_id: [True] * resolved + [False] * (11 - resolved) for problem_id, resolved in zip(GOAL_IDS, EPISODES[name])}}


def _change(mean, low, high, problems=160):
    return {"problems": problems, "mean": mean, "low": low, "high": high}


def test_the_settings_are_the_specs():
    assert {key: SETTINGS[key] for key in ("arm", "loop_arm", "base_arm", "minimum_solved_by_pre", "minimum_goal_set_again", "loss_share_of_rows",
                                           "standard_errors_allowed", "maximum_share_without_an_answer")} == {
        "arm": "t010_assembly_pre", "loop_arm": "t010", "base_arm": "t010_assembly", "minimum_solved_by_pre": 150, "minimum_goal_set_again": 80, "loss_share_of_rows": 0.1,
        "standard_errors_allowed": 2, "maximum_share_without_an_answer": 0.05}
    arms = CONFIG["ladder_loop"]["l2_assembly_arms"]
    # The arm is the base arm with a start adapter, the `loop` half as its candidates, no H0, and the map of the model it starts from: the
    # target rate and the rounds are the base arm's.
    assert arms[SETTINGS["arm"]] == {**arms[SETTINGS["base_arm"]], "start": "pre", "candidates": "loop_half", "h0": False, "map": "start_model"}
    assert arms[SETTINGS["base_arm"]] == {"target_rate": 0.10, "rounds": 6} and list(arms) == ["t010_assembly", "t010_assembly_pre", "t010_assembly_pre_r64"]
    # L4b's arm is L4's with another start and a training rule: nothing else of the arm moves (test_ladder_l4b_stage.py).
    assert arms["t010_assembly_pre_r64"] == {**arms[SETTINGS["arm"]], "start": "pre_r64", "rule": "old"} and SETTINGS["again"] == {"arm": "t010_assembly_pre_r64", "rank_16_arm": SETTINGS["arm"]}
    assert (PRE, START_MODEL, MAP_FILE, MAP_STEP) == ("pre", "start_model", "l4_map_pre.jsonl", "ladder_l4_pretrain_map")
    assert CONFIG["ladder_loop"]["base_map"] == {"problems": 4000, "episodes": 8, "seed": 20261004, "sampling_seed": 101}       # what the stored map was made with


# ------------------------------------------------------------------------------- the goal set drawn again
def test_the_goal_set_again_is_the_goal_problems_pre_does_not_solve_in_its_first_sampling():
    first, more = _made("pre")["goal"]
    assert goal_set_again(first, GOAL_IDS) == AGAIN                                                 # in the goal set's order
    assert goal_set_again(first, list(reversed(GOAL_IDS))) == list(reversed(AGAIN))
    assert goal_set_again(more, GOAL_IDS) == ["g8a", "g8b", "g8c"]                                  # another sampling draws another set: the FIRST is the one that chooses
    assert goal_set_again([_rung("g1a", 0, episodes=32), _rung("elsewhere", 0, episodes=32)], ["g1a", "g1b"]) == ["g1a", "g1b"]     # a goal problem with no row is not solved
    assert goal_set_again([], []) == []


def test_the_two_checks_of_the_pretraining_are_read_on_pre_and_each_can_fail():
    first, more = _made("pre")["goal"]
    on_g = [{**row, "resolved": row["resolved"] + other["resolved"], "episodes": 93} for row, other in zip(first, more)]
    checks = pretraining_checks(on_g, AGAIN, GOAL_IDS, 9, 4)
    assert list(checks) == ["the_pretraining_took", "the_goal_set_again_is_large_enough"] and all(check["passes"] for check in checks.values())
    assert (checks["the_pretraining_took"]["goal_problems_solved"], checks["the_pretraining_took"]["goal_problems"], checks["the_pretraining_took"]["minimum"]) == (9, 12, 9)
    assert (checks["the_goal_set_again_is_large_enough"]["problems"], checks["the_goal_set_again_is_large_enough"]["minimum"]) == (4, 4)       # exactly as many is enough
    assert [name for name, check in pretraining_checks(on_g, AGAIN, GOAL_IDS, 10, 4).items() if not check["passes"]] == ["the_pretraining_took"]
    assert [name for name, check in pretraining_checks(on_g, AGAIN, GOAL_IDS, 9, 5).items() if not check["passes"]] == ["the_goal_set_again_is_large_enough"]
    # A problem that is not a goal problem does not count as one solved.
    assert pretraining_checks([*on_g, _rung("elsewhere", 5, episodes=93)], AGAIN, GOAL_IDS, 9, 4)["the_pretraining_took"]["goal_problems_solved"] == 9
    assert pretraining_checks(on_g, AGAIN, GOAL_IDS, 150, 80)["the_pretraining_took"]["passes"] is False       # the real minimums, on this small world


# -------------------------------------------------------------------- the map of the model an arm starts from
BASE_MAP_IDS = ["m1", "m2", "m3", "m4", "m5"]


def _episodes(problem_id, resolved, episodes=8, sides=1):
    """One problem's row as an episode step stores it (more fields than a map holds)."""
    return {"problem_id": problem_id, "kind": "lean_workbook", "side": "true", "episodes": episodes, "resolved": resolved, "resolved_by_statement": resolved,
            "resolved_by_negation": 0, "sides": sides, "attempts_capped": 0, "attempts_without_an_answer": 0, "contradicted_side": "not_generated"}


def test_a_models_map_is_the_base_maps_problems_in_its_order_with_the_stored_maps_fields():
    results = [_episodes("m3", 8), _episodes("m1", 0), _episodes("m5", 2, sides=2), _episodes("m2", 5), _episodes("m4", 0)]
    rows = map_rows(results, BASE_MAP_IDS)
    assert [row["problem_id"] for row in rows] == BASE_MAP_IDS and all(list(row) == [*MAP_FIELDS, "set"] and row["set"] == "base_map" for row in rows)
    assert rows[4] == {"problem_id": "m5", "episodes": 8, "resolved": 2, "resolved_by_statement": 2, "resolved_by_negation": 0, "sides": 2, "set": "base_map"}
    assert MAP_FIELDS == ("problem_id", "episodes", "resolved", "resolved_by_statement", "resolved_by_negation", "sides")       # what the stored map's rows hold
    in_another_order = ["m4", "m1", "m5", "m3", "m2"]                           # the base map's own order is the rows' order, not the ids' sorted
    assert [row["problem_id"] for row in map_rows(results, in_another_order)] == in_another_order
    for wrong, match in ((results[:-1], "the map's episodes are of 4 problems and the base map holds 5 .1 of its problems have no result; first: m4"),
                         ([*results, _episodes("elsewhere", 1)], "the map's episodes are of 6 problems and the base map holds 5")):
        with pytest.raises(ValueError, match=match):
            map_rows(wrong, BASE_MAP_IDS)
    summary = map_summary(rows)
    assert summary == {"problems": 5, "episodes_each": 8, "problems_by_k": {"0": 2, "1": 0, "2": 1, "3": 0, "4": 0, "5": 1, "6": 0, "7": 0, "8": 1},
                       "mean_pass_rate": round((8 + 2 + 5) / 8 / 5, 5), "resolved_at_least_once": 3}
    assert map_summary([]) == {"problems": 0, "episodes_each": 0, "problems_by_k": {"0": 0}, "mean_pass_rate": None, "resolved_at_least_once": 0}


def test_a_start_models_map_is_refused_unless_it_is_the_stored_map_of_another_model():
    rows = map_rows([_episodes(key, index) for index, key in enumerate(BASE_MAP_IDS)], BASE_MAP_IDS)
    recorded = {"sampling_seed": 101, "episodes_each": 8}
    check_start_map(rows, recorded, BASE_MAP_IDS, 8, 101)                       # the same seed, attempts and problems: taken
    check_start_map(list(reversed(rows)), recorded, BASE_MAP_IDS, 8, 101)       # whatever the order of its rows
    for given, match in (
            ((rows, {**recorded, "sampling_seed": 102}, BASE_MAP_IDS, 8, 101), "it was made with sampling seed 102 and 8 attempts a problem; the stored map's are 101 and 8"),
            ((rows, {**recorded, "episodes_each": 4}, BASE_MAP_IDS, 8, 101), "it was made with sampling seed 101 and 4 attempts a problem; the stored map's are 101 and 8"),
            ((rows, {}, BASE_MAP_IDS, 8, 101), "it was made with sampling seed None and None attempts a problem"),
            ((rows[:-1], recorded, BASE_MAP_IDS, 8, 101), "it is of 4 problems and the base map holds 5; they are not the same problems .first that is in one alone: m5"),
            (([*rows[:-1], {**rows[-1], "problem_id": "elsewhere"}], recorded, BASE_MAP_IDS, 8, 101), "it is of 5 problems and the base map holds 5; they are not the same problems"),
            (([*rows[:-1], rows[0]], recorded, BASE_MAP_IDS, 8, 101), "they are not the same problems"),
            (([*rows, rows[0]], recorded, BASE_MAP_IDS, 8, 101), "it is of 6 problems and the base map holds 5; they are not the same problems .a problem stands twice."),
            (([*rows[:-1], {**rows[-1], "episodes": 7}], recorded, BASE_MAP_IDS, 8, 101), "1 of its problems do not have 8 attempts .first: m5")):
        with pytest.raises(ValueError, match=match):
            check_start_map(*given)


# ---------------------------------------------------------------------------------------------- the checks
OF_THE_PRETRAINING = {"the_pretraining_took": {"goal_problems_solved": 9, "goal_problems": 12, "minimum": 9, "passes": True},
                      "the_goal_set_again_is_large_enough": {"problems": 4, "minimum": 4, "passes": True}}
FALLING = lambda count: [1.5] * (count // 2) + [0.9] * (count - count // 2)      # noqa: E731
TRAININGS = {"M(1)": LOOP_HALF[:20], "M(2)": LOOP_HALF[:40], "`without`": LOOP_HALF[:30]}
LOSSES = {name: FALLING(len(problems)) for name, problems in TRAININGS.items()}
CHANGED = {"tensors": 4, "tensors_not_compared": 0, "elements": 1000, "elements_changed": 1000, "largest_change": 0.012}       # a trained adapter against the start adapter's file
MEASURED = {name: {"row_losses": LOSSES[name], "against_the_start_adapter": CHANGED} for name in ("M(2)", "`without`")}       # the two trainings whose models are measured
RUNGS_FINE = {name: [_rung(f"r{index}", 4) for index in range(10)] for name in ("pre", "with", "without")}
STEADY, A_LITTLE_UP, WELL_UP = [1.0, 1.2, 0.8, 1.0], [1.1, 1.3, 0.9, 1.1], [1.3, 1.5, 1.1, 1.3]       # a window of four rows: mean 1.0, 1.1 and 1.3, the same spread


def test_a_trainings_loss_did_not_rise_when_the_last_rows_are_not_above_the_first_by_more_than_two_standard_errors():
    # THE STANDARD ERROR: each window's sample variance (n - 1) over its rows, the two added, the square root.
    variance = (0.0 + 0.04 + 0.04 + 0.0) / 3
    error = math.sqrt(variance / 4 + variance / 4)
    within = loss_did_not_rise([*STEADY, 5.0, 5.0, *A_LITTLE_UP], 4, 2)        # the rows between the two windows are not read
    assert (within["rows"], within["rows_compared"], within["first"], within["last"], within["rise"]) == (10, 4, 1.0, 1.1, 0.1)
    assert within["standard_error"] == round(error, 5) == 0.11547 and within["allowed"] == round(2 * error, 5) and within["passes"] is True      # 0.1 is within 0.231
    above = loss_did_not_rise([*STEADY, *WELL_UP], 4, 2)
    assert (above["rise"], above["passes"]) == (0.3, False)                    # 0.3 is not
    assert loss_did_not_rise([*STEADY, *WELL_UP], 4, 3)["passes"] is True      # ... and would be within three
    # A loss that falls, or stays, did not rise: no fall is asked for (a model that starts from `pre` is trained on proofs it or its like wrote).
    assert loss_did_not_rise([*WELL_UP, *STEADY], 4, 2)["passes"] is True and loss_did_not_rise([1.0] * 40, 4, 2) == {
        "rows": 40, "rows_compared": 4, "first": 1.0, "last": 1.0, "rise": 0.0, "standard_error": 0.0, "allowed": 0.0, "passes": True}
    assert loss_did_not_rise(FALLING(40), 4, 2)["passes"] is True and loss_did_not_rise(list(reversed(FALLING(40))), 4, 2)["passes"] is False
    # A window of one row has no variance: nothing is allowed. A run shorter than two windows compares its halves; one with no row does not pass.
    assert loss_did_not_rise([1.0, 1.1], 1, 2)["passes"] is False and loss_did_not_rise([1.1, 1.0], 1, 2)["passes"] is True and loss_did_not_rise([1.0, 1.0], 1, 2)["allowed"] == 0.0
    assert loss_did_not_rise([*STEADY, *A_LITTLE_UP], 30, 2)["rows_compared"] == 4 and loss_did_not_rise([], 4, 2)["passes"] is False and loss_did_not_rise([1.0], 4, 2)["passes"] is False


def test_an_adapter_differs_from_the_start_when_a_number_of_it_is_another_and_a_count_cannot_pass_by_rounding():
    assert the_adapter_differs(CHANGED) == {"compared": True, "elements": 1000, "elements_changed": 1000, "largest_change": 0.012, "passes": True}
    assert the_adapter_differs({**CHANGED, "elements_changed": 1})["passes"] is True                       # one number is enough: it is not the same adapter
    assert the_adapter_differs({**CHANGED, "elements_changed": 0, "largest_change": 0.0})["passes"] is False       # the adapter saved IS the start adapter: no training happened
    for not_recorded in (None, {}, {"elements": 1000}):                                                    # no comparison was recorded (the stand-in trains nothing): not a pass
        assert the_adapter_differs(not_recorded) == {"compared": False, "elements": None, "elements_changed": None, "passes": False}
    ran = the_training_ran(FALLING(40), CHANGED, SETTINGS)
    assert ran["passes"] is True and ran["the_loss_did_not_rise"]["rows_compared"] == 4 and ran["the_adapter_differs_from_the_start"]["elements_changed"] == 1000
    assert the_training_ran(FALLING(40), {**CHANGED, "elements_changed": 0}, SETTINGS)["passes"] is False
    assert the_training_ran(list(reversed(FALLING(40))), CHANGED, SETTINGS)["passes"] is False            # both are asked
    # The allowance is the setting's: a rise of 0.1 where two standard errors are 0.231 passes, and with none allowed it does not.
    a_little_up = [*STEADY, *([1.0] * 32), *A_LITTLE_UP]
    assert SETTINGS["standard_errors_allowed"] == 2 and the_training_ran(a_little_up, CHANGED, SETTINGS)["the_loss_did_not_rise"]["allowed"] == 0.23094
    assert the_training_ran(a_little_up, CHANGED, SETTINGS)["passes"] is True and the_training_ran(a_little_up, CHANGED, {**SETTINGS, "standard_errors_allowed": 0})["passes"] is False


def _checks(of_the_pretraining=OF_THE_PRETRAINING, measured=MEASURED, rungs=RUNGS_FINE, trained_on=TRAININGS, heldout=frozenset({"held"})):
    checks = arm_checks(of_the_pretraining, measured, rungs, trained_on, heldout, 0, SETTINGS)
    return [name for name, check in checks.items() if not check["passes"]], checks


def test_the_arms_checks_are_the_two_of_the_pretraining_and_three_of_its_own_and_each_can_fail():
    failed, checks = _checks()
    assert failed == [] and list(checks) == ["the_pretraining_took", "the_goal_set_again_is_large_enough", "the_two_measured_trainings_ran",
                                             "each_measured_model_still_writes_proofs", "no_training_row_is_of_the_pretrain_half_or_held_out"]
    ran = checks["the_two_measured_trainings_ran"]
    assert set(ran) == {"what", "M(2)", "`without`", "passes"} and "by more than 2 standard errors of their difference" in ran["what"]
    assert (ran["M(2)"]["the_loss_did_not_rise"]["rows_compared"], ran["M(2)"]["the_loss_did_not_rise"]["first"], ran["M(2)"]["the_loss_did_not_rise"]["last"]) == (4, 1.5, 0.9)
    barred = checks["no_training_row_is_of_the_pretrain_half_or_held_out"]
    assert barred["rows"] == {"M(1)": 20, "M(2)": 40, "`without`": 30} and barred["barred_problems"] == 0 and barred["first"] == []
    # The pretraining's two checks are carried over as its report read them: one that failed there fails here.
    assert _checks({**OF_THE_PRETRAINING, "the_pretraining_took": {**OF_THE_PRETRAINING["the_pretraining_took"], "passes": False}})[0] == ["the_pretraining_took"]
    # A measured training whose adapter IS the start adapter's, or was not compared with it, or whose loss rose: the check fails on that one training.
    for broken in ({**MEASURED["`without`"], "against_the_start_adapter": {**CHANGED, "elements_changed": 0}}, {"row_losses": LOSSES["`without`"]},
                   {**MEASURED["`without`"], "row_losses": list(reversed(LOSSES["`without`"]))}):
        failed, checks = _checks(measured={**MEASURED, "`without`": broken})
        assert failed == ["the_two_measured_trainings_ran"] and checks["the_two_measured_trainings_ran"]["M(2)"]["passes"] is True
    # A loss that did not FALL is not a failure: flat losses on both pass (what the first version of this check would have failed).
    assert _checks(measured={name: {**entry, "row_losses": [1.0] * 40} for name, entry in MEASURED.items()})[0] == []
    # Both are needed: with one of the two, or none, the check does not pass.
    assert _checks(measured={"M(2)": MEASURED["M(2)"]})[0] == ["the_two_measured_trainings_ran"] and _checks(measured={})[0] == ["the_two_measured_trainings_ran"]
    # One measured model that stopped writing proofs (`pre` is one of the three).
    assert _checks(rungs={**RUNGS_FINE, "pre": [_rung(f"r{index}", 0, capped=8) for index in range(10)]})[0] == ["each_measured_model_still_writes_proofs"]
    assert _checks(rungs={**RUNGS_FINE, "with": [*[_rung(f"r{index}", 4) for index in range(9)], _rung("r9", 0, capped=4)]})[0] == ["each_measured_model_still_writes_proofs"]
    # A row of the `pretrain` half in ONE training (the twin's): its published proof was pretrained on. A held-out problem in another (any of the seven).
    failed, checks = _checks(trained_on={**TRAININGS, "`without`": [*TRAININGS["`without`"], OF_THE_OTHER_HALF]})
    assert failed == ["no_training_row_is_of_the_pretrain_half_or_held_out"]
    assert checks["no_training_row_is_of_the_pretrain_half_or_held_out"]["first"] == [OF_THE_OTHER_HALF]
    failed, checks = _checks(trained_on={**TRAININGS, "M(1)": [*TRAININGS["M(1)"], LOOP_HALF[50]]}, heldout={LOOP_HALF[50]})
    assert failed == ["no_training_row_is_of_the_pretrain_half_or_held_out"] and checks["no_training_row_is_of_the_pretrain_half_or_held_out"]["barred_problems"] == 1
    # No measured model at all is not a pass.
    assert _checks(rungs={})[0] == ["each_measured_model_still_writes_proofs"]


# ---------------------------------------------------------------------------------------------- the branch
PASSING = {"a": {"passes": True}}


def test_the_note_is_the_base_arms_own_gain_against_half_the_width_of_this_interval():
    note = the_note(_change(0.0002, -0.0010, 0.0014), {"mean": 0.0015})
    assert (note["resolves"], note["base_arms_gain"], note["seen"]) == (0.0012, 0.0015, True)
    assert the_note(_change(0.0002, -0.0010, 0.0014), {"mean": 0.0011})["seen"] is False
    assert the_note(_change(0.0002, -0.0010, 0.0014), {"mean": 0.0012})["seen"] is True              # it reaches what is resolved
    assert the_note(_change(0.0002, -0.0010, 0.0014), {"mean": -0.0020})["seen"] is True             # its size, whatever its sign
    for not_read in (None, {"mean": None}):
        note = the_note(_change(0.0002, -0.0010, 0.0014), not_read)
        assert note["seen"] is None and note["base_arms_gain"] is None and note["resolves"] == 0.0012 and "were not read" in note["what"]


def test_the_branch_is_the_specs():
    adds = l4_branch(PASSING, _change(0.0030, 0.0010, 0.0050))
    assert adds["name"] == ADDS == "THE LOOP ADDS ON TOP OF PRETRAINING" and "Two more seeds of the arm (the pretraining is not repeated: the same `pre`)" in adds["reason"]
    assert l4_branch(PASSING, _change(0.0004, 0.0001, 0.0007))["name"] == ADDS                      # however small: clear of zero and above
    costs = l4_branch(PASSING, _change(-0.0030, -0.0050, -0.0010))
    assert costs["name"] == COSTS == "THE ROUNDS COST THE PRETRAINED MODEL" and "The loop is not run on top of pretraining as it stands" in costs["reason"]
    shown = l4_branch(PASSING, _change(0.0002, -0.0010, 0.0014), {"mean": 0.0015})
    assert shown["name"] == NOT_SHOWN == "NOT SHOWN ON TOP OF PRETRAINING" and shown["the_note"]["seen"] is True
    assert "This run resolves 0.00120; the base arm's own gain, +0.00150, would have been seen here" in shown["reason"]
    assert "would not have been seen here" in l4_branch(PASSING, _change(0.0002, -0.0010, 0.0014), {"mean": 0.0005})["reason"]
    assert "the base arm's stored rows were not read" in l4_branch(PASSING, _change(0.0002, -0.0010, 0.0014))["reason"]
    assert l4_branch(PASSING, _change(0.001, 0.0, 0.002))["name"] == NOT_SHOWN and l4_branch(PASSING, _change(-0.001, -0.002, 0.0))["name"] == NOT_SHOWN       # ending at zero holds it
    assert l4_branch(PASSING, {"problems": 0, "mean": None, "low": None, "high": None})["name"] == NOT_READ
    failed = l4_branch({"a": {"passes": False}, "b": {"passes": True}}, _change(0.0030, 0.0010, 0.0050))
    assert failed["name"] == INCONCLUSIVE and failed["failed_checks"] == ["a"] and "nothing is said about what the loop adds on top of pretraining" in failed["reason"]
    assert l4_branch({"a": {"passes": False}}, {"problems": 0, "mean": None, "low": None, "high": None})["name"] == INCONCLUSIVE      # a failed check comes first
    assert {ADDS, NOT_SHOWN, COSTS, INCONCLUSIVE, NOT_READ} == set(BRANCHES)
