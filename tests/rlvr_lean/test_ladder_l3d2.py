"""L3d Step 2: the four checks, the note, the branch, the table by round and the report on hand-made rows with a known
answer. Spec: docs/spec/ladder-loop.spec.md, "L3d: train on what the episode reaches", "Step 2, made exact before
it is built" ("Primary", "Secondary", "Branches", "Can this run see a win"). Pure: no model, no Lean. The arm is
`test_ladder_assembly.py`; the stage end to end is `test_ladder_l3d2_stage.py`."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l3d1 import EVALUATION, GOAL_IDS, GROUPS, LENGTHS, _model, _rung  # noqa: E402 - Step 1's hand-made models: `with`, `without`, the stored model, the base

from rlvr_lean.domain.ladder_round.l3d2 import (  # noqa: E402
    BRANCHES,
    COST,
    INCONCLUSIVE,
    NOT_READ,
    NOT_SHOWN,
    TEACH,
    UNDETECTABLE,
    as_step_1_reads,
    can_this_run_see_a_win,
    l3d2_branch,
    round_table,
    the_note,
)
from rlvr_lean.reporting.ladder_l3d2 import build_l3d2_report  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
SETTINGS = CONFIG["ladder_loop"]["l3d"]["step_2"]
GAIN = SETTINGS["gain_of_a_published_proof"]
# `with`'s 40 rows in the order trained: 28 one-shot rows, 8 assembled in the rounds, 4 of H0. `without`: the 28, in that order.
ORIGINS = ["attempt", "assembled", "attempt", "attempt", "h0", "attempt", "attempt", "assembled", "attempt", "attempt"] * 4
WITH_ROWS = [{"id": f"row{index}", "origin": origin} for index, origin in enumerate(ORIGINS)]
ORDERS = {"with": WITH_ROWS, "without": [row for row in WITH_ROWS if row["origin"] == "attempt"]}
FALLING = lambda rows: [1.5] * (len(rows) // 2) + [0.9] * (len(rows) - len(rows) // 2)      # noqa: E731
RUNGS_FINE = {arm: [_rung(f"r{index}", 4) for index in range(10)] for arm in ("without", "with")}


def _trained(orders=ORDERS):
    return {arm: [row["id"] for row in rows] for arm, rows in orders.items()}


def _losses(trained=None):
    return {arm: {"row_losses": FALLING(rows), "rows_trained": (trained or _trained())[arm]} for arm, rows in ORDERS.items()}


def _change(mean, low, high):
    return {"problems": 230, "mean": mean, "low": low, "high": high}


def test_the_settings_are_the_specs():
    assert SETTINGS == {"arm": "t010_assembly", "loop_arm": "t010", "minimum_assembled": 150, "loss_share_of_rows": 0.1, "maximum_share_without_an_answer": 0.05,
                        "gain_of_a_published_proof": 0.000013}
    assert CONFIG["ladder_loop"]["l2_assembly_arms"][SETTINGS["arm"]] == {"target_rate": 0.10, "rounds": 6}
    assert CONFIG["ladder_loop"]["l2_arms"][SETTINGS["loop_arm"]]["target_rate"] == CONFIG["ladder_loop"]["l2_assembly_arms"][SETTINGS["arm"]]["target_rate"]


# ---------------------------------------------------------------------------------------------- the checks
def test_the_four_checks_pass_on_a_twin_that_is_with_less_its_assembled_rows_and_each_can_fail():
    fine = can_this_run_see_a_win(ORDERS, _trained(), {arm: FALLING(rows) for arm, rows in ORDERS.items()}, RUNGS_FINE, SETTINGS, 12)
    assert list(fine) == ["with_was_trained_on_enough_assembled_proofs", "both_trainings_took", "both_models_still_write_proofs",
                          "every_assembled_row_is_in_withs_record_once_and_none_in_withouts"] and all(check["passes"] for check in fine.values())
    enough = fine["with_was_trained_on_enough_assembled_proofs"]
    assert (enough["assembled_proofs"], enough["from_the_rounds"], enough["from_h0"], enough["minimum"]) == (12, 8, 4, 12)      # exactly as many is enough
    assert can_this_run_see_a_win(ORDERS, _trained(), {arm: FALLING(rows) for arm, rows in ORDERS.items()}, RUNGS_FINE, SETTINGS, 13)[
        "with_was_trained_on_enough_assembled_proofs"]["passes"] is False
    record = fine["every_assembled_row_is_in_withs_record_once_and_none_in_withouts"]
    assert (record["assembled_rows"], record["counted_once_in_with"], record["counted_in_without"], record["rows_trained"]) == (12, 12, 0, {"without": 28, "with": 40})
    assert as_step_1_reads(ORDERS)["with"][:5] == [{"id": "row0", "source": "rounds"}, {"id": "row1", "source": "h0"}, {"id": "row2", "source": "rounds"},
                                                   {"id": "row3", "source": "rounds"}, {"id": "row4", "source": "h0"}]

    def failed(trained=None, losses=None, rungs=RUNGS_FINE, minimum=12):
        checks = can_this_run_see_a_win(ORDERS, trained or _trained(), losses or {arm: FALLING(rows) for arm, rows in ORDERS.items()}, rungs, SETTINGS, minimum)
        return [name for name, check in checks.items() if not check["passes"]], checks

    # An assembled row in the twin's record: it is no longer `with`'s alone, and the twin was not trained as prepared.
    leaked, checks = failed({**_trained(), "without": [*_trained()["without"], "row1"]})
    assert leaked == ["with_was_trained_on_enough_assembled_proofs", "every_assembled_row_is_in_withs_record_once_and_none_in_withouts"]
    assert checks["with_was_trained_on_enough_assembled_proofs"]["assembled_proofs"] == 11
    # An assembled row missing from `with`'s record; the twin's rows in another order; a training that did not take; a model that stopped writing proofs.
    assert failed({**_trained(), "with": [key for key in _trained()["with"] if key != "row4"]})[0] == [
        "with_was_trained_on_enough_assembled_proofs", "every_assembled_row_is_in_withs_record_once_and_none_in_withouts"]
    assert failed({**_trained(), "without": list(reversed(_trained()["without"]))})[0] == ["every_assembled_row_is_in_withs_record_once_and_none_in_withouts"]
    assert failed(losses={"with": FALLING(WITH_ROWS), "without": [1.0] * 28})[0] == ["both_trainings_took"]
    assert failed(rungs={**RUNGS_FINE, "with": [_rung(f"r{index}", 0, capped=8) for index in range(10)]})[0] == ["both_models_still_write_proofs"]


# ---------------------------------------------------------------------------------------------- the branch
PASSING = {"a": {"passes": True}}


def test_the_note_is_the_ceilings_gain_for_each_proof_times_the_assembled_proofs_against_half_the_interval():
    assert GAIN == 0.000013
    note = the_note(_change(0.0002, -0.0010, 0.0014), 300, GAIN)
    assert (note["expected_here"], note["resolves"], note["seen"], note["assembled_proofs"]) == (0.0039, 0.0012, True, 300)
    assert the_note(_change(0.0002, -0.0010, 0.0014), 92, GAIN)["seen"] is False and the_note(_change(0.0002, -0.0010, 0.0014), 92, GAIN)["expected_here"] == 0.001196
    assert the_note(_change(0.0, -0.0013, 0.0013), 100, GAIN)["seen"] is True                 # 0.0013 expected, 0.0013 resolved: it reaches it


def test_the_branch_is_the_specs():
    teach = l3d2_branch(PASSING, _change(0.0030, 0.0010, 0.0050), 300, GAIN)
    assert teach["name"] == TEACH and "assembled proofs teach, at the rate the loop makes them. Two more seeds, and the loop with assembly is the loop from here" in teach["reason"]
    assert l3d2_branch(PASSING, _change(0.0004, 0.0001, 0.0007), 300, GAIN)["name"] == TEACH                    # however small: clear of zero and above
    cost = l3d2_branch(PASSING, _change(-0.0030, -0.0050, -0.0010), 300, GAIN)
    assert cost["name"] == COST and "the assembled proofs cost one-shot attempts" in cost["reason"]
    shown = l3d2_branch(PASSING, _change(0.0002, -0.0010, 0.0014), 300, GAIN)                                   # 300 proofs at the ceiling's rate: +0.0039, and 0.0012 is resolved
    assert shown["name"] == NOT_SHOWN and "could have been seen" in shown["reason"] and "Assembled proofs teach less than published ones, each" in shown["reason"]
    assert "300 assembled proofs at that would give +0.003900, and this run resolves 0.001200" in shown["reason"] and shown["could_have_been_seen"]["seen"] is True
    unseen = l3d2_branch(PASSING, _change(0.0002, -0.0010, 0.0014), 92, GAIN)                                   # 92 proofs: +0.0012 at most, under what is resolved
    assert unseen["name"] == UNDETECTABLE and "could not have been seen" in unseen["reason"] and "undetectable at this size" in unseen["reason"]
    assert l3d2_branch(PASSING, _change(0.001, 0.0, 0.002), 300, GAIN)["name"] == NOT_SHOWN                      # an interval that ends at zero holds it
    assert l3d2_branch(PASSING, {"problems": 0, "mean": None, "low": None, "high": None}, 300, GAIN)["name"] == NOT_READ
    failed = l3d2_branch({"a": {"passes": False}, "b": {"passes": True}}, _change(0.0030, 0.0010, 0.0050), 300, GAIN)
    assert failed["name"] == INCONCLUSIVE and failed["failed_checks"] == ["a"] and "nothing is said about what assembled proofs teach" in failed["reason"]
    assert {TEACH, NOT_SHOWN, UNDETECTABLE, COST, INCONCLUSIVE, NOT_READ} == set(BRANCHES)


# ------------------------------------------------------------------------------------------------ by round
def _picked(name, resolved, assembled=False):
    return {"problem_id": name, "side": "true", "episodes": 8, "resolved": resolved, "resolved_by_assembly": assembled}


def _trained_on(name, origin, side="statement"):
    return {"problem_id": name, "origin": origin, "side": side}


ROUNDS = {1: {"summary": {"assembly": {"lean_checks": 40}},
              "results": [_picked("a", 2), _picked("b", 0, True), _picked("c", 0), _picked("d", 0), _picked("e", 8), _picked("f", 0)],
              "examples": [_trained_on("a", "attempt"), _trained_on("e", "attempt", "negation"), _trained_on("b", "assembled"), _trained_on("h1", "h0"), _trained_on("h2", "h0")]},
          2: {"summary": {"assembly": {"lean_checks": 55}},
              "results": [_picked("g", 1), _picked("h", 0, True), _picked("i", 0, True), _picked("j", 0)],
              "examples": [_trained_on("g", "attempt"), _trained_on("h", "assembled"), _trained_on("i", "assembled", "negation"), _trained_on("h1", "h0")]}}


def test_the_table_by_round_says_what_the_attempts_resolved_and_what_only_assembly_did():
    table = round_table({number: {"results": entry["results"], "examples": entry["examples"], "assembly": entry["summary"]["assembly"]}
                         for number, entry in ROUNDS.items()}, 0.10)
    first, second = table
    assert (first["round"], first["picks"], first["resolved_by_an_attempt"], first["left_unresolved_by_the_attempts"], first["only_assembly_resolved"]) == (1, 6, 2, 4, 1)
    assert first["share_of_the_unresolved"] == 0.25 and first["mean_pass_rate"] == round((2 / 8 + 1) / 6, 5)        # the solver's own rate: no assembled problem in it
    assert first["training_rows"] == {"attempt": 2, "assembled": 1, "h0": 2} and first["share_of_refutations"] == round(1 / 3, 5) and first["assembly_lean_checks"] == 40
    assert (second["only_assembly_resolved"], second["share_of_the_unresolved"], second["mean_pass_rate"]) == (2, round(2 / 3, 5), round(1 / 8 / 4, 5))
    # The mean reward is the challenger's: k = 1 for a problem only assembly resolved (at t = 1/10 and 8 solvers, k = 1 is in the band).
    assert second["mean_reward"] > first["mean_reward"] > 0 and round_table({1: {"results": [_picked("x", 0)], "examples": []}}, 0.10)[0]["mean_reward"] == 0.0
    assert round_table({1: {"results": [_picked("x", 3)], "examples": []}}, 0.10)[0]["share_of_the_unresolved"] is None


# ---------------------------------------------------------------------------------- the report, hand-made rows
PREPARE = {"seed": 0, "stage": "l3d2", "arm": "t010_assembly", "rounds": [1, 2], "stand_in_engine": False, "attempts_a_goal_problem": 93, "rung_episodes": 8,
           "stored_runs": {"base": "ladder_l2_seed0", "loop": "ladder_l2_t010_seed0"}, "loop_arm": "t010", "loop_target_rate": 0.1, "minimum_assembled": 12,
           "sampling_seeds": {"rungs": 1002, "reach": 1001, "more": 1020}, "stored_measurements": {}, "contradicted_side_setting": "audit",
           "goal_samplings": [{"name": "reach", "episodes": 32, "sampling_seed": 1001}, {"name": "more", "episodes": 61, "sampling_seed": 1020}]}
ARM_PREPARE = {"arm": "t010_assembly", "target_rate": 0.1, "rounds": [1, 2], "problems_a_round": 6, "batches": 2, "solvers": 8, "h0_file": "harvest_h0.jsonl",
               "h0_file_sha256": "0" * 64, "h0_rows": 4, "sampling_seeds": {"round_1": 1011, "round_2": 1012}}
TRAINS = {"with": {"rows": 40, "stand_in_engine": False}, "without": {"rows": 28, "stand_in_engine": False}}


def _report(models=("without", "with", "loop"), prepare=PREPARE, losses=None, **changed):
    given = {name: changed.get(name) or _model(name) for name in models}
    return build_l3d2_report(prepare, ARM_PREPARE, TRAINS, losses or _losses(), ORDERS, ROUNDS, GROUPS, LENGTHS, _model("base"), given, SETTINGS, 0.10, EVALUATION)


def test_the_report_reads_the_checks_first_then_the_primary_the_branch_and_the_secondary_reads():
    report = _report()
    primary = report["primary"]
    assert (primary["problems"], primary["successes"], primary["successes_of_the_base"], primary["attempts_each"]) == (6, 23, 3, 558) and primary["low"] > 0
    assert report["branch"]["name"] == TEACH and report["ok"] is True and report["inconclusive"] is False and (report["arm"], report["rounds"]) == ("t010_assembly", [1, 2])
    lines = report["lines"]
    assert all(line.startswith("l3d2: ") for line in lines) and [line.split(": ", 1)[1].split(",")[0].split(".")[0].split(":")[0] for line in lines] == [
        "L3D STEP 2", "CHECK 1", "CHECK 2", "CHECK 3", "CHECK 4", "PRIMARY", "BRANCH", *["SECONDARY"] * 4, "SECONDARY", *["SECONDARY"] * 2, "SECONDARY", "SECONDARY",
        "SECONDARY", "SECONDARY"]
    assert "`with` is M(2), trained on 40 rows; `without` is its twin, trained from the base on the 28 one-shot rows among them, in the same order with the others left out" in lines[0]
    assert "CHECK 1, `with` was trained on enough assembled proofs that `without` was not: 12 (8 of the rounds, 4 of H0); at least 12 is asked: PASS" in lines[1]
    assert "CHECK 4, every assembled row stands once in the record of what `with` was trained on and none in `without`'s: 12 of 12 once in `with`'s 40 rows, 0 in `without`'s 28" in lines[4]
    assert lines[4].endswith("each record is the prepared order: yes: PASS") and "41.22 against 5.38 per 1,000 (23 successes against 3 in 558 attempts each)" in lines[5]
    assert lines[6].startswith("l3d2: BRANCH: ASSEMBLED PROOFS TEACH. the interval is clear of zero and above")
    text = "\n".join(lines)
    assert "BY ATTEMPTS ALONE, all of G (12): the base 8 / 0; `without` 8 / 3; `with` 12 / 4; the stored three-round model at t = 1/10 9 / 3. With assembly: not in this report" in text
    assert "by round (picks; resolved by an attempt; only assembly resolved, and its share of what the attempts left unresolved" in text
    assert "round 1: 6; 2; 1, 0.25; 0.20833; 0.33333; round 2: 4; 1; 2, 0.66667; 0.03125; 0.33333" in text
    by_length = report["secondary"]["by_length_group"]
    assert list(by_length["pairs"]) == ["with_minus_without", "with_minus_base", "without_minus_base", "loop_minus_base"] and by_length["problems"]["4_or_more"] == 6
    assert report["secondary"]["by_round"]["rows"][1]["only_assembly_resolved"] == 2 and report["the_arm"]["h0_rows"] == 4
    assert report["models"]["with"] == "`with`: M(2), the arm's last model" and "the twin of M(2)" in report["models"]["without"]
    assert "L3d Step 2 (the loop with assembly in the round, `with` against its twin) seed 0: ASSEMBLED PROOFS TEACH." in report["headline"]
    # The primary is against the TWIN, not the base: a twin with one more success there moves it, and the base's rows do not.
    other = _report(without=_model("without", on_g=([4, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0, 0], [5, 5, 5, 3, 3, 3, 1, 1, 0, 0, 0, 0])))
    assert (other["primary"]["successes"], other["primary"]["successes_of_the_base"]) == (23, 4)
    assert other["secondary"]["by_length_group"]["pairs"]["with_minus_base"]["4_or_more"]["successes_of_the_base"] == 3


def test_an_interval_that_holds_zero_takes_the_note_and_a_failed_check_says_inconclusive_and_nothing_else():
    mixed = _model("with", on_g=([4, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0, 0], [5, 5, 5, 3, 3, 3, 1, 0, 0, 0, 0, 0]))       # one problem up, one down
    through = _report(**{"with": mixed})
    assert through["primary"]["low"] < 0 < through["primary"]["high"] and through["branch"]["name"] == UNDETECTABLE     # 12 proofs at +0.000013: far under what is resolved
    note = through["branch"]["could_have_been_seen"]
    assert (note["assembled_proofs"], note["expected_here"], note["seen"]) == (12, 0.000156, False) and note["resolves"] == round((through["primary"]["high"] - through["primary"]["low"]) / 2, 6)
    assert any(line.startswith("l3d2: BRANCH: UNDETECTABLE AT THIS SIZE.") and "12 assembled proofs at that would give +0.000156" in line for line in through["lines"])
    # A twin whose record holds an assembled row: INCONCLUSIVE, the four checks, and nothing else.
    trained = _trained()
    trained["without"] = [*trained["without"], "row1"]
    broken = _report(losses=_losses(trained))
    assert broken["inconclusive"] is True and broken["branch"]["name"] == INCONCLUSIVE and broken["primary"] is None and broken["secondary"] is None
    assert [line.split(": ", 1)[1].split(",")[0].split(".")[0] for line in broken["lines"]] == ["L3D STEP 2", "CHECK 1", "CHECK 2", "CHECK 3", "CHECK 4", "INCONCLUSIVE"]
    assert broken["lines"][1].endswith("FAIL") and broken["lines"][4].endswith("NO: FAIL") and broken["measured_and_not_read"]["primary"]["successes"] == 23
    assert "PRIMARY" not in broken["headline"] and "Checks: assembled proofs FAIL (11 of at least 12)" in broken["headline"]
    # Too few assembled proofs is the same verdict: with 12 where 150 are asked the run cannot see a win.
    few = _report(prepare={**PREPARE, "minimum_assembled": 150})
    assert few["branch"]["failed_checks"] == ["with_was_trained_on_enough_assembled_proofs"] and "12 (8 of the rounds, 4 of H0); at least 150 is asked: FAIL" in few["lines"][1]


def test_a_smoke_run_has_no_stored_model_beside_and_a_set_lean_did_not_answer_is_not_to_be_read():
    prepare = {**PREPARE, "stored_runs": None, "loop_arm": None, "loop_target_rate": None, "attempts_a_goal_problem": 32,
               "goal_samplings": [{"name": "reach", "episodes": 32, "sampling_seed": 1001}]}
    first = lambda name: _model(name) | {"goal": [_model(name)["goal"][0]]}      # noqa: E731
    smoke = build_l3d2_report(prepare, ARM_PREPARE, TRAINS, _losses(), ORDERS, ROUNDS, GROUPS, LENGTHS, first("base"), {arm: first(arm) for arm in ("without", "with")},
                              SETTINGS, 0.10, EVALUATION)
    assert "loop" not in smoke["models"] and smoke["primary"]["attempts_each"] == 6 * 32 and any("L2's stored runs were not read (a smoke run)" in line for line in smoke["lines"])
    assert list(smoke["secondary"]["by_length_group"]["pairs"]) == ["with_minus_without", "with_minus_base", "without_minus_base"]
    unanswered = _model("with")
    unanswered["goal"][1][0]["attempts_without_an_answer"] = 20                 # of the 61 attempts on one problem: over 2% of that set's 732
    report = _report(**{"with": unanswered})
    assert report["ok"] is False and report["not_to_be_read"] == ["goal_with_2"] and "NOT TO BE READ" in report["lines"][-1] and "NOT TO BE READ" in report["headline"]
    assert len(GOAL_IDS) == 12
