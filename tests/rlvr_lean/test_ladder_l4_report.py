"""L4's two reports on hand-made rows with a known answer: the pretraining's (`pre` against the base, the goal set drawn
again, the two checks) and the arm's (the checks first, the primary on the goal set again over the second sampling alone,
the base arm's own gain beside it, the branch, the secondary reads). Spec: docs/spec/ladder-loop.spec.md, "L4: the
loop from a model pretrained on published proofs" ("The read, fixed before any run"). Pure: no model, no Lean. The rules
are `test_ladder_l4.py`; the two stages end to end are `test_ladder_l4_stage.py`."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l3d1 import EVALUATION, GROUPS, LENGTHS, RUNG_IDS, _model, _rung  # noqa: E402 - Step 1's hand-made world: 12 goal problems, 3 a length group
from test_ladder_l3d2 import ROUNDS  # noqa: E402 - Step 2's two hand-made rounds
from test_ladder_l4 import AGAIN, CHANGED, LOSSES, OF_THE_OTHER_HALF, OF_THE_PRETRAINING, ON_G, SETTINGS, TRAININGS, _episodes, _made  # noqa: E402 - the rules' hand-made world

from rlvr_lean.domain.ladder_round.l4 import ADDS, COSTS, INCONCLUSIVE, NOT_READ, NOT_SHOWN, map_rows  # noqa: E402
from rlvr_lean.reporting.ladder_l4 import LABEL, SAY, build_l4_report, build_pretrain_report  # noqa: E402


def test_every_line_says_what_the_model_is():
    assert (LABEL, SAY) == ("pretrained on published proofs", "l4 (pretrained on published proofs)")


# --------------------------------------------------------------------------------- the pretraining's report
SAMPLINGS = [{"name": "reach", "episodes": 32, "sampling_seed": 1001}, {"name": "more", "episodes": 61, "sampling_seed": 1020}]
PRETRAIN_PREPARE = {
    "seed": 0, "stage": "l4", "label": LABEL, "stand_in_engine": False, "attempts_a_goal_problem": 93, "rung_episodes": 8, "goal_samplings": SAMPLINGS,
    "stored_runs": {"base": "ladder_l2_seed0", "loop": "ladder_l2_t010_seed0"}, "loop_arm": "t010", "loop_target_rate": 0.1,
    "sampling_seeds": {"rungs": 1002, "reach": 1001, "more": 1020}, "stored_measurements": {}, "contradicted_side_setting": "audit",
    "pretraining_file": "pretraining.jsonl", "pretraining_file_sha256": "0" * 64, "rows": 24800, "half": "pretrain", "half_seed": 0,
    "rows_by_kind": {"lean_workbook": 8100, "stp_conjecture": 16700}, "rows_by_proof_lines": {"1": 9000, "2-3": 7000, "4-7": 5000, "8+": 3800}, "tokens": 9000000,
    "longest_example_tokens": 900, "max_sequence_tokens": 2048, "order": "the file's: no row is moved", "steps": 3100, "recipe": {"what": "the round's"},
    "ceiling": {"read": True, "run": "ladder_ceiling_seed0", "models": {"ceiling_small": {"rows": 2000}, "ceiling_full": {"rows": 8000}}}}
PRETRAIN_TRAIN = {"stage": "l4", "model": "pre", "rows": 24800, "steps": 3100, "stand_in_engine": False, "mean_loss_over_the_first_rows": 0.61, "mean_loss_over_the_last_rows": 0.34,
                  "rows_compared": 2480, "the_training_took": True}
# The base map's problems (ten here), attempted 8 times each by the base (the stored map) and again by `pre` (its own map).
MAP_IDS = [f"m{index}" for index in range(10)]
MAPS = {"sampling_seed": 101, "file": "l4_map_pre.jsonl",
        "base": map_rows([_episodes(key, resolved) for key, resolved in zip(MAP_IDS, (0, 0, 0, 0, 0, 0, 1, 2, 4, 8))], MAP_IDS),
        "pre": map_rows([_episodes(key, resolved) for key, resolved in zip(MAP_IDS, (0, 0, 1, 2, 4, 6, 8, 8, 8, 8))], MAP_IDS)}


def _pretrain_report(prepare=PRETRAIN_PREPARE, minimums=(9, 4), beside=("loop", "ceiling_small", "ceiling_full"), base=None, **changed):
    models = {"pre": changed.get("pre") or _made("pre")}
    for name in beside:
        models[name] = {**_model("loop" if name == "loop" else "with"), **(prepare["ceiling"]["models"].get(name) or {})}
    return build_pretrain_report(prepare, PRETRAIN_TRAIN, GROUPS, LENGTHS, base or _model("base"), models, minimums, SETTINGS, EVALUATION, MAPS)


def test_the_pretrainings_report_reads_pre_against_the_base_draws_the_goal_set_again_and_reads_the_two_checks():
    report = _pretrain_report()
    assert report["label"] == LABEL and report["ok"] is True and report["checks_pass"] is True and all(line.startswith(f"{SAY}: ") for line in report["lines"])
    again = report["goal_set_again"]
    assert again["problem_ids"] == AGAIN and again["problems"] == 4 and again["by_length_group"] == {"1": 0, "2-3": 0, "4-7": 1, "8+": 3, "4_or_more": 4, "all": 4}
    checks = report["can_this_run_see_a_win"]
    assert {name: check["passes"] for name, check in checks.items() if isinstance(check, dict)} == {"the_pretraining_took": True, "the_goal_set_again_is_large_enough": True}
    assert checks["the_pretraining_took"]["goal_problems_solved"] == 9                               # of the 12, in all 93 attempts
    lines = report["lines"]
    assert "L4, THE PRETRAINING, seed 0. `pre`: ONE pass from the base over 24,800 published proofs of the `pretrain` half of the pool (8,100 lean_workbook, 16,700 "\
           "stp_conjecture), in the file's order, 3,100 optimizer steps. Distillation of other provers. The adapter is KEPT" in lines[0]
    assert lines[1].endswith("CHECK 1, the pretraining took: `pre` solves 9 of the 12 goal problems in its 93 one-shot attempts a problem; at least 9 is asked: PASS")
    assert lines[2].endswith("G' (the goal problems `pre` does not solve in its 32-attempt sampling) holds 4; at least 4 is asked: PASS")
    assert "G' by the length of the shortest published proof: 1 line: 0 of 3; 2-3 lines: 0 of 3; 4-7 lines: 1 of 3; 8+ lines: 3 of 3; 4 lines or more: 4 of 6; all of G: 4 of 12" in lines[3]
    # `pre` against the base on all of G: 84 successes against 24 in 12 x 93 attempts.
    on_all = report["against_the_base"]["by_length_group"]["models"]["pre"]["all"]
    assert (on_all["successes"], on_all["successes_of_the_base"], on_all["attempts_each"], on_all["problems"]) == (84, 24, 1116, 12) and on_all["low"] > 0
    assert on_all["mean"] == pytest.approx(60 / 1116, abs=1e-5)
    solved = report["against_the_base"]["goal_problems_solved"]["models"]["pre"]["all"]
    assert (solved["resolved_after"], solved["resolved_before"], solved["gained"], solved["lost"]) == (9, 8, 1, 0)
    assert report["against_the_base"]["the_three_rungs"]["models"]["pre"]["above"]["mean"] == pytest.approx((1 / 8 + 1 / 8) / 2, abs=1e-5)
    # Beside `pre`: the stored three-round model and the ceiling's two, each against the base, with the reliable counts by attempts alone.
    assert list(report["against_the_base"]["by_length_group"]["models"]) == ["pre", "loop", "ceiling_small", "ceiling_full"]
    assert report["models"]["ceiling_full"] == "the ceiling's 8,000-proof model" and report["models"]["loop"] == "the stored three-round model at t = 1/10"
    assert "pretrained on published proofs" in report["models"]["pre"] and "24,800 proofs" in report["models"]["pre"]
    alone = report["goal_problems_solved_by_attempts_alone"]
    assert set(alone) == {"what", "base", "pre", "loop", "ceiling_small", "ceiling_full"} and (alone["pre"]["all"]["solved_at_least_once"], alone["pre"]["all"]["reliably"]) == (9, 6)
    text = "\n".join(lines)
    assert "BY ATTEMPTS ALONE, all of G (12): the base 8 / 0; `pre` 9 / 6; the stored three-round model at t = 1/10 9 / 3; the ceiling's 2,000-proof model 12 / 4" in text
    assert "goal problems solved at 93 attempts, `pre` against the base: all of G: 9 to 8, gained 1, lost 0" in text
    assert "L4's pretraining (pretrained on published proofs) seed 0: `pre`, one pass over 24,800 published proofs. The pretraining took PASS (9 goal problems solved, at "\
           "least 9 asked); G' PASS (4 problems, at least 4 asked)" in report["headline"]
    assert report["pretraining_file"]["rows"] == 24800 and report["the_ceiling"]["read"] is True
    # THE MAP the arm's challenger starts from is `pre`'s own, printed beside the base's stored one: the problems by k of 8, the mean pass rate.
    the_map = report["the_map_the_arm_starts_from"]
    assert (the_map["file"], the_map["sampling_seed"]) == ("l4_map_pre.jsonl", 101) and the_map["pre"]["problems"] == 10 == the_map["base"]["problems"]
    assert the_map["pre"]["problems_by_k"] == {"0": 2, "1": 1, "2": 1, "3": 0, "4": 1, "5": 0, "6": 1, "7": 0, "8": 4} and the_map["pre"]["mean_pass_rate"] == round(45 / 80, 5)
    assert the_map["base"]["problems_by_k"]["0"] == 6 and the_map["base"]["mean_pass_rate"] == round(15 / 80, 5) and the_map["pre"]["resolved_at_least_once"] == 8
    assert ("THE MAP THE ARM'S CHALLENGER STARTS FROM is `pre`'s own (l4_map_pre.jsonl): the base map's 10 problems attempted again by `pre`, 8 attempts each, sampling seed 101. "
            "Problems by k of 8: `pre` 0: 2, 1: 1, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0, 8: 4, mean pass rate 0.5625; the base's stored map 0: 6, 1: 1, 2: 1, 3: 0, 4: 1, 5: 0, 6: 0, "
            "7: 0, 8: 1, mean pass rate 0.1875") in text


def test_a_failed_check_of_the_pretraining_is_said_and_the_report_is_still_written_and_a_smoke_run_has_nothing_beside():
    failed = _pretrain_report(minimums=(150, 80))                                                   # the real minimums on this small world
    assert failed["checks_pass"] is False and failed["ok"] is True                                  # the report can be read: it says the arm is not to be run
    assert failed["lines"][1].endswith("at least 150 is asked: FAIL") and failed["lines"][2].endswith("at least 80 is asked: FAIL")
    assert "The pretraining took FAIL" in failed["headline"] and failed["goal_set_again"]["problem_ids"] == AGAIN
    prepare = {**PRETRAIN_PREPARE, "stored_runs": None, "loop_arm": None, "loop_target_rate": None, "attempts_a_goal_problem": 32, "goal_samplings": SAMPLINGS[:1],
               "ceiling": {"read": False, "why": "a smoke run"}}
    first = lambda model: {**model, "goal": [model["goal"][0]]}      # noqa: E731
    smoke = build_pretrain_report(prepare, PRETRAIN_TRAIN, GROUPS, LENGTHS, first(_model("base")), {"pre": first(_made("pre"))}, (0, 0), SETTINGS, EVALUATION, MAPS)
    assert list(smoke["models"]) == ["base", "pre"] and smoke["checks_pass"] is True and smoke["goal_set_again"]["problem_ids"] == AGAIN
    assert any("L2's stored runs were not read (a smoke run)" in line for line in smoke["lines"]) and smoke["the_ceiling"] == {"read": False, "why": "a smoke run"}
    assert smoke["can_this_run_see_a_win"]["the_pretraining_took"]["goal_problems_solved"] == 8     # in its 32 attempts alone
    unanswered = _made("pre")
    unanswered["goal"][1][0]["attempts_without_an_answer"] = 20                                      # of the 61 attempts on one problem: over 2% of that set's 732
    report = _pretrain_report(pre=unanswered)
    assert report["ok"] is False and report["not_to_be_read"] == ["goal_pre_2"] and "NOT TO BE READ" in report["lines"][-1] and "NOT TO BE READ" in report["headline"]


# ----------------------------------------------------------------------------------------- the arm's report
PREPARE = {"seed": 0, "stage": "l3d2", "arm": "t010_assembly_pre", "rounds": [1, 2], "stand_in_engine": False, "attempts_a_goal_problem": 93, "rung_episodes": 8,
           "stored_runs": {"base": "ladder_l2_seed0", "loop": "ladder_l2_t010_seed0"}, "loop_arm": "t010", "loop_target_rate": 0.1,
           "sampling_seeds": {"rungs": 1002, "reach": 1001, "more": 1020}, "stored_measurements": {}, "contradicted_side_setting": "audit", "goal_samplings": SAMPLINGS}
OWN = {"stage": "l4", "label": LABEL, "seed": 0, "arm": "t010_assembly_pre", "pretraining_run": "ladder_l4_pretrain_seed0", "start_adapter": "ladder_l4_pretrain_seed0/adapters/pre",
       "pretraining_rows": 24800, "pretraining_file_sha256": "0" * 64, "goal_set_again": 4, "the_two_checks_of_the_pretraining": OF_THE_PRETRAINING,
       "base_arm": "t010_assembly", "base_arm_run": "ladder_l2_t010_assembly_seed0"}
ARM_PREPARE = {"arm": "t010_assembly_pre", "target_rate": 0.1, "rounds": [1, 2], "problems_a_round": 6, "batches": 2, "solvers": 8, "candidates": "loop_half",
               "candidates_of_the_whole_pool": 51631, "start": "pre", "start_adapter": "ladder_l4_pretrain_seed0/adapters/pre", "h0_rows": 0,
               "sampling_seeds": {"round_1": 1011, "round_2": 1012}, "data": {"candidates": 26029}}
# Each training's own summary: what it was trained from, and what it recorded when it compared the adapter it saved with the start adapter's file.
TRAINS = {"M(1)": {"rows": 20, "trained_from": "the stored adapter pre", "stand_in_engine": False, "against_the_start_adapter": {"m1": CHANGED}},
          "M(2)": {"rows": 40, "trained_from": "the stored adapter pre", "rows_by_origin": {"attempt": 30, "assembled": 10, "h0": 0}, "stand_in_engine": False,
                   "against_the_start_adapter": {"m2": CHANGED}},
          "`without`": {"rows": 30, "trained_from": "the stored adapter pre", "stand_in_engine": False, "against_the_start_adapter": {"without": CHANGED}}}
BASE_ARM = _model("with")["goal"]                    # the base arm's last model on G: Step 1's hand-made `with`


def _report(prepare=PREPARE, own=OWN, again=AGAIN, losses=LOSSES, trained_on=TRAININGS, base_arm=BASE_ARM, with_loop=True, trains=TRAINS, **changed):
    models = {name: changed.get(name) or _made(name) for name in ("pre", "with", "without")}
    if with_loop:
        models["loop"] = _model("loop")
    return build_l4_report(prepare, own, ARM_PREPARE, again, trains, losses, trained_on, [3, 3, 5, 9, 4, 2, 6, 3, 3, 12], ROUNDS, GROUPS, LENGTHS, _model("base"), models,
                           base_arm, SETTINGS, 0.10, EVALUATION)


def test_the_primary_is_on_the_goal_set_again_over_the_second_sampling_alone_with_minus_pre():
    report = _report()
    primary = report["primary"]
    # G' is 4 problems; over the 61 attempts of the second sampling `with` has 3 + 2 + 1 + 0 successes there and `pre` 1 + 0 + 0 + 0.
    assert (primary["goal_set_again"], primary["problems"], primary["successes"], primary["successes_of_the_base"], primary["attempts_each"]) == (4, 4, 6, 1, 244)
    assert primary["mean"] == pytest.approx(5 / 244, abs=1e-5) and (primary["per_1000"], primary["per_1000_of_the_base"]) == (24.59, 4.1)
    assert "over the 61 attempts of the second sampling" in primary["what"] and "`with` minus `pre`" in primary["what"]
    # The FIRST sampling chose G' and is not in the primary: `with`'s successes there (1 + 1 on these problems) do not move it ...
    more_in_the_first = _made("with", on_g=([6, 6, 6, 4, 4, 4, 3, 2, 9, 9, 9, 9], ON_G["with"][1]))
    assert _report(**{"with": more_in_the_first})["primary"]["successes"] == 6
    # ... a success in the second does; and it is against `pre`, not the base or the twin.
    one_more = _made("with", on_g=(ON_G["with"][0], [9, 9, 9, 6, 6, 6, 4, 3, 3, 2, 1, 1]))
    assert _report(**{"with": one_more})["primary"]["successes"] == 7
    better_pre = _made("pre", on_g=(ON_G["pre"][0], [9, 9, 9, 6, 6, 6, 3, 2, 1, 1, 0, 0]))
    assert _report(pre=better_pre)["primary"]["successes_of_the_base"] == 2
    assert _report(without=_made("without", on_g=(ON_G["without"][0], [9] * 12)))["primary"] == primary
    # Another G' (from the pretraining's report) is another primary; one that names no goal problem of this run is refused.
    assert _report(again=["g8a", "g8b"])["primary"]["attempts_each"] == 122
    with pytest.raises(ValueError, match="1 problems of G' are not goal problems of this run .first: elsewhere"):
        _report(again=[*AGAIN, "elsewhere"])


def test_the_report_reads_the_checks_first_then_the_primary_the_base_arm_beside_it_the_branch_and_the_secondary_reads():
    report = _report()
    lines = report["lines"]
    assert all(line.startswith(f"{SAY}: ") for line in lines) and report["label"] == LABEL and report["ok"] is True and report["inconclusive"] is False
    assert [line.split(": ", 1)[1].split(",")[0].split(".")[0].split(":")[0] for line in lines][:10] == [
        "L4", "CHECK 1", "CHECK 2", "CHECK 3", "CHECK 4", "CHECK 5", "for information", "PRIMARY", "BESIDE IT", "BRANCH"]
    assert all(line.split(": ", 1)[1].startswith("SECONDARY") for line in lines[10:])
    assert "L4, THE LOOP FROM A MODEL PRETRAINED ON PUBLISHED PROOFS, seed 0. The arm t010_assembly_pre: 2 rounds with assembly after each batch, its candidates the `loop` "\
           "half of the pool, round 1 attempted by `pre` (24,800 published proofs of the `pretrain` half) and every model trained FROM `pre`" in lines[0]
    assert "`with` is M(2), `pre` trained one more pass on 40 rows of the rounds; `without` is its twin, from `pre` on the 30 one-shot rows among them" in lines[0]
    # CHECK 3 reads the TWO MEASURED trainings (`with`'s, M(2), and `without`'s): the adapter saved is not the start adapter's, and the loss did not rise.
    assert ("CHECK 3, the two measured trainings of the arm ran (the adapter saved is not the start adapter's; the mean loss over the last tenth of its rows is not above "
            "the first tenth's by more than 2 standard errors of their difference): M(2): 1,000 of the 1,000 numbers of its adapter are not the start adapter's; loss 0.9 over "
            "the last 4 rows against 1.5 over the first, a rise of -0.6 where 0.0 is allowed (standard error 0.0); `without`: 1,000 of the 1,000 numbers") in lines[3]
    assert lines[3].endswith("a rise of -0.6 where 0.0 is allowed (standard error 0.0): PASS") and "M(1)" not in lines[3]
    assert lines[5].endswith("0 such problems in 90 rows over 3 trainings: PASS")
    # The other trainings (here M(1)) are printed the same way and decide nothing.
    assert lines[6].startswith(f"{SAY}: for information, deciding nothing, the other 1 trainings read the same way: M(1): 1,000 of the 1,000 numbers")
    ran = report["can_this_run_see_a_win"]["the_two_measured_trainings_ran"]
    assert set(ran) == {"what", "M(2)", "`without`", "passes"} and set(report["the_other_trainings"]) == {"what", "M(1)"} and report["the_other_trainings"]["M(1)"]["passes"] is True
    assert "PRIMARY. G' (4 goal problems `pre` does not solve in its 32-attempt sampling), successes per attempt over the 61 attempts of the second sampling, `with` minus "\
           "`pre`" in lines[7] and "24.59 against 4.1 per 1,000 (6 successes against 1 in 244 attempts each)" in lines[7]
    # BESIDE IT: the base's own G' is the 5 goal problems it does not solve in its 32 attempts; the base arm's last model has 3 + 2 + 2 + 1 + 1 there over the 61, the base 1.
    beside = report["beside_the_primary"]
    assert (beside["goal_set_again_of_the_base"], beside["problems"], beside["successes"], beside["successes_of_the_base"], beside["attempts_each"]) == (5, 5, 9, 1, 305)
    assert beside["mean"] == pytest.approx(8 / 305, abs=1e-5) and beside["run"] == "ladder_l2_t010_assembly_seed0"
    assert "BESIDE IT, the base arm's own gain from its stored rows (the 5 goal problems the base does not solve in its 32-attempt sampling, its last model minus the base, "\
           "over the same 61 attempts)" in lines[8] and "29.51 against 3.28 per 1,000" in lines[8]
    assert report["branch"]["name"] in (ADDS, NOT_SHOWN) and lines[9].startswith(f"{SAY}: BRANCH: {report['branch']['name']}.")
    secondary = report["secondary"]
    assert list(secondary["by_length_group"]["pairs"]) == ["with_minus_pre", "with_minus_without", "pre_minus_base", "with_minus_base", "without_minus_base", "loop_minus_base"]
    assert secondary["by_length_group"]["pairs"]["with_minus_pre"]["pair"] == "`with` minus `pre`"
    on_all = secondary["by_length_group"]["pairs"]["with_minus_pre"]["all"]
    assert (on_all["successes"], on_all["successes_of_the_base"], on_all["attempts_each"]) == (95, 84, 1116)       # all of G, over all 93 attempts
    assert secondary["the_three_rungs"]["pairs"]["with_minus_pre"]["in"]["mean"] == pytest.approx(1 / 8 / 2, abs=1e-5)
    by_group = secondary["the_goal_set_again_by_length_group"]
    assert (by_group["4-7"]["problems"], by_group["8+"]["problems"], by_group["1"]["problems"], by_group["1"]["mean"]) == (1, 3, 0, None)
    assert (by_group["8+"]["successes"], by_group["8+"]["successes_of_the_base"]) == (3, 0) and by_group["4_or_more"]["successes"] == 6
    twin = secondary["with_minus_without"]
    assert (twin["on_the_goal_set_again"]["successes"], twin["on_the_goal_set_again"]["successes_of_the_base"]) == (6, 3)
    assert (twin["assembled_proofs_trained_on"], twin["assembled_in_the_rounds"]) == (10, 10)
    assert (twin["lines_once_minimised"]["median"], twin["lines_once_minimised"]["longest"], twin["lines_once_minimised"]["share_with_4_lines_or_more"]) == (3.5, 12, 0.5)
    alone = secondary["goal_problems_solved_by_attempts_alone"]
    assert set(alone) == {"what", "base", "pre", "with", "without", "loop"} and (alone["with"]["all"]["solved_at_least_once"], alone["with"]["all"]["reliably"]) == (11, 6)
    solved = secondary["goal_problems_solved"]["with_minus_pre"]
    assert solved["pair"] == "`with` against `pre`" and (solved["all"]["resolved_after"], solved["all"]["resolved_before"], solved["all"]["gained"]) == (11, 9, 2)
    assert [row["round"] for row in secondary["by_round"]["rows"]] == [1, 2] and secondary["by_round"]["rows"][1]["only_assembly_resolved"] == 2
    assert set(secondary["distinct_attempts"]) == {"what", "base", "pre", "with", "without", "loop"}
    text = "\n".join(lines)
    assert "SECONDARY, G' by the length of the shortest published proof, `with` minus `pre` over the 61 attempts of the second sampling: 1 line (0): not measured" in text
    assert "do assembled proofs teach at this strength: `with` minus `without` on G' over the 61 attempts of the second sampling" in text
    assert "`with` was trained on 10 assembled proofs that `without` was not; the 10 proofs the rounds assembled, once minimised: median 3.5 lines" in text
    assert "BY ATTEMPTS ALONE, all of G (12): the base 8 / 0; `pre` 9 / 6; `with` 11 / 6; `without` 10 / 6; the stored three-round model at t = 1/10 9 / 3. With assembly: not in" in text
    assert "goal problems solved at 93 attempts, `with` against `pre`: all of G: 11 to 9, gained 2, lost 0" in text
    assert "round 1: 6; 2; 1, 0.25; 0.20833; 0.33333; round 2: 4; 1; 2, 0.66667; 0.03125; 0.33333" in text
    assert "L4 (the loop from a model pretrained on published proofs) seed 0:" in report["headline"] and "the base arm's own gain beside it:" in report["headline"]
    assert "pretrained on published proofs (24,800 proofs of the pool's `pretrain` half)" in report["models"]["pre"] and "from `pre` on its one-shot rows alone" in report["models"]["without"]
    assert report["the_pretraining"]["pretraining_run"] == "ladder_l4_pretrain_seed0" and report["the_arm"]["candidates"] == "loop_half" and report["the_arm"]["start"] == "pre"
    assert report["trainings"]["M(1)"]["trained_from"] == "the stored adapter pre" and report["heldout"]["goal_set_again"] == 4


def test_the_branches_follow_the_primarys_interval_and_the_note_sets_the_base_arms_gain_against_it():
    # `with` clearly above `pre` on every problem of G': the loop adds.
    above = _made("with", on_g=(ON_G["with"][0], [9, 9, 9, 6, 6, 6, 4, 3, 5, 4, 4, 3]))
    adds = _report(**{"with": above})
    assert adds["primary"]["low"] > 0 and adds["branch"]["name"] == ADDS
    # One problem up and one down: the interval holds zero, and the note reads the base arm's own gain (8 / 305) against half its width.
    mixed = _made("with", on_g=(ON_G["with"][0], [9, 9, 9, 6, 6, 6, 4, 3, 0, 1, 0, 0]))
    through = _report(**{"with": mixed})
    assert through["primary"]["low"] < 0 < through["primary"]["high"] and through["branch"]["name"] == NOT_SHOWN
    note = through["branch"]["the_note"]
    assert note["base_arms_gain"] == through["beside_the_primary"]["mean"] and note["resolves"] == round((through["primary"]["high"] - through["primary"]["low"]) / 2, 6)
    assert note["seen"] == (abs(note["base_arms_gain"]) >= note["resolves"])
    # With no base arm's rows the note says so, and the branch is the same.
    alone = _report(**{"with": mixed}, base_arm=None)
    assert alone["beside_the_primary"] is None and alone["branch"]["name"] == NOT_SHOWN and alone["branch"]["the_note"]["seen"] is None
    assert any("BESIDE IT: the base arm's stored rows were not read" in line for line in alone["lines"])
    # `with` below `pre` on G': the rounds cost the pretrained model.
    stronger_pre = _made("pre", on_g=(ON_G["pre"][0], [9, 9, 9, 6, 6, 6, 3, 2, 5, 4, 4, 3]))
    below = _report(pre=stronger_pre, **{"with": _made("with", on_g=(ON_G["with"][0], [9, 9, 9, 6, 6, 6, 4, 3, 1, 0, 0, 0]))})
    assert below["primary"]["high"] < 0 and below["branch"]["name"] == COSTS


def test_a_failing_check_gives_inconclusive_and_nothing_else_is_said():
    for change, name, line in (
            ({"own": {**OWN, "the_two_checks_of_the_pretraining": {**OF_THE_PRETRAINING, "the_goal_set_again_is_large_enough": {"problems": 4, "minimum": 80, "passes": False}}}},
             "the_goal_set_again_is_large_enough", 2),
            ({"losses": {**LOSSES, "`without`": list(reversed(LOSSES["`without`"]))}}, "the_two_measured_trainings_ran", 3),              # the twin's loss rose
            ({"trains": {**TRAINS, "M(2)": {**TRAINS["M(2)"], "against_the_start_adapter": {"m2": {**CHANGED, "elements_changed": 0}}}}},
             "the_two_measured_trainings_ran", 3),                                                                                         # `with`'s adapter IS `pre`'s
            ({"trains": {**TRAINS, "`without`": {key: value for key, value in TRAINS["`without`"].items() if key != "against_the_start_adapter"}}},
             "the_two_measured_trainings_ran", 3),                                                                                         # the twin's was never compared
            ({"with": {**_made("with"), "rungs": [_rung(problem_id, 0, capped=8) for problem_id in RUNG_IDS]}}, "each_measured_model_still_writes_proofs", 4),
            ({"trained_on": {**TRAININGS, "M(2)": [*TRAININGS["M(2)"][:-1], OF_THE_OTHER_HALF]}}, "no_training_row_is_of_the_pretrain_half_or_held_out", 5)):
        broken = _report(**change)
        assert broken["inconclusive"] is True and broken["branch"]["name"] == INCONCLUSIVE and broken["branch"]["failed_checks"] == [name]
        assert broken["primary"] is None and broken["beside_the_primary"] is None and broken["secondary"] is None
        assert [text.split(": ", 1)[1].split(",")[0].split(".")[0] for text in broken["lines"]] == ["L4", "CHECK 1", "CHECK 2", "CHECK 3", "CHECK 4", "CHECK 5", "for information",
                                                                                                     "INCONCLUSIVE"]
        assert broken["lines"][line].endswith("FAIL") and broken["measured_and_not_read"]["primary"]["successes"] == 6      # kept for whoever repairs the run
        assert "Primary" not in broken["headline"] and "INCONCLUSIVE" in broken["headline"]
    # A loss that did not fall (flat on both measured trainings) is NOT a failure, and neither is anything about a training that is not measured: M(1)'s
    # loss rising, or its adapter never compared, is printed for information and decides nothing.
    flat = _report(losses={**LOSSES, "M(2)": [1.0] * 40, "`without`": [1.0] * 30})
    assert flat["inconclusive"] is False and flat["can_this_run_see_a_win"]["the_two_measured_trainings_ran"]["passes"] is True
    unmeasured = _report(losses={**LOSSES, "M(1)": list(reversed(LOSSES["M(1)"]))}, trains={**TRAINS, "M(1)": {"rows": 20}})
    assert unmeasured["inconclusive"] is False and unmeasured["the_other_trainings"]["M(1)"]["passes"] is False
    assert "M(1): its adapter was not compared with the start adapter's; loss 1.5 over the last 2 rows against 0.9 over the first, a rise of 0.6" in unmeasured["lines"][6]
    # A held-out problem among a training's rows is the same check.
    held = _report(trained_on={**TRAININGS, "M(1)": [*TRAININGS["M(1)"], "g1a"]})
    assert held["branch"]["failed_checks"] == ["no_training_row_is_of_the_pretrain_half_or_held_out"] and "1 such problems in 91 rows" in held["lines"][5]


def test_a_smoke_run_has_one_sampling_so_the_primary_is_not_read_and_a_set_lean_did_not_answer_is_not_to_be_read():
    prepare = {**PREPARE, "stored_runs": None, "loop_arm": None, "loop_target_rate": None, "attempts_a_goal_problem": 32, "goal_samplings": SAMPLINGS[:1]}
    first = lambda model: {**model, "goal": [model["goal"][0]]}      # noqa: E731
    models = {name: first(_made(name)) for name in ("pre", "with", "without")}
    smoke = build_l4_report(prepare, {**OWN, "base_arm_run": None}, ARM_PREPARE, AGAIN, TRAINS, LOSSES, TRAININGS, [], ROUNDS, GROUPS, LENGTHS, first(_model("base")), models,
                            None, SETTINGS, 0.10, EVALUATION)
    assert smoke["branch"]["name"] == NOT_READ and smoke["primary"]["mean"] is None and smoke["beside_the_primary"] is None and "loop" not in smoke["models"]
    assert any("G was attempted 32 times a problem in ONE sampling, so the primary has no fresh attempts to be read on" in line for line in smoke["lines"])
    assert smoke["secondary"]["with_minus_without"]["lines_once_minimised"]["proofs"] == 0 and smoke["secondary"]["by_length_group"]["pairs"]["with_minus_pre"]["all"]["attempts_each"] == 384
    unanswered = _made("with")
    unanswered["goal"][1][0]["attempts_without_an_answer"] = 20
    report = _report(**{"with": unanswered})
    assert report["ok"] is False and report["not_to_be_read"] == ["goal_with_2"] and "NOT TO BE READ" in report["lines"][-1] and "NOT TO BE READ" in report["headline"]
