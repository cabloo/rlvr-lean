"""The repair check's read and report on small hand-made inputs. Spec: docs/spec/ladder-loop.spec.md, "L3a: the
repair check, no training", "The read, fixed before the run". Pure. The rules for one failed proof are
`test_repair_cut.py`; the stage end to end is `test_ladder_l3a_stage.py`."""

import pytest

from rlvr_lean.domain.ladder_round.rounds import sign_test
from rlvr_lean.domain.repair import ARMS, BLIND, FIRST, RESUME_ARMS, RESUME_WITH_STATE, RESUME_WITHOUT_STATE
from rlvr_lean.domain.repair import read
from rlvr_lean.domain.repair.read import INCONCLUSIVE, NOT_SHOWN, STRONGER, WORSE_UNTRAINED
from rlvr_lean.reporting.ladder_l3a import build_l3a_report

LOOPS = 5
OK, BAD = "verified", "lean_error"
B, S, W = BLIND, RESUME_WITH_STATE, RESUME_WITHOUT_STATE
FOUR_BAD_LATER = ("lean_error",) * 4


def step(loop, status, how="blind", tokens=100, prompt=300, kept=0, repeats=None, no_state=None, requested=False, cut=None, sent=True, kept_lines_check=None):
    return {"loop": loop, "how": how, "had_state": how == "resume", "no_state": no_state, "status": status, "token_count": tokens, "prompt_tokens": prompt,
            "sent_to_lean": sent, "kept_lines": kept, "cut": cut, "repeats_failed_step": repeats, "state_requested": requested,
            "kept_lines_check": kept_lines_check}


def trimmed(before=(), fall_back=FOUR_BAD_LATER):
    """A resuming arm whose loop after `before` (failed resumed attempts) is TRIMMED: no goal was left at the cut and
    the kept lines verified. `fall_back`: what the same loop and the ones after it were in the reading without
    trimming (the blind attempt the stage also ran, then on from there)."""
    earlier = [step(index + 1, status, how="resume", requested=True, cut="body") for index, status in enumerate(before)]
    at = len(earlier) + 1
    own = step(at, OK, how="trimmed", tokens=0, prompt=0, kept=2, requested=True, cut="body", kept_lines_check="verified")
    others = [step(at + index, status, no_state="no_goals_at_the_cut" if index == 0 else None, requested=index == 0) for index, status in enumerate(fall_back)]
    without = next((entry["loop"] + 1 for entry in others if entry["status"] == OK), None)
    return {"resolved_at": at + 1, "loops": [*earlier, own], "trimmed_at": at, "without_trimming": {"resolved_at": without, "loops": [*earlier, *others[:LOOPS - at]]}}


def arm(*statuses, how="blind", **more):
    """An arm's attempts after a failed first one, to its first verified attempt."""
    steps = [step(index + 1, status, how=how, **more) for index, status in enumerate(statuses)]
    at = next((entry["loop"] + 1 for entry in steps if entry["status"] == OK), None)
    return {"resolved_at": at, "loops": steps}


def episode(problem, group, blind=(), state=(), without=(), number=0, first=BAD):
    """One episode. With a verified first attempt the arms have nothing to do; else each arm's statuses, loop by loop."""
    if first == OK:
        arms = {name: {"resolved_at": 1, "loops": []} for name in ARMS}
    else:
        arms = {B: arm(*blind) if not isinstance(blind, dict) else blind,
                S: arm(*state, how="resume", requested=True, cut="body") if not isinstance(state, dict) else state,
                W: arm(*without, how="resume", requested=True, cut="body") if not isinstance(without, dict) else without}
    return {"problem_id": problem, "group": group, "side": "true", "episode": number, "failed_first": first != OK,
            "first": {"status": first, "token_count": 90, "prompt_tokens": 200, "sent_to_lean": True}, "arms": arms}


FOUR_BAD = (BAD, BAD, BAD, BAD)


# ------------------------------------------------------------------------------------------------ episodes
def attempt(arm_name, problem, number, loop, status, **more):
    return {"arm": arm_name, "problem_id": problem, "episode": number, "loop": loop, "status": status, "audit": False, "how": "blind", "had_state": False,
            "no_state": None, "kept_lines": 0, "cut": None, "repeats_failed_step": None, "token_count": 50, "prompt_tokens": 120, "sent_to_lean": True, **more}


def test_an_episode_stops_at_its_first_verified_attempt_and_the_arms_share_the_first_attempt():
    problems = [{"problem_id": "p", "group": "goal", "side": "true"}]
    rows = [attempt(FIRST, "p", 0, 0, BAD), attempt(FIRST, "p", 1, 0, OK), attempt(FIRST, "p", 0, 0, OK, audit=True, side="negation")]
    for loop in (1, 2, 3, 4):
        rows.append(attempt(B, "p", 0, loop, BAD))
    rows += [attempt(S, "p", 0, 1, BAD, how="resume", had_state=True, kept_lines=2, cut="body", repeats_failed_step=True),
             attempt(S, "p", 0, 2, OK, how="resume", had_state=True, kept_lines=3, cut="body", repeats_failed_step=False),
             attempt(W, "p", 0, 1, BAD, how="resume", kept_lines=2, cut="body"), attempt(W, "p", 0, 2, BAD, no_state="state_timeout"),
             attempt(W, "p", 0, 3, BAD, how="resume", kept_lines=1, cut="whole_proof"), attempt(W, "p", 0, 4, OK, how="resume", kept_lines=4, cut="body")]
    states = [{"arm": FIRST, "problem_id": "p", "episode": 0, "after_loop": 0, "requested": True}, {"arm": S, "problem_id": "p", "episode": 0, "after_loop": 1, "requested": True},
              {"arm": W, "problem_id": "p", "episode": 0, "after_loop": 1, "requested": True}, {"arm": W, "problem_id": "p", "episode": 0, "after_loop": 2, "requested": False},
              {"arm": W, "problem_id": "p", "episode": 0, "after_loop": 3, "requested": True}]
    found = read.episode_rows(problems, rows, states, LOOPS)
    assert [(row["episode"], row["failed_first"]) for row in found] == [(0, True), (1, False)]          # the audited side's attempt is no episode
    failed, passed = found
    assert {name: failed["arms"][name]["resolved_at"] for name in ARMS} == {B: None, S: 3, W: 5}
    assert [len(failed["arms"][name]["loops"]) for name in ARMS] == [4, 2, 4]                           # at most four more, and none after a verified one
    assert all(passed["arms"][name] == {"resolved_at": 1, "loops": [], "trimmed_at": None, "without_trimming": None} for name in ARMS)
    assert all(failed["arms"][name]["trimmed_at"] is None and failed["arms"][name]["without_trimming"] is None for name in ARMS)
    assert {entry["kept_lines_check"] for name in ARMS for entry in failed["arms"][name]["loops"]} == {None}
    assert [entry["state_requested"] for entry in failed["arms"][W]["loops"]] == [True, True, False, True]      # loop 1 asks from the SHARED first attempt
    assert [entry["state_requested"] for entry in failed["arms"][S]["loops"]] == [True, True] and not any(entry["state_requested"] for entry in failed["arms"][B]["loops"])
    assert failed["arms"][W]["loops"][1]["no_state"] == "state_timeout" and failed["arms"][W]["loops"][1]["how"] == "blind"
    assert read.resolved(failed, S, 3) and not read.resolved(failed, S, 2) and not read.resolved(failed, B, LOOPS) and read.resolved(passed, B, 1)
    assert read.problem_rows(found, LOOPS) == [{"problem_id": "p", "group": "goal", "side": "true", "episodes": 2, "failed_first": 1,
                                                f"resolved_{B}": 1, f"resolved_{S}": 2, f"resolved_{W}": 2, f"trimmed_{S}": 0, f"trimmed_{W}": 0}]
    with pytest.raises(ValueError, match="whole episodes"):                                             # a loop that was never stored
        read.episode_rows(problems, [row for row in rows if not (row["arm"] == B and row["loop"] == 4)], states, LOOPS)


def test_a_trimmed_loop_resolves_the_episode_and_the_same_loop_as_a_blind_fall_back_is_read_beside_it():
    """Spec item 2a. No goal was left at the cut and the kept lines verified: that loop resolves the episode in
    the resuming arm with nothing generated. The stage also ran the loop as the blind fall-back it would have
    been, and went on from there (rows marked `untrimmed`): the reading without trimming."""
    problems = [{"problem_id": "p", "group": "below", "side": "true"}]
    cut = {"how": "trimmed", "token_count": 0, "prompt_tokens": 0, "kept_lines": 2, "cut": "body"}
    fall_back = {"untrimmed": True, "no_state": "no_goals_at_the_cut"}
    rows = [attempt(FIRST, "p", 0, 0, BAD), attempt(FIRST, "p", 1, 0, BAD)]
    rows += [attempt(B, "p", number, loop, BAD) for number in (0, 1) for loop in (1, 2, 3, 4)]
    # Episode 0: the first attempt itself had no goal left at its cut, so BOTH resuming arms trim at loop 1.
    rows += [attempt(S, "p", 0, 1, OK, **cut), attempt(S, "p", 0, 1, BAD, **fall_back), attempt(S, "p", 0, 2, OK, untrimmed=True, how="resume", had_state=True, kept_lines=1),
             attempt(W, "p", 0, 1, OK, **cut), *(attempt(W, "p", 0, loop, BAD, untrimmed=True, **({"no_state": "no_goals_at_the_cut"} if loop == 1 else {})) for loop in (1, 2, 3, 4))]
    # Episode 1: with the state the arm resumes and fails, then meets a cut with no goal left whose kept lines do NOT
    # verify (a blind attempt, as before), then one whose kept lines do: trimmed at loop 3.
    rows += [attempt(S, "p", 1, 1, BAD, how="resume", had_state=True, kept_lines=1), attempt(S, "p", 1, 2, BAD, no_state="no_goals_at_the_cut"),
             attempt(S, "p", 1, 3, OK, **cut), attempt(S, "p", 1, 3, BAD, **fall_back), attempt(S, "p", 1, 4, BAD, untrimmed=True),
             *(attempt(W, "p", 1, loop, BAD, how="resume", kept_lines=1) for loop in (1, 2, 3, 4))]
    asked = lambda who, number, after, **more: {"arm": who, "problem_id": "p", "episode": number, "after_loop": after, "requested": True, "trim_status": None, **more}      # noqa: E731
    states = [asked(FIRST, 0, 0, outcome="trimmed", trim_status="verified"), asked(S, 0, 1, outcome="state"),
              *(asked(W, 0, after, outcome="state") for after in (1, 2, 3)),
              asked(FIRST, 1, 0, outcome="state"), asked(S, 1, 1, outcome="no_goals_at_the_cut", trim_status="lean_error"),
              asked(S, 1, 2, outcome="trimmed", trim_status="verified"), asked(S, 1, 3, outcome="state"), *(asked(W, 1, after, outcome="state") for after in (1, 2, 3))]
    first, second = read.episode_rows(problems, rows, states, LOOPS)
    for name in (S, W):         # the spec's reading: resolved at the trimmed loop, with nothing generated
        own = first["arms"][name]
        assert (own["resolved_at"], own["trimmed_at"], len(own["loops"])) == (2, 1, 1)
        assert own["loops"][0] == {"loop": 1, "how": "trimmed", "had_state": False, "no_state": None, "status": OK, "token_count": 0, "prompt_tokens": 0,
                                   "sent_to_lean": True, "kept_lines": 2, "cut": "body", "repeats_failed_step": None, "state_requested": True,
                                   "kept_lines_check": "verified"}
    # The reading without trimming: that loop is the blind fall-back, and what followed it is read to its end.
    other = first["arms"][S]["without_trimming"]
    assert other["resolved_at"] == 3 and [(entry["loop"], entry["how"], entry["no_state"], entry["kept_lines_check"]) for entry in other["loops"]] == [
        (1, "blind", "no_goals_at_the_cut", None), (2, "resume", None, None)]
    assert first["arms"][W]["without_trimming"]["resolved_at"] is None and len(first["arms"][W]["without_trimming"]["loops"]) == 4
    assert first["arms"][B]["trimmed_at"] is None and first["arms"][B]["without_trimming"] is None
    own = second["arms"][S]
    assert (own["resolved_at"], own["trimmed_at"]) == (4, 3) and [entry["how"] for entry in own["loops"]] == ["resume", "blind", "trimmed"]
    assert [entry["kept_lines_check"] for entry in own["loops"]] == [None, "failed", "verified"]          # kept lines that did not verify: a blind attempt
    assert own["without_trimming"]["resolved_at"] is None and [entry["how"] for entry in own["without_trimming"]["loops"]] == ["resume", "blind", "blind", "blind"]
    assert {entry["kept_lines_check"] for entry in own["without_trimming"]["loops"]} == {None}             # no kept-lines check is part of that reading
    assert second["arms"][W]["trimmed_at"] is None
    assert read.resolved(first, S, 2) and not read.resolved(first, S, 2, trimming=False) and read.resolved(first, S, 3, trimming=False)
    assert read.resolved(first, W, LOOPS) and not read.resolved(first, W, LOOPS, trimming=False) and not read.resolved(second, S, LOOPS, trimming=False)
    assert read.problem_rows([first, second], LOOPS) == [{"problem_id": "p", "group": "below", "side": "true", "episodes": 2, "failed_first": 2,
                                                         f"resolved_{B}": 0, f"resolved_{S}": 2, f"resolved_{W}": 1, f"trimmed_{S}": 2, f"trimmed_{W}": 1}]
    with pytest.raises(ValueError, match="whole episodes"):             # the reading without trimming must be whole too
        read.episode_rows(problems, [row for row in rows if not (row["arm"] == W and row.get("untrimmed") and row["loop"] == 4)], states, LOOPS)


def test_the_primary_the_single_step_and_the_goal_set_are_read_with_the_trimmed_loops_and_without_them():
    rows = [episode("g1", "goal", FOUR_BAD, trimmed(), trimmed()),                          # resolved by trimming only: without it, never
            episode("g1", "goal", FOUR_BAD, FOUR_BAD, FOUR_BAD, number=1),
            episode("g2", "goal", FOUR_BAD, trimmed(fall_back=(BAD, OK)), trimmed(fall_back=(BAD, OK))),      # resolved either way, later without it
            episode("b1", "below", FOUR_BAD, trimmed(before=(BAD,)), (BAD, BAD, BAD, BAD)),   # the arms cut their own proofs after loop 1
            episode("a1", "above", (OK,), trimmed(fall_back=(OK,)), trimmed(fall_back=(OK,))), episode("a2", "above", (BAD, OK), (OK,), (OK,))]
    found = read.repair_read(rows, LOOPS, 500, 0)
    # With them (the spec's primary): g1 (1 + 0) / 2, g2 1, b1 1. Without: g1 0, g2 1, b1 0.
    assert found["primary"]["mean"] == round(2.5 / 3, 5) and found["primary"][S] == round(2.5 / 3, 5) and found["primary"][B] == 0.0
    without = found["without_trimming"]
    assert without["primary"]["mean"] == round(1 / 3, 5) and without["primary"][S] == round(1 / 3, 5) and without["primary"]["problems"] == 3
    assert found["comparisons"]["resume_with_state_minus_resume_without_state"]["hard"]["mean"] == round(1 / 3, 5)     # b1: only the arm with the state trimmed
    # The single step: the next attempt after a failed first one.
    assert found["single_step"]["hard"][S] == {"verified": 2, "share": 0.5, "of_which_trimmed": 2}
    assert found["single_step"]["hard"][B] == {"verified": 0, "share": 0.0, "of_which_trimmed": 0}
    assert without["single_step"]["hard"][S] == {"verified": 0, "share": 0.0, "of_which_trimmed": 0}
    assert found["single_step"]["above"][S] == {"verified": 2, "share": 1.0, "of_which_trimmed": 1} and without["single_step"]["above"][S]["verified"] == 2
    # The goal set by problem.
    with_them, without_them = found["goal_set_by_problem"]["resume_with_state_minus_blind"], without["goal_set_by_problem"]["resume_with_state_minus_blind"]
    assert (with_them["gained"], with_them["lost"], with_them[f"resolved_{S}"]) == (2, 0, 2) and (without_them["gained"], without_them[f"resolved_{S}"]) == (1, 1)
    # Reported apart, by arm, set and loop, with how many the arm resolved anyway.
    assert found["trimmed"][S] == {"resolutions": 4, "by_set": {"goal": 2, "below": 1, "in": 0, "above": 1, "hard": 3, "all": 4},
                                   "by_loop": {"1": 3, "2": 1}, "resolved_without_trimming_too": 2}
    assert found["trimmed"][W]["resolutions"] == 3 and found["trimmed"][W]["by_set"]["hard"] == 2 and B not in found["trimmed"]
    # A trimmed loop generates nothing and costs the state request and the kept-lines check.
    spent = read.arm_budget(read._of(rows, ("goal",)), S)
    assert spent == {"attempts": 6, "of_which_trimmed": 2, "generated_tokens": 400, "prompt_tokens": 1200, "attempts_sent_to_lean": 4, "state_requests": 6,
                     "kept_lines_checks": 2, "lean_checks": 12}
    assert found["diagnostics"][S]["trimmed"] == 4 and found["diagnostics"][S]["without_a_state_by_reason"] == {}
    # The above-band check counts a trimmed loop as the resolved loop it is, and says how many of its successes were trimmed.
    check = found["can_this_run_see_a_win"]["the_resumed_next_attempt_on_the_above_band_rung"]
    assert (check["resume_with_state_share_verified"], check["resume_with_state_verified"], check["of_which_trimmed"], check["passes"]) == (1.0, 2, 1, True)


# --------------------------------------------------------------------------------------- paired comparisons
def test_the_primary_is_paired_by_episode_with_each_problems_mean_over_its_episodes():
    rows = [episode("g1", "goal", FOUR_BAD, (BAD, OK), FOUR_BAD, number=0), episode("g1", "goal", FOUR_BAD, FOUR_BAD, FOUR_BAD, number=1),
            episode("g1", "goal", first=OK, number=2), episode("g1", "goal", (OK,), FOUR_BAD, (OK,), number=3),
            episode("b1", "below", FOUR_BAD, (OK,), FOUR_BAD), episode("b1", "below", FOUR_BAD, (BAD, BAD, BAD, OK), FOUR_BAD, number=1),
            episode("i1", "in", (OK,), FOUR_BAD, FOUR_BAD), episode("a1", "above", (OK,), (OK,), (OK,))]
    result = read.repair_read(rows, LOOPS, 500, 0)
    primary = result["primary"]
    # g1: (1 - 0, 0 - 0, 1 - 1, 0 - 1) / 4 = 0; b1: (1 + 1) / 2 = 1. The mean over the two problems is 0.5.
    assert (primary["problems"], primary["episodes"], primary["mean"]) == (2, 6, 0.5) and primary[S] == 0.75 and primary[B] == 0.25
    assert 0.0 <= primary["low"] <= primary["mean"] <= primary["high"] <= 1.0
    assert result["comparisons"]["resume_with_state_minus_blind"]["hard"] == primary
    by_set = result["comparisons"]["resume_with_state_minus_blind"]
    assert by_set["goal"]["mean"] == 0.0 and by_set["below"]["mean"] == 1.0 and by_set["in"]["mean"] == -1.0 and by_set["above"]["mean"] == 0.0
    assert by_set["all"]["problems"] == 4 and by_set["all"]["mean"] == 0.0
    assert result["comparisons"]["resume_without_state_minus_blind"]["hard"]["mean"] == 0.0
    assert result["comparisons"]["resume_with_state_minus_resume_without_state"]["hard"]["mean"] == 0.5
    # Within fewer attempts: the fourth loop's repair on b1 is not there yet.
    assert read.paired(read._of(rows, ("below",)), S, B, 4, 200, 0)["mean"] == 0.5
    within = result["resolved_within_attempts"]["hard"]
    assert within[S] == {"1": 0.125, "2": 0.375, "3": 0.5, "4": 0.5, "5": 0.75} and within[B]["5"] == 0.25 and within[B]["1"] == 0.125


def test_the_single_step_is_the_next_attempt_after_a_failed_first_one_by_arm():
    held = arm(OK, how="resume", requested=True)
    fell_back = {"resolved_at": None, "loops": [step(1, BAD, no_state="no_error_position")] + [step(loop, BAD) for loop in (2, 3, 4)]}
    rows = [episode("a1", "above", (OK,), held, (BAD, OK)), episode("a1", "above", (BAD, OK), fell_back, fell_back, number=1),
            episode("a1", "above", first=OK, number=2), episode("a2", "above", (OK,), (BAD, OK), (OK,))]
    found = read.single_step(rows)
    assert found["failed_first_attempts"] == 3 and found[B] == {"verified": 2, "share": 0.66667, "of_which_trimmed": 0}
    assert found[S] == {"verified": 1, "share": 0.33333, "of_which_trimmed": 0} and found[W] == {"verified": 1, "share": 0.33333, "of_which_trimmed": 0}
    # Only the next attempts whose prompt held a state, with the other arms on those same two episodes.
    assert found["where_the_prompt_held_a_state"] == {"attempts": 2, S: 0.5, W: 0.5, B: 1.0}
    assert read.single_step([])["failed_first_attempts"] == 0 and read.single_step([])[B]["share"] is None


def test_the_goal_set_is_read_by_problem_as_gained_against_lost_with_the_sign_test():
    rows = [episode("g1", "goal", FOUR_BAD, (OK,), FOUR_BAD), episode("g1", "goal", FOUR_BAD, FOUR_BAD, FOUR_BAD, number=1),
            episode("g2", "goal", (BAD, OK), FOUR_BAD, FOUR_BAD), episode("g3", "goal", (OK,), (OK,), FOUR_BAD),
            episode("g4", "goal", FOUR_BAD, (BAD, BAD, OK), FOUR_BAD), episode("g5", "goal", FOUR_BAD, FOUR_BAD, FOUR_BAD),
            episode("b1", "below", FOUR_BAD, (OK,), (OK,))]
    counted = read.repair_read(rows, LOOPS, 200, 0)["goal_set_by_problem"]["resume_with_state_minus_blind"]
    assert counted == {"problems": 5, f"resolved_{S}": 3, f"resolved_{B}": 2, "by_both": 1, "gained": 2, "lost": 1, "sign_test_p": sign_test(2, 1)}
    assert read.gained_against_lost(read._of(rows, ("goal",)), S, B, 2)["gained"] == 1          # within two attempts g4's third is not there


# -------------------------------------------------------------------------------------------------- budget
def test_the_budget_counts_tokens_and_lean_checks_by_arm_and_a_resume_loop_costs_one_more_check():
    capped = {"resolved_at": None, "loops": [step(1, "capped_tokens", how="resume", tokens=1024, requested=True, sent=False)]
              + [step(loop, BAD, tokens=100, no_state="no_error_position") for loop in (2, 3, 4)]}
    rows = [episode("g1", "goal", FOUR_BAD, capped, (BAD, OK)), episode("a1", "above", first=OK)]
    spent = read.budget(rows, read._of(rows, ("goal",)), LOOPS, 200, 0)
    assert spent["first_attempts"] == {"attempts": 2, "generated_tokens": 180, "prompt_tokens": 400, "lean_checks": 2}
    assert spent["by_arm"][B] == {"attempts": 4, "of_which_trimmed": 0, "generated_tokens": 400, "prompt_tokens": 1200, "attempts_sent_to_lean": 4,
                                  "state_requests": 0, "kept_lines_checks": 0, "lean_checks": 4}
    # The capped attempt was not sent; its loop still asked for a state; the three blind fall-backs asked for none.
    assert spent["by_arm"][S] == {"attempts": 4, "of_which_trimmed": 0, "generated_tokens": 1324, "prompt_tokens": 1200, "attempts_sent_to_lean": 3,
                                  "state_requests": 1, "kept_lines_checks": 0, "lean_checks": 4}
    # Kept lines that were checked on their own and did not verify are one more check on the blind attempt that loop then was.
    failed_check = {"resolved_at": None, "loops": [step(1, BAD, no_state="no_goals_at_the_cut", requested=True, kept_lines_check="failed")]
                    + [step(loop, BAD) for loop in (2, 3, 4)]}
    assert read.arm_budget([episode("g1", "goal", FOUR_BAD, failed_check, FOUR_BAD)], S)["lean_checks"] == 4 + 1 + 1
    assert spent["by_arm"][W]["lean_checks"] == 4 and spent["by_arm"][W]["attempts"] == 2


def test_the_read_at_equal_tokens_is_made_only_when_the_resume_arm_generates_more_than_ten_percent_more():
    def rows(resume_tokens):
        return [episode("g1", "goal", FOUR_BAD, arm(BAD, BAD, BAD, OK, how="resume", tokens=resume_tokens, requested=True), FOUR_BAD),
                episode("b1", "below", (BAD, BAD, BAD, OK), arm(OK, how="resume", tokens=resume_tokens, requested=True), FOUR_BAD)]
    same = read.repair_read(rows(110), LOOPS, 200, 0)["budget"]
    assert same["generated_tokens_resume_with_state_over_blind_on_the_primarys_problems"] == 0.6875 and same["resume_generates_more_than_10_percent_more"] is False
    assert isinstance(same["equal_tokens"], str) and "read as it stands" in same["equal_tokens"]
    more = read.repair_read(rows(300), LOOPS, 200, 0)["budget"]             # 5 x 300 against 8 x 100 tokens
    assert more["generated_tokens_resume_with_state_over_blind_on_the_primarys_problems"] == 1.875 and more["resume_generates_more_than_10_percent_more"] is True
    equal = more["equal_tokens"]
    assert equal["blind_attempts_after_the_first_that_match"] == 7.5 and equal["blind_attempts_after_the_first_that_were_sampled"] == 4
    assert equal["the_blind_arm_at_the_matching_number_can_be_read"] is False
    # Its mirror: the resume arm at the loops whose tokens match the blind arm's 800 (loop 1 is 600 tokens, loop 2 makes 900).
    assert equal["mirror"]["resume_loops_read"] == 1 and equal["mirror"]["resume_generated_tokens"] == 600 and equal["mirror"]["blind_generated_tokens"] == 800
    assert equal["mirror"]["primary"]["mean"] == 0.0 and equal["mirror"]["primary"][S] == 0.5 and equal["mirror"]["primary"][B] == 0.5


# ------------------------------------------------------------------------------- what the model did with it
def test_the_diagnostics_say_how_often_there_was_a_state_what_was_repeated_and_how_the_cut_moved():
    resumed = {"resolved_at": None, "loops": [step(1, BAD, how="resume", kept=2, repeats=True, requested=True, cut="body"),
                                             step(2, BAD, how="resume", kept=4, repeats=False, requested=True, cut="body"),
                                             step(3, BAD, no_state="error_after_the_sorry", requested=True),
                                             step(4, BAD, how="resume", kept=0, repeats=None, requested=True, cut="whole_proof")]}
    other = {"resolved_at": None, "loops": [step(1, BAD, how="resume", kept=3, repeats=True, requested=True, cut="body"),
                                           step(2, BAD, how="resume", kept=3, repeats=True, requested=True, cut="body"),
                                           step(3, BAD, how="resume", kept=1, repeats=False, requested=True, cut="body"),
                                           step(4, BAD, no_state="no_error_position")]}
    rows = [episode("g1", "goal", FOUR_BAD, resumed, resumed), episode("g2", "goal", FOUR_BAD, other, other)]
    found = read.diagnostics(rows)
    assert set(found) == set(RESUME_ARMS)
    own = found[S]
    assert (own["repair_loops"], own["with_a_state"], own["share_with_a_state"]) == (8, 6, 0.75)
    assert own["without_a_state_by_reason"] == {"error_after_the_sorry": 1, "no_error_position": 1}
    assert own["by_loop"]["1"] == {"loops": 2, "resumed": 2, "share_resumed": 1.0, "mean_kept_lines": 2.5} and own["by_loop"]["3"]["share_resumed"] == 0.5
    assert own["by_loop"]["4"]["mean_kept_lines"] == 0.0 and own["nothing_kept"] == 1 and own["whole_proof_kept"] == 1
    assert own["first_step_repeats_the_failed_step"] == {"loops_with_a_failed_step": 5, "repeats": 3, "share": 0.6}
    # Loop to loop where both resumed: 2 -> 4 in the first episode; 3 -> 3 and 3 -> 1 in the second.
    moved = own["cut_from_loop_to_loop"]
    assert (moved["pairs"], moved["mean_change_in_kept_lines"]) == (3, 0.0)
    assert (moved["share_further_along"], moved["share_at_the_same_line"], moved["share_further_back"]) == (0.33333, 0.33333, 0.33333)


# ------------------------------------------------------------------------------- can this run see a win
def above(blind_ok, state_ok, total=10, held=True):
    """`total` failed first attempts on the above-band rung whose next attempt verifies `blind_ok` times blind and
    `state_ok` times resumed."""
    rows = []
    for index in range(total):
        state = arm(OK if index < state_ok else BAD, *(() if index < state_ok else (BAD, BAD, BAD)), how="resume", requested=True) if held else \
            {"resolved_at": None, "loops": [step(loop, BAD, no_state="no_sorry_at_the_cut") for loop in (1, 2, 3, 4)]}
        rows.append(episode(f"a{index}", "above", (OK,) if index < blind_ok else FOUR_BAD, state, FOUR_BAD))
    return rows


def hard(state_wins, blind_wins, total=12):
    rows = []
    for index in range(total):
        rows.append(episode(f"g{index}", "goal", (OK,) if index < blind_wins else FOUR_BAD, (OK,) if total - index <= state_wins else FOUR_BAD, FOUR_BAD))
    return rows


def test_both_checks_must_pass_and_either_failing_makes_the_run_inconclusive():
    fine = read.repair_read(above(6, 3) + hard(6, 0), LOOPS, 500, 0)            # resumed 0.3 against blind 0.6: exactly half
    checks = fine["can_this_run_see_a_win"]
    assert checks["passes"] is True and checks["the_resumed_next_attempt_on_the_above_band_rung"]["passes"] is True
    assert checks["the_resumed_next_attempt_on_the_above_band_rung"]["blind_share_verified"] == 0.6
    assert checks["a_state_for_the_failed_first_attempts"] == {"failed_first_attempts": 22, "with_a_state": 22, "share": 1.0, "needed": "at least half", "passes": True}
    assert fine["branch"]["name"] == STRONGER and "L3b" in fine["branch"]["reason"]

    broken_prompt = read.repair_read(above(6, 2) + hard(6, 0), LOOPS, 500, 0)   # 0.2 against 0.6: less than half
    assert broken_prompt["can_this_run_see_a_win"]["the_resumed_next_attempt_on_the_above_band_rung"]["passes"] is False
    assert broken_prompt["branch"]["name"] == INCONCLUSIVE and "the prompt or the cut is broken" in broken_prompt["branch"]["reason"]
    assert "not a verdict" in broken_prompt["branch"]["reason"]

    few_states = above(6, 6, held=False) + hard(6, 0, total=4)                  # 10 of 14 failed first attempts got no state
    broken_cut = read.repair_read(few_states, LOOPS, 500, 0)
    states = broken_cut["can_this_run_see_a_win"]["a_state_for_the_failed_first_attempts"]
    assert (states["with_a_state"], states["share"], states["passes"]) == (4, 0.28571, False)
    assert broken_cut["branch"]["name"] == INCONCLUSIVE and "the cut is broken" in broken_cut["branch"]["reason"]

    # A check that cannot be read has not passed: no failed first attempt on the above-band rung, or a blind arm that verified none.
    assert read.repair_read(hard(6, 0), LOOPS, 200, 0)["branch"]["name"] == INCONCLUSIVE
    assert read.repair_read(above(0, 0) + hard(6, 0), LOOPS, 200, 0)["can_this_run_see_a_win"]["passes"] is False


def test_the_spec_counts_the_arms_blind_fall_backs_and_the_report_shows_the_resumed_attempts_beside_it():
    """Where a loop got no state the arm's attempt IS a blind one, so the arm's share can pass while the attempts
    that did hold a state verify far less often. The gate is the spec's; the other number stands beside it."""
    rows = []
    for index in range(10):                     # five fall-backs that verify as blind does, five resumed attempts that never do
        fell_back = {"resolved_at": 2, "loops": [step(1, OK, no_state="no_goals_at_the_cut")]}
        rows.append(episode(f"a{index}", "above", (OK,), fell_back if index < 5 else arm(*FOUR_BAD, how="resume", requested=True), FOUR_BAD))
    check = read.can_see_a_win(rows + hard(1, 0))["the_resumed_next_attempt_on_the_above_band_rung"]
    assert (check["blind_share_verified"], check["resume_with_state_share_verified"], check["passes"]) == (1.0, 0.5, True)
    assert check["only_where_the_prompt_held_a_state"] == {"attempts": 5, S: 0.0, W: 0.0, B: 1.0} and check["only_where_the_prompt_held_a_state_at_least_half_of_blind"] is False


def test_the_branch_is_read_from_the_primarys_interval_in_the_specs_words():
    passing = above(6, 6)
    assert read.repair_read(passing + hard(8, 0), LOOPS, 500, 0)["branch"]["name"] == STRONGER
    worse = read.repair_read(passing + hard(0, 8), LOOPS, 500, 0)["branch"]
    assert worse["name"] == WORSE_UNTRAINED and "not a verdict on repair" in worse["reason"] and "FOR THIS MODEL UNTRAINED" in worse["reason"]
    level = read.repair_read(passing + hard(3, 3), LOOPS, 500, 0)
    assert level["primary"]["mean"] == 0.0 and level["primary"]["low"] < 0 < level["primary"]["high"]
    assert level["branch"]["name"] == NOT_SHOWN and "needs a decision" in level["branch"]["reason"]
    checks = {"passes": True}
    assert read.repair_branch({"mean": 0.02, "low": 0.0, "high": 0.05}, checks).name == NOT_SHOWN          # an interval that touches zero is not clear of it
    assert read.repair_branch({"mean": 0.02, "low": 0.001, "high": 0.05}, checks).name == STRONGER
    assert read.repair_branch({"mean": None, "low": None, "high": None}, checks).name == INCONCLUSIVE
    assert set(read.BRANCHES) == {STRONGER, NOT_SHOWN, WORSE_UNTRAINED, INCONCLUSIVE}


# -------------------------------------------------------------------------------------------------- report
PREPARE = {"seed": 0, "fixture": False, "stand_in_engine": False, "goal_set": 12, "rungs": {"below": 0, "in": 0, "above": 10},
           "sizes": {"episodes_goal": 4, "episodes_rungs": 2, "loops": LOOPS, "max_new_tokens": 1024, "lean_seconds": 30, "sampling_seed": 1030},
           "problems_with_no_side_to_attempt": []}
STEP = {"problems": 22, "episodes": 22, "attempts": 200, "generated_tokens": 1000, "lean_checks_sent": 250, "loops": [], "stand_in_engine": False}
EVALUATION = {"bootstrap_resamples": 300, "bootstrap_seed": 0}


def test_the_report_answers_the_read_and_names_the_branch():
    report = build_l3a_report(PREPARE, above(6, 4) + hard(6, 1), STEP, {}, EVALUATION)
    assert report["branch"]["name"] == STRONGER and report["inconclusive"] is False and report["ok"] is True and report["seed"] == 0
    assert report["headline"].startswith("L3a seed 0: REPAIR IS A STRONGER SEARCH STEP. Primary (G and the below-band rung, 12 problems, 12 episodes resolved within 5 attempts")
    for needed in ("without the state minus blind", "with the state minus without it", "the next attempt after a failed first one (12 on these problems)",
                   "G by problem (12), with the state against blind: 6 to 1, gained 6, lost 1", "a state for 1.0 of the failed first attempts",
                   "above-band check: resumed 0.4 against blind 0.6"):
        assert needed in report["headline"], needed
    assert report["primary"]["mean"] == round(5 / 12, 5) and "paired by episode" in report["primary"]["what"]
    assert set(report["comparisons"]) == {"what", "resume_with_state_minus_blind", "resume_without_state_minus_blind", "resume_with_state_minus_resume_without_state"}
    assert set(report["comparisons"]["resume_with_state_minus_blind"]) == {"goal", "below", "in", "above", "hard", "all"}
    assert report["single_step"]["above"][B]["share"] == 0.6 and report["goal_set_by_problem"]["resume_with_state_minus_blind"]["gained"] == 6
    assert report["budget"]["by_arm"][S]["state_requests"] > 0 and report["budget"]["by_arm"][B]["state_requests"] == 0
    assert report["diagnostics"][S]["share_with_a_state"] == 1.0 and report["can_this_run_see_a_win"]["passes"] is True
    assert report["heldout"] == {"goal_set": 12, "rungs": {"below": 0, "in": 0, "above": 10}, "the_primarys_problems": 12, "problems_with_no_side_to_attempt": []}
    assert "up to 4 more attempts" in report["naming"] and report["attempts"][FIRST]["attempts"] == 22 and report["not_to_be_read"] == []
    # No loop was trimmed here: the reading without trimming is the same, and the report says both.
    assert report["trimmed"][S]["resolutions"] == 0 and report["without_trimming"]["primary"] == {key: value for key, value in report["primary"].items() if key != "what"}
    assert "trimmed resolutions (kept lines that verified on their own, nothing generated): 0 with the state (0 on these problems), 0 without it" in report["headline"]
    assert "(0 of its 4 successes trimmed)" in report["headline"] and "trimmed" in report["naming"]
    assert report["step"]["lean_checks_sent"] == 250 and report["model"].startswith("the base model")


def test_the_report_gives_the_trimmed_resolutions_apart_and_the_three_reads_without_them():
    rows = above(6, 4) + hard(6, 1)
    rows[0]["arms"][S] = trimmed(fall_back=(OK,))                   # above-band: resolved either way
    rows[10]["arms"][S] = trimmed()                                 # g0, which blind resolved at its next attempt: without trimming, never
    rows[10]["arms"][W] = trimmed()
    report = build_l3a_report(PREPARE, rows, STEP, {}, EVALUATION)
    assert report["trimmed"][S]["resolutions"] == 2 and report["trimmed"][S]["by_set"] == {"goal": 1, "below": 0, "in": 0, "above": 1, "hard": 1, "all": 2}
    assert report["trimmed"][W]["resolutions"] == 1 and report["trimmed"][S]["resolved_without_trimming_too"] == 1
    assert report["primary"]["mean"] == round(6 / 12, 5) and report["without_trimming"]["primary"]["mean"] == round(5 / 12, 5)
    assert set(report["without_trimming"]) == {"what", "primary", "single_step", "goal_set_by_problem"}
    assert report["single_step"]["hard"][S]["of_which_trimmed"] == 1 and report["without_trimming"]["single_step"]["hard"][S]["of_which_trimmed"] == 0
    assert report["goal_set_by_problem"]["resume_with_state_minus_blind"]["lost"] == 0 and report["without_trimming"]["goal_set_by_problem"]["resume_with_state_minus_blind"]["lost"] == 1
    for needed in ("trimmed resolutions (kept lines that verified on their own, nothing generated): 2 with the state (1 on these problems), 1 without it",
                   f"the primary without them: {round(5 / 12, 5)} [", "(1 of its 4 successes trimmed)"):
        assert needed in report["headline"], needed
    assert report["can_this_run_see_a_win"]["the_resumed_next_attempt_on_the_above_band_rung"]["of_which_trimmed"] == 1
    assert report["budget"]["by_arm"][S]["of_which_trimmed"] == 2 and report["budget"]["by_arm"][S]["kept_lines_checks"] == 2


def test_the_report_says_inconclusive_first_and_marks_counts_that_are_not_to_be_read():
    rows = above(6, 1) + hard(6, 1)
    for row in rows[:3]:
        row["arms"][B]["loops"][0]["status"] = "no_answer"
    report = build_l3a_report(PREPARE, rows, STEP, {}, EVALUATION)
    assert report["branch"]["name"] == INCONCLUSIVE and report["inconclusive"] is True and report["headline"].startswith("L3a seed 0: INCONCLUSIVE.")
    assert report["not_to_be_read"] == [B] and report["ok"] is False and "NOT TO BE READ: too many attempts without an answer in blind" in report["headline"]
    assert report["attempts"][B]["without_an_answer"] == 3
    # The sharper number beside the spec's check is said in the headline when the two disagree.
    rows = []
    for index in range(10):
        fell_back = {"resolved_at": 2, "loops": [step(1, OK, no_state="no_goals_at_the_cut")]}
        rows.append(episode(f"a{index}", "above", (OK,), fell_back if index < 5 else arm(*FOUR_BAD, how="resume", requested=True), FOUR_BAD))
    assert "BESIDE THE SPEC'S CHECK" in build_l3a_report(PREPARE, rows + hard(6, 1), STEP, {}, EVALUATION)["headline"]
