"""The second repair check's rules, read and report on small hand-made inputs. Spec: docs/spec/ladder-loop.spec.md,
"L3a2: one repair step after each fresh failure, then start over (no training)". Pure. What is done to one failed proof
is L3a's (`test_repair_cut.py`); the stage end to end is `test_ladder_l3a2_stage.py`."""

import pytest

from rlvr_lean.domain.ladder_round.rounds import sign_test
from rlvr_lean.domain.repair import BLIND, FIRST
from rlvr_lean.domain.repair import alternate as rules
from rlvr_lean.domain.repair import read
from rlvr_lean.domain.repair.alternate import ALTERNATE, ARMS, DID_NOT_REPEAT, KNOWN_COPY, REPAIR_ARMS, THE_EPISODE
from rlvr_lean.domain.repair.read import INCONCLUSIVE, NOT_SHOWN
from rlvr_lean.reporting.ladder_l3a2 import build_l3a2_report

LOOPS = 5
OK, BAD = "verified", "lean_error"
B, A = BLIND, ALTERNATE
FOUR_BAD = (BAD, BAD, BAD, BAD)


def step(loop, status, how="blind", tokens=100, prompt=300, kept=0, no_state=None, requested=False, cut=None, sent=None, kept_lines_check=None):
    return {"loop": loop, "how": how, "had_state": how == "resume", "no_state": no_state, "status": status, "token_count": tokens, "prompt_tokens": prompt,
            "sent_to_lean": status != KNOWN_COPY if sent is None else sent, "kept_lines": kept, "cut": cut, "repeats_failed_step": None,
            "state_requested": requested, "kept_lines_check": kept_lines_check}


def arm(steps):
    at = next((entry["loop"] + 1 for entry in steps if entry["status"] == OK), None)
    return {"resolved_at": at, "loops": steps, "trimmed_at": None, "without_trimming": None}


def blind(*statuses, tokens=100):
    """The blind arm after a failed first attempt: whole proofs, to its first verified attempt."""
    return arm([step(index + 1, status, tokens=tokens) for index, status in enumerate(statuses)])


def alternating(*statuses, tokens=100):
    """The alternate arm after a failed first attempt: attempts 2 and 4 (loops 1 and 3) are repair steps whose prompt
    held a state, attempts 3 and 5 whole proofs."""
    return arm([step(index + 1, status, tokens=tokens, **({"how": "resume", "requested": True, "cut": "body"} if rules.is_repair_step(index + 1) else {}))
                for index, status in enumerate(statuses)])


def trimmed(at=1, fall_back=FOUR_BAD):
    """The alternate arm whose repair step at loop `at` (1 or 3) is TRIMMED: no goal was left at the cut and the kept
    lines verified. `fall_back`: what that loop and the ones after it were in the reading without trimming."""
    earlier = alternating(*(BAD,) * (at - 1))["loops"]
    own = step(at, OK, how="trimmed", tokens=0, prompt=0, kept=2, requested=True, cut="body", kept_lines_check="verified")
    others = [step(at + index, status, no_state="no_goals_at_the_cut" if index == 0 else None, requested=index == 0) for index, status in enumerate(fall_back)][:LOOPS - at]
    without = next((entry["loop"] + 1 for entry in others if entry["status"] == OK), None)
    return {"resolved_at": at + 1, "loops": [*earlier, own], "trimmed_at": at, "without_trimming": {"resolved_at": without, "loops": [*earlier, *others]}}


def episode(problem, group, blind_arm=FOUR_BAD, alternate_arm=FOUR_BAD, number=0, first=BAD):
    """One episode. With a verified first attempt the arms have nothing to do; else each arm's statuses, loop by loop."""
    if first == OK:
        arms = {name: {"resolved_at": 1, "loops": [], "trimmed_at": None, "without_trimming": None} for name in ARMS}
    else:
        arms = {B: blind(*blind_arm) if not isinstance(blind_arm, dict) else blind_arm,
                A: alternating(*alternate_arm) if not isinstance(alternate_arm, dict) else alternate_arm}
    return {"problem_id": problem, "group": group, "side": "true", "episode": number, "failed_first": first != OK,
            "first": {"status": first, "token_count": 90, "prompt_tokens": 200, "sent_to_lean": True}, "arms": arms}


# --------------------------------------------------------------------------------------------- the episode
def test_attempts_2_and_4_are_the_repair_steps_and_the_others_start_over():
    assert [rules.is_repair_step(loop) for loop in range(LOOPS)] == [False, True, False, True, False]        # attempt n is loop n - 1
    assert ARMS == (BLIND, ALTERNATE) and REPAIR_ARMS == (ALTERNATE,)


def attempt(arm_name, problem, number, loop, status, proof="  simp\n", **more):
    return {"arm": arm_name, "problem_id": problem, "episode": number, "loop": loop, "status": status, "audit": False, "how": "blind", "had_state": False,
            "no_state": None, "kept_lines": 0, "cut": None, "repeats_failed_step": None, "token_count": 50, "prompt_tokens": 120,
            "sent_to_lean": status not in ("capped_tokens", "rejected_lexical", KNOWN_COPY), "proof": proof, **more}


def test_a_proof_is_a_known_copy_when_lean_already_rejected_that_text_in_the_episode():
    chain = [attempt(FIRST, "p", 0, 0, BAD, "  linarith\n"), attempt(A, "p", 0, 1, "timeout", "  nlinarith [sq_nonneg x]\n"),
             attempt(A, "p", 0, 2, "capped_tokens", "  simp\n" * 300), attempt(A, "p", 0, 3, "no_answer", "  omega\n"),
             attempt(A, "p", 0, 4, BAD, "  linarith\n\n")]
    already = rules.rejected(chain)
    # Sent and failed: an error, the Lean limit. Not the capped attempt (never sent), not the one Lean gave no answer for.
    assert already == {"  linarith": 0, "  nlinarith [sq_nonneg x]": 1}                               # the earliest attempt that wrote it
    assert rules.copy_of("  linarith\n", already) == 0 and rules.copy_of("  nlinarith [sq_nonneg x]", already) == 1
    # The text is the proof as it stands in the checked file: the white space at its end is dropped, nothing else.
    assert rules.copy_of("  linarith  \n\n", already) == 0 and rules.checked_text("  linarith \n") == "  linarith"
    assert rules.copy_of(" linarith\n", already) is None and rules.copy_of("\n  linarith\n", already) is None
    assert rules.copy_of("  omega\n", already) is None and rules.copy_of("  simp\n" * 300, already) is None
    assert rules.copy_of("  linarith\n", None) is None and rules.copy_of("  linarith\n", {}) is None       # a check that sends every proof
    # A known copy was not sent, and a verified proof was not rejected: neither adds to what Lean has rejected.
    assert rules.rejected([attempt(B, "p", 0, 1, KNOWN_COPY, "  ring\n"), attempt(B, "p", 0, 2, OK, "  norm_num\n"), attempt(B, "p", 0, 3, "rejected_lexical", "  sorry\n")]) == {}
    assert all(rules.is_rejected(attempt(B, "p", 0, 1, status)) for status in (BAD, "timeout", "uses_sorry", "forbidden_axiom"))


def test_the_episode_rows_hold_both_arms_and_a_repair_step_is_read_with_the_request_it_started_from():
    problems = [{"problem_id": "p", "group": "below", "side": "true"}]
    resume = {"how": "resume", "had_state": True, "kept_lines": 2, "cut": "body", "repeats_failed_step": False}
    rows = [attempt(FIRST, "p", 0, 0, BAD), attempt(FIRST, "p", 1, 0, OK), *(attempt(B, "p", 0, loop, BAD if loop != 2 else KNOWN_COPY) for loop in (1, 2, 3, 4)),
            attempt(A, "p", 0, 1, BAD, **resume), attempt(A, "p", 0, 2, KNOWN_COPY), attempt(A, "p", 0, 3, BAD, no_state=KNOWN_COPY), attempt(A, "p", 0, 4, OK)]
    states = [{"arm": FIRST, "problem_id": "p", "episode": 0, "after_loop": 0, "requested": True, "outcome": "state", "trim_status": None},
              {"arm": A, "problem_id": "p", "episode": 0, "after_loop": 2, "requested": False, "outcome": KNOWN_COPY, "trim_status": None}]
    failed, passed = read.episode_rows(problems, rows, states, LOOPS, ARMS, REPAIR_ARMS)
    assert set(failed["arms"]) == set(ARMS) and {name: failed["arms"][name]["resolved_at"] for name in ARMS} == {B: None, A: 5}
    assert all(passed["arms"][name] == {"resolved_at": 1, "loops": [], "trimmed_at": None, "without_trimming": None} for name in ARMS)
    own = failed["arms"][A]["loops"]
    assert [(entry["loop"], entry["how"], entry["status"]) for entry in own] == [(1, "resume", BAD), (2, "blind", KNOWN_COPY), (3, "blind", BAD), (4, "blind", OK)]
    # Attempt 2 started from the SHARED first attempt's request; attempt 4 from the one made after attempt 3, which was a
    # known copy (no file was sent for it); attempts 3 and 5 start from nothing.
    assert [entry["state_requested"] for entry in own] == [True, False, False, False] and own[2]["no_state"] == KNOWN_COPY
    assert [entry["sent_to_lean"] for entry in own] == [True, False, True, True]
    assert not any(entry["state_requested"] for entry in failed["arms"][B]["loops"])
    assert read.problem_rows([failed, passed], LOOPS, ARMS, REPAIR_ARMS) == [{"problem_id": "p", "group": "below", "side": "true", "episodes": 2, "failed_first": 1,
                                                                             f"resolved_{B}": 1, f"resolved_{A}": 2, f"trimmed_{A}": 0}]
    with pytest.raises(ValueError, match="whole episodes"):
        read.episode_rows(problems, [row for row in rows if not (row["arm"] == A and row["loop"] == 4)], states, LOOPS, ARMS, REPAIR_ARMS)


# ------------------------------------------------------------------------------------------------ the read
def mixed():
    return [episode("g1", "goal", FOUR_BAD, (OK,)), episode("g1", "goal", (OK,), FOUR_BAD, number=1), episode("g1", "goal", first=OK, number=2),
            episode("g1", "goal", number=3),
            episode("b1", "below", FOUR_BAD, (BAD, BAD, OK)), episode("b1", "below", FOUR_BAD, (BAD, BAD, BAD, OK), number=1),
            episode("i1", "in", (OK,), FOUR_BAD), episode("a1", "above", (OK,), (OK,))]


def test_the_primary_is_alternate_minus_blind_on_the_hard_problems_paired_by_episode():
    found = rules.alternate_read(mixed(), LOOPS, 500, 0)
    primary = found["primary"]
    # g1: (1 - 0, 0 - 1, 1 - 1, 0 - 0) / 4 = 0; b1: (1 + 1) / 2 = 1. The mean over the two problems is 0.5.
    assert (primary["problems"], primary["episodes"], primary["mean"]) == (2, 6, 0.5) and primary[A] == 0.75 and primary[B] == 0.25
    assert 0.0 <= primary["low"] <= primary["mean"] <= primary["high"] <= 1.0
    assert primary == read.paired(read._of(mixed(), ("goal", "below")), A, B, LOOPS, 500, 0)             # L3a's pairing, resamples and generator


def test_the_secondaries_are_the_same_read_within_fewer_attempts_and_by_set():
    found = rules.alternate_read(mixed(), LOOPS, 500, 0)
    within = found["within_attempts"]
    assert set(within) == {"goal", "below", "in", "above", "hard", "all"} and all(set(by) == {"2", "3", "4", "5"} for by in within.values())
    assert within["hard"]["5"] == found["primary"]
    # Within 2 attempts b1's repairs at attempts 4 and 5 are not there yet, and g1's two cancel; within 4 one of b1's is.
    assert [within["hard"][count]["mean"] for count in ("2", "3", "4")] == [0.0, 0.0, 0.25]
    assert within["goal"]["5"]["mean"] == 0.0 and within["below"]["5"]["mean"] == 1.0                  # G alone, the below-band rung alone
    assert within["in"]["5"]["mean"] == -1.0 and within["above"]["5"]["mean"] == 0.0 and within["above"]["5"]["problems"] == 1
    resolved = found["resolved_within_attempts"]["hard"]
    assert set(resolved) == set(ARMS) and resolved[A] == {"1": 0.125, "2": 0.25, "3": 0.25, "4": 0.5, "5": 0.75} and resolved[B]["5"] == 0.25


def test_the_goal_set_is_read_by_problem_as_gained_against_lost_with_the_sign_test():
    rows = [episode("g1", "goal", FOUR_BAD, (OK,)), episode("g1", "goal", number=1), episode("g2", "goal", (BAD, OK), FOUR_BAD),
            episode("g3", "goal", (OK,), (OK,)), episode("g4", "goal", FOUR_BAD, (BAD, BAD, BAD, OK)), episode("g5", "goal"), episode("b1", "below", FOUR_BAD, (OK,))]
    counted = rules.alternate_read(rows, LOOPS, 200, 0)["goal_set_by_problem"]
    assert counted == {"problems": 5, f"resolved_{A}": 3, f"resolved_{B}": 2, "by_both": 1, "gained": 2, "lost": 1, "sign_test_p": sign_test(2, 1)}


def test_the_verified_share_is_given_by_attempt_position_in_each_arm():
    fell_back = arm([step(1, OK, no_state="no_error_position"), ])
    rows = mixed() + [episode("g2", "goal", (BAD, KNOWN_COPY, BAD, BAD), fell_back), episode("g3", "goal", FOUR_BAD, trimmed(at=3))]
    found = rules.by_position(read._of(rows, ("goal", "below")), LOOPS)
    assert found["first_attempts"] == {"attempts": 8, "verified": 1, "share": 0.125}
    # Blind: of the 7 failed first attempts one verifies at position 2; the 6 others go on, and none verifies after.
    assert found[B] == {"2": {"attempts": 7, "verified": 1, "share": 0.14286, "known_copies": 0}, "3": {"attempts": 6, "verified": 0, "share": 0.0, "known_copies": 1},
                        "4": {"attempts": 6, "verified": 0, "share": 0.0, "known_copies": 0}, "5": {"attempts": 6, "verified": 0, "share": 0.0, "known_copies": 0}}
    # Alternate, position 2 (a repair step): g1's first episode verifies with a state; g2's had no state and verified as a blind attempt.
    assert found[A]["2"] == {"attempts": 7, "verified": 2, "share": 0.28571, "known_copies": 0, "repair_step": True, "with_a_state": 6, "verified_with_a_state": 1,
                             "of_which_trimmed": 0, "blind_for_want_of_a_state": 1}
    assert found[A]["3"] == {"attempts": 5, "verified": 0, "share": 0.0, "known_copies": 0}                 # a whole proof: no repair-step keys
    # Position 4 (the second repair step): b1's first episode verifies with a state, g3's is trimmed.
    assert found[A]["4"] == {"attempts": 5, "verified": 2, "share": 0.4, "known_copies": 0, "repair_step": True, "with_a_state": 4, "verified_with_a_state": 1,
                             "of_which_trimmed": 1, "blind_for_want_of_a_state": 0}
    assert found[A]["5"]["attempts"] == 3 and found[A]["5"]["verified"] == 1 and "repair_step" not in found[A]["5"]
    assert set(rules.alternate_read(rows, LOOPS, 100, 0)["by_position"]) == {"goal", "below", "in", "above", "hard", "all"}


def test_the_trimmed_resolutions_are_reported_apart_and_the_primary_is_read_without_them():
    rows = [episode("g1", "goal", FOUR_BAD, trimmed()),                                    # resolved by trimming only: without it, never
            episode("g1", "goal", number=1),
            episode("g2", "goal", FOUR_BAD, trimmed(fall_back=(BAD, OK))),                 # resolved either way, an attempt later without it
            episode("b1", "below", FOUR_BAD, trimmed(at=3, fall_back=(BAD, BAD))),         # the second repair step is the trimmed one
            episode("a1", "above", (OK,), trimmed(fall_back=(OK,)))]
    found = rules.alternate_read(rows, LOOPS, 500, 0)
    # With them (the spec's primary): g1 (1 + 0) / 2, g2 1, b1 1. Without: g1 0, g2 1, b1 0.
    assert found["primary"]["mean"] == round(2.5 / 3, 5) and found["primary"][B] == 0.0
    without = found["without_trimming"]
    assert set(without) == {"primary"} and without["primary"]["mean"] == round(1 / 3, 5) and without["primary"]["problems"] == 3
    assert found["trimmed"] == {"resolutions": 4, "by_set": {"goal": 2, "below": 1, "in": 0, "above": 1, "hard": 3, "all": 4},
                                "by_loop": {"1": 3, "3": 1}, "resolved_without_trimming_too": 2}
    # A trimmed step generates nothing and costs the state request and the kept-lines check.
    spent = rules.arm_spend(read._of(rows, ("goal",)), A)
    assert spent == {"attempts": 6, "of_which_trimmed": 2, "generated_tokens": 400, "prompt_tokens": 1200, "attempts_sent_to_lean": 4, "state_requests": 4,
                     "kept_lines_checks": 2, "lean_checks": 10, "known_copies": 0}
    # The position-2 check counts a trimmed step as the resolved step it is, and says how many of its successes were trimmed.
    check = found["can_this_run_see_a_win"]["the_repair_step_at_position_2_on_the_hard_problems"]
    assert (check["repair_step_verified"], check["of_which_trimmed"], check["with_a_state"], check["verified_with_a_state"]) == (2, 2, 2, 0)


# -------------------------------------------------------------------------------------------------- budget
def test_the_budget_counts_tokens_lean_checks_and_known_copies_by_arm():
    copies = arm([step(1, KNOWN_COPY, how="resume", requested=True, cut="body", tokens=40), step(2, KNOWN_COPY, tokens=80),
                  step(3, BAD, no_state=KNOWN_COPY, tokens=80), step(4, BAD, tokens=80)])
    rows = [episode("g1", "goal", (BAD, KNOWN_COPY, "capped_tokens", BAD), copies), episode("a1", "above", first=OK)]
    rows[0]["arms"][B]["loops"][2]["sent_to_lean"] = False                                  # the capped attempt was not sent either
    spent = rules.budget(rows, read._of(rows, ("goal",)), LOOPS, 200, 0)
    assert spent["first_attempts"] == {"attempts": 2, "generated_tokens": 180, "prompt_tokens": 400, "lean_checks": 2}
    # A known copy is an attempt whose tokens are counted and which is not a Lean check.
    assert spent["by_arm"][B] == {"attempts": 4, "of_which_trimmed": 0, "generated_tokens": 400, "prompt_tokens": 1200, "attempts_sent_to_lean": 2,
                                  "state_requests": 0, "kept_lines_checks": 0, "lean_checks": 2, "known_copies": 1}
    # The repair step at attempt 2 asked for a state (one more check) and wrote a known copy; attempt 4 would have started
    # from a known copy, so no state was asked for it.
    assert spent["by_arm"][A] == {"attempts": 4, "of_which_trimmed": 0, "generated_tokens": 280, "prompt_tokens": 1200, "attempts_sent_to_lean": 2,
                                  "state_requests": 1, "kept_lines_checks": 0, "lean_checks": 3, "known_copies": 2}
    assert spent["by_arm_on_the_primarys_problems"] == spent["by_arm"]
    assert spent["generated_tokens_alternate_over_blind_on_the_primarys_problems"] == 0.7 and spent["lean_checks_alternate_over_blind_on_the_primarys_problems"] == 1.5
    first_loop = rules.arm_spend(rows, A, through=1)                                         # the arm read at its first attempt after the first alone
    assert (first_loop["known_copies"], first_loop["generated_tokens"], first_loop["attempts"]) == (1, 40, 1) and rules.arm_spend(rows, A, through=2)["known_copies"] == 2


def test_the_read_at_equal_tokens_is_made_only_when_the_alternate_arm_generates_more_than_ten_percent_more():
    def rows(alternate_tokens):
        return [episode("g1", "goal", FOUR_BAD, alternating(BAD, BAD, BAD, OK, tokens=alternate_tokens)),
                episode("b1", "below", (BAD, BAD, BAD, OK), alternating(OK, tokens=alternate_tokens))]
    same = rules.alternate_read(rows(110), LOOPS, 200, 0)["budget"]
    assert same["generated_tokens_alternate_over_blind_on_the_primarys_problems"] == 0.6875 and same["alternate_generates_more_than_10_percent_more"] is False
    assert isinstance(same["equal_tokens"], str) and "read as it stands" in same["equal_tokens"]
    more = rules.alternate_read(rows(300), LOOPS, 200, 0)["budget"]            # 5 x 300 against 8 x 100 tokens
    assert more["generated_tokens_alternate_over_blind_on_the_primarys_problems"] == 1.875 and more["alternate_generates_more_than_10_percent_more"] is True
    equal = more["equal_tokens"]
    assert equal["blind_attempts_after_the_first_that_match"] == 7.5 and equal["blind_attempts_after_the_first_that_were_sampled"] == 4
    assert equal["the_blind_arm_at_the_matching_number_can_be_read"] is False
    # Its mirror, as in L3a: the alternate arm at the loops whose tokens match the blind arm's 800 (loop 1 is 600 tokens, loop 2 makes 900).
    assert equal["mirror"]["alternate_loops_read"] == 1 and equal["mirror"]["alternate_generated_tokens"] == 600 and equal["mirror"]["blind_generated_tokens"] == 800
    assert equal["mirror"]["primary"]["mean"] == 0.0 and equal["mirror"]["primary"][A] == 0.5 and equal["mirror"]["primary"][B] == 0.5


# ------------------------------------------------------------------------------- can this run see a win
def hard(alternate_wins, blind_wins, total=12, second_blind=2, second_repair=1):
    """`total` goal problems, one episode each. The blind arm's attempt at position 2 verifies on the first
    `second_blind` and the repair step at position 2 on the first `second_repair`; the blind arm resolves the next
    `blind_wins` at its attempt 5, and the alternate arm the LAST `alternate_wins` at its own attempt 5."""
    rows = []
    for index in range(total):
        late_blind = second_blind <= index < second_blind + blind_wins
        late_alternate = total - index <= alternate_wins and index >= second_repair
        rows.append(episode(f"g{index}", "goal", (OK,) if index < second_blind else (BAD, BAD, BAD, OK) if late_blind else FOUR_BAD,
                            (OK,) if index < second_repair else (BAD, BAD, BAD, OK) if late_alternate else FOUR_BAD))
    return rows


def test_both_checks_must_pass_and_either_failing_makes_the_run_inconclusive():
    fine = rules.alternate_read(hard(8, 0), LOOPS, 500, 0)                                  # position 2: the repair step 1 of 12, blind 2 of 12: exactly half
    checks = fine["can_this_run_see_a_win"]
    assert checks["passes"] is True and checks["the_repair_step_at_position_2_on_the_hard_problems"] == {
        "failed_first_attempts": 12, "blind_verified": 2, "blind_share_verified": 0.16667, "repair_step_verified": 1, "repair_step_share_verified": 0.08333,
        "of_which_trimmed": 0, "with_a_state": 12, "verified_with_a_state": 1, "needed": "at least half of the blind arm's share", "passes": True}
    states = checks["a_state_for_the_failed_attempts_a_repair_step_starts_from"]
    assert (states["repair_steps"], states["with_a_state"], states["share"], states["passes"]) == (12 + 11, 23, 1.0, True) and states["without_a_state_by_reason"] == {}
    assert fine["branch"]["name"] == THE_EPISODE

    broken_prompt = rules.alternate_read(hard(8, 0, second_blind=3), LOOPS, 500, 0)       # 1 of 12 against 3 of 12: less than half
    assert broken_prompt["can_this_run_see_a_win"]["the_repair_step_at_position_2_on_the_hard_problems"]["passes"] is False
    assert broken_prompt["branch"]["name"] == INCONCLUSIVE and "this build broke the prompt or the cut" in broken_prompt["branch"]["reason"]
    assert "at position 2" in broken_prompt["branch"]["reason"] and "not a verdict" in broken_prompt["branch"]["reason"]

    # A state for fewer than half of the failed attempts a repair step starts from: both repair steps count, by reason.
    rows = hard(8, 0)
    for row in rows[2:]:
        for entry in row["arms"][A]["loops"]:
            if rules.is_repair_step(entry["loop"]):
                entry.update({"how": "blind", "had_state": False, "no_state": "no_error_position" if entry["loop"] == 1 else KNOWN_COPY})
    broken_cut = rules.alternate_read(rows, LOOPS, 500, 0)
    states = broken_cut["can_this_run_see_a_win"]["a_state_for_the_failed_attempts_a_repair_step_starts_from"]
    assert (states["repair_steps"], states["with_a_state"], states["share"], states["passes"]) == (23, 3, 0.13043, False)
    assert states["without_a_state_by_reason"] == {"no_error_position": 10, KNOWN_COPY: 10}
    assert broken_cut["branch"]["name"] == INCONCLUSIVE and "a state came back for 0.13043 of the failed attempts a repair step starts from" in broken_cut["branch"]["reason"]

    # A check that cannot be read has not passed: no hard problem, or a blind arm that verified none at position 2.
    assert rules.alternate_read([episode("a1", "above", (OK,), (OK,))], LOOPS, 200, 0)["branch"]["name"] == INCONCLUSIVE
    assert rules.alternate_read(hard(8, 0, second_blind=0, second_repair=0), LOOPS, 200, 0)["can_this_run_see_a_win"]["passes"] is False
    assert rules.can_see_a_win([])["passes"] is False


def test_the_branch_is_read_from_the_primarys_interval_in_the_specs_words():
    won = rules.alternate_read(hard(8, 0), LOOPS, 500, 0)["branch"]
    assert won["name"] == THE_EPISODE and "this is the episode a round should use" in won["reason"] and "L3b" in won["reason"] and "needs a decision, with the numbers" in won["reason"]
    lost = rules.alternate_read(hard(0, 8), LOOPS, 500, 0)["branch"]
    assert lost["name"] == DID_NOT_REPEAT and "L3a's single step did not repeat on fresh attempts" in lost["reason"]
    level = rules.alternate_read(hard(3, 3, second_repair=2), LOOPS, 500, 0)
    assert level["primary"]["mean"] == 0.0 and level["primary"]["low"] < 0 < level["primary"]["high"]
    assert level["branch"]["name"] == NOT_SHOWN and "not worth its extra Lean check" in level["branch"]["reason"] and "which needs a decision" in level["branch"]["reason"]
    checks = {"passes": True}
    assert rules.alternate_branch({"mean": 0.02, "low": 0.0, "high": 0.05}, checks).name == NOT_SHOWN       # an interval that touches zero is not clear of it
    assert rules.alternate_branch({"mean": 0.02, "low": 0.001, "high": 0.05}, checks).name == THE_EPISODE
    assert rules.alternate_branch({"mean": -0.02, "low": -0.05, "high": 0.0}, checks).name == NOT_SHOWN
    assert rules.alternate_branch({"mean": -0.02, "low": -0.05, "high": -0.001}, checks).name == DID_NOT_REPEAT
    assert rules.alternate_branch({"mean": None, "low": None, "high": None}, checks).name == INCONCLUSIVE
    assert set(rules.BRANCHES) == {THE_EPISODE, NOT_SHOWN, DID_NOT_REPEAT, INCONCLUSIVE}


# -------------------------------------------------------------------------------------------------- report
PREPARE = {"seed": 0, "fixture": False, "stand_in_engine": False, "goal_set": 12, "rungs": {"below": 0, "in": 1, "above": 1},
           "sizes": {"episodes": {"goal": 12, "below": 12, "in": 2, "above": 2}, "episodes_from": "repair_alternate", "loops": LOOPS, "max_new_tokens": 1024,
                     "lean_seconds": 30, "sampling_seed": 1031},
           "problems_with_no_side_to_attempt": []}
STEP = {"problems": 14, "episodes": 14, "attempts": 120, "generated_tokens": 1000, "lean_checks_sent": 150, "loops": [], "stand_in_engine": False}
EVALUATION = {"bootstrap_resamples": 300, "bootstrap_seed": 0}
EASIER = [episode("i1", "in", (OK,), FOUR_BAD), episode("a1", "above", (OK,), (OK,))]


def test_the_report_answers_the_read_and_names_the_branch():
    report = build_l3a2_report(PREPARE, hard(8, 1) + EASIER, STEP, {}, EVALUATION)
    assert report["branch"]["name"] == THE_EPISODE and report["inconclusive"] is False and report["ok"] is True and report["seed"] == 0
    assert report["headline"].startswith("L3a2 seed 0: THIS IS THE EPISODE A ROUND SHOULD USE. Primary (G and the below-band rung, 12 problems, 12 episodes "
                                         "resolved within 5 attempts, alternate minus blind): ")
    # Of the 12: the blind arm resolves three (two at attempt 2, one at attempt 5) and the alternate arm nine (one at attempt 2, eight at 5).
    for needed in ("trimmed resolutions (kept lines that verified on their own, nothing generated): 0 (0 on these problems); the primary without them: 0.5 [",
                   "within 2 attempts -0.08333 [", "within 3 attempts -0.08333 [", "within 4 attempts -0.08333 [", "G alone 0.5 [",
                   "the below-band rung alone not measured", "the in-band rung -1.0 [", "the above-band rung 0.0 [",
                   "G by problem (12), alternate against blind: 9 to 3, gained 8, lost 2",
                   "verified by attempt position on these problems (2 and 4 are the repair steps), alternate: 2: 1/12, 3: 0/11, 4: 0/11, 5: 8/11; blind: 2: 2/12, 3: 0/10, 4: 0/10, 5: 1/10",
                   "a state for 1.0 of the failed attempts a repair step starts from", "known copies not sent to Lean: alternate 0, blind 0",
                   "generated tokens, alternate over blind: 1.0714", "Lean checks, alternate over blind: 1.619;",      # 45 attempts against 42; 23 state requests
                   "position-2 check: the repair step 0.08333 against blind 0.16667 (0 of its 1 successes trimmed)"):
        assert needed in report["headline"], needed
    assert report["primary"]["mean"] == 0.5 and "paired by episode" in report["primary"]["what"] and "alternate minus blind" in report["primary"]["what"]
    assert report["within_attempts"]["hard"]["5"] == {key: value for key, value in report["primary"].items() if key != "what"}
    assert report["goal_set_by_problem"]["gained"] == 8 and report["by_position"]["hard"][A]["2"]["repair_step"] is True
    assert report["budget"]["by_arm"][A]["state_requests"] > 0 and report["budget"]["by_arm"][B]["state_requests"] == 0 and "known_copies" in report["budget"]["by_arm"][B]
    assert report["can_this_run_see_a_win"]["passes"] is True and report["trimmed"]["resolutions"] == 0
    assert report["without_trimming"]["primary"] == {key: value for key, value in report["primary"].items() if key != "what"}
    assert report["heldout"] == {"goal_set": 12, "rungs": {"below": 0, "in": 1, "above": 1}, "the_primarys_problems": 12, "problems_with_no_side_to_attempt": []}
    assert "up to 4 more attempts" in report["naming"] and "known copy" in report["naming"] and report["attempts"][FIRST]["attempts"] == 14
    assert report["sizes"]["sampling_seed"] == 1031 and report["step"]["lean_checks_sent"] == 150 and report["model"].startswith("the base model")
    assert report["not_to_be_read"] == [] and "L3a2" in report["spec"]


def test_the_report_says_inconclusive_first_and_marks_counts_that_are_not_to_be_read():
    rows = hard(8, 1, second_blind=3) + EASIER
    for row in rows[3:6]:
        row["arms"][B]["loops"][0]["status"] = "no_answer"
    report = build_l3a2_report(PREPARE, rows, STEP, {}, EVALUATION)
    assert report["branch"]["name"] == INCONCLUSIVE and report["inconclusive"] is True and report["headline"].startswith("L3a2 seed 0: INCONCLUSIVE.")
    assert report["not_to_be_read"] == [B] and report["ok"] is False and "NOT TO BE READ: too many attempts without an answer in blind" in report["headline"]
    assert report["attempts"][B]["without_an_answer"] == 3
    # The trimmed steps are reported apart, and the headline gives the primary without them.
    rows = hard(8, 1) + EASIER
    rows[11]["arms"][A] = trimmed()
    report = build_l3a2_report(PREPARE, rows, STEP, {}, EVALUATION)
    assert report["trimmed"]["resolutions"] == 1 and report["trimmed"]["by_loop"] == {"1": 1} and report["primary"]["mean"] == 0.5
    assert report["without_trimming"]["primary"]["mean"] == round(5 / 12, 5)         # that episode's fall-back never verified
    assert f"nothing generated): 1 (1 on these problems); the primary without them: {round(5 / 12, 5)} [" in report["headline"]
    assert "(1 of its 2 successes trimmed)" in report["headline"]
