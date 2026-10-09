"""L4b on hand-made rows with a known answer: the three training rules inside the rounds (the old rule left as it is;
`reward_rows` and `rehearse` nested across the rounds; the twin; the kept share), the fifth check with rehearsal rows,
the breadth beside the primary, and the report with every branch, NARROWER, the rank-16 arm beside and the never-solved
ids. Spec: docs/spec/ladder-loop.spec.md, "L4b: the arm again, from the larger pretrained model". Pure: no model,
no Lean. The stage end to end is `test_ladder_l4b_stage.py`."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l3d1 import EVALUATION, GOAL_IDS, GROUPS, LENGTHS, _model  # noqa: E402 - Step 1's hand-made world: 12 goal problems, 3 a length group
from test_ladder_l4 import CHANGED, MEASURED, OF_THE_PRETRAINING, ON_G, RUNGS_FINE, SETTINGS, _made  # noqa: E402 - L4's hand-made `pre`, `with` and `without`
from test_ladder_l4_report import ARM_PREPARE, BASE_ARM, OWN, PREPARE, _report  # noqa: E402 - L4's own report on those rows

from rlvr_lean.domain.ladder_round.assembly import attempts_alone, training_order, training_set  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import ADDS, COSTS, INCONCLUSIVE, NOT_READ, NOT_SHOWN, PRETRAIN, arm_checks, half_of, l4_branch, pretraining_checks  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows import THE_MINIMUM, THE_RULE, rehearsal_rows, rehearse_set, smallest_decisive_split, uniform  # noqa: E402
from rlvr_lean.domain.ladder_round.l4b import NARROWER, OLD, ON_FEWER, REHEARSE, REWARD_ROWS, RULES, breadth_beside, by_origin, kept_share, record, rule_set, twin_of, with_breadth  # noqa: E402
from rlvr_lean.domain.ladder_round.rounds import sign_test  # noqa: E402
from rlvr_lean.domain.problem_pool.episodes import reward  # noqa: E402
from rlvr_lean.reporting.ladder_l4 import LABEL, SAY, build_l4_report  # noqa: E402
from rlvr_lean.reporting.ladder_l4b import build_l4b_report, of_the_rank_16_arm, the_rule_by_round  # noqa: E402

SEED, TARGET = 0, 0.10
PRETRAIN_HALF = [f"p{index}" for index in range(900) if half_of(f"p{index}", 0) == PRETRAIN]
LOOP_HALF = [f"p{index}" for index in range(900) if half_of(f"p{index}", 0) != PRETRAIN]      # problems a round of the arm may train on


def _pick(problem, k, assembled=False):
    return {"problem_id": problem, "side": "true", "episodes": 8, "resolved": k, "resolved_by_assembly": assembled}


def _row(problem, origin, number, lines=2):
    attempt = f"{problem}#round_r{number}_b1#statement#3" if origin == "attempt" else None
    return {"id": attempt or f"{problem}#assembled#r{number}", "problem_id": problem, "side": "statement", "attempt_id": attempt, "theorem": f"theorem {problem} : P := by\n",
            "completion": "".join(f"  step_{line}\n" for line in range(lines)), "verified_attempts": 2, "round": number, "batch": 1, "origin": origin}


# Three rounds of 40 picks: k cycles through 1 to 7 for the problems an attempt resolved (28 a round); every fifth problem only assembly resolved (8); four nothing did.
PROBLEMS = {number: LOOP_HALF[(number - 1) * 40:number * 40] for number in (1, 2, 3)}
K_OF = {problem: (0 if index % 8 == 7 or index % 5 == 4 else 1 + index % 8) for number in PROBLEMS for index, problem in enumerate(PROBLEMS[number])}
ASSEMBLED = {problem for number in PROBLEMS for index, problem in enumerate(PROBLEMS[number]) if index % 5 == 4}
PICKS = {number: [_pick(problem, K_OF[problem], problem in ASSEMBLED) for problem in PROBLEMS[number]] for number in PROBLEMS}
BY_ROUND = {number: [_row(problem, "assembled" if problem in ASSEMBLED else "attempt", number, lines=1 + index % 9) for index, problem in enumerate(PROBLEMS[number])
                     if K_OF[problem] > 0 or problem in ASSEMBLED] for number in PROBLEMS}
PRETRAINING = [{"problem_id": problem, "kind": "stp_conjecture", "proof_lines": 3 + index % 7, "lines": 3 + index % 7} for index, problem in enumerate(PRETRAIN_HALF[:300])]


def _picks(number):
    return [row for earlier in PICKS if earlier <= number for row in PICKS[earlier]]


def _set(rule, number, seed=SEED, at_least=0):
    return rule_set(BY_ROUND, _picks(number), number, rule, TARGET, seed, PRETRAINING, at_least)


def _k(row):
    return 1 if row["origin"] == "assembled" else K_OF[row["problem_id"]]


# ------------------------------------------------------------------------------------------- the three rules
def test_the_old_rule_is_what_every_arm_does_and_an_unknown_rule_is_refused():
    assert RULES == (OLD, REWARD_ROWS, REHEARSE) == ("old", "reward_rows", "rehearse")
    for number in (1, 2, 3):
        assert _set(OLD, number) == training_order(training_set(BY_ROUND, number), SEED)            # the same rows, the same order, nothing added to a row
    assert all("k" not in row and "lines" not in row for row in _set(OLD, 3)) and len(_set(OLD, 3)) == sum(len(rows) for rows in BY_ROUND.values()) == 3 * 36
    with pytest.raises(ValueError, match="'newest' is not a training rule: an arm's rule is one of .'old', 'reward_rows', 'rehearse'."):
        rule_set(BY_ROUND, _picks(1), 1, "newest", TARGET, SEED)
    # A rule gives a k to a round's own rows and to no other: an H0 row is refused, as L4t refuses it.
    with_h0 = {1: [*BY_ROUND[1], {**_row("elsewhere", "assembled", 1), "id": "elsewhere#h0", "origin": "h0"}]}
    for rule in (REWARD_ROWS, REHEARSE):
        with pytest.raises(ValueError, match="the row elsewhere#h0 is of origin 'h0': the rule gives a k to a round's one-shot and assembled rows, and to no other"):
            rule_set(with_h0, _picks(1), 1, rule, TARGET, SEED, PRETRAINING)


def test_reward_rows_keeps_a_row_by_a_hash_of_the_seed_and_its_id_against_r_of_k_so_a_row_kept_for_a_model_is_kept_for_every_later_one():
    sets = {number: _set(REWARD_ROWS, number) for number in (1, 2, 3)}
    every = training_order(training_set(BY_ROUND, 3), SEED)
    kept = {number: {row["id"] for row in sets[number]} for number in sets}
    # THE RULE, row by row: the draw is under the challenger's reward at the arm's target rate for the problem's k of 8 (an assembled row: k = 1).
    for row in every:
        draw, paid = uniform(SEED, "l4_rows_keep", row["id"]), reward(_k(row), 8, TARGET)
        assert (row["id"] in kept[3]) == (draw < paid)
    for row in sets[3]:
        assert (row["k"], row["n"], row["reward"], row["draw"], row["kept_by"]) == (_k(row), 8, round(reward(_k(row), 8, TARGET), 6), uniform(SEED, "l4_rows_keep", row["id"]), THE_RULE)
    # NESTED: M(r)'s set is M(3)'s rows of rounds 1 to r, in the same order. The hash does not depend on r.
    for number in (1, 2):
        assert [row["id"] for row in sets[number]] == [row["id"] for row in sets[3] if row["round"] <= number] and kept[number] < kept[number + 1]
    assert [row["id"] for row in sets[3]] == [row["id"] for row in every if row["id"] in kept[3]]                                    # the arm's order, rows left out
    # The share follows r(k): nearly every row at k = 1 (r = 0.97), about half at k = 2, none at k = 6 or more; the assembled rows count as k = 1.
    share = kept_share([{**row, "k": _k(row)} for row in every], kept[3])
    assert share["rows"] == len(every) and share["kept"] == len(sets[3]) and share["share"] == round(len(sets[3]) / len(every), 5) and list(share["by_k"]) == ["1", "2", "3", "4", "5", "6", "7"]
    assert share["by_k"]["1"]["share"] > 0.85 and share["by_k"]["1"]["share"] > share["by_k"]["3"]["share"] and share["by_k"]["6"]["kept"] == share["by_k"]["7"]["kept"] == 0
    assert sum(entry["rows"] for entry in share["by_k"].values()) == len(every) and reward(1, 8, TARGET) == pytest.approx(0.9701, abs=1e-4) and reward(8, 8, TARGET) == 0
    assert {row["id"] for row in every if row["origin"] == "assembled"} & kept[3]                                                   # assembled rows are kept as k = 1 rows are
    # Another seed is another draw; the rows themselves do not move a row's draw.
    assert {row["id"] for row in _set(REWARD_ROWS, 3, seed=1)} != kept[3]
    assert {row["id"] for row in rule_set({1: BY_ROUND[1][:10]}, PICKS[1], 1, REWARD_ROWS, TARGET, SEED)} == {row["id"] for row in BY_ROUND[1][:10]} & kept[1]
    # A smoke run's minimum: the rows the rule did not keep join by their draws, smallest first, each marked as NOT kept by the rule.
    filled = _set(REWARD_ROWS, 1, at_least=len(sets[1]) + 3)
    assert len(filled) == len(sets[1]) + 3 and sum(row["kept_by"] == THE_MINIMUM for row in filled) == 3 and {row["id"] for row in filled if row["kept_by"] == THE_RULE} == kept[1]
    said = record(REWARD_ROWS, filled, [{**row, "k": _k(row)} for row in training_set(BY_ROUND, 1)])
    assert (said["rule"], said["rows"], said["rows_of_the_rounds"], said["kept_by_the_runs_minimum"]) == (REWARD_ROWS, len(filled), len(BY_ROUND[1]), 3)
    assert said["rows_by_origin"] == by_origin(filled) and said["rows_by_origin"]["pretraining"] == 0 and said["kept_of_the_rounds_rows"]["kept"] == len(filled)


def test_rehearse_adds_as_many_pretraining_rows_as_the_model_has_one_shot_rows_and_each_models_are_the_first_of_the_next_models():
    sets = {number: _set(REHEARSE, number) for number in (1, 2, 3)}
    in_hash_order = [row["id"] for row in rehearsal_rows(PRETRAINING, len(PRETRAINING), SEED)]
    counts = {}
    for number, rows in sets.items():
        rounds, rehearsal = [row for row in rows if row["origin"] != "pretraining"], [row for row in rows if row["origin"] == "pretraining"]
        one_shot = sum(row["origin"] == "attempt" for row in rounds)
        counts[number] = one_shot
        # ALL the rounds' rows (one-shot and assembled), and as many rehearsal rows as one-shot rows.
        assert sorted(row["id"] for row in rounds) == sorted(row["id"] for row in training_set(BY_ROUND, number)) and len(rehearsal) == one_shot == 28 * number
        # THE NESTED DRAW: the first `one_shot` rows of the ONE hash order of the pretraining file's rows.
        assert sorted(row["id"] for row in rehearsal) == sorted(in_hash_order[:one_shot])
        assert [row["id"] for row in rows] == [row["id"] for row in training_order(rows, SEED)]                                    # one order, the arm's content hash
        # STORED BY ID AND PLACE, never with text.
        assert all({"theorem", "completion", "statement", "proof"}.isdisjoint(row) and row["k"] is None and PRETRAINING[row["row_of_the_file"]]["problem_id"] == row["problem_id"]
                   and row["id"] == f"{row['problem_id']}#pretraining" and half_of(row["problem_id"], 0) == PRETRAIN for row in rehearsal)
        assert all(row["k"] == _k(row) and row["n"] == 8 for row in rounds)
    assert counts[1] < counts[2] < counts[3]
    for number in (1, 2):       # M(r)'s rehearsal rows are a prefix of M(r + 1)'s, in the hash's order
        own = [row["id"] for row in sets[number] if row["origin"] == "pretraining"]
        assert set(own) < {row["id"] for row in sets[number + 1] if row["origin"] == "pretraining"} and sorted(own, key=in_hash_order.index) == in_hash_order[:len(own)]
    # THE TWIN is the same rule with the assembled rows left out: L4t's `rehearse` exactly (the one-shot rows and as many pretraining rows, one order).
    twin = twin_of(sets[3])
    one_shot = attempts_alone(sets[3])
    assert [row["id"] for row in twin] == [row["id"] for row in sets[3] if row["origin"] != "assembled"] == [
        row["id"] for row in rehearse_set(one_shot, rehearsal_rows(PRETRAINING, len(one_shot), SEED), SEED)]
    assert twin_of(_set(REWARD_ROWS, 3)) == attempts_alone(_set(REWARD_ROWS, 3)) and twin_of(_set(OLD, 3)) == attempts_alone(_set(OLD, 3))       # no rehearsal row: the one-shot rows alone
    said = record(REHEARSE, sets[3], [{**row, "k": _k(row)} for row in training_set(BY_ROUND, 3)])
    assert said["rows_by_origin"] == {"attempt": 84, "assembled": 24, "pretraining": 84} and said["kept_of_the_rounds_rows"]["share"] == 1.0 and said["kept_by_the_runs_minimum"] == 0
    # Refused: no rows to draw among; fewer rows than the model has one-shot rows (no row is drawn twice).
    with pytest.raises(ValueError, match="the rule `rehearse` draws its rehearsal rows among the rows the start model was pretrained on, and none were given"):
        rule_set(BY_ROUND, _picks(1), 1, REHEARSE, TARGET, SEED)
    with pytest.raises(ValueError, match="`pre` was trained on 10 rows and 28 are asked for the rehearsal"):
        rule_set(BY_ROUND, _picks(1), 1, REHEARSE, TARGET, SEED, PRETRAINING[:10])


def test_the_rules_kept_share_by_round_is_read_from_the_rounds_own_rows_and_the_last_models_set():
    rounds = {number: {"results": PICKS[number], "examples": BY_ROUND[number]} for number in (1, 2, 3)}
    for rule in RULES:
        last = _set(rule, 3)
        share = the_rule_by_round(rounds, last)
        of_the_rounds = [row for row in last if row["origin"] != "pretraining"]
        assert (share["rows"], share["kept"], share["set_by_origin"]) == (108, len(of_the_rounds), by_origin(last)) and list(share["by_round"]) == ["1", "2", "3"]
        assert all(entry["rows"] == 36 and entry["kept"] == sum(row["round"] == int(number) for row in of_the_rounds) for number, entry in share["by_round"].items())
        assert (share["share"] == 1.0) == (rule != REWARD_ROWS)


# --------------------------------------------------------------------------------------------- the fifth check
def test_the_fifth_check_names_the_rehearsal_rows_and_still_bars_a_rounds_row_of_the_pretrain_half_and_any_held_out_problem():
    rounds_rows = {"M(1)": LOOP_HALF[:20], "M(2)": LOOP_HALF[:40], "`without`": LOOP_HALF[:30]}
    rehearsed = {"M(1)": PRETRAIN_HALF[:15], "M(2)": PRETRAIN_HALF[:30], "`without`": PRETRAIN_HALF[:30]}
    checks = lambda trained_on=rounds_rows, **more: arm_checks(OF_THE_PRETRAINING, MEASURED, RUNGS_FINE, trained_on, frozenset({"held"}), 0, SETTINGS, **more)["no_training_row_is_of_the_pretrain_half_or_held_out"]      # noqa: E731
    named = checks(rehearsed=rehearsed)
    assert named["passes"] is True and named["barred_problems"] == 0 and named["rehearsal_rows_of_the_pretrain_half"] == {"M(1)": 15, "M(2)": 30, "`without`": 30}
    assert named["barred_rehearsal_rows"] == 0 and "named as such" in named["what_of_the_rehearsal_rows"] and named["rows"] == {"M(1)": 20, "M(2)": 40, "`without`": 30}
    # WITHOUT the rule (every other arm): the check is what it was, key for key, and a row of the `pretrain` half is barred whatever it is called.
    plain = checks()
    assert list(plain) == ["what", "rows", "barred_problems", "first", "passes"] and plain == {key: named[key] for key in plain}
    assert checks({**rounds_rows, "M(2)": [*LOOP_HALF[:40], *PRETRAIN_HALF[:3]]})["barred_problems"] == 3
    # A ROUND's row of the `pretrain` half is barred as ever; so is a held-out problem, among the rounds' rows or the rehearsal rows; so is a rehearsal row that is NOT of that half.
    a_rounds_row = checks({**rounds_rows, "M(2)": [*LOOP_HALF[:40], PRETRAIN_HALF[40]]}, rehearsed=rehearsed)
    assert (a_rounds_row["passes"], a_rounds_row["first"], a_rounds_row["barred_rehearsal_rows"]) == (False, [PRETRAIN_HALF[40]], 0)
    for wrong in ("held", LOOP_HALF[50]):
        of_the_rehearsal = checks(rehearsed={**rehearsed, "`without`": [*PRETRAIN_HALF[:30], wrong]})
        assert (of_the_rehearsal["passes"], of_the_rehearsal["first"], of_the_rehearsal["barred_rehearsal_rows"]) == (False, [wrong], 1)
    assert checks({**rounds_rows, "M(1)": [*LOOP_HALF[:20], "held"]}, rehearsed=rehearsed)["first"] == ["held"]
    # A held-out problem that IS of the `pretrain` half: nothing but its being held out bars it, and that is enough.
    held_out = PRETRAIN_HALF[400]
    of_that_half = arm_checks(OF_THE_PRETRAINING, MEASURED, RUNGS_FINE, rounds_rows, frozenset({held_out}), 0, SETTINGS, rehearsed={**rehearsed, "M(2)": [*PRETRAIN_HALF[:30], held_out]})[
        "no_training_row_is_of_the_pretrain_half_or_held_out"]
    assert half_of(held_out, 0) == PRETRAIN and (of_that_half["passes"], of_that_half["first"], of_that_half["barred_rehearsal_rows"]) == (False, [held_out], 1)


def test_the_first_two_checks_are_named_for_the_model_the_arm_starts_from():
    on_g = [{"problem_id": problem, "resolved": resolved} for problem, resolved in zip(GOAL_IDS, [3] * 9 + [0] * 3)]
    of_pre, of_another = pretraining_checks(on_g, GOAL_IDS[8:], GOAL_IDS, 9, 4), pretraining_checks(on_g, GOAL_IDS[8:], GOAL_IDS, 9, 4, "pre_r64")
    assert "`pre` solves at least 9" in of_pre["the_pretraining_took"]["what"] and "`pre_r64` solves at least 9" in of_another["the_pretraining_took"]["what"]
    assert "`pre_r64` does not solve" in of_another["the_goal_set_again_is_large_enough"]["what"]
    strip = lambda checks: {name: {key: value for key, value in check.items() if key != "what"} for name, check in checks.items()}      # noqa: E731
    assert strip(of_pre) == strip(of_another) and of_pre["the_pretraining_took"]["passes"] is True


# ------------------------------------------------------------------------------- the breadth, beside the primary
def _solved(gained, lost, both=100, problems=392):
    return {"problems": problems, "resolved_after": both + gained, "resolved_before": both + lost, "gained": gained, "lost": lost, "sign_test_p": sign_test(gained, lost)}


def test_narrower_is_lost_more_than_gained_with_the_sign_test_under_the_level_whatever_the_primary_says():
    narrowed = breadth_beside(_solved(25, 47), 0.71, 0.82, "pre_r64")
    assert narrowed["narrower"] is True and (narrowed["gained"], narrowed["lost"], narrowed["sign_test_p"]) == (25, 47, sign_test(25, 47)) and sign_test(25, 47) < 0.05
    assert (narrowed["solved_by_with"], narrowed["solved_by_the_start"], narrowed["share_of_distinct_attempts_on_g"]) == (125, 147, {"with": 0.71, "pre_r64": 0.82})
    assert narrowed["smallest_decisive_split"] == smallest_decisive_split(72) and "`with` against `pre_r64`" in narrowed["what"] and narrowed["counts"].startswith("gained 25, lost 47 (p = 0.0127")
    # Lost more, but not under the level; gained more, however lopsided; nothing changed: not narrower.
    assert sign_test(24, 36) > 0.05 and breadth_beside(_solved(24, 36), 0.7, 0.8, "pre_r64")["narrower"] is False
    assert sign_test(47, 25) < 0.05 and breadth_beside(_solved(47, 25), 0.7, 0.8, "pre_r64")["narrower"] is False
    assert sign_test(23, 39) > 0.05 > sign_test(23, 40) and breadth_beside(_solved(23, 39), 0.7, 0.8, "pre_r64")["narrower"] is False and breadth_beside(_solved(23, 40), 0.7, 0.8, "pre_r64")["narrower"] is True
    nothing = breadth_beside(_solved(0, 0), 0.8, 0.8, "pre_r64")
    assert nothing["narrower"] is False and nothing["counts"] == "gained 0, lost 0 (no problem changed)"
    # The branch's NAME is L4's and does not move; NARROWER is written beside it on every branch, and an "adds" is then "adds per attempt, on fewer problems".
    passing = {"a": {"passes": True}}
    for primary, name in (({"mean": 0.004, "low": 0.001, "high": 0.007}, ADDS), ({"mean": 0.001, "low": -0.002, "high": 0.004}, NOT_SHOWN), ({"mean": -0.004, "low": -0.007, "high": -0.001}, COSTS)):
        branch = l4_branch(passing, primary, None, "pre_r64")
        assert branch["name"] == name and "`pre`" not in branch["reason"]
        wide, narrow = with_breadth(branch, breadth_beside(_solved(30, 30), 0.8, 0.8, "pre_r64")), with_breadth(branch, narrowed)
        assert (wide["name"], wide["narrower"], wide["reported_as"], wide["reason"]) == (name, False, name, branch["reason"])
        assert (narrow["name"], narrow["narrower"]) == (name, True) and f"{NARROWER}: against the start model it lost more goal problems than it gained" in narrow["reason"]
        assert narrow["reported_as"] == (f"{name}, {NARROWER}: {ON_FEWER}" if name == ADDS else f"{name}, {NARROWER}") and (ON_FEWER in narrow["reason"]) == (name == ADDS)
    assert ON_FEWER == "adds per attempt, on fewer problems" and "the same `pre_r64`" in l4_branch(passing, {"mean": 0.004, "low": 0.001, "high": 0.007}, None, "pre_r64")["reason"]
    assert "the same `pre`)" in l4_branch(passing, {"mean": 0.004, "low": 0.001, "high": 0.007})["reason"]                         # L4's own reason is what it was
    # INCONCLUSIVE says nothing else, and a primary with nothing to be read on still has its breadth read.
    failed = with_breadth(l4_branch({"a": {"passes": False}}, {"mean": 0.004, "low": 0.001, "high": 0.007}, None, "pre_r64"), narrowed)
    assert (failed["name"], failed["narrower"], failed["reported_as"]) == (INCONCLUSIVE, None, INCONCLUSIVE) and NARROWER not in failed["reason"]
    unread = with_breadth(l4_branch(passing, {"mean": None}, None, "pre_r64"), narrowed)
    assert (unread["name"], unread["narrower"], unread["reported_as"]) == (NOT_READ, True, f"{NOT_READ}, {NARROWER}")


# ------------------------------------------------------------------------------------------------ the report
START = "pre_r64"
RECIPE = {"rank": 64, "alpha": 128, "rank_of_the_config": 16, "alpha_of_the_config": 32}
ROUNDS = {number: {"summary": {"assembly": {"lean_checks": 40}}, "results": PICKS[number], "examples": BY_ROUND[number]} for number in (1, 2)}
RANK_16 = {"read": True, "arm": "t010_assembly_pre", "run": "runs/ladder_l2_t010_assembly_pre_seed0", **of_the_rank_16_arm(_report())}       # L4's own hand-made report, as stored
NARROW = ([0] * 11 + [0], [0] * 11 + [5])                  # `with` solves ONE goal problem, the last (which the start model never does), and none of the nine it solves


def _own(rule=OLD, **changed):
    return {**OWN, "check": "L4b", "arm": "t010_assembly_pre_r64", "pretraining_run": "runs/ladder_l4_pretrain_r64_seed0", "start": START,
            "start_adapter": "runs/ladder_l4_pretrain_r64_seed0/adapters/pre_r64", "start_recipe": RECIPE, "rule": rule, **changed}


def _l4b(rule=OLD, again=("g4c", "g8a", "g8b", "g8c"), rank_16=RANK_16, never=None, own=None, losses=None, trained_rows=None, **models):
    sets = {2: rule_set(BY_ROUND, _picks(2), 2, rule, TARGET, SEED, PRETRAINING), 1: rule_set(BY_ROUND, _picks(1), 1, rule, TARGET, SEED, PRETRAINING)}
    trained = trained_rows or {"M(1)": sets[1], "M(2)": sets[2], "`without`": twin_of(sets[2])}
    trains = {name: {"rows": len(rows), "trained_from": "the stored adapter pre_r64", "stand_in_engine": False, "against_the_start_adapter": {name: CHANGED},
                     "rows_by_origin": {"attempt": sum(row["origin"] == "attempt" for row in rows), "assembled": sum(row["origin"] == "assembled" for row in rows), "h0": 0},
                     "rule": {"rule": rule}} for name, rows in trained.items()}
    falling = lambda count: [1.5] * (count // 2) + [0.9] * (count - count // 2)      # noqa: E731
    made = {START: models.get("start") or _made("pre"), "with": models.get("with") or _made("with"), "without": models.get("without") or _made("without"), "loop": _model("loop")}
    read = ({**PREPARE, "arm": "t010_assembly_pre_r64"}, None, {**ARM_PREPARE, "arm": "t010_assembly_pre_r64", "start": START}, list(again), trains,
            losses or {name: falling(len(rows)) for name, rows in trained.items()}, None, [3, 3, 5, 9, 4, 2], ROUNDS, GROUPS, LENGTHS, _model("base"), made, BASE_ARM, SETTINGS,
            TARGET, EVALUATION)
    return build_l4b_report(read, own or _own(rule), trained, sets[2], rank_16, never)


def test_the_report_is_l4s_with_the_start_model_in_pres_place_and_what_l4b_adds():
    report = _l4b()
    of_l4 = _report()                                                           # L4's own report on the same rows, from `pre`
    assert (report["check"], report["start"], report["rank"], report["alpha"], report["rule"], report["label"]) == ("L4b", START, 64, 128, OLD, LABEL)
    assert report["spec"].endswith("L4b: the arm again, from the larger pretrained model") and all(line.startswith(f"{SAY}: ") for line in report["lines"])
    # The first line says what this run is; then L4's lines, with `pre_r64` wherever L4's say `pre`; then what L4b adds.
    assert "L4b, THE ARM AGAIN, FROM THE LARGER PRETRAINED MODEL, seed 0. The start model is `pre_r64` (rank 64, alpha 128; the config's own are 16 and 32)" in report["lines"][0]
    assert "EVERY MODEL of the arm and its twin is trained from it AT ITS RANK. THE TRAINING RULE is `old`" in report["lines"][0]
    assert [line[len(SAY) + 2:][:7] for line in report["lines"][2:7]] == ["CHECK 1", "CHECK 2", "CHECK 3", "CHECK 4", "CHECK 5"]
    assert not [line for line in report["lines"][1:] if "`pre`" in line and "RANK-16 ARM" not in line]
    # THE PRIMARY is L4's: G', the second sampling alone, `with` minus the start model. The same rows give the same figure.
    assert report["primary"] == {**of_l4["primary"], "what": report["primary"]["what"]} and "`with` minus `pre_r64`" in report["primary"]["what"]
    # ... in numbers: on the 4 problems of G', over the 61 fresh attempts, `with` has 3 + 2 + 1 + 0 successes and THE START MODEL 1 + 0 + 0 + 0 (the twin has 2 + 1: not it).
    primary = report["primary"]
    assert (primary["goal_set_again"], primary["problems"], primary["successes"], primary["successes_of_the_base"], primary["attempts_each"]) == (4, 4, 6, 1, 244)
    better_start = _l4b(start=_made("pre", on_g=(ON_G["pre"][0], [9, 9, 9, 6, 6, 6, 3, 2, 1, 1, 1, 0])))["primary"]
    assert better_start["successes_of_the_base"] == 3 and _l4b(without=_made("without", on_g=(ON_G["without"][0], [9] * 12)))["primary"] == primary
    assert report["branch"]["name"] == of_l4["branch"]["name"] and set(report["models"]) == {"base", START, "with", "without", "loop"}
    assert list(report["secondary"]["by_length_group"]["pairs"]) == list(of_l4["secondary"]["by_length_group"]["pairs"])            # the pairs keep their names
    assert report["secondary"]["by_length_group"]["pairs"]["with_minus_pre"]["pair"] == "`with` minus `pre_r64`"
    # BESIDE THE PRIMARY, on every branch: the goal problems solved in the 93 attempts, `with` against the start model, and the two shares of distinct attempts.
    breadth = report["breadth_beside_the_primary"]
    solved = report["secondary"]["goal_problems_solved"]["with_minus_pre"]["all"]
    assert {key: breadth[key] for key in ("gained", "lost", "sign_test_p", "problems")} == {key: solved[key] for key in ("gained", "lost", "sign_test_p", "problems")}
    assert (breadth["solved_by_with"], breadth["solved_by_the_start"], breadth["narrower"]) == (solved["resolved_after"], solved["resolved_before"], False)
    assert breadth["share_of_distinct_attempts_on_g"] == {"with": 0.95, START: 0.95} and report["branch"]["narrower"] is False and report["branch"]["reported_as"] == report["branch"]["name"]
    line = next(line for line in report["lines"] if "WITHOUT OVERFITTING" in line)
    assert f"`with` against `pre_r64`: {solved['resolved_after']} to {solved['resolved_before']} of 12, gained {solved['gained']}, lost {solved['lost']}" in line and "not narrower by the sign test at 0.05" in line
    assert report["lines"].index(line) == next(index for index, text in enumerate(report["lines"]) if "BRANCH: " in text) - 1         # beside the primary, before the branch
    # L4's own report has none of it, and is what it is with the defaults said: the same report, line for line.
    assert "breadth_beside_the_primary" not in of_l4 and not any("WITHOUT OVERFITTING" in line for line in of_l4["lines"]) and "reported_as" not in of_l4["branch"]
    # THE RULE's kept share, in the rounds' table and by itself.
    rule = report["secondary"]["the_rule"]
    assert (rule["rule"], rule["rows"], rule["kept"], rule["share"]) == (OLD, 72, 72, 1.0) and [row["kept_by_the_rule"] for row in report["secondary"]["by_round"]["rows"]] == [
        rule["by_round"]["1"], rule["by_round"]["2"]]
    assert any("SECONDARY, THE RULE `old`: of the rounds' 72 rows M(2) was trained on 72 (1.0); by round: round 1 36 of 36; round 2 36 of 36; by k: k = 1 " in line for line in report["lines"])
    # THE RANK-16 ARM'S STORED FIGURES BESIDE EACH: here the same rows, so the same figures.
    beside = report["secondary"]["beside_the_rank_16_arm"]
    assert beside["read"] is True and beside["branch"] == of_l4["branch"]["name"] and set(beside["figures"]) == {
        "primary", "all_of_g_with_minus_start", "all_of_g_with_minus_without", "the_three_rungs_with_minus_start", "solved_at_least_once_and_reliably",
        "goal_problems_solved_with_against_start", "share_of_distinct_attempts_on_g", "by_round"}
    for name in ("all_of_g_with_minus_start", "all_of_g_with_minus_without", "the_three_rungs_with_minus_start", "solved_at_least_once_and_reliably", "goal_problems_solved_with_against_start",
                 "share_of_distinct_attempts_on_g"):
        assert beside["figures"][name]["here"] == beside["figures"][name]["at_rank_16"], name
    assert beside["figures"]["primary"]["at_rank_16"] == of_l4["primary"] and sum("BESIDE EACH, THE RANK-16 ARM" in line for line in report["lines"]) == 2
    # Without it (its run was not on the box): said, and nothing else moves.
    without = _l4b(rank_16={"read": False, "why": "FileNotFoundError: no such report"})
    assert without["secondary"]["beside_the_rank_16_arm"] == {"read": False, "why": "FileNotFoundError: no such report"} and without["branch"] == report["branch"]
    assert any("BESIDE EACH, THE RANK-16 ARM: NOT THERE (FileNotFoundError: no such report). Nothing is refused for it" in line for line in without["lines"])
    assert [line for line in without["lines"] if "RANK-16 ARM" not in line] == [line for line in report["lines"] if "RANK-16 ARM" not in line]
    # The goal problems nothing stored had solved: read when GIVEN, counted for each model in its 93 attempts.
    assert report["secondary"]["never_solved"]["given"] is False and any("nothing stored had solved: not given" in line for line in report["lines"])
    given = _l4b(never=["g8a", "g8b", "g8c", "elsewhere"])["secondary"]["never_solved"]
    in_93 = lambda name: [problem for problem, first, more in zip(GOAL_IDS, *ON_G[name]) if problem in ("g8a", "g8b", "g8c") and first + more > 0]      # noqa: E731
    assert (given["given"], given["problems"], given["of_this_runs_goal_set"]) == (True, 4, 3) and given["solved_problems"] == {START: in_93("pre"), "with": in_93("with"), "without": in_93("without")}
    assert given["solved"] == {name: len(found) for name, found in given["solved_problems"].items()} and given["solved"][START] == 0 < given["solved"]["with"] and len(report["what_it_cannot_say"]) == 3


def test_every_branch_of_the_report_and_narrower_beside_each():
    # `with` adds on the one problem of G' and solves NONE of the nine the start model solves: THE LOOP ADDS, NARROWER, "adds per attempt, on fewer problems".
    narrow = _l4b(again=["g8c"], **{"with": _made("with", on_g=NARROW)})
    breadth = narrow["breadth_beside_the_primary"]
    assert (breadth["gained"], breadth["lost"], breadth["narrower"]) == (1, 9, True) and sign_test(1, 9) < 0.05 and narrow["primary"]["low"] > 0
    assert narrow["branch"]["name"] == ADDS and narrow["branch"]["narrower"] is True and narrow["branch"]["reported_as"] == f"{ADDS}, NARROWER: adds per attempt, on fewer problems"
    assert any(f"BRANCH: {ADDS}, NARROWER: adds per attempt, on fewer problems. " in line for line in narrow["lines"]) and f": {ADDS}, NARROWER: adds per attempt, on fewer problems. " in narrow["headline"]
    assert any("WITHOUT OVERFITTING" in line and "1 to 9 of 12, gained 1, lost 9 (p = 0.0214844)" in line and line.endswith("NARROWER (lost more than gained with the sign test under 0.05)")
               for line in narrow["lines"])
    assert "two more seeds" in narrow["branch"]["reason"].lower() and "the same `pre_r64`" in narrow["branch"]["reason"]
    # The same breadth with a primary that holds zero, and with one below it: NOT SHOWN and COSTS, each NARROWER.
    held = _l4b(again=["g8c"], **{"with": _made("with", on_g=([0] * 12, [0] * 11 + [0])), "start": _made("pre", on_g=(ON_G["pre"][0], [9, 9, 9, 6, 6, 6, 3, 2, 1, 0, 0, 0]))})
    assert held["branch"]["name"] == NOT_SHOWN and held["branch"]["narrower"] is True and held["branch"]["reported_as"] == f"{NOT_SHOWN}, NARROWER" and "on fewer problems" not in held["branch"]["reported_as"]
    costs = _l4b(again=["g4c"], **{"with": _made("with", on_g=([0] * 12, [0] * 12))})
    assert costs["branch"]["name"] == COSTS and costs["branch"]["reported_as"] == f"{COSTS}, NARROWER" and costs["breadth_beside_the_primary"]["lost"] == 9
    # Broader and adding: the plain branch.
    assert _l4b()["branch"]["reported_as"] in (ADDS, NOT_SHOWN, COSTS)
    # A failing check: INCONCLUSIVE, and NOTHING ELSE IS SAID: no primary, no breadth, no rule, no rank 16. What was measured is kept.
    rising = lambda count: [0.9] * (count // 2) + [1.5] * (count - count // 2)      # noqa: E731
    sets = {2: rule_set(BY_ROUND, _picks(2), 2, OLD, TARGET, SEED), 1: rule_set(BY_ROUND, _picks(1), 1, OLD, TARGET, SEED)}
    trained = {"M(1)": sets[1], "M(2)": sets[2], "`without`": twin_of(sets[2])}
    failed = _l4b(again=["g8c"], losses={name: rising(len(rows)) for name, rows in trained.items()}, **{"with": _made("with", on_g=NARROW)})
    assert failed["branch"]["name"] == INCONCLUSIVE == failed["branch"]["reported_as"] and failed["branch"]["narrower"] is None and failed["primary"] is None
    assert failed["breadth_beside_the_primary"] is None and failed["secondary"] is None and failed["measured_and_not_read"]["breadth_beside_the_primary"]["narrower"] is True
    assert not any(word in line for line in failed["lines"] for word in ("PRIMARY", "SECONDARY", "WITHOUT OVERFITTING", "NARROWER", "RANK-16")) and "INCONCLUSIVE" in failed["headline"]
    assert failed["measured_and_not_read"]["secondary"]["the_rule"]["rule"] == OLD and failed["lines"][0].startswith(f"{SAY}: L4b, THE ARM AGAIN")
    # The start model's own checks, as the prepare step read them: a failing one is INCONCLUSIVE too.
    failing = {**OF_THE_PRETRAINING, "the_pretraining_took": {**OF_THE_PRETRAINING["the_pretraining_took"], "passes": False}}
    assert _l4b(own=_own(the_two_checks_of_the_pretraining=failing))["branch"]["name"] == INCONCLUSIVE


def test_the_report_with_the_rule_rehearse_names_its_rehearsal_rows_and_bars_what_is_not_one():
    report = _l4b(REHEARSE)
    barred = report["can_this_run_see_a_win"]["no_training_row_is_of_the_pretrain_half_or_held_out"]
    assert barred["passes"] is True and barred["rehearsal_rows_of_the_pretrain_half"] == {"M(1)": 28, "M(2)": 56, "`without`": 56} and barred["rows"] == {"M(1)": 36, "M(2)": 72, "`without`": 56}
    rule = report["secondary"]["the_rule"]
    assert (rule["rule"], rule["kept"], rule["rows"], rule["set_by_origin"]) == (REHEARSE, 72, 72, {"attempt": 56, "assembled": 16, "pretraining": 56})
    assert any("its set by origin: attempt 56, assembled 16, pretraining 56" in line for line in report["lines"])
    # The SAME rows under another rule are not rehearsal rows: a row of the `pretrain` half in a training of an arm without the rule is barred.
    sets = {2: rule_set(BY_ROUND, _picks(2), 2, REHEARSE, TARGET, SEED, PRETRAINING), 1: rule_set(BY_ROUND, _picks(1), 1, REHEARSE, TARGET, SEED, PRETRAINING)}
    smuggled = _l4b(OLD, trained_rows={"M(1)": sets[1], "M(2)": sets[2], "`without`": twin_of(sets[2])})
    barred = smuggled["can_this_run_see_a_win"]["no_training_row_is_of_the_pretrain_half_or_held_out"]
    assert smuggled["branch"]["name"] == INCONCLUSIVE and barred["passes"] is False and barred["barred_problems"] == 56 and "rehearsal_rows_of_the_pretrain_half" not in barred
    # `reward_rows`: the kept share is the rule's, by round and by k.
    kept = _l4b(REWARD_ROWS)["secondary"]["the_rule"]
    assert kept["rule"] == REWARD_ROWS and kept["kept"] < kept["rows"] == 72 and kept["by_k"]["1"]["share"] > kept["by_k"]["3"]["share"] and kept["set_by_origin"]["pretraining"] == 0


def test_l4s_own_report_is_what_it_was_with_the_new_arguments_at_their_defaults():
    models = {name: _made(name) for name in ("pre", "with", "without")}
    arguments = (PREPARE, OWN, ARM_PREPARE, ["g4c", "g8a", "g8b", "g8c"], {name: {"rows": 30, "against_the_start_adapter": {name: CHANGED}} for name in ("M(1)", "M(2)", "`without`")},
                 {name: [1.5] * 15 + [0.9] * 15 for name in ("M(1)", "M(2)", "`without`")}, {name: LOOP_HALF[:30] for name in ("M(1)", "M(2)", "`without`")}, [3, 4],
                 {number: {"summary": {}, "results": PICKS[number], "examples": BY_ROUND[number]} for number in (1, 2)}, GROUPS, LENGTHS, _model("base"), models, BASE_ARM, SETTINGS, TARGET,
                 EVALUATION)
    plain = build_l4_report(*arguments)
    assert plain == build_l4_report(*arguments, start="pre", breadth=False, rehearsed=None) and list(plain) == list(build_l4_report(*arguments, start="pre"))
    assert "breadth_beside_the_primary" not in plain and "reported_as" not in plain["branch"] and "narrower" not in plain["branch"]
