"""L4c on hand-made rows with a known answer: the arm's FOURTH training rule (`reward_rehearse`: the rows `reward_rows`
keeps and as many rehearsal rows as that kept set has one-shot rows, nested across the rounds, its twin), the one more
model of an arm with rehearsal rows (`rehearsal_alone`'s set and order), the separation's three outcomes and what makes
it NOT READ, SHORTER, how a branch is said with them, and the report (L4b's, unchanged, with the two reads beside the
primary for a rule with rehearsal rows and for no other). Spec: docs/spec/ladder-loop.spec.md, "L4c: the arm from
`pre_r64` with a rule L4t2 names". Pure: no model, no Lean. The stage end to end is `test_ladder_l4c_stage.py`."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l3d1 import EVALUATION, GOAL_IDS, GROUPS, LENGTHS, _model  # noqa: E402 - Step 1's hand-made world: 12 goal problems, 3 a length group
from test_ladder_l4 import CHANGED, ON_G, SETTINGS, _made  # noqa: E402 - L4's hand-made `pre`, `with` and `without`
from test_ladder_l4_report import ARM_PREPARE, BASE_ARM, OWN, PREPARE  # noqa: E402
from test_ladder_l4b import BY_ROUND, K_OF, PICKS, PRETRAINING, SEED, START, TARGET, _alone, _k, _l4b, _own, _picks, _set, _solved  # noqa: E402 - L4b's three hand-made rounds and its report

from rlvr_lean.domain.ladder_round import l4_rows2  # noqa: E402
from rlvr_lean.domain.ladder_round.assembly import training_order, training_set  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import ADDS, COSTS, INCONCLUSIVE, NOT_READ, NOT_SHOWN, PRETRAIN, half_of, l4_branch  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows import THE_MINIMUM, THE_RULE, rehearsal_rows, uniform  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows2 import HOLDS, KEY, three_reads  # noqa: E402
from rlvr_lean.domain.ladder_round.l4b import NARROWER, OLD, ON_FEWER, REHEARSE, REWARD_REHEARSE, REWARD_ROWS, RULES, WITH_REHEARSAL, breadth_beside, record, rule_set, twin_of, with_breadth  # noqa: E402
from rlvr_lean.domain.ladder_round.l4c import (  # noqa: E402
    NOT_THE_LOOPS,
    NOT_THE_ROUNDS,
    REHEARSAL_ALONE,
    ROUNDS_ADD,
    SEPARATION,
    SHORTER,
    THE_LOOPS,
    long_proofs_beside,
    rehearsal_alone_set,
    separation,
    with_the_long_proofs,
    with_the_separation,
)
from rlvr_lean.domain.ladder_round.rounds import sign_test  # noqa: E402
from rlvr_lean.domain.problem_pool.episodes import reward  # noqa: E402
from rlvr_lean.reporting.ladder_l4 import SAY, build_l4_report  # noqa: E402
from rlvr_lean.reporting.ladder_l4b import build_l4b_report, the_rule_by_round  # noqa: E402

NOT_SEPARATED = "NOT SEPARATED AT THIS SIZE"
IN_HASH_ORDER = [row["id"] for row in rehearsal_rows(PRETRAINING, len(PRETRAINING), SEED)]        # the ONE order every rehearsal row is drawn in


def _rehearsal(rows):
    return [row for row in rows if row["origin"] == "pretraining"]


def _of_the_rounds(rows):
    return [row for row in rows if row["origin"] != "pretraining"]


# ---------------------------------------------------------------------------------------------- the fourth rule
def test_the_arm_has_four_rules_and_two_of_them_hold_rehearsal_rows_the_two_l4t2_can_name():
    assert RULES == (OLD, REWARD_ROWS, REHEARSE, REWARD_REHEARSE) == ("old", "reward_rows", "rehearse", "reward_rehearse")
    assert WITH_REHEARSAL == (REHEARSE, REWARD_REHEARSE) and set(WITH_REHEARSAL) < set(RULES)
    # The rule L4t2's line names ("the arm's rule") is one of the arm's own, by the same word: `reward_rehearse` or `rehearse`, and both carry the separation.
    assert l4_rows2.REWARD_REHEARSE == REWARD_REHEARSE and l4_rows2.REHEARSE == REHEARSE
    passing = {"against": START, "failed_checks": [], "all_three": True, "fails": [], "not_read": [], "gained_minus_lost": 0, KEY[l4_rows2.NOT_NARROWER]: {"gained": 3, "lost": 3}}
    named = {l4_rows2.the_arms_rule({"name": l4_rows2.LOOPS_ROWS_ADD}, {**passing, "gained_minus_lost": mine}, passing)["rule"] for mine in (-1, 0, 1)}
    assert named == set(WITH_REHEARSAL)


def test_reward_rehearse_is_the_rows_reward_rows_keeps_and_as_many_rehearsal_rows_as_that_kept_set_has_one_shot_rows():
    sets = {number: _set(REWARD_REHEARSE, number) for number in (1, 2, 3)}
    kept = {number: _set(REWARD_ROWS, number) for number in (1, 2, 3)}
    of_rehearse = {number: _set(REHEARSE, number) for number in (1, 2, 3)}
    for number, rows in sets.items():
        own, rehearsal = _of_the_rounds(rows), _rehearsal(rows)
        # ITS ROUNDS' ROWS ARE `reward_rows`' SET: the same rows, each with its k, reward, draw and why it was kept, in the same order.
        assert own == kept[number] and all(row["kept_by"] == THE_RULE and row["draw"] == uniform(SEED, "l4_rows_keep", row["id"]) < reward(_k(row), 8, TARGET) for row in own)
        # AS MANY REHEARSAL ROWS AS THE KEPT SET HAS ONE-SHOT ROWS: not as it has rows (an assembled row brings none), and not as the rounds have one-shot rows.
        one_shot, assembled = sum(row["origin"] == "attempt" for row in own), sum(row["origin"] == "assembled" for row in own)
        assert len(rehearsal) == one_shot and assembled > 0 and len(rehearsal) != len(own) and one_shot < sum(row["origin"] == "attempt" for row in training_set(BY_ROUND, number))
        # THE FIRST OF THE HASH ORDER `rehearse` DRAWS FROM: so they are among `rehearse`'s own, and are its first.
        assert sorted(row["id"] for row in rehearsal) == sorted(IN_HASH_ORDER[:one_shot])
        theirs = [row["id"] for row in _rehearsal(of_rehearse[number])]
        assert {row["id"] for row in rehearsal} < set(theirs) and sorted(theirs, key=IN_HASH_ORDER.index)[:one_shot] == IN_HASH_ORDER[:one_shot]
        # ONE order, the arm's content hash; a rehearsal row is STORED BY ID AND PLACE, never with text, and is of the `pretrain` half.
        assert [row["id"] for row in rows] == [row["id"] for row in training_order(rows, SEED)]
        assert all({"theorem", "completion", "statement", "proof"}.isdisjoint(row) and row["k"] is None and PRETRAINING[row["row_of_the_file"]]["problem_id"] == row["problem_id"]
                   and row["id"] == f"{row['problem_id']}#pretraining" and half_of(row["problem_id"], 0) == PRETRAIN for row in rehearsal)
        l4_rows2.refuse_text(rows, "the set")
    # NESTED ACROSS THE ROUNDS, both parts: a row kept for M(r) is kept for every later model, and M(r)'s rehearsal rows are the FIRST of M(r + 1)'s.
    for number in (1, 2):
        assert [row["id"] for row in _of_the_rounds(sets[number])] == [row["id"] for row in _of_the_rounds(sets[3]) if row["round"] <= number]
        own = [row["id"] for row in _rehearsal(sets[number])]
        later = [row["id"] for row in _rehearsal(sets[number + 1])]
        assert set(own) < set(later) and sorted(later, key=IN_HASH_ORDER.index)[:len(own)] == sorted(own, key=IN_HASH_ORDER.index) == IN_HASH_ORDER[:len(own)]
    # THE TWIN is the same rule with the assembled rows left out: the last model's set, in its order, without them; which is the rule on the rounds without them.
    twin = twin_of(sets[3])
    assert [row["id"] for row in twin] == [row["id"] for row in sets[3] if row["origin"] != "assembled"] and sum(row["origin"] == "pretraining" for row in twin) == len(_rehearsal(sets[3]))
    without_assembled = {number: [row for row in rows if row["origin"] != "assembled"] for number, rows in BY_ROUND.items()}
    assert [row["id"] for row in rule_set(without_assembled, _picks(3), 3, REWARD_REHEARSE, TARGET, SEED, PRETRAINING)] == [row["id"] for row in twin]
    # What a training's summary says of the rule: the rows by origin, the kept share of the rounds' rows (it follows r(k), as `reward_rows`' does).
    said = record(REWARD_REHEARSE, sets[3], [{**row, "k": _k(row)} for row in training_set(BY_ROUND, 3)])
    assert said["rows_by_origin"] == {"attempt": len(_rehearsal(sets[3])), "assembled": len(_of_the_rounds(sets[3])) - len(_rehearsal(sets[3])), "pretraining": len(_rehearsal(sets[3]))}
    assert (said["rule"], said["rows"], said["kept_of_the_rounds_rows"]["kept"], said["kept_by_the_runs_minimum"]) == (REWARD_REHEARSE, len(sets[3]), len(kept[3]), 0)
    share = the_rule_by_round({number: {"results": PICKS[number], "examples": BY_ROUND[number]} for number in (1, 2, 3)}, sets[3])
    assert share["kept"] == len(kept[3]) < share["rows"] == 108 and share["set_by_origin"]["pretraining"] == len(_rehearsal(sets[3]))
    # Another seed is another draw of both parts; no rows to draw among is refused, naming this rule.
    other = _set(REWARD_REHEARSE, 3, seed=1)
    assert {row["id"] for row in _of_the_rounds(other)} != {row["id"] for row in kept[3]} and {row["id"] for row in _rehearsal(other)} != {row["id"] for row in _rehearsal(sets[3])}
    with pytest.raises(ValueError, match="the rule `reward_rehearse` draws its rehearsal rows among the rows the start model was pretrained on, and none were given"):
        rule_set(BY_ROUND, _picks(1), 1, REWARD_REHEARSE, TARGET, SEED)
    with pytest.raises(ValueError, match="`pre` was trained on 3 rows and .* are asked for the rehearsal"):
        rule_set(BY_ROUND, _picks(1), 1, REWARD_REHEARSE, TARGET, SEED, PRETRAINING[:3])
    # A SMOKE RUN'S MINIMUM (the rule keeps none of a fixture's rows): the rows it adds are rows of the kept set, marked as not the rule's, and a one-shot row among them brings a rehearsal row.
    filled = _set(REWARD_REHEARSE, 1, at_least=len(kept[1]) + 3)
    added = [row for row in _of_the_rounds(filled) if row["kept_by"] == THE_MINIMUM]
    assert len(added) == 3 and _of_the_rounds(filled) == _set(REWARD_ROWS, 1, at_least=len(kept[1]) + 3)
    assert len(_rehearsal(filled)) == sum(row["origin"] == "attempt" for row in _of_the_rounds(filled)) == len(_rehearsal(sets[1])) + sum(row["origin"] == "attempt" for row in added)
    assert all(K_OF[row["problem_id"]] >= 2 for row in added if row["origin"] == "attempt")       # (what the rule left: the problems solved more than once)


# --------------------------------------------------------------------------------------- the one more model
def test_rehearsal_alone_is_trained_on_the_rehearsal_rows_of_the_last_models_set_alone_in_that_sets_order():
    assert REHEARSAL_ALONE == "rehearsal_alone"
    for rule in WITH_REHEARSAL:
        last = _set(rule, 3)
        rows = rehearsal_alone_set(last)
        # Its rehearsal rows and nothing else, IN THE SET'S ORDER with the other rows left out: not the draw's order, and not the file's.
        assert [row["id"] for row in rows] == [row["id"] for row in last if row["origin"] == "pretraining"] and rows == _rehearsal(last) and len(rows) > 10
        assert [row["id"] for row in rows] != sorted((row["id"] for row in rows), key=IN_HASH_ORDER.index)
        assert [row["row_of_the_file"] for row in rows] != sorted(row["row_of_the_file"] for row in rows)
        assert len(rows) == sum(row["origin"] == "attempt" for row in last) < len(last) and not any("completion" in row or "theorem" in row for row in rows)
        # Of an earlier model's set it would be other rows: fewer, the first of these in the draw's order.
        assert {row["id"] for row in rehearsal_alone_set(_set(rule, 2))} < {row["id"] for row in rows}
    # Under `rehearse` the model is L4t2's `rehearse_only` on the twin's set: the same rows in the same order.
    assert rehearsal_alone_set(_set(REHEARSE, 3)) == l4_rows2.rehearse_only_set(twin_of(_set(REHEARSE, 3)))
    # A set with no rehearsal row (every set of the two other rules): there is nothing to train it on. A rehearsal row never carries text.
    for rule in (OLD, REWARD_ROWS):
        with pytest.raises(ValueError, match=r"the last model's set holds no rehearsal row among its \d+: there is nothing to train `rehearsal_alone` on"):
            rehearsal_alone_set(_set(rule, 3))
    last = _set(REWARD_REHEARSE, 3)
    carrying = [{**row, "proof": "a published proof"} if row["origin"] == "pretraining" and index % 7 == 0 else row for index, row in enumerate(last)]
    with pytest.raises(ValueError, match="rehearsal rows of `rehearsal_alone` carry a text field .* never stored with its text"):
        rehearsal_alone_set(carrying)


# ---------------------------------------------------------------------------------------------- the separation
def _interval(mean, low, high, problems=192):
    return {"problems": problems, "mean": mean, "low": low, "high": high}


def _adds(mean, low, high):
    """`rehearsal_alone`'s own read ADDS against the start model, as `three_reads` gives it."""
    return {**_interval(mean, low, high), "passes": None if mean is None else low > 0}


def test_the_separation_has_three_outcomes_in_the_specs_words():
    assert (ROUNDS_ADD, NOT_THE_ROUNDS, l4_rows2.NOT_SEPARATED) == ("THE ROUNDS ADD BESIDE THE REHEARSAL", "THE GAIN IS NOT SHOWN TO BE THE ROUNDS'", NOT_SEPARATED)
    assert SEPARATION == (ROUNDS_ADD, NOT_THE_ROUNDS, NOT_SEPARATED, NOT_READ)
    flat, adding = _adds(0.001, -0.002, 0.004), _adds(0.006, 0.002, 0.011)
    # `with` minus `rehearsal_alone` above zero: THE ROUNDS ADD BESIDE THE REHEARSAL, whatever `rehearsal_alone` itself does.
    for of_alone in (flat, adding):
        first = separation(_interval(0.005, 0.001, 0.009), of_alone, START)
        assert (first["name"], first["the_rounds_add"], first["failed_checks"], first["rehearsal_alone_adds"]) == (ROUNDS_ADD, True, [], of_alone["passes"])
        assert "`with` minus `rehearsal_alone` is above zero (+0.00500 [+0.00100, +0.00900]): the rounds add beside the rehearsal" in first["reason"]
        assert f"`rehearsal_alone` minus `{START}` on the same attempts: " in first["reason"]
    # Else, `rehearsal_alone` itself above zero against the start model: THE GAIN IS NOT SHOWN TO BE THE ROUNDS' (a second pass over published proofs gives it).
    second = separation(_interval(0.002, -0.003, 0.007), adding, START)
    assert (second["name"], second["the_rounds_add"], second["rehearsal_alone_adds"]) == (NOT_THE_ROUNDS, False, True)
    assert "is not above zero (+0.00200 [-0.00300, +0.00700]), and `rehearsal_alone` itself is above zero against `pre_r64`" in second["reason"]
    assert "a second pass over published proofs gives it" in second["reason"]
    # Else: NOT SEPARATED AT THIS SIZE, with what THIS difference resolves (half the width of its interval).
    third = separation(_interval(0.002, -0.003, 0.007), flat, START)
    assert (third["name"], third["the_rounds_add"], third["rehearsal_alone_adds"], third["resolves"]) == (NOT_SEPARATED, False, False, 0.005)
    assert "not separated at this size. This difference resolves 0.00500 per attempt" in third["reason"]
    # AT THE EDGE OF ZERO: an interval that ends at zero is not above it, on either side.
    assert separation(_interval(0.004, 0.0, 0.008), flat, START)["name"] == NOT_SEPARATED and separation(_interval(0.004, 0.00001, 0.008), flat, START)["name"] == ROUNDS_ADD
    assert separation(_interval(0.0, 0.0, 0.0), _adds(0.004, 0.0, 0.008), START)["name"] == NOT_SEPARATED
    assert separation(_interval(-0.004, -0.008, -0.001), adding, START)["name"] == NOT_THE_ROUNDS          # `with` BELOW it: still not shown to be the rounds'
    assert separation(_interval(-0.004, -0.008, -0.001), flat, START)["name"] == NOT_SEPARATED
    # NOT READ: a check of `rehearsal_alone` ITSELF failed (whatever the intervals are), or nothing was measured on both sides.
    failed = separation(_interval(0.005, 0.001, 0.009), adding, START, ["its_training_ran", "lean_answered"])
    assert (failed["name"], failed["failed_checks"], failed["the_rounds_add"]) == (NOT_READ, ["its_training_ran", "lean_answered"], None)
    assert "`rehearsal_alone` could not have been read (its_training_ran, lean_answered): the separation is not read, and nothing else of this report changes" in failed["reason"]
    unmeasured = separation({"problems": 0, "mean": None, "low": None, "high": None}, _adds(None, None, None), START)
    assert (unmeasured["name"], unmeasured["failed_checks"], unmeasured["the_rounds_add"]) == (NOT_READ, [], None) and "nothing to be read on" in unmeasured["reason"]


def test_a_primary_that_adds_is_reported_with_the_separation_after_it_and_is_the_loops_gain_only_when_the_rounds_add():
    passing = {"a": {"passes": True}}
    adds = l4_branch(passing, {"mean": 0.004, "low": 0.001, "high": 0.007}, None, START)
    reads = {ROUNDS_ADD: separation(_interval(0.005, 0.001, 0.009), _adds(0.001, -0.002, 0.004), START),
             NOT_THE_ROUNDS: separation(_interval(0.002, -0.003, 0.007), _adds(0.006, 0.002, 0.011), START),
             NOT_SEPARATED: separation(_interval(0.002, -0.003, 0.007), _adds(0.001, -0.002, 0.004), START),
             NOT_READ: separation(_interval(0.005, 0.001, 0.009), _adds(0.001, -0.002, 0.004), START, ["it_still_writes_proofs"])}
    for name, apart in reads.items():
        said = with_the_separation(adds, apart)
        assert (said["name"], said["apart_from_the_rehearsal"], said["the_loops_gain"]) == (ADDS, name, name == ROUNDS_ADD)       # the branch's NAME is L4's and does not move
        assert said["reported_as"] == f"{ADDS}; apart from the rehearsal: {name}" and said["reason"].startswith(adds["reason"] + ". APART FROM THE REHEARSAL, " + name + ": ")
        assert said["reason"].endswith(THE_LOOPS if name == ROUNDS_ADD else NOT_THE_LOOPS) and (THE_LOOPS, NOT_THE_LOOPS) == ("the gain is the loop's", "the gain is NOT called the loop's")
    # After NARROWER and SHORTER, when they are written: what L4b said is kept whole, and the two are added after it.
    narrowed = breadth_beside(_solved(25, 47), 0.7, 0.8, START)
    shorter = long_proofs_beside(three_reads(_interval(0.004, 0.001, 0.007), _solved(25, 47), _interval(-0.01, -0.019, -0.001, 230), START)[KEY[HOLDS]], START)
    assert sign_test(25, 47) < 0.05 and narrowed["narrower"] is True and shorter["shorter"] is True
    whole = with_the_separation(with_the_long_proofs(with_breadth(adds, narrowed), shorter), reads[NOT_SEPARATED])
    assert whole["reported_as"] == f"{ADDS}, {NARROWER}: {ON_FEWER}, {SHORTER}; apart from the rehearsal: {NOT_SEPARATED}"
    assert (whole["name"], whole["narrower"], whole["shorter"], whole["the_loops_gain"]) == (ADDS, True, True, False)
    # A primary that does NOT add is said as it was: there is no gain to call anyone's; the separation stands beside it in its own line.
    for primary, name in (({"mean": 0.001, "low": -0.002, "high": 0.004}, NOT_SHOWN), ({"mean": -0.004, "low": -0.007, "high": -0.001}, COSTS), ({"mean": None}, NOT_READ)):
        branch = l4_branch(passing, primary, None, START)
        for apart in reads.values():
            said = with_the_separation(branch, apart)
            assert (said["name"], said["reported_as"], said["reason"], said["the_loops_gain"], said["apart_from_the_rehearsal"]) == (name, name, branch["reason"], None, apart["name"])
    # INCONCLUSIVE says nothing else.
    failed = l4_branch({"a": {"passes": False}}, {"mean": 0.004, "low": 0.001, "high": 0.007}, None, START)
    said = with_the_separation(with_the_long_proofs(failed, shorter), reads[ROUNDS_ADD])
    assert (said["name"], said["reported_as"], said["reason"], said["shorter"], said["apart_from_the_rehearsal"], said["the_loops_gain"]) == (
        INCONCLUSIVE, INCONCLUSIVE, failed["reason"], None, None, None)


# ---------------------------------------------------------------------------------------------- the long proofs
def test_the_long_proofs_are_written_shorter_when_the_interval_is_wholly_below_zero():
    holds = lambda long: three_reads(_interval(0.004, 0.001, 0.007), _solved(30, 30), long, START)[KEY[HOLDS]]      # noqa: E731 - L4t2's own read, on `with`
    below, through, at_zero, above = (_interval(*figures, 230) for figures in ((-0.01, -0.019, -0.001), (-0.006, -0.018, 0.006), (-0.005, -0.01, 0.0), (0.008, 0.002, 0.02)))
    short = long_proofs_beside(holds(below), START)
    assert (short["shorter"], short["read"], short["problems"], short["mean"], short["low"], short["high"], short["interval"]) == (
        True, True, 230, -0.01, -0.019, -0.001, "-0.01000 [-0.01900, -0.00100]") and "`with` minus `pre_r64`" in short["what"] and SHORTER == "SHORTER"
    # Not wholly below zero (one that ends at zero is not wholly below it): not shorter. Nothing measured: not read, and not shorter.
    assert [long_proofs_beside(holds(long), START)["shorter"] for long in (through, at_zero, above)] == [False, False, False]
    unread = long_proofs_beside(holds({"problems": 0, "mean": None, "low": None, "high": None}), START)
    assert (unread["shorter"], unread["read"], unread["interval"]) == (False, False, "not measured")
    # Beside the branch AS NARROWER IS: after the branch's name on every branch, whatever the primary says; the name does not move.
    passing = {"a": {"passes": True}}
    for primary, name in (({"mean": 0.004, "low": 0.001, "high": 0.007}, ADDS), ({"mean": 0.001, "low": -0.002, "high": 0.004}, NOT_SHOWN),
                          ({"mean": -0.004, "low": -0.007, "high": -0.001}, COSTS), ({"mean": None}, NOT_READ)):
        branch = l4_branch(passing, primary, None, START)
        said, plain = with_the_long_proofs(branch, short), with_the_long_proofs(branch, long_proofs_beside(holds(through), START))
        assert (said["name"], said["shorter"], said["reported_as"]) == (name, True, f"{name}, {SHORTER}") and (plain["name"], plain["shorter"], plain["reported_as"], plain["reason"]) == (
            name, False, name, branch["reason"])
        assert said["reason"] == f"{branch['reason']}. {SHORTER}: on the goal problems of 4 lines or more it has fewer successes per attempt than the start model, the interval wholly below zero (-0.01000 [-0.01900, -0.00100])"
    failed = with_the_long_proofs(l4_branch({"a": {"passes": False}}, {"mean": 0.004, "low": 0.001, "high": 0.007}, None, START), short)
    assert (failed["name"], failed["shorter"], failed["reported_as"]) == (INCONCLUSIVE, None, INCONCLUSIVE) and SHORTER not in failed["reason"]


# ------------------------------------------------------------------------------------------------ the report
PRE_FIRST, PRE_MORE = ON_G["pre"]                           # the start model's successes by goal problem: its 32 attempts, its 61 more
SPLIT_WITH = (ON_G["with"][0], [9, 9, 9, 6, 6, 6, 4, 3, 3, 2, 2, 2])              # on G' (the last four) `with` has 2 more successes than the start model on EVERY problem
SPLIT_ALONE = (PRE_FIRST, [9, 9, 9, 6, 6, 6, 3, 2, 5, 0, 4, 0])                   # ... and `rehearsal_alone` 4 more on two of them and none on the two others
STRONG_START = ([6, 6, 6, 4, 4, 4, 9, 9, 9, 9, 9, 0], [9, 9, 9, 6, 6, 6, 9, 9, 9, 9, 9, 0])       # a start model that solves every long problem but the last
ONE_SUCCESS = ([0] * 12, [0] * 11 + [1])                                           # `with` solves the last goal problem once, in its fresh attempts, and nothing else
BESIDE = ("WITHOUT OVERFITTING", "THE LONG PROOFS", "ITS OWN CHECKS", "THE ROUNDS APART FROM THE REHEARSAL", "BRANCH: ")


def _line(report, word):
    found = [index for index, line in enumerate(report["lines"]) if word in line]
    assert len(found) == 1, (word, found)
    return found[0]


def _last(rule):
    return rule_set(BY_ROUND, _picks(2), 2, rule, TARGET, SEED, PRETRAINING)


@pytest.mark.parametrize("rule", WITH_REHEARSAL)
def test_the_report_of_an_arm_with_rehearsal_rows_reads_the_separation_and_the_long_proofs_beside_the_primary(rule):
    last = _last(rule)
    rehearsed = sum(row["origin"] == "pretraining" for row in last)
    # ---- `rehearsal_alone` has the start model's own successes: on G' over the 61 fresh attempts `with` has 3 + 2 + 1 + 0 and it has 1 + 0 + 0 + 0
    report = _l4b(rule)
    apart, long, branch = report["the_rounds_apart_from_the_rehearsal"], report["the_long_proofs_beside_the_primary"], report["branch"]
    assert (report["check"], report["rule"], report["spec_of_the_separation"]) == ("L4b", rule, "docs/spec/ladder-loop.spec.md, L4c: the arm from `pre_r64` with a rule L4t2 names")
    assert (apart["difference"]["successes"], apart["difference"]["successes_of_the_base"], apart["difference"]["attempts_each"], apart["difference"]["problems"]) == (6, 1, 244, 4)
    assert {key: apart["difference"][key] for key in ("mean", "low", "high")} == {key: report["primary"][key] for key in ("mean", "low", "high")} and apart["difference"]["low"] > 0
    assert (apart["name"], apart["the_rounds_add"], apart["failed_checks"], apart["rehearsal_alone_adds"]) == (ROUNDS_ADD, True, [], False)
    assert (apart["rehearsal_alone_minus_the_start"]["mean"], apart["rehearsal_alone_minus_the_start"]["passes"], apart["rehearsal_alone_minus_the_start"]["problems"]) == (0.0, False, 4)
    assert (branch["name"], branch["the_loops_gain"], branch["apart_from_the_rehearsal"], branch["shorter"], branch["narrower"]) == (ADDS, True, ROUNDS_ADD, False, False)
    assert branch["reported_as"] == f"{ADDS}; apart from the rehearsal: {ROUNDS_ADD}" and branch["reason"].endswith(THE_LOOPS)
    assert f": {branch['reported_as']}. " in report["headline"] and f"BRANCH: {branch['reported_as']}. {branch['reason']}" in report["lines"][_line(report, "BRANCH: ")]
    # ITS OWN CHECKS, the ones a measured model has: all three pass, and what it was trained on is the last model's rehearsal rows.
    checks = apart["checks_of_rehearsal_alone"]
    assert [name for name in checks if name != "what"] == ["its_training_ran", "it_still_writes_proofs", "lean_answered"] and all(checks[name]["passes"] for name in checks if name != "what")
    assert checks["lean_answered"]["share_without_an_answer"] == {"rungs_rehearsal_alone": 0.0, "goal_rehearsal_alone_1": 0.0, "goal_rehearsal_alone_2": 0.0}
    assert (apart["rehearsal_alone"]["rows"], apart["rehearsal_alone"]["rows_by_origin"], apart["rehearsal_alone"]["training"]["rows"]) == (rehearsed, {"pretraining": rehearsed}, rehearsed)
    # THE LONG PROOFS: the 6 goal problems of 4 lines or more over the 93 attempts, `with` minus the start model: L4's own figure for that pair, and not below zero here.
    of_l4 = report["secondary"]["by_length_group"]["pairs"]["with_minus_pre"]["4_or_more"]
    assert (long["problems"], long["mean"], long["low"], long["high"]) == (6, of_l4["mean"], of_l4["low"], of_l4["high"]) and (long["read"], long["shorter"]) == (True, False) and long["low"] > 0
    assert (of_l4["successes"], of_l4["successes_of_the_base"], of_l4["attempts_each"]) == (20, 9, 6 * 93)
    # THE LINES: what the run is, first; then beside the primary and before the branch, in this order and next to each other.
    assert f"THE TRAINING RULE is `{rule}`" in report["lines"][0] and "THE ARM CARRIES ITS OWN SEPARATION" in report["lines"][0]
    assert f"`rehearsal_alone`, is trained from `pre_r64` on the {rehearsed} rehearsal rows of M(2)'s set ALONE" in report["lines"][0]
    places = [_line(report, word) for word in BESIDE]
    assert places == list(range(places[0], places[0] + 5)) and _line(report, "PRIMARY. ") < places[0] and all(line.startswith(f"{SAY}: ") for line in report["lines"])
    assert "the 6 goal problems of 4 lines or more, successes per attempt over their 93 attempts, `with` minus `pre_r64`: " in report["lines"][places[1]]
    assert report["lines"][places[1]].endswith("not shorter (the interval is not wholly below zero)")
    assert f"`rehearsal_alone` (`pre_r64` trained one more pass on the {rehearsed} rehearsal rows of M(2)'s set ALONE" in report["lines"][places[2]]
    assert report["lines"][places[2]].count("PASS") == 3 and "FAIL" not in report["lines"][places[2]]
    assert f"on G' over the 61 attempts of the second sampling, `with` minus `rehearsal_alone`, paired by problem, 95% bootstrap over problems: " in report["lines"][places[3]]
    assert f": {ROUNDS_ADD}. on G' over the fresh attempts `with` minus `rehearsal_alone` is above zero" in report["lines"][places[3]]
    # `rehearsal_alone` IS NO MODEL OF ANY OTHER READ: L4's tables, its sets' health and its five checks do not know it; it is named among the models.
    assert set(report["secondary"]["goal_problems_solved_by_attempts_alone"]) == {"what", "base", START, "with", "without", "loop"} and not [name for name in report["attempts"] if REHEARSAL_ALONE in name]
    assert list(report["models"])[-1] == REHEARSAL_ALONE and "The arm's own separation" in report["models"][REHEARSAL_ALONE] and REHEARSAL_ALONE not in report["trainings"]
    assert REHEARSAL_ALONE not in str(report["can_this_run_see_a_win"]) and "named by L4t2" in report["what_it_cannot_say"][1] and "exploratory" in report["what_it_cannot_say"][1]

    # ---- `rehearsal_alone` has `with`'s own successes: the difference is zero, and it is itself above zero against the start model
    second_pass = _l4b(rule, alone=_alone(last, on_g=ON_G["with"]))
    apart = second_pass["the_rounds_apart_from_the_rehearsal"]
    assert (apart["name"], apart["the_rounds_add"], apart["rehearsal_alone_adds"], apart["difference"]["mean"]) == (NOT_THE_ROUNDS, False, True, 0.0)
    assert {key: apart["rehearsal_alone_minus_the_start"][key] for key in ("mean", "low", "high")} == {key: report["primary"][key] for key in ("mean", "low", "high")}
    assert second_pass["branch"]["reported_as"] == f"{ADDS}; apart from the rehearsal: {NOT_THE_ROUNDS}" and second_pass["branch"]["the_loops_gain"] is False
    assert second_pass["branch"]["reason"].endswith(NOT_THE_LOOPS) and f": {NOT_THE_ROUNDS}. " in second_pass["lines"][_line(second_pass, "THE ROUNDS APART")]
    # ---- `with` is above the start model on every problem of G'; `rehearsal_alone` on half of them only: neither difference is above zero
    split = _l4b(rule, alone=_alone(last, on_g=SPLIT_ALONE), **{"with": _made("with", on_g=SPLIT_WITH)})
    apart = split["the_rounds_apart_from_the_rehearsal"]
    assert split["primary"]["low"] > 0 and split["branch"]["name"] == ADDS and (split["primary"]["successes"], split["primary"]["successes_of_the_base"]) == (9, 1)
    assert (apart["difference"]["successes"], apart["difference"]["successes_of_the_base"], apart["difference"]["mean"]) == (9, 9, 0.0) and apart["difference"]["low"] < 0 < apart["difference"]["high"]
    assert apart["rehearsal_alone_minus_the_start"]["mean"] > 0 and apart["rehearsal_alone_minus_the_start"]["low"] == 0.0 and apart["rehearsal_alone_minus_the_start"]["passes"] is False
    assert (apart["name"], apart["the_rounds_add"], apart["rehearsal_alone_adds"]) == (NOT_SEPARATED, False, False)
    assert apart["resolves"] == round((apart["difference"]["high"] - apart["difference"]["low"]) / 2, 5) and split["branch"]["the_loops_gain"] is False
    assert split["branch"]["reported_as"] == f"{ADDS}; apart from the rehearsal: {NOT_SEPARATED}"
    # ---- the separation's reference is `rehearsal_alone`, and ITS reference the start model: the twin moves neither, the start model moves the second alone
    other_twin = _l4b(rule, without=_made("without", on_g=(ON_G["without"][0], [9] * 12)))
    assert other_twin["the_rounds_apart_from_the_rehearsal"] == report["the_rounds_apart_from_the_rehearsal"] and other_twin["the_long_proofs_beside_the_primary"] == long
    better_start = _l4b(rule, start=_made("pre", on_g=(PRE_FIRST, [9, 9, 9, 6, 6, 6, 3, 2, 1, 1, 1, 0])))["the_rounds_apart_from_the_rehearsal"]
    assert better_start["difference"] == report["the_rounds_apart_from_the_rehearsal"]["difference"] and better_start["rehearsal_alone_minus_the_start"]["mean"] < 0
    # ---- a primary that does not add: the separation is read beside it all the same, and the branch is said as it was
    held = _l4b(rule, **{"with": _made("with", on_g=ON_G["pre"])})
    assert held["branch"]["name"] == NOT_SHOWN == held["branch"]["reported_as"] and held["branch"]["the_loops_gain"] is None
    assert held["the_rounds_apart_from_the_rehearsal"]["name"] == held["branch"]["apart_from_the_rehearsal"] == NOT_SEPARATED and f": {NOT_SEPARATED}. " in held["lines"][_line(held, "THE ROUNDS APART")]


@pytest.mark.parametrize("rule", WITH_REHEARSAL)
def test_a_failed_check_of_rehearsal_alone_makes_the_separation_not_read_and_changes_nothing_else(rule):
    last = _last(rule)
    report = _l4b(rule)
    rising = [0.9] * (len(_alone(last)["rows"]) // 2) + [1.5] * (len(_alone(last)["rows"]) - len(_alone(last)["rows"]) // 2)
    capped = _alone(last)
    capped["rungs"] = [{**row, "attempts_capped": 2} for row in capped["rungs"]]                 # 12 of its 48 attempts on the rungs reached the token cap
    unanswered = _alone(last)
    unanswered["goal"][1][0]["attempts_without_an_answer"] = 20                                 # of the 61 attempts on one problem: over 2% of that set's 732
    not_compared = _alone(last)
    not_compared["train"] = {key: value for key, value in not_compared["train"].items() if key != "against_the_start_adapter"}      # a training that recorded no adapter comparison
    apart_from = ("the_rounds_apart_from_the_rehearsal", "branch", "lines", "headline")
    for alone, check in ((_alone(last, losses=rising), "its_training_ran"), (not_compared, "its_training_ran"), (capped, "it_still_writes_proofs"), (unanswered, "lean_answered")):
        failed = _l4b(rule, alone=alone)
        apart = failed["the_rounds_apart_from_the_rehearsal"]
        assert (apart["name"], apart["failed_checks"], apart["the_rounds_add"]) == (NOT_READ, [check], None) and apart["checks_of_rehearsal_alone"][check]["passes"] is False
        assert apart["difference"]["mean"] is not None                                          # what was measured is kept beside, and not read
        # NOTHING ELSE CHANGES: the checks of the run, the primary, the breadth, the long proofs, the secondary reads, whether the report is to be read.
        assert {key: value for key, value in failed.items() if key not in apart_from} == {key: value for key, value in report.items() if key not in apart_from}
        assert (failed["ok"], failed["not_to_be_read"], failed["inconclusive"], failed["branch"]["name"]) == (True, [], False, ADDS)
        assert failed["branch"]["reported_as"] == f"{ADDS}; apart from the rehearsal: {NOT_READ}" and failed["branch"]["the_loops_gain"] is False and failed["branch"]["reason"].endswith(NOT_THE_LOOPS)
        same = lambda lines: [line for line in lines if not any(word in line for word in ("ITS OWN CHECKS", "THE ROUNDS APART", "BRANCH: "))]      # noqa: E731
        assert same(failed["lines"]) == same(report["lines"]) and "FAIL" in failed["lines"][_line(failed, "ITS OWN CHECKS")] and f": {NOT_READ}. " in failed["lines"][_line(failed, "THE ROUNDS APART")]
        assert failed["lines"][_line(failed, "THE ROUNDS APART")].endswith("Queue the task again: its measure step samples that set again from the kept adapter") == (check == "lean_answered")
    assert _l4b(rule, alone=unanswered)["the_rounds_apart_from_the_rehearsal"]["checks_of_rehearsal_alone"]["lean_answered"]["sets_not_answered"] == ["goal_rehearsal_alone_2"]


@pytest.mark.parametrize("rule", WITH_REHEARSAL)
def test_shorter_is_written_beside_the_branch_as_narrower_is_and_an_inconclusive_run_says_nothing_else(rule):
    last = _last(rule)
    # `with` solves the last goal problem once, in its fresh attempts, and nothing else; the start model solves every long problem but that one.
    report = _l4b(rule, again=["g8c"], start=_made("pre", on_g=STRONG_START), **{"with": _made("with", on_g=ONE_SUCCESS)})
    long, breadth, branch = report["the_long_proofs_beside_the_primary"], report["breadth_beside_the_primary"], report["branch"]
    assert (long["problems"], long["shorter"], long["read"]) == (6, True, True) and long["high"] < 0 and (breadth["gained"], breadth["lost"], breadth["narrower"]) == (1, 11, True)
    assert report["primary"]["low"] > 0 and (branch["name"], branch["narrower"], branch["shorter"], branch["the_loops_gain"]) == (ADDS, True, True, True)
    assert branch["reported_as"] == f"{ADDS}, {NARROWER}: {ON_FEWER}, {SHORTER}; apart from the rehearsal: {ROUNDS_ADD}"
    assert report["lines"][_line(report, "THE LONG PROOFS")].endswith(f"{SHORTER} (the interval is wholly below zero)") and f"{SHORTER}: on the goal problems of 4 lines or more" in branch["reason"]
    assert f"BRANCH: {ADDS}, {NARROWER}: {ON_FEWER}, {SHORTER}; apart from the rehearsal: {ROUNDS_ADD}. " in report["lines"][_line(report, "BRANCH: ")] and branch["reported_as"] in report["headline"]
    # The same long proofs with a primary that holds zero: SHORTER all the same, after the branch's name.
    held = _l4b(rule, again=["g8c"], start=_made("pre", on_g=STRONG_START), **{"with": _made("with", on_g=([0] * 12, [0] * 12))})
    assert held["branch"]["name"] == NOT_SHOWN and held["branch"]["shorter"] is True and held["branch"]["reported_as"] == f"{NOT_SHOWN}, {NARROWER}, {SHORTER}"
    # A FAILING CHECK OF THE RUN: INCONCLUSIVE, and NOTHING ELSE IS SAID: no long proofs, no separation, no check of `rehearsal_alone`. What was measured is kept.
    trained = {"M(1)": _set(rule, 1), "M(2)": last, "`without`": twin_of(last)}
    rising = lambda count: [0.9] * (count // 2) + [1.5] * (count - count // 2)      # noqa: E731
    failed = _l4b(rule, again=["g8c"], start=_made("pre", on_g=STRONG_START), losses={name: rising(len(rows)) for name, rows in trained.items()}, **{"with": _made("with", on_g=ONE_SUCCESS)})
    assert failed["branch"]["name"] == INCONCLUSIVE == failed["branch"]["reported_as"] and (failed["branch"]["shorter"], failed["branch"]["the_loops_gain"], failed["branch"]["apart_from_the_rehearsal"]) == (None, None, None)
    assert failed["the_long_proofs_beside_the_primary"] is None and failed["the_rounds_apart_from_the_rehearsal"] is None and failed["primary"] is None
    kept = failed["measured_and_not_read"]
    assert kept["the_long_proofs_beside_the_primary"]["shorter"] is True and kept["the_rounds_apart_from_the_rehearsal"]["name"] == ROUNDS_ADD
    assert not any(word in line for line in failed["lines"] for word in ("THE LONG PROOFS", "ITS OWN CHECKS", "THE ROUNDS APART", SHORTER, ROUNDS_ADD)) and failed["lines"][0].count("THE ARM CARRIES ITS OWN SEPARATION") == 1


def test_under_the_two_other_rules_the_report_is_what_l4b_wrote_and_an_arm_with_rehearsal_rows_is_not_read_without_its_separation():
    for rule in (OLD, REWARD_ROWS):
        report = _l4b(rule)
        section = [report, report["branch"], *report["lines"], report["headline"], report["models"]]
        assert not [key for key in ("the_long_proofs_beside_the_primary", "the_rounds_apart_from_the_rehearsal", "spec_of_the_separation") if key in report]
        assert not [key for key in ("shorter", "the_loops_gain", "apart_from_the_rehearsal") if key in report["branch"]]
        assert not any(word in str(part) for part in section for word in (REHEARSAL_ALONE, "THE LONG PROOFS", "APART FROM THE REHEARSAL", SHORTER, "CARRIES ITS OWN SEPARATION"))
        assert report["what_it_cannot_say"][1].startswith("the rule was named by L4t on the rank-16 rounds' rows;") and len(report["what_it_cannot_say"]) == 3
        # ... and such an arm is refused the model: it has none.
        with pytest.raises(ValueError, match=f"the rule `{rule}` adds no rehearsal row and the report was given `rehearsal_alone`"):
            _l4b(rule, alone=_alone(_last(REHEARSE)))
    for rule in WITH_REHEARSAL:
        with pytest.raises(ValueError, match=f"the rule `{rule}` adds rehearsal rows and the report was not given `rehearsal_alone`: an arm with rehearsal rows carries its own separation"):
            _l4b(rule, alone=None)
    # The primary, the breadth and the first four checks do not move with the rule: they are read on the same measured rows.
    reports = {rule: _l4b(rule) for rule in RULES}
    for key in ("primary", "breadth_beside_the_primary", "beside_the_primary"):
        assert all(reports[rule][key] == reports[OLD][key] for rule in RULES), key
    assert {reports[rule]["branch"]["name"] for rule in RULES} == {ADDS} and [reports[rule]["branch"].get("reported_as") for rule in (OLD, REWARD_ROWS)] == [ADDS, ADDS]


def test_l4s_own_report_is_what_it_was_with_the_two_new_arguments_at_their_defaults():
    models = {name: _made(name) for name in ("pre", "with", "without")}
    loop_half = [f"p{index}" for index in range(900) if half_of(f"p{index}", 0) != PRETRAIN]
    arguments = (PREPARE, OWN, ARM_PREPARE, ["g4c", "g8a", "g8b", "g8c"], {name: {"rows": 30, "against_the_start_adapter": {name: CHANGED}} for name in ("M(1)", "M(2)", "`without`")},
                 {name: [1.5] * 15 + [0.9] * 15 for name in ("M(1)", "M(2)", "`without`")}, {name: loop_half[:30] for name in ("M(1)", "M(2)", "`without`")}, [3, 4],
                 {number: {"summary": {}, "results": PICKS[number], "examples": BY_ROUND[number]} for number in (1, 2)}, GROUPS, LENGTHS, _model("base"), models, BASE_ARM, SETTINGS, TARGET,
                 EVALUATION)
    plain = build_l4_report(*arguments)
    assert plain == build_l4_report(*arguments, long_proofs=False, rehearsal_alone=None) and "the_long_proofs_beside_the_primary" not in plain and "shorter" not in plain["branch"]
    # The long proofs can be read without the separation, and the separation without the long proofs: each adds its own key and line, and nothing else.
    long_alone = build_l4_report(*arguments, long_proofs=True)
    assert [key for key in long_alone if key not in plain] == ["the_long_proofs_beside_the_primary"] and len(long_alone["lines"]) == len(plain["lines"]) + 1
    assert long_alone["the_long_proofs_beside_the_primary"]["problems"] == 6 and "`with` minus `pre`" in long_alone["the_long_proofs_beside_the_primary"]["what"]
    assert sorted(GOAL_IDS)[6:] == sorted(problem for problem in GOAL_IDS if problem[1] in "48")        # the six problems of 4 lines or more
    assert build_l4b_report.__defaults__ == (None, None, None) and _own(REHEARSE)["rule"] == REHEARSE
