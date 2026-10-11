"""L4t2, two more one-change checks on what L4t left open: its two sets made from the two L4t's run stored (the rehearsal
rows alone; the reward rows with the first of the same hash order, and no published proof's text in either), the three
reads of a model against the start model, Question 1's three outcomes, Question 2, the rule the arm is run with, and the
report on hand-made rows with a known answer. Spec: docs/spec/ladder-loop.spec.md, "L4t2: two more one-change
checks, on what L4t left open". Pure: no model, no Lean. The stage end to end is `test_ladder_l4_rows2_stage.py`."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l3d1 import EVALUATION, GOAL_IDS, GROUPS, LENGTHS  # noqa: E402 - Step 1's hand-made world: 12 goal problems, 3 a length group
from test_ladder_l4 import CHANGED, LOOP_HALF, OF_THE_OTHER_HALF, _made  # noqa: E402 - L4's hand-made models
from test_ladder_l4_rows import FALLING, SAMPLINGS, _assembled, _attempt, _kept  # noqa: E402

from rlvr_lean.domain.ladder_round.assembly import training_order  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ, PRETRAIN, goal_set_again, half_of  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows import MODELS_FROM_ANOTHER_START, REHEARSAL_LABEL, SIGN_TEST_LEVEL, rehearsal_rows, rehearse_set  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows2 import (  # noqa: E402
    ADDS,
    ALL_THREE,
    EXPLORATORY,
    HOLDS,
    KEY,
    LOOPS_ROWS_ADD,
    MODELS,
    NO_ARM,
    NO_RULE,
    NOT_NARROWER,
    NOT_SEPARATED,
    QUESTION_1,
    READS,
    SECOND_PASS,
    STORED,
    TEXT_FIELDS,
    fails,
    question_1,
    question_2,
    refuse_text,
    rehearse_only_set,
    reward_rehearse_set,
    the_arms_rule,
    three_reads,
)
from rlvr_lean.domain.ladder_round.l4b import NARROWER, breadth_beside  # noqa: E402
from rlvr_lean.domain.ladder_round.rounds import sign_test  # noqa: E402
from rlvr_lean.domain.problem_pool.selection import rank  # noqa: E402
from rlvr_lean.reporting.ladder_ceiling import per_attempt, solved  # noqa: E402
from rlvr_lean.reporting.ladder_l4 import LABEL, SAY  # noqa: E402
from rlvr_lean.reporting.ladder_l4_rows import World, check_lines, secondary_lines, secondary_reads, the_four_checks, world_of  # noqa: E402
from rlvr_lean.reporting.ladder_l4_rows2 import CANNOT_SAY, IS, build_rows2_report  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
SETTINGS = CONFIG["ladder_loop"]["l4"]
SEED = 0


def test_the_names_and_the_settings_are_the_specs():
    assert MODELS == ("rehearse_only", "reward_rehearse") and STORED == ("old_rule", "reward_rows", "rehearse") == MODELS_FROM_ANOTHER_START
    assert READS == (ADDS, NOT_NARROWER, HOLDS) == ("ADDS", "NOT NARROWER", "HOLDS THE LONG PROOFS") and set(KEY) == set(READS)
    assert (LOOPS_ROWS_ADD, SECOND_PASS, NOT_SEPARATED) == ("THE LOOP'S ROWS ADD BESIDE THE REHEARSAL", "THE GAIN IS A SECOND PASS OVER PUBLISHED PROOFS", "NOT SEPARATED AT THIS SIZE")
    assert QUESTION_1 == (LOOPS_ROWS_ADD, SECOND_PASS, NOT_SEPARATED, INCONCLUSIVE, NOT_READ)
    assert ALL_THREE == "IT ADDS, KEEPS THE BREADTH AND HOLDS THE LONG PROOFS" and NO_ARM == "NO ARM WITH REHEARSAL ROWS" and NO_RULE == "NO RULE IS NAMED"
    assert fails([HOLDS]) == "IT FAILS: HOLDS THE LONG PROOFS" and fails([ADDS, NOT_NARROWER]) == "IT FAILS: ADDS AND NOT NARROWER"
    assert SETTINGS["rows2"] == {"start": "pre_r64", "models": list(MODELS)} and SETTINGS["rank_check"]["rank"] == 64 and SIGN_TEST_LEVEL == 0.05
    assert set(IS) == set(MODELS) and len(CANNOT_SAY) == 3 and EXPLORATORY in CANNOT_SAY[2][0].lower() + CANNOT_SAY[2][1:]


# ------------------------------------------------------------------------------------------------- the sets
PRETRAINING = [{"problem_id": f"q{index}", "kind": "stp", "proof_lines": 3 + index % 9, "lines": 3 + index % 9} for index in range(40)]      # the rows the start model was pretrained on, without text
TWIN = [{**_attempt(f"p{index}", number=1 + index % 2), "k": 1 + index % 8, "n": 8} for index in range(10)]                                 # the twin's ten one-shot rows
REHEARSE = rehearse_set(TWIN, rehearsal_rows(PRETRAINING, len(TWIN), SEED), SEED)                                                           # L4t's `rehearse`: 20 rows in one order
# L4t's `reward_rows`: four of the one-shot rows and two assembled ones, each kept by the rule, in the arm's order.
REWARD_ROWS = training_order([*({**row, "reward": 0.97, "draw": 0.1, "kept": True, "kept_by": "the rule"} for row in TWIN[:4]),
                              *({**_assembled(f"a{index}"), "k": 1, "n": 8, "reward": 0.97, "draw": 0.2, "kept": True, "kept_by": "the rule"} for index in range(2))], SEED)


def _draw_order(rows):
    """Rehearsal rows in the order L4t's rehearsal draw takes them: the smallest content hash of the seed and the row's id first."""
    return sorted(rows, key=lambda row: rank(SEED, REHEARSAL_LABEL, row["id"]))


def test_rehearse_only_is_the_rehearsal_rows_of_rehearse_and_nothing_else_in_that_sets_order():
    own = rehearse_only_set(REHEARSE)
    assert len(own) == 10 == len(REHEARSE) // 2 and all(row["origin"] == "pretraining" for row in own)
    assert own == [row for row in REHEARSE if row["origin"] == "pretraining"]                       # that set's order, the other rows left out, no row changed
    assert [row["id"] for row in own] == [row["id"] for row in training_order(own, SEED)]          # which is the arm's content-hash order of them
    assert not any(row["id"] in {twin["id"] for twin in TWIN} for row in own) and all(row["k"] is None for row in own)
    # NOTHING ELSE: a set of the twin's rows alone has nothing to train it on, and is refused.
    with pytest.raises(ValueError, match="L4t's `rehearse` set holds no rehearsal row among its 10: there is nothing to train `rehearse_only` on"):
        rehearse_only_set(TWIN)
    assert rehearse_only_set([*TWIN[:3], *own[:2]]) == own[:2]


def test_reward_rehearse_is_the_reward_rows_and_as_many_rehearsal_rows_as_it_has_one_shot_rows_the_first_of_the_same_hash_order():
    own = reward_rehearse_set(REWARD_ROWS, REHEARSE, PRETRAINING, SEED)
    of_the_rounds, rehearsal = [row for row in own if row["origin"] != "pretraining"], [row for row in own if row["origin"] == "pretraining"]
    # Every row of `reward_rows`, unchanged, and AS MANY rehearsal rows as it has ONE-SHOT rows (four: its two assembled rows bring none).
    assert sorted(row["id"] for row in of_the_rounds) == sorted(row["id"] for row in REWARD_ROWS) and all(row in REWARD_ROWS for row in of_the_rounds)
    assert len(rehearsal) == 4 == sum(row["origin"] == "attempt" for row in REWARD_ROWS) and len(own) == 6 + 4 and len({row["id"] for row in own}) == 10
    # THE FIRST OF THE SAME HASH ORDER: a prefix of `rehearse`'s rehearsal rows in the draw's order, so they are among `rehearse`'s own, by id and by place.
    drawn = _draw_order(row for row in REHEARSE if row["origin"] == "pretraining")
    assert _draw_order(rehearsal) == drawn[:4] and drawn == rehearsal_rows(PRETRAINING, 10, SEED) and _draw_order(rehearsal) == rehearsal_rows(PRETRAINING, 4, SEED)
    assert all(PRETRAINING[row["row_of_the_file"]]["problem_id"] == row["problem_id"] and row["id"] == f"{row['problem_id']}#pretraining" for row in rehearsal)
    assert {row["id"] for row in rehearsal} != {row["id"] for row in drawn[4:8]} and {row["id"] for row in rehearsal} < {row["id"] for row in drawn}
    # ONE order, the arm's content hash: not the rounds' rows and then the rehearsal rows.
    assert [row["id"] for row in own] == [row["id"] for row in training_order(own, SEED)] != [row["id"] for row in (*REWARD_ROWS, *_draw_order(rehearsal))]
    # A set with no one-shot row takes no rehearsal row; one with all ten takes all ten, which is `rehearse` with nothing left out.
    assert reward_rehearse_set([row for row in REWARD_ROWS if row["origin"] == "assembled"], REHEARSE, PRETRAINING, SEED) == [row for row in REWARD_ROWS if row["origin"] == "assembled"]
    assert sorted(row["id"] for row in reward_rehearse_set(TWIN, REHEARSE, PRETRAINING, SEED)) == sorted(row["id"] for row in REHEARSE)
    # ---- REFUSED: `rehearse` was not drawn from this file with this seed (a prefix of another draw would be other rows)
    with pytest.raises(ValueError, match="L4t's `rehearse` set holds 10 rehearsal rows, and they are not the 10 rows of the smallest hash among the 40 rows the start model was "
                                         "pretrained on .first that differs: q.*#pretraining.: that set was not drawn from this file with this seed"):
        reward_rehearse_set(REWARD_ROWS, rehearse_set(TWIN, rehearsal_rows(PRETRAINING, 10, SEED + 1), SEED), PRETRAINING, SEED)
    with pytest.raises(ValueError, match="that set was not drawn from this file with this seed"):
        reward_rehearse_set(REWARD_ROWS, REHEARSE, list(reversed(PRETRAINING)), SEED)              # the same problems at other places of the file
    with pytest.raises(ValueError, match="not the 9 rows of the smallest hash"):
        reward_rehearse_set(REWARD_ROWS, [row for row in REHEARSE if row["id"] != drawn[0]["id"]], PRETRAINING, SEED)      # a row of the draw is missing
    with pytest.raises(ValueError, match="`pre` was trained on 3 rows and 10 are asked for the rehearsal"):
        reward_rehearse_set(REWARD_ROWS, REHEARSE, PRETRAINING[:3], SEED)
    # ---- ... and an id that stands twice
    with pytest.raises(ValueError, match="1 ids stand twice in `reward_rehearse` .first: p0#round_r1_b1#statement#0."):
        reward_rehearse_set([*REWARD_ROWS, next(row for row in REWARD_ROWS if row["problem_id"] == "p0")], REHEARSE, PRETRAINING, SEED)


def test_the_rows_drawn_are_held_to_being_the_first_of_the_draw_were_the_draw_no_longer_nested(monkeypatch):
    """THE PREFIX is held, not assumed: a draw that gave OTHER rows for a smaller count (the last four of the order, say) is refused."""
    from rlvr_lean.domain.ladder_round import l4_rows2
    nested = l4_rows2.rehearsal_rows
    monkeypatch.setattr(l4_rows2, "rehearsal_rows", lambda pretraining, count, seed: nested(pretraining, 10, seed)[10 - count:] if count < 10 else nested(pretraining, count, seed))
    with pytest.raises(ValueError, match="the 4 rehearsal rows drawn for `reward_rehearse` are not the first 4 of the hash order `rehearse`'s were drawn in"):
        reward_rehearse_set(REWARD_ROWS, REHEARSE, PRETRAINING, SEED)
    monkeypatch.setattr(l4_rows2, "rehearsal_rows", nested)
    assert len(reward_rehearse_set(REWARD_ROWS, REHEARSE, PRETRAINING, SEED)) == 10


def test_a_rehearsal_row_is_named_by_its_id_and_its_place_and_one_that_carries_text_is_refused():
    own = reward_rehearse_set(REWARD_ROWS, REHEARSE, PRETRAINING, SEED)
    rehearsal = [row for row in own if row["origin"] == "pretraining"]
    assert all(set(row) == {"problem_id", "kind", "proof_lines", "lines", "id", "side", "origin", "k", "row_of_the_file"} for row in rehearsal)
    assert TEXT_FIELDS == ("theorem", "completion", "statement", "proof") and not any(field in row for row in rehearsal for field in TEXT_FIELDS)
    assert refuse_text(own, "the training set of `reward_rehearse`") is None and refuse_text(rehearse_only_set(REHEARSE), "the training set of `rehearse_only`") is None
    assert all(row["theorem"] and row["completion"] for row in own if row["origin"] != "pretraining")       # a round's own row is the model's proof: stored with its text, as ever
    for field in TEXT_FIELDS:
        with pytest.raises(ValueError, match=f"1 rehearsal rows of the training set of `rehearse_only` carry a text field .first: {rehearsal[0]['id']}.: a rehearsal row is a "
                                             "published proof, named by its id and its place in the file and never stored with its text"):
            refuse_text([{**rehearsal[0], field: "text"}, *rehearsal[1:]], "the training set of `rehearse_only`")


# ------------------------------------------------------------------------------------------ the three reads
def _solved(gained, lost, problems=392, before=245):
    return {"problems": problems, "resolved_after": before + gained - lost, "resolved_before": before, "gained": gained, "lost": lost, "sign_test_p": sign_test(gained, lost)}


ABOVE, THROUGH, BELOW = _kept(0.00734, 0.00205, 0.01358), _kept(0.00538, -0.001, 0.01272), _kept(-0.01014, -0.01903, -0.00117)
NOT_MEASURED = {"problems": 0, "mean": None, "low": None, "high": None}


def test_the_three_reads_are_the_arms_own_and_each_passes_or_fails_by_its_own_rule():
    # The spec's own figures, from L4t's stored rows: `rehearse` passes all three; `reward_rows` fails the third; `old_rule` fails the first two.
    rehearse = three_reads(_kept(0.00734, 0.00205, 0.01358), _solved(27, 38), _kept(0.00809, -0.00295, 0.01996), "pre_r64")
    assert (rehearse["passed"], rehearse["fails"], rehearse["not_read"], rehearse["all_three"], rehearse["gained_minus_lost"]) == (list(READS), [], [], True, -11)
    assert rehearse["said"] == ("ADDS: PASS (G', the fresh attempts: +0.00734 [+0.00205, +0.01358]); NOT NARROWER: PASS (gained 27, lost 38 (p = 0.214539)); HOLDS THE LONG PROOFS: "
                                "PASS (4 lines or more: +0.00809 [-0.00295, +0.01996]). ALL THREE")
    reward_rows = three_reads(_kept(0.00615, 0.00171, 0.01101), _solved(32, 36), _kept(-0.01014, -0.01903, -0.00117), "pre_r64")
    assert (reward_rows["fails"], reward_rows["all_three"], reward_rows["gained_minus_lost"]) == ([HOLDS], False, -4) and reward_rows["said"].endswith(". FAILS HOLDS THE LONG PROOFS")
    old_rule = three_reads(_kept(0.00538, 0.0, 0.01272), _solved(25, 53), _kept(-0.00636, -0.01786, 0.00622), "pre_r64")
    assert (old_rule["fails"], old_rule["passed"], old_rule["all_three"], old_rule["gained_minus_lost"]) == ([ADDS, NOT_NARROWER], [HOLDS], False, -28)
    assert old_rule["said"].endswith(". FAILS ADDS AND NOT NARROWER") and old_rule["against"] == "pre_r64" and old_rule["failed_checks"] == []
    # ---- ADDS: the interval on G' ABOVE zero; one that ends at zero is not above it
    for on_again, passes in ((ABOVE, True), (_kept(0.004, 0.00001, 0.011), True), (_kept(0.004, 0.0, 0.011), False), (THROUGH, False), (BELOW, False)):
        read = three_reads(on_again, _solved(30, 30), THROUGH, "pre_r64")
        assert read[KEY[ADDS]]["passes"] is passes and (ADDS in read["passed"]) is passes and (ADDS in read["fails"]) is (not passes)
        assert {key: read[KEY[ADDS]][key] for key in ("mean", "low", "high", "problems")} == on_again
    # ---- NOT NARROWER: NOT (lost more than gained with the sign test UNDER 0.05). It is L4b's own NARROWER, by its own function.
    assert sign_test(23, 39) > 0.05 > sign_test(23, 40)
    for breadth, passes in ((_solved(23, 40), False), (_solved(23, 39), True), (_solved(40, 23), True), (_solved(30, 30), True), (_solved(0, 0), True), (_solved(0, 6), False), (_solved(0, 5), True)):
        read = three_reads(ABOVE, breadth, THROUGH, "pre_r64")
        entry = read[KEY[NOT_NARROWER]]
        assert entry["passes"] is passes is (not breadth_beside(breadth, None, None, "pre_r64")["narrower"]) and entry["narrower"] is (not passes)
        assert (entry["gained"], entry["lost"], entry["gained_minus_lost"], read["gained_minus_lost"]) == (breadth["gained"], breadth["lost"], breadth["gained"] - breadth["lost"],
                                                                                                           breadth["gained"] - breadth["lost"])
    assert NARROWER == "NARROWER" and three_reads(ABOVE, _solved(23, 40), THROUGH, "pre_r64", level=0.01)[KEY[NOT_NARROWER]]["passes"] is True      # the level is the spec's 0.05
    # ---- HOLDS THE LONG PROOFS: the interval on the problems of 4 lines or more NOT WHOLLY BELOW zero; one that ends at zero is not wholly below it
    for long, passes in ((BELOW, False), (_kept(-0.01, -0.02, -0.00001), False), (_kept(-0.01, -0.02, 0.0), True), (THROUGH, True), (ABOVE, True)):
        read = three_reads(ABOVE, _solved(30, 30), long, "pre_r64")
        assert read[KEY[HOLDS]]["passes"] is passes and read["all_three"] is passes and read["fails"] == ([] if passes else [HOLDS])
    # ---- the three are read each on ITS figure: the interval of one is not the other's
    swapped = three_reads(BELOW, _solved(30, 30), ABOVE, "pre_r64")
    assert (swapped[KEY[ADDS]]["passes"], swapped[KEY[HOLDS]]["passes"], swapped["fails"]) == (False, True, [ADDS])
    # ---- nothing measured for a read (a run with one sampling of G; no problem of a known length): NOT READ, and the three are not all passed
    one_sampling = three_reads(NOT_MEASURED, _solved(0, 0, problems=1, before=1), NOT_MEASURED, "pre_r64")
    assert (one_sampling["passed"], one_sampling["fails"], one_sampling["not_read"], one_sampling["all_three"]) == ([NOT_NARROWER], [], [ADDS, HOLDS], None)
    assert one_sampling["said"].endswith(". NOT READ: ADDS, HOLDS THE LONG PROOFS") and "ADDS: not read (G', the fresh attempts: not measured)" in one_sampling["said"]
    assert three_reads(NOT_MEASURED, _solved(0, 6), ABOVE, "pre_r64")["all_three"] is False        # one that fails is a failure, whatever could not be read
    nothing = three_reads(NOT_MEASURED, {"problems": 0, "resolved_after": 0, "resolved_before": 0, "gained": 0, "lost": 0, "sign_test_p": None}, NOT_MEASURED, "pre_r64")
    assert (nothing["not_read"], nothing["all_three"], nothing["gained_minus_lost"]) == (list(READS), None, 0)
    # ---- A MODEL THAT FAILED A CHECK OF ITS OWN is INCONCLUSIVE: none of its reads is made, and what was measured is kept apart
    failed = three_reads(ABOVE, _solved(27, 38), ABOVE, "pre_r64", failed_checks=["its_training_ran"])
    assert (failed["failed_checks"], failed["passed"], failed["fails"], failed["not_read"], failed["all_three"], failed["gained_minus_lost"]) == (
        ["its_training_ran"], [], [], list(READS), None, None)
    assert all(failed[KEY[read]]["passes"] is None and set(failed[KEY[read]]) == {"what", "passes"} for read in READS)
    assert failed["said"] == "NOT READ: this model is INCONCLUSIVE (its_training_ran), and none of its reads is made" and failed["measured_and_not_read"][KEY[ADDS]]["low"] == 0.00205
    assert failed["measured_and_not_read"][KEY[NOT_NARROWER]]["gained_minus_lost"] == -11 and "measured_and_not_read" not in rehearse


# ---------------------------------------------------------------------------------------------- the branches
def _reads(adds=ABOVE, gained=30, lost=30, long=THROUGH, failed_checks=()):
    return three_reads(adds, _solved(gained, lost), long, "pre_r64", failed_checks)


def test_question_1_has_three_named_outcomes_and_is_read_on_rehearse_minus_rehearse_only():
    # ITS INTERVAL ABOVE ZERO: the loop's rows add beside the rehearsal, whatever `rehearse_only` does by itself.
    for alone in (_reads(ABOVE), _reads(THROUGH)):
        adds = question_1(_kept(0.005, 0.0012, 0.0091), alone, _reads())
        assert adds["name"] == LOOPS_ROWS_ADD and adds["rehearse_only_adds"] is alone[KEY[ADDS]]["passes"]
        assert "`rehearse` minus `rehearse_only` is above zero (+0.00500 [+0.00120, +0.00910]): the loop's rows add beside the rehearsal" in adds["reason"]
    # ELSE, IF `rehearse_only` ITSELF ADDS: the gain is a second pass over published proofs. An interval that ends at zero is not above it.
    for difference in (_kept(0.001, -0.004, 0.006), _kept(0.003, 0.0, 0.006), _kept(-0.004, -0.008, -0.001)):
        second = question_1(difference, _reads(ABOVE), _reads())
        assert second["name"] == SECOND_PASS and second["rehearse_only_adds"] is True and second["reason"].endswith("the gain is a second pass over published proofs, not shown to be the loop's")
        assert "`rehearse_only` itself ADDS (`rehearse_only` minus `pre_r64` on the same attempts: +0.00734 [+0.00205, +0.01358], above zero)" in second["reason"]
    # ELSE: not separated at this size, with what this difference resolves (half the width of its interval).
    for alone in (_reads(THROUGH), _reads(_kept(0.004, 0.0, 0.011)), _reads(BELOW)):
        neither = question_1(_kept(0.001, -0.004, 0.006), alone, _reads())
        assert neither["name"] == NOT_SEPARATED and neither["rehearse_only_adds"] is False and neither["resolves"] == 0.005
        assert "not separated at this size. This difference resolves 0.00500 per attempt (half the width of its interval)" in neither["reason"]
    # The difference is `rehearse` MINUS `rehearse_only`: read from the other side, the same rows give another outcome.
    assert question_1(_kept(-0.005, -0.0091, -0.0012), _reads(THROUGH), _reads())["name"] == NOT_SEPARATED
    # A model of the two that failed a check of its own: INCONCLUSIVE, by whichever it was. Nothing measured (one sampling of G): NOT READ.
    for only, both, names in ((_reads(failed_checks=["lean_answered"]), _reads(), ["rehearse_only"]), (_reads(), _reads(failed_checks=["its_training_ran"]), ["rehearse"]),
                              (_reads(failed_checks=["no_barred_row"]), _reads(failed_checks=["its_training_ran"]), ["rehearse", "rehearse_only"])):
        unread = question_1(_kept(0.005, 0.0012, 0.0091), only, both)
        assert unread["name"] == INCONCLUSIVE and list(unread["failed_checks"]) == names and "the difference stands on both models and is not read" in unread["reason"]
    assert question_1(NOT_MEASURED, _reads(NOT_MEASURED), _reads(NOT_MEASURED))["name"] == NOT_READ
    assert all(question_1(difference, alone, _reads())["name"] in QUESTION_1 for difference in (ABOVE, THROUGH, BELOW) for alone in (_reads(ABOVE), _reads(THROUGH)))


def test_question_2_is_the_three_reads_of_reward_rehearse_and_names_the_read_it_fails():
    assert question_2(_reads())["name"] == ALL_THREE and "it adds, keeps the breadth and holds the long proofs" in question_2(_reads())["reason"]
    for reads, failed in ((_reads(THROUGH), [ADDS]), (_reads(gained=23, lost=40), [NOT_NARROWER]), (_reads(long=BELOW), [HOLDS]), (_reads(THROUGH, long=BELOW), [ADDS, HOLDS]),
                          (_reads(THROUGH, 0, 9, BELOW), [ADDS, NOT_NARROWER, HOLDS])):
        second = question_2(reads)
        assert second["name"] == fails(failed) and second["fails"] == failed and second["name"].startswith("IT FAILS: ") and all(read in second["name"] for read in failed)
        assert "against `pre_r64`: ADDS: " in second["reason"]
    assert question_2(_reads(failed_checks=["it_still_writes_proofs"]))["name"] == INCONCLUSIVE and question_2(_reads(failed_checks=["x"]))["failed_checks"] == ["x"]
    # It fails no read that was made, and one could not be made: NOT READ. One that was made and failed is named, whatever was not read.
    assert question_2(_reads(NOT_MEASURED))["name"] == NOT_READ and "all three are asked" in question_2(_reads(NOT_MEASURED))["reason"]
    assert question_2(_reads(NOT_MEASURED, long=BELOW))["name"] == fails([HOLDS])


def test_the_arms_rule_is_the_specs_and_says_the_numbers_it_used():
    adds, not_separated = question_1(_kept(0.005, 0.0012, 0.0091), _reads(THROUGH), _reads()), question_1(_kept(0.001, -0.004, 0.006), _reads(THROUGH), _reads())
    second_pass = question_1(_kept(0.001, -0.004, 0.006), _reads(ABOVE), _reads())
    rehearse = _reads(ABOVE, 27, 38, _kept(0.00809, -0.00295, 0.01996))                           # as L4t's stored rows give it: all three, gained minus lost -11
    # ---- Question 1 reads A SECOND PASS: no arm with rehearsal rows, whatever `reward_rehearse` reads
    for candidate in (_reads(ABOVE, 40, 20), _reads(THROUGH), _reads(failed_checks=["its_training_ran"])):
        none = the_arms_rule(second_pass, candidate, rehearse)
        assert (none["name"], none["rule"]) == (NO_ARM, None) and "no arm with rehearsal rows is run, because it could not show that the loop adds" in none["reason"]
        assert "a second pass over published proofs helps" in none["reason"] and "exploratory" not in none
    # ---- OTHERWISE (the loop's rows add, or not separated): `reward_rehearse` if it passes the three reads AND its gained minus lost is AT LEAST `rehearse`'s
    for first in (adds, not_separated):
        for gained, lost in ((30, 34), (27, 38), (40, 20)):                                        # -4, -11 (a tie: AT LEAST), +20
            taken = the_arms_rule(first, _reads(ABOVE, gained, lost), rehearse)
            assert (taken["name"], taken["rule"], taken["exploratory"]) == ("reward_rehearse", "reward_rehearse", EXPLORATORY)
            assert f"Question 1 reads {first['name']} (not a second pass)" in taken["reason"] and f"`reward_rehearse` {gained - lost:+d} (gained {gained}, lost {lost}), `rehearse` -11 (gained 27, lost 38)" in taken["reason"]
            assert "`reward_rehearse` passes all three reads and its gained minus lost is at least `rehearse`'s: the arm's rule is `reward_rehearse`" in taken["reason"]
            assert "carries its own separation" in taken["reason"] and taken["numbers"]["rehearse"]["gained_minus_lost"] == -11
        # ... ELSE `rehearse`: it passes with a smaller gained minus lost (-12 against -11), or it fails a read
        under = the_arms_rule(first, _reads(ABOVE, 26, 38), rehearse)
        assert (under["name"], under["rule"]) == ("rehearse", "rehearse") and "`reward_rehearse` passes all three reads and its gained minus lost is under `rehearse`'s" in under["reason"]
        for candidate, said in ((_reads(ABOVE, 40, 20, BELOW), "it fails HOLDS THE LONG PROOFS"), (_reads(THROUGH, 40, 20), "it fails ADDS"), (_reads(ABOVE, 23, 40), "it fails NOT NARROWER")):
            kept = the_arms_rule(first, candidate, rehearse)
            assert (kept["name"], kept["rule"], kept["exploratory"]) == ("rehearse", "rehearse", EXPLORATORY) and f"`reward_rehearse` does not pass all three reads ({said})" in kept["reason"]
            assert "and `rehearse` passes all three: the arm's rule is `rehearse`" in kept["reason"]
        # ---- NO RULE IS NAMED, a case the section does not name: what the rule stands on could not be read, or neither rule passes all three
        for candidate, of_rehearse, said in ((_reads(failed_checks=["its_training_ran"]), rehearse, "the three reads of `reward_rehearse` could not all be made"),
                                             (_reads(ABOVE, 40, 20), _reads(failed_checks=["lean_answered"]), "the three reads of `rehearse` could not all be made"),
                                             (_reads(NOT_MEASURED), _reads(NOT_MEASURED), "the three reads of `reward_rehearse` and of `rehearse` could not all be made")):
            unnamed = the_arms_rule(first, candidate, of_rehearse)
            assert (unnamed["name"], unnamed["rule"]) == (NO_RULE, None) and said in unnamed["reason"] and "exploratory" not in unnamed
        neither = the_arms_rule(first, _reads(THROUGH), _reads(long=BELOW))
        assert neither["name"] == NO_RULE and "`reward_rehearse` fails ADDS and `rehearse` fails HOLDS THE LONG PROOFS" in neither["reason"]
        only_the_new = the_arms_rule(first, _reads(ABOVE, 20, 40 - 20), _reads(ABOVE, 30, 20, BELOW))
        assert only_the_new["name"] == NO_RULE and "so the section's rule names `rehearse`, which fails HOLDS THE LONG PROOFS" in only_the_new["reason"]
        assert the_arms_rule(first, _reads(ABOVE, 31, 20), _reads(ABOVE, 30, 20, BELOW))["name"] == "reward_rehearse"       # it passes, and is at least `rehearse`'s: named, whatever `rehearse` fails
    # Question 1 could not be read: no rule is named for an arm whose first question is open.
    for first in (question_1(NOT_MEASURED, _reads(NOT_MEASURED), _reads(NOT_MEASURED)), question_1(ABOVE, _reads(failed_checks=["x"]), _reads())):
        unnamed = the_arms_rule(first, _reads(ABOVE, 40, 20), rehearse)
        assert (unnamed["name"], unnamed["rule"]) == (NO_RULE, None) and f"Question 1 is {first['name']}" in unnamed["reason"]


# ------------------------------------------------------------------------------- the report, on hand-made rows
# Successes by problem in (the 32 attempts, the 61 more), in GOAL_IDS' order: three problems of 1 line, of 2-3, of 4-7, of 8 or more.
START = ([6, 6, 6, 4, 4, 4, 2, 2, 0, 0, 0, 0], [9, 9, 9, 6, 6, 6, 5, 4, 3, 2, 2, 1])               # the start model solves every problem; none of the last four in its first sampling
MORE_ON_AGAIN = ([6, 6, 6, 4, 4, 4, 2, 2, 0, 0, 0, 0], [9, 9, 9, 6, 6, 6, 5, 4, 9, 8, 8, 7])       # ... six more successes on each problem of G': it ADDS
LESS_ON_AGAIN = ([6, 6, 6, 4, 4, 4, 2, 2, 0, 0, 0, 0], [9, 9, 9, 6, 6, 6, 5, 4, 6, 5, 5, 4])       # ... three more: it ADDS, and less than the model above by three each
NO_LONG = ([6, 6, 6, 4, 4, 4, 0, 0, 0, 0, 0, 0], [9, 9, 9, 6, 6, 6, 0, 0, 0, 0, 0, 0])             # no problem of 4 lines or more solved: lost 6 (p = 0.03), and nothing on G'
PRETRAIN_HALF = [f"p{index}" for index in range(400) if half_of(f"p{index}", 0) == PRETRAIN]
TRAINED = {"rehearse_only": [{"id": f"{name}#pretraining", "problem_id": name, "origin": "pretraining", "k": None, "lines": 4 + index % 9} for index, name in enumerate(PRETRAIN_HALF[:30])],
           "reward_rehearse": [*({"id": f"{name}#a", "problem_id": name, "origin": "assembled" if index < 4 else "attempt", "k": 1 if index < 16 else 2, "lines": 2 + index % 7}
                                 for index, name in enumerate(LOOP_HALF[:24])),
                               *({"id": f"{name}#pretraining", "problem_id": name, "origin": "pretraining", "k": None, "lines": 4 + index % 9} for index, name in enumerate(PRETRAIN_HALF[:20]))]}
TRAINS = {name: {"model": name, "rows": len(own), "steps": -(-len(own) // 8), "stand_in_engine": False, "trained_from": "the stored adapter pre_r64",
                 "against_the_start_adapter": {name: CHANGED}} for name, own in TRAINED.items()}
LOSSES = {name: FALLING(len(own)) for name, own in TRAINED.items()}
PREPARE = {"stage": "l4", "label": LABEL, "check": "L4t2", "seed": 0, "start": "pre_r64", "start_adapter": "runs/ladder_l4_pretrain_r64_seed0/adapters/pre_r64",
           "l4t_run": "runs/ladder_l4_rows_pre_r64_seed0", "stand_in_engine": False, "models": list(MODELS), "start_recipe": {"rank": 64, "alpha": 128},
           "training_sets": {"rehearse_only": {"rows": 30, "rehearsal_rows": 30}, "reward_rehearse": {"rows": 44, "rows_of_reward_rows": 24, "rehearsal_rows": 20}},
           "goal_samplings": SAMPLINGS, "attempts_a_goal_problem": 93, "rung_episodes": 8, "sampling_seeds": {"rungs": 1002, "reach": 1001, "more": 1020},
           "contradicted_side_setting": "audit", "stored_models": {name: {"run": f"runs/{name}"} for name in ("pre_r64", *STORED)},
           "l4t_read": {"failed_checks": {name: [] for name in STORED}}}
AGAIN = goal_set_again(_made("pre", on_g=START)["goal"][0], GOAL_IDS)


def _model(on_g=START, **changed):
    return {**_made("pre", on_g=on_g), "stand_in_engine": False, **changed}


def _report(models=None, stored=None, prepare=PREPARE, trains=TRAINS, losses=LOSSES, trained=TRAINED, never=None, again=AGAIN):
    """The report with the start model's own rows standing in for every model that is not given."""
    models = {name: (models or {}).get(name) or _model() for name in MODELS}
    stored = {name: (stored or {}).get(name) or _model() for name in ("pre_r64", *STORED)}
    return build_rows2_report(prepare, trains, losses, trained, GROUPS, LENGTHS, stored, models, again, SETTINGS, EVALUATION, never)


def test_with_the_start_models_rows_standing_in_every_difference_is_zero_and_nothing_adds():
    report = _report()
    assert report["label"] == LABEL and report["stage"] == "l4" and report["check"] == "L4t2" and report["ok"] is True and all(line.startswith(f"{SAY}: ") for line in report["lines"])
    assert (report["start"], report["rank"], AGAIN, list(report["models"])) == ("pre_r64", 64, ["g4c", "g8a", "g8b", "g8c"], list(MODELS))
    for name in (*MODELS, *STORED):
        reads = report["the_three_reads"][name]
        assert (reads[KEY[ADDS]]["mean"], reads[KEY[ADDS]]["low"], reads[KEY[ADDS]]["high"], reads[KEY[ADDS]]["problems"]) == (0, 0, 0, 4)      # G', the 61 fresh attempts
        assert (reads[KEY[NOT_NARROWER]]["gained"], reads[KEY[NOT_NARROWER]]["lost"], reads[KEY[NOT_NARROWER]]["problems"]) == (0, 0, 12)
        assert (reads[KEY[HOLDS]]["mean"], reads[KEY[HOLDS]]["problems"]) == (0, 6)                # the six goal problems of 4 lines or more
        assert (reads["fails"], reads["passed"], reads["all_three"]) == ([ADDS], [NOT_NARROWER, HOLDS], False)       # an interval that ends at zero is not above it
    assert report["branches"] == {"question_1": NOT_SEPARATED, "question_2": fails([ADDS]), "the_arms_rule": NO_RULE}
    assert report["the_three_reads"]["goal_set_again"] == 4 and report["the_three_reads"]["goal_problems_of_4_lines_or_more"] == 6 and report["the_three_reads"]["fresh_attempts"] == 61
    assert report["question_1"]["difference"]["mean"] == 0 and report["question_1"]["resolves"] == 0 and "`reward_rehearse` fails ADDS and `rehearse` fails ADDS" in report["the_arms_rule"]["reason"]
    assert report["what_it_cannot_say"] == CANNOT_SAY and report["heldout"]["goal_set_again_by_length_group"] == {"1": 0, "2-3": 0, "4-7": 1, "8+": 3, "4_or_more": 4, "all": 4}


def test_the_reads_are_against_the_start_model_on_g_again_the_goal_set_and_the_long_proofs_and_the_stored_three_come_from_their_stored_rows():
    report = _report(models={"rehearse_only": _model(LESS_ON_AGAIN), "reward_rehearse": _model(MORE_ON_AGAIN)}, stored={"rehearse": _model(MORE_ON_AGAIN), "old_rule": _model(NO_LONG)})
    reads = report["the_three_reads"]
    # ADDS is read on G' over the 61 FRESH attempts alone (the second sampling), the model minus the start model: six more successes on each of four problems.
    adds = reads["rehearse"][KEY[ADDS]]
    assert (adds["problems"], adds["mean"], adds["passes"]) == (4, round(6 / 61, 5), True) and adds["low"] == adds["high"] == adds["mean"]
    assert reads["rehearse_only"][KEY[ADDS]]["mean"] == round(3 / 61, 5) and reads["rehearse_only"][KEY[ADDS]]["passes"] is True
    start, model = _model(), _model(MORE_ON_AGAIN)
    assert {key: adds[key] for key in ("mean", "low", "high", "problems")} == {key: per_attempt(model["goal"][1], start["goal"][1], AGAIN, 400, 0)[key] for key in ("mean", "low", "high", "problems")}
    # NOT NARROWER is read on the goal problems solved in ALL the attempts; HOLDS on the six problems of 4 lines or more over all the attempts.
    old = reads["old_rule"]
    assert (old[KEY[NOT_NARROWER]]["gained"], old[KEY[NOT_NARROWER]]["lost"], old[KEY[NOT_NARROWER]]["sign_test_p"], old[KEY[NOT_NARROWER]]["passes"]) == (0, 6, sign_test(0, 6), False)
    on_g = lambda entry: [{**first, "resolved": first["resolved"] + more["resolved"], "episodes": first["episodes"] + more["episodes"]} for first, more in zip(*entry["goal"])]      # noqa: E731
    assert {key: old[KEY[NOT_NARROWER]][key] for key in ("gained", "lost", "problems")} == {key: solved(on_g(_model(NO_LONG)), on_g(start), GOAL_IDS)[key] for key in ("gained", "lost", "problems")}
    long = per_attempt(on_g(_model(NO_LONG)), on_g(start), GOAL_IDS[6:], 400, 0)
    assert (old[KEY[HOLDS]]["problems"], old[KEY[HOLDS]]["mean"], old[KEY[HOLDS]]["high"], old[KEY[HOLDS]]["passes"]) == (6, long["mean"], long["high"], False) and long["high"] < 0
    assert old["fails"] == [ADDS, NOT_NARROWER, HOLDS] and old[KEY[ADDS]]["high"] < 0 and reads["rehearse"]["all_three"] is True and reads["reward_rows"]["fails"] == [ADDS]
    # THE STORED THREE are read from the rows given as stored, each under its own name; the two trained models from the rows measured here.
    lines = [line for line in report["lines"] if "THE THREE READS against `pre_r64`" in line]
    assert [line.split("`")[1] for line in lines] == [*MODELS, *STORED] and all("(measured by this run)" in line for line in lines[:2]) and all("(from the rows L4t's run stored)" in line for line in lines[2:])
    assert lines[4].endswith(f"`rehearse`, THE THREE READS against `pre_r64` (from the rows L4t's run stored): {reads['rehearse']['said']}") and reads["rehearse"]["said"].endswith(". ALL THREE")
    # QUESTION 1: `rehearse` minus `rehearse_only`, three more successes on each problem of G': above zero.
    first = report["question_1"]
    assert (first["name"], first["difference"]["mean"], first["difference"]["problems"]) == (LOOPS_ROWS_ADD, round(3 / 61, 5), 4) == (report["branches"]["question_1"], first["difference"]["low"], 4)
    # QUESTION 2 and THE ARM'S RULE: `reward_rehearse` passes all three and its gained minus lost (0) is at least `rehearse`'s (0).
    assert report["branches"]["question_2"] == ALL_THREE and report["the_arms_rule"]["rule"] == "reward_rehearse" == report["branches"]["the_arms_rule"]
    said = [line for line in report["lines"] if "QUESTION" in line or "THE ARM'S RULE" in line]
    assert len(said) == 3 and said[0].startswith(f"{SAY}: QUESTION 1, `rehearse` minus `rehearse_only` on G' over the 61 fresh attempts, paired by problem, 95% bootstrap over problems: "
                                                 "THE LOOP'S ROWS ADD BESIDE THE REHEARSAL. ")
    assert said[1].startswith(f"{SAY}: QUESTION 2, `reward_rehearse`, the three reads: IT ADDS, KEEPS THE BREADTH AND HOLDS THE LONG PROOFS. against `pre_r64`: ADDS: PASS")
    assert said[2].startswith(f"{SAY}: THE ARM'S RULE AFTER L4t2: `reward_rehearse`. Question 1 reads THE LOOP'S ROWS ADD BESIDE THE REHEARSAL (not a second pass); gained minus lost "
                              "against `pre_r64`: `reward_rehearse` +0 (gained 0, lost 0), `rehearse` +0 (gained 0, lost 0). ") and EXPLORATORY in said[2]
    assert "Question 1 (`rehearse` minus `rehearse_only` on G'): THE LOOP'S ROWS ADD BESIDE THE REHEARSAL. Question 2 (`reward_rehearse`): IT ADDS, KEEPS THE BREADTH AND HOLDS THE LONG PROOFS. " \
           "The arm's rule after L4t2: `reward_rehearse`." in report["headline"] and "`old_rule` fails ADDS and NOT NARROWER and HOLDS THE LONG PROOFS" in report["headline"]


def test_the_gain_is_a_second_pass_when_rehearse_only_adds_as_much_and_no_arm_with_rehearsal_rows_is_named():
    same = {"models": {"rehearse_only": _model(MORE_ON_AGAIN), "reward_rehearse": _model(MORE_ON_AGAIN)}, "stored": {"rehearse": _model(MORE_ON_AGAIN)}}
    report = _report(**same)
    assert report["question_1"]["difference"]["mean"] == 0 and report["question_1"]["rehearse_only_adds"] is True
    assert report["branches"] == {"question_1": SECOND_PASS, "question_2": ALL_THREE, "the_arms_rule": NO_ARM} and report["the_arms_rule"]["rule"] is None
    assert any(line.startswith(f"{SAY}: THE ARM'S RULE AFTER L4t2: NO ARM WITH REHEARSAL ROWS. Question 1 reads THE GAIN IS A SECOND PASS OVER PUBLISHED PROOFS: no arm") for line in report["lines"])
    # `rehearse_only` adds MORE than `rehearse` does: the difference lies below zero, and it is the same outcome (the loop's rows did not add beside it).
    below = _report(models={"rehearse_only": _model(MORE_ON_AGAIN)}, stored={"rehearse": _model(LESS_ON_AGAIN)})
    assert below["question_1"]["difference"]["high"] < 0 and below["branches"]["question_1"] == SECOND_PASS
    # Neither adds: not separated; the arm's rule is then by the three reads. Here `rehearse` (the start model's rows) does not add: no rule passes all three.
    assert _report()["branches"]["question_1"] == NOT_SEPARATED
    # `rehearse` adds and `rehearse_only` does not, the candidate fails a read: the arm's rule is `rehearse`.
    kept = _report(models={"reward_rehearse": _model(NO_LONG)}, stored={"rehearse": _model(MORE_ON_AGAIN)})
    assert kept["branches"] == {"question_1": LOOPS_ROWS_ADD, "question_2": fails([ADDS, NOT_NARROWER, HOLDS]), "the_arms_rule": "rehearse"} and kept["the_arms_rule"]["rule"] == "rehearse"


def test_each_trained_model_has_l4ts_four_checks_first_and_a_failed_one_makes_it_inconclusive_and_what_stands_on_it_not_read():
    report = _report(models={"rehearse_only": _model(LESS_ON_AGAIN), "reward_rehearse": _model(MORE_ON_AGAIN)}, stored={"rehearse": _model(MORE_ON_AGAIN)})
    for name in MODELS:
        own = report["models"][name]
        assert list(own["can_this_model_be_read"]) == ["what", "its_training_ran", "it_still_writes_proofs", "lean_answered", "no_barred_row"] and own["inconclusive"] is False
        assert all(entry["passes"] for key, entry in own["can_this_model_be_read"].items() if key != "what") and own["failed_checks"] == []
        assert own["the_three_reads"] is report["the_three_reads"][name] and own["model"].startswith(f"`{name}`: from `pre_r64`, one pass over {len(TRAINED[name])} rows: {IS[name]}")
        assert list(own["can_this_model_be_read"]["lean_answered"]["share_without_an_answer"]) == [f"rungs_{name}", f"goal_{name}_1", f"goal_{name}_2", "rungs_pre_r64", "goal_pre_r64_1", "goal_pre_r64_2"]
        lines = [line for line in report["lines"] if line.startswith(f"{SAY}: `{name}`")]
        assert all(f", CHECK {number}, " in line for number, line in enumerate(lines[:4], start=1)) and all(line.endswith(": PASS") for line in lines[:4])
        assert f"{len(TRAINED[name])} rows, {-(-len(TRAINED[name]) // 8)} optimizer steps, from `pre_r64`), CHECK 1, its training ran" in lines[0]
    # The rehearsal rows are of the `pretrain` half and are named as such; `rehearse_only` has no row of the rounds.
    barred = report["models"]["rehearse_only"]["can_this_model_be_read"]["no_barred_row"]
    assert (barred["rows"], barred["rows_of_the_rounds"], barred["rehearsal_rows_of_the_pretrain_half"], barred["passes"]) == (30, 0, 30, True)
    assert any("`rehearse_only`, CHECK 4" in line and "0 of the `pretrain` half among its 0 rounds' rows; its 30 rehearsal rows ARE of the `pretrain` half" in line for line in report["lines"])
    # ---- `rehearse_only` fails a check (a rehearsal row that is NOT of the `pretrain` half): INCONCLUSIVE, by itself; QUESTION 1 stands on it and is not read; no rule is named
    stray = [{**TRAINED["rehearse_only"][0], "problem_id": LOOP_HALF[0]}, *TRAINED["rehearse_only"][1:]]
    failed = _report(models={"rehearse_only": _model(LESS_ON_AGAIN), "reward_rehearse": _model(MORE_ON_AGAIN)}, stored={"rehearse": _model(MORE_ON_AGAIN)},
                     trained={**TRAINED, "rehearse_only": stray})
    own = failed["models"]["rehearse_only"]
    assert (own["inconclusive"], own["failed_checks"], own["secondary"]) == (True, ["no_barred_row"], None) and own["measured_and_not_read"]["secondary"]["trained_on"]["rows"] == 30
    assert own["the_three_reads"]["all_three"] is None and own["the_three_reads"]["said"].startswith("NOT READ: this model is INCONCLUSIVE (no_barred_row)")
    assert failed["branches"] == {"question_1": INCONCLUSIVE, "question_2": ALL_THREE, "the_arms_rule": NO_RULE} and failed["question_1"]["failed_checks"] == {"rehearse_only": ["no_barred_row"]}
    assert len([line for line in failed["lines"] if line.startswith(f"{SAY}: `rehearse_only`")]) == 5 and not any("`rehearse_only`, SECONDARY" in line for line in failed["lines"])
    assert failed["models"]["reward_rehearse"]["inconclusive"] is False and any("`reward_rehearse`, SECONDARY" in line for line in failed["lines"])       # the other model is still read
    assert "`rehearse_only` INCONCLUSIVE" in failed["headline"] and failed["ok"] is True
    # ---- `reward_rehearse` fails a check (its loss rose): Question 2 is INCONCLUSIVE and no rule is named; Question 1 is still read
    risen = _report(models={"rehearse_only": _model(LESS_ON_AGAIN), "reward_rehearse": _model(MORE_ON_AGAIN)}, stored={"rehearse": _model(MORE_ON_AGAIN)},
                    losses={**LOSSES, "reward_rehearse": list(reversed(FALLING(44)))})
    assert risen["branches"] == {"question_1": LOOPS_ROWS_ADD, "question_2": INCONCLUSIVE, "the_arms_rule": NO_RULE} and risen["models"]["reward_rehearse"]["failed_checks"] == ["its_training_ran"]
    # ---- a row of the rounds that is a held-out problem, or of the `pretrain` half: the fourth check fails
    for problem_id in (GOAL_IDS[0], OF_THE_OTHER_HALF):
        barred_row = [{**TRAINED["reward_rehearse"][5], "problem_id": problem_id}, *TRAINED["reward_rehearse"][:5], *TRAINED["reward_rehearse"][6:]]
        assert _report(trained={**TRAINED, "reward_rehearse": barred_row})["models"]["reward_rehearse"]["failed_checks"] == ["no_barred_row"]
    # ---- A STORED MODEL L4t's OWN REPORT READ AS INCONCLUSIVE is not read here either: its three reads, and what stands on it
    of_l4t = {**PREPARE, "l4t_read": {"failed_checks": {"old_rule": [], "reward_rows": ["its_training_ran"], "rehearse": ["lean_answered"]}}}
    unread = _report(models={"rehearse_only": _model(LESS_ON_AGAIN), "reward_rehearse": _model(MORE_ON_AGAIN)}, stored={"rehearse": _model(MORE_ON_AGAIN)}, prepare=of_l4t)
    assert unread["the_three_reads"]["rehearse"]["failed_checks"] == ["lean_answered"] and unread["the_three_reads"]["reward_rows"]["said"].startswith("NOT READ: this model is INCONCLUSIVE (its_training_ran)")
    assert unread["the_three_reads"]["old_rule"]["failed_checks"] == [] and unread["branches"] == {"question_1": INCONCLUSIVE, "question_2": ALL_THREE, "the_arms_rule": NO_RULE}
    assert unread["question_1"]["failed_checks"] == {"rehearse": ["lean_answered"]} and "`rehearse` INCONCLUSIVE" in unread["headline"]
    assert all(not report_["models"][name]["inconclusive"] for report_ in (unread,) for name in MODELS)         # the two trained models' own checks are their own


def test_a_set_lean_did_not_answer_is_not_to_be_read_and_the_model_on_it_is_inconclusive():
    unanswered = _model(MORE_ON_AGAIN)
    unanswered["goal"][1] = [{**row, "attempts_without_an_answer": 20} for row in unanswered["goal"][1]]
    report = _report(models={"reward_rehearse": unanswered})
    assert report["ok"] is False and report["not_to_be_read"] == ["goal_reward_rehearse_2"] and "NOT TO BE READ" in report["lines"][-1] and "NOT TO BE READ" in report["headline"]
    assert report["models"]["reward_rehearse"]["failed_checks"] == ["lean_answered"] and report["models"]["rehearse_only"]["failed_checks"] == []
    assert report["branches"]["question_2"] == INCONCLUSIVE and set(report["attempts"]) == {f"{part}_{name}" for name in ("pre_r64", *STORED, *MODELS) for part in ("rungs", "goal") if part == "rungs"} | {
        f"goal_{name}_{index}" for name in ("pre_r64", *STORED, *MODELS) for index in (1, 2)}
    # The START MODEL's set stands under every read: one Lean did not answer makes both trained models INCONCLUSIVE.
    start = _model()
    start["rungs"] = [{**row, "attempts_without_an_answer": 8} for row in start["rungs"]]
    report = _report(stored={"pre_r64": start})
    assert report["not_to_be_read"] == ["rungs_pre_r64"] and all(report["models"][name]["failed_checks"] == ["lean_answered"] for name in MODELS)


def test_the_secondary_reads_are_l4ts_of_each_trained_model_by_itself_against_the_start_model_with_l4ts_three_beside():
    report = _report(models={"rehearse_only": _model(LESS_ON_AGAIN), "reward_rehearse": _model(MORE_ON_AGAIN)}, stored={"rehearse": _model(MORE_ON_AGAIN), "old_rule": _model(NO_LONG)},
                     never=["g8a", "g8c", "not_a_goal_problem"])
    for name in MODELS:
        secondary = report["models"][name]["secondary"]
        # L4t's own secondary reads, of a model read BY ITSELF against the start model: no read against a model in `without`'s place.
        assert list(secondary) == ["the_goal_set_again", "by_length_group", "goal_problems_solved_by_attempts_alone", "the_three_rungs", "distinct_attempts", "verified_proof_lines",
                                   "never_solved_before", "trained_on"]
        assert set(secondary["by_length_group"]) == {"what", "problems", "minus_pre_r64", "against_pre_r64"} and set(secondary["the_three_rungs"]) == {"what", "problems", "minus_pre_r64"}
        assert set(secondary["distinct_attempts"]) == set(secondary["verified_proof_lines"]) == set(secondary["goal_problems_solved_by_attempts_alone"]) == {"what", "pre_r64", *STORED, name}
        # The three reads are the SAME figures its secondary reads give: G' over the fresh attempts, and the problems of 4 lines or more.
        reads = report["the_three_reads"][name]
        assert {key: reads[KEY[ADDS]][key] for key in ("mean", "low", "high", "problems")} == {key: secondary["the_goal_set_again"][key] for key in ("mean", "low", "high", "problems")}
        assert {key: reads[KEY[HOLDS]][key] for key in ("mean", "low", "high", "problems")} == {key: secondary["by_length_group"]["minus_pre_r64"]["4_or_more"][key] for key in ("mean", "low", "high", "problems")}
        assert "seen_twice" not in secondary["the_goal_set_again"] and secondary["trained_on"]["rows"] == len(TRAINED[name])
        unsolved = secondary["never_solved_before"]
        assert (unsolved["given"], unsolved["goal_problems"], unsolved["solved"], unsolved["not_goal_problems_of_this_run"]) == (True, 2, 2, 1)
        assert (unsolved["solved_by_old_rule"], unsolved["solved_by_reward_rows"], unsolved["solved_by_rehearse"]) == (0, 2, 2)
        assert unsolved["what"].endswith("; `old_rule`'s, `reward_rows`'s and `rehearse`'s own counts stand beside")
        lines = [line for line in report["lines"] if line.startswith(f"{SAY}: `{name}`, SECONDARY, ")]
        assert len(lines) == 9 and f"G' (4 goal problems `pre_r64` does not solve in its first sampling), successes per attempt over the 61 attempts of the second sampling, `{name}` minus `pre_r64`" in lines[0]
        assert "4 lines or more (6): " in lines[1] and f"goal problems solved at least once, `{name}` against `pre_r64`: " in lines[2] and "the three rungs (8 episodes a problem)" in lines[4]
        assert all(f"`{other}` " in lines[3] and f"`{other}` " in lines[5] and f"`{other}` " in lines[6] for other in ("pre_r64", *STORED, name))
        assert f"of the 2 listed, `{name}` solves 2 (g8a, g8c); `old_rule` solves 0 and `reward_rows` 2 and `rehearse` 2; 1 more ids of the file are not goal problems of this run" in lines[7]
        assert "the rows it was trained on: " in lines[8] and ("by k none" in lines[8]) is (name == "rehearse_only")
    assert report["models"]["rehearse_only"]["secondary"]["trained_on"]["rows_by_origin"] == {"pretraining": 30} and report["models"]["reward_rehearse"]["secondary"]["trained_on"]["rehearsal_rows"] == 20
    # THE ORDER OF THE LINES: the first; each model's four checks; the three reads of the five; the two questions; the arm's rule; then the secondary reads.
    heads = [("CHECK" if ", CHECK " in line else "READS" if "THE THREE READS against" in line else "QUESTION 1" if "QUESTION 1," in line else "QUESTION 2" if "QUESTION 2," in line else
              "RULE" if "THE ARM'S RULE AFTER L4t2:" in line else "SECONDARY" if ", SECONDARY, " in line else "FIRST") for line in report["lines"]]
    assert heads == ["FIRST", *["CHECK"] * 8, *["READS"] * 5, "QUESTION 1", "QUESTION 2", "RULE", *["SECONDARY"] * 18]
    first = report["lines"][0]
    assert "L4t2, TWO MORE ONE-CHANGE CHECKS, ON WHAT L4T LEFT OPEN, seed 0. NO NEW ROUND" in first and "EVERY MODEL IS AT RANK 64, the start adapter's, trained FROM `pre_r64`" in first
    assert "`rehearse_only`: the 30 rehearsal rows of L4t's `rehearse` set and nothing else" in first and "`reward_rehearse`: 44 rows, L4t's `reward_rows` set (24) and as many rehearsal rows as it has one-shot rows (20" in first
    assert "ADDS (G', 4 goal problems `pre_r64` does not solve in its first sampling, successes per attempt over the 61 fresh attempts: the interval above zero)" in first
    assert "HOLDS THE LONG PROOFS (the 6 goal problems of 4 lines or more, successes per attempt over the 93 attempts: the interval not wholly below zero)" in first


def test_a_run_with_one_sampling_of_g_reads_no_fresh_attempts_and_names_no_rule():
    one = {**PREPARE, "goal_samplings": SAMPLINGS[:1], "attempts_a_goal_problem": 32}
    first_alone = lambda entry: {**entry, "goal": entry["goal"][:1]}      # noqa: E731
    report = build_rows2_report(one, TRAINS, LOSSES, TRAINED, GROUPS, LENGTHS, {name: first_alone(_model()) for name in ("pre_r64", *STORED)},
                                {name: first_alone(_model(MORE_ON_AGAIN)) for name in MODELS}, AGAIN, SETTINGS, EVALUATION, None)
    assert report["ok"] is True and report["branches"] == {"question_1": NOT_READ, "question_2": NOT_READ, "the_arms_rule": NO_RULE} and report["the_three_reads"]["fresh_attempts"] is None
    assert all(report["the_three_reads"][name]["not_read"] == [ADDS] and report["the_three_reads"][name]["all_three"] is None for name in (*MODELS, *STORED))
    assert any("the stored models have ONE sampling of G here (a smoke run)" in line and "ADDS and Question 1 are not read" in line for line in report["lines"])
    assert any("QUESTION 1, `rehearse` minus `rehearse_only` on G' over fresh attempts, which this run does not have" in line for line in report["lines"])
    with pytest.raises(ValueError, match="1 problems of G' are not goal problems of this run .first: nowhere."):
        _report(again=[*AGAIN, "nowhere"])


def test_the_two_reports_read_a_model_by_the_same_functions():
    """L4t2's report calls L4t's own: the world a model is read on, its four checks, its secondary reads. One body each."""
    reporting = {path.name: path.read_text() for path in (PACKAGE / "reporting").glob("ladder_l4_rows*.py")}
    for name in ("def world_of(", "def the_four_checks(", "def check_lines(", "def secondary_reads(", "def secondary_lines("):
        assert [module for module, text in reporting.items() if name in text] == ["ladder_l4_rows.py"], name
        assert name[4:] in reporting["ladder_l4_rows2.py"] and reporting["ladder_l4_rows.py"].count(name[4:]) == 2          # defined once, called once by L4t's own report
    assert isinstance(world_of(PREPARE, {"pre_r64": _model()}, GROUPS, LENGTHS, AGAIN, SETTINGS, EVALUATION), World)
    assert callable(the_four_checks) and callable(check_lines) and callable(secondary_reads) and callable(secondary_lines)
    # No second copy of a rule: the sign test, the interval and G' are not written again in L4t2's two modules.
    domain = (PACKAGE / "domain" / "ladder_round" / "l4_rows2.py").read_text()
    assert "breadth_beside(solved, None, None, start, level)" in domain and "rehearsal_rows(pretraining, len(one_shot), seed)" in domain and "training_order(rows, seed)" in domain
    for written_again in ("math.comb", "paired_bootstrap", "sign_test(", "hashlib", "random"):
        assert written_again not in domain and written_again not in reporting["ladder_l4_rows2.py"], written_again
