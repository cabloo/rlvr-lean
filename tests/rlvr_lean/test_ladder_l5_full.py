"""L5f, the rules on hand-made rows: the names and the numbers; the seed and the escalation (L5s's own two); the check
that this run's sets are, id for id and order for order, the ones L5d's run of the seed stored and trained, and its
refusal; THE STACK at its edges, on whole numbers, INCONCLUSIVE and NOT READ; THE LEVEL with L5d's run read and not
read; what follows. Spec: docs/spec/ladder-loop.spec.md, "L5f: the three kinds again, from the twice-pretrained
model". Pure: no file, no GPU, no network. The stage end to end is `test_ladder_l5_full_stage.py`."""

from fractions import Fraction

import pytest

from rlvr_lean.domain.ladder_round import l5_full, l5_stock
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ
from rlvr_lean.domain.ladder_round.l5 import DEEP, MODELS, PUBLISHED, SHALLOW
from rlvr_lean.domain.ladder_round.l5_full import (
    A_COST,
    COSTS,
    ESCALATE,
    FURTHER_SEEDS,
    NO_FURTHER_SEED,
    NOT_SHOWN_AT_ONE,
    START,
    hold_to_theirs,
    the_level,
    the_sets_are_theirs,
    the_stack,
    what_follows,
)


def test_the_names_and_the_numbers_are_the_sections():
    assert START == "second_full" and MODELS == (PUBLISHED, SHALLOW, DEEP)                    # L5d's three kinds, from L4p's whole second pass
    assert ESCALATE == "THE MODEL'S OWN PROOFS DO NOT FALL BEHIND PUBLISHED ONES AFTER A SECOND PASS: ESCALATE"
    assert (COSTS, NOT_SHOWN_AT_ONE) == ("THEY COST AFTER A SECOND PASS AT THIS SEED", "NOT SHOWN AT ONE SEED")
    assert l5_full.STACK == (ESCALATE, COSTS, NOT_SHOWN_AT_ONE, INCONCLUSIVE, NOT_READ) and l5_full.FOLLOWS == (FURTHER_SEEDS, NO_FURTHER_SEED, NOT_READ)
    assert A_COST == Fraction(85, 10000) and (l5_full.OF_A_FIGURE, l5_full.OF_THE_PRIMARY) == (0.0034, 0.0042)      # "below zero by more than 0.0085 (twice the standard deviation above)"
    # The scout's seed rule and the read against a start model are L5s's own functions, not copies.
    assert l5_full.the_seed is l5_stock.the_seed and l5_full.fixed_seeds is l5_stock.fixed_seeds and l5_full.against_the_start is l5_stock.against_the_start
    assert l5_full.fixed_seeds([3], [4, 5], False) == (3,) and l5_full.the_seed(3, [3], [4, 5], False) is None
    with pytest.raises(ValueError, match=r"seed 4 is one of the further seeds \[4, 5\]"):
        l5_full.the_seed(4, [3], [4, 5], False)


# ------------------------------------------------------------------------------------------------ the check
def _rows(*ids):
    return [{"id": key, "problem_id": key.split("#")[0]} for key in ids]


SETS = {"published": _rows("p1", "p2", "p3"), "published_1": _rows("p2", "p1", "p3"), "published_2": _rows("p3", "p2", "p1"),
        "deep": _rows("d1#l5", "p1", "p2"), "deep_1": _rows("p2", "d1#l5", "p1"), "deep_2": _rows("p1", "p2", "d1#l5")}
STORED = {name: [row["id"] for row in own] for name, own in SETS.items()}
RECORDED = {name: sorted(STORED[name]) for name in SETS if name[-1].isdigit()}                # a training loop's record: the set's rows, each once, in the order it trained


def test_the_sets_are_l5ds_when_every_set_is_its_stored_one_id_for_id_and_order_for_order_and_each_of_its_trainings_was_of_them():
    held = the_sets_are_theirs(SETS, STORED, RECORDED)
    assert (held["sets"], held["trainings_of_theirs"], held["rows"], held["sets_that_are_not_the_stored_ones"], held["first_that_differs"], held["sets_that_are_not_what_was_trained"],
            held["passes"]) == (6, 4, {name: 3 for name in SETS}, [], None, [], True)
    assert hold_to_theirs(SETS, STORED, RECORDED) == held
    assert the_sets_are_theirs({}, {}, RECORDED)["passes"] is False and the_sets_are_theirs(SETS, STORED, {})["passes"] is False      # nothing held is no hold


@pytest.mark.parametrize("stored, first", [
    ({"published_2": ["p1", "p2", "p3"]}, {"set": "published_2", "place": 0, "here": "p3", "theirs": "p1", "rows_here": 3, "rows_of_theirs": 3}),      # the same rows in another ORDER
    ({"deep": ["d1#l5", "p1", "p9"]}, {"set": "deep", "place": 2, "here": "p2", "theirs": "p9", "rows_here": 3, "rows_of_theirs": 3}),                # another row
    ({"deep_1": ["p2", "d1#l5"]}, {"set": "deep_1", "place": 2, "here": "p1", "theirs": None, "rows_here": 3, "rows_of_theirs": 2}),                   # a row short
    ({"published": None}, {"set": "published", "place": 0, "here": "p1", "theirs": None, "rows_here": 3, "rows_of_theirs": 0})])                      # no such set in that run
def test_a_set_that_is_not_the_stored_one_fails_the_check_naming_the_first_place_and_the_first_step_refuses(stored, first):
    held = the_sets_are_theirs(SETS, {**STORED, **stored}, RECORDED)
    assert (held["passes"], held["sets_that_are_not_the_stored_ones"], held["first_that_differs"], held["sets_that_are_not_what_was_trained"]) == (False, [first["set"]], first, [])
    with pytest.raises(ValueError, match=rf"the sets \['{first['set']}'\] are not the ones L5d's run of this seed stored .`{first['set']}`: 3 rows here and {first['rows_of_theirs']} "
                                         rf"there; first that differs at place {first['place']}: .*its sets are held to be that run's, id for id and order for order. Refused"):
        hold_to_theirs(SETS, {**STORED, **stored}, RECORDED)


@pytest.mark.parametrize("recorded", [{"deep_1": ["p2", "d1#l5", "p9"]},              # a row that is none of the set's
                                      {"deep_1": ["p2", "d1#l5", "p1", "p1"]},        # a row twice
                                      {"deep_1": ["p2", "d1#l5"]},                    # a row not trained on
                                      {"deep_1": None},                               # no record of that training
                                      {"wide_1": ["p1", "p2", "p3"]}])                # a training of theirs that is of no set here
def test_a_training_of_theirs_that_was_not_of_these_rows_each_once_fails_the_check(recorded):
    held = the_sets_are_theirs(SETS, STORED, {**RECORDED, **recorded})
    assert (held["passes"], held["sets_that_are_not_the_stored_ones"], held["first_that_differs"], held["sets_that_are_not_what_was_trained"]) == (False, [], None, list(recorded))
    with pytest.raises(ValueError, match=rf"L5d's trainings of \['{list(recorded)[0]}'\] were not of these rows, each once"):
        hold_to_theirs(SETS, STORED, {**RECORDED, **recorded})


# ------------------------------------------------------------------------------------------------- the reads
def _of(successes, attempts=1000):
    return {"successes": successes, "attempts": attempts, "per_attempt": successes / attempts, "per_1000": 1000 * successes / attempts, "sampling_noise": 0.01}


def _figures(published, shallow, deep, attempts=1000):
    return {PUBLISHED: _of(published, attempts), SHALLOW: _of(shallow, attempts), DEEP: _of(deep, attempts)}


@pytest.mark.parametrize("shallow, deep, name", [(100, 110, ESCALATE), (90, 110, ESCALATE),              # above zero, and EXACTLY zero: at or above zero
                                                 (99, 100, NOT_SHOWN_AT_ONE), (90, 93, NOT_SHOWN_AT_ONE),  # just below zero, and EXACTLY 0.0085 below it: not "by more than"
                                                 (90, 92, COSTS), (0, 0, COSTS)])                          # below zero by more than 0.0085
def test_the_stack_has_three_outcomes_and_each_edge_is_read_on_whole_numbers(shallow, deep, name):
    stack = the_stack([], _figures(100, shallow, deep))
    assert (stack["name"], stack["difference"], stack["a_cost"], stack["standard_deviation_of_the_one_seed_difference"]) == (name, round(((shallow + deep) / 2 - 100) / 1000, 6), 0.0085, 0.0042)
    assert (stack[PUBLISHED], stack[SHALLOW], stack[DEEP]) == ({"successes": 100, "attempts": 1000}, {"successes": shallow, "attempts": 1000}, {"successes": deep, "attempts": 1000})
    assert "the mean of `shallow` and `deep`, minus `published`, all three from `second_full`, per 1,000 attempts on all of G: " f"{(shallow + deep) / 2 - 100:+.2f}" in stack["reason"]
    follows = what_follows(stack, [4, 5])
    assert follows["name"] == (FURTHER_SEEDS if name == ESCALATE else NO_FURTHER_SEED) and (follows.get("seeds") == [4, 5]) is (name == ESCALATE)


def test_the_stack_is_decided_on_exact_fractions_never_on_a_rounded_figure():
    # 392 goal problems of 32 attempts: 12,544 attempts, so 0.0085 an attempt is 106.624 successes. Half a success under zero: NOT SHOWN; 106.5 under: NOT SHOWN; 107 under: THEY COST.
    names = [the_stack([], _figures(1382, 1247, deep, 12544))["name"] for deep in (1517, 1516, 1304, 1303)]
    assert names == [ESCALATE, NOT_SHOWN_AT_ONE, NOT_SHOWN_AT_ONE, COSTS]
    # The rates are compared, not the counts: models of other numbers of attempts (never so in a run that is read) are read by their shares.
    mixed = {PUBLISHED: _of(10, 100), SHALLOW: _of(100, 1000), DEEP: _of(200, 2000)}
    assert the_stack([], mixed)["name"] == ESCALATE and the_stack([], {**mixed, DEEP: _of(199, 2000)})["name"] == NOT_SHOWN_AT_ONE


def test_the_stack_is_inconclusive_when_a_check_failed_and_not_read_without_attempts_and_nothing_then_follows():
    failed = the_stack(["`deep`: lean_answered"], _figures(100, 100, 100))
    assert failed == {"name": INCONCLUSIVE, "reason": "a check of this run failed (`deep`: lean_answered): the stack is not read"}
    not_read = the_stack([], {PUBLISHED: _of(100), SHALLOW: {"successes": 0, "attempts": 0}, DEEP: _of(100)})
    assert not_read == {"name": NOT_READ, "reason": "a model has no attempt on the goal set: the stack is NOT READ"}
    assert the_stack(["x"], {PUBLISHED: _of(100), SHALLOW: {"successes": 0, "attempts": 0}, DEEP: _of(100)})["name"] == INCONCLUSIVE      # a failed check is said first
    for stack in (failed, not_read):
        assert what_follows(stack, [4, 5]) == {"name": NOT_READ, "reason": f"the stack reads {stack['name']}: nothing is named for what follows"}
    assert what_follows({"name": COSTS}, [4, 5]) == {"name": NO_FURTHER_SEED, "reason": f"the stack reads {COSTS}: no further seed; the outcome needs a decision, with its figures"}
    escalated = what_follows({"name": ESCALATE}, [4, 5])
    assert "seeds [4, 5] of this stage, each after L5d's own run of that seed; three seeds conclude, with L5d's primary rule and names" in escalated["reason"]
    assert "a step that is not built here" in escalated["reason"]


def _theirs(published=91, shallow=99, deep=110):
    return {"read": True, "run": "ladder_l5_draws_pre_r64_seed3", "figures": _figures(published, shallow, deep), "the_primarys_difference": 0.0135}


def test_the_level_is_each_model_minus_the_start_model_and_minus_the_same_kind_of_l5ds_run_and_names_nothing():
    level = the_level(_figures(112, 115, 118), _of(110), _of(78), _theirs())
    assert level["minus_the_start"] == {PUBLISHED: 0.002, SHALLOW: 0.005, DEEP: 0.008}
    assert level["minus_the_same_kind_of_l5d"] == {PUBLISHED: 0.021, SHALLOW: 0.016, DEEP: 0.008} and (level["l5d_read"], level["why"]) == (True, None)
    assert level["highest"] == {"model": DEEP, "per_attempt": 0.118}
    assert level["beside"] == {"pre_r64": 0.078, "second_full": 0.11, "l5d": {PUBLISHED: 0.091, SHALLOW: 0.099, DEEP: 0.11}}
    assert "name" not in level and "One seed names nothing" in level["what"]
    # L5d's run of the seed NOT READ: the second difference is not made, and why is said; the first needs no other run.
    alone = the_level(_figures(112, 115, 118), _of(110), _of(78), {"read": False, "why": "it is not on the box"})
    assert alone["minus_the_start"] == level["minus_the_start"] and alone["minus_the_same_kind_of_l5d"] == {PUBLISHED: None, SHALLOW: None, DEEP: None}
    assert (alone["l5d_read"], alone["why"], alone["beside"]["l5d"], alone["highest"]) == (False, "it is not on the box", None, level["highest"])
    # Two models level: the first of the three kinds' order is named. A model without an attempt gives no difference and is never the highest.
    assert the_level(_figures(118, 118, 100), _of(110), _of(78), _theirs())["highest"]["model"] == PUBLISHED
    empty = the_level({**_figures(112, 115, 118), DEEP: {"successes": 0, "attempts": 0}}, _of(110), _of(78), _theirs())
    assert (empty["minus_the_start"][DEEP], empty["minus_the_same_kind_of_l5d"][DEEP], empty["highest"]["model"]) == (None, None, SHALLOW)
