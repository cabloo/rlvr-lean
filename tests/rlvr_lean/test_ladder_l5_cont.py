"""L5c, the rules on hand-made rows: the names and the numbers; the seed and the escalation (L5s's own two); the three
sets along the seed's order of the pretraining file and what they are held to (L5f's stored `published`, L5s's rule,
L5f's four trainings as four orders of `wide`); each set in the orders of its order seeds, `wide`'s four beside L5f's
four; the first check and what fails it; why L5f's run cannot carry the stage; the hold of its five adapters; WHAT MADE
IT and the three reads at their edges, on whole numbers, INCONCLUSIVE and NOT READ; what follows and its candidate.
Spec: docs/spec/ladder-loop.spec.md, "L5c: the continuation". Pure: no file, no GPU, no network. The stage end to
end is `test_ladder_l5_cont_stage.py`."""

from fractions import Fraction

import pytest

from rlvr_lean.domain.ladder_round import l5_cont, l5_stock
from rlvr_lean.domain.ladder_round.assembly import training_order
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ
from rlvr_lean.domain.ladder_round.l4_rows import rehearsal_rows
from rlvr_lean.domain.ladder_round.l5_cont import (
    A_LINE,
    ADDS,
    AGAIN,
    AVERAGED,
    BAR,
    COSTS,
    FURTHER_SEEDS,
    MODELS,
    MORE,
    NOT_SHOWN_AT_ONE,
    OF_ONE,
    OF_THEIRS,
    ONE_CONTINUATION,
    SINGLE,
    START,
    THE_AVERAGE_IS_PART,
    THE_RECIPE_STANDS,
    TRAINED,
    WIDE,
    hold_the_members,
    hold_the_sets,
    in_their_orders,
    of_l5f,
    the_read,
    the_sets,
    the_sets_are_the_rule,
    theirs_are_held,
    what_follows,
    what_made_it,
)

SEED = 3
PRETRAINING = [{"problem_id": f"p{place}"} for place in range(12)]            # the rows the first start was pretrained on, in the file's order, WITHOUT their text
ORDER = [row["id"] for row in rehearsal_rows(PRETRAINING, 12, SEED)]           # the seed's order of the file: what a set's rows are drawn along


def test_the_names_and_the_numbers_are_the_sections():
    assert (SINGLE, WIDE, AGAIN, MORE) == ("single", "wide", "again", "more") == MODELS and AVERAGED == (WIDE, AGAIN, MORE)
    assert (START, BAR) == ("second_full", "published")                                    # the start of `wide` and `more`; L5f's stored averaged model, the bar and `again`'s start
    assert l5_cont.ORDERS_OF == {WIDE: (5, 6, 7, 8), AGAIN: (1, 2, 3, 4), MORE: (1, 2, 3, 4)}
    assert TRAINED == ("wide_5", "wide_6", "wide_7", "wide_8", "again_1", "again_2", "again_3", "again_4", "more_1", "more_2", "more_3", "more_4")
    assert OF_THEIRS == ("published_1", "published_2", "published_3", "published_4") and l5_cont.THE_SINGLE == "published_1"
    assert OF_ONE == Fraction(13, 1000) and A_LINE == Fraction(85, 10000)                  # "below by more than 13 per 1,000"; "+8.5 per 1,000 ... below -8.5"
    assert (THE_AVERAGE_IS_PART, ONE_CONTINUATION) == ("THE AVERAGE IS PART OF THE GAIN", "ONE CONTINUATION GIVES IT")
    assert (ADDS, COSTS, NOT_SHOWN_AT_ONE) == ("ADDS: ESCALATE", "COSTS", "NOT SHOWN AT ONE SEED")
    assert l5_cont.MADE_IT == (THE_AVERAGE_IS_PART, ONE_CONTINUATION, INCONCLUSIVE, NOT_READ) and l5_cont.READS == (ADDS, COSTS, NOT_SHOWN_AT_ONE, INCONCLUSIVE, NOT_READ)
    assert l5_cont.FOLLOWS == (FURTHER_SEEDS, THE_RECIPE_STANDS, NOT_READ) and l5_cont.ASKED == {WIDE: "A WIDER AVERAGE", AGAIN: "A SECOND STEP", MORE: "MORE ROWS"}
    # The scout's seed rule and the read against a stored model are L5s's own functions, not copies.
    assert l5_cont.the_seed is l5_stock.the_seed and l5_cont.fixed_seeds is l5_stock.fixed_seeds and l5_cont.against_the_start is l5_stock.against_the_start
    assert l5_cont.fixed_seeds([3], [4, 5], False) == (3,) and l5_cont.the_seed(3, [3], [4, 5], False) is None
    with pytest.raises(ValueError, match=r"seed 4 is one of the further seeds \[4, 5\]"):
        l5_cont.the_seed(4, [3], [4, 5], False)


# -------------------------------------------------------------------------------------------------- the sets
def _ids(rows):
    return [row["id"] for row in rows]


SETS = the_sets(PRETRAINING, SEED, 4, 12)
STORED = {BAR: _ids(SETS[WIDE]), **{name: _ids(training_order(SETS[WIDE], order)) for order, name in enumerate(OF_THEIRS, start=1)}}       # L5f's run: its set, and each of its four orders
RECORDED = {name: sorted(STORED[name]) for name in OF_THEIRS}                  # a training loop's own record: the set's rows, each once


def test_the_three_sets_are_the_first_rows_the_next_rows_and_the_first_more_rows_of_the_seeds_order_each_in_the_seeds_training_order():
    assert sorted(_ids(SETS[WIDE])) == sorted(ORDER[:4]) and sorted(_ids(SETS[AGAIN])) == sorted(ORDER[4:8]) and sorted(_ids(SETS[MORE])) == sorted(ORDER)
    drawn = rehearsal_rows(PRETRAINING, 12, SEED)
    assert SETS[WIDE] == training_order(drawn[:4], SEED) and SETS[AGAIN] == training_order(drawn[4:8], SEED) and SETS[MORE] == training_order(drawn, SEED)
    # A published row by id and place, with its origin and no text: what `l4_rows.rehearsal_rows` gives.
    assert all(row["origin"] == "pretraining" and row["id"] == f"{row['problem_id']}#pretraining" and PRETRAINING[row["row_of_the_file"]]["problem_id"] == row["problem_id"]
               for own in SETS.values() for row in own)
    assert not set(_ids(SETS[WIDE])) & set(_ids(SETS[AGAIN])) and set(_ids(SETS[WIDE])) < set(_ids(SETS[MORE]))
    # `more` shorter than twice `wide` (the smoke run's sizes): the order is still drawn far enough for `again`.
    small = the_sets(PRETRAINING, SEED, 2, 3)
    assert sorted(_ids(small[WIDE])) == sorted(ORDER[:2]) and sorted(_ids(small[AGAIN])) == sorted(ORDER[2:4]) and sorted(_ids(small[MORE])) == sorted(ORDER[:3])
    # Another seed is another order of the file.
    assert _ids(the_sets(PRETRAINING, 4, 4, 12)[MORE]) != _ids(SETS[MORE])
    for first, more, said in ((0, 4, "`wide` is asked 0 rows and `more` 4"), (4, 3, "`wide` is asked 4 rows and `more` 3: `wide` is L5f's `published` set, at least one row, and `more` begins with it")):
        with pytest.raises(ValueError, match=said):
            the_sets(PRETRAINING, SEED, first, more)
    with pytest.raises(ValueError, match="trained on 12 rows and 14 are asked"):       # `again` would need rows the file does not hold
        the_sets(PRETRAINING, SEED, 7, 8)


def test_the_sets_are_held_to_l5fs_stored_published_to_l5ss_rule_and_to_hold_no_text():
    held = hold_the_sets(SETS, STORED[BAR], _ids(SETS[MORE]))
    assert held["rows"] == {WIDE: 4, AGAIN: 4, MORE: 12} and held["passes"] is True
    # `wide` IS NOT THE SET L5f's RUN STORED: the same rows in another order; another row; no such set.
    for theirs, place in (([*reversed(STORED[BAR])], 0), ([*STORED[BAR][:3], "p99#pretraining"], 3), (None, 0)):
        with pytest.raises(ValueError, match=rf"`wide` is not the set L5f's run of this seed stored as `published`, id for id \(4 rows here and {None if theirs is None else 4} there; "
                                             rf"first that differs at place {place}: "):
            hold_the_sets(SETS, theirs, _ids(SETS[MORE]))
    # `more` IS NOT L5s's RULE's: a row short.
    with pytest.raises(ValueError, match=r"`more` is not L5s's `published_stock` by its rule for this seed, id for id \(12 rows here and 11 there; first that differs at place 11: "):
        hold_the_sets(SETS, STORED[BAR], _ids(SETS[MORE])[:11])
    # A ROW WITH TEXT, a row that is no published row, an `again` that shares a row with `wide`: each refused.
    with_text = {**SETS, AGAIN: [{**SETS[AGAIN][0], "proof": "by simp"}, *SETS[AGAIN][1:]]}
    own = {**SETS, AGAIN: [{**SETS[AGAIN][0], "origin": "deep"}, *SETS[AGAIN][1:]]}
    shared = {**SETS, AGAIN: [SETS[WIDE][0], *SETS[AGAIN][1:]]}
    for sets, said in ((with_text, "1 rows are no published row or carry text"), (own, "1 rows are no published row or carry text"), (shared, "1 of them shared")):
        with pytest.raises(ValueError, match=said):
            hold_the_sets(sets, STORED[BAR], _ids(SETS[MORE]))


def test_l5fs_four_trainings_are_held_to_be_four_orders_of_wide_as_stored_and_as_trained():
    held = hold_the_members(SETS[WIDE], STORED, RECORDED)
    assert (held["trainings"], held["rows"], held["passes"]) == (list(OF_THEIRS), 4, True)
    # A STORED SET IN ANOTHER ORDER than its order seed's; a training whose loop recorded a row twice; a training with no record.
    with pytest.raises(ValueError, match=r"the stored sets of \['published_2'\] are not `wide`'s 4 rows in their order seed's order, and the trainings \[\] did not record"):
        hold_the_members(SETS[WIDE], {**STORED, "published_2": STORED["published_1"]}, RECORDED)
    with pytest.raises(ValueError, match=r"the stored sets of \[\] are not .* and the trainings \['published_3'\] did not record those rows, each once"):
        hold_the_members(SETS[WIDE], STORED, {**RECORDED, "published_3": [*RECORDED["published_3"], RECORDED["published_3"][0]]})
    with pytest.raises(ValueError, match=r"the trainings \['published_4'\] did not record"):
        hold_the_members(SETS[WIDE], STORED, {**RECORDED, "published_4": None})


def test_each_set_stands_in_the_order_of_each_of_its_order_seeds_and_wides_four_are_held_beside_l5fs_four():
    in_orders, coincide = in_their_orders(SETS)
    assert tuple(in_orders) == TRAINED
    for name, own in in_orders.items():
        model, _, order = name.rpartition("_")
        assert _ids(own) == _ids(training_order(SETS[model], int(order))) and [row["row"] for row in own] == list(range(len(SETS[model])))
    assert all(coincide[model]["alike"] == [] for model in AVERAGED) and set(coincide) == set(AVERAGED)
    # THE ORDER SEED THAT IS THE TASK'S SEED gives the seed's own order: named, and trained all the same. For `wide` it is one of L5f's four, held beside its own.
    assert coincide[WIDE]["the_order_of_the_tasks_seed"] == [3] == coincide[AGAIN]["the_order_of_the_tasks_seed"] == coincide[MORE]["the_order_of_the_tasks_seed"]
    assert "wide_3" not in in_orders and _ids(in_orders["again_3"]) == _ids(SETS[AGAIN])
    # TWO ROWS HAVE TWO ORDERS: eight order seeds of them hold pairs alike, refused but in a smoke run.
    small = the_sets(PRETRAINING, SEED, 2, 4)
    with pytest.raises(ValueError, match=r"of the order seeds \[1, 2, 3, 4, 5, 6, 7, 8\] of `wide`, the pairs .* give one order of these 2 rows: an averaged adapter is the mean of "
                                         "trainings that differ by their order. Refused"):
        in_their_orders(small)
    loosely, found = in_their_orders(small, smoke=True)
    assert tuple(loosely) == TRAINED and found[WIDE]["alike"] != [] and all(len(loosely[f"wide_{order}"]) == 2 for order in (5, 6, 7, 8))


def test_the_first_check_makes_each_set_again_and_fails_for_a_set_an_order_a_stored_set_of_l5fs_or_a_count_that_is_not_the_rules():
    in_orders, _ = in_their_orders(SETS)
    check = the_sets_are_the_rule(SETS, PRETRAINING, SEED, in_orders, STORED, RECORDED, 12)
    assert check["passes"] is True and check["rows"] == {WIDE: 4, AGAIN: 4, MORE: 12} and check["refused"] is None
    assert (check["sets_that_are_not_the_rule"], check["rows_with_text_or_not_published"], check["sets_not_in_their_order_seeds_order"], check["wide_is_l5fs_stored_set"],
            check["first_that_differs_from_l5fs"], check["l5fs_trainings_that_are_not_orders_of_wide"], check["l5fs_trainings_not_of_these_rows"], check["rows_of_more_by_l5ss_rule"],
            check["more_is_l5ss_rule"]) == ([], 0, [], True, None, [], [], 12, True)
    swapped = lambda own: [own[1], own[0], *own[2:]]      # noqa: E731
    failing = lambda **changed: the_sets_are_the_rule(**{"sets": SETS, "pretraining": PRETRAINING, "seed": SEED, "in_orders": in_orders, "stored": STORED, "recorded": RECORDED,      # noqa: E731
                                                         "more_by_the_rule": 12, **changed})
    # A SET THAT IS NOT ITS RULE (two rows changed places): named; and each of its orders still is its rows in that order.
    failed = failing(sets={**SETS, AGAIN: swapped(SETS[AGAIN])})
    assert (failed["passes"], failed["sets_that_are_not_the_rule"], failed["sets_not_in_their_order_seeds_order"]) == (False, [AGAIN], [])
    # ANOTHER SEED's rows: none of the three is the rule's.
    assert failing(seed=4)["sets_that_are_not_the_rule"] == list(AVERAGED)
    # A SET OUT OF ITS ORDER SEED'S ORDER.
    failed = failing(in_orders={**in_orders, "more_2": swapped(in_orders["more_2"])})
    assert (failed["passes"], failed["sets_not_in_their_order_seeds_order"], failed["sets_that_are_not_the_rule"]) == (False, ["more_2"], [])
    # L5f's RUN AS IT IS NOW: its stored set is another; one of its trainings is not an order of `wide`; one was not of these rows; it holds none.
    failed = failing(stored={**STORED, BAR: swapped(STORED[BAR])})
    assert (failed["passes"], failed["wide_is_l5fs_stored_set"], failed["first_that_differs_from_l5fs"]["place"]) == (False, False, 0)
    assert failing(stored={**STORED, "published_4": STORED["published_1"]})["l5fs_trainings_that_are_not_orders_of_wide"] == ["published_4"]
    assert failing(recorded={**RECORDED, "published_1": RECORDED["published_1"][:3]})["l5fs_trainings_not_of_these_rows"] == ["published_1"]
    gone = failing(stored={name: None for name in STORED}, recorded={name: None for name in RECORDED})
    assert (gone["passes"], gone["wide_is_l5fs_stored_set"], gone["l5fs_trainings_that_are_not_orders_of_wide"], gone["l5fs_trainings_not_of_these_rows"]) == (
        False, False, list(OF_THEIRS), list(OF_THEIRS))
    # `more` IS NOT THE ROWS L5s's RULE GIVES, counted again; and a row with text.
    assert (failing(more_by_the_rule=10)["passes"], failing(more_by_the_rule=10)["more_is_l5ss_rule"], failing(more_by_the_rule=None)["passes"]) == (False, False, False)
    failed = failing(sets={**SETS, WIDE: [{**SETS[WIDE][0], "statement": "theorem x"}, *SETS[WIDE][1:]]})
    assert (failed["passes"], failed["rows_with_text_or_not_published"]) == (False, 1)
    # ROWS THE FILE DOES NOT HOLD: the rule cannot be made again, and says why.
    failed = failing(pretraining=PRETRAINING[:8])
    assert failed["passes"] is False and "8 rows and 12 are asked" in failed["refused"] and failed["sets_that_are_not_the_rule"] == list(AVERAGED)


# ------------------------------------------------------------------------------------------------ L5f's run
MEASUREMENT = {"sampling": "reach", "episodes": 32, "sampling_seed": 1001, "problems": 392}
THEIRS = {"seed": 3, "start": "second_full", "recipe": "mean_constant", "ok": True, "inconclusive": False, "failed_checks": [], "the_measurement": dict(MEASUREMENT),
          "figures": {"published": {"all_of_g": {"successes": 1816, "attempts": 12544}}}}


@pytest.mark.parametrize("changed, why", [
    ({}, None),
    ({"there": False, "why": "the run is not on the box"}, "the run is not on the box"),
    ({"seed": 4}, "it is a report of seed 4"),
    ({"start": "pre_r64"}, "it is a run from `pre_r64` and this stage's bar is from `second_full`"),
    ({"recipe": "constant"}, "its recipe is `constant` and this run's `mean_constant`"),
    ({"ok": False}, "its report is not to be read"),
    ({"inconclusive": True, "failed_checks": ["`deep`: lean_answered"]}, "it reads INCONCLUSIVE (`deep`: lean_answered)"),
    ({"figures": None}, "it reads INCONCLUSIVE ()"),
    ({"the_measurement": {**MEASUREMENT, "episodes": 16}}, "it was measured otherwise ({'sampling': 'reach', 'episodes': 16, 'sampling_seed': 1001, 'problems': 392}) than this run")])
def test_why_l5fs_run_of_the_seed_cannot_carry_the_stage(changed, why):
    said = of_l5f({**THEIRS, **changed}, 3, "mean_constant", MEASUREMENT)
    assert said is None if why is None else said.startswith(why)


def _saved(rank=64, alpha=128):
    return {"rank": rank, "alpha": alpha, "ranks": [rank]}


def _training(name, changed=5, **more):
    return {"adapter": f"/runs/l5f/adapters/{name}", "adapter_saved": _saved(), "differs": {"compared": True, "elements_changed": changed, "passes": changed > 0}, **more}


TRAININGS = {name: _training(name) for name in OF_THEIRS}
AVERAGING = {"adapter": "/runs/l5f/adapters/published", "of": list(OF_THEIRS), "adapter_saved": _saved(), "averaged": True,
             "read_back": {"is_the_mean": True, "is_none_of_them": True, "passes": True}, "one_of_its_models_in_a_smoke_run": False}


def test_l5fs_five_adapters_are_held_as_its_own_steps_recorded_them_and_on_a_gpu_box_as_the_files_read():
    held = theirs_are_held(TRAININGS, AVERAGING, 64, 128)
    assert (held["passes"], held["recorded"], held["read_now"], held["rank"], held["alpha"]) == (True, True, None, 64, 128)
    assert held["trainings"]["published_2"] == {"adapter": "/runs/l5f/adapters/published_2", "read_back": _saved(), "at_its_rank": True, "numbers_not_the_start_adapters": 5}
    assert held["the_averaged"] == {"adapter": "/runs/l5f/adapters/published", "of": list(OF_THEIRS), "read_back": _saved(), "at_its_rank": True, "is_the_mean": True,
                                    "is_none_of_them": True, "one_of_its_models_in_a_smoke_run": False, "averaged": True}
    failing = lambda trainings=TRAININGS, averaging=AVERAGING, now=None: theirs_are_held(trainings, averaging, 64, 128, now)["passes"]      # noqa: E731
    # A TRAINING AT ANOTHER RANK, one that is the start adapter's number for number, one not read back, one missing.
    assert failing({**TRAININGS, "published_1": {**_training("published_1"), "adapter_saved": _saved(32, 64)}}) is False
    assert failing({**TRAININGS, "published_3": _training("published_3", changed=0)}) is False
    assert failing({**TRAININGS, "published_4": {**_training("published_4"), "adapter_saved": None}}) is False
    assert failing({name: own for name, own in TRAININGS.items() if name != "published_2"}) is False
    # THE AVERAGED ONE: not averaged (a pre-flight without a GPU), not the mean, one of its models, of other members, at another rank.
    assert failing(averaging={**AVERAGING, "averaged": False}) is False
    assert failing(averaging={**AVERAGING, "read_back": {**AVERAGING["read_back"], "is_the_mean": False}}) is False
    assert failing(averaging={**AVERAGING, "read_back": {**AVERAGING["read_back"], "is_none_of_them": False}}) is False
    assert failing(averaging={**AVERAGING, "of": list(OF_THEIRS[:3])}) is False and failing(averaging={**AVERAGING, "adapter_saved": _saved(16, 32)}) is False
    # ... but in a SMOKE run of L5f the mean may be each of its four, and that run said so: held all the same.
    smoke = {**AVERAGING, "read_back": {**AVERAGING["read_back"], "is_none_of_them": False}, "one_of_its_models_in_a_smoke_run": True}
    assert failing(averaging=smoke) is True
    # ON A GPU BOX the files themselves are read: one that does not pass fails the hold, whatever was recorded.
    now = {name: {"passes": True} for name in (*OF_THEIRS, BAR)}
    assert failing(now=now) is True and theirs_are_held(TRAININGS, AVERAGING, 64, 128, now)["read_now"] == now
    assert failing(now={**now, "published_3": {"passes": False}}) is False


# ------------------------------------------------------------------------------------------------- the reads
def _figure(successes, attempts=2000):
    return {"successes": successes, "attempts": attempts}


BAR_FIGURE = _figure(200)                                                      # 100 per 1,000


@pytest.mark.parametrize("successes, name, difference", [
    (200, ONE_CONTINUATION, 0.0), (230, ONE_CONTINUATION, 0.015), (175, ONE_CONTINUATION, -0.0125),
    (174, ONE_CONTINUATION, -0.013),               # EXACTLY 13 per 1,000 below is not "below by MORE than 13"
    (173, THE_AVERAGE_IS_PART, -0.0135), (100, THE_AVERAGE_IS_PART, -0.05)])
def test_what_made_it_is_read_on_whole_numbers_at_the_edge_of_thirteen_per_thousand(successes, name, difference):
    made = what_made_it([], _figure(successes), BAR_FIGURE)
    assert (made["name"], made["difference"], made["line"], made[SINGLE], made[BAR]) == (name, difference, 0.013, _figure(successes), BAR_FIGURE)
    assert made["reason"].startswith(f"`single` minus L5f's stored `published`, per 1,000 attempts on all of G: {difference * 1000:+.2f} ({successes} of 2000 against 200 of 2000); ")
    assert ("it is below zero by more than 13 per 1,000" in made["reason"]) is (name == THE_AVERAGE_IS_PART)


@pytest.mark.parametrize("successes, name, difference", [
    (300, ADDS, 0.05), (217, ADDS, 0.0085),        # AT plus 8.5 per 1,000 it adds
    (216, NOT_SHOWN_AT_ONE, 0.008), (200, NOT_SHOWN_AT_ONE, 0.0), (184, NOT_SHOWN_AT_ONE, -0.008),
    (183, NOT_SHOWN_AT_ONE, -0.0085),              # EXACTLY 8.5 per 1,000 below is not "below -8.5"
    (182, COSTS, -0.009), (100, COSTS, -0.05)])
def test_each_of_the_three_reads_is_on_whole_numbers_at_the_edges_of_plus_and_minus_eight_and_a_half_per_thousand(successes, name, difference):
    for model in AVERAGED:
        read = the_read([], model, _figure(successes), BAR_FIGURE)
        assert (read["name"], read["difference"], read["line"], read[model], read[BAR]) == (name, difference, 0.0085, _figure(successes), BAR_FIGURE)
        assert read["reason"] == (f"`{model}` minus L5f's stored `published`, per 1,000 attempts on all of G: {difference * 1000:+.2f} ({successes} of 2000 against 200 of 2000); "
                                  "the lines are +8.5 and -8.5")


def test_a_failed_check_is_inconclusive_and_no_attempt_is_not_read():
    failed = ["the run: l5fs_run_of_the_seed_carries_this_stage"]
    made, read = what_made_it(failed, _figure(100), BAR_FIGURE), the_read(failed, WIDE, _figure(300), BAR_FIGURE)
    assert (made["name"], read["name"]) == (INCONCLUSIVE, INCONCLUSIVE) and "difference" not in made and "difference" not in read
    assert "what made it is not read" in made["reason"] and "`wide` is not read" in read["reason"]
    for own, bar in ((None, BAR_FIGURE), (_figure(0, 0), BAR_FIGURE), (_figure(10), None), (_figure(10), _figure(0, 0))):
        assert what_made_it([], own, bar)["name"] == NOT_READ == the_read([], MORE, own, bar)["name"]


def _reads(**successes):
    figures = {SINGLE: _figure(successes.get(SINGLE, 190)), **{model: _figure(successes.get(model, 200)) for model in AVERAGED}}
    return what_made_it([], figures[SINGLE], BAR_FIGURE), {model: the_read([], model, figures[model], BAR_FIGURE) for model in AVERAGED}, figures


def test_what_follows_names_the_further_seeds_of_a_model_that_adds_or_that_the_recipe_stands_and_the_candidate():
    # ONE ADDS: two further seeds of it and of the bar; the candidate is the highest model.
    made, reads, figures = _reads(wide=220, more=260)
    follows = what_follows(made, reads, figures, BAR_FIGURE, [4, 5])
    assert (follows["name"], follows["adds"], follows["seeds"], follows["candidate"]) == (FURTHER_SEEDS, [WIDE, MORE], [4, 5], {"model": MORE, "of": "this stage", "per_attempt": 0.13})
    assert "`wide`, `more` read ADDS: ESCALATE: seeds [4, 5] of each and of the bar" in follows["reason"] and "is `more` of this stage, 130.00 per 1,000" in follows["reason"]
    made, reads, figures = _reads(again=217)
    follows = what_follows(made, reads, figures, BAR_FIGURE, [4, 5])
    assert (follows["name"], follows["adds"], follows["candidate"]["model"]) == (FURTHER_SEEDS, [AGAIN], AGAIN) and "`again` reads ADDS: ESCALATE" in follows["reason"]
    # NOTHING ADDS: L5f's `published` recipe stands. A model above the bar and under the line is still the candidate; with none above it, the bar is.
    made, reads, figures = _reads(wide=210, again=150)
    follows = what_follows(made, reads, figures, BAR_FIGURE, [4, 5])
    assert (follows["name"], follows["adds"], follows["candidate"]) == (THE_RECIPE_STANDS, [], {"model": WIDE, "of": "this stage", "per_attempt": 0.105}) and "seeds" not in follows
    assert "none of `wide`, `again`, `more` reads ADDS: ESCALATE: L5f's `published` recipe" in follows["reason"] and f"`single` alone ({ONE_CONTINUATION})" in follows["reason"]
    made, reads, figures = _reads(single=150, wide=199, again=150, more=200)
    follows = what_follows(made, reads, figures, BAR_FIGURE, [4, 5])
    assert (follows["name"], follows["candidate"]) == (THE_RECIPE_STANDS, {"model": BAR, "of": "L5f's stored run of the seed", "per_attempt": 0.1})
    assert f"`single` alone ({THE_AVERAGE_IS_PART})" in follows["reason"]
    # `single` itself may be the highest: one of L5f's own four trainings, as this stage measured it.
    made, reads, figures = _reads(single=215, wide=205)
    assert what_follows(made, reads, figures, BAR_FIGURE, [4, 5])["candidate"] == {"model": SINGLE, "of": "this stage", "per_attempt": 0.1075}
    # A READ THAT IS NOT MADE: nothing is named.
    made, reads, figures = _reads(wide=260)
    for changed in ({"made": {"name": INCONCLUSIVE}}, {"reads": {**reads, AGAIN: {"name": NOT_READ}}}, {"bar": None}, {"bar": _figure(0, 0)}):
        given = {"made": made, "reads": reads, "figures": figures, "bar": BAR_FIGURE, "more": [4, 5], **changed}
        follows = what_follows(**given)
        assert (follows["name"], set(follows)) == (NOT_READ, {"name", "reason"})
