"""L6b, the rules on hand-made rows: the names and the numbers; the setting and the seed; K; the next rows of the seed's
order and their hold to L5c's stored `again`; the two sets, what they are held to and their four orders; the check that
makes them again and what fails it; the check of the searched set; why L5c's run cannot carry the stage; THE PRIMARY and
WHOSE PROOFS at their edges, on whole numbers, INCONCLUSIVE and NOT READ; the search's yield; what follows.
Spec: docs/spec/ladder-loop.spec.md, "L6b: a second round, from the best model". Pure: no file, no GPU, no network.
The stage end to end is `test_ladder_l6b_stage.py`."""

from fractions import Fraction

import pytest

from rlvr_lean.domain.ladder_round import l6b
from rlvr_lean.domain.ladder_round.assembly import training_order
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ
from rlvr_lean.domain.ladder_round.l4_rows import PRETRAINING, rehearsal_rows
from rlvr_lean.domain.ladder_round.l5 import capped
from rlvr_lean.domain.ladder_round.l5_stock import NOT_SHOWN_AT_ONE
from rlvr_lean.domain.ladder_round.l6 import PART_A, PART_B
from rlvr_lean.domain.ladder_round.l6b import (
    A_LINE,
    ADD,
    BAR,
    COST,
    MODELS,
    NEW,
    NEW_OWN,
    OLD,
    OLD_OWN,
    SAID_AS_IT_IS,
    START,
    TEACH_LESS,
    TEACH_MORE,
    THE_CYCLE_TURNS,
    THE_SEARCH_IS_THE_FINDING,
    TO_A_DECISION,
    TRAINED,
    hold_the_next,
    hold_the_setting,
    hold_the_sets,
    in_their_orders,
    of_l5c,
    published_next,
    the_dose,
    the_primary,
    the_searched_is_l6s,
    the_seed,
    the_sets,
    the_sets_are_the_rule,
    the_yield,
    under_half,
    what_follows,
    whose_proofs,
)

SEED = 3
FIRST = 8                                                                       # the rows of L5f's `published` set, and of each set here
FILE = [{"problem_id": f"p{place}"} for place in range(24)]                     # the rows the first start was pretrained on, in the file's order, WITHOUT their text
ORDER = rehearsal_rows(FILE, 2 * FIRST, SEED)                                    # the seed's order of the file: L5f's `published` is its first rows, L5c's `again` the next
NEXT = ORDER[FIRST:]
AGAIN = [row["id"] for row in training_order(NEXT, SEED)]                       # the ids L5c's run stores as `again`, in its order


def _own(prefix, count, origin):
    return [{"id": f"{prefix}{index}", "problem_id": f"{prefix}_problem{index}", "origin": origin, "k": 1 + index, "n": 64, "lines": 3 + index, "first_success": 1 + index}
            for index in range(count)]


NEW_SOLVES, OLD_ROWS = _own("new", 6, NEW), _own("old", 4, OLD)


def _ids(rows):
    return [row["id"] for row in rows]


def test_the_names_and_the_numbers_are_the_sections():
    assert MODELS == ("new_own", "old_own") and (START, BAR) == ("published", "again")      # B is L5f's averaged `published`; the bar is L5c's `again`
    assert (NEW, OLD) == ("new", "deep") and l6b.OWN_OF == {NEW_OWN: NEW, OLD_OWN: OLD}
    assert TRAINED == ("new_own_1", "new_own_2", "new_own_3", "new_own_4", "old_own_1", "old_own_2", "old_own_3", "old_own_4")
    assert A_LINE == Fraction(85, 10000)                                                    # "+8.5 per 1,000 ... below zero by more than 8.5"
    assert ADD == "THE MODEL'S OWN NEW PROOFS ADD FROM THE BEST MODEL" and COST == "THEY COST THERE AT THIS SEED" and NOT_SHOWN_AT_ONE == "NOT SHOWN AT ONE SEED"
    assert TEACH_MORE == "PROOFS THE STRONGER MODEL FOUND TEACH MORE THAN THE WEAKER MODEL'S" and TEACH_LESS == "THEY TEACH LESS"


def test_the_setting_is_l5fs_published_l5cs_again_and_seeds_of_l5cs():
    assert hold_the_setting({"start": "published", "bar": "again", "seeds": [3]}, (3,)) == (3,)
    for settings, why in (({"start": "published", "bar": "again", "seeds": [3], "rows": 800}, "for nothing else"),
                          ({"start": "published", "bar": "again"}, "for nothing else"),
                          ({"start": "second_full", "bar": "again", "seeds": [3]}, "from L5f's stored averaged `published`"),
                          ({"start": "published", "bar": "published", "seeds": [3]}, "the bar is L5c's stored `again`"),
                          ({"start": "published", "bar": "again", "seeds": []}, "at least one"),
                          ({"start": "published", "bar": "again", "seeds": [3, 3]}, "distinct"),
                          ({"start": "published", "bar": "again", "seeds": [True]}, "whole numbers"),
                          ({"start": "published", "bar": "again", "seeds": [4]}, "a seed L5c's setting fixes")):
        with pytest.raises(ValueError, match=why):
            hold_the_setting(settings, (3,))
    the_seed(3, (3,))
    with pytest.raises(ValueError, match="a task of another seed is not run"):
        the_seed(4, (3,))


def test_k_is_the_smaller_of_half_a_set_the_old_own_rows_and_the_new_solves():
    assert the_dose(800, 400, 520) == 400 and the_dose(800, 400, 167) == 167            # "the smaller of 400 and N"
    assert the_dose(800, 300, 520) == 300 and the_dose(2, 1, 2) == 1                     # fewer old own rows than half a set; the smoke run's sizes
    for sizes in ((1, 4, 4), (8, 0, 4), (8, 4, 0)):
        with pytest.raises(ValueError, match="K = 0"):
            the_dose(*sizes)


def test_the_next_rows_are_the_rows_after_l5fs_of_the_seeds_order_and_are_held_to_l5cs_stored_again():
    rows = published_next(FILE, SEED, FIRST)
    assert _ids(rows) == _ids(ORDER[FIRST:]) and len(rows) == FIRST                     # rows 9 to 16 of the order, in the draw's own order
    assert not set(_ids(rows)) & set(_ids(ORDER[:FIRST]))                               # none of L5f's
    assert all(row["origin"] == PRETRAINING and "completion" not in row and isinstance(row["row_of_the_file"], int) for row in rows)       # by id and place, never with text
    assert hold_the_next(rows, SEED, AGAIN) == {"what": hold_the_next(rows, SEED, AGAIN)["what"], "rows": FIRST, "passes": True}
    with pytest.raises(ValueError, match="no row asked|at least one row"):
        published_next(FILE, SEED, 0)
    with pytest.raises(ValueError, match="no row is drawn twice"):
        published_next(FILE, SEED, 13)                                                  # 26 rows of a file of 24
    for theirs, why in ((None, "0 rows here|None there"), (AGAIN[:-1], "8 rows here and 7 there"), ([*AGAIN[1:], AGAIN[0]], "first that differs at place 0")):
        with pytest.raises(ValueError, match=f"not the set L5c's run of this seed stored as `again`.*({why})"):
            hold_the_next(rows, SEED, theirs)
    with pytest.raises(ValueError, match="carry text"):
        hold_the_next([{**rows[0], "completion": "by simp"}, *rows[1:]], SEED, AGAIN)


def test_the_two_sets_are_k_own_rows_and_the_same_first_next_rows_each_in_the_seeds_training_order():
    dose = the_dose(FIRST, len(OLD_ROWS), len(NEW_SOLVES))
    assert dose == 4
    sets = the_sets(NEW_SOLVES, OLD_ROWS, NEXT, SEED, dose)
    published = NEXT[:FIRST - dose]
    assert _ids(sets[NEW_OWN]) == _ids(training_order([*capped(NEW_SOLVES, dose, SEED), *published], SEED))       # a seeded draw of K of the six new solves
    assert _ids(sets[OLD_OWN]) == _ids(training_order([*OLD_ROWS[:dose], *published], SEED))                      # the FIRST K old own rows, in their stored order
    assert all(len(sets[model]) == FIRST for model in MODELS)
    of = lambda model, origin: sorted(row["id"] for row in sets[model] if (row["origin"] == PRETRAINING) == (origin == PRETRAINING))      # noqa: E731
    assert of(NEW_OWN, PRETRAINING) == of(OLD_OWN, PRETRAINING) == sorted(_ids(published))                         # THE SAME published rows
    assert len(of(NEW_OWN, NEW)) == len(of(OLD_OWN, OLD)) == dose and set(of(NEW_OWN, NEW)) < set(_ids(NEW_SOLVES))
    assert hold_the_sets(sets, NEXT, SEED, dose)["passes"]
    fewer = the_sets(NEW_SOLVES[:2], OLD_ROWS, NEXT, SEED, 2)                                                      # N under the old own rows: every new solve, the first two old ones
    assert sorted(row["id"] for row in fewer[NEW_OWN] if row["origin"] == NEW) == ["new0", "new1"]
    assert sorted(row["id"] for row in fewer[OLD_OWN] if row["origin"] == OLD) == ["old0", "old1"] and len(fewer[OLD_OWN]) == FIRST
    for wrong in (0, 5, 7):                                                                                        # no row; more than the old own rows hold; more than half a set
        with pytest.raises(ValueError, match="K is at least 1, at most each of the two, and at most half a set"):
            the_sets(NEW_SOLVES, [*OLD_ROWS, *_own("older", 4, OLD)] if wrong == 7 else OLD_ROWS, NEXT, SEED, wrong)


def test_the_sets_are_held_to_their_sizes_origins_order_published_rows_and_to_hold_no_text():
    dose = 4
    sets = the_sets(NEW_SOLVES, OLD_ROWS, NEXT, SEED, dose)
    swap = lambda model, own: {**sets, model: own}      # noqa: E731
    with pytest.raises(ValueError, match="each is K = 3 rows"):
        hold_the_sets(sets, NEXT, SEED, 3)
    with pytest.raises(ValueError, match="rows of another origin than `new`"):
        hold_the_sets(swap(NEW_OWN, [{**row, "origin": OLD} if row["origin"] == NEW else row for row in sets[NEW_OWN]]), NEXT, SEED, dose)
    with pytest.raises(ValueError, match="does not stand in the content-hash order"):
        hold_the_sets(swap(OLD_OWN, list(reversed(sets[OLD_OWN]))), NEXT, SEED, dose)
    other = training_order([*OLD_ROWS[:dose], *NEXT[1:FIRST - dose + 1]], SEED)                                    # as many published rows, but not the first ones
    with pytest.raises(ValueError, match=r"the published rows of \['old_own'\] are not the first 4 of the next rows"):
        hold_the_sets(swap(OLD_OWN, other), NEXT, SEED, dose)
    with pytest.raises(ValueError, match="1 published rows carry text"):
        hold_the_sets(swap(NEW_OWN, [{**row, "completion": "by simp"} if row["id"] == NEXT[0]["id"] else row for row in sets[NEW_OWN]]), NEXT, SEED, dose)
    of_both = next(row["problem_id"] for row in sets[NEW_OWN] if row["origin"] == NEW)                            # a problem round 1 solved is no new solve
    with pytest.raises(ValueError, match="1 problems are in both sets' own rows"):
        hold_the_sets(swap(OLD_OWN, [{**row, "problem_id": of_both} if row["id"] == "old0" else row for row in sets[OLD_OWN]]), NEXT, SEED, dose)


def test_each_set_stands_in_the_order_of_each_order_seed_and_two_orders_alike_are_refused_but_in_a_smoke_run():
    sets = the_sets(NEW_SOLVES, OLD_ROWS, NEXT, SEED, 4)
    in_orders, coincide = in_their_orders(sets)
    assert sorted(in_orders) == sorted(TRAINED) and sorted(coincide) == sorted(MODELS)
    for name, own in in_orders.items():
        model, _, order = name.rpartition("_")
        assert _ids(own) == _ids(training_order(sets[model], int(order))) and sorted(_ids(own)) == sorted(_ids(sets[model]))
    assert _ids(in_orders["new_own_3"]) == _ids(sets[NEW_OWN])                                                     # order seed 3 at seed 3 is the seed's own order: trained all the same
    two = {model: own[:1] for model, own in sets.items()}                                                          # one row has one order
    with pytest.raises(ValueError, match="give one order"):
        in_their_orders(two)
    assert sorted(in_their_orders(two, smoke=True)[0]) == sorted(TRAINED)


def _stored():
    sets = the_sets(NEW_SOLVES, OLD_ROWS, NEXT, SEED, 4)
    return sets, in_their_orders(sets)[0]


def test_the_check_makes_each_set_again_and_fails_for_a_set_an_order_a_new_solve_or_a_stored_again_that_is_not_the_rules():
    sets, in_orders = _stored()
    held = the_sets_are_the_rule(sets, in_orders, NEW_SOLVES, OLD_ROWS, FILE, SEED, AGAIN)
    assert held["passes"] and (held["K"], held["new_solves"], held["old_own_rows"], held["rows"]) == (4, 6, 4, {NEW_OWN: FIRST, OLD_OWN: FIRST})
    assert held["refused"] is None and held["the_next_rows_are_l5cs_again"] and held["the_two_hold_the_same_published_rows"]
    # a new solve of the set that the stored search no longer gives: the rule made again draws other rows
    in_the_set = next(row["id"] for row in sets[NEW_OWN] if row["origin"] == NEW)
    fewer = the_sets_are_the_rule(sets, in_orders, [row for row in NEW_SOLVES if row["id"] != in_the_set], OLD_ROWS, FILE, SEED, AGAIN)
    assert not fewer["passes"] and fewer["sets_that_are_not_the_rule"] == [NEW_OWN] and fewer["new_solves"] == 5
    # L5d's stored set as it is now holds other old rows
    moved = the_sets_are_the_rule(sets, in_orders, NEW_SOLVES, list(reversed(OLD_ROWS)), FILE, SEED, AGAIN)
    assert moved["passes"]                                                                                         # the same four rows: the set is their content-hash order
    other = the_sets_are_the_rule(sets, in_orders, NEW_SOLVES, [*_own("older", 1, OLD), *OLD_ROWS], FILE, SEED, AGAIN)
    assert not other["passes"] and other["sets_that_are_not_the_rule"] == [OLD_OWN]
    # L5c's stored `again` is not the next rows
    not_again = the_sets_are_the_rule(sets, in_orders, NEW_SOLVES, OLD_ROWS, FILE, SEED, AGAIN[:-1])
    assert not not_again["passes"] and not not_again["the_next_rows_are_l5cs_again"] and not_again["first_that_differs_from_l5cs"]["rows_of_theirs"] == FIRST - 1
    assert not the_sets_are_the_rule(sets, in_orders, NEW_SOLVES, OLD_ROWS, FILE, SEED, None)["passes"]
    # a stored order that is not its order seed's
    out = the_sets_are_the_rule(sets, {**in_orders, "old_own_2": in_orders["old_own_1"]}, NEW_SOLVES, OLD_ROWS, FILE, SEED, AGAIN)
    assert not out["passes"] and out["sets_not_in_their_order_seeds_order"] == ["old_own_2"]
    # a published row with text
    text = {**sets, NEW_OWN: [{**row, "completion": "by simp"} if row["origin"] == PRETRAINING else row for row in sets[NEW_OWN]]}
    assert not the_sets_are_the_rule(text, in_orders, NEW_SOLVES, OLD_ROWS, FILE, SEED, AGAIN)["passes"]
    # a file that no longer holds the rows
    assert "no row is drawn twice" in the_sets_are_the_rule(sets, in_orders, NEW_SOLVES, OLD_ROWS, FILE[:10], SEED, AGAIN)["refused"]


def _searched():
    return [{"problem_id": "a1", "part": PART_A}, {"problem_id": "a2", "part": PART_A}, {"problem_id": "b1", "part": PART_B}]


def _results(ids, episodes=64):
    return [{"problem_id": problem_id, "episodes": episodes, "resolved": 0} for problem_id in ids]


def test_the_searched_set_is_l6s_rule_made_again_with_no_barred_problem_and_sampled_as_prepared():
    stored = _searched()
    ids = [row["problem_id"] for row in stored]
    held = the_searched_is_l6s(stored, _searched(), None, _results(ids), 64, ["h1"], ["g1"], ["s1"], ["solved1"])
    assert held["passes"] and held["problems"] == 3 and held["by_part"] == {PART_A: 2, PART_B: 1} and held["is_the_rule_made_again"]
    for changed, key in (({"again": [*_searched(), {"problem_id": "b2", "part": PART_B}]}, "is_the_rule_made_again"),
                         ({"again": [{**_searched()[0], "part": PART_B}, *_searched()[1:]]}, "is_the_rule_made_again"),
                         ({"again": None, "refused": "the map is not there"}, "is_the_rule_made_again"),
                         ({"results": _results(ids[:2])}, "sampled_as_prepared"), ({"results": _results(ids, 32)}, "attempts_a_problem_that_are_not_the_settings")):
        given = {"again": _searched(), "refused": None, "results": _results(ids), **changed}
        own = the_searched_is_l6s(stored, given["again"], given["refused"], given["results"], 64, ["h1"], ["g1"], ["s1"], ["solved1"])
        assert not own["passes"] and own[key] in (False, [32]), key
    for barred, key in (((["a1"], [], [], []), "held_out"), (([], ["a2"], [], []), "of_the_goal_set"), (([], [], ["b1"], []), "of_the_held_aside_set"),
                        (([], [], [], ["a1"]), "solved_by_search_one")):
        own = the_searched_is_l6s(stored, _searched(), None, _results(ids), 64, *barred)
        assert not own["passes"] and own[key] == 1 and own["first"] == [barred[("held_out", "of_the_goal_set", "of_the_held_aside_set", "solved_by_search_one").index(key)][0]]
    assert not the_searched_is_l6s([], [], None, [], 64, [], [], [], [])["passes"]                                 # no problem searched


MEASUREMENT = {"sampling": "reach", "episodes": 32, "sampling_seed": 1001, "problems": 392}


def _entry(**changed):
    return {"there": True, "seed": 3, "start": "second_full", "recipe": "mean_constant", "ok": True, "inconclusive": False, "failed_checks": [], "figures": {"again": {}},
            "the_measurement": dict(MEASUREMENT), **changed}


@pytest.mark.parametrize("changed, why", [
    ({"there": False, "why": "it is not on the box"}, "it is not on the box"), ({"seed": 4}, "a report of seed 4"), ({"start": "pre_r64"}, "a run from `pre_r64`"),
    ({"recipe": "constant"}, "its recipe is `constant`"), ({"ok": False}, "not to be read"),
    ({"inconclusive": True, "failed_checks": ["`again`: the_adapter"]}, "it reads INCONCLUSIVE"), ({"figures": None}, "it reads INCONCLUSIVE"),
    ({"the_measurement": {**MEASUREMENT, "sampling_seed": 1002}}, "it was measured otherwise")])
def test_why_l5cs_run_of_the_seed_cannot_carry_the_stage(changed, why):
    assert of_l5c(_entry(), 3, "mean_constant", MEASUREMENT) is None
    assert why in of_l5c(_entry(**changed), 3, "mean_constant", MEASUREMENT)


def test_a_smoke_run_reads_an_inconclusive_report_of_l5c_all_the_same_and_never_one_not_to_be_read():
    inconclusive = _entry(inconclusive=True, figures=None, failed_checks=["`again`: the_loss"])
    assert of_l5c(inconclusive, 3, "mean_constant", MEASUREMENT, though_inconclusive=True) is None
    assert "not to be read" in of_l5c({**inconclusive, "ok": False}, 3, "mean_constant", MEASUREMENT, though_inconclusive=True)
    assert "a report of seed 4" in of_l5c({**inconclusive, "seed": 4}, 3, "mean_constant", MEASUREMENT, though_inconclusive=True)


ATTEMPTS = 2000                                                                 # 17 successes of 2,000 are 8.5 per 1,000 exactly


def _figure(successes, attempts=ATTEMPTS):
    return {"successes": successes, "attempts": attempts}


@pytest.mark.parametrize("successes, name, difference", [
    (117, ADD, 0.0085), (116, NOT_SHOWN_AT_ONE, 0.008), (100, NOT_SHOWN_AT_ONE, 0.0), (83, NOT_SHOWN_AT_ONE, -0.0085), (82, COST, -0.009)])
def test_the_primary_is_read_on_whole_numbers_at_the_edges_of_plus_and_minus_eight_and_a_half_per_thousand(successes, name, difference):
    read = the_primary([], _figure(successes), _figure(100))
    assert (read["name"], read["difference"], read["line"]) == (name, difference, 0.0085)
    assert read["new_own"] == _figure(successes) and read["L5c's stored again"] == _figure(100)


@pytest.mark.parametrize("successes, name", [(117, TEACH_MORE), (116, NOT_SHOWN_AT_ONE), (83, NOT_SHOWN_AT_ONE), (82, TEACH_LESS)])
def test_whose_proofs_is_read_at_the_same_lines(successes, name):
    read = whose_proofs([], _figure(successes), _figure(100))
    assert read["name"] == name and read["new_own"] == _figure(successes) and read["old_own"] == _figure(100)


def test_a_figure_is_not_rounded_before_a_branch_is_read():
    # 12,544 attempts: 107 successes more are 8.53 per 1,000 (ADD), 106 are 8.45 (not shown); a figure rounded to two decimals of a per-1,000 would not tell them apart from the line
    assert the_primary([], _figure(1907, 12544), _figure(1800, 12544))["name"] == ADD
    assert the_primary([], _figure(1906, 12544), _figure(1800, 12544))["name"] == NOT_SHOWN_AT_ONE
    assert the_primary([], _figure(1693, 12544), _figure(1800, 12544))["name"] == COST
    assert the_primary([], _figure(1694, 12544), _figure(1800, 12544))["name"] == NOT_SHOWN_AT_ONE


def test_a_failed_check_is_inconclusive_and_no_attempt_is_not_read():
    for read in (the_primary, whose_proofs):
        assert read(["the run: the_sets"], _figure(200), _figure(100))["name"] == INCONCLUSIVE
        assert read([], None, _figure(100))["name"] == read([], _figure(100), None)["name"] == read([], _figure(0, 0), _figure(100))["name"] == NOT_READ


def test_the_search_is_a_finding_of_its_own():
    solves = [{"k": 1, "n": 64}, {"k": 31, "n": 64}, {"k": 32, "n": 64}, {"k": 64, "n": 64}]
    assert under_half(solves) == 2                                                                                 # 32 of 64 is half, not under it
    table = {"problems": 865, "share_solved": 0.00462, PART_A: {"problems": 480, "solved": 3}, PART_B: {"problems": 385, "solved": 1}, "episodes_each": 64, "per_1000": 1.2,
             "what": "L6's own words"}
    found = the_yield(table, solves, {"minimum": 150, "passes": False}, {"new_solves": {"median": 4}})
    assert (found["new_solves"], found["minimum"], found["enough"], found["problems"], found["solved_in_under_half_of_its_attempts"]) == (4, 150, False, 865, 2)
    assert (found[PART_A]["solved"], found[PART_A]["problems"], found[PART_B]["solved"], found[PART_B]["problems"]) == (3, 480, 1, 385)
    assert found["lines"] == {"new_solves": {"median": 4}} and "L6's own words" not in found["what"]


def test_what_follows_is_named_by_the_primary_and_says_what_whose_proofs_reads():
    read = lambda name: {"name": name}      # noqa: E731
    for primary, name in ((ADD, THE_CYCLE_TURNS), (NOT_SHOWN_AT_ONE, SAID_AS_IT_IS), (COST, TO_A_DECISION)):
        for whose, said in ((TEACH_MORE, "the part worth scaling"), (TEACH_LESS, "teach less"), (NOT_SHOWN_AT_ONE, "not shown at one seed")):
            follows = what_follows(read(primary), read(whose), True)
            assert follows["name"] == name and follows["whose_proofs"] == whose and said in follows["reason"]
    assert "two further seeds" in what_follows(read(ADD), read(TEACH_MORE), True)["reason"]
    assert "a third round is not run" in what_follows(read(NOT_SHOWN_AT_ONE), read(TEACH_MORE), True)["reason"]
    assert "the full-size cycle is not proposed" in what_follows(read(COST), read(TEACH_MORE), True)["reason"]
    for name in (INCONCLUSIVE, NOT_READ):
        assert what_follows(read(name), read(name), True)["name"] == NOT_READ
    assert what_follows(read(NOT_READ), read(NOT_READ), False)["name"] == THE_SEARCH_IS_THE_FINDING            # too few new solves: nothing was trained
