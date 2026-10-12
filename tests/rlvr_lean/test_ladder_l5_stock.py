"""L5s, the rules on hand-made rows: the seed and the escalation; the two sets (every deep solve, every ordinary solve
and as many published rows; twice as many published rows), what they are held to and that they hold a seed's rows of
L5d's three sets; the first check; THE DOSE at its edges, INCONCLUSIVE and NOT READ; the two differences beside it; a
model against the start model, paired by problem; what follows. Spec: docs/spec/ladder-loop.spec.md, "L5s: the
dose. Every own solve in stock that the model finds less than half the time, against the same number of published
proofs". Pure: no file, no GPU, no network. The stage end to end is `test_ladder_l5_stock_stage.py`."""

import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l5 import DEEP_ROWS, PRETRAINING, SEED, SHALLOW_ROWS  # noqa: E402 - hand-made deep rows, ordinary solves and a pretraining file's rows
from test_ladder_l5_draws import G_PARTS, START, _figures, _on_g  # noqa: E402 - four goal problems, the start model's own first sampling, a seed's figures

from rlvr_lean.domain.ladder_round import l5  # noqa: E402
from rlvr_lean.domain.ladder_round import l5_stock  # noqa: E402
from rlvr_lean.domain.ladder_round.assembly import training_order  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows import rehearsal_rows  # noqa: E402
from rlvr_lean.domain.ladder_round.l5 import DEEP, PUBLISHED, SHALLOW  # noqa: E402
from rlvr_lean.domain.ladder_round.l5_draws import ALL, on_the_parts  # noqa: E402
from rlvr_lean.domain.ladder_round.l5_noise import ORDERS, numbered  # noqa: E402
from rlvr_lean.domain.ladder_round.l5_stock import (  # noqa: E402
    A_COST,
    COSTS,
    ESCALATE,
    FURTHER_SEEDS,
    MODELS,
    NO_FURTHER_SEED,
    NOT_SHOWN_AT_ONE,
    PUBLISHED_STOCK,
    STOCK,
    against_the_start,
    beside,
    fixed_seeds,
    hold_the_sets,
    holds_the_seeds_rows,
    of_l5d,
    ordinary_of_the_rule,
    the_dose,
    the_seed,
    the_sets,
    the_sets_are_the_rule,
    what_follows,
)

MEASURED = {"sampling": "reach", "episodes": 32, "sampling_seed": 1001, "problems": 4}
DEEP_SOLVES, ORDINARY_SOLVES = [row["problem_id"] for row in DEEP_ROWS], [row["problem_id"] for row in SHALLOW_ROWS]
placed = lambda row: (row["id"], row["problem_id"], row["row_of_the_file"])      # noqa: E731
of_the_file = lambda rows: [row for row in rows if row["origin"] == "pretraining"]      # noqa: E731


def test_the_names_and_the_numbers_are_the_sections():
    assert MODELS == (STOCK, PUBLISHED_STOCK) == ("stock", "published_stock")                # trained, averaged and measured in this order
    assert (ESCALATE, COSTS, NOT_SHOWN_AT_ONE) == ("THE SOLVE RATE DOES NOT FALL WITH THE DOSE: ESCALATE", "MORE OF ITS OWN PROOFS COSTS AT THIS SEED", "NOT SHOWN AT ONE SEED")
    assert l5_stock.DOSE == (ESCALATE, COSTS, NOT_SHOWN_AT_ONE, INCONCLUSIVE, NOT_READ) and l5_stock.FOLLOWS == (FURTHER_SEEDS, NO_FURTHER_SEED, NOT_READ)
    assert A_COST == Fraction(1, 100) and l5_stock.STANDARD_DEVIATION == 0.0048              # "below zero by more than 0.010 (twice the standard deviation above)"
    assert set(l5_stock.OWN) == {DEEP, SHALLOW}                                              # an own row is of one of L5's two origins


# -------------------------------------------------------------------------------------------------- the seed
def test_a_tasks_seed_is_the_scouts_and_a_further_one_waits_for_the_escalation_to_be_recorded():
    assert fixed_seeds([3], [4, 5], False) == (3,) and fixed_seeds([3], [4, 5], True) == (3, 4, 5)
    assert the_seed(3, [3], [4, 5], False) is None and the_seed(5, [3], [4, 5], True) is None
    with pytest.raises(ValueError, match=r"seed 4 is one of the further seeds \[4, 5\], which run once the scout's read at \[3\] names the escalation and the setting records it "
                                         r"\(escalated is false\): one seed is the scout"):
        the_seed(4, [3], [4, 5], False)
    with pytest.raises(ValueError, match=r"seed 6 is none of this stage's seeds \[3\] \(nor of \[4, 5\], which wait for the escalation\)"):
        the_seed(6, [3], [4, 5], False)
    with pytest.raises(ValueError, match=r"seed 6 is none of this stage's seeds \[3\] and \[4, 5\]: its sets hold the rows of L5d's own run of its seed"):
        the_seed(6, [3], [4, 5], True)


# -------------------------------------------------------------------------------------------------- the sets
def test_an_ordinary_solve_is_a_map_problem_solved_in_l5s_own_range_and_none_the_search_or_the_held_aside_set_took():
    map_rows = [{"problem_id": f"m{k}", "resolved": k} for k in range(9)]
    assert ordinary_of_the_rule(map_rows, 1, 3) == ["m1", "m2", "m3"] and ordinary_of_the_rule(map_rows, 1, 3, {"m2", "m7"}) == ["m1", "m3"]
    assert ordinary_of_the_rule(map_rows, 4, 8) == ["m4", "m5", "m6", "m7", "m8"] and ordinary_of_the_rule([], 1, 3) == []


def test_stock_is_every_own_row_and_as_many_published_rows_and_the_bar_twice_as_many_published_rows_of_the_one_order():
    sets = the_sets(DEEP_ROWS, SHALLOW_ROWS, PRETRAINING, SEED)
    assert list(sets) == list(MODELS) and (len(sets[STOCK]), len(sets[PUBLISHED_STOCK])) == (16, 16)                 # N = 8 own rows: 2N rows a set
    own = [row for row in sets[STOCK] if row["origin"] != "pretraining"]
    assert sorted(row["id"] for row in own) == sorted(row["id"] for row in (*DEEP_ROWS, *SHALLOW_ROWS)) and of_the_file(sets[PUBLISHED_STOCK]) == sets[PUBLISHED_STOCK]
    # THE PUBLISHED ROWS are the first N and the first 2N of the ONE hash order L5's and L5d's published rows are the first of: the bar holds `stock`'s.
    order = rehearsal_rows(PRETRAINING, 16, SEED)
    assert sorted(map(placed, of_the_file(sets[STOCK]))) == sorted(map(placed, order[:8])) and sorted(map(placed, sets[PUBLISHED_STOCK])) == sorted(map(placed, order))
    assert order[:8] == rehearsal_rows(PRETRAINING, 8, SEED) and all(sets[model] == training_order(sets[model], SEED) for model in MODELS)      # each set in the seed's content-hash order
    assert hold_the_sets(sets, PRETRAINING, SEED) == {"own_rows": 8, "deep_solves": 4, "ordinary_solves": 4, "rows": {"stock": 16, "published_stock": 16}}
    assert hold_the_sets(sets, PRETRAINING, SEED, DEEP_SOLVES, ORDINARY_SOLVES)["own_rows"] == 8                       # THE RULE MADE AGAIN: every one, and no other
    # IT HOLDS THAT SEED's ROWS of L5d's three sets (D deep rows and D ordinary solves of these, the first D and 2D published rows), by id.
    theirs = l5.the_sets(DEEP_ROWS[:2], SHALLOW_ROWS[:2], PRETRAINING, SEED)
    assert holds_the_seeds_rows(sets, theirs)["rows"] == {"published": 4, "shallow": 4, "deep": 4}
    # No deep solve, or no ordinary solve: no `stock`.
    for deep, ordinary in (([], SHALLOW_ROWS), (DEEP_ROWS, [])):
        with pytest.raises(ValueError, match="it holds every one of both, at least 1 of each. Refused"):
            the_sets(deep, ordinary, PRETRAINING, SEED)
    with pytest.raises(ValueError):                                                                                     # the bar asks for more rows than the file holds
        the_sets(DEEP_ROWS, SHALLOW_ROWS, PRETRAINING[:12], SEED)


def test_the_sets_are_held_to_the_rule_and_each_way_of_not_being_it_is_refused():
    sets = the_sets(DEEP_ROWS, SHALLOW_ROWS, PRETRAINING, SEED)
    stock, bar = sets[STOCK], sets[PUBLISHED_STOCK]
    own, published = [row for row in stock if row["origin"] != "pretraining"], of_the_file(stock)
    again = lambda rows: training_order(rows, SEED)      # noqa: E731
    for broken, said in (
            ({STOCK: again(stock[1:]), PUBLISHED_STOCK: bar}, "the two sets hold .rows of the model's own, published rows."),                         # a row short
            ({STOCK: stock, PUBLISHED_STOCK: again([*bar[1:], own[0]])}, "`published_stock` 2N published rows and no other"),                       # an own row in the bar
            ({STOCK: again([row for row in stock if row["origin"] != DEEP]), PUBLISHED_STOCK: bar}, "at least 1 of each"),                           # no deep solve
            ({STOCK: again([*stock[:-1], {**stock[-1], "origin": "round_2"}]) if stock[-1]["origin"] != "pretraining" else again([{**own[0], "origin": "round_2"}, *own[1:], *published]),
              PUBLISHED_STOCK: bar}, "the own rows by origin"),                                                                                       # an origin that is none of L5's two
            ({STOCK: list(reversed(stock)), PUBLISHED_STOCK: bar}, "`stock` does not stand in the content-hash order of this seed"),
            ({STOCK: stock, PUBLISHED_STOCK: again([bar[0], *bar[:-1]])}, "`published_stock` holds 1 ids twice"),
            ({STOCK: again([{**own[0], "problem_id": own[1]["problem_id"]}, *own[1:], *published]), PUBLISHED_STOCK: bar}, "1 problems have two own rows in `stock`"),
            ({STOCK: again([{**own[0], "problem_id": PRETRAINING[0]["problem_id"]}, *own[1:], *published]), PUBLISHED_STOCK: bar}, "1 problems of its own rows are rows of the pretraining file"),
            ({STOCK: again([*own, *of_the_file(bar)[-8:]]) if sorted(map(placed, of_the_file(bar)[-8:])) != sorted(map(placed, published)) else stock, PUBLISHED_STOCK: bar},
             "the 8 published rows of `stock` are not the first 8 of the hash order of this seed among the 40 rows")):
        with pytest.raises(ValueError, match=said):
            hold_the_sets(broken, PRETRAINING, SEED)
    with pytest.raises(ValueError, match="`stock` does not stand in the content-hash order of this seed"):
        hold_the_sets(sets, PRETRAINING, SEED + 1)                                                                      # another seed's sets are not this seed's: their order is said first
    with pytest.raises(ValueError, match="not the first 8 of the hash order of this seed"):                             # and, put into this seed's order, their published rows are not its draw
        hold_the_sets({model: training_order(rows, SEED + 1) for model, rows in {STOCK: [*own, *of_the_file(bar)[-8:]], PUBLISHED_STOCK: bar}.items()}, PRETRAINING, SEED + 1)
    # THE RULE MADE AGAIN gives a deep solve, or an ordinary solve, the set does not hold; or does not give one it holds: refused, naming the first.
    with pytest.raises(ValueError, match="`stock` holds 4 rows of origin `deep` and the rule gives 5 .every deep solve of the search; first that differs: b7.: it holds every one"):
        hold_the_sets(sets, PRETRAINING, SEED, [*DEEP_SOLVES, "b7"], ORDINARY_SOLVES)
    with pytest.raises(ValueError, match="`stock` holds 4 rows of origin `shallow` and the rule gives 3 .every ordinary solve the rule gives; first that differs: b8."):
        hold_the_sets(sets, PRETRAINING, SEED, DEEP_SOLVES, ORDINARY_SOLVES[1:])
    # A SEED's ROWS IT DOES NOT HOLD: L5d's sets of another draw (a deep solve that is none of the search's, published rows of another seed's order).
    other = l5.the_sets([{**DEEP_ROWS[0], "id": "zz#l5_deep_search#statement#0", "attempt_id": "zz#l5_deep_search#statement#0", "problem_id": "zz"}, DEEP_ROWS[1]], SHALLOW_ROWS[:2], PRETRAINING, SEED)
    with pytest.raises(ValueError, match="1 rows of that seed's `deep` are no rows of `stock` .first: zz#l5_deep_search#statement#0.: the sets of this stage hold the rows of L5d's three sets"):
        holds_the_seeds_rows(sets, other)
    with pytest.raises(ValueError, match="rows of that seed's `published` are no rows of `published_stock`"):
        holds_the_seeds_rows({STOCK: stock, PUBLISHED_STOCK: bar[:4]}, l5.the_sets(DEEP_ROWS[:2], SHALLOW_ROWS[:2], PRETRAINING, SEED))


def test_the_first_check_is_the_rule_made_again_with_no_text_and_each_order_seeds_order():
    sets = {model: numbered(own) for model, own in the_sets(DEEP_ROWS, SHALLOW_ROWS, PRETRAINING, SEED).items()}
    in_orders = {f"{model}_{order}": numbered(training_order(sets[model], order)) for model in MODELS for order in ORDERS}
    held = the_sets_are_the_rule(sets, PRETRAINING, SEED, DEEP_SOLVES, ORDINARY_SOLVES, in_orders)
    assert (held["counted"], held["refused"], held["published_rows_with_text"], held["sets_not_in_their_order_seeds_order"], held["passes"]) == (
        {"own_rows": 8, "deep_solves": 4, "ordinary_solves": 4, "rows": {"stock": 16, "published_stock": 16}}, None, 0, [], True)
    # THE RULE NOW GIVES ANOTHER ORDINARY SOLVE; another seed's draw; a set that is not there: the check fails and says why.
    more = the_sets_are_the_rule(sets, PRETRAINING, SEED, DEEP_SOLVES, [*ORDINARY_SOLVES, "b20"], in_orders)
    assert more["passes"] is False and more["counted"] is None and "the rule gives 5" in more["refused"]
    assert the_sets_are_the_rule(sets, PRETRAINING, SEED + 1, DEEP_SOLVES, ORDINARY_SOLVES, in_orders)["passes"] is False
    assert the_sets_are_the_rule({STOCK: sets[STOCK]}, PRETRAINING, SEED, DEEP_SOLVES, ORDINARY_SOLVES, {})["passes"] is False
    # A PUBLISHED ROW WITH ITS TEXT; a set that is not in its order seed's order, or is another model's, or is of no model.
    with_text = {**sets, PUBLISHED_STOCK: [{**sets[PUBLISHED_STOCK][0], "proof": "by simp"}, *sets[PUBLISHED_STOCK][1:]]}
    found = the_sets_are_the_rule(with_text, PRETRAINING, SEED, DEEP_SOLVES, ORDINARY_SOLVES, in_orders)
    assert (found["published_rows_with_text"], found["passes"]) == (1, False)
    for broken in ({**in_orders, "stock_2": in_orders["stock_3"]}, {**in_orders, "stock_2": in_orders["stock_2"][1:]}, {**in_orders, "stock_2": in_orders["published_stock_2"]},
                   {**in_orders, "wide_2": in_orders["stock_2"]}):
        held = the_sets_are_the_rule(sets, PRETRAINING, SEED, DEEP_SOLVES, ORDINARY_SOLVES, broken)
        assert held["passes"] is False and held["counted"]["own_rows"] == 8 and len(held["sets_not_in_their_order_seeds_order"]) == 1


# ------------------------------------------------------------------------------------------------- the reads
def _entry(published=(8, 8, 0, 0), shallow=(12, 8, 4, 0), deep=(16, 8, 8, 0), **changed):
    """What L5d's report says of its own seed 3: its three models' figures on four goal problems of 32 attempts."""
    return {"seed": 3, "recipe": "mean_constant", "ok": True, "inconclusive": False, "failed_checks": [], "the_measurement": dict(MEASURED),
            "figures": _figures(published, shallow, deep), "there": True, "run": "ladder_l5_draws_pre_r64_seed3", **changed}


def test_l5ds_run_of_the_seed_is_read_when_it_is_there_and_was_measured_alike_and_not_read_with_why_otherwise():
    read = of_l5d(_entry(), 3, "mean_constant", MEASURED)
    assert read["read"] is True and read["run"] == "ladder_l5_draws_pre_r64_seed3" and list(read["figures"]) == [PUBLISHED, SHALLOW, DEEP]
    assert (read["figures"][DEEP]["successes"], read["figures"][DEEP]["attempts"], read["figures"][PUBLISHED]["per_attempt"]) == (32, 128, 0.125)
    assert read["the_primarys_difference"] == round((24 / 128 + 32 / 128) / 2 - 16 / 128, 6)                         # that seed's own primary: the mean of `shallow` and `deep`, minus `published`
    assert of_l5d({"there": False, "why": "it is not on the box", "run": "x"}, 3, "mean_constant", MEASURED) == {"read": False, "why": "it is not on the box", "run": "x"}
    for changed, why in (({"seed": 4}, "it is a report of seed 4"), ({"recipe": "constant"}, "its recipe is `constant` and this run's `mean_constant`"), ({"ok": False}, "its report is not to be read"),
                         ({"inconclusive": True, "failed_checks": ["`deep`: lean_answered"]}, "it reads INCONCLUSIVE .`deep`: lean_answered."), ({"figures": None}, "it reads INCONCLUSIVE"),
                         ({"the_measurement": {**MEASURED, "episodes": 61}}, "it was measured otherwise")):
        not_read = of_l5d(_entry(**changed), 3, "mean_constant", MEASURED)
        assert not_read["read"] is False and not_read["why"].startswith("L5d's run of seed 3 does not stand beside this one: ") and pytest.importorskip("re").search(why, not_read["why"])


def _of(successes, attempts=1000):
    return {"successes": successes, "attempts": attempts, "per_attempt": successes / attempts, "per_1000": 1000 * successes / attempts, "sampling_noise": 0.01}


def _theirs(deep, published=90, shallow=100):
    return {"read": True, "run": "r", "figures": {PUBLISHED: _of(published), SHALLOW: _of(shallow), DEEP: _of(deep)}, "the_primarys_difference": 0.015}


@pytest.mark.parametrize("stock, name", [(111, ESCALATE), (110, ESCALATE),              # above zero, and EXACTLY zero: at or above zero
                                         (109, NOT_SHOWN_AT_ONE), (100, NOT_SHOWN_AT_ONE),  # just below zero, and EXACTLY 0.010 below it: not "by more than"
                                         (99, COSTS), (0, COSTS)])                          # below zero by more than 0.010
def test_the_dose_has_three_outcomes_and_each_edge_is_read_on_whole_numbers(stock, name):
    dose = the_dose([], _of(stock), _theirs(110))
    assert (dose["name"], dose["difference"], dose["stock"], dose["deep"], dose["a_cost"]) == (name, round((stock - 110) / 1000, 6), {"successes": stock, "attempts": 1000},
                                                                                               {"successes": 110, "attempts": 1000}, 0.01)
    assert f"{stock - 110:+.2f}" in dose["reason"] and "`stock` minus `deep` of L5d's run of this seed" in dose["reason"]
    follows = what_follows(dose, [4, 5])
    assert follows["name"] == (FURTHER_SEEDS if name == ESCALATE else NO_FURTHER_SEED) and (follows.get("seeds") == [4, 5]) is (name == ESCALATE)


def test_the_dose_is_decided_on_exact_fractions_never_on_a_rounded_figure():
    # 392 goal problems of 32 attempts: 12,544 attempts. One success under L5d's `deep` is 0.08 per 1,000 below zero: NOT SHOWN. 125 under is 0.009965: NOT SHOWN; 126 under: IT COSTS.
    theirs = {"read": True, "figures": {PUBLISHED: _of(1140, 12544), SHALLOW: _of(1247, 12544), DEEP: _of(1382, 12544)}}
    assert [the_dose([], _of(1382 - under, 12544), theirs)["name"] for under in (0, 1, 125, 126)] == [ESCALATE, NOT_SHOWN_AT_ONE, NOT_SHOWN_AT_ONE, COSTS]
    # Another number of attempts on each side (never so in a run that is read: `of_l5d` holds the measurement): the rates are compared, not the counts.
    assert the_dose([], _of(11, 100), theirs)["name"] == NOT_SHOWN_AT_ONE and the_dose([], _of(12, 100), theirs)["name"] == ESCALATE


def test_the_dose_is_inconclusive_when_a_check_failed_and_not_read_when_l5ds_run_is_not_and_nothing_then_follows():
    failed = the_dose(["`stock`: lean_answered"], _of(110), _theirs(110))
    assert failed == {"name": INCONCLUSIVE, "reason": "a check of this run failed (`stock`: lean_answered): the dose is not read"}
    not_read = the_dose([], _of(110), {"read": False, "why": "it is not on the box"})
    assert not_read == {"name": NOT_READ, "reason": "`stock` minus `deep` of L5d's run of this seed is NOT READ: it is not on the box"}
    assert the_dose(["x"], _of(110), {"read": False, "why": "y"})["name"] == INCONCLUSIVE                               # a failed check is said first
    for dose in (failed, not_read):
        assert what_follows(dose, [4, 5]) == {"name": NOT_READ, "reason": f"the dose reads {dose['name']}: nothing is named for what follows"}
    assert "no further seed; the dose is not the lever at this size" in what_follows({"name": COSTS}, [4, 5])["reason"]
    assert "seeds [4, 5] of this stage, each after L5d's own run of that seed" in what_follows({"name": ESCALATE}, [4, 5])["reason"]


def test_two_differences_stand_beside_the_dose_and_name_nothing():
    figures = {STOCK: _of(110), PUBLISHED_STOCK: _of(95)}
    found = beside(figures, _theirs(110))
    assert found["own_proofs_at_this_dose"] == {"what": found["own_proofs_at_this_dose"]["what"], "difference": 0.015, "l5ds_primary_at_this_seed": 0.015}
    assert found["the_steps_without_its_own_proofs"] == {"what": found["the_steps_without_its_own_proofs"]["what"], "read": True, "why": None, "difference": 0.005}       # 95 against that seed's `published` 90
    assert "name" not in found["own_proofs_at_this_dose"] and "name" not in found["the_steps_without_its_own_proofs"]
    alone = beside(figures, {"read": False, "why": "it is not on the box"})
    assert alone["own_proofs_at_this_dose"]["difference"] == 0.015 and alone["own_proofs_at_this_dose"]["l5ds_primary_at_this_seed"] is None      # this run's own difference needs no other run
    assert alone["the_steps_without_its_own_proofs"] == {"what": alone["the_steps_without_its_own_proofs"]["what"], "read": False, "why": "it is not on the box", "difference": None}


def test_a_model_is_read_against_the_start_models_stored_first_sampling_paired_by_problem():
    # The start model solves g1 and g2 (16 and 1 of 32); the model loses g2, gains g3, and g2 and g3 are the long ones.
    found = against_the_start(_on_g((20, 0, 2, 0)), START, G_PARTS, ["g2", "g3"])
    assert found["per_attempt"]["all_of_g"] == {"problems": 4, "model": round(22 / 128, 6), "start": round(17 / 128, 6), "difference": round(5 / 128, 6)}
    assert found["per_attempt"]["solved_by_the_start"] == {"problems": 2, "model": round(20 / 64, 6), "start": round(17 / 64, 6), "difference": round(3 / 64, 6)}
    assert found["per_attempt"]["goal_set_again"] == {"problems": 2, "model": round(2 / 64, 6), "start": 0.0, "difference": round(2 / 64, 6)}
    assert (found["solved"], found["solved_by_the_start"], found["gained"], found["lost"]) == (2, 2, 1, 1)
    assert found["long"] == {"problems": 2, "solved": 1, "gained": 1, "lost": 1} and found["long_solved_by_the_start"] == 1
    # The same rows as the start model's: nothing gained, nothing lost, every difference zero.
    same = against_the_start(START, START, G_PARTS, [])
    assert (same["gained"], same["lost"], same["per_attempt"]["all_of_g"]["difference"]) == (0, 0, 0.0) and same["long"] == {"problems": 0, "solved": 0, "gained": 0, "lost": 0}
    # A part that holds no problem gives no figure on it.
    empty = against_the_start(_on_g((1, 1, 1, 1)), _on_g((1, 1, 1, 1)), {**G_PARTS, "goal_set_again": []}, [])
    assert empty["per_attempt"]["goal_set_again"] == {"problems": 0, "model": None, "start": None, "difference": None}
    assert on_the_parts(_on_g((20, 0, 2, 0)), G_PARTS)[ALL]["successes"] == 22
