"""L5n, the rules and the report on hand-made rows: the learning-rate factor of the recipe as it is and with the decay, step
by step, and what a training recorded held to it; the fixed rows held to the file, to L5's draw and to what that training
recorded; each order another, and none the one trained before; a model's figure and its binomial noise; the four reads,
each with its threshold and its names; what follows; the checks; the report. Spec: docs/spec/ladder-loop.spec.md,
"L5n: what is one training worth? The spread between trainings, and a recipe that lowers it". The stage in the scripted
world is `test_ladder_l5_noise_stage.py`; the mean of adapters is `test_adapter_files.py`."""

import math
import statistics
from pathlib import Path

import pytest
import yaml

from rlvr_lean.domain.ladder_round.assembly import training_order
from rlvr_lean.domain.ladder_round.dose import RATE_TOLERANCE, decayed, held_to_the_schedule, learning_rates, scheduler_factor, steps_in_all, warmed
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ
from rlvr_lean.domain.ladder_round.l4_rows import rehearsal_rows
from rlvr_lean.domain.ladder_round.l5_noise import (
    AS_IT_IS,
    AT_LEAST_THE_MEAN,
    AVERAGE_OF_FOUR,
    AVERAGED,
    AVERAGING,
    DECAY_COSTS,
    DECAY_LOWERS,
    F_CRITICAL,
    FOLLOWS,
    LESS_THAN_THE_MEAN,
    LEVEL,
    LOWERS,
    MODELS,
    NAMED_NOTHING,
    ORDER_ALONE,
    ORDERS,
    RECIPES,
    ROWS_DRAWN,
    T_CRITICAL,
    THE_RECIPES_OWN,
    TRAINED,
    WITH_THE_DECAY,
    averaging,
    decay_lowers,
    figure,
    its_level,
    mean_of,
    model_of,
    models_of,
    numbered,
    only_the_fixed_rows,
    parts_of,
    per_attempt,
    sampling_noise,
    spread,
    spread_as_it_is,
    the_orders,
    the_rows,
    what_follows,
)
from rlvr_lean.reporting.ladder_l4 import SAY
from rlvr_lean.reporting.ladder_l5_noise import LEAN_ANSWERED, THE_ADAPTER, THE_MEAN, THE_ROWS, build_l5_noise_report, the_checks

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
RATE, WARMUP = CONFIG["training"]["learning_rate"], CONFIG["training"]["warmup_steps"]
NOT_SHOWN = "NOT SHOWN"


# ------------------------------------------------------------------------------------- the learning rate
def test_the_recipe_as_it_is_rises_over_the_warm_up_and_stays_and_the_decay_falls_in_a_line_to_zero_after_the_last_step():
    assert (RATE, WARMUP) == (1e-4, 5)
    # THE RECIPE AS IT IS: a fifth more each warm-up step, one at its last and ever after. It is what the training has always handed its scheduler, whose `step` is the
    # updates made so far (the optimizer step less one): the very expression, number for number.
    assert [warmed(step, 5) for step in (1, 2, 3, 4, 5, 6, 50, 100)] == [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]
    as_it_is = scheduler_factor(5)
    assert all(as_it_is(step - 1) == min(1.0, ((step - 1) + 1) / 5) == warmed(step, 5) for step in range(1, 201))
    # WITH THE DECAY, 100 steps: the warm-up is the recipe's own; then a straight line from one at the warm-up's last step to zero at the step AFTER the last.
    assert [decayed(step, 5, 100) for step in (1, 2, 3, 4, 5)] == [warmed(step, 5) for step in (1, 2, 3, 4, 5)] == [0.2, 0.4, 0.6, 0.8, 1.0]
    assert (decayed(6, 5, 100), decayed(50, 5, 100), decayed(99, 5, 100), decayed(100, 5, 100)) == (95 / 96, 51 / 96, 2 / 96, 1 / 96)
    assert decayed(100, 5, 100) > 0 and decayed(101, 5, 100) == 0 == decayed(140, 5, 100)             # THE LAST STEP IS STILL A STEP; past it there is nothing
    falls = [decayed(step, 5, 100) for step in range(5, 102)]
    assert all(math.isclose(before - after, 1 / 96) for before, after in zip(falls, falls[1:]))      # one line, the same fall every step
    with_decay = scheduler_factor(5, 100)
    assert all(with_decay(step - 1) == decayed(step, 5, 100) for step in range(1, 102)) and with_decay(5) != as_it_is(5)
    # A training no longer than its warm-up has no step after it: every factor is the recipe's own (a smoke run's one step).
    for steps in (1, 3, 5):
        assert [decayed(step, 5, steps) for step in range(1, steps + 1)] == [warmed(step, 5) for step in range(1, steps + 1)]
    assert [decayed(step, 5, 6) for step in range(1, 7)] == [0.2, 0.4, 0.6, 0.8, 1.0, 0.5] and [decayed(step, 5, 8) for step in (6, 7, 8)] == [0.75, 0.5, 0.25]
    # The steps of a whole training, as the loop makes them: so many passes, or one pass over each order given.
    assert (steps_in_all(800, 8, 1), steps_in_all(801, 8, 1), steps_in_all(800, 8, 3), steps_in_all(5, 8, 1, [list(range(800))]), steps_in_all(5, 8, 9, [[0] * 9, [0] * 16])) == (100, 101, 300, 100, 4)
    # Each step's learning rate: the config's times the factor.
    assert learning_rates(RATE, 5, 100, False) == [RATE * warmed(step, 5) for step in range(1, 101)] and learning_rates(RATE, 5, 100, True) == [
        RATE * decayed(step, 5, 100) for step in range(1, 101)]
    assert learning_rates(RATE, 5, 100, True)[:5] == learning_rates(RATE, 5, 100, False)[:5] and learning_rates(RATE, 5, 100, True)[5:] != learning_rates(RATE, 5, 100, False)[5:]
    assert sum(learning_rates(1.0, 5, 100, False)) == pytest.approx(98.0) and sum(learning_rates(1.0, 5, 100, True)) == pytest.approx(50.5)        # about half the recipe's total movement


def test_what_a_training_recorded_of_its_learning_rate_is_held_to_the_schedule_it_was_asked_for():
    constant, decay = learning_rates(RATE, 5, 100, False), learning_rates(RATE, 5, 100, True)
    held = held_to_the_schedule(decay, RATE, 5, True)
    assert (held["decay"], held["steps"], held["warmup_steps"], held["learning_rate"]) == (True, 100, 5, RATE) and "straight line" in held["what"]
    assert (held["of_the_first_step"], held["of_the_warmups_last_step"], held["of_the_step_after_the_warmup"], held["of_the_last_step"]) == (RATE / 5, RATE, RATE * (95 / 96), RATE * (1 / 96))
    assert held["factor_of_the_last_step"] == pytest.approx(1 / 96) and held["largest"] == RATE
    as_it_is = held_to_the_schedule(constant, RATE, 5, False)
    assert (as_it_is["decay"], as_it_is["of_the_step_after_the_warmup"], as_it_is["of_the_last_step"], as_it_is["factor_of_the_last_step"]) == (False, RATE, RATE, 1.0) and "as it is" in as_it_is["what"]
    # A TRAINING WITH THE DECAY WHOSE RATE DID NOT FALL is the recipe under another name, and one without it whose rate fell is no training of the recipe: refused.
    with pytest.raises(ValueError, match="the learning rate recorded at step 6 is 0.0001 and the schedule with the decay gives 9.895833333333334e-05 .95 of 100 steps are not the schedule's."):
        held_to_the_schedule(constant, RATE, 5, True)
    with pytest.raises(ValueError, match="recorded at step 6 is 9.895833333333334e-05 and the schedule of the recipe as it is gives 0.0001 .95 of 100 steps"):
        held_to_the_schedule(decay, RATE, 5, False)
    for nothing in (None, []):
        with pytest.raises(ValueError, match="the training recorded no learning rate"):
            held_to_the_schedule(nothing, RATE, 5, False)
    # ONE step off is refused, at any step: the warm-up's, the middle, the last; the allowance is a billionth of the rate and nothing more.
    for step in (1, 5, 50, 100):
        off = list(decay)
        off[step - 1] += 2 * RATE_TOLERANCE * RATE
        with pytest.raises(ValueError, match=f"recorded at step {step} is .* .1 of 100 steps are not the schedule's."):
            held_to_the_schedule(off, RATE, 5, True)
        off[step - 1] = decay[step - 1] + 0.5 * RATE_TOLERANCE * RATE
        assert held_to_the_schedule(off, RATE, 5, True)["steps"] == 100
    # A one-step training (a smoke run's): inside the warm-up, the decay's rate is the recipe's.
    assert held_to_the_schedule([RATE / 5], RATE, 5, True)["of_the_last_step"] == held_to_the_schedule([RATE / 5], RATE, 5, False)["of_the_last_step"] == RATE / 5


# -------------------------------------------------------------------------------------------- the models
def test_the_models_are_each_order_with_each_recipe_and_one_averaged_adapter_a_recipe():
    assert (RECIPES, ORDERS) == (("constant", "decay"), (1, 2, 3, 4)) and CONFIG["ladder_loop"]["l5_noise"]["orders"] == list(ORDERS) and CONFIG["ladder_loop"]["l5_noise"]["recipes"] == list(RECIPES)
    assert TRAINED == ("constant_1", "decay_1", "constant_2", "decay_2", "constant_3", "decay_3", "constant_4", "decay_4") and AVERAGED == ("mean_constant", "mean_decay")
    assert MODELS == (*TRAINED, *AVERAGED) == models_of(ORDERS) and models_of((1, 2)) == ("constant_1", "decay_1", "constant_2", "decay_2", "mean_constant", "mean_decay")
    assert (model_of("decay", 3), mean_of("decay"), parts_of("decay_3"), parts_of("constant_12")) == ("decay_3", "mean_decay", ("decay", 3), ("constant", 12))
    for name in ("mean_constant", "published", "decay", "decay_x", "constant_"):
        with pytest.raises(ValueError, match="is no trained model of L5n"):
            parts_of(name)
    assert len(ORDERS) - 1 in T_CRITICAL and (T_CRITICAL[3], F_CRITICAL[3]) == (3.182, 9.28)            # the section's two figures, at four orders


# ---------------------------------------------------------------------------------------------- the rows
PRETRAINING = [{"problem_id": f"p{index}", "kind": "stp_conjecture", "proof_lines": 2 + index % 5, "lines": 2 + index % 5} for index in range(30)]


def _stored(count=8, seed=0):
    """The set L5's own task stores for `published`: the first `count` rows of the hash order of its seed, in the order it trains them in, each with its place in the set."""
    return numbered(training_order(rehearsal_rows(PRETRAINING, count, seed), seed))


def test_the_fixed_rows_are_l5s_stored_set_held_to_the_file_to_the_rule_and_to_what_that_training_recorded():
    stored = _stored()
    ids = [row["id"] for row in stored]
    rows = the_rows(stored, ids, PRETRAINING, 0)
    assert rows == [{key: value for key, value in row.items() if key != "row"} for row in stored] and len(rows) == 8 and all("row" not in row for row in rows)
    assert all(row["origin"] == "pretraining" and PRETRAINING[row["row_of_the_file"]]["problem_id"] == row["problem_id"] and row["id"] == f"{row['problem_id']}#pretraining" for row in rows)
    assert sorted(row["id"] for row in rows) == sorted(row["id"] for row in rehearsal_rows(PRETRAINING, 8, 0)) != sorted(row["id"] for row in rehearsal_rows(PRETRAINING, 8, 1))

    def refused(match, own=stored, trained=ids, seed=0):
        with pytest.raises(ValueError, match=match):
            the_rows(own, trained, PRETRAINING, seed)

    # A published row is NEVER stored with its text.
    refused("1 rehearsal rows of the stored set of L5's `published` carry a text field", [{**stored[0], "proof": "by simp"}, *stored[1:]])
    # No row, a row that is no published row, an id twice.
    refused("holds 0 rows, 0 that are no published rows and 0 ids twice: the fixed rows are published rows, each once", [])
    refused("holds 8 rows, 1 that are no published rows and 0 ids twice", [{**stored[0], "origin": "deep"}, *stored[1:]])
    refused("holds 9 rows, 0 that are no published rows and 1 ids twice", [*stored, stored[0]])
    # BY ID AND PLACE: a row whose place in the file is another problem's, lies beyond the file, or whose id is not its problem's.
    refused("1 rows of the stored set of L5's `published` are not the rows of the pretraining file at their place", [{**stored[0], "row_of_the_file": (stored[0]["row_of_the_file"] + 1) % 30}, *stored[1:]])
    refused("1 rows of the stored set .* are not the rows of the pretraining file at their place", [{**stored[0], "row_of_the_file": 30}, *stored[1:]])
    refused("1 rows of the stored set .* are not the rows of the pretraining file at their place", [{**stored[0], "id": "another#pretraining"}, *stored[1:]])
    # THE RULE'S DRAW: a set drawn with another seed, or holding a row the rule does not draw, was made otherwise.
    of_seed_1 = _stored(seed=1)
    refused("are not the first 8 of the hash order of seed 0 among the 30 rows the start model was pretrained on .first that differs: .*: that set was made otherwise", of_seed_1, [row["id"] for row in of_seed_1])
    later = next(row for row in rehearsal_rows(PRETRAINING, 30, 0) if row["id"] not in ids)
    refused("are not the first 8 of the hash order of seed 0", [later, *stored[1:]], [later["id"], *ids[1:]])
    # THE ORDER it was trained in, and the training's OWN record: these rows in this order.
    refused("does not stand in the content-hash order of seed 0, or its training's own record .8 rows. is not these rows in this order", [stored[1], stored[0], *stored[2:]], [ids[1], ids[0], *ids[2:]])
    refused("its training's own record .8 rows. is not these rows in this order", stored, [ids[1], ids[0], *ids[2:]])
    refused("its training's own record .7 rows. is not these rows in this order", stored, ids[:-1])
    assert numbered([{"row": 9, "id": "a"}, {"id": "b"}]) == [{"row": 0, "id": "a"}, {"row": 1, "id": "b"}]


def test_each_order_seed_gives_the_fixed_rows_another_order_and_none_is_the_order_trained_before():
    rows = the_rows(_stored(), [row["id"] for row in _stored()], PRETRAINING, 0)
    by_order, coincide = the_orders(rows, ORDERS)
    assert list(by_order) == list(ORDERS) and coincide == {"the_order_trained_before": [], "alike": []}
    for order, own in by_order.items():                 # each order is ITS seed's: the content hash of that seed and each row's id; the same rows; never the one trained before
        assert own == list(training_order(rows, order)) and sorted(row["id"] for row in own) == sorted(row["id"] for row in rows) and [row["id"] for row in own] != [row["id"] for row in rows]
    assert len({tuple(row["id"] for row in own) for own in by_order.values()}) == 4 and [row["id"] for row in training_order(rows, 0)] == [row["id"] for row in rows]
    # THE ORDER L5's OWN TASK TRAINED IN (its seed's) is refused among the orders; so are two orders that are one.
    with pytest.raises(ValueError, match=r"of the order seeds \[0, 1\], \[0\] give the order L5's own task trained these 8 rows in, and the pairs \[\] give one order"):
        the_orders(rows, (0, 1))
    with pytest.raises(ValueError, match=r"of the order seeds \[1, 2, 1\], \[\] give the order L5's own task trained these 8 rows in, and the pairs \[\[1, 1\]\] give one order"):
        the_orders(rows, (1, 2, 1))
    # A SMOKE RUN ALONE: a handful of rows has few orders, and such an order is trained all the same, recorded.
    loose, found = the_orders(rows, (0, 1, 1), loosely=True)
    assert found == {"the_order_trained_before": [0], "alike": [[1, 1]]} and list(loose) == [0, 1]
    two = rows[:2]
    assert the_orders(two, (1, 2), loosely=True)[1]["the_order_trained_before"] == [order for order in (1, 2) if [row["id"] for row in training_order(two, order)] == [row["id"] for row in two]]


def test_the_check_of_a_trained_models_rows_no_row_is_anything_but_one_of_the_fixed_rows_and_each_stands_once():
    fixed = the_rows(_stored(), [row["id"] for row in _stored()], PRETRAINING, 0)
    trained = [{key: row[key] for key in ("id", "problem_id", "origin", "row_of_the_file")} for row in training_order(fixed, 3)]
    found = only_the_fixed_rows(trained, fixed)
    assert found == {"what": found["what"], "rows": 8, "fixed_rows": 8, "rows_that_are_none_of_them": 0, "rows_twice": 0, "fixed_rows_not_trained_on": 0, "first": [], "passes": True}
    stranger = {"id": "p29#pretraining", "problem_id": "p29", "origin": "pretraining", "row_of_the_file": 29}
    assert stranger["id"] not in {row["id"] for row in fixed}
    for rows_, key, first in (([stranger, *trained[1:]], "rows_that_are_none_of_them", stranger["id"]), ([*trained, trained[0]], "rows_twice", trained[0]["id"]),
                              (trained[:-1], "fixed_rows_not_trained_on", trained[-1]["id"])):
        failed = only_the_fixed_rows(rows_, fixed)
        assert failed["passes"] is False and failed[key] == 1 and first in failed["first"]
    beside = only_the_fixed_rows([*trained, stranger], fixed)             # every fixed row once, and ONE MORE that is none of them
    assert (beside["passes"], beside["rows"], beside["rows_that_are_none_of_them"], beside["rows_twice"], beside["fixed_rows_not_trained_on"]) == (False, 9, 1, 0, 0)
    # BY PLACE TOO: a row under a fixed row's id at another place of the file is none of them.
    moved = only_the_fixed_rows([{**trained[0], "row_of_the_file": 29}, *trained[1:]], fixed)
    assert (moved["passes"], moved["rows_that_are_none_of_them"], moved["fixed_rows_not_trained_on"]) == (False, 1, 1) and only_the_fixed_rows([], fixed)["passes"] is False


# -------------------------------------------------------------------------------------------- the figure
def _rows(successes, attempts=32, timed_out=0, sides=1, without=0):
    """One sampling's per-problem rows: `successes[i]` of `attempts` on problem i."""
    return [{"problem_id": f"g{index}", "resolved": count, "episodes": attempts, "sides": sides, "attempts_timed_out": timed_out, "attempts_without_an_answer": without,
             "attempts_capped": 0} for index, count in enumerate(successes)]


def _totalling(total, problems=4, attempts=32):
    """Rows of `problems` goal problems whose successes add up to `total`."""
    counts = []
    for _ in range(problems):
        counts.append(min(attempts, total))
        total -= counts[-1]
    return _rows(counts, attempts)


def test_a_models_figure_is_its_successes_per_attempt_on_all_of_g_with_its_binomial_sampling_noise():
    rows = _rows([0, 8, 16, 32], timed_out=2)
    assert per_attempt(rows) == 56 / 128 and per_attempt([]) is None and sampling_noise([]) is None
    # From each problem's own rate p over its n attempts: the square root of the sum of p(1 - p) / n, over the number of problems. A problem never or always solved adds nothing.
    by_hand = math.sqrt(0.25 * 0.75 / 32 + 0.5 * 0.5 / 32) / 4
    assert sampling_noise(rows) == pytest.approx(by_hand) and sampling_noise(_rows([0, 32, 32])) == 0.0
    assert sampling_noise(_rows([16], attempts=32)) == pytest.approx(math.sqrt(0.25 / 32))               # one problem at a half: the binomial's own standard error
    assert sampling_noise(_rows([8] * 4)) == pytest.approx(sampling_noise(_rows([8] * 16)) * 2)          # four times the problems, half the noise
    own = figure(rows)
    assert own == {"problems": 4, "successes": 56, "attempts": 128, "per_attempt": 0.4375, "per_1000": 437.5, "sampling_noise": round(by_hand, 6), "goal_problems_solved": 3,
                   "timed_out_in_lean": 8, "share_timed_out_in_lean": round(8 / 128, 5)}
    assert figure(_rows([1, 0], sides=2, timed_out=4))["share_timed_out_in_lean"] == round(8 / 128, 5)   # an audited problem is attempted on both sides
    # ... and its EPISODES are its attempts in the figure: 8 of 32, whichever side each was won on (the timeouts alone are counted over both sides).
    assert per_attempt(_rows([8], sides=2)) == 8 / 32 == per_attempt(_rows([8])) and figure(_rows([8, 0], sides=2))["attempts"] == 64
    assert spread([0.1, 0.2, 0.3, 0.4]) == pytest.approx(statistics.stdev([0.1, 0.2, 0.3, 0.4])) == pytest.approx(0.1290994) and spread([0.3]) is None and spread([]) is None


# --------------------------------------------------------------------------------------------- the reads
def test_the_spread_of_the_recipe_as_it_is_and_whether_order_alone_makes_it():
    constant, noises = [0.125, 0.15625, 0.1875, 0.21875], [0.003, 0.004, 0.003, 0.004]
    own = statistics.stdev(constant)
    read = spread_as_it_is([], constant, noises, [0.10, 0.10 + 2 * own, 0.10 + own])      # L5's `published` over three seeds: ITS spread is the constant one's (three evenly spaced)
    assert read["name"] == ORDER_ALONE and read["standard_deviation"] == round(own, 6) and read["trainings"] == 4 and read["seeds"] == 3
    assert read["sampling_noise_of_one_figure"] == round(math.sqrt((2 * 0.003 ** 2 + 2 * 0.004 ** 2) / 4), 6)        # the root of the mean square of the four
    assert "is at least half of the second" in read["reason"] and "rows and order both drawn" in read["reason"]
    # AT LEAST HALF of the second: exactly half is order alone; a hair under is the rows drawn.
    of_l5 = lambda sd: [0.2 - sd, 0.2, 0.2 + sd]      # noqa: E731 - three seeds whose standard deviation is `sd`
    assert spread(of_l5(2 * own)) == pytest.approx(2 * own)
    exactly = spread_as_it_is([], constant, noises, [0.0, 2 * own, 4 * own])
    assert (exactly["of_published_over_the_seeds"], exactly["half_of_it"], exactly["name"]) == (round(2 * own, 6), round(own, 6), ORDER_ALONE)
    under = spread_as_it_is([], constant, noises, of_l5(2.01 * own))
    assert under["name"] == ROWS_DRAWN and "is less than half of the second" in under["reason"] and spread_as_it_is([], constant, noises, of_l5(0.2 * own))["name"] == ORDER_ALONE
    # L5's seeds NOT THERE: the spread is given and the name is NOT READ, with which seed and why.
    absent = spread_as_it_is([], constant, noises, None, "seed 2: its run is not on the box")
    assert (absent["name"], absent["standard_deviation"], absent["of_published_over_the_seeds"]) == (NOT_READ, round(own, 6), None)
    assert "NOT THERE (seed 2: its run is not on the box): whether order alone makes the spread is not read" in absent["reason"]
    assert spread_as_it_is([], constant, noises, [0.1])["name"] == NOT_READ                              # one seed has no spread
    failed = spread_as_it_is(["`decay_2`: lean_answered"], constant, noises, of_l5(own))
    assert failed == {"name": INCONCLUSIVE, "failed_checks": ["`decay_2`: lean_answered"],
                      "reason": "the run could not have been read (`decay_2`: lean_answered): the spread of the recipe as it is is not read"}
    assert all(spread_as_it_is(checks, constant, noises, seeds)["name"] in AS_IT_IS for checks in ([], ["x"]) for seeds in (None, of_l5(own), of_l5(9 * own)))


def test_does_decay_lower_it_at_most_half_and_what_four_against_four_can_show():
    constant = [0.125, 0.15625, 0.1875, 0.21875]
    half = [0.15625, 0.171875, 0.1875, 0.203125]                    # the same four, half as far apart
    assert spread(half) == spread(constant) / 2
    lowers = decay_lowers([], constant, half)
    assert lowers["name"] == DECAY_LOWERS and (lowers["of_decay_over_of_constant"], lowers["of_constant"], lowers["of_decay"]) == (0.5, round(spread(constant), 6), round(spread(half), 6))
    assert "at most half. With 4 against 4 only a ratio of about 3.0 between two standard deviations is beyond chance (F at 3 and 3 degrees, 9.28)" in lowers["reason"]
    assert (lowers["f_at_equal_degrees"], lowers["ratio_beyond_chance"]) == (9.28, round(math.sqrt(9.28), 3))
    # AT MOST HALF: a hair more than half is NOT SHOWN; so is the same spread, and a wider one.
    a_hair_more = [0.15625 - 1e-6, 0.171875, 0.1875, 0.203125 + 1e-6]
    assert decay_lowers([], constant, a_hair_more)["name"] == NOT_SHOWN and "more than half" in decay_lowers([], constant, a_hair_more)["reason"]
    assert decay_lowers([], constant, constant)["name"] == NOT_SHOWN and decay_lowers([], half, constant)["name"] == NOT_SHOWN
    assert decay_lowers([], constant, [0.2] * 4)["name"] == DECAY_LOWERS                                  # no spread at all with the decay
    # A recipe with no spread has none to lower; fewer than two trainings have none to compare.
    none = decay_lowers([], [0.2] * 4, [0.2] * 4)
    assert none["name"] == NOT_SHOWN and "there is no spread to lower" in none["reason"] and decay_lowers([], [0.2], [0.3])["name"] == NOT_SHOWN
    two = decay_lowers([], [0.1, 0.3], [0.2, 0.21])                 # a smoke run's two orders: F at 1 and 1
    assert two["name"] == DECAY_LOWERS and two["f_at_equal_degrees"] == 161.4 and "F at 1 and 1 degrees, 161.4" in two["reason"]
    assert decay_lowers(["x"], constant, half)["name"] == INCONCLUSIVE and all(decay_lowers(checks, constant, other)["name"] in LOWERS for checks in ([], ["x"]) for other in (half, constant))


def test_the_level_of_the_decay_is_the_paired_difference_with_its_t_interval_at_three_degrees():
    constant, decay = [0.100, 0.120, 0.090, 0.110], [0.098, 0.121, 0.084, 0.105]
    differences = [after - before for after, before in zip(decay, constant)]
    mean, half_width = statistics.fmean(differences), 3.182 * statistics.stdev(differences) / 2
    level = its_level([], list(zip(decay, constant)))
    assert level["differences"] == [round(value, 6) for value in differences] and (level["degrees_of_freedom"], level["critical_value"]) == (3, 3.182)
    assert (level["mean"], level["low"], level["high"]) == (round(mean, 6), round(mean - half_width, 6), round(mean + half_width, 6))
    assert level["name"] == NAMED_NOTHING and level["low"] < 0 < level["high"] and "it is not below zero: the level is given and named nothing" in level["reason"]
    # DECAY COSTS when the interval is BELOW ZERO: every order lower, by about as much.
    costs = its_level([], [(before - 0.010 - 0.001 * index, before) for index, before in enumerate(constant)])
    assert costs["name"] == DECAY_COSTS and costs["high"] < 0 and "it is below zero" in costs["reason"]
    # Above zero names nothing: the level is given. So does an interval that ends AT zero.
    assert its_level([], [(before + 0.010 + 0.001 * index, before) for index, before in enumerate(constant)])["name"] == NAMED_NOTHING
    at_zero = its_level([], [(1.0 - 1.0, 0.0), (0.0, 1.0 - 0.0), (0.0, 1.0), (0.0, 0.0)])            # differences 0, -1, -1, 0
    assert at_zero["mean"] == -0.5 and at_zero["name"] == NAMED_NOTHING and at_zero["high"] > 0
    assert its_level([], [(0.0, 1.0)] * 4)["high"] == -1.0 and its_level([], [(0.0, 1.0)] * 4)["name"] == DECAY_COSTS       # the same at every order: no width, below zero
    # The order's own pair: `decay_k` minus `constant_k`, not the other way round.
    assert its_level([], [(0.2, 0.1)] * 4)["mean"] == 0.1
    # Two orders (a smoke run): one degree, 12.706. More than four: its critical value is not given here.
    assert its_level([], [(0.2, 0.1), (0.3, 0.1)])["critical_value"] == 12.706 and its_level([], [(0.2, 0.1)] * 3)["critical_value"] == 4.303
    with pytest.raises(ValueError, match=r"the t interval of 5 paired differences is not given here: its critical values are those of \[2, 3, 4\] orders"):
        its_level([], [(0.2, 0.1)] * 5)
    assert its_level(["x"], [])["name"] == INCONCLUSIVE and all(its_level([], pairs)["name"] in LEVEL for pairs in ([(0.2, 0.1)] * 4, [(0.1, 0.2)] * 4))


def test_averaging_gives_at_least_the_mean_of_its_models_unless_it_is_below_it_by_more_than_twice_its_noise():
    models = [0.100, 0.120, 0.090, 0.110]                           # mean 0.105, best 0.120
    gives = averaging([], 0.104, 0.003, models, "constant")
    assert gives["name"] == AT_LEAST_THE_MEAN and (gives["model"], gives["mean_of_its_models"], gives["best_of_its_models"]) == ("mean_constant", 0.105, 0.12)
    assert (gives["minus_the_mean"], gives["minus_the_best"], gives["allowed_below_the_mean"], gives["sampling_noise"]) == (-0.001, -0.016, 0.006, 0.003)
    assert "the mean of its 4 models" in gives["reason"] and "`mean_constant` has 0.10400 per attempt against 0.10500" in gives["reason"]
    # NOT BELOW THE MEAN BY MORE THAN TWICE ITS OWN NOISE: exactly twice below still gives; a hair more does not. Above the best it gives, of course.
    mean = statistics.fmean(models)
    assert averaging([], mean - 0.25, 0.125, [mean] * 4, "decay")["name"] == AT_LEAST_THE_MEAN            # exactly twice, in numbers a float holds exactly
    assert averaging([], 0.25, 0.125, [0.5] * 4, "decay")["name"] == AT_LEAST_THE_MEAN and averaging([], 0.249, 0.125, [0.5] * 4, "decay")["name"] == LESS_THAN_THE_MEAN
    below = averaging([], 0.098, 0.003, models, "decay")
    assert below["name"] == LESS_THAN_THE_MEAN and below["model"] == "mean_decay" and averaging([], 0.130, 0.003, models, "decay")["name"] == AT_LEAST_THE_MEAN
    # ITS noise, not twice something else: a noisier figure is allowed further below.
    assert averaging([], 0.098, 0.004, models, "decay")["name"] == AT_LEAST_THE_MEAN and averaging([], 0.098, 0.0034, models, "decay")["name"] == LESS_THAN_THE_MEAN
    # Against the MEAN of its models, not their best or their least.
    assert averaging([], 0.103, 0.0005, [0.100, 0.100, 0.100, 0.120], "constant")["name"] == LESS_THAN_THE_MEAN and averaging([], 0.1045, 0.0005, [0.100, 0.100, 0.100, 0.120], "constant")["name"] == AT_LEAST_THE_MEAN
    failed = averaging(["x"], 0.1, 0.003, models, "decay")
    assert failed["name"] == INCONCLUSIVE and "the averaged adapter `mean_decay` is not read" in failed["reason"]
    assert all(averaging([], value, 0.003, models, "decay")["name"] in AVERAGING for value in (0.05, 0.105, 0.2))


def test_what_follows_by_the_sections_three_cases_in_its_order():
    named = lambda name: {"name": name}      # noqa: E731
    # DECAY LOWERS THE SPREAD and does not cost: every later stage trains with it, whatever the averaging reads.
    for of_the_average in (AT_LEAST_THE_MEAN, LESS_THAN_THE_MEAN):
        first = what_follows(named(DECAY_LOWERS), named(NAMED_NOTHING), named(of_the_average))
        assert first["name"] == WITH_THE_DECAY and "L5's three kinds are trained again with it before L6 runs" in first["reason"]
    # NOT THAT (it does not lower it, OR it lowers it and costs), and averaging gives at least the mean: a later stage trains each model four times and averages.
    for lowers, level in ((NOT_SHOWN, NAMED_NOTHING), (NOT_SHOWN, DECAY_COSTS), (DECAY_LOWERS, DECAY_COSTS)):
        assert what_follows(named(lowers), named(level), named(AT_LEAST_THE_MEAN))["name"] == AVERAGE_OF_FOUR
        # NEITHER: the spread is the recipe's own at this size, and the choice needs a decision.
        neither = what_follows(named(lowers), named(level), named(LESS_THAN_THE_MEAN))
        assert neither["name"] == THE_RECIPES_OWN and "needs a decision, with these figures" in neither["reason"]
    for failed in ((INCONCLUSIVE, NAMED_NOTHING, AT_LEAST_THE_MEAN), (DECAY_LOWERS, INCONCLUSIVE, AT_LEAST_THE_MEAN), (DECAY_LOWERS, NAMED_NOTHING, INCONCLUSIVE)):
        assert what_follows(*map(named, failed))["name"] == NOT_READ
    assert {what_follows(named(lowers), named(level), named(average))["name"] for lowers in (DECAY_LOWERS, NOT_SHOWN) for level in (DECAY_COSTS, NAMED_NOTHING)
            for average in (AT_LEAST_THE_MEAN, LESS_THAN_THE_MEAN)} == set(FOLLOWS) - {NOT_READ}


# -------------------------------------------------------------------------------------------- the report
FIXED = the_rows(_stored(), [row["id"] for row in _stored()], PRETRAINING, 0)
SAVED = {"rank": 64, "alpha": 128, "ranks": [64], "numbers": 64}
# Successes of 128 attempts (four goal problems, 32 each): the recipe as it is spread wide, the decay half as wide and level with it; each averaged model at its models' mean.
TOTALS = {"constant_1": 16, "decay_1": 20, "constant_2": 20, "decay_2": 22, "constant_3": 24, "decay_3": 24, "constant_4": 28, "decay_4": 26, "mean_constant": 22, "mean_decay": 23}
PUBLISHED_SEEDS = {0: 26, 1: 14, 2: 20}                             # L5's `published` at its three seeds, on the same attempts


def _prepare(orders=ORDERS, smoke=None):
    models = models_of(orders)
    return {"seed": 0, "start": "pre_r64", "orders": list(orders), "recipes": list(RECIPES), "models": list(models), "trained": [name for name in models if not name.startswith("mean_")],
            "averaged": [mean_of(recipe) for recipe in RECIPES], "stand_in_engine": False, "start_recipe": {"rank": 64, "alpha": 128},
            "the_rows": {"rows": len(FIXED), "model": "published", "run": "runs/ladder_l5_deep_pre_r64_seed0", "seed": 0}, "the_orders": {"order_seeds": list(orders)},
            "the_sampling": {"name": "reach", "episodes": 32, "sampling_seed": 1001}, "the_published_of_l5": {"seed": 0},
            "training_sets": {model_of(recipe, order): {"order_seed": order, "recipe": recipe} for order in orders for recipe in RECIPES}, "recipe": {"learning_rate": 1e-4}, "smoke": smoke}


def _report(totals=TOTALS, seeds=PUBLISHED_SEEDS, orders=ORDERS, change=lambda world: None, smoke=None):
    """The report on hand-made rows. `totals`: by model, its successes over the four goal problems. `seeds`: by seed of L5, its `published`'s successes, or why it is not
    there. `change`: what a test moves in the world before it is read."""
    prepare = _prepare(orders, smoke)
    trained_models, averaged_models = prepare["trained"], prepare["averaged"]
    world = {
        "prepare": prepare,
        "trains": {name: {"rows": 8, "steps": 1, "stand_in_engine": False, "adapter_saved": dict(SAVED),
                          "against_the_start_adapter": {name: {"elements": 64, "elements_changed": 60, "largest_change": 0.01}}} for name in trained_models},
        "trained": {name: [{key: row[key] for key in ("id", "problem_id", "origin", "row_of_the_file")} for row in training_order(FIXED, parts_of(name)[1])] for name in trained_models},
        "averaged": {name: {"of": [model_of(name.removeprefix("mean_"), order) for order in orders], "adapter": f"adapters/{name}", "adapter_saved": dict(SAVED), "stand_in_engine": False,
                            "read_back": {"elements": 64, "elements_not_the_mean": 0, "elements_not_those_of_each_adapter": [40] * len(orders), "is_the_mean": True,
                                          "is_none_of_them": True, "passes": True}} for name in averaged_models},
        "measured": {name: {"adapter": f"adapters/{name}", "stand_in_engine": False, "distinct_attempts": {"goal": {"prompts": 4, "mean_share_distinct": 0.9}}} for name in prepare["models"]},
        "rows": {name: _totalling(totals[name]) for name in prepare["models"]}, "fixed": FIXED,
        "published": {seed: ({"there": True, "run": f"runs/seed{seed}", "rows": _totalling(own), "inconclusive": False, "share_of_distinct_attempts": 0.91} if isinstance(own, int)
                             else {"there": False, "run": f"runs/seed{seed}", "why": own}) for seed, own in seeds.items()}}
    change(world)
    return build_l5_noise_report(*(world[key] for key in ("prepare", "trains", "trained", "averaged", "measured", "rows", "fixed", "published")))


def test_the_report_gives_the_checks_first_then_each_figure_with_its_noise_and_the_reads_by_their_names():
    report = _report()
    assert (report["label"], report["check"], report["start"], report["rank"], report["models"], report["orders"], report["recipes"]) == (
        "pretrained on published proofs", "L5n", "pre_r64", 64, list(MODELS), [1, 2, 3, 4], ["constant", "decay"])
    assert report["inconclusive"] is False and report["failed_checks"] == [] and report["ok"] is True and report["not_to_be_read"] == []
    lines = report["lines"]
    assert all(line.startswith(f"{SAY}: ") for line in lines) and [line.split(": ")[1].split(",")[0].split(":")[0] for line in lines[:5]] == ["L5n", "CHECK", "CHECK", "CHECK", "CHECK"]
    assert "L5n, WHAT IS ONE TRAINING WORTH? THE SPREAD BETWEEN TRAININGS, AND A RECIPE THAT LOWERS IT. Seed 0. ONE FIXED SET OF ROWS: the 8 published rows L5's own `published` of seed 0" in lines[0]
    assert "32 attempts at each of 4 goal problems, sampling seed 1001 (the one `pre_r64`'s run recorded): no rung, no second sampling, no held-aside set" in lines[0]
    # THE CHECKS, first: one line a kind, every model in it.
    assert "CHECK, each trained adapter is not the start adapter's and is read back at its rank (64): `constant_1`: 60 of the 64 numbers of its adapter are not the start adapter's; read back at rank 64 and alpha 128, matrices of rank [64]" in lines[1] and lines[1].endswith(": PASS")
    assert "CHECK, no row a model was trained on is anything but one of the 8 fixed rows" in lines[2] and "`decay_4` 8 rows, 0 none of them, 0 twice, 0 not trained on: PASS" in lines[2]
    assert ("CHECK, each averaged adapter's numbers ARE the mean of its models' (read back), it is none of them, and it is served: `mean_constant` (the mean of `constant_1`, `constant_2`, "
            "`constant_3`, `constant_4`): 0 of its 64 numbers are not the mean") in lines[3] and lines[3].endswith("its set was sampled from it: PASS")
    assert "CHECK, Lean answered (at most 0.02 of a set without an answer): `constant_1` 0.0; " in lines[4] and lines[4].endswith("`mean_decay` 0.0: PASS")
    assert list(report["can_this_run_be_read"]) == ["what", *MODELS] and list(report["can_this_run_be_read"]["constant_1"]) == [THE_ADAPTER, THE_ROWS, LEAN_ANSWERED]
    assert list(report["can_this_run_be_read"]["mean_decay"]) == [THE_MEAN, LEAN_ANSWERED]
    # THE FIGURES: each model's successes per attempt over ITS OWN rows, with its own sampling noise.
    figures = report["figures"]
    for name in MODELS:
        own = figure(_totalling(TOTALS[name]))
        assert figures[name] == {**own, "share_of_distinct_attempts": 0.9} and own["per_attempt"] == round(TOTALS[name] / 128, 6)
    assert "THE FIGURES, successes per 1,000 attempts on all of G (4 problems) over the 32 attempts" in lines[5] and f"`constant_1` 125.00 ({1000 * figures['constant_1']['sampling_noise']:.2f}); `decay_1` 156.25" in lines[5]
    # THE SPREAD OF THE RECIPE AS IT IS: the four `constant` figures (none of the decay's, none of the averaged); beside it L5's `published` over its three seeds.
    constant, decay = [TOTALS[f"constant_{order}"] / 128 for order in ORDERS], [TOTALS[f"decay_{order}"] / 128 for order in ORDERS]
    as_it_is = report["the_spread_of_the_recipe_as_it_is"]
    assert as_it_is == spread_as_it_is([], constant, [figures[f"constant_{order}"]["sampling_noise"] for order in ORDERS], [PUBLISHED_SEEDS[seed] / 128 for seed in (0, 1, 2)])
    assert as_it_is["standard_deviation"] == round(statistics.stdev(constant), 6) and as_it_is["of_published_over_the_seeds"] == round(statistics.stdev([26 / 128, 14 / 128, 20 / 128]), 6)
    assert as_it_is["name"] == ORDER_ALONE and lines[6].startswith(f"{SAY}: THE SPREAD OF THE RECIPE AS IT IS: ORDER ALONE MAKES THE SPREAD. the standard deviation of the 4 `constant` figures is")
    # DOES DECAY LOWER IT: the four `decay` figures against the four `constant` ones.
    assert report["does_decay_lower_it"] == decay_lowers([], constant, decay) and report["does_decay_lower_it"]["name"] == DECAY_LOWERS and lines[7].startswith(f"{SAY}: DOES DECAY LOWER IT: DECAY LOWERS THE SPREAD. ")
    # AND ITS LEVEL: `decay_k` minus `constant_k`, order by order.
    assert report["and_its_level"] == its_level([], list(zip(decay, constant))) and report["and_its_level"]["differences"] == [round((b - a) / 128, 6) for a, b in ((16, 20), (20, 22), (24, 24), (28, 26))]
    assert report["and_its_level"]["name"] == NAMED_NOTHING and lines[8].startswith(f"{SAY}: AND ITS LEVEL: NAMED NOTHING. `decay` minus `constant` over the 4 orders is +0.00781 per attempt")
    # AVERAGING: each averaged model against ITS OWN four, with ITS OWN noise.
    assert report["averaging"] == {"mean_constant": averaging([], 22 / 128, figures["mean_constant"]["sampling_noise"], constant, "constant"),
                                   "mean_decay": averaging([], 23 / 128, figures["mean_decay"]["sampling_noise"], decay, "decay")}
    assert [line.split(": ")[1] for line in lines[9:11]] == ["AVERAGING", "AVERAGING"] and "`mean_constant` has 0.17188 per attempt against 0.17188" in lines[9] and "`mean_decay` has 0.17969" in lines[10]
    # WHAT FOLLOWS, in one line, and the names stored.
    assert report["what_follows"]["name"] == WITH_THE_DECAY and lines[11].startswith(f"{SAY}: WHAT FOLLOWS: EVERY LATER STAGE TRAINS WITH THE DECAY. ")
    assert report["branches"] == {"the_spread_of_the_recipe_as_it_is": ORDER_ALONE, "does_decay_lower_it": DECAY_LOWERS, "and_its_level": NAMED_NOTHING,
                                  "averaging": {"mean_constant": AT_LEAST_THE_MEAN, "mean_decay": AT_LEAST_THE_MEAN}, "what_follows": WITH_THE_DECAY}
    # BESIDE, naming nothing: goal problems solved and their spread; the shares; L5's own `published` of seed 0 as one more draw of `constant`, in NO standard deviation.
    beside = report["beside"]
    solved = lambda name: figures[name]["goal_problems_solved"]      # noqa: E731
    assert beside["goal_problems_solved"]["constant"] == {"by_order": {str(order): solved(f"constant_{order}") for order in ORDERS},
                                                          "standard_deviation": round(statistics.stdev([solved(f"constant_{order}") for order in ORDERS]), 3)}
    assert "BESIDE, naming nothing: goal problems solved in the 32 attempts, the share of distinct attempts, the share of attempts that timed out in Lean: `constant_1` 1, 0.9, 0.0; " in lines[12]
    assert beside["one_more_draw_of_constant"] == "0" and beside["the_published_of_l5"]["0"] == {"there": True, "run": "runs/seed0", "inconclusive": False, **figure(_totalling(26)),
                                                                                              "share_of_distinct_attempts": 0.91}
    assert "L5's own `published` of seed 0 (the same rows in that seed's order, the recipe as it is), from its stored first sampling: one more draw of `constant`, in no standard deviation: 203.12 per 1,000" in lines[13]
    assert "at each of its seeds (rows and order both drawn), on the same 32 attempts: seed 0: 203.12 per 1,000, 1 solved; seed 1: 109.38 per 1,000, 1 solved; seed 2: 156.25 per 1,000, 1 solved" in lines[14]
    assert len(lines) == 15 and report["headline"].startswith("L5n (what is one training worth; pretrained on published proofs) seed 0, from `pre_r64`, 8 fixed rows in 4 orders: the spread of the recipe as it is ")
    assert "What follows: EVERY LATER STAGE TRAINS WITH THE DECAY" in report["headline"] and len(report["what_it_cannot_say"]) == 5 and report["smoke"] is None
    assert report["models_made"]["constant_3"]["model"].startswith("`constant_3`: from `pre_r64`, one pass over 8 rows in the order of order seed 3: the recipe as it is")
    assert report["models_made"]["mean_decay"]["model"] == "`mean_decay`: trained by nothing: the mean of `decay_1`, `decay_2`, `decay_3`, `decay_4`"


def test_the_reports_reads_follow_the_figures_and_a_seed_of_l5_that_is_not_there_leaves_the_first_name_not_read():
    # The decay as wide as the recipe and level with it; the averaged adapter of the recipe as it is well below its models: neither.
    wide = _report({**TOTALS, "decay_1": 16, "decay_2": 20, "decay_3": 24, "decay_4": 28, "mean_constant": 2, "mean_decay": 22})
    assert wide["branches"] == {"the_spread_of_the_recipe_as_it_is": ORDER_ALONE, "does_decay_lower_it": NOT_SHOWN, "and_its_level": NAMED_NOTHING,
                                "averaging": {"mean_constant": LESS_THAN_THE_MEAN, "mean_decay": AT_LEAST_THE_MEAN}, "what_follows": THE_RECIPES_OWN}
    # WHAT FOLLOWS reads the averaged adapter OF THE RECIPE AS IT IS: `mean_decay` giving at least its mean does not make the second case, and `mean_constant` does.
    other = _report({**TOTALS, "decay_1": 16, "decay_2": 20, "decay_3": 24, "decay_4": 28, "mean_constant": 22, "mean_decay": 2})
    assert other["branches"]["averaging"] == {"mean_constant": AT_LEAST_THE_MEAN, "mean_decay": LESS_THAN_THE_MEAN} and other["branches"]["what_follows"] == AVERAGE_OF_FOUR
    # The decay narrower but BELOW the recipe at every order: it lowers the spread and COSTS; the second case, by the average.
    costs = _report({**TOTALS, "decay_1": 8, "decay_2": 10, "decay_3": 12, "decay_4": 14})
    assert (costs["branches"]["does_decay_lower_it"], costs["branches"]["and_its_level"], costs["branches"]["what_follows"]) == (DECAY_LOWERS, DECAY_COSTS, AVERAGE_OF_FOUR)
    assert costs["and_its_level"]["differences"] == [round(-(8 + 2 * index) / 128, 6) for index in range(4)] and costs["and_its_level"]["high"] < 0
    # L5's `published` far more spread over its seeds than the orders are: the rows drawn make most of it.
    drawn = _report(seeds={0: 60, 1: 2, 2: 100})
    assert drawn["branches"]["the_spread_of_the_recipe_as_it_is"] == ROWS_DRAWN and drawn["branches"]["what_follows"] == WITH_THE_DECAY
    # A SEED OF L5 THAT IS NOT THERE: the spread is given, the name is NOT READ, and nothing else moves.
    absent = _report(seeds={0: 26, 1: 14, 2: "ladder_l5_deep_pre_r64_seed2 does not hold ['ladder_l5_report.done.json']"})
    assert absent["branches"]["the_spread_of_the_recipe_as_it_is"] == NOT_READ and absent["the_spread_of_the_recipe_as_it_is"]["standard_deviation"] == _report()["the_spread_of_the_recipe_as_it_is"]["standard_deviation"]
    assert {key: value for key, value in absent["branches"].items() if key != "the_spread_of_the_recipe_as_it_is"} == {key: value for key, value in _report()["branches"].items() if key != "the_spread_of_the_recipe_as_it_is"}
    assert "NOT THERE (seed 2: ladder_l5_deep_pre_r64_seed2 does not hold ['ladder_l5_report.done.json'])" in absent["lines"][6] and "seed 2: NOT THERE (ladder_l5_deep_pre_r64_seed2 does not hold" in absent["lines"][14]
    assert absent["beside"]["the_published_of_l5"]["2"] == {"there": False, "why": "ladder_l5_deep_pre_r64_seed2 does not hold ['ladder_l5_report.done.json']"} and absent["inconclusive"] is False
    # The one more draw of `constant` is in NO standard deviation: moving it moves no read of the orders.
    assert _report(seeds={0: 99, 1: 14, 2: "x"})["the_spread_of_the_recipe_as_it_is"]["standard_deviation"] == absent["the_spread_of_the_recipe_as_it_is"]["standard_deviation"]
    # A seed of L5 whose own report read INCONCLUSIVE is read, and said.
    said = _report(change=lambda world: world["published"][1].update(inconclusive=True))
    assert "seed 1: 109.38 per 1,000, 1 solved (its own report reads INCONCLUSIVE)" in said["lines"][14] and said["branches"] == _report()["branches"]
    # A SMOKE RUN of two orders says so, and reads by the same rules at one degree of freedom.
    smoke = {"orders": 2, "orders_of_the_setting": [1, 2, 3, 4], "l5_read_though_inconclusive": True, "l5s_failed_checks": ["`shallow`: its_training_ran"], "the_order_trained_before": [2], "alike": []}
    small = _report(orders=(1, 2), seeds={0: 26, 1: "not there", 2: "not there"}, smoke=smoke)
    assert small["models"] == ["constant_1", "decay_1", "constant_2", "decay_2", "mean_constant", "mean_decay"] and small["and_its_level"]["critical_value"] == 12.706
    assert "A SMOKE RUN: L5's smoke run reads INCONCLUSIVE (`shallow`: its_training_ran) and was read all the same; 2 of the setting's 4 orders are trained; the orders that are the one L5 trained in ([2])" in small["lines"][1]
    assert small["branches"]["the_spread_of_the_recipe_as_it_is"] == NOT_READ and small["models_made"]["mean_constant"]["model"].endswith("the mean of `constant_1`, `constant_2`")


def test_any_failed_check_makes_the_run_inconclusive_and_nothing_else_is_said():
    def failing(change, named, said):
        report = _report(change=change)
        assert report["inconclusive"] is True and report["failed_checks"] == [named]
        assert report["branches"] == {"the_spread_of_the_recipe_as_it_is": INCONCLUSIVE, "does_decay_lower_it": INCONCLUSIVE, "and_its_level": INCONCLUSIVE,
                                      "averaging": {"mean_constant": INCONCLUSIVE, "mean_decay": INCONCLUSIVE}, "what_follows": NOT_READ}
        assert len(report["lines"]) == 6 and report["lines"][5] == f"{SAY}: INCONCLUSIVE. the run could not have been read ({named}): the spread of the recipe as it is is not read"
        assert any(said in line and line.endswith(": FAIL") for line in report["lines"][1:5]) and sum(line.endswith(": PASS") for line in report["lines"][1:5]) == 3
        # What was measured is kept apart, and not one figure is printed.
        assert list(report["figures"]) == ["what", "measured_and_not_read"] and set(report["figures"]["measured_and_not_read"]) == set(MODELS) and list(report["beside"]) == ["what", "measured_and_not_read"]
        assert not any("per 1,000" in line or "THE FIGURES" in line for line in report["lines"]) and report["headline"].endswith(f"INCONCLUSIVE. the run could not have been read ({named}): the spread of the recipe as it is is not read")
        return report

    # A trained adapter that IS the start adapter's; one that was not compared; one read back at another rank, or with another alpha, or not read back at all.
    failing(lambda world: world["trains"]["decay_2"]["against_the_start_adapter"]["decay_2"].update(elements_changed=0), f"`decay_2`: {THE_ADAPTER}", "`decay_2`: 0 of the 64 numbers of its adapter are not the start adapter's")
    failing(lambda world: world["trains"]["constant_1"].pop("against_the_start_adapter"), f"`constant_1`: {THE_ADAPTER}", "`constant_1`: its adapter was not compared with the start adapter's")
    for saved in ({**SAVED, "rank": 16, "ranks": [16]}, {**SAVED, "alpha": 32}, {**SAVED, "ranks": [64, 16]}):
        failing(lambda world, saved=saved: world["trains"]["constant_3"].update(adapter_saved=saved), f"`constant_3`: {THE_ADAPTER}", f"read back at rank {saved['rank']} and alpha {saved['alpha']}")
    failing(lambda world: world["trains"]["constant_3"].pop("adapter_saved"), f"`constant_3`: {THE_ADAPTER}", "`constant_3`: 60 of the 64 numbers of its adapter are not the start adapter's; not read back")
    # A row that is none of the fixed rows; a fixed row not trained on.
    stranger = {"id": "p29#pretraining", "problem_id": "p29", "origin": "pretraining", "row_of_the_file": 29}
    failing(lambda world: world["trained"]["decay_4"].__setitem__(0, stranger), f"`decay_4`: {THE_ROWS}", "`decay_4` 8 rows, 1 none of them, 0 twice, 1 not trained on")
    failing(lambda world: world["trained"]["constant_2"].pop(), f"`constant_2`: {THE_ROWS}", "`constant_2` 7 rows, 0 none of them, 0 twice, 1 not trained on")
    # An averaged adapter that is not the mean; one that is one of its models; one at another rank; one that was not served (its set sampled from another adapter); none averaged.
    failing(lambda world: world["averaged"]["mean_decay"]["read_back"].update(elements_not_the_mean=3, is_the_mean=False, passes=False), f"`mean_decay`: {THE_MEAN}", "`mean_decay` (the mean of `decay_1`, `decay_2`, `decay_3`, `decay_4`): 3 of its 64 numbers are not the mean")
    failing(lambda world: world["averaged"]["mean_constant"]["read_back"].update(elements_not_those_of_each_adapter=[40, 0, 40, 40], is_none_of_them=False, passes=False), f"`mean_constant`: {THE_MEAN}",
            "the numbers that are not each model's: 40, 0, 40, 40")
    failing(lambda world: world["averaged"]["mean_constant"].update(adapter_saved={**SAVED, "rank": 16, "ranks": [16]}), f"`mean_constant`: {THE_MEAN}", "NOT at the start adapter's rank")
    # ... A SMOKE RUN ALONE keeps an averaged adapter that is one of its models (its trainings are one training), and SAYS so; one that is NOT THE MEAN fails there too.
    smoke = {"orders": 4, "orders_of_the_setting": [1, 2, 3, 4], "l5_read_though_inconclusive": False, "l5s_failed_checks": [], "the_order_trained_before": [], "alike": []}
    one_of_them = lambda world: world["averaged"]["mean_constant"]["read_back"].update(elements_not_those_of_each_adapter=[0, 0, 0, 0], is_none_of_them=False, passes=False)      # noqa: E731
    kept = _report(change=one_of_them, smoke=smoke)
    assert kept["inconclusive"] is False and kept["can_this_run_be_read"]["mean_constant"][THE_MEAN]["one_of_its_models_in_a_smoke_run"] is True
    assert "the numbers that are not each model's: 0, 0, 0, 0 (IT IS ONE OF ITS MODELS: a smoke run's trainings of one optimizer step on the same rows are one training; kept all the same)" in kept["lines"][4]
    assert _report(change=one_of_them)["inconclusive"] is True and _report()["can_this_run_be_read"]["mean_constant"][THE_MEAN]["one_of_its_models_in_a_smoke_run"] is False
    not_the_mean = _report(change=lambda world: world["averaged"]["mean_decay"]["read_back"].update(elements_not_the_mean=3, is_the_mean=False, is_none_of_them=False, passes=False), smoke=smoke)
    assert not_the_mean["failed_checks"] == [f"`mean_decay`: {THE_MEAN}"]
    failing(lambda world: world["measured"]["mean_decay"].update(adapter="adapters/decay_1"), f"`mean_decay`: {THE_MEAN}", "it was NOT served")
    failing(lambda world: world["averaged"]["mean_decay"].update(read_back=None, adapter_saved=None), f"`mean_decay`: {THE_MEAN}", "`mean_decay`: no adapter was averaged; its set was sampled")
    # LEAN DID NOT ANSWER a set (more than 0.02 of its attempts without an answer): INCONCLUSIVE, and the report is NOT TO BE READ (the step fails, and that set is sampled again).
    silent = _report(change=lambda world: world["rows"]["constant_4"].__setitem__(0, {**world["rows"]["constant_4"][0], "attempts_without_an_answer": 3}))
    assert silent["inconclusive"] is True and silent["failed_checks"] == [f"`constant_4`: {LEAN_ANSWERED}"] and silent["ok"] is False and silent["not_to_be_read"] == ["constant_4"]
    assert silent["lines"][-1].startswith(f"{SAY}: NOT TO BE READ: too many attempts without a verdict from Lean in the sets of `constant_4`") and "`constant_4` 0.02344; " in silent["lines"][4]
    at_the_edge = _report(change=lambda world: world["rows"]["constant_4"].__setitem__(0, {**world["rows"]["constant_4"][0], "attempts_without_an_answer": 2}))      # 2 of 128: not above 0.02
    assert at_the_edge["inconclusive"] is False and at_the_edge["ok"] is True
    # The checks alone, as a function: every model has its own, and each kind is one entry.
    world = {}
    _report(change=world.update)
    checks = the_checks(*(world[key] for key in ("prepare", "trains", "trained", "averaged", "measured", "rows", "fixed")))
    assert all(entry["passes"] for own in checks.values() for entry in own.values()) and checks["mean_constant"][THE_MEAN]["served"] is True and checks["constant_1"][THE_ADAPTER]["at_its_rank"] is True


def test_the_report_refuses_rows_that_are_not_this_runs_models_each_on_the_one_goal_set():
    with pytest.raises(ValueError, match="the report is given the rows of .*: it reads this run's models"):
        _report(change=lambda world: world["rows"].pop("mean_decay"))
    with pytest.raises(ValueError, match="the report is given the rows of .*: it reads this run's models"):
        _report(change=lambda world: world["trains"].pop("decay_1"))
    with pytest.raises(ValueError, match="the models were not measured on the same problems"):
        _report(change=lambda world: world["rows"].__setitem__("decay_1", _totalling(20, problems=5)))
