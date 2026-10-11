"""L5d's rules, its read over the seeds and the command that makes that read again, on hand-made rows: the recipe held to
L5n's report; the seeds; the checks of a seed's sets; the figure on the three parts of G and the counts; the interval over
SEEDS; the primary's four outcomes and their edges; the differences beside it; where it sits; the spread; L5's own seeds
beside the fresh ones; what follows; a seed missing, a seed whose check failed, seeds of two recipes. Spec:
docs/spec/ladder-loop.spec.md, "L5d: the three kinds again, over fresh draws, measured on the goal set's first
sampling". The stage end to end in the scripted world is `test_ladder_l5_draws_stage.py`."""

import json
import math
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l5 import DEEP_ROWS, PRETRAINING, SEED, SHALLOW_ROWS  # noqa: E402 - hand-made deep rows, ordinary solves and a pretraining file's rows

from rlvr_lean.domain.ladder_round import l5_draws  # noqa: E402
from rlvr_lean.domain.ladder_round.assembly import training_order  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ  # noqa: E402
from rlvr_lean.domain.ladder_round.l5 import DEEP, MODELS, NOT_SHOWN, PUBLISHED, REACH_TOO, RELIABILITY, SHALLOW, THEY_COST, the_sets  # noqa: E402
from rlvr_lean.domain.ladder_round.l5_draws import (  # noqa: E402
    AGAIN,
    ALL,
    ASKED_BY,
    BEAT,
    BESIDE,
    CONSTANT,
    DECAY,
    L6_RUNS,
    MEAN_CONSTANT,
    NAMED_NOTHING,
    NOT_CONFIRMED,
    PARTS,
    RECIPES,
    SOLVED,
    STOP,
    T_CRITICAL,
    THE_EFFECT_FOUND,
    THE_PRIMARY,
    UNDETECTABLE,
    differences,
    fixed_seeds,
    four_orders,
    half_widths,
    interval,
    its_trainings,
    named,
    no_problem_of,
    on_the_parts,
    over_the_seeds,
    pooled,
    primary,
    refuse_problems_of,
    the_counts,
    the_parts,
    the_recipe,
    the_seed,
    the_sets_are_the_rules,
    the_start_is_the_settings,
    trained_models,
    trained_on_its_set,
    what_follows,
    where_it_sits,
)
from rlvr_lean.domain.ladder_round.l5_noise import AVERAGE_OF_FOUR, ORDERS, THE_RECIPES_OWN, WITH_THE_DECAY, numbered  # noqa: E402
from rlvr_lean.reporting.ladder_l4 import SAY  # noqa: E402
from rlvr_lean.reporting.ladder_l5_draws import lines_over_the_seeds, read_over_the_seeds  # noqa: E402
from rlvr_lean.tools import ladder_l5_draws as tool  # noqa: E402

FOUND = {0: 0.0038, 1: 0.0242, 2: 0.0215}          # the primary's difference at L5's three seeds, per attempt: the section's +3.8, +24.2, +21.5 per 1,000
MEASURED = {"sampling": "reach", "episodes": 32, "sampling_seed": 1001, "problems": 4}
THERE = {"there": True, "run": "ladder_l5_noise_pre_r64_seed0", "ok": True, "inconclusive": False}


# ------------------------------------------------------------------------------------------------ the names
def test_the_names_and_the_numbers_are_the_sections():
    assert BEAT == "THE MODEL'S OWN PROOFS BESIDE PUBLISHED ONES BEAT PUBLISHED ONES ALONE ON THE GOAL SET" and UNDETECTABLE == "UNDETECTABLE AT THIS NUMBER OF SEEDS"
    assert (THEY_COST, NOT_SHOWN, RELIABILITY, REACH_TOO) == ("THEY COST", "NOT SHOWN", "RELIABILITY", "REACH TOO")
    assert l5_draws.PRIMARY == (BEAT, THEY_COST, UNDETECTABLE, NOT_SHOWN, INCONCLUSIVE, NOT_READ) and l5_draws.SITS == (RELIABILITY, REACH_TOO, NAMED_NOTHING, NOT_READ)
    assert l5_draws.FOLLOWS == (L6_RUNS, NOT_CONFIRMED, STOP, NOT_READ) and RECIPES == ("constant", "decay", "mean_constant") == (CONSTANT, DECAY, MEAN_CONSTANT)
    assert ASKED_BY == {WITH_THE_DECAY: "decay", AVERAGE_OF_FOUR: "mean_constant", THE_RECIPES_OWN: "constant"} and l5_draws.OF_L5 == "constant"
    assert (THE_EFFECT_FOUND, T_CRITICAL[2], T_CRITICAL[5], T_CRITICAL[8]) == (0.0165, 4.303, 2.571, 2.306) and sorted(T_CRITICAL) == list(range(1, 9))     # t 4.303 at three seeds, 2.571 at six
    assert PARTS == (ALL, SOLVED, AGAIN) and list(BESIDE.values()) == [(DEEP, PUBLISHED), (SHALLOW, PUBLISHED), (DEEP, SHALLOW)]
    assert trained_models(CONSTANT) == trained_models(DECAY) == MODELS == ("published", "shallow", "deep")
    assert trained_models(MEAN_CONSTANT) == tuple(f"{model}_{order}" for model in MODELS for order in (1, 2, 3, 4))
    assert its_trainings(DECAY, DEEP) == ("deep",) and its_trainings(MEAN_CONSTANT, DEEP) == ("deep_1", "deep_2", "deep_3", "deep_4")


# ------------------------------------------------------------------------------------------------ the recipe
@pytest.mark.parametrize("name, asked", [(WITH_THE_DECAY, DECAY), (AVERAGE_OF_FOUR, MEAN_CONSTANT), (THE_RECIPES_OWN, CONSTANT)])
def test_the_recipe_is_the_one_l5ns_report_names_and_any_other_is_refused_naming_both(name, asked):
    found = the_recipe(asked, {**THERE, "what_follows": name})
    assert found == {"recipe": asked, "asked_for_by_l5n": asked, "found": f"L5n's report names {name}", "l5n_named_it": True, "said_beside_every_read": None}
    for other in set(RECIPES) - {asked}:
        with pytest.raises(ValueError, match=f"the setting names the recipe `{other}` and L5n's report names {name}.*which asks for `{asked}`"):
            the_recipe(other, {**THERE, "what_follows": name})


@pytest.mark.parametrize("of_l5n, said", [
    ({"there": False, "run": "r", "why": "r does not hold ['report_ladder_l5_noise.json']"}, "L5n's run is not there or not finished (r does not hold ['report_ladder_l5_noise.json'])"),
    ({**THERE, "inconclusive": True, "what_follows": NOT_READ}, f"L5n's report reads {INCONCLUSIVE}"),
    ({**THERE, "inconclusive": True, "what_follows": WITH_THE_DECAY}, f"L5n's report reads {INCONCLUSIVE}"),     # an INCONCLUSIVE report names nothing, whatever stands in it
    ({**THERE, "ok": False, "what_follows": WITH_THE_DECAY}, "L5n's report is not to be read"),
    ({**THERE, "what_follows": NOT_READ}, f"L5n's report names no case for what follows ({NOT_READ})"),
    ({**THERE, "what_follows": None}, "L5n's report names no case for what follows (None)")])
def test_with_l5n_not_there_unfinished_not_read_or_inconclusive_the_recipe_is_the_recipe_as_it_is_and_that_is_said(of_l5n, said):
    found = the_recipe(CONSTANT, of_l5n)
    assert (found["recipe"], found["asked_for_by_l5n"], found["found"], found["l5n_named_it"]) == (CONSTANT, CONSTANT, said, False)
    assert found["said_beside_every_read"] == f"{said}: the recipe as it is, one training a model, is trained with, and L5n did not name it"
    for other in (DECAY, MEAN_CONSTANT):
        with pytest.raises(ValueError, match=f"the setting names the recipe `{other}` and .*which asks for `constant`.*Change the setting, or finish L5n's run"):
            the_recipe(other, of_l5n)
    with pytest.raises(ValueError, match="the recipe `cosine` is none of"):
        the_recipe("cosine", of_l5n)


def test_the_four_orders_are_l5ns_and_an_order_seed_that_is_the_tasks_seed_is_trained_all_the_same_and_named():
    rows = numbered(training_order(the_sets(DEEP_ROWS, SHALLOW_ROWS, PRETRAINING, 3)[DEEP], 3))       # a set in the order of the task's seed, 3
    by_order, coincide = four_orders(rows)
    assert list(by_order) == list(ORDERS) and all([row["id"] for row in by_order[order]] == [row["id"] for row in training_order(rows, order)] for order in ORDERS)
    assert coincide == {"the_order_of_the_tasks_seed": [3], "alike": []} and [row["id"] for row in by_order[3]] == [row["id"] for row in rows]
    assert len({tuple(row["id"] for row in own) for own in by_order.values()}) == 4
    with pytest.raises(ValueError, match=r"the pairs \[\[1, 2\], \[1, 3\], \[1, 4\], \[2, 3\], \[2, 4\], \[3, 4\]\] give one order of these 1 rows"):
        four_orders(rows[:1])                                                               # one row has one order: the four trainings would be one
    assert four_orders(rows[:1], smoke=True)[1]["alike"] == [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]      # a SMOKE run alone


# ------------------------------------------------------------------------------------------------- the seeds
def test_a_tasks_seed_is_a_fresh_seed_of_the_setting_and_the_further_ones_need_explicit_approval():
    seeds, more, of_l5 = (3, 4, 5), (6, 7, 8), (0, 1, 2)
    assert fixed_seeds(seeds, more, False) == (3, 4, 5) and fixed_seeds(seeds, more, True) == (3, 4, 5, 6, 7, 8)
    for seed in seeds:
        the_seed(seed, seeds, more, False, of_l5)
    for seed in (*seeds, *more):
        the_seed(seed, seeds, more, True, of_l5)
    for seed in of_l5:
        with pytest.raises(ValueError, match=f"seed {seed} is one of L5's own .*those seeds found the sign"):
            the_seed(seed, seeds, more, True, of_l5)
    for seed in more:
        with pytest.raises(ValueError, match=f"seed {seed} is one of the further seeds .*the approval is not recorded .more_seeds_approved is false."):
            the_seed(seed, seeds, more, False, of_l5)
    with pytest.raises(ValueError, match=r"seed 9 is none of the fresh seeds \[3, 4, 5\] .nor of \[6, 7, 8\], which need explicit approval."):
        the_seed(9, seeds, more, False, of_l5)
    with pytest.raises(ValueError, match=r"seed 9 is none of the fresh seeds \[3, 4, 5\] and \[6, 7, 8\]: the number of seeds is fixed before the read"):
        the_seed(9, seeds, more, True, of_l5)


# ------------------------------------------------------------------------------------------------ the checks
def test_the_sets_are_held_to_l5s_rule_made_again_with_no_text_and_each_order_seeds_order():
    sets = {model: numbered(own) for model, own in the_sets(DEEP_ROWS, SHALLOW_ROWS, PRETRAINING, SEED).items()}
    held = the_sets_are_the_rules(sets, PRETRAINING, SEED)
    assert (held["rows_of_each_kind"], held["refused"], held["published_rows_with_text"], held["sets_not_in_their_order_seeds_order"], held["passes"]) == (4, None, 0, [], True)
    # ANOTHER SEED's draw is not this seed's; a set in another order is not the rule's; a published row with its text is refused.
    other = the_sets_are_the_rules(sets, PRETRAINING, SEED + 1)
    assert other["passes"] is False and other["rows_of_each_kind"] is None and "Refused" in other["refused"]
    assert the_sets_are_the_rules({**sets, PUBLISHED: list(reversed(sets[PUBLISHED]))}, PRETRAINING, SEED)["passes"] is False
    with_text = {**sets, PUBLISHED: [{**sets[PUBLISHED][0], "proof": "by simp"}, *sets[PUBLISHED][1:]]}
    assert (the_sets_are_the_rules(with_text, PRETRAINING, SEED)["published_rows_with_text"], the_sets_are_the_rules(with_text, PRETRAINING, SEED)["passes"]) == (1, False)
    assert the_sets_are_the_rules({model: sets[model] for model in (PUBLISHED, DEEP)}, PRETRAINING, SEED)["passes"] is False          # a set that is not there
    # THE AVERAGED RECIPE: each of a kind's four sets is that kind's rows in its order seed's order.
    in_orders = {f"{model}_{order}": numbered(training_order(sets[model], order)) for model in MODELS for order in ORDERS}
    assert the_sets_are_the_rules(sets, PRETRAINING, SEED, in_orders)["passes"] is True
    for broken in ({**in_orders, "deep_2": in_orders["deep_3"]}, {**in_orders, "deep_2": in_orders["deep_2"][1:]}, {**in_orders, "deep_2": in_orders["shallow_2"]},
                   {**in_orders, "wide_2": in_orders["deep_2"]}):
        held = the_sets_are_the_rules(sets, PRETRAINING, SEED, broken)
        assert held["passes"] is False and held["rows_of_each_kind"] == 4 and len(held["sets_not_in_their_order_seeds_order"]) == 1


def test_no_row_is_of_a_problem_of_g_or_of_the_held_aside_set_and_a_model_was_trained_on_its_set_alone():
    rows = [{"id": f"{key}#x", "problem_id": key} for key in ("a", "b", "c")]
    assert no_problem_of(rows, {"g1"}, {"h1"})["passes"] is True and no_problem_of([], {"g1"}, {"h1"})["passes"] is False
    of_g, of_the_set = no_problem_of(rows, {"a"}, {"h1"}), no_problem_of(rows, {"g1"}, {"b", "c"})
    assert (of_g["problems_of_the_goal_set"], of_g["problems_of_the_held_aside_set"], of_g["first"], of_g["passes"]) == (1, 0, ["a"], False)
    assert (of_the_set["problems_of_the_goal_set"], of_the_set["problems_of_the_held_aside_set"], of_the_set["first"], of_the_set["passes"]) == (0, 2, ["b", "c"], False)
    assert refuse_problems_of(rows, {"g1"}, {"h1"}, "the training set of `deep`")["passes"] is True
    with pytest.raises(ValueError, match="1 problems of the training set of `deep` are goal problems and 0 are held-aside problems .first: a.: refused, nothing was written"):
        refuse_problems_of(rows, {"a"}, {"h1"}, "the training set of `deep`")
    with pytest.raises(ValueError, match="0 problems of the training set of `deep` are goal problems and 1 are held-aside problems .first: c."):
        refuse_problems_of(rows, set(), {"c"}, "the training set of `deep`")
    ids = [row["id"] for row in rows]
    assert trained_on_its_set(ids, rows)["passes"] is True and trained_on_its_set(list(reversed(ids)), rows)["passes"] is True       # the order is the first check's
    for trained, key in ((ids[:2], "rows_not_trained_on"), ([*ids, ids[0]], "rows_twice"), ([*ids, "z#x"], "rows_that_are_none_of_them")):
        found = trained_on_its_set(trained, rows)
        assert found["passes"] is False and found[key] == 1 and sum(found[name] for name in ("rows_not_trained_on", "rows_twice", "rows_that_are_none_of_them")) == 1
    assert trained_on_its_set([], rows)["passes"] is False and trained_on_its_set([], [])["passes"] is False     # a training of nothing on a set of nothing passes nothing


def test_the_start_models_stored_first_sampling_is_held_to_the_settings_problems_seed_and_attempts():
    rows = [{"problem_id": key, "episodes": 32, "resolved": 0} for key in ("g1", "g2")]
    recorded = {"name": "reach", "episodes": 32, "sampling_seed": 1001}
    assert the_start_is_the_settings(rows, ["g2", "g1"], recorded, "reach", 32)["passes"] is True
    for changed in ((rows[:1], ["g1", "g2"], recorded, "reach", 32), ([*rows, rows[0]], ["g1", "g2"], recorded, "reach", 32), (rows, ["g1", "g3"], recorded, "reach", 32),
                    ([rows[0], {**rows[1], "episodes": 61}], ["g1", "g2"], recorded, "reach", 32), (rows, ["g1", "g2"], recorded, "more", 32), (rows, ["g1", "g2"], recorded, "reach", 61),
                    (rows, ["g1", "g2"], {**recorded, "sampling_seed": None}, "reach", 32), ([], [], recorded, "reach", 32)):
        assert the_start_is_the_settings(*changed)["passes"] is False
    assert the_start_is_the_settings([rows[0], {**rows[1], "episodes": 61}], ["g1", "g2"], recorded, "reach", 32)["attempts_a_problem_that_are_not_the_settings"] == [61]


# ----------------------------------------------------------------------------------------------- the figures
def _on_g(successes):
    """One model's per-problem rows on four goal problems, 32 attempts each: g1 and g2 the start model solves, g3 and g4 it does not (G')."""
    return [{"problem_id": key, "episodes": 32, "resolved": count, "sides": 1, "attempts_timed_out": 0} for key, count in zip(("g1", "g2", "g3", "g4"), successes)]


START = _on_g((16, 1, 0, 0))                        # the start model's own first sampling: it solves g1 and g2
G_PARTS = {ALL: ["g1", "g2", "g3", "g4"], SOLVED: ["g1", "g2"], AGAIN: ["g3", "g4"]}


def test_the_parts_of_g_are_the_start_models_own_first_sampling_and_are_held_to_the_goal_set_again():
    assert the_parts(START, ["g4", "g3"]) == G_PARTS
    with pytest.raises(ValueError, match="leaves 2 goal problems unsolved and G' holds 1: the two are not one sampling's"):
        the_parts(START, ["g3"])
    figures = on_the_parts(_on_g((16, 8, 4, 0)), G_PARTS)
    assert [(figures[part]["problems"], figures[part]["successes"], figures[part]["attempts"]) for part in PARTS] == [(4, 28, 128), (2, 24, 64), (2, 4, 64)]
    assert (figures[ALL]["per_attempt"], figures[ALL]["per_1000"], figures[ALL]["goal_problems_solved"]) == (0.21875, 218.75, 3)
    assert on_the_parts(_on_g((1, 1, 1, 1)), {ALL: ["g1"], SOLVED: [], AGAIN: ["g1"]})[SOLVED]["attempts"] == 0


def test_the_counts_are_the_goal_problems_solved_gained_and_lost_against_published_and_of_the_long_ones():
    rows = {PUBLISHED: _on_g((9, 1, 0, 0)), SHALLOW: _on_g((9, 0, 2, 0)), DEEP: _on_g((9, 3, 1, 1))}
    counts = the_counts(rows, ["g2", "g3"])
    assert counts[PUBLISHED] == {"solved": 2, "gained": 0, "lost": 0, "long": {"problems": 2, "solved": 1, "gained": 0, "lost": 0}}
    assert counts[SHALLOW] == {"solved": 2, "gained": 1, "lost": 1, "long": {"problems": 2, "solved": 1, "gained": 1, "lost": 1}}
    assert counts[DEEP] == {"solved": 4, "gained": 2, "lost": 0, "long": {"problems": 2, "solved": 2, "gained": 1, "lost": 0}}
    assert the_counts(rows, [])[DEEP]["long"] == {"problems": 0, "solved": 0, "gained": 0, "lost": 0}


def _figures(published, shallow, deep):
    """A seed's figures from each model's successes on the four goal problems."""
    return {model: on_the_parts(_on_g(successes), G_PARTS) for model, successes in ((PUBLISHED, published), (SHALLOW, shallow), (DEEP, deep))}


def test_a_seeds_differences_are_from_whole_numbers_of_successes_and_attempts():
    own = differences(_figures((8, 8, 0, 0), (12, 8, 4, 0), (16, 8, 8, 0)))
    assert own == {THE_PRIMARY: (24 / 128 + 32 / 128) / 2 - 16 / 128, "deep_minus_published": 16 / 128, "shallow_minus_published": 8 / 128, "deep_minus_shallow": 8 / 128}
    on_again = differences(_figures((8, 8, 0, 0), (12, 8, 4, 0), (16, 8, 8, 0)), AGAIN)
    assert on_again[THE_PRIMARY] == (4 / 64 + 8 / 64) / 2 and differences(_figures((8, 8, 0, 0), (12, 8, 4, 0), (16, 8, 8, 0)), SOLVED)[THE_PRIMARY] == (20 / 64 + 24 / 64) / 2 - 16 / 64
    empty = {model: on_the_parts(_on_g((1, 1, 1, 1)), {ALL: ["g1"], SOLVED: [], AGAIN: ["g1"]}) for model in MODELS}
    assert differences(empty, SOLVED) == {key: None for key in (THE_PRIMARY, *BESIDE)} and differences(empty)[THE_PRIMARY] == 0.0


# ------------------------------------------------------------------------------------------------- the reads
def test_the_interval_is_over_seeds_the_mean_plus_and_minus_students_t_times_the_standard_deviation_over_the_root_of_the_seeds(monkeypatch):
    own = interval(FOUND)
    deviation = statistics.stdev(FOUND.values())
    assert (own["seeds"], own["degrees_of_freedom"], own["critical_value"]) == (3, 2, 4.303) and own["at_each_seed"] == {"0": 0.0038, "1": 0.0242, "2": 0.0215}
    assert own["mean"] == 0.0165 and own["standard_deviation"] == round(deviation, 6) == 0.011081                # the section's +16.5 per 1,000 and 11.1
    assert own["half_width"] == round(4.303 * deviation / math.sqrt(3), 6) == 0.027529                           # ... and its half-width at three seeds, 27.5
    assert (own["low"], own["high"]) == (round(0.0165 - own["exact"]["half_width"], 6), round(0.0165 + own["exact"]["half_width"], 6))
    six = interval({seed: value for seed, value in zip(range(3, 9), (0.01, 0.02, 0.03, 0.01, 0.02, 0.03))})
    assert (six["seeds"], six["critical_value"]) == (6, 2.571) and six["half_width"] == round(2.571 * statistics.stdev((0.01, 0.02, 0.03) * 2) / math.sqrt(6), 6)
    for count in (1, 10):
        with pytest.raises(ValueError, match=f"the t interval over {count} seeds is not given here"):
            interval({seed: 0.01 * seed for seed in range(count)})
    # WHAT A NUMBER OF SEEDS WOULD RESOLVE with the standard deviation found (11.1): the section's 27.5 at three and 11.6 at six.
    resolved = half_widths(0.0111)
    assert (round(1000 * resolved["by_seeds"]["3"], 1), round(1000 * resolved["by_seeds"]["6"], 1)) == (27.6, 11.7) and list(resolved["by_seeds"]) == [str(count) for count in range(2, 10)]
    assert resolved["seeds_at_which_it_is_at_most_the_effect_found"] == 5 and half_widths(1.0)["seeds_at_which_it_is_at_most_the_effect_found"] is None
    # A HALF-WIDTH THAT IS EXACTLY THE EFFECT FOUND is at most it (here the effect found is made the half-width at four seeds, to the last bit).
    monkeypatch.setattr(l5_draws, "THE_EFFECT_FOUND", 3.182 * 0.01 / math.sqrt(4))
    assert half_widths(0.01)["seeds_at_which_it_is_at_most_the_effect_found"] == 4


def test_the_primary_has_four_outcomes_an_interval_that_ends_at_zero_holds_zero_and_a_half_width_at_the_threshold_is_not_above_it():
    assert named(0.001, 0.05, 0.0245) == BEAT and named(-0.05, -0.001, 0.0245) == THEY_COST
    assert named(0.0, 0.05, 0.025) == UNDETECTABLE and named(-0.05, 0.0, 0.025) == UNDETECTABLE                 # AN INTERVAL THAT ENDS AT ZERO HOLDS ZERO: neither BEAT nor COST
    assert named(0.0, 0.02, 0.01) == NOT_SHOWN and named(-0.02, 0.0, 0.01) == NOT_SHOWN
    assert named(-0.0165, 0.0165, THE_EFFECT_FOUND) == NOT_SHOWN                                               # A HALF-WIDTH EXACTLY AT THE EFFECT FOUND is not above it
    assert named(-0.0165, 0.0166, 0.01655) == UNDETECTABLE and named(-0.001, 0.001, 0.001) == NOT_SHOWN
    beat = primary({3: 0.020, 4: 0.021, 5: 0.022})
    assert (beat["name"], beat["seeds"], beat["seeds_at_which_it_is_positive"], beat["the_effect_found"]) == (BEAT, 3, 3, 0.0165) and beat["low"] > 0 and "exact" not in beat
    cost = primary({3: -0.020, 4: -0.021, 5: -0.022})
    assert (cost["name"], cost["seeds_at_which_it_is_positive"]) == (THEY_COST, 0) and cost["high"] < 0
    # L5's OWN THREE SEEDS, read by this rule: the interval holds zero and is wider than the effect found (which is why they are not a result).
    theirs = primary(FOUND)
    assert theirs["name"] == UNDETECTABLE and theirs["low"] < 0 < theirs["high"] and theirs["half_width"] > THE_EFFECT_FOUND
    assert "it holds zero, and its half-width (27.53) is above the 16.5 per 1,000 found on L5's seeds" in theirs["reason"] and "mean" not in theirs["reason"].split(":", 1)[1]
    shown = primary({3: -0.001, 4: 0.0, 5: 0.002})
    assert shown["name"] == NOT_SHOWN and shown["half_width"] < THE_EFFECT_FOUND and shown["low"] < 0 < shown["high"]
    assert shown["seeds_at_which_it_is_positive"] == 1                                                         # a seed at which the difference is zero is not positive
    assert all(primary(at_each)["name"] in l5_draws.PRIMARY for at_each in (FOUND, {3: 0.0, 4: 0.0, 5: 0.0}))
    # THE OUTCOME IS NAMED FROM THE INTERVAL AS COMPUTED, not as it is printed (six places): one that begins three ten-millionths above zero is above zero, and prints 0.0; a
    # half-width four hundred-millionths above the effect found is above it, and prints as the effect found.
    half = 4.303 * 0.001 / math.sqrt(3)
    barely = primary({3: half + 3e-7 - 0.001, 4: half + 3e-7, 5: half + 3e-7 + 0.001})
    assert (barely["name"], barely["low"]) == (BEAT, 0.0)
    wide = (THE_EFFECT_FOUND + 4e-8) * math.sqrt(3) / 4.303
    just_above = primary({3: -wide, 4: 0.0, 5: wide})
    assert (just_above["name"], just_above["half_width"]) == (UNDETECTABLE, THE_EFFECT_FOUND)


def test_where_it_sits_is_read_only_when_the_primary_names_the_first_outcome_and_names_two_cases():
    above, zero, below = {3: 0.020, 4: 0.021, 5: 0.022}, {3: -0.001, 4: 0.0, 5: 0.001}, {3: -0.020, 4: -0.021, 5: -0.022}
    for name in (THEY_COST, UNDETECTABLE, NOT_SHOWN, INCONCLUSIVE, NOT_READ):
        assert where_it_sits(name, above, above) == {"name": NOT_READ, "reason": f"the primary reads {name}: where it sits is read only when the model's own proofs beat"}
    assert where_it_sits(BEAT, above, zero)["name"] == RELIABILITY and where_it_sits(BEAT, above, above)["name"] == REACH_TOO
    assert where_it_sits(BEAT, zero, above)["name"] == REACH_TOO and where_it_sits(BEAT, below, above)["name"] == REACH_TOO          # (b) above zero is REACH TOO, whatever (a) is
    for solved, again in ((zero, zero), (above, below), (zero, below), (below, zero)):                          # NEITHER OF THE SECTION'S TWO CASES: given, and named nothing
        assert where_it_sits(BEAT, solved, again)["name"] == NAMED_NOTHING
    sits = where_it_sits(BEAT, above, zero)
    assert sits["on_the_problems_the_start_solves"]["low"] > 0 and sits["on_the_goal_set_again"]["low"] < 0 < sits["on_the_goal_set_again"]["high"]
    assert where_it_sits(BEAT, above, {3: 0.0, 4: 0.0, 5: 0.0})["name"] == RELIABILITY                         # (b) that BEGINS at zero holds zero: it is not above it
    assert where_it_sits(BEAT, {3: None, 4: None, 5: None}, above)["name"] == NOT_READ and where_it_sits(BEAT, above, {3: 0.1, 4: None, 5: 0.1})["name"] == NOT_READ


def test_what_follows_is_the_sections_three_cases_and_nothing_while_the_primary_is_not_read():
    assert what_follows(primary({3: 0.020, 4: 0.021, 5: 0.022}))["name"] == L6_RUNS and what_follows(primary({3: -0.020, 4: -0.021, 5: -0.022}))["name"] == STOP
    for at_each in (FOUND, {3: -0.001, 4: 0.0, 5: 0.002}):
        follows = what_follows(primary(at_each))
        assert follows["name"] == NOT_CONFIRMED and "L6 does not run on it" in follows["reason"] and list(follows["seeds_would_resolve"]["by_seeds"]) == [str(n) for n in range(2, 10)]
    assert "with as many seeds an arm as the standard deviation measured here (1.00 per 1,000 over 3 seeds) asks" in what_follows(primary({3: 0.020, 4: 0.021, 5: 0.022}))["reason"]
    for name in (INCONCLUSIVE, NOT_READ):
        assert what_follows({"name": name}) == {"name": NOT_READ, "reason": f"the primary reads {name}: nothing is named for what follows"}


# ------------------------------------------------------------------------------------- the read over the seeds
def _seed(seed, published=(8, 8, 0, 0), shallow=(12, 8, 4, 0), deep=(16, 8, 8, 0), **changed):
    """What a seed's report says of its own seed: its figures from each model's successes on the four goal problems."""
    rows = {PUBLISHED: _on_g(published), SHALLOW: _on_g(shallow), DEEP: _on_g(deep)}
    return {"seed": seed, "recipe": CONSTANT, "ok": True, "inconclusive": False, "failed_checks": [], "the_measurement": dict(MEASURED),
            "figures": {model: on_the_parts(own, G_PARTS) for model, own in rows.items()}, "counts": the_counts(rows, ["g2", "g3"]), **changed}


THREE = {3: _seed(3), 4: _seed(4, shallow=(12, 9, 4, 0)), 5: _seed(5, deep=(16, 8, 8, 1))}       # the primary's difference: 12/128, 12.5/128, 12.5/128


def test_the_read_is_over_the_fresh_seeds_alone_each_figure_from_its_whole_numbers():
    read = over_the_seeds(THREE, (3, 4, 5))
    at_each = {3: 12 / 128, 4: 12.5 / 128, 5: 12.5 / 128}
    assert (read["read"], read["fixed_seeds"], read["seeds_there"], read["seeds_missing"], read["seeds_whose_checks_failed"], read["recipe"]) == (True, [3, 4, 5], [3, 4, 5], [], {}, CONSTANT)
    assert read["primary"] == primary(at_each) and read["primary"]["name"] == BEAT and read["primary"]["at_each_seed"] == {"3": 0.09375, "4": 0.097656, "5": 0.097656}
    assert list(read["beside"]) == list(BESIDE) and read["beside"]["deep_minus_published"]["at_each_seed"] == {"3": 0.125, "4": 0.125, "5": 0.132812}
    assert read["beside"]["deep_minus_shallow"]["mean"] == round((8 + 7 + 9) / 3 / 128, 6) and "exact" not in read["beside"]["deep_minus_shallow"]
    # WHERE IT SITS: on the two problems the start model solves the difference is 6/64, 6.5/64, 6/64 (above zero); on G' 6/64, 6/64, 6.5/64 (above zero): REACH TOO.
    sits = read["where_it_sits"]
    assert sits["name"] == REACH_TOO and sits["on_the_problems_the_start_solves"]["at_each_seed"] == {"3": 0.09375, "4": 0.101562, "5": 0.09375}
    assert sits["on_the_goal_set_again"]["at_each_seed"] == {"3": 0.09375, "4": 0.09375, "5": 0.101562}
    spread = read["the_spread"]
    assert spread[PUBLISHED] == {"by_seed": {"3": 0.125, "4": 0.125, "5": 0.125}, "mean": 0.125, "standard_deviation": 0.0, "sampling_noise_of_one_figure": spread[PUBLISHED]["sampling_noise_of_one_figure"]}
    assert spread[DEEP]["by_seed"] == {"3": 0.25, "4": 0.25, "5": 0.257812} and spread[DEEP]["standard_deviation"] == round(statistics.stdev((32 / 128, 32 / 128, 33 / 128)), 6)
    assert spread[PUBLISHED]["sampling_noise_of_one_figure"] == THREE[3]["figures"][PUBLISHED][ALL]["sampling_noise"] > 0
    assert read["figures"]["4"][SHALLOW][ALL] == {key: THREE[4]["figures"][SHALLOW][ALL][key] for key in ("successes", "attempts", "per_1000", "sampling_noise")}
    assert read["counts"] == {str(seed): THREE[seed]["counts"] for seed in (3, 4, 5)} and read["what_follows"]["name"] == L6_RUNS
    assert read["exploratory"]["read"] is False and read["exploratory"]["exploratory"] is True and "L5's seeds are not there" in read["exploratory"]["why"]


def test_a_seed_of_the_fixed_number_that_is_not_there_makes_every_read_not_read_and_says_which():
    by_seed = {3: THREE[3], 4: {"there": False, "why": "ladder_l5_draws_pre_r64_seed4 does not hold ['report_ladder_l5_draws.json']"}}
    read = over_the_seeds(by_seed, (3, 4, 5))
    assert (read["read"], read["seeds_there"], read["seeds_missing"]) == (False, [3], [4, 5]) and read["primary"]["name"] == read["where_it_sits"]["name"] == read["what_follows"]["name"] == NOT_READ
    assert read["primary"]["reason"] == "the reads are fixed over the seeds [3, 4, 5] and the runs of [4, 5] are not there: nothing is read over the seeds"
    assert read["beside"] is None and read["the_spread"] is None and read["exploratory"] is None and list(read["figures"]) == ["3"]
    # SIX SEEDS, once the approval is recorded: three that are there do not make the read.
    assert over_the_seeds(THREE, (3, 4, 5, 6, 7, 8))["seeds_missing"] == [6, 7, 8] and over_the_seeds(THREE, (3, 4, 5, 6, 7, 8))["primary"]["name"] == NOT_READ
    six = {**THREE, **{seed: _seed(seed) for seed in (6, 7, 8)}}
    assert over_the_seeds(six, (3, 4, 5, 6, 7, 8))["primary"]["critical_value"] == 2.571 and over_the_seeds(six, (3, 4, 5, 6, 7, 8))["primary"]["seeds"] == 6
    # A SEED THAT IS THERE AND NOT OF THE FIXED NUMBER is a new replication: named, and left out.
    more = over_the_seeds(six, (3, 4, 5))
    assert more["seeds_not_of_the_fixed_number"] == [6, 7, 8] and more["primary"] == over_the_seeds(THREE, (3, 4, 5))["primary"] and list(more["figures"]) == ["3", "4", "5"]


def test_a_seed_whose_check_failed_is_left_out_and_the_run_is_inconclusive_until_it_is_run_again():
    failed = {**THREE, 4: {**THREE[4], "inconclusive": True, "failed_checks": ["`deep`: no_barred_row"], "figures": None, "counts": None}}
    read = over_the_seeds(failed, (3, 4, 5))
    assert (read["read"], read["seeds_whose_checks_failed"], read["primary"]["name"], read["what_follows"]["name"]) == (False, {"4": ["`deep`: no_barred_row"]}, INCONCLUSIVE, NOT_READ)
    assert "a check failed at seed 4 (`deep`: no_barred_row): such a seed is left out, and the run is INCONCLUSIVE until it is run again" in read["primary"]["reason"]
    assert list(read["figures"]) == ["3", "5"] and read["where_it_sits"]["name"] == NOT_READ
    not_ok = over_the_seeds({**THREE, 5: {**THREE[5], "ok": False}}, (3, 4, 5))
    assert not_ok["primary"]["name"] == INCONCLUSIVE and not_ok["seeds_whose_checks_failed"] == {"5": ["its report is not to be read"]}
    # A SEED MISSING AND A SEED FAILED: nothing is read, and both are said.
    both = over_the_seeds({3: THREE[3], 4: failed[4]}, (3, 4, 5))
    assert both["primary"]["name"] == NOT_READ and "the runs of [5] are not there (and the checks of [4] failed)" in both["primary"]["reason"]


def test_seeds_of_two_recipes_or_measured_otherwise_are_not_pooled():
    with pytest.raises(ValueError, match="the seeds were trained with two recipes .seed 3: `constant`, seed 4: `decay`, seed 5: `constant`.: seeds of two recipes are not pooled"):
        over_the_seeds({**THREE, 4: {**THREE[4], "recipe": DECAY}}, (3, 4, 5))
    with pytest.raises(ValueError, match="the seeds were not measured alike"):
        over_the_seeds({**THREE, 5: {**THREE[5], "the_measurement": {**MEASURED, "sampling_seed": 1020}}}, (3, 4, 5))
    refused = read_over_the_seeds({**THREE, 4: {**THREE[4], "recipe": DECAY}}, (3, 4, 5))
    assert (refused["read"], refused["primary"]["name"], refused["what_follows"]["name"], refused["seeds_there"]) == (False, NOT_READ, NOT_READ, [3, 4, 5])
    assert refused["primary"]["reason"].endswith("seeds of two recipes are not pooled: nothing is read over the seeds") and "two recipes" in refused["refused"]
    lines = lines_over_the_seeds(refused)
    assert len(lines) == 1 and lines[0].startswith(f"{SAY}: THE PRIMARY, over the fresh seeds [3, 4, 5] alone") and f": {NOT_READ}. the seeds were trained with two recipes" in lines[0]
    # A seed that is not there has no recipe: the two that are there, of one recipe, are read as far as they go.
    assert read_over_the_seeds({3: THREE[3], 4: {"there": False, "why": "not given"}, 5: THREE[5]}, (3, 4, 5))["seeds_missing"] == [4]


def test_l5s_own_seeds_stand_beside_the_fresh_ones_as_exploratory_under_the_recipe_l5_trained_with_alone():
    of_l5 = {0: {"there": True, "figures": _figures((8, 8, 0, 0), (9, 8, 0, 0), (9, 8, 0, 0))}, 1: {"there": True, "figures": _figures((8, 8, 0, 0), (12, 8, 4, 0), (12, 8, 4, 0))},
             2: {"there": True, "figures": _figures((8, 8, 0, 0), (12, 8, 0, 0), (12, 8, 8, 0))}}
    read = over_the_seeds(THREE, (3, 4, 5), of_l5)
    exploratory = read["exploratory"]
    assert (exploratory["read"], exploratory["exploratory"], exploratory["seeds_that_found_the_sign"]) == (True, True, [0, 1, 2]) and exploratory["what"].startswith("EXPLORATORY: ")
    assert exploratory["of_l5"] == {"0": round(1 / 128, 6), "1": 0.0625, "2": 0.0625}
    assert exploratory["pooled"]["seeds"] == 6 and exploratory["pooled"]["critical_value"] == 2.571 and list(exploratory["pooled"]["at_each_seed"]) == ["0", "1", "2", "3", "4", "5"]
    # THE POOL holds L5's own differences and the fresh seeds' PRIMARY ones, each on all of G: nothing of a part, and no seed's figure in another's place.
    assert exploratory["pooled"]["at_each_seed"] == {**exploratory["of_l5"], **read["primary"]["at_each_seed"]} == {
        "0": round(1 / 128, 6), "1": 0.0625, "2": 0.0625, "3": 0.09375, "4": 0.097656, "5": 0.097656}
    assert read["primary"] == over_the_seeds(THREE, (3, 4, 5))["primary"]                                       # THE PRIMARY IS THE FRESH SEEDS' ALONE, whatever stands beside
    six = {**THREE, **{seed: _seed(seed) for seed in (6, 7, 8)}}
    nine = over_the_seeds(six, (3, 4, 5, 6, 7, 8), of_l5)["exploratory"]["pooled"]
    assert (nine["seeds"], nine["critical_value"]) == (9, 2.306)                                                # THE NINE POOLED
    at_each = {3: 0.01, 4: 0.02, 5: 0.03}
    assert pooled(DECAY, at_each, of_l5) == {"exploratory": True, "what": exploratory["what"], "read": False,
                                             "why": "the recipe is `decay` and L5's seeds were trained with `constant`: they do not stand beside these"}
    absent = pooled(CONSTANT, at_each, {**of_l5, 2: {"there": False, "why": "ladder_l5_deep_pre_r64_seed2 does not hold ['report_ladder_l5.json']"}})
    assert absent["read"] is False and absent["why"] == "L5's seeds are not there: seed 2: ladder_l5_deep_pre_r64_seed2 does not hold ['report_ladder_l5.json']"
    assert pooled(CONSTANT, at_each, None)["read"] is False and pooled(CONSTANT, at_each, {})["read"] is False
    assert pooled(CONSTANT, at_each, {3: of_l5[0], 6: of_l5[1]})["read"] is False                               # a seed that is both is no seed more


def test_the_lines_of_the_read_begin_with_the_label_and_say_what_is_said_beside_every_read():
    of_l5 = {seed: {"there": True, "figures": _figures((8, 8, 0, 0), (12, 8, 4, 0), (12, 8, 4 + seed, 0))} for seed in (0, 1, 2)}
    read = read_over_the_seeds({**THREE, 6: _seed(6)}, (3, 4, 5), of_l5)
    said = "L5n's run is not there or not finished (x): the recipe as it is, one training a model, is trained with, and L5n did not name it"
    lines = lines_over_the_seeds(read, said)
    assert all(line.startswith(f"{SAY}: ") for line in lines) and [line.split(": ", 2)[1].split(",")[0].split(" (")[0] for line in lines] == [
        "THE PRIMARY", "the seeds [6] are there and are not of the fixed number", "BESIDE", "WHERE IT SITS", "THE SPREAD", "EXPLORATORY", "WHAT FOLLOWS",
        "what a number of seeds would resolve"]
    assert f": {BEAT}. " in lines[0] and lines[0].endswith(f" NOTE: {said}") and lines[3].endswith(f" NOTE: {said}") and lines[6].endswith(f" NOTE: {said}")
    assert "`deep` minus `published` " in lines[2] and "`shallow` minus `published` " in lines[2] and "`deep` minus `shallow` " in lines[2] and "positive at 3 of the 3 fresh seeds" in lines[2]
    assert "L5's own seeds [0, 1, 2] FOUND THE SIGN" in lines[5] and "pooled with the fresh seeds, 6 seeds" in lines[5] and "confirms nothing" in lines[5]
    plain = lines_over_the_seeds(read_over_the_seeds(THREE, (3, 4, 5)))
    assert not any("NOTE:" in line for line in plain) and "L5's seeds beside the fresh ones are NOT READ" in plain[4]
    missing = lines_over_the_seeds(read_over_the_seeds({3: THREE[3]}, (3, 4, 5)), said)
    assert len(missing) == 1 and f": {NOT_READ}. the reads are fixed over the seeds [3, 4, 5] and the runs of [4, 5] are not there" in missing[0]


# ------------------------------------------------------------------------------------------------ the command
def _pulled(root, seed, entry, name=None, **more):
    """A pulled task directory of one seed, with the report its report step wrote (what the command reads of it)."""
    steps = root / (name or f"ladder_l5_draws_seed{seed}_r1") / "steps"
    steps.mkdir(parents=True)
    (steps / "report_ladder_l5_draws.json").write_text(json.dumps({"check": "L5d", "seed": seed, "of_this_seed": entry, "the_seeds_of_l5": {}, "the_recipe": {"said_beside_every_read": None}, **more}))
    return steps.parent


def test_the_command_reads_the_pulled_task_directories_that_are_there_and_prints_the_read(tmp_path, capsys):
    of_l5 = {str(seed): {"there": True, "figures": _figures((8, 8, 0, 0), (12, 8, 4, 0), (12, 8, 4 + seed, 0))} for seed in (0, 1, 2)}
    directories = [_pulled(tmp_path, 3, THREE[3]), _pulled(tmp_path, 4, THREE[4]), _pulled(tmp_path, 5, THREE[5], the_seeds_of_l5=of_l5)]
    out = tmp_path / "out" / "read.json"
    assert tool.main([*map(str, directories), "--out", str(out)]) == 0
    printed, written = capsys.readouterr().out.splitlines(), json.loads(out.read_text())
    expected = read_over_the_seeds({seed: {**THREE[seed], "there": True, "run": ""} for seed in (3, 4, 5)}, (3, 4, 5), {int(seed): own for seed, own in of_l5.items()})
    assert printed == written["lines"] == lines_over_the_seeds(expected) and written["primary"] == json.loads(json.dumps(expected["primary"])) and written["primary"]["name"] == BEAT
    assert written["directories"] == {str(seed): str(directory / "steps") for seed, directory in zip((3, 4, 5), directories)} and written["exploratory"]["read"] is True
    # THE STEPS DIRECTORY may be named in a task directory's place; a seed whose directory is not given makes the read NOT READ, naming it.
    assert tool.main([str(directories[0] / "steps"), str(directories[2])]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == 1 and f": {NOT_READ}. the reads are fixed over the seeds [3, 4, 5] and the runs of [4] are not there" in printed[0]
    # THE SEEDS THE READS ARE OVER are the setting's: with explicit approval recorded (a config given by --config) the three directories are three of six.
    text, approved = tool.CONFIG.read_text(), tmp_path / "out" / "approved.yaml"
    assert text.count("more_seeds_approved: false") == 1
    approved.write_text(text.replace("more_seeds_approved: false", "more_seeds_approved: true"))
    assert tool.main([*map(str, directories), "--config", str(approved)]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == 1 and f": {NOT_READ}. the reads are fixed over the seeds [3, 4, 5, 6, 7, 8] and the runs of [6, 7, 8] are not there" in printed[0]
    assert not list(tmp_path.glob("*.json")) and sorted(path.name for path in tmp_path.iterdir()) == sorted([*(directory.name for directory in directories), "out"])


def test_the_command_refuses_a_directory_that_is_not_a_finished_task_of_the_stage_and_two_of_one_seed(tmp_path, capsys):
    good = _pulled(tmp_path, 3, THREE[3])
    empty = tmp_path / "ladder_l5_draws_seed4_r1"
    empty.mkdir()
    assert tool.main([str(good), str(empty)]) == 2 and "does not hold report_ladder_l5_draws.json" in capsys.readouterr().err
    (empty / "report_ladder_l5_draws.json").write_text(json.dumps({"check": "L5n", "of_this_seed": THREE[4]}))
    assert tool.main([str(good), str(empty)]) == 2 and "is not a report of L5d with what it says of its own seed" in capsys.readouterr().err
    (empty / "report_ladder_l5_draws.json").write_text(json.dumps({"check": "L5d", "of_this_seed": {**THREE[4], "seed": True}}))
    assert tool.main([str(empty)]) == 2 and "is not a report of L5d" in capsys.readouterr().err
    twice = _pulled(tmp_path, 3, THREE[3], name="ladder_l5_draws_seed3_r2")
    assert tool.main([str(good), str(twice)]) == 2 and "are both of seed 3: one task directory a seed" in capsys.readouterr().err
    # SEEDS OF TWO RECIPES are not refused by the command: the read says NOT READ, with the reason.
    other = _pulled(tmp_path, 4, {**THREE[4], "recipe": DECAY}, name="other")
    assert tool.main([str(good), str(other), str(_pulled(tmp_path, 5, THREE[5]))]) == 0 and "seeds of two recipes are not pooled" in capsys.readouterr().out
