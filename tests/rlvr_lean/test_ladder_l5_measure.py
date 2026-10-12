"""L5m, the rules on hand-made rows: the names; the candidate the settings may name and the task's seed; why the run that
made the candidate cannot carry the stage; the candidate is the adapter that stage measured (and the one case a smoke run
alone passes); the hold of the two adapters; every set is the setting's; the two reads at their edges (an interval that
ends at zero spans it), INCONCLUSIVE and NOT READ; what follows, with the one outcome the section does not name; the
problems solved reliably and the ones never solved. Spec: docs/spec/ladder-loop.spec.md, "L5m: the best model,
measured in full". Pure: no file, no GPU, no network. The stage end to end is `test_ladder_l5_measure_stage.py`."""

import pytest

from rlvr_lean.domain.ladder_round import l5_measure
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ
from rlvr_lean.domain.ladder_round.l5_measure import (
    BAR,
    CANDIDATE,
    DOES_NOT_HOLD,
    DOES_NOT_HOLD_THERE,
    HOLDS_ON_FRESH,
    HOLDS_ON_THE_SET,
    L5C,
    L5F,
    MODELS_OF,
    NOT_NAMED,
    NOT_SHOWN_ON_THE_SET,
    ON_FRESH,
    ON_THE_SET,
    SELECTION_ON_G,
    THE_BEST_MEASURED,
    adapters_are_held,
    by_the_interval,
    is_the_one_measured,
    never_solved,
    of_its_run,
    reliably,
    sets_are_the_settings,
    the_candidate,
    the_seed,
    what_follows,
)

SEEDS_OF = {L5F: (3,), L5C: (3,)}
MEASUREMENT = {"sampling": "reach", "episodes": 32, "sampling_seed": 1001, "problems": 392}
ENTRY = {"seed": 3, "start": "second_full", "recipe": "mean_constant", "ok": True, "inconclusive": False, "failed_checks": [], "figures": {"published": 0.14477},
         "the_measurement": dict(MEASUREMENT)}
ADAPTER = "/runs/theirs/adapters/published"
AVERAGING = {"averaged": True, "adapter": ADAPTER, "of": ["published_1", "published_2", "published_3", "published_4"], "adapter_saved": {"rank": 64, "alpha": 128},
             "read_back": {"is_the_mean": True, "is_none_of_them": True, "elements": 100, "elements_not_the_mean": 0}}
MEASURING = {"adapter": ADAPTER}


def test_the_names_are_the_sections():
    assert (L5F, L5C) == ("l5_full", "l5_cont") and (BAR, CANDIDATE) == ("second_full", "candidate")
    assert MODELS_OF == {"l5_full": ("published", "shallow", "deep"), "l5_cont": ("wide", "again", "more")}         # `single` of L5c is one training, not an averaged adapter
    assert ON_THE_SET == ("THE GAIN HOLDS ON THE HELD-ASIDE SET", "NOT SHOWN ON THE HELD-ASIDE SET", "IT DOES NOT HOLD THERE")
    assert ON_FRESH == ("THE GAIN HOLDS ON FRESH ATTEMPTS", "NOT SHOWN", "IT DOES NOT HOLD")
    assert l5_measure.FOLLOWS == (THE_BEST_MEASURED, SELECTION_ON_G, NOT_NAMED, NOT_READ) and l5_measure.HALF == 2


# ---------------------------------------------------------------------------------------------- the candidate
@pytest.mark.parametrize("of, model", [(L5F, "published"), (L5F, "shallow"), (L5F, "deep"), (L5C, "wide"), (L5C, "again"), (L5C, "more")])
def test_a_candidate_is_an_averaged_adapter_of_one_of_the_two_stages(of, model):
    the_candidate(of, 3, model, SEEDS_OF)


@pytest.mark.parametrize("of, seed, model, said", [
    ("l5_draws", 3, "published", "of the stage 'l5_draws'"),             # another stage: L5d's models are from `pre_r64`, not the bar
    ("l5_stock", 3, "stock", "of the stage 'l5_stock'"),
    (L5C, 3, "single", "'single' of `l5_cont`"),                         # one training, not an averaged adapter
    (L5C, 3, "published", "'published' of `l5_cont`"),                   # a model of the other stage
    (L5F, 3, "wide", "'wide' of `l5_full`"),
    (L5F, 4, "published", "seed is 4"),                                  # a seed that stage's settings do not list
    (L5F, "3", "published", "seed is '3'"),
    (L5F, True, "published", "seed is True"),
])
def test_a_candidate_that_is_anything_else_is_refused(of, seed, model, said):
    with pytest.raises(ValueError, match=said):
        the_candidate(of, seed, model, SEEDS_OF)


def test_a_task_of_another_seed_than_the_candidates_is_refused():
    the_seed(3, 3)
    with pytest.raises(ValueError, match="this task's seed is 4 and the candidate is of seed 3"):
        the_seed(4, 3)


# ---------------------------------------------------------------------------------------------- its run
def test_a_run_that_made_the_candidate_and_can_be_read_carries_the_stage():
    assert of_its_run(ENTRY, 3, "second_full", "mean_constant", MEASUREMENT) is None


@pytest.mark.parametrize("change, said", [
    ({"there": False, "why": "its report is not on the box"}, "its report is not on the box"),
    ({"seed": 4}, "a report of seed 4"),
    ({"start": "pre_r64"}, "a run from `pre_r64`"),
    ({"recipe": "constant"}, "its recipe is `constant`"),
    ({"ok": False}, "not to be read"),
    ({"inconclusive": True, "failed_checks": ["lean_answered"]}, f"it reads {INCONCLUSIVE} \\(lean_answered\\)"),
    ({"figures": {}}, f"it reads {INCONCLUSIVE}"),
    ({"the_measurement": {**MEASUREMENT, "sampling_seed": 1002}}, "it was measured otherwise"),
    ({"the_measurement": {**MEASUREMENT, "episodes": 16}}, "it was measured otherwise"),
])
def test_why_the_run_that_made_the_candidate_cannot_carry_the_stage(change, said):
    import re

    why = of_its_run({**ENTRY, **change}, 3, "second_full", "mean_constant", MEASUREMENT)
    assert why and re.search(said, why)


def test_the_candidate_is_the_adapter_its_stage_averaged_and_measured():
    held = is_the_one_measured(AVERAGING, MEASURING, ADAPTER, 64, 128)
    assert held["passes"] and (held["is_the_mean"], held["at_its_rank"], held["one_adapter"], held["served_here"]) == (True, True, True, True)
    assert (held["adapter"], held["averaged_as"], held["measured_there_from"], held["served_here_from"]) == (ADAPTER, ADAPTER, ADAPTER, [])
    assert held["of"] == AVERAGING["of"] and (held["elements"], held["elements_not_the_mean"]) == (100, 0) and held["read_back"] == {"rank": 64, "alpha": 128}
    assert is_the_one_measured(AVERAGING, MEASURING, ADAPTER, 64, 128, [ADAPTER, ADAPTER, ADAPTER])["passes"]         # ... and every measurement of this run was sampled from it


@pytest.mark.parametrize("averaging, measuring, adapter, rank, alpha, served, failing", [
    ({**AVERAGING, "averaged": False}, MEASURING, ADAPTER, 64, 128, (), "is_the_mean"),
    ({**AVERAGING, "read_back": {**AVERAGING["read_back"], "is_the_mean": False}}, MEASURING, ADAPTER, 64, 128, (), "is_the_mean"),
    ({**AVERAGING, "read_back": {**AVERAGING["read_back"], "is_none_of_them": False}}, MEASURING, ADAPTER, 64, 128, (), "is_the_mean"),        # one of its members
    ({key: value for key, value in AVERAGING.items() if key != "read_back"}, MEASURING, ADAPTER, 64, 128, (), "is_the_mean"),                # no record of the mean
    ({**AVERAGING, "adapter_saved": {"rank": 16, "alpha": 32}}, MEASURING, ADAPTER, 64, 128, (), "at_its_rank"),
    (AVERAGING, MEASURING, ADAPTER, None, None, (), "at_its_rank"),
    ({**AVERAGING, "adapter_saved": {"rank": None, "alpha": None}}, MEASURING, ADAPTER, None, None, (), "at_its_rank"),                    # nothing against nothing does not pass
    (AVERAGING, {"adapter": ADAPTER + "_1"}, ADAPTER, 64, 128, (), "one_adapter"),                                     # that stage measured another adapter
    (AVERAGING, MEASURING, ADAPTER + "_1", 64, 128, (), "one_adapter"),                                                # this run is handed another
    (AVERAGING, {}, ADAPTER, 64, 128, (), "one_adapter"),
    (AVERAGING, MEASURING, ADAPTER, 64, 128, (ADAPTER, None), "served_here"),                                          # a measurement of this run from no adapter
    (AVERAGING, MEASURING, ADAPTER, 64, 128, (ADAPTER, ADAPTER + "_1"), "served_here"),
])
def test_what_fails_the_check_that_the_candidate_is_the_one_measured(averaging, measuring, adapter, rank, alpha, served, failing):
    held = is_the_one_measured(averaging, measuring, adapter, rank, alpha, served)
    assert not held["passes"] and not held[failing]
    assert all(held[other] for other in ("is_the_mean", "at_its_rank", "one_adapter", "served_here") if other != failing)


def test_an_averaged_adapter_that_is_one_of_its_models_passes_in_a_smoke_run_of_this_stage_and_nowhere_else():
    theirs = {**AVERAGING, "one_of_its_models_in_a_smoke_run": True, "read_back": {**AVERAGING["read_back"], "is_none_of_them": False}}
    assert not is_the_one_measured(theirs, MEASURING, ADAPTER, 64, 128)["passes"]                   # a real run of this stage: the record that stage kept it does not help
    passed = is_the_one_measured(theirs, MEASURING, ADAPTER, 64, 128, smoke=True)
    assert passed["passes"] and passed["one_of_its_models_in_a_smoke_run"]
    assert not is_the_one_measured({**theirs, "one_of_its_models_in_a_smoke_run": False}, MEASURING, ADAPTER, 64, 128, smoke=True)["passes"]     # a smoke run still needs that stage's record
    assert is_the_one_measured(AVERAGING, MEASURING, ADAPTER, 64, 128, smoke=True)["one_of_its_models_in_a_smoke_run"] is False


def test_the_two_adapters_are_held():
    recorded = {"adapter_saved": {"rank": 64, "alpha": 128}, "numbers": 100, "numbers_not_its_start_adapters": 97}
    bar = {"adapter": "/runs/l4p/adapters/second_full", "read_now": None, "passes": True}
    held = adapters_are_held({"adapter": ADAPTER, "recorded": recorded, "read_now": None}, bar, 64, 128)
    assert held["passes"] and held["the_candidate"]["passes"] and held["the_bar"]["passes"] and (held["rank"], held["alpha"]) == (64, 128)
    assert held["the_candidate"]["adapter"] == ADAPTER and held["the_bar"]["adapter"] == bar["adapter"]
    on_a_gpu_box = adapters_are_held({"adapter": ADAPTER, "recorded": recorded, "read_now": {"passes": True}}, {**bar, "read_now": {"passes": True}}, 64, 128)
    assert on_a_gpu_box["passes"] and on_a_gpu_box["the_candidate"]["read_now"] == {"passes": True}
    failing = [
        ({"adapter": ADAPTER, "recorded": {**recorded, "numbers_not_its_start_adapters": 0}, "read_now": None}, bar, 64, 128),     # number for number its start adapter
        ({"adapter": ADAPTER, "recorded": {**recorded, "adapter_saved": {"rank": 16, "alpha": 32}}, "read_now": None}, bar, 64, 128),
        ({"adapter": ADAPTER, "recorded": {}, "read_now": None}, bar, 64, 128),                                                    # no record
        ({"adapter": ADAPTER, "recorded": recorded, "read_now": {"passes": False}}, bar, 64, 128),                                 # the files themselves
        ({"adapter": ADAPTER, "recorded": recorded, "read_now": None}, bar, None, None),
    ]
    for of_the_candidate, of_the_bar, rank, alpha in failing:
        held = adapters_are_held(of_the_candidate, of_the_bar, rank, alpha)
        assert not held["passes"] and not held["the_candidate"]["passes"] and held["the_bar"]["passes"]
    held = adapters_are_held({"adapter": ADAPTER, "recorded": recorded, "read_now": None}, {**bar, "passes": False}, 64, 128)
    assert not held["passes"] and held["the_candidate"]["passes"] and not held["the_bar"]["passes"]


# ---------------------------------------------------------------------------------------------- the sets
def _rows(ids, episodes, resolved=0):
    return [{"problem_id": problem, "episodes": episodes, "resolved": resolved} for problem in ids]


def _a_set(ids=("a", "b", "c"), attempts=32, seed=1001, **change):
    return {"rows": _rows(ids, attempts), "problem_ids": list(ids), "attempts": attempts, "sampling_seed": seed, "recorded_attempts": attempts, "recorded_seed": seed, **change}


def test_every_set_is_the_settings():
    held = sets_are_the_settings({"the first sampling": _a_set(), "the held-aside set": _a_set(("x", "y"), 16, 1051)})
    assert held["passes"] and held["not_the_settings"] == [] and list(held["sets"]) == ["the first sampling", "the held-aside set"]
    assert held["sets"]["the held-aside set"] == {"problems": 2, "problems_of_the_setting": 2, "the_problems": True, "attempts": 16,
                                                  "attempts_a_problem_that_are_not_the_settings": [], "sampling_seed": 1051, "recorded_sampling_seed": 1051,
                                                  "recorded_attempts": 16, "passes": True}
    assert not sets_are_the_settings({})["passes"]                                                      # no set is not every set


@pytest.mark.parametrize("change", [
    {"rows": _rows(("a", "b"), 32)},                                    # a problem of the setting is not there
    {"rows": _rows(("a", "b", "c", "d"), 32)},                          # a problem that is not the setting's
    {"rows": _rows(("a", "b", "c", "c"), 32), "problem_ids": ["a", "b", "c", "c"]},        # a problem twice
    {"rows": []},
    {"rows": [*_rows(("a", "b"), 32), *_rows(("c",), 31)]},             # a problem with another number of attempts
    {"recorded_seed": 1002},                                            # sampled with another seed
    {"recorded_seed": None},
    {"recorded_attempts": 16},
])
def test_a_set_that_is_not_the_settings_fails_the_check_and_is_named(change):
    held = sets_are_the_settings({"the first sampling": _a_set(), "fresh attempts": _a_set(**change)})
    assert not held["passes"] and held["not_the_settings"] == ["fresh attempts"] and held["sets"]["the first sampling"]["passes"]


# ---------------------------------------------------------------------------------------------- the reads
def _difference(mean, low, high, problems=600):
    return {"mean": mean, "low": low, "high": high, "per_1000": 60.0, "per_1000_of_the_base": 50.0, "problems": problems}


@pytest.mark.parametrize("names", [ON_THE_SET, ON_FRESH])
def test_a_read_is_named_by_where_its_interval_lies(names):
    above, spanning, below = names
    assert by_the_interval([], _difference(0.01, 0.002, 0.018), names, "the set")["name"] == above
    assert by_the_interval([], _difference(0.01, -0.002, 0.018), names, "the set")["name"] == spanning
    assert by_the_interval([], _difference(-0.01, -0.018, -0.002), names, "the set")["name"] == below
    assert by_the_interval([], _difference(0.01, 0.0, 0.018), names, "the set")["name"] == spanning           # AN INTERVAL THAT ENDS AT ZERO SPANS IT, at either end
    assert by_the_interval([], _difference(-0.01, -0.018, 0.0), names, "the set")["name"] == spanning
    assert by_the_interval([], _difference(0.0, 0.0, 0.0), names, "the set")["name"] == spanning


def test_a_read_says_what_it_read_and_what_a_spanning_interval_resolves():
    read = by_the_interval([], _difference(0.01, 0.002, 0.018), ON_THE_SET, "the held-aside set")
    assert read == {"name": HOLDS_ON_THE_SET, "failed_checks": [], "mean": 0.01, "low": 0.002, "high": 0.018,
                    "reason": "on the held-aside set, the candidate minus `second_full`, successes per attempt, paired by problem: +0.01000 [+0.00200, +0.01800], "
                              "60.0 against 50.0 per 1,000 over 600 problems; the interval is above zero"}
    spanning = by_the_interval([], _difference(0.004, -0.006, 0.014), ON_FRESH, "G's second sampling")
    assert spanning["name"] == "NOT SHOWN" and spanning["resolves"] == 0.01 and "resolves 0.01000 per attempt (half the width of its interval)" in spanning["reason"]
    assert "resolves" not in read and "the interval spans zero" in spanning["reason"]
    below = by_the_interval([], _difference(-0.01, -0.018, -0.002), ON_THE_SET, "the held-aside set")
    assert below["name"] == DOES_NOT_HOLD_THERE and below["reason"].endswith("the interval is below zero")


def test_a_read_is_inconclusive_after_a_failed_check_and_not_read_with_nothing_paired():
    failed = by_the_interval(["lean_answered", "every_set_is_the_settings"], _difference(0.01, 0.002, 0.018), ON_THE_SET, "the held-aside set")
    assert failed == {"name": INCONCLUSIVE, "failed_checks": ["lean_answered", "every_set_is_the_settings"],
                      "reason": "a check of this run failed (lean_answered; every_set_is_the_settings): the held-aside set is not read"}
    unpaired = by_the_interval([], {"mean": None}, ON_FRESH, "G's second sampling")
    assert unpaired == {"name": NOT_READ, "failed_checks": [], "reason": "no problem of G's second sampling was measured on both sides: it is NOT READ"}
    assert by_the_interval([], {}, ON_FRESH, "G's second sampling")["name"] == NOT_READ


# ---------------------------------------------------------------------------------------------- what follows
@pytest.mark.parametrize("on_the_set, on_fresh, name", [
    (HOLDS_ON_THE_SET, HOLDS_ON_FRESH, THE_BEST_MEASURED),              # holds on both: said without the caution of selection
    (NOT_SHOWN_ON_THE_SET, HOLDS_ON_FRESH, SELECTION_ON_G),             # the primary decides: not shown there ...
    (NOT_SHOWN_ON_THE_SET, "NOT SHOWN", SELECTION_ON_G),
    (NOT_SHOWN_ON_THE_SET, DOES_NOT_HOLD, SELECTION_ON_G),
    (DOES_NOT_HOLD_THERE, HOLDS_ON_FRESH, SELECTION_ON_G),              # ... or not holding there
    (DOES_NOT_HOLD_THERE, DOES_NOT_HOLD, SELECTION_ON_G),
    (HOLDS_ON_THE_SET, "NOT SHOWN", NOT_NAMED),                         # THE SECTION NAMES NOTHING for a gain that holds on the set and not on fresh attempts
    (HOLDS_ON_THE_SET, DOES_NOT_HOLD, NOT_NAMED),
    (INCONCLUSIVE, INCONCLUSIVE, NOT_READ),
    (NOT_READ, HOLDS_ON_FRESH, NOT_READ),
    (HOLDS_ON_THE_SET, NOT_READ, NOT_READ),
])
def test_what_follows(on_the_set, on_fresh, name):
    follows = what_follows({"name": on_the_set}, {"name": on_fresh})
    assert follows["name"] == name and follows["reason"]
    if name == NOT_NAMED:
        assert "nothing is claimed of it, and it needs a decision" in follows["reason"]
    if name == SELECTION_ON_G:
        assert f"on the held-aside set the read is {on_the_set}" in follows["reason"] and "selection on G until a further seed shows otherwise" in follows["reason"]


# ---------------------------------------------------------------------------------------------- beside
def test_the_problems_solved_reliably_and_the_ones_never_solved():
    rows = [{"problem_id": "z", "episodes": 93, "resolved": 47}, {"problem_id": "a", "episodes": 93, "resolved": 46}, {"problem_id": "b", "episodes": 93, "resolved": 0},
            {"problem_id": "c", "episodes": 4, "resolved": 2}, {"problem_id": "d", "episodes": 0, "resolved": 0}, {"problem_id": "e", "episodes": 93, "resolved": 93}]
    everything = {"a", "b", "c", "d", "e", "z"}
    assert reliably(rows, everything) == ["c", "e", "z"]                # HALF ITS ATTEMPTS OR MORE (47 of 93, 2 of 4; not 46 of 93), at least once, sorted
    assert reliably(rows, {"z", "a"}) == ["z"] and reliably(rows, set()) == []
    assert never_solved(rows, everything) == ["b", "d"] and never_solved(rows, {"d", "e"}) == ["d"]          # in the order of the rows
