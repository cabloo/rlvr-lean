"""The ceiling, a labelled diagnostic: what its training file may hold, its schedule by rows, the three checks, the
branch and the report on hand-made rows with a known answer. Spec: docs/spec/ladder-loop.spec.md, "The ceiling:
a labelled diagnostic". Pure: no model, no Lean. The stage end to end is `test_ladder_ceiling_stage.py`."""

import json
from pathlib import Path

import pytest
import yaml

from rlvr_lean.data.heldout_proof_lines import length_group
from rlvr_lean.domain.ladder_round import ceiling
from rlvr_lean.domain.ladder_round.ceiling import (
    AS_THE_LOOP,
    BRANCHES,
    CEILING,
    CHECKPOINTS,
    FEWER,
    INCONCLUSIVE,
    LEARNS,
    NOT_READ,
    NOT_SHOWN,
    TWO_MORE_SEEDS,
    can_this_run_see_a_win,
    ceiling_branch,
    check_examples_fit,
    checkpoint_steps,
    doses,
    not_broken,
    still_writes_proofs,
    the_training_took,
    training_example,
    training_rows,
)
from rlvr_lean.domain.ladder_round.dose import epoch_orders, run_dose
from rlvr_lean.domain.repair.accumulate import proof_line_count
from rlvr_lean.gpu.model_utils import training_text
from rlvr_lean.reporting.ladder_ceiling import (
    ALL,
    FOUR_PLUS,
    attempts_on_g,
    build_ceiling_report,
    goal_sets,
    lean_did_not_answer,
    per_attempt,
    solved,
    verified_lines,
)

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
SETTINGS = CONFIG["ladder_loop"]["ceiling"]
FIXTURE = PACKAGE / "data" / "ladder_ceiling_fixture" / "training.jsonl"
TRAINING = PACKAGE / "data" / "ladder_ceiling" / "training.jsonl"
EVALUATION = {"bootstrap_resamples": 400, "bootstrap_seed": 0}


def _file_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _row(problem_id, **changes):
    return {"problem_id": problem_id, "kind": "stp_conjecture", "side": "true", "statement": f"theorem {problem_id} : P := by\n", "proof": "  step_one\n  done\n",
            "proof_lines": 2, "predicted_rate": 0.01, "certificate_source": "stp", **changes}


# ------------------------------------------------------------------------------------------------ the file
def test_the_settings_are_the_specs():
    assert SETTINGS["checkpoints"] == [2000, 8000] and SETTINGS["loop_arm"] == "t010" and SETTINGS["times_the_loops_own_gain"] == 2
    assert SETTINGS["loops_own_gain"] == 0.00064                    # the three seeds pooled, computed from the stored rows before the run
    assert SETTINGS["training_file"] == {"below_predicted_rate": 0.05, "stp_sample": 4800, "proofs_tried": 3, "maximum_characters": 2400, "seed": 0, "lean_seconds": 60}
    assert (SETTINGS["loss_window_rows"], SETTINGS["maximum_share_without_an_answer"], SETTINGS["minimum_share_of_the_bases_pass_rate_above_the_band"]) == (500, 0.05, 0.5)
    assert CONFIG["ladder_loop"]["l2_arms"][SETTINGS["loop_arm"]] == {"target_rate": 0.10}
    assert CONFIG["ladder_loop"]["measure"] == {"reach_episodes": 32, "rung_episodes": 8} and CHECKPOINTS == ("small", "full") and CEILING == "ceiling"


def test_the_first_rows_of_the_file_are_trained_on_in_its_order_and_a_short_file_is_refused():
    rows = [_row(f"pool{index}") for index in range(10)]
    assert training_rows(rows, set(), set(), 10) == rows and training_rows(rows, {"other"}, {"another"}, 6) == rows[:6]
    with pytest.raises(ValueError, match="has 10 rows and the last checkpoint is after 11"):
        training_rows(rows, set(), set(), 11)


def test_a_held_out_problem_anywhere_in_the_file_is_refused_and_so_is_one_of_the_base_map():
    rows = [_row(f"pool{index}") for index in range(10)]
    with pytest.raises(ValueError, match=r"1 rows of the ceiling's training file are held-out problems \(first: pool9\)"):
        training_rows(rows, {"pool9"}, set(), 4)           # beyond the rows that would be trained on: refused all the same
    with pytest.raises(ValueError, match=r"2 rows of the ceiling's training file are problems of the base map \(first: pool1\)"):
        training_rows(rows, set(), {"pool1", "pool3"}, 4)


def test_a_row_on_another_side_a_repeated_problem_and_a_row_that_is_not_a_statement_and_a_proof_are_refused():
    rows = [_row(f"pool{index}") for index in range(4)]
    with pytest.raises(ValueError, match="not on the side `true`"):
        training_rows([*rows, _row("refuted", side="false")], set(), set(), 4)
    with pytest.raises(ValueError, match=r"1 problems appear twice .*first: pool2"):
        training_rows([*rows, _row("pool2")], set(), set(), 4)
    with pytest.raises(ValueError, match=r"row 4 .* has no \['proof_lines'\]"):
        training_rows([*rows, {key: value for key, value in _row("bare").items() if key != "proof_lines"}], set(), set(), 4)
    for broken in (_row("open", statement="theorem open : P :=\n"), _row("empty", proof="  \n")):
        with pytest.raises(ValueError, match="is not a statement up to `:= by` and a proof"):
            training_rows([*rows, broken], set(), set(), 4)


def test_a_row_is_the_rounds_training_example_and_one_the_recipe_would_cut_is_refused():
    row = _row("pool1")
    example = training_example(row)
    assert example == {"problem_id": "pool1", "side": "statement", "theorem": row["statement"], "completion": row["proof"]}
    prompt, target = training_text(example["theorem"], example["completion"])       # the round's builder: the proof, then the closing fence
    assert prompt.endswith(row["statement"]) and target == "  step_one\n  done\n```"
    check_examples_fit(["a", "b"], [100, 2048], 2048)
    with pytest.raises(ValueError, match=r"1 rows .* longer than the 2048 tokens .*first: b, 2049 tokens"):
        check_examples_fit(["a", "b"], [100, 2049], 2048)


def test_the_committed_fixture_is_rows_the_stage_takes_and_holds_no_held_out_problem():
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    rows = _file_rows(FIXTURE)
    heldout = {row["problem_id"] for row in _file_rows(PACKAGE / "data" / "ladder_l0" / "heldout.jsonl")}
    base_map = {row["problem_id"] for row in _file_rows(PACKAGE / "data" / "ladder_l0" / "base_map.jsonl")}
    assert len(heldout) == 2000 and len(base_map) == 4000
    assert training_rows(rows, heldout, base_map, 24) == rows and {row["kind"] for row in rows} == {"lean_workbook", "stp_conjecture"}
    assert all(row["proof_lines"] == proof_line_count(row["proof"]) and row["proof"].endswith("\n") and row["statement"].endswith(":= by\n") for row in rows)
    assert {length_group(row["proof_lines"]) for row in rows} == {"1", "2-3", "4-7", "8+"}


@pytest.mark.skipif(not TRAINING.exists(), reason="needs the published-proof training file, which is not distributed")
def test_the_training_file_passes_what_the_prepare_step_checks_before_it_counts_tokens():
    rows = _file_rows(TRAINING)
    heldout = {row["problem_id"] for row in _file_rows(PACKAGE / "data" / "ladder_l0" / "heldout.jsonl")}
    base_map = {row["problem_id"] for row in _file_rows(PACKAGE / "data" / "ladder_l0" / "base_map.jsonl")}
    chosen = training_rows(rows, heldout, base_map, SETTINGS["checkpoints"][-1])          # no held-out problem, none of the base map, the side true, no problem twice
    assert len(chosen) == SETTINGS["checkpoints"][-1] == len(rows)


# -------------------------------------------------------------------------------------------- the schedule
def test_a_checkpoint_is_saved_after_the_first_step_at_or_after_its_rows():
    batch = CONFIG["training"]["effective_batch"]
    assert batch == 8 and checkpoint_steps(SETTINGS["checkpoints"], batch) == {
        "small": {"rows": 2000, "step": 250, "rows_seen": 2000}, "full": {"rows": 8000, "step": 1000, "rows_seen": 8000}}
    # Rows that do not fall on a step: the step at or after them, and the rows seen by then are recorded.
    assert checkpoint_steps([12, 24], 8) == {"small": {"rows": 12, "step": 2, "rows_seen": 16}, "full": {"rows": 24, "step": 3, "rows_seen": 24}}
    assert checkpoint_steps([10, 20], 8) == {"small": {"rows": 10, "step": 2, "rows_seen": 16}, "full": {"rows": 20, "step": 3, "rows_seen": 20}}
    for wrong in ([2000], [8000, 2000], [2000, 2000], [0, 8], [1, 2, 3]):
        with pytest.raises(ValueError, match="one increasing number of rows for each"):
            checkpoint_steps(wrong, 8)


def test_the_training_loop_takes_the_files_order_and_gives_each_rows_own_loss():
    seen, saved = [], []

    def train_step(batch):
        seen.append(list(batch))
        return [[float(index), 1.0, 0.5, 0.25] for index in batch]

    schedule = checkpoint_steps([12, 24], 8)
    result = run_dose(24, 8, 1, 0, [], {name: entry["step"] for name, entry in schedule.items()}, train_step, lambda step: {},
                      lambda name, step: saved.append((name, step, len(seen))), orders=[list(range(24))], per_example=True)
    assert seen == [list(range(0, 8)), list(range(8, 16)), list(range(16, 24))] and seen[0] != epoch_orders(24, 1, 0)[0][:8]     # the round's order is a shuffle
    assert saved == [("small", 2, 2), ("full", 3, 3)] and result["steps"] == 3 and result["readings"] == []
    losses = [loss for row in result["training_steps"] for loss in row["example_losses"]]
    assert losses == [round((index + 1.75) / 4, 5) for index in range(24)]        # row i's own mean loss, in the file's order
    # Without the two arguments the loop is what it was: the round's order, and no per-example losses.
    plain = run_dose(24, 8, 1, 0, [], {"small": 2, "full": 3}, lambda batch: [[1.0, 1.0, 0.5, 0.25] for _ in batch], lambda step: {}, lambda name, step: None)
    assert all("example_losses" not in row for row in plain["training_steps"])


def test_a_dose_is_counted_by_kind_and_by_the_length_of_its_own_proofs():
    rows = [_row(f"pool{index}", kind="lean_workbook" if index % 3 == 0 else "stp_conjecture", proof_lines=[1, 2, 5, 9][index % 4]) for index in range(24)]
    counted = doses(rows, list(range(100, 124)), checkpoint_steps([12, 24], 8), length_group)
    assert counted["small"]["rows"] == 16 and counted["small"]["by_kind"] == {"lean_workbook": 6, "stp_conjecture": 10}
    assert counted["small"]["by_proof_lines"] == {"1": 4, "2-3": 4, "4-7": 4, "8+": 4} and counted["small"]["longest_tokens"] == 115
    assert counted["full"]["rows"] == 24 and counted["full"]["by_proof_lines"] == {"1": 6, "2-3": 6, "4-7": 6, "8+": 6} and counted["full"]["tokens"] == sum(range(100, 124))


# ---------------------------------------------------------------------------------------------- the checks
def _rung(problem_id, resolved, episodes=8, capped=0, no_answer=0, sides=1):
    return {"problem_id": problem_id, "kind": "lean_workbook", "side": "true", "episodes": episodes, "resolved": resolved, "sides": sides,
            "attempts_capped": capped, "attempts_timed_out": 0, "attempts_without_an_answer": no_answer, "attempts_not_checked": 0}


def test_the_training_took_when_the_last_rows_cost_less_than_the_first():
    falling = [2.0 - index / 1000 for index in range(8000)]
    took = the_training_took(falling, 500)
    assert took["passes"] is True and took["rows_compared"] == 500 and took["first"] == pytest.approx(1.7505) and took["last"] == pytest.approx(-5.7495, abs=1e-4)
    assert the_training_took(list(reversed(falling)), 500)["passes"] is False and the_training_took([1.0] * 2000, 500)["passes"] is False     # equal is not below
    short = the_training_took([3.0] * 12 + [1.0] * 12, 500)                    # a smoke run: its first half against its second
    assert short["rows_compared"] == 12 and (short["first"], short["last"], short["passes"]) == (3.0, 1.0, True)
    assert the_training_took([], 500)["passes"] is False


def test_the_model_still_writes_proofs_while_few_attempts_reach_the_cap_or_get_no_verdict():
    rows = [_rung(f"r{index}", 4) for index in range(10)]                       # 80 attempts
    assert still_writes_proofs(rows, 0.05) == {**still_writes_proofs(rows, 0.05), "attempts": 80, "share": 0.0, "passes": True}
    some = [*rows[:8], _rung("r8", 4, capped=2), _rung("r9", 4, no_answer=1)]
    counted = still_writes_proofs(some, 0.05)
    assert (counted["capped_at_the_token_limit"], counted["without_a_verdict_from_lean"], counted["share"], counted["passes"]) == (2, 1, 0.0375, True)
    assert still_writes_proofs([*rows[:9], _rung("r9", 0, capped=4)], 0.05)["passes"] is False       # 4 of 80 is not under 5%
    two_sided = still_writes_proofs([_rung("audited", 8, sides=2, capped=1)], 0.05)                  # an audited problem's ruled-out side is attempted too
    assert two_sided["attempts"] == 16 and two_sided["passes"] is False and still_writes_proofs([], 0.05)["passes"] is False


def test_training_has_not_broken_the_model_while_it_keeps_half_the_bases_pass_rate_above_the_band():
    base = [_rung("a1", 6), _rung("a2", 8), _rung("b1", 1)]
    kept = not_broken([_rung("a1", 3), _rung("a2", 4), _rung("b1", 0)], base, {"a1", "a2"}, 0.5)
    assert (kept["pass_rate"], kept["pass_rate_of_the_base"], kept["minimum"], kept["passes"]) == (0.4375, 0.875, 0.4375, True)     # exactly half is at least half
    assert not_broken([_rung("a1", 3), _rung("a2", 3), _rung("b1", 8)], base, {"a1", "a2"}, 0.5)["passes"] is False                # the below-band rung does not count
    assert not_broken([], base, set(), 0.5)["passes"] is False


def test_any_failed_check_is_named_and_makes_the_run_inconclusive():
    rungs = [_rung("a1", 6), _rung("b1", 1)]
    fine = can_this_run_see_a_win([2.0] * 500 + [1.0] * 500, rungs, rungs, {"a1"}, SETTINGS)
    assert fine["all_pass"] is True and fine["failed"] == [] and set(fine) == {
        "the_training_took", "the_model_still_writes_proofs", "training_has_not_broken_it", "failed", "all_pass"}
    flat = can_this_run_see_a_win([1.0] * 1000, rungs, rungs, {"a1"}, SETTINGS)
    assert flat["all_pass"] is False and flat["failed"] == ["the_training_took"]
    branch = ceiling_branch(flat, {"mean": 0.02, "low": 0.01, "high": 0.03}, 0.00064, 2)
    assert branch["name"] == INCONCLUSIVE and "the training took: FAIL" in branch["reason"] and "nothing is said about the model" in branch["reason"]


# ---------------------------------------------------------------------------------------------- the branch
PASSING = {"all_pass": True, "failed": []}
OWN_GAIN = SETTINGS["loops_own_gain"]          # the loop's own gain: ONE number, the three seeds pooled, fixed before the run


def _change(mean, low, high):
    return {"problems": 230, "mean": mean, "low": low, "high": high}


def test_the_branch_is_the_specs_and_is_read_against_the_loops_own_gain_a_fixed_number():
    assert OWN_GAIN == 0.00064 and SETTINGS["times_the_loops_own_gain"] == 2
    learns = ceiling_branch(PASSING, _change(0.00128, 0.0002, 0.0024), OWN_GAIN, 2)       # clear of zero, and a point of exactly twice the loop's own
    assert learns["name"] == LEARNS and "the model can learn longer proofs from examples; L3d is worth building" in learns["reason"]
    assert "+0.00128 [+0.00020, +0.00240]" in learns["reason"]
    assert "the loop's own gain there is +0.00064, the three seeds pooled, fixed before the run; 2 times it is +0.00128" in learns["reason"]
    about = ceiling_branch(PASSING, _change(0.00127, 0.0002, 0.0024), OWN_GAIN, 2)        # clear of zero, a point under +0.00128
    assert about["name"] == AS_THE_LOOP and "longer proofs help about as the loop's own do" in about["reason"] and "which this cannot tell" in about["reason"]
    assert ceiling_branch(PASSING, _change(0.0003, 0.0001, 0.0005), OWN_GAIN, 2)["name"] == AS_THE_LOOP      # however small: an interval clear of zero is read, not sent for seeds
    through = ceiling_branch(PASSING, _change(0.00064, -0.0003, 0.0016), OWN_GAIN, 2)     # the interval holds zero, the point not above the loop's own gain
    assert through["name"] == NOT_SHOWN and "not shown to be learnable at this size" in through["reason"] and "reach stays in the episode" in through["reason"]
    seeds = ceiling_branch(PASSING, _change(0.00065, -0.0003, 0.0016), OWN_GAIN, 2)       # ... and above it: one seed does not conclude
    assert seeds["name"] == TWO_MORE_SEEDS and "two more seeds are run before anything is concluded" in seeds["reason"] and "+0.00064" in seeds["reason"]
    assert ceiling_branch(PASSING, _change(0.001, 0.0, 0.002), OWN_GAIN, 2)["name"] == TWO_MORE_SEEDS        # an interval that ends at zero holds it
    assert ceiling_branch(PASSING, _change(-0.0002, -0.0012, 0.0008), OWN_GAIN, 2)["name"] == NOT_SHOWN      # a point below zero, an interval through it
    below = ceiling_branch(PASSING, _change(-0.004, -0.007, -0.001), OWN_GAIN, 2)         # the case the read did not name
    assert below["name"] == FEWER and "fewer of these problems per attempt than the base" in below["reason"] and "named no branch" in below["reason"]
    empty = ceiling_branch(PASSING, {"problems": 0, "mean": None, "low": None, "high": None}, OWN_GAIN, 2)
    assert empty["name"] == NOT_READ and "could not be computed" in empty["reason"]
    assert {LEARNS, AS_THE_LOOP, NOT_SHOWN, TWO_MORE_SEEDS, FEWER, INCONCLUSIVE, NOT_READ} == set(BRANCHES)


# ---------------------------------------------------------------------------------- the report, hand-made rows
GROUPS = [{"problem_id": f"g{lines}{letter}", "group": "goal"} for lines in ("1", "2", "4", "8") for letter in "abc"] + \
         [{"problem_id": f"{rung}{index}", "group": rung} for rung in ("below", "in", "above") for index in (1, 2)] + [{"problem_id": "nowhere", "group": None}]
LENGTHS = {f"g{lines}{letter}": {"length_group": group} for lines, group in (("1", "1"), ("2", "2-3"), ("4", "4-7"), ("8", "8+")) for letter in "abc"}
GOAL_IDS = [row["problem_id"] for row in GROUPS if row["group"] == "goal"]
RUNG_IDS = [row["problem_id"] for row in GROUPS if row["group"] in ("below", "in", "above")]
# Successes by problem in (the 32 attempts, the 61 more), in GOAL_IDS' order: 1 line, 2-3 lines, 4-7 lines, 8 or more.
ON_G = {"base": ([2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0], [3, 3, 3, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
        "full": ([4, 4, 4, 2, 2, 2, 2, 1, 1, 1, 0, 0], [6, 6, 6, 2, 2, 2, 3, 2, 1, 1, 1, 0]),
        "small": ([3, 3, 3, 2, 2, 2, 1, 0, 0, 0, 0, 0], [3, 3, 3, 2, 2, 2, 1, 1, 0, 0, 0, 0]),
        "loop": ([4, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0, 0], [5, 5, 5, 3, 3, 3, 1, 1, 0, 0, 0, 1])}
ON_THE_RUNGS = {"base": [1, 1, 2, 3, 6, 7], "full": [1, 2, 3, 3, 6, 6], "small": [1, 1, 3, 3, 7, 7], "loop": [2, 2, 3, 4, 7, 7]}
PREPARE = {"seed": 0, "stand_in_engine": False, "attempts_a_goal_problem": 93, "rung_episodes": 8, "stored_runs": {"base": "ladder_l2_seed0", "loop": "ladder_l2_t010_seed0"},
           "loop_arm": "t010", "loop_target_rate": 0.1, "sampling_seeds": {"rungs": 1002, "reach": 1001, "more": 1020}, "stored_measurements": {},
           "goal_samplings": [{"name": "reach", "episodes": 32, "sampling_seed": 1001}, {"name": "more", "episodes": 61, "sampling_seed": 1020}],
           "contradicted_side_setting": "audit", "checkpoints": checkpoint_steps([2000, 8000], 8), "doses": {}, "recipe": {"what": "the round's"},
           "training_file": "training.jsonl", "training_file_sha256": "0" * 64, "training_file_rows": 8000, "rows_trained_on": 8000, "order": "the file's",
           "longest_example_tokens": 900, "max_sequence_tokens": 2048}
LOSSES = {"row_losses": [1.5] * 4000 + [0.9] * 4000}


def _model(name, rows=None, rungs=None):
    first, more = ON_G[name]
    return {"rows": rows, "rungs": [_rung(problem_id, resolved) for problem_id, resolved in zip(RUNG_IDS, rungs or ON_THE_RUNGS[name])],
            "goal": [[_rung(problem_id, resolved, episodes=32) for problem_id, resolved in zip(GOAL_IDS, first)],
                     [_rung(problem_id, resolved, episodes=61) for problem_id, resolved in zip(GOAL_IDS, more)]],
            "verified_proof_lines": {"goal": {"1": 6, "2": 2, "5": 1, "9": 1}, "rungs": {"1": 20, "3": 4}},
            "distinct_attempts": {"goal": {"prompts": 12, "mean_share_distinct": 0.95}, "rungs": {"prompts": 6, "mean_share_distinct": 0.9}}}


def _report(models=("small", "full", "loop"), losses=LOSSES, prepare=PREPARE, **changed):
    given = {name: changed.get(name) or _model(name, rows={"small": 2000, "full": 8000}.get(name)) for name in models}
    return build_ceiling_report(prepare, {"stand_in_engine": False, "steps": 1000, "checkpoints": []}, losses, GROUPS, LENGTHS, _model("base"), given, SETTINGS, EVALUATION)


def test_the_goal_set_is_read_by_the_length_of_the_shortest_published_proof():
    sets = goal_sets(GOAL_IDS, LENGTHS)
    assert list(sets) == ["1", "2-3", "4-7", "8+", FOUR_PLUS, ALL] and sets[FOUR_PLUS] == ["g4a", "g4b", "g4c", "g8a", "g8b", "g8c"] and sets[ALL] == GOAL_IDS
    assert list(goal_sets(["g1a", "fixture_h3"], LENGTHS)) == ["1", "2-3", "4-7", "8+", "not_known", FOUR_PLUS, ALL]     # a problem the lengths do not hold
    summed = attempts_on_g(_model("base")["goal"])
    assert [(row["resolved"], row["episodes"]) for row in summed][:4] == [(5, 93), (5, 93), (5, 93), (2, 93)] and attempts_on_g([_model("base")["goal"][0]])[0]["episodes"] == 32
    with pytest.raises(ValueError, match="do not hold the same problems"):
        attempts_on_g([_model("base")["goal"][0], _model("base")["goal"][1][:-1]])
    # Two sides with different numbers of attempts on a problem are never compared.
    with pytest.raises(ValueError, match="the comparison needs the same number"):
        per_attempt(attempts_on_g(_model("full")["goal"]), attempts_on_g([_model("base")["goal"][0]]), GOAL_IDS, 100, 0)
    assert verified_lines({"1": 6, "2": 2, "5": 1, "9": 1}) == {"proofs": 10, "median": 1.0, "mean": 2.4, "share_with_8_lines_or_more": 0.1, "longest": 9,
                                                                 "share_with_4_lines_or_more": 0.2}
    assert verified_lines({})["proofs"] == 0 and verified_lines({})["share_with_4_lines_or_more"] is None


def test_the_primary_is_successes_per_attempt_on_the_problems_of_four_lines_or_more_paired_by_problem():
    report = _report()
    primary = report["primary"]
    # Full: 5, 3, 2 and 2, 1, 0 of 93 on the six problems; the base: 2, 1, 0 and 0, 0, 0. Ten more successes in 558 attempts.
    assert primary["problems"] == 6 and primary["mean"] == round(10 / 558, 5) and (primary["successes"], primary["successes_of_the_base"], primary["attempts_each"]) == (13, 3, 558)
    assert (primary["per_1000"], primary["per_1000_of_the_base"]) == (23.3, 5.38) and primary["low"] > 0 and primary["model"] == "the 8,000-proof model"
    beside = report["beside_the_primary"]
    assert beside["mean"] == round(2 / 558, 5) and (beside["successes"], beside["successes_of_the_base"]) == (5, 3) and beside["model"] == "the three-round model at t = 1/10"
    assert report["branch"]["name"] == LEARNS and report["inconclusive"] is False and report["can_this_run_see_a_win"]["all_pass"] is True
    # The branch is read against ONE fixed number. This seed's three-round model stands beside as information and decides nothing.
    assert {key: report["the_loops_own_gain"][key] for key in ("per_attempt", "times", "times_it")} == {"per_attempt": 0.00064, "times": 2, "times_it": 0.00128}
    assert "deciding nothing" in beside["what"] and "+0.00064" in report["branch"]["reason"] and "+0.00358" not in report["branch"]["reason"]
    assert _report(loop=_model("full"))["branch"] == report["branch"] == _report(models=("small", "full"))["branch"]
    secondary = report["secondary"]
    by_length = secondary["by_length_group"]
    assert by_length["problems"] == {"1": 3, "2-3": 3, "4-7": 3, "8+": 3, FOUR_PLUS: 6, ALL: 12} and list(by_length["models"]) == ["full", "small", "loop"]
    assert by_length["models"]["full"]["1"]["mean"] == round(5 / 93, 5) and by_length["models"]["full"]["8+"]["mean"] == round(3 / 279, 5)
    assert by_length["models"]["small"][FOUR_PLUS]["mean"] == 0.0 and by_length["models"]["small"]["model"] == "the 2,000-proof model"       # the smaller dose, the same quantity
    assert by_length["models"]["loop"]["4-7"]["mean"] == round(1 / 279, 5) and by_length["models"]["loop"]["8+"]["successes"] == 1
    # Problems solved at 93 attempts, gained against lost: full solves every 4-7 problem and two of 8 or more; the base two of 4-7.
    solved_ = secondary["goal_problems_solved"]["models"]
    assert (solved_["full"][FOUR_PLUS]["resolved_after"], solved_["full"][FOUR_PLUS]["resolved_before"], solved_["full"][FOUR_PLUS]["gained"], solved_["full"][FOUR_PLUS]["lost"]) == (5, 2, 3, 0)
    assert solved_["full"]["8+"]["gained_problems"] == ["g8a", "g8b"] and solved_["loop"]["8+"]["gained"] == 1 and solved_["small"][ALL]["gained"] == 0
    assert solved(attempts_on_g(_model("full")["goal"]), attempts_on_g(_model("base")["goal"]), ["g4c"])["gained"] == 1
    assert secondary["the_three_rungs"]["models"]["full"]["above"]["mean"] == -0.0625 and secondary["the_three_rungs"]["models"]["loop"]["below"]["mean"] == 0.125
    assert secondary["verified_proof_lines"]["full"]["goal"]["share_with_4_lines_or_more"] == 0.2 and set(secondary["verified_proof_lines"]) == {"what", "base", "full", "small", "loop"}
    assert secondary["distinct_attempts"]["base"]["rungs"]["mean_share_distinct"] == 0.9
    assert report["models"]["base"].startswith("the base model, from stored attempts") and report["heldout"]["goal_set_by_length_group"][FOUR_PLUS] == 6


def test_the_report_prints_the_read_in_the_specs_order_and_every_line_says_ceiling():
    report = _report()
    lines = report["lines"]
    assert all(line.startswith("ceiling: ") for line in lines) and "labelled diagnostic" in report["headline"] and report["diagnostic"] == "ceiling"
    order = ["THE CEILING, A LABELLED DIAGNOSTIC", "PRIMARY.", "the loop's own gain there", "beside it,", "SECONDARY, the same by the length", "SECONDARY, goal problems solved",
             "SECONDARY, the three rungs", "SECONDARY, the lines of the proofs", "SECONDARY, the share of distinct attempts", "CHECK 1", "CHECK 2", "CHECK 3", "BRANCH: "]
    places = [next(index for index, line in enumerate(lines) if marker in line) for marker in order]
    assert places == sorted(places) and len(set(places)) == len(places)
    assert "+0.01792 [+" in lines[places[1]] and "13 successes against 3 in 558 attempts each" in lines[places[1]] and "23.3 against 5.38 per 1,000" in lines[places[1]]
    assert "the ONE number the branch is read against: +0.00064 per attempt" in lines[places[2]] and "2 times it is +0.00128" in lines[places[2]]
    assert "for information and deciding nothing, this seed's three-round model at t = 1/10 minus the base, from the stored rows" in lines[places[3]]
    assert [line.count("PASS") for line in lines[places[9]:places[12]]] == [1, 1, 1] and lines[places[12]].startswith(f"ceiling: BRANCH: {LEARNS}.")
    assert "no model trained this way is kept" in lines[0] and "8,000 published proofs" in lines[0]


def test_a_failed_check_makes_the_report_inconclusive_and_it_says_nothing_else_about_the_model():
    broken = _report(full=_model("full", rows=8000, rungs=[1, 2, 3, 3, 2, 2]))        # above the band: 0.25 against the base's 0.8125
    assert broken["branch"]["name"] == INCONCLUSIVE and broken["inconclusive"] is True
    assert broken["can_this_run_see_a_win"]["failed"] == ["training_has_not_broken_it"] and broken["can_this_run_see_a_win"]["training_has_not_broken_it"]["pass_rate"] == 0.25
    assert broken["primary"] is None and broken["beside_the_primary"] is None and broken["secondary"] is None
    assert broken["measured_and_not_read"]["primary"]["mean"] == round(10 / 558, 5) and "say nothing about the model" in broken["measured_and_not_read"]["why"]
    text = "\n".join(broken["lines"])
    assert "PRIMARY" not in text and "SECONDARY" not in text and "BRANCH" not in text and "0.01792" not in text and "0.01792" not in broken["headline"]
    assert [line.split(":")[1].strip().split(",")[0] for line in broken["lines"][1:4]] == ["CHECK 1", "CHECK 2", "CHECK 3"]
    assert broken["lines"][3].endswith("FAIL") and broken["lines"][4].startswith("ceiling: INCONCLUSIVE. ") and broken["headline"].split(": ")[1].startswith("INCONCLUSIVE")
    # The training loss that did not fall, and a model that reaches the token cap too often, are the other two.
    assert _report(losses={"row_losses": [1.0] * 8000})["can_this_run_see_a_win"]["failed"] == ["the_training_took"]
    capped = _model("full", rows=8000)
    capped["rungs"][0]["attempts_capped"] = 3                                         # 3 of 48 attempts
    assert _report(full=capped)["can_this_run_see_a_win"]["failed"] == ["the_model_still_writes_proofs"]
    # The smaller dose is beside the checks for information and decides nothing.
    small = _model("small", rows=2000, rungs=[1, 1, 3, 3, 1, 1])
    fine = _report(small=small)
    assert fine["inconclusive"] is False and fine["can_this_run_see_a_win"]["for_information_the_smaller_dose"]["training_has_not_broken_it"]["passes"] is False


def test_a_run_without_l2s_stored_attempts_still_reads_its_branch_and_a_set_lean_did_not_answer_is_not_to_be_read():
    one_sampling = {name: {**_model(name, rows=rows), "goal": [_model(name)["goal"][0]]} for name, rows in (("small", 12), ("full", 24))}
    base = {**_model("base"), "goal": [_model("base")["goal"][0]]}
    prepare = {**PREPARE, "stored_runs": None, "loop_arm": None, "loop_target_rate": None, "attempts_a_goal_problem": 32, "goal_samplings": PREPARE["goal_samplings"][:1]}
    report = build_ceiling_report(prepare, {"stand_in_engine": True}, {"row_losses": [2.0] * 12 + [1.0] * 12}, GROUPS, LENGTHS, base, one_sampling, SETTINGS, EVALUATION)
    # Four of the six problems gain one success in 32: clear of zero, and far above twice the fixed number. No three-round model is needed to read it.
    assert report["branch"]["name"] == LEARNS and report["beside_the_primary"] is None and report["stand_in_engine"] is True
    assert report["primary"]["attempts_each"] == 6 * 32 and list(report["secondary"]["by_length_group"]["models"]) == ["full", "small"]
    assert any("L2's stored attempts were not read (a smoke run)" in line for line in report["lines"]) and any("no stored three-round model was read" in line for line in report["lines"])
    silent = _model("full", rows=8000)
    silent["goal"][1][0]["attempts_without_an_answer"] = 30                           # of 12 x 61 attempts: over 2% with no verdict
    unread = _report(full=silent)
    assert unread["ok"] is False and unread["not_to_be_read"] == ["goal_full_2"] and "NOT TO BE READ" in unread["lines"][-1] and _report()["ok"] is True
    assert lean_did_not_answer(silent["goal"][1]) and not lean_did_not_answer(silent["goal"][0]) and not lean_did_not_answer([])
    assert unread["label"] == ceiling.LABEL and "no model trained this way is kept" in ceiling.LABEL
