"""L4p, how much more pretraining, the dose of a second pass: its three sets along the ONE hash order L4t's rehearsal draw
uses (`matched_1262`'s rows, which are the published rows of L4t2's `reward_rehearse`, first; then `rehearse_only`'s, then
`second_half`'s, inside `second_full`'s; every row of the file once, no published proof's text), the read of the rounds'
hard rows beside the same published rows, the primary's three branches and its edge at zero, the overfitting reads, and
the report on hand-made rows with a known answer, with all three models measured and with `matched_1262` alone. Spec: docs/spec/ladder-loop.spec.md, "L4p: how much more pretraining? The dose of a second
pass". Pure: no model, no Lean. The stage end to end is `test_ladder_l4_dose_stage.py`."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l3d1 import EVALUATION, GOAL_IDS, GROUPS, LENGTHS  # noqa: E402 - Step 1's hand-made world: 12 goal problems, 3 a length group
from test_ladder_l4 import CHANGED, LOOP_HALF, ON_THE_RUNGS, _made  # noqa: E402 - L4's hand-made models
from test_ladder_l4_rows import FALLING, SAMPLINGS, _kept  # noqa: E402
from test_ladder_l4_rows2 import PRETRAINING, REHEARSE, REWARD_ROWS, TWIN, _draw_order, _solved  # noqa: E402 - L4t2's hand-made sets: 40 rows the start model was pretrained on, `rehearse`'s 20

from rlvr_lean.domain.ladder_round import l4_dose  # noqa: E402
from rlvr_lean.domain.ladder_round.assembly import training_order  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ, PRETRAIN, goal_set_again, half_of  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_dose import (  # noqa: E402
    BRANCHES,
    DOSES,
    FIRST_SEVENTH,
    HARD_ROWS_ADD,
    MATCHED_BRANCHES,
    MODELS,
    MORE_COSTS,
    MORE_HELPS,
    STORED,
    THEY_COST,
    dose_branch,
    dose_sets,
    matched_branch,
    overfitting,
    rows_of,
)
from rlvr_lean.domain.ladder_round.l4_rows import SIGN_TEST_LEVEL, rehearsal_rows, rehearse_set  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows2 import ADDS, HOLDS, KEY, NOT_NARROWER, NOT_SEPARATED, TEXT_FIELDS, refuse_text, rehearse_only_set, reward_rehearse_set, three_reads  # noqa: E402
from rlvr_lean.domain.ladder_round.l4b import breadth_beside  # noqa: E402
from rlvr_lean.domain.ladder_round.read import paired_change  # noqa: E402
from rlvr_lean.domain.ladder_round.rounds import sign_test  # noqa: E402
from rlvr_lean.reporting.ladder_ceiling import per_attempt, solved  # noqa: E402
from rlvr_lean.reporting.ladder_l4 import LABEL, SAY  # noqa: E402
from rlvr_lean.reporting.ladder_l4_dose import CANNOT_SAY, IS, READ_AGAINST, STEPS, build_dose_report, step_name  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
SETTINGS = CONFIG["ladder_loop"]["l4"]
SEED = 0
MATCHED, HALF, FULL = "matched_1262", "second_half", "second_full"


def test_the_names_and_the_settings_are_the_specs():
    assert MODELS == (MATCHED, HALF, FULL) and DOSES == (MATCHED, "rehearse_only", HALF, FULL)     # `matched_1262` trained FIRST; the dose along one order, the smallest first
    assert STORED == ("rehearse_only", "reward_rehearse")                                          # L4t2's two, read from the rows its run stored
    assert (HARD_ROWS_ADD, NOT_SEPARATED, THEY_COST) == ("THE ROUNDS' HARD ROWS ADD BESIDE THE SAME PUBLISHED ROWS", "NOT SEPARATED AT THIS SIZE", "THEY COST")
    assert MATCHED_BRANCHES == (HARD_ROWS_ADD, NOT_SEPARATED, THEY_COST, INCONCLUSIVE, NOT_READ)
    assert (MORE_HELPS, FIRST_SEVENTH, MORE_COSTS) == ("MORE OF THE SECOND PASS HELPS", "THE FIRST SEVENTH GAVE WHAT A SECOND PASS GIVES", "MORE COSTS")
    assert BRANCHES == (MORE_HELPS, FIRST_SEVENTH, MORE_COSTS, INCONCLUSIVE, NOT_READ)
    assert SETTINGS["dose"] == {"start": "pre_r64", "models": list(MODELS)} and SETTINGS["dose"]["start"] == SETTINGS["rows2"]["start"] and SIGN_TEST_LEVEL == 0.05
    # `second_full` is every row of the file, `second_half` half of them: 24,866 and 12,433, the section's two figures; half of an odd file is rounded down.
    assert (rows_of(FULL, 24866), rows_of(HALF, 24866)) == (24866, 12433) and (rows_of(FULL, 12), rows_of(HALF, 12), rows_of(HALF, 13), rows_of(HALF, 1)) == (12, 6, 6, 0)
    for other in ("rehearse_only", MATCHED):                                                       # `matched_1262`'s size is not a share of the file: it is `reward_rehearse`'s published rows
        with pytest.raises(ValueError, match=f"'{other}' is not a dose by a share of the file: it is one of .'second_half', 'second_full'."):
            rows_of(other, 12)
    assert set(IS) == set(MODELS) and len(CANNOT_SAY) == 3 and "MORE DATA" in CANNOT_SAY[0] and "decides no arm by itself" in CANNOT_SAY[2]
    assert STEPS == (("rehearse_only", MATCHED), (HALF, "rehearse_only"), (FULL, HALF)) and step_name(FULL, HALF) == "second_full_minus_second_half"
    assert READ_AGAINST == {MATCHED: ("reward_rehearse", "rehearse_only"), HALF: ("rehearse_only",), FULL: ("rehearse_only",)}


# ------------------------------------------------------------------------------------------------- the sets
REHEARSE_ONLY = rehearse_only_set(REHEARSE)                # L4t2's stored set: the ten rehearsal rows of L4t's `rehearse`, in that set's order
REWARD_REHEARSE = reward_rehearse_set(REWARD_ROWS, REHEARSE, PRETRAINING, SEED)      # L4t2's other stored set: six rows of the rounds and four published rows, in one order
placed = lambda row: (row["id"], row["problem_id"], row["row_of_the_file"])      # noqa: E731


def _sets(rehearse_only, pretraining, seed, reward_rehearse=None):
    """`dose_sets`, with L4t2's `reward_rehearse` set as it stored it unless another is given."""
    return dose_sets(rehearse_only, REWARD_REHEARSE if reward_rehearse is None else reward_rehearse, pretraining, seed)


def test_the_two_sets_are_nested_along_the_one_hash_order_and_second_full_is_every_row_of_the_file_once():
    sets = _sets(REHEARSE_ONLY, PRETRAINING, SEED)
    half, full, matched = sets[HALF], sets[FULL], sets[MATCHED]
    assert list(sets) == list(MODELS) and (len(matched), len(REHEARSE_ONLY), len(half), len(full)) == (4, 10, 20, 40)
    order = _draw_order(full)                                                                      # the hash order L4t's rehearsal draw uses: the smallest hash first
    assert order == rehearsal_rows(PRETRAINING, 40, SEED) and _draw_order(half) == rehearsal_rows(PRETRAINING, 20, SEED)
    # `rehearse_only`'s rows are the FIRST of `second_half`'s, which are the first of `second_full`'s: by id and by place in the file.
    assert list(map(placed, _draw_order(REHEARSE_ONLY))) == list(map(placed, order[:10])) == list(map(placed, _draw_order(half)[:10]))
    assert list(map(placed, _draw_order(half))) == list(map(placed, order[:20])) and {row["id"] for row in half} != {row["id"] for row in order[20:]}
    assert {row["id"] for row in REHEARSE_ONLY} < {row["id"] for row in half} < {row["id"] for row in full}
    # `second_full` holds EVERY row of the pretraining file exactly once, each naming the problem at its place.
    assert sorted(row["row_of_the_file"] for row in full) == list(range(40)) and len({row["id"] for row in full}) == 40
    assert all(PRETRAINING[row["row_of_the_file"]]["problem_id"] == row["problem_id"] and row["id"] == f"{row['problem_id']}#pretraining" for row in full)
    # Each is TRAINED in the arm's content-hash order, as `rehearse_only` was: a smaller dose is the larger one's order with rows left out. It is not the draw's order.
    for own in (half, full):
        assert [row["id"] for row in own] == [row["id"] for row in training_order(own, SEED)] != [row["id"] for row in _draw_order(own)]
    assert [row["id"] for row in REHEARSE_ONLY] == [row["id"] for row in half if row["id"] in {given["id"] for given in REHEARSE_ONLY}]
    assert [row["id"] for row in half] == [row["id"] for row in full if row["id"] in {given["id"] for given in half}]
    # No row carries a published proof's text, and none has a k.
    assert all(set(row) == {"problem_id", "kind", "proof_lines", "lines", "id", "side", "origin", "k", "row_of_the_file"} and row["origin"] == "pretraining" and row["k"] is None
               for row in (*half, *full))
    assert not any(field in row for row in (*half, *full) for field in TEXT_FIELDS) and refuse_text(full, "the training set of `second_full`") is None
    with pytest.raises(ValueError, match="1 rehearsal rows of the training set of `second_full` carry a text field"):
        refuse_text([{**full[0], "proof": "text"}, *full[1:]], "the training set of `second_full`")
    # A file of an odd number of rows: half of it, rounded down; `rehearse_only` as large as `second_half` is its whole.
    odd = _sets(training_order(rehearsal_rows(PRETRAINING[:21], 10, SEED), SEED), PRETRAINING[:21], SEED, training_order(rehearsal_rows(PRETRAINING[:21], 3, SEED), SEED))
    assert (len(odd[MATCHED]), len(odd[HALF]), len(odd[FULL])) == (3, 10, 21)


def test_matched_1262_is_exactly_the_published_rows_of_reward_rehearse_the_first_of_the_order_and_nothing_else():
    sets = _sets(REHEARSE_ONLY, PRETRAINING, SEED)
    matched, published = sets[MATCHED], [row for row in REWARD_REHEARSE if row["origin"] == "pretraining"]
    order = rehearsal_rows(PRETRAINING, 40, SEED)
    # AS MANY ROWS AS `reward_rehearse` HAS PUBLISHED ROWS (four of its ten; its six rows of the rounds bring none), and they ARE those rows: by id and by place in the file.
    assert (len(matched), len(published), len(REWARD_REHEARSE)) == (4, 4, 10) and sorted(map(placed, matched)) == sorted(map(placed, published))
    assert all(row["origin"] == "pretraining" and row["k"] is None for row in matched) and not {row["id"] for row in matched} & {row["id"] for row in REWARD_ROWS}
    # THE FIRST OF THE ONE ORDER: so they are the first of `rehearse_only`'s, of `second_half`'s and of `second_full`'s.
    assert list(map(placed, _draw_order(matched))) == list(map(placed, order[:4])) == list(map(placed, _draw_order(REHEARSE_ONLY)[:4])) == list(map(placed, _draw_order(sets[HALF])[:4]))
    assert {row["id"] for row in matched} < {row["id"] for row in REHEARSE_ONLY} and {row["id"] for row in matched} != {row["id"] for row in order[4:8]}
    # Trained in the content-hash order: `reward_rehearse`'s own order with its rounds' rows left out, none moved; and `rehearse_only`'s with rows left out.
    assert [row["id"] for row in matched] == [row["id"] for row in published] == [row["id"] for row in training_order(matched, SEED)]
    assert [row["id"] for row in matched] == [row["id"] for row in REHEARSE_ONLY if row["id"] in {given["id"] for given in matched}]
    assert not any(field in row for row in matched for field in TEXT_FIELDS) and refuse_text(matched, "the training set of `matched_1262`") is None
    # THE COUNT IS THAT SET'S, NOT A NUMBER: with two published rows it is two; with every one of `rehearse_only`'s it is `rehearse_only`.
    two = [*(row for row in REWARD_REHEARSE if row["origin"] != "pretraining"), *(row for row in published if row["id"] in {order[0]["id"], order[1]["id"]})]
    assert list(map(placed, _draw_order(_sets(REHEARSE_ONLY, PRETRAINING, SEED, training_order(two, SEED))[MATCHED]))) == list(map(placed, order[:2]))
    assert [row["id"] for row in _sets(REHEARSE_ONLY, PRETRAINING, SEED, REHEARSE_ONLY)[MATCHED]] == [row["id"] for row in REHEARSE_ONLY]
    # ---- REFUSED: no published row; more than `rehearse_only` holds (it is the smallest dose)
    with pytest.raises(ValueError, match="L4t2's `reward_rehearse` set holds 0 published rows among its 6 and `rehearse_only` holds 10: `matched_1262` is those published rows and "
                                         "nothing else, and the smallest dose"):
        _sets(REHEARSE_ONLY, PRETRAINING, SEED, [row for row in REWARD_REHEARSE if row["origin"] != "pretraining"])
    with pytest.raises(ValueError, match="L4t2's `reward_rehearse` set holds 11 published rows among its 11 and `rehearse_only` holds 10"):
        _sets(REHEARSE_ONLY, PRETRAINING, SEED, training_order(order[:11], SEED))
    # ---- ... its published rows are not the first of the order: another row of the file in the place of one; the right id at another place; a row missing with one beyond
    swapped = training_order([*(row for row in REWARD_REHEARSE if row["id"] != order[0]["id"]), order[7]], SEED)
    with pytest.raises(ValueError, match="L4t2's `reward_rehearse` set holds 4 published rows, and they are not the first 4 of the hash order among the 40 rows the start model was "
                                         f"pretrained on .first that differs: {order[0]['id']}|{order[7]['id']}.: `matched_1262` would not be the control of that set"):
        _sets(REHEARSE_ONLY, PRETRAINING, SEED, swapped)
    moved = [{**row, "row_of_the_file": (row["row_of_the_file"] + 1) % 40} if row["id"] == order[0]["id"] else row for row in REWARD_REHEARSE]
    with pytest.raises(ValueError, match="they are not the first 4 of the hash order among the 40 rows"):
        _sets(REHEARSE_ONLY, PRETRAINING, SEED, moved)
    another_seed = reward_rehearse_set(REWARD_ROWS, rehearse_set(TWIN, rehearsal_rows(PRETRAINING, 10, SEED + 1), SEED), PRETRAINING, SEED + 1)
    with pytest.raises(ValueError, match="`matched_1262` would not be the control of that set"):
        _sets(REHEARSE_ONLY, PRETRAINING, SEED, another_seed)
    # ---- ... or they do not stand in the content-hash order (the control is that set with its rounds' rows left out, none moved)
    with pytest.raises(ValueError, match="the published rows of L4t2's `reward_rehearse` set do not stand in the content-hash order of this seed: `matched_1262` is that set with its "
                                         "rounds' rows left out and none moved"):
        _sets(REHEARSE_ONLY, PRETRAINING, SEED, list(reversed(REWARD_REHEARSE)))


def test_matched_1262_is_held_to_the_first_of_the_order_were_the_draw_no_longer_nested(monkeypatch):
    nested = l4_dose.rehearsal_rows
    monkeypatch.setattr(l4_dose, "rehearsal_rows", lambda pretraining, count, seed: nested(pretraining, 10, seed)[10 - count:] if count < 10 else nested(pretraining, count, seed))
    with pytest.raises(ValueError, match="the 4 rows drawn for `matched_1262` are not the first 4 of the hash order `second_full`'s 40 stand in"):
        _sets(REHEARSE_ONLY, PRETRAINING, SEED)
    monkeypatch.setattr(l4_dose, "rehearsal_rows", nested)
    assert len(_sets(REHEARSE_ONLY, PRETRAINING, SEED)[MATCHED]) == 4


def test_the_dose_is_refused_when_it_would_not_be_along_one_order():
    another_seed = rehearse_only_set(rehearse_set(TWIN, rehearsal_rows(PRETRAINING, 10, SEED + 1), SEED))
    with pytest.raises(ValueError, match="L4t2's `rehearse_only` set holds 10 rows, and they are not the first 10 of the hash order among the 40 rows the start model was "
                                         "pretrained on .first that differs: q.*#pretraining.: that set was not drawn from this file with this seed, and the dose would not be "
                                         "along one order"):
        _sets(another_seed, PRETRAINING, SEED)
    with pytest.raises(ValueError, match="they are not the first 10 of the hash order"):
        _sets(REHEARSE_ONLY, list(reversed(PRETRAINING)), SEED)                                 # the same problems at other places of the file
    # One row of it is a row of `second_half` that is NOT among the first ten: still not the first of the order.
    order = rehearsal_rows(PRETRAINING, 40, SEED)
    later = training_order([*(row for row in REHEARSE_ONLY if row["id"] != order[0]["id"]), order[15]], SEED)
    with pytest.raises(ValueError, match="they are not the first 10 of the hash order"):
        _sets(later, PRETRAINING, SEED)
    with pytest.raises(ValueError, match="they are not the first 9 of the hash order"):
        _sets([row for row in REHEARSE_ONLY if row["id"] != order[0]["id"]], PRETRAINING, SEED)                # a row of the draw is missing
    # A set larger than `second_half` cannot be its first rows; one that holds a row of a round, or no row, is not a dose of the pretraining file.
    with pytest.raises(ValueError, match="L4t2's `rehearse_only` set holds 10 rows and `second_half` 6 .half of the 12 the start model was pretrained on.: `rehearse_only`'s rows "
                                         "are the first of `second_half`'s, and a larger set cannot be"):
        _sets(REHEARSE_ONLY, PRETRAINING[:12], SEED)
    with pytest.raises(ValueError, match="L4t2's `rehearse_only` set holds 11 rows and 1 of them are not rehearsal rows .first: p0#round_r1_b1#statement#0.: the dose is of rows of "
                                         "the pretraining file and of no other"):
        _sets([*REHEARSE_ONLY, TWIN[0]], PRETRAINING, SEED)
    with pytest.raises(ValueError, match="L4t2's `rehearse_only` set holds 0 rows and 0 of them are not rehearsal rows: the dose is of rows of the pretraining file"):
        _sets([], PRETRAINING, SEED)
    # Its rows must stand in the content-hash order the two models are trained in.
    with pytest.raises(ValueError, match="L4t2's `rehearse_only` set does not stand in the content-hash order of this seed: the models here are trained in that order"):
        _sets(list(reversed(REHEARSE_ONLY)), PRETRAINING, SEED)
    with pytest.raises(ValueError, match="1 problems stand twice among the rows `pre` was trained on"):
        _sets(REHEARSE_ONLY, [*PRETRAINING, PRETRAINING[0]], SEED)


def test_both_nestings_and_every_row_once_are_held_were_the_draw_no_longer_what_it_is(monkeypatch):
    """HELD, not assumed: a draw that gave OTHER rows for the half, or not every row for the whole, is refused."""
    nested = l4_dose.rehearsal_rows
    # ... the half is the LAST twenty of the order
    monkeypatch.setattr(l4_dose, "rehearsal_rows", lambda pretraining, count, seed: nested(pretraining, 40, seed)[40 - count:] if count < 40 else nested(pretraining, count, seed))
    with pytest.raises(ValueError, match="the 20 rows drawn for `second_half` are not the first 20 of the hash order `second_full`'s 40 stand in"):
        _sets(REHEARSE_ONLY, PRETRAINING, SEED)
    # ... the whole draw leaves a row of the file out, or holds one twice; or its 40 ids stand at 39 places of the file; or its 40 places hold 39 ids
    for broken, said in ((lambda rows: rows[:-1], "39 rows .39 ids, 39 places of the file."), (lambda rows: [*rows[:-1], rows[0]], "40 rows .39 ids, 39 places of the file."),
                         (lambda rows: [*rows[:-1], {**rows[-1], "row_of_the_file": rows[0]["row_of_the_file"]}], "40 rows .40 ids, 39 places of the file."),
                         (lambda rows: [*rows[:-1], {**rows[-1], "id": rows[0]["id"]}], "40 rows .39 ids, 40 places of the file.")):
        monkeypatch.setattr(l4_dose, "rehearsal_rows", lambda pretraining, count, seed, broken=broken: broken(nested(pretraining, count, seed)) if count == 40 else nested(pretraining, count, seed))
        with pytest.raises(ValueError, match=f"`second_full` holds {said} and the start model was pretrained on 40: it is every row of the pretraining file exactly once"):
            _sets(REHEARSE_ONLY, PRETRAINING, SEED)
    # ... the whole draw is in another order: `rehearse_only`'s rows are no longer its first
    monkeypatch.setattr(l4_dose, "rehearsal_rows", lambda pretraining, count, seed: list(reversed(nested(pretraining, 40, seed)))[:count])
    with pytest.raises(ValueError, match="L4t2's `rehearse_only` set holds 10 rows, and they are not the first 10 of the hash order"):
        _sets(REHEARSE_ONLY, PRETRAINING, SEED)
    monkeypatch.setattr(l4_dose, "rehearsal_rows", nested)
    assert len(_sets(REHEARSE_ONLY, PRETRAINING, SEED)[FULL]) == 40


# ------------------------------------------------------------------------------------------------ the branch
ABOVE, THROUGH, BELOW = _kept(0.0114, 0.0026, 0.0203), _kept(-0.0068, -0.01569, 0.00233), _kept(-0.01, -0.02, -0.001)
NOT_MEASURED = {"problems": 0, "mean": None, "low": None, "high": None}


def test_the_primary_has_three_branches_and_an_interval_that_ends_at_zero_holds_zero():
    helps = dose_branch([], ABOVE, "pre_r64")
    assert helps["name"] == MORE_HELPS and helps["failed_checks"] == [] and "resolves" not in helps
    assert helps["reason"] == ("all of G, `second_full` minus `rehearse_only`: the interval is above zero (+0.01140 [+0.00260, +0.02030]): more of the second pass helps, and a "
                               "third pass is specified next")
    holds = dose_branch([], THROUGH, "pre_r64")
    assert holds["name"] == FIRST_SEVENTH and holds["resolves"] == 0.00901 and "the interval holds zero (-0.00680 [-0.01569, +0.00233]): the first seventh gave what a second pass gives" in holds["reason"]
    assert "The gain is not the amount of training; the recipe is one pass and a short second one. This difference resolves 0.00901 per attempt (half the width of its interval)" in holds["reason"]
    costs = dose_branch([], BELOW, "pre_r64")
    assert costs["name"] == MORE_COSTS and "the interval is below zero (-0.01000 [-0.02000, -0.00100]): more of the second pass costs" in costs["reason"] and "`pre_r64`" in costs["reason"]
    # THE EDGE: an interval that ENDS at zero holds zero, from either side; a hair clear of it does not.
    for primary, name in ((_kept(0.004, 0.0, 0.011), FIRST_SEVENTH), (_kept(0.004, 0.00001, 0.011), MORE_HELPS), (_kept(-0.004, -0.011, 0.0), FIRST_SEVENTH),
                          (_kept(-0.004, -0.011, -0.00001), MORE_COSTS), (_kept(0.0, 0.0, 0.0), FIRST_SEVENTH), (_kept(0.003, -0.001, 0.002), FIRST_SEVENTH)):
        assert dose_branch([], primary, "pre_r64")["name"] == name, primary
    # The branch is read on the INTERVAL, never on the point: a point above zero with an interval through it holds zero.
    assert dose_branch([], _kept(0.009, -0.0001, 0.02), "pre_r64")["name"] == FIRST_SEVENTH and dose_branch([], _kept(-0.009, -0.02, 0.0001), "pre_r64")["name"] == FIRST_SEVENTH
    # A check of `second_full` failed: INCONCLUSIVE, whatever was measured. `rehearse_only` INCONCLUSIVE where it was measured, or nothing measured: NOT READ.
    for primary in (ABOVE, THROUGH, BELOW, NOT_MEASURED):
        failed = dose_branch(["its_training_ran", "lean_answered"], primary, "pre_r64", ["no_barred_row"])
        assert (failed["name"], failed["failed_checks"]) == (INCONCLUSIVE, ["its_training_ran", "lean_answered"]) and "`second_full` could not have been read (its_training_ran, lean_answered)" in failed["reason"]
        unread = dose_branch([], primary, "pre_r64", ["lean_answered"])
        assert (unread["name"], unread["not_read_against"], unread["failed_checks"]) == (NOT_READ, "rehearse_only", []) and "is INCONCLUSIVE in the report of the run that measured it (lean_answered)" in unread["reason"]
    nothing = dose_branch([], NOT_MEASURED, "pre_r64")
    assert nothing["name"] == NOT_READ and "not_read_against" not in nothing and "no goal problem was measured on both sides" in nothing["reason"]
    assert all(dose_branch(failed, primary, "pre_r64", of)["name"] in BRANCHES for failed in ([], ["x"]) for primary in (ABOVE, THROUGH, BELOW, NOT_MEASURED) for of in ([], ["y"]))
    # `second_full` NOT MEASURED YET (the report of the job that trains and measures `matched_1262` alone): NOT READ, said as such, and never INCONCLUSIVE.
    for failed, of in (([], []), (["x"], ["y"])):
        waiting = dose_branch(failed, NOT_MEASURED, "pre_r64", of, measured=False)
        assert (waiting["name"], waiting["not_measured"], waiting["failed_checks"]) == (NOT_READ, "second_full", []) and "not_read_against" not in waiting
        assert waiting["reason"] == ("`second_full` is not measured yet: this is the report of the job that trains and measures `matched_1262` alone, and the dose's primary is "
                                     "read by the report of the whole stage")
    assert "not_measured" not in dose_branch([], ABOVE, "pre_r64") and dose_branch([], ABOVE, "pre_r64", measured=True)["name"] == MORE_HELPS


def test_the_rounds_hard_rows_have_three_outcomes_and_an_interval_that_ends_at_zero_holds_zero():
    adds = matched_branch([], _kept(0.005, 0.0012, 0.0091))
    assert adds["name"] == HARD_ROWS_ADD and adds["failed_checks"] == [] and "resolves" not in adds
    assert adds["reason"].startswith("on G' over the fresh attempts `reward_rehearse` minus `matched_1262` is above zero (+0.00500 [+0.00120, +0.00910]): the rounds' hard rows "
                                     "add beside the same published rows. It decides no arm by itself")
    holds = matched_branch([], _kept(0.003, -0.003, 0.009))
    assert (holds["name"], holds["resolves"]) == (NOT_SEPARATED, 0.006) and "holds zero (+0.00300 [-0.00300, +0.00900]): not separated at this size. This difference resolves 0.00600 per attempt (half the width of its interval)" in holds["reason"]
    costs = matched_branch([], _kept(-0.005, -0.0091, -0.0012))
    assert costs["name"] == THEY_COST and "is below zero (-0.00500 [-0.00910, -0.00120]): beside the same published rows the rounds' hard rows cost" in costs["reason"]
    # THE EDGE: an interval that ENDS at zero holds zero, from either side; the outcome is read on the interval, never on the point.
    for difference, name in ((_kept(0.004, 0.0, 0.011), NOT_SEPARATED), (_kept(0.004, 0.00001, 0.011), HARD_ROWS_ADD), (_kept(-0.004, -0.011, 0.0), NOT_SEPARATED),
                             (_kept(-0.004, -0.011, -0.00001), THEY_COST), (_kept(0.0, 0.0, 0.0), NOT_SEPARATED), (_kept(0.009, -0.0001, 0.02), NOT_SEPARATED),
                             (_kept(-0.009, -0.02, 0.0001), NOT_SEPARATED)):
        assert matched_branch([], difference)["name"] == name, difference
    # A check of `matched_1262` failed: INCONCLUSIVE. `reward_rehearse` INCONCLUSIVE where it was measured, or no fresh attempts (one sampling of G): NOT READ.
    for difference in (ABOVE, THROUGH, BELOW, NOT_MEASURED):
        failed = matched_branch(["its_training_ran"], difference, ["lean_answered"])
        assert (failed["name"], failed["failed_checks"]) == (INCONCLUSIVE, ["its_training_ran"]) and "`matched_1262` could not have been read (its_training_ran)" in failed["reason"]
        unread = matched_branch([], difference, ["lean_answered"])
        assert (unread["name"], unread["not_read_against"]) == (NOT_READ, "reward_rehearse") and "is INCONCLUSIVE in the report of the run that measured it (lean_answered)" in unread["reason"]
    nothing = matched_branch([], NOT_MEASURED)
    assert nothing["name"] == NOT_READ and "not_read_against" not in nothing and "G' holds no problem with fresh attempts on both sides" in nothing["reason"]
    assert all(matched_branch(failed, difference, of)["name"] in MATCHED_BRANCHES for failed in ([], ["x"]) for difference in (ABOVE, THROUGH, BELOW, NOT_MEASURED) for of in ([], ["y"]))


# -------------------------------------------------------------------------------------------- overfitting
def _change(mean, low, high, problems=2):
    return {"problems": problems, "mean": mean, "low": low, "high": high}


FLAT = {"below": _change(0.0, -0.05, 0.05), "in": _change(0.01, -0.02, 0.04), "above": _change(-0.02, -0.05, 0.01)}


def test_overfitting_is_narrower_by_l4bs_function_names_a_rung_wholly_below_zero_and_says_the_distinct_shares():
    # NARROWER: lost more than gained with the sign test UNDER 0.05. It is L4b's own, by its function, and the reverse of L4t2's read NOT NARROWER.
    assert sign_test(23, 39) > 0.05 > sign_test(23, 40)
    for found, narrower in ((_solved(23, 40), True), (_solved(23, 39), False), (_solved(40, 23), False), (_solved(30, 30), False), (_solved(0, 0), False), (_solved(0, 6), True), (_solved(0, 5), False)):
        read = overfitting(found, FLAT, 0.81, 0.82, "pre_r64")
        assert read["narrower"] is narrower is breadth_beside(found, None, None, "pre_r64")["narrower"] and (read["gained"], read["lost"], read["sign_test_p"]) == (
            found["gained"], found["lost"], found["sign_test_p"])
        assert read["narrower"] is (not three_reads(ABOVE, found, THROUGH, "pre_r64")[KEY[NOT_NARROWER]]["passes"])
        assert read["said"].startswith(f"{'NARROWER' if narrower else 'not narrower'} (gained {found['gained']}, lost {found['lost']}")
    assert overfitting(_solved(23, 40), FLAT, 0.81, 0.82, "pre_r64", level=0.01)["narrower"] is False                # the level is the spec's 0.05
    # THE RUNGS: one whose interval is WHOLLY below zero is named; one that ends at zero is not wholly below it; a point below zero is not enough.
    for rungs, named in ((FLAT, []), ({**FLAT, "in": _change(-0.2, -0.3, -0.00001)}, ["in"]), ({**FLAT, "in": _change(-0.2, -0.3, 0.0)}, []),
                         ({**FLAT, "below": _change(-0.1, -0.2, -0.05), "above": _change(-0.3, -0.4, -0.1)}, ["below", "above"]), ({**FLAT, "above": _change(-0.2, -0.5, 0.1)}, []),
                         ({**FLAT, "in": _change(0.2, 0.1, 0.3)}, []), ({**FLAT, "in": NOT_MEASURED}, [])):
        read = overfitting(_solved(30, 30), rungs, 0.81, 0.82, "pre_r64")
        assert read["rungs_wholly_below_zero"] == named and read["the_three_rungs"] == rungs
        assert (": WHOLLY BELOW ZERO on " + ", ".join(named) in read["said"]) is bool(named) and (": none wholly below zero" in read["said"]) is (not named)
    # THE SHARE OF DISTINCT ATTEMPTS on G: the model's and the start model's, said and not branched on.
    read = overfitting(_solved(30, 30), FLAT, 0.61, 0.82, "pre_r64")
    assert read["share_of_distinct_attempts_on_g"] == {"of_the_model": 0.61, "of_pre_r64": 0.82} and read["said"].endswith("; the share of distinct attempts on G 0.61 against 0.82")
    assert (read["against"], read["failed_checks"], read["narrower"]) == ("pre_r64", [], False) and "below +0.00000 [-0.05000, +0.05000]; in +0.01000 [-0.02000, +0.04000]" in read["said"]
    # A MODEL THAT FAILED A CHECK OF ITS OWN is INCONCLUSIVE: none of it is read, and what was measured is kept apart.
    failed = overfitting(_solved(23, 40), {**FLAT, "in": _change(-0.2, -0.3, -0.1)}, 0.61, 0.82, "pre_r64", ["its_training_ran"])
    assert (failed["narrower"], failed["rungs_wholly_below_zero"], failed["failed_checks"]) == (None, None, ["its_training_ran"])
    assert failed["said"] == "NOT READ: this model is INCONCLUSIVE (its_training_ran), and none of its reads is made"
    assert failed["measured_and_not_read"]["narrower"] is True and failed["measured_and_not_read"]["rungs_wholly_below_zero"] == ["in"]


# ------------------------------------------------------------------------------- the report, on hand-made rows
# Successes by problem in (the 32 attempts, the 61 more), in GOAL_IDS' order: three problems of 1 line, of 2-3, of 4-7, of 8 or more.
START = ([6, 6, 6, 4, 4, 4, 2, 2, 0, 0, 0, 0], [9, 9, 9, 6, 6, 6, 5, 4, 3, 2, 2, 1])               # the start model solves every problem; none of the last four in its first sampling
A_LITTLE = ([6, 6, 6, 4, 4, 4, 2, 2, 1, 1, 1, 1], [9, 9, 9, 6, 6, 6, 5, 4, 4, 3, 3, 2])            # two more successes on each of the last four problems
MORE = ([7, 7, 7, 5, 5, 5, 3, 3, 2, 2, 2, 2], [10, 10, 10, 7, 7, 7, 6, 5, 5, 4, 4, 3])             # ... and two more than that on EVERY problem: above `A_LITTLE` on all of G
NO_LONG = ([6, 6, 6, 4, 4, 4, 0, 0, 0, 0, 0, 0], [9, 9, 9, 6, 6, 6, 0, 0, 0, 0, 0, 0])             # no problem of 4 lines or more solved: lost 6 (p = 0.03)
PRETRAIN_HALF = [f"p{index}" for index in range(400) if half_of(f"p{index}", 0) == PRETRAIN]
TRAINED = {name: [{"id": f"{problem}#pretraining", "problem_id": problem, "origin": "pretraining", "k": None, "lines": 4 + index % 9} for index, problem in enumerate(PRETRAIN_HALF[:rows])]
           for name, rows in ((MATCHED, 4), (HALF, 30), (FULL, 60))}
TRAINS = {name: {"model": name, "rows": len(own), "steps": -(-len(own) // 8), "stand_in_engine": False, "trained_from": "the stored adapter pre_r64",
                 "against_the_start_adapter": {name: CHANGED}} for name, own in TRAINED.items()}
LOSSES = {name: FALLING(len(own)) for name, own in TRAINED.items()}
PREPARE = {"stage": "l4", "label": LABEL, "check": "L4p", "seed": 0, "start": "pre_r64", "start_adapter": "runs/ladder_l4_pretrain_r64_seed0/adapters/pre_r64",
           "l4t2_run": "runs/ladder_l4_rows2_pre_r64_seed0", "stand_in_engine": False, "models": list(MODELS), "start_recipe": {"rank": 64, "alpha": 128},
           "rows_the_start_was_pretrained_on": 60, "rehearse_only": {"rows": 8, "optimizer_steps": 1}, "reward_rehearse": {"rows": 10, "published_rows": 4},
           "training_sets": {MATCHED: {"rows": 4}, HALF: {"rows": 30}, FULL: {"rows": 60}},
           "goal_samplings": SAMPLINGS, "attempts_a_goal_problem": 93, "rung_episodes": 8, "sampling_seeds": {"rungs": 1002, "reach": 1001, "more": 1020},
           "contradicted_side_setting": "audit", "stored_models": {name: {"run": f"runs/{name}"} for name in ("pre_r64", *STORED)},
           "l4t2_read": {"failed_checks": {name: [] for name in STORED}}}
AGAIN = goal_set_again(_made("pre", on_g=START)["goal"][0], GOAL_IDS)


def _model(on_g=START, rungs=None, **changed):
    return {**_made("pre", on_g=on_g, rungs=rungs), "stand_in_engine": False, **changed}


def _report(models=None, stored=None, prepare=PREPARE, trains=TRAINS, losses=LOSSES, trained=TRAINED, never=None, again=AGAIN, measured=MODELS):
    """The report with the start model's own rows standing in for every model that is not given. `measured`: the
    models of the stage that are trained and measured (all three; `matched_1262` alone for its own job's report)."""
    models = {name: (models or {}).get(name) or _model() for name in measured}
    stored = {name: (stored or {}).get(name) or _model() for name in ("pre_r64", *STORED)}
    return build_dose_report(prepare, trains, losses, trained, GROUPS, LENGTHS, stored, models, again, SETTINGS, EVALUATION, never)


def _on_g(entry):
    return [{**first, "resolved": first["resolved"] + more["resolved"], "episodes": first["episodes"] + more["episodes"]} for first, more in zip(*entry["goal"])]


def test_with_the_start_models_rows_standing_in_every_difference_is_zero_and_the_interval_holds_zero():
    report = _report()
    assert report["label"] == LABEL and report["stage"] == "l4" and report["check"] == "L4p" and report["ok"] is True and all(line.startswith(f"{SAY}: ") for line in report["lines"])
    assert (report["start"], report["rank"], AGAIN, list(report["models"]), report["inconclusive"]) == ("pre_r64", 64, ["g4c", "g8a", "g8b", "g8c"], list(MODELS), False)
    primary = report["primary"]
    assert (primary["mean"], primary["low"], primary["high"], primary["problems"], primary["attempts_each"]) == (0, 0, 0, 12, 12 * 93)       # ALL of G, the 93 attempts
    assert report["branch"]["name"] == FIRST_SEVENTH and report["branch"]["resolves"] == 0                 # an interval that ends at zero holds zero
    for name in MODELS:
        assert (report["overfitting"][name]["narrower"], report["overfitting"][name]["rungs_wholly_below_zero"], report["overfitting"][name]["gained"]) == (False, [], 0)
        assert report["the_three_reads"][name]["fails"] == [ADDS] and report["the_three_reads"][name]["passed"] == [NOT_NARROWER, HOLDS]
    dose = report["the_dose"]
    assert dose["rows"] == {MATCHED: 4, "rehearse_only": 8, HALF: 30, FULL: 60} and dose["share_of_a_second_pass"] == {MATCHED: round(4 / 60, 5), "rehearse_only": round(8 / 60, 5), HALF: 0.5, FULL: 1.0}
    assert all(dose["against_the_start"][name]["read"] and dose["against_the_start"][name]["per_attempt"]["mean"] == 0 for name in DOSES) and list(dose["against_the_start"]) == list(DOSES)
    assert list(dose["steps"]) == ["rehearse_only_minus_matched_1262", "second_half_minus_rehearse_only", "second_full_minus_second_half"] and all(
        step["read"] and step["mean"] == 0 for step in dose["steps"].values())
    assert [step["rows_more"] for step in dose["steps"].values()] == [4, 22, 30] and (report["measured"], report["not_measured_yet"]) == (list(MODELS), [])
    # The rounds' hard rows: the stored `reward_rehearse` minus `matched_1262` is zero here, and an interval that ends at zero is not above it.
    hard = report["the_rounds_hard_rows"]
    assert (hard["name"], hard["difference"]["mean"], hard["difference"]["problems"], hard["resolves"], hard["published_rows"], hard["rows_of_reward_rehearse"]) == (NOT_SEPARATED, 0, 4, 0, 4, 10)
    assert report["branches"] == {"the_dose": FIRST_SEVENTH, "the_rounds_hard_rows": NOT_SEPARATED}
    assert report["what_it_cannot_say"] == CANNOT_SAY and report["heldout"]["goal_set_again_by_length_group"] == {"1": 0, "2-3": 0, "4-7": 1, "8+": 3, "4_or_more": 4, "all": 4}
    first = report["lines"][0]
    assert "L4p, HOW MUCH MORE PRETRAINING? THE DOSE OF A SECOND PASS, seed 0. A LABELLED CHECK OF THE PRETRAINING: no round, and no row of a round" in first
    assert "EVERY MODEL IS AT RANK 64, the start adapter's, trained FROM `pre_r64`" in first and "`rehearse_only` its first 8 (L4t2's model: stored in runs/ladder_l4_rows2_pre_r64_seed0, read and never trained again)" in first
    assert "`second_half` its first 30, `second_full` all 60: the dose at 7%, 13%, 50% and 100% of a second pass" in first
    assert "`matched_1262` its first 4 (EXACTLY the published rows of L4t2's `reward_rehearse` set, which holds 10 rows: its control)" in first
    assert "THE ROUNDS' HARD ROWS are read on G' over the 61 fresh attempts, the stored `reward_rehearse` minus `matched_1262`" in first
    assert "THE DOSE'S PRIMARY is on all of G (12 problems, the 93 attempts), successes per attempt, `second_full` minus `rehearse_only`" in first


def test_the_primary_is_second_full_minus_rehearse_only_on_all_of_g_over_all_the_attempts():
    # `second_full` above `rehearse_only` on EVERY problem of G: more of the second pass helps.
    helps = _report(models={FULL: _model(MORE), HALF: _model(A_LITTLE)}, stored={"rehearse_only": _model(A_LITTLE)})
    primary = helps["primary"]
    expected = per_attempt(_on_g(_model(MORE)), _on_g(_model(A_LITTLE)), GOAL_IDS, 400, 0)
    assert {key: primary[key] for key in expected} == expected and (primary["problems"], primary["mean"], primary["low"]) == (12, round(2 / 93, 5), round(2 / 93, 5))
    assert helps["branch"]["name"] == MORE_HELPS and "specified next" in helps["branch"]["reason"]
    said = [line for line in helps["lines"] if "PRIMARY." in line or "BRANCH: " in line]
    assert said[0].startswith(f"{SAY}: PRIMARY. All of G (12 problems), successes per attempt over the 93 attempts a problem, `second_full` minus `rehearse_only`, paired by problem, "
                              f"95% bootstrap over problems: {primary['mean']:+.5f} [") and said[1].startswith(f"{SAY}: BRANCH: MORE OF THE SECOND PASS HELPS. all of G")
    assert f"({primary['successes']} successes against {primary['successes_of_the_base']} in {12 * 93:,} attempts each)" in said[0] and len(said) == 2
    # ITS REFERENCE IS `rehearse_only`, NOT THE START MODEL: `second_full` with the START model's own rows is zero against it, and BELOW `rehearse_only`: more costs.
    costs = _report(stored={"rehearse_only": _model(MORE)})
    assert costs["branch"]["name"] == MORE_COSTS and costs["primary"]["high"] < 0 and costs["the_dose"]["against_the_start"][FULL]["per_attempt"]["mean"] == 0
    assert costs["primary"]["mean"] == -per_attempt(_on_g(_model(MORE)), _on_g(_model()), GOAL_IDS, 400, 0)["mean"]
    # ... and it is `second_full`'s, not `second_half`'s: the smaller dose above `rehearse_only` changes nothing of the primary.
    half_alone = _report(models={HALF: _model(MORE)}, stored={"rehearse_only": _model(A_LITTLE)})
    assert half_alone["primary"] == _report(stored={"rehearse_only": _model(A_LITTLE)})["primary"] and half_alone["branch"]["name"] != MORE_HELPS
    assert half_alone["the_dose"]["steps"]["second_half_minus_rehearse_only"]["low"] > 0 > half_alone["primary"]["mean"]
    # ALL of G, not G' and not the fresh attempts: more successes on G' alone, on four of twelve problems, is an interval over the twelve.
    on_again = _report(models={FULL: _model(A_LITTLE)})
    assert on_again["primary"]["problems"] == 12 and on_again["primary"]["mean"] == round(sum([2 / 93] * 4) / 12, 5) and on_again["the_three_reads"][FULL][KEY[ADDS]]["problems"] == 4
    # The interval is the 95% bootstrap over problems the reports share (`per_attempt`), with the config's resamples and seed.
    assert on_again["primary"]["low"] == per_attempt(_on_g(_model(A_LITTLE)), _on_g(_model()), GOAL_IDS, 400, 0)["low"] != on_again["primary"]["mean"]
    assert "Primary (all of G, 12 problems, 93 attempts each, `second_full` minus `rehearse_only`): " in on_again["headline"] and on_again["headline"].startswith(
        f"L4p (how much more pretraining; {LABEL}) seed 0, from `pre_r64`: {on_again['branch']['name']}. ")


def test_overfitting_is_read_on_every_branch_for_both_models_against_the_start_model():
    lower_in = [2, 3, 2, 2, 7, 8]                                                                  # the rung `in`: two and three fewer of 8 on its two problems
    for of_rehearse_only, of_full, branch in ((A_LITTLE, MORE, MORE_HELPS), (MORE, MORE, FIRST_SEVENTH), (MORE, START, MORE_COSTS)):
        full = _model(of_full, rungs=lower_in)
        report = _report(models={FULL: full, HALF: _model(NO_LONG)}, stored={"rehearse_only": _model(of_rehearse_only)})
        assert report["branch"]["name"] == branch
        of_half, of_full = report["overfitting"][HALF], report["overfitting"][FULL]
        # `second_half` solves no problem of 4 lines or more: lost 6 against the START model (p = 0.03): NARROWER, on whatever branch.
        assert (of_half["narrower"], of_half["gained"], of_half["lost"], of_half["sign_test_p"], of_half["against"]) == (True, 0, 6, sign_test(0, 6), "pre_r64")
        assert {key: of_half[key] for key in ("gained", "lost", "problems")} == {key: solved(_on_g(_model(NO_LONG)), _on_g(_model()), GOAL_IDS)[key] for key in ("gained", "lost", "problems")}
        assert of_full["narrower"] is False and of_full["rungs_wholly_below_zero"] == ["in"]
        # The rungs are the model minus the START model, paired by problem (the function the secondary reads use).
        rungs = {rung: paired_change(full["rungs"], _model()["rungs"], [f"{rung}1", f"{rung}2"], 400, 0) for rung in ("below", "in", "above")}
        assert of_full["the_three_rungs"] == rungs == report["models"][FULL]["secondary"]["the_three_rungs"]["minus_pre_r64"] and rungs["in"]["high"] < 0 == rungs["below"]["high"]
        assert of_full["share_of_distinct_attempts_on_g"] == {"of_the_model": 0.95, "of_pre_r64": 0.95}
        lines = [line for line in report["lines"] if ": OVERFITTING, " in line]
        assert len(lines) == 3 and lines[1].startswith(f"{SAY}: OVERFITTING, `second_half` against `pre_r64` (read on every branch): NARROWER (gained 0, lost 6 (p = ")
        assert lines[2].startswith(f"{SAY}: OVERFITTING, `second_full` against `pre_r64` (read on every branch): not narrower (") and ": WHOLLY BELOW ZERO on in; " in lines[2]
        assert lines[0].startswith(f"{SAY}: OVERFITTING, `matched_1262` against `pre_r64` (read on every branch): not narrower (gained 0, lost 0")
        assert "Overfitting against `pre_r64`: `matched_1262` not narrower (" in report["headline"] and "; `second_half` NARROWER (gained 0, lost 6 (p = " in report["headline"]
        assert "`second_full` not narrower" in report["headline"]
        assert ", wholly below zero on the rung in" in report["headline"]
        assert report["models"][HALF]["overfitting"] is of_half and report["models"][FULL]["overfitting"] is of_full
    # A rung whose interval ENDS at zero is not wholly below it: one problem lower, the other the same.
    edge = _report(models={FULL: _model(rungs=[2, 3, 2, 5, 7, 8])})["overfitting"][FULL]
    assert (edge["the_three_rungs"]["in"]["high"], edge["the_three_rungs"]["in"]["mean"] < 0, edge["rungs_wholly_below_zero"]) == (0, True, [])
    # The share of distinct attempts on G is each model's own, from what it wrote; the one beside it is the START model's, never `rehearse_only`'s.
    narrow, of_the_start, least = _model(), _model(), _model()
    for entry, share in ((narrow, 0.41), (of_the_start, 0.77), (least, 0.33)):
        entry["distinct_attempts"] = {**entry["distinct_attempts"], "goal": {"prompts": 12, "mean_share_distinct": share}}
    assert _report(models={FULL: narrow})["overfitting"][FULL]["share_of_distinct_attempts_on_g"] == {"of_the_model": 0.41, "of_pre_r64": 0.95}
    shares = _report(models={FULL: narrow}, stored={"pre_r64": of_the_start, "rehearse_only": least})["overfitting"]
    assert shares[FULL]["share_of_distinct_attempts_on_g"] == {"of_the_model": 0.41, "of_pre_r64": 0.77} and all(
        shares[name]["share_of_distinct_attempts_on_g"] == {"of_the_model": 0.95, "of_pre_r64": 0.77} for name in (MATCHED, HALF))
    # NARROWER is counted against the START model, never against `rehearse_only`: with a `rehearse_only` that solves six problems fewer, a model that solves what the start
    # model solves has gained none and lost none (against `rehearse_only` it would have gained six).
    against = _report(stored={"rehearse_only": _model(NO_LONG)})
    assert all((against["overfitting"][name]["gained"], against["overfitting"][name]["lost"], against["overfitting"][name]["narrower"]) == (0, 0, False) for name in MODELS)
    assert all(against["the_dose"]["against_the_start"][name]["solved"]["gained"] == 0 for name in MODELS) and against["the_dose"]["against_the_start"]["rehearse_only"]["solved"]["lost"] == 6
    assert solved(_on_g(_model()), _on_g(_model(NO_LONG)), GOAL_IDS)["gained"] == 6
    assert ON_THE_RUNGS["pre"] == [2, 3, 4, 5, 7, 8]


def test_the_dose_is_the_three_against_the_start_model_and_its_two_steps_with_rehearse_only_from_its_stored_rows():
    report = _report(models={HALF: _model(MORE), FULL: _model(NO_LONG)}, stored={"rehearse_only": _model(A_LITTLE)})
    dose, start = report["the_dose"], _on_g(_model())
    for name, rows_ in (("rehearse_only", A_LITTLE), (HALF, MORE), (FULL, NO_LONG)):
        entry = dose["against_the_start"][name]
        assert entry["per_attempt"] == per_attempt(_on_g(_model(rows_)), start, GOAL_IDS, 400, 0) and entry["solved"] == solved(_on_g(_model(rows_)), start, GOAL_IDS) and entry["read"] is True
        assert (entry["rows"], entry["share_of_a_second_pass"]) == (dose["rows"][name], dose["share_of_a_second_pass"][name])
    # THE SHAPE: each dose minus the one BEFORE it, on all of G.
    half_step, full_step = dose["steps"]["second_half_minus_rehearse_only"], dose["steps"]["second_full_minus_second_half"]
    assert {key: half_step[key] for key in ("mean", "low", "high", "problems")} == {key: per_attempt(_on_g(_model(MORE)), _on_g(_model(A_LITTLE)), GOAL_IDS, 400, 0)[key] for key in ("mean", "low", "high", "problems")}
    assert {key: full_step[key] for key in ("mean", "low", "high", "problems")} == {key: per_attempt(_on_g(_model(NO_LONG)), _on_g(_model(MORE)), GOAL_IDS, 400, 0)[key] for key in ("mean", "low", "high", "problems")}
    assert half_step["low"] > 0 > full_step["high"] and report["branch"]["name"] == MORE_COSTS
    lines = [line for line in report["lines"] if ": THE DOSE, " in line or ": THE SHAPE OF THE DOSE, " in line]
    assert len(lines) == 2 and lines[0].startswith(f"{SAY}: THE DOSE, all of G (12 problems, 93 attempts a problem), successes per attempt, each dose minus `pre_r64`: `matched_1262` (4 rows, "
                                                   "7% of a second pass): ") and lines[0].endswith(". `rehearse_only` is from the rows L4t2's run stored")
    assert "; `rehearse_only` (8 rows, 13% of a second pass): " in lines[0] and "; `second_half` (30 rows, 50% of a second pass): " in lines[0]
    assert "; `second_full` (60 rows, 100% of a second pass): " in lines[0] and "goal problems solved 6 to 12, gained 0, lost 6" in lines[0]
    assert "each dose minus the one before it: `rehearse_only` minus `matched_1262` (4 rows more): " in lines[1] and "; `second_half` minus `rehearse_only` (22 rows more): " in lines[1]
    assert "; `second_full` minus `second_half` (30 rows more): " in lines[1]
    # The first step is `rehearse_only` (stored) minus `matched_1262` (the start model's rows here): what `A_LITTLE` has over the start model.
    first_step = dose["steps"]["rehearse_only_minus_matched_1262"]
    assert {key: first_step[key] for key in ("mean", "low", "high", "problems")} == {key: per_attempt(_on_g(_model(A_LITTLE)), start, GOAL_IDS, 400, 0)[key] for key in ("mean", "low", "high", "problems")}
    # L4t2's THREE READS of each trained model against the start model, by its own function, on the figures the secondary reads give.
    for name, rows_ in ((HALF, MORE), (FULL, NO_LONG)):
        reads, model = report["the_three_reads"][name], _model(rows_)
        expected = three_reads(per_attempt(model["goal"][1], _model()["goal"][1], AGAIN, 400, 0), solved(_on_g(model), start, GOAL_IDS), per_attempt(_on_g(model), start, GOAL_IDS[6:], 400, 0), "pre_r64")
        assert reads == expected and reads is report["models"][name]["the_three_reads"]
    assert report["the_three_reads"][HALF]["all_three"] is True and report["the_three_reads"][FULL]["fails"] == [ADDS, NOT_NARROWER, HOLDS]
    assert (report["the_three_reads"]["goal_set_again"], report["the_three_reads"]["goal_problems_of_4_lines_or_more"], report["the_three_reads"]["fresh_attempts"]) == (4, 6, 61)
    said = [line for line in report["lines"] if "THE THREE READS of L4t2 against `pre_r64`" in line]
    assert [line.split("`")[1] for line in said] == list(MODELS) and said[1].endswith(report["the_three_reads"][HALF]["said"]) and "ADDS on G' over the 61 fresh attempts" in said[1]
    # NARROWER and the read NOT NARROWER are one rule on one count: they never disagree.
    assert all(report["overfitting"][name]["narrower"] is (not report["the_three_reads"][name][KEY[NOT_NARROWER]]["passes"]) for name in MODELS)


def test_each_trained_model_has_l4ts_four_checks_first_and_a_failed_one_makes_it_inconclusive_and_what_stands_on_it_not_read():
    given = {"models": {HALF: _model(A_LITTLE), FULL: _model(MORE)}, "stored": {"rehearse_only": _model(A_LITTLE)}}
    report = _report(**given)
    for name in MODELS:
        own = report["models"][name]
        assert list(own["can_this_model_be_read"]) == ["what", "its_training_ran", "it_still_writes_proofs", "lean_answered", "no_barred_row"] and own["inconclusive"] is False
        assert all(entry["passes"] for key, entry in own["can_this_model_be_read"].items() if key != "what") and own["failed_checks"] == []
        assert own["model"].startswith(f"`{name}`: from `pre_r64`, one pass over {len(TRAINED[name])} rows: {IS[name]}")
        assert list(own["can_this_model_be_read"]["lean_answered"]["share_without_an_answer"]) == [
            key for who in (name, "pre_r64", *READ_AGAINST[name]) for key in (f"rungs_{who}", f"goal_{who}_1", f"goal_{who}_2")]
        lines = [line for line in report["lines"] if line.startswith(f"{SAY}: `{name}`")]
        assert all(f", CHECK {number}, " in line for number, line in enumerate(lines[:4], start=1)) and all(line.endswith(": PASS") for line in lines[:4])
        barred = own["can_this_model_be_read"]["no_barred_row"]
        assert (barred["rows"], barred["rows_of_the_rounds"], barred["rehearsal_rows_of_the_pretrain_half"], barred["passes"]) == (len(TRAINED[name]), 0, len(TRAINED[name]), True)
    # ---- `second_full` fails a check (a row that is NOT of the `pretrain` half): INCONCLUSIVE; the primary, its overfitting, its reads and the step that stands on it are not read
    stray = [{**TRAINED[FULL][0], "problem_id": LOOP_HALF[0]}, *TRAINED[FULL][1:]]
    failed = _report(**given, trained={**TRAINED, FULL: stray})
    own = failed["models"][FULL]
    assert (failed["branch"]["name"], failed["inconclusive"], own["inconclusive"], own["failed_checks"], own["secondary"]) == (INCONCLUSIVE, True, True, ["no_barred_row"], None)
    assert failed["primary"]["mean"] is None and failed["primary"]["measured_and_not_read"]["mean"] == round(2 / 93, 5) and not any("PRIMARY." in line for line in failed["lines"])
    assert any(line.startswith(f"{SAY}: INCONCLUSIVE. `second_full` could not have been read (no_barred_row)") for line in failed["lines"]) and "Primary (" not in failed["headline"]
    assert own["overfitting"]["said"].startswith("NOT READ: this model is INCONCLUSIVE (no_barred_row)") and own["the_three_reads"]["all_three"] is None
    assert own["measured_and_not_read"]["secondary"]["trained_on"]["rows"] == 60 and "`second_full` INCONCLUSIVE" in failed["headline"]
    dose = failed["the_dose"]
    assert dose["against_the_start"][FULL]["read"] is False and dose["against_the_start"][FULL]["failed_checks"] == {FULL: ["no_barred_row"]}
    assert dose["steps"]["second_full_minus_second_half"]["read"] is False and dose["steps"]["second_full_minus_second_half"]["measured_and_not_read"]["mean"] is not None
    assert "`second_full` minus `second_half`: NOT READ (`second_full` is INCONCLUSIVE (no_barred_row): not read)" in next(line for line in failed["lines"] if "THE SHAPE" in line)
    assert "`second_full` (60 rows, 100% of a second pass): NOT READ" in next(line for line in failed["lines"] if ": THE DOSE, " in line)
    # ... and `second_half` is still read, by itself: its checks, its overfitting, its step, its secondary reads.
    assert failed["models"][HALF]["inconclusive"] is False and failed["overfitting"][HALF]["narrower"] is False and dose["steps"]["second_half_minus_rehearse_only"]["read"] is True
    assert any("`second_half`, SECONDARY" in line for line in failed["lines"]) and not any("`second_full`, SECONDARY" in line for line in failed["lines"]) and failed["ok"] is True
    # ---- `second_half` fails a check (its loss rose): the PRIMARY is still read (it does not stand on it); both steps of the shape stand on it and are not read
    risen = _report(**given, losses={**LOSSES, HALF: list(reversed(FALLING(30)))})
    assert risen["branch"]["name"] == MORE_HELPS and risen["models"][HALF]["failed_checks"] == ["its_training_ran"] and risen["models"][FULL]["failed_checks"] == []
    assert [step["read"] for step in risen["the_dose"]["steps"].values()] == [True, False, False] and risen["the_dose"]["against_the_start"][FULL]["read"] is True
    assert risen["overfitting"][HALF]["narrower"] is None and risen["overfitting"][FULL]["narrower"] is False and any("PRIMARY." in line for line in risen["lines"])
    # ---- `rehearse_only` read as INCONCLUSIVE by the report of the run that measured it: the primary is NOT READ against it; the models' own reads against the start model stand
    of_l4t2 = {**PREPARE, "l4t2_read": {"failed_checks": {"rehearse_only": ["lean_answered"], "reward_rehearse": []}}}
    unread = _report(**given, prepare=of_l4t2)
    assert (unread["branch"]["name"], unread["branch"]["not_read_against"], unread["inconclusive"]) == (NOT_READ, "rehearse_only", False) and unread["primary"]["mean"] is None
    assert not any("PRIMARY." in line for line in unread["lines"]) and any(line.startswith(f"{SAY}: BRANCH: NOT READ. `rehearse_only`, which the primary is read against") for line in unread["lines"])
    assert unread["the_dose"]["against_the_start"]["rehearse_only"]["read"] is False and [step["read"] for step in unread["the_dose"]["steps"].values()] == [False, False, True]
    assert all(not unread["models"][name]["inconclusive"] and unread["overfitting"][name]["narrower"] is False for name in MODELS)
    assert unread["the_rounds_hard_rows"]["name"] != NOT_READ                      # the rounds' hard rows do not stand on `rehearse_only`


def test_a_set_lean_did_not_answer_is_not_to_be_read_and_the_model_on_it_is_inconclusive():
    unanswered = _model(MORE)
    unanswered["goal"][1] = [{**row, "attempts_without_an_answer": 20} for row in unanswered["goal"][1]]
    report = _report(models={FULL: unanswered})
    assert report["ok"] is False and report["not_to_be_read"] == ["goal_second_full_2"] and "NOT TO BE READ" in report["lines"][-1] and "NOT TO BE READ" in report["headline"]
    assert report["models"][FULL]["failed_checks"] == ["lean_answered"] and report["models"][HALF]["failed_checks"] == [] and report["branch"]["name"] == INCONCLUSIVE
    everyone = ("pre_r64", *STORED, *MODELS)
    assert set(report["attempts"]) == {f"rungs_{name}" for name in everyone} | {f"goal_{name}_{index}" for name in everyone for index in (1, 2)}
    # The start model's sets and `rehearse_only`'s stand under every model's reads: one Lean did not answer makes all three INCONCLUSIVE. `reward_rehearse`'s stand under
    # `matched_1262`'s alone.
    for who, failing in (("pre_r64", MODELS), ("rehearse_only", MODELS), ("reward_rehearse", (MATCHED,))):
        stored = _model()
        stored["rungs"] = [{**row, "attempts_without_an_answer": 8} for row in stored["rungs"]]
        report = _report(stored={who: stored})
        assert report["not_to_be_read"] == [f"rungs_{who}"] and all(report["models"][name]["failed_checks"] == (["lean_answered"] if name in failing else []) for name in MODELS)
    # `second_half`'s own set is not one `second_full`'s reads stand on.
    half = _model()
    half["goal"][0] = [{**row, "attempts_without_an_answer": 10} for row in half["goal"][0]]
    report = _report(models={HALF: half})
    assert report["models"][HALF]["failed_checks"] == ["lean_answered"] and report["models"][FULL]["failed_checks"] == [] and report["ok"] is False
    assert report["not_to_be_read"] == ["goal_second_half_1"] and report["branch"]["name"] == FIRST_SEVENTH          # the primary does not stand on `second_half`


def test_the_secondary_reads_are_l4ts_of_each_trained_model_by_itself_with_the_start_model_and_rehearse_only_beside():
    report = _report(models={HALF: _model(A_LITTLE), FULL: _model(MORE)}, stored={"rehearse_only": _model(A_LITTLE)}, never=["g8a", "g8c", "not_a_goal_problem"])
    for name in MODELS:
        secondary = report["models"][name]["secondary"]
        assert list(secondary) == ["the_goal_set_again", "by_length_group", "goal_problems_solved_by_attempts_alone", "the_three_rungs", "distinct_attempts", "verified_proof_lines",
                                   "never_solved_before", "trained_on"]
        assert set(secondary["by_length_group"]) == {"what", "problems", "minus_pre_r64", "against_pre_r64"} and set(secondary["the_three_rungs"]) == {"what", "problems", "minus_pre_r64"}
        assert set(secondary["distinct_attempts"]) == set(secondary["verified_proof_lines"]) == set(secondary["goal_problems_solved_by_attempts_alone"]) == {"what", "pre_r64", *READ_AGAINST[name], name}
        reads = report["the_three_reads"][name]
        assert {key: reads[KEY[ADDS]][key] for key in ("mean", "low", "high", "problems")} == {key: secondary["the_goal_set_again"][key] for key in ("mean", "low", "high", "problems")}
        assert {key: reads[KEY[HOLDS]][key] for key in ("mean", "low", "high", "problems")} == {key: secondary["by_length_group"]["minus_pre_r64"]["4_or_more"][key] for key in ("mean", "low", "high", "problems")}
        assert secondary["trained_on"]["rows"] == len(TRAINED[name]) and secondary["trained_on"]["rows_by_origin"] == {"pretraining": len(TRAINED[name])}
        unsolved = secondary["never_solved_before"]
        assert (unsolved["given"], unsolved["goal_problems"], unsolved["solved"], unsolved["not_goal_problems_of_this_run"], unsolved["solved_by_rehearse_only"]) == (True, 2, 2, 1, 2)
        lines = [line for line in report["lines"] if line.startswith(f"{SAY}: `{name}`, SECONDARY, ")]
        assert len(lines) == 9 and "4 lines or more (6): " in lines[1] and "the three rungs (8 episodes a problem)" in lines[4] and "by k none" in lines[8]
        assert all(f"`{other}` " in lines[3] and f"`{other}` " in lines[5] and f"`{other}` " in lines[6] for other in ("pre_r64", *READ_AGAINST[name], name))
        assert f"of the 2 listed, `{name}` solves 2 (g8a, g8c); " + ("`reward_rehearse` solves 2 and `rehearse_only` 2" if name == MATCHED else "`rehearse_only` solves 2") \
            + "; 1 more ids of the file are not goal problems of this run" in lines[7]
    assert _report()["models"][FULL]["secondary"]["never_solved_before"]["given"] is False
    # THE ORDER OF THE LINES: the first; each model's four checks; the primary and its branch; overfitting; the dose and its shape; the three reads; then the secondary reads.
    heads = [("CHECK" if ", CHECK " in line else "HARD ROWS" if ": THE ROUNDS' HARD ROWS" in line or ": BESIDE IT, deciding nothing. " in line else "PRIMARY" if ": PRIMARY. " in line else
              "BRANCH" if ": BRANCH: " in line else "OVERFITTING" if ": OVERFITTING, " in line else "DOSE" if ": THE DOSE, " in line else
              "SHAPE" if ": THE SHAPE OF THE DOSE, " in line else "READS" if "THE THREE READS of L4t2" in line else "SECONDARY" if ", SECONDARY, " in line else "FIRST")
             for line in report["lines"]]
    assert heads == ["FIRST", *["CHECK"] * 12, *["HARD ROWS"] * 3, "PRIMARY", "BRANCH", *["OVERFITTING"] * 3, "DOSE", "SHAPE", *["READS"] * 3, *["SECONDARY"] * 27]


def test_a_run_with_one_sampling_of_g_reads_the_primary_on_it_and_no_fresh_attempts():
    one = {**PREPARE, "goal_samplings": SAMPLINGS[:1], "attempts_a_goal_problem": 32}
    first_alone = lambda entry: {**entry, "goal": entry["goal"][:1]}      # noqa: E731
    report = build_dose_report(one, TRAINS, LOSSES, TRAINED, GROUPS, LENGTHS, {"pre_r64": first_alone(_model()), "rehearse_only": first_alone(_model(A_LITTLE)), "reward_rehearse": first_alone(_model())},
                               {name: first_alone(_model(MORE)) for name in MODELS}, AGAIN, SETTINGS, EVALUATION, None)
    assert report["ok"] is True and report["branch"]["name"] == MORE_HELPS and report["primary"]["attempts_each"] == 12 * 32 and report["the_three_reads"]["fresh_attempts"] is None
    assert all(report["the_three_reads"][name]["not_read"] == [ADDS] for name in MODELS)
    assert any("the stored models have ONE sampling of G here (a smoke run)" in line and "the rounds' hard rows and ADDS are not read" in line for line in report["lines"])
    assert report["the_rounds_hard_rows"]["name"] == NOT_READ and "G' holds no problem with fresh attempts on both sides" in report["the_rounds_hard_rows"]["reason"]
    assert any("THE THREE READS of L4t2 against `pre_r64` (ADDS on G' over fresh attempts, which this run does not have" in line for line in report["lines"])
    with pytest.raises(ValueError, match="1 problems of G' are not goal problems of this run .first: nowhere."):
        _report(again=[*AGAIN, "nowhere"])


def test_no_rule_is_written_a_second_time():
    """The report and the rules call the functions the other reports use: the world a model is read on, its four checks,
    its secondary reads, the three reads, NARROWER, the interval, the sign test, the draw."""
    reporting = (PACKAGE / "reporting" / "ladder_l4_dose.py").read_text()
    for name in ("world_of(", "the_four_checks(", "check_lines(", "secondary_reads(", "secondary_lines(", "three_reads(", "dose_branch(", "matched_branch(", "overfitting(", "per_attempt(",
                 "solved(", "never_solved("):
        assert name in reporting and f"def {name}" not in reporting, name
    domain = (PACKAGE / "domain" / "ladder_round" / "l4_dose.py").read_text()
    assert "breadth_beside(solved, distinct, distinct_of_the_start, start, level)" in domain and domain.count("rehearsal_rows(pretraining, rows_of(") == 2
    assert "rehearsal_rows(pretraining, len(published), seed)" in domain and "published = rehearsal_of(reward_rehearse)" in domain and "training_order(matched, seed)" in domain
    assert "training_order(half, seed)" in domain and "training_order(full, seed)" in domain
    for written_again in ("math.comb", "paired_bootstrap", "sign_test(", "hashlib", "random"):
        assert written_again not in domain and written_again not in reporting, written_again
    gpu = (PACKAGE / "gpu" / "ladder_l4_dose.py").read_text()
    assert gpu.count("train_a_model(") == 1 and gpu.count("measure_a_model(") == 1 and gpu.count("read_the_trained(") == 1
    assert not [name for name in ("one_pass(", "measure_model(", "examples_of(", "the_adapter_saved(", "hot_config(", "max_lora_rank", "rehearsal_rows(") if name in gpu]


def test_the_rounds_hard_rows_are_reward_rehearse_minus_matched_1262_on_g_again_over_the_fresh_attempts():
    # `reward_rehearse` (stored) above `matched_1262` on every problem of G': the rounds' hard rows add beside the same published rows.
    adds = _report(models={MATCHED: _model(A_LITTLE)}, stored={"reward_rehearse": _model(MORE)})
    hard = adds["the_rounds_hard_rows"]
    expected = per_attempt(_model(MORE)["goal"][1], _model(A_LITTLE)["goal"][1], AGAIN, 400, 0)                # G', the 61 FRESH attempts alone: the second sampling
    assert hard["difference"] == expected and (hard["name"], hard["difference"]["problems"], hard["difference"]["mean"], hard["difference"]["low"]) == (HARD_ROWS_ADD, 4, round(1 / 61, 5), round(1 / 61, 5))
    said = [line for line in adds["lines"] if ": THE ROUNDS' HARD ROWS" in line or ": BESIDE IT, deciding nothing. " in line]
    assert len(said) == 3 and said[0].startswith(f"{SAY}: THE ROUNDS' HARD ROWS BESIDE THE SAME PUBLISHED ROWS. G' (4 goal problems `pre_r64` does not solve in its first sampling), successes per "
                                                 "attempt over the 61 fresh attempts, `reward_rehearse` (stored: the rounds' hard rows and these 4 published rows) minus `matched_1262` (these "
                                                 f"published rows alone), paired by problem, 95% bootstrap over problems: {expected['mean']:+.5f} [")
    assert said[1].startswith(f"{SAY}: THE ROUNDS' HARD ROWS: THE ROUNDS' HARD ROWS ADD BESIDE THE SAME PUBLISHED ROWS. on G' over the fresh attempts `reward_rehearse` minus `matched_1262` is above zero")
    assert "The rounds' hard rows beside the same published rows (`reward_rehearse` minus `matched_1262` on G'): THE ROUNDS' HARD ROWS ADD BESIDE THE SAME PUBLISHED ROWS (" in adds["headline"]
    # ITS REFERENCE IS `matched_1262`, NOT THE START MODEL AND NOT `rehearse_only`: `reward_rehearse` above the start model and BELOW `matched_1262` costs.
    costs = _report(models={MATCHED: _model(MORE)}, stored={"reward_rehearse": _model(A_LITTLE), "rehearse_only": _model()})
    assert costs["the_rounds_hard_rows"]["name"] == THEY_COST and costs["the_rounds_hard_rows"]["difference"]["mean"] == round(-1 / 61, 5)
    assert per_attempt(_model(A_LITTLE)["goal"][1], _model()["goal"][1], AGAIN, 400, 0)["low"] > 0
    # ... and it is `reward_rehearse`'s stored rows on the other side, not `rehearse_only`'s, and `matched_1262`'s, not another dose's.
    others = _report(models={HALF: _model(MORE), FULL: _model(MORE)}, stored={"rehearse_only": _model(MORE)})
    assert others["the_rounds_hard_rows"]["name"] == NOT_SEPARATED and others["the_rounds_hard_rows"]["difference"]["mean"] == 0
    # G' AND THE FRESH ATTEMPTS, not all of G: more on the problems outside G' alone is nothing here, and stands beside.
    outside = ([7, 7, 7, 5, 5, 5, 3, 3, 0, 0, 0, 0], [10, 10, 10, 7, 7, 7, 6, 5, 3, 2, 2, 1])                  # one more in each sampling on the eight problems that are not of G'
    beside = _report(stored={"reward_rehearse": _model(outside)})
    assert beside["the_rounds_hard_rows"]["name"] == NOT_SEPARATED and beside["the_rounds_hard_rows"]["difference"]["mean"] == 0
    # BESIDE IT, DECIDING NOTHING: the same difference on ALL of G over the 93 attempts; the goal problems solved; solved reliably; the never-solved when given.
    given = _report(models={MATCHED: _model(NO_LONG)}, stored={"reward_rehearse": _model(outside)}, never=["g8a", "g8c", "not_a_goal_problem"])
    it = given["the_rounds_hard_rows"]["beside"]
    assert it["on_all_of_g"] == per_attempt(_on_g(_model(outside)), _on_g(_model(NO_LONG)), GOAL_IDS, 400, 0) and it["on_all_of_g"]["attempts_each"] == 12 * 93
    assert it["solved"] == solved(_on_g(_model(outside)), _on_g(_model(NO_LONG)), GOAL_IDS) and (it["solved"]["gained"], it["solved"]["lost"]) == (6, 0)
    assert it["solved_reliably"] == {name: given["models"][MATCHED]["secondary"]["goal_problems_solved_by_attempts_alone"][name]["all"]["reliably"] for name in ("reward_rehearse", MATCHED)}
    assert it["never_solved_before"] == {"given": True, "goal_problems": 2, "reward_rehearse": 2, MATCHED: 0}
    line = next(line for line in given["lines"] if ": BESIDE IT, deciding nothing. " in line)
    assert "All of G over the 93 attempts, `reward_rehearse` minus `matched_1262`: " in line and "goal problems solved 12 to 6, gained 6, lost 0" in line
    assert "solved reliably `reward_rehearse` " in line and "the goal problems nothing stored had solved: of the 2 listed, `reward_rehearse` 2, `matched_1262` 0" in line
    assert "the goal problems nothing stored had solved: not given" in next(line for line in adds["lines"] if ": BESIDE IT, deciding nothing. " in line)
    # It decides nothing of the dose: the primary is what it was.
    assert given["branch"]["name"] == FIRST_SEVENTH and given["branches"] == {"the_dose": FIRST_SEVENTH, "the_rounds_hard_rows": given["the_rounds_hard_rows"]["name"]}
    # ---- `matched_1262` fails a check: INCONCLUSIVE, the difference not read (kept apart); the dose's primary is still read
    risen = _report(models={MATCHED: _model(A_LITTLE)}, stored={"reward_rehearse": _model(MORE)}, losses={**LOSSES, MATCHED: list(reversed(FALLING(4)))})
    hard = risen["the_rounds_hard_rows"]
    assert (hard["name"], hard["failed_checks"], hard["difference"]["mean"], hard["difference"]["measured_and_not_read"]["mean"]) == (INCONCLUSIVE, ["its_training_ran"], None, round(1 / 61, 5))
    assert "measured_and_not_read" in hard["beside"] and risen["branch"]["name"] == FIRST_SEVENTH and risen["models"][MATCHED]["inconclusive"] is True
    assert len([line for line in risen["lines"] if ": THE ROUNDS' HARD ROWS" in line or ": BESIDE IT, deciding nothing. " in line]) == 1
    assert [step["read"] for step in risen["the_dose"]["steps"].values()] == [False, True, True] and risen["the_dose"]["against_the_start"][MATCHED]["read"] is False
    # ---- `reward_rehearse` read as INCONCLUSIVE by the report of the run that measured it: NOT READ against it; `matched_1262`'s own reads against the start model stand
    of_l4t2 = {**PREPARE, "l4t2_read": {"failed_checks": {"rehearse_only": [], "reward_rehearse": ["its_training_ran"]}}}
    unread = _report(models={MATCHED: _model(A_LITTLE)}, stored={"reward_rehearse": _model(MORE)}, prepare=of_l4t2)
    assert (unread["the_rounds_hard_rows"]["name"], unread["the_rounds_hard_rows"]["not_read_against"]) == (NOT_READ, "reward_rehearse") and unread["models"][MATCHED]["inconclusive"] is False
    assert unread["the_rounds_hard_rows"]["difference"] == {"mean": None, "measured_and_not_read": per_attempt(_model(MORE)["goal"][1], _model(A_LITTLE)["goal"][1], AGAIN, 400, 0)}
    assert set(unread["the_rounds_hard_rows"]["beside"]) == {"what", "measured_and_not_read"} and "(+0.01639 [" not in unread["headline"]
    assert [line for line in unread["lines"] if ": THE ROUNDS' HARD ROWS" in line or ": BESIDE IT, deciding nothing. " in line] == [
        f"{SAY}: THE ROUNDS' HARD ROWS: NOT READ. {unread['the_rounds_hard_rows']['reason']}"]
    assert unread["the_three_reads"][MATCHED]["all_three"] is True and all(step["read"] for step in unread["the_dose"]["steps"].values()) and unread["branch"]["name"] == FIRST_SEVENTH


def test_the_report_with_matched_1262_alone_measured_reads_its_read_and_says_the_rest_is_not_measured_yet():
    alone = _report(models={MATCHED: _model(A_LITTLE)}, stored={"reward_rehearse": _model(MORE), "rehearse_only": _model(MORE)}, measured=(MATCHED,), never=["g8a"])
    assert (alone["measured"], alone["not_measured_yet"], alone["ok"], alone["inconclusive"]) == ([MATCHED], [HALF, FULL], True, False)
    # THE DOSE'S PRIMARY IS NOT READ, said as such, and not INCONCLUSIVE; no figure of it is given.
    assert (alone["branch"]["name"], alone["branch"]["not_measured"], alone["primary"]["mean"]) == (NOT_READ, FULL, None) and "measured_and_not_read" not in alone["primary"]
    assert alone["branches"] == {"the_dose": NOT_READ, "the_rounds_hard_rows": HARD_ROWS_ADD} and not any(": PRIMARY. " in line for line in alone["lines"])
    assert any(line.startswith(f"{SAY}: BRANCH: NOT READ. `second_full` is not measured yet: this is the report of the job that trains and measures `matched_1262` alone") for line in alone["lines"])
    assert alone["lines"][1].startswith(f"{SAY}: THIS IS THE REPORT OF THE JOB THAT TRAINS AND MEASURES `matched_1262` ALONE: `second_half` and `second_full` are NOT MEASURED YET.")
    assert "NOT READ here (not INCONCLUSIVE: nothing of them failed)" in alone["lines"][1] and INCONCLUSIVE not in alone["headline"]
    assert alone["headline"].startswith(f"L4p (how much more pretraining; {LABEL}) seed 0, from `pre_r64`: NOT READ (THE DOSE'S PRIMARY: `second_half` and `second_full` not measured yet). "
                                        "The rounds' hard rows beside the same published rows (`reward_rehearse` minus `matched_1262` on G'): THE ROUNDS' HARD ROWS ADD BESIDE THE SAME PUBLISHED ROWS (")
    assert "`second_half` not measured yet; `second_full` not measured yet" in alone["headline"]
    # WHAT STANDS ON `matched_1262` AND THE STORED MODELS ALONE IS READ: its read, its checks, its overfitting, its three reads, its secondary reads, its dose and the first step.
    whole = _report(models={MATCHED: _model(A_LITTLE)}, stored={"reward_rehearse": _model(MORE), "rehearse_only": _model(MORE)}, never=["g8a"])
    assert alone["the_rounds_hard_rows"] == whole["the_rounds_hard_rows"] and alone["models"][MATCHED] == whole["models"][MATCHED] and alone["overfitting"][MATCHED] == whole["overfitting"][MATCHED]
    assert alone["the_three_reads"][MATCHED] == whole["the_three_reads"][MATCHED] and alone["the_three_reads"][MATCHED]["all_three"] is True
    dose = alone["the_dose"]
    assert dose["against_the_start"][MATCHED] == whole["the_dose"]["against_the_start"][MATCHED] and dose["against_the_start"]["rehearse_only"]["read"] is True
    assert dose["steps"]["rehearse_only_minus_matched_1262"] == whole["the_dose"]["steps"]["rehearse_only_minus_matched_1262"] and dose["steps"]["rehearse_only_minus_matched_1262"]["low"] > 0
    # ... and what stands on a model that is not measured yet is NOT READ, with nothing of it computed.
    for name in (HALF, FULL):
        assert alone["models"][name]["measured"] is False and alone["models"][name]["said"] == f"NOT READ: `{name}` is not measured yet (the job that trains and measures `matched_1262` alone writes this report)"
        assert alone["overfitting"][name]["said"] == alone["models"][name]["said"] == alone["the_three_reads"][name]["said"] and "narrower" not in alone["overfitting"][name]
        assert dose["against_the_start"][name] == {"rows": PREPARE["training_sets"][name]["rows"], "share_of_a_second_pass": dose["share_of_a_second_pass"][name], "read": False,
                                                   "not_measured": [name], "why": f"`{name}` is not measured yet"}
        assert f"`{name}`: not measured yet (the job that trains and measures `matched_1262` alone)" in alone["not_measured"]
    assert dose["steps"]["second_half_minus_rehearse_only"] == {"rows_more": 22, "read": False, "not_measured": [HALF], "why": "`second_half` is not measured yet"}
    assert dose["steps"]["second_full_minus_second_half"] == {"rows_more": 30, "read": False, "not_measured": [FULL, HALF], "why": "`second_full` and `second_half` are not measured yet"}
    assert set(alone["attempts"]) == {f"rungs_{name}" for name in ("pre_r64", *STORED, MATCHED)} | {f"goal_{name}_{index}" for name in ("pre_r64", *STORED, MATCHED) for index in (1, 2)}
    heads = [("CHECK" if ", CHECK " in line else "HARD ROWS" if ": THE ROUNDS' HARD ROWS" in line or ": BESIDE IT, deciding nothing. " in line else "BRANCH" if ": BRANCH: " in line else
              "OVERFITTING" if ": OVERFITTING, " in line else "DOSE" if ": THE DOSE, " in line else "SHAPE" if ": THE SHAPE OF THE DOSE, " in line else
              "READS" if "THE THREE READS of L4t2" in line else "SECONDARY" if ", SECONDARY, " in line else "ALONE" if "THIS IS THE REPORT OF THE JOB" in line else "FIRST")
             for line in alone["lines"]]
    assert heads == ["FIRST", "ALONE", *["CHECK"] * 4, *["HARD ROWS"] * 3, "BRANCH", *["OVERFITTING"] * 3, "DOSE", "SHAPE", *["READS"] * 3, *["SECONDARY"] * 9]
    assert "`second_half` (30 rows, 50% of a second pass): NOT READ (`second_half` is not measured yet)" in next(line for line in alone["lines"] if ": THE DOSE, " in line)
    assert "`second_half` minus `rehearse_only`: NOT READ (`second_half` is not measured yet)" in next(line for line in alone["lines"] if ": THE SHAPE OF THE DOSE, " in line)
    # A model that is MEASURED and failed a check is INCONCLUSIVE, as ever: the two are never said as one.
    risen = _report(models={MATCHED: _model(A_LITTLE)}, measured=(MATCHED, HALF), losses={**LOSSES, HALF: list(reversed(FALLING(30)))})
    assert (risen["not_measured_yet"], risen["models"][HALF]["inconclusive"], risen["models"][HALF]["measured"], risen["branch"]["not_measured"]) == ([FULL], True, True, FULL)
    assert risen["overfitting"][HALF]["said"].startswith("NOT READ: this model is INCONCLUSIVE (its_training_ran)") and "`second_half` INCONCLUSIVE; `second_full` not measured yet" in risen["headline"]
