"""L4t, what should a round train on: its two row rules and its report on hand-made rows with a known answer (the draw,
the reward rule, the rehearsal draw, what is barred, the sign test's smallest decisive split, each branch in the spec's
words, the checks first and each model by itself, the table of `hot`). Spec: docs/spec/ladder-loop.spec.md, "L4t:
what should a round train on? Three one-change checks on the rounds already made". Pure: no model, no Lean. The stage end
to end is `test_ladder_l4_rows_stage.py`."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l3d1 import EVALUATION, GOAL_IDS, GROUPS, LENGTHS, _rung  # noqa: E402 - Step 1's hand-made world: 12 goal problems, 3 a length group
from test_ladder_l4 import AGAIN, CHANGED, LOOP_HALF, OF_THE_OTHER_HALF, _made  # noqa: E402 - L4's hand-made `pre`, `with` and `without`

from rlvr_lean.domain.ladder_round.assembly import training_order  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ, PRETRAIN, half_of  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows import (  # noqa: E402
    BRANCHES,
    GIVES_BACK,
    MODELS,
    NOT_NAMED,
    NOT_THIS_LEVER,
    REHEARSE,
    REWARD_ROWS,
    THE_MINIMUM,
    THE_RULE,
    UNDOES,
    barred,
    kept_rows,
    no_barred_row,
    refuse_barred,
    rehearsal_id,
    rehearsal_rows,
    rehearse_set,
    reward_draws,
    rows_branch,
    smallest_decisive_split,
    uniform,
    with_k,
)
from rlvr_lean.domain.ladder_round.rounds import sign_test  # noqa: E402
from rlvr_lean.domain.problem_pool.episodes import reward  # noqa: E402
from rlvr_lean.domain.problem_pool.selection import rank  # noqa: E402
from rlvr_lean.reporting.ladder_l4 import LABEL, SAY  # noqa: E402
from rlvr_lean.reporting.ladder_l4_rows import CANNOT_SAY, build_rows_report, on_a_set, trained_on  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
SETTINGS = CONFIG["ladder_loop"]["l4"]
TARGET = CONFIG["ladder_loop"]["l2_assembly_arms"][SETTINGS["rows"]["arm_run"]]["target_rate"]


# ------------------------------------------------------------------------------------------------ the draws
def _attempt(problem_id, number=1, batch=1, sample=0):
    key = f"{problem_id}#round_r{number}_b{batch}#statement#{sample}"
    return {"id": key, "problem_id": problem_id, "side": "statement", "attempt_id": key, "theorem": f"theorem {problem_id} : P := by\n", "completion": "  step\n  done\n",
            "verified_attempts": 1, "round": number, "batch": batch, "origin": "attempt", "lines": 2}


def _assembled(problem_id, number=1, batch=1):
    return {"id": f"{problem_id}#assembled#r{number}", "problem_id": problem_id, "side": "statement", "attempt_id": None, "theorem": f"theorem {problem_id} : P := by\n",
            "completion": "  have h : A := by simp\n  exact h\n", "verified_attempts": 0, "round": number, "batch": batch, "origin": "assembled", "lines": 2}


def _pick(problem_id, resolved, by_assembly=False, episodes=8):
    return {"problem_id": problem_id, "episodes": episodes, "resolved": resolved, "resolved_by_assembly": by_assembly}


def test_a_draw_is_a_number_in_0_1_made_from_the_content_hash_and_is_the_same_whenever_it_is_asked():
    draws = [uniform(0, "l4_rows_keep", f"row{index}") for index in range(2000)]
    assert all(0 <= draw < 1 for draw in draws) and draws == [uniform(0, "l4_rows_keep", f"row{index}") for index in range(2000)]
    # It is the hash every draw of the loop uses (`selection.rank`), its first 13 hexadecimal digits over 16 ** 13.
    assert uniform(0, "l4_rows_keep", "row7") == int(rank(0, "l4_rows_keep", "row7")[:13], 16) / 16 ** 13
    # Uniform: about a tenth of the draws in each tenth; another seed or another label gives another draw.
    assert all(150 < sum(tenth / 10 <= draw < (tenth + 1) / 10 for draw in draws) < 250 for tenth in range(10))
    assert uniform(1, "l4_rows_keep", "row7") != uniform(0, "l4_rows_keep", "row7") != uniform(0, "l4_rows_rehearsal", "row7")


# -------------------------------------------------------------------------------------------- reward_rows
def test_the_reward_is_the_challengers_own_at_the_arms_target_rate():
    assert TARGET == 0.10 and [round(reward(k, 8, TARGET), 2) for k in range(0, 9)] == [0.0, 0.97, 0.48, 0.14, 0.03, 0.0, 0.0, 0.0, 0.0]      # the spec's figures


def test_a_row_is_kept_when_its_draw_is_under_the_reward_of_its_problems_k_and_an_assembled_row_is_k_1():
    rows = [_attempt("p1"), _attempt("p2"), _assembled("p3"), _attempt("p4")]
    picks = [_pick("p1", 1), _pick("p2", 4), _pick("p3", 0, by_assembly=True), _pick("p4", 8), _pick("p5", 0)]      # p5: picked, resolved by nothing, no row
    read = with_k(rows, picks)
    assert [(row["id"], row["k"], row["n"]) for row in read] == [(rows[0]["id"], 1, 8), (rows[1]["id"], 4, 8), ("p3#assembled#r1", 1, 8), (rows[3]["id"], 8, 8)]
    assert all({key: value for key, value in row.items() if key not in ("k", "n")} == given for row, given in zip(read, rows))      # nothing else of a row moves
    draws = reward_draws(read, TARGET, seed=0)
    for row in draws:
        assert row["reward"] == round(reward(row["k"], row["n"], TARGET), 6) and row["draw"] == uniform(0, "l4_rows_keep", row["id"])
        assert row["kept"] is (row["draw"] < reward(row["k"], row["n"], TARGET))
    paid = [round(reward(k, 8, TARGET), 6) for k in (1, 4, 1, 8)]
    assert [row["reward"] for row in draws] == paid and paid[0] == paid[2] > 0.97 and paid[3] == 0.0      # the assembled row is paid as k = 1; a row at k = 8 is never kept
    assert draws[3]["kept"] is False
    # DETERMINISTIC AND REPRODUCIBLE: the same rows give the same draws; a row's draw does not depend on which other rows there are; another seed gives other draws.
    assert reward_draws(read, TARGET, seed=0) == draws and reward_draws(read[2:], TARGET, seed=0) == draws[2:]
    assert [row["draw"] for row in reward_draws(read, TARGET, seed=1)] != [row["draw"] for row in draws]


def test_over_many_rows_about_97_in_100_are_kept_at_k_1_and_about_3_at_k_4():
    for k, expected in ((1, 0.970), (2, 0.485), (3, 0.141), (4, 0.025), (5, 0.002), (8, 0.0)):
        rows = [{**_attempt(f"p{index}"), "k": k, "n": 8} for index in range(4000)]
        kept = sum(row["kept"] for row in reward_draws(rows, TARGET, seed=0))
        assert abs(kept / 4000 - expected) < 0.012, (k, kept)
    assembled = [{**_assembled(f"p{index}"), "k": 1, "n": 8} for index in range(4000)]
    assert abs(sum(row["kept"] for row in reward_draws(assembled, TARGET, seed=0)) / 4000 - 0.970) < 0.012


def test_a_row_without_a_k_is_refused():
    rows, picks = [_attempt("p1"), _assembled("p3")], [_pick("p1", 2), _pick("p3", 0, by_assembly=True)]
    assert [row["k"] for row in with_k(rows, picks)] == [2, 1]
    for changed_rows, changed_picks, match in (
            (rows, [*picks, _pick("p1", 3)], "p1 was picked twice by the rounds: a problem has one k"),
            ([*rows, _attempt("p9")], picks, "is of p9, which no round picked: it has no k"),
            ([*rows, {**_assembled("p1"), "origin": "h0", "id": "p1#h0"}], picks, "is of origin 'h0': the rule gives a k to a round's one-shot and assembled rows, and to no other"),
            (rows, [_pick("p1", 0), picks[1]], "is of origin 'attempt' and its problem's pick has 0 verified attempts: the stored rounds do not agree"),
            (rows, [picks[0], _pick("p3", 0)], "is of origin 'assembled' and its problem's pick has 0 verified attempts: the stored rounds do not agree"),
            (rows, [picks[0], _pick("p3", 2, by_assembly=True)], "is of origin 'assembled' and its problem's pick has 2 verified attempts")):
        with pytest.raises(ValueError, match=match):
            with_k(changed_rows, changed_picks)


def test_the_kept_rows_are_in_the_arms_order_and_a_smoke_runs_minimum_is_filled_by_the_smallest_draws_and_marked():
    draws = reward_draws([{**_attempt(f"p{index}"), "k": 1 if index < 30 else 8, "n": 8} for index in range(40)], TARGET, seed=0)
    kept = kept_rows(draws, seed=0)
    assert {row["id"] for row in kept} == {row["id"] for row in draws if row["kept"]} and 25 <= len(kept) <= 30 and all(row["kept_by"] == THE_RULE for row in kept)
    assert [row["id"] for row in kept] == [row["id"] for row in training_order([row for row in draws if row["kept"]], 0)]       # the arm's content-hash order
    assert kept_rows(draws, seed=0, at_least=len(kept)) == kept                                    # a minimum the rule meets changes nothing
    # A smoke run: the rule keeps none of a fixture's rows (every problem at k = 8), and the run still needs rows to train on.
    none = reward_draws([{**_attempt(f"p{index}"), "k": 8, "n": 8} for index in range(5)], TARGET, seed=0)
    assert kept_rows(none, seed=0) == []
    filled = kept_rows(none, seed=0, at_least=2)
    assert len(filled) == 2 and all(row["kept_by"] == THE_MINIMUM and row["kept"] is False for row in filled)
    assert {row["id"] for row in filled} == {row["id"] for row in sorted(none, key=lambda row: row["draw"])[:2]}
    assert len(kept_rows(none, seed=0, at_least=9)) == 5                                           # never more rows than there are


# ------------------------------------------------------------------------------------------------- rehearse
def test_the_rehearsal_is_as_many_rows_of_the_pretraining_file_as_the_twin_has_by_the_hash_and_no_row_twice():
    pretraining = [{"problem_id": f"q{index}", "kind": "lean_workbook", "proof_lines": 1 + index % 9, "lines": 1 + index % 9} for index in range(50)]
    drawn = rehearsal_rows(pretraining, 12, seed=0)
    assert len(drawn) == 12 == len({row["problem_id"] for row in drawn}) and all(row["id"] == rehearsal_id(row["problem_id"]) == f"{row['problem_id']}#pretraining" for row in drawn)
    assert all((row["origin"], row["k"], row["side"]) == ("pretraining", None, "statement") and pretraining[row["row_of_the_file"]]["problem_id"] == row["problem_id"] for row in drawn)
    assert not any("theorem" in row or "completion" in row or "proof" in row for row in drawn)     # no text: the published proofs stay in the pretraining file
    # By the content hash of the task's seed and each row's id: the 12 of the smallest hash, whatever the file's order; another seed draws others.
    by_hash = sorted(pretraining, key=lambda row: rank(0, "l4_rows_rehearsal", f"{row['problem_id']}#pretraining"))[:12]
    assert [row["problem_id"] for row in drawn] == [row["problem_id"] for row in by_hash]
    assert {row["problem_id"] for row in rehearsal_rows(list(reversed(pretraining)), 12, seed=0)} == {row["problem_id"] for row in drawn}
    assert {row["problem_id"] for row in rehearsal_rows(pretraining, 12, seed=1)} != {row["problem_id"] for row in drawn}
    assert len(rehearsal_rows(pretraining, 50, seed=0)) == 50 and rehearsal_rows(pretraining, 0, seed=0) == []
    with pytest.raises(ValueError, match="`pre` was trained on 50 rows and 51 are asked for the rehearsal .as many as the twin's.: no row is drawn twice"):
        rehearsal_rows(pretraining, 51, seed=0)
    with pytest.raises(ValueError, match="1 problems stand twice among the rows `pre` was trained on .first: q3.: a rehearsal row is drawn once"):
        rehearsal_rows([*pretraining, dict(pretraining[3])], 12, seed=0)
    # `rehearse`: the twin's rows and the rehearsal's, in ONE order, the arm's content hash.
    twin = [{**_attempt(f"p{index}"), "k": 2} for index in range(12)]
    together = rehearse_set(twin, drawn, seed=0)
    assert len(together) == 24 and [row["id"] for row in together] == [row["id"] for row in training_order([*twin, *drawn], 0)]
    origins = [row["origin"] for row in together]
    assert origins.count("attempt") == origins.count("pretraining") == 12 and origins != sorted(origins)                    # mixed, not one block after the other
    with pytest.raises(ValueError, match="the rehearsal holds 11 rows and the twin 12: `rehearse` is the twin's rows and AS MANY of the pretraining file"):
        rehearse_set(twin, drawn[:11], seed=0)
    with pytest.raises(ValueError, match="1 ids stand twice in `rehearse`"):
        rehearse_set([*twin[:11], twin[0]], drawn, seed=0)


# --------------------------------------------------------------------------------------------- what is barred
def test_a_training_set_with_a_held_out_problem_or_a_rounds_row_of_the_pretrain_half_is_refused():
    of_the_pretrain_half = [f"p{index}" for index in range(200) if half_of(f"p{index}", 0) == PRETRAIN][:6]
    rounds = [_attempt(name) for name in LOOP_HALF[:8]]
    rehearsal = [{"id": rehearsal_id(name), "problem_id": name, "origin": "pretraining"} for name in of_the_pretrain_half]
    fine = refuse_barred([*rounds, *rehearsal], {"held"}, 0, "the training set of `rehearse`")
    assert fine == {"rows": 14, "rows_of_the_rounds": 8, "rehearsal_rows": 6, "held_out": [], "of_the_pretrain_half_among_the_rounds_rows": [], "rehearsal_rows_not_of_the_pretrain_half": []}
    assert barred([*rounds, *rehearsal], {"held"}, 0) == fine
    # THE REHEARSAL ROWS ARE OF THE `pretrain` HALF, and that is what they must be; a ROUND's row of that half is barred.
    for rows, heldout, match in (
            ([*rounds, _attempt("held")], {"held"}, "1 problems of the training set of `rehearse` are held-out problems .first: held.: refused, nothing was written"),
            ([*rounds, *rehearsal, {**rehearsal[0], "problem_id": "held"}], {"held"}, "are held-out problems .first: held"),
            ([*rounds, _attempt(OF_THE_OTHER_HALF)], set(), f"1 problems of the training set of `rehearse` are rounds' rows of the `pretrain` half .first: {OF_THE_OTHER_HALF}"),
            ([*rounds, {**rehearsal[0], "problem_id": LOOP_HALF[0]}], set(), f"are rehearsal rows that are not of the `pretrain` half .first: {LOOP_HALF[0]}")):
        with pytest.raises(ValueError, match=match):
            refuse_barred(rows, heldout, 0, "the training set of `rehearse`")
    # The fourth check reads the rows the training loop recorded, and names the rehearsal rows as what they are.
    check = no_barred_row([*rounds, *rehearsal], {"held"}, 0)
    assert check["passes"] is True and (check["rows"], check["rows_of_the_rounds"], check["rehearsal_rows_of_the_pretrain_half"]) == (14, 8, 6)
    failed = no_barred_row([*rounds, _attempt(OF_THE_OTHER_HALF), _attempt("held")], {"held"}, 0)
    assert failed["passes"] is False and (failed["held_out_problems"], failed["problems_of_the_pretrain_half_among_the_rounds_rows"]) == (1, 1) and failed["first"] == ["held", OF_THE_OTHER_HALF]
    assert no_barred_row([], {"held"}, 0)["passes"] is False                                       # a training with no recorded row does not pass


# ----------------------------------------------------------------------------------------------- the branch
def test_the_smallest_decisive_split_is_the_least_number_gained_at_which_the_sign_test_is_under_the_level():
    for changed in range(0, 6):
        assert smallest_decisive_split(changed)["gained"] is None                                  # five problems all one way: p = 0.0625
    assert sign_test(5, 0) == 0.0625 and sign_test(6, 0) == 0.03125
    assert smallest_decisive_split(6) == {"changed": 6, "gained": 6, "lost": 0, "sign_test_p": 0.03125, "level": 0.05}
    for changed in (9, 20, 67, 100):
        split = smallest_decisive_split(changed)
        gained = split["gained"]
        assert gained > changed - gained and split["lost"] == changed - gained and sign_test(gained, changed - gained) == split["sign_test_p"] < 0.05
        assert sign_test(gained - 1, changed - gained + 1) >= 0.05                                 # one fewer gained is not decisive
    assert (smallest_decisive_split(67)["gained"], smallest_decisive_split(67)["lost"]) == (42, 25)       # the arm's own 28 gained and 39 lost: 67 changed


PASSING = {"its_training_ran": {"passes": True}, "it_still_writes_proofs": {"passes": True}, "lean_answered": {"passes": True}, "no_barred_row": {"passes": True}}


def _breadth(gained, lost, problems=392):
    return {"problems": problems, "gained": gained, "lost": lost, "sign_test_p": sign_test(gained, lost)}


def _kept(mean, low, high):
    return {"problems": 392, "mean": mean, "low": low, "high": high}


def test_the_branch_is_the_specs():
    assert BRANCHES == (GIVES_BACK, UNDOES, NOT_THIS_LEVER, NOT_NAMED, INCONCLUSIVE, NOT_READ)
    above, through, below = _kept(0.0114, 0.0026, 0.0203), _kept(0.004, -0.002, 0.011), _kept(-0.01, -0.02, -0.001)
    # Gained more than lost with the sign test under 0.05 AND the interval for all of G against `pre` above zero.
    both = rows_branch(PASSING, _breadth(40, 20), above)
    assert both["name"] == GIVES_BACK == "THE RULE GIVES THE BREADTH BACK AND KEEPS THE GAIN" and (both["the_first"], both["the_second"]) == (True, True)
    assert both["reason"].endswith("the rule gives the breadth back and keeps the gain; it replaces the old rule in the next arm") and "gained 40, lost 20, p = 0.0134893" in both["reason"]
    # The first and not the second (an interval that holds zero, or lies below it).
    for kept in (through, below, _kept(0.004, 0.0, 0.011)):                                        # an interval that ENDS at zero is not above it
        first = rows_branch(PASSING, _breadth(40, 20), kept)
        assert first["name"] == UNDOES and first["reason"].endswith("it gives the breadth back by undoing the training, and is not taken")
    # Neither: with the count that would have been seen.
    for breadth in (_breadth(28, 39), _breadth(34, 33), _breadth(36, 24)):                         # lost more; about even; gained more, p = 0.155
        neither = rows_branch(PASSING, breadth, through)
        changed = breadth["gained"] + breadth["lost"]
        assert neither["name"] == NOT_THIS_LEVER and neither["reason"].startswith("neither (") and "not this lever at this size" in neither["reason"]
        assert neither["smallest_decisive_split"] == smallest_decisive_split(changed)
        assert f"with the {changed} problems that changed, the sign test's smallest decisive split is {neither['smallest_decisive_split']['gained']} gained to" in neither["reason"]
    assert "smallest decisive split is 42 gained to 25 lost" in rows_branch(PASSING, _breadth(28, 39), through)["reason"]
    # The sign test must be UNDER 0.05, and gained must be MORE than lost: the mirror split is not the first.
    assert rows_branch(PASSING, _breadth(20, 40), above)["the_first"] is False and sign_test(20, 40) < 0.05
    assert sign_test(39, 23) > 0.05 > sign_test(40, 23) and rows_branch(PASSING, _breadth(39, 23), above)["name"] == NOT_NAMED and rows_branch(PASSING, _breadth(40, 23), above)["name"] == GIVES_BACK
    # THE SECOND AND NOT THE FIRST is a case the spec's branches do not name: reported as that, with the same count.
    second = rows_branch(PASSING, _breadth(28, 39), above)
    assert second["name"] == NOT_NAMED and (second["the_first"], second["the_second"]) == (False, True) and "it keeps the gain and does not give the breadth back" in second["reason"] and second["name"] == "IT KEEPS THE GAIN AND DOES NOT GIVE THE BREADTH BACK"
    assert "smallest decisive split is 42 gained to 25 lost" in second["reason"]
    # No problem changed (the same rows on both sides): nothing could have been seen, and it says so.
    same = rows_branch(PASSING, _breadth(0, 0), through)
    assert same["name"] == NOT_THIS_LEVER and "gained 0, lost 0, no problem changed" in same["reason"] and "with the 0 problems that changed, no split is decisive at 0.05" in same["reason"]
    # A failed check: INCONCLUSIVE, by itself, whatever was measured.
    failed = rows_branch({**PASSING, "lean_answered": {"passes": False}, "no_barred_row": {"passes": False}}, _breadth(40, 20), above)
    assert failed["name"] == INCONCLUSIVE and failed["failed_checks"] == ["lean_answered", "no_barred_row"] and "The other models are read by themselves" in failed["reason"]
    assert rows_branch(PASSING, {"problems": 0, "gained": 0, "lost": 0, "sign_test_p": None}, {"mean": None})["name"] == NOT_READ


# ------------------------------------------------------------------------------------------------ the report
SAMPLINGS = [{"name": "reach", "episodes": 32, "sampling_seed": 1001}, {"name": "more", "episodes": 61, "sampling_seed": 1020}]
HOT_SAMPLINGS = [{"name": "reach", "episodes": 32, "sampling_seed": 1040}, {"name": "more", "episodes": 61, "sampling_seed": 1041}]
PREPARE = {"stage": "l4", "label": LABEL, "seed": 0, "start": "pre", "start_adapter": "runs/ladder_l4_pretrain_seed0/adapters/pre", "arm": "t010_assembly_pre",
           "arm_run": "runs/ladder_l2_t010_assembly_pre_seed0", "pretraining_run": "runs/ladder_l4_pretrain_seed0", "target_rate": 0.1, "stand_in_engine": False,
           "training_sets": {"rehearse": {"rows": 60, "twin_rows": 30, "rehearsal_rows": 30}, "reward_rows": {"rows": 24, "rows_of_the_rounds": 40, "kept_by_the_rule": 24,
                                                                                                           "kept_by_the_runs_minimum": 0}},
           "hot": {"temperature": 1.2, "temperature_of_every_other_measurement": 1.0, "models": ["with", "pre"], "samplings": HOT_SAMPLINGS, "attempts_a_goal_problem": 93},
           "goal_samplings": SAMPLINGS, "attempts_a_goal_problem": 93, "rung_episodes": 8, "sampling_seeds": {"rungs": 1002, "reach": 1001, "more": 1020},
           "contradicted_side_setting": "audit", "stored_models": {name: {"run": f"runs/{name}"} for name in ("pre", "with", "without")}}
FALLING = lambda count: [2.0 - index / count for index in range(count)]      # noqa: E731
TRAINS = {name: {"model": name, "rows": rows, "steps": -(-rows // 8), "stand_in_engine": False, "trained_from": "the stored adapter pre", "against_the_start_adapter": {name: CHANGED}}
          for name, rows in (("rehearse", 60), ("reward_rows", 24))}
LOSSES = {"rehearse": FALLING(60), "reward_rows": FALLING(24)}
PRETRAIN_HALF = [f"p{index}" for index in range(400) if half_of(f"p{index}", 0) == PRETRAIN]
TRAINED = {"rehearse": [*({"id": f"{name}#a", "problem_id": name, "origin": "attempt", "k": 1 + index % 4, "lines": 1 + index % 9} for index, name in enumerate(LOOP_HALF[:30])),
                        *({"id": f"{name}#pretraining", "problem_id": name, "origin": "pretraining", "k": None, "lines": 4 + index % 9} for index, name in enumerate(PRETRAIN_HALF[:30]))],
           "reward_rows": [{"id": f"{name}#a", "problem_id": name, "origin": "assembled" if index < 4 else "attempt", "k": 1 if index < 16 else 2, "lines": 2 + index % 7}
                           for index, name in enumerate(LOOP_HALF[:24])]}
STORED = {name: _made(name) for name in ("pre", "with", "without")}
BROADER = ([6, 6, 6, 4, 4, 4, 3, 2, 1, 1, 1, 1], [9, 9, 9, 6, 6, 6, 4, 3, 3, 2, 1, 1])             # every goal problem solved: `without` leaves three unsolved
HEADS = lambda lines: [line.split(": ", 1)[1].split(".")[0] for line in lines]      # noqa: E731


def _model(on_g=None, name="without", **changed):
    return {**_made(name, on_g=on_g), "stand_in_engine": False, **changed}


def _hot(name, on_g=None):
    return {key: value for key, value in _model(on_g, name).items() if key != "rungs"}


def _report(models=None, hot=None, stored=STORED, prepare=PREPARE, trains=TRAINS, losses=LOSSES, trained=TRAINED, never=None, again=AGAIN):
    models = models or {name: _model() for name in MODELS}
    return build_rows_report(prepare, trains, losses, trained, GROUPS, LENGTHS, stored, models, hot or {"with": _hot("with"), "pre": _hot("pre")}, again, SETTINGS, EVALUATION, never)


def test_stored_rows_standing_in_for_a_model_read_gained_0_and_lost_0_against_without_and_the_same_interval_as_without():
    report = _report()
    assert report["label"] == LABEL and report["stage"] == "l4" and report["check"] == "L4t" and report["ok"] is True and all(line.startswith(f"{SAY}: ") for line in report["lines"])
    for name in MODELS:
        read = report["models"][name]
        assert (read["primary"]["gained"], read["primary"]["lost"], read["primary"]["sign_test_p"]) == (0, 0, None) and read["primary"]["problems"] == 12
        assert {key: read["beside_the_primary"][key] for key in ("mean", "low", "high")} == {key: report["what_the_old_rule_gave"]["without_minus_pre"][key] for key in ("mean", "low", "high")}
        assert read["beside_the_primary"]["of_without"] == report["what_the_old_rule_gave"]["without_minus_pre"]
        assert all(entry["mean"] == entry["low"] == entry["high"] == 0 for entry in read["secondary"]["the_three_rungs"]["minus_without"].values())
        assert read["branch"]["name"] in (NOT_THIS_LEVER, NOT_NAMED) and report["branches"][name] == read["branch"]["name"]
    # `hot` standing in by the stored rows themselves: the two temperatures read the same.
    for name in ("with", "pre"):
        entry = report["hot"][name]
        assert entry["hot"]["all_of_g"] == entry["stored"]["all_of_g"] and (entry["hot_against_stored"]["solved"]["gained"], entry["hot_against_stored"]["solved"]["lost"]) == (0, 0)
        assert entry["hot_against_stored"]["per_attempt"]["mean"] == 0


def test_each_model_is_read_by_itself_the_checks_first_then_the_primary_what_stands_beside_it_the_branch_and_the_secondary_reads():
    report = _report({"rehearse": _model(BROADER), "reward_rows": _model()}, never=["g8a", "g8c", "not_a_goal_problem"])
    lines = report["lines"]
    assert lines[0].startswith(f"{SAY}: L4t, WHAT SHOULD A ROUND TRAIN ON? Three one-change checks on the rounds already made, seed 0. NO NEW ROUND: from the stored rounds of the arm "
                               "t010_assembly_pre") and "`rehearse`: 60 rows (30 of the twin and as many published proofs of the pretraining file)" in lines[0]
    assert "`reward_rows`: 24 of the rounds' 40 rows, each kept with the probability of its problem's reward at t = 0.1" in lines[0] and "by a smoke run's minimum" not in lines[0]
    assert lines[1].startswith(f"{SAY}: from the stored rows, what the old rule gave, on all of G over 93 attempts a problem: `pre` 9 of 12 solved")
    of = lambda name: [line for line in lines if line.startswith(f"{SAY}: `{name}`")]      # noqa: E731
    for name in MODELS:
        own = of(name)
        assert all(f", CHECK {number}, " in line for number, line in enumerate(own[:4], start=1)) and len(own) == 17
        assert "PRIMARY, breadth" in own[4] and "BESIDE IT, reliability kept" in own[5] and f"`{name}`: BRANCH: " in own[6] and all(f"`{name}`, SECONDARY, " in line for line in own[7:])
        assert all(line.endswith(": PASS") for line in own[:4])
    rehearse, reward_rows = report["models"]["rehearse"], report["models"]["reward_rows"]
    # THE PRIMARY: goal problems solved at least once in the 93 attempts, the model against `without`: gained, lost, the two-sided sign test.
    without_solves = sum(first + more > 0 for first, more in zip(*[[row["resolved"] for row in sampling] for sampling in STORED["without"]["goal"]]))
    assert without_solves == 10 and (rehearse["primary"]["resolved_after"], rehearse["primary"]["resolved_before"], rehearse["primary"]["gained"], rehearse["primary"]["lost"]) == (12, 10, 2, 0)
    assert rehearse["primary"]["sign_test_p"] == sign_test(2, 0) == 0.5 and rehearse["primary"]["gained_problems"] == ["g8b", "g8c"]
    assert "PRIMARY, breadth. The goal problems solved at least once in the 93 attempts, `rehearse` against `without`, paired by problem: 12 to 10 of 12, gained 2, lost 0 (p = 0.5)" in of("rehearse")[4]
    # BESIDE IT: all of G, successes per attempt over the 93 attempts, the model minus `pre`; `without`'s own stands beside.
    beside = rehearse["beside_the_primary"]
    assert (beside["successes"], beside["successes_of_the_base"], beside["attempts_each"]) == (sum(BROADER[0]) + sum(BROADER[1]), 84, 12 * 93)
    assert "`rehearse` minus `pre`, 95% bootstrap over problems: " in of("rehearse")[5] and "; `without`'s is " in of("rehearse")[5]
    # THE BRANCH: two problems changed, and no split of two is decisive.
    assert rehearse["branch"]["name"] in (NOT_THIS_LEVER, NOT_NAMED) and rehearse["branch"]["smallest_decisive_split"]["gained"] is None
    assert "with the 2 problems that changed, no split is decisive at 0.05" in rehearse["branch"]["reason"]
    # THE SECONDARY reads.
    secondary = rehearse["secondary"]
    again = secondary["the_goal_set_again"]
    assert again["goal_set_again"] == 4 == again["problems"] and again["attempts_each"] == 4 * 61 and again["successes_of_the_base"] == sum(STORED["pre"]["goal"][1][GOAL_IDS.index(name)]["resolved"] for name in AGAIN)
    assert "G' (4 goal problems `pre` does not solve in its first sampling), successes per attempt over the 61 attempts of the second sampling, `rehearse` minus `pre` (the arm's primary)" in of("rehearse")[7]
    assert list(secondary["by_length_group"]["minus_pre"]) == ["1", "2-3", "4-7", "8+", "4_or_more"] and secondary["by_length_group"]["against_without"]["8+"]["gained"] == 2
    assert set(secondary["goal_problems_solved_by_attempts_alone"]) == {"what", "pre", "with", "without", "rehearse"} and set(secondary["the_three_rungs"]["minus_pre"]) == {"below", "in", "above"}
    assert set(secondary["distinct_attempts"]) == set(secondary["verified_proof_lines"]) == {"what", "pre", "with", "without", "rehearse"}
    # The goal problems nothing stored had solved are GIVEN: two of the three ids are goal problems; `rehearse` solves both, `without` one.
    unsolved = secondary["never_solved_before"]
    assert (unsolved["given"], unsolved["ids"], unsolved["goal_problems"], unsolved["not_goal_problems_of_this_run"], unsolved["solved"], unsolved["problem_ids"]) == (True, 3, 2, 1, 2, ["g8a", "g8c"])
    assert (unsolved["solved_by_with"], unsolved["solved_by_without"], reward_rows["secondary"]["never_solved_before"]["solved"]) == (1, 1, 1)
    assert "of the 2 listed, `rehearse` solves 2 (g8a, g8c); `with` solves 1 and `without` 1; 1 more ids of the file are not goal problems of this run" in of("rehearse")[15]
    # The rows it was trained on, from the training loop's own record.
    record = secondary["trained_on"]
    assert (record["rows"], record["rows_by_origin"], record["rehearsal_rows"]) == (60, {"attempt": 30, "pretraining": 30}, 30) and sum(record["rows_by_k"].values()) == 30
    assert record == trained_on(TRAINED["rehearse"]) and record["lines"]["proofs"] == 60
    assert reward_rows["secondary"]["trained_on"]["rows_by_origin"] == {"assembled": 4, "attempt": 20} and reward_rows["secondary"]["trained_on"]["rows_by_k"] == {"1": 16, "2": 8}
    assert "the rows it was trained on: 60, by origin attempt 30, pretraining 30; by k 1: 8, 2: 8, 3: 7, 4: 7; their proofs: 60 proofs" in of("rehearse")[16]
    # CHECK 4 names the rehearsal rows as what they are.
    assert "its 30 rehearsal rows ARE of the `pretrain` half (published proofs `pre` was trained on), and are named as such: PASS" in of("rehearse")[3] and "rehearsal rows" not in of("reward_rows")[3]
    assert report["what_it_cannot_say"] == CANNOT_SAY and len(CANNOT_SAY) == 3 and "the training rule alone and not the loop under it" in CANNOT_SAY[0]
    assert report["headline"].startswith("L4t (what should a round train on; pretrained on published proofs) seed 0, from `pre`") and "`hot` at 1.2, read and not branched" in report["headline"]


def test_the_branches_follow_the_two_reads_and_rehearse_says_so_when_it_beats_pre_on_the_goal_set_again():
    # Eight more goal problems than a `without` that solves four, and more successes per attempt than `pre`: BOTH.
    narrow = ([6, 6, 6, 4, 0, 0, 0, 0, 0, 0, 0, 0], [9, 9, 9, 6, 0, 0, 0, 0, 0, 0, 0, 0])
    much_broader = ([8, 8, 8, 6, 6, 6, 4, 3, 2, 2, 2, 2], [12, 12, 12, 9, 9, 9, 6, 5, 4, 4, 3, 3])
    stored = {**STORED, "without": _made("without", on_g=narrow)}
    report = _report({"rehearse": _model(much_broader), "reward_rows": _model(narrow)}, stored=stored)
    rehearse = report["models"]["rehearse"]
    assert (rehearse["primary"]["gained"], rehearse["primary"]["lost"], rehearse["primary"]["sign_test_p"]) == (8, 0, sign_test(8, 0)) and sign_test(8, 0) < 0.05
    assert rehearse["beside_the_primary"]["low"] > 0 and rehearse["branch"]["name"] == GIVES_BACK and report["branches"] == {"rehearse": GIVES_BACK, "reward_rows": rehearse["branch"]["name"] and report["branches"]["reward_rows"]}
    assert any(line.startswith(f"{SAY}: `rehearse`: BRANCH: THE RULE GIVES THE BREADTH BACK AND KEEPS THE GAIN. ") for line in report["lines"])
    # `rehearse` beats `pre` on G' with an interval above zero: the note says what else would show exactly this.
    again = rehearse["secondary"]["the_goal_set_again"]
    assert again["low"] > 0 and "It trains again on published proofs `pre` has seen: a model that only gains from seeing them twice would show exactly this" in again["seen_twice"]
    assert any("NOTE: `rehearse` beats `pre` on G' with an interval above zero" in line for line in report["lines"])
    # The same rows as `reward_rows`: the note is `rehearse`'s alone.
    other = _report({"rehearse": _model(narrow), "reward_rows": _model(much_broader)}, stored=stored)
    assert other["branches"]["reward_rows"] == GIVES_BACK and "seen_twice" not in other["models"]["reward_rows"]["secondary"]["the_goal_set_again"]
    assert not any("NOTE: `rehearse` beats" in line for line in other["lines"])
    # Broader, and no more reliable than `pre`: the breadth is given back by undoing the training.
    thin = ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    undone = _report({"rehearse": _model(thin), "reward_rows": _model(narrow)}, stored=stored)
    assert undone["models"]["rehearse"]["beside_the_primary"]["high"] < 0 and undone["branches"]["rehearse"] == UNDOES
    assert all(name in BRANCHES for built in (report, other, undone) for name in built["branches"].values())


def test_a_model_whose_check_fails_is_inconclusive_by_itself_and_the_other_is_still_read():
    barred_row = [{**TRAINED["reward_rows"][0], "problem_id": OF_THE_OTHER_HALF}, *TRAINED["reward_rows"][1:]]
    report = _report({"rehearse": _model(BROADER), "reward_rows": _model()}, trained={**TRAINED, "reward_rows": barred_row})
    broken, fine = report["models"]["reward_rows"], report["models"]["rehearse"]
    assert broken["inconclusive"] is True and broken["branch"]["name"] == INCONCLUSIVE and broken["branch"]["failed_checks"] == ["no_barred_row"]
    assert broken["primary"] is None and broken["beside_the_primary"] is None and broken["secondary"] is None and broken["measured_and_not_read"]["primary"]["problems"] == 12
    own = [line for line in report["lines"] if line.startswith(f"{SAY}: `reward_rows`")]
    assert len(own) == 5 and own[3].endswith(": FAIL") and own[4].startswith(f"{SAY}: `reward_rows`: INCONCLUSIVE. this model could not have been read (no_barred_row)")
    assert not any("PRIMARY" in line or "SECONDARY" in line for line in own)
    # THE OTHER MODEL IS STILL READ, and the report is to be read.
    assert fine["inconclusive"] is False and fine["primary"]["gained"] == 2 and report["ok"] is True and report["branches"]["reward_rows"] == INCONCLUSIVE
    assert len([line for line in report["lines"] if line.startswith(f"{SAY}: `rehearse`")]) == 17
    # Each check fails by itself: an adapter that is the start adapter's; a loss that rose; a model that stopped writing proofs.
    same = {**TRAINS, "rehearse": {**TRAINS["rehearse"], "against_the_start_adapter": {"rehearse": {**CHANGED, "elements_changed": 0}}}}
    assert _report(trains=same)["models"]["rehearse"]["branch"]["failed_checks"] == ["its_training_ran"]
    assert _report(trains={**TRAINS, "rehearse": {key: value for key, value in TRAINS["rehearse"].items() if key != "against_the_start_adapter"}})["branches"]["rehearse"] == INCONCLUSIVE
    assert _report(losses={**LOSSES, "rehearse": list(reversed(FALLING(60)))})["models"]["rehearse"]["branch"]["failed_checks"] == ["its_training_ran"]
    silent = _model()
    silent["rungs"] = [{**row, "attempts_capped": 4} for row in silent["rungs"]]
    stopped = _report({"rehearse": silent, "reward_rows": _model()})
    assert stopped["models"]["rehearse"]["branch"]["failed_checks"] == ["it_still_writes_proofs"] and stopped["branches"]["reward_rows"] != INCONCLUSIVE


def test_a_set_lean_did_not_answer_is_not_to_be_read_and_fails_the_models_that_stand_on_it():
    unanswered = _model()
    unanswered["goal"][1] = [{**row, "attempts_without_an_answer": 20} for row in unanswered["goal"][1]]
    report = _report({"rehearse": unanswered, "reward_rows": _model()})
    assert report["ok"] is False and report["not_to_be_read"] == ["goal_rehearse_2"] and "NOT TO BE READ: too many attempts without a verdict from Lean in goal_rehearse_2" in report["lines"][-1]
    assert report["models"]["rehearse"]["branch"]["failed_checks"] == ["lean_answered"] and report["models"]["reward_rows"]["inconclusive"] is False
    # A hot set Lean did not answer: the report is not to be read, the table says which, and no trained model's read stands on it.
    hot = _hot("with")
    hot["goal"][0] = [{**row, "attempts_without_an_answer": 20} for row in hot["goal"][0]]
    report = _report(hot={"with": hot, "pre": _hot("pre")})
    assert report["ok"] is False and report["not_to_be_read"] == ["goal_hot_with_1"] and report["hot"]["with"]["not_to_be_read"] == ["goal_hot_with_1"] and report["hot"]["pre"]["not_to_be_read"] == []
    assert not any(read["inconclusive"] for read in report["models"].values())
    # A stored set of `without`: both trained models are read against it, and both are INCONCLUSIVE.
    without = _made("without")
    without["rungs"] = [{**row, "attempts_without_an_answer": 4} for row in without["rungs"]]
    report = _report(stored={**STORED, "without": without})
    assert report["not_to_be_read"] == ["rungs_without"] and all(read["branch"]["failed_checks"] == ["lean_answered"] for read in report["models"].values())


def test_the_table_of_hot_gives_both_models_at_both_temperatures_on_g_and_on_the_goal_set_again():
    hotter_with = ([5, 5, 5, 3, 3, 3, 3, 2, 1, 1, 1, 1], [8, 8, 8, 5, 5, 5, 4, 3, 3, 2, 1, 1])      # at the hot temperature `with` solves every goal problem, with fewer successes
    report = _report(hot={"with": _hot("with", hotter_with), "pre": _hot("pre")})
    table = report["hot"]
    assert (table["temperature"], table["temperature_of_the_stored_rows"], table["samplings"]) == (1.2, 1.0, HOT_SAMPLINGS) and "read, not branched" in table["what"]
    entry = table["with"]
    stored_with = [first + more for first, more in zip(*[[row["resolved"] for row in sampling] for sampling in STORED["with"]["goal"]])]
    assert entry["stored"]["all_of_g"] == {"problems": 12, "solved_at_least_once": sum(count > 0 for count in stored_with), "successes": sum(stored_with), "attempts": 12 * 93,
                                           "per_1000": round(1000 * sum(stored_with) / (12 * 93), 2)} and entry["stored"]["temperature"] == 1.0
    assert entry["hot"]["all_of_g"] == {"problems": 12, "solved_at_least_once": 12, "successes": sum(hotter_with[0]) + sum(hotter_with[1]), "attempts": 12 * 93,
                                        "per_1000": round(1000 * (sum(hotter_with[0]) + sum(hotter_with[1])) / (12 * 93), 2)} and entry["hot"]["temperature"] == 1.2
    assert entry["hot"]["all_of_g"] == on_a_set([{**row, "resolved": first + more, "episodes": 93} for row, first, more in zip(STORED["with"]["goal"][0], *hotter_with)], GOAL_IDS)
    # ... the same on G', and over the second sampling alone (at 1.0 `pre`'s first sampling chose G').
    assert entry["hot"]["the_goal_set_again"]["problems"] == 4 and entry["hot"]["the_goal_set_again"]["attempts"] == 4 * 93
    assert entry["hot"]["the_goal_set_again_over_the_second_sampling"]["attempts"] == 4 * 61 and table["pre"]["stored"]["the_goal_set_again"]["successes"] == sum(
        row["resolved"] for row in STORED["pre"]["goal"][1] if row["problem_id"] in AGAIN)                                 # none of `pre`'s first sampling
    assert entry["hot"]["distinct_attempts_on_g"] == entry["stored"]["distinct_attempts_on_g"] == {"prompts": 12, "mean_share_distinct": 0.95}
    changed = entry["hot_against_stored"]
    assert (changed["solved"]["gained"], changed["solved"]["lost"]) == (12 - sum(count > 0 for count in stored_with), 0) and changed["per_attempt"]["mean"] < 0
    lines = [line for line in report["lines"] if line.startswith(f"{SAY}: `hot`")]
    assert len(lines) == 4 and lines[0].startswith(f"{SAY}: `hot`, read and not branched. `with` on all of G (12 problems, 93 attempts a problem at 1.2, 93 at 1.0): at 1.0 (stored) ")
    assert "; at 1.2 12 of 12 solved, " in lines[0] and "the same on G' (4 problems; at 1.0 `pre`'s first sampling CHOSE them" in lines[1] and "`pre` on all of G" in lines[2]
    assert "over the second sampling alone, at 1.0 " in lines[1]
    assert f"`with` solves 12 goal problems against {sum(count > 0 for count in stored_with)} at 1.0" in report["headline"]
    assert not any("BRANCH" in line for line in lines)                                             # read, not branched


def test_a_smoke_run_has_one_sampling_and_says_what_its_minimum_kept():
    one = lambda model: {**model, "goal": model["goal"][:1]}      # noqa: E731
    prepare = {**PREPARE, "goal_samplings": SAMPLINGS[:1], "attempts_a_goal_problem": 32,
               "hot": {**PREPARE["hot"], "samplings": HOT_SAMPLINGS[:1], "attempts_a_goal_problem": 32},
               "training_sets": {**PREPARE["training_sets"], "reward_rows": {"rows": 2, "rows_of_the_rounds": 3, "kept_by_the_rule": 0, "kept_by_the_runs_minimum": 2}}}
    report = _report({name: one(_model()) for name in MODELS}, hot={name: one(_hot(name)) for name in ("with", "pre")}, stored={name: one(model) for name, model in STORED.items()},
                     prepare=prepare)
    assert "(2 of them by a smoke run's minimum and NOT by the rule)" in report["lines"][0] and "the stored models have ONE sampling of G here (a smoke run)" in report["lines"][2]
    for name in MODELS:
        read = report["models"][name]
        assert read["secondary"]["the_goal_set_again"]["mean"] is None and read["primary"]["episodes_after"] == 32 and read["beside_the_primary"]["attempts_each"] == 12 * 32
    assert report["hot"]["with"]["hot"]["the_goal_set_again_over_the_second_sampling"] is None and report["ok"] is True
    # A world with no goal problem: the primary has nothing to be read on.
    groups = [row for row in GROUPS if row["group"] != "goal"]
    empty = build_rows_report(PREPARE, TRAINS, LOSSES, TRAINED, groups, LENGTHS, STORED, {name: _model() for name in MODELS}, {"with": _hot("with"), "pre": _hot("pre")}, [],
                              SETTINGS, EVALUATION)
    assert set(empty["branches"].values()) == {NOT_READ}
    with pytest.raises(ValueError, match="1 problems of G' are not goal problems of this run .first: nowhere."):
        _report(again=[*AGAIN, "nowhere"])


def test_every_goal_problem_is_a_row_of_the_hand_made_world():
    assert len(GOAL_IDS) == 12 and all(_rung(name, 1)["problem_id"] == name for name in GOAL_IDS) and set(AGAIN) < set(GOAL_IDS)
