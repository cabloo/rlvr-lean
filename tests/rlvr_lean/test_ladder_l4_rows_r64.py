"""L4t FROM ANOTHER START THAN `pre`: the model of the check of the adapter's rank, `pre_r64`. The start resolved to its
run and its adapter; the rank and alpha of the START ADAPTER (never the config's) reaching every training, each saved
adapter read back and held to them, and the model server's largest adapter rank raised for this stage's measure steps
alone; the third trained model `old_rule` (the twin's rows, in the twin's order), FIRST; the references of every read
switched (`old_rule` in `without`'s place, the start model in `pre`'s); G' made for the start model; `old_rule` read by
itself, with its three named outcomes; the refusals; the stage end to end in the scripted world from a rank-64 start; its
smoke stage; and a run from `pre` left as it was. Spec: docs/spec/ladder-loop.spec.md, "L4t: what should
a round train on?", "L4r's first branch was taken: what L4t runs". The engine and Lean are scripted; nothing touches a GPU
or the network."""

import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_assembly import arm  # noqa: E402, F401
from test_ladder_l2_stage import ARM, loop  # noqa: E402, F401
from test_ladder_l3d1 import EVALUATION, GOAL_IDS, GROUPS, LENGTHS  # noqa: E402 - Step 1's hand-made world: 12 goal problems, 3 a length group
from test_ladder_l4 import LOOP_HALF, OF_THE_OTHER_HALF, _made  # noqa: E402 - L4's hand-made `pre`, `with` and `without`
from test_ladder_l4_rank_stage import _adapter, _check  # noqa: E402 - an adapter's two files written by hand; the check's own steps
from test_ladder_l4_rows import FALLING, HOT_SAMPLINGS, PASSING, SAMPLINGS, _breadth, _kept  # noqa: E402
from test_ladder_l4_rows_stage import _all, rows  # noqa: E402, F401 - a run from `pre`, as it always was
from test_ladder_l4_stage import CHANGED, FIXTURE, _file_rows, _on_the_gpu, _run, l4  # noqa: E402, F401
from test_ladder_round import _run_stage, stage  # noqa: E402, F401

from rlvr_lean.domain.ladder_round.dose import run_dose  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ, PRETRAIN, goal_set_again, half_of  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows import (  # noqa: E402
    BRANCHES,
    GIVES_BACK,
    MODELS,
    MODELS_FROM_ANOTHER_START,
    NOT_NAMED,
    NOT_READ_AS_EITHER,
    NOT_THIS_LEVER,
    UNDOES,
    models_of,
    narrows,
    no_narrowing,
    old_rule_branch,
    old_rule_outcomes,
    reference_of,
    rows_branch,
    smallest_decisive_split,
)
from rlvr_lean.domain.ladder_round.rounds import sign_test  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import ladder_ceiling, ladder_dose, ladder_l2, ladder_l3d2, ladder_l4, ladder_l4_rank, ladder_l4_rows, ladder_l4_start, ladder_loop, ladder_round, milestone2, pipeline  # noqa: E402
from rlvr_lean.gpu.ladder_ceiling import MORE, REACH  # noqa: E402
from rlvr_lean.gpu.stand_in_engine import stand_in_parameters  # noqa: E402
from rlvr_lean.reporting.ladder_l4 import LABEL, SAY  # noqa: E402
from rlvr_lean.reporting.ladder_l4_rows import CANNOT_SAY, build_rows_report, old_rule_figures  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
SETTINGS = CONFIG["ladder_loop"]["l4"]
START, START_CHECKS = ladder_l4_rows.START_VARIABLE, ladder_l4_rows.START_CHECKS_VARIABLE
RUN, ARM_RUN, MINIMUM = ladder_l4_rows.RUN_VARIABLE, ladder_l4_rows.ARM_RUN_VARIABLE, ladder_l4_rows.MINIMUM_VARIABLE
CHECK, RANK_RUN = ladder_l4_rank.CHECK_VARIABLE, ladder_l4_rank.RUN_VARIABLE
SEEDS = "RLVR_LEAN_TRAINING_SEEDS"
STEPS = ["ladder_l4_rows_prepare", "ladder_l4_rows_train_old_rule", "ladder_l4_rows_measure_old_rule", "ladder_l4_rows_train_reward_rows", "ladder_l4_rows_measure_reward_rows",
         "ladder_l4_rows_train_rehearse", "ladder_l4_rows_measure_rehearse", "ladder_l4_rows_measure_hot_old_rule", "ladder_l4_rows_measure_hot_start", "ladder_l4_rows_report"]
TRAININGS = ("ladder_l4_rows_train_old_rule", "ladder_l4_rows_train_reward_rows", "ladder_l4_rows_train_rehearse")
RANK_64 = {"rank": 64, "alpha": 128}
# What a run from `pre_r64` leaves in its run directory, beside the episodes of the sets it samples.
OWN_FILES = [
    *(f"adapters/{model}/{name}" for model in ("old_rule", "rehearse", "reward_rows") for name in ("adapter_config.json", "adapter_model.safetensors")),
    "l4_goal_set_again.jsonl", "l4_heldout_groups.jsonl", "l4_rows_loss_old_rule.json", "l4_rows_loss_rehearse.json", "l4_rows_loss_reward_rows.json", "l4_rows_reward_draws.jsonl",
    "l4_rows_stored_models.json", "l4_rows_training_old_rule.jsonl", "l4_rows_training_rehearse.jsonl", "l4_rows_training_reward_rows.jsonl", "l4_stored_pre_more.jsonl",
    "l4_stored_pre_r64_more.jsonl", "l4_stored_pre_r64_reach.jsonl", "l4_stored_pre_r64_rungs.jsonl", "l4_stored_pre_reach.jsonl", "l4_stored_without_more.jsonl",
    "l4_stored_without_reach.jsonl", "ladder_l4_rows_measure_hot_old_rule.done.json", "ladder_l4_rows_measure_hot_start.done.json", "ladder_l4_rows_measure_old_rule.done.json",
    "ladder_l4_rows_measure_rehearse.done.json", "ladder_l4_rows_measure_reward_rows.done.json", "ladder_l4_rows_prepare.done.json", "ladder_l4_rows_report.done.json",
    "ladder_l4_rows_train_old_rule.done.json", "ladder_l4_rows_train_rehearse.done.json", "ladder_l4_rows_train_reward_rows.done.json", "problems.jsonl", "report_ladder_l4_rows.json"]
# A run from `pre` is what it was before another start was built: the keys of what its prepare step and its report hold, in their order.
PREPARE_KEYS_FROM_PRE = ["stage", "label", "check", "seed", "start", "start_adapter", "arm", "arm_run", "pretraining_run", "with_adapter", "rounds", "batches", "solvers", "target_rate",
                         "models", "stand_in_engine", "effective_batch", "training_sets", "pretraining_file", "pretraining_file_sha256", "recipe", "hot", "goal_set", "goal_set_again",
                         "rung_problems", "sampling_seeds", "rung_episodes", "goal_samplings", "attempts_a_goal_problem", "parts", "stored_models", "never_solved_file",
                         "contradicted_side_setting"]
REPORT_KEYS_FROM_PRE = ["spec", "stage", "label", "check", "headline", "lines", "ok", "seed", "stand_in_engine", "branches", "models", "what_the_old_rule_gave", "hot",
                        "what_it_cannot_say", "the_stored_models", "training_sets", "heldout", "sizes", "attempts", "not_to_be_read", "not_measured", "adapters"]
BESIDE = ["reference", "start_run", "start_recipe", "start_run_read", "rows_made_by", "goal_set_again_made_here", "rank_16"]       # what a run from another start records beside


# ------------------------------------------------------------------------------------------------ the rules
def test_from_another_start_the_old_rule_is_trained_first_and_stands_in_withouts_place():
    assert (models_of("pre"), reference_of("pre")) == (MODELS, "without") == (("rehearse", "reward_rows"), "without")
    for start in ("pre_r64", "pre_r32", "anything_else"):
        assert (models_of(start), reference_of(start)) == (MODELS_FROM_ANOTHER_START, "old_rule") == (("old_rule", "reward_rows", "rehearse"), "old_rule")
    # The spec's two names are for rank 64; the rank in them is the start adapter's.
    assert (narrows(64), no_narrowing(64), NOT_READ_AS_EITHER) == ("THE OLD RULE NARROWS AT RANK 64 TOO", "NO NARROWING SHOWN AT RANK 64", "NOT READ AS EITHER")
    assert (narrows(32), no_narrowing(32)) == ("THE OLD RULE NARROWS AT RANK 32 TOO", "NO NARROWING SHOWN AT RANK 32")
    assert old_rule_outcomes(64) == (narrows(64), no_narrowing(64), NOT_READ_AS_EITHER, INCONCLUSIVE, NOT_READ)
    assert CONFIG["ladder_loop"]["l4"]["rows"]["models_from_another_start"] == list(MODELS_FROM_ANOTHER_START) and CONFIG["ladder_loop"]["l4"]["rows"]["start"] == "pre"


def test_the_old_rule_by_itself_has_three_named_outcomes():
    above, through, below = _kept(0.0114, 0.0026, 0.0203), _kept(0.004, -0.002, 0.011), _kept(-0.01, -0.02, -0.001)
    # LOST MORE THAN GAINED with the sign test under 0.05: it narrows at this rank too, whatever its gain per attempt is.
    for kept in (above, through, below):
        narrowed = old_rule_branch(PASSING, _breadth(25, 47), kept, 64, "pre_r64")
        assert narrowed["name"] == "THE OLD RULE NARROWS AT RANK 64 TOO" and (narrowed["lost_more_than_gained"], narrowed["rank"]) == (True, 64)
        assert "lost more than gained against `pre_r64` with the sign test under 0.05 (gained 25, lost 47, p = 0.0127746" in narrowed["reason"]
        assert narrowed["reason"].endswith("the old rule narrows at rank 64 too. The cap was not the cause; the training rule is")
    # Otherwise, with the interval for all of G per attempt above zero: no narrowing shown, with the smallest split at which one would have been seen.
    for breadth in (_breadth(30, 30), _breadth(40, 20), _breadth(24, 36), _breadth(0, 0)):       # even; gained more; lost more, p = 0.155; nothing changed
        shown = old_rule_branch(PASSING, breadth, above, 64, "pre_r64")
        changed = breadth["gained"] + breadth["lost"]
        assert shown["name"] == "NO NARROWING SHOWN AT RANK 64" and (shown["lost_more_than_gained"], shown["the_gain_is_above_zero"]) == (False, True)
        assert "no narrowing shown at rank 64; the old rule may stand at this rank" in shown["reason"] and shown["smallest_decisive_split"] == smallest_decisive_split(changed)
    assert "a narrowing would have been seen at 39 lost to 21 gained (the sign test's smallest decisive split with the 60 problems that changed" in old_rule_branch(
        PASSING, _breadth(24, 36), above, 64, "pre_r64")["reason"] and sign_test(21, 39) < 0.05 <= sign_test(22, 38)
    assert "with the 0 problems that changed, no split is decisive at 0.05: a narrowing could not have been seen at this size" in old_rule_branch(
        PASSING, _breadth(0, 0), above, 64, "pre_r64")["reason"]
    # Otherwise (not narrowed, and the interval holds zero, ends at it or lies below it): not read as either.
    for kept in (through, below, _kept(0.004, 0.0, 0.011)):
        neither = old_rule_branch(PASSING, _breadth(30, 30), kept, 64, "pre_r64")
        assert neither["name"] == NOT_READ_AS_EITHER and "not read as either" in neither["reason"] and (neither["lost_more_than_gained"], neither["the_gain_is_above_zero"]) == (False, False)
    # The sign test must be UNDER 0.05 and LOST must be more than gained: the mirror split is not a narrowing.
    assert sign_test(23, 39) > 0.05 > sign_test(23, 40)
    assert old_rule_branch(PASSING, _breadth(23, 39), above, 64, "pre_r64")["name"] == no_narrowing(64) and old_rule_branch(PASSING, _breadth(23, 40), above, 64, "pre_r64")["name"] == narrows(64)
    assert old_rule_branch(PASSING, _breadth(47, 25), through, 64, "pre_r64")["name"] == NOT_READ_AS_EITHER and sign_test(47, 25) < 0.05
    # The rank in the name is the start adapter's, and a failed check or a world with no goal problem is said as for every model.
    assert old_rule_branch(PASSING, _breadth(25, 47), above, 32, "pre_r32")["name"] == "THE OLD RULE NARROWS AT RANK 32 TOO"
    failed = old_rule_branch({**PASSING, "its_training_ran": {"passes": False}}, _breadth(25, 47), above, 64, "pre_r64")
    assert failed["name"] == INCONCLUSIVE and failed["failed_checks"] == ["its_training_ran"] and "the models read against it are not read" in failed["reason"]
    assert old_rule_branch(PASSING, {"problems": 0, "gained": 0, "lost": 0, "sign_test_p": None}, {"mean": None}, 64, "pre_r64")["name"] == NOT_READ
    assert all(old_rule_branch(PASSING, breadth, kept, 64, "pre_r64")["name"] in old_rule_outcomes(64) for breadth in (_breadth(25, 47), _breadth(30, 30)) for kept in (above, through))


def test_the_branch_of_the_two_others_names_the_models_it_is_read_against_and_is_not_read_against_an_inconclusive_old_rule():
    above, through = _kept(0.0114, 0.0026, 0.0203), _kept(0.004, -0.002, 0.011)
    for breadth, kept, name in ((_breadth(40, 20), above, GIVES_BACK), (_breadth(40, 20), through, UNDOES), (_breadth(28, 39), above, NOT_NAMED), (_breadth(28, 39), through, NOT_THIS_LEVER)):
        from_pre, other = rows_branch(PASSING, breadth, kept), rows_branch(PASSING, breadth, kept, against="old_rule", start="pre_r64")
        assert from_pre["name"] == other["name"] == name                           # the four names are the same
        assert "`without`" in from_pre["reason"] and "`pre`" in from_pre["reason"] and "old_rule" not in from_pre["reason"]
        assert other["reason"] == from_pre["reason"].replace("`without`", "`old_rule`").replace("`pre`", "`pre_r64`")
        assert {key: value for key, value in other.items() if key != "reason"} == {key: value for key, value in from_pre.items() if key != "reason"}
    # `old_rule` is INCONCLUSIVE: the model is NOT READ against it, and says so; its own failed check comes first.
    unread = rows_branch(PASSING, _breadth(40, 20), above, against="old_rule", start="pre_r64", against_failed=["its_training_ran"])
    assert unread["name"] == NOT_READ and unread["not_read_against"] == "old_rule" and "`old_rule`, which stands in `without`'s place, is INCONCLUSIVE (its_training_ran)" in unread["reason"]
    own = rows_branch({**PASSING, "no_barred_row": {"passes": False}}, _breadth(40, 20), above, against="old_rule", start="pre_r64", against_failed=["its_training_ran"])
    assert own["name"] == INCONCLUSIVE and "not_read_against" not in own
    assert rows_branch(PASSING, _breadth(40, 20), above, against="old_rule", start="pre_r64", against_failed=[])["name"] == GIVES_BACK


# ------------------------------------------------------------------------------- the report, on hand-made rows
OLD_NARROWS = ([20] * 6 + [0] * 6, [40] * 6 + [0] * 6)                                              # `old_rule` solves six goal problems, each far more often: `pre` solves nine
OLD_HOLDS = ([8, 8, 8, 6, 6, 6, 4, 3, 1, 1, 0, 0], [12, 12, 12, 9, 9, 9, 6, 4, 2, 1, 0, 0])        # ... every problem `pre` solves and one more, each more often
BROADER = ([8, 8, 8, 6, 6, 6, 4, 4, 3, 2, 2, 2], [12, 12, 12, 9, 9, 9, 6, 5, 4, 4, 3, 3])          # every goal problem solved
PRETRAIN_HALF = [f"p{index}" for index in range(400) if half_of(f"p{index}", 0) == PRETRAIN]
TRAINED = {"old_rule": [{"id": f"{name}#a", "problem_id": name, "origin": "attempt", "k": 1 + index % 8, "lines": 1 + index % 9} for index, name in enumerate(LOOP_HALF[:30])],
           "reward_rows": [{"id": f"{name}#a", "problem_id": name, "origin": "assembled" if index < 4 else "attempt", "k": 1 if index < 16 else 2, "lines": 2 + index % 7}
                           for index, name in enumerate(LOOP_HALF[:24])],
           "rehearse": [*({"id": f"{name}#a", "problem_id": name, "origin": "attempt", "k": 1 + index % 4, "lines": 1 + index % 9} for index, name in enumerate(LOOP_HALF[:30])),
                        *({"id": f"{name}#pretraining", "problem_id": name, "origin": "pretraining", "k": None, "lines": 4 + index % 9} for index, name in enumerate(PRETRAIN_HALF[:30]))]}
TRAINS = {name: {"model": name, "rows": len(own), "steps": -(-len(own) // 8), "stand_in_engine": False, "trained_from": "the stored adapter pre_r64",
                 "against_the_start_adapter": {name: CHANGED}} for name, own in TRAINED.items()}
LOSSES = {name: FALLING(len(own)) for name, own in TRAINED.items()}
PREPARE = {"stage": "l4", "label": LABEL, "seed": 0, "start": "pre_r64", "start_adapter": "runs/ladder_l4_pretrain_r64_seed0/adapters/pre_r64", "arm": "t010_assembly_pre",
           "arm_run": "runs/ladder_l2_t010_assembly_pre_seed0", "pretraining_run": "runs/ladder_l4_pretrain_seed0", "target_rate": 0.1, "stand_in_engine": False,
           "models": ["old_rule", "reward_rows", "rehearse"], "reference": "old_rule", "start_recipe": {"rank": 64, "alpha": 128}, "rows_made_by": {"start": "pre", "rank": 16},
           "rank_16": {"read": True},
           "training_sets": {"old_rule": {"rows": 30, "twin_rows": 30}, "rehearse": {"rows": 60, "twin_rows": 30, "rehearsal_rows": 30},
                             "reward_rows": {"rows": 24, "rows_of_the_rounds": 40, "kept_by_the_rule": 24, "kept_by_the_runs_minimum": 0}},
           "hot": {"temperature": 1.2, "temperature_of_every_other_measurement": 1.0, "models": ["old_rule", "pre_r64"], "samplings": HOT_SAMPLINGS, "attempts_a_goal_problem": 93},
           "goal_samplings": SAMPLINGS, "attempts_a_goal_problem": 93, "rung_episodes": 8, "sampling_seeds": {"rungs": 1002, "reach": 1001, "more": 1020},
           "contradicted_side_setting": "audit", "stored_models": {"pre_r64": {"run": "runs/ladder_l4_pretrain_r64_seed0"}}}
START_MODEL = _made("pre")                                                         # the start model's stored rows: the hand-made `pre`'s, under the name `pre_r64`
AGAIN = goal_set_again(START_MODEL["goal"][0], GOAL_IDS)                           # ITS G': what it does not solve in its first sampling


def _wrote(distinct, lines):
    return {"distinct_attempts": {"rungs": {"prompts": 6, "mean_share_distinct": 0.9}, "goal": {"prompts": 12, "mean_share_distinct": distinct}},
            "verified_proof_lines": {"rungs": {"1": 4}, "goal": lines}}


def _model(on_g=None, name="pre", **changed):
    return {**_made(name, on_g=on_g), "stand_in_engine": False, **changed}


def _hot(model):
    return {key: value for key, value in model.items() if key != "rungs"}


def _report(models=None, rank_16="given", prepare=PREPARE, trains=TRAINS, losses=LOSSES, trained=TRAINED, never=None, stored=None):
    models = {name: (models or {}).get(name) or _model() for name in MODELS_FROM_ANOTHER_START}
    stored = stored or {"pre_r64": START_MODEL}
    of_16 = {name: {key: value for key, value in _made(name).items() if key != "rungs"} for name in ("pre", "without")} if rank_16 == "given" else rank_16
    return build_rows_report(prepare, trains, losses, trained, GROUPS, LENGTHS, stored, models, {"old_rule": _hot(models["old_rule"]), "pre_r64": _hot(stored["pre_r64"])}, AGAIN,
                             SETTINGS, EVALUATION, never, of_16)


def test_the_old_rule_is_read_by_itself_against_the_start_model_with_rank_16s_stored_figures_beside():
    narrow = _model(OLD_NARROWS, **_wrote(0.70, {"1": 80, "2": 10, "8": 10}))
    start = {**START_MODEL, **_wrote(0.82, {"1": 50, "4": 25, "9": 25})}
    report = _report({"old_rule": narrow}, stored={"pre_r64": start})
    assert report["label"] == LABEL and report["check"] == "L4t" and report["ok"] is True and all(line.startswith(f"{SAY}: ") for line in report["lines"])
    assert (report["start"], report["reference"], report["rank"], report["read_against_the_reference"]) == ("pre_r64", "old_rule", 64, ["reward_rows", "rehearse"])
    assert list(report["branches"]) == ["old_rule", "reward_rows", "rehearse"] == list(report["models"])
    own = report["models"]["old_rule"]
    figures = own["by_itself"]
    # The four figures: the goal problems solved in the 93 attempts (gained, lost, the sign test), the two shares, all of G per attempt.
    assert (figures["solved"]["resolved_after"], figures["solved"]["resolved_before"], figures["solved"]["gained"], figures["solved"]["lost"], figures["solved"]["problems"]) == (6, 9, 0, 3, 12)
    assert figures["solved"] == {key: own["primary"][key] for key in figures["solved"]} and figures["solved"]["sign_test_p"] == sign_test(0, 3) == 0.25
    assert figures["share_of_distinct_attempts_on_g"] == {"of_the_old_rule": 0.70, "of_its_start": 0.82}
    assert figures["share_of_verified_proofs_with_8_lines_or_more"] == {"of_the_old_rule": 0.1, "of_its_start": 0.25} and figures["verified_proofs"] == {"of_the_old_rule": 100, "of_its_start": 100}
    assert figures["per_attempt"] == {key: own["beside_the_primary"][key] for key in figures["per_attempt"]}
    assert figures["per_attempt"]["successes"] == sum(OLD_NARROWS[0]) + sum(OLD_NARROWS[1]) and figures["per_attempt"]["successes_of_the_base"] == 84
    assert figures == {"what": figures["what"], **old_rule_figures(*(_on_g(entry) for entry in (narrow, start)), narrow, start, GOAL_IDS, 400, 0), "at_the_rank_of_the_rounds": figures["at_the_rank_of_the_rounds"]}
    # Three problems changed: no split of three is decisive, and its gain per attempt is above zero. NO NARROWING SHOWN, and the line says what could have been seen.
    assert own["branch"]["name"] == "NO NARROWING SHOWN AT RANK 64" == report["branches"]["old_rule"] and own["beside_the_primary"]["low"] > 0
    assert "with the 3 problems that changed, no split is decisive at 0.05" in own["branch"]["reason"]
    assert report["the_old_rule_by_itself"]["outcome"] == own["branch"] and report["the_old_rule_by_itself"]["model"] == "old_rule"
    # BESIDE EACH, the stored `without` against `pre`, from their own rows: the old rule at the rank the rounds were made at.
    of_16 = report["what_the_old_rule_gave"]
    assert of_16 is figures["at_the_rank_of_the_rounds"] and (of_16["read"], of_16["rank"]) == (True, 16) and set(of_16["on_all_of_g"]) == {"pre", "without"}
    assert (of_16["solved"]["resolved_after"], of_16["solved"]["resolved_before"], of_16["solved"]["gained"], of_16["solved"]["lost"]) == (10, 9, 1, 0)
    assert of_16["per_attempt"]["successes"] == 86 and of_16["per_attempt"]["successes_of_the_base"] == 84
    # Its lines: the four checks, the read by itself, what stands beside, the outcome, then its secondary reads against the start model alone.
    lines = [line for line in report["lines"] if line.startswith(f"{SAY}: `old_rule`")]
    assert all(f", CHECK {number}, " in line for number, line in enumerate(lines[:4], start=1)) and all(line.endswith(": PASS") for line in lines[:4]) and len(lines) == 16
    assert lines[4].startswith(f"{SAY}: `old_rule`, READ BY ITSELF against `pre_r64`, the model it was trained from (the cap's question; 93 attempts a goal problem): goal problems solved at "
                               "least once, `old_rule` against `pre_r64`: 6 to 9 of 12, gained 0, lost 3 (p = 0.25); the share of distinct attempts on G 0.7 against 0.82; the share of "
                               "verified proofs with 8 lines or more 0.1 against 0.25; all of G, successes per attempt, `old_rule` minus `pre_r64`: ")
    assert lines[5].startswith(f"{SAY}: `old_rule`, BESIDE IT, the old rule at rank 16, where the rounds were made, from the stored rows: goal problems solved at least once, `without` "
                               "against `pre`: 10 to 9 of 12, gained 1, lost 0 (p = 1.0); ")
    assert lines[6].startswith(f"{SAY}: `old_rule`: OUTCOME: NO NARROWING SHOWN AT RANK 64. ") and all("`old_rule`, SECONDARY, " in line for line in lines[7:])
    assert not any("PRIMARY" in line or "BRANCH" in line or "minus `old_rule`" in line for line in lines) and not any("`without`" in line for line in (*lines[1:5], *lines[6:]))
    assert "the rows it was trained on: 30, by origin attempt 30; by k " in lines[15]
    # The first line says the rank, the third model and whose rows these are; the limit is said with what it cannot say.
    assert "EVERY MODEL IS AT RANK 64, the start adapter's. `old_rule`, FIRST: the twin's 30 one-shot rows, in the twin's order (what `without` is, from `pre_r64`)" in report["lines"][0]
    assert "`hot`: `old_rule` and `pre_r64` sampled again on G at temperature 1.2. `old_rule` is read BY ITSELF against the stored `pre_r64`, which is the cap's question; each of the two " \
           "others BY ITSELF against `old_rule`, which stands in `without`'s place, and `pre_r64`, which stands in `pre`'s. THE ROWS ARE THE RANK-16 ROUNDS'" in report["lines"][0]
    assert report["what_it_cannot_say"][:3] == CANNOT_SAY and len(report["what_it_cannot_say"]) == 4 and "not the rounds `pre_r64` would have made" in report["what_it_cannot_say"][3]
    assert "by itself: NO NARROWING SHOWN AT RANK 64 (against `pre_r64`: gained 0, lost 3 (p = 0.25); all of G per attempt: " in report["headline"] and "from `pre_r64`" in report["headline"]
    # G' is the START MODEL's, given by length group here (a held-out proof's length is read by the report and by nothing before it).
    assert report["the_goal_set_again_of_the_start"]["problems"] == len(AGAIN) == 4 and report["the_goal_set_again_of_the_start"]["by_length_group"]["8+"] == 3
    assert report["heldout"]["goal_set_again"] == 4 and set(report["the_stored_models"]) == {"pre_r64", "runs", "rank_16"}
    # Without rank 16's rows (they were not on the box): said to be not there, and nothing else moves.
    without = _report({"old_rule": narrow}, stored={"pre_r64": start}, rank_16=None, prepare={**PREPARE, "rank_16": {"read": False, "why": "FileNotFoundError: no such rows"}})
    assert without["what_the_old_rule_gave"]["read"] is False and without["branches"] == report["branches"]
    assert [line for line in without["lines"] if "BESIDE IT, the old rule at the rank the rounds were made at (the stored `without` against `pre`): NOT THERE (FileNotFoundError: no such rows)" in line]
    assert [line for line in without["lines"] if "NOT THERE" not in line] == [line for line in report["lines"] if "BESIDE IT, the old rule at rank 16" not in line]


def _on_g(entry):
    return [{**first, "resolved": first["resolved"] + more["resolved"], "episodes": first["episodes"] + more["episodes"]} for first, more in zip(*entry["goal"])]


def test_the_three_outcomes_of_the_old_rule_on_hand_made_rows():
    # Nine goal problems the start model solves, and `old_rule` none of them: lost 9, gained 0, p = 0.004. IT NARROWS, whatever else it gained.
    lost_nine = ([0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 20])
    narrowed = _report({"old_rule": _model(lost_nine)})
    assert narrowed["models"]["old_rule"]["primary"]["lost"] == 9 and narrowed["models"]["old_rule"]["primary"]["gained"] == 3 and sign_test(3, 9) > 0.05
    lost_all = ([0] * 12, [0] * 12)
    narrowed = _report({"old_rule": _model(lost_all)})
    own = narrowed["models"]["old_rule"]
    assert (own["primary"]["gained"], own["primary"]["lost"], own["primary"]["sign_test_p"]) == (0, 9, sign_test(0, 9)) and sign_test(0, 9) < 0.05
    assert narrowed["branches"]["old_rule"] == "THE OLD RULE NARROWS AT RANK 64 TOO" and any(line.startswith(f"{SAY}: `old_rule`: OUTCOME: THE OLD RULE NARROWS AT RANK 64 TOO. ") for line in narrowed["lines"])
    # It keeps what the start model solves and succeeds more often: NO NARROWING SHOWN.
    held = _report({"old_rule": _model(OLD_HOLDS)})
    assert held["models"]["old_rule"]["primary"]["lost"] == 0 and held["models"]["old_rule"]["beside_the_primary"]["low"] > 0 and held["branches"]["old_rule"] == "NO NARROWING SHOWN AT RANK 64"
    # The start model's own rows standing in: nothing changed and nothing gained. NOT READ AS EITHER.
    same = _report()
    assert (same["models"]["old_rule"]["primary"]["gained"], same["models"]["old_rule"]["primary"]["lost"]) == (0, 0) and same["models"]["old_rule"]["beside_the_primary"]["mean"] == 0
    assert same["branches"] == {"old_rule": NOT_READ_AS_EITHER, "reward_rows": NOT_THIS_LEVER, "rehearse": NOT_THIS_LEVER}
    assert all(branch in old_rule_outcomes(64) for branch in (narrowed["branches"]["old_rule"], held["branches"]["old_rule"], same["branches"]["old_rule"]))
    # The rank in the name is the one the prepare step recorded of the start adapter.
    assert _report({"old_rule": _model(lost_all)}, prepare={**PREPARE, "start_recipe": {"rank": 32, "alpha": 64}})["branches"]["old_rule"] == "THE OLD RULE NARROWS AT RANK 32 TOO"


def test_the_two_others_are_read_against_the_old_rule_and_the_start_model():
    # `old_rule` narrows to four goal problems; `rehearse` solves all twelve and succeeds more often than the start model; `reward_rows` is `old_rule`'s rows again.
    narrow = ([8, 8, 8, 6, 0, 0, 0, 0, 0, 0, 0, 0], [12, 12, 12, 9, 0, 0, 0, 0, 0, 0, 0, 0])
    report = _report({"old_rule": _model(narrow), "rehearse": _model(BROADER), "reward_rows": _model(narrow)}, never=["g8a", "g8c", "not_a_goal_problem"])
    rehearse, reward_rows = report["models"]["rehearse"], report["models"]["reward_rows"]
    # THE PRIMARY is against `old_rule`'s measured rows: eight goal problems gained on its four, none lost.
    assert (rehearse["primary"]["resolved_after"], rehearse["primary"]["resolved_before"], rehearse["primary"]["gained"], rehearse["primary"]["lost"]) == (12, 4, 8, 0)
    assert "`rehearse` against `old_rule` (the same rounds' rows by the old rule)" in rehearse["primary"]["what"]
    # BESIDE IT, reliability is against the START MODEL (84 successes), and `old_rule`'s own stands beside.
    assert rehearse["beside_the_primary"]["successes_of_the_base"] == 84 and rehearse["beside_the_primary"]["successes"] == sum(BROADER[0]) + sum(BROADER[1])
    assert {key: rehearse["beside_the_primary"]["of_old_rule"][key] for key in ("mean", "low", "high")} == {
        key: report["models"]["old_rule"]["beside_the_primary"][key] for key in ("mean", "low", "high")} and "of_without" not in rehearse["beside_the_primary"]
    assert rehearse["branch"]["name"] == GIVES_BACK and "against `old_rule`" in rehearse["branch"]["reason"] and "against `pre_r64`" in rehearse["branch"]["reason"]
    assert (reward_rows["primary"]["gained"], reward_rows["primary"]["lost"]) == (0, 0) and report["branches"]["reward_rows"] in (NOT_THIS_LEVER, NOT_NAMED)
    assert all(name in BRANCHES for name in (report["branches"]["rehearse"], report["branches"]["reward_rows"]))
    # THE SECONDARY reads: G' and the rungs against the start model; the breadth by length against `old_rule`; the models named beside are the start model and `old_rule`.
    secondary = rehearse["secondary"]
    assert list(secondary) == ["the_goal_set_again", "by_length_group", "goal_problems_solved_against_pre_r64", "goal_problems_solved_by_attempts_alone", "the_three_rungs",
                               "distinct_attempts", "verified_proof_lines", "never_solved_before", "trained_on"]
    assert secondary["the_goal_set_again"]["goal_set_again"] == 4 and secondary["the_goal_set_again"]["attempts_each"] == 4 * 61 and "`pre_r64` does not solve" in secondary["the_goal_set_again"]["what"]
    assert set(secondary["by_length_group"]) == {"what", "problems", "minus_pre_r64", "against_old_rule"} and secondary["by_length_group"]["against_old_rule"]["8+"]["gained"] == 3
    assert set(secondary["the_three_rungs"]) == {"what", "problems", "minus_pre_r64", "minus_old_rule"}
    assert set(secondary["distinct_attempts"]) == set(secondary["verified_proof_lines"]) == set(secondary["goal_problems_solved_by_attempts_alone"]) == {"what", "pre_r64", "old_rule", "rehearse"}
    unsolved = secondary["never_solved_before"]
    assert (unsolved["given"], unsolved["goal_problems"], unsolved["solved"], unsolved["solved_by_old_rule"]) == (True, 2, 2, 0) and "solved_by_with" not in unsolved
    lines = [line for line in report["lines"] if line.startswith(f"{SAY}: `rehearse`")]
    assert len(lines) == 17 and "PRIMARY, breadth. The goal problems solved at least once in the 93 attempts, `rehearse` against `old_rule`, paired by problem: 12 to 4 of 12, gained 8, lost 0" in lines[4]
    assert "`rehearse` minus `pre_r64`, 95% bootstrap over problems: " in lines[5] and "; `old_rule`'s is " in lines[5]
    assert "of the 2 listed, `rehearse` solves 2 (g8a, g8c); `old_rule` solves 0; 1 more ids of the file are not goal problems of this run" in lines[15]
    assert "CHECK 3, Lean answered" in lines[2] and all(key in lines[2] for key in ("rungs_rehearse", "goal_rehearse_2", "rungs_pre_r64", "goal_old_rule_1"))
    assert not any("`without`" in line or "`with`" in line for line in lines[4:])
    # `hot`: `old_rule` (its rows at 1.0 measured by this run) and the start model (stored).
    hot = [line for line in report["lines"] if line.startswith(f"{SAY}: `hot`")]
    assert len(hot) == 4 and "`old_rule` on all of G (12 problems, 93 attempts a problem at 1.2, 93 at 1.0): at 1.0 (measured here) 4 of 12 solved" in hot[0]
    assert "`pre_r64` on all of G" in hot[2] and "at 1.0 (stored) 9 of 12 solved" in hot[2] and "at 1.0 `pre_r64`'s first sampling CHOSE them, so `pre_r64` has no success there in it" in hot[1]
    assert set(report["hot"]) >= {"old_rule", "pre_r64"} and "with" not in report["hot"] and "`old_rule`'s rows measured by this run, `pre_r64`'s stored" in report["hot"]["what"]
    assert "`rehearse`: THE RULE GIVES THE BREADTH BACK AND KEEPS THE GAIN (against `old_rule`: gained 8, lost 0" in report["headline"]


def test_an_inconclusive_old_rule_is_not_read_and_the_two_others_are_not_read_against_it_and_say_so():
    barred_row = [{**TRAINED["old_rule"][0], "problem_id": OF_THE_OTHER_HALF}, *TRAINED["old_rule"][1:]]
    report = _report({"rehearse": _model(BROADER)}, trained={**TRAINED, "old_rule": barred_row})
    old = report["models"]["old_rule"]
    assert old["inconclusive"] is True and old["branch"]["name"] == INCONCLUSIVE and old["branch"]["failed_checks"] == ["no_barred_row"] and old["by_itself"] is None and old["primary"] is None
    assert old["measured_and_not_read"]["by_itself"]["solved"]["problems"] == 12
    for name in ("reward_rows", "rehearse"):
        own = report["models"][name]
        assert own["branch"]["name"] == NOT_READ and own["branch"]["not_read_against"] == "old_rule" and own["inconclusive"] is False
        assert own["primary"] is None and own["secondary"] is None and "`old_rule`, which this model is read against, is INCONCLUSIVE" in own["measured_and_not_read"]["why"]
        assert all(check["passes"] for key, check in own["can_this_model_be_read"].items() if key != "what")           # its own four checks passed
        lines = [line for line in report["lines"] if line.startswith(f"{SAY}: `{name}`")]
        assert len(lines) == 5 and lines[4].startswith(f"{SAY}: `{name}`: BRANCH: NOT READ. `old_rule`, which stands in `without`'s place, is INCONCLUSIVE (no_barred_row)")
        assert not any("PRIMARY" in line or "SECONDARY" in line for line in lines)
    assert report["branches"] == {"old_rule": INCONCLUSIVE, "reward_rows": NOT_READ, "rehearse": NOT_READ} and report["ok"] is True
    assert "`old_rule`: INCONCLUSIVE. `reward_rows`: NOT READ (`old_rule` is INCONCLUSIVE)" in report["headline"]
    assert len([line for line in report["lines"] if line.startswith(f"{SAY}: `old_rule`")]) == 5 and len([line for line in report["lines"] if line.startswith(f"{SAY}: `hot`")]) == 4
    # A set of `old_rule` Lean did not answer: the two others stand on it too, and each is INCONCLUSIVE by its own third check.
    unanswered = _model()
    unanswered["goal"][1] = [{**row, "attempts_without_an_answer": 20} for row in unanswered["goal"][1]]
    report = _report({"old_rule": unanswered})
    assert report["ok"] is False and report["not_to_be_read"] == ["goal_hot_old_rule_2", "goal_old_rule_2"] and set(report["branches"].values()) == {INCONCLUSIVE}
    assert all(read["branch"]["failed_checks"] == ["lean_answered"] for read in report["models"].values())
    # A model of its own whose check fails is INCONCLUSIVE by itself; `old_rule` and the other are read.
    report = _report({"rehearse": _model(BROADER)}, losses={**LOSSES, "reward_rows": list(reversed(FALLING(24)))})
    assert report["branches"]["reward_rows"] == INCONCLUSIVE and report["branches"]["old_rule"] == NOT_READ_AS_EITHER and report["models"]["rehearse"]["primary"]["gained"] == 3


# --------------------------------------------------------------------------- the stage, in the scripted world
def _trainings(monkeypatch, calls, saves=None):
    """The training's GPU path with the model taken out: `_train_with_readings` is the schedule alone (stand-in losses).
    AS THE REAL ONE DOES, it attaches the adapter the config it is HANDED asks for (`config["lora"]`): each adapter it
    saves is a weights file of that rank with that alpha beside it; and a training from a start adapter records what of
    each adapter it saved is not the start adapter's. `saves`: by model, another (rank, alpha, rank in the config file)
    to save than the config's (a training the start adapter's rank did not reach)."""
    def train_with_readings(config, examples, sample, pairs, schedule, seed, directory, tensorboard_run, orders=None, per_example=False, positions=False, **more):
        calls.append({"lora": dict(config["lora"]), "max_lora_rank": config["vllm"]["max_lora_rank"], "examples": examples, "schedule": schedule, "seed": seed,
                      "directory": directory, "tensorboard_run": tensorboard_run, "orders": orders, "more": more})
        train_step, read, _ = ladder_dose._stand_in_calls(len(examples), [], [])
        saved = lambda name: (saves or {}).get(name) or (config["lora"]["rank"], config["lora"]["alpha"], None)      # noqa: E731
        return {**run_dose(len(examples), config["training"]["effective_batch"], schedule["passes"], seed, schedule["reading_steps"], schedule["checkpoint_steps"],
                           train_step, read, lambda name, step_number: _adapter(directory / name, *saved(name)), orders=orders, per_example=per_example,
                           positions=positions), "target_format": "native", "seconds": 1.0, "peak_allocated_gb": 12.3,
                **({"against_the_start_adapter": {name: dict(CHANGED) for name in schedule["checkpoint_steps"]}} if "start" in more else {})}

    monkeypatch.setattr(ladder_dose, "_train_with_readings", train_with_readings)
    monkeypatch.setattr(pipeline, "_cap_torch_memory", lambda config: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None, memory_allocated=lambda: 0,
                                                                                   max_memory_reserved=lambda: 12_640_000_000)))


def _stage(config, monkeypatch, until=None, stage_name="ladder_l4_rows_r64"):
    """The stage's steps in its order, the trainings by their GPU path."""
    summaries = {}
    for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
        if environment != "gpu" or step not in ladder_l4_rows.STEPS:
            continue
        run = ladder_l4_rows.STEPS[step]
        summaries[step] = _on_the_gpu(monkeypatch, run, config) if step in TRAININGS else run(config)
        if step == until:
            break
    return summaries


@pytest.fixture
def r64(l4, monkeypatch):  # noqa: F811
    """L4's world with what the stage reads on the box from `pre_r64`: `pre`'s run (rank 16), the arm's from it (two
    rounds of two problems), and the run of the check of the adapter's rank (`pre_r64`, rank 64, measured as `pre`
    was); every training by its GPU path, so that each adapter is two files on disk at the rank its config asked for.
    Then the task's own variables, as the stage `ladder_l4_rows_r64` sets them. `kits`: what every engine was built
    with from here on."""
    for name in (RUN, ARM_RUN, MINIMUM, SEEDS, START, START_CHECKS, CHECK, RANK_RUN):
        monkeypatch.delenv(name, raising=False)
    config, calls, kits = l4.config, [], []
    l4.stored()
    _trainings(monkeypatch, calls)
    ladder_l4.ladder_l4_pretrain_prepare(config)
    _on_the_gpu(monkeypatch, ladder_l4.ladder_l4_pretrain_train, config)
    for step in (ladder_l4.ladder_l4_pretrain_measure, ladder_l4.ladder_l4_pretrain_map, ladder_l4.ladder_l4_pretrain_report):
        step(config)
    l4.the_arm()
    monkeypatch.setattr(ladder_l2, "_start_request", lambda start: ("the start adapter", start))
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (
        kits.append({"temperature": given["sampling"]["temperature"], "adapter": adapter, "max_lora_rank": given["vllm"]["max_lora_rank"], "rank": given["lora"]["rank"],
                     "vllm": given["vllm"], "lora": given["lora"]}), (l4.solver, stand_in_parameters))[1]))
    with monkeypatch.context() as patched:                                     # the trainings, and only they, take their GPU path
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        _run(config, "ladder_l4", rounds=2, until="ladder_l3d2_train_without")
    _run(config, "ladder_l4", rounds=2)
    arm_run, pre = l4.store().root, l4.pretrain_store().root
    monkeypatch.delenv(ARM)                                                    # a task of this stage names no arm and no number of rounds
    monkeypatch.delenv(ladder_l2.L2_ROUNDS_VARIABLE)
    monkeypatch.setenv(CHECK, "rank")                                          # the check's own task
    _check(config, monkeypatch)
    rank_run = ladder_l4_rank._store(config, ladder_l4_rank.the_check(config)).root
    monkeypatch.delenv(CHECK)
    assert [call["lora"]["rank"] for call in calls] == [16, 16, 16, 16, 64]    # `pre`, the arm's two models and its twin at the config's rank; `pre_r64` at the check's
    options = entry.step_fields(entry.STAGES["ladder_l4_rows_r64"][2])[2]
    assert options == {"environment": {START: "pre_r64"}} and all(entry.step_fields(step)[2] == options for step in entry.STAGES["ladder_l4_rows_r64"])
    for name, value in options["environment"].items():                         # THE STAGE names the start; the setting stays `pre`
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(MINIMUM, "2")                                           # the fixtures' rounds make a handful of rows: as the smoke run, at least two are kept
    assert [kit["max_lora_rank"] for kit in kits[-3:]] == [64] * 3 and {kit["max_lora_rank"] for kit in kits[:-3]} == {16}      # the check's own measurement, and no step before it, at 64
    del calls[:], kits[:]
    settings = lambda: ladder_l4_rows.the_settings(config)      # noqa: E731
    return SimpleNamespace(config=config, l4=l4, calls=calls, kits=kits, arm=arm_run, pre=pre, rank=rank_run, store=lambda: ladder_l4_rows._store(config, settings()))


def test_the_start_is_a_setting_that_names_a_pretrained_model_and_where_it_lives(r64, monkeypatch):
    config = r64.config
    settings = config["ladder_loop"]["l4"]["rows"]
    assert settings["start"] == "pre"                                          # the setting is what it was: the stage names another
    rows_ = ladder_l4_rows.the_settings(config)
    start = ladder_l4_rows.the_start(config, rows_)
    assert (rows_.start, rows_.models) == ("pre_r64", ("old_rule", "reward_rows", "rehearse")) and ladder_l4_rows.run_name(rows_, 0) == "ladder_l4_rows_pre_r64_seed0"
    # `pre_r64`: the run and the adapter of the check of the adapter's rank, by that module's own functions; its rank and alpha are that check's.
    check = ladder_l4_rank.check_of(config, "pre_r64")
    assert (check.which, check.rank, check.alpha, check.name, check.stage) == ("rank", 64, 128, "pre_r64", "ladder_l4_rank") and start.check == check
    assert start.directory == r64.rank == ladder_l4_rank.run_directory(config, check, 0) and start.directory.name == "ladder_l4_pretrain_r64_seed0"
    assert start.adapter == ladder_l4_rank.adapter_directory(config, check, 0) == r64.rank / "adapters" / "pre_r64" == ladder_l4_rows.start_adapter(config, rows_)
    assert "the task of stage `ladder_l4_rank` for seed 0 (`python -m rlvr_lean.runner.entry --stage ladder_l4_rank --seeds 0`; for the smoke run, stage `ladder_l4_rank_smoke`)" == start.task
    assert ladder_l4_rows.start_set("rungs", start) == "l4_rungs_pre_r64" and ladder_l4_rows.start_set(REACH, start) == "l4_reach_pre_r64"
    assert ladder_l4_rows.hot_of(rows_) == (("old_rule", "old_rule"), ("start", "pre_r64")) and ladder_l4_rows.steps_of("pre_r64") == STEPS
    assert (ladder_l4_rank.check_of(config, "pre_r32").which, ladder_l4_rank.check_of(config, "pre"), ladder_l4_rank.check_of(config, "pre_r128")) == ("fallback", None, None)
    # The setting itself naming it (no variable) is the same start: `ladder_loop.l4.rows.start: pre_r64` works.
    monkeypatch.delenv(START)
    monkeypatch.setitem(settings, "start", "pre_r64")
    assert ladder_l4_rows.the_settings(config) == rows_ and ladder_l4_rows.the_start(config, ladder_l4_rows.the_settings(config)) == start
    # `pre`: the ONE pretraining's run and adapter, the config's own rank (no check), the two models, `with` and `pre` sampled again.
    monkeypatch.setitem(settings, "start", "pre")
    from_pre = ladder_l4_rows.the_settings(config)
    start = ladder_l4_rows.the_start(config, from_pre)
    assert (from_pre.start, from_pre.models, start.name, start.check) == ("pre", ("rehearse", "reward_rows"), "pre", None) and ladder_l4_rows.run_name(from_pre, 0) == "ladder_l4_rows_seed0"
    assert (start.directory, start.adapter) == (r64.pre, r64.pre / "adapters" / "pre") and ladder_l4_rows.start_set(MORE, start) == "l4_more_pre"
    assert ladder_l4_rows.hot_of(from_pre) == (("with", "with"), ("pre", "pre")) and "`ladder_l4_pretrain` for seed 0 (`python -m rlvr_lean.runner.entry --stage ladder_l4_pretrain --seeds 0`" in start.task
    # From `pre` the config is handed on AS IT IS; from `pre_r64` a training gets the start adapter's rank and alpha, and a measure step the model server's limit too.
    assert ladder_l4_rows.config_from(config, {}, start) is config and ladder_l4_rows.config_from(config, {}, start, serving=True) is config
    start = ladder_l4_rows.the_start(config, rows_)
    training = ladder_l4_rows.config_from(config, {"start_recipe": RANK_64}, start)
    serving = ladder_l4_rows.config_from(config, {"start_recipe": RANK_64}, start, serving=True)
    assert training["lora"] == serving["lora"] == {**config["lora"], **RANK_64} and training["vllm"] is config["vllm"] and serving["vllm"] == {**config["vllm"], "max_lora_rank": 64}
    assert {key: value for key, value in serving.items() if key not in ("lora", "vllm")} == {key: value for key, value in config.items() if key not in ("lora", "vllm")}
    assert (config["lora"]["rank"], config["lora"]["alpha"], config["vllm"]["max_lora_rank"]) == (16, 32, 16)      # and the config itself is not changed
    with pytest.raises(RuntimeError, match="prepared from `pre_r64` at rank 32 and alpha 64 .what the run that made it recorded., and ladder_loop.l4.rank_check now gives it rank 64 and alpha 128"):
        ladder_l4_rows.config_from(config, {"start_recipe": {"rank": 32, "alpha": 64}}, start)
    # The models of another start are their own setting, held to the stage's steps as `pre`'s are.
    monkeypatch.setenv(START, "pre_r64")
    monkeypatch.setitem(settings, "models_from_another_start", ["old_rule", "rehearse", "reward_rows"])
    with pytest.raises(ValueError, match="ladder_loop.l4.rows.models_from_another_start is .'old_rule', 'rehearse', 'reward_rows'. and the stage trains and measures "
                                         ".'old_rule', 'reward_rows', 'rehearse'.: change both together"):
        ladder_l4_rows.the_settings(config)


def test_the_stage_from_pre_r64_trains_the_old_rule_first_every_model_at_rank_64_and_reads_against_the_old_rule_and_pre_r64(r64, monkeypatch):
    config, calls, kits = r64.config, r64.calls, r64.kits
    arm_run, pre, rank_run = r64.arm, r64.pre, r64.rank
    assert (arm_run.name, pre.name, rank_run.name) == ("ladder_l2_t010_assembly_pre_seed0", "ladder_l4_pretrain_seed0", "ladder_l4_pretrain_r64_seed0")
    read_only = {path: _all(path) for path in (arm_run, pre, rank_run)}
    summaries = _stage(config, monkeypatch)
    store = r64.store()
    assert store.root.name == "ladder_l4_rows_pre_r64_seed0" and store.root.parent == pre.parent and list(summaries) == STEPS == ladder_l4_rows.steps_of("pre_r64")
    assert {path: _all(path) for path in read_only} == read_only               # THE THREE RUNS WERE ONLY READ: file for file, adapters included
    assert (config["lora"]["rank"], config["vllm"]["max_lora_rank"], config["sampling"]["temperature"]) == (16, 16, 1.0)      # and the config itself was not changed
    assert not (store.root.parent / "ladder_l4_rows_seed0").exists()

    # ---- prepare: the start, its rank, the three training sets, G' of the start model
    prepare = summaries["ladder_l4_rows_prepare"]
    start = rank_run / "adapters" / "pre_r64"
    assert list(prepare) == [*PREPARE_KEYS_FROM_PRE, *BESIDE]                  # what a run from `pre` records, then what this start has beside
    assert (prepare["start"], prepare["start_adapter"], prepare["start_run"], prepare["with_adapter"], prepare["reference"]) == ("pre_r64", str(start), str(rank_run), None, "old_rule")
    assert (prepare["arm_run"], prepare["pretraining_run"], prepare["models"]) == (str(arm_run), str(pre), ["old_rule", "reward_rows", "rehearse"]) and list(prepare["training_sets"]) == prepare["models"]
    # THE RANK AND ALPHA ARE THE START ADAPTER'S, as its own run recorded them: not the config's.
    of_start = json.loads((rank_run / "ladder_l4_rank_prepare.done.json").read_text())
    recipe = prepare["start_recipe"]
    assert (recipe["rank"], recipe["alpha"], recipe["rank_of_the_config"], recipe["alpha_of_the_config"]) == (64, 128, 16, 32) == (
        of_start["recipe"]["lora"]["rank"], of_start["recipe"]["lora"]["alpha"], config["lora"]["rank"], config["lora"]["alpha"])
    assert recipe["adapter_saved"]["ranks"] == [64] and (recipe["check"], recipe["stage"]) == ("rank", "ladder_l4_rank") and prepare["recipe"]["lora"] == {**config["lora"], **RANK_64}
    assert prepare["start_run_read"]["checks_waived_for_a_smoke_run"] is False and prepare["start_run_read"]["pretraining_file_sha256"] == prepare["pretraining_file_sha256"]
    # THE ROWS ARE THE ARM'S, MADE FROM `pre` at rank 16: recorded as that.
    of_the_arm = json.loads((arm_run / "ladder_l2_prepare.done.json").read_text())
    made_by = prepare["rows_made_by"]
    assert (made_by["start"], made_by["start_adapter"], made_by["rank"], made_by["alpha"]) == ("pre", of_the_arm["start_adapter"], 16, 32) and "not the rounds `pre_r64` would have made" in made_by["what"]
    # `old_rule`: THE TWIN'S ROWS EXACTLY, in the twin's order, each with its k and its text.
    twin = _file_rows(arm_run / ladder_l3d2.TRAINING_WITHOUT_FILE)
    old_rule = store.read_rows("l4_rows_training_old_rule.jsonl")
    picks = {row["problem_id"]: row for number in (1, 2) for batch in (1, 2) for row in _file_rows(arm_run / f"episodes_round_r{number}_b{batch}_problems.jsonl")}
    assert twin and [row["id"] for row in old_rule] == [row["id"] for row in twin] and [row["row"] for row in old_rule] == list(range(len(twin)))
    assert all({key: row[key] for key in ("problem_id", "side", "theorem", "completion")} == {key: given[key] for key in ("problem_id", "side", "theorem", "completion")}
               and row["origin"] == "attempt" and row["k"] == picks[row["problem_id"]]["resolved"] for row, given in zip(old_rule, twin))
    sets = prepare["training_sets"]
    assert (sets["old_rule"]["rows"], sets["old_rule"]["twin_rows"], sets["old_rule"]["rows_by_origin"], sets["old_rule"]["optimizer_steps"]) == (
        len(twin), len(twin), {"attempt": len(twin)}, -(-len(twin) // 8))
    assert sets["old_rule"]["twin_file"] == "l3d2_training_without.jsonl" and sets["old_rule"]["barred"]["held_out"] == []
    rehearse, reward_rows = store.read_rows("l4_rows_training_rehearse.jsonl"), store.read_rows("l4_rows_training_reward_rows.jsonl")
    assert sorted(row["id"] for row in rehearse if row["origin"] == "attempt") == sorted(row["id"] for row in twin) and len(rehearse) == 2 * len(twin) and len(reward_rows) >= 2
    # NO PUBLISHED PROOF IS COPIED INTO THE RUN.
    file_rows = _file_rows(FIXTURE)
    assert not any(proof in content.decode(errors="ignore") for proof in (*(json.dumps(row["proof"], ensure_ascii=False)[1:-1] for row in file_rows), *(row["proof"].strip() for row in file_rows))
                   for content in _all(store.root).values())
    # G' IS THE START MODEL'S, made here by the pretraining's rule from its stored first sampling, and stored in this run.
    first = _file_rows(rank_run / "episodes_l4_reach_pre_r64_problems.jsonl")
    goal = [row["problem_id"] for row in store.read_rows(ladder_l4.GROUPS_FILE) if row["group"] == "goal"]
    again = [row["problem_id"] for row in store.read_rows(ladder_l4.AGAIN_FILE)]
    assert again == goal_set_again(first, goal) and prepare["goal_set_again"] == len(again) == len(goal) - sum(row["resolved"] > 0 for row in first)
    assert prepare["goal_set_again_made_here"] == {"what": prepare["goal_set_again_made_here"]["what"], "problems": len(again), "solved_in_the_first_sampling": len(goal) - len(again),
                                                   "file": "l4_goal_set_again.jsonl"} and not (rank_run / ladder_l4.AGAIN_FILE).exists()
    # The stored rows every difference is read against are `pre_r64`'s; beside, the stored `without` and `pre` on G (rank 16). `with` is not read.
    for part in ("rungs", REACH, MORE):
        assert store.read_rows(ladder_ceiling.stored_file("pre_r64", part, "l4")) == _file_rows(rank_run / f"episodes_l4_{part}_pre_r64_problems.jsonl")
    for part in (REACH, MORE):
        assert store.read_rows(ladder_ceiling.stored_file("pre", part, "l4")) == _file_rows(pre / f"episodes_l4_{part}_pre_problems.jsonl")
        assert store.read_rows(ladder_ceiling.stored_file("without", part, "l4")) == _file_rows(arm_run / f"episodes_l3d2_{part}_without_problems.jsonl")
    assert prepare["rank_16"]["read"] is True and list(prepare["stored_models"]) == ["pre_r64"] and prepare["stored_models"]["pre_r64"]["sets"]["rungs"] == "l4_rungs_pre_r64"
    assert set(json.loads(store.path(ladder_l4_rows.STORED_MODELS_FILE).read_text())) == {"stage", "label", "pre_r64", "pre", "without"}
    of_pre = json.loads((pre / "ladder_l4_pretrain_prepare.done.json").read_text())
    assert all(prepare[key] == of_pre[key] == of_start[key] for key in ("sampling_seeds", "rung_episodes", "goal_samplings"))
    hot = prepare["hot"]
    assert (hot["temperature"], hot["models"]) == (1.2, ["old_rule", "pre_r64"]) and "`old_rule` and `pre_r64` each sampled again on G alone" in hot["what"]

    # ---- train: each model FROM `pre_r64`, AT ITS RANK, `old_rule` first and in the twin's order; the adapter saved read back
    assert [next(iter(call["schedule"]["checkpoint_steps"])) for call in calls] == ["old_rule", "reward_rows", "rehearse"]
    for call, model, set_rows in zip(calls, ("old_rule", "reward_rows", "rehearse"), (old_rule, reward_rows, rehearse)):
        train = summaries[f"ladder_l4_rows_train_{model}"]
        assert call["lora"] == {**config["lora"], **RANK_64} and call["max_lora_rank"] == 16      # the adapter's rank is the start adapter's; a training starts no model server
        assert call["more"] == {"start": start} and call["orders"] == [list(range(len(set_rows)))] and call["tensorboard_run"] == f"ladder_l4_rows_pre_r64_seed0_{model}"
        assert (train["model"], train["rows"], train["trained_from"], train["adapter"]) == (model, len(set_rows), f"the stored adapter {start}", str(store.root / "adapters" / model))
        saved = train["adapter_saved"]
        assert (saved["rank"], saved["alpha"], saved["ranks"], train["peak_reserved_gb"], train["against_the_start_adapter"]) == (64, 128, [64], 12.64, {model: CHANGED})
        assert json.loads((store.root / "adapters" / model / "adapter_config.json").read_text())["r"] == 64
        assert json.loads(store.path(f"l4_rows_loss_{model}.json").read_text())["rows_trained"] == [row["id"] for row in set_rows]
    assert calls[0]["examples"] == [{key: row[key] for key in ("problem_id", "side", "theorem", "completion")} for row in twin] and "the twin's own" in summaries["ladder_l4_rows_train_old_rule"]["order"]
    assert len(calls) == 3

    # ---- measure: as `with` was, with `pre`'s sampling seeds, the model server's largest adapter rank RAISED to 64; then `old_rule` and `pre_r64` on G alone at 1.2
    assert len(kits) == 3 * 3 + 2 + 2 and [kit["temperature"] for kit in kits] == [1.0] * 9 + [1.2] * 4
    assert all(kit["max_lora_rank"] == 64 and kit["rank"] == 64 and {**kit["vllm"], "max_lora_rank": 16} == config["vllm"] for kit in kits)
    for model in ("old_rule", "reward_rows", "rehearse"):
        measured = summaries[f"ladder_l4_rows_measure_{model}"]
        for of, own_set, start_set in ((measured["rungs"], f"l4_rungs_{model}", "l4_rungs_pre_r64"), (measured["goal"][REACH], f"l4_reach_{model}", "l4_reach_pre_r64"),
                                       (measured["goal"][MORE], f"l4_more_{model}", "l4_more_pre_r64")):
            marker = json.loads((rank_run / f"episodes_{start_set}.done.json").read_text())
            assert of["set"] == own_set and (of["sampling_seed"], of["episodes_each"], of["problems"]) == (marker["sampling_seed"], marker["episodes_each"], marker["problems"])
    for step, who, adapter in (("old_rule", "old_rule", store.root / "adapters" / "old_rule"), ("start", "pre_r64", start)):
        measured = summaries[f"ladder_l4_rows_measure_hot_{step}"]
        assert (measured["model"], measured["temperature"], measured["adapter"]) == (who, 1.2, str(adapter)) and "rungs" not in measured and set(measured["goal"]) == {REACH, MORE}
        assert [measured["goal"][sampling["name"]]["set"] for sampling in hot["samplings"]] == [f"l4_{sampling['name']}_hot_{who}" for sampling in hot["samplings"]]
        assert [measured["goal"][sampling["name"]]["sampling_seed"] for sampling in hot["samplings"]] == [1040, 1041]
    assert {row["set"] for row in store.read_rows("problems.jsonl")} == {f"l4_{part}_{model}" for part in ("rungs", REACH, MORE) for model in ("old_rule", "reward_rows", "rehearse")} | {
        f"l4_{part}_hot_{who}" for part in (REACH, MORE) for who in ("old_rule", "pre_r64")}

    # ---- the report: `old_rule` by itself first, then the two others against it and the start model, then the table of `hot`
    report = summaries["ladder_l4_rows_report"]
    assert list(report) == [*REPORT_KEYS_FROM_PRE[:-1], "start", "reference", "rank", "read_against_the_reference", "the_old_rule_by_itself", "the_goal_set_again_of_the_start",
                            "start_recipe", "rows_made_by", "adapters"]
    assert report["label"] == "pretrained on published proofs" and report["ok"] is True and all(line.startswith(f"{SAY}: ") for line in report["lines"]) and "from `pre_r64`" in report["headline"]
    assert list(report["branches"]) == ["old_rule", "reward_rows", "rehearse"] and report["branches"]["old_rule"] in old_rule_outcomes(64)
    assert all(report["branches"][model] in BRANCHES for model in ("reward_rows", "rehearse")) and (report["rank"], report["reference"]) == (64, "old_rule")
    total = lambda name: sum(row["resolved"] for part in (REACH, MORE) for row in store.read_rows(name(part)))      # noqa: E731
    of_the_start = total(lambda part: ladder_ceiling.stored_file("pre_r64", part, "l4"))
    solved_by = lambda model: sum(sum(counts) > 0 for counts in zip(*([row["resolved"] for row in store.read_rows(f"episodes_l4_{part}_{model}_problems.jsonl")] for part in (REACH, MORE))))      # noqa: E731
    for model in ("old_rule", "reward_rows", "rehearse"):
        read = report["models"][model]
        read = read["measured_and_not_read"] if read["primary"] is None else read
        assert read["primary"]["problems"] == len(goal) == 1 and read["beside_the_primary"]["attempts_each"] == prepare["attempts_a_goal_problem"]
        assert read["beside_the_primary"]["successes"] == total(lambda part: f"episodes_l4_{part}_{model}_problems.jsonl") and read["beside_the_primary"]["successes_of_the_base"] == of_the_start
        assert read["primary"]["resolved_after"] == solved_by(model)
        # THE REFERENCE: `old_rule` against the start model's stored rows; the two others against `old_rule`'s measured rows.
        against = sum(sum(counts) > 0 for counts in zip(*([row["resolved"] for row in store.read_rows(ladder_ceiling.stored_file("pre_r64", part, "l4"))] for part in (REACH, MORE))))
        assert read["primary"]["resolved_before"] == (against if model == "old_rule" else solved_by("old_rule"))
        assert read["secondary"]["trained_on"]["rows"] == sets[model]["rows"] and ("by_itself" in read) is (model == "old_rule")
    assert report["what_the_old_rule_gave"]["read"] is True and report["what_the_old_rule_gave"]["rank"] == 16 and report["the_goal_set_again_of_the_start"]["problems"] == len(again)
    assert set(report["hot"]) >= {"old_rule", "pre_r64"} and report["hot"]["pre_r64"]["hot"]["all_of_g"]["attempts"] == hot["attempts_a_goal_problem"]
    assert json.loads(store.path(ladder_l4_rows.REPORT_FILE).read_text()) == json.loads(json.dumps(report, default=str))
    assert store.done_summary(ladder_l4_rows.REPORT) == {"stage": "l4", "label": "pretrained on published proofs", "check": "L4t", "headline": report["headline"],
                                                         "branches": report["branches"], "ok": True}
    assert report["adapters"]["there"] == ["old_rule", "rehearse", "reward_rows"] and "`pre_r64` is its own run's and was only read" in report["adapters"]["what"]
    # What it leaves: its own files; nothing of it is in another run's directory.
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == sorted(OWN_FILES)
    # A rerun returns what is stored: nothing is trained or sampled again.
    sent, sampled = len(r64.l4.lean.sources), r64.l4.solver.calls
    rerun = _stage(config, monkeypatch)
    assert len(calls) == 3 and (len(r64.l4.lean.sources), r64.l4.solver.calls) == (sent, sampled) and {path: _all(path) for path in read_only} == read_only
    assert {name: summary for name, summary in rerun.items() if "report" not in name} == {name: summary for name, summary in summaries.items() if "report" not in name}
    # A step of a run from `pre` is not one of this run: refused.
    for name in ("ladder_l4_rows_measure_hot_with", "ladder_l4_rows_measure_hot_pre"):
        with pytest.raises(RuntimeError, match=f"the step {name} is not one of a run from `pre_r64` "):
            ladder_l4_rows.STEPS[name](config)


def test_as_on_the_gpu_each_measure_step_serves_its_adapter_under_a_model_server_started_for_rank_64_and_no_other_stage_does(r64, monkeypatch):
    config, kits = r64.config, r64.kits
    _stage(config, monkeypatch, until="ladder_l4_rows_measure_rehearse")
    store, start = r64.store(), r64.rank / "adapters" / "pre_r64"
    del kits[:]
    with monkeypatch.context() as patched:
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        for name in ("vllm", "vllm.lora"):
            patched.setitem(sys.modules, name, SimpleNamespace())
        patched.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda name, number, path: (name, number, path)))
        ladder_l4_rows.STEPS["ladder_l4_rows_measure_hot_old_rule"](config)
        ladder_l4_rows.STEPS["ladder_l4_rows_measure_hot_start"](config)
        # Without its adapter a measurement cannot be made: the start model's is its own run's, `old_rule`'s is this stage's.
        for step, adapter, match in (("ladder_l4_rows_measure_hot_start", start, "adapters/pre_r64 is not there: the adapter of `pre_r64` is its own run's, only read here"),
                                     ("ladder_l4_rows_measure_hot_old_rule", store.root / "adapters" / "old_rule", "adapters/old_rule is not there: the adapter `old_rule` is kept by this stage")):
            marker = store.path(f"{step}.done.json")
            kept = marker.read_bytes()
            marker.unlink()
            adapter.rename(adapter.with_name("elsewhere"))
            with pytest.raises(RuntimeError, match=match):
                ladder_l4_rows.STEPS[step](config)
            adapter.with_name("elsewhere").rename(adapter)
            marker.write_bytes(kept)
    assert [(kit["temperature"], kit["adapter"], kit["max_lora_rank"]) for kit in kits] == [(1.2, ("ladder_l4_rows_hot_old_rule", 1, str(store.root / "adapters" / "old_rule")), 64)] * 2 + [
        (1.2, ("ladder_l4_rows_hot_pre_r64", 1, str(start)), 64)] * 2
    # ---- NO OTHER STAGE: the pretraining's and the arm's own measurements, run again by a task that carries this stage's variables, serve under the config's own limit
    del kits[:]
    for directory, markers in ((r64.pre, (ladder_l4.MEASURE, "episodes_l4_rungs_pre", "episodes_l4_reach_pre", "episodes_l4_more_pre")),
                               (r64.arm, (ladder_l3d2.measure_marker("with"), "episodes_l3d2_rungs_with", "episodes_l3d2_reach_with", "episodes_l3d2_more_with"))):
        for marker in markers:
            (directory / f"{marker}.done.json").unlink()
        for path in [*directory.glob("episodes_l4_*_pre_block_*.done.json"), *directory.glob("episodes_l3d2_*_with_block_*.done.json")]:
            path.unlink()
    ladder_l4.ladder_l4_pretrain_measure(config)
    monkeypatch.setenv(ARM, "t010_assembly_pre")
    monkeypatch.setenv(ladder_l2.L2_ROUNDS_VARIABLE, "2")
    ladder_l3d2.STEPS["ladder_l3d2_measure_with"](config)
    assert len(kits) == 6 and all(kit["max_lora_rank"] == 16 and kit["rank"] == 16 and kit["vllm"] is config["vllm"] and kit["lora"] is config["lora"] for kit in kits)
    # ---- where a rank is put into a config is still ONE function, of the check's module; this stage calls it once, and names no limit of the model server itself
    gpu = {path.name: path.read_text() for path in (PACKAGE / "gpu").glob("*.py")}
    assert sorted(name for name, text in gpu.items() if re.search(r"[\"']max_lora_rank[\"']\s*:", text)) == ["ladder_l4_rank.py"]
    assert not [name for name in ("ladder_l4_rows.py", "ladder_l4_start.py", "ladder_l4b.py", "ladder_l2.py", "ladder_l3d2.py", "ladder_assembly.py") if "max_lora_rank" in gpu[name]]
    # `config_from` is ONE function of the module L4t and an arm of the loop both ask (`ladder_l4_start.py`: L4b built the arm from `pre_r64`), and it alone calls `config_of`.
    assert gpu["ladder_l4_start.py"].count("config_of(") == 1 and "ladder_l4_rank.config_of(config, start.check)" in gpu["ladder_l4_start.py"].split("def config_from")[1].split("\ndef ")[0]
    assert "config_of(" not in gpu["ladder_l4_rows.py"] and "def config_from" not in gpu["ladder_l4_rows.py"] and ladder_l4_rows.config_from is ladder_l4_start.config_from
    assert gpu["ladder_l4_rows.py"].count("config_from(") == 4                 # the prepare step's recorded recipe; the training; the two kinds of measure step
    # An arm asks it in ONE place of its own (`ladder_l2.config_for`); the map of the arm's start model once more (`ladder_l4b_map`); no other stage's module does.
    assert gpu["ladder_l2.py"].count("config_from(") == 1 and "ladder_l4_start.config_from(" in gpu["ladder_l2.py"].split("def config_for")[1].split("\ndef ")[0]
    assert gpu["ladder_l4b.py"].count("config_from(") == 1 and not [name for name in ("ladder_l4.py", "ladder_l3d2.py", "ladder_ceiling.py", "ladder_assembly.py") if "config_from(" in gpu[name]]
    assert not [name for name in ("ladder_l4.py", "ladder_l3d2.py", "ladder_l2.py", "ladder_ceiling.py", "ladder_l4_start.py", "ladder_l4b.py") if START in gpu[name]]
    assert sorted(str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if START in path.read_text()) == ["gpu/ladder_l4_rows.py", "runner/entry.py"]


def test_a_training_whose_saved_adapter_is_not_the_start_adapters_rank_fails_its_step_and_nothing_of_it_is_measured(r64, monkeypatch):
    config, calls = r64.config, r64.calls
    ladder_l4_rows.ladder_l4_rows_prepare(config)
    store = r64.store()
    train, measure = ladder_l4_rows.STEPS["ladder_l4_rows_train_old_rule"], ladder_l4_rows.STEPS["ladder_l4_rows_measure_old_rule"]
    for saves, match in (((16, 32, None), r"the adapter `old_rule` saved at .*adapters/old_rule has rank 16 and alpha 32 in its adapter_config.json and matrices of rank .16.; the start "
                                          r"adapter `pre_r64` is rank 64 and alpha 128, and every model here is at the start adapter's rank. Nothing of this model is measured"),
                         ((64, 32, None), r"has rank 64 and alpha 32 in its adapter_config.json and matrices of rank .64.; the start adapter `pre_r64` is rank 64 and alpha 128"),
                         ((16, 128, 64), r"has rank 64 and alpha 128 in its adapter_config.json and matrices of rank .16.; the start adapter `pre_r64` is rank 64 and alpha 128")):
        _trainings(monkeypatch, calls, saves={"old_rule": saves})
        with pytest.raises(RuntimeError, match=match):
            _on_the_gpu(monkeypatch, train, config)
        assert not store.is_done("ladder_l4_rows_train_old_rule")              # the step is NOT marked done
        with pytest.raises(RuntimeError, match="the measurement of L4t's model `old_rule` needs the step ladder_l4_rows_train_old_rule of this run, which is not done"):
            measure(config)
        shutil.rmtree(store.root / "adapters" / "old_rule")
    # The training as it is: the start adapter's rank reaches it, and the step is marked done with what it read back.
    _trainings(monkeypatch, calls)
    done = _on_the_gpu(monkeypatch, train, config)
    assert done["adapter_saved"] == {"rank": 64, "alpha": 128, "modules": 2, "tensors": 4, "ranks": [64], "numbers": 64 * 25, "numbers_a_unit_of_rank": 25} and store.is_done("ladder_l4_rows_train_old_rule")
    # The stand-in trains and saves nothing: nothing is read back, and the summary says nothing of an adapter saved.
    trained_so_far = len(calls)
    assert "adapter_saved" not in ladder_l4_rows.STEPS["ladder_l4_rows_train_reward_rows"](config) and len(calls) == trained_so_far
    # The check's setting moved after the run was prepared (another alpha for the same model): refused before anything is trained or sampled.
    monkeypatch.setitem(config["ladder_loop"]["l4"]["rank_check"], "alpha", 64)
    monkeypatch.setitem(config["ladder_loop"]["l4"]["rank_check"], "fallback_alpha", 32)
    for step in ("ladder_l4_rows_train_rehearse", "ladder_l4_rows_measure_old_rule", "ladder_l4_rows_measure_hot_start"):
        with pytest.raises(RuntimeError, match="this run was prepared from `pre_r64` at rank 64 and alpha 128 .what the run that made it recorded., and ladder_loop.l4.rank_check now "
                                               "gives it rank 64 and alpha 64: a model here is trained and served at the start adapter's own rank"):
            _on_the_gpu(monkeypatch, ladder_l4_rows.STEPS[step], config) if "train" in step else ladder_l4_rows.STEPS[step](config)


def test_the_prepare_step_from_pre_r64_refuses_before_anything_is_written(r64, monkeypatch, tmp_path):
    config, arm_run, pre, rank_run = r64.config, r64.arm, r64.pre, r64.rank
    as_stored = {path: _all(path) for path in (arm_run, pre, rank_run)}
    root = r64.store().root
    task = "the task of stage `ladder_l4_rank` for seed 0 .`python -m rlvr_lean.runner.entry --stage ladder_l4_rank --seeds 0`; for the smoke run, stage `ladder_l4_rank_smoke`."

    def refused(match, error=RuntimeError):
        with pytest.raises(error, match=match):
            ladder_l4_rows.ladder_l4_rows_prepare(config)
        assert root.name == "ladder_l4_rows_pre_r64_seed0" and not _all(root)  # nothing was written: no marker, no file

    def with_a_change(directory, change, match, error=RuntimeError):
        """One of the three runs with one thing of it changed, refused; then put back as it was."""
        change()
        refused(match, error)
        for name, content in as_stored[directory].items():
            (directory / name).write_bytes(content)

    def rewritten(directory, name, change):
        return lambda: (directory / name).write_text(json.dumps(change(json.loads(as_stored[directory][name]))))

    def rows_of(directory, name, change):
        own = [json.loads(line) for line in as_stored[directory][name].decode().splitlines() if line.strip()]
        return lambda: (directory / name).write_text("".join(json.dumps(row) + "\n" for row in change(own)))

    # ---- THE CHECK'S RUN: its report, what its three steps recorded, the start model's rows and their markers; each missing one is refused, naming the task to run
    for name in ("report_ladder_l4_rank.json", "ladder_l4_rank_prepare.done.json", "ladder_l4_rank_train.done.json", "ladder_l4_rank_measure.done.json"):
        with_a_change(rank_run, (rank_run / name).unlink, f"ladder_l4_pretrain_r64_seed0 does not hold .'{name}'.: the report of the check that made `pre_r64`, or what its prepare, train "
                                                          f"or measure step recorded. L4t makes no round and no pretraining: it reads what {task} wrote on this box. Run that task to "
                                                          "its end first; nothing was written")
    for name in ("episodes_l4_rungs_pre_r64_problems.jsonl", "episodes_l4_reach_pre_r64.done.json", "episodes_l4_more_pre_r64_problems.jsonl"):
        with_a_change(rank_run, (rank_run / name).unlink, f"ladder_l4_pretrain_r64_seed0 does not hold .'{name}'.: `pre_r64`'s per-problem rows on a set it was measured on, or that "
                                                          f"set's own marker. L4t makes no round and no pretraining: it reads what {task} wrote")
    # ---- a report that is NOT TO BE READ, or whose CHECKS FAILED: refused, naming the task
    with_a_change(rank_run, rewritten(rank_run, "report_ladder_l4_rank.json", lambda report: {**report, "ok": False}),
                  f"the run ladder_l4_pretrain_r64_seed0 cannot carry L4t: its report is not to be read .Lean gave no verdict on too much of a set.. `pre_r64`'s rows stand on the other "
                  f"side of every comparison here. Queue {task} again; nothing was written")
    failing = lambda report: {**report, "inconclusive": True, "can_this_run_see_a_win": {**report["can_this_run_see_a_win"], "the_training_took": {"passes": False}}}      # noqa: E731
    with_a_change(rank_run, rewritten(rank_run, "report_ladder_l4_rank.json", failing), "the run ladder_l4_pretrain_r64_seed0 cannot carry L4t: its checks failed .the_training_took.")
    with_a_change(rank_run, rewritten(rank_run, "report_ladder_l4_rank.json", lambda report: {**report, "inconclusive": True}), "cannot carry L4t: its checks failed .INCONCLUSIVE.")
    # ---- a run that is not the one it is named for: another seed, another model, another rank or alpha than the check's, an adapter saved at another rank
    change_of = lambda **changed: (lambda recorded: {**recorded, **changed})      # noqa: E731
    with_a_change(rank_run, rewritten(rank_run, "ladder_l4_rank_prepare.done.json", change_of(seed=3)),
                  "the run ladder_l4_pretrain_r64_seed0 recorded seed 3 and the model 'pre_r64'; `pre_r64` is the model the check of the adapter's rank made at seed 0")
    with_a_change(rank_run, rewritten(rank_run, "ladder_l4_rank_prepare.done.json", lambda recorded: {**recorded, "rank_check": {**recorded["rank_check"], "model": "pre_r32"}}),
                  "recorded seed 0 and the model 'pre_r32'; `pre_r64` is the model")
    for lora, said in (({"rank": 32, "alpha": 128}, "rank 32 and alpha 128"), ({"rank": 64, "alpha": 64}, "rank 64 and alpha 64")):
        with_a_change(rank_run, rewritten(rank_run, "ladder_l4_rank_prepare.done.json", lambda recorded, lora=lora: {**recorded, "recipe": {**recorded["recipe"], "lora": {
            **recorded["recipe"]["lora"], **lora}}}), f"the run ladder_l4_pretrain_r64_seed0 recorded an adapter of {said} for `pre_r64`; ladder_loop.l4.rank_check gives that model "
                                                      "rank 64 and alpha 128. Every model here is trained and served at the START ADAPTER's rank, which is the one its own run recorded")
    with_a_change(rank_run, rewritten(rank_run, "ladder_l4_rank_train.done.json", lambda recorded: {**recorded, "adapter_saved": {**recorded["adapter_saved"], "ranks": [16]}}),
                  "recorded an adapter of rank 64 and alpha 128 for `pre_r64` and saved one of rank 64 and alpha 128 with matrices of rank .16.; ladder_loop.l4.rank_check gives")
    # ---- `pre_r64` must have been trained on the file `pre` was: the SHA-256 and the rows both prepare steps recorded
    with_a_change(rank_run, rewritten(rank_run, "ladder_l4_rank_prepare.done.json", change_of(pretraining_file_sha256="0" * 64)),
                  "the run ladder_l4_pretrain_r64_seed0 trained `pre_r64` on a file of SHA-256 0{64} .12 rows. and `pre` was trained on .* .12 rows.: the rehearsal rows are drawn among "
                  "the rows `pre` was trained on, and they must be rows `pre_r64` was trained on too")
    with_a_change(rank_run, rewritten(rank_run, "ladder_l4_rank_prepare.done.json", change_of(rows=11)), "trained `pre_r64` on a file of SHA-256 .* .11 rows. and `pre` was trained on")
    # ---- the start model's stored rows must pair with this run: its seeds, its problems, Lean having answered
    marker = json.loads(as_stored[rank_run]["episodes_l4_reach_pre_r64.done.json"])
    with_a_change(rank_run, rewritten(rank_run, "episodes_l4_reach_pre_r64.done.json", change_of(sampling_seed=5)),
                  f"ladder_l4_pretrain_r64_seed0 measured `pre_r64` .reach. with sampling seed 5 and {marker['episodes_each']} episodes; this config gives {marker['sampling_seed']} and "
                  f"{marker['episodes_each']}: L4t's models would not pair with the stored results")
    with_a_change(rank_run, rows_of(rank_run, "episodes_l4_rungs_pre_r64_problems.jsonl", lambda own: own[:-1]), "measured `pre_r64` .rungs. on other problems than L1's run holds for it")
    with_a_change(rank_run, rows_of(rank_run, "episodes_l4_more_pre_r64_problems.jsonl", lambda own: [{**own[0], "attempts_without_an_answer": 60}, *own[1:]]),
                  "measured `pre_r64` .more. with too many attempts without a verdict from Lean: that set stands on the other side of a comparison")
    # ---- THE ARM'S RUN MUST HAVE STARTED FROM `pre`: the rows are the rank-16 rounds', whatever this run starts from
    of_the_arm = lambda **changed: rewritten(arm_run, "ladder_l2_prepare.done.json", change_of(**changed))      # noqa: E731
    for changed in ({"start": "pre_r64"}, {"start": None}, {"seed": 3}, {"arm": "t010_assembly"}):
        with_a_change(arm_run, of_the_arm(**changed), "the arm's run ladder_l2_t010_assembly_pre_seed0 recorded seed .* this task's seed is 0 and ladder_loop.l4.rows names the arm "
                                                      "'t010_assembly_pre', whose rounds were made from `pre`: from `pre_r64` the rows are still the rounds that arm made from `pre`")
    with_a_change(arm_run, of_the_arm(start_adapter=str(rank_run / "adapters" / "pre_r64")),
                  "was trained from .*ladder_l4_pretrain_r64_seed0/adapters/pre_r64 and the rounds read here are those made from .*ladder_l4_pretrain_seed0/adapters/pre: not the same adapter")
    for name, what in (("l3d2_training_without.jsonl", "the twin's rows"), ("training_examples_r2.jsonl", "a round's training examples"), ("ladder_l2_prepare.done.json", "what the arm's steps recorded")):
        with_a_change(arm_run, (arm_run / name).unlink, f"ladder_l2_t010_assembly_pre_seed0 does not hold .'{name}'.: .*{what}.*the task of stage `ladder_l4` for seed 0")
    # ---- of `pre`'s run: what its prepare step recorded, and the rows it was trained on
    for name in ("ladder_l4_pretrain_prepare.done.json", "l4_pretraining_rows.jsonl"):
        with_a_change(pre, (pre / name).unlink, f"ladder_l4_pretrain_seed0 does not hold .'{name}'.: what `pre`'s prepare step recorded, or the rows it was trained on .the rehearsal is "
                                                "drawn among them.. L4t makes no round and no pretraining: it reads what the task of stage `ladder_l4_pretrain` for seed 0")
    with_a_change(pre, rows_of(pre, "l4_pretraining_rows.jsonl", lambda own: own[:-1]), "stored 11 rows `pre` was trained on and L4's pretraining file holds 12")
    # ---- as on the GPU: without the start adapter nothing is prepared (the arm's last model is not read from this start)
    adapter = rank_run / "adapters" / "pre_r64"
    adapter.rename(adapter.with_name("elsewhere"))
    with monkeypatch.context() as patched:
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        refused(f"adapters/pre_r64 is not there: the adapter `pre_r64` is what every model here is trained from. Run {task} again; nothing was written")
    adapter.with_name("elsewhere").rename(adapter)
    shutil.rmtree(arm_run / "adapters" / "m2")
    with monkeypatch.context() as patched:
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        assert ladder_l4_rows.ladder_l4_rows_prepare(config)["with_adapter"] is None
    shutil.rmtree(root)
    # ---- A SMOKE RUN ALONE goes on from a check whose own checks failed, and records it; a report that is not to be read is refused for it too
    rewritten(rank_run, "report_ladder_l4_rank.json", failing)()
    monkeypatch.setenv(START_CHECKS, "smoke")
    waived = ladder_l4_rows.ladder_l4_rows_prepare(config)["start_run_read"]
    assert (waived["checks_failed"], waived["checks_waived_for_a_smoke_run"]) == (["the_training_took"], True)
    shutil.rmtree(root)
    rewritten(rank_run, "report_ladder_l4_rank.json", lambda report: {**report, "ok": False})()
    with pytest.raises(RuntimeError, match="cannot carry L4t: its report is not to be read"):
        ladder_l4_rows.ladder_l4_rows_prepare(config)
    monkeypatch.delenv(START_CHECKS)
    (rank_run / "report_ladder_l4_rank.json").write_bytes(as_stored[rank_run]["report_ladder_l4_rank.json"])
    # ---- WHAT STANDS BESIDE IS REFUSED FOR NOTHING: without `pre`'s rows on G, or `without`'s, the run prepares and says the rank-16 figures are not there
    for directory, name in ((pre, "episodes_l4_more_pre_problems.jsonl"), (arm_run, "ladder_l3d2_measure_without.done.json"), (pre, "report_ladder_l4_pretrain.json"),
                            (pre, "l4_goal_set_again.jsonl")):
        (directory / name).unlink()
        prepared = ladder_l4_rows.ladder_l4_rows_prepare(config)
        needed = name not in ("report_ladder_l4_pretrain.json", "l4_goal_set_again.jsonl")       # `pre`'s report and ITS G' are not read from this start at all
        assert prepared["rank_16"]["read"] is (not needed) and (name in prepared["rank_16"].get("why", "")) is needed
        assert sorted(name for name in _all(r64.store().root) if "l4_stored_" in name and "pre_r64" not in name) == ([] if needed else [
            "l4_stored_pre_more.jsonl", "l4_stored_pre_reach.jsonl", "l4_stored_without_more.jsonl", "l4_stored_without_reach.jsonl"])
        shutil.rmtree(root)
        (directory / name).write_bytes(as_stored[directory][name])


def test_the_goal_set_again_is_the_start_models_made_from_its_first_sampling_and_never_pres(r64, monkeypatch):
    config, rank_run, pre = r64.config, r64.rank, r64.pre
    goal = [row["problem_id"] for row in _file_rows(r64.arm / "l3d2_heldout_groups.jsonl") if row["group"] == "goal"]
    stored = {part: _file_rows(rank_run / f"episodes_l4_{part}_pre_r64_problems.jsonl") for part in (REACH, MORE)}
    of_pre = [row["problem_id"] for row in _file_rows(pre / ladder_l4.AGAIN_FILE)]

    def prepared_with(first, more):
        """The start model's stored rows with these successes on the goal problem in its first sampling and in its second."""
        for part, resolved in ((REACH, first), (MORE, more)):
            (rank_run / f"episodes_l4_{part}_pre_r64_problems.jsonl").write_text("".join(json.dumps({**row, "resolved": resolved}) + "\n" for row in stored[part]))
        summary = ladder_l4_rows.ladder_l4_rows_prepare(config)
        again = [row["problem_id"] for row in r64.store().read_rows(ladder_l4.AGAIN_FILE)]
        shutil.rmtree(r64.store().root)
        return summary, again

    # Not solved in its FIRST sampling (whatever its second gave): the problem is of G'. Solved there: it is not.
    summary, again = prepared_with(first=0, more=5)
    assert again == goal and (summary["goal_set_again"], summary["goal_set_again_made_here"]["problems"], summary["goal_set_again_made_here"]["solved_in_the_first_sampling"]) == (1, 1, 0)
    summary, again = prepared_with(first=3, more=0)
    assert again == [] and (summary["goal_set_again"], summary["goal_set_again_made_here"]["solved_in_the_first_sampling"]) == (0, 1)
    # `pre`'s own G' is not what is stored: with `pre`'s file as it is, the start model's G' follows the start model's rows alone.
    assert [prepared_with(first=0, more=0)[1], prepared_with(first=1, more=0)[1]] == [goal, []] and of_pre in (goal, [])
    assert _file_rows(pre / ladder_l4.AGAIN_FILE) == [{"problem_id": problem_id} for problem_id in of_pre]             # and `pre`'s run is not written to


def test_the_report_of_a_run_without_rank_16s_rows_says_they_are_not_there(r64, monkeypatch):
    config = r64.config
    (r64.pre / "episodes_l4_reach_pre_problems.jsonl").unlink()                # `pre`'s rows on G were not on the box
    summaries = _stage(config, monkeypatch)
    report = summaries["ladder_l4_rows_report"]
    assert summaries["ladder_l4_rows_prepare"]["rank_16"]["read"] is False and report["what_the_old_rule_gave"]["read"] is False and report["ok"] is True
    assert "episodes_l4_reach_pre_problems.jsonl" in report["what_the_old_rule_gave"]["why"] and list(report["branches"]) == ["old_rule", "reward_rows", "rehearse"]
    assert sorted(name for name in _all(r64.store().root) if "l4_stored_" in name) == ["l4_stored_pre_r64_more.jsonl", "l4_stored_pre_r64_reach.jsonl", "l4_stored_pre_r64_rungs.jsonl"]


# ----------------------------------------------------------------------------- a run from `pre` is what it was
def test_a_run_from_pre_writes_what_it_wrote_before_another_start_was_built(rows, monkeypatch):  # noqa: F811
    config = rows.config
    monkeypatch.delenv(START, raising=False)
    summaries = {}
    for environment, step, options in map(entry.step_fields, entry.STAGES["ladder_l4_rows"]):
        assert options == {}                                                   # the stage names no start: the setting's, `pre`
        if environment == "gpu" and step in ladder_l4_rows.STEPS:
            summaries[step] = _on_the_gpu(monkeypatch, ladder_l4_rows.STEPS[step], config) if "_train_" in step else ladder_l4_rows.STEPS[step](config)
    store = rows.store()
    prepare, report = summaries["ladder_l4_rows_prepare"], summaries["ladder_l4_rows_report"]
    assert store.root.name == "ladder_l4_rows_seed0" and list(summaries) == ladder_l4_rows.steps_of("pre") == list(ladder_l4_rows.STEPS)[:8]
    # Key for key, in their order: nothing of another start is recorded, and no file of `old_rule` is written.
    assert list(prepare) == PREPARE_KEYS_FROM_PRE and list(report) == REPORT_KEYS_FROM_PRE and not [key for key in (*BESIDE, "rank", "the_old_rule_by_itself") if key in prepare or key in report]
    assert (prepare["start"], prepare["models"], list(prepare["training_sets"]), prepare["hot"]["models"], list(prepare["stored_models"])) == (
        "pre", ["rehearse", "reward_rows"], ["rehearse", "reward_rows"], ["with", "pre"], ["pre", "with", "without"])
    assert prepare["recipe"]["lora"] == config["lora"] and prepare["start_adapter"].endswith("ladder_l4_pretrain_seed0/adapters/pre") and prepare["with_adapter"].endswith("adapters/m2")
    assert list(report["branches"]) == ["rehearse", "reward_rows"] == list(report["models"]) and set(report["what_the_old_rule_gave"]) == {
        "what", "on_all_of_g", "without_minus_pre", "with_minus_pre", "with_against_pre", "without_against_pre"}
    assert not [name for name in _all(store.root) if "old_rule" in name or "pre_r64" in name or "hot_start" in name]
    assert all("adapter_saved" not in summaries[f"ladder_l4_rows_train_{model}"] and "peak_reserved_gb" not in summaries[f"ladder_l4_rows_train_{model}"] for model in MODELS)
    # Every training was handed the config's own adapter settings and every engine the config's own model server's: the SAME objects, nothing copied or raised.
    assert [call["more"] for call in rows.calls] == [{"start": rows.pre / "adapters" / "pre"}] * 2
    assert not [line for line in report["lines"] if "old_rule" in line or "RANK" in line or "pre_r64" in line]


# ------------------------------------------------------------------------------------------- the smoke stage
def test_the_smoke_stage_runs_from_the_smoke_pre_r64_at_rank_64_in_the_l4_smoke_runs_world(arm, monkeypatch):  # noqa: F811
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    config, calls, kits = arm.config, [], []
    monkeypatch.delenv(ARM)
    for name in (RUN, ARM_RUN, MINIMUM, START, START_CHECKS, CHECK, RANK_RUN):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ladder_round.ROUND_RUN_VARIABLE, "ladder_l1_smoke")
    _run_stage(config)                                                         # what the L1 smoke task leaves on the box
    monkeypatch.delenv(ladder_round.ROUND_RUN_VARIABLE)
    _trainings(monkeypatch, calls)
    # ---- the pretraining's smoke task, the arm's, then the check's: what the three smoke tasks leave on the box
    for stage_name in ("ladder_l4_pretrain_smoke", "ladder_l4_smoke", "ladder_l4_rank_smoke"):
        options = entry.step_fields(entry.STAGES[stage_name][2])[2]
        with monkeypatch.context() as patched:                                 # one task's environment: gone when the task is
            patched.delenv(ladder_loop.DATA_VARIABLE)
            for name, value in options["environment"].items():
                patched.setenv(name, value)
            if stage_name == "ladder_l4_smoke":
                arm.with_lean()
                patched.setattr(ladder_l2, "_start_request", lambda start: ("the start adapter", start))
            steps = {**ladder_l2.STEPS, **ladder_l3d2.STEPS, **ladder_l4.STEPS, **ladder_l4_rank.STEPS}
            for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
                if environment == "gpu" and step in steps:
                    _on_the_gpu(patched, steps[step], config) if "_train" in step else steps[step](config)
    runs = pipeline.STORE / "runs-v4.27"
    pre, arm_run, rank_run = runs / "ladder_l4_pretrain_smoke", runs / "ladder_l4_smoke", runs / "ladder_l4_rank_smoke"
    assert (rank_run / "adapters" / "pre_r64" / "adapter_config.json").exists() and (arm_run / "adapters" / "m2").is_dir() and (pre / "adapters" / "pre").is_dir()
    as_stored = {path: _all(path) for path in (pre, arm_run, rank_run)}
    # ---- this stage's smoke task: the smoke stage of a run from `pre`, with the start named, the check's SMOKE run as where it lives, and a run directory of its own
    options = entry.step_fields(entry.STAGES["ladder_l4_rows_r64_smoke"][2])[2]
    assert all(entry.step_fields(step)[2] == options for step in entry.STAGES["ladder_l4_rows_r64_smoke"])
    of_the_smoke = entry.step_fields(entry.STAGES["ladder_l4_rows_smoke"][2])[2]["environment"]
    assert options["environment"] == {**of_the_smoke, START: "pre_r64", RANK_RUN: "ladder_l4_rank_smoke", RUN: "ladder_l4_rows_r64_smoke", START_CHECKS: "smoke"}
    assert (of_the_smoke[ladder_l4.RUN_VARIABLE], of_the_smoke[ARM_RUN], of_the_smoke[MINIMUM]) == ("ladder_l4_pretrain_smoke", "ladder_l4_smoke", "2")
    monkeypatch.delenv(ladder_loop.DATA_VARIABLE)
    for name, value in options["environment"].items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (
        kits.append((given["sampling"]["temperature"], given["vllm"]["max_lora_rank"])), (arm.solver, stand_in_parameters))[1]))
    del calls[:]
    summaries = _stage(config, monkeypatch, stage_name="ladder_l4_rows_r64_smoke")
    store = ladder_l4_rows._store(config, ladder_l4_rows.the_settings(config))
    assert store.root.name == "ladder_l4_rows_r64_smoke" and {path: _all(path) for path in as_stored} == as_stored and list(summaries) == STEPS
    assert not [path.name for path in runs.iterdir() if path.name.startswith("ladder_l4_rows_") and path.name != "ladder_l4_rows_r64_smoke"]
    prepare = summaries["ladder_l4_rows_prepare"]
    assert (prepare["start"], prepare["start_run"], prepare["start_adapter"], prepare["arm_run"], prepare["pretraining_run"]) == (
        "pre_r64", str(rank_run), str(rank_run / "adapters" / "pre_r64"), str(arm_run), str(pre))
    assert [sampling["name"] for sampling in prepare["goal_samplings"]] == [REACH] and prepare["attempts_a_goal_problem"] == 32 and prepare["start_recipe"]["rank"] == 64
    sets = prepare["training_sets"]
    assert list(sets) == ["old_rule", "reward_rows", "rehearse"] and sets["old_rule"]["rows"] == sets["old_rule"]["twin_rows"] > 0 and sets["rehearse"]["rows"] == 2 * sets["old_rule"]["rows"]
    assert sets["reward_rows"]["minimum_of_a_smoke_run"] == 2 and prepare["goal_set_again"] in (0, 1)
    # ALL THREE models are trained from the smoke `pre_r64` AT RANK 64 and measured under a model server started for rank 64; the two hot measurements at the hot temperature.
    assert [(call["lora"]["rank"], call["lora"]["alpha"], call["more"]) for call in calls] == [(64, 128, {"start": rank_run / "adapters" / "pre_r64"})] * 3
    assert [list(call["schedule"]["checkpoint_steps"]) for call in calls] == [["old_rule"], ["reward_rows"], ["rehearse"]]
    assert sorted(path.name for path in (store.root / "adapters").iterdir()) == ["old_rule", "rehearse", "reward_rows"]
    assert kits == [(1.0, 64)] * 6 + [(1.2, 64)] * 2                           # the rungs and G for each trained model; G alone for `old_rule`, then for `pre_r64`
    assert [summaries[f"ladder_l4_rows_measure_hot_{step}"]["goal"][REACH]["set"] for step in ("old_rule", "start")] == ["l4_reach_hot_old_rule", "l4_reach_hot_pre_r64"]
    report = summaries["ladder_l4_rows_report"]
    assert report["label"] == "pretrained on published proofs" and all(line.startswith(f"{SAY}: ") for line in report["lines"]) and store.is_done(ladder_l4_rows.REPORT)
    assert report["branches"]["old_rule"] in old_rule_outcomes(64) and all(report["branches"][model] in BRANCHES for model in ("reward_rows", "rehearse"))
    assert any("the stored models have ONE sampling of G here (a smoke run)" in line for line in report["lines"])


# -------------------------------------------------------------------------------------------- registration
def test_the_stage_from_pre_r64_follows_the_settings_order_with_a_guard_before_every_gpu_step():
    whole = milestone2.load_config(PACKAGE / "config" / "experiment.yaml")
    settings = whole["ladder_loop"]["l4"]["rows"]
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l4_rows_r64"]]
    assert [(environment, step) for environment, step, _ in steps] == [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", STEPS[0]),
                                                                       *(pair for step in STEPS[1:] for pair in (("guard", None), ("gpu", step)))]
    # THE ORDER IS THE SETTING'S: each model trained, then measured, before the next; `old_rule` first, so that the cap's question is answered first.
    in_order = [step for _, step, _ in steps if step and "_train_" in step]
    assert in_order == [f"ladder_l4_rows_train_{model}" for model in settings["models_from_another_start"]] == list(TRAININGS)
    assert STEPS == ladder_l4_rows.steps_of("pre_r64") and set(STEPS) <= set(ladder_l4_rows.STEPS) and list(ladder_l4_rows.STEPS)[8:] == [
        "ladder_l4_rows_train_old_rule", "ladder_l4_rows_measure_old_rule", "ladder_l4_rows_measure_hot_old_rule", "ladder_l4_rows_measure_hot_start"]
    assert all(options == {"environment": {START: "pre_r64"}} for _, _, options in steps)
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l4_rows_r64_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in steps] and all(options == smoke[0][2] for _, _, options in smoke)
    # The start is named by these two stages alone; the smoke alone waives the checks of the run its start model was made by.
    names = lambda variable: sorted(name for name, own in entry.STAGES.items() if any(variable in entry.step_fields(step)[2].get("environment", {}) for step in own))      # noqa: E731
    assert names(START) == ["ladder_l4_rows_r64", "ladder_l4_rows_r64_smoke"] and names(START_CHECKS) == ["ladder_l4_rows_r64_smoke"]
    assert entry.step_fields(entry.STAGES["ladder_l4_rows_r64_smoke"][2])[2]["environment"][START_CHECKS] == "smoke"
    # The setting: the start stays `pre`; the stage's start is a model of the check's setting, at its rank.
    rank_check = whole["ladder_loop"]["l4"]["rank_check"]
    assert settings["start"] == "pre" and (rank_check["rank"], rank_check["alpha"]) == (64, 128) and (whole["lora"]["rank"], whole["vllm"]["max_lora_rank"]) == (16, 16)
    assert ladder_l4_rank.check_of(whole, "pre_r64").rank == 64 and ladder_l4_rows.models_of("pre_r64") == tuple(settings["models_from_another_start"])
