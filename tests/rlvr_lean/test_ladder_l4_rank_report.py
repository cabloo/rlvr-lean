"""L4r, is the pretrained model capped by the size of its adapter: its rules and its report on hand-made rows with a known
answer (the check, the ONE change, an adapter's numbers, the loss by twentieth, the three checks, the primary against
`pre`, each branch in the spec's words, the secondary reads). Spec: docs/spec/ladder-loop.spec.md, "L4r: is the
pretrained model capped by the size of its adapter?" ("The read, fixed before the run"). Pure: no model, no Lean. The
stage end to end is `test_ladder_l4_rank_stage.py`."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l3d1 import EVALUATION, GROUPS, LENGTHS, RUNG_IDS, _rung  # noqa: E402 - Step 1's hand-made world: 12 goal problems, 3 a length group
from test_ladder_l4 import ON_G, _made  # noqa: E402 - L4's hand-made `pre`, and its `with` (here: the rows of a model that reaches further than `pre`)

from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rank import (  # noqa: E402
    BRANCHES,
    CAPPED,
    CHECKS,
    NOT_THE_CAP,
    THE_SAME,
    WORSE,
    adapter_numbers,
    by_twentieth,
    model_name,
    never_solved,
    rank_and_alpha,
    rank_branch,
    the_one_change,
    training_checks,
)
from rlvr_lean.reporting.ladder_l4 import LABEL, SAY  # noqa: E402
from rlvr_lean.reporting.ladder_l4_rank import NAMED_IN_A_LINE, _named_problems, build_rank_report, cannot_say, timed_out  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
SETTINGS = CONFIG["ladder_loop"]["l4"]
LORA = CONFIG["lora"]


# --------------------------------------------------------------------------------------------- the check
def test_the_check_is_rank_64_with_alpha_128_and_its_fallback_rank_32_with_alpha_64():
    check = SETTINGS["rank_check"]
    assert check == {"rank": 64, "alpha": 128, "fallback_rank": 32, "fallback_alpha": 64} and CHECKS == ("rank", "fallback")
    assert rank_and_alpha(check, "rank") == (64, 128) and rank_and_alpha(check, "fallback") == (32, 64)
    assert (model_name(64), model_name(32)) == ("pre_r64", "pre_r32")
    # alpha / rank stays `pre`'s in both: the adapter's scale is `pre`'s.
    assert 128 / 64 == 64 / 32 == LORA["alpha"] / LORA["rank"] == 2 and (LORA["rank"], LORA["alpha"]) == (16, 32)
    with pytest.raises(ValueError, match="'rank_64' is not a check of ladder_loop.l4.rank_check: it is one of .'rank', 'fallback'."):
        rank_and_alpha(check, "rank_64")
    for changed, match in (({"rank": 0}, "gives the check `rank` rank 0 and alpha 128: each is a positive whole number"),
                           ({"alpha": "128"}, "gives the check `rank` rank 64 and alpha '128'"), ({"fallback_alpha": 64.0}, "gives the check `fallback` rank 32 and alpha 64.0"),
                           ({"rank": True}, "gives the check `rank` rank True"),
                           ({"fallback_rank": 64}, "fallback_rank is 64 and its rank is 64: the fallback is the SMALLER adapter, run when the box cannot hold the rank"),
                           ({"fallback_rank": 128}, "fallback_rank is 128 and its rank is 64")):
        for which in CHECKS:                                                                       # a setting that is no check is refused whichever check is asked for
            with pytest.raises(ValueError, match=match):
                rank_and_alpha({**check, **changed}, which)


# --------------------------------------------------------------------------------------- the ONE change
def _prepared(rank=16, alpha=32, **changed):
    """What a prepare step records of a pass over the pretraining file (the part the ONE change is held on)."""
    recorded = {"seed": 0, "pretraining_file_sha256": "a8d6" * 16, "rows": 24866, "order": "the file's: no row is moved", "effective_batch": 8, "steps": 3109,
                "max_sequence_tokens": 2048, "tokens": 7943531, "pretraining_file": "/somewhere/pretraining.jsonl",
                "recipe": {"what": "the round's", "learning_rate": 1.0e-4, "warmup_steps": 5, "effective_batch": 8, "adapter_seed": 0,
                           "lora": {"rank": rank, "alpha": alpha, "dropout": 0.0, "target_modules": list(LORA["target_modules"])}}}
    for key, value in changed.items():
        where = recorded
        *path, last = key.split(".")
        for part in path:
            where = where[part]
        where[last] = value
    return recorded


def test_the_one_change_is_the_rank_and_the_alpha_and_anything_else_that_differs_is_refused_and_named():
    change = the_one_change(_prepared(), _prepared(64, 128))
    assert (change["rank"], change["alpha"], change["alpha_over_rank"]) == ({"pre": 16, "check": 64}, {"pre": 32, "check": 128}, 2.0)
    assert change["held_the_same"] == [*THE_SAME, "recipe.adapter_seed", "recipe.effective_batch", "recipe.learning_rate", "recipe.warmup_steps", "recipe.what",
                                       "recipe.lora.dropout", "recipe.lora.target_modules"]
    assert THE_SAME == ("seed", "pretraining_file_sha256", "rows", "order", "effective_batch", "steps", "max_sequence_tokens")
    assert the_one_change(_prepared(), _prepared(32, 64))["rank"] == {"pre": 16, "check": 32}       # the fallback is one change too
    # What is NOT held: where the file is on the box, and how its tokens were counted (the stand-in counts otherwise; the SHA-256 is the file).
    assert the_one_change(_prepared(), _prepared(64, 128, tokens=1, pretraining_file="/elsewhere/pretraining.jsonl"))["rank"]["check"] == 64
    for key, value, said in (("seed", 1, "seed is 1 here and 0 in `pre`'s run"), ("pretraining_file_sha256", "0" * 64, "pretraining_file_sha256 is '0{64}' here and '(a8d6){16}' in `pre`'s run"),
                             ("rows", 8000, "rows is 8000 here and 24866"), ("order", "shuffled", "order is 'shuffled' here"), ("effective_batch", 16, "effective_batch is 16 here and 8"),
                             ("steps", 1555, "steps is 1555 here and 3109"), ("max_sequence_tokens", 4096, "max_sequence_tokens is 4096 here and 2048"),
                             ("recipe.learning_rate", 2.0e-4, "recipe.learning_rate is 0.0002 here and 0.0001"), ("recipe.warmup_steps", 0, "recipe.warmup_steps is 0 here and 5"),
                             ("recipe.adapter_seed", 1, "recipe.adapter_seed is 1 here and 0"), ("recipe.lora.dropout", 0.05, "recipe.lora.dropout is 0.05 here and 0.0"),
                             ("recipe.lora.target_modules", ["q_proj", "v_proj"], "recipe.lora.target_modules is .'q_proj', 'v_proj'. here")):
        with pytest.raises(ValueError, match="the check is `pre`'s pass again with ONE change, the adapter's rank and alpha, and 1 other thing is not `pre`'s: " + said):
            the_one_change(_prepared(), _prepared(64, 128, **{key: value}))
    with pytest.raises(ValueError, match="and 2 other things are not `pre`'s: seed is 1 here and 0 in `pre`'s run; recipe.learning_rate is 0.0002 here and 0.0001 in `pre`'s run"):
        the_one_change(_prepared(), _prepared(64, 128, **{"seed": 1, "recipe.learning_rate": 2.0e-4}))
    # A key one of the two does not have is a difference, not something to pass over.
    with pytest.raises(ValueError, match="1 other thing is not `pre`'s: recipe.lora.use_rslora is True here and None in `pre`'s run"):
        the_one_change(_prepared(), _prepared(64, 128, **{"recipe.lora.use_rslora": True}))
    # The rank must be ABOVE `pre`'s, and alpha / rank `pre`'s: the adapter's scale would be a second change.
    for rank, alpha, match in ((16, 32, "the check's rank is 16 and `pre`'s is 16: the check asks whether a LARGER adapter reaches further"),
                               (8, 16, "the check's rank is 8 and `pre`'s is 16"),
                               (64, 32, "the check's alpha / rank is 32 / 64 and `pre`'s is 32 / 16: the adapter's scale would be a second change"),
                               (64, 64, "the check's alpha / rank is 64 / 64 and `pre`'s is 32 / 16"), (64, 256, "the check's alpha / rank is 256 / 64")):
        with pytest.raises(ValueError, match=match):
            the_one_change(_prepared(), _prepared(rank, alpha))


# ------------------------------------------------------------------------------------------- an adapter
def _header(rank, layers=30, wide=4096, middle=11008):
    """The header of a LoRA adapter's weights file for a model of `layers` layers, `wide` wide and `middle` in its feed-forward block, on the config's target modules."""
    inside = {"q_proj": (wide, wide), "k_proj": (wide, wide), "v_proj": (wide, wide), "o_proj": (wide, wide), "gate_proj": (wide, middle), "up_proj": (wide, middle),
              "down_proj": (middle, wide)}
    header = {}
    for layer in range(layers):
        for name in LORA["target_modules"]:
            inputs, outputs = inside[name]
            where = "self_attn" if name in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
            header[f"base_model.model.model.layers.{layer}.{where}.{name}.lora_A.weight"] = {"dtype": "F32", "shape": [rank, inputs], "data_offsets": [0, 0]}
            header[f"base_model.model.model.layers.{layer}.{where}.{name}.lora_B.weight"] = {"dtype": "F32", "shape": [outputs, rank], "data_offsets": [0, 0]}
    return header


def test_an_adapters_numbers_are_its_rank_times_the_inputs_and_outputs_of_its_modules():
    small = {"x.q_proj.lora_A.weight": {"shape": [16, 6]}, "x.q_proj.lora_B.weight": {"shape": [10, 16]}, "x.down_proj.lora_A.weight": {"shape": [16, 5]},
             "x.down_proj.lora_B.weight": {"shape": [4, 16]}}
    assert adapter_numbers(small) == {"modules": 2, "tensors": 4, "ranks": [16], "numbers": 16 * 25, "numbers_a_unit_of_rank": 25}
    assert adapter_numbers(small, 64) == {**adapter_numbers(small), "rank": 64, "numbers_at_rank": 64 * 25}
    mixed = {**small, "x.down_proj.lora_B.weight": {"shape": [4, 8]}}
    assert adapter_numbers(mixed)["ranks"] == [8, 16] and adapter_numbers({})["numbers"] == 0       # an adapter that says one rank and holds another is seen
    # THE MODEL'S OWN SHAPES (30 layers, 4,096 wide, 11,008 in the middle; seven target modules): the count the box made of a rank-16 file, 37,478,400 numbers
    # in 420 tensors, and what ranks 32 and 64 hold. In training each number is held four times as a 32-bit float: the weight, its gradient, AdamW's two moments.
    of_pre = adapter_numbers(_header(16), 64)
    assert (of_pre["modules"], of_pre["tensors"], of_pre["ranks"], of_pre["numbers"], of_pre["numbers_a_unit_of_rank"], of_pre["numbers_at_rank"]) == (
        210, 420, [16], 37_478_400, 2_342_400, 149_913_600)
    assert adapter_numbers(_header(16), 32)["numbers_at_rank"] == 74_956_800 and adapter_numbers(_header(64))["numbers"] == 149_913_600
    held = {rank: rank * 2_342_400 * 4 * 4 for rank in (16, 32, 64)}
    assert {rank: round(count / 1e9, 2) for rank, count in held.items()} == {16: 0.6, 32: 1.2, 64: 2.4}
    assert (round((held[64] - held[16]) / 2 ** 30, 2), round((held[32] - held[16]) / 2 ** 30, 2)) == (1.68, 0.56)       # what the larger adapters add to `pre`'s, in GiB


# -------------------------------------------------------------------------------------------- the read
def test_the_loss_by_twentieth_is_the_mean_over_twenty_consecutive_parts_of_the_rows():
    assert by_twentieth([float(index) for index in range(40)]) == [index * 2 + 0.5 for index in range(20)]
    # 24,866 rows: twenty parts of 1,243 or 1,244 rows, none left out, in order.
    rows = [1.0] * 12433 + [3.0] * 12433
    assert by_twentieth(rows) == [1.0] * 10 + [3.0] * 10 and [24866 * index // 20 for index in (1, 2, 10, 20)] == [1243, 2486, 12433, 24866]
    uneven = by_twentieth([float(index) for index in range(30)])
    assert len(uneven) == 20 and uneven[:3] == [0.0, 1.5, 3.0] and uneven[-1] == 28.5              # parts of one row and of two, by turns: rows 30 x i // 20 up to 30 x (i + 1) // 20
    # A run of fewer rows than parts has parts with no row.
    assert by_twentieth([2.0, 4.0]) == [None] * 9 + [2.0] + [None] * 9 + [4.0] and by_twentieth([]) == [None] * 20
    assert by_twentieth([1.0, 2.0, 3.0, 4.0], parts=2) == [1.5, 3.5]


def test_the_first_two_checks_are_read_on_the_checks_model():
    falling, flat = [1.6] * 20 + [0.8] * 20, [1.0] * 40
    rungs = [_rung(problem_id, 4) for problem_id in RUNG_IDS]
    checks = training_checks(falling, rungs, SETTINGS)
    assert list(checks) == ["the_training_took", "it_still_writes_proofs"] and all(check["passes"] for check in checks.values())
    assert (checks["the_training_took"]["rows_compared"], checks["the_training_took"]["first"], checks["the_training_took"]["last"]) == (4, 1.6, 0.8)      # a tenth of 40 rows
    assert training_checks(flat, rungs, SETTINGS)["the_training_took"]["passes"] is False           # BELOW is asked: a flat loss did not take
    assert training_checks(list(reversed(falling)), rungs, SETTINGS)["the_training_took"]["passes"] is False
    # 48 attempts on the rungs: 2 without an answer is under 5%, 3 is not.
    two = [_rung(RUNG_IDS[0], 4, capped=1, no_answer=1), *rungs[1:]]
    three = [_rung(RUNG_IDS[0], 4, capped=2, no_answer=1), *rungs[1:]]
    assert training_checks(falling, two, SETTINGS)["it_still_writes_proofs"]["passes"] is True and training_checks(falling, two, SETTINGS)["it_still_writes_proofs"]["share"] == round(2 / 48, 5)
    writes = training_checks(falling, three, SETTINGS)["it_still_writes_proofs"]
    assert (writes["passes"], writes["attempts"], writes["capped_at_the_token_limit"], writes["without_a_verdict_from_lean"], writes["maximum"]) == (False, 48, 2, 1, 0.05)


PASSING = {"the_training_took": {"passes": True}, "it_still_writes_proofs": {"passes": True}, "lean_answered": {"passes": True}}


def _change(mean, low, high, problems=392):
    return {"problems": problems, "mean": mean, "low": low, "high": high}


def test_the_branch_is_the_specs():
    assert BRANCHES == ("THE ADAPTER'S SIZE CAPPED THE PRETRAINED MODEL", "NOT THE CAP AT THIS SIZE", "THE LARGER ADAPTER IS WORSE AFTER ONE PASS AT THIS LEARNING RATE",
                        "INCONCLUSIVE", "NOT READ")
    capped = rank_branch(PASSING, _change(0.012, 0.004, 0.021), "pre_r64", 16)
    assert capped["name"] == CAPPED and capped["failed_checks"] == [] and capped["reason"] == (
        "the interval is clear of zero and above (+0.01200 [+0.00400, +0.02100]): the adapter's size capped the pretrained model. The arm is then run again from `pre_r64` "
        "(its own map, the same halves, every model of the arm at that rank) before more seeds are bought at rank 16, and every result of this project that reads \"not shown\" "
        "at rank 16 is marked as read under that cap")
    through = rank_branch(PASSING, _change(0.002, -0.006, 0.01), "pre_r64", 16)
    assert through["name"] == NOT_THE_CAP and through["half_width"] == 0.008 and through["reason"] == (
        "the interval holds zero (+0.00200 [-0.00600, +0.01000]): not the cap at this size; rank 16 stays. The size this run could have seen is the interval's half-width, "
        "0.00800 per attempt")
    worse = rank_branch(PASSING, _change(-0.01, -0.02, -0.001), "pre_r64", 16)
    assert worse["name"] == WORSE and worse["reason"] == ("the interval lies below zero (-0.01000 [-0.02000, -0.00100]): the larger adapter is worse after one pass at this "
                                                          "learning rate; rank 16 stays and nothing is concluded about a larger adapter trained longer")
    # An interval that ENDS at zero holds it, at either end; one of no width at zero holds it.
    assert rank_branch(PASSING, _change(0.004, 0.0, 0.009), "pre_r64", 16)["name"] == NOT_THE_CAP and rank_branch(PASSING, _change(-0.004, -0.009, 0.0), "pre_r64", 16)["name"] == NOT_THE_CAP
    exact = rank_branch(PASSING, _change(0.0, 0.0, 0.0), "pre_r64", 16)
    assert exact["name"] == NOT_THE_CAP and exact["half_width"] == 0.0
    # The fallback's words name its own model.
    assert "run again from `pre_r32`" in rank_branch(PASSING, _change(0.012, 0.004, 0.021), "pre_r32", 16)["reason"]
    # A failed check: INCONCLUSIVE whatever the primary says, with what failed; nothing to be read on: NOT READ.
    for failed in ("the_training_took", "it_still_writes_proofs", "lean_answered"):
        branch = rank_branch({**PASSING, failed: {"passes": False}}, _change(0.012, 0.004, 0.021), "pre_r64", 16)
        assert branch["name"] == INCONCLUSIVE and branch["failed_checks"] == [failed] and branch["reason"] == (
            f"this run could not have seen a win ({failed}): nothing is said about the size of the adapter")
    both = rank_branch({**PASSING, "the_training_took": {"passes": False}, "lean_answered": {"passes": False}}, _change(0.012, 0.004, 0.021), "pre_r64", 16)
    assert both["failed_checks"] == ["the_training_took", "lean_answered"]
    assert rank_branch(PASSING, {"problems": 0, "mean": None, "low": None, "high": None}, "pre_r64", 16)["name"] == NOT_READ


def test_the_goal_problems_nothing_stored_has_ever_solved_are_given_and_counted_on_the_checks_model():
    on_g = [_rung("g1a", 9, episodes=93), _rung("g8a", 2, episodes=93), _rung("g8b", 0, episodes=93), _rung("g8c", 1, episodes=93)]
    of_pre = [_rung("g1a", 9, episodes=93), _rung("g8a", 0, episodes=93), _rung("g8b", 0, episodes=93), _rung("g8c", 0, episodes=93)]
    goal = ["g1a", "g8a", "g8b", "g8c"]
    not_given = never_solved(None, on_g, of_pre, goal)
    assert not_given == {"given": False, "what": not_given["what"]} and "no file of their ids was given in the run directory" in not_given["what"]
    given = never_solved(["g8a", "g8b", "g8c"], on_g, of_pre, goal)
    assert (given["given"], given["ids"], given["goal_problems"], given["not_goal_problems_of_this_run"], given["solved"], given["problem_ids"], given["solved_by_pre"]) == (
        True, 3, 3, 0, 2, ["g8a", "g8c"], 0)
    # An id that is not a goal problem of this run is counted apart and reads nothing; an id given twice is one id; a list `pre` solves some of is seen to be wrong.
    stray = never_solved(["g8a", "elsewhere", "g8a", "g1a"], on_g, of_pre, goal)
    assert (stray["ids"], stray["goal_problems"], stray["not_goal_problems_of_this_run"], stray["solved"], stray["solved_by_pre"]) == (3, 2, 1, 2, 1)
    assert never_solved([], on_g, of_pre, goal)["solved"] == 0 and never_solved([], on_g, of_pre, goal)["given"] is True


# ------------------------------------------------------------------------------------------ the report
SAMPLINGS = [{"name": "reach", "episodes": 32, "sampling_seed": 1001}, {"name": "more", "episodes": 61, "sampling_seed": 1020}]
RANK_CHECK = {"what": "L4r", "check": "rank", "stage": "ladder_l4_rank", "model": "pre_r64", "rank": 64, "alpha": 128, "in_the_place_of_rank": None, "max_lora_rank": 64,
              "max_lora_rank_of_every_other_stage": 16, "trained_parameters": 149_913_600, "trained_parameters_of_pre": 37_478_400, "modules": 210,
              "counted_from": "the shapes of `pre`'s stored adapter", "the_one_change": the_one_change(_prepared(), _prepared(64, 128)),
              "pre": {"run": "runs/ladder_l4_pretrain_seed0", "seed": 0, "rows": 24866, "parts": ["rungs", "reach", "more"], "report_checks_pass": True, "rank": 16, "alpha": 32},
              "never_solved_file": "l4_rank_never_solved.jsonl"}
PREPARE = {**_prepared(64, 128), "stage": "l4", "label": LABEL, "stand_in_engine": False, "attempts_a_goal_problem": 93, "rung_episodes": 8, "goal_samplings": SAMPLINGS,
           "sampling_seeds": {"rungs": 1002, "reach": 1001, "more": 1020}, "contradicted_side_setting": "audit", "half": "pretrain", "half_seed": 0,
           "rows_by_kind": {"lean_workbook": 8155, "stp_conjecture": 16711}, "rows_by_proof_lines": {"1": 2107, "2-3": 4105, "4-7": 6234, "8+": 12420},
           "longest_example_tokens": 1536, "rank_check": RANK_CHECK}
TRAIN = {"stage": "l4", "label": LABEL, "model": "pre_r64", "rows": 24866, "steps": 3109, "stand_in_engine": False, "peak_allocated_gb": 12.3, "peak_reserved_gb": 12.6,
         "adapter_saved": {"rank": 64, "alpha": 128, "modules": 210, "tensors": 420, "ranks": [64], "numbers": 149_913_600, "numbers_a_unit_of_rank": 2_342_400}}
LOSSES = {"pre": [1.5] * 20 + [0.9] * 20, "pre_r64": [1.6] * 20 + [0.8] * 20}
PRE = _made("pre")                                  # 84 successes in its 12 x 93 attempts on G; the rungs 2, 3, 4, 5, 7, 8 of 8
FURTHER = ON_G["with"]                              # a model that reaches further than `pre` on the long problems: 95 successes
HEADS = lambda report: [line.split(": ", 1)[1].split(",")[0].split(".")[0].split(":")[0] for line in report["lines"]]      # noqa: E731


def _model(on_g=FURTHER, rungs=None, **changed):
    return {**_made("with", on_g=on_g, rungs=rungs), "stand_in_engine": False, **changed}


def _report(model=None, pre=PRE, prepare=PREPARE, train=TRAIN, losses=LOSSES, never=None):
    return build_rank_report(prepare, train, losses, GROUPS, LENGTHS, pre, _model() if model is None else model, SETTINGS, EVALUATION, never)


def test_the_report_reads_the_three_checks_first_then_the_primary_against_pre_the_branch_and_the_secondary_reads():
    report = _report()
    lines = report["lines"]
    assert report["label"] == LABEL and report["stage"] == "l4" and report["ok"] is True and report["inconclusive"] is False and all(line.startswith(f"{SAY}: ") for line in lines)
    assert HEADS(report) == ["L4r", "CHECK 1", "CHECK 2", "CHECK 3", "PRIMARY", "BRANCH", *["SECONDARY"] * 7]
    assert lines[0] == (
        f"{SAY}: L4r, IS THE PRETRAINED MODEL CAPPED BY THE SIZE OF ITS ADAPTER? A labelled check of the pretraining, seed 0. `pre_r64`: `pre`'s pass again (from the base, ONE "
        "pass over the same 24,866 published proofs in the file's order, 3,109 optimizer steps; the same seed, learning rate, batch, sequence limit and target modules) with "
        "ONE change: the adapter's rank is 64 and its alpha 128, where `pre` has 16 and 32 (149,913,600 trained numbers). It is measured as `pre` was, with `pre`'s sampling "
        "seeds, and read against `pre`'s stored rows (runs/ladder_l4_pretrain_seed0), which are only read. No map is made")
    # ---- the three checks, each with its number and PASS or FAIL
    assert lines[1] == f"{SAY}: CHECK 1, the training took: the mean loss over the last tenth of its rows (4) is 0.8, over the first tenth it was 1.6: PASS"
    assert lines[2] == (f"{SAY}: CHECK 2, it still writes proofs: 0.0 of `pre_r64`'s 48 attempts on the three rungs got no answer (0 reached the token cap, 0 had no verdict from "
                        "Lean); under 0.05 is asked: PASS")
    assert lines[3] == (f"{SAY}: CHECK 3, Lean answered: the share of attempts without an answer in each set read (at most 0.02 is asked): rungs_pre 0.0 of 48; goal_pre_1 0.0 of "
                        "384; goal_pre_2 0.0 of 732; rungs_pre_r64 0.0 of 48; goal_pre_r64_1 0.0 of 384; goal_pre_r64_2 0.0 of 732: PASS")
    checks = report["can_this_run_see_a_win"]
    assert [name for name in checks if name != "what"] == ["the_training_took", "it_still_writes_proofs", "lean_answered"] and all(checks[name]["passes"] for name in checks if name != "what")
    assert checks["lean_answered"]["maximum"] == 0.02 and checks["lean_answered"]["sets_not_answered"] == [] and len(checks["lean_answered"]["share_without_an_answer"]) == 6
    # ---- THE PRIMARY: all of G, successes per attempt over the 93 attempts, `pre_r64` minus `pre`, paired by problem: 95 successes against 84 in 12 x 93 attempts
    primary = report["primary"]
    assert (primary["problems"], primary["successes"], primary["successes_of_the_base"], primary["attempts_each"]) == (12, 95, 84, 1116)
    assert primary["mean"] == pytest.approx(11 / 1116, abs=1e-5) and (primary["per_1000"], primary["per_1000_of_the_base"]) == (85.13, 75.27) and primary["low"] > 0
    assert "all of G: successes per attempt over the 93 attempts a problem, `pre_r64` minus `pre`, paired by problem, a 95% bootstrap interval over problems" in primary["what"]
    assert lines[4].startswith(f"{SAY}: PRIMARY. All of G (12 problems), successes per attempt over the 93 attempts a problem, `pre_r64` minus `pre`, paired by problem, 95% "
                               "bootstrap over problems: +0.00986 [") and lines[4].endswith("85.13 against 75.27 per 1,000 (95 successes against 84 in 1,116 attempts each)")
    # It is against `pre`, on every attempt of both samplings: a success of the check's model in either moves it, and so does one of `pre`.
    one_more = _model(on_g=(FURTHER[0], [9, 9, 9, 6, 6, 6, 4, 3, 3, 2, 1, 1]))
    assert _report(one_more)["primary"]["successes"] == 96 and _report(_model(on_g=([7, *FURTHER[0][1:]], FURTHER[1])))["primary"]["successes"] == 96
    assert _report(pre=_made("pre", on_g=(ON_G["pre"][0], [9, 9, 9, 6, 6, 6, 3, 2, 1, 1, 0, 0])))["primary"]["successes_of_the_base"] == 85
    # ---- the branch, in the spec's words
    assert report["branch"]["name"] == CAPPED and lines[5].startswith(f"{SAY}: BRANCH: THE ADAPTER'S SIZE CAPPED THE PRETRAINED MODEL. the interval is clear of zero and above")
    assert "The arm is then run again from `pre_r64` (its own map, the same halves, every model of the arm at that rank) before more seeds are bought at rank 16" in lines[5]
    # ---- the secondary reads, in the spec's order
    secondary = report["secondary"]
    assert list(secondary) == ["by_length_group", "goal_problems_solved", "goal_problems_solved_by_attempts_alone", "the_three_rungs", "loss_by_twentieth", "timed_out_in_lean",
                               "distinct_attempts", "never_solved_before"]
    by_length = secondary["by_length_group"]
    assert by_length["problems"] == {"1": 3, "2-3": 3, "4-7": 3, "8+": 3, "4_or_more": 6}
    assert {group: (by_length[group]["successes"], by_length[group]["successes_of_the_base"], by_length[group]["attempts_each"]) for group in by_length["problems"]} == {
        "1": (45, 45, 279), "2-3": (30, 30, 279), "4-7": (16, 9, 279), "8+": (4, 0, 279), "4_or_more": (20, 9, 558)}
    assert by_length["4_or_more"]["mean"] == pytest.approx(11 / 558, abs=1e-5) and by_length["1"]["mean"] == 0
    assert lines[6].startswith(f"{SAY}: SECONDARY, the same by the length of the shortest published proof, `pre_r64` minus `pre`: 1 line (3): +0.00000 [+0.00000, +0.00000], 161.29 "
                               "against 161.29 per 1,000; 2-3 lines (3): ") and "4 lines or more (6): " in lines[6] and "all of G" not in lines[6]
    solved = secondary["goal_problems_solved"]["all"]
    assert (solved["resolved_after"], solved["resolved_before"], solved["gained"], solved["lost"], solved["sign_test_p"], solved["gained_problems"]) == (11, 9, 2, 0, 0.5, ["g8a", "g8b"])
    assert lines[7] == (f"{SAY}: SECONDARY, goal problems solved at 93 attempts, `pre_r64` against `pre`: all of G: 11 to 9, gained 2, lost 0 (p = 0.5); 4 lines or more: 5 to 3, "
                        "gained 2, lost 0 (p = 0.5)")
    alone = secondary["goal_problems_solved_by_attempts_alone"]
    assert (alone["pre"]["all"]["solved_at_least_once"], alone["pre"]["all"]["reliably"], alone["pre_r64"]["all"]["solved_at_least_once"], alone["pre_r64"]["all"]["reliably"]) == (9, 6, 11, 6)
    assert lines[8] == (f"{SAY}: SECONDARY, goal problems solved in at least one episode of 8 one-shot attempts / reliably (in at least half of their episodes): all of G (12): "
                        "`pre` 9 / 6, `pre_r64` 11 / 6; 4 lines or more (6): `pre` 3 / 0, `pre_r64` 5 / 0")
    rungs = secondary["the_three_rungs"]
    assert rungs["problems"] == {"below": 2, "in": 2, "above": 2} and rungs["in"]["mean"] == pytest.approx(1 / 8 / 2, abs=1e-5) and rungs["below"]["mean"] == 0 == rungs["above"]["mean"]
    assert lines[9].startswith(f"{SAY}: SECONDARY, the three rungs (8 episodes a problem), `pre_r64` minus `pre`: below +0.00000 [+0.00000, +0.00000]; in +0.06250 [")
    assert secondary["loss_by_twentieth"]["pre"] == [1.5] * 10 + [0.9] * 10 and secondary["loss_by_twentieth"]["pre_r64"] == [1.6] * 10 + [0.8] * 10
    assert lines[10] == (f"{SAY}: SECONDARY, the loss on rows not yet trained on, by twentieth of the pass: `pre` " + ", ".join(["1.500"] * 10 + ["0.900"] * 10) + "; `pre_r64` "
                         + ", ".join(["1.600"] * 10 + ["0.800"] * 10))
    assert lines[11] == (f"{SAY}: SECONDARY, the share of attempts that time out in Lean (on the rungs, on G): `pre` 0.0, 0.0; `pre_r64` 0.0, 0.0. The share of distinct attempts "
                         "(on the rungs, on G): `pre` 0.9, 0.95; `pre_r64` 0.9, 0.95")
    assert secondary["never_solved_before"]["given"] is False and lines[12] == (
        f"{SAY}: SECONDARY, the goal problems nothing stored has ever solved: not given (no file of their ids is in the run directory; the stage does not compute it)")
    # ---- the headline, and what stands beside the read
    assert report["headline"].startswith("L4r (is the pretrained model capped by the size of its adapter; pretrained on published proofs) seed 0, `pre_r64` at rank 64: THE ADAPTER'S "
                                         "SIZE CAPPED THE PRETRAINED MODEL. Primary (all of G, 12 problems, 93 attempts each, `pre_r64` minus `pre`): +0.00986 [")
    assert "Checks: the training took PASS (0.8 against 1.6), still writes proofs PASS (0.0 without an answer), Lean answered PASS." in report["headline"]
    assert report["what_it_cannot_say"] == cannot_say("pre_r64", 64, 16) == [
        "Rank 64 is not the whole model: a null does not show that full training would not help",
        "One learning rate and one pass: an adapter 4 times the size may want either changed, and this run changes neither",
        "One seed, a first run by the seed rule; a positive read is confirmed by what is built on it (the arm from `pre_r64`), not by a second pretraining"]
    of_the_check = report["the_check"]
    assert (of_the_check["check"], of_the_check["model"], of_the_check["rank"], of_the_check["alpha"], of_the_check["max_lora_rank"]) == ("rank", "pre_r64", 64, 128, 64)
    assert of_the_check["adapter_saved"] == TRAIN["adapter_saved"] and (of_the_check["peak_allocated_gb"], of_the_check["peak_reserved_gb"]) == (12.3, 12.6)
    assert of_the_check["the_one_change"]["rank"] == {"pre": 16, "check": 64} and of_the_check["pre"]["run"] == "runs/ladder_l4_pretrain_seed0"
    assert "rank 16" in report["models"]["pre"] and "nothing here measured it again" in report["models"]["pre"] and "kept until the read" in report["models"]["pre_r64"]
    assert report["heldout"] == {"goal_set": 12, "goal_set_by_length_group": {"1": 3, "2-3": 3, "4-7": 3, "8+": 3, "4_or_more": 6, "all": 12}, "rungs": {"below": 2, "in": 2, "above": 2}}
    assert report["sizes"]["sampling_seeds"] == {"rungs": 1002, "reach": 1001, "more": 1020} and report["not_to_be_read"] == [] and report["pretraining_file"]["rows"] == 24866


def test_the_same_rows_on_both_sides_read_exactly_zero_with_an_interval_of_no_width_and_not_the_cap():
    same = _report(model={**PRE, "stand_in_engine": False}, losses={"pre": LOSSES["pre"], "pre_r64": LOSSES["pre"]})
    primary = same["primary"]
    assert (primary["mean"], primary["low"], primary["high"], primary["successes"], primary["successes_of_the_base"]) == (0.0, 0.0, 0.0, 84, 84)
    assert same["branch"]["name"] == NOT_THE_CAP and same["branch"]["half_width"] == 0.0 and same["inconclusive"] is False
    solved = same["secondary"]["goal_problems_solved"]["all"]
    assert (solved["gained"], solved["lost"], solved["sign_test_p"]) == (0, 0, None) and all(same["secondary"]["the_three_rungs"][rung]["mean"] == 0 for rung in ("below", "in", "above"))
    assert all(entry["mean"] == 0 == entry["low"] == entry["high"] for group, entry in same["secondary"]["by_length_group"].items() if group not in ("what", "problems"))
    assert same["secondary"]["loss_by_twentieth"]["pre"] == same["secondary"]["loss_by_twentieth"]["pre_r64"]


def test_the_branches_follow_the_primarys_interval():
    # One long problem up and one down: the interval holds zero, and the note gives its half-width as the size this run could have seen.
    mixed = _report(_model(on_g=(ON_G["pre"][0], [9, 9, 9, 6, 6, 6, 3, 2, 0, 1, 0, 0])))
    primary = mixed["primary"]
    assert primary["low"] < 0 < primary["high"] and mixed["branch"]["name"] == NOT_THE_CAP and mixed["branch"]["half_width"] == round((primary["high"] - primary["low"]) / 2, 6)
    assert f"the interval's half-width, {mixed['branch']['half_width']:.5f} per attempt" in mixed["lines"][5] and "rank 16 stays" in mixed["lines"][5]
    # The check's model below `pre` on the problems `pre` solves: the larger adapter is worse after one pass.
    below = _report(_model(on_g=([5, 5, 5, 3, 3, 3, 1, 0, 0, 0, 0, 0], [8, 8, 8, 5, 5, 5, 2, 1, 0, 0, 0, 0])))
    assert below["primary"]["high"] < 0 and below["branch"]["name"] == WORSE and below["lines"][5].startswith(
        f"{SAY}: BRANCH: THE LARGER ADAPTER IS WORSE AFTER ONE PASS AT THIS LEARNING RATE. the interval lies below zero")
    assert "nothing is concluded about a larger adapter trained longer" in below["headline"]


def test_a_failing_check_gives_inconclusive_and_nothing_else_is_said():
    flat = {"pre": LOSSES["pre"], "pre_r64": [1.0] * 40}
    capped = _model(rungs=None)
    capped["rungs"] = [_rung(problem_id, 0, capped=8) for problem_id in RUNG_IDS]
    unanswered = _model()
    unanswered["goal"][1][0]["attempts_without_an_answer"] = 20                                    # of the 61 attempts on one problem: over 2% of that set's 732
    for change, name, line, ok in (({"losses": flat}, "the_training_took", 1, True), ({"model": capped}, "it_still_writes_proofs", 2, True),
                                   ({"model": unanswered}, "lean_answered", 3, False)):
        broken = _report(**change)
        assert broken["inconclusive"] is True and broken["branch"]["name"] == INCONCLUSIVE and broken["branch"]["failed_checks"] == [name] and broken["ok"] is ok
        assert broken["primary"] is None and broken["secondary"] is None and broken["lines"][line].endswith("FAIL")
        assert HEADS(broken)[:5] == ["L4r", "CHECK 1", "CHECK 2", "CHECK 3", "INCONCLUSIVE"] and len(broken["lines"]) == (5 if ok else 6)
        assert broken["lines"][4] == f"{SAY}: INCONCLUSIVE. this run could not have seen a win ({name}): nothing is said about the size of the adapter"
        assert "Primary" not in broken["headline"] and "INCONCLUSIVE" in broken["headline"]
        assert list(broken["measured_and_not_read"]) == ["why", "primary", "secondary"] and broken["measured_and_not_read"]["primary"]["successes"] == 95      # kept for whoever repairs the run
    # A set Lean did not answer is NOT TO BE READ: the step fails (`ok`), and the task queued again samples it again.
    report = _report(model=unanswered)
    assert report["not_to_be_read"] == ["goal_pre_r64_2"] and report["can_this_run_see_a_win"]["lean_answered"]["sets_not_answered"] == ["goal_pre_r64_2"]
    assert report["lines"][3].endswith("goal_pre_r64_2 0.02732 of 732: FAIL") and "NOT TO BE READ: too many attempts without a verdict from Lean in goal_pre_r64_2" in report["lines"][-1]
    assert "the task queued again measures such a set of this run's own again from the kept adapter" in report["lines"][-1] and "NOT TO BE READ" in report["headline"]
    # Exactly 2% of a set without an answer is "at most 2%": it passes (8 of a set's 400 attempts: one problem of it has 48 attempts here, on both sides).
    at_the_line, of_pre = _model(), _made("pre")
    for side in (at_the_line, of_pre):
        side["goal"][0][0]["episodes"] = 48
    at_the_line["goal"][0][0]["attempts_without_an_answer"] = 8
    exact = _report(model=at_the_line, pre=of_pre)
    assert exact["attempts"]["goal_pre_r64_1"]["share_without_an_answer"] == 0.02 and exact["can_this_run_see_a_win"]["lean_answered"]["passes"] is True and exact["ok"] is True
    at_the_line["goal"][0][0]["attempts_without_an_answer"] = 9
    assert _report(model=at_the_line, pre=of_pre)["can_this_run_see_a_win"]["lean_answered"]["passes"] is False


def test_the_fallbacks_report_says_which_ran_and_the_ids_given_are_read():
    fallback = {**PREPARE, "recipe": _prepared(32, 64)["recipe"],
                "rank_check": {**RANK_CHECK, "check": "fallback", "stage": "ladder_l4_rank_fallback", "model": "pre_r32", "rank": 32, "alpha": 64, "in_the_place_of_rank": 64,
                               "max_lora_rank": 32, "trained_parameters": 74_956_800, "the_one_change": the_one_change(_prepared(), _prepared(32, 64))}}
    train = {**TRAIN, "model": "pre_r32", "adapter_saved": {**TRAIN["adapter_saved"], "rank": 32, "alpha": 64, "ranks": [32], "numbers": 74_956_800}}
    report = _report(prepare=fallback, train=train, losses={"pre": LOSSES["pre"], "pre_r32": LOSSES["pre_r64"]}, never=["g8a", "g8c", "elsewhere"])
    assert all(line.startswith(f"{SAY}: ") for line in report["lines"])
    assert "`pre_r32`: `pre`'s pass again" in report["lines"][0] and "the adapter's rank is 32 and its alpha 64, where `pre` has 16 and 32 (74,956,800 trained numbers). THIS IS THE "\
           "FALLBACK: rank 32 is run in the place of rank 64, which the box could not hold (that run is VOID). It is measured as `pre` was" in report["lines"][0]
    assert "`pre_r32` at rank 32" in report["headline"] and report["the_check"]["in_the_place_of_rank"] == 64 and "run again from `pre_r32`" in report["branch"]["reason"]
    assert report["what_it_cannot_say"][0].startswith("Rank 32 is not the whole model") and "an adapter 2 times the size" in report["what_it_cannot_say"][1]
    assert set(report["secondary"]["loss_by_twentieth"]) == {"what", "pre", "pre_r32"} and "pre_r32" in report["models"]
    # THE IDS GIVEN: two of the three are goal problems of this run; the check's model solves one of them (g8a, 3 of its 93 attempts), `pre` none.
    unsolved = report["secondary"]["never_solved_before"]
    assert (unsolved["given"], unsolved["ids"], unsolved["goal_problems"], unsolved["not_goal_problems_of_this_run"], unsolved["solved"], unsolved["problem_ids"],
            unsolved["solved_by_pre"]) == (True, 3, 2, 1, 1, ["g8a"], 0)
    assert report["lines"][-1] == (f"{SAY}: SECONDARY, the goal problems nothing stored has ever solved: of the 2 listed, `pre_r32` solves 1 (g8a); `pre` solves 0 of them; 1 more "
                                   "ids of the file are not goal problems of this run and are not counted")
    none = _report(never=["g8c"])
    assert none["lines"][-1] == f"{SAY}: SECONDARY, the goal problems nothing stored has ever solved: of the 1 listed, `pre_r64` solves 0; `pre` solves 0 of them"
    # A line names at most twenty solved problems and says how many more there are; the report's file holds every one.
    many = [f"p{index}" for index in range(23)]
    assert (_named_problems([]), _named_problems(["a", "b"]), NAMED_IN_A_LINE) == ("", " (a, b)", 20) and _named_problems(many[:20]) == f" ({', '.join(many[:20])})"
    assert _named_problems(many) == f" ({', '.join(many[:20])}, and 3 more: all are in the report's file)"
    # With no count of the adapter's numbers (a stand-in's run) the heading says none.
    uncounted = _report(prepare={**PREPARE, "rank_check": {**RANK_CHECK, "trained_parameters": None}}, train={**TRAIN, "adapter_saved": None})
    assert "where `pre` has 16 and 32. It is measured as `pre` was" in uncounted["lines"][0] and uncounted["the_check"]["adapter_saved"] is None


def test_the_share_of_attempts_that_time_out_in_lean_is_read_on_the_rungs_and_on_g():
    model = _model()
    model["rungs"][0]["attempts_timed_out"], model["goal"][0][0]["attempts_timed_out"], model["goal"][1][1]["attempts_timed_out"] = 2, 5, 10
    assert timed_out(model) == {"rungs": {"attempts": 48, "timed_out_in_lean": 2, "share": round(2 / 48, 5)}, "goal": {"attempts": 1116, "timed_out_in_lean": 15, "share": round(15 / 1116, 5)}}
    report = _report(model=model)
    assert report["secondary"]["timed_out_in_lean"]["pre_r64"] == timed_out(model) and report["secondary"]["timed_out_in_lean"]["pre"]["goal"]["share"] == 0.0
    assert "`pre` 0.0, 0.0; `pre_r64` 0.04167, 0.01344. The share of distinct attempts" in report["lines"][11]
    # A problem attempted on both sides counts its attempts on both.
    model["rungs"][1]["sides"] = 2
    assert timed_out(model)["rungs"]["attempts"] == 56


def test_a_smoke_run_has_one_sampling_and_a_world_with_no_goal_problem_is_not_read():
    one = {**PREPARE, "attempts_a_goal_problem": 32, "goal_samplings": SAMPLINGS[:1]}
    first = lambda model: {**model, "goal": [model["goal"][0]]}      # noqa: E731
    smoke = build_rank_report(one, TRAIN, LOSSES, GROUPS, LENGTHS, first(PRE), first(_model()), SETTINGS, EVALUATION)
    assert smoke["primary"]["attempts_each"] == 12 * 32 and (smoke["primary"]["successes"], smoke["primary"]["successes_of_the_base"]) == (37, 33)
    assert "successes per attempt over the 32 attempts a problem" in smoke["lines"][4] and list(smoke["attempts"]) == ["rungs_pre", "goal_pre_1", "rungs_pre_r64", "goal_pre_r64_1"]
    no_goal = [row for row in GROUPS if row["group"] != "goal"]
    empty = lambda model: {**model, "goal": [[], []], "episodes_of_8": {}}      # noqa: E731
    unread = build_rank_report(PREPARE, TRAIN, LOSSES, no_goal, LENGTHS, empty(PRE), empty(_model()), SETTINGS, EVALUATION)
    assert unread["branch"]["name"] == NOT_READ and unread["primary"]["mean"] is None and unread["inconclusive"] is False and "PRIMARY. All of G (0 problems)" in unread["lines"][4]
