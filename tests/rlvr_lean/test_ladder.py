"""Spec §13b: the ladder's first rung. Its sets, its branches (fixed before the run), the census arithmetic, the
names its fresh samples are stored under, the task's parts, and its report over a tiny store."""

import pytest

from rlvr_lean.domain.evaluation.bootstrap import BootstrapInterval
from rlvr_lean.domain.evaluation.ladder import (
    ESCALATE,
    SECONDARY_ABOVE,
    SECONDARY_BELOW,
    STOP,
    UNDETECTABLE,
    VOID,
    below_band_ids,
    census_summary,
    ladder_branch,
    never_proved_ids,
    trained_on,
)
from rlvr_lean.domain.evaluation.loss_parts import pair_loss_row
from rlvr_lean.domain.training.arms import FRESH_SETS, LADDER_ARM, FreshNames, arm_names
from rlvr_lean.domain.training.target_format import NATIVE
from rlvr_lean.runner.entry import STAGES, part_is_blocked, step_fields
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.reporting.ladder import LADDER_REPORT_FILE, build_ladder_report, fresh_counts, ladder_scalars

REUSED = "learning_progress_cosine"
RANGE = (0.083, 0.188)
ZERO = BootstrapInterval(0.0, -0.01, 0.01)


def interval(low, high):
    return BootstrapInterval((low + high) / 2, low, high)


# --- the sets, from the base's stored counts -------------------------------------------------------------------

def test_below_the_band_is_one_or_two_stored_proofs_and_never_proved_is_none():
    counts = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 12, "f": 0}
    assert below_band_ids(counts, 1, 2) == ["b", "c"]
    assert never_proved_ids(counts) == ["a", "f"]
    with pytest.raises(ValueError):
        below_band_ids(counts, 0, 2)                                  # never-proved conjectures are not "below the band"


def test_the_primary_s_conjectures_must_not_be_in_the_training_set():
    assert trained_on(["b", "c"], ["d", "e"]) == []
    assert trained_on(["b", "c"], ["c", "d"]) == ["c"]


# --- the branches ----------------------------------------------------------------------------------------------

def test_a_primary_entirely_above_zero_escalates():
    branch = ladder_branch(interval(0.01, 0.09), 0.13, RANGE, {"never": ZERO})
    assert branch.name == ESCALATE and "second rung" in branch.action


def test_a_primary_entirely_below_zero_stops_for_diagnosis():
    branch = ladder_branch(interval(-0.09, -0.01), 0.13, RANGE, {"never": interval(0.01, 0.02)})
    assert branch.name == STOP and branch.action == "stop and diagnose"


def test_a_primary_containing_zero_with_secondaries_inside_the_base_s_luck_is_undetectable():
    branch = ladder_branch(interval(-0.02, 0.05), 0.13, RANGE, {"never": interval(-0.005, 0.004), "workbook": interval(0.0, 0.01)})
    assert branch.name == UNDETECTABLE and "4,609 never sampled" in branch.action and "not more seeds" in branch.action


def test_a_primary_containing_zero_with_a_secondary_above_zero_escalates_as_exploratory_and_names_it():
    branch = ladder_branch(interval(-0.02, 0.05), 0.13, RANGE, {"never": interval(0.002, 0.02), "workbook": ZERO})
    assert branch.name == SECONDARY_ABOVE and "never" in branch.reason and "workbook" not in branch.reason
    assert "ESCALATE to seeds 1 and 2 with never" in branch.action and "exploratory" in branch.action


def test_a_primary_containing_zero_with_a_secondary_below_zero_stops_even_if_another_is_above():
    branch = ladder_branch(interval(-0.02, 0.05), 0.13, RANGE, {"never": interval(0.002, 0.02), "workbook": interval(-0.02, -0.001)})
    assert branch.name == SECONDARY_BELOW and "workbook" in branch.reason and branch.action == "stop and diagnose"


@pytest.mark.parametrize("rate", [0.0829, 0.1881, 0.0, 0.5])
def test_a_base_fresh_rate_outside_the_pre_registered_range_is_void_whatever_the_primary(rate):
    assert ladder_branch(interval(0.05, 0.09), rate, RANGE, {}).name == VOID


@pytest.mark.parametrize("rate", [0.083, 0.132, 0.188])
def test_the_range_s_own_bounds_are_inside_it(rate):
    assert ladder_branch(interval(0.05, 0.09), rate, RANGE, {}).name == ESCALATE


def test_an_interval_that_touches_zero_contains_it():
    assert ladder_branch(interval(0.0, 0.09), 0.13, RANGE, {}).name == UNDETECTABLE
    assert ladder_branch(interval(-0.09, 0.0), 0.13, RANGE, {}).name == UNDETECTABLE
    assert ladder_branch(interval(-0.01, 0.01), 0.13, RANGE, {"never": interval(0.0, 0.02)}).name == UNDETECTABLE


# --- the census --------------------------------------------------------------------------------------------------

def test_the_census_counts_disproved_conjectures_and_keeps_inexact_negations_apart():
    statements = [{"conjecture_id": "a", "built": True, "compiles": True, "exact": True},
                  {"conjecture_id": "b", "built": True, "compiles": True, "exact": False},
                  {"conjecture_id": "c", "built": True, "compiles": False, "exact": False},
                  {"conjecture_id": "d", "built": False, "compiles": False, "exact": False},
                  {"conjecture_id": "e", "built": True, "compiles": True, "exact": True}]
    summary = census_summary(statements, {"a": 3, "b": 1, "c": 5})          # "c" was never sampled: its count is ignored
    assert (summary["never_proved"], summary["sampled"], summary["negation_not_built"], summary["negation_does_not_compile"]) == (5, 3, 1, 1)
    assert (summary["disproved"], summary["disproved_among_exact"], summary["negation_inexact"]) == (2, 1, 1)
    assert summary["disproved_share_of_never_proved"] == pytest.approx(0.4) and summary["undetermined"] == 3
    with pytest.raises(ValueError):
        census_summary([], {})


# --- names -------------------------------------------------------------------------------------------------------

def test_the_ladder_arm_is_native_trains_on_the_half_pass_rate_selection_and_has_its_own_names():
    names = arm_names(LADDER_ARM, 0, REUSED)
    assert LADDER_ARM == "ladder_half_native" and names.target_format == NATIVE and names.examples_arm == "half_pass_rate"
    assert names.key == names.adapter_directory == names.tensorboard_run == "ladder_half_native_seed0"
    assert names.train_marker == "train_adapter_ladder_half_native_seed0" and names.variant == "eval_ladder_half_native_s0"
    assert names.key != arm_names("half_pass_rate", 0, REUSED).key


def test_fresh_samples_are_stored_per_set_and_model_under_names_no_other_artifact_uses():
    base, adapter = FreshNames("base", "below_band"), FreshNames("ladder_half_native_seed0", "below_band")
    assert (base.marker, base.attempts_file, base.verification_file, base.variant) == (
        "fresh_samples_below_band_base", "fresh_attempts_below_band_base.jsonl", "fresh_verification_below_band_base.jsonl", "fresh_base")
    assert adapter.variant == "fresh_ladder_half_native_seed0" and adapter.attempts_file != base.attempts_file
    every = {name for who in ("base", "ladder_half_native_seed0") for set_name in FRESH_SETS
             for name in (FreshNames(who, set_name).marker, FreshNames(who, set_name).attempts_file, FreshNames(who, set_name).verification_file)}
    assert len(every) == 2 * 3 * 3 and all(name.startswith("fresh_") for name in every)
    assert FRESH_SETS == ("below_band", "workbook_unsolved", "never_proved")
    with pytest.raises(ValueError):
        FreshNames("base", "band")


# --- the task's parts ----------------------------------------------------------------------------------------------

def test_a_plain_stage_entry_has_no_options_and_an_entry_s_options_are_a_copy():
    assert step_fields(("gpu", "train_adapter")) == ("gpu", "train_adapter", {})
    options = {"arms": "x", "part": 1}
    assert step_fields(("gpu", "report", options)) == ("gpu", "report", options) and step_fields(("gpu", "report", options))[2] is not options


def test_a_failed_part_blocks_its_own_later_steps_and_the_parts_that_need_it_only():
    assert part_is_blocked({"part": 1}, {1}) and not part_is_blocked({"part": 2}, {1})
    assert part_is_blocked({"part": 3, "needs": (1, 2)}, {2}) and not part_is_blocked({"part": 3, "needs": (1, 2)}, set())
    assert not part_is_blocked({}, {1, 2})


def test_the_task_runs_13b_then_the_scout_s_seeds_then_the_reach_which_needs_both():
    steps = [step_fields(entry) for entry in STAGES["ladder_13b"]]
    work = [(step, options.get("part"), options.get("arms"), options.get("seeds")) for environment, step, options in steps if environment == "gpu"]
    assert work == [("fix_tokenizers", None, None, None),
                    ("train_adapter", 1, "ladder_half_native", "0"), ("ladder_probes", 1, "ladder_half_native", "0"),
                    ("evaluate_sampling", 1, "ladder_half_native", "0"), ("evaluate_loss", 1, "ladder_half_native", "0"),
                    ("negation_census", 1, "ladder_half_native", "0"), ("report", 1, "ladder_half_native", "0"),
                    ("train_adapter", 2, "native_same_picks", "0,1,2"), ("evaluate_sampling", 2, "native_same_picks", "0,1,2"),
                    ("evaluate_loss", 2, "native_same_picks", "0,1,2"), ("report", 2, "native_same_picks", "0,1,2"),
                    ("base_reach", 3, None, None)]
    labels = [options.get("label", step) for environment, step, options in steps if environment == "gpu"]
    assert len(set(labels)) == len(labels)                                   # no step's result file overwrites another's
    assert next(options for _, step, options in steps if step == "base_reach")["needs"] == (1, 2)
    assert all(environment == "guard" for environment, _, _ in steps[2::2])  # the card is checked idle before every GPU step


# --- the report over a tiny store ------------------------------------------------------------------------------------

BELOW = [f"b{i}" for i in range(10)]
NEVER = [f"n{i}" for i in range(20)]
TRAINED = [f"t{i}" for i in range(4)]
WORKBOOK = [f"w{i}" for i in range(20)]
MINIF2F = [f"m{i}" for i in range(4)]


def _rows(counts: dict, variant: str, status_of_failure: str = "lean_error"):
    attempts, verification = [], []
    for statement_id, (drawn, verified) in counts.items():
        for index in range(drawn):
            attempt_id = f"{statement_id}#{variant}#{index}"
            attempts.append({"attempt_id": attempt_id, "statement_id": statement_id, "variant": variant, "completion": f"proof {index % 3}",
                             "first_token_id": 100000})
            verification.append({"attempt_id": attempt_id, "status": "verified" if index < verified else status_of_failure, "seconds": 0.5})
    return attempts, verification


def ladder_config(void_range=(0.083, 0.188)) -> dict:
    return {"evaluation": {"bootstrap_resamples": 300, "bootstrap_seed": 0}, "profiles": {"smoke": {"minif2f_samples": 2}},
            "selection": {"phase_a_method": REUSED},
            "ladder": {"seed": 0, "void_base_fresh_rate": list(void_range), "never_proved_luck_expected": 25, "never_proved_luck_range": [13, 39]}}


def _fresh(store, who, set_name, counts, **kwargs):
    fresh = FreshNames(who, set_name)
    attempts, verification = _rows(counts, fresh.variant, **kwargs)
    store.write_rows(fresh.attempts_file, attempts)
    store.write_rows(fresh.verification_file, verification)
    store.mark_done(fresh.marker, {})


def tiny_store(tmp_path, adapter_below: int, base_below: int = 1, adapter_never_solved: int = 1) -> ArtifactStore:
    """Stored: below-band conjectures 1 of 12, never-proved 0 of 12, trained 6 of 12. Fresh: the base proves
    `base_below` of 8 on each below-band conjecture and the adapter `adapter_below`; of the never-proved ones the
    base proves 1 and the adapter the first `adapter_never_solved`."""
    store = ArtifactStore(tmp_path)
    names = arm_names(LADDER_ARM, 0, REUSED)
    store.write_rows("conjectures.jsonl", [{"conjecture_id": i, "holdout": index % 5 == 0} for index, i in enumerate(BELOW + NEVER + TRAINED)])
    store.write_rows("statements_reward.jsonl", [{"statement_id": i, "half": "validation"} for i in WORKBOOK])
    store.write_rows("minif2f_eval.jsonl", [{"statement_id": i} for i in MINIF2F])
    attempts, verification = _rows({**{i: (12, 1) for i in BELOW}, **{i: (12, 0) for i in NEVER}, **{i: (12, 6) for i in TRAINED},
                                    **{i: (8, 0) for i in WORKBOOK}}, "base")
    store.write_rows("proof_attempts.jsonl", attempts)
    store.write_rows("verification.jsonl", verification)
    attempts, verification = _rows({i: (2, 1) for i in MINIF2F}, "eval_base")
    store.write_rows("eval_attempts_base.jsonl", attempts)
    store.write_rows("eval_verification_base.jsonl", verification)
    store.write_rows("selection_half_pass_rate.jsonl", [{"method": "half_pass_rate", "conjecture_ids": TRAINED}])
    _fresh(store, "base", "below_band", {i: (8, base_below) for i in BELOW})
    _fresh(store, names.key, "below_band", {i: (8, adapter_below) for i in BELOW})
    _fresh(store, "base", "never_proved", {i: (8, 1 if index == 0 else 0) for index, i in enumerate(NEVER)})
    _fresh(store, names.key, "never_proved", {i: (8, 2 if index < adapter_never_solved else 0) for index, i in enumerate(NEVER)}, status_of_failure="timeout")
    _fresh(store, "base", "workbook_unsolved", {i: (4, 0) for i in WORKBOOK})
    _fresh(store, names.key, "workbook_unsolved", {i: (4, 0) for i in WORKBOOK})
    attempts, verification = _rows({**{i: (12, 2) for i in BELOW[::5] + NEVER[::5] + TRAINED[::5]}, **{i: (8, 1) for i in WORKBOOK}, **{i: (2, 2) for i in MINIF2F}}, names.variant)
    store.write_rows(names.eval_attempts_file, attempts)
    store.write_rows(names.eval_verification_file, verification)
    store.mark_done(names.sampling_marker, {})
    for name, body in (("heldout_loss_nf4_native_base.jsonl", 0.2), ("heldout_loss_fp8_native_base.jsonl", 0.19), (names.nf4_loss_file, 0.18), (names.fp8_loss_file, 0.17)):
        store.write_rows(name, [pair_loss_row(i, [0.0] + [body] * 10 + [0.1, 0.0]) for i in WORKBOOK])
    store.mark_done(names.loss_marker, {})
    store.mark_done(names.train_marker, {"arm": LADDER_ARM, "seed": 0, "steps": 18, "epochs": 1, "sequence_start_token_id": 100000, "losses": [0.25, 0.1],
                                         "heldout_parts_curve": [{"step": 0, "mean_loss": 0.2454, "body_per_token": 0.19,
                                                                  "nats_per_proof": {"first": 0.0, "body": 6.0, "newline": 0.1, "fence": 0.05},
                                                                  "contribution": {"first": 0.0, "body": 0.23, "newline": 0.01, "fence": 0.0}}]})
    store.mark_done("negation_census", {"never_proved": 20, "sampled": 19, "disproved": 4, "disproved_among_exact": 3, "negation_inexact": 2,
                                        "negation_does_not_compile": 1})
    return store


def test_an_adapter_above_the_base_on_the_untrained_band_escalates(tmp_path):
    store = tiny_store(tmp_path, adapter_below=3)                              # base 1 of 8 = 0.125, adapter 0.375
    report = build_ladder_report(store, ladder_config(), "smoke", [arm_names(LADDER_ARM, 0, REUSED)])
    primary = report["primary"]
    assert (primary["problems"], primary["base"], primary["adapter"]) == (10, 0.125, 0.375) and primary["delta"] == pytest.approx(0.25)
    assert primary["below_band_conjectures"] == 10 and primary["of_those_in_the_training_set"] == 0 and primary["training_examples"] == 4
    assert primary["stored_verified_counts"] == {"1": 10} and primary["base_stored_mean_rate"] == pytest.approx(0.0833, abs=1e-4)
    assert primary["base_fresh_mean_rate"] == 0.125 and primary["base_fresh_rate_in_range"] is True
    assert report["branch"]["name"] == ESCALATE and "ESCALATE" in report["headline"]
    assert (tmp_path / LADDER_REPORT_FILE).exists()


def test_the_secondaries_count_statements_proved_at_least_once_and_decide_when_the_primary_does_not(tmp_path):
    store = tiny_store(tmp_path, adapter_below=1, adapter_never_solved=9)        # primary: no difference at all
    report = build_ladder_report(store, ladder_config(), "smoke", [arm_names(LADDER_ARM, 0, REUSED)])
    never = report["secondaries"]["never_proved"]
    assert (never["statements"], never["samples_each"], never["base_proved_at_least_once"], never["adapter_proved_at_least_once"], never["proved_by_both"]) == (20, 8, 1, 9, 1)
    assert never["delta"] == pytest.approx(0.4) and never["label"] == "improved"
    assert report["secondaries"]["workbook_unsolved"]["adapter_proved_at_least_once"] == 0
    assert report["primary"]["delta"] == 0 and report["branch"]["name"] == SECONDARY_ABOVE and "never-proved conjectures" in report["branch"]["reason"]


def test_no_difference_anywhere_is_undetectable_and_a_base_out_of_range_is_void(tmp_path):
    store = tiny_store(tmp_path, adapter_below=1, adapter_never_solved=1)
    assert build_ladder_report(store, ladder_config(), "smoke", [arm_names(LADDER_ARM, 0, REUSED)])["branch"]["name"] == UNDETECTABLE
    report = build_ladder_report(store, ladder_config(void_range=(0.2, 0.3)), "smoke", [arm_names(LADDER_ARM, 0, REUSED)])
    assert report["branch"]["name"] == VOID and report["primary"]["base_fresh_rate_in_range"] is False


def test_the_report_states_how_each_model_s_fresh_samples_were_checked_and_carries_the_rest(tmp_path):
    store = tiny_store(tmp_path, adapter_below=3)
    report = build_ladder_report(store, ladder_config(), "smoke", [arm_names(LADDER_ARM, 0, REUSED)])
    checks = report["lean_checks_fresh"]["never_proved"]
    assert checks["base"] == {"checks": 160, "timeout_share": 0.0, "server_error_share": 0.0, "median_check_seconds": 0.5}
    assert checks["adapter"]["timeout_share"] == pytest.approx(158 / 160, abs=1e-5)       # planted: every adapter failure a timeout
    assert report["standard_evaluation"]["workbook_holdout_pass@1"]["adapter"] == 0.125
    assert report["solved_at_least_once"]["workbook_holdout"] == {"base": 0, "adapter": 20, "new": 20, "lost": 0}
    assert report["heldout_loss"]["nf4"]["adapter_by_part"]["body_per_token"] == pytest.approx(0.18)
    assert report["census"]["disproved"] == 4 and report["sequence_start_token_first_share"] == 1.0
    assert report["distinct_attempts_conjecture_holdout"]["adapter"] == pytest.approx(3 / 12)
    scalars = ladder_scalars(report)
    assert scalars["ladder/primary_delta"] == pytest.approx(0.25) and scalars["ladder/branch/ESCALATE"] == 1.0
    assert scalars["ladder/never_proved/timeout_share_adapter"] == pytest.approx(158 / 160, abs=1e-5) and scalars["census/disproved"] == 4.0
    assert all(isinstance(value, float) for value in scalars.values())


def test_fresh_counts_are_empty_until_the_set_is_marked_done_and_the_rung_is_one_run(tmp_path):
    store = tiny_store(tmp_path, adapter_below=3)
    assert fresh_counts(store, "base", "below_band")["b0"] == (8, 1)
    (tmp_path / "fresh_samples_below_band_base.done.json").unlink()
    assert fresh_counts(store, "base", "below_band") == {}
    with pytest.raises(RuntimeError):
        build_ladder_report(store, ladder_config(), "smoke", [arm_names(LADDER_ARM, 0, REUSED)])
    with pytest.raises(ValueError):
        build_ladder_report(store, ladder_config(), "smoke", [arm_names(LADDER_ARM, 1, REUSED)])
