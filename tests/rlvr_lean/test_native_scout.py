"""Spec §13a: the scout's branches (fixed before the run), its "also read" figures, and its report over a tiny store."""

import pytest

from rlvr_lean.domain.evaluation.loss_parts import pair_loss_row
from rlvr_lean.domain.evaluation.scout import ESCALATE, TOKEN_HABIT, VOID, control_reproduces, distinct_share, scout_branch, share_starting_with
from rlvr_lean.domain.training.arms import SCOUT_ARM, arm_names
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.reporting.native_scout import (
    SCOUT_REPORT_FILE,
    SCOUT_REPORT_MARKDOWN,
    SCOUT_SEEDS_REPORT_FILE,
    build_native_scout_report,
    raw_difference,
    scout_scalars,
)

REUSED = "learning_progress_cosine"
LINE = -0.004


# --- the pre-registered branches ---------------------------------------------------------------------------

@pytest.mark.parametrize("difference", [0.0, 0.0001, 0.0099, 0.05])
def test_a_difference_of_zero_or_more_escalates(difference):
    branch = scout_branch(difference, control_reproduced=True, token_habit_below=LINE)
    assert branch.name == ESCALATE and "seeds 1 and 2" in branch.action


@pytest.mark.parametrize("difference", [-0.0041, -0.01, -0.0744])
def test_a_difference_below_the_line_reads_the_token_habit_and_stops(difference):
    branch = scout_branch(difference, control_reproduced=True, token_habit_below=LINE)
    assert branch.name == TOKEN_HABIT and branch.action.startswith("stop and diagnose")


@pytest.mark.parametrize("difference", [-0.004, -0.0039, -0.0001])
def test_between_the_line_and_zero_is_no_measurable_change_at_one_seed_and_escalates(difference):
    branch = scout_branch(difference, control_reproduced=True, token_habit_below=LINE)
    assert branch.name == ESCALATE and "no measurable change at one seed" in branch.reason


@pytest.mark.parametrize("difference", [0.02, -0.02, -0.002])
def test_a_control_that_does_not_reproduce_is_void_whatever_the_difference(difference):
    branch = scout_branch(difference, control_reproduced=False, token_habit_below=LINE)
    assert branch.name == VOID and branch.action == "fix and re-run one seed"


def test_the_line_lies_below_zero_and_the_difference_is_a_number():
    with pytest.raises(ValueError):
        scout_branch(0.01, True, 0.004)
    with pytest.raises(ValueError):
        scout_branch(float("nan"), True, LINE)


def test_the_control_reproduces_to_the_four_decimals_the_pre_registration_quotes():
    assert control_reproduces(0.07441, 0.0744) and control_reproduces(0.074449, 0.0744)
    assert not control_reproduces(0.07446, 0.0744) and not control_reproduces(0.0843, 0.0744)


# --- the "also read" figures ---------------------------------------------------------------------------------

def test_distinct_share_counts_different_texts_per_statement_over_all_attempts():
    assert distinct_share({"a": ["simp", "simp", "rfl", "simp"], "b": ["x", "y"]}) == pytest.approx(4 / 6)
    assert distinct_share({"a": ["simp"] * 12}) == pytest.approx(1 / 12)
    with pytest.raises(ValueError):
        distinct_share({})


def test_the_share_of_attempts_that_start_with_the_sequence_start_token():
    assert share_starting_with([100, 100, 207, None], 100) == 0.5
    with pytest.raises(ValueError):
        share_starting_with([], 100)


def test_the_raw_difference_is_unrounded_and_over_the_problems_both_have():
    base = {"a": (8, 1), "b": (8, 0), "c": (8, 8)}
    adapted = {"a": (8, 2), "b": (8, 0)}
    assert raw_difference(base, adapted, 1) == pytest.approx((1 / 8 + 0) / 2)
    assert raw_difference(base, {}, 1) is None


# --- the report over a tiny store ----------------------------------------------------------------------------

HOLDOUT = [f"h{i}" for i in range(10)]
WORKBOOK = [f"w{i}" for i in range(20)]
MINIF2F = [f"m{i}" for i in range(6)]
BOS = 100000


def _rows(counts: dict, variant: str, first_token_id=None, texts=("simp", "rfl", "omega", "linarith")):
    attempts, verification = [], []
    for statement_id, (drawn, verified) in counts.items():
        for index in range(drawn):
            attempt_id = f"{statement_id}#{variant}#{index}"
            row = {"attempt_id": attempt_id, "statement_id": statement_id, "variant": variant, "completion": texts[index % len(texts)]}
            if first_token_id is not None:
                row["first_token_id"] = first_token_id
            attempts.append(row)
            verification.append({"attempt_id": attempt_id, "status": "verified" if index < verified else "lean_error"})
    return attempts, verification


def scout_config(base: float, legacy: float) -> dict:
    return {"evaluation": {"bootstrap_resamples": 300, "bootstrap_seed": 0}, "profiles": {"smoke": {"minif2f_samples": 2}},
            "selection": {"phase_a_method": REUSED},
            "scout": {"seed": 0, "token_habit_below": LINE, "control_base_workbook_pass_at_1": base, "control_legacy_workbook_pass_at_1": legacy}}


def tiny_store(tmp_path, native_workbook_verified: int) -> ArtifactStore:
    """Base: 1 of 4 on every holdout problem (pass@1 0.25). Legacy adapter: 2 of 4 (0.5). Native adapter:
    `native_workbook_verified` of 4 on the workbook holdout."""
    store = ArtifactStore(tmp_path)
    store.write_rows("conjectures.jsonl", [{"conjecture_id": i, "holdout": True} for i in HOLDOUT])
    store.write_rows("statements_reward.jsonl", [{"statement_id": i, "half": "validation"} for i in WORKBOOK])
    store.write_rows("minif2f_eval.jsonl", [{"statement_id": i} for i in MINIF2F])
    attempts, verification = _rows({i: (4, 1) for i in HOLDOUT + WORKBOOK}, "base")
    store.write_rows("proof_attempts.jsonl", attempts)
    store.write_rows("verification.jsonl", verification)
    attempts, verification = _rows({i: (2, 1) for i in MINIF2F}, "eval_base")
    store.write_rows("eval_attempts_base.jsonl", attempts)
    store.write_rows("eval_verification_base.jsonl", verification)
    legacy, native = arm_names(REUSED, 0, REUSED), arm_names(SCOUT_ARM, 0, REUSED)
    attempts, verification = _rows({**{i: (4, 2) for i in HOLDOUT + WORKBOOK}, **{i: (2, 1) for i in MINIF2F}}, legacy.variant, texts=("simp", "simp", "rfl", "simp"))
    store.write_rows(legacy.eval_attempts_file, attempts)
    store.write_rows(legacy.eval_verification_file, verification)
    attempts, verification = _rows({**{i: (4, 2) for i in HOLDOUT}, **{i: (4, native_workbook_verified) for i in WORKBOOK},
                                    **{i: (2, 2) for i in MINIF2F}}, native.variant, first_token_id=BOS, texts=("simp", "rfl", "rfl", "simp"))
    attempts[0]["first_token_id"] = 207                               # one attempt that did not start with the token
    store.write_rows(native.eval_attempts_file, attempts)
    store.write_rows(native.eval_verification_file, verification)
    # Held-out loss. Legacy rows as Phase A stored them (the mean only); native rows by part.
    store.write_rows("heldout_loss_nf4_base.jsonl", [{"statement_id": i, "loss": 0.94} for i in WORKBOOK])
    store.write_rows("heldout_loss_fp8_base.jsonl", [{"statement_id": i, "loss": 0.96} for i in WORKBOOK])
    store.write_rows(legacy.nf4_loss_file, [{"statement_id": i, "loss": 0.22} for i in WORKBOOK])
    store.write_rows(legacy.fp8_loss_file, [{"statement_id": i, "loss": 0.21} for i in WORKBOOK])
    for name, body in (("heldout_loss_nf4_native_base.jsonl", 0.2), ("heldout_loss_fp8_native_base.jsonl", 0.19),
                       (native.nf4_loss_file, 0.18), (native.fp8_loss_file, 0.17)):
        store.write_rows(name, [pair_loss_row(i, [0.0] + [body] * 10 + [0.1, 0.0]) for i in WORKBOOK])
    store.mark_done(legacy.train_marker, {"arm": REUSED, "seed": 0, "steps": 54, "first_loss": 2.2324, "last_loss": 0.0183, "losses": [2.2324, 0.0183]})
    store.mark_done(native.train_marker, {"arm": SCOUT_ARM, "seed": 0, "steps": 54, "target_format": "native", "sequence_start_token_id": BOS,
                                          "first_loss": 0.25, "last_loss": 0.002, "losses": [0.25, 0.002],
                                          "heldout_parts_curve": [{"step": 0, "mean_loss": 0.2454, "body_per_token": 0.19,
                                                                   "nats_per_proof": {"first": 0.0003, "body": 6.0, "newline": 0.1, "fence": 0.05},
                                                                   "contribution": {"first": 0.0, "body": 0.23, "newline": 0.01, "fence": 0.0}}]})
    store.mark_done(native.sampling_marker, {})
    store.mark_done(native.loss_marker, {})
    return store


def test_a_native_adapter_above_base_escalates_with_the_legacy_control_beside_it(tmp_path):
    store = tiny_store(tmp_path, native_workbook_verified=3)
    report = build_native_scout_report(store, scout_config(0.25, 0.5), "smoke", [arm_names(SCOUT_ARM, 0, REUSED)])
    primary = report["results"]["workbook_holdout_pass@1"]
    assert primary["native_minus_base"]["problems"] == 20
    assert (primary["native_minus_base"]["base"], primary["native_minus_base"]["adapter"]) == (0.25, 0.75)
    assert primary["native_minus_base"]["delta"] == pytest.approx(0.5) and primary["native_minus_base"]["label"] == "improved"
    assert primary["legacy_minus_base"]["adapter"] == 0.5 and primary["legacy_minus_base"]["delta"] == pytest.approx(0.25)
    assert primary["native_minus_legacy"]["delta"] == pytest.approx(0.25)
    assert report["control"]["reproduced"] is True
    assert report["branch"]["name"] == ESCALATE and report["branch"]["difference"] == pytest.approx(0.5)
    assert "ESCALATE" in report["headline"] and "reproduced" in report["headline"]
    assert report["results"]["minif2f_pass@2"]["native_minus_base"]["adapter"] == 1.0


def test_a_native_adapter_below_the_line_reads_the_token_habit(tmp_path):
    store = tiny_store(tmp_path, native_workbook_verified=0)
    report = build_native_scout_report(store, scout_config(0.25, 0.5), "smoke", [arm_names(SCOUT_ARM, 0, REUSED)])
    assert report["branch"]["name"] == TOKEN_HABIT and report["branch"]["difference"] == pytest.approx(-0.25)
    assert report["results"]["workbook_holdout_pass@1"]["native_minus_base"]["label"] == "degraded"


def test_a_control_that_reads_differently_from_the_pre_registration_is_void(tmp_path):
    store = tiny_store(tmp_path, native_workbook_verified=3)
    report = build_native_scout_report(store, scout_config(0.25, 0.4321), "smoke", [arm_names(SCOUT_ARM, 0, REUSED)])
    assert report["control"]["reproduced"] is False and report["branch"]["name"] == VOID
    assert "NOT REPRODUCED" in report["headline"]


def test_the_report_reads_the_loss_by_part_the_variety_and_the_first_token(tmp_path):
    store = tiny_store(tmp_path, native_workbook_verified=3)
    report = build_native_scout_report(store, scout_config(0.25, 0.5), "smoke", [arm_names(SCOUT_ARM, 0, REUSED)])
    native_nf4, legacy_nf4 = report["heldout_loss"]["nf4"]["native"], report["heldout_loss"]["nf4"]["legacy"]
    assert native_nf4["base_by_part"]["body_per_token"] == pytest.approx(0.2) and native_nf4["adapter_by_part"]["body_per_token"] == pytest.approx(0.18)
    assert native_nf4["base_by_part"]["nats_per_proof"]["first"] == 0.0
    assert native_nf4["comparison"]["loss_reduction"] == pytest.approx(10 * 0.02 / 13, abs=1e-4)
    assert legacy_nf4["comparison"]["loss_reduction"] == pytest.approx(0.72) and legacy_nf4["base_by_part"] is None     # stored without parts
    assert report["heldout_loss"]["fp8"]["native"]["adapter_by_part"]["body_per_token"] == pytest.approx(0.17)
    assert report["heldout_loss_by_part_during_training"][0]["step"] == 0
    assert report["distinct_attempts_conjecture_holdout"] == {"base": 1.0, "legacy": 0.5, "native": 0.5}
    start = report["sequence_start_token_when_sampling"]
    attempts = 4 * 10 + 4 * 20 + 2 * 6
    assert start["attempts"] == attempts and start["native_share"] == pytest.approx((attempts - 1) / attempts, abs=1e-5)
    assert start["native_share_by_set"]["workbook_holdout"] == 1.0
    assert "heldout_parts_curve" not in report["training"]["native"] and report["training"]["legacy"]["first_loss"] == 2.2324


def test_the_report_writes_its_own_files_and_leaves_phase_a_s_and_phase_b_s_alone(tmp_path):
    store = tiny_store(tmp_path, native_workbook_verified=3)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    build_native_scout_report(store, scout_config(0.25, 0.5), "smoke", [arm_names(SCOUT_ARM, 0, REUSED)])
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert set(after) - set(before) == {SCOUT_REPORT_FILE, SCOUT_REPORT_MARKDOWN}
    assert all(after[name] == content for name, content in before.items())
    assert not {"report.jsonl", "report.md", "phase_b_report.jsonl", "phase_b_report.md"} & set(after)
    assert "ESCALATE" in (tmp_path / SCOUT_REPORT_MARKDOWN).read_text()


def test_the_scout_needs_its_pre_registered_seed_and_every_run_evaluated(tmp_path):
    store = tiny_store(tmp_path, native_workbook_verified=3)
    with pytest.raises(ValueError):                                    # the scout's own seed is missing
        build_native_scout_report(store, scout_config(0.25, 0.5), "smoke", [arm_names(SCOUT_ARM, 1, REUSED)])
    with pytest.raises(ValueError):                                    # another arm is not the scout
        build_native_scout_report(store, scout_config(0.25, 0.5), "smoke", [arm_names(SCOUT_ARM, 0, REUSED), arm_names("random", 0, REUSED)])
    with pytest.raises(RuntimeError):                                  # seed 1 is listed and not evaluated
        build_native_scout_report(store, scout_config(0.25, 0.5), "smoke", [arm_names(SCOUT_ARM, 0, REUSED), arm_names(SCOUT_ARM, 1, REUSED)])
    (tmp_path / "evaluate_loss_native_same_picks_seed0.done.json").unlink()
    with pytest.raises(RuntimeError):
        build_native_scout_report(store, scout_config(0.25, 0.5), "smoke", [arm_names(SCOUT_ARM, 0, REUSED)])


# --- the read over several seeds (the escalation the scout earned) ---------------------------------------------

def add_seed(store: ArtifactStore, seed: int, native_workbook_verified: int, legacy_workbook_verified: int) -> None:
    legacy, native = arm_names(REUSED, seed, REUSED), arm_names(SCOUT_ARM, seed, REUSED)
    attempts, verification = _rows({**{i: (4, 2) for i in HOLDOUT}, **{i: (4, legacy_workbook_verified) for i in WORKBOOK},
                                    **{i: (2, 1) for i in MINIF2F}}, legacy.variant)
    store.write_rows(legacy.eval_attempts_file, attempts)
    store.write_rows(legacy.eval_verification_file, verification)
    attempts, verification = _rows({**{i: (4, 2) for i in HOLDOUT}, **{i: (4, native_workbook_verified) for i in WORKBOOK},
                                    **{i: (2, 2) for i in MINIF2F}}, native.variant, first_token_id=BOS)
    store.write_rows(native.eval_attempts_file, attempts)
    store.write_rows(native.eval_verification_file, verification)
    for name, body in ((native.nf4_loss_file, 0.18), (native.fp8_loss_file, 0.17)):
        store.write_rows(name, [pair_loss_row(i, [0.0] + [body] * 10 + [0.1, 0.0]) for i in WORKBOOK])
    store.mark_done(native.train_marker, {"arm": SCOUT_ARM, "seed": seed, "steps": 54, "sequence_start_token_id": BOS, "losses": [0.25, 0.002]})
    store.mark_done(native.sampling_marker, {})
    store.mark_done(native.loss_marker, {})


def test_over_three_seeds_a_problem_s_score_is_the_mean_of_its_seeds_and_the_scout_s_own_report_stays(tmp_path):
    store = tiny_store(tmp_path, native_workbook_verified=3)                  # seed 0: native 0.75, legacy 0.5, base 0.25
    one_seed = build_native_scout_report(store, scout_config(0.25, 0.5), "smoke", [arm_names(SCOUT_ARM, 0, REUSED)])
    scout_file = (tmp_path / SCOUT_REPORT_FILE).read_bytes()
    add_seed(store, 1, native_workbook_verified=1, legacy_workbook_verified=2)     # native 0.25, legacy 0.5
    add_seed(store, 2, native_workbook_verified=2, legacy_workbook_verified=4)     # native 0.5, legacy 1.0
    runs = [arm_names(SCOUT_ARM, seed, REUSED) for seed in (0, 1, 2)]
    report = build_native_scout_report(store, scout_config(0.25, 0.5), "smoke", runs)
    primary = report["over_seeds"]["results"]["workbook_holdout_pass@1"]
    assert report["seeds"] == [0, 1, 2] and primary["native_minus_base"]["seeds"] == 3
    assert primary["native_minus_base"]["adapter"] == pytest.approx(0.5) and primary["native_minus_base"]["delta"] == pytest.approx(0.25)
    assert primary["legacy_minus_base"]["adapter"] == pytest.approx(0.6667, abs=1e-4)
    assert primary["native_minus_legacy"]["delta"] == pytest.approx(0.5 - 0.6667, abs=1e-4) and primary["native_minus_legacy"]["label"] == "degraded"
    assert report["over_seeds"]["per_seed"]["workbook_holdout_pass@1"]["native_minus_base"] == [0.5, 0.0, 0.25]
    assert report["one_seed"]["headline"] == one_seed["headline"] and report["one_seed"]["branch"]["name"] == ESCALATE
    assert (tmp_path / SCOUT_REPORT_FILE).read_bytes() == scout_file              # the one-seed scout's file is not rewritten
    assert (tmp_path / SCOUT_SEEDS_REPORT_FILE).exists() and "seeds [0, 1, 2]" in report["headline"]
    scalars = scout_scalars(report)
    assert scalars["scout/seeds_workbook_holdout_pass@1/native_minus_base_delta"] == pytest.approx(0.25)
    assert scalars["scout/one_seed_branch/ESCALATE"] == 1.0 and scalars["scout/primary_native_minus_base_seed1"] == 0.0
    assert all(isinstance(value, float) for value in scalars.values())
