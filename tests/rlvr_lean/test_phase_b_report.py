"""Spec fixture 14 (§13): the Phase B comparison arithmetic, its labels, and the report over a tiny store."""

import pytest

from rlvr_lean.domain.training.arms import arm_names
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.reporting.phase_b import arm_difference, build_phase_b_report, evaluation_scalars, loss_difference

REUSED = "learning_progress_cosine"


def test_an_arm_better_on_every_problem_reads_lp_worse():
    learning_progress = [{f"p{i}": (4, 1) for i in range(30)}]
    other = [{f"p{i}": (4, 3) for i in range(30)}]
    result = arm_difference(learning_progress, other, 1, resamples=500, seed=0)
    assert result["delta"] == pytest.approx(-0.5)
    assert result["label"] == "LP worse"


def test_identical_arms_read_no_measurable_difference():
    counts = [{f"p{i}": (4, i % 5) for i in range(30)}]
    assert arm_difference(counts, counts, 1, resamples=500, seed=0)["label"] == "no measurable difference"


def test_scores_are_per_problem_means_over_seeds_and_problems_must_exist_at_every_seed():
    learning_progress = [{"a": (4, 4), "b": (4, 0), "c": (4, 4)}, {"a": (4, 0), "b": (4, 4)}]     # "c" missing at seed 1
    other = [{"a": (4, 1), "b": (4, 1), "c": (4, 0)}, {"a": (4, 1), "b": (4, 1), "c": (4, 0)}]
    result = arm_difference(learning_progress, other, 1, resamples=200, seed=0)
    assert result["problems"] == 2 and result["seeds"] == 2
    assert result["learning_progress"] == pytest.approx(0.5) and result["other"] == pytest.approx(0.25)
    assert result["delta"] == pytest.approx(0.25)


def test_the_prompt_rule_uses_its_own_labels():
    learning_progress = [{f"p{i}": (4, 4) for i in range(20)}]
    other = [{f"p{i}": (4, 0) for i in range(20)}]
    assert arm_difference(learning_progress, other, 4, resamples=200, seed=0, prompt_rule=True)["label"] == "promising"


def test_a_positive_loss_difference_means_lp_lowered_the_loss_more():
    base = {f"s{i}": 1.0 for i in range(20)}
    result = loss_difference(base, [{key: 0.2 for key in base}], [{key: 0.6 for key in base}], resamples=200, seed=0)
    assert result["delta"] == pytest.approx(0.4)
    assert (result["learning_progress_reduction"], result["other_reduction"]) == (pytest.approx(0.8), pytest.approx(0.4))
    assert result["label"] == "LP better"


# --- the report over a tiny store ------------------------------------------------------------------------

def _attempts(store, file, statement_counts, variant):
    attempts, verification = [], []
    for statement_id, (drawn, verified) in statement_counts.items():
        for index in range(drawn):
            attempt_id = f"{statement_id}#{variant}#{index}"
            attempts.append({"attempt_id": attempt_id, "statement_id": statement_id, "variant": variant})
            verification.append({"attempt_id": attempt_id, "status": "verified" if index < verified else "lean_error"})
    return attempts, verification


def tiny_store(tmp_path):
    store = ArtifactStore(tmp_path)
    holdout = [f"h{i}" for i in range(12)]
    workbook = [f"w{i}" for i in range(12)]
    minif2f = [f"m{i}" for i in range(6)]
    store.write_rows("conjectures.jsonl", [{"conjecture_id": i, "holdout": True} for i in holdout]
                     + [{"conjecture_id": "t0", "holdout": False}, {"conjecture_id": "t1", "holdout": False}])
    store.write_rows("statements_reward.jsonl", [{"statement_id": i, "half": "validation"} for i in workbook])
    store.write_rows("minif2f_eval.jsonl", [{"statement_id": i} for i in minif2f])
    base_attempts, base_verification = _attempts(store, None, {**{i: (4, 1) for i in holdout + workbook}}, "base")
    store.write_rows("proof_attempts.jsonl", base_attempts)
    store.write_rows("verification.jsonl", base_verification)
    eval_attempts, eval_verification = _attempts(store, None, {i: (2, 1) for i in minif2f}, "eval_base")
    store.write_rows("eval_attempts_base.jsonl", eval_attempts)
    store.write_rows("eval_verification_base.jsonl", eval_verification)
    store.write_rows("heldout_loss_nf4_base.jsonl", [{"statement_id": i, "loss": 1.0} for i in workbook])
    store.write_rows("heldout_loss_fp8_base.jsonl", [{"statement_id": i, "loss": 1.0} for i in workbook])
    store.write_rows("pool.jsonl", [{"conjecture_id": i, "sample_count": 12, "verified_count": 6, "canonical_proof_tokens": 10,
                                     "canonical_proof_id": f"{i}#base#0", "learning_progress": 1.0, "cosine": 0.1} for i in ("t0", "t1")])
    store.write_rows("selections.jsonl", [{"method": "random", "conjecture_ids": ["t1"]}])
    store.write_rows("training_examples.jsonl", [{"conjecture_id": "t0", "proof_attempt_id": "t0#base#0"}])
    # LP solves 3 of 4 everywhere, random 2 of 4: LP is better on the workbook holdout.
    for arm, verified, loss in ((REUSED, 3, 0.3), ("random", 2, 0.5)):
        names = arm_names(arm, 0, REUSED)
        attempts, verification = _attempts(store, None, {**{i: (4, verified) for i in holdout + workbook}, **{i: (2, 2) for i in minif2f}}, names.variant)
        store.write_rows(names.eval_attempts_file, attempts)
        store.write_rows(names.eval_verification_file, verification)
        store.write_rows(names.nf4_loss_file, [{"statement_id": i, "loss": loss} for i in workbook])
        store.write_rows(names.fp8_loss_file, [{"statement_id": i, "loss": loss} for i in workbook])
        for marker in (names.train_marker, names.sampling_marker, names.loss_marker):
            store.mark_done(marker, {"steps": 3, "losses": [1.0, 0.5, 0.1]})
    return store


CONFIG = {"evaluation": {"bootstrap_resamples": 300, "bootstrap_seed": 0}, "selection": {"phase_a_method": REUSED},
          "profiles": {"smoke": {"minif2f_samples": 2}}}


def test_report_over_a_tiny_store(tmp_path):
    store = tiny_store(tmp_path)
    runs = [arm_names(REUSED, 0, REUSED), arm_names("random", 0, REUSED)]
    report = build_phase_b_report(store, CONFIG, "smoke", runs)
    assert report["seeds"] == [0] and report["arms"] == [REUSED, "random"]
    deciding = report["deciding"]["random"]
    assert deciding["delta"] == pytest.approx(0.25) and deciding["label"] == "LP better"
    assert report["comparisons"]["random"]["heldout_loss_nf4_reduction"]["label"] == "LP better"
    assert report["against_base"]["random"]["conjecture_holdout_pass@1"]["label"] == "improved"
    assert report["stops"] == []
    assert "three seeds" in report["headline"]
    assert store.path("phase_b_report.md").read_text().startswith("# rlvr_lean Phase B")


def test_evaluation_scalars_for_base_and_an_arm(tmp_path):
    store = tiny_store(tmp_path)
    base = evaluation_scalars(store, CONFIG, "smoke", None)
    assert base["eval/pass@1_workbook_holdout"] == pytest.approx(0.25)
    assert base["eval/pass@2_minif2f"] == pytest.approx(1.0) and base["eval/pass@1_minif2f"] == pytest.approx(0.5)
    adapter = evaluation_scalars(store, CONFIG, "smoke", arm_names("random", 0, REUSED))
    assert adapter["eval/pass@1_conjecture_holdout"] == pytest.approx(0.5)
    assert adapter["eval/heldout_loss_nf4"] == pytest.approx(0.5)
