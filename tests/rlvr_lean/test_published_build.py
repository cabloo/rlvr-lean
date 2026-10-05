"""Candidates from the published files: which statements have an answer, which are left out and why.
Spec: docs/spec/ladder-loop.spec.md, "A problem, and the pool" (fixtures 2 and 3). The files here are tiny
stand-ins written in the publishers' own shapes; nothing touches the network or Lean."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from rlvr_lean.data.published import (
    DISPROOF_UNPARSED,
    DISPROOF_WITH_HYPOTHESES,
    NO_USABLE_CERTIFICATE,
    PUBLISHED_BOTH_WAYS,
    PublishedFile,
    build_candidates,
    candidate_from_row,
    candidate_row,
    published_files,
    verified,
)
from rlvr_lean.domain.conjecturing.dedup import content_id, normalize_statement
from rlvr_lean.domain.problem_pool import FALSE_SIDE, GOEDEL, INTERNLM_PROOFS, INTERNLM_ROWS, LEAN_WORKBOOK, STP, STP_CONJECTURE, TRUE_SIDE
from rlvr_lean.domain.problem_pool.certificates import STP_HEADER, STP_PROMPT_HEAD
from rlvr_lean.domain.verification import LEAN_HEADER, lean_pin

pyarrow = pytest.importorskip("pyarrow", reason="the `data` dependency group (pyarrow) is not installed")
import pyarrow.parquet  # noqa: E402

CONFIG = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean" / "config" / "experiment.yaml"
SETTINGS = {"certificates": {"source_order": [GOEDEL, STP, INTERNLM_ROWS, INTERNLM_PROOFS], "per_source": 2, "stp_conjectures_checked": 2},
            "heldout": {"seed": 5}}


def _workbook_row(name, rest, natural_language, proofs=()):
    return {"formal_statement": f"theorem {name} {rest}  :=  by sorry", "natural_language_statement": natural_language,
            "proof": list(proofs), "split": "lean_workbook"}


WORKBOOK = [
    _workbook_row("wb_1", "(x : ℝ) : x ^ 2 ≥ 0", "squares", ["nlinarith [sq_nonneg x]", "positivity", "exact sq_nonneg x -- long way round"]),
    _workbook_row("wb_2", "(n : ℕ) : ∑ i in Finset.range n, i = n * (n - 1) / 2", "a sum"),
    _workbook_row("wb_3", ": ∀ a : ℝ, a > 0", "false, no binders"),
    _workbook_row("wb_4", "(x y : ℝ) : x + y = 2", "false, variables only"),
    _workbook_row("wb_5", "(x : ℝ) (h : 0 < x) : x < 0", "false, with a hypothesis"),
    _workbook_row("wb_6", "(a b : ℝ) (h₁ : a < b) (h₂ : b < a) : a = 1", "published both ways"),
    _workbook_row("wb_7", "∀ x > 0, x = 1", "malformed: no colon after the name"),
    _workbook_row("wb_8", "(x : ℕ) : x + 0 = x", "no published answer"),
    _workbook_row("wb_9", "(x : ℕ) : x * 1 = x", "holdout, no published answer"),
    _workbook_row("wb_10", "(x : ℕ) : x + 1 > x", "only a filtered proof"),
    _workbook_row("wb_11", "{x : ℝ} : x = 1", "false, an implicit variable"),
    {"formal_statement": "def not_a_theorem : ℕ := by sorry", "natural_language_statement": "a def", "proof": [], "split": "lean_workbook"},
]


def _rows(name, status, steps):
    return [{"id": name, "status": status, "tactic": tactic, "state_before": before, "state_after": after} for tactic, before, after in steps]


INTERNLM_ROWS_FILE = (
    _rows("wb_2", "proved", [("induction n <;> simp_all [Finset.sum_range_succ]", "n : ℕ\n⊢ goal", "S1"), ("omega", "S1", "no goals")])
    + _rows("wb_3", "disproved", [("push_neg", "⊢ ¬∀ (a : ℝ), a > 0", "⊢ ∃ a, a ≤ 0"), ("exact ⟨0, le_refl 0⟩", "⊢ ∃ a, a ≤ 0", "no goals")])
    + _rows("wb_4", "disproved", [("intro h", "x y : ℝ\n⊢ ¬x + y = 2", "x y : ℝ\nh : x + y = 2\n⊢ False"),
                                  ("nlinarith", "x y : ℝ\nh : x + y = 2\n⊢ False", "no goals")])
    + _rows("wb_5", "disproved", [("linarith", "x : ℝ\nh : 0 < x\n⊢ ¬x < 0", "no goals")])
    + _rows("wb_6", "disproved", [("linarith", "a b : ℝ\nh₁ : a < b\nh₂ : b < a\n⊢ ¬a = 1", "no goals")])
    + _rows("wb_7", "disproved", [("norm_num", "⊢ ¬∀ x > 0, x = 1", "no goals")])
    + _rows("wb_11", "disproved", [("simp", "x : ℝ\n⊢ ¬x = 1", "no goals")])
)

GOEDEL_FILE = [
    {"problem_id": "wb_1", "full_proof": LEAN_HEADER + "/- squares -/\ntheorem wb_1 (x : ℝ) : x ^ 2 ≥ 0  := by\n  -- a square\n  positivity\n"},
    {"problem_id": "wb_6", "full_proof": LEAN_HEADER + "theorem wb_6 (a b : ℝ) (h₁ : a < b) (h₂ : b < a) : a = 1 := by\n  linarith\n"},
    {"problem_id": "wb_10", "full_proof": LEAN_HEADER + "theorem wb_10 (x : ℕ) : x + 1 > x := by\n  sorry\n"},
    {"problem_id": "wb_8", "full_proof": LEAN_HEADER + "theorem wb_8 (y : ℕ) : y = y := by\n  rfl\n"},       # another statement under this name
]

CONJECTURE_A = "theorem lean_workbook_p1 (x : ℝ) : x * 0 = 0 := by"
CONJECTURE_B = "theorem amgm (a b : ℝ) : a ^ 2 + b ^ 2 ≥ 2 * a * b := by"
CONJECTURE_C = "theorem third (n : ℕ) : n ≤ n ^ 2 := by"
MINIF2F = "theorem mathd_algebra_1 (x : ℝ) : x + 1 = 1 + x := by\n"


def _stp(body, target, tag):
    return {"prompt": STP_PROMPT_HEAD + STP_HEADER + body, "target": target, "tag": f"['{tag}']"}


STP_FILE = [
    _stp("theorem wb_1 (x : ℝ) : x ^ 2 ≥ 0  :=  by", "\n  nlinarith [sq_nonneg x, sq_nonneg (x + 1)]", "statement"),
    _stp("theorem wb_1 (x : ℝ) : x ^ 2 ≥ 0  :=  by", "\n  positivity", "statement"),
    _stp("theorem wb_1 (x : ℝ) : x ^ 2 ≥ 0  :=  by", "\n  exact sq_nonneg x", "statement"),
    _stp("theorem mathd_algebra_1 (x : ℝ) : x + 1 = 1 + x := by", "\n  ring", "statement"),               # not a workbook statement
    _stp(CONJECTURE_A, "\n  simp", "conjecture"),
    _stp(CONJECTURE_A.replace("lean_workbook_p1", "renamed"), "\n  ring", "conjecture"),                  # the same conjecture, another name
    _stp(CONJECTURE_B, " nlinarith [sq_nonneg (a - b)]", "conjecture"),
    _stp(CONJECTURE_C, "\n  nlinarith [sq_nonneg n]", "conjecture"),
    _stp("theorem copy_of_minif2f (x : ℝ) : x + 1 = 1 + x := by", "\n  ring", "conjecture"),              # equal to a miniF2F statement
    _stp("theorem cut (x : ℝ) : x = x :=", " by rfl", "conjecture"),                                     # the prompt stops at `:=`
    {"prompt": STP_PROMPT_HEAD + "/- Copyright -/\nimport Mathlib\ntheorem m : 1 = 1 := by", "target": "\n  rfl", "tag": "['mathlib']"},
]


def _parquet(path: Path, rows: list[dict]) -> Path:
    pyarrow.parquet.write_table(pyarrow.Table.from_pylist(rows), path)
    return path


@pytest.fixture
def build(tmp_path):
    workbook = tmp_path / "lean_workbook.json"
    workbook.write_text(json.dumps(WORKBOOK))
    files = {"internlm_workbook": workbook, "internlm_rows": _parquet(tmp_path / "rows.parquet", INTERNLM_ROWS_FILE),
             "goedel": _parquet(tmp_path / "goedel.parquet", GOEDEL_FILE), "stp:00": _parquet(tmp_path / "stp0.parquet", STP_FILE[:6]),
             "stp:01": _parquet(tmp_path / "stp1.parquet", STP_FILE[6:])}
    return build_candidates(files, SETTINGS, lean_pin("v4.27"), {normalize_statement(MINIF2F)}, {"wb_1", "wb_8", "wb_9", "wb_5"}, log=lambda message: None)


def _by_id(build):
    return {candidate.problem_id: candidate for candidate in build.candidates}


def _reasons(build):
    return {row["problem_id"]: row["reason"] for row in build.excluded}


def test_a_proved_statement_gets_its_certificates_in_the_configured_order_shortest_first(build):
    candidate = _by_id(build)["wb_1"]
    assert (candidate.kind, candidate.side, candidate.group[:3]) == (LEAN_WORKBOOK, TRUE_SIDE, "nl_")
    assert [(certificate.source, certificate.proof) for certificate in candidate.certificates] == [
        (GOEDEL, "  positivity"),
        (STP, "  positivity"), (STP, "  exact sq_nonneg x"),                       # two of STP's three, the shortest
        (INTERNLM_PROOFS, "  positivity"), (INTERNLM_PROOFS, "  exact sq_nonneg x"),   # two of the three; its comment removed
    ]
    assert all(certificate.theorem == "theorem wb_1 (x : ℝ) : x ^ 2 ≥ 0 := by\n" for certificate in candidate.certificates)


def test_the_old_sum_notation_is_rewritten_for_the_pin_and_marked(build):
    candidate = _by_id(build)["wb_2"]
    assert candidate.statement == "theorem wb_2 (n : ℕ) : ∑ i ∈ Finset.range n, i = n * (n - 1) / 2 := by\n"
    assert candidate.statement_published == "theorem wb_2 (n : ℕ) : ∑ i in Finset.range n, i = n * (n - 1) / 2 := by\n"
    assert candidate.rewritten == 1 and build.counts["candidates_with_a_rewritten_statement"] == 1
    assert [(certificate.source, certificate.proof) for certificate in candidate.certificates] == [
        (INTERNLM_ROWS, "  induction n <;> simp_all [Finset.sum_range_succ]\n  omega")]
    assert candidate.certificates[0].theorem == candidate.statement


def test_a_disproved_statement_without_hypotheses_is_a_known_false_candidate(build):
    closed, variables = _by_id(build)["wb_3"], _by_id(build)["wb_4"]
    assert closed.side == FALSE_SIDE and closed.statement == "theorem wb_3 : ∀ a : ℝ, a > 0 := by\n"
    assert closed.certificates[0].theorem == "theorem negation_of_wb_3 : ¬ (∀ a : ℝ, a > 0) := by\n"
    assert closed.certificates[0].proof == "  push_neg\n  exact ⟨0, le_refl 0⟩"
    assert variables.certificates[0].theorem == "theorem negation_of_wb_4 : ¬ (∀ (x y : ℝ), x + y = 2) := by\n"
    assert "exact published_for_every_value default default (claimed_for_every_value default default)" in variables.certificates[0].proof
    assert build.counts["negations_with_variables"] == 1 and build.counts["candidates_known_false"] == 2


def test_what_is_left_out_is_counted_by_reason(build):
    """Fixture 3: published both ways is out; a known-false statement whose exact negation is not built is out."""
    reasons = _reasons(build)
    assert reasons == {"wb_5": DISPROOF_WITH_HYPOTHESES, "wb_6": PUBLISHED_BOTH_WAYS, "wb_7": DISPROOF_UNPARSED,
                       "wb_10": NO_USABLE_CERTIFICATE, "wb_11": "negation_wrapper_not_built", "wb_8": NO_USABLE_CERTIFICATE}
    assert not set(reasons) & set(_by_id(build))
    assert build.counts["certificates_rejected_lexically"] == 1                          # the proof that is `sorry`
    assert build.counts["goedel_proofs_unreadable_or_of_another_statement"] == 1         # wb_8's file states another theorem
    assert build.counts["workbook_rows_not_a_theorem"] == 1
    assert build.counts["excluded_published_both_ways"] == 1


def test_stp_conjectures_are_named_by_content_and_taken_in_the_seeded_order(build):
    conjectures = [candidate for candidate in build.candidates if candidate.kind == STP_CONJECTURE]
    assert build.counts["stp_conjectures_with_a_published_proof"] == 3 and len(conjectures) == 2      # the first two of three
    assert [row["position"] for row in build.stp_index] == [0, 1, 2]
    assert [candidate.problem_id for candidate in conjectures] == [row["problem_id"] for row in build.stp_index[:2]]
    identifier = f"stp_{content_id(CONJECTURE_A + chr(10))}"
    by_id = {row["problem_id"]: row for row in build.stp_index}
    assert len(by_id[identifier]["proofs"]) == 2                                          # both names are one conjecture
    for candidate in conjectures:
        assert candidate.statement.startswith(f"theorem {candidate.problem_id} ") and candidate.group == candidate.problem_id
        assert all(certificate.source == STP and certificate.theorem == candidate.statement for certificate in candidate.certificates)


def test_stp_rows_the_pool_does_not_use_are_counted(build):
    assert build.counts["stp_statements_not_lean_workbook"] == 1
    assert build.counts["stp_conjectures_equal_to_minif2f"] == 1
    assert build.counts["stp_prompts_unreadable_conjecture"] == 1
    assert build.counts["stp_rows_mathlib"] == 1
    assert not any("mathd" in candidate.problem_id or "minif2f" in candidate.statement for candidate in build.candidates)


def test_a_proof_that_continues_the_prompts_line_is_put_under_the_theorem(build):
    amgm = f"stp_{content_id(CONJECTURE_B + chr(10))}"
    candidate = _by_id(build).get(amgm)
    if candidate is not None:                       # it is in the first two of the seeded order or it is not
        assert candidate.certificates[0].proof == "  nlinarith [sq_nonneg (a - b)]"


def test_the_present_holdout_without_a_published_answer_is_set_aside(build):
    # wb_9 only: wb_8 HAS a published answer (an unreadable one), wb_5 a published disproof, wb_1 is a candidate.
    assert [row["statement_id"] for row in build.set_aside] == ["wb_9"]
    assert build.counts["present_holdout"] == 4 and build.counts["present_holdout_candidates"] == 1
    assert all(row["statement"].endswith(":= by\n") and row["group"].startswith("nl_") for row in build.set_aside)


def test_candidates_survive_being_stored(build):
    for candidate in build.candidates:
        assert candidate_from_row(json.loads(json.dumps(candidate_row(candidate), ensure_ascii=False))) == candidate


# ------------------------------------------------------------------------------------------- pinned sources
def test_every_source_in_the_config_is_pinned_by_revision_and_hash():
    files = published_files(yaml.safe_load(CONFIG.read_text()))
    assert len(files) == 13 and {published.source for published in files} == {"internlm_workbook", "goedel_workbook_proofs", "stp"}
    for published in files:
        assert len(published.revision) == 40 and len(published.sha256) == 64 and int(published.sha256, 16) >= 0
        assert published.url == f"https://huggingface.co/datasets/{published.repo}/resolve/{published.revision}/{published.path}"
    assert len({published.local(Path("/store")) for published in files}) == 13


def test_a_file_with_another_hash_is_refused_and_one_with_the_pinned_hash_is_read(tmp_path):
    content = b"published bytes"
    good = PublishedFile("stp", "publisher/repo", "a" * 40, "data/part.parquet", hashlib.sha256(content).hexdigest())
    good.local(tmp_path).parent.mkdir(parents=True)
    good.local(tmp_path).write_bytes(content)
    assert verified(good, tmp_path, download=False) == good.local(tmp_path)
    good.local(tmp_path).write_bytes(content + b"!")
    with pytest.raises(ValueError, match="refused"):
        verified(good, tmp_path, download=False)
    with pytest.raises(FileNotFoundError):
        verified(PublishedFile("stp", "publisher/repo", "b" * 40, "data/other.parquet", "0" * 64), tmp_path, download=False)
