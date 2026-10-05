"""The renaming of library names in published proofs. Spec: docs/spec/ladder-loop.spec.md, "A published proof
is brought to our pin mechanically, and Lean still judges it". Pure: no Lean, no network."""

from pathlib import Path

import yaml

from rlvr_lean.domain.problem_pool import FALSE_SIDE, GOEDEL, INTERNLM_ROWS, TRUE_SIDE, Certificate, certificate_source
from rlvr_lean.domain.problem_pool.certificates import negation_certificate
from rlvr_lean.domain.problem_pool.renames import (
    identifiers,
    opened_namespaces,
    rename_library_names,
    renamed_certificate,
    tabled_name,
    written_forms,
)

TABLE = {"le_div_iff": "le_div_iff₀", "div_le_iff": "div_le_iff₀", "Real.sqrt_eq_iff_sq_eq": "Real.sqrt_eq_iff_eq_sq",
         "Nat.dvd_sub'": "Nat.dvd_sub", "Set.eq_empty_iff_forall_not_mem": "Set.eq_empty_iff_forall_notMem"}
CONFIG = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean" / "config" / "experiment.yaml"


def _names(text):
    return [name for _, _, name in identifiers(text)]


def test_an_identifier_is_the_whole_dotted_name():
    assert _names("rw [div_le_iff hc, Nat.le_div_iff_mul_le]") == ["rw", "div_le_iff", "hc", "Nat.le_div_iff_mul_le"]
    assert _names("exact le_div_iff₀.mpr h'") == ["exact", "le_div_iff₀.mpr", "h'"]
    assert _names("(h x).le_div_iff") == ["h", "x"]              # a projection of a parenthesis is not an identifier
    assert _names("x.1.le") == ["x"]
    assert _names("2le") == []


def test_a_tabled_name_is_renamed_wherever_a_proof_cites_it():
    cases = {
        "  rw [div_le_iff hc]": "  rw [div_le_iff₀ hc]",
        "  rw [← le_div_iff (by positivity)] at h": "  rw [← le_div_iff₀ (by positivity)] at h",
        "  simp [le_div_iff, div_le_iff] at *": "  simp [le_div_iff₀, div_le_iff₀] at *",
        "  nlinarith [(le_div_iff hc).mp h, sq_nonneg x]": "  nlinarith [(le_div_iff₀ hc).mp h, sq_nonneg x]",
        "  exact (@div_le_iff ℝ _ a b c hc).2 h": "  exact (@div_le_iff₀ ℝ _ a b c hc).2 h",
        "  apply Nat.dvd_sub' h₁ h₂": "  apply Nat.dvd_sub h₁ h₂",
    }
    for published, expected in cases.items():
        text, changed = rename_library_names(published, TABLE)
        assert text == expected and len(changed) >= 1


def test_a_name_followed_by_a_projection_is_that_name():
    assert rename_library_names("  exact Set.eq_empty_iff_forall_not_mem.mpr h", TABLE) == (
        "  exact Set.eq_empty_iff_forall_notMem.mpr h", ("Set.eq_empty_iff_forall_not_mem",))
    assert tabled_name("le_div_iff.mpr", TABLE) == "le_div_iff" and tabled_name("le_div_iff", TABLE) == "le_div_iff"


def test_another_name_that_holds_a_tabled_name_is_left_alone():
    for text in ("  rw [Nat.le_div_iff_mul_le hc]",          # extends it
                 "  rw [Nat.le_div_iff hc]",                  # ends in it: another lemma
                 "  rw [NNReal.div_le_iff hr]",
                 "  rw [le_div_iff₀ hc]",                     # the new name itself
                 "  rw [le_div_iff' hc]",
                 "  rw [div_le_iff_of_neg hc]",
                 "  exact h.le_div_iff",                      # a field of a local hypothesis
                 "  rw [my_le_div_iff]",
                 "  rw [Real.sqrt_eq_iff_sq_eq_extra]",
                 "  apply Nat.dvd_sub h₁ h₂",
                 "  simp [sqrt_eq_iff_sq_eq]"):               # the short form is not a key unless the table lists it
        assert rename_library_names(text, TABLE) == (text, ())
    assert tabled_name("Nat.le_div_iff", TABLE) is None and tabled_name("x.le_div_iff.mpr", TABLE) is None


def test_every_occurrence_is_renamed_and_the_changed_names_are_listed_once():
    text, changed = rename_library_names("  rw [le_div_iff h, le_div_iff h']\n  exact (div_le_iff h).2 (le_div_iff h |>.1 x)", TABLE)
    assert text == "  rw [le_div_iff₀ h, le_div_iff₀ h']\n  exact (div_le_iff₀ h).2 (le_div_iff₀ h |>.1 x)"
    assert changed == ("div_le_iff", "le_div_iff")


def test_renaming_twice_changes_nothing_more():
    once, _ = rename_library_names("  rw [le_div_iff h]", TABLE)
    assert rename_library_names(once, TABLE) == (once, ())


def test_a_renamed_certificate_is_marked_and_its_theorem_is_untouched():
    theorem = "theorem t (a b c : ℝ) (hc : 0 < c) (le_div_iff : a ≤ b) : a ≤ b / c * c := by\n"      # a hypothesis NAMED like the lemma
    certificate = Certificate("t", TRUE_SIDE, GOEDEL, theorem, "  rw [div_le_iff hc]\n  linarith", rewritten=1)
    renamed = renamed_certificate(certificate, TABLE)
    assert renamed.theorem == theorem and renamed.proof == "  rw [div_le_iff₀ hc]\n  linarith"
    assert renamed.renamed == ("div_le_iff",) and renamed.rewritten == 1 and renamed.source == GOEDEL
    assert certificate.renamed == ()                                             # the original stays as published
    assert certificate_source(renamed, False) != certificate_source(certificate, False)      # another text: another check


def test_a_certificate_without_a_tabled_name_gets_no_renamed_copy():
    certificate = Certificate("t", TRUE_SIDE, GOEDEL, "theorem t : 1 = 1 := by\n", "  rw [le_div_iff₀ hc]\n  norm_num")
    assert renamed_certificate(certificate, TABLE) is None


def test_the_wrapper_of_a_negation_keeps_the_statements_type():
    # The variables-only wrapper restates the statement's type on its first line; a rename never reaches it.
    statement = "theorem t (x : ℝ) : le_div_iff = x := by\n"          # (not Lean that means anything: the type holds the token)
    theorem, proof = negation_certificate(statement, "  intro h\n  exact absurd (le_div_iff hc) h", "negation_of_t")
    certificate = Certificate("t", FALSE_SIDE, INTERNLM_ROWS, theorem, proof)
    renamed = renamed_certificate(certificate, TABLE)
    assert renamed.theorem == theorem
    lines = renamed.proof.split("\n")
    assert lines[0] == "  have published_for_every_value (x : ℝ) : ¬ (le_div_iff = x) := by"       # untouched
    assert lines[1:3] == ["    intro h", "    exact absurd (le_div_iff₀ hc) h"]
    assert lines[3:] == proof.split("\n")[3:]


def test_a_reported_name_may_be_written_without_an_opened_namespace():
    namespaces = opened_namespaces()
    assert {"Real", "Nat"} <= set(namespaces)
    assert written_forms("Real.sqrt_eq_iff_sq_eq", namespaces) == ("Real.sqrt_eq_iff_sq_eq", "sqrt_eq_iff_sq_eq")
    assert written_forms("le_div_iff", namespaces) == ("le_div_iff",)
    assert written_forms("Complex.abs_apply", namespaces) == ("Complex.abs_apply",)        # Complex is not opened


def test_the_configs_table_names_distinct_whole_identifiers():
    table = yaml.safe_load(CONFIG.read_text())["ladder_loop"]["certificates"]["renames"]
    assert table["div_le_div_iff"] == "div_le_div_iff₀" and table["le_div_iff"] == "le_div_iff₀" and table["div_le_iff"] == "div_le_iff₀"
    assert table["Nat.dvd_sub'"] == "Nat.dvd_sub"
    for old, new in table.items():
        assert _names(old) == [old] and _names(new) == [new] and old != new      # each side is ONE identifier
        assert new not in table                                                  # no chain: a new name is never itself renamed
        assert rename_library_names(f"  exact {old} h", table) == (f"  exact {new} h", (old,))
