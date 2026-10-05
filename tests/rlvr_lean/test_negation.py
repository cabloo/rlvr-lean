"""Spec §13b, the census: the negation of a conjecture is `¬ (∀ binders, statement)`, built from its own signature."""

import pytest

from rlvr_lean.domain.conjecturing.negation import build_negation_exactness_source, negate_statement, negation_name, split_signature
from rlvr_lean.domain.verification.lean_source import LEAN_HEADER, theorem_name_of


def test_binders_move_under_a_universal_quantifier_and_the_whole_claim_is_negated():
    statement = "theorem conjecture_ab12 (x : ℝ) (hx : 0 < x) : ∃ y : ℝ, y^2 = x := by\n"
    assert negate_statement(statement, "negated_conjecture_ab12") == (
        "theorem negated_conjecture_ab12 : ¬ (∀ (x : ℝ) (hx : 0 < x), ∃ y : ℝ, y^2 = x) := by\n")


def test_a_statement_without_binders_is_negated_as_it_stands():
    assert negate_statement("theorem t : (Real.sqrt 2 + 1) * (Real.sqrt 2 - 1) = 1 := by\n", "n") == (
        "theorem n : ¬ ((Real.sqrt 2 + 1) * (Real.sqrt 2 - 1) = 1) := by\n")


def test_an_already_negated_statement_gets_a_second_negation():
    assert negate_statement("theorem t : ¬(∀ (x : ℝ), x = 0) := by\n", "n") == "theorem n : ¬ (¬(∀ (x : ℝ), x = 0)) := by\n"


def test_every_binder_form_is_copied_as_written():
    statement = "theorem t {α : Type*} [Fintype α] ⦃s : Finset α⦄ (h : s.card = 0) x : s = ∅ ∨ x = x := by\n"
    assert split_signature(statement) == ("t", "{α : Type*} [Fintype α] ⦃s : Finset α⦄ (h : s.card = 0) x", "s = ∅ ∨ x = x")
    assert negate_statement(statement, "n") == "theorem n : ¬ (∀ {α : Type*} [Fintype α] ⦃s : Finset α⦄ (h : s.card = 0) x, s = ∅ ∨ x = x) := by\n"


def test_the_type_colon_is_the_first_one_outside_every_bracket():
    statement = "theorem t (f : ℕ → ℕ) (h : ∀ n : ℕ, f n = n) : ∃ g : ℕ → ℕ, ∀ n : ℕ, g n = f n := by\n"
    name, binders, statement_type = split_signature(statement)
    assert (name, binders) == ("t", "(f : ℕ → ℕ) (h : ∀ n : ℕ, f n = n)")
    assert statement_type == "∃ g : ℕ → ℕ, ∀ n : ℕ, g n = f n"


def test_a_statement_over_several_lines_keeps_its_text():
    statement = "theorem t (m n : ℕ) :\n  ∃ x : ℕ, (m = 10 ∧ n = 2) ∨ (m = 2 ∧ n = 10) ↔ x = 6 := by\n"
    assert negate_statement(statement, "n") == "theorem n : ¬ (∀ (m n : ℕ), ∃ x : ℕ, (m = 10 ∧ n = 2) ∨ (m = 2 ∧ n = 10) ↔ x = 6) := by\n"


def test_an_assignment_inside_brackets_is_not_the_statement_s_end():
    statement = "theorem t (x : ℕ) : (let y := x + 1; y) = x + 1 := by\n"
    assert split_signature(statement)[2] == "(let y := x + 1; y) = x + 1"


def test_the_negation_is_a_theorem_the_verifier_can_name():
    negated = negate_statement("theorem conjecture_ab12 (x : ℝ) : x = x := by\n", negation_name("ab12"))
    assert negation_name("ab12") == "negated_conjecture_ab12"
    assert theorem_name_of(negated) == "negated_conjecture_ab12" and negated.endswith(":= by\n")


@pytest.mark.parametrize("statement", [
    "def t (x : ℝ) : ℝ := x\n",                      # not a theorem
    "theorem t (x : ℝ) : x = x\n",                   # no `:= by`
    "theorem t (x : ℝ) : x = x := trivial\n",        # not a tactic proof
    "theorem t (x : ℝ) := by\n",                     # no type
    "theorem (x : ℝ) : x = x := by\n",               # no name
])
def test_a_statement_that_is_not_a_theorem_with_a_type_and_a_tactic_proof_is_refused(statement):
    with pytest.raises(ValueError):
        negate_statement(statement, "n")


def test_the_new_name_must_be_a_lean_identifier():
    with pytest.raises(ValueError):
        negate_statement("theorem t : 1 = 1 := by\n", "not a name")


def test_the_exactness_file_states_the_built_negation_equal_to_the_negated_type_of_the_conjecture():
    statement = "theorem conjecture_ab12 (h : 0 < x) : (-x : ℝ) < 0 := by\n"
    source = build_negation_exactness_source(statement)
    assert source == (LEAN_HEADER + statement + "  sorry\n\n"
                      "example : (¬ (∀ (h : 0 < x), (-x : ℝ) < 0)) = (¬ (type_of% @conjecture_ab12)) := rfl\n")
