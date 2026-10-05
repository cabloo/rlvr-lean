"""The one notation rewrite between Lean pins: `∑ x in s, f x` to `∑ x ∈ s, f x` (spec ladder-loop, decided
2026-10-04; o2a-version-tax-RESULT.md). Pure."""

from rlvr_lean.domain.verification import lean_pin
from rlvr_lean.domain.verification.notation import membership_binders


def test_a_sum_over_a_set_is_rewritten_and_counted():
    assert membership_binders("∑ i in Finset.range n, f i") == ("∑ i ∈ Finset.range n, f i", 1)
    assert membership_binders("∏ k in s, (1 + a k)") == ("∏ k ∈ s, (1 + a k)", 1)


def test_text_without_the_old_form_is_returned_unchanged():
    for text in ("∑ i ∈ Finset.range n, f i", "∑ i, f i", "∑ i : Fin n, f i", "theorem t (x : ℝ) : x = x := by", "∀ᶠ x in atTop, f x < 1", ""):
        assert membership_binders(text) == (text, 0)


def test_every_operator_is_rewritten_nested_ones_too():
    text = "theorem t (n : ℕ) : ∑ i in Finset.range n, ∑ j in Finset.range i, (i + j) = ∏ k in s, k := by\n"
    rewritten, places = membership_binders(text)
    assert places == 3
    assert rewritten == "theorem t (n : ℕ) : ∑ i ∈ Finset.range n, ∑ j ∈ Finset.range i, (i + j) = ∏ k ∈ s, k := by\n"


def test_only_the_binders_own_in_is_touched():
    # An `in` inside the set's brackets, after the comma, or after another operator's binder list is not this operator's.
    assert membership_binders("∑ i in (Finset.filter (fun x => x in_range) s), f i")[1] == 1
    assert membership_binders("∑ i, (∑ j in t, g j)") == ("∑ i, (∑ j ∈ t, g j)", 1)
    assert membership_binders("(∑ i ∈ s, f i) + (∑ j in t, g j)") == ("(∑ i ∈ s, f i) + (∑ j ∈ t, g j)", 1)
    assert membership_binders("∑ x, f x in y") == ("∑ x, f x in y", 0)


def test_a_binder_split_over_lines_is_rewritten():
    assert membership_binders("∑ i\n  in Finset.Icc 1 n, a i") == ("∑ i\n  ∈ Finset.Icc 1 n, a i", 1)


def test_an_identifier_that_starts_or_ends_with_in_is_not_the_word():
    for text in ("∑ index, f index", "∑ i inside, f i", "∑ main, f main"):
        assert membership_binders(text) == (text, 0)


def test_rewriting_twice_changes_nothing_more():
    once, places = membership_binders("∑ i in s, ∏ j in t, f i j")
    assert places == 2 and membership_binders(once) == (once, 0)


def test_only_the_later_pin_rewrites_and_source_never_does():
    old = "theorem t : ∑ i in Finset.range 3, i = 3 := by\n"
    assert lean_pin("v4.9").rewrite(old) == (old, 0)
    assert lean_pin("v4.27").rewrite(old) == ("theorem t : ∑ i ∈ Finset.range 3, i = 3 := by\n", 1)
    # A file as it is SENT is never rewritten: a proof the model wrote is judged as written.
    assert lean_pin("v4.27").source("import Mathlib\n" + old) == "import Mathlib\n" + old
