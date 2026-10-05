"""Spec fixture 6 and §3 items 1-2: statement normalization and deduplication, reading a statement out of a
generated completion, and the canonical `conjecture_<id>` name."""

import re

import pytest

from rlvr_lean.domain.conjecturing.dedup import (
    StatementDeduplicator,
    content_id,
    normalize_statement,
    strip_lean_comments,
)
from rlvr_lean.domain.conjecturing.parsing import canonical_statement, parse_generated_statement, rename_theorem

# miniF2F-valid `mathd_algebra_478`, as DeepSeek's datasets/minif2f.jsonl writes it.
MINIF2F_STATEMENT = (
    "theorem mathd_algebra_478 (b h v : ℝ) (h₀ : 0 < b ∧ 0 < h ∧ 0 < v) (h₁ : v = 1 / 3 * (b * h))\n"
    "    (h₂ : b = 30) (h₃ : h = 13 / 2) : v = 65 := by\n"
)

BASE = "theorem t1 (x : ℝ) (h₀ : x > 2) : x ^ 2 > 4 := by\n"
SAME_STATEMENT_VARIANTS = [
    "theorem t1 (x : ℝ) (h₀ : x > 2) : x ^ 2 > 4 := by\n",
    "theorem renamed_one (x : ℝ) (h₀ : x > 2) : x ^ 2 > 4 := by\n",                    # name
    "lemma conjecture_0123abcd (x : ℝ) (h₀ : x > 2) : x ^ 2 > 4 := by\n",               # keyword and name
    "theorem t1  (x : ℝ)\n    (h₀ : x > 2) :\n\tx ^ 2 > 4   :=   by\n",                # whitespace
    "theorem t1 (x : ℝ) -- a real\n  (h₀ : x > 2) /- block -/ : x ^ 2 > 4 := by\n",     # comments
    "/-- The square exceeds 4. /- nested -/ still the docstring -/\ntheorem t1 (x : ℝ) (h₀ : x > 2) : x ^ 2 > 4 := by\n",
    "theorem t1 (x : ℝ) (h₀ : x > 2) : x ^ 2 > 4 := by sorry",                          # proof stubs
    "theorem t1 (x : ℝ) (h₀ : x > 2) : x ^ 2 > 4 := by\n  sorry\n",
    "theorem t1 (x : ℝ) (h₀ : x > 2) : x ^ 2 > 4 :=",
    "theorem t1 (x : ℝ) (h₀ : x > 2) : x ^ 2 > 4",
    "@[simp] private theorem t1 (x : ℝ) (h₀ : x > 2) : x ^ 2 > 4 := by\n",              # attribute, modifier
    "theorem t1 (x : ℝ) (h₀ : x > 2) : x ^ 2 > 4 := by\n",
]


@pytest.mark.parametrize("variant", SAME_STATEMENT_VARIANTS)
def test_name_whitespace_comment_and_stub_variants_normalize_equal(variant):
    assert normalize_statement(variant) == normalize_statement(BASE) == "(x : ℝ) (h₀ : x > 2) : x ^ 2 > 4"
    assert content_id(variant) == content_id(BASE)


def test_unicode_is_nfc_normalized():
    composed, decomposed = "theorem t (é : ℕ) : é = é := by", "theorem t (é : ℕ) : é = é := by"
    assert composed != decomposed and normalize_statement(composed) == normalize_statement(decomposed)


@pytest.mark.parametrize("different", [
    "theorem t1 (x : ℝ) (h₀ : x > 2) : x ^ 2 > 3 := by\n",       # another constant
    "theorem t1 (x : ℕ) (h₀ : x > 2) : x ^ 2 > 4 := by\n",       # another type
    "theorem t1 (x : ℝ) (h₀ : x ≥ 2) : x ^ 2 > 4 := by\n",
])
def test_different_statements_stay_different(different):
    assert normalize_statement(different) != normalize_statement(BASE)
    assert content_id(different) != content_id(BASE)


def test_content_id_is_sixteen_hex_digits():
    assert re.fullmatch(r"[0-9a-f]{16}", content_id(BASE))


def test_comment_stripping_keeps_every_code_position():
    text = "theorem a /- x\n /- y -/ -/ (n : ℕ) -- tail := 1\n : n = n"
    stripped = strip_lean_comments(text)
    assert len(stripped) == len(text) and stripped.count("\n") == text.count("\n")
    assert stripped.split() == ["theorem", "a", "(n", ":", "ℕ)", ":", "n", "=", "n"]
    assert [index for index, character in enumerate(text) if character == "("] == [stripped.index("(")]


def test_deduplicator_rejects_a_statement_equal_to_a_minif2f_statement():
    deduplicator = StatementDeduplicator(seen_statements=[MINIF2F_STATEMENT])
    as_generated = ("theorem conjecture_5f3a (b h v : ℝ) (h₀ : 0 < b ∧ 0 < h ∧ 0 < v) "
                    "(h₁ : v = 1 / 3 * (b * h)) (h₂ : b = 30) (h₃ : h = 13 / 2) : v = 65 := by\n")
    assert deduplicator.add(as_generated) is False
    assert deduplicator.seen_count == 1


def test_deduplicator_records_new_statements_and_rejects_their_variants():
    deduplicator = StatementDeduplicator()
    assert deduplicator.add(BASE) is True
    assert all(deduplicator.add(variant) is False for variant in SAME_STATEMENT_VARIANTS)
    assert deduplicator.add("theorem t2 (x : ℝ) (h₀ : x > 3) : x ^ 2 > 9 := by\n") is True
    assert deduplicator.seen_count == 2


def test_deduplicator_uses_the_type_fingerprint_when_given():
    deduplicator = StatementDeduplicator(seen_fingerprints=[1234])
    renamed_binders = "theorem f2 (y : ℝ) (hy : y > 2) : y ^ 2 > 4 := by\n"   # a string-new restatement
    assert deduplicator.add(renamed_binders, type_fingerprint=1234) is False
    assert deduplicator.add(renamed_binders) is True                          # no fingerprint: string key only
    assert deduplicator.add("theorem f3 (z : ℝ) (hz : z > 2) : z ^ 2 > 4 := by", type_fingerprint=99) is True
    assert deduplicator.add("theorem f4 (w : ℝ) (hw : w > 2) : w ^ 2 > 4 := by", type_fingerprint=99) is False


# --- parsing a generated statement ----------------------------------------------------------------------

@pytest.mark.parametrize("generated,expected", [
    ("lean_workbook_conj (x : ℝ) (h₀ : x > 2) : x ^ 2 > 4 := by\n  nlinarith",
     "theorem lean_workbook_conj (x : ℝ) (h₀ : x > 2) : x ^ 2 > 4 := by\n"),
    ("foo_α₁' {n : ℕ} [Fact (1 < n)]\n    (h : n % 2 = 1) :\n    n ≠ 2 :=",
     "theorem foo_α₁' {n : ℕ} [Fact (1 < n)]\n    (h : n % 2 = 1) :\n    n ≠ 2 := by\n"),
    # a `:=` inside a binder is not the statement's own
    ("with_let (h : let y := 2; y = 2) : True := by trivial", "theorem with_let (h : let y := 2; y = 2) : True := by\n"),
    # comments are removed, including one that hides a `:=`
    ("commented (x : ℕ) -- note: x := 1 here\n  : x = x := by rfl", "theorem commented (x : ℕ)\n  : x = x := by\n"),
    ("Nat.my_conj: ∀ n : ℕ, n + 0 = n := by simp", "theorem Nat.my_conj: ∀ n : ℕ, n + 0 = n := by\n"),
    ("  leading_space : 1 = 1 := by", "theorem leading_space : 1 = 1 := by\n"),
])
def test_parse_accepts_representative_generations(generated, expected):
    assert parse_generated_statement(generated) == expected


def test_parse_takes_the_whole_text_when_generation_stopped_at_the_assignment():
    generated = "stopped (a b : ℕ) (h : a < b) : a + 1 ≤ b "
    assert parse_generated_statement(generated) is None
    assert parse_generated_statement(generated, stopped_at_assignment=True) == \
        "theorem stopped (a b : ℕ) (h : a < b) : a + 1 ≤ b := by\n"


@pytest.mark.parametrize("generated", [
    "no_assignment (x : ℕ) : x = x",                      # never reached `:=` (accepted only if stopped there)
    "no_type (x : ℕ) (h : x = 1) := by simp",              # no top-level `:` type separator
    "only_colon_no_type : := by",
    "123bad : 1 = 1 := by",                                # not an identifier
    "(x : ℕ) : x = x := by",                               # no name at all
    ":= by trivial",                                       # empty body
    "unbalanced (x : ℕ : x = x := by",                     # `:=` sits inside an open bracket
    "closes_early (x : ℕ)) : x = x := by",
    "/- runs off the end := by",                           # unterminated comment hides everything
    "too_long : " + " ∧ ".join(["1 = 1"] * 400) + " := by",
])
def test_parse_rejects_unusable_generations(generated):
    assert parse_generated_statement(generated) is None
    if not generated.startswith("no_assignment"):
        # The same text cut by a `:=` stop sequence is just as unusable.
        assert parse_generated_statement(generated.split(":=")[0], stopped_at_assignment=True) is None


def test_parse_length_limit_is_on_the_body():
    body = "edge : " + "1" * (2_000 - len("edge : "))
    assert parse_generated_statement(body + " := by") == f"theorem {body} := by\n"
    assert parse_generated_statement(body + "1 := by") is None


# --- renaming and the canonical form --------------------------------------------------------------------

def test_rename_replaces_only_the_declared_name():
    assert rename_theorem("theorem foo (foo : ℕ) : foo = foo := by\n", "bar") == "theorem bar (foo : ℕ) : foo = foo := by\n"
    assert rename_theorem("/-- theorem fake -/\n@[simp] lemma real_name : True := by", "x.y") == \
        "/-- theorem fake -/\n@[simp] lemma x.y : True := by"
    with pytest.raises(ValueError):
        rename_theorem("example : True := by trivial", "bar")
    with pytest.raises(ValueError):
        rename_theorem("theorem foo : True := by", "not an identifier")


def test_canonical_statement_names_the_conjecture_by_its_content():
    conjecture_id, renamed = canonical_statement(BASE)
    assert conjecture_id == content_id(BASE)
    assert renamed == f"theorem conjecture_{conjecture_id} (x : ℝ) (h₀ : x > 2) : x ^ 2 > 4 := by\n"
    assert canonical_statement(renamed) == (conjecture_id, renamed)          # idempotent
    assert canonical_statement("theorem other_name (x : ℝ) (h₀ : x > 2) : x ^ 2 > 4 := by\n") == (conjecture_id, renamed)


def test_generated_text_to_deduplicated_conjecture():
    deduplicator = StatementDeduplicator(seen_statements=[MINIF2F_STATEMENT])
    generations = [
        "gen_a (x : ℝ) (h₀ : x > 2) : x ^ 2 > 4 := by nlinarith",
        "gen_b (x : ℝ)  (h₀ : x > 2) : x ^ 2 > 4 :=",                         # duplicate of gen_a
        "gen_c (b h v : ℝ) (h₀ : 0 < b ∧ 0 < h ∧ 0 < v) (h₁ : v = 1 / 3 * (b * h)) (h₂ : b = 30) "
        "(h₃ : h = 13 / 2) : v = 65 := by",                                   # equals the miniF2F statement
        "gen_d broken",
    ]
    kept = []
    for generated in generations:
        statement = parse_generated_statement(generated)
        if statement is not None and deduplicator.add(statement):
            kept.append(canonical_statement(statement))
    assert [renamed.split()[1] for _, renamed in kept] == [f"conjecture_{content_id(BASE)}"]
