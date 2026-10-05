"""Spec fixtures for the verification context (the first experiment's spec §4,
Fixtures item 7): one recorded server response per status, the lexical filter, and the Lean source."""

import pytest

from rlvr_lean.domain.verification import (
    LEAN_HEADER,
    VerificationStatus,
    build_proof_source,
    classify_check_result,
    find_forbidden_token,
    rejected_lexically,
    theorem_name_of,
)

AXIOM_REPORT = {"severity": "info", "data": "'demo' depends on axioms: [propext, Quot.sound, Classical.choice]"}
NO_AXIOMS_REPORT = {"severity": "info", "data": "'demo' does not depend on any axioms"}


def response(messages, sorries=()):
    return {"id": "a", "time": 1.5, "response": {"env": 3, "messages": list(messages), "sorries": list(sorries)},
            "diagnostics": {"repl_uuid": "x", "cpu_max": 99.0, "memory_max": 2_500_000_000}}


def status_of(check_result):
    return classify_check_result("a", check_result).status


def test_verified_with_standard_axioms_and_with_none():
    result = classify_check_result("a", response([AXIOM_REPORT]))
    assert result.status is VerificationStatus.VERIFIED and result.is_verified
    assert set(result.axioms) == {"propext", "Quot.sound", "Classical.choice"}
    assert result.verification_seconds == 1.5 and result.peak_memory_bytes == 2_500_000_000
    assert status_of(response([NO_AXIOMS_REPORT])) is VerificationStatus.VERIFIED


def test_lean_error_message_is_lean_error_even_with_a_clean_axiom_report():
    messages = [{"severity": "error", "data": "unsolved goals"}, AXIOM_REPORT]
    assert status_of(response(messages)) is VerificationStatus.LEAN_ERROR


def test_warning_containing_failed_is_lean_error():
    messages = [{"severity": "warning", "data": "aesop: failed to prove the goal"}, AXIOM_REPORT]
    assert status_of(response(messages)) is VerificationStatus.LEAN_ERROR


def test_the_failed_warning_rule_can_be_switched_off_for_the_v4_27_pipeline():
    """The OEIS Open spec, O1: a recovered tactic failure warns while the proof is complete (one gold proof)."""
    messages = [{"severity": "warning", "data": "aesop: failed to prove the goal after exhaustive search."}, AXIOM_REPORT]
    assert classify_check_result("a", response(messages), failed_warning_is_error=False).status is VerificationStatus.VERIFIED
    errors = [{"severity": "error", "data": "unsolved goals"}, AXIOM_REPORT]
    assert classify_check_result("a", response(errors), failed_warning_is_error=False).status is VerificationStatus.LEAN_ERROR


def test_repl_level_error_is_lean_error():
    assert status_of({"id": "a", "time": 0.1, "response": {"message": "Could not parse as a valid JSON command"}}) \
        is VerificationStatus.LEAN_ERROR


def test_sorry_is_caught_by_the_sorries_array_or_by_the_warning_alone():
    with_array = response([{"severity": "warning", "data": "declaration uses 'sorry'"}], sorries=[{"goal": "⊢ False"}])
    assert status_of(with_array) is VerificationStatus.USES_SORRY
    # `by admit` on Kimina: the warning is present but the `sorries` array is empty (upstream issue #75).
    warning_only = response([{"severity": "warning", "data": "declaration uses 'sorry'"},
                             {"severity": "info", "data": "'demo' depends on axioms: [sorryAx]"}])
    assert status_of(warning_only) is VerificationStatus.USES_SORRY


def test_axiom_outside_the_whitelist_is_forbidden():
    for axioms in ("[sorryAx]", "[propext, my_axiom]", "[Lean.ofReduceBool]"):
        result = classify_check_result("a", response([{"severity": "info", "data": f"'demo' depends on axioms: {axioms}"}]))
        assert result.status is VerificationStatus.FORBIDDEN_AXIOM and not result.is_verified


def test_missing_axiom_report_does_not_count_as_verified():
    assert status_of(response([])) is VerificationStatus.FORBIDDEN_AXIOM


def test_timeout_is_distinct_from_server_error():
    assert status_of({"id": "a", "time": 60, "error": "Lean REPL command timed out in 60 seconds"}) is VerificationStatus.TIMEOUT
    assert status_of({"id": "a", "time": 60, "error": "Lean REPL header command timed out in 60 seconds"}) \
        is VerificationStatus.TIMEOUT
    assert status_of({"id": "a", "time": None, "error": "server_error: no answer after 4 attempts: ReadTimeout('timed out')"}) \
        is VerificationStatus.SERVER_ERROR
    assert status_of({"id": "a", "time": 0.2, "error": "JSON decode error"}) is VerificationStatus.SERVER_ERROR
    assert status_of({"id": "a", "time": 0.2}) is VerificationStatus.SERVER_ERROR


@pytest.mark.parametrize("completion,token", [
    ("  sorry", "sorry"), ("  exact?\n  admit", "admit"), ("  nlinarith\naxiom cheat : False", "axiom"),
    ("  norm_num\nimport Lean", "import"), ("  native_decide", "native_decide"),
    ('  simp\n#eval IO.println "x"', "#eval"), ("  run_cmd Lean.logInfo \"x\"", "run_cmd"),
    ("  exact IO.FS.readFile x", "IO."), ("  unsafe foo", "unsafe"),
    ("  run_tac do pure ()", "run_tac"), ("  run_elab pure ()", "run_elab"), ("  run_meta pure ()", "run_meta"),
    ("elab \"pwn\" : tactic => pure ()", "elab"), ("macro \"x\" : term => `(1)", "macro"),
    ("syntax \"x\" : term", "syntax"), ("initialize foo : IO.Ref Nat ← IO.mkRef 0", "initialize"),
    # A generated conjecture STATEMENT goes through the same filter before its compile check.
    ("theorem t (h : (by run_tac pure (); exact True)) : True := by", "run_tac"),
])
def test_lexical_filter_finds_forbidden_tokens(completion, token):
    assert find_forbidden_token(completion) == token
    assert rejected_lexically("a", token).status is VerificationStatus.REJECTED_LEXICAL


@pytest.mark.parametrize("completion", [
    "  nlinarith [sq_nonneg (a - b), sq_nonneg (a + b)]",
    "  exact sorry_free_lemma h",          # `sorry` inside an identifier is not the token
    "  simp [admits_bound, important]",    # nor `admit`/`import` inside identifiers
    "  have ratio := h.axiomatic'\n  linarith",
    "  decide",
])
def test_lexical_filter_passes_ordinary_proofs(completion):
    assert find_forbidden_token(completion) is None


def test_theorem_name_and_source_assembly():
    statement = "theorem mathd_algebra_1 (x : ℝ) (h₀ : x + 1 = 2) : x = 1 := by\n"
    assert theorem_name_of(statement) == "mathd_algebra_1"
    assert theorem_name_of("/-- doc -/\nlemma foo_bar' {n : ℕ} : n = n := by") == "foo_bar'"
    source = build_proof_source(statement, "  linarith\n\n")
    assert source == LEAN_HEADER + statement + "  linarith\n\n#print axioms mathd_algebra_1\n"
    # The two import lines must come first and unchanged: the server reuses a worker per import header.
    assert source.startswith("import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n")
    with pytest.raises(ValueError):
        theorem_name_of("example : 1 = 1 := by")


@pytest.mark.parametrize("lean_file,name", [
    # Shapes that the first live pin-gate run got wrong: prose about "the theorem" in comments.
    ("/-- Show that the theorem is true. -/\ntheorem real_name (x : ℕ) : x = x := by\n  rfl", "real_name"),
    ("theorem main_result : True := by\n  /- the lemma states that ... -/\n  trivial", "main_result"),
    ("-- this theorem which we prove\ntheorem t1 : True := by trivial", "t1"),
    ("/- outer /- nested theorem axioms -/ still comment -/\nlemma l2 : True := by trivial", "l2"),
    # A helper lemma before the main theorem: the LAST declaration is the one reported on.
    ("lemma helper : 1 = 1 := rfl\n\n@[simp] theorem main' : 2 = 2 := by rfl", "main'"),
    ("private theorem hidden : True := by trivial", "hidden"),
])
def test_theorem_name_ignores_comments_and_takes_the_last_declaration(lean_file, name):
    assert theorem_name_of(lean_file) == name
