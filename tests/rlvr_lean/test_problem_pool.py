"""The pool of problems with a published answer: certificates as Lean text, when a problem is settled, what is
held out. Spec: docs/spec/ladder-loop.spec.md (fixtures 2, 3, 6, 7 and 10 as far as L0's data half goes).
Pure: no Lean, no network."""

import pytest

from rlvr_lean.domain.conjecturing.dedup import normalize_statement
from rlvr_lean.domain.problem_pool import (
    CERTIFICATE_CHECK,
    EXACTNESS_CHECK,
    FALSE_SIDE,
    GOEDEL,
    IN,
    INTERNLM_ROWS,
    LEAN_WORKBOOK,
    OPEN,
    OUT,
    STP,
    STP_CONJECTURE,
    TRUE_SIDE,
    Candidate,
    Certificate,
    SoundnessAlarm,
    certificate_source,
    SLICE_ALL,
    SLICE_PRESENT,
    SLICE_STP,
    SLICE_WORKBOOK,
    draw_base_map,
    draw_heldout,
    more_candidates_needed,
    pool_apart_from,
    pool_row,
    raise_on_both_sides,
    settle,
    slice_parts,
    steps,
)
from rlvr_lean.domain.problem_pool.certificates import (
    HAS_HYPOTHESES,
    NO_HYPOTHESES,
    UNPARSED,
    block_after_by,
    chain_order,
    hypotheses_reading,
    negation_certificate,
    proof_from_rows,
    published_file_proof,
    stp_proof,
    stp_statement,
    tactic_block,
    workbook_statement,
)
from rlvr_lean.domain.problem_pool.certificates import STP_HEADER, STP_PROMPT_HEAD
from rlvr_lean.domain.problem_pool.selection import NEGATION_NOT_EXACT, NO_CERTIFICATE_VERIFIES
from rlvr_lean.domain.verification import LEAN_HEADER

STATEMENT = "theorem lean_workbook_7 (x : ℝ) (h : 0 < x) : x ^ 2 > 0 := by\n"


# ------------------------------------------------------------------------------- published proofs as Lean text
def test_a_workbook_row_becomes_a_statement_and_a_non_theorem_is_refused():
    assert workbook_statement("theorem t (x : ℝ) : x = x  :=  by sorry") == "theorem t (x : ℝ) : x = x := by\n"
    assert workbook_statement("theorem t :\n  1 = 1 := by sorry\n") == "theorem t :\n  1 = 1 := by\n"
    assert workbook_statement("def f (x : ℕ) : ℕ := by sorry") is None


def test_a_tactic_script_is_indented_as_a_block():
    assert tactic_block("have h : 0 ≤ x := by\n  positivity\nnlinarith [h]") == "  have h : 0 ≤ x := by\n    positivity\n  nlinarith [h]"


def _row(tactic, before, after):
    return {"tactic": tactic, "state_before": before, "state_after": after}


def test_tactic_rows_are_ordered_by_their_goals_not_by_the_file():
    rows = [_row("nlinarith", "S2", "no goals"), _row("intro x", "S0", "S1"), _row("norm_num", "S1", "S2")]
    assert [row["tactic"] for row in chain_order(rows)] == ["intro x", "norm_num", "nlinarith"]
    assert proof_from_rows(rows) == "  intro x\n  norm_num\n  nlinarith"


def test_rows_that_are_not_one_finished_chain_give_no_proof():
    assert proof_from_rows([_row("intro x", "S0", "S1"), _row("simp", "S1", "S2")]) is None                     # never reaches "no goals"
    assert proof_from_rows([_row("simp", "S0", "S1"), _row("ring", "S0", "no goals")]) is None                  # two roots: a branch
    assert proof_from_rows([_row("simp", "S0", "S1"), _row("ring", "S1", "no goals"), _row("omega", "S1", "no goals")]) is None
    assert proof_from_rows([_row("ring", "S0", "no goals"), _row("simp", "S5", "S6")]) is None                  # a stray row


def test_a_multi_line_tactic_keeps_its_own_shape_inside_the_block():
    rows = [_row("constructor\n· simp\n· ring", "S0", "no goals")]
    assert proof_from_rows(rows) == "  constructor\n  · simp\n  · ring"


GOEDEL_FILE = (LEAN_HEADER + "/- Prove that x^2 > 0. We use the axiom of choice, sorry for the pun. -/\n"
               "theorem lean_workbook_7 (x : ℝ) (h : 0 < x) : x ^ 2 > 0  := by\n"
               "  /-\n  A paragraph of prose, with := and by in it.\n  -/\n"
               "  -- square is positive\n"
               "  positivity\n")


def test_a_published_file_gives_its_statement_and_its_proof_without_comments():
    statement, proof = published_file_proof(GOEDEL_FILE, "lean_workbook_7")
    assert normalize_statement(statement) == normalize_statement(STATEMENT)
    assert proof == "  positivity"            # the comments are gone, and so are the blank lines they left


def test_a_published_file_for_another_theorem_or_without_a_tactic_proof_is_refused():
    assert published_file_proof(GOEDEL_FILE, "lean_workbook_70") is None
    assert published_file_proof(GOEDEL_FILE, "lean_workbook") is None            # a prefix of the name is not the name
    assert published_file_proof(LEAN_HEADER + "theorem t : 1 = 1 := rfl\n", "t") is None
    assert published_file_proof(LEAN_HEADER + "theorem t : 1 = 1 := by\n  -- nothing\n", "t") is None


def test_a_proof_that_continues_the_by_line_is_moved_under_it():
    assert block_after_by(" simp\n  <;> linarith") == "  simp\n  <;> linarith"
    assert block_after_by(" nlinarith [h]") == "  nlinarith [h]"
    assert block_after_by("\n  intro x\n\n  simp\n") == "  intro x\n  simp"
    assert block_after_by("simp\nlinarith") == "  simp\n  linarith"


def _prompt(body):
    return STP_PROMPT_HEAD + STP_HEADER + body


def test_an_stp_prompt_gives_its_statement():
    assert stp_statement(_prompt("theorem t (x : ℕ) : x + 0 = x:= by")) == "theorem t (x : ℕ) : x + 0 = x := by\n"
    assert stp_statement(_prompt("-- a remark\ntheorem t (x : ℕ) :\n  x + 0 = x   :=  by")) == "theorem t (x : ℕ) :\n  x + 0 = x := by\n"


def test_an_stp_prompt_that_is_not_one_theorem_stopping_at_by_is_refused():
    assert stp_statement(_prompt("theorem t (G : Type*) [Group G] :\n  IsOpen A :=")) is None                 # cut at `:=`
    assert stp_statement(_prompt("def f (x : ℕ) : ℕ := by")) is None
    assert stp_statement(_prompt("theorem a : 1 = 1 := by\n  rfl\ntheorem b : 2 = 2 := by")) is None
    assert stp_statement(STP_PROMPT_HEAD + "/-\nCopyright (c) 2017\n-/\nimport Mathlib\ntheorem t : 1 = 1 := by") is None


def test_an_stp_target_is_the_proof_without_comments():
    assert stp_proof("\n  intro x -- name it\n  simp\n") == "  intro x\n  simp"
    assert stp_proof(" simp") == "  simp"
    assert stp_proof("\n  -- nothing here\n") is None


# ------------------------------------------------------------------------------------------- the exact negation
def test_hypotheses_are_read_from_the_statement_and_from_the_first_row():
    assert hypotheses_reading("theorem t : ∀ a b : ℝ, a + b = 2 := by\n", "⊢ ¬∀ (a b : ℝ), a + b = 2") == NO_HYPOTHESES
    assert hypotheses_reading("theorem t (x y : ℝ) : x + y = 2 := by\n", "x y : ℝ\n⊢ ¬x + y = 2") == NO_HYPOTHESES
    assert hypotheses_reading(STATEMENT, "x : ℝ\nh : 0 < x\n⊢ ¬x ^ 2 > 0") == HAS_HYPOTHESES
    # Either reading alone is enough to keep a problem out.
    assert hypotheses_reading("theorem t (x : ℝ) : x = 1 := by\n", "x : ℝ\nh : 0 < x\n⊢ ¬x = 1") == HAS_HYPOTHESES
    assert hypotheses_reading("theorem t ∀ x > 0, x = 1 := by\n", "⊢ ¬∀ x > 0, x = 1") == UNPARSED


def test_a_statement_without_binders_is_negated_by_the_published_proof_itself():
    theorem, proof = negation_certificate("theorem t : ∀ a : ℝ, a > 0 := by\n", "  push_neg\n  exact ⟨0, by norm_num⟩", "negation_of_t")
    assert theorem == "theorem negation_of_t : ¬ (∀ a : ℝ, a > 0) := by\n"
    assert proof == "  push_neg\n  exact ⟨0, by norm_num⟩"


def test_variables_without_hypotheses_are_wrapped_and_instantiated_at_default():
    theorem, proof = negation_certificate("theorem t (x y : ℝ) (n : ℕ) : x + y = n + 1 := by\n", "  intro h\n  linarith", "negation_of_t")
    assert theorem == "theorem negation_of_t : ¬ (∀ (x y : ℝ) (n : ℕ), x + y = n + 1) := by\n"
    assert proof == ("  have published_for_every_value (x y : ℝ) (n : ℕ) : ¬ (x + y = n + 1) := by\n"
                     "    intro h\n    linarith\n"
                     "  intro claimed_for_every_value\n"
                     "  exact published_for_every_value default default default (claimed_for_every_value default default default)")


def test_a_binder_that_is_not_an_explicit_variable_gets_no_wrapper():
    assert negation_certificate("theorem t {x : ℝ} : x = 1 := by\n", "  simp", "negation_of_t") is None
    assert negation_certificate("theorem t (α : Type) [Inhabited α] (a : α) : a = a := by\n", "  simp", "negation_of_t") is None


# ------------------------------------------------------------------------------------- the checks of a candidate
def _certificate(source, proof="  positivity", side=TRUE_SIDE, problem="lean_workbook_7", theorem=STATEMENT):
    return Certificate(problem, side, source, theorem, proof)


def _candidate(certificates, side=TRUE_SIDE, problem="lean_workbook_7", statement=STATEMENT, kind=LEAN_WORKBOOK, group="g7"):
    return Candidate(problem, kind, side, statement, statement, 0, group, tuple(certificates))


def test_a_certificate_file_is_header_theorem_proof_and_the_axiom_report():
    source = certificate_source(_certificate(GOEDEL), with_fingerprint=False)
    assert source == LEAN_HEADER + STATEMENT + "  positivity\n\n#print axioms lean_workbook_7\n"
    assert "rlvr-type-fingerprint" in certificate_source(_certificate(GOEDEL), with_fingerprint=True)


def test_a_known_true_candidate_is_checked_certificate_by_certificate():
    candidate = _candidate([_certificate(GOEDEL), _certificate(STP, "  nlinarith [h]")])
    checks = steps(candidate)
    assert [(check.kind, check.source) for check in checks] == [(CERTIFICATE_CHECK, GOEDEL), (CERTIFICATE_CHECK, STP)]
    assert len({check.sha for check in checks}) == 2 and all("rlvr-type-fingerprint" in check.lean_file for check in checks)


def test_settling_asks_one_check_at_a_time_and_stops_at_the_first_that_verifies():
    candidate = _candidate([_certificate(GOEDEL), _certificate(STP, "  nlinarith [h]")])
    first, second = steps(candidate)
    assert settle(candidate, [first, second], {}).next_check == first
    failed = {first.sha: {"status": "lean_error", "fingerprint": None}}
    assert settle(candidate, [first, second], failed).next_check == second
    verdict = settle(candidate, [first, second], {first.sha: {"status": "verified", "fingerprint": 41}})
    assert (verdict.state, verdict.certificate.source, verdict.fingerprint, verdict.next_check) == (IN, GOEDEL, 41, None)
    verdict = settle(candidate, [first, second], {**failed, second.sha: {"status": "verified", "fingerprint": 42}})
    assert (verdict.state, verdict.certificate.source, verdict.certificate_sha, verdict.checks_answered) == (IN, STP, second.sha, 2)


def test_a_candidate_whose_certificates_all_fail_is_out_and_never_in_the_pool():
    """Fixture 2: no certificate checked under the pin, no problem."""
    candidate = _candidate([_certificate(GOEDEL), _certificate(STP, "  nlinarith [h]")])
    checks = steps(candidate)
    answers = {check.sha: {"status": status} for check, status in zip(checks, ("lean_error", "timeout"))}
    verdict = settle(candidate, checks, answers)
    assert (verdict.state, verdict.reason, verdict.certificate) == (OUT, NO_CERTIFICATE_VERIFIES, None)
    assert settle(candidate, checks, {checks[0].sha: {"status": "uses_sorry"}}).state == OPEN


def test_a_check_that_was_given_up_counts_as_failed():
    candidate = _candidate([_certificate(GOEDEL), _certificate(STP, "  nlinarith [h]")])
    first, second = steps(candidate)
    assert settle(candidate, [first, second], {}, abandoned={first.sha}).next_check == second
    assert settle(candidate, [first, second], {}, abandoned={first.sha, second.sha}).state == OUT


NEGATED = "theorem negation_of_lean_workbook_9 : ¬ (∀ a : ℝ, a > 0) := by\n"


def _false_candidate():
    statement = "theorem lean_workbook_9 : ∀ a : ℝ, a > 0 := by\n"
    certificate = _certificate(INTERNLM_ROWS, "  push_neg\n  exact ⟨0, by norm_num⟩", FALSE_SIDE, "lean_workbook_9", NEGATED)
    return _candidate([certificate], FALSE_SIDE, "lean_workbook_9", statement, group="g9")


def test_a_known_false_candidate_needs_its_negation_shown_exact_first():
    """Fixture 3: a known-false problem whose exact negation cannot be built or has no certificate is not in the pool."""
    candidate = _false_candidate()
    exactness, certificate = steps(candidate)
    assert (exactness.kind, certificate.kind) == (EXACTNESS_CHECK, CERTIFICATE_CHECK)
    assert "type_of% @lean_workbook_9" in exactness.lean_file and "rlvr-type-fingerprint" in exactness.lean_file
    assert "rlvr-type-fingerprint" not in certificate.lean_file and "#print axioms negation_of_lean_workbook_9" in certificate.lean_file
    assert settle(candidate, [exactness, certificate], {}).next_check == exactness
    not_exact = {exactness.sha: {"compiles": False}, certificate.sha: {"status": "verified"}}
    assert settle(candidate, [exactness, certificate], not_exact).reason == NEGATION_NOT_EXACT
    exact = {exactness.sha: {"compiles": True, "fingerprint": 9}}
    assert settle(candidate, [exactness, certificate], exact).next_check == certificate
    verdict = settle(candidate, [exactness, certificate], {**exact, certificate.sha: {"status": "verified", "fingerprint": None}})
    assert (verdict.state, verdict.fingerprint) == (IN, 9)          # the STATEMENT's fingerprint, from the exactness check
    assert settle(candidate, [exactness, certificate], {**exact, certificate.sha: {"status": "lean_error"}}).reason == NO_CERTIFICATE_VERIFIES
    # An exactness check that was given up is never stood in for by a certificate.
    assert settle(candidate, [exactness, certificate], {certificate.sha: {"status": "verified"}}, abandoned={exactness.sha}).state == OUT


def test_a_pool_row_names_its_certificate_and_holds_no_published_proof():
    """Fixture 7: a published proof never appears as a training target. The pool file is what later stages
    read, and no proof text is in it."""
    candidate = _candidate([_certificate(GOEDEL, "  nlinarith [sq_nonneg (x - 1), h]")])
    verdict = settle(candidate, steps(candidate), {steps(candidate)[0].sha: {"status": "verified", "fingerprint": 5}})
    row = pool_row(candidate, verdict)
    assert row["certificate_source"] == GOEDEL and row["certificate_sha"] == steps(candidate)[0].sha and row["type_fingerprint"] == 5
    assert "nlinarith" not in repr(row) and not {"proof", "certificate", "certificates"} & set(row)


# ---------------------------------------------------------------------------------------------- both sides
def _pool(problem, statement, side=TRUE_SIDE, kind=LEAN_WORKBOOK, group=None, fingerprint=None):
    return {"problem_id": problem, "kind": kind, "side": side, "statement": statement, "group": group or problem, "type_fingerprint": fingerprint}


def test_one_statement_verified_both_ways_raises_the_alarm():
    """Fixture 10, at L0: a proof of both sides stops everything."""
    true_row = _pool("a", "theorem a (x : ℕ) : x = x := by\n", fingerprint=1)
    raise_on_both_sides([true_row, _pool("b", "theorem b (y : ℕ) : y + 0 = y := by\n", fingerprint=2)])
    with pytest.raises(SoundnessAlarm):         # the same text under another name
        raise_on_both_sides([true_row, _pool("c", "theorem c (x : ℕ) : x = x := by\n", FALSE_SIDE)])
    with pytest.raises(SoundnessAlarm):         # another text, the same elaborated type
        raise_on_both_sides([true_row, _pool("d", "theorem d (y : ℕ) : y = y := by\n", FALSE_SIDE, fingerprint=1)])
    raise_on_both_sides([true_row, _pool("e", "theorem e (x : ℕ) : x = x := by\n", fingerprint=1)])     # twice true is a copy, not an alarm


# ------------------------------------------------------------------------------------------------ held out
def _rows():
    workbook = [_pool(f"lean_workbook_{n}", f"theorem lean_workbook_{n} (x : ℕ) : x + {n} = {n} + x := by\n", fingerprint=100 + n) for n in range(40)]
    conjectures = [_pool(f"stp_{n:04d}", f"theorem stp_{n:04d} (x : ℝ) : x * {n} = {n} * x := by\n", kind=STP_CONJECTURE, fingerprint=500 + n) for n in range(30)]
    return workbook + conjectures


def test_the_draw_is_a_function_of_the_seed_and_the_rows():
    rows = _rows()
    first = draw_heldout(rows, {"lean_workbook_3"}, [], seed=7, workbook_problems=10, stp_conjectures=5)
    again = draw_heldout(list(reversed(rows)), {"lean_workbook_3"}, [], seed=7, workbook_problems=10, stp_conjectures=5)
    other = draw_heldout(rows, {"lean_workbook_3"}, [], seed=8, workbook_problems=10, stp_conjectures=5)
    assert [row["problem_id"] for row in first.heldout] == [row["problem_id"] for row in again.heldout]
    assert [row["problem_id"] for row in first.heldout] != [row["problem_id"] for row in other.heldout]
    assert first.counts["heldout_lean_workbook"] == 10 and first.counts["heldout_stp_conjecture"] == 5
    assert [row["heldout_part"] for row in first.heldout] == [LEAN_WORKBOOK] * 10 + [STP_CONJECTURE] * 5


def test_the_present_holdout_is_taken_first():
    present = {"lean_workbook_3", "lean_workbook_17", "lean_workbook_31", "not_in_the_rows"}
    drawn = draw_heldout(_rows(), present, [], seed=7, workbook_problems=5, stp_conjectures=0)
    assert {row["problem_id"] for row in drawn.heldout[:3]} == present - {"not_in_the_rows"}
    assert drawn.counts["heldout_workbook_from_present_holdout"] == 3


def test_no_held_out_problem_and_no_copy_of_one_is_in_the_pool():
    """Fixture 6: no problem of H is in the pool; nor is the same problem under another name, another
    formalization of its natural-language problem, or a statement with its elaborated type."""
    rows = _rows()
    held = "lean_workbook_3"
    rows += [_pool("same_text", "theorem same_text (x : ℕ) : x + 3 = 3 + x := by\n", fingerprint=9001),
             _pool("same_group", "theorem same_group (y : ℕ) : y + 3 = 3 + y + 0 := by\n", group=held, fingerprint=9002),
             _pool("same_type", "theorem same_type (z : ℕ) : z + 3 = 3 + z := by\n", fingerprint=103)]
    drawn = draw_heldout(rows, {held}, [], seed=7, workbook_problems=1, stp_conjectures=0)
    assert [row["problem_id"] for row in drawn.heldout] == [held]
    in_pool = {row["problem_id"] for row in drawn.pool}
    assert not {held, "same_text", "same_group", "same_type"} & in_pool
    assert drawn.counts["dropped_same_problem_as_held_out_or_set_aside"] == 3 and len(drawn.pool) == len(rows) - 4


def test_no_set_aside_statement_and_no_copy_of_one_is_in_the_pool():
    """Fixture 6: none of the statements with no published answer is in the pool, in any form."""
    rows = _rows()
    set_aside = [{"statement_id": "lean_workbook_900", "statement": "theorem lean_workbook_900 (x : ℕ) : x + 5 = 5 + x := by\n",
                  "group": "g900", "type_fingerprint": 110}]
    rows.append(_pool("sibling", "theorem sibling (q : ℕ) : q * 2 = 2 * q := by\n", group="g900", fingerprint=9003))
    drawn = draw_heldout(rows, set(), set_aside, seed=7, workbook_problems=0, stp_conjectures=0)
    in_pool = {row["problem_id"] for row in drawn.pool}
    assert not {"lean_workbook_5", "lean_workbook_10", "sibling"} & in_pool         # same text, same type, same group
    assert drawn.counts["dropped_same_problem_as_held_out_or_set_aside"] == 3


def test_two_members_of_h_are_never_one_problem_and_a_short_draw_says_so():
    rows = [_pool("a", "theorem a (x : ℕ) : x = x := by\n", fingerprint=1), _pool("b", "theorem b (x : ℕ) : x = x := by\n", fingerprint=2),
            _pool("c", "theorem c (y : ℕ) : y = y + 0 := by\n", group="a", fingerprint=3)]
    drawn = draw_heldout(rows, set(), [], seed=1, workbook_problems=3, stp_conjectures=2)
    assert len(drawn.heldout) == 1 and drawn.pool == []
    assert drawn.counts["heldout_lean_workbook_short_by"] == 2 and drawn.counts["heldout_stp_conjecture_short_by"] == 2


def test_a_later_copy_inside_the_pool_is_dropped():
    rows = [_pool("a", "theorem a (x : ℕ) : x = x := by\n", fingerprint=1), _pool("b", "theorem b (x : ℕ) : x = x := by\n", fingerprint=2)]
    drawn = draw_heldout(rows, set(), [], seed=1, workbook_problems=0, stp_conjectures=0)
    assert [row["problem_id"] for row in drawn.pool] == ["a"] and drawn.counts["dropped_copy_within_pool"] == 1


# ------------------------------------------------------------------------------------------------ the slice
def _candidates():
    return [(f"lean_workbook_{n}", LEAN_WORKBOOK) for n in range(40)] + [(f"stp_{n:04d}", STP_CONJECTURE) for n in range(30)]


SIZES = {SLICE_WORKBOOK: 12, SLICE_STP: 9, SLICE_ALL: 20}


def test_the_slice_takes_the_present_holdout_and_three_seeded_samples():
    present = {"lean_workbook_3", "lean_workbook_17", "not_a_candidate"}
    parts = slice_parts(_candidates(), present, 7, 9, SIZES)
    assert sorted(parts[SLICE_PRESENT]) == ["lean_workbook_17", "lean_workbook_3"]
    assert len(parts[SLICE_WORKBOOK]) == 12 and not set(parts[SLICE_WORKBOOK]) & present and all(name.startswith("lean_workbook_") for name in parts[SLICE_WORKBOOK])
    assert len(parts[SLICE_STP]) == 9 and all(name.startswith("stp_") for name in parts[SLICE_STP])
    assert len(parts[SLICE_ALL]) == 20 and {name[:3] for name in parts[SLICE_ALL]} == {"lea", "stp"}
    assert parts == slice_parts(list(reversed(_candidates())), present, 7, 9, SIZES)            # a function of the seeds, not of the order given
    assert parts[SLICE_WORKBOOK] != slice_parts(_candidates(), present, 8, 9, SIZES)[SLICE_WORKBOOK]
    assert parts[SLICE_ALL] != slice_parts(_candidates(), present, 7, 10, SIZES)[SLICE_ALL] and parts[SLICE_STP] == slice_parts(_candidates(), present, 7, 10, SIZES)[SLICE_STP]


def test_a_larger_slice_holds_the_smaller_one_as_its_beginning():
    small = slice_parts(_candidates(), {"lean_workbook_3"}, 7, 9, SIZES)
    large = slice_parts(_candidates(), {"lean_workbook_3"}, 7, 9, {SLICE_WORKBOOK: 20, SLICE_STP: 15, SLICE_ALL: 33})
    for part in (SLICE_WORKBOOK, SLICE_STP, SLICE_ALL):
        assert large[part][:len(small[part])] == small[part]


def test_h_drawn_from_the_slice_is_h_drawn_from_everything():
    """The slice's held-out samples are the beginning of the very orders `draw_heldout` draws in, so the first
    verified of the slice are the first verified of the whole: the same draw, made early."""
    rows = _rows()                                                                # 40 Lean Workbook and 30 STP rows, every one verified
    verified = [row for index, row in enumerate(rows) if index % 4 != 1]              # a quarter did not verify
    present = {"lean_workbook_3", "lean_workbook_17"}
    whole = draw_heldout(verified, present, [], 7, 8, 5)
    parts = slice_parts([(row["problem_id"], row["kind"]) for row in rows], present, 7, 9, SIZES)
    eligible = set(parts[SLICE_PRESENT]) | set(parts[SLICE_WORKBOOK]) | set(parts[SLICE_STP])
    early = draw_heldout([row for row in verified if row["problem_id"] in eligible], present, [], 7, 8, 5)
    assert [row["problem_id"] for row in early.heldout] == [row["problem_id"] for row in whole.heldout] and len(whole.heldout) == 13
    assert early.counts["heldout_lean_workbook_short_by"] == 0 and early.counts["heldout_stp_conjecture_short_by"] == 0


def test_the_base_map_is_the_first_verified_of_its_order_outside_h():
    rows = {row["problem_id"]: row for row in _rows()}
    order = slice_parts([(row["problem_id"], row["kind"]) for row in rows.values()], set(), 7, 9, SIZES)[SLICE_ALL]
    heldout = [rows[order[0]]]                                                    # the first of the order is a member of H
    twin = {**rows[order[1]], "problem_id": "twin_of_h", "statement": heldout[0]["statement"].replace(order[0], "twin_of_h"), "group": "twin"}
    verified = {key: row for key, row in rows.items() if key != order[2]}             # the third did not verify
    verified["twin_of_h"] = twin
    sample, counts = draw_base_map(verified, [order[0], "twin_of_h", *order[1:]], heldout, [], 5)
    assert [row["problem_id"] for row in sample] == [order[1], *order[3:7]]
    assert (counts["in_h"], counts["same_problem_as_held_out_or_set_aside"], counts["not_verified"], counts["base_map"], counts["base_map_short_by"]) == (1, 1, 1, 5, 0)
    assert counts["considered"] == 8                                              # it stops at the fifth it keeps


def test_a_copy_inside_the_sample_and_a_copy_of_a_set_aside_statement_are_skipped():
    a = _pool("a", "theorem a (x : ℕ) : x = x := by\n", fingerprint=1)
    copy = _pool("b", "theorem b (x : ℕ) : x = x := by\n", fingerprint=2)
    aside = {"statement_id": "s", "statement": "theorem s (y : ℕ) : y + 1 = 1 + y := by\n", "group": "gs", "type_fingerprint": 77}
    like_aside = _pool("c", "theorem c (z : ℕ) : z + 1 = 1 + z := by\n", fingerprint=77)
    sample, counts = draw_base_map({"a": a, "b": copy, "c": like_aside}, ["a", "b", "c"], [], [aside], 5)
    assert [row["problem_id"] for row in sample] == ["a"]
    assert (counts["copy_within_the_sample"], counts["same_problem_as_held_out_or_set_aside"], counts["base_map_short_by"]) == (1, 1, 4)


def test_a_problem_drawn_for_the_base_map_is_never_the_copy_that_is_dropped():
    first = _pool("early_in_the_file", "theorem early_in_the_file (x : ℕ) : x = x := by\n", fingerprint=1)
    drawn = _pool("drawn", "theorem drawn (x : ℕ) : x = x := by\n", fingerprint=2)
    other = _pool("other", "theorem other (x : ℕ) : x + 1 = 1 + x := by\n", fingerprint=3)
    assert [row["problem_id"] for row in pool_apart_from([first, drawn, other], [], [])[0]] == ["early_in_the_file", "other"]
    pool, counts = pool_apart_from([first, drawn, other], [], [], kept_first=["drawn"])
    assert [row["problem_id"] for row in pool] == ["drawn", "other"] and counts["dropped_copy_within_pool"] == 1


def test_a_short_draw_asks_for_more_candidates_at_the_share_that_was_taken():
    assert more_candidates_needed(100, 400, 800) == 251                           # half of the candidates were taken: 200, and a quarter more
    assert more_candidates_needed(10, 0, 50) >= 10 and more_candidates_needed(1, 9, 10) >= 1
    assert more_candidates_needed(100, 900, 1000) < more_candidates_needed(100, 300, 1000)
