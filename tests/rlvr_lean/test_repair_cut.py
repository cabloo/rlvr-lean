"""The repair check's rules for ONE failed proof: the cut, the file that asks Lean for the state, the state read from
its answer, the prompt, the proof that is checked. Spec: docs/spec/ladder-loop.spec.md, "L3a: the repair check, no
training", "How Lean's answer is fed back", items 1 to 3. Pure: hand-written proofs and answers in the shape the pool
returns (a `response` with `messages`, each `severity`, `data`, `pos` = {line, column} and `endPos`, and `sorries`,
each `pos` and `goal`), as measured on the pool 2026-10-05."""

from rlvr_lean.domain.proving import build_prover_prompt
from rlvr_lean.domain.repair import cut as rules
from rlvr_lean.domain.verification.lean_source import LEAN_HEADER, build_proof_source

STATEMENT = "theorem demo (a b : ℝ) (h : a ≤ b) : a + 0 ≤ b := by\n"
FIRST = rules.proof_first_line(STATEMENT)          # the file line of the proof's first line
UNSOLVED = "unsolved goals\na b : ℝ\nh : a ≤ b\n⊢ a + 0 ≤ b"


def error(line, column, text, end=None):
    """An error message as the pool returns it; `line` is counted in the PROOF (1 is its first line, 0 the theorem's)."""
    message = {"severity": "error", "data": text, "pos": {"line": FIRST + line - 1, "column": column}}
    if end is not None:
        message["endPos"] = {"line": FIRST + end[0] - 1, "column": end[1]}
    return message


def answer(*messages, sorries=()):
    return {"response": {"messages": list(messages), "sorries": [{"pos": {"line": line, "column": column}, "goal": goal} for line, column, goal in sorries]}}


def cut_of(proof, *messages, statement=STATEMENT):
    return rules.plan_cut(statement, proof, rules.errors_of(answer(*messages)))


# ------------------------------------------------------------------------------------------- Lean's answer
def test_the_proof_begins_on_the_line_after_the_statement_whatever_its_length():
    assert FIRST == LEAN_HEADER.count("\n") + 2 == 9
    source = build_proof_source(STATEMENT, "  linarith\n")
    assert source.split("\n")[FIRST - 1] == "  linarith" and source.split("\n")[FIRST - 2].endswith(":= by")
    two_lines = "theorem demo (a b : ℝ)\n    (h : a ≤ b) : a + 0 ≤ b := by\n"
    assert rules.proof_first_line(two_lines) == FIRST + 1 and rules.proof_first_line(two_lines.rstrip("\n")) == FIRST + 1
    # The checked file drops the white space at the proof's end, and the lines are read the same way.
    assert rules.proof_lines("  simp\n  linarith\n\n") == ["  simp", "  linarith"] and rules.proof_lines(" \n") == []
    assert rules.proof_lines("\n  norm_num\n") == ["", "  norm_num"]


def test_errors_are_read_with_their_positions_the_first_by_position_first():
    raw = answer(error(3, 4, "linarith failed"), {"severity": "warning", "data": "unused variable", "pos": {"line": 1, "column": 0}},
                 error(2, 6, "Unknown identifier `foo`", end=(2, 9)), {"severity": "error", "data": "no position"},
                 {"severity": "info", "data": "'demo' depends on axioms: [sorryAx]", "pos": {"line": 20, "column": 0}}, error(2, 2, "earlier on the line"))
    errors = rules.errors_of(raw)
    assert [(entry["line"] - FIRST + 1, entry["column"]) for entry in errors] == [(2, 2), (2, 6), (3, 4)]
    assert errors[1]["end_line"] == FIRST + 1 and errors[1]["end_column"] == 9 and errors[0]["end_line"] is None
    assert rules.errors_without_a_position(raw) == 1
    assert rules.errors_of({"error": "Lean REPL command timed out in 30 seconds"}) == [] and rules.errors_of({"response": {"message": "refused"}}) == []


# ------------------------------------------------------------------------------------------------- the cut
def test_the_cut_keeps_every_line_before_the_line_of_the_first_error():
    proof = "  intro x\n  simp_all\n  nlinarith [sq_nonneg x]\n"
    cut, why = cut_of(proof, error(3, 2, "linarith failed"), error(2, 2, "simp_all made no progress"))
    assert why is None and cut.kind == rules.BODY and cut.kept == ("  intro x",) and cut.indentation == "  "
    assert cut.failed_step == "simp_all" and cut.error["text"] == "simp_all made no progress" and cut.moved_back == 0 and cut.kept_lines == 1


def test_an_error_inside_a_nested_block_is_cut_at_its_own_indentation():
    """Measured on the pool: a placeholder at the outer indentation after `have h : X := by` gives "expected '{'
    or indented tactic sequence". It must sit at the cut line's own indentation. The position returned is that of
    the `sorry` inside `all_goals sorry`, which is where Lean reports the goals."""
    proof = ("  have h₁ : 0 ≤ a := by\n"
             "    have h₂ : a * 2 > 0 := by\n"
             "      nlinarith\n"
             "    linarith\n"
             "  linarith\n")
    cut, _ = cut_of(proof, error(3, 6, "linarith failed"), error(0, 40, UNSOLVED))
    assert cut.kept == ("  have h₁ : 0 ≤ a := by", "    have h₂ : a * 2 > 0 := by") and cut.indentation == "      " and cut.failed_step == "nlinarith"
    source, line, column = rules.state_source(STATEMENT, cut)
    assert source.split("\n")[line - 1] == "      all_goals sorry" and (line, column) == (FIRST + 2, 6 + len("all_goals "))
    assert rules.STATE_PLACEHOLDER == "all_goals sorry"                # spec item 2: every goal open at the cut, not only the first
    # A failed inner block that ends before the cut: the cut line is back at the outer indentation.
    cut, _ = cut_of(proof, error(5, 2, "linarith failed"))
    assert cut.kept_lines == 4 and cut.indentation == "  "


def test_bullets_arms_and_chains_are_cut_before_the_line_the_error_is_on():
    bullets = "  constructor\n  · simp\n  · norm_num\n    linarith\n"
    cut, _ = cut_of(bullets, error(4, 4, "linarith failed"))
    assert cut.kept == ("  constructor", "  · simp", "  · norm_num") and cut.indentation == "    "
    cut, _ = cut_of(bullets, error(3, 4, "norm_num failed"))        # the error is inside the bullet's own line
    assert cut.kept == ("  constructor", "  · simp") and cut.indentation == "  " and cut.failed_step == "· norm_num"
    arms = "  induction n with\n  | zero => simp\n  | succ n ih =>\n    simp\n    omega\n"
    cut, _ = cut_of(arms, error(5, 4, "omega could not prove the goal"))
    assert cut.kept_lines == 4 and cut.indentation == "    " and cut.failed_step == "omega"
    chain = "  cases' le_total 0 a with h₁ h₁ <;>\n    cases' le_total 0 b with h₂ h₂ <;>\n      nlinarith\n"
    cut, _ = cut_of(chain, error(3, 6, "linarith failed"), error(3, 6, "linarith failed"))
    assert cut.kept_lines == 2 and cut.kept[-1].endswith("<;>") and cut.indentation == "      "


def test_an_error_on_the_first_proof_line_leaves_nothing_kept_and_that_is_allowed():
    cut, why = cut_of("  nlinarith [sq_nonneg (a - b)]\n", error(1, 2, "linarith failed"))
    assert why is None and cut.kind == rules.BODY and cut.kept == () and cut.indentation == "  " and cut.failed_step == "nlinarith [sq_nonneg (a - b)]"
    source, line, column = rules.state_source(STATEMENT, cut)
    assert (line, column) == (FIRST, 12) and source == build_proof_source(STATEMENT, "  all_goals sorry\n")       # the state is then the theorem's own goal


def test_the_theorems_own_unsolved_goals_keep_the_whole_proof_with_the_sorry_after_it():
    proof = "  intro x\n  simp\n"
    cut, why = cut_of(proof, error(0, 40, UNSOLVED, end=(2, 6)))
    assert why is None and cut.kind == rules.WHOLE_PROOF and cut.kept == ("  intro x", "  simp") and cut.failed_step is None and cut.indentation == "  "
    source, line, column = rules.state_source(STATEMENT, cut)
    assert source.split("\n")[FIRST - 1:FIRST + 2] == ["  intro x", "  simp", "  all_goals sorry"] and (line, column) == (FIRST + 2, 12)
    # The placeholder sits with the proof's own steps, whatever the last line's indentation.
    nested = "    intro x\n    have h : 0 ≤ x := by\n      positivity\n"
    cut, _ = cut_of(nested, error(0, 40, UNSOLVED))
    assert cut.kind == rules.WHOLE_PROOF and cut.indentation == "    " and rules.own_indentation(["", "  -- a comment", "   simp"]) == "   "
    assert rules.own_indentation([]) == rules.DEFAULT_INDENTATION


def test_a_syntax_error_is_cut_where_it_is_and_not_read_as_a_proof_that_ran_out():
    """Lean also reports the theorem's goals as unsolved when a proof cannot be parsed to its end: that error is
    first by position (it is on the theorem's line), and the proof did not run out. The cut is at the error in
    the proof."""
    proof = "  intro x\n  simp only [foo] at\n  linarith\n"
    cut, why = cut_of(proof, error(0, 40, UNSOLVED), error(3, 2, "unexpected identifier; expected term"))
    assert why is None and cut.kind == rules.BODY and cut.kept == ("  intro x", "  simp only [foo] at") and cut.failed_step == "linarith"


def test_an_error_on_a_later_line_of_a_step_moves_the_cut_to_where_the_step_begins():
    """A name Lean does not know inside `nlinarith [...,\\n    ...]` is reported on that later line; a proof cut
    there does not parse. The step begins on the line that opens the bracket still open."""
    proof = ("  have h₁ : x = 1 - y := by linarith\n"
             "  simp only [sub_eq_add_neg, add_assoc, (by simp : (1 : ℝ) = 1),\n"
             "    add_left_neg, add_zero]  -- (a stray bracket in a comment\n"
             "  positivity\n")
    cut, why = cut_of(proof, error(3, 4, "Unknown identifier `add_left_neg`"), error(4, 2, "failed to prove positivity"))
    assert why is None and cut.kept == ("  have h₁ : x = 1 - y := by linarith",) and cut.indentation == "  " and cut.moved_back == 1
    assert cut.failed_step == "simp only [sub_eq_add_neg, add_assoc, (by simp : (1 : ℝ) = 1)," and cut.error["line"] == FIRST + 2
    assert rules.opens_unclosed(["  simp [a, (b +", "    c), ⟨d,", "    e⟩,", "    f"]) == 0
    assert rules.opens_unclosed(["  simp [a]", "  exact ⟨1, by simp⟩"]) is None
    assert rules.opens_unclosed(["  /- (a comment [with brackets", "  still the comment -/ simp"]) is None
    assert rules.opens_unclosed(['  exact foo "(" (bar', "    baz)"]) is None and rules.opens_unclosed(["  simp [a] -- (", "  ring"]) is None
    assert rules.step_begins(["  intro x", "  nlinarith [sq_nonneg x,", "    foo]"], 2) == 1 and rules.step_begins(["  intro x", "  simp"], 1) == 1


def test_a_last_step_that_never_ended_is_cut_where_it_begins():
    """A proof whose last step is left open is reported after the proof's last line (at `#print axioms`)."""
    proof = "  intro x\n  nlinarith [sq_nonneg x,\n"
    cut, why = cut_of(proof, error(0, 40, UNSOLVED), error(4, 0, "unexpected token '#print'; expected term"))
    assert why is None and cut.kind == rules.BODY and cut.kept == ("  intro x",) and cut.failed_step == "nlinarith [sq_nonneg x," and cut.moved_back == 1
    closed = "  intro x\n  exact\n"
    cut, why = cut_of(closed, error(4, 0, "unexpected token '#print'; expected term"))
    assert why is None and cut.kind == rules.WHOLE_PROOF and cut.kept_lines == 2 and cut.failed_step is None


def test_a_proof_with_no_error_position_or_an_error_only_on_the_statement_cannot_be_cut():
    assert rules.plan_cut(STATEMENT, "  simp\n", []) == (None, rules.NO_ERROR_POSITION)
    assert rules.plan_cut(STATEMENT, "  simp\n", rules.errors_of({"error": "Lean REPL command timed out in 30 seconds"})) == (None, rules.NO_ERROR_POSITION)
    assert rules.plan_cut(STATEMENT, "  simp\n", rules.errors_of(answer({"severity": "error", "data": "no position"}))) == (None, rules.NO_ERROR_POSITION)
    assert cut_of("  simp\n", error(0, 8, "(kernel) declaration has metavariables")) == (None, rules.ERROR_ON_THE_STATEMENT)
    two_lines = "theorem demo (a b : ℝ)\n    (h : a ≤ b) : a + 0 ≤ b := by\n"
    cut, why = cut_of("  simp\n  linarith\n", error(2, 2, "linarith failed"), statement=two_lines)       # line 2 of THIS file is the first proof line
    assert why is None and cut.kept == () and cut.failed_step == "simp"


# ----------------------------------------------------------------------------------------------- the state
PLACE = (FIRST + 1, 12)             # the `sorry` of `  all_goals sorry` on the second proof line: the placeholder begins at column 2


def state_of(raw, line=PLACE[0], column=PLACE[1]):
    return rules.read_state(raw, line, column)


def test_the_state_is_the_goal_lean_reports_at_the_sorry():
    goal = "a b : ℝ\nh : a ≤ b\n⊢ a + 0 ≤ b"
    raw = answer({"severity": "warning", "data": "declaration uses 'sorry'", "pos": {"line": FIRST - 1, "column": 8}}, sorries=[(*PLACE, goal)])
    assert state_of(raw) == (goal, None)
    # A placeholder inside an inner block: the block around it reports its goals as unsolved, and that one error is allowed.
    around = error(0, 40, UNSOLVED, end=(3, 0))
    assert state_of(answer(around, sorries=[(*PLACE, goal)])) == (goal, None)
    assert state_of(answer(error(0, 40, UNSOLVED), sorries=[(*PLACE, goal)])) == (goal, None)       # no end position: taken as around it


def test_every_goal_open_at_the_cut_is_in_the_state_joined_as_lean_prints_several_goals():
    """`all_goals sorry` runs a `sorry` on each goal open at the cut, and Lean reports each at the same place (spec
    item 2: a plain `sorry` reported only the goal it closed)."""
    raw = answer(sorries=[(*PLACE, "case inl\n⊢ P"), (*PLACE, "case inr\n⊢ Q"), (FIRST + 5, 12, "elsewhere")])
    assert state_of(raw) == ("case inl\n⊢ P\n\ncase inr\n⊢ Q", None)


def test_no_state_can_be_had_from_a_file_with_another_error_and_the_reason_says_where():
    goal = "⊢ a + 0 ≤ b"
    there = [(*PLACE, goal)]
    assert state_of(answer(error(1, 2, "failed to infer `have` declaration type"), sorries=there)) == (None, rules.ERROR_BEFORE_THE_SORRY)
    # "unsolved goals" of a block that ENDED before the placeholder is an error of the kept lines, not the block around it.
    assert state_of(answer(error(1, 20, UNSOLVED, end=(1, 30)), sorries=there)) == (None, rules.ERROR_BEFORE_THE_SORRY)
    # At the placeholder, from where it begins (`all_goals`) to its `sorry`: it does not fit where the cut put it.
    assert state_of(answer(error(2, 2, "expected '{' or indented tactic sequence"))) == (None, rules.ERROR_AT_THE_SORRY)
    assert state_of(answer(error(2, 2, "Unknown identifier `all_goals`"), error(0, 40, UNSOLVED))) == (None, rules.ERROR_AT_THE_SORRY)      # a cut where a TERM stood
    assert state_of(answer(error(2, 12, "No goals to be solved"), error(0, 40, UNSOLVED))) == (None, rules.NO_GOALS_AT_THE_CUT)
    assert state_of(answer(error(4, 0, "unexpected token '#print'; expected ']'"), sorries=there)) == (None, rules.ERROR_AFTER_THE_SORRY)
    assert state_of(answer(error(4, 0, "after"), error(1, 2, "before"), sorries=there)) == (None, rules.ERROR_BEFORE_THE_SORRY)
    assert state_of(answer({"severity": "error", "data": "an error with no position"}, sorries=there)) == (None, rules.ERROR_BEFORE_THE_SORRY)


def test_a_sound_file_in_which_lean_ran_no_sorry_had_no_goal_left_at_the_cut():
    """Spec item 2a: with no goal open, `all_goals` runs nothing and Lean reports no `sorry` at all. The kept
    lines then close the block they end in: the whole proof, or an inner block (the theorem's goals stay open)."""
    closed = answer({"severity": "info", "data": "'demo' depends on axioms: [propext]", "pos": {"line": FIRST + 3, "column": 0}})
    assert state_of(closed) == (None, rules.NO_GOALS_AT_THE_CUT) and state_of(answer()) == (None, rules.NO_GOALS_AT_THE_CUT)
    assert state_of(answer(error(0, 40, UNSOLVED, end=(3, 0)))) == (None, rules.NO_GOALS_AT_THE_CUT)           # an inner block was closed
    # Not when the file is unsound, and not when Lean ran a `sorry` somewhere else than where ours was put.
    assert state_of(answer(error(1, 2, "linarith failed"))) == (None, rules.ERROR_BEFORE_THE_SORRY)
    assert state_of(answer(sorries=[(FIRST + 3, 12, "⊢ another place")])) == (None, rules.NO_SORRY_AT_THE_CUT)
    assert state_of(answer(sorries=[(*PLACE, "  ")])) == (None, rules.NO_SORRY_AT_THE_CUT)
    # The kept lines as a proof of their own: what is then checked, assembled as an attempt's proof is.
    assert rules.trimmed_proof(("  intro x", "  nlinarith [sq_nonneg x]")) == "  intro x\n  nlinarith [sq_nonneg x]\n" and rules.trimmed_proof(()) == ""
    assert build_proof_source(STATEMENT, rules.trimmed_proof(("  simp",))) == build_proof_source(STATEMENT, "  simp\n")


def test_a_timeout_and_no_answer_are_no_state():
    assert state_of({"error": "Lean REPL command timed out in 30 seconds"}) == (None, rules.STATE_TIMEOUT)
    assert state_of({"error": "server_error: no answer after 2 attempts: HTTP 500"}) == (None, rules.STATE_NO_ANSWER)
    assert state_of({"response": {"message": "could not parse"}}) == (None, rules.STATE_NO_ANSWER) and state_of({}) == (None, rules.STATE_NO_ANSWER)
    assert {rules.NO_ERROR_POSITION, rules.STATE_TIMEOUT, rules.NO_SORRY_AT_THE_CUT, rules.NO_GOALS_AT_THE_CUT, rules.PROMPT_TOO_LONG} <= set(rules.NO_STATE_REASONS)


# ---------------------------------------------------------------------------------------------- the prompt
def test_the_state_comment_is_the_provers_own_format():
    """DeepSeek-Prover-V1.5, prover/lean/proof.py, `Proof.segmentation`: a comment block at the indentation,
    its lines two spaces further in, a newline after the closing mark."""
    state = "a b : ℝ\nh : a ≤ b\n⊢ a + 0 ≤ b"
    assert rules.state_comment(state, "  ") == "  /- tactic state:\n    a b : ℝ\n    h : a ≤ b\n    ⊢ a + 0 ≤ b\n  -/\n"
    assert rules.state_comment("⊢ P", "      ") == "      /- tactic state:\n        ⊢ P\n      -/\n"
    indent_len, goal = 4, "case inl\n⊢ P\n\ncase inr\n⊢ Q"                     # the authors' own expression, for any state
    newline_with_indent = "\n" + " " * indent_len
    theirs = newline_with_indent.join([" " * indent_len + "/- tactic state:", "  " + goal.replace("\n", newline_with_indent + "  "), "-/\n"])
    assert rules.state_comment(goal, " " * indent_len) == theirs


def test_the_prompt_is_the_blind_prompt_the_kept_lines_and_the_state_and_the_other_arm_has_no_state():
    kept, state = ("  intro x", "  have h₁ : 0 ≤ x := by"), "x : ℝ\n⊢ 0 ≤ x"
    with_state = rules.resume_prompt(STATEMENT, kept, "    ", state)
    assert with_state == (build_prover_prompt(STATEMENT) + "  intro x\n  have h₁ : 0 ≤ x := by\n" + "    /- tactic state:\n      x : ℝ\n      ⊢ 0 ≤ x\n    -/\n")
    without = rules.resume_prompt(STATEMENT, kept, "    ", None)
    assert without == build_prover_prompt(STATEMENT) + "  intro x\n  have h₁ : 0 ≤ x := by\n" and with_state.startswith(without)
    # Nothing kept: the state is the theorem's own goal, and the arm without the state has the blind prompt itself.
    assert rules.resume_prompt(STATEMENT, (), "  ", None) == build_prover_prompt(STATEMENT)
    assert rules.resume_prompt(STATEMENT, (), "  ", "⊢ P") == build_prover_prompt(STATEMENT) + "  /- tactic state:\n    ⊢ P\n  -/\n"


def test_the_proof_that_is_checked_is_the_kept_lines_and_the_continuation_without_the_comment():
    kept = ("  intro x", "  have h₁ : 0 ≤ x := by")
    proof = rules.resumed_proof(kept, "    positivity\n  linarith\n")
    assert proof == "  intro x\n  have h₁ : 0 ≤ x := by\n    positivity\n  linarith\n" and "tactic state" not in proof
    assert rules.resumed_proof((), "  simp\n") == "  simp\n"
    # A proof resumed twice is cut by the same rules: its lines are the kept ones and the continuation's.
    assert rules.proof_lines(proof)[:2] == list(kept)


def test_a_continuation_that_begins_with_the_failed_step_is_seen():
    assert rules.repeats_failed_step("    nlinarith  [sq_nonneg x]\n  linarith\n", "nlinarith [sq_nonneg x]") is True
    assert rules.repeats_failed_step("\n    -- try again\n    /- tactic state:\n      ⊢ P\n    -/\n    nlinarith [sq_nonneg x]\n", "nlinarith [sq_nonneg x]") is True
    assert rules.repeats_failed_step("    positivity\n    nlinarith [sq_nonneg x]\n", "nlinarith [sq_nonneg x]") is False
    assert rules.repeats_failed_step("", "simp") is False and rules.repeats_failed_step("  simp\n", None) is None
    assert rules.repeats_failed_step("  -- nothing\n", "-- a step that is only a comment") is None       # no step to repeat: never "repeated"
    assert rules.first_step("  -- only a comment\n") is None and rules.first_step("\n   rw  [h]   at h₂\n") == "rw [h] at h₂"
