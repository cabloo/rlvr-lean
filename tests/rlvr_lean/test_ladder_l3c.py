"""The accumulating episode's rules, read and report on small hand-made inputs. Spec: docs/spec/ladder-loop.spec.md,
"L3c: an episode that keeps what verified (no training)" and its "Made exact before the build (2026-10-07)". Pure. The cut
at a proof's first error is L3a's (`test_repair_cut.py`), the known copies are L3a2's (`test_ladder_l3a2.py`); the stage
end to end is `test_ladder_l3c_stage.py`."""

import pytest

from rlvr_lean.data.heldout_proof_lines import LENGTH_GROUPS, length_group, proof_lines
from rlvr_lean.domain.ladder_round.rounds import sign_test
from rlvr_lean.domain.repair import BLIND, FIRST
from rlvr_lean.domain.repair import accumulate as rules
from rlvr_lean.domain.repair import cut, read
from rlvr_lean.domain.repair.accumulate import ACCUMULATE, ARMS, ASSEMBLED, CONTINUE, FRESH, OWN_PROOF, REACHES, RESOLVES_FEWER, STANDS, THE_SOLVERS_EPISODE
from rlvr_lean.domain.repair.alternate import KNOWN_COPY
from rlvr_lean.domain.repair.read import INCONCLUSIVE, NOT_SHOWN
from rlvr_lean.domain.verification.lean_source import build_proof_source
from rlvr_lean.reporting import ladder_l3c as reporting
from rlvr_lean.reporting.ladder_l3c import build_l3c_report

GENERATIONS = 8
OK, BAD = "verified", "lean_error"
B, A = BLIND, ACCUMULATE
STATEMENT = "theorem demo (a b : ℝ) (h : a ≤ b) : a + 0 ≤ b := by\n"
FIRST_LINE = cut.proof_first_line(STATEMENT)


def texts(found):
    return [block["text"] for block in found.blocks]


def lemma(text, name, statement, generation=1):
    return {"text": text, "name": name, "statement": statement, "generation": generation}


# ----------------------------------------------------------------------------------------------- the steps
def test_a_proofs_top_level_steps_are_read_by_indentation_and_comments_are_no_step():
    proof = ("  have h₁ : 0 ≤ a := by\n"
             "    nlinarith\n"
             "  -- now the other one\n"
             "\n"
             "  have h₂ : 0 ≤ b := by linarith  -- from h\n"
             "  constructor\n"
             "  · exact h₁\n"
             "  all_goals linarith\n")
    code, base, steps = rules.top_steps(cut.proof_lines(proof))
    assert base == 2 and steps == [(0, 2, False), (4, 5, False), (5, 8, True)]            # the comment and the blank line are in no step
    assert code[2] == "" and code[4] == "  have h₂ : 0 ≤ b := by linarith" and rules.GOES_ON == ("<;>", "·", "|", ".", "all_goals", "any_goals")
    # A line at the steps' own indentation that begins with `<;>` goes on with the step before it, and says so.
    assert rules.top_steps(["  have h : P := by simp", "  <;> linarith", "  linarith"])[2] == [(0, 2, True), (2, 3, False)]
    assert rules.top_steps(["  induction n with", "  | zero => simp", "  | succ n ih =>", "    omega"])[2] == [(0, 4, True)]
    # A block comment is taken out where it stands; a line of code left of the first one is a layout that is not understood.
    assert rules.top_steps(["    /- a note", "  that goes on -/", "    simp", "    linarith"])[1:] == (4, [(2, 3, False), (3, 4, False)])
    assert rules.top_steps(["    simp", "  linarith"]) is None and rules.top_steps(["  · simp"]) is None and rules.top_steps(["", "  -- nothing"]) is None


def test_the_name_a_have_binds_is_read_from_its_first_line():
    assert rules.bound_name("  have h₁ : 0 ≤ a := by") == "h₁" and rules.bound_name("have h' := foo a") == "h'" and rules.bound_name("  have h(x : ℝ) : P x := by") == "h"
    assert rules.bound_name("  have : 0 ≤ a := by positivity") == "" and rules.bound_name("  have := sq_nonneg a") == "" and rules.bound_name("  have: P := h") == ""
    # A pattern, another kind of `have`, a `have` with nothing after it, and any other tactic bind no one name.
    for line in ("  have ⟨x, hx⟩ := h", "  have (x) : P := h", "  haveI : Fact p := ⟨hp⟩", "  have' h := foo", "  have", "  obtain ⟨x, hx⟩ := h", "  nlinarith [h]", "  haven : P := h"):
        assert rules.bound_name(line) is None, line


def test_names_are_read_as_their_new_ones_in_one_pass_and_only_as_whole_names():
    text = "nlinarith [h, h₁, h₁₀, h', x.h, h.le, sq_nonneg (h - 1)]"
    assert rules.renamed(text, {"h": "h_g2", "h₁": "h₁_g2"}) == "nlinarith [h_g2, h₁_g2, h₁₀, h', x.h, h_g2.le, sq_nonneg (h_g2 - 1)]"
    # One pass: a name the model wrote that looks like a pooled one is not read twice.
    assert rules.renamed("exact h_g2.trans h", {"h": "h_g2", "h_g2": "h_g2_g2"}) == "exact h_g2_g2.trans h_g2" and rules.renamed("simp", {}) == "simp"
    assert rules.token("h₁").search("linarith [h₁]") and not rules.token("h₁").search("linarith [h₁₀, h₁', a.h₁]")


# ----------------------------------------------------------------------------------------------- the harvest
def test_the_leading_lemmas_before_the_first_error_join_the_pool_under_names_of_their_generation():
    proof = ("  have h₁ : 0 ≤ (a - b) ^ 2 := by\n"
             "    nlinarith [sq_nonneg (a - b)]\n"
             "  have h₂ : 0 ≤ b - a := by linarith [h, h₁]\n"
             "  nlinarith [h₁, h₂,\n"
             "    mul_self_nonneg a]\n")
    found = rules.harvest(proof, 3, [], 3, 12)                        # the first error is on line 4: three lines are kept
    assert texts(found) == ["  have h₁_g3 : 0 ≤ (a - b) ^ 2 := by\n    nlinarith [sq_nonneg (a - b)]", "  have h₂_g3 : 0 ≤ b - a := by linarith [h, h₁_g3]"]
    assert [(block["name"], block["statement"], block["generation"]) for block in found.blocks] == [("h₁_g3", ": 0 ≤ (a - b) ^ 2", 3), ("h₂_g3", ": 0 ≤ b - a", 3)]
    # The proof was lemmas and then ONE closing step: that step is the closer, its names read as the pooled ones.
    assert found.closer == {"text": "  nlinarith [h₁_g3, h₂_g3,\n    mul_self_nonneg a]", "needs": ["h₁_g3", "h₂_g3"], "generation": 3}
    assert (found.duplicates, found.over_the_size, found.ended, found.own_proof) == (0, 0, rules.NOT_A_HAVE, True)
    # The theorem's own hypothesis `h` is not a lemma of the proof: it keeps its name. So does a name inside the lemma that binds it.
    shadows = rules.harvest("  have h : 0 ≤ b - a := by linarith [h]\n  linarith [h]\n", 1, [], 1, 12)
    assert texts(shadows) == ["  have h_g1 : 0 ≤ b - a := by linarith [h]"] and shadows.closer["text"] == "  linarith [h_g1]"
    # A lemma that holds the first error, and what follows it, is not harvested; with a step after it there is no closer.
    stopped = rules.harvest(proof, 2, [], 3, 12)
    assert texts(stopped) == ["  have h₁_g3 : 0 ≤ (a - b) ^ 2 := by\n    nlinarith [sq_nonneg (a - b)]"] and stopped.closer is None and stopped.ended == rules.THE_ERROR
    nothing = rules.harvest(proof, 1, [], 3, 12)                      # the error is inside the first lemma
    assert (nothing.blocks, nothing.closer, nothing.ended) == ((), None, rules.THE_ERROR)
    # The whole proof kept (it ran out with goals open): every leading lemma is harvested, and the last step is still the closer.
    assert len(rules.harvest(proof, 5, [], 3, 12).blocks) == 2 and rules.harvest(proof, 5, [], 3, 12).closer is not None


def test_only_a_have_with_a_proof_of_its_own_that_binds_one_name_is_a_lemma():
    def ends(proof, kept=9):
        found = rules.harvest(proof, kept, [], 1, 12)
        return len(found.blocks), found.ended, found.closer is not None

    first = "  have h₁ : 0 ≤ a := by positivity\n"
    assert ends(first + "  obtain ⟨x, hx⟩ := h\n  have h₂ : 0 ≤ x := by linarith\n  linarith\n") == (1, rules.NOT_A_HAVE, False)        # what follows may depend on a goal it changed
    assert ends(first + "  have ⟨x, hx⟩ := h\n  linarith\n") == (1, rules.NOT_A_HAVE, False)              # a `have` that binds a pattern
    assert ends(first + "  haveI : Fact (0 < a) := ⟨by positivity⟩\n  linarith\n") == (1, rules.NOT_A_HAVE, False)
    assert ends(first + "  have h₂ : 0 ≤ b\n  linarith\n  linarith\n") == (1, rules.NO_PROOF, False)         # it opens a goal of its own: no `:=`
    assert ends(first + "  have h₂ : 0 ≤ b\n  · linarith\n  linarith\n") == (1, rules.HAVE_GOES_ON, False)     # ... and a bullet at the steps' indentation goes on with it
    assert ends(first + "  have h₂ : 0 ≤ b := by simp\n  <;> linarith\n  linarith\n") == (1, rules.HAVE_GOES_ON, False)
    assert ends(first + "  have h₂ : 0 ≤ b := by\n    linarith\n") == (2, rules.THE_END, False)            # every step was a lemma: no closer
    assert ends("  nlinarith [sq_nonneg (a - b)]\n", 0) == (0, rules.NOT_A_HAVE, True)                   # ONE closing step and no lemma: still a closer
    assert ends("  intro x\n  nlinarith [sq_nonneg x]\n", 1) == (0, rules.NOT_A_HAVE, False)              # two steps are no closer
    assert ends("  have h₁ : 0 ≤ a := by positivity\n  have h₂ : 0 ≤ b := by bad\n", 1) == (1, rules.THE_ERROR, False)      # the last step is a `have`: no closer
    # A `:=` inside brackets is not the lemma's own: `have h : P (x := 1)` has no proof.
    assert ends("  have h : foo (x := 1)\n  linarith\n") == (0, rules.NO_PROOF, False)
    assert ends("    simp\n  linarith\n") == (0, rules.LAYOUT, False) and ends("", 0) == (0, rules.LAYOUT, False)


def test_a_name_bound_twice_in_one_proof_is_two_pooled_names():
    proof = "  have h : 0 ≤ a := by positivity\n  have h : 0 ≤ a + 1 := by linarith [h]\n  have k : 0 ≤ a + 2 := by linarith [h]\n  linarith [h, k]\n"
    found = rules.harvest(proof, 3, [], 2, 12)
    assert texts(found) == ["  have h_g2 : 0 ≤ a := by positivity", "  have h_g2' : 0 ≤ a + 1 := by linarith [h_g2]", "  have k_g2 : 0 ≤ a + 2 := by linarith [h_g2']"]
    assert found.closer == {"text": "  linarith [h_g2', k_g2]", "needs": ["h_g2'", "k_g2"], "generation": 2}
    # An anonymous lemma stays anonymous, and from it on `this` is that lemma, whatever a `have this` bound before.
    proof = "  have this : 0 ≤ a := by positivity\n  have : 0 ≤ b := by linarith [this]\n  have k := add_nonneg this h\n  linarith [this]\n"
    found = rules.harvest(proof, 3, [], 1, 12)
    assert texts(found) == ["  have this_g1 : 0 ≤ a := by positivity", "  have : 0 ≤ b := by linarith [this_g1]", "  have k_g1 := add_nonneg this h"]
    assert [(block["name"], block["statement"]) for block in found.blocks] == [("this_g1", ": 0 ≤ a"), ("", ": 0 ≤ b"), ("k_g1", ":= add_nonneg this h")]
    assert found.closer == {"text": "  linarith [this]", "needs": [], "generation": 1}


def test_a_statement_the_pool_holds_is_not_added_and_its_name_is_read_as_the_pooled_one():
    pool = [lemma("  have h₁_g1 : 0 ≤ a := by positivity", "h₁_g1", ": 0 ≤ a"), lemma("  have : 0 ≤ b := by positivity", "", ": 0 ≤ b"),
            lemma("  have k_g1 := sq_nonneg (a - b)", "k_g1", ":= sq_nonneg (a - b)")]
    proof = ("  have ha : 0  ≤ a := by\n    positivity\n"                       # the pool holds it (white space apart), under another name
             "  have hab : 0 ≤ a + b := by linarith [ha]\n"
             "  have s := sq_nonneg (a - b)\n"                                  # a `have` with no type: the same term is the same lemma
             "  have : 0 ≤ b := by linarith\n"                                  # anonymous, held, and nothing after it says `this`
             "  nlinarith [ha, hab, s]\n")
    found = rules.harvest(proof, 5, pool, 4, 12)
    assert texts(found) == ["  have hab_g4 : 0 ≤ a + b := by linarith [h₁_g1]"] and found.duplicates == 3
    assert found.closer == {"text": "  nlinarith [h₁_g1, hab_g4, k_g1]", "needs": ["hab_g4", "h₁_g1", "k_g1"], "generation": 4}
    assert found.own_proof is False                                 # with a lemma left out the pool and the closer are not this proof
    # An anonymous lemma is added all the same when the proof goes on to say `this`: it must be the nearest one.
    said = rules.harvest("  have : 0 ≤ b := by linarith\n  have q : 0 ≤ b + 1 := by linarith [this]\n  linarith\n", 2, pool, 4, 12)
    assert texts(said) == ["  have : 0 ≤ b := by linarith", "  have q_g4 : 0 ≤ b + 1 := by linarith [this]"] and said.duplicates == 0
    # A named and an anonymous lemma of one statement are both kept: `this` cannot stand for a name, nor a name for `this`.
    both = rules.harvest("  have hb : 0 ≤ b := by linarith\n  have : 0 ≤ a := by linarith\n  linarith\n", 2, pool, 4, 12)
    assert [block["name"] for block in both.blocks] == ["hb_g4", ""] and both.duplicates == 0
    # The same statement twice in ONE proof: the second is read as the first.
    twice = rules.harvest("  have p : 0 ≤ a + 2 := by linarith\n  have q : 0 ≤ a + 2 := by linarith\n  linarith [q]\n", 2, [], 2, 12)
    assert texts(twice) == ["  have p_g2 : 0 ≤ a + 2 := by linarith"] and twice.duplicates == 1 and twice.closer["text"] == "  linarith [p_g2]"


def test_the_pool_only_grows_and_a_harvest_past_its_size_is_cut_at_its_tail():
    pool = [lemma(f"  have h{index}_g1 : 0 ≤ a + {index} := by linarith", f"h{index}_g1", f": 0 ≤ a + {index}") for index in range(10)]
    proof = "".join(f"  have k{index} : 0 ≤ b + {index} := by linarith\n" for index in range(4)) + "  have h5' : 0 ≤ a + 5 := by linarith\n  nlinarith [k0, k3, h5']\n"
    found = rules.harvest(proof, 5, pool, 2, 12)
    assert [block["name"] for block in found.blocks] == ["k0_g2", "k1_g2"] and (found.over_the_size, found.duplicates) == (2, 1)       # room for two; a held statement needs none
    # The closer names a lemma that did not reach the pool: it is kept with what it needs, and is not tried.
    assert found.closer["needs"] == ["h5_g1", "k0_g2", "k3_g2"] and found.own_proof is False
    grown = [*pool, *found.blocks]
    assert rules.closers_to_check(grown, [found.closer], True, found.closer) == [] and rules.pooled_names(grown) >= {"h5_g1", "k0_g2", "k1_g2"}
    full = rules.harvest(proof, 5, grown, 3, 12)                                   # a full pool takes no more
    assert (full.blocks, full.over_the_size, full.duplicates) == ((), 2, 3) and full.closer["needs"] == ["h5_g1", "k0_g2", "k3_g3"]


def test_a_continuing_proof_is_harvested_from_what_the_model_wrote_after_the_pool():
    pool = [lemma("  have h₁_g1 : 0 ≤ a := by positivity", "h₁_g1", ": 0 ≤ a")]
    continuation = "  have h₂ : 0 ≤ a + 1 := by linarith [h₁_g1]\n  have h₁_g1 : 0 ≤ a + 2 := by linarith\n  nlinarith [h₁_g1, h₂]\n"
    found = rules.harvest(continuation, 2, pool, 2, 12, continues=True)
    # Its lemmas were verified with the pool above them. A pooled name the model bound again is a new name: the pool's own stays what it was.
    assert texts(found) == ["  have h₂_g2 : 0 ≤ a + 1 := by linarith [h₁_g1]", "  have h₁_g1_g2 : 0 ≤ a + 2 := by linarith"]
    assert found.closer == {"text": "  nlinarith [h₁_g1_g2, h₂_g2]", "needs": ["h₁_g1_g2", "h₂_g2"], "generation": 2}
    # The pool, these lemmas and this closer ARE the proof Lean has just rejected, under its pooled names.
    assert found.own_proof is True and rules.harvest(continuation, 2, pool, 2, 12).own_proof is False       # a FRESH proof below a pool is not
    assert rules.harvest("  linarith [h₁_g1]\n", 0, pool, 2, 12, continues=True).own_proof is True          # ONE step after the pool: the same text, even
    # A continuation that does not stand where the pool's lemmas do is not understood as steps after the pool.
    off = rules.harvest("    have h₂ : 0 ≤ a + 1 := by linarith\n    linarith\n", 1, pool, 2, 12, continues=True)
    assert (off.blocks, off.closer, off.ended) == ((), None, rules.OFF_THE_POOL)
    # A fresh proof is moved to the pool's indentation, whatever its own.
    deep = rules.harvest("      have h : 0 ≤ a := by\n        positivity\n      nlinarith [h,\n         sq_nonneg a]\n", 2, [], 5, 12)
    assert texts(deep) == ["  have h_g5 : 0 ≤ a := by\n    positivity"] and deep.closer["text"] == "  nlinarith [h_g5,\n     sq_nonneg a]"
    flat = rules.harvest("have h : 0 ≤ a := by\n  positivity\nlinarith\n", 2, [], 5, 12)
    assert texts(flat) == ["  have h_g5 : 0 ≤ a := by\n    positivity"] and flat.closer["text"] == "  linarith"


def test_comments_are_left_out_of_the_pool_and_a_proofs_lines_are_counted_as_the_published_ones():
    proof = "  -- first the easy part\n  have h₁ : 0 ≤ a := by positivity  -- clear\n\n  /- then the rest -/\n  linarith [h₁]\n"
    found = rules.harvest(proof, 4, [], 1, 12)
    assert texts(found) == ["  have h₁_g1 : 0 ≤ a := by positivity"] and found.closer["text"] == "  linarith [h₁_g1]"
    for text in (proof, "  simp\n", "\n  nlinarith [a,\n    b]\n\n  -- done\n", ""):
        assert rules.proof_line_count(text) == proof_lines(text)
    assert rules.proof_line_count(proof) == 3 and rules.LONG_PROOF == 8 and length_group(rules.LONG_PROOF) == "8+" == LENGTH_GROUPS[-1] == reporting.LENGTH_GROUPS[-1]


# -------------------------------------------------------------------------------------------------- the pool
def answer(*messages, sorries=()):
    return {"response": {"messages": list(messages), "sorries": [{"pos": {"line": line, "column": column}, "goal": goal} for line, column, goal in sorries]}}


def error(line, column, text, end=None):
    message = {"severity": "error", "data": text, "pos": {"line": FIRST_LINE + line - 1, "column": column}}
    if end is not None:
        message["endPos"] = {"line": FIRST_LINE + end[0] - 1, "column": end[1]}
    return message


def test_a_pool_is_checked_with_all_goals_sorry_after_its_lemmas():
    pool = [lemma("  have h₁_g1 : 0 ≤ a := by\n    positivity", "h₁_g1", ": 0 ≤ a"), lemma("  have : 0 ≤ b := by linarith", "", ": 0 ≤ b", 2)]
    assert rules.pool_lines(pool) == ["  have h₁_g1 : 0 ≤ a := by", "    positivity", "  have : 0 ≤ b := by linarith"] and rules.pooled_names(pool) == {"h₁_g1"}
    assert rules.pool_text(pool) == "  have h₁_g1 : 0 ≤ a := by\n    positivity\n  have : 0 ≤ b := by linarith\n"
    source, line, column = rules.pool_source(STATEMENT, pool)
    assert source == build_proof_source(STATEMENT, rules.pool_text(pool) + "  all_goals sorry\n") and (line, column) == (FIRST_LINE + 3, 12)
    assert rules.POOL_INDENTATION == "  " and cut.STATE_PLACEHOLDER == "all_goals sorry"
    # The proof a kept closer makes after it: the lemmas, then the closing step.
    assert rules.assembled_proof(pool, {"text": "  nlinarith [h₁_g1,\n    this]"}) == rules.pool_text(pool) + "  nlinarith [h₁_g1,\n    this]\n"


def test_a_pool_stands_when_lean_reports_no_error_and_one_goal_at_the_sorry():
    line, column = FIRST_LINE + 3, 12
    goal = "a b : ℝ\nh : a ≤ b\nh₁_g1 : 0 ≤ a\nthis : 0 ≤ b\n⊢ a + 0 ≤ b"
    warning = {"severity": "warning", "data": "declaration uses 'sorry'", "pos": {"line": FIRST_LINE - 1, "column": 8}}
    assert rules.read_pool(answer(warning, sorries=[(line, column, goal)]), line, column) == (goal, STANDS) and STANDS == "state"
    # Any error at all means the pool does not stand: one in a lemma, and the "unsolved goals" of a `have` left open around the `sorry`.
    assert rules.read_pool(answer(error(2, 4, "linarith failed"), sorries=[(line, column, goal)]), line, column) == (None, cut.ERROR_BEFORE_THE_SORRY)
    open_have = answer(error(3, 20, "unsolved goals\n⊢ a + 0 ≤ b", end=(4, 17)), sorries=[(line, column, "⊢ 0 ≤ b")])
    assert cut.read_state(open_have, line, column) == ("⊢ 0 ≤ b", None) and rules.read_pool(open_have, line, column) == (None, rules.ERROR_IN_THE_POOL)
    # More than one goal: a pooled step opened a goal of its own. No goal: the "lemmas" closed the theorem's. Neither is a pool of lemmas.
    assert rules.read_pool(answer(sorries=[(line, column, "⊢ 0 ≤ b"), (line, column, goal)]), line, column) == (None, rules.SEVERAL_GOALS)
    assert rules.read_pool(answer(), line, column) == (None, cut.NO_GOALS_AT_THE_CUT)
    assert rules.read_pool({"error": "Lean REPL command timed out in 30 seconds"}, line, column) == (None, cut.STATE_TIMEOUT)
    assert rules.read_pool({"error": "HTTP 500"}, line, column) == (None, cut.STATE_NO_ANSWER)


def test_the_closers_tried_after_a_pool_are_the_kept_ones_whose_pooled_names_exist():
    pool = [lemma("  have h₁_g1 : 0 ≤ a := by positivity", "h₁_g1", ": 0 ≤ a"), lemma("  have k_g3 : 0 ≤ b := by linarith", "k_g3", ": 0 ≤ b", 3)]
    plain, named, missing = ({"text": "  nlinarith", "needs": [], "generation": 1}, {"text": "  linarith [h₁_g1, k_g3]", "needs": ["h₁_g1", "k_g3"], "generation": 3},
                             {"text": "  linarith [q_g2]", "needs": ["q_g2"], "generation": 2})
    # The pool has just grown: every kept closer, but for the one that names a lemma the pool does not hold.
    assert rules.closers_to_check(pool, [plain, missing, named], True, named) == [plain, named]
    # The pool has not: only the closer just kept (the others were checked after this pool already), and none when none was kept.
    assert rules.closers_to_check(pool, [plain, missing, named], False, named) == [named] and rules.closers_to_check(pool, [plain, named], False, None) == []
    assert rules.closers_to_check(pool, [plain, missing], False, missing) == []
    assert rules.closers_to_check([], [plain], True, plain) == []                       # there is nothing to check a closer after


# ---------------------------------------------------------------------------------------------- the episode
def attempt(arm, problem, number, loop, status, proof="  simp\n", kind=FRESH, how="blind", **more):
    return {"arm": arm, "problem_id": problem, "episode": number, "loop": loop, "status": status, "audit": False, "kind": kind, "how": how, "had_state": how == "resume",
            "no_state": None, "token_count": 50, "prompt_tokens": 120, "sent_to_lean": status not in ("capped_tokens", KNOWN_COPY), "proof": proof,
            "pool_blocks": 0, "shared": kind == FRESH, "kept_closer": None, **more}


def pool_state(problem, number, loop, blocks, stands=True, outcome=STANDS, harvested=1):
    return {"problem_id": problem, "episode": number, "loop": loop, "generation": loop + 1, "blocks": blocks, "stands": stands, "outcome": outcome,
            "state": "⊢ goal" if stands else None, "harvested": harvested}


def closer_check(problem, number, loop, status, proof, sent=True, from_generation=1, pool_blocks=1):
    return {"problem_id": problem, "episode": number, "loop": loop, "status": status, "sent_to_lean": sent, "proof": proof, "from_generation": from_generation, "pool_blocks": pool_blocks}


def test_what_an_episode_holds_is_read_from_its_stored_rows():
    first = lemma("  have h_g1 : 0 ≤ a := by positivity", "h_g1", ": 0 ≤ a")
    second = lemma("  have k_g2 : 0 ≤ b := by linarith", "k_g2", ": 0 ≤ b", 2)
    closer = {"text": "  linarith [h_g1]", "needs": ["h_g1"], "generation": 1}
    chain = [attempt(FIRST, "p", 0, 0, BAD, "  have h : 0 ≤ a := by positivity\n  linarith [h]\n", kept_closer=closer)]
    assert rules.empty_state() == {"pool": [], "state": None, "made_at": None, "changed": False, "closers": [], "rejected": {}, "assembled": {}}
    assert rules.kind_of(rules.empty_state()) == FRESH and rules.kind_of(rules.episode_state(chain, [], [])) == FRESH        # an empty pool: fresh
    pools = [pool_state("p", 0, 0, [first])]
    own = [closer_check("p", 0, 0, OWN_PROOF, "  have h_g1 : 0 ≤ a := by positivity\n  linarith [h_g1]\n", sent=False)]
    held = rules.episode_state(chain, pools, own)
    assert (held["pool"], held["state"], held["made_at"], held["changed"], held["closers"]) == ([first], "⊢ goal", 0, True, [closer]) and rules.kind_of(held) == CONTINUE
    # What Lean has rejected: the attempt's text, and the same proof under its pooled names (it was not sent, and is known all the same).
    assert held["rejected"] == {"  have h : 0 ≤ a := by positivity\n  linarith [h]": 0, "  have h_g1 : 0 ≤ a := by positivity\n  linarith [h_g1]": 0}
    assert held["assembled"] == {"  have h_g1 : 0 ≤ a := by positivity\n  linarith [h_g1]": 0}
    # Generation 2 continued and left nothing: the pool has not changed since, and generation 3 is fresh.
    chain.append(attempt(A, "p", 0, 1, BAD, "  have h_g1 : 0 ≤ a := by positivity\n  simp\n", CONTINUE, "resume"))
    assert rules.kind_of(rules.episode_state(chain, pools, own)) == FRESH
    # Generation 3's harvest was taken back out: the pool is what it was, and still unchanged.
    chain.append(attempt(A, "p", 0, 2, BAD, "  have q : 0 ≤ b := by weird\n  simp\n"))
    pools.append(pool_state("p", 0, 2, [first, lemma("  have q_g3 : 0 ≤ b := by weird", "q_g3", ": 0 ≤ b", 3)], stands=False, outcome=cut.ERROR_BEFORE_THE_SORRY))
    held = rules.episode_state(chain, pools, own)
    assert (held["pool"], held["made_at"], rules.kind_of(held)) == ([first], 0, FRESH)
    # Generation 4's harvest stands: the pool has changed since the last continuing generation, and generation 5 continues.
    chain.append(attempt(A, "p", 0, 3, "timeout", "  have k : 0 ≤ b := by linarith\n  slow\n"))
    pools.append(pool_state("p", 0, 3, [first, second]))
    own += [closer_check("p", 0, 3, BAD, "  have h_g1 : 0 ≤ a := by positivity\n  have k_g2 : 0 ≤ b := by linarith\n  linarith [h_g1]\n", pool_blocks=2),
            closer_check("p", 0, 3, "no_answer", "the pool gave no answer for this one\n"), closer_check("p", 0, 3, KNOWN_COPY, "  simp\n", sent=False)]
    held = rules.episode_state(chain, pools, own)
    assert (held["pool"], held["made_at"], held["changed"], rules.kind_of(held)) == ([first, second], 3, True, CONTINUE)
    # A closer check Lean rejected is a rejected text too; one Lean gave no answer for, and a known copy, add nothing.
    assert set(held["assembled"]) == {"  have h_g1 : 0 ≤ a := by positivity\n  linarith [h_g1]", "  have h_g1 : 0 ≤ a := by positivity\n  have k_g2 : 0 ≤ b := by linarith\n  linarith [h_g1]"}
    assert len(held["rejected"]) == 6 and held["rejected"]["  have k : 0 ≤ b := by linarith\n  slow"] == 3 and "  simp" not in held["assembled"]
    assert rules.is_refused(own[0]) and rules.is_refused(own[1]) and not rules.is_refused(own[2]) and not rules.is_refused(own[3])


def test_the_episode_rows_say_how_and_when_each_arm_resolved():
    problems = [{"problem_id": "p", "group": "below", "side": "true"}]
    block = lemma("  have h_g1 : 0 ≤ a := by positivity", "h_g1", ": 0 ≤ a")
    assembled = "  have h_g1 : 0 ≤ a := by positivity\n  have k_g3 : 0 ≤ b := by linarith\n  linarith [h_g1]\n"
    rows = [attempt(FIRST, "p", 0, 0, BAD, kept_closer={"text": "  linarith [h_g1]", "needs": ["h_g1"], "generation": 1}), attempt(FIRST, "p", 1, 0, OK, "  simp\n  linarith\n"),
            attempt(FIRST, "p", 2, 0, BAD), attempt(FIRST, "p", 0, 0, OK, audit=True),
            *(attempt(B, "p", 0, loop, BAD) for loop in range(1, 8)),
            attempt(A, "p", 0, 1, BAD, kind=CONTINUE, how="resume", pool_blocks=1, shared=False), attempt(A, "p", 0, 2, KNOWN_COPY),
            *(attempt(B, "p", 2, loop, BAD if loop < 4 else OK) for loop in range(1, 5)),
            *(attempt(A, "p", 2, loop, BAD) for loop in range(1, 3)), attempt(A, "p", 2, 3, OK, "  have h_g1 : 0 ≤ a := by positivity\n\n  -- so\n  linarith\n", CONTINUE, "resume", pool_blocks=1)]
    pools = [pool_state("p", 0, 0, [block]), pool_state("p", 0, 2, [block, block], harvested=1), pool_state("p", 2, 1, [block]), pool_state("p", 2, 2, [block, block], False, "several_goals")]
    closers = [closer_check("p", 0, 0, OWN_PROOF, "own\n", sent=False), closer_check("p", 0, 2, BAD, "other\n", pool_blocks=2), closer_check("p", 0, 2, OK, assembled, pool_blocks=2)]
    one, two, three = rules.episode_rows(problems, rows, pools, closers, GENERATIONS)
    # Episode 0: the blind arm never resolves it; the accumulate arm does at generation 3, by a kept closer that verified after the pool grew.
    assert (one["failed_first"], one["arms"][B]["resolved_at"], one["arms"][B]["resolved_by"], len(one["arms"][B]["loops"])) == (True, None, None, 7)
    own = one["arms"][A]
    assert (own["resolved_at"], own["resolved_by"], own["assembled_proof"], own["proof_lines"], own["pool_blocks"], own["kept_closers"]) == (3, ASSEMBLED, assembled, 3, 2, 1)
    assert [(step["loop"], step["kind"], step["how"], step["status"], step["pool_blocks"], step["shared"]) for step in own["loops"]] == [
        (1, CONTINUE, "resume", BAD, 1, False), (2, FRESH, "blind", KNOWN_COPY, 0, True)]
    assert own["pool_checks"] == [{"loop": 0, "outcome": STANDS, "stands": True, "blocks": 1, "harvested": 1}, {"loop": 2, "outcome": STANDS, "stands": True, "blocks": 2, "harvested": 1}]
    assert [(check["loop"], check["status"], check["sent_to_lean"]) for check in own["closer_checks"]] == [(0, OWN_PROOF, False), (2, BAD, True), (2, OK, True)]
    # Episode 1: the shared first generation verified: both arms resolved it there, with the same proof.
    assert two["failed_first"] is False and all((two["arms"][arm]["resolved_at"], two["arms"][arm]["resolved_by"], two["arms"][arm]["proof_lines"], two["arms"][arm]["loops"]) == (1, FRESH, 2, [])
                                                for arm in ARMS)
    # Episode 2: the blind arm at generation 5 by a fresh proof, the accumulate arm at 4 by a continuing one; a harvest taken back out leaves the pool.
    assert (three["arms"][B]["resolved_at"], three["arms"][B]["resolved_by"]) == (5, FRESH)
    assert (three["arms"][A]["resolved_at"], three["arms"][A]["resolved_by"], three["arms"][A]["proof_lines"], three["arms"][A]["pool_blocks"], three["arms"][A]["assembled_proof"]) == (4, CONTINUE, 2, 1, None)
    assert [check["stands"] for check in three["arms"][A]["pool_checks"]] == [True, False]
    assert [read.resolved(one, A, within) for within in (2, 3)] == [False, True] and read.resolved(three, B, 4) is False
    assert rules.problem_rows([one, two, three], GENERATIONS) == [{"problem_id": "p", "group": "below", "side": "true", "episodes": 3, "failed_first": 2, f"resolved_{B}": 2,
                                                                 f"resolved_{A}": 3, f"{CONTINUE}_{A}": 1, f"{ASSEMBLED}_{A}": 1, "with_a_pool": 2}]
    with pytest.raises(ValueError, match="whole episodes"):
        rules.episode_rows(problems, [row for row in rows if not (row["arm"] == B and row["episode"] == 0 and row["loop"] == 7)], pools, closers, GENERATIONS)
    with pytest.raises(ValueError, match="whole episodes"):                                 # the accumulate arm, with no closer that resolved it
        rules.episode_rows(problems, rows, pools, closers[:2], GENERATIONS)


# ------------------------------------------------------------------------------------------------ the read
SEVEN_BAD = (BAD,) * 7


def generation(loop, status, kind=FRESH, tokens=100, prompt=300, pool_blocks=0, no_state=None):
    how = "resume" if kind == CONTINUE and no_state is None else "blind"
    return {"loop": loop, "kind": kind, "how": how, "had_state": how == "resume", "no_state": no_state, "status": status, "token_count": tokens, "prompt_tokens": prompt,
            "sent_to_lean": status != KNOWN_COPY, "pool_blocks": pool_blocks if how == "resume" else 0, "shared": kind == FRESH}


def blind(*statuses, lines=2):
    steps = [generation(index + 1, status) for index, status in enumerate(statuses)]
    at = next((step["loop"] + 1 for step in steps if step["status"] == OK), None)
    return {"resolved_at": at, "resolved_by": FRESH if at else None, "proof_lines": lines if at else None, "loops": steps}


def accumulating(*statuses, continues=(1,), assembled_at=None, lines=2, pool=1, tokens=100):
    """The accumulate arm after a failed first generation: its generations (the ones at the loops `continues` have the
    pool in their prompt), a pool that stood after generation 1 (`pool` blocks; 0: none), and (`assembled_at`, a
    generation) a kept closer that verified after that generation's harvest."""
    steps = [generation(index + 1, status, CONTINUE if pool and index + 1 in continues else FRESH, tokens=tokens, pool_blocks=pool) for index, status in enumerate(statuses)]
    at = next((step["loop"] + 1 for step in steps if step["status"] == OK), None)
    by = None if at is None else CONTINUE if steps[at - 2]["how"] == "resume" else FRESH
    if assembled_at is not None:
        at, by = assembled_at, ASSEMBLED
    checks = [{"loop": 0, "status": OWN_PROOF, "sent_to_lean": False, "from_generation": 1, "pool_blocks": pool}] if pool else []
    if assembled_at is not None:
        checks.append({"loop": assembled_at - 1, "status": OK, "sent_to_lean": True, "from_generation": 1, "pool_blocks": pool})
    return {"resolved_at": at, "resolved_by": by, "proof_lines": lines if at else None, "loops": steps, "assembled_proof": "  the pool\n  the closer\n" if by == ASSEMBLED else None,
            "pool_blocks": pool, "pool_checks": [{"loop": 0, "outcome": STANDS, "stands": True, "blocks": pool, "harvested": pool}] if pool else [],
            "kept_closers": 1 if pool else 0, "closer_checks": checks}


def episode(problem, group, blind_arm=SEVEN_BAD, accumulate_arm=SEVEN_BAD, number=0, first=BAD):
    """One episode. With a verified first generation the arms have nothing to do; else each arm's statuses, loop by loop."""
    if first == OK:
        arms = {B: {"resolved_at": 1, "resolved_by": FRESH, "proof_lines": 1, "loops": []},
                A: {"resolved_at": 1, "resolved_by": FRESH, "proof_lines": 1, "loops": [], "assembled_proof": None, "pool_blocks": 0, "pool_checks": [], "kept_closers": 0, "closer_checks": []}}
    else:
        arms = {B: blind(*blind_arm) if not isinstance(blind_arm, dict) else blind_arm,
                A: accumulating(*accumulate_arm) if not isinstance(accumulate_arm, dict) else accumulate_arm}
    return {"problem_id": problem, "group": group, "side": "true", "episode": number, "failed_first": first != OK,
            "first": {"status": first, "token_count": 90, "prompt_tokens": 200, "sent_to_lean": True, "proof_lines": 1 if first == OK else None}, "arms": arms}


def mixed():
    return [episode("g1", "goal", SEVEN_BAD, (OK,)), episode("g1", "goal", (BAD, OK), SEVEN_BAD, number=1), episode("g1", "goal", first=OK, number=2),
            episode("g1", "goal", number=3),
            episode("b1", "below", SEVEN_BAD, accumulating(BAD, BAD, assembled_at=3, lines=11)), episode("b1", "below", SEVEN_BAD, (BAD, BAD, BAD, BAD, BAD, OK), number=1),
            episode("i1", "in", (OK,), SEVEN_BAD), episode("a1", "above", (OK,), (OK,))]


def test_the_primary_is_accumulate_minus_blind_on_the_hard_problems_paired_by_episode():
    found = rules.accumulate_read(mixed(), GENERATIONS, 500, 0)
    primary = found["primary"]
    # g1: (1 - 0, 0 - 1, 1 - 1, 0 - 0) / 4 = 0; b1: (1 + 1) / 2 = 1. The mean over the two problems is 0.5.
    assert (primary["problems"], primary["episodes"], primary["mean"]) == (2, 6, 0.5) and primary[A] == 0.75 and primary[B] == 0.25
    assert primary == read.paired(read._of(mixed(), ("goal", "below")), A, B, GENERATIONS, 500, 0)       # L3a's pairing, resamples and generator
    within = found["within_generations"]
    assert set(within) == {"goal", "below", "in", "above", "hard", "all"} and all(set(by) == {str(count) for count in range(2, 9)} for by in within.values())
    assert within["hard"]["8"] == primary and [within["hard"][count]["mean"] for count in ("2", "3", "6", "7")] == [0.125, 0.25, 0.25, 0.5]
    assert within["in"]["8"]["mean"] == -1.0 and within["above"]["8"]["mean"] == 0.0 and within["goal"]["8"]["mean"] == 0.0


def test_the_arms_are_compared_by_running_totals():
    totals = rules.accumulate_read(mixed(), GENERATIONS, 200, 0)["running_totals"]["hard"]
    assert set(totals) == {str(count) for count in range(1, 9)}
    assert totals["1"] == {A: 1, B: 1, "lead": 0, f"only_{A}": 0, f"only_{B}": 0, "sign_test_p": None}                 # the shared first generation
    assert totals["2"] == {A: 2, B: 1, "lead": 1, f"only_{A}": 1, f"only_{B}": 0, "sign_test_p": sign_test(1, 0)}
    assert totals["3"] == {A: 3, B: 2, "lead": 1, f"only_{A}": 2, f"only_{B}": 1, "sign_test_p": sign_test(2, 1)}       # the assembled proof is within generation 3
    assert (totals["7"][A], totals["8"][A], totals["8"][B], totals["8"]["lead"]) == (4, 4, 2, 2)
    # A total never falls, and no number of this read is a rate among the episodes an arm has left.
    assert all(totals[str(count)][arm] <= totals[str(count + 1)][arm] for count in range(1, 8) for arm in ARMS)
    resolved = rules.accumulate_read(mixed(), GENERATIONS, 200, 0)["resolved_within_generations"]["hard"]
    assert resolved[A]["8"] == 0.75 and resolved[B]["8"] == 0.25 and resolved[A]["1"] == resolved[B]["1"] == 0.125


def test_reach_is_the_goal_set_by_problem_as_gained_against_lost_with_the_sign_test():
    rows = [episode("g1", "goal", SEVEN_BAD, (OK,)), episode("g1", "goal", number=1), episode("g2", "goal", (BAD, OK), SEVEN_BAD),
            episode("g3", "goal", (OK,), (OK,)), episode("g4", "goal", SEVEN_BAD, accumulating(BAD, BAD, BAD, assembled_at=4)), episode("g5", "goal"),
            episode("b1", "below", SEVEN_BAD, (OK,))]
    reach = rules.accumulate_read(rows, GENERATIONS, 200, 0)["reach"]
    assert reach == {"problems": 5, f"resolved_{A}": 3, f"resolved_{B}": 2, "by_both": 1, "gained": 2, "lost": 1, "sign_test_p": sign_test(2, 1)}
    assert rules.reach_is_ahead(reach) is False and rules.reach_is_ahead({"gained": 9, "lost": 1, "sign_test_p": sign_test(9, 1)}) is True
    assert rules.reach_is_ahead({"gained": 1, "lost": 9, "sign_test_p": sign_test(1, 9)}) is False and rules.reach_is_ahead({"gained": 0, "lost": 0, "sign_test_p": None}) is False


def test_how_the_resolutions_came_and_how_long_the_verified_proofs_are():
    found = rules.accumulate_read(mixed(), GENERATIONS, 200, 0)
    assert found["resolutions"]["hard"] == {"episodes": 6, "resolved": 4, FRESH: 2, "of_which_at_the_first_generation": 1, CONTINUE: 1, ASSEMBLED: 1, f"resolved_{B}": 2}
    assert found["resolutions"]["in"][f"resolved_{B}"] == 1 and found["resolutions"]["in"]["resolved"] == 0
    lines = found["verified_proof_lines"]["hard"]
    # The accumulate arm verified four proofs: of 2 lines (a continuing generation), 1 (the shared first generation), 11 (assembled) and 2 (a
    # fresh proof at generation 7); the blind arm two.
    assert lines[A] == {"proofs": 4, "median": 2.0, "mean": 4.0, "share_with_8_lines_or_more": 0.25, "longest": 11,
                        "after_the_first_generation": {"proofs": 3, "median": 2, "mean": 5.0, "share_with_8_lines_or_more": 0.33333, "longest": 11},
                        "by_how": {FRESH: {"proofs": 1, "median": 2, "mean": 2.0, "share_with_8_lines_or_more": 0.0, "longest": 2},
                                   CONTINUE: {"proofs": 1, "median": 2, "mean": 2.0, "share_with_8_lines_or_more": 0.0, "longest": 2},
                                   ASSEMBLED: {"proofs": 1, "median": 11, "mean": 11.0, "share_with_8_lines_or_more": 1.0, "longest": 11}}}
    assert (lines[B]["proofs"], lines[B]["median"], lines[B]["share_with_8_lines_or_more"], lines[B]["after_the_first_generation"]["proofs"]) == (2, 1.5, 0.0, 1)


def test_what_the_arm_did_is_given_by_generation_and_is_no_comparison_of_the_arms():
    rows = mixed() + [episode("g2", "goal", (BAD, KNOWN_COPY, BAD, BAD, BAD, BAD, BAD), accumulating(*SEVEN_BAD, pool=0))]
    found = rules.accumulate_read(rows, GENERATIONS, 200, 0)
    by = found["by_generation"]["hard"]
    assert by[B]["2"] == {"generations": 6, "verified": 0, "known_copies": 0} and by[B]["3"] == {"generations": 6, "verified": 1, "known_copies": 1}
    # Generation 2 of the accumulate arm: five episodes held a pool and continued (one verified); the sixth had none and drew the blind arm's sample.
    assert by[A]["2"] == {"generations": 6, "verified": 1, "known_copies": 0, FRESH: 1, CONTINUE: 5, "continuing_with_the_pool_in_the_prompt": 5, "of_which_verified": 1,
                          "of_which_known_copies": 0, "shared_with_the_blind_arm": 1, "resolved_by_an_assembled_proof_after_it": 0}
    assert (by[A]["3"]["generations"], by[A]["3"]["resolved_by_an_assembled_proof_after_it"], by[A]["3"][CONTINUE]) == (5, 1, 0)
    # The first continuing generation, like for like: the same failed first generations behind both arms.
    second = found["second_generation"]["hard"]
    assert second == {"failed_first_generations": 6, "with_the_pool_in_the_second_prompt": 5, f"verified_{A}": 1, f"verified_{B}": 0, f"only_{A}": 1, f"only_{B}": 0,
                      "sign_test_p": sign_test(1, 0), "known_copies": 0}
    pools = found["pools"]["hard"]
    assert (pools["episodes"], pools["with_a_pool_at_the_end"], pools["share_with_a_pool_at_the_end"], pools["pool_checks"], pools["harvests_taken_back_out"]) == (7, 5, 0.71429, 5, 0)
    assert pools["closer_checks_by_status"] == {OWN_PROOF: 5, OK: 1} and pools["closer_checks_sent_to_lean"] == 1 and pools["episodes_resolved_by_an_assembled_proof"] == 1
    assert pools["kept_closers"] == 5 and pools["lemmas_pooled"] == 5 and pools["mean_blocks_where_there_is_a_pool"] == 1.0


# -------------------------------------------------------------------------------------------------- budget
def test_the_budget_counts_generations_tokens_and_lean_checks_by_arm():
    own = accumulating(BAD, KNOWN_COPY, BAD, BAD, BAD, BAD, BAD, continues=(1, 3), tokens=40)
    own["pool_checks"].append({"loop": 2, "outcome": "state_timeout", "stands": False, "blocks": 2, "harvested": 1})
    own["closer_checks"] += [{"loop": 2, "status": BAD, "sent_to_lean": True, "from_generation": 1, "pool_blocks": 1}, {"loop": 4, "status": KNOWN_COPY, "sent_to_lean": False, "from_generation": 1, "pool_blocks": 1}]
    rows = [episode("g1", "goal", (BAD, KNOWN_COPY, "capped_tokens", BAD, BAD, BAD, BAD), own), episode("a1", "above", first=OK)]
    rows[0]["arms"][B]["loops"][2]["sent_to_lean"] = False                                  # the capped attempt was not sent either
    spent = rules.budget(rows, read._of(rows, ("goal",)), GENERATIONS, 200, 0)
    assert spent["first_generations"] == {"generations": 2, "generated_tokens": 180, "prompt_tokens": 400, "lean_checks": 2}
    assert spent["by_arm"][B] == {"generations": 7, "generated_tokens": 700, "prompt_tokens": 2100, "attempts_sent_to_lean": 5, "pool_checks": 0, "closer_checks": 0,
                                  "lean_checks": 5, "known_copies": 1}
    # The accumulate arm: six attempts sent, two pool checks (the one after generation 1 and one that timed out) and one closer check
    # sent (the proof's own and the known copy were not).
    assert spent["by_arm"][A] == {"generations": 7, "generated_tokens": 280, "prompt_tokens": 2100, "attempts_sent_to_lean": 6, "pool_checks": 2, "closer_checks": 1,
                                  "lean_checks": 9, "known_copies": 1}
    assert spent["by_arm_on_the_primarys_problems"] == spent["by_arm"]
    assert spent["generated_tokens_accumulate_over_blind_on_the_primarys_problems"] == 0.4 and spent["lean_checks_accumulate_over_blind_on_the_primarys_problems"] == 1.8
    through = rules.arm_spend(rows, A, through=1)                                            # the arm read at its first generation after the first alone
    assert (through["generations"], through["generated_tokens"], through["pool_checks"], through["closer_checks"], through["lean_checks"]) == (1, 40, 1, 0, 2)
    assert rules.arm_spend(rows, A, through=2)["lean_checks"] == 1 + 1 + 1 + 1                # one more attempt was a known copy; a pool check and a closer check


def test_the_read_at_equal_tokens_is_made_only_when_the_accumulate_arm_generates_more_than_ten_percent_more():
    def rows(tokens):
        return [episode("g1", "goal", SEVEN_BAD, accumulating(BAD, BAD, BAD, BAD, BAD, BAD, OK, tokens=tokens)),
                episode("b1", "below", (BAD, BAD, BAD, BAD, BAD, BAD, OK), accumulating(OK, tokens=tokens))]
    same = rules.accumulate_read(rows(110), GENERATIONS, 200, 0)["budget"]
    assert same["generated_tokens_accumulate_over_blind_on_the_primarys_problems"] == 0.6286 and same["accumulate_generates_more_than_10_percent_more"] is False
    assert isinstance(same["equal_tokens"], str) and "read as it stands" in same["equal_tokens"]
    little = rules.accumulate_read(rows(185), GENERATIONS, 200, 0)["budget"]          # 8 x 185 against 14 x 100 tokens: 5.7% more is within the 10%
    assert little["generated_tokens_accumulate_over_blind_on_the_primarys_problems"] == 1.0571 and little["accumulate_generates_more_than_10_percent_more"] is False
    assert "read as it stands" in little["equal_tokens"]
    more = rules.accumulate_read(rows(350), GENERATIONS, 200, 0)["budget"]            # 8 x 350 against 14 x 100 tokens
    assert more["generated_tokens_accumulate_over_blind_on_the_primarys_problems"] == 2.0 and more["accumulate_generates_more_than_10_percent_more"] is True
    equal = more["equal_tokens"]
    assert equal["blind_generations_after_the_first_that_match"] == 14.0 and equal["blind_generations_after_the_first_that_were_sampled"] == 7
    assert equal["the_blind_arm_at_the_matching_number_can_be_read"] is False
    # Its mirror, as in L3a2: the accumulate arm at the loops whose tokens match the blind arm's 1,400 (three loops are 1,400 tokens: within 10%).
    assert equal["mirror"]["accumulate_loops_read"] == 3 and equal["mirror"]["accumulate_generated_tokens"] == 1400 and equal["mirror"]["blind_generated_tokens"] == 1400
    assert equal["mirror"]["primary"]["mean"] == 0.0 and equal["mirror"]["primary"][A] == 0.5 and equal["mirror"]["primary"][B] == 0.5


# ------------------------------------------------------------------------------- can this run see a win
def hard(accumulate_wins, blind_wins, total=12, with_a_pool=6):
    """`total` goal problems, one episode each; the first `with_a_pool` hold a pool after generation 1 and continue at
    generation 2. The blind arm resolves the first `blind_wins` at its generation 8, the accumulate arm the LAST
    `accumulate_wins` at its own generation 8."""
    late = (BAD,) * 6 + (OK,)
    return [episode(f"g{index}", "goal", late if index < blind_wins else SEVEN_BAD,
                    accumulating(*(late if total - index <= accumulate_wins else SEVEN_BAD), pool=1 if index < with_a_pool else 0)) for index in range(total)]


def test_all_three_checks_must_pass_and_any_failing_makes_the_run_inconclusive():
    fine = rules.accumulate_read(hard(8, 0, total=10, with_a_pool=3), GENERATIONS, 500, 0)          # a pool in 3 of 10 hard episodes: exactly 30%
    checks = fine["can_this_run_see_a_win"]
    assert checks["passes"] is True and checks["a_pool_in_the_hard_episodes"] == {
        "hard_episodes": 10, "with_a_pool_by_their_last_generation": 3, "share": 0.3, "hard_episodes_whose_first_generation_failed": 10, "share_of_those": 0.3,
        "needed": "at least 30%", "passes": True}
    assert checks["a_state_for_the_continuing_generations"] == {"continuing_generations": 3, "with_a_state": 3, "share": 1.0, "without_a_state_by_reason": {},
                                                                "needed": "at least half", "passes": True}
    assert checks["the_pool_stands"] == {"pool_checks": 3, "stand": 3, "share": 1.0, "by_outcome": {STANDS: 3}, "needed": "at least 95%", "passes": True}
    assert fine["branch"]["name"] == THE_SOLVERS_EPISODE

    few = rules.accumulate_read(hard(8, 0, total=10, with_a_pool=2), GENERATIONS, 500, 0)           # 2 of 10: the episode is too short for a pool, or the harvest is broken
    assert few["can_this_run_see_a_win"]["a_pool_in_the_hard_episodes"]["passes"] is False and few["branch"]["name"] == INCONCLUSIVE
    assert "a pool held a lemma in 0.2 of the hard episodes" in few["branch"]["reason"] and "not a verdict" in few["branch"]["reason"]
    # An episode the first generation resolved counts among the hard episodes: it has no pool.
    counted = rules.can_see_a_win(hard(0, 0, total=7, with_a_pool=3) + [episode(f"r{index}", "goal", first=OK) for index in range(3)])["a_pool_in_the_hard_episodes"]
    assert (counted["hard_episodes"], counted["share"], counted["hard_episodes_whose_first_generation_failed"], counted["share_of_those"], counted["passes"]) == (10, 0.3, 7, 0.42857, True)
    short = rules.can_see_a_win(hard(0, 0, total=7, with_a_pool=3) + [episode(f"r{index}", "goal", first=OK) for index in range(4)])["a_pool_in_the_hard_episodes"]
    assert (short["hard_episodes"], short["share"], short["share_of_those"], short["passes"]) == (11, 0.27273, 0.42857, False)           # 3 of 11, whatever the share of the 7

    # A state for fewer than half of the continuing generations: the prompt did not hold the pool.
    rows = hard(8, 0)
    for row in rows[2:6]:
        row["arms"][A]["loops"][0].update({"how": "blind", "had_state": False, "no_state": cut.PROMPT_TOO_LONG, "pool_blocks": 0})
    stateless = rules.accumulate_read(rows, GENERATIONS, 500, 0)
    states = stateless["can_this_run_see_a_win"]["a_state_for_the_continuing_generations"]
    assert (states["continuing_generations"], states["with_a_state"], states["share"], states["passes"]) == (6, 2, 0.33333, False)
    assert states["without_a_state_by_reason"] == {cut.PROMPT_TOO_LONG: 4} and "a state came back for 0.33333 of the continuing generations" in stateless["branch"]["reason"]

    # The pool stands in fewer than 95% of its checks: the harvest takes steps that are not lemmas.
    rows = hard(8, 0)
    rows[0]["arms"][A]["pool_checks"].append({"loop": 3, "outcome": cut.ERROR_BEFORE_THE_SORRY, "stands": False, "blocks": 2, "harvested": 1})
    falls = rules.accumulate_read(rows, GENERATIONS, 500, 0)
    stands = falls["can_this_run_see_a_win"]["the_pool_stands"]
    assert (stands["pool_checks"], stands["stand"], stands["share"], stands["passes"]) == (7, 6, 0.85714, False) and stands["by_outcome"] == {STANDS: 6, cut.ERROR_BEFORE_THE_SORRY: 1}
    assert falls["branch"]["name"] == INCONCLUSIVE and "the pool stood in 0.85714 of the checks made of it" in falls["branch"]["reason"]

    # A check that cannot be read has not passed: no hard episode, no continuing generation, no pool check.
    assert rules.accumulate_read([episode("a1", "above", (OK,), (OK,))], GENERATIONS, 200, 0)["branch"]["name"] == INCONCLUSIVE
    assert rules.can_see_a_win(hard(8, 0, with_a_pool=0))["passes"] is False and rules.can_see_a_win([])["passes"] is False


def test_the_branch_is_read_from_the_primarys_interval_and_from_reach_in_the_specs_words():
    won = rules.accumulate_read(hard(8, 0), GENERATIONS, 500, 0)["branch"]
    assert won["name"] == THE_SOLVERS_EPISODE and "the accumulating episode is the solver's episode from here" in won["reason"] and "L3d trains on the proofs it assembles" in won["reason"]
    level = rules.accumulate_read(hard(3, 3), GENERATIONS, 500, 0)
    assert level["primary"]["mean"] == 0.0 and level["primary"]["low"] < 0 < level["primary"]["high"]
    assert level["branch"]["name"] == NOT_SHOWN and "reach is not ahead" in level["branch"]["reason"] and "the size of the pool is the next thing to vary" in level["branch"]["reason"]
    assert "which needs a decision" in level["branch"]["reason"] and "gained 3, lost 3" in level["branch"]["reason"]
    checks, zero = {"passes": True}, {"mean": 0.004, "low": -0.003, "high": 0.011}
    ahead, behind, even = ({"gained": 12, "lost": 2, "sign_test_p": sign_test(12, 2)}, {"gained": 2, "lost": 12, "sign_test_p": sign_test(2, 12)},
                           {"gained": 8, "lost": 4, "sign_test_p": sign_test(8, 4)})
    # Reach ahead (gained more than lost, sign test under 0.05) with the primary's interval through zero.
    reaches = rules.accumulate_branch(zero, ahead, checks)
    assert reaches.name == REACHES and "reaches problems without resolving more episodes" in reaches.reason and "L3d is still the next step" in reaches.reason
    assert rules.accumulate_branch(zero, even, checks).name == NOT_SHOWN and rules.accumulate_branch(zero, behind, checks).name == NOT_SHOWN       # p = 0.39; and behind is not ahead
    assert rules.accumulate_branch({"mean": 0.02, "low": 0.001, "high": 0.05}, behind, checks).name == THE_SOLVERS_EPISODE              # the primary decides first
    assert rules.accumulate_branch({"mean": 0.02, "low": 0.0, "high": 0.05}, even, checks).name == NOT_SHOWN                            # an interval that touches zero is not clear of it
    # Below zero with an interval clear of zero: the spec names no branch for it; it is reported as that, whatever reach says.
    below = rules.accumulate_branch({"mean": -0.02, "low": -0.05, "high": -0.001}, ahead, checks)
    assert below.name == RESOLVES_FEWER and "The spec names no branch for this" in below.reason and "needs a decision" in below.reason
    assert rules.accumulate_branch({"mean": -0.02, "low": -0.05, "high": 0.0}, even, checks).name == NOT_SHOWN
    assert rules.accumulate_branch({"mean": None, "low": None, "high": None}, even, checks).name == INCONCLUSIVE
    assert set(rules.BRANCHES) == {THE_SOLVERS_EPISODE, NOT_SHOWN, REACHES, RESOLVES_FEWER, INCONCLUSIVE} and rules.REACH_AHEAD_BELOW == 0.05


# -------------------------------------------------------------------------------------------------- report
PREPARE = {"seed": 0, "fixture": False, "stand_in_engine": False, "goal_set": 12, "rungs": {"below": 0, "in": 1, "above": 1},
           "sizes": {"episodes": {"goal": 6, "below": 6, "in": 2, "above": 2}, "pilot_problems": None, "generations": GENERATIONS, "loops": GENERATIONS, "pool_blocks": 12,
                     "kept_closers": 8, "max_new_tokens": 1024, "lean_seconds": 30, "sampling_seed": 1032},
           "problems_with_no_side_to_attempt": []}
STEP = {"problems": 14, "episodes": 14, "attempts": 120, "generated_tokens": 1000, "lean_checks_sent": 150, "loops": [], "stand_in_engine": False}
EVALUATION = {"bootstrap_resamples": 300, "bootstrap_seed": 0}
EASIER = [episode("i1", "in", (OK,), SEVEN_BAD), episode("a1", "above", (OK,), (OK,))]
# The shortest published proof of each goal problem, as the report step reads it: g0 to g2 one line, g3 to g5 2-3, g6 to g8 4-7, g9 to g11 8 or more.
LENGTHS = {f"g{index}": {"problem_id": f"g{index}", "shortest_published_proof_lines": lines, "length_group": length_group(lines)}
           for index, lines in enumerate((1, 1, 1, 2, 3, 3, 4, 5, 7, 8, 9, 20))}


def test_the_report_answers_the_read_and_names_the_branch():
    report = build_l3c_report(PREPARE, hard(8, 1) + EASIER, STEP, {}, EVALUATION, LENGTHS)
    assert report["branch"]["name"] == THE_SOLVERS_EPISODE and report["inconclusive"] is False and report["ok"] is True and (report["seed"], report["pilot"]) == (0, False)
    assert report["headline"].startswith("L3c seed 0: THE ACCUMULATING EPISODE IS THE SOLVER'S EPISODE FROM HERE. Primary (G and the below-band rung, 12 problems, 12 "
                                         "episodes resolved within 8 generations, accumulate minus blind): ")
    # Of the 12: the blind arm resolves g0 at generation 8, the accumulate arm g4 to g11 at generation 8.
    for needed in ("reach, G by problem (12), accumulate against blind: 8 to 1, gained 8, lost 1 (p = 0.0390625)",
                   "where the shortest published proof is 4 lines or more (6): 6 to 0, gained 6, lost 0 (p = 0.03125)",
                   "hard episodes resolved within k generations, accumulate/blind: 2: 0/0, 3: 0/0, 4: 0/0, 5: 0/0, 6: 0/0, 7: 0/0, 8: 8/1",
                   "within 2 generations 0.0 [0.0, 0.0]", "within 7 generations 0.0 [0.0, 0.0]",
                   "by the shortest published proof: 1 line -0.33333 [", "2-3 lines 0.66667 [", "4-7 lines 1.0 [1.0, 1.0], 8+ lines 1.0 [1.0, 1.0]",
                   "the accumulate arm's resolutions on these problems: fresh 8 (0 at the first generation), continuing 0, assembled 0",
                   "lines of the verified proofs (median, share with 8 or more): accumulate 2.0, 0.0; blind 2, 0.0",
                   "the in-band rung -1.0 [", "the above-band rung 0.0 [", "a pool in 0.5 of the hard episodes by their last generation",
                   "a state for 1.0 of the continuing generations", "the pool stands in 1.0 of its checks", "known copies not sent to Lean: accumulate 0, blind 0",
                   "generated tokens, accumulate over blind: 1.0", "Lean checks, accumulate over blind: 1.0714"):        # 84 attempts each; 6 pool checks
        assert needed in report["headline"], needed
    assert "within 8 generations" not in report["headline"].split("accumulate/blind")[1]                 # the last one is the primary
    assert report["primary"]["mean"] == round(7 / 12, 5) and "paired by episode" in report["primary"]["what"] and "accumulate minus blind" in report["primary"]["what"]
    assert report["within_generations"]["hard"]["8"] == {key: value for key, value in report["primary"].items() if key != "what"}
    assert report["reach"]["gained"] == 8 and report["reach"]["at_4_lines_or_more"]["gained"] == 6 and "read on all of G" in report["reach"]["what"]
    by_length = report["by_proof_length"]
    assert by_length["hard_problems"] == {"1": 3, "2-3": 3, "4-7": 3, "8+": 3} and set(by_length["primary"]) == set(LENGTH_GROUPS) == set(by_length["goal_set_by_problem"])
    assert [by_length["primary"][group]["problems"] for group in LENGTH_GROUPS] == [3, 3, 3, 3] and by_length["goal_set_by_problem"]["1"]["lost"] == 1
    assert by_length["goal_set_by_problem_at_4_lines_or_more"] == {key: value for key, value in report["reach"]["at_4_lines_or_more"].items()}
    assert report["running_totals"]["hard"]["8"][A] == 8 and "never by a rate among the episodes each arm has left" in report["running_totals"]["what"]
    assert report["by_generation"]["what"].startswith("NOT a comparison of the arms.") and "by_generation" not in report["headline"]
    assert report["budget"]["by_arm_on_the_primarys_problems"][A]["pool_checks"] == 6 and report["budget"]["by_arm"][A]["pool_checks"] == 8       # two more on the rungs
    assert report["budget"]["by_arm"][B]["pool_checks"] == 0 and "known_copies" in report["budget"]["by_arm"][B]
    assert report["can_this_run_see_a_win"]["passes"] is True and report["pools"]["hard"]["with_a_pool_at_the_end"] == 6
    assert report["heldout"] == {"goal_set": 12, "rungs": {"below": 0, "in": 1, "above": 1}, "the_primarys_problems": 12, "problems_with_no_side_to_attempt": []}
    assert "up to 7 more" in report["naming"] and "known copy" in report["naming"] and "running total" in report["naming"] and report["attempts"][FIRST]["attempts"] == 14
    assert report["sizes"]["sampling_seed"] == 1032 and report["step"]["lean_checks_sent"] == 150 and report["model"].startswith("the base model")
    assert report["not_to_be_read"] == [] and "L3c" in report["spec"]


def test_the_report_says_inconclusive_first_marks_what_is_not_to_be_read_and_names_a_pilot():
    rows = hard(8, 1, with_a_pool=2) + EASIER
    for row in rows[3:6]:
        row["arms"][B]["loops"][0]["status"] = "no_answer"
    report = build_l3c_report(PREPARE, rows, STEP, {}, EVALUATION, LENGTHS)
    assert report["branch"]["name"] == INCONCLUSIVE and report["inconclusive"] is True and report["headline"].startswith("L3c seed 0: INCONCLUSIVE.")
    assert report["not_to_be_read"] == [B] and report["ok"] is False and "NOT TO BE READ: too many attempts without an answer in blind" in report["headline"]
    # A problem the lengths do not hold (the fixture's) is its own group: the report does not guess a length.
    unknown = build_l3c_report(PREPARE, hard(8, 1) + EASIER, STEP, {}, EVALUATION, {key: value for key, value in LENGTHS.items() if key != "g7"})
    assert unknown["by_proof_length"]["hard_problems"] == {"1": 3, "2-3": 3, "4-7": 2, "8+": 3, "not_known": 1} and unknown["by_proof_length"]["primary"]["not_known"]["problems"] == 1
    assert unknown["reach"]["at_4_lines_or_more"]["problems"] == 5 and "8+ lines 1.0 [1.0, 1.0], length not known 1.0 [" in unknown["headline"]
    # The pilot's report says what it is before anything else.
    pilot = build_l3c_report({**PREPARE, "sizes": {**PREPARE["sizes"], "pilot_problems": 40}}, hard(8, 1) + EASIER, STEP, {}, EVALUATION, LENGTHS)
    assert pilot["pilot"] is True and pilot["headline"].startswith("L3c PILOT seed 0: THE ACCUMULATING EPISODE IS THE SOLVER'S EPISODE FROM HERE. THE PILOT'S EPISODES ARE NOT READ AS A RESULT. Primary (")
    # An assembled proof is counted in the headline's resolutions and its lines.
    rows = hard(8, 1) + EASIER
    rows[2]["arms"][A] = accumulating(BAD, BAD, assembled_at=3, lines=14)
    report = build_l3c_report(PREPARE, rows, STEP, {}, EVALUATION, LENGTHS)
    assert "fresh 8 (0 at the first generation), continuing 0, assembled 1" in report["headline"] and "accumulate/blind: 2: 0/0, 3: 1/0," in report["headline"]
    assert report["verified_proof_lines"]["hard"][A]["by_how"][ASSEMBLED]["longest"] == 14 and report["resolutions"]["hard"][ASSEMBLED] == 1
