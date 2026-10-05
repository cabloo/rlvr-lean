"""What the repair check does to ONE failed proof. Spec: docs/spec/ladder-loop.spec.md, "L3a: the repair check, no
training", "How Lean's answer is fed back", items 1 to 3. Pure.

  errors_of             the errors of one Lean answer that carry a position, the first by position first
  leading_error         the error a failed proof is read by (the first in the proof, when it has one)
  plan_cut              where the proof is cut (item 1), or why it cannot be
  state_source          the file that asks Lean for the proof state at the cut (item 2): the kept lines, then
                        `all_goals sorry`
  read_state            the state out of that file's answer, or why none can be had
  state_comment         the state in the prover's own comment format: the ONE place that format is written
  resume_prompt         the prompt (item 3), with the state or without it
  resumed_proof         the proof that is checked: the kept lines, then what the model wrote
  trimmed_proof         the kept lines as a proof of their own, when no goal was left at the cut (item 2a)
  repeats_failed_step   whether what the model wrote begins with the step that had just failed

Positions are Lean's: a line is 1-based in the FILE that was sent (`build_proof_source`: the header, the statement, the
proof, `#print axioms`), a column is 0-based. A proof's lines are the lines of that file's proof part.

THE FORMAT is DeepSeek-Prover-V1.5's own ("truncate and resume"), read 2026-10-05 in its repository
(github.com/deepseek-ai/DeepSeek-Prover-V1.5 at commit 2c4ba9119eef74d0d611f494261b2c5bae98c69a, the commit
`data.deepseek_prover_commit` names):
  prover/lean/proof.py, `Proof.segmentation`, builds the comment:
      newline_with_indent = '\\n' + ' ' * indent_len
      state_comment = newline_with_indent.join([' ' * indent_len + '/- tactic state:',
                                                '  ' + goal.replace('\\n', newline_with_indent + '  '), '-/\\n'])
  prover/algorithms/rmax_tree_search.py, `_tactic_tree_generate_proof`: the prompt is the plain prompt, then the kept
      code (`tactic_code`, whole lines, ending with a newline), then that comment (prover/workers/generator.py joins
      them with nothing between);
  the same file, `_tactic_tree_parse_proof`: `code = code_prefix['tactic_code'] + code`. The proof that is checked is
      the kept code and what the model wrote: the comment is NOT in it. `resumed_proof` does the same.
Where this differs from the authors' search, each because of the spec's own words:
  * their state is every open goal after the last kept tactic (the REPL's `stateAfter`); ours is every goal Lean
    reports at `all_goals sorry` put at the cut (`STATE_PLACEHOLDER`), joined as Lean prints several goals;
  * their comment sits at the indentation of the line the last kept tactic begins on; ours sits where the `sorry` sat
    (the cut line's indentation). The two agree but for a first step inside a bullet (`· tac`), where theirs is the
    bullet's column;
  * they resume only after a finished tactic; the spec keeps every line before the error's line, so a block that
    was opened (`have h : X := by`) and whose first step failed is kept open, and the state is that block's goal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rlvr_lean.domain.proving import build_prover_prompt
from rlvr_lean.domain.verification.lean_source import LEAN_HEADER, _strip_comments, build_proof_source

UNSOLVED_GOALS = "unsolved goals"       # how Lean begins the error of a proof (or a block) that ran out with goals open
# What stands for the rest of the proof in the file that asks for the state (spec item 2, made exact 2026-10-05):
# `all_goals sorry`, so that Lean runs a `sorry` on EVERY goal open at the cut and reports each. A plain `sorry` reports
# only the goal it closes: in the dry run on 600 stored failed attempts that left other open goals unshown in 38 of 592
# states, and the prover's own format shows them all. With no goal open there `all_goals` runs nothing and Lean reports
# no `sorry` at all: that is how "no goals left at the cut" is read (item 2a). `state_source` finds the `sorry` in this.
STATE_PLACEHOLDER = "all_goals sorry"
DEFAULT_INDENTATION = "  "              # of a proof's own steps when the proof holds no line to read it from

BODY = "body"                           # the cut is before a line of the proof
WHOLE_PROOF = "whole_proof"             # the proof ran out with goals open: all of it is kept

# Why a failed proof cannot be cut (the loop is then a blind attempt, counted as "no state").
NO_ERROR_POSITION = "no_error_position"             # Lean named no error with a position (a timeout, the token cap, no answer)
ERROR_ON_THE_STATEMENT = "error_on_the_statement"   # the only errors are on the theorem's own lines and are not its open goals
# Why a cut gave no state (the same: a blind attempt, counted).
STATE_TIMEOUT = "state_timeout"                     # the file that asks for the state ran into the Lean limit
STATE_NO_ANSWER = "state_no_answer"                 # the pool gave no answer for it
ERROR_BEFORE_THE_SORRY = "error_before_the_sorry"   # an error lies before the `sorry`: the kept lines do not stand alone
ERROR_AT_THE_SORRY = "error_at_the_sorry"           # Lean refused the `sorry` itself (it does not fit where the cut put it)
NO_GOALS_AT_THE_CUT = "no_goals_at_the_cut"         # nothing was left to prove there: the kept lines already close the goal
                                                    # (item 2a: they are then checked on their own, and may resolve the episode)
ERROR_AFTER_THE_SORRY = "error_after_the_sorry"     # the file does not end where the `sorry` ends: the cut is inside a structure
NO_SORRY_AT_THE_CUT = "no_sorry_at_the_cut"         # Lean reports a `sorry` somewhere, and none with a goal where ours was put
PROMPT_TOO_LONG = "prompt_too_long"                 # the state came back and the prompt would not fit the model's context
NO_STATE_REASONS = (NO_ERROR_POSITION, ERROR_ON_THE_STATEMENT, STATE_TIMEOUT, STATE_NO_ANSWER, ERROR_BEFORE_THE_SORRY, ERROR_AT_THE_SORRY,
                    NO_GOALS_AT_THE_CUT, ERROR_AFTER_THE_SORRY, NO_SORRY_AT_THE_CUT, PROMPT_TOO_LONG)

_OPENING, _CLOSING = "([{⟨", ")]}⟩"


# ------------------------------------------------------------------------------------------- Lean's answer
def errors_of(raw: Mapping[str, Any]) -> list[dict]:
    """The errors of one check's answer that carry a position, the first by position first: each
    `{line, column, end_line, end_column, text}`. An answer with no `response` (a timeout, no answer) has none."""
    response = raw.get("response")
    if not isinstance(response, Mapping):
        return []
    found = []
    for message in response.get("messages") or []:
        position = message.get("pos")
        if message.get("severity") != "error" or not isinstance(position, Mapping) or position.get("line") is None:
            continue
        end = message.get("endPos") if isinstance(message.get("endPos"), Mapping) else {}
        found.append({"line": int(position["line"]), "column": int(position.get("column") or 0),
                      "end_line": None if end.get("line") is None else int(end["line"]),
                      "end_column": None if end.get("column") is None else int(end["column"]),
                      "text": str(message.get("data", ""))})
    return sorted(found, key=lambda error: (error["line"], error["column"]))


def errors_without_a_position(raw: Mapping[str, Any]) -> int:
    response = raw.get("response")
    if not isinstance(response, Mapping):
        return 0
    return sum(message.get("severity") == "error" and not (isinstance(message.get("pos"), Mapping) and message["pos"].get("line") is not None)
               for message in response.get("messages") or [])


def is_unsolved_goals(error: Mapping) -> bool:
    return str(error["text"]).lstrip().startswith(UNSOLVED_GOALS)


# -------------------------------------------------------------------------------------------- the proof's lines
def proof_first_line(statement: str) -> int:
    """The 1-based line, in the file `build_proof_source` assembles, on which the proof begins. The line before it
    is the one that ends the statement (the theorem's `:= by`)."""
    statement_text = statement if statement.endswith("\n") else statement + "\n"
    return (LEAN_HEADER + statement_text).count("\n") + 1


def proof_lines(proof: str) -> list[str]:
    """The proof's lines as they stand in the file that was checked (which drops the white space at its end)."""
    text = proof.rstrip()
    return text.split("\n") if text else []


def indentation_of(line: str) -> str:
    return line[:len(line) - len(line.lstrip())]


def own_indentation(lines: Sequence[str]) -> str:
    """The indentation of the proof's own steps: that of its first line of code."""
    code = _strip_comments("\n".join(lines)).split("\n")
    for line, stripped in zip(lines, code):
        if stripped.strip():
            return indentation_of(line)
    return DEFAULT_INDENTATION


def opens_unclosed(lines: Sequence[str]) -> int | None:
    """The index of the line that opens the OUTERMOST bracket these lines leave open, or None when they leave none
    open. Comments (`--`, `/- -/`, nested) and string literals are skipped; the kinds of bracket are not matched."""
    stack: list[int] = []
    depth, in_string = 0, False
    for index, line in enumerate(lines):
        position = 0
        while position < len(line):
            pair, character = line[position:position + 2], line[position]
            if depth:
                depth += 1 if pair == "/-" else -1 if pair == "-/" else 0
                position += 2 if pair in ("/-", "-/") else 1
            elif in_string:
                in_string = character != '"'
                position += 2 if character == "\\" else 1
            elif pair == "--":
                break
            elif pair == "/-":
                depth, position = 1, position + 2
            else:
                if character == '"':
                    in_string = True
                elif character in _OPENING:
                    stack.append(index)
                elif character in _CLOSING and stack:
                    stack.pop()
                position += 1
    return stack[0] if stack else None


def step_begins(lines: Sequence[str], index: int) -> int:
    """The line on which the step that fails at line `index` begins. Lean puts some errors on a later line of a
    step that spans several (a name it does not know inside `nlinarith [...,\\n    ...]`), and a proof cut there
    does not parse. Such a step keeps a bracket open across the lines before `index`: it begins on the line that
    opens the outermost one. (The spec's words for the cut: "the same proof cut before the failing step".)"""
    opener = opens_unclosed(lines[:index])
    return index if opener is None else opener


# ------------------------------------------------------------------------------------------------- the cut
@dataclass(frozen=True)
class Cut:
    kind: str                       # BODY or WHOLE_PROOF
    kept: tuple[str, ...]           # the proof lines that are kept
    indentation: str                # where the `sorry`, and the state comment, are put
    failed_step: str | None         # the first line the cut removed (BODY); None when the whole proof is kept
    error: Mapping                  # the error the cut was made at
    moved_back: int = 0             # lines from where the failing step begins to the line Lean put the error on

    @property
    def kept_lines(self) -> int:
        return len(self.kept)


IN_THE_PROOF, AFTER_THE_PROOF, THE_THEOREMS_OWN_GOALS, ON_THE_STATEMENT = "in_the_proof", "after_the_proof", "the_theorems_own_goals", "on_the_statement"


def leading_error(statement: str, proof: str, errors: Sequence[Mapping]) -> tuple[str | None, Mapping | None]:
    """(where it lies, the error a failed proof is read by), or (None, None) when no error carries a position. In
    this order: the first by position in the proof; else the first placed after the proof's last line; else the
    theorem's own "unsolved goals"; else the first on the theorem's lines. `plan_cut` says why."""
    errors = sorted(errors, key=lambda error: (error["line"], error["column"]))
    if not errors:
        return None, None
    first = proof_first_line(statement)
    last = first + len(proof_lines(proof)) - 1
    in_proof = [error for error in errors if first <= error["line"] <= last]
    after = [error for error in errors if error["line"] > last]
    own = [error for error in errors if error["line"] < first and is_unsolved_goals(error)]
    if in_proof:
        return IN_THE_PROOF, in_proof[0]
    if after:
        return AFTER_THE_PROOF, after[0]
    if own:
        return THE_THEOREMS_OWN_GOALS, own[0]
    return ON_THE_STATEMENT, errors[0]


def plan_cut(statement: str, proof: str, errors: Sequence[Mapping]) -> tuple[Cut | None, str | None]:
    """(the cut, None), or (None, why the proof cannot be cut). Spec item 1.

    The first error by position that lies in the proof: every line before the one its step begins on is kept
    (`step_begins`). Nothing kept is allowed. An error on the theorem's own lines is not in the proof: the
    theorem's "unsolved goals" is read as "the proof ran out with goals open" (the whole proof is kept) only when
    the proof itself has no error. With one, the proof did not run out: it stopped there (Lean also reports the
    theorem's goals as unsolved when a proof cannot be parsed to its end), and the cut is there. An error placed
    after the proof's last line is a last step that never ended: the cut is where that step begins."""
    where, error = leading_error(statement, proof, errors)
    if error is None:
        return None, NO_ERROR_POSITION
    lines, first = proof_lines(proof), proof_first_line(statement)
    if where == IN_THE_PROOF:
        at = error["line"] - first
    elif where == AFTER_THE_PROOF:
        at = len(lines)
    elif where == THE_THEOREMS_OWN_GOALS:
        return Cut(WHOLE_PROOF, tuple(lines), own_indentation(lines), None, error), None
    else:
        return None, ERROR_ON_THE_STATEMENT
    begins = step_begins(lines, at)
    if begins == len(lines):        # an error after the proof whose last step is closed: all of it is kept
        return Cut(WHOLE_PROOF, tuple(lines), own_indentation(lines), None, error), None
    return Cut(BODY, tuple(lines[:begins]), indentation_of(lines[begins]), lines[begins].strip(), error, moved_back=at - begins), None


# ----------------------------------------------------------------------------------------------- the state
def kept_text(kept: Sequence[str]) -> str:
    return "".join(line + "\n" for line in kept)


def state_source(statement: str, cut: Cut) -> tuple[str, int, int]:
    """(the Lean file that asks for the proof state at the cut, the 1-based line of its `sorry`, its 0-based column).
    Spec item 2: the kept lines, then `all_goals sorry` at the cut's indentation. It is the attempt's own file with
    the rest of the proof replaced."""
    source = build_proof_source(statement, kept_text(cut.kept) + cut.indentation + STATE_PLACEHOLDER + "\n")
    return source, proof_first_line(statement) + len(cut.kept), len(cut.indentation) + STATE_PLACEHOLDER.rindex("sorry")


def read_state(raw: Mapping[str, Any], line: int, column: int) -> tuple[str | None, str | None]:
    """(the proof state Lean reports at the `sorry` at `line`, `column`; None), or (None, why there is none).

    The state is the goal of every `sorry` Lean ran at that place: one for each goal open at the cut (`all_goals
    sorry`), joined as Lean prints several goals. It is taken only from a file that is otherwise sound: the one
    error allowed is "unsolved goals" of a block the `sorry` is inside (the theorem's own, when the cut is inside
    an inner block). Any other error means the kept lines do not stand alone (before the placeholder), the
    placeholder does not fit where it was put (at it), or the cut is inside a structure the kept lines do not
    close (after it). A sound file in which Lean ran no `sorry` at all had no goal left at the cut: the kept lines
    already close the block they end in (`NO_GOALS_AT_THE_CUT`; spec item 2a)."""
    if raw.get("error"):
        text = str(raw["error"])
        return None, STATE_TIMEOUT if text.startswith("Lean REPL") and "timed out" in text else STATE_NO_ANSWER
    response = raw.get("response")
    if not isinstance(response, Mapping) or ("messages" not in response and "sorries" not in response):
        return None, STATE_NO_ANSWER
    place = (line, column)
    begins = (line, column - STATE_PLACEHOLDER.rindex("sorry"))         # where the placeholder itself begins
    worst = None
    for error in errors_of(raw):
        start = (error["line"], error["column"])
        end = (error["end_line"], error["end_column"] or 0) if error["end_line"] is not None else None
        if is_unsolved_goals(error) and start < begins and (end is None or end >= place):
            continue
        if begins <= start <= place:
            reason = NO_GOALS_AT_THE_CUT if "no goals" in error["text"].lower() else ERROR_AT_THE_SORRY
        else:
            reason = ERROR_BEFORE_THE_SORRY if start < place else ERROR_AFTER_THE_SORRY
        # One reason a file: an error before the placeholder says the most, then one at it, then one after it.
        order = (ERROR_BEFORE_THE_SORRY, NO_GOALS_AT_THE_CUT, ERROR_AT_THE_SORRY, ERROR_AFTER_THE_SORRY)
        worst = reason if worst is None or order.index(reason) < order.index(worst) else worst
    if worst is None and errors_without_a_position(raw):
        worst = ERROR_BEFORE_THE_SORRY
    if worst is not None:
        return None, worst
    entries = response.get("sorries") or []
    goals = []
    for entry in entries:
        position = entry.get("pos") if isinstance(entry.get("pos"), Mapping) else {}
        if (position.get("line"), position.get("column")) == place and str(entry.get("goal") or "").strip():
            goals.append(str(entry["goal"]).strip())
    if goals:
        return "\n\n".join(goals), None
    return None, NO_SORRY_AT_THE_CUT if entries else NO_GOALS_AT_THE_CUT


# ---------------------------------------------------------------------------------------------- the prompt
def state_comment(state: str, indentation: str) -> str:
    """The proof state as the prover was trained to read it: a comment block at `indentation`, its lines two
    spaces further in. The authors' own expression (this module's docstring), with their `' ' * indent_len` as
    `indentation`. THE ONE PLACE the format is written: change it here."""
    newline = "\n" + indentation
    return newline.join([indentation + "/- tactic state:", "  " + state.replace("\n", newline + "  "), "-/\n"])


def resume_prompt(statement: str, kept: Sequence[str], indentation: str, state: str | None) -> str:
    """Spec item 3: the blind prompt, the kept proof lines, the state in the model's own comment format; the
    model continues. `state` None: the same prompt with no state comment (the arm that resumes without it)."""
    return build_prover_prompt(statement) + kept_text(kept) + (state_comment(state, indentation) if state is not None else "")


def resumed_proof(kept: Sequence[str], continuation: str) -> str:
    """The proof that is checked: the kept lines, then what the model wrote. The state comment is not part of it
    (the authors' search leaves it out too), so both resuming arms check a proof of the same shape."""
    return kept_text(kept) + continuation


def trimmed_proof(kept: Sequence[str]) -> str:
    """Spec item 2a: when Lean reports no goal left at the cut, the kept lines are a whole proof with something
    extra after it. This is that proof: the kept lines alone. It is checked exactly as an attempt is (the same
    file, the axioms and all), and only if it verifies does it resolve anything."""
    return kept_text(kept)


def first_step(text: str) -> str | None:
    """The first line of code in what the model wrote (comments and blank lines skipped), white space squeezed."""
    for line in _strip_comments(text).split("\n"):
        if line.strip():
            return " ".join(line.split())
    return None


def repeats_failed_step(continuation: str, failed_step: str | None) -> bool | None:
    """Whether a continuation begins with the very step that had just failed (its first line, when it spans
    several). None when no step failed (the whole proof was kept)."""
    failed = first_step(failed_step) if failed_step is not None else None
    if failed is None:
        return None
    return first_step(continuation) == failed
