"""The Lean source sent to the verifier for one proof attempt, and the lexical filter applied first.

Spec §4: the source is `header + statement + completion + "#print axioms <name>"`, and a completion
containing a forbidden token is rejected without being sent.
"""

from __future__ import annotations

import re

# The header DeepSeek-Prover-V1.5 was trained and evaluated with (prover/utils.py in its repo, read
# 2026-10-02). It is part of the model's prompt, so it must not be edited. Note `maxHeartbeats 0`:
# Lean has no internal work limit under this header, so the verifier's wall-clock timeout is the only
# bound on a proof check.
LEAN_HEADER = (
    "import Mathlib\n"
    "import Aesop\n"
    "\n"
    "set_option maxHeartbeats 0\n"
    "\n"
    "open BigOperators Real Nat Topology Rat\n"
    "\n"
)

# Tokens a proof or a theorem statement never needs. They either fake a proof (`sorry`, `admit`,
# `axiom`, `native_decide`); or run arbitrary code ON THE VERIFICATION SERVER while Lean elaborates
# the file (`#eval`, `run_cmd`, `run_tac`, `run_elab`, `run_meta`, `IO.`, `unsafe`, and the
# metaprogramming commands that define new syntax or elaborators: `elab`, `elab_rules`, `macro`,
# `macro_rules`, `syntax`, `initialize`, `builtin_initialize`); or change the environment the
# statement is checked in (`import`). This is defence in depth, not the boundary: a Lean file that
# reaches the server can run programs there (measured 2026-10-02: a shell command ran in the
# container), so the container's own limits are what contain it. Each pattern matches the token only
# as a whole word, so `sorry_lemma` or `admits` in an identifier is not a hit.
_WORD_TOKENS = (
    "sorry", "admit", "axiom", "import", "unsafe", "native_decide",
    "run_cmd", "run_tac", "run_elab", "run_meta",
    "elab", "elab_rules", "macro", "macro_rules", "syntax", "initialize", "builtin_initialize",
)
_FORBIDDEN_TOKEN_PATTERNS: dict[str, re.Pattern[str]] = {
    **{token: re.compile(rf"(?<![\w.']){token}(?![\w'])") for token in _WORD_TOKENS},
    "#eval": re.compile(r"#eval(?![\w'])"),
    "IO.": re.compile(r"(?<![\w.'])IO\."),
}

# A declaration starts a line, optionally after attributes and modifiers. Matched on text with comments
# removed: comments and docstrings routinely say "the theorem is ..." (the first pin-gate run named the
# theorem `is`, `axioms` and `that` 14 times out of 489 before this was anchored).
_THEOREM_NAME_PATTERN = re.compile(
    r"^[ \t]*(?:@\[[^\]]*\][ \t]*)?(?:(?:private|protected|noncomputable|nonrec)[ \t]+)*"
    r"(?:theorem|lemma)[ \t]+([^\s:({\[]+)",
    re.MULTILINE,
)
_LINE_COMMENT = re.compile(r"--[^\n]*")


def _strip_comments(lean_text: str) -> str:
    """Lean text with `--` line comments and `/- ... -/` block comments (nested, docstrings included)
    replaced by spaces, keeping line structure. String literals are not special-cased: a statement
    never needs one containing comment markers, and the result is only used to find declarations."""
    out, depth, index = [], 0, 0
    while index < len(lean_text):
        pair = lean_text[index:index + 2]
        if pair == "/-":
            depth, index = depth + 1, index + 2
            out.append("  ")
        elif pair == "-/" and depth:
            depth, index = depth - 1, index + 2
            out.append("  ")
        else:
            character = lean_text[index]
            out.append(character if depth == 0 or character == "\n" else " ")
            index += 1
    return _LINE_COMMENT.sub("", "".join(out))


def find_forbidden_token(completion: str) -> str | None:
    """Return the first forbidden token in model-written Lean text, or None when it is clean.

    Applied to EVERYTHING the model writes before it reaches the server: proof completions, and the
    statements of generated conjectures (their compile check runs them through Lean too)."""
    for token, pattern in _FORBIDDEN_TOKEN_PATTERNS.items():
        if pattern.search(completion):
            return token
    return None


def theorem_name_of(statement: str) -> str:
    """The name of the theorem a statement declares (the LAST declaration, if it declares several).

    `#print axioms` needs it. A statement with no `theorem` or `lemma` keyword is a caller bug.
    """
    names = _THEOREM_NAME_PATTERN.findall(_strip_comments(statement))
    if not names:
        raise ValueError(f"no theorem or lemma declaration found in statement: {statement[:120]!r}")
    return names[-1]


def build_proof_source(statement: str, completion: str) -> str:
    """Assemble the complete Lean file for one attempt.

    `statement` is the theorem up to and including `:= by` (as in the model's prompt); `completion`
    is the proof the model wrote after it. The trailing `#print axioms` makes Lean report every axiom
    the finished theorem rests on, which is how `sorry` smuggled through a tactic, `admit`, and
    declared axioms are detected (the verification server itself has no axiom check).
    """
    theorem_name = theorem_name_of(statement)
    statement_text = statement if statement.endswith("\n") else statement + "\n"
    completion_text = completion.rstrip() + "\n"
    return f"{LEAN_HEADER}{statement_text}{completion_text}\n#print axioms {theorem_name}\n"


def imports_first(lean_file: str) -> str:
    """The same file with its leading `import` lines moved above everything else (the OEIS Open spec, O1).

    Kimina takes a file's header to be the `import` lines it starts with. The benchmark's files start
    with a license comment, which Lean itself accepts before the imports, so Kimina saw no header and
    the REPL rejected each `import` as "must be used in the beginning of the file" (30 of the 38 gold
    proofs at the first gate run). Only imports in the leading region (before the first command) move,
    in their order; comments stay where they were; an `import` inside a comment is not an import.
    """
    original = lean_file.splitlines(keepends=True)
    code = _strip_comments(lean_file).splitlines(keepends=True)
    if len(original) != len(code):           # the comment stripper keeps line structure; if not, leave it
        return lean_file
    imports, rest, in_header = [], [], True
    for raw_line, code_line in zip(original, code):
        content = code_line.strip()
        if in_header and content.startswith("import "):
            imports.append(raw_line if raw_line.endswith("\n") else raw_line + "\n")
            continue
        if content:
            in_header = False
        rest.append(raw_line)
    return "".join(imports) + "".join(rest) if imports else lean_file
