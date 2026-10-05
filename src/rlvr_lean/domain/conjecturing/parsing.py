"""Reading a generated conjecture out of the model's completion, and giving it its canonical name.
Spec §3 items 1-2.

The conjecture prompt is a Lean file of seed statements that ends in an open `theorem `, so the model
continues with a NAME, binders, `:` and a type, and normally stops at (or runs on past) the `:=` that
would start a proof. The statement is everything up to that `:=`.
"""

from __future__ import annotations

import re

from rlvr_lean.domain.conjecturing.dedup import DECLARATION_HEAD_PATTERN, content_id, strip_lean_comments

MAXIMUM_STATEMENT_BODY_CHARACTERS = 2_000

# Pairs that may enclose a `:=` or `:` which is NOT the statement's own: binders `( )`, `{ }`, `[ ]`,
# `⦃ ⦄`; a `let x := ...` or structure instance inside them; anonymous constructors `⟨ ⟩`; quotients `⟦ ⟧`.
_OPENING_BRACKETS = frozenset("([{⟨⦃⟦")
_CLOSING_BRACKETS = frozenset(")]}⟩⦄⟧")

# A Lean identifier: dot-separated atoms, each a letter or `_` followed by letters, digits, `_`, `'`,
# `!`, `?` (Python's Unicode `\w` covers Lean's Greek letters, `ℕ`-style letterlikes and subscripts),
# or any «guillemet-quoted» text.
_IDENTIFIER_ATOM = r"(?:«[^»\n]+»|[^\W\d][\w'!?]*)"
_IDENTIFIER = re.compile(rf"{_IDENTIFIER_ATOM}(?:\.{_IDENTIFIER_ATOM})*")
# The name is the body's first token: it ends at whitespace, a binder bracket or the type colon.
_NAME_TOKEN = re.compile(r"«[^»\n]*»|[^\s:({\[⦃]+")


def _is_lean_identifier(text: str) -> bool:
    return _IDENTIFIER.fullmatch(text) is not None


def _first_index_outside_brackets(text: str, token: str) -> int | None:
    """Index of the first occurrence of `token` that no bracket pair encloses, or None."""
    depth = 0
    for index, character in enumerate(text):
        if character in _OPENING_BRACKETS:
            depth += 1
        elif character in _CLOSING_BRACKETS:
            depth -= 1
        elif depth == 0 and text.startswith(token, index):
            return index
    return None


def _brackets_balance(text: str) -> bool:
    depth = 0
    for character in text:
        if character in _OPENING_BRACKETS:
            depth += 1
        elif character in _CLOSING_BRACKETS:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _has_type(body: str) -> bool:
    """The body declares a type: a `:` outside every bracket pair, with something after it."""
    separator = _first_index_outside_brackets(body, ":")
    return separator is not None and body[separator + 1:].strip() != ""


def _statement_body(code: str, stopped_at_assignment: bool) -> str | None:
    """The text before the statement's own `:=`; the whole text if generation stopped exactly there."""
    assignment = _first_index_outside_brackets(code, ":=")
    if assignment is not None:
        return code[:assignment]
    return code if stopped_at_assignment else None


def parse_generated_statement(
    generated_text: str,
    *,
    stopped_at_assignment: bool = False,
    maximum_body_characters: int = MAXIMUM_STATEMENT_BODY_CHARACTERS,
) -> str | None:
    """`theorem <body> := by\\n` for the statement the model wrote after `theorem `, or None if unusable.

    `stopped_at_assignment` says generation was cut by a `:=` stop sequence, which the returned text
    omits. Comments are removed (a line comment would otherwise swallow the `:= by` appended after it,
    and a comment can trip the lexical filter for nothing). None when: there is no `:=` and generation
    did not stop at one; the body is empty or longer than `maximum_body_characters`; its brackets do not
    balance; it has no type after a top-level `:`; or its name is not a Lean identifier.
    """
    body = _statement_body(strip_lean_comments(generated_text), stopped_at_assignment)
    if body is None:
        return None
    body = "\n".join(line.rstrip() for line in body.strip().split("\n"))
    if not body or len(body) > maximum_body_characters:
        return None
    if not _brackets_balance(body) or not _has_type(body):
        return None
    name = _NAME_TOKEN.match(body)
    if name is None or not _is_lean_identifier(name.group()):
        return None
    return f"theorem {body} := by\n"


def rename_theorem(statement: str, new_name: str) -> str:
    """The statement with its declared name replaced by `new_name`; everything else is left as written."""
    if not _is_lean_identifier(new_name):
        raise ValueError(f"not a Lean identifier: {new_name!r}")
    # Comments are blanked in place, so the name's span in the blanked text is its span in the original.
    head = DECLARATION_HEAD_PATTERN.match(strip_lean_comments(statement))
    if head is None:
        raise ValueError(f"statement does not start with a theorem or lemma declaration: {statement[:120]!r}")
    start, end = head.span("name")
    return statement[:start] + new_name + statement[end:]


def canonical_statement(statement: str) -> tuple[str, str]:
    """(conjecture_id, the statement renamed `conjecture_<conjecture_id>`), spec §3 item 2. The id is
    computed from the normalized text, which ignores the name, so the renamed statement has the same id."""
    conjecture_id = content_id(statement)
    return conjecture_id, rename_theorem(statement, f"conjecture_{conjecture_id}")
