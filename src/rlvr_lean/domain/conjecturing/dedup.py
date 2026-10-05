"""Deduplication of statements, and the content id a conjecture is named by. Spec §3 items 2 and 4.

Two keys decide whether a statement was seen before: its NORMALIZED TEXT (comments and the theorem's
name removed, whitespace collapsed, Unicode NFC, the trailing proof stub dropped), and, when the compile
check produced one, the hash of its ELABORATED TYPE (`parse_type_fingerprint`), which also ignores
bound-variable names and notation. Either key matching makes it a duplicate.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable

# A statement's declaration head: optional attributes and modifiers, the keyword, then the name (a plain
# identifier, possibly dotted, or a «guillemet-quoted» one). Matched at the very start of comment-free
# text, so the declared name is never confused with a name mentioned in a docstring.
DECLARATION_HEAD_PATTERN = re.compile(
    r"\A\s*(?:@\[[^\]]*\]\s*)*(?:(?:private|protected|noncomputable|nonrec)\s+)*(?:theorem|lemma)\s+"
    r"(?P<name>«[^»]*»|[^\s:({\[⦃]+)"
)
_WHITESPACE_RUN = re.compile(r"\s+")
# What follows the type in our statement formats: `:= by` (miniF2F, conjectures), `:= by sorry` (Lean
# Workbook, seed statements), or a bare `:=`.
_TRAILING_PROOF_STUB = re.compile(r"\s*:=(?:\s*by(?:\s+sorry)?)?\s*\Z")


def strip_lean_comments(lean_text: str) -> str:
    """Lean text with every comment blanked out: `--` line comments, and `/- ... -/` block comments,
    which nest (docstrings `/-- -/` and module docs `/-! -/` are block comments too).

    Each comment character becomes a space and newlines are kept, so the result has the same length and
    every code character keeps its position; spans found in it are valid in the original. As in
    `lean_source`, string literals are not special-cased: a statement never needs one containing a
    comment marker.
    """
    if "--" not in lean_text and "/-" not in lean_text:      # no comment can start: the loop below would copy it
        return lean_text
    output: list[str] = []
    block_depth, index = 0, 0
    while index < len(lean_text):
        pair = lean_text[index:index + 2]
        if block_depth == 0 and pair == "--":
            line_end = lean_text.find("\n", index)
            line_end = len(lean_text) if line_end == -1 else line_end
            output.append(" " * (line_end - index))
            index = line_end
        elif pair == "/-":
            block_depth, index = block_depth + 1, index + 2
            output.append("  ")
        elif block_depth > 0 and pair == "-/":
            block_depth, index = block_depth - 1, index + 2
            output.append("  ")
        else:
            character = lean_text[index]
            output.append(character if block_depth == 0 or character == "\n" else " ")
            index += 1
    return "".join(output)


def normalize_statement(statement: str) -> str:
    """The statement's binders and type only, in a canonical spelling: the string dedup key."""
    text = strip_lean_comments(unicodedata.normalize("NFC", statement))
    text = _WHITESPACE_RUN.sub(" ", text).strip()
    text = DECLARATION_HEAD_PATTERN.sub("", text, count=1)
    return _TRAILING_PROOF_STUB.sub("", text).strip()


def content_id(statement: str) -> str:
    """First 16 hex digits of the SHA-256 of the normalized statement (spec §3 item 2). It does not
    depend on the theorem's name, so renaming a statement to `conjecture_<id>` keeps its id."""
    return hashlib.sha256(normalize_statement(statement).encode("utf-8")).hexdigest()[:16]


class StatementDeduplicator:
    """Remembers statements by both dedup keys. Pre-seed it with everything a conjecture must not repeat:
    the seed statements, the reward pool and all miniF2F statements (spec §3 item 4)."""

    def __init__(self, seen_statements: Iterable[str] = (), seen_fingerprints: Iterable[int] = ()) -> None:
        self._seen_normalized_statements = {normalize_statement(statement) for statement in seen_statements}
        self._seen_type_fingerprints = set(seen_fingerprints)

    def add(self, statement: str, type_fingerprint: int | None = None) -> bool:
        """Record `statement` and return True if it is new; return False, recording nothing, if its
        normalized text or (when given) its type fingerprint was already seen."""
        normalized = normalize_statement(statement)
        if normalized in self._seen_normalized_statements:
            return False
        if type_fingerprint is not None and type_fingerprint in self._seen_type_fingerprints:
            return False
        self._seen_normalized_statements.add(normalized)
        if type_fingerprint is not None:
            self._seen_type_fingerprints.add(type_fingerprint)
        return True

    @property
    def seen_count(self) -> int:
        """Distinct normalized statements seen so far, pre-seeded ones included."""
        return len(self._seen_normalized_statements)
