"""Notation that one Lean pin accepts and another does not, rewritten. Spec: docs/spec/ladder-loop.spec.md
(decided 2026-10-04: rewrite, with rewritten statements marked) and docs/spec/o2a-version-tax-RESULT.md.

One rule so far. Mathlib once wrote a sum or product over a finite set as `∑ x in s, f x`; it now writes
`∑ x ∈ s, f x`, and by Lean v4.27 the old form no longer parses. The texts this project reads (Lean Workbook,
miniF2F, the published proofs) are from 2024 and use the old form: 8% of our statements and 63 of miniF2F's
488 do not compile at v4.27 for this alone.
"""

from __future__ import annotations

import re

_BIG_OPERATORS = "∑∏"
_OPENING = "([{⟨⦃⟦"
_CLOSING = ")]}⟩⦄⟧"
_IN_WORD = re.compile(r"(?<=\s)in(?=\s)")
MAXIMUM_BINDER_CHARACTERS = 400      # a binder and its set; the scan gives up past this, so a stray `∑` costs nothing


def _old_membership_span(text: str, operator_index: int) -> tuple[int, int] | None:
    """The span of the `in` of `∑ x in s,` whose operator is at `operator_index`, or None when that operator
    is not in the old form: its binder already has `∈`, or it has no set (`∑ x, f x`), or no comma follows."""
    depth = 0
    end = min(len(text), operator_index + 1 + MAXIMUM_BINDER_CHARACTERS)
    index = operator_index + 1
    while index < end:
        character = text[index]
        if character in _OPENING:
            depth += 1
        elif character in _CLOSING:
            depth -= 1
            if depth < 0:               # the operator's own enclosing bracket closed: there was no binder list
                return None
        elif depth == 0:
            if character in ",∈" or character in _BIG_OPERATORS:
                return None             # `∑ x, f`, `∑ x ∈ s, f`, or another operator before any `in`
            if character == "i":
                word = _IN_WORD.match(text, index)
                if word is not None:
                    return word.span()
        index += 1
    return None


def membership_binders(lean_text: str) -> tuple[str, int]:
    """(`lean_text` with every `∑ x in s,` and `∏ x in s,` written `∑ x ∈ s,`, how many were rewritten).

    Only the first `in` after the operator and before its comma is touched, and only outside brackets, so an
    `in` inside the binder's own type or inside the set is left alone. Text with none of the old form comes
    back unchanged, with 0."""
    spans = []
    for index, character in enumerate(lean_text):
        if character in _BIG_OPERATORS:
            span = _old_membership_span(lean_text, index)
            if span is not None:
                spans.append(span)
    if not spans:
        return lean_text, 0
    pieces, cursor = [], 0
    for start, stop in spans:
        pieces.append(lean_text[cursor:start])
        pieces.append("∈")
        cursor = stop
    pieces.append(lean_text[cursor:])
    return "".join(pieces), len(spans)
