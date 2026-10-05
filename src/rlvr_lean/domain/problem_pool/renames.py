"""Library names Mathlib has since renamed, rewritten in a published proof. Spec: docs/spec/ladder-loop.spec.md,
"A published proof is brought to our pin mechanically, and Lean still judges it".

The published proofs were written for a 2024 Mathlib. A lemma they cite by a name that is gone (`le_div_iff`,
now `le_div_iff₀`) fails for that alone. The table (config `ladder_loop.certificates.renames`, old name to new
name, each entry from Mathlib's own deprecation record) is applied to CERTIFICATES only, never to a statement.
A renamed certificate is a NEW certificate of its problem, marked with the names that changed; the original's
failure stays on record. A wrong entry costs a certificate and nothing else: Lean accepts or refuses the result.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping

from rlvr_lean.domain.conjecturing.parsing import _IDENTIFIER, _first_index_outside_brackets
from rlvr_lean.domain.problem_pool.certificates import Certificate
from rlvr_lean.domain.verification.lean_source import LEAN_HEADER

# The line the variables-only negation wrapper opens with (`certificates.negation_certificate`). It restates
# the STATEMENT's type, so it is never renamed: only the published proof under it is.
WRAPPER_HEAD = "  have published_for_every_value "
_BEFORE_AN_IDENTIFIER_PART = frozenset("'!?.")       # with letters, digits and `_`: the token started earlier


def opened_namespaces(header: str = LEAN_HEADER) -> tuple[str, ...]:
    """The namespaces the pipeline's header opens: a proof may write `sqrt_nonneg` for `Real.sqrt_nonneg`."""
    return tuple(name for line in header.split("\n") if line.startswith("open ") for name in line.split()[1:])


def identifiers(lean_text: str) -> list[tuple[int, int, str]]:
    """(start, end, text) of every WHOLE identifier in Lean text: the maximal dotted name, so `Nat.le_div_iff`
    is one identifier and `le_div_iff` is not found inside it, nor inside `le_div_iff₀`, `le_div_iff'` or
    `x.le_div_iff`. A field access on a parenthesis (`(h x).le`) is not an identifier of its own."""
    found = []
    for match in _IDENTIFIER.finditer(lean_text):
        before = lean_text[match.start() - 1] if match.start() else " "
        if before.isalnum() or before == "_" or before in _BEFORE_AN_IDENTIFIER_PART:
            continue
        found.append((match.start(), match.end(), match.group()))
    return found


def tabled_name(identifier: str, table: Mapping[str, str]) -> str | None:
    """The key of `table` this identifier names: itself, or its longest dotted prefix that is a key, the rest
    then being projections of that lemma (`le_div_iff.mpr`, `Set.eq_empty_iff_forall_not_mem.mp`), which is how
    Lean itself reads a dotted identifier. A name that merely ENDS in a key (`Nat.le_div_iff`) or extends one
    without a dot (`le_div_iff₀`, `le_div_iff_mul_le`) is another name and is not matched."""
    if identifier in table:
        return identifier
    cut = identifier.rfind(".")
    while cut > 0:
        if identifier[:cut] in table:
            return identifier[:cut]
        cut = identifier.rfind(".", 0, cut)
    return None


def rename_library_names(lean_text: str, table: Mapping[str, str]) -> tuple[str, tuple[str, ...]]:
    """(`lean_text` with every whole identifier that names a key of `table` rewritten to the key's value, the
    keys that were replaced, sorted). Text with none of them comes back unchanged, with an empty tuple."""
    pieces, cursor, changed = [], 0, set()
    for start, end, name in identifiers(lean_text):
        old = tabled_name(name, table)
        if old is None or table[old] == old:
            continue
        pieces.append(lean_text[cursor:start])
        pieces.append(table[old] + name[len(old):])
        cursor = end
        changed.add(old)
    if not changed:
        return lean_text, ()
    pieces.append(lean_text[cursor:])
    return "".join(pieces), tuple(sorted(changed))


def _published_part(proof: str) -> int:
    """Where the published proof starts inside a certificate's proof: 0, or past the wrapper's opening line."""
    if not proof.startswith(WRAPPER_HEAD):
        return 0
    assignment = _first_index_outside_brackets(proof, ":=")       # the wrapper's own: its binders and type are bracketed
    line_end = proof.find("\n", assignment if assignment is not None else 0)
    return len(proof) if assignment is None or line_end == -1 else line_end + 1


def renamed_certificate(certificate: Certificate, table: Mapping[str, str]) -> Certificate | None:
    """The certificate with the table applied to its published proof, marked with the names that changed; None
    when the proof holds none of them. The theorem (the statement, or its negation) is never touched."""
    start = _published_part(certificate.proof)
    published, changed = rename_library_names(certificate.proof[start:], table)
    if not changed:
        return None
    return dataclasses.replace(certificate, proof=certificate.proof[:start] + published,
                               renamed=tuple(sorted(set(certificate.renamed) | set(changed))))


def written_forms(name: str, namespaces: tuple[str, ...]) -> tuple[str, ...]:
    """The ways a proof can write the library name Lean reports as `name`: itself, and without a leading
    namespace the header opens (`Real.sqrt_eq_iff_sq_eq` as `sqrt_eq_iff_sq_eq`)."""
    forms = [name]
    for namespace in namespaces:
        if name.startswith(namespace + ".") and len(name) > len(namespace) + 1:
            forms.append(name[len(namespace) + 1:])
    return tuple(forms)
