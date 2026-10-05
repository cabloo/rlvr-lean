"""The negation of a conjecture, as a theorem to prove. Spec §13b (the training-data census).

A conjecture `theorem NAME BINDERS : TYPE := by` claims `∀ BINDERS, TYPE`. Its negation is the theorem
`¬ (∀ BINDERS, TYPE)`, with no binders of its own: a verified proof of it shows the conjecture is FALSE as
stated, hypotheses included. The binders are copied as written, so every binder form Lean accepts after
`∀` (explicit, implicit, instance, strict-implicit, bare names) carries over; one it does not accept (a
binder with a default value, say) fails the compile check the census runs before sampling, and is counted.
"""

from __future__ import annotations

from rlvr_lean.domain.conjecturing.parsing import _NAME_TOKEN, _brackets_balance, _first_index_outside_brackets, _is_lean_identifier

_KEYWORDS = ("theorem ", "lemma ")


def split_signature(statement: str) -> tuple[str, str, str]:
    """(name, binders, type) of a statement written as `theorem NAME BINDERS : TYPE := by`.

    The statement's own `:=` and type colon are the first ones no bracket pair encloses, which is how the
    statement was cut out of the model's text in the first place (`parse_generated_statement`)."""
    text = statement.strip()
    keyword = next((word for word in _KEYWORDS if text.startswith(word)), None)
    if keyword is None:
        raise ValueError(f"not a theorem or lemma statement: {statement[:80]!r}")
    assignment = _first_index_outside_brackets(text, ":=")
    if assignment is None or text[assignment + 2:].strip() != "by":
        raise ValueError(f"the statement does not end in `:= by`: {statement[-80:]!r}")
    body = text[len(keyword):assignment].strip()
    name = _NAME_TOKEN.match(body)
    if name is None or not _is_lean_identifier(name.group()):
        raise ValueError(f"the statement's name is not a Lean identifier: {body[:80]!r}")
    rest = body[name.end():]
    colon = _first_index_outside_brackets(rest, ":")
    if colon is None:
        raise ValueError(f"the statement has no type: {statement[:80]!r}")
    binders, statement_type = rest[:colon].strip(), rest[colon + 1:].strip()
    if not statement_type:
        raise ValueError(f"the statement's type is empty: {statement[:80]!r}")
    if not _brackets_balance(binders) or not _brackets_balance(statement_type):
        raise ValueError(f"the statement's brackets do not balance: {statement[:80]!r}")
    return name.group(), binders, statement_type


def negate_statement(statement: str, negation_name: str) -> str:
    """`theorem <negation_name> : ¬ (∀ BINDERS, TYPE) := by\\n` for the statement `theorem NAME BINDERS : TYPE := by`;
    `¬ (TYPE)` when it has no binders."""
    if not _is_lean_identifier(negation_name):
        raise ValueError(f"not a Lean identifier: {negation_name!r}")
    _, binders, statement_type = split_signature(statement)
    claim = f"∀ {binders}, {statement_type}" if binders else statement_type
    return f"theorem {negation_name} : ¬ ({claim}) := by\n"


def negation_name(conjecture_id: str) -> str:
    return f"negated_conjecture_{conjecture_id}"


def build_negation_exactness_source(statement: str) -> str:
    """A Lean file that compiles iff the built negation IS `¬ (the conjecture's own type)`: the conjecture
    (closed with `sorry`, never sent as a proof attempt), then the two propositions stated equal by `rfl`.

    Why it can fail: a statement may use a variable it never binds (`theorem t (h : 0 < x) : -x < 0`), which
    Lean binds for the whole theorem. Copied under `¬ (∀ ...)`, that variable is bound OUTSIDE the negation,
    so the built theorem says "for every x, not ..." and not "not for every x ...". A proof of it still shows
    the conjecture false (it is the stronger claim), but a conjecture false only for some x cannot be
    disproved that way, so the census counts such negations apart."""
    from rlvr_lean.domain.verification.lean_source import LEAN_HEADER

    name, binders, statement_type = split_signature(statement)
    claim = f"∀ {binders}, {statement_type}" if binders else statement_type
    statement_text = statement if statement.endswith("\n") else statement + "\n"
    return f"{LEAN_HEADER}{statement_text}  sorry\n\nexample : (¬ ({claim})) = (¬ (type_of% @{name})) := rfl\n"
