"""Lean sources for the two statement-level checks of spec §3: the compile check (which also prints a
fingerprint of the elaborated statement, for deduplication) and the vacuity check.

Both take a statement as the model's prompt carries it: the theorem up to and including `:= by`.
"""

from __future__ import annotations

import re

from rlvr_lean.domain.verification.lean_source import LEAN_HEADER, theorem_name_of

FINGERPRINT_MARKER = "rlvr-type-fingerprint"

# Vacuity: try to derive `False` from the hypotheses alone. `intros` moves the binders of the goal
# (`∀ x, P x → ...`) into the context, `exfalso` replaces the goal with `False`, and each alternative is
# a cheap closer that either finishes or fails. If this verifies, the hypotheses contradict each other
# and the statement is trivially true: discard it. Sound (a verified proof of False from the hypotheses
# IS a contradiction) and incomplete (it misses contradictions these tactics cannot find).
VACUITY_PROOF = (
    "  intros\n"
    "  exfalso\n"
    "  first\n"
    "  | contradiction\n"
    "  | (simp_all; done)\n"
    "  | omega\n"
    "  | linarith\n"
    "  | nlinarith\n"
    "  | (norm_num at *; done)\n"
)

# Prints the hash of the theorem's ELABORATED type. Lean's `Expr` hash is structural over the elaborated
# term, so it does not see how the statement was written (whitespace, notation, the theorem's own name)
# and, if Lean's binder representation is as expected, not hypothesis or variable names either. That is
# checked by a live fixture rather than assumed (tests/rlvr_lean/test_kimina_conjecture_fixtures.py).
_FINGERPRINT_COMMAND = (
    "open Lean in\n"
    "#eval show Elab.Command.CommandElabM Unit from do\n"
    "  match (← getEnv).find? `{name} with\n"
    "  | some info => logInfo m!\"" + FINGERPRINT_MARKER + " {{info.type.hash}}\"\n"
    "  | none => logInfo \"" + FINGERPRINT_MARKER + " missing\"\n"
)
_FINGERPRINT_PATTERN = re.compile(re.escape(FINGERPRINT_MARKER) + r" (\d+)")


def _statement_text(statement: str) -> str:
    return statement if statement.endswith("\n") else statement + "\n"


def fingerprint_command(name: str) -> str:
    """The command that prints the fingerprint of the declared theorem `name`, to append to a Lean file."""
    return _FINGERPRINT_COMMAND.format(name=name)


def build_compile_check_source(statement: str) -> str:
    """Statement + `sorry`, then the fingerprint command. Compiles iff the statement elaborates."""
    name = theorem_name_of(statement)
    return f"{LEAN_HEADER}{_statement_text(statement)}  sorry\n\n" + fingerprint_command(name)


def build_vacuity_source(statement: str) -> str:
    """Statement + the vacuity proof + an axiom report, classified like any proof attempt."""
    name = theorem_name_of(statement)
    return f"{LEAN_HEADER}{_statement_text(statement)}{VACUITY_PROOF}\n#print axioms {name}\n"


def parse_type_fingerprint(info_messages: list[str]) -> int | None:
    """The fingerprint printed by the compile check, or None if it is absent (the check failed)."""
    for message in info_messages:
        match = _FINGERPRINT_PATTERN.search(message)
        if match:
            return int(match.group(1))
    return None
