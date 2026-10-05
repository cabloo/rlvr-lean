"""What one verification attempt resolved to. Spec §4, "Status of one attempt, first match wins".

The verification server (Kimina) reports Lean's messages; it does not decide validity for us. Its own
client classifies `by admit` as valid and has no axiom check, so the rules live here, in the domain.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

# The axioms ordinary Mathlib mathematics rests on. Anything else in `#print axioms` output means the
# proof assumed something: `sorryAx` (a `sorry` or `admit` anywhere in the term), a declared `axiom`,
# or `Lean.ofReduceBool` (`native_decide`, which trusts the compiler).
ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})

_AXIOM_LIST_PATTERN = re.compile(r"depends on axioms:\s*\[(?P<axioms>[^\]]*)\]", re.DOTALL)
_NO_AXIOMS_TEXT = "does not depend on any axioms"
_SORRY_WARNING_TEXT = "declaration uses 'sorry'"


class VerificationStatus(str, enum.Enum):
    """Statuses in the order the spec checks them. Only VERIFIED counts as a proof."""

    REJECTED_LEXICAL = "rejected_lexical"   # forbidden token in the completion; never sent to Lean
    TIMEOUT = "timeout"                     # wall-clock limit hit; recorded distinctly, never retried
    SERVER_ERROR = "server_error"           # the server could not give an answer, even after retries
    LEAN_ERROR = "lean_error"               # Lean reported an error (or a "failed" warning)
    USES_SORRY = "uses_sorry"               # Lean accepted the file only by assuming a `sorry`
    FORBIDDEN_AXIOM = "forbidden_axiom"     # the proof rests on an axiom outside ALLOWED_AXIOMS
    VERIFIED = "verified"


@dataclass(frozen=True)
class VerificationResult:
    attempt_id: str
    status: VerificationStatus
    messages: tuple[str, ...] = ()
    axioms: tuple[str, ...] = ()
    verification_seconds: float | None = None
    peak_memory_bytes: int | None = None
    detail: str = field(default="", compare=False)

    @property
    def is_verified(self) -> bool:
        return self.status is VerificationStatus.VERIFIED


def rejected_lexically(attempt_id: str, forbidden_token: str) -> VerificationResult:
    """The result for a completion that was filtered out before reaching Lean."""
    return VerificationResult(
        attempt_id=attempt_id,
        status=VerificationStatus.REJECTED_LEXICAL,
        detail=f"completion contains the forbidden token {forbidden_token!r}",
    )


def _parse_axioms(info_messages: list[str]) -> tuple[str, ...] | None:
    """Axioms reported by `#print axioms`, or None when Lean printed no axiom report at all.

    Lean prints either "'name' depends on axioms: [a, b]" or "'name' does not depend on any axioms".
    The list order differs between Lean versions, so callers must treat it as a set.
    """
    for message in info_messages:
        match = _AXIOM_LIST_PATTERN.search(message)
        if match:
            names = (name.strip() for name in match.group("axioms").split(","))
            return tuple(sorted(name for name in names if name))
        if _NO_AXIOMS_TEXT in message:
            return ()
    return None


def classify_check_result(attempt_id: str, check_result: Mapping[str, Any], *,
                          failed_warning_is_error: bool = True) -> VerificationResult:
    """Turn one element of Kimina's `/api/check` `results` into a VerificationResult.

    A result carries exactly one of `error` (the server could not run the snippet: a timeout, or a
    crashed worker) or `response` (what Lean said). The order of the checks below is the spec's.

    `failed_warning_is_error` keeps DeepSeek's and STP's convention that a warning containing "failed" is
    an incomplete proof: on for the Lean v4.9 pipeline, so Phase A and B are judged as they always were.
    The Lean v4.27 (OEIS Open) pipeline turns it off: there a tactic that fails inside a combinator that
    recovers (`aesop: failed to prove the goal ...` under `first`) warns while the proof is complete,
    and one of the benchmark's 38 gold proofs was rejected for exactly that; the axiom check below and
    Comparator are what decide.
    """
    seconds = check_result.get("time")
    diagnostics = check_result.get("diagnostics") or {}
    peak_memory = diagnostics.get("memory_max")
    common = {"attempt_id": attempt_id, "verification_seconds": seconds, "peak_memory_bytes": peak_memory}

    error_text = check_result.get("error")
    if error_text:
        # The server's two timeout strings are "Lean REPL command timed out in N seconds" and
        # "Lean REPL header command timed out in ...". Anything else in `error` (a crashed worker, or
        # our own client giving up, whose text may itself mention an HTTP timeout) is a server error.
        text = str(error_text)
        is_lean_timeout = text.startswith("Lean REPL") and "timed out" in text
        status = VerificationStatus.TIMEOUT if is_lean_timeout else VerificationStatus.SERVER_ERROR
        return VerificationResult(status=status, detail=text, **common)

    response = check_result.get("response")
    if not isinstance(response, Mapping):
        return VerificationResult(status=VerificationStatus.SERVER_ERROR, detail="result has neither error nor response", **common)

    # The REPL itself rejected the command (for example it could not parse the request).
    if "message" in response and "messages" not in response:
        return VerificationResult(status=VerificationStatus.LEAN_ERROR, detail=str(response["message"]), **common)

    lean_messages = response.get("messages") or []
    message_texts = tuple(f"{message.get('severity')}: {message.get('data')}" for message in lean_messages)
    by_severity: dict[str, list[str]] = {"error": [], "warning": [], "info": []}
    for message in lean_messages:
        by_severity.setdefault(str(message.get("severity")), []).append(str(message.get("data", "")))
    common["messages"] = message_texts

    # DeepSeek's and STP's verifiers also treat a warning containing "failed" as an incomplete proof.
    failed_warning = failed_warning_is_error and any("failed" in warning for warning in by_severity["warning"])
    if by_severity["error"] or failed_warning:
        return VerificationResult(status=VerificationStatus.LEAN_ERROR, **common)

    if response.get("sorries") or any(_SORRY_WARNING_TEXT in warning for warning in by_severity["warning"]):
        return VerificationResult(status=VerificationStatus.USES_SORRY, **common)

    axioms = _parse_axioms(by_severity["info"])
    if axioms is None:
        # No axiom report means we cannot show the proof is axiom-clean, so it does not count.
        return VerificationResult(status=VerificationStatus.FORBIDDEN_AXIOM, detail="no `#print axioms` report in Lean's output", **common)
    if not set(axioms) <= ALLOWED_AXIOMS:
        return VerificationResult(status=VerificationStatus.FORBIDDEN_AXIOM, axioms=axioms, **common)

    return VerificationResult(status=VerificationStatus.VERIFIED, axioms=axioms, **common)
