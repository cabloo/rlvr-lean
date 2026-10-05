"""Verification context: what is sent to Lean for one proof attempt, and what counts as verified."""

from rlvr_lean.domain.verification.lean_source import (
    LEAN_HEADER,
    build_proof_source,
    find_forbidden_token,
    imports_first,
    theorem_name_of,
)
from rlvr_lean.domain.verification.pin import DEFAULT_LEAN_PIN, LEAN_PINS, LeanPin, lean_pin, lean_pin_from_config
from rlvr_lean.domain.verification.status import (
    ALLOWED_AXIOMS,
    VerificationResult,
    VerificationStatus,
    classify_check_result,
    rejected_lexically,
)

__all__ = [
    "ALLOWED_AXIOMS",
    "DEFAULT_LEAN_PIN",
    "LEAN_HEADER",
    "LEAN_PINS",
    "LeanPin",
    "VerificationResult",
    "VerificationStatus",
    "build_proof_source",
    "classify_check_result",
    "find_forbidden_token",
    "imports_first",
    "lean_pin",
    "lean_pin_from_config",
    "rejected_lexically",
    "theorem_name_of",
]
