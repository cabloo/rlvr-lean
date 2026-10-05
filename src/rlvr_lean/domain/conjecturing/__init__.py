"""Conjecturing context: what makes a generated statement usable (compiles, is new, is not vacuous)."""

from rlvr_lean.domain.conjecturing.statement_checks import (
    FINGERPRINT_MARKER,
    VACUITY_PROOF,
    build_compile_check_source,
    build_vacuity_source,
    fingerprint_command,
    parse_type_fingerprint,
)

__all__ = [
    "FINGERPRINT_MARKER",
    "VACUITY_PROOF",
    "build_compile_check_source",
    "build_vacuity_source",
    "fingerprint_command",
    "parse_type_fingerprint",
]
