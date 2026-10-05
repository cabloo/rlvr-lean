"""The Lean pin: which Lean and Mathlib a run is checked against, and everything that choice changes.
Spec: the OEIS Open spec, O2a items 9a, 9c and 9d.

`lean.pin` in the config is the ONE setting. What depends on it is a field of `LeanPin` here, or is built
from one by `infrastructure.verification_service` (the endpoint, the Lean timeout and the number of requests
in flight, which are the config's). Nothing else in the pipeline asks which pin is in use: it asks the pin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from rlvr_lean.domain.verification.lean_source import imports_first
from rlvr_lean.domain.verification.notation import membership_binders
from rlvr_lean.domain.verification.status import VerificationResult, classify_check_result


@dataclass(frozen=True)
class LeanPin:
    name: str
    # A warning containing "failed" is an incomplete proof (DeepSeek's and STP's convention; `status.py`).
    # On at v4.9, so Phase A and B are judged as they always were; off at v4.27, where a tactic that fails
    # inside a combinator that recovers warns while the proof is complete (O1: one gold proof of 38).
    failed_warning_is_error: bool
    # Move a file's leading `import` lines above a leading comment before it is sent (`imports_first`; O1:
    # 30 of the 38 gold proofs). A file the pipeline assembles starts with its imports, so this changes
    # nothing for it; it is what lets a benchmark file through.
    imports_above_comments: bool
    # True: the endpoint is a lean-pool front door (the config's `lean.pool`): HTTPS, the pool's own
    # certificate authority and no other, the certificate checked against the pool's NAME. False: the single
    # server of the config's `kimina` section, plain HTTP, addressed directly.
    through_pool: bool
    # Where this pin's runs live under the store root (item 9c): one directory per profile inside it. Each
    # pin has its own, so no file, marker or adapter of one pin is read as another's or written over.
    runs_directory: str
    # Item 9d: this pin's seed and reward statements are the named pin's, filtered again by compiling each
    # under this pin. None: the statements are chosen from the ranked workbook candidates, as always.
    statements_from_pin: str | None = None
    # True: this pin's Mathlib no longer parses `∑ x in s, f x` (spec ladder-loop, decided 2026-10-04). A
    # published statement or proof is rewritten to `∑ x ∈ s, f x` by `rewrite`, and the caller marks it.
    membership_binders_only: bool = False

    def rewrite(self, lean_text: str) -> tuple[str, int]:
        """(a PUBLISHED statement or proof in this pin's notation, how many places were rewritten). Asked for
        explicitly by whoever reads published text; `source` never rewrites, so a proof the model wrote is
        judged as the model wrote it."""
        return membership_binders(lean_text) if self.membership_binders_only else (lean_text, 0)

    def source(self, lean_file: str) -> str:
        """`lean_file` as it is sent to this pin's server."""
        return imports_first(lean_file) if self.imports_above_comments else lean_file

    def classify(self, attempt_id: str, check_result: Mapping[str, Any]) -> VerificationResult:
        """One element of the server's `results`, read by this pin's rules."""
        return classify_check_result(attempt_id, check_result, failed_warning_is_error=self.failed_warning_is_error)


DEFAULT_LEAN_PIN = "v4.9"      # today's behaviour, unchanged; the default until the port is accepted (item 9a)

LEAN_PINS: dict[str, LeanPin] = {
    # Lean v4.9.0-rc1 and the 2024 Mathlib fork the prover was trained on: Phase A's and B's verifier.
    "v4.9": LeanPin(name="v4.9", failed_warning_is_error=True, imports_above_comments=False, through_pool=False,
                    runs_directory="runs"),
    # Lean v4.27.0 and Mathlib v4.27.0 in a formal-conjectures checkout: OEIS Open's pin (O1), behind the pool.
    "v4.27": LeanPin(name="v4.27", failed_warning_is_error=False, imports_above_comments=True, through_pool=True,
                     runs_directory="runs-v4.27", statements_from_pin="v4.9", membership_binders_only=True),
}


def lean_pin(name: str) -> LeanPin:
    """The pin called `name`. An unknown name is a configuration error, refused before anything is checked."""
    if name not in LEAN_PINS:
        raise ValueError(f"lean.pin is {name!r}; it must be one of {', '.join(sorted(LEAN_PINS))}")
    return LEAN_PINS[name]


def lean_pin_from_config(config: Mapping[str, Any]) -> LeanPin:
    """The config's `lean.pin`. A config without the setting is one from before the port: the default pin."""
    return lean_pin(str((config.get("lean") or {}).get("pin", DEFAULT_LEAN_PIN)))
