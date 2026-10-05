"""Integration test against the LIVE Kimina Lean Server (spec fixture 8). It skips itself unless both

    RLVR_LEAN_KIMINA_URL       e.g. http://lean-server.example:18000
    RLVR_LEAN_KIMINA_API_KEY   the server's API key

are set. It sends raw Lean sources, bypassing the lexical filter on purpose: the point is to see what the
SERVER and our classifier do with `sorry`, `admit`, a declared axiom and a file that outlives the timeout.

NOT YET RUN: the server does not exist until its installer has been run. Milestone 1's acceptance is
this file passing from the dev machine and from a probe task on the GPU box.
"""

import asyncio
import os

import pytest

httpx = pytest.importorskip("httpx", reason="rlvr_lean's client dependency; see pyproject.toml")

from rlvr_lean.domain.verification import (  # noqa: E402
    LEAN_HEADER,
    VerificationStatus,
    build_proof_source,
    classify_check_result,
)
from rlvr_lean.infrastructure.kimina_client import KiminaClientSettings, KiminaVerifier, LeanSnippet  # noqa: E402

KIMINA_URL = os.environ.get("RLVR_LEAN_KIMINA_URL")
KIMINA_API_KEY = os.environ.get("RLVR_LEAN_KIMINA_API_KEY")
pytestmark = pytest.mark.skipif(not (KIMINA_URL and KIMINA_API_KEY), reason="live Kimina server not configured")

STATEMENT = "theorem rlvr_check (x : ℝ) (h₀ : x + 1 = 2) : x = 1 := by\n"
FALSE_STATEMENT = "theorem rlvr_check (x : ℝ) (h₀ : x + 1 = 2) : x = 5 := by\n"
AXIOM_STATEMENT = "axiom rlvr_cheat : False\n\ntheorem rlvr_check (x : ℝ) (h₀ : x + 1 = 2) : x = 5 := by\n"
# A file that simply waits longer than the Lean timeout: deterministic, and light on CPU and memory.
SLEEPING_SOURCE = LEAN_HEADER + "#eval IO.sleep 30000\n"

CASES = {
    "valid": (build_proof_source(STATEMENT, "  linarith\n"), {VerificationStatus.VERIFIED}),
    "invalid": (build_proof_source(FALSE_STATEMENT, "  linarith\n"), {VerificationStatus.LEAN_ERROR}),
    "sorry": (build_proof_source(FALSE_STATEMENT, "  sorry\n"), {VerificationStatus.USES_SORRY}),
    # Kimina's own client calls `admit` valid; ours must not, by the warning or by the axiom report.
    "admit": (build_proof_source(FALSE_STATEMENT, "  admit\n"),
              {VerificationStatus.USES_SORRY, VerificationStatus.FORBIDDEN_AXIOM}),
    "axiom": (build_proof_source(AXIOM_STATEMENT, "  exact absurd rlvr_cheat id\n"), {VerificationStatus.FORBIDDEN_AXIOM}),
    "timeout": (SLEEPING_SOURCE, {VerificationStatus.TIMEOUT}),
}


def run_cases(lean_timeout_seconds):
    async def run():
        settings = KiminaClientSettings(base_url=KIMINA_URL, api_key=KIMINA_API_KEY,
                                        lean_timeout_seconds=lean_timeout_seconds, snippets_per_request=3)
        async with KiminaVerifier(settings) as verifier:
            assert await verifier.is_healthy()
            return await verifier.check([LeanSnippet(name, source) for name, (source, _) in CASES.items()])

    return {result["id"]: classify_check_result(result["id"], result) for result in asyncio.run(run())}


def test_live_server_statuses():
    results = run_cases(lean_timeout_seconds=20)
    wrong = {name: (result.status.value, result.detail, result.messages)
             for name, result in results.items() if result.status not in CASES[name][1]}
    assert not wrong, wrong
    assert set(results["valid"].axioms) <= {"propext", "Classical.choice", "Quot.sound"}
    assert "rlvr_cheat" in results["axiom"].axioms
