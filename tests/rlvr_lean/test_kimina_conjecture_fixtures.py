"""Spec fixtures 9 and 10 against the LIVE server: the vacuity filter and the type fingerprint.
Skips unless RLVR_LEAN_KIMINA_URL and RLVR_LEAN_KIMINA_API_KEY are set (see test_kimina_integration.py)."""

import asyncio
import os

import pytest

httpx = pytest.importorskip("httpx", reason="rlvr_lean's client dependency; see pyproject.toml")

from rlvr_lean.domain.conjecturing import build_compile_check_source, build_vacuity_source, parse_type_fingerprint  # noqa: E402
from rlvr_lean.domain.verification import VerificationStatus, classify_check_result  # noqa: E402
from rlvr_lean.infrastructure.kimina_client import KiminaClientSettings, KiminaVerifier, LeanSnippet  # noqa: E402

KIMINA_URL = os.environ.get("RLVR_LEAN_KIMINA_URL")
KIMINA_API_KEY = os.environ.get("RLVR_LEAN_KIMINA_API_KEY")
pytestmark = pytest.mark.skipif(not (KIMINA_URL and KIMINA_API_KEY), reason="live Kimina server not configured")

VACUITY_CASES = {
    # name: (statement, should be flagged vacuous)
    "contradictory_hypotheses": ("theorem v1 (x : ℝ) (h₀ : x > 2) (h₁ : x < 1) : x = 7 := by\n", True),
    "contradiction_as_implications": ("theorem v2 (x : ℝ) : x > 2 → x < 1 → x = 7 := by\n", True),
    "true_negation_goal": ("theorem v3 (x : ℝ) (h₀ : x > 2) : ¬ (x < 1) := by\n", False),
    "consistent_hypotheses": ("theorem v4 (x : ℝ) (h₀ : x > 2) : x > 1 := by\n", False),
}
FINGERPRINT_CASES = {
    "original": "theorem f1 (a b : ℝ) (h : a < b) : a + 1 < b + 1 := by\n",
    "renamed": "theorem f2 (x y : ℝ) (hxy : x < y) : x + 1 < y + 1 := by\n",
    "reformatted": "theorem f3 (a b : ℝ)\n    (h : a < b) :\n    a + 1 < b + 1 := by\n",
    "different": "theorem f4 (a b : ℝ) (h : a < b) : a + 2 < b + 2 := by\n",
}


def run(snippets):
    async def go():
        settings = KiminaClientSettings(base_url=KIMINA_URL, api_key=KIMINA_API_KEY, lean_timeout_seconds=60)
        async with KiminaVerifier(settings) as verifier:
            return await verifier.check(snippets)

    return {result["id"]: result for result in asyncio.run(go())}


def test_vacuity_filter():
    results = run([LeanSnippet(name, build_vacuity_source(statement)) for name, (statement, _) in VACUITY_CASES.items()])
    flagged = {name: classify_check_result(name, result).status is VerificationStatus.VERIFIED
               for name, result in results.items()}
    assert flagged == {name: expected for name, (_, expected) in VACUITY_CASES.items()}


def test_type_fingerprint_ignores_names_and_layout_but_not_content():
    results = run([LeanSnippet(name, build_compile_check_source(statement)) for name, statement in FINGERPRINT_CASES.items()])
    fingerprints = {}
    for name, result in results.items():
        messages = [str(message.get("data")) for message in (result.get("response") or {}).get("messages", [])
                    if message.get("severity") == "info"]
        fingerprints[name] = parse_type_fingerprint(messages)
    assert all(value is not None for value in fingerprints.values()), (fingerprints, results)
    assert fingerprints["original"] == fingerprints["reformatted"]
    assert fingerprints["original"] != fingerprints["different"]
    assert fingerprints["original"] == fingerprints["renamed"], "hash depends on binder names: dedup must normalise them"
