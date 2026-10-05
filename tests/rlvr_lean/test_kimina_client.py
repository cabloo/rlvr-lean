"""The async Kimina client against a fake server (no network): batching, bounded concurrency,
retry with backoff, batch splitting, and the rule that a Lean timeout is an answer and not a failure.
Spec: the first experiment's spec §7, "Kimina client"."""

import asyncio
import json
import random

import pytest

httpx = pytest.importorskip("httpx", reason="rlvr_lean's client dependency; see pyproject.toml")

from rlvr_lean.infrastructure.kimina_client import KiminaClientSettings, KiminaVerifier, LeanSnippet  # noqa: E402


def make_verifier(handler, **overrides):
    settings = KiminaClientSettings(base_url="http://kimina.test", api_key="secret", **overrides)
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(base_url=settings.base_url, transport=transport,
                                    headers={"Authorization": "Bearer secret"})
    waits = []

    async def record_sleep(seconds):
        waits.append(seconds)

    verifier = KiminaVerifier(settings, http_client=http_client, sleep=record_sleep, random_source=random.Random(0))
    return verifier, waits


def ok_reply(request):
    snippets = json.loads(request.content)["snippets"]
    return httpx.Response(200, json={"results": [
        {"id": snippet["id"], "time": 0.1, "response": {"env": 0, "messages": [], "sorries": []}} for snippet in snippets]})


def snippets(count):
    return [LeanSnippet(snippet_id=f"s{index}", code=f"-- {index}") for index in range(count)]


def test_results_come_back_in_input_order_in_batches_with_the_api_key():
    seen_batch_sizes, seen_auth = [], set()

    def handler(request):
        body = json.loads(request.content)
        seen_batch_sizes.append(len(body["snippets"]))
        seen_auth.add(request.headers.get("Authorization"))
        assert body["timeout"] == 60 and request.url.path == "/api/check"
        return ok_reply(request)

    verifier, _ = make_verifier(handler, snippets_per_request=4)
    results = asyncio.run(verifier.check(snippets(10)))
    assert [result["id"] for result in results] == [f"s{index}" for index in range(10)]
    assert sorted(seen_batch_sizes) == [2, 4, 4] and seen_auth == {"Bearer secret"}


def test_concurrency_never_exceeds_the_limit():
    in_flight = peak = 0

    async def handler(request):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return ok_reply(request)

    verifier, _ = make_verifier(handler, snippets_per_request=1, concurrent_requests=3)
    asyncio.run(verifier.check(snippets(12)))
    assert peak == 3


def test_retryable_failures_are_retried_with_growing_backoff():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, text="No available REPLs")
        if len(calls) == 2:
            raise httpx.ConnectError("connection refused")
        return ok_reply(request)

    verifier, waits = make_verifier(handler, backoff_initial_seconds=1.0, snippets_per_request=4)
    results = asyncio.run(verifier.check(snippets(2)))
    assert len(calls) == 3 and len(results) == 2
    assert len(waits) == 2 and 0 <= waits[0] <= 1.0 and 0 <= waits[1] <= 2.0   # jittered, capped exponential


def test_a_poison_snippet_is_isolated_and_its_neighbours_still_get_answers():
    def handler(request):
        ids = [snippet["id"] for snippet in json.loads(request.content)["snippets"]]
        if "s2" in ids:
            return httpx.Response(500, text="JSON decode error")   # a crashed worker fails the whole request
        return ok_reply(request)

    verifier, _ = make_verifier(handler, snippets_per_request=4, max_attempts_per_batch=2)
    results = asyncio.run(verifier.check(snippets(4)))
    by_id = {result["id"]: result for result in results}
    assert [result["id"] for result in results] == ["s0", "s1", "s2", "s3"]
    assert by_id["s2"]["error"].startswith("server_error:")
    assert all("response" in by_id[name] for name in ("s0", "s1", "s3"))


def test_a_crashed_worker_splits_the_batch_at_once_instead_of_repeating_the_crash():
    calls = []

    def handler(request):
        ids = [snippet["id"] for snippet in json.loads(request.content)["snippets"]]
        calls.append(ids)
        return httpx.Response(500, text="JSON decode error") if "s2" in ids else ok_reply(request)

    verifier, _ = make_verifier(handler, snippets_per_request=4, max_attempts_per_batch=4, max_attempts_after_crash=2)
    asyncio.run(verifier.check(snippets(4)))
    # batch of 4 once, each half once, the culprit twice on its own, its neighbour once.
    assert sorted(map(tuple, calls)) == sorted([("s0", "s1", "s2", "s3"), ("s0", "s1"), ("s2", "s3"),
                                                ("s2",), ("s2",), ("s3",)])


def test_an_overloaded_server_is_retried_whole_not_split():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, text="busy") if len(calls) < 3 else ok_reply(request)

    verifier, waits = make_verifier(handler, snippets_per_request=4)
    asyncio.run(verifier.check(snippets(4)))
    assert len(calls) == 3 and len(waits) == 2


def test_a_lean_timeout_is_returned_once_and_never_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"results": [
            {"id": "s0", "time": 60, "error": "Lean REPL command timed out in 60 seconds"}]})

    verifier, waits = make_verifier(handler)
    results = asyncio.run(verifier.check(snippets(1)))
    assert len(calls) == 1 and waits == [] and "timed out" in results[0]["error"]


def test_client_errors_are_raised_not_retried():
    def handler(request):
        return httpx.Response(401, json={"detail": "Invalid API key"})

    verifier, waits = make_verifier(handler)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(verifier.check(snippets(1)))
    assert waits == []


def test_duplicate_snippet_ids_are_rejected_before_any_request():
    verifier, _ = make_verifier(ok_reply)
    with pytest.raises(ValueError):
        asyncio.run(verifier.check([LeanSnippet("same", "a"), LeanSnippet("same", "b")]))


def test_http_timeout_covers_queue_wait_header_and_body():
    settings = KiminaClientSettings(base_url="http://kimina.test", lean_timeout_seconds=60,
                                    server_queue_wait_seconds=120, http_margin_seconds=60)
    assert settings.http_timeout_seconds == 120 + 2 * 60 + 60
