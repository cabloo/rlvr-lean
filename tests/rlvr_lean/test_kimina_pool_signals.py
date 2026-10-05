"""The client's side of a lean-pool's two signals (lean_pool/README.md, "Background work and the pool's size"): priority, and the pool stating
its size (advisory): the priority header is sent only by a client configured with it; with `auto`
the requests in flight follow the size the pool states and fall back, saying so once, on anything unreadable;
the limit changes while requests are in flight without losing or doubling one; a background client waits out a
503 from a pool that is up. No network: a fake server inside httpx."""

import asyncio
import importlib.util
import json
import random
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx", reason="rlvr_lean's client dependency; see pyproject.toml")

from rlvr_lean.infrastructure import kimina_client  # noqa: E402
from rlvr_lean.infrastructure.kimina_client import (  # noqa: E402
    BACKGROUND_PRIORITY,
    POOL_WORKERS_FIELD,
    POOL_WORKERS_HEADER,
    PRIORITY_HEADER,
    KiminaClientSettings,
    KiminaVerifier,
    LeanSnippet,
    RequestSlots,
    pool_workers,
)
from rlvr_lean.infrastructure.verification_service import (  # noqa: E402
    AUTO_IN_FLIGHT,
    LeanCheckSettings,
    following_the_pool,
    lean_settings,
    waiting_out_the_proxy_queue,
)

ROOT = Path(__file__).resolve().parents[2]


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make_verifier(handler, clock=None, **overrides):
    settings = KiminaClientSettings(base_url="http://pool.test", api_key="secret", **overrides)
    http_client = httpx.AsyncClient(base_url=settings.base_url, transport=httpx.MockTransport(handler),
                                    headers={"Authorization": "Bearer secret"})
    waits, notices = [], []

    async def record_sleep(seconds):
        waits.append(seconds)

    verifier = KiminaVerifier(settings, http_client=http_client, sleep=record_sleep, random_source=random.Random(0),
                              on_notice=notices.append, clock=clock or Clock())
    return verifier, waits, notices


def results_of(request, headers=None):
    snippets = json.loads(request.content)["snippets"]
    return httpx.Response(200, headers=headers or {}, json={"results": [
        {"id": snippet["id"], "time": 0.1, "response": {"env": 0, "messages": [], "sorries": []}} for snippet in snippets]})


def snippets(count, tag="s"):
    return [LeanSnippet(snippet_id=f"{tag}{index}", code=f"-- {tag} {index}") for index in range(count)]


# ------------------------------------------------------------------------------------------------ the two names

def test_the_names_are_the_pools_own():
    """This package imports nothing from the lean_pool project, so the header names are written twice."""
    spec = importlib.util.spec_from_file_location("leanpool_signals", ROOT / "lean_pool/leanpool/signals.py")
    signals = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = signals            # dataclasses resolve the module of a class they decorate
    try:
        spec.loader.exec_module(signals)
    finally:
        del sys.modules[spec.name]
    assert (PRIORITY_HEADER, BACKGROUND_PRIORITY) == (signals.PRIORITY_HEADER, signals.BACKGROUND_PRIORITY)
    assert (POOL_WORKERS_HEADER, POOL_WORKERS_FIELD) == (signals.WORKERS_HEADER, signals.WORKERS_FIELD)


@pytest.mark.parametrize("value,workers", [("28", 28), (" 4 ", 4), (12, 12), ("1", 1)])
def test_a_stated_size_is_a_whole_number_of_at_least_one(value, workers):
    assert pool_workers(value) == workers


@pytest.mark.parametrize("value", [None, "", " ", "0", 0, "-3", -3, "2.5", 2.5, "many", "1e2", True, "9" * 12, [28], {"workers": 28}])
def test_anything_else_states_no_size(value):
    assert pool_workers(value) is None


# -------------------------------------------------------------------------------------------------- priority

def test_the_priority_header_is_sent_only_by_a_client_configured_with_it():
    seen = []

    def handler(request):
        seen.append(request.headers.get(PRIORITY_HEADER))
        return results_of(request)

    for priority in (None, BACKGROUND_PRIORITY):
        verifier, _, _ = make_verifier(handler, **({"priority": priority} if priority else {}))
        asyncio.run(verifier.check(snippets(2)))
    assert seen == [None, None, BACKGROUND_PRIORITY, BACKGROUND_PRIORITY]


def test_a_background_client_waits_out_a_busy_pool_and_never_records_the_503():
    """The pool's queue timed the check out behind work that goes first. The pool is up: pause, ask again, as
    often as it takes; the pauses are not attempts and the 503 is never a result."""
    calls = {"check": 0, "health": 0}

    def handler(request):
        if request.url.path == "/health":
            calls["health"] += 1
            return httpx.Response(200, json={"status": "ok"})
        calls["check"] += 1
        if calls["check"] <= 9:                 # more refusals than a normal client would ever sit through
            return httpx.Response(503, text="<html><body><h1>503 Service Unavailable</h1>")
        return results_of(request)

    verifier, waits, _ = make_verifier(handler, priority=BACKGROUND_PRIORITY, busy_pause_seconds=30.0)
    (result,) = asyncio.run(verifier.check(snippets(1)))
    assert "error" not in result and result["response"]["env"] == 0
    assert calls == {"check": 10, "health": 9}
    assert waits == [30.0] * 9 and verifier.busy_pauses == 9


def test_a_normal_client_treats_a_503_as_it_always_did():
    def handler(request):
        return httpx.Response(503, text="no server is available")

    verifier, waits, _ = make_verifier(handler)
    (result,) = asyncio.run(verifier.check(snippets(1)))
    assert result["error"].startswith("server_error: no answer after 4 attempts") and len(waits) == 3
    assert verifier.busy_pauses == 0


def test_a_background_client_does_not_wait_on_a_pool_that_is_down():
    """503 with a /health that fails is a pool with no Lean server: the ordinary bounded retries, then no answer."""
    def handler(request):
        return httpx.Response(503, text="no server is available")       # /health too

    verifier, waits, _ = make_verifier(handler, priority=BACKGROUND_PRIORITY)
    (result,) = asyncio.run(verifier.check(snippets(1)))
    assert result["error"].startswith("server_error: no answer after 4 attempts")
    assert verifier.busy_pauses == 0 and 30.0 not in waits


def test_the_busy_pauses_are_bounded():
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(503, text="busy")

    verifier, waits, _ = make_verifier(handler, priority=BACKGROUND_PRIORITY, busy_pauses_allowed=5)
    (result,) = asyncio.run(verifier.check(snippets(1)))
    assert result["error"].startswith("server_error") and verifier.busy_pauses == 5


# ----------------------------------------------------------------------------------------- the slots themselves

def test_the_limit_changes_while_requests_are_in_flight_and_waiting():
    async def scenario():
        slots, running, peak, done = RequestSlots(2), 0, 0, []
        release = asyncio.Event()

        async def holder(name):
            nonlocal running, peak
            async with slots:
                running += 1
                peak = max(peak, running)
                await release.wait()
                running -= 1
                done.append(name)

        tasks = [asyncio.create_task(holder(index)) for index in range(6)]
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert (running, slots.in_use) == (2, 2)
        slots.set_limit(5)                      # raised: three waiters are let in at once
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert (running, slots.in_use) == (5, 5)
        slots.set_limit(1)                      # lowered: nobody is thrown out, nobody is let in
        await asyncio.sleep(0)
        assert (running, slots.in_use) == (5, 5)
        release.set()
        await asyncio.gather(*tasks)
        assert sorted(done) == list(range(6)) and (slots.in_use, peak) == (0, 5)

    asyncio.run(scenario())


def test_after_the_limit_is_lowered_nobody_enters_until_enough_have_left():
    async def scenario():
        slots, entered = RequestSlots(3), []
        gates = [asyncio.Event() for _ in range(5)]

        async def holder(index):
            async with slots:
                entered.append(index)
                await gates[index].wait()

        tasks = [asyncio.create_task(holder(index)) for index in range(5)]
        await asyncio.sleep(0.01)
        assert entered == [0, 1, 2]
        slots.set_limit(1)
        gates[0].set()
        gates[1].set()
        await asyncio.sleep(0.01)
        assert entered == [0, 1, 2] and slots.in_use == 1      # two left, one still holds: the limit is met
        gates[2].set()
        await asyncio.sleep(0.01)
        assert entered == [0, 1, 2, 3] and slots.in_use == 1   # first come, first served
        gates[3].set()
        gates[4].set()
        await asyncio.gather(*tasks)
        assert slots.in_use == 0

    asyncio.run(scenario())


def test_a_cancelled_waiter_loses_no_slot_and_takes_none():
    async def scenario():
        slots = RequestSlots(1)
        gate = asyncio.Event()

        async def holder():
            async with slots:
                await gate.wait()

        first = asyncio.create_task(holder())
        await asyncio.sleep(0)
        waiting, later = asyncio.create_task(holder()), asyncio.create_task(holder())
        await asyncio.sleep(0)
        waiting.cancel()                        # cancelled while it waits
        await asyncio.sleep(0)
        gate.set()
        await asyncio.gather(first, later)
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert slots.in_use == 0
        # Cancelled in the very moment the slot is handed over: the slot goes back.
        await slots.acquire()                    # this test holds the one slot itself
        handed = asyncio.create_task(holder())
        await asyncio.sleep(0)                   # it waits
        slots.release()                          # handed to it and counted; it has not run yet
        assert slots.in_use == 1
        handed.cancel()
        with pytest.raises(asyncio.CancelledError):
            await handed
        assert slots.in_use == 0
        async with slots:                        # and the slot is usable again
            assert slots.in_use == 1

    asyncio.run(scenario())


def test_a_storm_of_checks_under_a_limit_that_keeps_changing_loses_and_doubles_nothing():
    async def scenario():
        randomness = random.Random(7)
        slots, running, peak, breaches, finished = RequestSlots(4), 0, 0, 0, 0

        async def one(delay):
            nonlocal running, peak, breaches, finished
            async with slots:
                running += 1
                peak = max(peak, running)
                # A holder always has a counted slot (a slot handed to a waiter is counted before it runs,
                # so the count may be ahead of the holders, never behind).
                breaches += running > slots.in_use
                try:
                    await asyncio.sleep(delay)
                finally:
                    running -= 1                                   # also when cancelled while holding its slot
                finished += 1

        async def meddle():
            for _ in range(200):
                slots.set_limit(randomness.randint(1, 12))
                await asyncio.sleep(0.0005)

        tasks = [asyncio.create_task(one(randomness.random() * 0.002)) for _ in range(400)]
        meddler = asyncio.create_task(meddle())
        await asyncio.sleep(0.003)
        for index in (5, 50, 90, 200, 333, 399):
            tasks[index].cancel()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        await meddler
        cancelled = sum(isinstance(outcome, asyncio.CancelledError) for outcome in outcomes)
        assert finished + cancelled == 400 and cancelled >= 1          # every check ran once or was cancelled
        assert breaches == 0 and peak <= 12                            # none ran without a slot; never above a limit
        assert (slots.in_use, running) == (0, 0)                       # and no slot was lost

    asyncio.run(scenario())


# -------------------------------------------------------------------------------------- following the pool's size

def test_auto_asks_health_once_and_keeps_the_stated_workers_times_the_margin_in_flight():
    health, in_flight, peak = [], 0, 0

    async def handler(request):
        nonlocal in_flight, peak
        if request.url.path == "/health":
            health.append(1)
            return httpx.Response(200, json={"status": "ok", "workers": 8, "queued": 0, "servers": 2})
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.005)
        in_flight -= 1
        return results_of(request, {POOL_WORKERS_HEADER: "8"})

    verifier, _, notices = make_verifier(handler, follow_pool_size=True, concurrent_requests=3)

    async def scenario():
        await verifier.check(snippets(40, "a"))
        await verifier.check(snippets(40, "b"))

    asyncio.run(scenario())
    assert len(health) == 1 and verifier.requests_in_flight_limit == 10 and peak == 10      # ceil(8 x 1.25)
    assert notices == ["the pool states 8 workers: 10 requests in flight (was 3)"]


def test_auto_follows_a_size_that_changes_at_most_once_per_refresh_period_and_never_above_the_ceiling():
    clock, stated = Clock(), {"workers": "28"}

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "workers": 28, "queued": 0, "servers": 3})
        return results_of(request, {POOL_WORKERS_HEADER: stated["workers"]})

    verifier, _, notices = make_verifier(handler, clock=clock, follow_pool_size=True, concurrent_requests=40,
                                         pool_size_ceiling=48, pool_size_refresh_seconds=30.0)

    async def scenario():
        limits = []
        await verifier.check(snippets(3, "a"))
        limits.append(verifier.requests_in_flight_limit)       # 35 = ceil(28 x 1.25)
        stated["workers"] = "34"                                # a server joined
        clock.now += 10
        await verifier.check(snippets(3, "b"))
        limits.append(verifier.requests_in_flight_limit)       # still 35: read 10 s ago
        clock.now += 25
        await verifier.check(snippets(3, "c"))
        limits.append(verifier.requests_in_flight_limit)       # 43 = ceil(34 x 1.25)
        stated["workers"] = "200"
        clock.now += 31
        await verifier.check(snippets(3, "d"))
        limits.append(verifier.requests_in_flight_limit)       # the ceiling, whatever the pool says
        stated["workers"] = "13"                                # two servers dropped
        clock.now += 31
        await verifier.check(snippets(3, "e"))
        limits.append(verifier.requests_in_flight_limit)       # 17 = ceil(13 x 1.25)
        return limits

    assert asyncio.run(scenario()) == [35, 35, 43, 48, 17]
    assert len(notices) == 4


@pytest.mark.parametrize("header", [None, "", "0", "lots", "2.5", "-4"])
def test_a_pool_that_states_no_usable_size_leaves_the_configured_number_and_is_mentioned_once(header):
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})            # the pool as it is installed today
        return results_of(request, {} if header is None else {POOL_WORKERS_HEADER: header})

    verifier, _, notices = make_verifier(handler, follow_pool_size=True, concurrent_requests=40)

    async def scenario():
        first = await verifier.check(snippets(5, "a"))
        second = await verifier.check(snippets(5, "b"))
        return first + second

    results = asyncio.run(scenario())
    assert len(results) == 10 and all("error" not in result for result in results)
    assert verifier.requests_in_flight_limit == 40
    assert len(notices) == 1 and "keeping 40 requests in flight (said once)" in notices[0]


@pytest.mark.parametrize("health", ["refused", "500", "not json", "a list", "no number", "503 no server"])
def test_a_health_that_fails_or_says_nothing_costs_no_check(health):
    def handler(request):
        if request.url.path == "/health":
            if health == "refused":
                raise httpx.ConnectError("connection refused")
            if health == "500":
                return httpx.Response(500, text="oops")
            if health == "not json":
                return httpx.Response(200, text="<html>ok</html>")
            if health == "a list":
                return httpx.Response(200, json=[1, 2, 3])
            if health == "503 no server":
                return httpx.Response(503, json={"status": "no Lean server is up", "workers": 0, "queued": 0, "servers": 0})
            return httpx.Response(200, json={"status": "ok", "workers": "plenty"})
        return results_of(request)

    verifier, _, notices = make_verifier(handler, follow_pool_size=True, concurrent_requests=7)
    results = asyncio.run(verifier.check(snippets(4)))
    assert len(results) == 4 and all("error" not in result for result in results)
    assert verifier.requests_in_flight_limit == 7 and len(notices) == 1


def test_a_client_that_does_not_follow_ignores_what_the_pool_states_and_asks_no_health():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return results_of(request, {POOL_WORKERS_HEADER: "28"})

    verifier, _, notices = make_verifier(handler, concurrent_requests=5)
    asyncio.run(verifier.check(snippets(6)))
    assert verifier.requests_in_flight_limit == 5 and "/health" not in paths and notices == []


def test_the_size_is_followed_on_an_answer_that_is_an_error_too():
    """HAProxy sets the headers on every answer that leaves the front door, its own 503 included."""
    calls = []

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, headers={POOL_WORKERS_HEADER: "16"}, text="queue timeout")
        return results_of(request, {POOL_WORKERS_HEADER: "16"})

    verifier, _, _ = make_verifier(handler, follow_pool_size=True, concurrent_requests=4)
    asyncio.run(verifier.check(snippets(1)))
    assert verifier.requests_in_flight_limit == 20


# ------------------------------------------------------------------------------------------ the settings helpers

def pool_settings(**overrides):
    return LeanCheckSettings(base_url="https://192.0.2.1:18100", api_key="k", concurrent_requests=8, **overrides)


def test_a_number_is_a_number_and_auto_follows_the_pool_with_a_fallback():
    settings = pool_settings()
    assert following_the_pool(settings, None) is settings and following_the_pool(settings, 0) is settings
    assert following_the_pool(settings, 24).concurrent_requests == 24
    assert following_the_pool(settings, 24).follow_pool_size is False
    auto = following_the_pool(settings, AUTO_IN_FLIGHT, fallback=40)
    assert (auto.follow_pool_size, auto.concurrent_requests) == (True, 40)
    assert following_the_pool(settings, AUTO_IN_FLIGHT).concurrent_requests == 8
    for wrong in ("automatic", -1, 2.5, True, "12"):
        with pytest.raises(ValueError):
            following_the_pool(settings, wrong)


def test_a_client_that_waits_in_the_pools_queue_waits_as_long_as_the_queue_may_hold_it():
    from rlvr_lean.domain.verification import LEAN_PINS
    config = {"lean": {"pool": {"proxy_queue_seconds": 630, "server_wait_seconds": 120}}}
    through_pool = next(pin for pin in LEAN_PINS.values() if pin.through_pool)
    direct = next(pin for pin in LEAN_PINS.values() if not pin.through_pool)
    waiting = waiting_out_the_proxy_queue(pool_settings(pin=through_pool, lean_timeout_seconds=30), config)
    assert waiting.server_queue_wait_seconds == 750.0 and waiting.http_timeout_seconds == 870.0
    unchanged = pool_settings(pin=direct)
    assert waiting_out_the_proxy_queue(unchanged, config) is unchanged
    assert waiting_out_the_proxy_queue(pool_settings(pin=through_pool), {"lean": {"pool": {}}}).server_queue_wait_seconds == 120.0


def test_the_configs_margin_and_ceiling_reach_a_pool_client(monkeypatch, tmp_path):
    import yaml
    from rlvr_lean.infrastructure import verification_service

    config = yaml.safe_load((ROOT / "src/rlvr_lean/config/experiment.yaml").read_text())
    config["lean"]["pin"] = "v4.27"
    monkeypatch.setattr(verification_service, "resolve_lan_host", lambda name, server: ("192.0.2.9", name))
    certificate = tmp_path / "ca.crt"
    certificate.write_text("not read here")
    settings = lean_settings(config, api_key="k", ca_file=str(certificate))
    assert (settings.pool_size_margin, settings.pool_size_ceiling) == (1.25, 256)
    assert settings.follow_pool_size is False and settings.priority is None        # nothing changes unasked
    assert config["ladder_loop"]["lean_in_flight_fallback"] == 40
    assert kimina_client.KiminaClientSettings(base_url="").pool_size_margin == 1.25


# ------------------------------------------------------------------------------ the bulk certificate check

def _tool_arguments(tmp_path, in_flight):
    import argparse
    key, certificate = tmp_path / "key", tmp_path / "ca.crt"
    key.write_text("pool-key\n")
    certificate.write_text("not read here")
    return argparse.Namespace(api_key_file=key, ca_file=certificate, in_flight=in_flight)


def _tool_config(monkeypatch):
    import yaml
    from rlvr_lean.infrastructure import verification_service

    monkeypatch.setattr(verification_service, "resolve_lan_host", lambda name, server: ("192.0.2.9", name))
    return yaml.safe_load((ROOT / "src/rlvr_lean/config/experiment.yaml").read_text())


def test_in_flight_is_a_number_or_auto():
    import argparse
    from rlvr_lean.tools.ladder_pool import in_flight_argument

    assert in_flight_argument("14") == 14 and in_flight_argument("auto") == AUTO_IN_FLIGHT
    assert in_flight_argument(" AUTO ") == AUTO_IN_FLIGHT
    for wrong in ("0", "-2", "many", "2.5", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            in_flight_argument(wrong)


def test_the_bulk_check_is_a_background_client_that_waits_out_the_pools_queue(monkeypatch, tmp_path):
    from rlvr_lean.tools import ladder_pool

    config = _tool_config(monkeypatch)
    bulk = ladder_pool.pin_settings(_tool_arguments(tmp_path, 14), config, background=True)
    assert (bulk.priority, bulk.concurrent_requests, bulk.follow_pool_size) == (BACKGROUND_PRIORITY, 14, False)
    assert bulk.server_queue_wait_seconds == 750.0          # the proxy's queue (630) and a server's own wait (120)
    assert "14 in flight" == ladder_pool.in_flight_text(bulk)
    auto = ladder_pool.pin_settings(_tool_arguments(tmp_path, AUTO_IN_FLIGHT), config, background=True)
    assert (auto.priority, auto.follow_pool_size, auto.concurrent_requests) == (BACKGROUND_PRIORITY, True, 8)
    assert "follow the pool's stated size x 1.25 (8 until it states one, at most 256)" in ladder_pool.in_flight_text(auto)
    # The live sample is a measurement with its own budget: a normal client, as it was.
    sample = ladder_pool.pin_settings(_tool_arguments(tmp_path, None), config)
    assert (sample.priority, sample.follow_pool_size, sample.concurrent_requests) == (None, False, 8)
    assert sample.server_queue_wait_seconds == KiminaClientSettings(base_url="").server_queue_wait_seconds


def test_the_episode_steps_send_no_priority_and_take_a_number_or_auto(monkeypatch, tmp_path):
    from rlvr_lean.gpu import ladder_loop
    from rlvr_lean.infrastructure import verification_service

    config = _tool_config(monkeypatch)
    certificate = tmp_path / "ca.crt"
    certificate.write_text("not read here")
    monkeypatch.setenv(verification_service.API_KEY_VARIABLE, "pool-key")
    monkeypatch.setenv(verification_service.CA_FILE_VARIABLE, str(certificate))
    # As shipped since the pool states its size (2026-10-04): `auto`, starting from the fallback.
    shipped = ladder_loop._lean_settings(config, lean_seconds=30)
    assert (shipped.priority, shipped.follow_pool_size, shipped.concurrent_requests) == (None, True, 40)
    assert (shipped.lean_timeout_seconds, shipped.http_timeout_seconds) == (30, 870.0)
    config["ladder_loop"]["lean_in_flight"] = 24
    fixed = ladder_loop._lean_settings(config, lean_seconds=30)
    assert (fixed.priority, fixed.follow_pool_size, fixed.concurrent_requests) == (None, False, 24)
    config["ladder_loop"]["lean_in_flight"] = AUTO_IN_FLIGHT
    auto = ladder_loop._lean_settings(config, lean_seconds=30)
    assert (auto.priority, auto.follow_pool_size, auto.concurrent_requests) == (None, True, 40)   # the fallback
    assert (auto.pool_size_margin, auto.pool_size_ceiling, auto.http_timeout_seconds) == (1.25, 256, 870.0)


def test_the_checker_keeps_as_many_checks_in_flight_as_its_client_does_now(monkeypatch, tmp_path):
    """`run_checks` has one worker per possible request; those beyond the client's present limit stand by, and
    join in when the limit rises (a server joined) without a restart."""
    from rlvr_lean.domain.verification import LEAN_PINS
    from rlvr_lean.tools import version_tax

    through_pool = next(pin for pin in LEAN_PINS.values() if pin.through_pool)
    state = {"limit": 2, "in_flight": 0, "answered": 0, "peak_before": 0, "peak_after": 0}

    class FollowingStandIn:
        busy_pauses = 0

        def __init__(self, settings, on_request_failure=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exception_info):
            return None

        async def is_healthy(self):
            return True

        @property
        def requests_in_flight_limit(self):
            return state["limit"]

        async def check(self, snippets):
            state["in_flight"] += 1
            phase = "peak_before" if state["limit"] == 2 else "peak_after"
            state[phase] = max(state[phase], state["in_flight"])
            await asyncio.sleep(0.002)
            state["in_flight"] -= 1
            state["answered"] += 1
            if state["answered"] == 20:
                state["limit"] = 6                     # the pool grew
            return [{"id": snippets[0].snippet_id, "time": 0.1, "response": {"env": 0, "messages": []}}]

    monkeypatch.setattr(version_tax, "KiminaVerifier", FollowingStandIn)
    monkeypatch.setattr(version_tax, "STANDBY_SECONDS", 0.001)
    settings = LeanCheckSettings(base_url="https://192.0.2.9:18100", pin=through_pool, concurrent_requests=2,
                                 follow_pool_size=True, pool_size_ceiling=16)
    jobs = [version_tax.Job(f"sha{index}", "proof", f"theorem t{index} : True := by trivial") for index in range(400)]
    session = asyncio.run(version_tax.run_checks(settings, jobs, tmp_path, retry_passes=0))
    assert session["answered"] == 400 and session["in_flight"] == 6 and session["in_flight_follows_the_pool"] is True
    assert state["peak_before"] == 2 and state["peak_after"] == 6
