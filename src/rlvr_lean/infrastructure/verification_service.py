"""Verification as a service the samplers can feed while they keep sampling (spec §1 item 10, §4).

`VerificationService.submit(attempts)` returns immediately; a background thread runs each batch through
the Kimina client (its own event loop per batch) while the caller samples the next chunk on the GPU.
`results()` waits for everything and returns attempt id -> VerificationResult. The lexical filter runs
first, so filtered completions never reach the server.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from rlvr_lean.domain.verification import (
    DEFAULT_LEAN_PIN,
    LEAN_PINS,
    LeanPin,
    VerificationResult,
    build_proof_source,
    find_forbidden_token,
    lean_pin_from_config,
    rejected_lexically,
)
from rlvr_lean.infrastructure.kimina_client import KiminaClientSettings, KiminaVerifier, LeanSnippet
from rlvr_lean.infrastructure.lan_resolver import resolve_lan_host

# How a step is told the key and the authority's certificate: the entry shim sets both (`--kimina-api-key`,
# `--kimina-ca-certificate`), and neither is ever printed. The second is a FILE's path: the entry writes the
# certificate it was handed into the store.
API_KEY_VARIABLE = "RLVR_LEAN_KIMINA_API_KEY"
CA_FILE_VARIABLE = "RLVR_LEAN_KIMINA_CA_FILE"


@dataclass(frozen=True)
class ProofAttemptToVerify:
    attempt_id: str
    statement: str       # the theorem up to `:= by`
    completion: str      # the model's proof


@dataclass(frozen=True)
class LeanCheckSettings(KiminaClientSettings):
    """Where one Lean pin's checks are sent (the client's settings) and the pin whose rules write and read
    them. Built from the config's ONE setting, `lean.pin` (the OEIS Open spec, O2a item 9a)."""

    pin: LeanPin = LEAN_PINS[DEFAULT_LEAN_PIN]


def lean_settings(config: dict, api_key: str, ca_file: str | None = None) -> LeanCheckSettings:
    """The settings of the pin the config selects: its endpoint, Lean timeout and requests in flight.

    v4.9: the single server of the `kimina` section, plain HTTP, at its resolved address (unchanged).
    v4.27: the pool of `lean.pool` (item 9b): HTTPS to the address the pool's NAME resolves to, the
    certificate checked against that name (and the name sent for SNI), trusting `ca_file`'s authority and
    no other."""
    pin = lean_pin_from_config(config)
    kimina = config["kimina"]
    if not pin.through_pool:
        address, _ = resolve_lan_host(kimina["host"], kimina["lan_dns_server"])
        return LeanCheckSettings(base_url=f"http://{address}:{kimina['port']}", api_key=api_key,
                                 lean_timeout_seconds=kimina["lean_timeout_seconds"],
                                 concurrent_requests=kimina["concurrent_requests"], pin=pin)
    if not ca_file:
        raise RuntimeError(f"Lean pin {pin.name} is checked through the pool, whose certificate only its own "
                           f"authority vouches for: set {CA_FILE_VARIABLE} to that authority's certificate file "
                           "(a stage run is handed it with the entry's --kimina-ca-certificate)")
    pool = config["lean"]["pool"]
    address, _ = resolve_lan_host(pool["name"], kimina["lan_dns_server"])
    defaults = KiminaClientSettings(base_url="")
    return LeanCheckSettings(base_url=f"https://{address}:{pool['port']}", api_key=api_key,
                             lean_timeout_seconds=pool["lean_timeout_seconds"],
                             concurrent_requests=pool["concurrent_requests"],
                             ca_file=ca_file, tls_server_name=pool["name"], pin=pin,
                             # Used only by a client that follows the pool's stated size (`follow_pool_size`).
                             pool_size_margin=float(pool.get("size_margin", defaults.pool_size_margin)),
                             pool_size_ceiling=int(pool.get("in_flight_ceiling", defaults.pool_size_ceiling)))


AUTO_IN_FLIGHT = "auto"


def following_the_pool(settings: KiminaClientSettings, in_flight: object, fallback: int | None = None) -> KiminaClientSettings:
    """`settings` with the requests in flight a caller asked for: a number, as always, or `auto`: follow the
    size the pool states (lean_pool/README.md, "Background work and the pool's size"), with `fallback` (default: the settings' own
    number) in flight until it states one and whenever it states none. None or 0 changes nothing."""
    if in_flight == AUTO_IN_FLIGHT:
        return dataclasses.replace(settings, follow_pool_size=True,
                                   concurrent_requests=int(fallback) if fallback else settings.concurrent_requests)
    if not in_flight:
        return settings
    if isinstance(in_flight, bool) or not isinstance(in_flight, int) or in_flight < 1:
        raise ValueError(f"requests in flight must be a whole number of at least 1, or '{AUTO_IN_FLIGHT}', not {in_flight!r}")
    return dataclasses.replace(settings, concurrent_requests=in_flight)


def waiting_out_the_proxy_queue(settings: KiminaClientSettings, config: dict) -> KiminaClientSettings:
    """`settings` whose HTTP timeout covers the wait in the pool's own queue. A client that keeps more requests
    in flight than the pool has workers, or whose checks wait behind others (a background client), must wait
    as long as the proxy may hold a request AND a Lean server may hold it for a free worker, or it gives up
    on (and sends again) a request that is still queued. Unchanged for a pin that is not behind a pool."""
    pool = (config.get("lean") or {}).get("pool") or {}
    if not pin_of(settings).through_pool or pool.get("proxy_queue_seconds") is None:
        return settings
    return dataclasses.replace(settings, server_queue_wait_seconds=float(pool["proxy_queue_seconds"] + pool.get("server_wait_seconds", 0)))


def kimina_settings_from_config(config: dict) -> LeanCheckSettings:
    return lean_settings(config, api_key=os.environ[API_KEY_VARIABLE], ca_file=os.environ.get(CA_FILE_VARIABLE))


def pin_of(settings: KiminaClientSettings) -> LeanPin:
    """The pin whose rules apply to checks sent with `settings`. Bare client settings (a tool that names its
    own server) are read by the default pin's rules, as before the port."""
    return settings.pin if isinstance(settings, LeanCheckSettings) else LEAN_PINS[DEFAULT_LEAN_PIN]


def check_lean_sources(settings: KiminaClientSettings, sources: dict[str, str]) -> dict[str, dict]:
    """Synchronous helper: snippet id -> the server's raw result."""
    pin = pin_of(settings)

    async def run() -> list[dict]:
        async with KiminaVerifier(settings) as verifier:
            return await verifier.check([LeanSnippet(snippet_id, pin.source(code)) for snippet_id, code in sources.items()])

    return {raw["id"]: raw for raw in asyncio.run(run())} if sources else {}


class VerificationService:
    def __init__(self, settings: KiminaClientSettings) -> None:
        self._settings = settings
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="verification")
        self._futures: list[Future] = []
        self._lexical: dict[str, VerificationResult] = {}
        self.checked = 0
        self.check_seconds = 0.0

    def submit(self, attempts: list[ProofAttemptToVerify]) -> None:
        sources = {}
        for attempt in attempts:
            token = find_forbidden_token(attempt.completion)
            if token is not None:
                self._lexical[attempt.attempt_id] = rejected_lexically(attempt.attempt_id, token)
            else:
                sources[attempt.attempt_id] = build_proof_source(attempt.statement, attempt.completion)
        self._futures.append(self._executor.submit(self._verify, sources))

    def _verify(self, sources: dict[str, str]) -> dict[str, VerificationResult]:
        started = time.monotonic()
        raw_results = check_lean_sources(self._settings, sources)
        self.check_seconds += time.monotonic() - started
        self.checked += len(raw_results)
        pin = pin_of(self._settings)
        return {attempt_id: pin.classify(attempt_id, raw) for attempt_id, raw in raw_results.items()}

    def results(self) -> dict[str, VerificationResult]:
        combined = dict(self._lexical)
        for future in self._futures:
            combined.update(future.result())
        self._executor.shutdown(wait=True)
        return combined


class LeanCheckPool:
    """ONE client for all the checks of a step, however many callers feed it and however many are waiting.

    `VerificationService` gives each caller a client of its own, so two callers at once put twice
    `concurrent_requests` in flight, and one caller's batches are checked one after the other (a batch's last slow
    proof holds every other slot idle). Here one event loop runs in a background thread with one `KiminaVerifier`:
    every batch any session submits competes for the SAME `concurrent_requests` slots, first come first served, so
    the requests in flight stay at that number for as long as anything is waiting.

    `session()` gives a caller its own `submit` and `results`, the shape of `VerificationService`. `close()` cancels
    what is still waiting and stops the loop. Safe to use from several threads.
    """

    def __init__(self, settings: KiminaClientSettings, verifier_factory: Callable[[], KiminaVerifier] | None = None) -> None:
        self.settings = settings
        self._verifier_factory = verifier_factory or (lambda: KiminaVerifier(settings, on_request_failure=self._note_failure))
        self._verifier: KiminaVerifier | None = None
        self._lock = threading.Lock()
        self._closed = False
        self.checked = 0                            # Lean files answered (or given up on), over every session
        self.request_failures = 0                   # failed HTTP requests, the ones a retry recovered included
        self.in_flight_limit = settings.concurrent_requests     # as of the latest answer; it moves with the pool's
                                                                # stated size when the settings follow it
        self.first_submitted: float | None = None   # time.monotonic() of the first submission
        self.last_answered: float | None = None     # ... and of the latest batch that came back
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="lean-checks", daemon=True)
        self._thread.start()

    def _note_failure(self, error: Exception) -> None:
        with self._lock:
            self.request_failures += 1

    async def _check(self, sources: dict[str, str]) -> list[dict]:
        if self._verifier is None:      # made on the loop's own thread: its slots and its connections belong to this loop
            self._verifier = self._verifier_factory()
        pin = pin_of(self.settings)
        raw = await self._verifier.check([LeanSnippet(snippet_id, pin.source(code)) for snippet_id, code in sources.items()])
        with self._lock:
            self.checked += len(raw)
            self.last_answered = time.monotonic()
            self.in_flight_limit = getattr(self._verifier, "requests_in_flight_limit", self.in_flight_limit)
        return raw

    def submit_sources(self, sources: dict[str, str]) -> Future:
        """Start checking `sources` (snippet id -> Lean file); the future gives the server's raw results."""
        with self._lock:
            if self._closed:
                raise RuntimeError("this Lean check pool is closed")
            if self.first_submitted is None:
                self.first_submitted = time.monotonic()
        return asyncio.run_coroutine_threadsafe(self._check(sources), self._loop)

    def session(self) -> "CheckSession":
        return CheckSession(self)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True

        async def shut_down() -> None:
            waiting = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in waiting:
                task.cancel()
            await asyncio.gather(*waiting, return_exceptions=True)
            if self._verifier is not None:
                await self._verifier.close()

        try:
            asyncio.run_coroutine_threadsafe(shut_down(), self._loop).result(timeout=30)
        except Exception:  # noqa: BLE001 - closing must not hide the failure that led here
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=30)
        if not self._loop.is_running():
            self._loop.close()


class CheckSession:
    """One caller's checks through a `LeanCheckPool`. `submit` returns at once; `results` waits for this
    session's checks only. The lexical filter runs first, as in `VerificationService`."""

    def __init__(self, pool: LeanCheckPool) -> None:
        self._pool = pool
        self._futures: list[Future] = []
        self._lexical: dict[str, VerificationResult] = {}

    def submit(self, attempts: list[ProofAttemptToVerify]) -> None:
        sources = {}
        for attempt in attempts:
            token = find_forbidden_token(attempt.completion)
            if token is not None:
                self._lexical[attempt.attempt_id] = rejected_lexically(attempt.attempt_id, token)
            else:
                sources[attempt.attempt_id] = build_proof_source(attempt.statement, attempt.completion)
        if sources:
            self._futures.append(self._pool.submit_sources(sources))

    def results(self) -> dict[str, VerificationResult]:
        pin = pin_of(self._pool.settings)
        combined = dict(self._lexical)
        for future in self._futures:
            for raw in future.result():
                combined[raw["id"]] = pin.classify(raw["id"], raw)
        return combined
