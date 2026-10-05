"""Async client for the Kimina Lean Server's `/api/check` endpoint. Spec §7, "Kimina client".

What it guarantees to its caller:
  * bounded load: at most `concurrent_requests` HTTP requests in flight, `snippets_per_request` Lean
    files each. Their product should be about the server's worker count; more only queues server-side.
  * every snippet gets exactly one result, in input order, whatever happens.
  * transport errors, HTTP 429 and HTTP 5xx are retried with exponential backoff and jitter.
  * one bad snippet cannot fail its neighbours: the server answers a whole request with one status,
    so a batch that keeps failing is split in half and each half retried, down to single snippets.
  * a Lean TIMEOUT is an answer, not a failure: it arrives as HTTP 200 and is never retried.
  * a snippet that still has no answer becomes a `server_error`-shaped result, recorded distinctly.

Through a lean-pool (https://github.com/cabloo/lean-pool; its README, "Background work and the pool's size") two more things, both optional
and both harmless against a server that knows neither:
  * `priority="background"`: the pool serves these checks only when no normal check is waiting. A 503 from a
    pool that is up then means "wait": the client pauses and asks again, without counting an attempt.
  * `follow_pool_size`: the requests in flight follow the worker count the pool states (`/health` at the
    start, a response header after that), times a margin. A pool that states nothing, or states nonsense,
    leaves the limit at `concurrent_requests`.
"""

from __future__ import annotations

import asyncio
import math
import random
import ssl
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence

import httpx

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# The pool's own names (lean-pool's `leanpool/signals.py`). This package imports nothing from that
# project, so they are stated again here; tests/rlvr_lean/test_kimina_pool_signals.py pins them.
PRIORITY_HEADER = "X-Lean-Priority"
BACKGROUND_PRIORITY = "background"
POOL_WORKERS_HEADER = "X-Lean-Pool-Workers"
POOL_WORKERS_FIELD = "workers"            # the same number in the body of GET /health
_LARGEST_POOL = 1_000_000


def pool_workers(value: object) -> int | None:
    """The worker count a pool stated (a response header's text, or a number from `/health`'s body): a whole
    number of at least 1. Anything else (nothing, an empty text, zero, a fraction, words) is None: the caller
    then keeps the number it was configured with."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1 <= value <= _LARGEST_POOL else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text.isascii() or not text.isdigit():
        return None
    return pool_workers(int(text))


class RequestSlots:
    """At most `limit` holders at a time, first come first served, and the limit may change while some hold a
    slot and others wait. Raising it lets waiters in at once; lowering it lets nobody in until enough holders
    have left. A holder is never thrown out. For one event loop (like `asyncio.Semaphore`, whose rules for a
    cancelled waiter this keeps: a slot that was handed to a waiter which is then cancelled goes back)."""

    def __init__(self, limit: int) -> None:
        self._limit = max(1, int(limit))
        self._in_use = 0
        self._waiters: deque[asyncio.Future] = deque()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_use(self) -> int:
        return self._in_use

    def set_limit(self, limit: int) -> None:
        self._limit = max(1, int(limit))
        self._hand_over()

    def _hand_over(self) -> None:
        while self._waiters and self._in_use < self._limit:
            waiter = self._waiters.popleft()
            if not waiter.done():
                self._in_use += 1               # counted here, so nobody takes the slot before the waiter runs
                waiter.set_result(None)

    async def acquire(self) -> None:
        if self._in_use < self._limit and not self._waiters:
            self._in_use += 1
            return
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        try:
            await waiter
        except asyncio.CancelledError:
            if waiter.done() and not waiter.cancelled():    # the slot had been handed over: give it back
                self._in_use -= 1
                self._hand_over()
            else:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass
            raise

    def release(self) -> None:
        self._in_use -= 1
        self._hand_over()

    async def __aenter__(self) -> "RequestSlots":
        await self.acquire()
        return self

    async def __aexit__(self, *exception_info: object) -> None:
        self.release()


@dataclass(frozen=True)
class LeanSnippet:
    """One Lean file to check. `snippet_id` must be unique within a call to `check`."""

    snippet_id: str
    code: str


@dataclass(frozen=True)
class KiminaClientSettings:
    base_url: str
    api_key: str | None = None
    lean_timeout_seconds: int = 60          # the server kills a check after this long (header and body separately)
    # One snippet per request by default. The server answers a request only when its slowest snippet is
    # done, so with several per request one slow proof idles the workers its neighbours finished on
    # (measured on the first pin-gate run). Keep `concurrent_requests` equal to the server's worker count.
    snippets_per_request: int = 1
    concurrent_requests: int = 12
    max_attempts_per_batch: int = 4
    max_attempts_after_crash: int = 2       # for a single snippet whose worker died (HTTP 500)
    backoff_initial_seconds: float = 1.0
    backoff_maximum_seconds: float = 30.0
    server_queue_wait_seconds: float = 120.0  # the server's own wait for a free worker (LEAN_SERVER_MAX_WAIT)
    http_margin_seconds: float = 60.0
    request_debug_diagnostics: bool = True   # ask for per-snippet cpu/memory figures
    # For an https `base_url` whose certificate was signed by a private authority (the lean-pool front door,
    # lean-pool's README, "TLS"): that authority's certificate file, and then ONLY it
    # is trusted. None trusts the system's public authorities, which never vouch for the pool.
    ca_file: str | None = None
    # For an https `base_url` that holds an ADDRESS (a GPU task reaches the pool by a resolved LAN address,
    # and the pool's certificate names hosts, not addresses; the OEIS Open spec, O2a item 9b):
    # the NAME the server's certificate must carry. The client still connects to the address in `base_url`;
    # the name is what the certificate is checked against, in place of the URL's host, and what is sent for
    # SNI. None checks the certificate against the URL's host, as any https client does.
    tls_server_name: str | None = None
    # --- through a lean-pool only (lean-pool's README, "Background work and the pool's size"); a single server ignores all of it.
    # "background": sent as `X-Lean-Priority`, and the pool takes these checks only when no normal check is
    # waiting. None sends no such header (a normal check). Any other text is sent as it is and means normal.
    priority: str | None = None
    # A background check the pool's queue timed out (HTTP 503 from a pool whose /health still answers) is asked
    # again after this pause, without counting as an attempt, at most `busy_pauses_allowed` times per batch.
    busy_pause_seconds: float = 30.0
    busy_pauses_allowed: int = 2000
    # Follow the worker count the pool states: requests in flight = ceil(workers x margin), never above the
    # ceiling, re-read at most every `pool_size_refresh_seconds`. `concurrent_requests` is the number until the
    # pool states one, and for as long as it states none.
    follow_pool_size: bool = False
    pool_size_margin: float = 1.25
    pool_size_ceiling: int = 256
    pool_size_refresh_seconds: float = 30.0

    @property
    def http_timeout_seconds(self) -> float:
        """Long enough for the slowest legitimate answer: a queue wait, a header import and a body,
        each allowed the full Lean timeout, plus a margin. Shorter would turn slow answers into retries."""
        return self.server_queue_wait_seconds + 2 * self.lean_timeout_seconds + self.http_margin_seconds


def _trust(ca_file: str | None) -> ssl.SSLContext | bool:
    """What an https server must prove: a certificate from `ca_file`'s authority and from no other, or
    (None) from the system's. A missing or unreadable file raises here, before any request is sent."""
    if ca_file is None:
        return True
    # With a cafile, Python loads that file and not the system's authorities; the certificate and the host
    # name are both checked.
    return ssl.create_default_context(cafile=ca_file)


class KiminaRequestError(Exception):
    """A request that failed in a way worth retrying (transport error, 429, 5xx).

    `worker_crashed` marks HTTP 500: the server's answer when a Lean worker died on one of the snippets
    (typically its memory limit). That is usually deterministic for the snippet, so retrying the same
    batch repeats the crash; the batch is split at once instead, so only the culprit pays again."""

    def __init__(self, message: str, worker_crashed: bool = False, status_code: int | None = None) -> None:
        super().__init__(message)
        self.worker_crashed = worker_crashed
        self.status_code = status_code      # the HTTP status of a refused request; None for a transport error


class KiminaVerifier:
    """Sends Lean snippets to Kimina and returns the server's raw per-snippet results."""

    def __init__(
        self,
        settings: KiminaClientSettings,
        http_client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        random_source: random.Random | None = None,
        on_request_failure: Callable[[KiminaRequestError], None] | None = None,
        on_notice: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        # One line each time the requests in flight change with the pool's size, and ONE line if the pool
        # states no size. Default: standard error, so a step's log shows what its client did.
        self._notice = on_notice or (lambda text: print(f"[lean client] {text}", file=sys.stderr, flush=True))
        self._clock = clock
        self._asked_health = False
        self._sized_at: float | None = None
        self._said_no_size = False
        self.busy_pauses = 0                # 503s from a pool that was up, waited out (a background client only)
        # Told of EVERY failed request, including the ones a retry then recovers: a caller that measures the
        # server (a pool under its first sustained load) counts them; the results never show a recovered one.
        self._on_request_failure = on_request_failure
        if settings.tls_server_name and not settings.base_url.startswith("https://"):
            raise ValueError("tls_server_name names the certificate to expect, so base_url must be https")
        # httpx hands a request's `sni_hostname` to the TLS handshake as THE server's name: Python's ssl sends
        # it for SNI and checks the certificate against it (`check_hostname`), instead of the URL's host.
        # Connections are pooled by URL, and every request of this client carries the same name.
        self._request_extensions = {"sni_hostname": settings.tls_server_name} if settings.tls_server_name else {}
        headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
        self._http_client = http_client or httpx.AsyncClient(
            base_url=settings.base_url, headers=headers, timeout=settings.http_timeout_seconds,
            verify=_trust(settings.ca_file),
        )
        self._owns_http_client = http_client is None
        # On each check, not on the http client: a caller may hand in its own client, and the priority is this
        # verifier's, whoever made the connection.
        self._check_headers = {PRIORITY_HEADER: settings.priority} if settings.priority else {}
        self._sleep = sleep
        self._random = random_source or random.Random()
        self._request_slots = RequestSlots(settings.concurrent_requests)

    @property
    def requests_in_flight_limit(self) -> int:
        """How many requests this client keeps in flight right now."""
        return self._request_slots.limit

    def _is_background(self) -> bool:
        return (self._settings.priority or "").strip().lower() == BACKGROUND_PRIORITY

    def _resize(self, workers: int) -> None:
        settings = self._settings
        limit = min(max(1, settings.pool_size_ceiling), max(1, math.ceil(workers * settings.pool_size_margin)))
        self._sized_at = self._clock()
        if limit != self._request_slots.limit:
            self._notice(f"the pool states {workers} workers: {limit} requests in flight (was {self._request_slots.limit})")
            self._request_slots.set_limit(limit)

    def _no_size(self, what: str) -> None:
        if not self._said_no_size:
            self._said_no_size = True
            self._notice(f"{what}: keeping {self._request_slots.limit} requests in flight (said once)")

    def _follow_pool_size(self, response: Any) -> None:
        """Take the pool's size from an answer's headers, at most once per refresh period. Never raises: a
        pool that states nothing, or nonsense, leaves the limit where it is."""
        if not self._settings.follow_pool_size:
            return
        try:
            workers = pool_workers(response.headers.get(POOL_WORKERS_HEADER))
        except Exception:  # noqa: BLE001 - an advisory number must never fail a check
            workers = None
        if workers is None:
            self._no_size(f"the pool's answers carry no usable {POOL_WORKERS_HEADER}")
            return
        if self._sized_at is None or self._clock() - self._sized_at >= self._settings.pool_size_refresh_seconds:
            self._resize(workers)

    async def size_from_pool(self) -> int | None:
        """Ask `/health` for the pool's size and take it. Returns the workers it stated, or None (and the limit
        stays) when it cannot be asked, answers something else, or states no usable number."""
        try:
            response = await self._http_client.get("/health", timeout=10.0, extensions=self._request_extensions)
            workers = pool_workers(response.json().get(POOL_WORKERS_FIELD)) if response.status_code == 200 else None
        except Exception:  # noqa: BLE001 - advisory: any failure to read it is "the pool states no size"
            workers = None
        if workers is None:
            self._no_size("the pool's /health states no size")
            return None
        self._resize(workers)
        return workers

    async def close(self) -> None:
        if self._owns_http_client:
            await self._http_client.aclose()

    async def __aenter__(self) -> "KiminaVerifier":
        return self

    async def __aexit__(self, *exception_info: object) -> None:
        await self.close()

    async def is_healthy(self) -> bool:
        """True when the web server answers. It does not prove a Lean worker can start."""
        try:
            response = await self._http_client.get("/health", timeout=10.0, extensions=self._request_extensions)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def check(self, snippets: Sequence[LeanSnippet]) -> list[dict[str, Any]]:
        """Check every snippet; return one raw result per snippet, in input order."""
        snippet_ids = [snippet.snippet_id for snippet in snippets]
        if len(set(snippet_ids)) != len(snippet_ids):
            raise ValueError("snippet ids must be unique within one call to check()")
        if self._settings.follow_pool_size and not self._asked_health:
            self._asked_health = True       # once: before the first check, so the first burst is already sized
            await self.size_from_pool()
        batch_size = self._settings.snippets_per_request
        batches = [list(snippets[start:start + batch_size]) for start in range(0, len(snippets), batch_size)]
        batch_results = await asyncio.gather(*(self._check_batch(batch) for batch in batches))
        results_by_id = {result["id"]: result for results in batch_results for result in results}
        return [results_by_id[snippet_id] for snippet_id in snippet_ids]

    async def _check_batch(self, batch: list[LeanSnippet]) -> list[dict[str, Any]]:
        """One batch, with retries; a batch that keeps failing is split so the failure is isolated.

        A crashed worker (HTTP 500) splits a multi-snippet batch immediately, and a single snippet
        gets `max_attempts_after_crash` tries in total: a crash is rarely the server's weather."""
        failure: Exception | None = None
        attempts_allowed = self._settings.max_attempts_per_batch
        attempt_number = 0
        busy_pauses = 0
        while attempt_number < attempts_allowed:
            attempt_number += 1
            try:
                async with self._request_slots:
                    return await self._post_batch(batch)
            except KiminaRequestError as error:
                failure = error
                if self._on_request_failure is not None:
                    self._on_request_failure(error)
                # A background check that waited out the pool's queue: the pool is up and busy with work that
                # goes first. Not an attempt and not an answer: pause (outside the slot), then ask again.
                if (error.status_code == 503 and self._is_background()
                        and busy_pauses < self._settings.busy_pauses_allowed and await self.is_healthy()):
                    busy_pauses += 1
                    self.busy_pauses += 1
                    attempt_number -= 1
                    await self._sleep(self._settings.busy_pause_seconds)
                    continue
                if error.worker_crashed:
                    if len(batch) > 1:
                        break
                    attempts_allowed = min(attempts_allowed, self._settings.max_attempts_after_crash)
                if attempt_number < attempts_allowed:
                    await self._sleep(self._backoff_seconds(attempt_number))
        if len(batch) > 1:
            middle = len(batch) // 2
            first_half, second_half = await asyncio.gather(
                self._check_batch(batch[:middle]), self._check_batch(batch[middle:])
            )
            return first_half + second_half
        return [_server_error_result(batch[0], f"no answer after {attempt_number} attempts: {failure}")]

    def _backoff_seconds(self, attempt_number: int) -> float:
        """Exponential backoff with full jitter: a uniformly random wait up to the capped exponential."""
        ceiling = min(
            self._settings.backoff_maximum_seconds,
            self._settings.backoff_initial_seconds * (2 ** (attempt_number - 1)),
        )
        return self._random.uniform(0.0, ceiling)

    async def _post_batch(self, batch: list[LeanSnippet]) -> list[dict[str, Any]]:
        body = {
            "snippets": [{"id": snippet.snippet_id, "code": snippet.code} for snippet in batch],
            "timeout": self._settings.lean_timeout_seconds,
            "debug": self._settings.request_debug_diagnostics,
            "reuse": True,
        }
        priority = {"headers": self._check_headers} if self._check_headers else {}
        try:
            response = await self._http_client.post("/api/check", json=body, extensions=self._request_extensions, **priority)
        except httpx.TransportError as error:
            raise KiminaRequestError(f"transport error: {error!r}") from error
        self._follow_pool_size(response)
        if response.status_code in RETRYABLE_STATUS_CODES:
            raise KiminaRequestError(f"HTTP {response.status_code}: {response.text[:200]}",
                                     worker_crashed=response.status_code == 500, status_code=response.status_code)
        # 401 (bad key), 422 (malformed request) and the like are our bug, not the server's weather.
        response.raise_for_status()
        results = response.json().get("results")
        expected_ids = {snippet.snippet_id for snippet in batch}
        if not isinstance(results, list) or {result.get("id") for result in results} != expected_ids:
            raise KiminaRequestError("the reply does not contain exactly the requested snippet ids")
        return results


def _server_error_result(snippet: LeanSnippet, reason: str) -> dict[str, Any]:
    """A result in the server's own shape for a snippet the server never answered."""
    return {"id": snippet.snippet_id, "time": None, "error": f"server_error: {reason}"}
