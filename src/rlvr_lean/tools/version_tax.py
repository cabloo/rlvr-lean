"""Version tax: re-check a finished run's stored proof attempts under the NEXT Lean pin and report what the
move costs. Spec: the OEIS Open spec, O2a items 9d and 9e. No GPU: only the Lean pool is used.

What it does, in this order:
  statements  every seed, reward and conjecture statement of the run is compiled under the later pin, with
              the pipeline's own compile check (item 9d, and the "compiles at both pins" filter of 9e);
  attempts    every stored attempt is checked again, with its own statement, the pipeline's header and the
              pipeline's lexical filter, and read by the later pin's rules;
  report      `version_tax.json` and `version_tax.md`: both pins' pass rates on IDENTICAL texts with intervals,
              the later pin's failure classes, the same by base pass-rate band, and the pool's behaviour.

Rules it keeps:
  * resumable: each answer is appended to `checks.jsonl` as it arrives, and a rerun skips what is answered.
    Identical Lean files (a third of the attempts repeat a proof) are checked once.
  * bounded load: the pin's `concurrent_requests` in flight, one file per request.
  * a pool or transport failure is NO ANSWER, never a Lean failure: it is retried at the end of the run and
    on every rerun, and whatever still has no answer is left out of BOTH pins' counts and reported.
  * every failed request is logged with its time (`request_failures.jsonl`), including the ones a retry
    recovered, so a pool that flaps under load shows.

    PYTHONPATH=src python -m rlvr_lean.tools.version_tax --run-steps <run>/steps --out <dir> \\
        --api-key-file <the pool's key file> --ca-file <the pool authority's certificate> [--limit 500]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import signal
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from rlvr_lean.domain.conjecturing import build_compile_check_source
from rlvr_lean.domain.evaluation.version_tax import (
    NO_ANSWER_STATUS,
    OTHER_ERROR,
    TIMEOUT,
    UNKNOWN_IDENTIFIER,
    Outcome,
    build_version_tax,
    error_kind,
    failure_class,
    is_echo_of_a_failed_declaration,
    names_something_unknown,
    wilson_interval,
)
from rlvr_lean.domain.verification import (
    LEAN_PINS,
    build_proof_source,
    find_forbidden_token,
    rejected_lexically,
    theorem_name_of,
)
from rlvr_lean.gpu.pipeline import _compiles, _lean_messages        # the pipeline's own rule for "this statement compiles"
from rlvr_lean.infrastructure.kimina_client import KiminaRequestError, KiminaVerifier, LeanSnippet
from rlvr_lean.infrastructure.verification_service import LeanCheckSettings, lean_settings

EARLIER_PIN, LATER_PIN = "v4.9", "v4.27"
CONFIG = Path(__file__).resolve().parents[1] / "config" / "experiment.yaml"
STATEMENT_GROUPS = {"seed": ("statements_seed.jsonl", "statement_id"), "reward": ("statements_reward.jsonl", "statement_id"),
                    "conjecture": ("conjectures.jsonl", "conjecture_id")}
ORDER_SEED = 0                 # attempts are checked in one seeded order, so a run that stops early is a random sample
EXAMPLES_PER_CLASS = 3
MESSAGES_KEPT = 8              # error messages stored per check (each cut at 600 characters)
KINDS_LISTED = 15              # error kinds listed in the finer breakdown
MAX_CRASH_RUNS = 2             # runs in which a check may crash its worker before it is left as "no answer"
UNHEALTHY_WAIT_SECONDS = 1800  # how long a run waits for the pool's /health before it gives up
STANDBY_SECONDS = 1.0          # how often a worker beyond the client's present requests in flight looks again


LOG_LABEL = "version_tax"      # a tool that borrows `run_checks` puts its own name on the lines it prints


def log(message: str) -> None:
    print(f"[{LOG_LABEL} {time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:        # a line cut short by a kill: its check is simply done again
            continue
    return rows


@dataclass(frozen=True)
class Job:
    sha: str                   # of the Lean file sent: identical files are one check
    kind: str                  # "statement" or "attempt"
    source: str


def sha_of(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


class StoredRun:
    """A finished run's files, read only: its statements, its attempts and the earlier pin's verdicts."""

    def __init__(self, steps: Path) -> None:
        self.statements: dict[str, dict[str, str]] = {}                    # group -> statement id -> statement
        for group, (name, key) in STATEMENT_GROUPS.items():
            self.statements[group] = {row[key]: row["statement"] for row in read_rows(steps / name)}
        self.statement_of = {**self.statements["reward"], **self.statements["conjecture"]}
        self.group_of = {**{key: "reward" for key in self.statements["reward"]},
                         **{key: "conjecture" for key in self.statements["conjecture"]}}
        attempts = sorted(read_rows(steps / "proof_attempts.jsonl"), key=lambda row: row["attempt_id"])
        random.Random(ORDER_SEED).shuffle(attempts)
        self.attempts = attempts
        self.earlier = {row["attempt_id"]: row for row in read_rows(steps / "verification.jsonl")}
        unknown = [row["attempt_id"] for row in attempts if row["statement_id"] not in self.statement_of]
        if unknown or set(self.earlier) != {row["attempt_id"] for row in attempts}:
            raise ValueError(f"{steps} is not one run's files: {len(unknown)} attempts have no statement, and the "
                             f"verdicts cover {len(self.earlier)} attempts of {len(attempts)}")

    def attempt_source(self, attempt: dict, pin) -> str | None:
        """The Lean file for one attempt as the pipeline sends it, or None if the lexical filter stops it."""
        if find_forbidden_token(attempt["completion"]) is not None:
            return None
        return pin.source(build_proof_source(self.statement_of[attempt["statement_id"]], attempt["completion"]))


def plan(run: StoredRun, pin, limit: int | None) -> tuple[list[dict], dict[str, dict[str, str]], list[Job]]:
    """(the attempts in scope, statement id -> sha per group, the unique checks in the order they are run)."""
    attempts = run.attempts if limit is None else run.attempts[:limit]
    in_scope = {attempt["statement_id"] for attempt in attempts}
    jobs: dict[str, Job] = {}
    statement_sha: dict[str, dict[str, str]] = {}
    for group, statements in run.statements.items():
        statement_sha[group] = {}
        for statement_id, statement in statements.items():
            if limit is not None and statement_id not in in_scope:
                continue                                    # a sample only needs its own statements
            source = pin.source(build_compile_check_source(statement))
            statement_sha[group][statement_id] = sha_of(source)
            jobs.setdefault(sha_of(source), Job(sha_of(source), "statement", source))
    for attempt in attempts:
        source = run.attempt_source(attempt, pin)
        if source is not None:
            jobs.setdefault(sha_of(source), Job(sha_of(source), "attempt", source))
    return attempts, statement_sha, list(jobs.values())


def answer_row(job: Job, raw: dict, pin) -> dict:
    """One answer as stored: the later pin's verdict and what the report needs to explain it."""
    result = pin.classify(job.sha, raw)
    errors = _lean_messages(raw, "error")
    return {"sha": job.sha, "kind": job.kind, "status": result.status.value, "seconds": result.verification_seconds,
            "compiles": _compiles(raw) if job.kind == "statement" else None,
            "errors": [message[:600] for message in errors[:MESSAGES_KEPT]], "error_count": len(errors),
            "failed_warning": any("failed" in warning for warning in _lean_messages(raw, "warning")),
            "detail": result.detail[:300], "cached": bool(raw.get("cached")), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def crashed_its_worker(row: dict) -> bool:
    """A no-answer row whose last refusal was HTTP 500: the server's answer when the Lean worker died on this
    very file (in practice its memory limit: `norm_num` on `2 ^ (Nat.fib 2012 - 1)`). Unlike a pool that was
    briefly down, asking again kills another worker, so such a check is asked again in a LATER run only, and
    after `MAX_CRASH_RUNS` runs it is left as no answer."""
    return row.get("status") == NO_ANSWER_STATUS and "HTTP 500" in row.get("detail", "")


def answered(checks_file: Path) -> dict[str, dict]:
    """sha -> its stored ANSWER. A no-answer row stays in the file as a record and its check is tried again."""
    rows = {}
    if checks_file.exists():
        for row in read_rows(checks_file):
            if row.get("status") != NO_ANSWER_STATUS:
                rows[row["sha"]] = row
    return rows


def given_up(checks_file: Path) -> set[str]:
    """The checks that crashed their worker in `MAX_CRASH_RUNS` runs and have no answer: not asked again."""
    crashes = Counter(row["sha"] for row in read_rows(checks_file) if crashed_its_worker(row)) if checks_file.exists() else Counter()
    return {sha for sha, count in crashes.items() if count >= MAX_CRASH_RUNS} - set(answered(checks_file))


async def run_checks(settings: LeanCheckSettings, jobs: list[Job], out: Path, retry_passes: int, answer_of=None, on_answer=None) -> dict:
    """Check `jobs`, `settings.concurrent_requests` at a time, appending each answer as it arrives.

    `answer_of(job, raw, pin)` is the row stored for one answer (default: this tool's `answer_row`), and
    `on_answer(row)` is told of each one: another tool that keeps this file's rules (resumable, bounded,
    a transport failure is no answer) reads more out of an answer than the version tax does."""
    pin = settings.pin
    answer_of = answer_of or answer_row
    stop = asyncio.Event()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(signal_number, stop.set)
    counts = Counter()
    started = time.monotonic()

    with (out / "checks.jsonl").open("a") as checks, (out / "request_failures.jsonl").open("a") as failures:
        def on_failure(error: KiminaRequestError) -> None:
            kind = "transport" if error.status_code is None else f"http_{error.status_code}"
            counts[f"failed_request:{kind}"] += 1
            failures.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind,
                                       "message": str(error)[:300]}) + "\n")
            failures.flush()

        async with KiminaVerifier(settings, on_request_failure=on_failure) as verifier:
            healthy = asyncio.Lock()

            async def wait_until_healthy() -> None:
                """After a check got no answer: hold every worker until the pool's /health answers again."""
                async with healthy:
                    deadline = time.monotonic() + UNHEALTHY_WAIT_SECONDS
                    while not await verifier.is_healthy():
                        if stop.is_set() or time.monotonic() > deadline:
                            stop.set()
                            return
                        counts["health_waits"] += 1
                        await asyncio.sleep(15)

            def in_flight_limit() -> int:
                """The client's requests in flight right now (it moves when the client follows the pool's size)."""
                return getattr(verifier, "requests_in_flight_limit", settings.concurrent_requests)

            def busy_pauses() -> int:
                return getattr(verifier, "busy_pauses", 0)

            async def worker(index: int, queue: asyncio.Queue, unanswered: list[Job]) -> None:
                while not stop.is_set():
                    # A client that follows the pool's stated size has more of these than it has requests in
                    # flight at any one time: one beyond the limit takes no job, so a stop is still answered
                    # after the checks in flight and not after every one of them has had a job.
                    if index >= in_flight_limit():
                        if queue.empty():
                            return
                        await asyncio.sleep(STANDBY_SECONDS)
                        continue
                    try:
                        job = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    raw = (await verifier.check([LeanSnippet(job.sha, job.source)]))[0]
                    row = answer_of(job, raw, pin)
                    checks.write(json.dumps(row, ensure_ascii=False) + "\n")
                    checks.flush()
                    if on_answer is not None:
                        on_answer(row)
                    counts["answered" if row["status"] != NO_ANSWER_STATUS else "no_answer"] += 1
                    counts["cached"] += row["cached"]
                    if row["status"] == NO_ANSWER_STATUS:
                        if not crashed_its_worker(row):       # a retry in this run is for a pool that was down
                            unanswered.append(job)
                        await wait_until_healthy()

            async def progress(total: int) -> None:
                while True:
                    await asyncio.sleep(60)
                    elapsed = time.monotonic() - started
                    failed = {key.split(":", 1)[1]: value for key, value in counts.items() if key.startswith("failed_request:")}
                    waited = f", waited out a busy pool {busy_pauses()} times" if busy_pauses() else ""
                    log(f"{counts['answered']}/{total} answered ({counts['answered'] / elapsed:.2f}/s, {counts['cached']} from the "
                        f"cache), no answer {counts['no_answer']}, failed requests {failed or 0}, "
                        f"{in_flight_limit()} in flight{waited}")

            reporter = asyncio.create_task(progress(len(jobs)))
            # As many workers as there can ever be requests in flight: the configured number, or the ceiling of
            # a client that follows the pool's stated size (the workers beyond the present limit stand by).
            workers = settings.pool_size_ceiling if settings.follow_pool_size else settings.concurrent_requests
            try:
                pending = jobs
                for attempt_number in range(1 + retry_passes):
                    if not pending or stop.is_set():
                        break
                    if attempt_number:
                        log(f"retrying {len(pending)} checks that got no answer (pass {attempt_number} of {retry_passes})")
                    queue: asyncio.Queue = asyncio.Queue()
                    for job in pending:
                        queue.put_nowait(job)
                    unanswered: list[Job] = []
                    await asyncio.gather(*(worker(index, queue, unanswered) for index in range(workers)))
                    pending = unanswered + [queue.get_nowait() for _ in range(queue.qsize())]
            finally:
                reporter.cancel()
                in_flight_at_the_end, busy_pauses_in_all = in_flight_limit(), busy_pauses()
    seconds = time.monotonic() - started
    session = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - seconds)), "seconds": round(seconds, 1),
               "checks_wanted": len(jobs), "answered": counts["answered"], "from_cache": counts["cached"],
               "no_answer_rows": counts["no_answer"], "stopped_early": stop.is_set(), "health_waits": counts["health_waits"],
               "failed_requests": {key.split(":", 1)[1]: value for key, value in counts.items() if key.startswith("failed_request:")},
               "answers_per_second": round(counts["answered"] / seconds, 3) if seconds else None,
               "in_flight": in_flight_at_the_end, "in_flight_follows_the_pool": settings.follow_pool_size,
               "priority": settings.priority, "busy_pauses": busy_pauses_in_all,
               "lean_timeout_seconds": settings.lean_timeout_seconds}
    with (out / "sessions.jsonl").open("a") as sessions:
        sessions.write(json.dumps(session) + "\n")
    return session


# ------------------------------------------------------------------------------------------------ the report
def own_errors(row: dict, statement: str) -> list[str]:
    """A check's stored error messages without the echo of its own failed declaration (`#print axioms` not
    finding the theorem), which would otherwise read as a missing library name."""
    name = theorem_name_of(statement)
    return [message for message in row["errors"] if not is_echo_of_a_failed_declaration(message, name)]


def deciding_error(errors: list[str], detail: str = "") -> str:
    """The message that classes a failure: the first that names something unknown, else the first."""
    return next((message for message in errors if names_something_unknown(message)), errors[0] if errors else detail)


def later_outcome(run: StoredRun, attempt: dict, pin, rows: dict[str, dict]) -> tuple[Outcome | None, dict | None]:
    """The later pin's outcome of one stored attempt (None while its check has no row at all), and that row."""
    source = run.attempt_source(attempt, pin)
    if source is None:
        token = find_forbidden_token(attempt["completion"])
        return Outcome(rejected_lexically(attempt["attempt_id"], token).status.value), None
    row = rows.get(sha_of(source))
    if row is None:
        return None, None
    return Outcome(row["status"], tuple(own_errors(row, run.statement_of[attempt["statement_id"]])), row["seconds"]), row


def statement_report(run: StoredRun, statement_sha: dict[str, dict[str, str]], rows: dict[str, dict]) -> tuple[dict, set[str]]:
    """Item 9d: how many of the earlier pin's statements compile under the later pin, per group. Every one of
    them compiled under the earlier pin (the earlier run kept nothing else), so a survivor compiles at both."""
    report, compiling = {}, set()
    for group, shas in statement_sha.items():
        verdicts = {statement_id: rows.get(sha) for statement_id, sha in shas.items()}
        answered_rows = {statement_id: row for statement_id, row in verdicts.items() if row is not None}
        survivors = sorted(statement_id for statement_id, row in answered_rows.items() if row["compiles"])
        failed = {statement_id: (row, own_errors(row, run.statements[group][statement_id]))
                  for statement_id, row in answered_rows.items() if not row["compiles"]}
        # A statement that does not compile is classed like a failed attempt; its status is never "verified"
        # (the check ends in `sorry`), so only the timeout and the messages decide.
        classes = Counter(failure_class(TIMEOUT if row["status"] == TIMEOUT else "lean_error", errors) for row, errors in failed.values())
        kinds = Counter(error_kind(deciding_error(errors, row["detail"])) for row, errors in failed.values())
        entry = {"earlier_pin_statements": len(run.statements[group]), "checked": len(shas), "answered": len(answered_rows),
                 "no_answer": len(shas) - len(answered_rows), "compile": len(survivors), "do_not_compile": len(failed),
                 "do_not_compile_by_class": {name: classes.get(name, 0) for name in (UNKNOWN_IDENTIFIER, OTHER_ERROR, TIMEOUT)},
                 "do_not_compile_by_error_kind": dict(kinds.most_common(KINDS_LISTED)),
                 "examples": [{"statement_id": statement_id, "error": deciding_error(errors, row["detail"])[:300],
                               "statement": run.statements[group][statement_id][:300]}
                              for statement_id, (row, errors) in sorted(failed.items())[:EXAMPLES_PER_CLASS * 2]]}
        if answered_rows:
            low, high = wilson_interval(len(survivors), len(answered_rows))
            entry.update({"survival": len(survivors) / len(answered_rows), "low": low, "high": high})
        report[group] = entry
        if group != "seed":
            compiling.update(survivors)
    return report, compiling


def examples(run: StoredRun, attempts: list[dict], outcomes: dict[str, tuple], compiling: set[str]) -> dict:
    """Up to three attempts per class that the earlier pin verified and the later one does not."""
    found: dict[str, list[dict]] = {UNKNOWN_IDENTIFIER: [], OTHER_ERROR: [], TIMEOUT: []}
    for attempt in sorted(attempts, key=lambda row: row["attempt_id"]):
        if run.earlier[attempt["attempt_id"]]["status"] != "verified" or attempt["statement_id"] not in compiling:
            continue
        outcome, row = outcomes[attempt["attempt_id"]]
        name = failure_class(outcome.status, outcome.error_messages) if outcome is not None else None
        if row is None or name not in found or len(found[name]) >= EXAMPLES_PER_CLASS:
            continue
        if any(example["statement_id"] == attempt["statement_id"] for example in found[name]):
            continue                                        # three statements, not three samples of one
        found[name].append({"attempt_id": attempt["attempt_id"], "statement_id": attempt["statement_id"],
                            "statement": run.statement_of[attempt["statement_id"]].strip()[:400],
                            "completion": attempt["completion"][:400],
                            "later_error": deciding_error(list(outcome.error_messages), row["detail"])[:400],
                            "later_seconds": row["seconds"]})
    return found


def error_kinds(attempts: list[dict], outcomes: dict[str, tuple], keep) -> dict[str, int]:
    """The most frequent kinds of deciding error among the later pin's Lean errors on attempts `keep` accepts."""
    kinds: Counter = Counter()
    for attempt in attempts:
        outcome, row = outcomes[attempt["attempt_id"]]
        if row is not None and outcome.status == "lean_error" and keep(attempt):
            kinds[error_kind(deciding_error(list(outcome.error_messages), row["detail"]))] += 1
    return dict(kinds.most_common(KINDS_LISTED))


def build_report(run: StoredRun, settings_summary: dict, pin, limit: int | None, out: Path) -> dict:
    attempts, statement_sha, jobs = plan(run, pin, limit)
    rows = answered(out / "checks.jsonl")
    statements, compiling = statement_report(run, statement_sha, rows)
    outcomes = {attempt["attempt_id"]: later_outcome(run, attempt, pin, rows) for attempt in attempts}
    earlier, later, considered, waiting = {}, {}, [], 0
    for attempt in attempts:
        outcome = outcomes[attempt["attempt_id"]][0]
        if outcome is None:
            # No row at all: not run yet, or its last row was "no answer". Recorded as no answer under the later pin.
            outcome, waiting = Outcome(NO_ANSWER_STATUS), waiting + 1
        stored = run.earlier[attempt["attempt_id"]]
        earlier[attempt["attempt_id"]] = Outcome(stored["status"], (stored["first_error"],) if stored["status"] == "lean_error" else (),
                                                 stored["seconds"])
        later[attempt["attempt_id"]] = outcome
        considered.append((attempt["attempt_id"], attempt["statement_id"]))
    # A statement's base pass rate is the earlier run's own: verified over samples drawn, all stored attempts.
    drawn, verified = Counter(), Counter()
    for attempt in run.attempts:
        drawn[attempt["statement_id"]] += 1
        verified[attempt["statement_id"]] += run.earlier[attempt["attempt_id"]]["status"] == "verified"
    tax = build_version_tax(considered, earlier, later, compiling, group_of=run.group_of,
                            base_pass_rate={statement: verified[statement] / drawn[statement] for statement in drawn})
    verified_later = [row for attempt in attempts if attempt["statement_id"] in compiling
                      for outcome, row in [outcomes[attempt["attempt_id"]]] if outcome is not None and outcome.verified and row is not None]
    # Statements the later pin ANSWERED and that do not compile under it: no proof of one can verify (a plumbing check).
    broken = {statement_id for group in ("reward", "conjecture") for statement_id, sha in statement_sha[group].items()
              if sha in rows and not rows[sha]["compiles"]}
    transitions = Counter(f"{earlier[attempt_id].status} -> {later[attempt_id].status}" for attempt_id, statement in considered
                          if statement in compiling)
    earlier_limit = settings_summary["earlier_lean_timeout_seconds"]
    abandoned = given_up(out / "checks.jsonl")
    unanswered = [attempt for attempt in attempts if outcomes[attempt["attempt_id"]][0] is None]
    report = {
        "spec": "the OEIS Open spec, O2a items 9d and 9e",
        "pins": {"earlier": EARLIER_PIN, "later": LATER_PIN}, "settings": settings_summary,
        # Complete: every check has an answer, or crashed its worker in every run allowed and is left as no answer.
        "complete": limit is None and all(job.sha in rows or job.sha in abandoned for job in jobs),
        "scope": {"attempts_stored": len(run.attempts), "attempts_in_scope": len(attempts), "limit": limit,
                  "unique_checks": len(jobs), "unique_checks_answered": sum(job.sha in rows for job in jobs),
                  "unique_checks_that_crashed_their_worker_every_time": sum(job.sha in abandoned for job in jobs),
                  "attempts_without_a_later_answer": waiting,
                  # What the earlier pin said of them: one it VERIFIED would be a loss this report cannot see.
                  "attempts_without_a_later_answer_by_earlier_status": dict(Counter(
                      run.earlier[attempt["attempt_id"]]["status"] for attempt in unanswered)),
                  "attempts_rejected_lexically": sum(run.attempt_source(attempt, pin) is None for attempt in attempts)},
        "statements": statements,
        "tax": tax,
        "transitions": dict(sorted(transitions.items(), key=lambda item: -item[1])),
        "sensitivity": {
            # The two rules that differ between the pins, applied the earlier pin's way to the later pin's answers.
            "later_verified_with_a_failed_warning": sum(row["failed_warning"] for row in verified_later),
            f"later_verified_only_after_more_than_{earlier_limit}_seconds": sum((row["seconds"] or 0) > earlier_limit for row in verified_later),
            "later_verified_on_statements_that_do_not_compile": sum(
                1 for attempt in attempts if attempt["statement_id"] in broken
                and outcomes[attempt["attempt_id"]][0] is not None and outcomes[attempt["attempt_id"]][0].verified),
            "attempts_on_statements_that_do_not_compile": sum(attempt["statement_id"] in broken for attempt in attempts),
        },
        "examples_lost": examples(run, attempts, outcomes, compiling),
        # The classes above, finer: the deciding error's first line with names removed, most frequent first.
        "error_kinds": {
            "lost": error_kinds(attempts, outcomes, lambda attempt: attempt["statement_id"] in compiling
                                and run.earlier[attempt["attempt_id"]]["status"] == "verified"),
            "all_later_failures": error_kinds(attempts, outcomes, lambda attempt: attempt["statement_id"] in compiling),
        },
        "pool": {"sessions": read_rows(out / "sessions.jsonl") if (out / "sessions.jsonl").exists() else [],
                 "failed_requests": pool_failures(out / "request_failures.jsonl")},
    }
    return report


def pool_failures(path: Path) -> dict:
    """Every failed request, by kind and by minute: a burst is a pool that flapped."""
    rows = read_rows(path) if path.exists() else []
    by_minute = Counter(row["at"][:16] for row in rows)
    return {"total": len(rows), "by_kind": dict(Counter(row["kind"] for row in rows)),
            "first": rows[0]["at"] if rows else None, "last": rows[-1]["at"] if rows else None,
            "by_minute": dict(sorted(by_minute.items())), "sample_messages": sorted({row["message"][:160] for row in rows})[:6]}


def _percent(figure: dict) -> str:
    if figure.get("value") is None:
        return "n/a"
    if figure.get("low") is None:
        return f"{100 * figure['value']:.2f}%"
    return f"{100 * figure['value']:.2f}% [{100 * figure['low']:.2f}, {100 * figure['high']:.2f}]"


def _share(figure: dict) -> str:
    if figure.get("value") is None:
        return "n/a"
    return f"{figure['value']:.3f} [{figure['low']:.3f}, {figure['high']:.3f}]" if figure.get("low") is not None else f"{figure['value']:.3f}"


def _tax_row(name: str, section: dict) -> str:
    if not section.get("attempts"):
        return f"| {name} | 0 | 0 | | | | | | |"
    attempt, statement = section["per_attempt"], section["per_statement"]
    return (f"| {name} | {section['statements']} | {section['attempts']} | {_percent(attempt['earlier'])} | {_percent(attempt['later'])} | "
            f"{_share(attempt['later_as_share_of_earlier'])} | {_percent(statement['earlier'])} | {_percent(statement['later'])} | "
            f"{_share(statement['later_as_share_of_earlier'])} |")


def markdown(report: dict) -> str:
    tax, read, scope = report["tax"], report["tax"]["read"], report["scope"]
    earlier, later = report["pins"]["earlier"], report["pins"]["later"]
    lines = [f"# Version tax: {earlier} to {later}, identical proof texts", "",
             f"{'COMPLETE' if report['complete'] else 'PARTIAL'}: {scope['attempts_in_scope']} of {scope['attempts_stored']} stored attempts in scope, "
             f"{scope['unique_checks_answered']} of {scope['unique_checks']} unique checks answered, "
             f"{scope['attempts_without_a_later_answer']} attempts without an answer at {later}.", "",
             f"**Read (pre-registered: {read['rule']}):** "
             + (f"{later} verifies {_share(tax['overall']['per_attempt']['later_as_share_of_earlier'])} of what {earlier} verifies, "
                f"a relative loss of {100 * read['relative_loss']:.1f}%: the tax is **{read['read']}**"
                f"{'' if read['interval_on_one_side_of_the_threshold'] else ' (the interval crosses the threshold)'}."
                if read["later_as_share_of_earlier"] is not None else "undefined (nothing verified at the earlier pin)."), "",
             "## Statements that still compile (9d)", "", "| Set | Checked | Compile | Survival [95%] | Unknown identifier | Other error | Timeout | No answer |",
             "|---|---|---|---|---|---|---|---|"]
    for group, entry in report["statements"].items():
        classes = entry["do_not_compile_by_class"]
        survival = _percent({"value": entry.get("survival"), "low": entry.get("low"), "high": entry.get("high")})
        lines.append(f"| {group} | {entry['checked']} | {entry['compile']} | {survival} | {classes[UNKNOWN_IDENTIFIER]} | "
                     f"{classes[OTHER_ERROR]} | {classes[TIMEOUT]} | {entry['no_answer']} |")
    counts = tax["attempts"]
    lines += ["", "## Pass rates on the statements that compile at both pins (9e)", "",
              f"{counts['compared']} attempts compared ({counts['on_statements_compiling_at_both_pins']} on such statements, minus "
              f"{counts['no_answer_earlier']} with no answer at {earlier} and {counts['no_answer_later']} at {later}). Intervals: 95%, "
              "bootstrap over statements.", "",
              f"| | Statements | Attempts | Per attempt, {earlier} | Per attempt, {later} | {later} / {earlier} | Proved at least once, {earlier} | "
              f"Proved at least once, {later} | {later} / {earlier} |", "|---|---|---|---|---|---|---|---|---|",
              _tax_row("All", tax["overall"])]
    lines += [_tax_row(group, section) for group, section in tax.get("by_group", {}).items()]
    lines += [_tax_row(f"base pass rate {band}", section) for band, section in tax["by_band"].items()]
    lines += ["", f"## Failures at {later}", "", f"| | Unknown identifier | Other error | Timeout | Rejected before Lean | Verified at {later} only |",
              "|---|---|---|---|---|---|"]
    overall = tax["overall"]
    lines.append(f"| All attempts not verified at {later} | {overall['later_failures'][UNKNOWN_IDENTIFIER]} | {overall['later_failures'][OTHER_ERROR]} | "
                 f"{overall['later_failures'][TIMEOUT]} | {overall['later_failures']['rejected_lexical']} | |")
    lines.append(f"| Verified at {earlier}, not at {later} (the tax) | {overall['lost'][UNKNOWN_IDENTIFIER]} | {overall['lost'][OTHER_ERROR]} | "
                 f"{overall['lost'][TIMEOUT]} | {overall['lost']['rejected_lexical']} | {overall['gained']} |")
    lines += ["", "## Checks that could move the read", ""]
    lines += [f"- {key.replace('_', ' ')}: {value}" for key, value in report["sensitivity"].items()]
    failed = report["pool"]["failed_requests"]
    lines += ["", "## The pool", "", f"- failed requests (retried): {failed['total']} {failed['by_kind'] or ''}"
              + (f", from {failed['first']} to {failed['last']}" if failed["total"] else ""),
              f"- checks that crashed their worker every time and are left as no answer: {scope['unique_checks_that_crashed_their_worker_every_time']}; "
              f"attempts without an answer at {later}, by what {earlier} said: {scope['attempts_without_a_later_answer_by_earlier_status'] or 'none'}"]
    for session in report["pool"]["sessions"]:
        lines.append(f"- session {session['started']}: {session['answered']} answers in {session['seconds']} s "
                     f"({session['answers_per_second']}/s, {session['from_cache']} from the cache), {session['no_answer_rows']} without an answer, "
                     f"{session['in_flight']} in flight, Lean timeout {session['lean_timeout_seconds']} s")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-steps", type=Path, required=True, help="the earlier pin's finished run: its steps/ directory (read only)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, required=True, help="the pool's key; read, never printed")
    parser.add_argument("--ca-file", type=Path, required=True, help="the pool authority's certificate")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--limit", type=int, default=None, help="only the first N attempts of the seeded order (a sample to read by eye)")
    parser.add_argument("--retry-passes", type=int, default=2, help="extra passes over checks that got no answer")
    parser.add_argument("--report-only", action="store_true", help="send nothing: rebuild the report from what is stored")
    arguments = parser.parse_args()
    if arguments.limit is not None and arguments.limit < 1:
        parser.error("--limit must be at least 1")
    arguments.out.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(arguments.config.read_text())
    earlier_limit = config["kimina"]["lean_timeout_seconds"]
    config.setdefault("lean", {})["pin"] = LATER_PIN            # this tool IS the later pin's check; the earlier verdicts are stored
    pin = LEAN_PINS[LATER_PIN]
    run = StoredRun(arguments.run_steps)
    summary = {"lean_timeout_seconds": config["lean"]["pool"]["lean_timeout_seconds"], "earlier_lean_timeout_seconds": earlier_limit,
               "in_flight": config["lean"]["pool"]["concurrent_requests"], "pool": f"{config['lean']['pool']['name']}:{config['lean']['pool']['port']}",
               "run_steps": str(arguments.run_steps), "order_seed": ORDER_SEED}
    if not arguments.report_only:
        settings = lean_settings(config, api_key=arguments.api_key_file.read_text().strip(), ca_file=str(arguments.ca_file))
        _, _, jobs = plan(run, pin, arguments.limit)
        done, abandoned = answered(arguments.out / "checks.jsonl"), given_up(arguments.out / "checks.jsonl")
        todo = [job for job in jobs if job.sha not in done and job.sha not in abandoned]
        log(f"{len(jobs)} unique checks for {arguments.limit or len(run.attempts)} attempts; {sum(job.sha in done for job in jobs)} already "
            f"answered, {sum(job.sha in abandoned for job in jobs)} left as no answer (they crashed their worker in {MAX_CRASH_RUNS} runs), "
            f"{len(todo)} to run, {settings.concurrent_requests} in flight, Lean timeout {settings.lean_timeout_seconds} s")
        session = asyncio.run(run_checks(settings, todo, arguments.out, arguments.retry_passes))
        log(f"session: {json.dumps(session)}")
    report = build_report(run, summary, pin, arguments.limit, arguments.out)
    (arguments.out / "version_tax.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (arguments.out / "version_tax.md").write_text(markdown(report))
    print(markdown(report))
    return 0 if report["complete"] or arguments.limit is not None else 1


if __name__ == "__main__":
    sys.exit(main())
