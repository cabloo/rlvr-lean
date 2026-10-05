"""The ladder loop's pool: problems with a published answer that our own Lean has checked again.
Spec: docs/spec/ladder-loop.spec.md, milestone L0 (the data half). No GPU; only the Lean pool is used.

Commands, in the order they are run. `--store` is one directory that holds everything:

  fetch    the published files, pinned by revision and SHA-256, into <store>/sources (a wrong hash is refused)
  build    <store>/steps: candidates.jsonl (every statement with a published answer and its certificates in the
           order they are tried), set_aside.jsonl, excluded.jsonl, stp_index.jsonl. Asks Lean nothing.
  check    sends certificates through the pin's Lean pool and appends each answer to <store>/checks/checks.jsonl.
           For one problem the first certificate that verifies settles it; the rest are never sent.
           Resumable: kill it and run it again. Progress: <store>/checks/progress.json, rewritten every 30 s.
  sample   a small stratified sample (one named certificate per sampled problem), to read survival by source
  names    the library names the failed certificates report as unknown, by name and source, and what the config's
           table (`ladder_loop.certificates.renames`) renames each to. Prints; writes only where `--out` says.
  slice    FIRST, ahead of the rest: a seeded slice of candidates (`ladder_loop.slice`) through BOTH passes, so H
           and the base-map sample exist hours before the whole pool is checked. Each part of the slice is a
           prefix of the seeded order its draw uses, so the draw is the whole pool's, made early. Reuses every
           stored answer; resumable; `--dry-run` only counts. Progress: <store>/checks/slice_progress.json.
  rename   the SECOND pass over the whole pool, after `check` has finished: for every problem left with no
           certificate that verifies, its certificates with the table applied (new, marked certificates; the
           originals' failures stay on record), into <store>/steps/renamed.jsonl, then the same checker over them.
  pool     <store>/steps: heldout.jsonl, base_map.jsonl and pool.jsonl. Needs the SLICE settled through both
           passes (every candidate, if no slice was taken). H and the base-map sample are each drawn ONCE and
           kept; a short draw stops with how many more candidates the slice needs, writing nothing. The pool
           file holds what is verified so far and grows on later runs; a later copy of a member of H never enters.
  export   after `pool`: H and the base-map sample into `--out`, a data directory of the package that travels
           with the code (a stage on the GPU box cannot see this store). Statements and ids, no published proof.
  status   counts, without sending anything

Rules it keeps (the version tax tool's, whose checker it runs): a pool or transport failure is NO ANSWER, never
a Lean failure, and is asked again on the next run; identical Lean files are checked once; a check that crashes
its worker twice is left unanswered and counts as failed for its problem.

`check`, `slice` and `rename` are BACKGROUND work to the pool (lean-pool's README, "Background work and the pool's size"):
their checks carry `X-Lean-Priority: background`, so the pool takes them only when no solver's check is
waiting. A 503 from a pool that is up (the queue timed the check out behind the solver's) is waited out with a
pause and asked again, never stored. `--in-flight auto` follows the size the pool states (item 9b); against a
pool that states none it is `lean.pool.concurrent_requests`.

    PYTHONPATH=src python -m rlvr_lean.tools.ladder_pool check --store <dir> \\
        --api-key-file <the pool's key file> --ca-file <the pool authority's certificate> [--in-flight 11 | auto]
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from rlvr_lean.data.published import (
    Build,
    build_candidates,
    candidate_from_row,
    candidate_row,
    certificate_entry,
    certificate_from_entry,
    published_files,
    verified,
)
from rlvr_lean.data.ladder_export import exported_problem, write_export
from rlvr_lean.data.sources import all_minif2f_normalized
from rlvr_lean.domain.conjecturing import parse_type_fingerprint
from rlvr_lean.domain.evaluation.version_tax import (
    NO_ANSWER_STATUS,
    error_kind,
    failure_class,
    is_echo_of_a_failed_declaration,
    wilson_interval,
)
from rlvr_lean.domain.problem_pool import (
    CERTIFICATE_CHECK,
    EXACTNESS_CHECK,
    FALSE_SIDE,
    GOEDEL,
    IN,
    INTERNLM_PROOFS,
    INTERNLM_ROWS,
    LEAN_WORKBOOK,
    OPEN,
    OUT,
    SLICE_ALL,
    SLICE_PARTS,
    SLICE_PRESENT,
    SLICE_STP,
    SLICE_WORKBOOK,
    SOURCES,
    STP,
    STP_CONJECTURE,
    Candidate,
    Check,
    SoundnessAlarm,
    draw_base_map,
    draw_heldout,
    more_candidates_needed,
    pool_apart_from,
    pool_row,
    raise_on_both_sides,
    rank,
    settle,
    slice_parts,
    statement_check,
    steps,
)
from rlvr_lean.domain.problem_pool.renames import identifiers, opened_namespaces, renamed_certificate, written_forms
from rlvr_lean.domain.problem_pool.selection import BASE_MAP_LABEL, NO_CERTIFICATE_VERIFIES
from rlvr_lean.domain.verification import lean_pin, theorem_name_of
from rlvr_lean.gpu.pipeline import _compiles, _lean_messages        # the pipeline's own rule for "this file compiles"
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.infrastructure.kimina_client import BACKGROUND_PRIORITY
from rlvr_lean.infrastructure.verification_service import (
    AUTO_IN_FLIGHT,
    following_the_pool,
    lean_settings,
    waiting_out_the_proxy_queue,
)
from rlvr_lean.tools import version_tax
from rlvr_lean.tools.version_tax import Job, answered, given_up, read_rows, run_checks

CONFIG = Path(__file__).resolve().parents[1] / "config" / "experiment.yaml"
PROGRESS_EVERY_SECONDS = 30
MESSAGES_KEPT = 4
# The strata of the live sample: each names ONE certificate per sampled problem, so the table reads survival by
# source and not "does any of its proofs verify".
STRATA = ("internlm_proofs", "internlm_rows_proved", "goedel", "stp_statement", "stp_conjecture",
          "disproved_no_variables", "disproved_with_variables")
WRAPPER_MARK = "published_for_every_value"      # the name the variables-only negation wrapper introduces


def log(message: str) -> None:
    print(f"[ladder_pool {time.strftime('%H:%M:%S')}] {message}", flush=True)


def settings_of(config: dict) -> dict:
    settings = config["ladder_loop"]
    missing = set(SOURCES) - set(settings["certificates"]["source_order"])
    if missing:
        raise ValueError(f"ladder_loop.certificates.source_order does not place {sorted(missing)}")
    return settings


def local_files(config: dict, store: Path, download: bool) -> dict[str, Path]:
    """The verified local path of every pinned file, under the names `build_candidates` reads them by."""
    files, shard = {}, 0
    for published in published_files(config):
        path = verified(published, store / "sources", download=download)
        if published.source == "stp":
            files[f"stp:{shard:02d}"], shard = path, shard + 1
        elif published.source == "goedel_workbook_proofs":
            files["goedel"] = path
        else:
            files["internlm_workbook" if published.path.endswith(".json") else "internlm_rows"] = path
    return files


def present_holdout_of(run_steps: Path) -> set[str]:
    """The workbook holdout of the stored run: the validation half of its reward statements."""
    return {row["statement_id"] for row in read_rows(run_steps / "statements_reward.jsonl") if row.get("half") == "validation"}


# ------------------------------------------------------------------------------------------------ commands
def fetch(arguments, config: dict) -> int:
    for name, path in local_files(config, arguments.store, download=True).items():
        log(f"verified {name}: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


def build(arguments, config: dict) -> int:
    settings = settings_of(config)
    steps_store = ArtifactStore(arguments.store / "steps")
    if steps_store.is_done("build") and not arguments.again:
        log(f"already built: {json.dumps(steps_store.done_summary('build')['counts'])}")
        return 0
    files = local_files(config, arguments.store, download=False)
    minif2f = all_minif2f_normalized(arguments.store / "sources", config["data"]["deepseek_prover_commit"])
    result: Build = build_candidates(files, settings, lean_pin(settings["lean_pin"]), minif2f,
                                     present_holdout_of(arguments.present_run_steps), log)
    steps_store.write_rows("candidates.jsonl", (candidate_row(candidate) for candidate in result.candidates))
    steps_store.write_rows("set_aside.jsonl", result.set_aside)
    steps_store.write_rows("excluded.jsonl", result.excluded)
    steps_store.write_rows("stp_index.jsonl", result.stp_index)
    steps_store.write_rows("present_holdout.jsonl", ({"statement_id": name} for name in sorted(result.present_holdout)))
    steps_store.mark_done("build", {"counts": result.counts, "lean_pin": settings["lean_pin"], "settings": settings["certificates"],
                                    "present_run_steps": str(arguments.present_run_steps)})
    log(f"built: {json.dumps(result.counts, indent=1)}")
    return 0


RENAMED_STEP = "renamed"        # the marker of the renaming pass: its certificates are built (`renamed.jsonl`)


class Plan:
    """The candidates, the checks each may need, and the answers so far.

    Once the renaming pass has built its certificates they are part of every plan: each follows its problem's
    published certificates, so the first pass's answers are found where they were and only the new ones are
    open. `with_renamed=False` is the first pass alone, which is what decides who the renaming pass is for."""

    def __init__(self, store: Path, with_renamed: bool = True) -> None:
        self.steps_store = ArtifactStore(store / "steps")
        self.checks_directory = store / "checks"
        self.checks_directory.mkdir(parents=True, exist_ok=True)
        self.candidates: list[Candidate] = [candidate_from_row(row) for row in self.steps_store.read_rows("candidates.jsonl")]
        self.renamed_certificates = 0
        if with_renamed and self.steps_store.is_done(RENAMED_STEP):
            extra = {row["problem_id"]: row["certificates"] for row in self.steps_store.read_rows("renamed.jsonl")}
            self.renamed_certificates = sum(len(entries) for entries in extra.values())
            self.candidates = [candidate if candidate.problem_id not in extra else dataclasses.replace(
                candidate, certificates=candidate.certificates + tuple(
                    certificate_from_entry(candidate.problem_id, candidate.side, entry) for entry in extra[candidate.problem_id]))
                for candidate in self.candidates]
        self.checks: dict[str, list[Check]] = {candidate.problem_id: steps(candidate) for candidate in self.candidates}
        self.set_aside = self.steps_store.read_rows("set_aside.jsonl")
        self.statement_checks = [statement_check(row["statement_id"], row["statement"]) for row in self.set_aside]
        self.reload()

    def reload(self) -> None:
        self.answers = answered(self.checks_directory / "checks.jsonl")
        self.abandoned = given_up(self.checks_directory / "checks.jsonl")

    def verdicts(self) -> dict[str, object]:
        return {candidate.problem_id: settle(candidate, self.checks[candidate.problem_id], self.answers, self.abandoned)
                for candidate in self.candidates}

    def next_checks(self, scope: set[str] | None = None) -> list[Check]:
        """The one check each open candidate needs now (of `scope` only, when given: the slice), then the
        set-aside statements not yet answered."""
        wanted = [verdict.next_check for problem_id, verdict in self.verdicts().items()
                  if verdict.state == OPEN and (scope is None or problem_id in scope)]
        wanted += [check for check in self.statement_checks if check.sha not in self.answers and check.sha not in self.abandoned]
        unique: dict[str, Check] = {}
        for check in wanted:
            unique.setdefault(check.sha, check)
        return list(unique.values())


def answer_of(checks_by_sha: dict[str, Check]):
    """The row stored for one answer: the pin's verdict, whether the file compiled, the statement's fingerprint."""
    def row_of(job: Job, raw: dict, pin) -> dict:
        check = checks_by_sha[job.sha]
        result = pin.classify(job.sha, raw)
        errors = _lean_messages(raw, "error")
        return {"sha": job.sha, "kind": job.kind, "problem_id": check.problem_id, "source": check.source,
                "status": result.status.value, "compiles": _compiles(raw), "seconds": result.verification_seconds,
                "fingerprint": parse_type_fingerprint(_lean_messages(raw, "info")),
                "errors": [message[:600] for message in errors[:MESSAGES_KEPT]], "error_count": len(errors),
                "detail": result.detail[:300], "cached": bool(raw.get("cached")),
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return row_of


class Progress:
    """`progress.json`: what a watcher reads. Rewritten at most every `PROGRESS_EVERY_SECONDS`."""

    def __init__(self, path: Path, plan: Plan, started: float, scope: set[str] | None = None) -> None:
        self.path, self.plan, self.started, self.scope = path, plan, started, scope
        self.session_answers, self.session_no_answer, self.last_written, self.round = 0, 0, 0.0, 0
        self.settled_at_start = self._settled()

    def _verdicts(self) -> dict:
        """The verdicts this run is about: every candidate's, or the scope's (the slice)."""
        verdicts = self.plan.verdicts()
        return verdicts if self.scope is None else {key: verdict for key, verdict in verdicts.items() if key in self.scope}

    def _settled(self) -> int:
        return sum(verdict.state != OPEN for verdict in self._verdicts().values())

    def on_answer(self, row: dict) -> None:
        if row["status"] == NO_ANSWER_STATUS:
            self.session_no_answer += 1
        else:
            self.session_answers += 1
            self.plan.answers[row["sha"]] = row
        if time.monotonic() - self.last_written >= PROGRESS_EVERY_SECONDS:
            self.write("running")

    def write(self, state: str) -> dict:
        self.last_written = time.monotonic()
        elapsed = time.monotonic() - self.started
        verdicts = self._verdicts()
        states = Counter(verdict.state for verdict in verdicts.values())
        settled_now = states[IN] + states[OUT] - self.settled_at_start
        # Checks still to send: an open problem costs what a settled one has cost on average (most are settled
        # by their first certificate), never less than one.
        settled_costs = [verdict.checks_answered for verdict in verdicts.values() if verdict.state != OPEN and verdict.checks_answered]
        per_problem = max(1.0, sum(settled_costs) / len(settled_costs)) if settled_costs else 1.0
        rate = self.session_answers / elapsed if elapsed > 0 else 0.0
        remaining = states[OPEN] * per_problem
        progress = {"state": state, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "round": self.round,
                    "problems": len(verdicts), "in_pool": states[IN], "out": states[OUT], "open": states[OPEN],
                    "settled_this_session": settled_now, "answers_this_session": self.session_answers,
                    "no_answer_this_session": self.session_no_answer, "answers_stored": len(self.plan.answers),
                    "answers_per_second": round(rate, 3), "checks_per_problem": round(per_problem, 3),
                    "checks_remaining_estimate": round(remaining),
                    "eta_seconds": round(remaining / rate) if rate > 0 else None, "elapsed_seconds": round(elapsed)}
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=1))
        temporary.replace(self.path)
        return progress


def in_flight_argument(text: str):
    """`--in-flight`: a whole number of at least 1, or `auto` (follow the size the pool states)."""
    if text.strip().lower() == AUTO_IN_FLIGHT:
        return AUTO_IN_FLIGHT
    try:
        number = int(text)
    except ValueError:
        number = 0
    if number < 1:
        raise argparse.ArgumentTypeError(f"a whole number of at least 1, or '{AUTO_IN_FLIGHT}'")
    return number


def pin_settings(arguments, config: dict, background: bool = False):
    """The pin's settings with this run's requests in flight. `background` (the bulk certificate check: `check`,
    `slice`, `rename`) marks every check as background work (lean-pool's README, "Background work and the pool's size"): the pool then
    takes them only when no solver's check is waiting, so the client waits as long as the pool's queue may
    hold a check, and a 503 from a pool that is up is waited out, not recorded. `sample` is a measurement
    with a budget of its own and stays a normal client."""
    config.setdefault("lean", {})["pin"] = settings_of(config)["lean_pin"]
    settings = lean_settings(config, api_key=arguments.api_key_file.read_text().strip(), ca_file=str(arguments.ca_file))
    settings = following_the_pool(settings, arguments.in_flight)
    if not background:
        return settings
    return dataclasses.replace(waiting_out_the_proxy_queue(settings, config), priority=BACKGROUND_PRIORITY)


def in_flight_text(settings) -> str:
    if not settings.follow_pool_size:
        return f"{settings.concurrent_requests} in flight"
    return (f"requests in flight follow the pool's stated size x {settings.pool_size_margin} "
            f"({settings.concurrent_requests} until it states one, at most {settings.pool_size_ceiling})")


def run(settings, plan: Plan, wanted: list[Check], progress: Progress, retry_passes: int) -> dict:
    """Send `wanted`, minus what is already answered or given up."""
    todo = [check for check in wanted if check.sha not in plan.answers and check.sha not in plan.abandoned]
    by_sha = {check.sha: check for check in todo}
    jobs = [Job(check.sha, check.kind, settings.pin.source(check.lean_file)) for check in todo]
    session = asyncio.run(run_checks(settings, jobs, plan.checks_directory, retry_passes, answer_of(by_sha), progress.on_answer))
    plan.reload()
    return session


def check(arguments, config: dict, scope: set[str] | None = None, progress_file: str = "progress.json") -> int:
    """Send what is open, round after round, until nothing is. `scope` limits it to those problems (the
    slice), with a progress file of its own; the answers go to the one `checks.jsonl` either way."""
    version_tax.LOG_LABEL = "ladder_pool"
    plan = Plan(arguments.store)
    settings = pin_settings(arguments, config, background=True)
    progress = Progress(plan.checks_directory / progress_file, plan, time.monotonic(), scope)
    log(f"{len(plan.candidates) if scope is None else len(scope)} candidates{'' if scope is None else ' (the slice)'}, "
        f"{len(plan.statement_checks)} set-aside statements, {len(plan.answers)} answers stored; "
        f"{in_flight_text(settings)}, priority {settings.priority}, Lean timeout {settings.lean_timeout_seconds} s, "
        f"pin {settings.pin.name}")
    budget = arguments.max_checks
    while True:
        wanted = plan.next_checks(scope)
        if budget is not None:
            wanted = wanted[:budget]
        if not wanted:
            break
        progress.round += 1
        log(f"round {progress.round}: {len(wanted)} checks ({Counter(check.kind for check in wanted).most_common()})")
        before = (len(plan.answers), len(plan.abandoned))
        session = run(settings, plan, wanted, progress, arguments.retry_passes)
        if budget is not None:
            budget -= len(wanted)
        try:
            alarm_check(plan)
        except SoundnessAlarm as alarm:
            progress.write("SOUNDNESS ALARM")
            log(f"SOUNDNESS ALARM: {alarm}")
            return 3
        if session["stopped_early"] or (len(plan.answers), len(plan.abandoned)) == before or (budget is not None and budget <= 0):
            break               # a signal; or a round that settled nothing (the pool is down: run again later)
    final = progress.write("finished" if not plan.next_checks(scope) else "stopped")
    log(f"{final['state']}: {json.dumps(final)}")
    return 0 if final["state"] == "finished" else 1


def verified_rows(plan: Plan) -> list[dict]:
    verdicts = plan.verdicts()
    return [pool_row(candidate, verdicts[candidate.problem_id]) for candidate in plan.candidates
            if verdicts[candidate.problem_id].state == IN]


def alarm_check(plan: Plan) -> None:
    raise_on_both_sides(verified_rows(plan))


def stratum_of(candidate: Candidate, certificate_index: int) -> str:
    certificate = candidate.certificates[certificate_index]
    if candidate.side == FALSE_SIDE:
        return "disproved_with_variables" if WRAPPER_MARK in certificate.proof else "disproved_no_variables"
    if candidate.kind == STP_CONJECTURE:
        return "stp_conjecture"
    return {INTERNLM_PROOFS: "internlm_proofs", INTERNLM_ROWS: "internlm_rows_proved", GOEDEL: "goedel", STP: "stp_statement"}[certificate.source]


def sample_plan(plan: Plan, per_stratum: dict[str, int], seed: int) -> dict[str, list[tuple[Candidate, list[Check]]]]:
    """Per stratum: the sampled problems, each with the checks that decide ONE named certificate (the first of
    its source), plus the exactness check for a known-false problem."""
    members: dict[str, list[tuple[Candidate, list[Check]]]] = defaultdict(list)
    for candidate in plan.candidates:
        checks = plan.checks[candidate.problem_id]
        offset = len(checks) - len(candidate.certificates)          # 1 for a known-false problem: its exactness check
        seen = set()
        for index in range(len(candidate.certificates)):
            stratum = stratum_of(candidate, index)
            if stratum not in seen:
                seen.add(stratum)
                members[stratum].append((candidate, checks[:offset] + [checks[offset + index]]))
    return {stratum: sorted(members[stratum], key=lambda item: rank(seed, f"sample:{stratum}", item[0].problem_id))[:per_stratum.get(stratum, 0)]
            for stratum in STRATA}


_MISSING_NAME = re.compile(r"[Uu]nknown (?:identifier|constant) `([^`]+)`")


def own_errors(row: dict, check: Check) -> list[str]:
    """A failed certificate's stored errors without the echo of its own failed theorem (`#print axioms` not
    finding it), which would otherwise read as a missing library name."""
    name = theorem_name_of(check.lean_file)
    return [message for message in row["errors"] if not is_echo_of_a_failed_declaration(message, name)]


def sample(arguments, config: dict) -> int:
    version_tax.LOG_LABEL = "ladder_pool"
    plan = Plan(arguments.store)
    sizes = dict(zip(STRATA, arguments.sizes))
    chosen = sample_plan(plan, sizes, arguments.seed)
    wanted = [check for members in chosen.values() for _, checks in members for check in checks]
    total = len({check.sha for check in wanted})
    if total > arguments.max_checks:
        raise SystemExit(f"the sample is {total} checks, more than --max-checks {arguments.max_checks}")
    if not arguments.report_only:
        settings = pin_settings(arguments, config)
        progress = Progress(plan.checks_directory / "sample_progress.json", plan, time.monotonic())
        log(f"sample: {total} checks, {settings.concurrent_requests} in flight, pin {settings.pin.name}")
        session = run(settings, plan, wanted, progress, arguments.retry_passes)
        log(f"session: {json.dumps(session)}")
    report = {"seed": arguments.seed, "strata": {}}
    missing_names: Counter = Counter()
    for stratum, members in chosen.items():
        statuses, classes, kinds, verified_count, answered_count, inexact = Counter(), Counter(), Counter(), 0, 0, 0
        seconds = []
        for candidate, checks in members:
            rows = [plan.answers.get(check.sha) for check in checks]
            if any(row is None for row in rows):
                statuses["no_answer"] += 1
                continue
            answered_count += 1
            exact = all(row["compiles"] for check, row in zip(checks, rows) if check.kind == EXACTNESS_CHECK)
            certificate = rows[-1]
            seconds.append(certificate["seconds"] or 0.0)
            if not exact:
                inexact += 1
            if exact and certificate["status"] == "verified":
                verified_count += 1
                statuses["verified"] += 1
            else:
                statuses[certificate["status"] if exact else "negation_not_exact"] += 1
                if certificate["status"] != "verified":
                    errors = own_errors(certificate, checks[-1])
                    classes[failure_class(certificate["status"], errors) or certificate["status"]] += 1
                    kinds[error_kind(errors[0] if errors else certificate["detail"])] += 1
                    missing_names.update({name for message in errors for name in _MISSING_NAME.findall(message)})
        low, high = wilson_interval(verified_count, answered_count) if answered_count else (None, None)
        report["strata"][stratum] = {"sampled": len(members), "answered": answered_count, "verified": verified_count,
                                     "survival": verified_count / answered_count if answered_count else None, "low": low, "high": high,
                                     "statuses": dict(statuses), "failure_classes": dict(classes),
                                     "error_kinds": dict(kinds.most_common(6)), "negation_not_exact": inexact,
                                     "median_seconds": sorted(seconds)[len(seconds) // 2] if seconds else None}
    # The library names failed certificates could not find: a published proof is written for the Mathlib of its day.
    report["library_names_not_found"] = dict(missing_names.most_common(20))
    sessions = read_rows(plan.checks_directory / "sessions.jsonl") if (plan.checks_directory / "sessions.jsonl").exists() else []
    report["sessions"] = sessions
    (plan.checks_directory / "sample_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


# ----------------------------------------------------------------------------------------- the renaming pass
def renames_table(config: dict) -> dict[str, str]:
    """The config's table, old library name -> new. An entry that maps a name to itself, or to nothing, is a
    configuration error: it would mark certificates as renamed that are not."""
    table = dict(settings_of(config)["certificates"].get("renames") or {})
    bad = sorted(old for old, new in table.items() if not isinstance(new, str) or not new.strip() or new == old)
    if bad:
        raise ValueError(f"ladder_loop.certificates.renames has entries with no new name: {bad}")
    return table


def certificates_by_sha(plan: Plan) -> dict[str, tuple[Candidate, object, Check]]:
    """check sha -> (its candidate, its certificate, the check), for every certificate check of the plan."""
    found = {}
    for candidate in plan.candidates:
        checks = plan.checks[candidate.problem_id]
        offset = len(checks) - len(candidate.certificates)          # 1 for a known-false problem: its exactness check
        for certificate, check in zip(candidate.certificates, checks[offset:]):
            found[check.sha] = (candidate, certificate, check)
    return found


def names_report(plan: Plan, table: dict[str, str]) -> dict:
    """The library names the FAILED certificates report as unknown, from the stored Lean messages: how many
    certificates and problems name each, by source, how the proofs write it (as reported, or without a
    namespace the header opens), and what the table renames it to. This is what the table is built from."""
    namespaces = opened_namespaces()
    by_sha = certificates_by_sha(plan)
    names: dict[str, dict] = {}
    failed = naming = more_errors_than_stored = covered = 0
    for sha, row in plan.answers.items():
        if row["kind"] != CERTIFICATE_CHECK or row["status"] == "verified" or sha not in by_sha:
            continue
        candidate, certificate, check = by_sha[sha]
        failed += 1
        more_errors_than_stored += row.get("error_count", 0) > len(row["errors"])
        reported = {name for message in own_errors(row, check) for name in _MISSING_NAME.findall(message)}
        naming += bool(reported)
        # Every identifier of the proof and its dotted prefixes: `le_div_iff.mpr` cites `le_div_iff`.
        written = {text[:cut] for _, _, text in identifiers(certificate.proof)
                   for cut in [index for index, character in enumerate(text) if character == "."] + [len(text)]}
        covered += any(form in table for name in reported for form in written_forms(name, namespaces) if form in written)
        for name in reported:
            entry = names.setdefault(name, {"certificates": 0, "problems": set(), "by_source": Counter(), "written_as": Counter()})
            entry["certificates"] += 1
            entry["problems"].add(candidate.problem_id)
            entry["by_source"][f"{candidate.kind}/{certificate.source}" + ("/renamed" if certificate.renamed else "")] += 1
            for form in written_forms(name, namespaces):
                if form in written:
                    entry["written_as"][form] += 1
    rows = []
    for name, entry in sorted(names.items(), key=lambda item: (-item[1]["certificates"], item[0])):
        forms = [form for form in written_forms(name, namespaces) if entry["written_as"][form]] or [name]
        rows.append({"name": name, "certificates": entry["certificates"], "problems": len(entry["problems"]),
                     "by_source": dict(entry["by_source"].most_common()), "written_as": dict(entry["written_as"].most_common()),
                     "renamed_to": {form: table[form] for form in forms if form in table},
                     "written_forms_not_in_table": [form for form in forms if form not in table]})
    return {"failed_certificates": failed, "naming_an_unknown_name": naming,
            "naming_a_name_the_table_renames": covered, "with_more_errors_than_were_stored": more_errors_than_stored,
            "distinct_names": len(rows), "table_entries": len(table), "names": rows}


def names(arguments, config: dict) -> int:
    """Print the tabulation; write it only where `--out` says (the store is not touched)."""
    report = names_report(Plan(arguments.store), renames_table(config))
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if arguments.out is not None:
        arguments.out.write_text(text)
    print(text)
    return 0


def renamed_rows(first_pass: Plan, table: dict[str, str], scope: set[str] | None = None) -> tuple[list[dict], dict]:
    """(the rows of `renamed.jsonl`, counts) from the FIRST pass alone: for every problem it left with no
    certificate that verifies (of `scope` only, when given: the slice), each of its certificates with the
    table applied, where that changes the proof and gives a text the problem does not already have. A
    problem's rows depend on that problem alone, so the slice's rows are the whole pass's rows for it."""
    verdicts = first_pass.verdicts()
    rows, counts, by_name = [], Counter(), Counter()
    for candidate in first_pass.candidates:
        if scope is not None and candidate.problem_id not in scope:
            continue
        verdict = verdicts[candidate.problem_id]
        counts[f"first_pass_{verdict.state}"] += 1
        if verdict.state != OUT or verdict.reason != NO_CERTIFICATE_VERIFIES:
            continue
        counts["unsettled_no_certificate_verifies"] += 1
        texts = {certificate.proof for certificate in candidate.certificates}
        added = []
        for certificate in candidate.certificates:
            renamed = renamed_certificate(certificate, table)
            if renamed is not None and renamed.proof not in texts:
                texts.add(renamed.proof)
                added.append(renamed)
        if added:
            counts["problems_with_a_renamed_certificate"] += 1
            counts[f"problems_with_a_renamed_certificate_{candidate.kind}"] += 1
            counts["renamed_certificates"] += len(added)
            by_name.update({name for certificate in added for name in certificate.renamed})
            rows.append({"problem_id": candidate.problem_id, "certificates": [certificate_entry(certificate) for certificate in added]})
    return rows, {**dict(sorted(counts.items())), "problems_by_renamed_name": dict(by_name.most_common())}


def rename(arguments, config: dict) -> int:
    """The second pass: build the renamed certificates of what the first pass left unsettled, then check them
    with the same checker (same resume rules, same alarm). `--dry-run` counts from what is stored now and
    writes nothing."""
    table = renames_table(config)
    first_pass = Plan(arguments.store, with_renamed=False)
    rows, counts = renamed_rows(first_pass, table)
    if arguments.dry_run:
        print(json.dumps({"dry_run": True, "table_entries": len(table), **counts}, indent=1, ensure_ascii=False))
        return 0
    if counts.get(f"first_pass_{OPEN}", 0):
        raise SystemExit(f"the first pass is not finished: {counts[f'first_pass_{OPEN}']} candidates are still open. Run `check` to its end "
                         "first; the renaming pass is for what it leaves unsettled.")
    first_pass.steps_store.write_rows("renamed.jsonl", rows)
    first_pass.steps_store.mark_done(RENAMED_STEP, {"table": table, "scope": WHOLE_POOL, "counts": counts})
    log_expected_renames("renaming pass", Plan(arguments.store), counts)
    return check(arguments, config)


def log_expected_renames(label: str, plan: Plan, counts: dict) -> None:
    sent = sum(check.sha in plan.answers or check.sha in plan.abandoned for _, certificate, check in certificates_by_sha(plan).values()
               if certificate.renamed)
    log(f"{label}: {counts.get('unsettled_no_certificate_verifies', 0)} problems the first pass left unsettled; "
        f"{counts.get('problems_with_a_renamed_certificate', 0)} of them have a certificate that carries a tabled name; "
        f"EXPECTED CHECKS: at most {counts.get('renamed_certificates', 0)} (one per renamed certificate), at least "
        f"{counts.get('problems_with_a_renamed_certificate', 0)} (a problem stops at its first that verifies); {sent} already answered")


# ------------------------------------------------------------------------------------------------ the slice
SLICE_STEP = "slice"            # the marker of the slice: its members are written (`slice.jsonl`)
WHOLE_POOL, THE_SLICE = "all", "slice"          # what a renaming pass was built for


def slice_settings(config: dict) -> dict:
    settings = settings_of(config)
    return {"sizes": dict(settings["slice"]), "heldout_seed": settings["heldout"]["seed"], "base_map_seed": settings["base_map"]["seed"]}


def stored_slice(steps_store: ArtifactStore) -> dict[str, list[str]] | None:
    """The slice as written: its parts, each in its seeded order. None when no slice was taken."""
    if not steps_store.is_done(SLICE_STEP):
        return None
    parts: dict[str, list[str]] = {part: [] for part in SLICE_PARTS}
    for row in steps_store.read_rows("slice.jsonl"):
        parts[row["part"]].append(row["problem_id"])
    return parts


def members_of(parts: dict[str, list[str]]) -> set[str]:
    return {problem_id for ids in parts.values() for problem_id in ids}


def slice_preview(plan: Plan, parts: dict[str, list[str]]) -> dict:
    """Per part of the slice: its candidates, how many are verified, failed or still open."""
    verdicts = plan.verdicts()
    return {part: {"candidates": len(ids), **{state: sum(verdicts[problem_id].state == state for problem_id in ids) for state in (IN, OUT, OPEN)}}
            for part, ids in parts.items()}


def slice_first(arguments, config: dict) -> int:
    """`slice`: take the slice through BOTH passes ahead of the rest of the pool (spec: "The held-out sets do
    not wait for the whole pool"). The first pass on the slice's candidates and the set-aside statements,
    then the renaming pass on what it left unsettled among them, with the same checker, the same resume rules
    and the same alarm, reusing every answer already stored. `--dry-run` counts and writes nothing."""
    table = renames_table(config)
    first_pass = Plan(arguments.store, with_renamed=False)
    steps_store = first_pass.steps_store
    present = {row["statement_id"] for row in steps_store.read_rows("present_holdout.jsonl")}
    wanted = slice_settings(config)
    parts = slice_parts(((candidate.problem_id, candidate.kind) for candidate in first_pass.candidates), present,
                        wanted["heldout_seed"], wanted["base_map_seed"], wanted["sizes"])
    members = members_of(parts)
    if arguments.dry_run:
        verdicts = first_pass.verdicts()
        open_now = [problem_id for problem_id in members if verdicts[problem_id].state == OPEN]
        statements = sum(check.sha not in first_pass.answers and check.sha not in first_pass.abandoned for check in first_pass.statement_checks)
        _, renames = renamed_rows(first_pass, table, members)
        print(json.dumps({"dry_run": True, "members": len(members), "by_part": slice_preview(first_pass, parts),
                          "first_pass_open": len(open_now), "set_aside_statements_to_check": statements,
                          "new_checks_at_least": len(open_now) + statements,
                          "renamed_certificates_so_far": renames.get("renamed_certificates", 0)}, indent=1))
        return 0
    if steps_store.is_done("heldout") and steps_store.is_done(SLICE_STEP) and steps_store.done_summary(SLICE_STEP)["settings"] != wanted:
        raise SystemExit("H is already drawn from the slice that is written, and the config now describes another slice: refused. "
                         "Nothing was changed; the rest of the pool is checked with `check`.")
    marker = steps_store.done_summary(RENAMED_STEP) if steps_store.is_done(RENAMED_STEP) else None
    whole_pass_built = marker is not None and marker.get("scope", WHOLE_POOL) == WHOLE_POOL
    if whole_pass_built and marker["table"] != table:
        raise SystemExit("the renaming pass was built for the whole pool with another table: run `rename` again; the slice needs nothing more.")
    steps_store.write_rows("slice.jsonl", ({"part": part, "position": position, "problem_id": problem_id}
                                           for part, ids in parts.items() for position, problem_id in enumerate(ids)))
    steps_store.mark_done(SLICE_STEP, {"settings": wanted, "counts": {part: len(ids) for part, ids in parts.items()}, "members": len(members)})
    log(f"slice: {len(members)} candidates ({', '.join(f'{part} {len(ids)}' for part, ids in parts.items())}); first pass")
    code = check(arguments, config, scope=members, progress_file="slice_progress.json")
    if code != 0:
        return code
    if not whole_pass_built:
        first_pass.reload()
        rows, counts = renamed_rows(first_pass, table, members)
        steps_store.write_rows("renamed.jsonl", rows)
        steps_store.mark_done(RENAMED_STEP, {"table": table, "scope": THE_SLICE, "slice": wanted, "counts": counts})
        log_expected_renames("slice, renaming pass", Plan(arguments.store), counts)
        code = check(arguments, config, scope=members, progress_file="slice_progress.json")
    if code == 0:
        log(f"slice settled through both passes: {json.dumps(slice_preview(Plan(arguments.store), parts))}. Next: `pool`, then `export`.")
    return code


def _stored_draw(steps_store: ArtifactStore, step: str, file: str, wanted: dict, verified_ids: set[str], what: str) -> list[dict] | None:
    """A draw that was already made (H, the base-map sample), kept as written; None when it was not made yet.
    It is fixed once: other settings, or a member the stored answers no longer verify, is refused."""
    if not steps_store.is_done(step):
        return None
    stored = steps_store.done_summary(step)
    if stored["settings"] != wanted:
        raise SystemExit(f"{what} was drawn with {stored['settings']} and the config now says {wanted}: refused. It is fixed once; "
                         "nothing was changed.")
    rows = steps_store.read_rows(file)
    lost = sorted({row["problem_id"] for row in rows} - verified_ids)
    if lost:
        raise SystemExit(f"{len(lost)} members of {what} are not verified by the stored answers (first: {lost[0]}): refused; nothing was changed.")
    log(f"{what} is already drawn ({len(rows)} problems): kept as written")
    return rows


def pool(arguments, config: dict) -> int:
    """H, the base-map sample and the pool file.

    What must be settled through BOTH passes first is the SLICE when one was taken (spec: "The held-out sets
    do not wait for the whole pool"), else every candidate. H and the base-map sample are each drawn once and
    kept; the pool file is written with what is verified so far and grows on later runs, and a later copy of
    a member of H never enters it. A draw that comes out short of what the config wants stops here, saying
    how many more candidates the slice needs, with nothing written."""
    settings = settings_of(config)
    plan = Plan(arguments.store)
    steps_store = plan.steps_store
    parts = stored_slice(steps_store)
    scope = None if parts is None else members_of(parts)
    if not steps_store.is_done(RENAMED_STEP):
        raise SystemExit("the renaming pass has not run: the held-out sets are drawn only after BOTH passes. Run `slice` (the slice "
                         "through both passes) or `rename` (the whole pool; it needs the first pass finished) first.")
    marker = steps_store.done_summary(RENAMED_STEP)
    if marker["table"] != renames_table(config):
        raise SystemExit("the renames table changed since the renaming pass was built: run `slice` or `rename` again, then `pool`.")
    if marker.get("scope", WHOLE_POOL) != WHOLE_POOL and (parts is None or marker.get("slice") != steps_store.done_summary(SLICE_STEP)["settings"]):
        raise SystemExit("the renaming pass was built for another slice than the one that is written: run `slice` again, then `pool`.")
    verdicts = plan.verdicts()
    still_open = sum(verdict.state == OPEN for problem_id, verdict in verdicts.items() if scope is None or problem_id in scope)
    unanswered = [check for check in plan.statement_checks if check.sha not in plan.answers and check.sha not in plan.abandoned]
    if still_open or unanswered:
        raise SystemExit(f"{still_open} candidates{'' if scope is None else ' of the slice'} and {len(unanswered)} set-aside statements are not "
                         f"settled: run `{'check' if scope is None else 'slice'}` to its end first (a draw is made once, from a settled set)")
    rows = verified_rows(plan)
    raise_on_both_sides(rows)
    by_id = {row["problem_id"]: row for row in rows}
    set_aside = []
    for row, statement in zip(plan.set_aside, plan.statement_checks):
        answer = plan.answers.get(statement.sha, {})
        set_aside.append({**row, "compiles": bool(answer.get("compiles")), "type_fingerprint": answer.get("fingerprint")})
    present = {row["statement_id"] for row in steps_store.read_rows("present_holdout.jsonl")}
    wanted, wanted_map = settings["heldout"], settings["base_map"]
    short: list[str] = []

    # ---- H: kept if drawn; else drawn from the slice's held-out parts (or from every verified row)
    heldout = _stored_draw(steps_store, "heldout", "heldout.jsonl", wanted, set(by_id), "H")
    new_heldout = heldout is None
    if new_heldout:
        eligible = None if parts is None else set(parts[SLICE_PRESENT]) | set(parts[SLICE_WORKBOOK]) | set(parts[SLICE_STP])
        drawn = draw_heldout([row for row in rows if eligible is None or row["problem_id"] in eligible], present, set_aside,
                             wanted["seed"], wanted["workbook_problems"], wanted["stp_conjectures"])
        heldout = drawn.heldout
        heldout_counts = {key: value for key, value in drawn.counts.items() if key.startswith("heldout_")}
        if parts is not None:
            from_sample = heldout_counts[f"heldout_{LEAN_WORKBOOK}"] - heldout_counts["heldout_workbook_from_present_holdout"]
            for part, kind, taken in ((SLICE_WORKBOOK, LEAN_WORKBOOK, from_sample), (SLICE_STP, STP_CONJECTURE, heldout_counts[f"heldout_{STP_CONJECTURE}"])):
                missing = heldout_counts[f"heldout_{kind}_short_by"]
                if missing > 0:
                    short.append(f"H is {missing} {kind} problems short: raise ladder_loop.slice.{part} from {len(parts[part])} by about "
                                 f"{more_candidates_needed(missing, taken, len(parts[part]))}")
    else:
        heldout_counts = steps_store.done_summary("heldout")["counts"]

    # ---- the base-map sample: kept if drawn; else the first verified of its seeded order outside H
    sample = _stored_draw(steps_store, "base_map", "base_map.jsonl", wanted_map, set(by_id), "the base-map sample")
    new_sample = sample is None
    if new_sample:
        order = parts[SLICE_ALL] if parts is not None else sorted(by_id, key=lambda problem_id: rank(wanted_map["seed"], BASE_MAP_LABEL, problem_id))
        sample, sample_counts = draw_base_map(by_id, order, heldout, set_aside, wanted_map["problems"])
        if parts is not None and sample_counts["base_map_short_by"] > 0:
            short.append(f"the base-map sample is {sample_counts['base_map_short_by']} problems short: raise ladder_loop.slice.{SLICE_ALL} from "
                         f"{len(parts[SLICE_ALL])} by about {more_candidates_needed(sample_counts['base_map_short_by'], len(sample), len(parts[SLICE_ALL]))}")
    else:
        sample_counts = steps_store.done_summary("base_map")["counts"]
    if short:
        raise SystemExit("; ".join(short) + ". Then run `slice` again (it checks only the candidates that are new) and `pool`. Nothing was written.")

    remaining, dropped = pool_apart_from(rows, heldout, set_aside, kept_first=[row["problem_id"] for row in sample])
    out_reasons = Counter(verdict.reason for verdict in verdicts.values() if verdict.state == OUT)
    by_source = Counter((row["kind"], row["side"], row["certificate_source"]) for row in rows)
    summary = {"candidates": len(plan.candidates), "verified": len(rows), "not_verified_by_reason": dict(out_reasons),
               "candidates_still_open": sum(verdict.state == OPEN for verdict in verdicts.values()),
               "drawn_from": "the whole pool" if parts is None else "the slice",
               "slice": None if parts is None else slice_preview(plan, parts),
               "verified_by_kind_side_source": {" ".join(key): value for key, value in sorted(by_source.items())},
               "verified_with_a_rewritten_statement": sum(row["rewritten"] > 0 for row in rows),
               "verified_by_a_renamed_certificate": sum(bool(row["certificate_renamed"]) for row in rows),
               "renaming_pass": {"built_for": marker.get("scope", WHOLE_POOL), **marker["counts"]},
               "set_aside": len(set_aside), "set_aside_compiling": sum(row["compiles"] for row in set_aside),
               "heldout": heldout_counts, "heldout_settings": wanted, "base_map": sample_counts, "base_map_settings": wanted_map, "pool": dropped,
               "pool_by_kind_side": {" ".join(key): value for key, value in sorted(Counter((row["kind"], row["side"]) for row in remaining).items())},
               "checks_given_up": len(plan.abandoned), "answers": len(plan.answers)}
    if new_heldout:
        steps_store.write_rows("heldout.jsonl", heldout)
        steps_store.mark_done("heldout", {"settings": wanted, "counts": heldout_counts})
    if new_sample:
        steps_store.write_rows("base_map.jsonl", sample)
        steps_store.mark_done("base_map", {"settings": wanted_map, "counts": sample_counts})
    steps_store.write_rows("pool.jsonl", remaining)
    steps_store.write_rows("set_aside_checked.jsonl", set_aside)
    steps_store.mark_done("pool", summary)
    print(json.dumps(summary, indent=1))
    return 0


def status(arguments, config: dict) -> int:
    plan = Plan(arguments.store)
    verdicts = plan.verdicts()
    states = Counter(verdict.state for verdict in verdicts.values())
    out = {"candidates": len(plan.candidates), "in_pool": states[IN], "out": states[OUT], "open": states[OPEN],
           "out_by_reason": dict(Counter(verdict.reason for verdict in verdicts.values() if verdict.state == OUT)),
           "answers_stored": len(plan.answers), "checks_given_up": len(plan.abandoned),
           "next_checks": len(plan.next_checks()),
           "certificates_at_most": sum(len(checks) for checks in plan.checks.values()) + len(plan.statement_checks),
           "renaming_pass_built": plan.steps_store.is_done(RENAMED_STEP), "renamed_certificates": plan.renamed_certificates,
           "renaming_pass_built_for": plan.steps_store.done_summary(RENAMED_STEP).get("scope", WHOLE_POOL) if plan.steps_store.is_done(RENAMED_STEP) else None,
           "in_pool_by_a_renamed_certificate": sum(verdict.state == IN and bool(verdict.certificate.renamed) for verdict in verdicts.values()),
           "heldout_drawn": plan.steps_store.is_done("heldout"), "base_map_drawn": plan.steps_store.is_done("base_map")}
    parts = stored_slice(plan.steps_store)
    if parts is not None:
        out["slice"] = slice_preview(plan, parts)
        out["slice_open"] = sum(verdicts[problem_id].state == OPEN for problem_id in members_of(parts))
    print(json.dumps(out, indent=1))
    return 0


def export(arguments, config: dict) -> int:
    """H and the base-map sample of the pool, into a data directory that travels with the code: a stage on the
    GPU box cannot see this store. Statements and ids only (`data/ladder_export.py`)."""
    settings = settings_of(config)
    steps_store = ArtifactStore(arguments.store / "steps")
    if arguments.out is None:
        raise SystemExit("export needs --out: the data directory to write (src/rlvr_lean/data/ladder_l0)")
    if not steps_store.is_done("pool") or not steps_store.is_done("base_map"):
        raise SystemExit("`pool` has not run: there is no H, no base-map sample and no pool to export")
    present = {row["statement_id"] for row in steps_store.read_rows("present_holdout.jsonl")}
    heldout = [exported_problem(row, heldout_part=row["heldout_part"], in_present_holdout=row["problem_id"] in present)
               for row in steps_store.read_rows("heldout.jsonl")]
    wanted = settings["base_map"]
    sample = [exported_problem(row) for row in steps_store.read_rows("base_map.jsonl")]       # drawn once, by `pool`
    summary = write_export(arguments.out, heldout, sample, {
        "fixture": False, "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "lean_pin": settings["lean_pin"],
        "heldout_settings": settings["heldout"], "base_map_settings": wanted, "pool": steps_store.done_summary("pool")})
    log(f"exported to {arguments.out}: {json.dumps(summary['files'])}"
        + ("" if len(sample) == wanted["problems"] else f"; the base-map sample holds {len(sample)} of the {wanted['problems']} problems wanted"))
    return 0


COMMANDS = {"fetch": fetch, "build": build, "check": check, "sample": sample, "names": names, "slice": slice_first, "rename": rename,
            "pool": pool, "export": export, "status": status}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=list(COMMANDS))
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--present-run-steps", type=Path, help="build: the stored run whose workbook holdout is the present holdout (read only)")
    parser.add_argument("--again", action="store_true", help="build: rebuild even though candidates are stored")
    parser.add_argument("--api-key-file", type=Path, help="check, sample: the pool's key; read, never printed")
    parser.add_argument("--ca-file", type=Path, help="check, sample: the pool authority's certificate")
    parser.add_argument("--in-flight", type=in_flight_argument, default=None,
                        help="requests in flight: a number (default: lean.pool.concurrent_requests), or `auto` to follow the size "
                             "the pool states on /health and on its answers (x lean.pool.size_margin; the default number until "
                             "it states one). check, slice and rename send their checks as background work, which the pool takes "
                             "only when no solver's check is waiting")
    parser.add_argument("--retry-passes", type=int, default=2, help="extra passes over checks that got no answer")
    parser.add_argument("--max-checks", type=int, default=None, help="check: stop after this many checks; sample: refuse a larger sample")
    parser.add_argument("--sizes", type=int, nargs=len(STRATA), default=[80, 80, 80, 80, 80, 60, 30],
                        help=f"sample: problems per stratum, in this order: {' '.join(STRATA)}")
    parser.add_argument("--seed", type=int, default=0, help="sample: the seeded order problems are drawn in")
    parser.add_argument("--report-only", action="store_true", help="sample: send nothing, rebuild the report from what is stored")
    parser.add_argument("--dry-run", action="store_true", help="slice, rename: count from what is stored now; write nothing, send nothing")
    parser.add_argument("--out", type=Path, default=None, help="names: also write the tabulation to this file; export: the directory to write")
    arguments = parser.parse_args()
    config = yaml.safe_load(arguments.config.read_text())
    if arguments.command == "build" and arguments.present_run_steps is None:
        parser.error("build needs --present-run-steps")
    sends = arguments.command in ("check", "sample", "slice", "rename") and not arguments.report_only and not arguments.dry_run
    if sends and (arguments.api_key_file is None or arguments.ca_file is None):
        parser.error(f"{arguments.command} needs --api-key-file and --ca-file")
    if arguments.command == "sample" and arguments.max_checks is None:
        arguments.max_checks = 600
    return COMMANDS[arguments.command](arguments, config)


if __name__ == "__main__":
    sys.exit(main())
