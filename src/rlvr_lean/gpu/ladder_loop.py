"""The ladder loop's GPU steps. Spec: docs/spec/ladder-loop.spec.md, milestone L0 (the GPU half), "An episode",
"Held-out sets", "The reward and the band". Each is run as its own process by `rlvr_lean.runner.entry`.

  ladder_prepare             the problems that travel with the code (H and the base-map sample of the pool):
                             files checked against their recorded hashes, each problem's exact negation built and
                             shown exact by Lean. No GPU.
  ladder_episodes_heldout    the base model's `goal.base_episodes` episodes on every problem of H
  ladder_episodes_base_map   its `base_map.episodes` episodes on the base-map sample
  ladder_l0_report           G, the rungs and the base map: what L0 decides

An episode is one attempt at the statement and one at its exact negation (at most `episode.max_new_tokens` new
tokens and `episode.lean_seconds` of Lean each); it resolves the problem when Lean verifies either. A problem
whose negation is not built or not exact has one side only, and that is recorded. A verified proof on the side
the problem's certificate contradicts is a soundness alarm: the step stops (`SoundnessAlarm`, exit code 3).

The steps run at the ladder loop's own Lean pin (`ladder_loop.lean_pin`), whatever `lean.pin` says, in a run
directory of their own. Work is stored every `episode.block_problems` problems; a rerun resumes at the blocks
that are not done.

Sampling and checking ROLL across blocks (`run_episodes`). The generator samples block after block, in chunks of
`episode.chunk_attempts` attempts, and hands every chunk to Lean as it comes; a block is settled off the
generating thread as soon as its own checks are back, so Lean is never left idle waiting for the next block and
the GPU never waits for a block to be judged. At most `episode.blocks_in_flight` blocks are sampled and not yet
settled: when Lean is the slower side the generator waits. Every check of a step goes through ONE client
(`LeanCheckPool`), so `ladder_loop.lean_in_flight` is the number of requests in flight whatever is waiting.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterator

from rlvr_lean.domain.conjecturing.negation import build_negation_exactness_source, negate_statement
from rlvr_lean.domain.problem_pool.certificates import negation_of
from rlvr_lean.domain.problem_pool.episodes import (
    CAPPED_TOKENS,
    CHECKED,
    NEGATION_NOT_BUILT,
    NEGATION_NOT_EXACT,
    NEGATION_REJECTED,
    NEGATION_SIDE,
    NO_ANSWER,
    NOT_CHECKED,
    STATEMENT_SIDE,
    attempt_id,
    problem_result,
    raise_on_contradiction,
    side_plan,
)
from rlvr_lean.domain.problem_pool.selection import SoundnessAlarm
from rlvr_lean.domain.proving import build_prover_prompt, completion_from_output
from rlvr_lean.domain.verification import VerificationStatus, find_forbidden_token
from rlvr_lean.domain.verification.pin import lean_pin_from_config
from rlvr_lean.gpu import pipeline
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.infrastructure.verification_service import (
    LeanCheckPool,
    ProofAttemptToVerify,
    check_lean_sources,
    following_the_pool,
    kimina_settings_from_config,
    waiting_out_the_proxy_queue,
)

DATA_VARIABLE = "RLVR_LEAN_LADDER_DATA"          # another data directory than the package's (the fixture, a pre-flight)
RUN_VARIABLE = "RLVR_LEAN_LADDER_RUN"            # another run directory than `ladder_l0` (a smoke run never marks the real one)
STAND_IN_VARIABLE = "RLVR_LEAN_STAND_IN_ENGINE"  # a pre-flight without a GPU (`stand_in_engine.py`)
PACKAGE_DATA = Path(__file__).resolve().parents[1] / "data" / "ladder_l0"
DATA_FILES = ("heldout.jsonl", "base_map.jsonl")
HELDOUT, BASE_MAP = "heldout", "base_map"
PREPARE = "ladder_prepare"
REPORT = "ladder_l0_report"
MAXIMUM_EXACTNESS_WITHOUT_AN_ANSWER = 0.01       # above this share the pool is unwell: fail and run again, do not record


def data_directory() -> Path:
    override = os.environ.get(DATA_VARIABLE)
    return Path(override) if override else PACKAGE_DATA


def ladder_config(config: dict) -> dict:
    """The config with the ladder loop's own Lean pin selected."""
    return {**config, "lean": {**(config.get("lean") or {}), "pin": config["ladder_loop"]["lean_pin"]}}


def _store(config: dict) -> ArtifactStore:
    pin = lean_pin_from_config(ladder_config(config))
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    return ArtifactStore(pipeline.STORE / pin.runs_directory / os.environ.get(RUN_VARIABLE, "ladder_l0"), Path(mirror) if mirror else None)


def _lean_settings(config: dict, lean_seconds: int | None = None):
    """The pin's settings with the ladder loop's requests in flight, and an episode's own Lean limit.

    `lean_in_flight` may be above the pool's worker count: the extra requests wait in the proxy's queue. The
    client must then wait for an answer as long as the proxy may hold a request there AND a Lean server may hold
    it for a free worker, or it would give up on (and send again) a request that is still queued.

    `lean_in_flight: auto` follows the size the pool states (lean-pool's README, "Background work and the pool's size"): workers x
    `lean.pool.size_margin`, re-read while the step runs, so a server that joins or drops is followed. Until
    the pool states a size, and against a pool that states none, `lean_in_flight_fallback` is the number."""
    ladder = config["ladder_loop"]
    settings = following_the_pool(kimina_settings_from_config(ladder_config(config)), ladder.get("lean_in_flight"),
                                  fallback=ladder.get("lean_in_flight_fallback"))
    if lean_seconds is not None:
        settings = dataclasses.replace(settings, lean_timeout_seconds=lean_seconds)
    return waiting_out_the_proxy_queue(settings, config)


@contextlib.contextmanager
def lean_sessions(config: dict) -> Iterator[Callable]:
    """A factory of check sessions for an episode step. They share ONE client, so the step's requests in flight
    are `ladder_loop.lean_in_flight` however many blocks are waiting on Lean."""
    pool = LeanCheckPool(_lean_settings(config, lean_seconds=config["ladder_loop"]["episode"]["lean_seconds"]))
    try:
        yield pool.session
    finally:
        pool.close()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_problems(directory: Path) -> tuple[list[dict], list[dict], dict]:
    """(H, the base-map sample, the export's summary) from a data directory, refused unless each file has the
    hash and the row count the summary recorded, ids are unique, and no problem of H is in the base-map sample."""
    summary = json.loads((directory / "summary.json").read_text())
    rows = {}
    for name in DATA_FILES:
        recorded = summary["files"][name]
        found = _sha256(directory / name)
        if found != recorded["sha256"]:
            raise ValueError(f"{directory / name} has SHA-256 {found} and the export recorded {recorded['sha256']}: refused")
        rows[name] = _read_jsonl(directory / name)
        if len(rows[name]) != recorded["rows"]:
            raise ValueError(f"{directory / name} has {len(rows[name])} rows and the export recorded {recorded['rows']}")
    heldout, base_map = rows["heldout.jsonl"], rows["base_map.jsonl"]
    ids = [row["problem_id"] for row in heldout + base_map]
    if len(set(ids)) != len(ids):
        repeated = sorted(key for key, count in Counter(ids).items() if count > 1)
        raise ValueError(f"{len(repeated)} problems appear twice in {directory} (first: {repeated[0]}): H is never in the pool")
    return heldout, base_map, summary


# ------------------------------------------------------------------------------------------- ladder_prepare
def build_problems(heldout: list[dict], base_map: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """(the problems as the episode steps read them, the exactness check of each built negation by problem id).
    A negation that cannot be built or holds a forbidden token is recorded as that problem's one-side reason."""
    problems, exactness = [], {}
    for set_name, rows in ((HELDOUT, heldout), (BASE_MAP, base_map)):
        for row in rows:
            problem = {"problem_id": row["problem_id"], "set": set_name, "kind": row["kind"], "side": row["side"],
                       "heldout_part": row.get("heldout_part"), "statement": row["statement"], "negation": None, "one_side_reason": None}
            try:
                negation = negate_statement(row["statement"], negation_of(row["problem_id"]))
            except ValueError:
                problem["one_side_reason"] = NEGATION_NOT_BUILT
            else:
                if find_forbidden_token(negation) is not None:
                    problem["one_side_reason"] = NEGATION_REJECTED
                else:
                    problem["negation"] = negation
                    exactness[row["problem_id"]] = build_negation_exactness_source(row["statement"])
            problems.append(problem)
    return problems, exactness


def ladder_prepare(config: dict) -> dict:
    store = _store(config)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    directory = data_directory()
    heldout, base_map, data_summary = load_problems(directory)
    problems, exactness = build_problems(heldout, base_map)
    settings = _lean_settings(config)
    raw = check_lean_sources(settings, exactness)
    without_an_answer = sorted(key for key, answer in raw.items() if answer.get("error"))
    if len(without_an_answer) > MAXIMUM_EXACTNESS_WITHOUT_AN_ANSWER * max(1, len(exactness)):
        raise RuntimeError(f"{len(without_an_answer)} of {len(exactness)} exactness checks got no answer from the Lean pool "
                           f"(first: {raw[without_an_answer[0]].get('error')}): nothing was recorded, run the step again")
    for problem in problems:
        if problem["negation"] is not None and not pipeline._compiles(raw[problem["problem_id"]]):
            problem["negation"], problem["one_side_reason"] = None, NEGATION_NOT_EXACT
    store.write_rows("problems.jsonl", problems)
    reasons = Counter(problem["one_side_reason"] for problem in problems if problem["one_side_reason"])
    summary = {"data_directory": str(directory), "fixture": bool(data_summary.get("fixture")), "lean_pin": settings.pin.name,
               "problems": {HELDOUT: len(heldout), BASE_MAP: len(base_map)},
               "by_side": dict(Counter(f"{problem['set']}/{problem['side']}" for problem in problems)),
               "two_sided": sum(problem["negation"] is not None for problem in problems), "one_sided_by_reason": dict(reasons),
               # A known-false problem with one side only can never be resolved: its statement cannot verify.
               "known_false_with_one_side": sum(problem["side"] == "false" and problem["negation"] is None for problem in problems),
               "exactness_checks": len(exactness), "exactness_without_an_answer": len(without_an_answer)}
    store.mark_done(PREPARE, summary)
    return summary


# ------------------------------------------------------------------------------------------------ episodes
def _engine_kit(config: dict) -> tuple[object, Callable]:
    """(the sampling engine, a function (samples, seed, max tokens) -> its sampling parameters)."""
    if os.environ.get(STAND_IN_VARIABLE):
        from rlvr_lean.gpu.stand_in_engine import StandInEngine, stand_in_parameters

        print("STAND-IN ENGINE: no model is loaded; attempts are drawn from fixed proofs (a pre-flight, not a measurement)", flush=True)
        return StandInEngine(), stand_in_parameters
    engine = pipeline._engine(config, enable_lora=False)
    return engine, lambda samples, seed, max_tokens: pipeline._sampling_parameters(config, samples, seed=seed, max_tokens=max_tokens)


def _status(result) -> str:
    """One attempt's status from Lean's verdict. A server error, and a Lean HEADER that timed out (a cold worker
    loading Mathlib, not this proof), are no answer; a proof that ran into the Lean limit is a timeout."""
    if result.status is VerificationStatus.SERVER_ERROR or _header_timed_out(result):
        return NO_ANSWER
    return result.status.value


def _header_timed_out(result) -> bool:
    return result.status is VerificationStatus.TIMEOUT and "header" in result.detail


class LeanPoolRefused(RuntimeError):
    """The pool did not take some checks (its queue timed out, no server was up), again after being asked again.
    Nothing is recorded for the block: a rerun of the step resumes there."""


# What the client's last failed request says when the POOL did not take a check: the proxy answers 503 when a
# request waited out its queue or no server is up, 502 and 504 are gateway failures, and a transport error is the
# pool not answering at all. None of them says anything about the proof. (A crashed worker is HTTP 500: that one
# is usually the proof's doing, is not asked again, and stays "no answer".)
_POOL_REFUSALS = ("HTTP 503", "HTTP 502", "HTTP 504", "transport error")


def _pool_refused(result) -> bool:
    return result.status is VerificationStatus.SERVER_ERROR and any(mark in result.detail for mark in _POOL_REFUSALS)


def chunk_prompts(settings: dict, episodes: int) -> int:
    """Prompts per call to the engine: `episode.chunk_attempts` attempts, whatever the samples per prompt."""
    return max(1, settings.get("chunk_attempts", 512) // episodes)


@dataclasses.dataclass
class SampledBlock:
    """A block that is sampled, with its checks on their way to Lean."""
    problems: list[dict]
    attempts: list[dict]
    sent: dict[str, ProofAttemptToVerify]
    service: object
    generated_tokens: int
    generation_seconds: float
    first_submitted: float                  # time.monotonic() when its first chunk was handed to Lean


def sample_block(engine, parameters_of: Callable, config: dict, problems: list[dict], episodes: int, variant: str,
                 sampling_seed: int, service_factory: Callable, should_stop: Callable[[], bool] | None = None) -> SampledBlock | None:
    """Sample `episodes` episodes for each problem, chunk by chunk, handing each chunk to Lean as it comes. An
    attempt that reached the token cap is recorded as capped and never sent. Returns without waiting for Lean.
    None when `should_stop()` turned true between two chunks: the block is abandoned."""
    settings = config["ladder_loop"]["episode"]
    items = []
    for problem in problems:
        plan = side_plan(problem, settings, sampling_seed)          # which sides are sampled, and which of them are checked
        for side, text in ((STATEMENT_SIDE, problem["statement"]), (NEGATION_SIDE, problem.get("negation"))):
            if side in plan:
                items.append((problem, side, text, plan[side] == CHECKED))
    parameters = parameters_of(episodes, sampling_seed, settings["max_new_tokens"])
    chunk_size = chunk_prompts(settings, episodes)
    service = service_factory()
    attempts, sent, generated_tokens, generation_seconds, first_submitted = [], {}, 0, 0.0, None
    for start in range(0, len(items), chunk_size):
        if should_stop is not None and should_stop():
            return None
        chunk = items[start:start + chunk_size]
        began = time.monotonic()
        outputs = engine.generate([build_prover_prompt(text) for _, _, text, _ in chunk], parameters, lora_request=None)
        generation_seconds += time.monotonic() - began
        to_verify = []
        for (problem, side, text, checked), output in zip(chunk, outputs):
            if len(output.outputs) != episodes:
                raise RuntimeError(f"the engine returned {len(output.outputs)} samples for {problem['problem_id']} ({side}); {episodes} were asked")
            for index, sample in enumerate(output.outputs):
                completion = completion_from_output(sample.text)
                identifier = attempt_id(problem["problem_id"], variant, side, index)
                capped = sample.finish_reason == "length"
                generated_tokens += len(sample.token_ids)
                attempts.append({"attempt_id": identifier, "problem_id": problem["problem_id"], "side": side, "episode": index,
                                 "completion": completion, "token_count": len(sample.token_ids), "finish_reason": sample.finish_reason,
                                 "status": CAPPED_TOKENS if capped else None if checked else NOT_CHECKED, "seconds": None, "first_error": ""})
                if checked and not capped:
                    sent[identifier] = ProofAttemptToVerify(identifier, text, completion)
                    to_verify.append(sent[identifier])
        if first_submitted is None:
            first_submitted = time.monotonic()
        service.submit(to_verify)
    return SampledBlock(problems, attempts, sent, service, generated_tokens, generation_seconds,
                        first_submitted if first_submitted is not None else time.monotonic())


def settle_block(sampled: SampledBlock, config: dict, service_factory: Callable) -> tuple[list[dict], dict]:
    """Wait for a sampled block's checks and give every attempt its status. An attempt whose Lean header timed
    out is asked once more. An attempt the POOL did not take is asked again, up to `episode.pool_refusal_rounds`
    times, and is never recorded as a failed proof or as no answer: if the pool still refuses, `LeanPoolRefused`."""
    settings = config["ladder_loop"]["episode"]
    results = sampled.service.results()

    def ask_again(identifiers: list[str]) -> None:
        again = service_factory()
        again.submit([sampled.sent[identifier] for identifier in identifiers])
        results.update(again.results())

    cold = [identifier for identifier, result in results.items() if _header_timed_out(result)]
    if cold:
        ask_again(cold)
    asked_again = 0
    for _ in range(settings.get("pool_refusal_rounds", 2)):
        refused = [identifier for identifier, result in results.items() if _pool_refused(result)]
        if not refused:
            break
        asked_again += len(refused)
        ask_again(refused)
    refused = [identifier for identifier, result in results.items() if _pool_refused(result)]
    if refused:
        raise LeanPoolRefused(f"the Lean pool did not take {len(refused)} of {len(sampled.sent)} checks of a block, asked again "
                              f"{settings.get('pool_refusal_rounds', 2)} times (first: {results[refused[0]].detail[:200]}): nothing "
                              "was recorded for the block, run the step again")
    checked_seconds = time.monotonic() - sampled.first_submitted
    for attempt in sampled.attempts:
        if attempt["status"] is None:
            result = results[attempt["attempt_id"]]
            attempt["status"] = _status(result)
            attempt["seconds"] = result.verification_seconds
            attempt["first_error"] = next((message for message in result.messages if message.startswith("error")), result.detail)[:300]
    statuses = Counter(attempt["status"] for attempt in sampled.attempts)
    return sampled.attempts, {"problems": len(sampled.problems), "attempts": len(sampled.attempts), "statuses": dict(statuses),
                              "generated_tokens": sampled.generated_tokens, "generation_seconds": round(sampled.generation_seconds, 1),
                              "sent_to_lean": len(sampled.sent), "first_submission_to_last_result_seconds": round(checked_seconds, 1),
                              "header_timeouts_asked_again": len(cold), "pool_refusals_asked_again": asked_again}


def run_block(engine, parameters_of: Callable, config: dict, problems: list[dict], episodes: int, variant: str,
              sampling_seed: int, service_factory: Callable) -> tuple[list[dict], dict]:
    """One block, sampled and then settled (`sample_block`, `settle_block`)."""
    return settle_block(sample_block(engine, parameters_of, config, problems, episodes, variant, sampling_seed, service_factory),
                        config, service_factory)


def episode_results(problems: list[dict], attempts: list[dict], episodes: int, plan_of: Callable | None = None) -> list[dict]:
    """k of n for every problem of a block. The alarm is raised first: nothing is counted on a block in which
    Lean verified a proof on the side a certificate contradicts. `plan_of(problem)` is the sides that were
    sampled for it (default: every side it has)."""
    by_problem: dict[str, list[dict]] = {}
    for attempt in attempts:
        by_problem.setdefault(attempt["problem_id"], []).append(attempt)
    for problem in problems:
        raise_on_contradiction(problem, by_problem.get(problem["problem_id"], []))
    return [problem_result(problem, by_problem.get(problem["problem_id"], []), episodes, plan_of(problem) if plan_of else None)
            for problem in problems]


def run_episodes(config: dict, set_name: str, episodes: int, sampling_seed: int, store: ArtifactStore,
                 engine_kit: Callable, service_factory: Callable) -> dict:
    """Every problem of one set, as a pipeline that rolls across blocks.

    The calling thread samples block after block. A sampled block is settled by a finaliser thread as soon as its
    own checks are back: its attempts are written BEFORE it is judged, so an alarm leaves its evidence, and it is
    marked done only once its results are stored. Only whole blocks are ever marked done, in whatever order they
    finish; a kill loses at most the blocks in flight. At most `episode.blocks_in_flight` blocks are sampled and
    not yet settled: beyond that the generator waits for Lean. A failure in a finaliser (a soundness alarm, a
    pool that refuses) stops the sampling at the next chunk; the blocks already sampled are settled, then the
    failure is raised (an alarm before anything else)."""
    marker = f"episodes_{set_name}"
    if store.is_done(marker):
        store.mirror(f"{marker}_problems.jsonl")      # a rerun is another task: its out/ gets the results too
        return store.done_summary(marker)
    problems = [row for row in store.read_rows("problems.jsonl") if row["set"] == set_name]
    if not problems:
        raise RuntimeError(f"`problems.jsonl` has no problem of the set {set_name}: has {PREPARE} run on the right data directory?")
    episode_settings = config["ladder_loop"]["episode"]
    block_size = episode_settings["block_problems"]
    blocks = [problems[start:start + block_size] for start in range(0, len(problems), block_size)]
    pending = [index for index in range(len(blocks)) if not store.is_done(f"{marker}_block_{index:04d}")]
    in_flight = max(1, episode_settings.get("blocks_in_flight", 3))
    slots, lock, stop = threading.Semaphore(in_flight), threading.Lock(), threading.Event()
    failures: list[tuple[int, BaseException]] = []
    session = {"waited": 0.0, "first_submitted": None, "last_result": None, "sent": 0, "settled": 0}
    started = time.monotonic()

    def finalise(index: int, sampled: SampledBlock) -> None:
        block_marker = f"{marker}_block_{index:04d}"
        try:
            attempts, stats = settle_block(sampled, config, service_factory)
            store.write_rows(f"{marker}_attempts_{index:04d}.jsonl", attempts)
            store.write_rows(f"{marker}_problems_{index:04d}.jsonl", episode_results(
                sampled.problems, attempts, episodes, lambda problem: side_plan(problem, episode_settings, sampling_seed)))
            store.mark_done(block_marker, stats)
            with lock:
                session["settled"] += 1
                session["sent"] += stats["sent_to_lean"]
                session["last_result"] = time.monotonic()
                done = len(blocks) - len(pending) + session["settled"]
            print(f"{block_marker}: {stats['statuses']} ({done} of {len(blocks)} blocks, {round(time.monotonic() - started)} s)", flush=True)
        except BaseException as error:  # noqa: BLE001 - whatever stopped a block stops the step, on the calling thread
            with lock:
                failures.append((index, error))
            stop.set()
        finally:
            slots.release()

    def take_a_slot() -> bool:
        """Wait for room to sample one more block (back-pressure: Lean is the slower side). False once stopped."""
        waiting_since = time.monotonic()
        while not slots.acquire(timeout=0.1):
            if stop.is_set():
                return False
        session["waited"] += time.monotonic() - waiting_since
        if stop.is_set():
            slots.release()
            return False
        return True

    engine, parameters_of, generator_failure = None, None, None
    with ThreadPoolExecutor(max_workers=in_flight, thread_name_prefix="ladder-settle") as finalisers:
        try:
            for index in pending:
                if not take_a_slot():
                    break
                sampled = None
                try:
                    if engine is None:
                        engine, parameters_of = engine_kit(config)
                    sampled = sample_block(engine, parameters_of, config, blocks[index], episodes, set_name, sampling_seed,
                                           service_factory, stop.is_set)
                finally:
                    if sampled is None:                     # it failed, or was stopped between two chunks: the block is abandoned
                        slots.release()
                if sampled is None:
                    break
                with lock:
                    if session["first_submitted"] is None:
                        session["first_submitted"] = sampled.first_submitted
                finalisers.submit(finalise, index, sampled)
        except BaseException as error:  # noqa: BLE001 - the generator itself failed (the engine died): settle what is sampled, then raise
            generator_failure = error
            stop.set()
    del engine
    if failures:
        alarms = [error for _, error in sorted(failures, key=lambda failure: failure[0]) if isinstance(error, SoundnessAlarm)]
        raise alarms[0] if alarms else min(failures, key=lambda failure: failure[0])[1]
    if generator_failure is not None:
        raise generator_failure
    results, totals, statuses = [], Counter(), Counter()
    for index in range(len(blocks)):
        results.extend(store.read_rows(f"{marker}_problems_{index:04d}.jsonl"))
        stats = store.done_summary(f"{marker}_block_{index:04d}")
        statuses.update(stats["statuses"])
        totals.update({key: stats.get(key, 0) for key in ("attempts", "generated_tokens", "generation_seconds", "header_timeouts_asked_again",
                                                          "pool_refusals_asked_again")})
    store.write_rows(f"{marker}_problems.jsonl", results)
    checking = (session["last_result"] - session["first_submitted"]) if session["first_submitted"] is not None and session["last_result"] is not None else 0.0
    summary = {"set": set_name, "problems": len(results), "episodes_each": episodes, "sampling_seed": sampling_seed,
               "max_new_tokens": episode_settings["max_new_tokens"], "lean_seconds": episode_settings["lean_seconds"],
               "resolved_at_least_once": sum(row["resolved"] > 0 for row in results), "blocks": len(blocks),
               # What was done with the side each problem's certificate rules out (`episode.contradicted_side`).
               "contradicted_side_setting": episode_settings.get("contradicted_side", "all"),
               "contradicted_side": dict(Counter(str(row["contradicted_side"]) for row in results)),
               "attempts": totals["attempts"], "statuses": dict(statuses), "generated_tokens": totals["generated_tokens"],
               "generation_seconds": round(totals["generation_seconds"], 1), "header_timeouts_asked_again": totals["header_timeouts_asked_again"],
               "pool_refusals_asked_again": totals["pool_refusals_asked_again"],
               # Where the time went, to show the next bottleneck. The first two are over every block; the rest are
               # this run of the step (a resumed step counts only the blocks it did itself).
               "tokens_per_second": round(totals["generated_tokens"] / totals["generation_seconds"], 1) if totals["generation_seconds"] else None,
               "pipeline": {"chunk_attempts": episode_settings.get("chunk_attempts", 512), "prompts_per_chunk": chunk_prompts(episode_settings, episodes),
                            "blocks_in_flight": in_flight, "lean_in_flight": config["ladder_loop"].get("lean_in_flight"),
                            "blocks_settled_in_this_run": session["settled"], "wall_seconds": round(time.monotonic() - started, 1),
                            "generator_waited_on_lean_seconds": round(session["waited"], 1),
                            "first_submission_to_last_result_seconds": round(checking, 1), "sent_to_lean": session["sent"],
                            "checks_per_second": round(session["sent"] / checking, 2) if checking > 0 else None},
               "stand_in_engine": bool(os.environ.get(STAND_IN_VARIABLE))}
    store.mark_done(marker, summary)
    return summary


def _episodes_step(config: dict, set_name: str, episodes: int, sampling_seed: int) -> dict:
    with lean_sessions(config) as sessions:
        return run_episodes(config, set_name, episodes, sampling_seed, _store(config), _engine_kit, sessions)


def ladder_episodes_heldout(config: dict) -> dict:
    goal = config["ladder_loop"]["goal"]
    return _episodes_step(config, HELDOUT, goal["base_episodes"], goal["sampling_seed"])


def ladder_episodes_base_map(config: dict) -> dict:
    base_map = config["ladder_loop"]["base_map"]
    return _episodes_step(config, BASE_MAP, base_map["episodes"], base_map["sampling_seed"])


# -------------------------------------------------------------------------------------------------- report
def ladder_l0_report(config: dict) -> dict:
    from rlvr_lean.reporting.ladder_l0 import build_l0_report

    store = _store(config)
    _, _, data_summary = load_problems(data_directory())
    report = build_l0_report(store.read_rows(f"episodes_{HELDOUT}_problems.jsonl"), store.read_rows(f"episodes_{BASE_MAP}_problems.jsonl"),
                             data_summary, config["ladder_loop"],
                             {HELDOUT: store.done_summary(f"episodes_{HELDOUT}"), BASE_MAP: store.done_summary(f"episodes_{BASE_MAP}"),
                              PREPARE: store.done_summary(PREPARE)})
    store.path("report_ladder_l0.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    if mirror:
        Path(mirror, "report_ladder_l0.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    store.mark_done(REPORT, {"headline": report["headline"]})
    return report


STEPS = {
    "ladder_prepare": ladder_prepare,
    "ladder_episodes_heldout": ladder_episodes_heldout,
    "ladder_episodes_base_map": ladder_episodes_base_map,
    "ladder_l0_report": ladder_l0_report,
}
