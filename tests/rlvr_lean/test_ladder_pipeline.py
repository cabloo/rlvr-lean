"""The ladder loop's episode steps as a pipeline that rolls across blocks: chunks sized by attempts, a bounded
look-ahead, blocks settled off the generating thread, one Lean client for a step, and a check the POOL did not take
asked again. The engine and Lean are stand-ins; nothing touches a GPU or the network."""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rlvr_lean.domain.problem_pool import LEAN_WORKBOOK, TRUE_SIDE, SoundnessAlarm
from rlvr_lean.domain.problem_pool.episodes import NEGATION_SIDE, NO_ANSWER
from rlvr_lean.domain.verification import VerificationResult, VerificationStatus
from rlvr_lean.domain.verification.pin import LEAN_PINS
from rlvr_lean.gpu import ladder_loop
from rlvr_lean.gpu.stand_in_engine import stand_in_parameters
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.infrastructure.verification_service import LeanCheckPool, LeanCheckSettings, ProofAttemptToVerify

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
WAIT = 20          # seconds a test waits for something that must happen; a pipeline that hangs fails here


def _config(**episode):
    config = json.loads(json.dumps(CONFIG))
    config["ladder_loop"]["episode"].update({"block_problems": 1, "contradicted_side": "all", **episode})
    return config


def _problem(name):
    return {"problem_id": name, "set": "heldout", "kind": LEAN_WORKBOOK, "side": TRUE_SIDE, "heldout_part": LEAN_WORKBOOK,
            "statement": f"theorem {name} (x : ℝ) : x = x := by\n", "negation": f"theorem negation_of_{name} : ¬ (∀ (x : ℝ), x = x) := by\n",
            "one_side_reason": None}


def _store(tmp_path, names):
    store = ArtifactStore(tmp_path / "run")
    store.write_rows("problems.jsonl", [_problem(name) for name in names])
    return store


class Engine:
    """`generate` with vLLM's shape: every sample of a prompt is the script's text for the theorem it names
    (by default a proof Lean accepts for a statement and one it refuses for a negation)."""

    def __init__(self, script=None, dies_on=None, before_dying=None):
        self.script, self.dies_on, self.before_dying = script or {}, dies_on, before_dying
        self.calls, self.sampled = [], []

    def generate(self, prompts, parameters, lora_request=None):
        names = [prompt.rsplit("theorem ", 1)[1].split()[0] for prompt in prompts]
        if self.dies_on in names:
            if self.before_dying:
                self.before_dying()
            raise RuntimeError("the engine died")
        self.calls.append(names)
        self.sampled.extend(names)
        return [SimpleNamespace(outputs=[SimpleNamespace(text=self.script.get(name, "  nope" if name.startswith("negation_of_") else "  good"),
                                                         token_ids=[0] * 5, finish_reason="stop")
                                         for _ in range(parameters.n)]) for name in names]


def _kit(engine):
    return lambda config: (engine, stand_in_parameters)


def _name(attempt):
    return attempt.statement.split()[1]


class Lean:
    """A check session by proof text: `good` verifies, anything else is a Lean error. `before(name)` runs in
    `results()` for each theorem the session holds, on the thread that settles the block: a test's gate."""

    before = staticmethod(lambda name: None)
    sessions = []

    def __init__(self):
        self.attempts = []
        Lean.sessions.append(self)

    def submit(self, attempts):
        self.attempts.extend(attempts)

    def results(self):
        for name in dict.fromkeys(_name(attempt) for attempt in self.attempts):
            Lean.before(name)
        return {attempt.attempt_id: VerificationResult(attempt.attempt_id, VerificationStatus.VERIFIED if "good" in attempt.completion
                                                       else VerificationStatus.LEAN_ERROR, verification_seconds=0.1,
                                                       messages=() if "good" in attempt.completion else ("error: unsolved goals",))
                for attempt in self.attempts}


@pytest.fixture(autouse=True)
def _fresh():
    Lean.before, Lean.sessions = staticmethod(lambda name: None), []


def _done(store, index):
    return store.is_done(f"episodes_heldout_block_{index:04d}")


def _in_thread(function):
    """Run `function` on a thread; returns (thread, a dict that gets its `result` or `error`)."""
    outcome = {}

    def run():
        try:
            outcome["result"] = function()
        except BaseException as error:  # noqa: BLE001 - handed to the test
            outcome["error"] = error

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


def _until(condition, what):
    deadline = time.monotonic() + WAIT
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.01)


# ------------------------------------------------------------------------------------------------- chunks
def test_a_chunk_is_sized_by_attempts_whatever_the_samples_per_prompt():
    settings = CONFIG["ladder_loop"]["episode"]
    assert settings["chunk_attempts"] == 512
    assert ladder_loop.chunk_prompts(settings, 32) == 16 and ladder_loop.chunk_prompts(settings, 8) == 64
    assert ladder_loop.chunk_prompts(settings, 4096) == 1                   # never less than one prompt
    engine = Engine()
    config = _config(chunk_attempts=8, block_problems=8, contradicted_side_audit_share=0.0, contradicted_side="audit",
                     skip_generating_contradicted_side=True)
    attempts, stats = ladder_loop.run_block(engine, stand_in_parameters, config, [_problem(f"p{i}") for i in range(5)], 4, "heldout", 0, Lean)
    assert [len(call) for call in engine.calls] == [2, 2, 1]                # 8 attempts a call at 4 samples a prompt
    assert len(attempts) == 20 and stats["sent_to_lean"] == 20 and stats["statuses"] == {"verified": 20}
    # Each chunk went to Lean as it was sampled, in one session for the block.
    assert len(Lean.sessions) == 1 and len(Lean.sessions[0].attempts) == 20


# ------------------------------------------------------------------------------------------ back-pressure
def test_the_generator_waits_when_lean_is_the_slower_side(tmp_path):
    names = [f"b{index}" for index in range(6)]
    store, engine, gate = _store(tmp_path, names), Engine(), threading.Event()
    Lean.before = staticmethod(lambda name: gate.wait(WAIT))                # Lean answers nothing until the gate opens
    thread, outcome = _in_thread(lambda: ladder_loop.run_episodes(_config(blocks_in_flight=2), "heldout", 2, 0, store, _kit(engine), Lean))
    _until(lambda: len(engine.sampled) >= 4, "two blocks to be sampled")    # two sides a block
    time.sleep(0.4)
    assert engine.sampled == ["b0", "negation_of_b0", "b1", "negation_of_b1"]      # and no third block: the look-ahead is bounded
    assert not any(_done(store, index) for index in range(6))
    gate.set()
    thread.join(WAIT)
    assert "error" not in outcome, outcome.get("error")
    summary = outcome["result"]
    assert all(_done(store, index) for index in range(6)) and summary["problems"] == 6
    pipeline = summary["pipeline"]
    assert pipeline["generator_waited_on_lean_seconds"] >= 0.3 and pipeline["blocks_in_flight"] == 2
    assert pipeline["blocks_settled_in_this_run"] == 6 and pipeline["sent_to_lean"] == 24 and pipeline["checks_per_second"] > 0
    assert pipeline["first_submission_to_last_result_seconds"] >= 0.3 and summary["tokens_per_second"] is None      # the stand-in takes no time
    assert [row["problem_id"] for row in store.read_rows("episodes_heldout_problems.jsonl")] == names


def test_blocks_may_settle_out_of_order_and_only_whole_blocks_are_marked(tmp_path):
    names = [f"b{index}" for index in range(4)]
    store, engine = _store(tmp_path, names), Engine()
    # Block 0's checks come back only once block 1 has been settled and marked.
    Lean.before = staticmethod(lambda name: _until(lambda: _done(store, 1), "block 1 to settle first") if name == "b0" else None)
    seen = {}
    real = store.mark_done

    def mark_done(phase, summary):
        if phase == "episodes_heldout_block_0001":
            seen["block_0_done_when_block_1_settled"] = _done(store, 0)
        real(phase, summary)

    store.mark_done = mark_done
    summary = ladder_loop.run_episodes(_config(blocks_in_flight=3), "heldout", 2, 0, store, _kit(engine), Lean)
    assert seen == {"block_0_done_when_block_1_settled": False}
    assert [row["problem_id"] for row in store.read_rows("episodes_heldout_problems.jsonl")] == names     # the set's results are in the problems' order
    assert summary["attempts"] == 16 and summary["statuses"] == {"verified": 8, "lean_error": 8}


# -------------------------------------------------------------------------------------------------- alarm
def test_an_alarm_stops_the_sampling_while_later_blocks_are_in_flight_and_leaves_its_evidence(tmp_path):
    names = [f"b{index}" for index in range(6)]
    store = _store(tmp_path, names)
    engine = Engine({"negation_of_b1": "  good"})                           # Lean will verify the negation of a known-true statement
    evidence_file = store.path("episodes_heldout_attempts_0001.jsonl")

    def before(name):
        if name == "b1":                    # block 1's checks come back once blocks 2 and 3 are sampled too: three in flight
            _until(lambda: "b3" in engine.sampled, "blocks 2 and 3 to be in flight")
        elif name in ("b2", "b3"):          # and theirs only after block 1's alarm (its evidence is written just before it is judged)
            _until(evidence_file.exists, "block 1 to be judged")
            time.sleep(0.2)

    Lean.before = staticmethod(before)
    with pytest.raises(SoundnessAlarm, match="b1 is known true"):
        ladder_loop.run_episodes(_config(blocks_in_flight=3), "heldout", 2, 0, store, _kit(engine), Lean)
    assert "b4" not in engine.sampled and "b5" not in engine.sampled         # the sampling stopped
    assert not _done(store, 1) and not store.is_done("episodes_heldout")
    evidence = store.read_rows("episodes_heldout_attempts_0001.jsonl")       # written before it was judged
    assert [attempt["status"] for attempt in evidence if attempt["side"] == NEGATION_SIDE] == ["verified", "verified"]
    assert not store.path("episodes_heldout_problems_0001.jsonl").exists()
    assert _done(store, 0) and _done(store, 2) and _done(store, 3)           # whole blocks already sampled were settled


# ------------------------------------------------------------------------------------------------- resume
def test_a_kill_with_blocks_in_flight_loses_those_blocks_and_a_rerun_does_only_them(tmp_path):
    names = [f"b{index}" for index in range(5)]
    store, killed = _store(tmp_path, names), threading.Event()

    def before(name):
        if name in ("b1", "b2"):            # in flight when the process dies: their checks never come back
            killed.wait(WAIT)
            raise RuntimeError("the process was killed")

    Lean.before = staticmethod(before)
    dying = Engine(dies_on="b3", before_dying=killed.set)
    with pytest.raises(RuntimeError, match="killed|engine died"):
        ladder_loop.run_episodes(_config(blocks_in_flight=3), "heldout", 2, 0, store, _kit(dying), Lean)
    assert _done(store, 0) and not any(_done(store, index) for index in (1, 2, 3, 4)) and not store.is_done("episodes_heldout")
    Lean.before = staticmethod(lambda name: None)
    engine = Engine()
    summary = ladder_loop.run_episodes(_config(blocks_in_flight=3), "heldout", 2, 0, store, _kit(engine), Lean)
    assert [name for name in engine.sampled if not name.startswith("negation_of_")] == ["b1", "b2", "b3", "b4"]       # not block 0
    assert summary["pipeline"]["blocks_settled_in_this_run"] == 4 and summary["problems"] == 5 and summary["blocks"] == 5
    assert [row["problem_id"] for row in store.read_rows("episodes_heldout_problems.jsonl")] == names
    # A finished set is not sampled again: the engine is not even built.
    assert ladder_loop.run_episodes(_config(), "heldout", 2, 0, store, lambda config: pytest.fail("the engine was built"), Lean) == summary


def test_store_writes_of_many_blocks_settling_at_once_do_not_collide(tmp_path):
    names = [f"b{index}" for index in range(40)]
    store = ArtifactStore(tmp_path / "run", tmp_path / "mirror")             # with the mirror a job runner collects
    store.write_rows("problems.jsonl", [_problem(name) for name in names])
    barrier = threading.Barrier(8)

    def together(name):                     # eight blocks' checks come back at the same instant, five times over
        try:
            barrier.wait(2)
        except threading.BrokenBarrierError:
            pass

    Lean.before = staticmethod(together)
    summary = ladder_loop.run_episodes(_config(blocks_in_flight=8), "heldout", 3, 0, store, _kit(Engine()), Lean)
    assert summary["attempts"] == 240 and summary["pipeline"]["blocks_settled_in_this_run"] == 40
    for index in range(40):
        attempts = store.read_rows(f"episodes_heldout_attempts_{index:04d}.jsonl")
        assert len(attempts) == 6 and {attempt["problem_id"] for attempt in attempts} == {names[index]}
        assert (tmp_path / "mirror" / f"episodes_heldout_block_{index:04d}.done.json").exists()
    assert not list((tmp_path / "run").glob("*.tmp"))


# ---------------------------------------------------------------------------- a check the POOL did not take
def _refused(attempt, status=503):
    return VerificationResult(attempt.attempt_id, VerificationStatus.SERVER_ERROR,
                              detail=f"server_error: no answer after 4 attempts: HTTP {status}: no server is available to handle this request")


class RefusingLean(Lean):
    """The pool does not take the checks of `refused` in the first `rounds` sessions that hold them."""

    refused, rounds, status = set(), 1, 503

    def results(self):
        found = super().results()
        earlier = sum(1 for session in Lean.sessions[:Lean.sessions.index(self)] if session.attempts)
        for attempt in self.attempts:
            if _name(attempt) in RefusingLean.refused and earlier < RefusingLean.rounds:
                found[attempt.attempt_id] = _refused(attempt, RefusingLean.status)
        return found


def test_a_check_the_pool_did_not_take_is_asked_again_and_is_never_a_failed_proof(tmp_path):
    RefusingLean.refused, RefusingLean.rounds, RefusingLean.status = {"b0"}, 1, 503
    config = _config(contradicted_side="audit", contradicted_side_audit_share=0.0, skip_generating_contradicted_side=True)
    attempts, stats = ladder_loop.run_block(Engine(), stand_in_parameters, config, [_problem("b0")], 3, "heldout", 0, RefusingLean)
    assert [attempt["status"] for attempt in attempts] == ["verified"] * 3 and stats["pool_refusals_asked_again"] == 3
    assert [len(session.attempts) for session in Lean.sessions] == [3, 3]     # the same three checks, sent a second time


def test_a_pool_that_keeps_refusing_fails_the_block_unrecorded(tmp_path):
    RefusingLean.refused, RefusingLean.rounds, RefusingLean.status = {"b1"}, 99, 503
    store = _store(tmp_path, ["b0", "b1", "b2"])
    config = _config(blocks_in_flight=1, pool_refusal_rounds=2, contradicted_side="audit", contradicted_side_audit_share=0.0,
                     skip_generating_contradicted_side=True)
    with pytest.raises(ladder_loop.LeanPoolRefused, match="did not take 2 of 2 checks"):
        ladder_loop.run_episodes(config, "heldout", 2, 0, store, _kit(Engine()), RefusingLean)
    assert _done(store, 0) and not _done(store, 1) and not store.path("episodes_heldout_attempts_0001.jsonl").exists()
    assert sum(1 for session in Lean.sessions if session.attempts and _name(session.attempts[0]) == "b1") == 3      # asked, and asked again twice
    # The pool is well again: a rerun does the blocks that are left, and nothing was ever recorded as "no answer".
    RefusingLean.refused = set()
    summary = ladder_loop.run_episodes(config, "heldout", 2, 0, store, _kit(Engine()), RefusingLean)
    assert summary["statuses"] == {"verified": 6} and NO_ANSWER not in summary["statuses"]


@pytest.mark.parametrize("detail,again", [
    ("server_error: no answer after 4 attempts: HTTP 503: no server is available", True),          # the proxy's queue timed out, or no server is up
    ("server_error: no answer after 4 attempts: HTTP 504: the Lean servers did not answer in time", True),
    ("server_error: no answer after 4 attempts: transport error: ReadTimeout('')", True),
    ("server_error: no answer after 2 attempts: HTTP 500: worker crashed", False),               # usually the proof's doing: not asked again
])
def test_which_server_errors_are_the_pools_and_which_are_the_proofs(detail, again):
    result = VerificationResult("a", VerificationStatus.SERVER_ERROR, detail=detail)
    assert ladder_loop._pool_refused(result) is again and ladder_loop._status(result) == NO_ANSWER
    lean_timeout = VerificationResult("a", VerificationStatus.TIMEOUT, detail="Lean REPL command timed out in 30 seconds")
    assert not ladder_loop._pool_refused(lean_timeout) and ladder_loop._status(lean_timeout) == "timeout"


def test_a_crashed_worker_is_not_asked_again_and_stays_no_answer():
    RefusingLean.refused, RefusingLean.rounds, RefusingLean.status = {"b0"}, 99, 500
    config = _config(contradicted_side="audit", contradicted_side_audit_share=0.0, skip_generating_contradicted_side=True)
    attempts, stats = ladder_loop.run_block(Engine(), stand_in_parameters, config, [_problem("b0")], 2, "heldout", 0, RefusingLean)
    assert [attempt["status"] for attempt in attempts] == [NO_ANSWER, NO_ANSWER] and stats["pool_refusals_asked_again"] == 0
    assert len(Lean.sessions) == 1


# ------------------------------------------------------------------- requests in flight above the pool's size
def test_the_client_waits_as_long_as_the_proxys_queue_may_hold_a_request(monkeypatch):
    pool = LeanCheckSettings(base_url="https://192.0.2.9:18100", api_key="k", lean_timeout_seconds=120, concurrent_requests=8,
                             ca_file="ca.crt", tls_server_name="lean-pool.example", pin=LEAN_PINS["v4.27"])
    monkeypatch.setattr(ladder_loop, "kimina_settings_from_config", lambda config: pool)
    settings = ladder_loop._lean_settings(CONFIG, lean_seconds=CONFIG["ladder_loop"]["episode"]["lean_seconds"])
    # `auto` as shipped: it starts from the fallback and then follows the size the pool states.
    assert CONFIG["ladder_loop"]["lean_in_flight"] == "auto" and settings.concurrent_requests == CONFIG["ladder_loop"]["lean_in_flight_fallback"] == 40
    queue = CONFIG["lean"]["pool"]["proxy_queue_seconds"] + CONFIG["lean"]["pool"]["server_wait_seconds"]
    assert queue == 750 and settings.server_queue_wait_seconds == 750 and settings.lean_timeout_seconds == 30
    # The proxy's queue, a Lean server's wait for a worker, the header and the body, and a margin.
    assert settings.http_timeout_seconds == 750 + 2 * 30 + 60
    single = LeanCheckSettings(base_url="http://192.0.2.9:18000", api_key="k", pin=LEAN_PINS["v4.9"])
    monkeypatch.setattr(ladder_loop, "kimina_settings_from_config", lambda config: single)
    assert ladder_loop._lean_settings(CONFIG, lean_seconds=30).server_queue_wait_seconds == 120      # no proxy in front: unchanged


# ------------------------------------------------------------------------------ one client for a whole step
def test_one_check_pool_bounds_the_requests_in_flight_across_callers_and_keeps_them_apart():
    httpx = pytest.importorskip("httpx", reason="rlvr_lean's client dependency; see pyproject.toml")
    import asyncio

    from rlvr_lean.infrastructure.kimina_client import KiminaVerifier

    state = {"now": 0, "most": 0, "requests": 0}

    async def handler(request):
        state["now"] += 1
        state["most"] = max(state["most"], state["now"])
        state["requests"] += 1
        await asyncio.sleep(0.002)
        state["now"] -= 1
        body = json.loads(request.content)
        return httpx.Response(200, json={"results": [{"id": snippet["id"], "time": 0.01, "response": {"messages": [
            {"severity": "info", "data": f"'{snippet['id']}' depends on axioms: [propext]"}]}} for snippet in body["snippets"]]})

    settings = LeanCheckSettings(base_url="http://kimina.test", api_key="k", concurrent_requests=7, pin=LEAN_PINS["v4.9"])
    pool = LeanCheckPool(settings, verifier_factory=lambda: KiminaVerifier(
        settings, http_client=httpx.AsyncClient(base_url=settings.base_url, transport=httpx.MockTransport(handler))))
    found, errors = {}, []

    def caller(number):
        try:
            session = pool.session()
            for chunk in range(3):          # three batches each, submitted without waiting
                session.submit([ProofAttemptToVerify(f"c{number}-{chunk}-{index}", f"theorem t{number} : True := by\n", "  trivial\n") for index in range(20)])
            session.submit([ProofAttemptToVerify(f"c{number}-filtered", f"theorem t{number} : True := by\n", "  sorry\n")])
            found[number] = session.results()
        except BaseException as error:  # noqa: BLE001 - handed to the test
            errors.append(error)

    threads = [threading.Thread(target=caller, args=(number,)) for number in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT)
    try:
        assert not errors, errors
        assert state["requests"] == 360 and state["most"] == 7              # never above the bound, and at it: ONE set of slots
        assert pool.checked == 360 and pool.first_submitted is not None and pool.last_answered >= pool.first_submitted
        for number in range(6):             # every caller got its own results, all of them, and no one else's
            assert len(found[number]) == 61 and all(key.startswith(f"c{number}-") for key in found[number])
            assert found[number][f"c{number}-filtered"].status is VerificationStatus.REJECTED_LEXICAL
    finally:
        pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.submit_sources({"late": "theorem late : True := by trivial"})
    pool.close()                            # closing twice is harmless
