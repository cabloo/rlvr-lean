"""The version-tax tool (the OEIS Open spec, O2a item 9e) against a stand-in for the pool: what
it sends, that it resumes, that its load is bounded, and that a pool failure is "no answer", never a Lean
failure. The report's arithmetic is tested in test_version_tax.py."""

import asyncio
import json

import pytest

pytest.importorskip("httpx", reason="rlvr_lean's client dependency; see pyproject.toml")
pytest.importorskip("yaml")

from rlvr_lean.domain.verification import LEAN_PINS  # noqa: E402
from rlvr_lean.infrastructure.kimina_client import KiminaRequestError  # noqa: E402
from rlvr_lean.infrastructure.verification_service import LeanCheckSettings  # noqa: E402
from rlvr_lean.tools import version_tax  # noqa: E402

PIN = LEAN_PINS["v4.27"]
STATEMENTS = {
    "seed": [("w_seed", "theorem w_seed : True := by\n")],
    "reward": [("w_ok", "theorem w_ok (x : ℕ) : x = x := by\n"), ("w_gone", "theorem w_gone : ∑ i in s, i = 0 := by\n")],
    "conjecture": [("c1", "theorem conjecture_c1 : 1 = 1 := by\n")],
}
# (attempt id, statement id, completion, the earlier pin's status)
ATTEMPTS = [
    ("w_ok#base#0", "w_ok", "  rfl\n", "verified"),
    ("w_ok#base#1", "w_ok", "  rfl\n", "verified"),                       # the same text: one check
    ("w_ok#base#2", "w_ok", "  exact old_lemma x\n", "verified"),         # a name the later pin does not have
    ("w_ok#base#3", "w_ok", "  sorry\n", "rejected_lexical"),             # never sent
    ("w_gone#base#0", "w_gone", "  simp\n", "verified"),                  # its statement no longer compiles
    ("c1#base#0", "c1", "  norm_num\n", "lean_error"),
    ("c1#base#1", "c1", "  slow_tactic\n", "timeout"),
]


def write_run(steps):
    steps.mkdir()
    for group, rows in STATEMENTS.items():
        name, key = version_tax.STATEMENT_GROUPS[group]
        (steps / name).write_text("".join(json.dumps({key: statement_id, "statement": statement}) + "\n" for statement_id, statement in rows))
    (steps / "proof_attempts.jsonl").write_text("".join(
        json.dumps({"attempt_id": attempt_id, "statement_id": statement_id, "completion": completion}) + "\n"
        for attempt_id, statement_id, completion, _ in ATTEMPTS))
    (steps / "verification.jsonl").write_text("".join(
        json.dumps({"attempt_id": attempt_id, "status": status, "seconds": 1.0, "first_error": "error: x" if status == "lean_error" else ""}) + "\n"
        for attempt_id, _, _, status in ATTEMPTS))
    return steps


def lean_reply(code):
    """What the later pin's server says to one file."""
    name = code.split("theorem ", 1)[1].split()[0]
    if "∑ i in" in code:
        messages = [{"severity": "error", "data": "unexpected token 'in'; expected ','"},
                    {"severity": "error", "data": f"Unknown constant `{name}`"}]
    elif "old_lemma" in code:
        messages = [{"severity": "error", "data": "Unknown identifier `old_lemma`"}]
    elif "slow_tactic" in code:
        return {"time": 120.0, "error": "Lean REPL command timed out in 120 seconds"}
    elif "sorry" in code:
        messages = [{"severity": "warning", "data": "declaration uses 'sorry'"}, {"severity": "info", "data": "rlvr-type-fingerprint 7"}]
    else:
        messages = [{"severity": "info", "data": f"'{name}' depends on axioms: [propext]"}]
    return {"time": 0.5, "response": {"env": 0, "messages": messages}}


class PoolStandIn:
    """Stands in for the client. `failing` holds texts whose next answers are a pool failure."""
    sent: list = []
    failing: dict = {}
    crashing: tuple = ()               # texts whose file kills the Lean worker, every time
    in_flight = 0
    most_in_flight = 0

    def __init__(self, settings, on_request_failure=None):
        self.on_request_failure = on_request_failure

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exception_info):
        return None

    async def is_healthy(self):
        return True

    async def check(self, snippets):
        cls = type(self)
        assert len(snippets) == 1                                        # one file per request
        cls.in_flight += 1
        cls.most_in_flight = max(cls.most_in_flight, cls.in_flight)
        await asyncio.sleep(0.001)
        cls.in_flight -= 1
        snippet = snippets[0]
        cls.sent.append(snippet.code)
        for text, times in cls.failing.items():
            if text in snippet.code and times > 0:
                cls.failing[text] = times - 1
                self.on_request_failure(KiminaRequestError("HTTP 503: no server available", status_code=503))
                return [{"id": snippet.snippet_id, "time": None, "error": "server_error: no answer after 4 attempts: HTTP 503"}]
        if any(text in snippet.code for text in cls.crashing):
            self.on_request_failure(KiminaRequestError('HTTP 500: {"detail":"JSON decode error"}', worker_crashed=True, status_code=500))
            return [{"id": snippet.snippet_id, "time": None,
                     "error": 'server_error: no answer after 2 attempts: HTTP 500: {"detail":"JSON decode error"}'}]
        return [{"id": snippet.snippet_id, **lean_reply(snippet.code)}]


@pytest.fixture
def tool(monkeypatch, tmp_path):
    monkeypatch.setattr(version_tax, "KiminaVerifier", PoolStandIn)
    PoolStandIn.sent, PoolStandIn.failing, PoolStandIn.in_flight, PoolStandIn.most_in_flight = [], {}, 0, 0
    PoolStandIn.crashing = ()
    run = version_tax.StoredRun(write_run(tmp_path / "steps"))
    out = tmp_path / "out"
    out.mkdir()
    settings = LeanCheckSettings(base_url="https://pool.test", concurrent_requests=2, lean_timeout_seconds=120, pin=PIN)
    summary = {"earlier_lean_timeout_seconds": 60}

    def check(limit=None, retry_passes=2):
        _, _, jobs = version_tax.plan(run, PIN, limit)
        done, abandoned = version_tax.answered(out / "checks.jsonl"), version_tax.given_up(out / "checks.jsonl")
        todo = [job for job in jobs if job.sha not in done and job.sha not in abandoned]
        session = asyncio.run(version_tax.run_checks(settings, todo, out, retry_passes))
        return session, version_tax.build_report(run, summary, PIN, limit, out)

    return run, out, check


def test_identical_files_are_one_check_and_a_lexically_rejected_proof_is_never_sent(tool):
    run, _, check = tool
    session, report = check()
    assert session["checks_wanted"] == session["answered"] == 9          # 4 statements + 5 distinct proof files
    assert len(PoolStandIn.sent) == 9 and not any("sorry\n\n#print" in code for code in PoolStandIn.sent)
    assert sum("  rfl\n" in code for code in PoolStandIn.sent) == 1
    assert all(code.startswith("import Mathlib\nimport Aesop\n") for code in PoolStandIn.sent)       # the pipeline's header
    assert report["scope"]["attempts_rejected_lexically"] == 1 and report["complete"]


def test_load_is_bounded_by_the_pins_requests_in_flight(tool):
    _, _, check = tool
    check()
    assert PoolStandIn.most_in_flight == 2


def test_a_rerun_sends_nothing_that_is_answered(tool):
    _, out, check = tool
    check()
    rows_before = (out / "checks.jsonl").read_text()
    session, report = check()
    assert session["checks_wanted"] == 0 and len(PoolStandIn.sent) == 9
    assert (out / "checks.jsonl").read_text() == rows_before and report["complete"]


def test_a_sample_is_a_prefix_of_the_full_order_and_the_full_run_reuses_it(tool):
    run, _, check = tool
    session, report = check(limit=3)
    assert report["scope"]["attempts_in_scope"] == 3 and not report["complete"]
    assert [attempt["attempt_id"] for attempt in version_tax.plan(run, PIN, 3)[0]] == [attempt["attempt_id"] for attempt in run.attempts[:3]]
    sent_by_sample = len(PoolStandIn.sent)
    session, report = check()
    assert session["checks_wanted"] == 9 - sent_by_sample and len(PoolStandIn.sent) == 9 and report["complete"]


def test_the_report_compares_identical_texts_on_statements_that_compile_at_both_pins(tool):
    _, _, check = tool
    _, report = check()
    statements = report["statements"]
    assert (statements["reward"]["compile"], statements["reward"]["do_not_compile"]) == (1, 1)
    assert statements["reward"]["do_not_compile_by_class"] == {"unknown_identifier": 0, "other_error": 1, "timeout": 0}
    assert statements["seed"]["compile"] == 1 and statements["conjecture"]["compile"] == 1
    counts = report["tax"]["attempts"]
    assert counts == {"given": 7, "on_statements_compiling_at_both_pins": 6, "no_answer_earlier": 0, "no_answer_later": 0, "compared": 6}
    overall = report["tax"]["overall"]
    assert (overall["verified_earlier"], overall["verified_later"], overall["gained"]) == (3, 3, 1)      # c1's norm_num is new
    assert overall["lost"] == {"unknown_identifier": 1, "other_error": 0, "timeout": 0, "rejected_lexical": 0}
    assert overall["later_failures"] == {"unknown_identifier": 1, "other_error": 0, "timeout": 1, "rejected_lexical": 1}
    assert report["examples_lost"]["unknown_identifier"][0]["later_error"] == "Unknown identifier `old_lemma`"
    # The proof of the statement that no longer compiles is not a tax, and its echo is not a missing name.
    assert report["sensitivity"]["attempts_on_statements_that_do_not_compile"] == 1
    assert report["sensitivity"]["later_verified_on_statements_that_do_not_compile"] == 0
    assert report["transitions"]["verified -> verified"] == 2


def test_a_pool_failure_is_no_answer_retried_and_logged_never_a_lean_failure(tool):
    _, out, check = tool
    PoolStandIn.failing = {"old_lemma": 1, "norm_num": 99}                # one recovers on the retry pass, one never answers
    session, report = check(retry_passes=1)
    assert session["no_answer_rows"] == 3 and session["failed_requests"] == {"http_503": 3}
    failures = [json.loads(line) for line in (out / "request_failures.jsonl").read_text().splitlines()]
    assert len(failures) == 3 and {failure["kind"] for failure in failures} == {"http_503"} and all(failure["at"] for failure in failures)
    counts = report["tax"]["attempts"]
    assert counts["no_answer_later"] == 1 and counts["compared"] == 5 and not report["complete"]
    assert report["scope"]["attempts_without_a_later_answer"] == 1
    assert report["tax"]["overall"]["lost"]["unknown_identifier"] == 1    # the recovered check counts; the unanswered one is left out
    assert sum(report["tax"]["overall"]["later_failures"].values()) == 3
    assert report["pool"]["failed_requests"]["total"] == 3 and report["pool"]["failed_requests"]["by_kind"] == {"http_503": 3}
    # A later run asks again, and with the pool back the report is complete.
    PoolStandIn.failing = {}
    session, report = check()
    assert session["checks_wanted"] == 1 and report["complete"] and report["tax"]["attempts"]["compared"] == 6


def test_a_file_that_kills_its_worker_is_not_asked_again_in_the_same_run_and_is_given_up_after_two(tool):
    """HTTP 500 is the server's answer when the Lean worker died on the file itself. Asking again kills
    another worker, so it is asked once per run, in at most two runs, and then left as no answer."""
    _, out, check = tool
    PoolStandIn.crashing = ("slow_tactic",)
    session, report = check(retry_passes=2)
    assert sum("slow_tactic" in code for code in PoolStandIn.sent) == 1          # no retry pass for it
    assert session["no_answer_rows"] == 1 and session["failed_requests"] == {"http_500": 1} and not report["complete"]
    session, report = check(retry_passes=2)
    assert session["checks_wanted"] == 1 and sum("slow_tactic" in code for code in PoolStandIn.sent) == 2
    assert report["complete"]                                                    # nothing more can be asked
    assert report["scope"]["unique_checks_that_crashed_their_worker_every_time"] == 1
    assert report["scope"]["attempts_without_a_later_answer_by_earlier_status"] == {"timeout": 1}
    assert report["tax"]["attempts"]["no_answer_later"] == 1 and report["tax"]["overall"]["later_failures"]["timeout"] == 0
    session, _ = check()
    assert session["checks_wanted"] == 0 and sum("slow_tactic" in code for code in PoolStandIn.sent) == 2


def test_files_that_are_not_one_runs_are_refused(tmp_path):
    steps = write_run(tmp_path / "steps")
    (steps / "verification.jsonl").write_text(json.dumps({"attempt_id": "w_ok#base#0", "status": "verified", "seconds": 1, "first_error": ""}) + "\n")
    with pytest.raises(ValueError, match="verdicts cover 1 attempts of 7"):
        version_tax.StoredRun(steps)


def test_the_summary_names_the_read_and_both_pins(tool):
    _, _, check = tool
    _, report = check()
    report["settings"], report["pins"] = {}, {"earlier": "v4.9", "later": "v4.27"}
    text = version_tax.markdown(report)
    assert "# Version tax: v4.9 to v4.27, identical proof texts" in text and "COMPLETE" in text
    assert "small if later/earlier >= 0.8" in text and "| reward | 2 | 1 |" in text
