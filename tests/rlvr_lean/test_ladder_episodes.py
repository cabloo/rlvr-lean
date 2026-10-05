"""The ladder loop's L0, the GPU half: episodes, k of n, the reward, the band and the rungs, the alarm, the report,
the export. Spec: docs/spec/ladder-loop.spec.md (fixtures 1, 4, 6, 7, 9, 10, 13). The engine and Lean are
scripted stand-ins; nothing touches a GPU or the network."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rlvr_lean.data.ladder_export import PROBLEM_KEYS, base_map_sample, exported_problem, write_export
from rlvr_lean.domain.problem_pool import FALSE_SIDE, LEAN_WORKBOOK, STP_CONJECTURE, TRUE_SIDE, SoundnessAlarm
from rlvr_lean.domain.problem_pool.episodes import (
    ABOVE_BAND,
    BELOW_BAND,
    CAPPED_TOKENS,
    CHECK_AUDIT,
    CHECKED,
    GENERATED_ONLY,
    IN_BAND,
    NEGATION_NOT_BUILT,
    NEGATION_NOT_EXACT,
    NEGATION_SIDE,
    NO_ANSWER,
    NOT_CHECKED,
    STATEMENT_SIDE,
    attempt_id,
    audited,
    band,
    problem_result,
    raise_on_contradiction,
    reward,
    rung,
    side_plan,
)
from rlvr_lean.domain.verification import VerificationResult, VerificationStatus
from rlvr_lean.runner import entry
from rlvr_lean.gpu import ladder_loop
from rlvr_lean.gpu.stand_in_engine import StandInEngine, stand_in_parameters
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.reporting.ladder_l0 import build_l0_report

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
SETTINGS = CONFIG["ladder_loop"]
T, FLOOR = SETTINGS["challenger"]["target_rate"], SETTINGS["challenger"]["band_reward"]


# ------------------------------------------------------------------------------- the reward and the band
def test_the_reward_at_the_default_target_is_the_specs_table():
    """Fixture 4."""
    assert (T, FLOOR) == (0.25, 0.8)
    assert [round(reward(k, 8, 1 / 4), 2) for k in range(9)] == [0, 0.79, 1.00, 0.87, 0.59, 0.31, 0.11, 0.02, 0]
    assert [round(reward(k, 8, 1 / 3), 2) for k in range(9)] == [0, 0.65, 0.95, 0.99, 0.84, 0.59, 0.32, 0.09, 0]


def test_the_reward_is_zero_at_none_and_all_and_one_at_the_target():
    for target in (1 / 4, 1 / 3, 1 / 2):
        for n in (8, 12, 32):
            assert reward(0, n, target) == 0 and reward(n, n, target) == 0
        assert reward(3, 12, 1 / 4) == pytest.approx(1.0) and reward(4, 12, 1 / 3) == pytest.approx(1.0)


def test_at_one_half_the_reward_is_the_symmetric_one():
    for k in range(9):
        assert reward(k, 8, 1 / 2) == pytest.approx(4 * (k / 8) * (1 - k / 8)) == pytest.approx(reward(8 - k, 8, 1 / 2))


def test_a_reward_is_refused_for_counts_that_are_not_k_of_n():
    for k, n, target in ((9, 8, 0.25), (-1, 8, 0.25), (0, 0, 0.25), (1, 8, 0.0), (1, 8, 1.0)):
        with pytest.raises(ValueError):
            reward(k, n, target)


def test_the_band_computed_from_the_target_is_the_specs_interval():
    """Fixture 4: 0.13 to 0.41 at t = 1/4."""
    low, high = band(1 / 4, 0.8)
    assert (round(low, 2), round(high, 2)) == (0.13, 0.41)
    assert low < 1 / 4 < high and reward(2, 8, 1 / 4) >= 0.8 > reward(1, 8, 1 / 4)
    low_half, high_half = band(1 / 2, 0.8)
    assert low_half + high_half == pytest.approx(1.0)                  # symmetric at one half


def test_a_rung_holds_only_problems_with_a_success_and_its_edges_come_from_the_target():
    """Fixture 13: 1 to 4 successes of 32 below the band, 5 to 13 in it, 14 or more above; 2 or 3 of 8 in it."""
    placed = {k: rung(k, 32, T, FLOOR) for k in range(33)}
    assert placed[0] is None
    assert [k for k in placed if placed[k] == BELOW_BAND] == [1, 2, 3, 4]
    assert [k for k in placed if placed[k] == IN_BAND] == list(range(5, 14))
    assert [k for k in placed if placed[k] == ABOVE_BAND] == list(range(14, 33))
    assert [rung(k, 8, T, FLOOR) for k in range(9)] == [None, BELOW_BAND, IN_BAND, IN_BAND] + [ABOVE_BAND] * 5
    assert [k for k in range(1, 33) if rung(k, 32, 1 / 2, FLOOR) == IN_BAND] == list(range(9, 24))      # another t, other edges


# ---------------------------------------------------------------------------------------------- episodes
def _problem(problem_id="p", side=TRUE_SIDE, negation=True, kind=LEAN_WORKBOOK, part=LEAN_WORKBOOK, reason=None):
    return {"problem_id": problem_id, "set": "heldout", "kind": kind, "side": side, "heldout_part": part,
            "statement": f"theorem {problem_id} (x : ℝ) : x = x := by\n",
            "negation": f"theorem negation_of_{problem_id} : ¬ (∀ (x : ℝ), x = x) := by\n" if negation else None, "one_side_reason": reason}


def _attempts(problem, statuses_by_side):
    """Attempts of one problem from {side: [status of episode 0, 1, ...]}."""
    return [{"attempt_id": attempt_id(problem["problem_id"], "v", side, index), "problem_id": problem["problem_id"], "side": side,
             "episode": index, "status": status}
            for side, statuses in statuses_by_side.items() for index, status in enumerate(statuses)]


def test_an_episode_resolves_by_the_statement_or_by_the_negation_and_not_by_neither():
    """Fixture 1."""
    true = _problem("p", TRUE_SIDE)
    by_statement = problem_result(true, _attempts(true, {STATEMENT_SIDE: ["verified", "lean_error", "lean_error", "verified"],
                                                         NEGATION_SIDE: ["lean_error"] * 4}), 4)
    assert (by_statement["resolved"], by_statement["resolved_by_statement"], by_statement["resolved_by_negation"]) == (2, 2, 0)
    false = _problem("q", FALSE_SIDE)
    by_negation = problem_result(false, _attempts(false, {STATEMENT_SIDE: ["lean_error"] * 4,
                                                          NEGATION_SIDE: ["lean_error", "verified", "verified", "verified"]}), 4)
    assert (by_negation["resolved"], by_negation["resolved_by_statement"], by_negation["resolved_by_negation"]) == (3, 0, 3)
    neither = problem_result(true, _attempts(true, {STATEMENT_SIDE: ["lean_error"] * 4, NEGATION_SIDE: ["timeout"] * 4}), 4)
    assert neither["resolved"] == 0 and neither["attempts_timed_out"] == 4 and (neither["episodes"], neither["sides"]) == (4, 2)


def test_k_counts_episodes_not_attempts():
    # Not reachable for a real problem (it would be the alarm); the count itself is per EPISODE.
    problem = _problem()
    result = problem_result(problem, _attempts(problem, {STATEMENT_SIDE: ["verified", "verified"], NEGATION_SIDE: ["verified", "lean_error"]}), 2)
    assert result["resolved"] == 2 and (result["resolved_by_statement"], result["resolved_by_negation"]) == (2, 1)


def test_a_capped_attempt_is_a_failure_and_an_attempt_without_a_verdict_is_counted():
    """Fixture 9: the token cap and the Lean time cap."""
    problem = _problem()
    result = problem_result(problem, _attempts(problem, {STATEMENT_SIDE: [CAPPED_TOKENS, "timeout", NO_ANSWER, "verified"],
                                                         NEGATION_SIDE: [CAPPED_TOKENS, "lean_error", "lean_error", "lean_error"]}), 4)
    assert result["resolved"] == 1
    assert (result["attempts_capped"], result["attempts_timed_out"], result["attempts_without_an_answer"]) == (2, 1, 1)


def test_a_problem_with_one_side_only_is_counted_on_that_side_and_says_why():
    problem = _problem(negation=False, reason=NEGATION_NOT_EXACT)
    result = problem_result(problem, _attempts(problem, {STATEMENT_SIDE: ["verified", "lean_error"]}), 2)
    assert (result["resolved"], result["sides"], result["one_side_reason"]) == (1, 1, NEGATION_NOT_EXACT)


def test_missing_attempts_are_refused_rather_than_counted_as_failures():
    problem = _problem()
    with pytest.raises(ValueError, match="incomplete"):
        problem_result(problem, _attempts(problem, {STATEMENT_SIDE: ["verified", "verified"], NEGATION_SIDE: ["lean_error"]}), 2)
    with pytest.raises(ValueError, match="incomplete"):
        problem_result(problem, _attempts(problem, {STATEMENT_SIDE: ["verified"], NEGATION_SIDE: ["lean_error"]}), 2)


def test_a_verified_proof_on_the_contradicted_side_is_the_alarm():
    """Fixture 10."""
    true, false = _problem("p", TRUE_SIDE), _problem("q", FALSE_SIDE)
    raise_on_contradiction(true, _attempts(true, {STATEMENT_SIDE: ["verified"], NEGATION_SIDE: ["lean_error"]}))
    raise_on_contradiction(false, _attempts(false, {STATEMENT_SIDE: ["lean_error"], NEGATION_SIDE: ["verified"]}))
    with pytest.raises(SoundnessAlarm, match="known true"):
        raise_on_contradiction(true, _attempts(true, {STATEMENT_SIDE: ["lean_error"], NEGATION_SIDE: ["verified"]}))
    with pytest.raises(SoundnessAlarm, match="known false"):
        raise_on_contradiction(false, _attempts(false, {STATEMENT_SIDE: ["verified"], NEGATION_SIDE: ["lean_error"]}))


# ------------------------------------------------------------------------------- the step, with stand-ins
class ScriptedEngine:
    """`generate` with vLLM's shape. Each prompt gets its `n` samples from the script, by the theorem it names."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def generate(self, prompts, parameters, lora_request=None):
        self.calls.append((list(prompts), parameters))
        outputs = []
        for prompt in prompts:
            name = prompt.rsplit("theorem ", 1)[1].split()[0]
            samples = self.script.get(name, ["  nope"])
            outputs.append(SimpleNamespace(outputs=[SimpleNamespace(
                text=samples[index % len(samples)].replace("<cap>", ""), token_ids=[0] * 7,
                finish_reason="length" if "<cap>" in samples[index % len(samples)] else "stop") for index in range(parameters.n)]))
        return outputs


class ScriptedLean:
    """A verification service by proof text: `good` verifies, `slow` runs into the Lean limit, `down` gets no
    answer, `cold` has its HEADER time out the first time it is asked and verifies the second."""

    instances = []

    def __init__(self):
        self.submitted, self.number = [], len(ScriptedLean.instances)
        ScriptedLean.instances.append(self)

    def submit(self, attempts):
        self.submitted.extend(attempts)

    def results(self):
        found = {}
        for attempt in self.submitted:
            text = attempt.completion
            if "good" in text or ("cold" in text and self.number > 0 and "always" not in text):
                found[attempt.attempt_id] = VerificationResult(attempt.attempt_id, VerificationStatus.VERIFIED, verification_seconds=0.5)
            elif "cold" in text:
                found[attempt.attempt_id] = VerificationResult(attempt.attempt_id, VerificationStatus.TIMEOUT,
                                                               detail="Lean REPL header command timed out in 30 seconds")
            elif "slow" in text:
                found[attempt.attempt_id] = VerificationResult(attempt.attempt_id, VerificationStatus.TIMEOUT,
                                                               detail="Lean REPL command timed out in 30 seconds")
            elif "down" in text:
                found[attempt.attempt_id] = VerificationResult(attempt.attempt_id, VerificationStatus.SERVER_ERROR, detail="server_error: no answer")
            else:
                found[attempt.attempt_id] = VerificationResult(attempt.attempt_id, VerificationStatus.LEAN_ERROR, messages=("error: unsolved goals",))
        return found


@pytest.fixture(autouse=True)
def _fresh_lean():
    ScriptedLean.instances = []


def _kit(engine):
    return lambda config: (engine, stand_in_parameters)


def test_a_block_gives_every_problem_one_attempt_per_side_per_episode():
    problems = [_problem("a"), _problem("b", negation=False, reason=NEGATION_NOT_BUILT)]
    engine = ScriptedEngine({"a": ["  good", "  nope"], "negation_of_a": ["  nope"], "b": ["  good"]})
    # Both sides, as for an audited problem (the config in use does not attempt the ruled-out side of the others).
    attempts, stats = ladder_loop.run_block(engine, stand_in_parameters, _episode_config(contradicted_side="all"), problems, 4, "heldout", 100, ScriptedLean)
    assert [(a["problem_id"], a["side"], a["episode"]) for a in attempts] == (
        [("a", STATEMENT_SIDE, i) for i in range(4)] + [("a", NEGATION_SIDE, i) for i in range(4)] + [("b", STATEMENT_SIDE, i) for i in range(4)])
    assert [a["status"] for a in attempts[:4]] == ["verified", "lean_error", "verified", "lean_error"]
    assert len({a["attempt_id"] for a in attempts}) == 12 and stats["attempts"] == 12 and stats["problems"] == 2
    # The prompts are the prover's own, on the statement and on the exact negation; the caps are the episode's.
    prompts, parameters = engine.calls[0]
    assert prompts[1].endswith("theorem negation_of_a : ¬ (∀ (x : ℝ), x = x) := by\n") and prompts[0].startswith("Complete the following Lean 4 code:")
    assert (parameters.n, parameters.seed, parameters.max_tokens) == (4, 100, SETTINGS["episode"]["max_new_tokens"])
    results = ladder_loop.episode_results(problems, attempts, 4)
    assert [(r["problem_id"], r["resolved"], r["sides"]) for r in results] == [("a", 2, 2), ("b", 4, 1)]


def test_an_attempt_that_reached_the_token_cap_is_never_sent_to_lean():
    """Fixture 9: a capped attempt has failed, whatever Lean would have said of its text."""
    engine = ScriptedEngine({"a": ["  good<cap>", "  good"]})
    attempts, stats = ladder_loop.run_block(engine, stand_in_parameters, CONFIG, [_problem("a", negation=False)], 2, "heldout", 0, ScriptedLean)
    assert [a["status"] for a in attempts] == [CAPPED_TOKENS, "verified"] and attempts[0]["finish_reason"] == "length"
    assert [sent.attempt_id for sent in ScriptedLean.instances[0].submitted] == [attempts[1]["attempt_id"]]
    assert ladder_loop.episode_results([_problem("a", negation=False)], attempts, 2)[0]["resolved"] == 1


def test_a_proof_that_runs_into_the_lean_limit_has_failed_and_no_answer_is_not_a_failure_of_the_proof():
    engine = ScriptedEngine({"a": ["  slow", "  down", "  good"]})
    attempts, stats = ladder_loop.run_block(engine, stand_in_parameters, CONFIG, [_problem("a", negation=False)], 3, "heldout", 0, ScriptedLean)
    assert [a["status"] for a in attempts] == ["timeout", NO_ANSWER, "verified"]
    assert stats["statuses"] == {"timeout": 1, NO_ANSWER: 1, "verified": 1} and stats["header_timeouts_asked_again"] == 0


def test_a_header_that_timed_out_is_asked_once_more():
    # A cold worker loading Mathlib is not the proof's failure: asked again once; still no verdict, then no answer.
    engine = ScriptedEngine({"a": ["  cold", "  cold always", "  nope"]})
    attempts, stats = ladder_loop.run_block(engine, stand_in_parameters, CONFIG, [_problem("a", negation=False)], 3, "heldout", 0, ScriptedLean)
    assert [a["status"] for a in attempts] == ["verified", NO_ANSWER, "lean_error"] and stats["header_timeouts_asked_again"] == 2
    assert len(ScriptedLean.instances) == 2 and len(ScriptedLean.instances[1].submitted) == 2


def test_the_negation_is_built_exact_or_the_problem_has_one_side():
    rows = [{"problem_id": "ok", "kind": LEAN_WORKBOOK, "side": TRUE_SIDE, "statement": "theorem ok (x : ℝ) (h : 0 < x) : x ≠ 0 := by\n", "heldout_part": LEAN_WORKBOOK},
            {"problem_id": "odd", "kind": LEAN_WORKBOOK, "side": TRUE_SIDE, "statement": "theorem odd ∀ x > 0, x = x := by\n", "heldout_part": LEAN_WORKBOOK}]
    problems, exactness = ladder_loop.build_problems(rows, [{"problem_id": "m", "kind": STP_CONJECTURE, "side": FALSE_SIDE, "statement": "theorem m : ∀ a : ℝ, a > 0 := by\n"}])
    by_id = {problem["problem_id"]: problem for problem in problems}
    assert by_id["ok"]["negation"] == "theorem negation_of_ok : ¬ (∀ (x : ℝ) (h : 0 < x), x ≠ 0) := by\n" and by_id["ok"]["set"] == "heldout"
    assert (by_id["odd"]["negation"], by_id["odd"]["one_side_reason"]) == (None, NEGATION_NOT_BUILT)
    assert by_id["m"]["set"] == "base_map" and by_id["m"]["heldout_part"] is None
    assert sorted(exactness) == ["m", "ok"] and "type_of% @ok" in exactness["ok"]


def _prepared(tmp_path, problems):
    store = ArtifactStore(tmp_path / "run")
    store.write_rows("problems.jsonl", problems)
    return store


def test_episodes_are_stored_block_by_block_and_a_rerun_resumes(tmp_path):
    config = json.loads(json.dumps(CONFIG))
    config["ladder_loop"]["episode"]["block_problems"] = 2
    problems = [_problem(name, negation=False) for name in "abcde"]
    store = _prepared(tmp_path, problems)
    engine = ScriptedEngine({"a": ["  good"], "c": ["  good", "  nope"], "e": ["  nope"]})
    # The third block's engine call dies: the first two blocks are stored and marked.
    calls = {"count": 0}

    def dying_kit(config):
        class Dying(ScriptedEngine):
            def generate(self, prompts, parameters, lora_request=None):
                calls["count"] += 1
                if calls["count"] == 3:
                    raise RuntimeError("the engine died")
                return ScriptedEngine.generate(self, prompts, parameters, lora_request)
        return Dying(engine.script), stand_in_parameters

    with pytest.raises(RuntimeError, match="engine died"):
        ladder_loop.run_episodes(config, "heldout", 2, 0, store, dying_kit, ScriptedLean)
    assert store.is_done("episodes_heldout_block_0000") and store.is_done("episodes_heldout_block_0001")
    assert not store.is_done("episodes_heldout_block_0002") and not store.is_done("episodes_heldout")
    summary = ladder_loop.run_episodes(config, "heldout", 2, 0, store, _kit(engine), ScriptedLean)
    assert len(engine.calls) == 1 and [prompt.rsplit("theorem ", 1)[1].split()[0] for prompt in engine.calls[0][0]] == ["e"]      # only the block not done
    results = store.read_rows("episodes_heldout_problems.jsonl")
    assert [(row["problem_id"], row["resolved"]) for row in results] == [("a", 2), ("b", 0), ("c", 1), ("d", 0), ("e", 0)]
    assert (summary["problems"], summary["episodes_each"], summary["blocks"], summary["attempts"], summary["resolved_at_least_once"]) == (5, 2, 3, 10, 2)
    assert summary["statuses"] == {"verified": 3, "lean_error": 7} and summary["stand_in_engine"] is False
    # A finished set is not sampled again: the engine is not even built.
    assert ladder_loop.run_episodes(config, "heldout", 2, 0, store, lambda config: pytest.fail("the engine was built"), ScriptedLean) == summary
    # A rerun is ANOTHER task with an output directory of its own (the first run of L0's GPU half failed after
    # both attempt steps and delivered no step files): the finished set's results reach the new mirror too.
    later = ArtifactStore(tmp_path / "run", tmp_path / "the-rerun-s-out")
    assert ladder_loop.run_episodes(config, "heldout", 2, 0, later, lambda config: pytest.fail("the engine was built"), ScriptedLean) == summary
    assert (tmp_path / "the-rerun-s-out" / "episodes_heldout_problems.jsonl").read_text() == store.path("episodes_heldout_problems.jsonl").read_text()
    assert ArtifactStore(tmp_path / "run").mirror("episodes_heldout_problems.jsonl") is False          # no mirror: nothing to do
    assert later.mirror("not-there.jsonl") is False


def test_a_proof_on_the_contradicted_side_stops_the_step_and_leaves_its_evidence(tmp_path):
    """Fixture 10: the round stops. The alarm sits on an AUDITED problem: only those have their ruled-out side
    attempted under the config in use, so here every problem is audited."""
    config = json.loads(json.dumps(CONFIG))
    config["ladder_loop"]["episode"]["block_problems"] = 1
    config["ladder_loop"]["episode"]["contradicted_side_audit_share"] = 1.0
    store = _prepared(tmp_path, [_problem("a"), _problem("b")])
    engine = ScriptedEngine({"a": ["  good"], "negation_of_a": ["  nope"], "b": ["  nope"], "negation_of_b": ["  nope", "  good"]})
    with pytest.raises(SoundnessAlarm, match="b is known true"):
        ladder_loop.run_episodes(config, "heldout", 2, 0, store, _kit(engine), ScriptedLean)
    assert store.is_done("episodes_heldout_block_0000") and not store.is_done("episodes_heldout_block_0001")
    evidence = store.read_rows("episodes_heldout_attempts_0001.jsonl")                 # written before it was judged
    assert [a["status"] for a in evidence if a["side"] == NEGATION_SIDE] == ["lean_error", "verified"]
    assert not store.path("episodes_heldout_problems_0001.jsonl").exists() and not store.is_done("episodes_heldout")


# ------------------------------------------- the side the certificate rules out (decided 2026-10-04: not attempted until L3)
def _episode_config(**episode):
    config = json.loads(json.dumps(CONFIG))
    config["ladder_loop"]["episode"].update(episode)
    return config


def test_the_config_in_use_does_not_attempt_the_ruled_out_side_and_all_checks_both():
    episode = SETTINGS["episode"]
    assert (episode["contradicted_side"], episode["skip_generating_contradicted_side"], episode["contradicted_side_audit_share"]) == (CHECK_AUDIT, True, 0.02)
    assert not audited("p", 0, 0.02) and not audited("q", 0, 0.02)                  # two problems outside the audit share
    assert side_plan(_problem("p", TRUE_SIDE), episode, 0) == {STATEMENT_SIDE: CHECKED}
    assert side_plan(_problem("q", FALSE_SIDE), episode, 0) == {NEGATION_SIDE: CHECKED}
    episode = {**episode, "contradicted_side": "all"}                               # the setting an audited problem is run with
    assert side_plan(_problem("p", TRUE_SIDE), episode, 0) == {STATEMENT_SIDE: CHECKED, NEGATION_SIDE: CHECKED}
    assert side_plan(_problem("q", FALSE_SIDE), episode, 0) == {STATEMENT_SIDE: CHECKED, NEGATION_SIDE: CHECKED}
    assert side_plan(_problem("r", negation=False), episode, 0) == {STATEMENT_SIDE: CHECKED}
    problem = _problem()
    result = problem_result(problem, _attempts(problem, {STATEMENT_SIDE: ["verified"], NEGATION_SIDE: ["lean_error"]}), 1)
    assert result["contradicted_side"] == CHECKED and result["attempts_not_checked"] == 0
    assert problem_result(_problem(negation=False), _attempts(problem, {STATEMENT_SIDE: ["verified"]}), 1)["contradicted_side"] is None


def test_with_the_switch_on_the_contradicted_side_is_checked_only_for_the_audit_share():
    none, everyone = {"contradicted_side": CHECK_AUDIT, "contradicted_side_audit_share": 0.0}, {"contradicted_side": CHECK_AUDIT, "contradicted_side_audit_share": 1.0}
    assert side_plan(_problem("p", TRUE_SIDE), none, 0) == {STATEMENT_SIDE: CHECKED, NEGATION_SIDE: GENERATED_ONLY}
    assert side_plan(_problem("q", FALSE_SIDE), none, 0) == {STATEMENT_SIDE: GENERATED_ONLY, NEGATION_SIDE: CHECKED}      # a known-false problem: its STATEMENT is ruled out
    assert side_plan(_problem("p", TRUE_SIDE), everyone, 0) == {STATEMENT_SIDE: CHECKED, NEGATION_SIDE: CHECKED}
    assert side_plan(_problem("p", TRUE_SIDE), {**none, "skip_generating_contradicted_side": True}, 0) == {STATEMENT_SIDE: CHECKED}
    assert side_plan(_problem("q", FALSE_SIDE), {**none, "skip_generating_contradicted_side": True}, 0) == {NEGATION_SIDE: CHECKED}
    assert side_plan(_problem("p", TRUE_SIDE), {**everyone, "skip_generating_contradicted_side": True}, 0) == {STATEMENT_SIDE: CHECKED, NEGATION_SIDE: CHECKED}
    with pytest.raises(ValueError, match="must be 'all' or 'audit'"):
        side_plan(_problem(), {"contradicted_side": "none"}, 0)


def test_the_audit_share_is_a_seeded_draw_of_problems():
    names = [f"problem_{index}" for index in range(4000)]
    chosen = [name for name in names if audited(name, 100, 0.02)]
    assert 50 <= len(chosen) <= 110 and chosen == [name for name in names if audited(name, 100, 0.02)]      # about 2%, the same every time
    assert chosen != [name for name in names if audited(name, 101, 0.02)]
    assert not any(audited(name, 100, 0.0) for name in names) and all(audited(name, 100, 1.0) for name in names)


def test_an_unaudited_attempt_on_the_contradicted_side_is_neither_sent_nor_counted():
    config = _episode_config(contradicted_side=CHECK_AUDIT, contradicted_side_audit_share=0.0, skip_generating_contradicted_side=False)
    problems = [_problem("a", TRUE_SIDE), _problem("f", FALSE_SIDE)]
    # Texts that the scripted Lean WOULD verify on the ruled-out sides: they are never asked.
    engine = ScriptedEngine({"a": ["  good", "  nope"], "negation_of_a": ["  good"], "f": ["  good"], "negation_of_f": ["  good", "  nope"]})
    attempts, stats = ladder_loop.run_block(engine, stand_in_parameters, config, problems, 2, "heldout", 100, ScriptedLean)
    by_side = {(a["problem_id"], a["side"]): [] for a in attempts}
    for attempt in attempts:
        by_side[(attempt["problem_id"], attempt["side"])].append(attempt["status"])
    assert by_side == {("a", STATEMENT_SIDE): ["verified", "lean_error"], ("a", NEGATION_SIDE): [NOT_CHECKED, NOT_CHECKED],
                       ("f", STATEMENT_SIDE): [NOT_CHECKED, NOT_CHECKED], ("f", NEGATION_SIDE): ["verified", "lean_error"]}
    sent = {attempt.attempt_id for attempt in ScriptedLean.instances[0].submitted}
    assert sent == {a["attempt_id"] for a in attempts if a["status"] != NOT_CHECKED} and len(sent) == 4 and len(engine.calls[0][0]) == 4
    results = ladder_loop.episode_results(problems, attempts, 2, lambda problem: side_plan(problem, config["ladder_loop"]["episode"], 100))
    assert [(r["problem_id"], r["resolved"], r["sides"], r["contradicted_side"], r["attempts_not_checked"]) for r in results] == [
        ("a", 1, 2, GENERATED_ONLY, 2), ("f", 1, 2, GENERATED_ONLY, 2)]
    assert stats["statuses"][NOT_CHECKED] == 4


def test_the_audited_share_is_still_checked_and_still_raises_the_alarm():
    config = _episode_config(contradicted_side=CHECK_AUDIT, contradicted_side_audit_share=1.0)
    problems = [_problem("a", TRUE_SIDE)]
    engine = ScriptedEngine({"a": ["  nope"], "negation_of_a": ["  good"]})
    attempts, _ = ladder_loop.run_block(engine, stand_in_parameters, config, problems, 2, "heldout", 100, ScriptedLean)
    assert [a["status"] for a in attempts if a["side"] == NEGATION_SIDE] == ["verified", "verified"]
    with pytest.raises(SoundnessAlarm):
        ladder_loop.episode_results(problems, attempts, 2, lambda problem: side_plan(problem, config["ladder_loop"]["episode"], 100))


def test_the_second_switch_does_not_even_sample_the_contradicted_side(tmp_path):
    config = _episode_config(contradicted_side=CHECK_AUDIT, contradicted_side_audit_share=0.0, skip_generating_contradicted_side=True, block_problems=8)
    problems = [_problem("a", TRUE_SIDE), _problem("f", FALSE_SIDE), _problem("b", TRUE_SIDE, negation=False, reason=NEGATION_NOT_EXACT)]
    store = _prepared(tmp_path, problems)
    engine = ScriptedEngine({"a": ["  good"], "negation_of_f": ["  good", "  nope"], "b": ["  nope"]})
    summary = ladder_loop.run_episodes(config, "heldout", 2, 100, store, _kit(engine), ScriptedLean)
    prompts = [prompt.rsplit("theorem ", 1)[1].split()[0] for prompt in engine.calls[0][0]]
    assert prompts == ["a", "negation_of_f", "b"]                                  # neither `negation_of_a` nor `f` was sampled
    results = {row["problem_id"]: row for row in store.read_rows("episodes_heldout_problems.jsonl")}
    assert [(results[name]["resolved"], results[name]["sides"], results[name]["contradicted_side"]) for name in ("a", "f", "b")] == [
        (2, 1, "not_generated"), (1, 1, "not_generated"), (0, 1, None)]
    assert summary["attempts"] == 6 and summary["contradicted_side_setting"] == CHECK_AUDIT
    assert summary["contradicted_side"] == {"not_generated": 2, "None": 1}


def test_a_run_that_checks_both_sides_records_that_the_contradicted_side_was_checked(tmp_path):
    store = _prepared(tmp_path, [_problem("a", TRUE_SIDE)])
    engine = ScriptedEngine({"a": ["  good"], "negation_of_a": ["  nope"]})
    summary = ladder_loop.run_episodes(_episode_config(contradicted_side="all"), "heldout", 2, 100, store, _kit(engine), ScriptedLean)
    assert summary["contradicted_side_setting"] == "all" and summary["contradicted_side"] == {CHECKED: 1} and summary["attempts"] == 4
    assert len(ScriptedLean.instances[0].submitted) == 4                           # both sides of both episodes went to Lean


def test_the_step_exits_with_the_alarms_own_code(tmp_path, monkeypatch):
    from rlvr_lean.gpu import __main__ as gpu_main

    def alarmed(config):
        raise SoundnessAlarm("p is known true and its negation verified")

    monkeypatch.setitem(ladder_loop.STEPS, "ladder_episodes_heldout", alarmed)
    monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", "ladder_episodes_heldout", "--out", str(tmp_path / "step.json")])
    assert gpu_main.main() == gpu_main.SOUNDNESS_ALARM_EXIT == entry.SOUNDNESS_ALARM_EXIT == 3
    result = json.loads((tmp_path / "step.json").read_text())
    assert result["ok"] is False and "its negation verified" in result["soundness_alarm"]

    monkeypatch.setitem(ladder_loop.STEPS, "ladder_episodes_heldout", lambda config: 1 / 0)
    assert gpu_main.main() == 1 and "soundness_alarm" not in json.loads((tmp_path / "step.json").read_text())


# ----------------------------------------------------------------------------- prepare, with a scripted Lean
def _data_directory(tmp_path, heldout, base_map, **summary):
    directory = tmp_path / "data"
    write_export(directory, heldout, base_map, {"fixture": True, "pool": {}, **summary})
    return directory


def _row(problem_id, statement, side=TRUE_SIDE, kind=LEAN_WORKBOOK, **extra):
    return {"problem_id": problem_id, "kind": kind, "side": side, "statement": statement, "rewritten": 0, **extra}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    heldout = [_row("h1", "theorem h1 (x : ℝ) : x = x := by\n", heldout_part=LEAN_WORKBOOK),
               _row("h2", "theorem h2 (h : 0 < x) : -x < 0 := by\n", heldout_part=LEAN_WORKBOOK),          # x is never bound: not exact
               _row("h3", "theorem h3 : ∀ a : ℝ, a > 0 := by\n", FALSE_SIDE, heldout_part=LEAN_WORKBOOK)]
    base_map = [_row("p1", "theorem p1 (n : ℕ) : n = n := by\n", kind=STP_CONJECTURE)]
    monkeypatch.setenv(ladder_loop.DATA_VARIABLE, str(_data_directory(tmp_path, heldout, base_map)))
    store = ArtifactStore(tmp_path / "run")
    monkeypatch.setattr(ladder_loop, "_store", lambda config: store)
    monkeypatch.setattr(ladder_loop, "_lean_settings", lambda config, lean_seconds=None: SimpleNamespace(pin=SimpleNamespace(name="the pin")))
    answers = {"calls": 0, "no_answer": set()}

    def check(settings, sources):
        answers["calls"] += 1
        return {key: ({"error": "server_error: no answer"} if key in answers["no_answer"] else
                      {"response": {"messages": [{"severity": "error", "data": "type mismatch"}] if key == "h2" else []}}) for key in sources}

    monkeypatch.setattr(ladder_loop, "check_lean_sources", check)
    return SimpleNamespace(store=store, answers=answers, directory=tmp_path / "data")


def test_prepare_keeps_a_negation_only_when_lean_shows_it_exact(prepared):
    summary = ladder_loop.ladder_prepare(CONFIG)
    by_id = {problem["problem_id"]: problem for problem in prepared.store.read_rows("problems.jsonl")}
    assert by_id["h1"]["negation"] is not None and by_id["h3"]["negation"] is not None
    assert (by_id["h2"]["negation"], by_id["h2"]["one_side_reason"]) == (None, NEGATION_NOT_EXACT)
    assert summary["problems"] == {"heldout": 3, "base_map": 1} and summary["two_sided"] == 3
    assert summary["one_sided_by_reason"] == {NEGATION_NOT_EXACT: 1} and summary["known_false_with_one_side"] == 0 and summary["fixture"] is True
    assert ladder_loop.ladder_prepare(CONFIG) == summary and prepared.answers["calls"] == 1          # done: Lean is not asked again


def test_prepare_records_nothing_when_the_pool_gives_no_answers(prepared):
    prepared.answers["no_answer"] = {"h1", "h3"}
    with pytest.raises(RuntimeError, match="got no answer"):
        ladder_loop.ladder_prepare(CONFIG)
    assert not prepared.store.path("problems.jsonl").exists() and not prepared.store.is_done(ladder_loop.PREPARE)


def test_a_data_directory_that_is_not_what_the_export_recorded_is_refused(prepared):
    heldout, base_map, summary = ladder_loop.load_problems(prepared.directory)
    assert (len(heldout), len(base_map), summary["fixture"]) == (3, 1, True)
    path = prepared.directory / "heldout.jsonl"
    original = path.read_bytes()
    path.write_bytes(original.replace(b"h1 (x", b"h1 (y"))
    with pytest.raises(ValueError, match="refused"):
        ladder_loop.load_problems(prepared.directory)
    path.write_bytes(original)


def test_a_problem_of_h_in_the_base_map_sample_is_refused(tmp_path):
    """Fixture 6, on the GPU box too: H is never in the pool."""
    row = _row("same", "theorem same : 1 = 1 := by\n")
    directory = _data_directory(tmp_path, [{**row, "heldout_part": LEAN_WORKBOOK}], [row])
    with pytest.raises(ValueError, match="H is never in the pool"):
        ladder_loop.load_problems(directory)


def test_the_committed_fixture_is_what_its_summary_recorded():
    heldout, base_map, summary = ladder_loop.load_problems(PACKAGE / "data" / "ladder_l0_fixture")
    assert (len(heldout), len(base_map), summary["fixture"]) == (6, 4, True)
    assert {row["side"] for row in heldout} == {TRUE_SIDE, FALSE_SIDE} and {row["heldout_part"] for row in heldout} == {LEAN_WORKBOOK, STP_CONJECTURE}
    problems, exactness = ladder_loop.build_problems(heldout, base_map)
    assert all(problem["negation"] for problem in problems) and len(exactness) == 10


# ------------------------------------------------------------------------------------------------ the report
def _result(problem_id, resolved, episodes, part=LEAN_WORKBOOK, side=TRUE_SIDE, kind=None, by_negation=0, sides=2, **extra):
    return {"problem_id": problem_id, "kind": kind or part or LEAN_WORKBOOK, "side": side, "heldout_part": part, "episodes": episodes,
            "resolved": resolved, "resolved_by_statement": resolved - by_negation, "resolved_by_negation": by_negation, "sides": sides,
            "one_side_reason": None if sides == 2 else NEGATION_NOT_EXACT, "attempts_capped": 0, "attempts_timed_out": 0,
            "attempts_without_an_answer": 0, **extra}


STEPS = {"heldout": {"stand_in_engine": False}, "base_map": {"stand_in_engine": False}, "ladder_prepare": {}}


def _report(heldout, base_map, minimum=2):
    settings = json.loads(json.dumps(SETTINGS))
    settings["goal"]["rung_minimum"] = minimum
    return build_l0_report(heldout, base_map, {"fixture": False, "pool": {"candidates": 100, "verified": 80}}, settings, STEPS)


def test_the_goal_set_is_the_workbook_problems_the_base_never_resolved():
    heldout = ([_result(f"g{i}", 0, 32) for i in range(3)] + [_result("gf", 0, 32, side=FALSE_SIDE)]
               + [_result("s0", 0, 32, part=STP_CONJECTURE)]                      # a conjecture with no success: in no rung, not in G
               + [_result("low", 3, 32), _result("mid", 9, 32, part=STP_CONJECTURE), _result("high", 30, 32)])
    report = _report(heldout, [])
    assert report["decides"]["goal_set"] == {"problems": 4, "known_true": 3, "known_false": 1, "of_lean_workbook_problems_held_out": 6}
    assert report["heldout"]["stp_conjectures_with_no_success"] == 1
    assert {name: report["decides"]["rungs"][name]["problems"] for name in ("below", "in", "above")} == {"below": 1, "in": 1, "above": 1}
    assert report["rung_edges_in_successes"] == {"below": [1, 4], "in": [5, 13], "above": [14, 32]}
    assert report["band"] == {"low": round(band(T, FLOOR)[0], 4), "high": round(band(T, FLOOR)[1], 4)}
    assert report["decides"]["certificates_surviving_the_pin"]["verified"] == 80


def test_a_rung_below_the_minimum_is_l0s_finding_and_is_named():
    heldout = [_result("low", 2, 32)] + [_result(f"mid{i}", 8, 32) for i in range(3)] + [_result(f"high{i}", 20, 32) for i in range(2)]
    report = _report(heldout, [], minimum=2)
    assert report["decides"]["rungs_below_the_minimum"] == ["below"] and "BELOW THE MINIMUM: below" in report["headline"]
    assert report["decides"]["rungs"]["in"]["at_least_the_minimum"] is True and report["ok"] is True


def test_the_base_map_reads_the_pool_against_the_band():
    base_map = ([_result(f"z{i}", 0, 8, part=None, kind=STP_CONJECTURE) for i in range(4)] + [_result("one", 1, 8, part=None, kind=LEAN_WORKBOOK)]
                + [_result("two", 2, 8, part=None, kind=LEAN_WORKBOOK), _result("three", 3, 8, part=None, kind=STP_CONJECTURE, by_negation=3, side=FALSE_SIDE)]
                + [_result("all", 8, 8, part=None, kind=LEAN_WORKBOOK)])
    report = _report([], base_map)["base_map"]
    assert report["k_histogram"] == {"0": 4, "1": 1, "2": 1, "3": 1, "4": 0, "5": 0, "6": 0, "7": 0, "8": 1}
    assert (report["in_the_band"], report["share_in_the_band"], report["below_the_band"], report["above_the_band"]) == (2, 0.25, 1, 1)
    assert (report["never_resolved"], report["always_resolved"]) == (4, 1)
    assert report["mean_reward_of_a_random_draw"] == round((reward(1, 8, T) + reward(2, 8, T) + reward(3, 8, T)) / 8, 4)
    assert report["share_in_the_band_by_kind"] == {LEAN_WORKBOOK: round(1 / 3, 4), STP_CONJECTURE: 0.2}
    assert report["resolutions"]["share_of_verified_attempts_that_are_negations"] == round(3 / 14, 4)


def test_too_many_attempts_without_an_answer_and_the_report_is_not_to_be_read():
    heldout = [_result("a", 0, 32, attempts_without_an_answer=10), _result("b", 5, 32)]
    report = _report(heldout, [])
    assert report["not_to_be_read"] == ["heldout"] and report["ok"] is False and "NOT TO BE READ" in report["headline"]
    assert _report([_result("a", 0, 32, attempts_without_an_answer=1), _result("b", 5, 32)], [])["ok"] is True


# ------------------------------------------------------------------------------------------------ the export
def test_an_exported_problem_carries_its_statement_and_no_proof():
    """Fixture 7: nothing that reaches the GPU box is a published proof."""
    pool_row = {"problem_id": "wb_1", "kind": LEAN_WORKBOOK, "side": TRUE_SIDE, "statement": "theorem wb_1 : 1 = 1 := by\n", "statement_published": "...",
                "rewritten": 0, "group": "nl_1", "type_fingerprint": 5, "certificate_source": "goedel", "certificate_sha": "ab" * 32,
                "certificate_renamed": ["le_div_iff"]}
    exported = exported_problem(pool_row, heldout_part=LEAN_WORKBOOK)
    assert set(exported) == set(PROBLEM_KEYS) | {"heldout_part"} and not any("certificate" in key or "proof" in key for key in exported)


def test_the_base_map_sample_is_a_seeded_draw_of_the_pool():
    pool = [{"problem_id": f"p{i}"} for i in range(50)]
    first = [row["problem_id"] for row in base_map_sample(pool, 7, 10)]
    assert first == [row["problem_id"] for row in base_map_sample(list(reversed(pool)), 7, 10)] and len(set(first)) == 10
    assert first != [row["problem_id"] for row in base_map_sample(pool, 8, 10)]
    assert len(base_map_sample(pool, 7, 500)) == 50


def test_the_export_writes_what_the_step_reads(tmp_path):
    heldout = [_row("h1", "theorem h1 : ∑ i ∈ Finset.range 3, i = 3 := by\n", heldout_part=LEAN_WORKBOOK, in_present_holdout=True)]
    summary = write_export(tmp_path / "out", heldout, [_row("p1", "theorem p1 : 2 = 2 := by\n")], {"fixture": False, "pool": {"verified": 2}})
    assert summary["files"]["heldout.jsonl"]["rows"] == 1 and len(summary["files"]["base_map.jsonl"]["sha256"]) == 64
    read_heldout, read_map, read_summary = ladder_loop.load_problems(tmp_path / "out")
    assert read_heldout == heldout and read_map[0]["problem_id"] == "p1" and read_summary["pool"] == {"verified": 2}


# ----------------------------------------------------------------------------------------- stage and engine
def test_every_step_of_the_stage_is_a_registered_step_and_the_smoke_run_has_its_own_data_and_directory():
    from rlvr_lean.gpu import milestone2

    registered = {**milestone2.STEPS, **ladder_loop.STEPS}
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l0b"]]
    assert [step for environment, step, _ in steps if environment == "gpu"] == [
        "fix_tokenizers", "ladder_prepare", "ladder_episodes_heldout", "ladder_episodes_base_map", "ladder_l0_report"]
    assert all(step in registered for environment, step, _ in steps if environment == "gpu")
    assert all(options == {} for _, _, options in steps)                           # no parts: a failure ends the stage
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l0b_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in steps]
    variables = entry.child_environment("gpu", "key", smoke[2][2])
    assert variables[ladder_loop.DATA_VARIABLE].endswith("src/rlvr_lean/data/ladder_l0_fixture") and Path(variables[ladder_loop.DATA_VARIABLE]).is_dir()
    assert variables[ladder_loop.RUN_VARIABLE] == "ladder_l0_smoke" and variables["RLVR_LEAN_KIMINA_API_KEY"] == "key"
    assert ladder_loop.DATA_VARIABLE not in entry.child_environment("gpu", "key", steps[2][2])


def test_the_steps_run_at_the_ladder_loops_own_pin_in_a_directory_of_their_own(tmp_path, monkeypatch):
    from rlvr_lean.gpu import pipeline

    monkeypatch.setattr(pipeline, "STORE", tmp_path)
    monkeypatch.delenv(ladder_loop.RUN_VARIABLE, raising=False)
    monkeypatch.delenv("RLVR_LEAN_STEP_DIR", raising=False)
    assert CONFIG["lean"]["pin"] != SETTINGS["lean_pin"]                            # the config's default pin is the other one
    assert ladder_loop.ladder_config(CONFIG)["lean"]["pin"] == SETTINGS["lean_pin"] and CONFIG["lean"]["pin"] != SETTINGS["lean_pin"]
    assert ladder_loop._store(CONFIG).root == tmp_path / "runs-v4.27" / "ladder_l0"
    monkeypatch.setenv(ladder_loop.RUN_VARIABLE, "ladder_l0_smoke")
    assert ladder_loop._store(CONFIG).root == tmp_path / "runs-v4.27" / "ladder_l0_smoke"


def test_the_stand_in_engine_is_deterministic_and_sometimes_runs_into_the_cap():
    engine, parameters = StandInEngine(), stand_in_parameters(32, 100, 1024)
    prompts = [f"theorem t{i} : 1 = 1 := by\n" for i in range(6)]
    first, second = engine.generate(prompts, parameters), engine.generate(prompts, parameters)
    assert [[s.text for s in output.outputs] for output in first] == [[s.text for s in output.outputs] for output in second]
    assert all(len(output.outputs) == 32 for output in first)
    reasons = {sample.finish_reason for output in first for sample in output.outputs}
    assert reasons == {"stop", "length"} and len({sample.text for output in first for sample in output.outputs}) > 5
    other = engine.generate(prompts, stand_in_parameters(32, 101, 1024))
    assert [[s.text for s in output.outputs] for output in other] != [[s.text for s in output.outputs] for output in first]
