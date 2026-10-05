"""The ladder loop's L3a2, the stage end to end on what an L1 run stored: the blind first attempts, the two arms that go
on from each failed one (blind; alternate: a repair step, a whole proof, a repair step, a whole proof), the known
copies that are not sent to Lean, the report; a rerun, a run resumed after an interruption, the refusals, the alarm, the
registration. Spec: docs/spec/ladder-loop.spec.md, "L3a2: one repair step after each fresh failure, then start over
(no training)". The loops are L3a's, and so are the scripted engine and the scripted Lean (`test_ladder_l3a_stage.py`);
nothing touches a GPU or the network. The read is `test_ladder_l3a2.py`."""

import contextlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l3a_stage import STATE_FILE, ScriptedEngine, ScriptedPool  # noqa: E402 - L3a's scripted engine and Lean
from test_ladder_round import KNOWN_FALSE, stage  # noqa: E402, F401 - the L1 fixture stage, with stand-ins

from rlvr_lean.domain.problem_pool import SoundnessAlarm  # noqa: E402
from rlvr_lean.domain.proving import build_prover_prompt  # noqa: E402
from rlvr_lean.domain.repair import BLIND, FIRST  # noqa: E402
from rlvr_lean.domain.repair import alternate as check_rules  # noqa: E402
from rlvr_lean.domain.repair import cut as rules  # noqa: E402
from rlvr_lean.domain.repair.alternate import ALTERNATE, ARMS, BRANCHES, KNOWN_COPY, REPAIR_ARMS  # noqa: E402
from rlvr_lean.domain.repair.read import episode_rows  # noqa: E402
from rlvr_lean.domain.verification.lean_source import build_proof_source  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import ladder_l2, ladder_l3a, ladder_l3a2, ladder_round  # noqa: E402
from rlvr_lean.gpu.stand_in_engine import stand_in_parameters  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
SEED = 1031                                     # round.sampling_seed + 100 x the task's seed + the stage's place, one after L3a's
NOT_SENT_ANYWAY = ("capped_tokens", "rejected_lexical")


@pytest.fixture
def alternating(stage, monkeypatch):  # noqa: F811
    """What L1's prepare step stores on the fixtures (one goal problem, five rung problems), a scripted engine and
    a scripted Lean; twelve first attempts on the goal problem and on each below-band problem, six on the others."""
    for name in (ladder_l3a2.L3A2_RUN_VARIABLE, ladder_l3a2.L3A2_SOURCE_VARIABLE, ladder_l3a2.L3A2_EPISODES_VARIABLE):
        monkeypatch.delenv(name, raising=False)
    stage.config["ladder_loop"]["repair_alternate"].update({"episodes_goal": 12, "episodes_below": 12, "episodes_rungs": 6})
    engine, pool = ScriptedEngine(), ScriptedPool()
    monkeypatch.setattr(ladder_l3a, "engine_kit", lambda: lambda config: (engine, stand_in_parameters))      # the loops are L3a's: so are the engine and
    monkeypatch.setattr(ladder_l3a, "lean_pool", lambda config: contextlib.nullcontext(pool))                # the Lean client they are given
    return SimpleNamespace(config=stage.config, engine=engine, pool=pool, store=lambda: ladder_l3a2._store(stage.config))


def _run(config):
    return {step: ladder_l3a2.STEPS[step](config) for environment, step, _ in map(entry.step_fields, entry.STAGES["ladder_l3a2"])
            if environment == "gpu" and step in ladder_l3a2.STEPS}


def _files(directory):
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir()) if path.is_file()}


def _attempts(store, loops=5):
    return [row for loop in range(loops) for row in store.read_rows(ladder_l3a.attempts_file(loop))]


def test_the_whole_stage_runs_on_what_an_l1_run_stored_and_writes_a_run_of_its_own(alternating):
    config = alternating.config
    with pytest.raises(RuntimeError, match="L3a2 reads the run directory .*stage `ladder_l1` with --seeds 0"):        # nothing to read yet: refused
        ladder_l3a2.ladder_l3a2_prepare(config)
    assert not ladder_l3a2.source_directory(config).exists()                              # and the refusal created nothing
    ladder_round.ladder_l1_prepare(config)
    source = ladder_l3a2.source_directory(config)
    before = _files(source)
    summaries = _run(config)
    assert _files(source) == before                                                        # L1's run directory was only read
    store = alternating.store()
    assert store.root.name == "ladder_l3a2_seed0" and store.root.parent == source.parent and not (source.parent / "ladder_l3a_seed0").exists()
    prepare = summaries["ladder_l3a2_prepare"]
    assert prepare["goal_set"] == 1 and prepare["rungs"] == {"below": 2, "in": 2, "above": 1} and prepare["first_attempts"] == 12 + 2 * 12 + 3 * 6
    assert prepare["sizes"] == {"episodes": {"goal": 12, "below": 12, "in": 6, "above": 6}, "episodes_from": "repair_alternate", "loops": 5,
                                "max_new_tokens": 1024, "lean_seconds": 30, "sampling_seed": SEED}
    problems = {row["problem_id"]: row for row in store.read_rows("problems.jsonl")}
    assert {(row["group"], row["episodes"]) for row in problems.values()} == {("goal", 12), ("below", 12), ("in", 6), ("above", 6)}

    # ---- the first attempts: blind, on the side the certificate allows, shared by the two arms
    rows = _attempts(store)
    firsts = [row for row in rows if row["loop"] == 0]
    assert {row["arm"] for row in firsts} == {FIRST} and all(row["how"] == "blind" and row["kept_lines"] == 0 and row["copy_of"] is None for row in firsts)
    own = {(row["problem_id"], row["episode"]): row for row in firsts if not row["audit"]}
    assert len(own) == prepare["first_attempts"]
    assert all(row["side"] == ("negation" if row["problem_id"] in KNOWN_FALSE else "statement") for row in own.values())
    assert {row["arm"] for row in rows if row["loop"] > 0} == set(ARMS) and not [row for row in rows if row["loop"] > 0 and row["audit"]]

    # ---- every failed first attempt goes on in the two arms, each to its first verified attempt and at most four more
    later, beside = {}, {}
    for row in rows:
        if row["loop"] > 0:
            (beside if row["untrimmed"] else later).setdefault((row["arm"], row["problem_id"], row["episode"]), []).append(row)
    failed = {key for key, row in own.items() if row["status"] != "verified"}
    assert set(later) == {(arm, *key) for arm in ARMS for key in failed} and failed
    trimmed_rows = []
    for (arm, problem_id, episode), chain in later.items():
        assert [row["loop"] for row in chain] == list(range(1, len(chain) + 1)) and len(chain) <= 4
        assert all(row["status"] != "verified" for row in chain[:-1]) and (chain[-1]["status"] == "verified" or len(chain) == 4)
        previous = own[problem_id, episode]
        for row in chain:
            if row["how"] == "trimmed":
                # L3a's item 2a: no goal was left at the cut and the kept lines verified: the repair step resolves the episode with nothing generated.
                assert arm == ALTERNATE and check_rules.is_repair_step(row["loop"]) and row is chain[-1] and row["copy_of"] is None
                assert (row["status"], row["token_count"], row["prompt_tokens"], row["completion"], row["seed"], row["job"]) == ("verified", 0, 0, "", None, None)
                assert row["proof"] == "".join(line + "\n" for line in rules.proof_lines(previous["proof"])[:row["kept_lines"]])
                trimmed_rows.append(row)
                continue
            # Every attempt of one loop of one episode is sampled with the same seed, whatever the arm; none is one of L3a's.
            assert row["seed"] == ladder_l3a.attempt_seed(SEED, problem_id, row["side"], episode, row["loop"])
            if arm == ALTERNATE and check_rules.is_repair_step(row["loop"]):
                # Attempts 2 and 4: ONE repair step, from the attempt just before, which is a fresh one (the first attempt, or attempt 3).
                assert previous["how"] == "blind" and previous["kept_lines"] == 0 and previous["loop"] == row["loop"] - 1
                if row["how"] == "resume":
                    assert row["no_state"] is None and row["had_state"] is True
                    kept = rules.proof_lines(previous["proof"])[:row["kept_lines"]]
                    assert row["proof"] == "".join(line + "\n" for line in kept) + row["completion"] and "tactic state" not in row["proof"]
                    assert previous["cut_line"] == (row["kept_lines"] + 1 if row["cut"] == rules.BODY else None)
                else:
                    assert row["how"] == "blind" and row["no_state"] in (*rules.NO_STATE_REASONS, KNOWN_COPY) and row["had_state"] is False
            else:
                # The blind arm throughout, and attempts 3 and 5 of the alternate arm: a whole proof from the plain prompt.
                assert (row["how"], row["no_state"], row["kept_lines"], row["had_state"]) == ("blind", None, 0, False) and row["proof"] == row["completion"]
            previous = row
    resumed = [row for row in rows if row["how"] == "resume"]
    assert resumed and {row["arm"] for row in resumed} == {ALTERNATE} and {row["loop"] for row in resumed} == {1, 3}
    assert {row["cut"] for row in resumed} == {rules.BODY, rules.WHOLE_PROOF}
    fell_back = Counter(row["no_state"] for row in rows if row["arm"] == ALTERNATE and row["no_state"])
    assert {rules.ERROR_AFTER_THE_SORRY, rules.NO_ERROR_POSITION, rules.NO_GOALS_AT_THE_CUT, KNOWN_COPY} <= set(fell_back)

    # ---- the prompts: a whole proof is asked for with the plain prompt, a repair step with the kept lines and the state
    plain = {build_prover_prompt(ladder_l3a.statement_of(problems[row["problem_id"]], row["side"])) for row in firsts}
    with_state = [prompt for prompt, _ in alternating.engine.calls if prompt not in plain]
    assert with_state and all("/- tactic state:" in prompt and prompt.endswith("-/\n") for prompt in with_state)
    assert len(alternating.engine.calls) == len(set(alternating.engine.calls)) == summaries["ladder_l3a2_attempts"]["generations"]

    # ---- one generation where the two arms ask the same thing: attempts 3 and 5, and a repair step that had no state
    by_place = {}
    for row in rows:
        if row["loop"] > 0 and row["how"] == "blind":
            by_place.setdefault((row["problem_id"], row["episode"], row["loop"]), {})[row["arm"]] = row
    both = {place: pair for place, pair in by_place.items() if len(pair) == 2}
    assert {loop for _, _, loop in both} == {1, 2, 3, 4}
    assert all(pair[BLIND]["job"] == pair[ALTERNATE]["job"] and pair[BLIND]["completion"] == pair[ALTERNATE]["completion"] for pair in both.values())

    # ---- known copies: a proof Lean already rejected in the same arm of the same episode is a failed attempt that is not sent
    chains = [[own[problem_id, episode], *chain] for (_, problem_id, episode), chain in later.items()]
    chains += [[own[problem_id, episode], *[row for row in later[arm, problem_id, episode] if row["how"] != "trimmed"], *chain]       # the reading without trimming
               for (arm, problem_id, episode), chain in beside.items()]
    found = {}
    for chain in chains:
        already = {}
        for row in chain:
            repeated = None if row["status"] in NOT_SENT_ANYWAY else already.get(row["proof"].rstrip())
            assert row["copy_of"] == repeated
            if repeated is None:
                assert row["status"] != KNOWN_COPY
            else:
                assert (row["status"], row["sent_to_lean"], row["seconds"], row["errors"]) == (KNOWN_COPY, False, None, []) and row["token_count"] > 0
                assert chain[repeated]["proof"].rstrip() == row["proof"].rstrip() and f"attempt {repeated + 1}" in row["first_error"]
                found[id(row)] = row            # a row before a trimmed step is in both readings: counted once
            if row["sent_to_lean"] and row["status"] not in ("verified", "no_answer"):
                already.setdefault(row["proof"].rstrip(), row["loop"])
    copies = list(found.values())
    assert len(copies) == sum(row["status"] == KNOWN_COPY for row in rows) and {row["arm"] for row in copies} == set(ARMS) and {row["loop"] for row in copies} == {1, 2, 3, 4} and {row["copy_of"] for row in copies} >= {0, 1, 2}
    # The same generation may be a known copy in one arm and a checked attempt in the other: each arm has its own episode so far.
    by_job = {}
    for row in rows:
        if row["job"]:
            by_job.setdefault(row["job"], []).append(row)
    assert any({KNOWN_COPY} < {row["status"] for row in shared} for shared in by_job.values())
    # Lean was sent each checked generation ONCE, and no generation that was a known copy wherever it was an attempt.
    sent_jobs = {job: shared[0] for job, shared in by_job.items() if any(row["sent_to_lean"] for row in shared)}
    assert all(row["status"] in (KNOWN_COPY, *NOT_SENT_ANYWAY) for job, shared in by_job.items() if job not in sent_jobs for row in shared)
    step = summaries["ladder_l3a2_attempts"]
    attempt_files = Counter(build_proof_source(ladder_l3a.statement_of(problems[row["problem_id"]], row["side"]), row["proof"]) for row in sent_jobs.values())
    state_files = [source for source in alternating.pool.sources if STATE_FILE.search(source)]
    assert len(state_files) == step["state_files_sent_to_lean"] > 0 and step["kept_lines_checks_sent_to_lean"] > 0
    assert len(alternating.pool.sources) == step["lean_checks_sent"] == len(sent_jobs) + len(state_files) + step["kept_lines_checks_sent_to_lean"]
    others = Counter(source for source in alternating.pool.sources if not STATE_FILE.search(source)) - attempt_files       # what is left: the kept-lines checks
    assert sum(others.values()) == step["kept_lines_checks_sent_to_lean"]

    # ---- the state requests: one for every failed attempt a repair step starts from (the first attempt; the alternate arm's attempt 3)
    states = [row for loop in range(4) for row in store.read_rows(ladder_l3a.states_file(loop))]
    assert not store.path(ladder_l3a.states_file(4)).exists() and store.read_rows(ladder_l3a.states_file(1)) == store.read_rows(ladder_l3a.states_file(3)) == []
    assert {(row["arm"], row["after_loop"]) for row in states} == {(FIRST, 0), (ALTERNATE, 2)}
    assert {(row["problem_id"], row["episode"]) for row in states if row["after_loop"] == 0} == failed
    asked = {(row["arm"], row["problem_id"], row["episode"], row["after_loop"]): row for row in states}
    assert all((row["state"] is not None) == (row["outcome"] == "state") and row["requested"] == (row["cut"] is not None) for row in states)
    assert all((row["outcome"] == "trimmed") == (row["trim_status"] == "verified") for row in states)
    # A repair step that would start from a known copy: Lean was not asked about that proof, so there is no state, and the step is a blind attempt.
    from_copies = [row for row in states if row["outcome"] == KNOWN_COPY]
    assert from_copies and all((row["arm"], row["after_loop"], row["requested"], row["state"], row["cut"]) == (ALTERNATE, 2, False, None, None) for row in from_copies)
    third = {(row["problem_id"], row["episode"], row["untrimmed"]): row for row in rows if row["arm"] == ALTERNATE and row["loop"] == 2}
    fourth = [row for row in rows if row["arm"] == ALTERNATE and row["loop"] == 3 and row["how"] != "trimmed"]
    before_it = lambda row: third.get((row["problem_id"], row["episode"], row["untrimmed"])) or third[row["problem_id"], row["episode"], False]      # noqa: E731
    assert fourth and all((row["no_state"] == KNOWN_COPY) == (before_it(row)["status"] == KNOWN_COPY) for row in fourth)
    assert {(row["problem_id"], row["episode"]) for row in from_copies} == {(row["problem_id"], row["episode"]) for row in fourth if row["no_state"] == KNOWN_COPY}

    # ---- trimmed steps at attempt 2 and at attempt 4; and beside each, the step as the blind attempt it would have been, with what followed
    assert {row["loop"] for row in trimmed_rows} == {1, 3} and set(beside) == {(row["arm"], row["problem_id"], row["episode"]) for row in trimmed_rows}
    for row in trimmed_rows:
        chain = beside[row["arm"], row["problem_id"], row["episode"]]
        assert [other["loop"] for other in chain] == list(range(row["loop"], row["loop"] + len(chain))) and chain[-1]["loop"] <= 4
        assert all(other["status"] != "verified" for other in chain[:-1]) and (chain[-1]["status"] == "verified" or chain[-1]["loop"] == 4)
        assert (chain[0]["how"], chain[0]["no_state"], chain[0]["kept_lines"]) == ("blind", rules.NO_GOALS_AT_THE_CUT, 0)
    assert {row["how"] for chain in beside.values() for row in chain} == {"blind", "resume"}             # a later repair step is made in that reading too
    assert not [row for row in rows if row["untrimmed"] and row["arm"] != ALTERNATE]

    # ---- the episodes and the report
    episodes = store.read_rows("repair_episodes.jsonl")
    assert episodes == episode_rows(list(problems.values()), rows, states, 5, ARMS, REPAIR_ARMS) and len(episodes) == prepare["first_attempts"]
    assert all(set(row["arms"]) == set(ARMS) for row in episodes)
    assert step["episodes"] == len(episodes) and step["failed_first_attempts"] == len(failed) and step["attempts"] == len(rows)
    assert step["episodes_resolved"] == {arm: sum(row["arms"][arm]["resolved_at"] is not None for row in episodes) for arm in ARMS}
    assert step["episodes_resolved_by_a_trimmed_loop"] == {ALTERNATE: len(trimmed_rows)}
    assert step["attempts_of_the_reading_without_trimming_only"] == sum(len(chain) for chain in beside.values())
    assert {arm: step["statuses_by_arm"][arm].get(KNOWN_COPY, 0) for arm in ARMS} == {arm: sum(row["arm"] == arm and not row["untrimmed"] for row in copies) for arm in ARMS}
    assert [entry["loop"] for entry in step["loops"]] == [0, 1, 2, 3, 4] and step["loops_done_in_this_run"] == [0, 1, 2, 3, 4]
    assert {row["problem_id"] for row in store.read_rows("repair_problems.jsonl")} == set(problems)
    report = summaries["ladder_l3a2_report"]
    assert report["fixture"] is True and report["stand_in_engine"] is True and report["branch"]["name"] in BRANCHES
    assert report["heldout"]["goal_set"] == 1 and report["primary"]["problems"] == 3 and report["primary"]["episodes"] == 3 * 12
    assert report["headline"].startswith("L3a2 seed 0: ") and "alternate minus blind" in report["headline"]
    budget = report["budget"]["by_arm"]
    for arm in ARMS:
        chains_of = [chain for (name, _, _), chain in later.items() if name == arm]
        assert budget[arm]["attempts"] == sum(len(chain) for chain in chains_of) and budget[arm]["generated_tokens"] == sum(row["token_count"] for chain in chains_of for row in chain)
        assert budget[arm]["known_copies"] == sum(row["status"] == KNOWN_COPY for chain in chains_of for row in chain) > 0
        assert budget[arm]["attempts_sent_to_lean"] == sum(row["sent_to_lean"] and row["how"] != "trimmed" for chain in chains_of for row in chain)
    assert budget[BLIND]["state_requests"] == 0 and budget[BLIND]["lean_checks"] == budget[BLIND]["attempts_sent_to_lean"] and budget[BLIND]["of_which_trimmed"] == 0
    # The alternate arm's Lean checks: its attempts that were sent, one more for each repair step whose failed proof could be cut, and one
    # more where the kept lines were checked on their own; a trimmed step is an attempt with no tokens.
    steps = [row for (name, _, _), chain in later.items() if name == ALTERNATE for row in chain if check_rules.is_repair_step(row["loop"])]
    request_of = lambda row: asked[FIRST if row["loop"] == 1 else ALTERNATE, row["problem_id"], row["episode"], row["loop"] - 1]      # noqa: E731
    requested, kept_lines = sum(request_of(row)["requested"] for row in steps), sum(request_of(row)["trim_status"] is not None for row in steps)
    assert budget[ALTERNATE]["state_requests"] == requested > 0 and budget[ALTERNATE]["kept_lines_checks"] == kept_lines > 0
    assert budget[ALTERNATE]["lean_checks"] == budget[ALTERNATE]["attempts_sent_to_lean"] + requested + kept_lines
    assert budget[ALTERNATE]["of_which_trimmed"] == report["trimmed"]["resolutions"] == len(trimmed_rows) and report["trimmed"]["by_loop"].keys() == {"1", "3"}
    check = report["can_this_run_see_a_win"]["a_state_for_the_failed_attempts_a_repair_step_starts_from"]
    assert (check["repair_steps"], check["with_a_state"]) == (len(steps), sum(row["how"] == "resume" for row in steps)) and check["without_a_state_by_reason"][KNOWN_COPY] > 0
    positions = report["by_position"]["all"]
    assert positions[ALTERNATE]["2"]["attempts"] == positions[BLIND]["2"]["attempts"] == len(failed) and positions[ALTERNATE]["4"]["repair_step"] is True
    assert json.loads(store.path("report_ladder_l3a2.json").read_text())["headline"] == report["headline"]

    # ---- a rerun returns what is stored: nothing is sampled or checked again
    calls, sources = len(alternating.engine.calls), len(alternating.pool.sources)
    again = _run(config)
    assert (len(alternating.engine.calls), len(alternating.pool.sources)) == (calls, sources) and again == summaries
    assert not ladder_round._store(config).is_done("ladder_l3a2_prepare") and _files(source) == before


def test_a_run_that_stopped_resumes_at_the_first_loop_not_done_and_reads_the_same(alternating, monkeypatch, tmp_path):
    config = alternating.config
    ladder_round.ladder_l1_prepare(config)
    ladder_l3a2.ladder_l3a2_prepare(config)
    whole = ladder_l3a.run_loop

    def stops(config, store, problems, own, loop, *rest):
        if loop == 3:
            raise RuntimeError("interrupted")
        return whole(config, store, problems, own, loop, *rest)

    monkeypatch.setattr(ladder_l3a, "run_loop", stops)
    with pytest.raises(RuntimeError, match="interrupted"):
        ladder_l3a2.ladder_l3a2_attempts(config)
    store = alternating.store()
    assert [store.is_done(ladder_l3a.loop_marker(loop, ladder_l3a2.L3A2)) for loop in range(5)] == [True, True, True, False, False]
    assert not store.is_done(ladder_l3a2.ATTEMPTS) and not [path for path in store.root.iterdir() if path.name.startswith("ladder_l3a_")]
    monkeypatch.setattr(ladder_l3a, "run_loop", whole)
    out = tmp_path / "rerun_out" / "steps"                               # the rerun is another task, with an output directory of its own
    monkeypatch.setenv("RLVR_LEAN_STEP_DIR", str(out))
    calls = len(alternating.engine.calls)
    summary = ladder_l3a2.ladder_l3a2_attempts(config)
    # The repair steps of loop 3 start from the state requests and the rejected proofs that loops 0 to 2 STORED.
    assert summary["loops_done_in_this_run"] == [3, 4] and len(alternating.engine.calls) > calls
    assert len(alternating.engine.calls) == len(set(alternating.engine.calls))                           # nothing of loops 0 to 2 was sampled again
    delivered = {path.name for path in out.iterdir()}
    assert {"repair_attempts_0.jsonl", "repair_states_0.jsonl", "repair_states_2.jsonl", "repair_attempts_4.jsonl", "ladder_l3a2_loop_0.done.json",
            "ladder_l3a2_loop_4.done.json", "repair_episodes.jsonl", "repair_problems.jsonl", "ladder_l3a2_attempts.done.json"} <= delivered
    resumed = store.read_rows("repair_episodes.jsonl")
    # The report step hands a rerun the per-episode and per-problem rows and the markers again, not the attempts.
    again = tmp_path / "report_out" / "steps"
    monkeypatch.setenv("RLVR_LEAN_STEP_DIR", str(again))
    ladder_l3a2.ladder_l3a2_report(config)
    handed = {path.name for path in again.iterdir()}
    assert {"report_ladder_l3a2.json", "repair_episodes.jsonl", "repair_problems.jsonl", "problems.jsonl", "heldout_groups.jsonl", "ladder_l3a2_prepare.done.json",
            "ladder_l3a2_loop_3.done.json", "ladder_l3a2_attempts.done.json", "ladder_l3a2_report.done.json"} <= handed
    assert not [name for name in handed if name.startswith(("repair_attempts_", "repair_states_"))]
    # The same run made in one go reads the same.
    monkeypatch.delenv("RLVR_LEAN_STEP_DIR")
    shutil.rmtree(store.root)
    ladder_l3a2.ladder_l3a2_prepare(config)
    ladder_l3a2.ladder_l3a2_attempts(config)
    assert alternating.store().read_rows("repair_episodes.jsonl") == resumed


def test_a_proof_already_rejected_in_an_episode_is_not_sent_for_it_and_gives_its_repair_step_no_state(alternating):
    """One generation, by hand: the alternate arm's attempt 3 and the blind arm's, the same sample. What is done
    with it is decided for each arm by what Lean has already rejected in THAT arm's episode."""
    episode_settings = alternating.config["ladder_loop"]["episode"]
    statement, proof = "theorem demo (x : ℝ) : x = x := by\n", "  step_one\n  bad_step\n  done\n"
    seen = {check_rules.checked_text(proof): 1}

    def settled(blind, alternate, finish="stop"):
        consumers = [{"arm": arm, "problem_id": "demo", "group": "below", "side": "statement", "episode": 0, "audit": False, "untrimmed": False, "how": "blind",
                      "had_state": False, "no_state": None, "kept_lines": 0, "cut": None, "failed_step": None, "rejected": already}
                     for arm, already in ((BLIND, blind), (ALTERNATE, alternate))]
        job = ladder_l3a.Job(build_prover_prompt(statement), 7, statement, (), True, True, consumers)        # a state is wanted: attempt 4 is a repair step
        job.completion, job.finish_reason, job.token_count = proof, finish, 9
        sent = len(alternating.pool.sources)
        stats = ladder_l3a.settle_chunk(alternating.pool, [job], episode_settings, REPAIR_ARMS)
        by_arm = {row["arm"]: row for row in ladder_l3a.attempt_rows(job, 2)}
        return stats, alternating.pool.sources[sent:], by_arm, ladder_l3a.state_rows(job, 2, REPAIR_ARMS)

    # Rejected before in both arms: nothing is sent, and the repair step that would start from it has no state.
    stats, sources, rows, states = settled(seen, {check_rules.checked_text(proof): 0})
    assert stats == {"sent": 0, "state_files": 0, "kept_lines_checks": 0} and sources == []
    assert [(rows[arm]["status"], rows[arm]["copy_of"], rows[arm]["sent_to_lean"], rows[arm]["token_count"]) for arm in ARMS] == [(KNOWN_COPY, 1, False, 9), (KNOWN_COPY, 0, False, 9)]
    assert [(row["arm"], row["outcome"], row["requested"], row["state"]) for row in states] == [(ALTERNATE, KNOWN_COPY, False, None)]
    # Rejected before in the alternate arm only: it is checked for the blind arm, and still not the alternate arm's to cut.
    stats, sources, rows, states = settled({}, seen)
    assert stats == {"sent": 1, "state_files": 0, "kept_lines_checks": 0} and sources == [build_proof_source(statement, proof)]
    assert (rows[BLIND]["status"], rows[BLIND]["copy_of"], rows[BLIND]["sent_to_lean"]) == ("lean_error", None, True) and rows[BLIND]["first_error_line"] == 2
    assert (rows[ALTERNATE]["status"], rows[ALTERNATE]["copy_of"], rows[ALTERNATE]["sent_to_lean"], rows[ALTERNATE]["errors"]) == (KNOWN_COPY, 1, False, [])
    assert [(row["arm"], row["outcome"], row["requested"]) for row in states] == [(ALTERNATE, KNOWN_COPY, False)]
    # Rejected before in the blind arm only: it is checked for the alternate arm, whose repair step gets its state.
    stats, sources, rows, states = settled(seen, {})
    assert stats == {"sent": 1, "state_files": 1, "kept_lines_checks": 0} and len(sources) == 2
    assert (rows[BLIND]["status"], rows[BLIND]["sent_to_lean"]) == (KNOWN_COPY, False) and (rows[ALTERNATE]["status"], rows[ALTERNATE]["copy_of"]) == ("lean_error", None)
    assert [(row["arm"], row["outcome"], row["requested"], row["kept_lines"]) for row in states] == [(ALTERNATE, "state", True, 1)]
    # Rejected in neither: checked once for both.
    stats, sources, rows, states = settled({}, {})
    assert stats == {"sent": 1, "state_files": 1, "kept_lines_checks": 0} and {rows[arm]["status"] for arm in ARMS} == {"lean_error"}
    # An attempt that reached the token cap is not sent in any case, and is counted as that: not as a known copy.
    stats, sources, rows, states = settled(seen, seen, finish="length")
    assert stats["sent"] == 0 and [(rows[arm]["status"], rows[arm]["copy_of"]) for arm in ARMS] == [("capped_tokens", None)] * 2
    assert [(row["outcome"], row["requested"]) for row in states] == [(rules.NO_ERROR_POSITION, False)]


def test_a_run_keeps_the_sizes_and_the_seed_it_was_prepared_with(alternating, monkeypatch):
    config = alternating.config
    ladder_round.ladder_l1_prepare(config)
    ladder_l3a2.ladder_l3a2_prepare(config)
    config["ladder_loop"]["repair_alternate"]["episodes_below"] = 7
    with pytest.raises(RuntimeError, match="a run keeps the sizes it began with"):
        ladder_l3a2.ladder_l3a2_attempts(config)
    with pytest.raises(RuntimeError, match="a run keeps the sizes it began with"):
        ladder_l3a2.ladder_l3a2_prepare(config)
    config["ladder_loop"]["repair_alternate"]["episodes_below"] = 12
    with pytest.raises(RuntimeError, match="needs the step ladder_l3a2_attempts of this run, which is not done: the stage `ladder_l3a2`"):
        ladder_l3a2.ladder_l3a2_report(config)
    # Another seed is another run directory, reading that seed's L1 run, with sampling seeds of its own.
    monkeypatch.setenv("RLVR_LEAN_TRAINING_SEEDS", "2")
    assert ladder_l3a2._store(config).root.name == "ladder_l3a2_seed2" and ladder_l3a2.source_directory(config).name == "ladder_l1_seed2"
    assert ladder_l3a2.sizes(config)["sampling_seed"] == 1231
    monkeypatch.setenv(ladder_l3a2.L3A2_RUN_VARIABLE, "ladder_l3a2_smoke")
    monkeypatch.setenv(ladder_l3a2.L3A2_SOURCE_VARIABLE, "ladder_l1_smoke")
    assert ladder_l3a2._store(config).root.name == "ladder_l3a2_smoke" and ladder_l3a2.source_directory(config).name == "ladder_l1_smoke"
    # L3a's own variables do not move this stage, and this stage's do not move L3a.
    assert ladder_l3a._store(config).root.name == "ladder_l3a_seed2" and ladder_l3a.source_directory(config).name == "ladder_l1_seed2"


def test_the_episodes_are_the_configs_by_group_and_a_smoke_run_has_l3as(alternating, monkeypatch):
    config = alternating.config
    config["ladder_loop"]["repair_alternate"] = {"episodes_goal": 12, "episodes_below": 12, "episodes_rungs": 2}        # the spec's sizes
    own = ladder_l3a2.sizes(config)
    assert own["episodes"] == {"goal": 12, "below": 12, "in": 2, "above": 2} and own["episodes_from"] == "repair_alternate"
    assert [ladder_l3a2.L3A2.episodes(own, group) for group in ("goal", "below", "in", "above")] == [12, 12, 2, 2]
    # The smoke stage names L3a's section: the sizes L3a's smoke run has on the fixture (4 on the goal problem, 2 on each rung problem).
    monkeypatch.setenv(ladder_l3a2.L3A2_EPISODES_VARIABLE, "repair")
    config["ladder_loop"]["repair"] = {"episodes_goal": 4, "episodes_rungs": 2}
    smoke = ladder_l3a2.sizes(config)
    assert smoke["episodes"] == {"goal": 4, "below": 2, "in": 2, "above": 2} and smoke["episodes_from"] == "repair" and smoke["sampling_seed"] == own["sampling_seed"]
    monkeypatch.setenv(ladder_l3a2.L3A2_EPISODES_VARIABLE, "round")
    with pytest.raises(ValueError, match="names the section 'round'"):
        ladder_l3a2.sizes(config)
    monkeypatch.delenv(ladder_l3a2.L3A2_EPISODES_VARIABLE)
    config["ladder_loop"]["episode"]["loops"] = 1
    with pytest.raises(ValueError, match="a first attempt and at least one more"):
        ladder_l3a2.sizes(config)


def test_the_sampling_seed_is_the_stages_own_and_no_draw_is_one_of_l3as(alternating):
    config = alternating.config
    base = ladder_l3a2.sizes(config)["sampling_seed"]
    others = set(ladder_l2.sampling_seeds(config).values()) | {ladder_round.sampling_seed(config, name) for name in ladder_round.SAMPLED_SETS}
    others.add(ladder_l3a.sizes(config)["sampling_seed"])
    assert base == 1000 + ladder_l3a2.SEED_PLACE == SEED and ladder_l3a2.SEED_PLACE != ladder_l3a.SEED_PLACE and base not in others
    places = [(problem, side, episode, loop) for problem in ("p", "q") for side in ("statement", "negation") for episode in range(12) for loop in range(5)]
    own = {ladder_l3a.attempt_seed(base, *place) for place in places}
    of_l3a = {ladder_l3a.attempt_seed(ladder_l3a.sizes(config)["sampling_seed"], *place) for place in places}
    assert len(own) == len(places) and not own & of_l3a and min(own) >= 2 ** 31 > max(others)      # the first attempts and every later draw


def test_a_proof_on_the_side_a_certificate_contradicts_stops_the_step_and_leaves_its_evidence(alternating):
    config = alternating.config
    config["ladder_loop"]["episode"]["contradicted_side"] = "all"       # every problem's ruled-out side is attempted and checked
    ladder_round.ladder_l1_prepare(config)
    known_true = next(row["problem_id"] for row in ladder_l3a._rows(ladder_l3a2.source_directory(config) / "heldout_groups.jsonl") if row["side"] == "true")
    alternating.pool.alarm = known_true
    ladder_l3a2.ladder_l3a2_prepare(config)
    with pytest.raises(SoundnessAlarm, match=f"{known_true} is known true"):
        ladder_l3a2.ladder_l3a2_attempts(config)
    store = alternating.store()
    assert not store.is_done(ladder_l3a.loop_marker(0, ladder_l3a2.L3A2)) and not store.is_done(ladder_l3a2.ATTEMPTS)
    audited = [row for row in store.read_rows(ladder_l3a.attempts_file(0)) if row["audit"]]
    assert any(row["status"] == "verified" and row["problem_id"] == known_true and row["side"] == "negation" for row in audited)
    # Without the alarm the ruled-out side's attempts are first attempts only: they are checked and never continued.
    alternating.pool.alarm = None
    shutil.rmtree(store.root)
    _run(config)
    rows = _attempts(alternating.store())
    assert [row for row in rows if row["audit"]] and all(row["loop"] == 0 and row["status"] != "verified" for row in rows if row["audit"])
    assert len(alternating.store().read_rows("repair_episodes.jsonl")) == 12 + 2 * 12 + 3 * 6


def test_the_stage_is_registered_with_a_guard_before_each_gpu_step():
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l3a2"]]
    assert [(environment, step) for environment, step, _ in steps] == [
        ("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_l3a2_prepare"), ("guard", None), ("gpu", "ladder_l3a2_attempts"),
        ("guard", None), ("gpu", "ladder_l3a2_report")]
    assert set(ladder_l3a2.STEPS) == {"ladder_l3a2_prepare", "ladder_l3a2_attempts", "ladder_l3a2_report"} and all(options == {} for _, _, options in steps)
    assert not set(ladder_l3a2.STEPS) & set(ladder_l3a.STEPS)
    assert [ladder_l3a2.L3A2.step(part) for part in ("prepare", "attempts", "report")] == [ladder_l3a2.PREPARE, ladder_l3a2.ATTEMPTS, ladder_l3a2.REPORT]
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l3a2_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in steps]
    variables = entry.child_environment("gpu", "key", smoke[2][2])
    assert variables[ladder_l3a2.L3A2_SOURCE_VARIABLE] == "ladder_l1_smoke" and variables[ladder_l3a2.L3A2_RUN_VARIABLE] == "ladder_l3a2_smoke"
    assert variables[ladder_l3a2.L3A2_EPISODES_VARIABLE] == "repair"        # the sizes L3a's smoke run has
    assert ladder_l3a.L3A_RUN_VARIABLE not in variables and ladder_l3a2.L3A2_RUN_VARIABLE not in entry.child_environment(
        "gpu", "key", entry.step_fields(entry.STAGES["ladder_l3a_smoke"][2])[2])
