"""The ladder loop's L3a, the stage end to end on what an L1 run stored: the blind first attempts, the three arms that
go on from each failed one, the state requests, the trimmed loops, the report; a rerun, a run resumed after an
interruption, the refusals, the alarm, the registration. Spec: docs/spec/ladder-loop.spec.md, "L3a: the repair
check, no training". The engine and Lean are scripted (Lean answers with positions and `sorries`, as the pool does, and
with no `sorry` at all where no goal is left at a cut); nothing touches a GPU or the network. The rules for one proof
are `test_repair_cut.py`, the read is `test_ladder_l3a.py`."""

import contextlib
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_round import KNOWN_FALSE, stage  # noqa: E402, F401 - the L1 fixture stage, with stand-ins

from rlvr_lean.domain.problem_pool import SoundnessAlarm  # noqa: E402
from rlvr_lean.domain.proving import build_prover_prompt  # noqa: E402
from rlvr_lean.domain.repair import ARMS, BLIND, FIRST, RESUME_ARMS, RESUME_WITH_STATE, RESUME_WITHOUT_STATE  # noqa: E402
from rlvr_lean.domain.repair import cut as rules  # noqa: E402
from rlvr_lean.domain.repair.read import BRANCHES, episode_rows  # noqa: E402
from rlvr_lean.domain.verification.lean_source import build_proof_source  # noqa: E402
from rlvr_lean.domain.verification.pin import LEAN_PINS  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import ladder_l2, ladder_l3a, ladder_round  # noqa: E402
from rlvr_lean.gpu.stand_in_engine import StandInOutput, StandInSample, stand_in_parameters  # noqa: E402
from rlvr_lean.infrastructure.verification_service import LeanCheckSettings  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
COMMENT = re.compile(r"[ ]*/- tactic state:\n(?:.*\n)*?[ ]*-/\n")
FIRST_PROOFS = ("  step_one\n  bad_step\n  done\n",                 # an error on the second line: one line kept
                "  step_one\n  done\n",                             # verified
                "  step_one\n  step_two\n",                         # the proof runs out with goals open: all of it is kept
                "  bad_step\n  done\n",                             # an error on the first line: nothing kept
                None,                                               # the token cap
                "  open_structure\n  bad_step\n  done\n",           # a cut the kept lines do not close: no state
                "  have h : P := by\n    bad_step\n  done\n",       # an error inside a nested block
                "  step_one\n  closes_goal\n  bad_step\n",          # a whole proof with something extra after it: no goal left at the cut
                "  have h : P := by\n    closes_inner\n    bad_step\n  done\n",     # no goal left in an INNER block: the kept lines prove nothing
                "  two_goals\n  bad_step\n  done\n")                # two goals open at the cut
CONTINUATIONS = ("  bad_step\n  done\n",                            # repeats the step that had just failed
                 "  done\n",                                        # verified
                 "  step_more\n",                                   # runs out again
                 "  slow_step\n  done\n",                           # runs into the Lean limit: nothing to cut at the next loop
                 "  closes_goal\n  bad_step\n")                     # proves it and goes on: no goal left at the next cut
PLACEHOLDER = "all_goals sorry"
STATE_FILE = re.compile(r"^ +all_goals sorry$", re.MULTILINE)


class ScriptedEngine:
    """One sample for each prompt, chosen by its seed: a whole proof for the plain prompt, a continuation for a
    prompt that holds kept lines. The same for the same prompt and seed."""

    def __init__(self):
        self.calls = []

    def generate(self, prompts, parameters, lora_request=None):
        assert isinstance(parameters, list) and len(parameters) == len(prompts) and all(one.n == 1 and one.max_tokens == 1024 for one in parameters)
        outputs = []
        for prompt, one in zip(prompts, parameters):
            self.calls.append((prompt, one.seed))
            plain = prompt.rstrip("\n").endswith(":= by") and prompt.count(":= by") == 1
            text = FIRST_PROOFS[one.seed % len(FIRST_PROOFS)] if plain else CONTINUATIONS[one.seed % len(CONTINUATIONS)]
            if text is None:
                outputs.append(StandInOutput([StandInSample("  step_one\n" * 40, list(range(1024)), "length")]))
            else:
                outputs.append(StandInOutput([StandInSample(text, list(range(len(text) // 4)), "stop")]))
        return outputs


class ScriptedPool:
    """Lean by the text of a proof, answering as the pool does: an error carries the position of its line; a file
    that ends in `all_goals sorry` comes back with every goal open there, each at the `sorry`; and with no goal
    open there it comes back with no `sorry` at all."""

    def __init__(self):
        self.settings = LeanCheckSettings(base_url="http://scripted", pin=LEAN_PINS["v4.27"])
        self.sources, self.alarm = [], None

    def close(self):
        pass

    def submit_sources(self, sources):
        self.sources.extend(sources.values())
        answers = [{"id": key, "time": 0.1, **self.answer(source)} for key, source in sources.items()]
        return SimpleNamespace(result=lambda: answers)

    def answer(self, source):
        lines = source.split("\n")
        theorem = next(index for index, line in enumerate(lines) if line.startswith("theorem "))
        name = lines[theorem].split()[1]
        last = lines.index(f"#print axioms {name}") - 1                 # the blank line after the proof
        proof = [(index + 1, lines[index]) for index in range(theorem + 1, last)]
        unsolved = {"severity": "error", "data": f"unsolved goals\n⊢ the goal of {name}", "pos": {"line": theorem + 1, "column": 40},
                    "endPos": {"line": last + 1, "column": 0}}
        proved = {"severity": "info", "data": f"'{name}' does not depend on any axioms", "pos": {"line": last + 2, "column": 0}}
        asks = next(((line, text) for line, text in proof if text.strip() == PLACEHOLDER), None)
        if asks is not None:
            line, text = asks
            indentation, kept = len(text) - len(text.lstrip()), [other for number, other in proof if number < line]
            column = indentation + len("all_goals ")                    # Lean reports the goals at the `sorry`
            if any("open_structure" in other for other in kept):
                return {"response": {"messages": [{"severity": "error", "data": "unexpected token '#print'; expected ']'", "pos": {"line": last + 2, "column": 0}}],
                                     "sorries": [{"pos": {"line": line, "column": column}, "goal": "⊢ ?m.1"}]}}
            if kept and kept[-1].strip() == "closes_goal":              # no goal is left: `all_goals` runs nothing, and the file is a proof
                return {"response": {"messages": [proved]}}
            if kept and kept[-1].strip() == "closes_inner":             # none is left in the inner block; the theorem's own goal is still open
                return {"response": {"messages": [unsolved]}}
            goals = [f"kept : {len(kept)} lines\n⊢ the goal of {name}"]
            if any("two_goals" in other for other in kept):
                goals = [f"case left\n⊢ the first goal of {name}", f"case right\n⊢ the second goal of {name}"]
            return {"response": {"messages": [unsolved] if indentation > 2 else [],
                                 "sorries": [{"pos": {"line": line, "column": column}, "goal": goal} for goal in goals]}}
        if self.alarm is not None and name == f"negation_of_{self.alarm}":      # a verifier gone wrong: it accepts a proof of the ruled-out side
            return {"response": {"messages": [proved]}}
        if any("slow_step" in text for _, text in proof):
            return {"error": "Lean REPL command timed out in 30 seconds"}
        bad = next(((line, text) for line, text in proof if "bad_step" in text), None)
        if bad is not None:
            return {"response": {"messages": [unsolved, {"severity": "error", "data": "bad_step failed", "pos": {"line": bad[0], "column": len(bad[1]) - len(bad[1].lstrip())}}]}}
        base = name[len("negation_of_"):] if name.startswith("negation_of_") else name
        allowed = (name.startswith("negation_of_") == (base in KNOWN_FALSE)) or base == self.alarm
        if allowed and proof and proof[-1][1].strip() in ("done", "closes_goal"):
            return {"response": {"messages": [proved]}}
        return {"response": {"messages": [unsolved]}}


@pytest.fixture
def repair(stage, monkeypatch):  # noqa: F811
    """What L1's prepare step stores on the fixtures (one goal problem, five rung problems), a scripted engine and
    a scripted Lean; six first attempts on the goal problem and four on each rung problem."""
    for name in (ladder_l3a.L3A_RUN_VARIABLE, ladder_l3a.L3A_SOURCE_VARIABLE):
        monkeypatch.delenv(name, raising=False)
    stage.config["ladder_loop"]["repair"].update({"episodes_goal": 6, "episodes_rungs": 4})
    engine, pool = ScriptedEngine(), ScriptedPool()
    monkeypatch.setattr(ladder_l3a, "engine_kit", lambda: lambda config: (engine, stand_in_parameters))
    monkeypatch.setattr(ladder_l3a, "lean_pool", lambda config: contextlib.nullcontext(pool))
    return SimpleNamespace(config=stage.config, engine=engine, pool=pool, store=lambda: ladder_l3a._store(stage.config))


def _run(config):
    return {step: ladder_l3a.STEPS[step](config) for environment, step, _ in map(entry.step_fields, entry.STAGES["ladder_l3a"])
            if environment == "gpu" and step in ladder_l3a.STEPS}


def _files(directory):
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir()) if path.is_file()}


def _attempts(store, loops=5):
    return [row for loop in range(loops) for row in store.read_rows(ladder_l3a.attempts_file(loop))]


def test_the_whole_stage_runs_on_what_an_l1_run_stored_and_writes_a_run_of_its_own(repair):
    config = repair.config
    with pytest.raises(RuntimeError, match="stage `ladder_l1` with --seeds 0"):           # nothing to read yet: refused, and it says what to run
        ladder_l3a.ladder_l3a_prepare(config)
    assert not ladder_l3a.source_directory(config).exists()                               # and the refusal created nothing
    ladder_round.ladder_l1_prepare(config)
    source = ladder_l3a.source_directory(config)
    before = _files(source)
    summaries = _run(config)
    assert _files(source) == before                                                        # L1's run directory was only read
    store = repair.store()
    assert store.root.name == "ladder_l3a_seed0" and store.root.parent == source.parent
    prepare = summaries["ladder_l3a_prepare"]
    assert prepare["goal_set"] == 1 and prepare["rungs"] == {"below": 2, "in": 2, "above": 1} and prepare["first_attempts"] == 6 + 5 * 4
    assert prepare["sizes"]["loops"] == 5 and prepare["sizes"]["sampling_seed"] == 1030 and prepare["problems_with_no_side_to_attempt"] == []
    problems = {row["problem_id"]: row for row in store.read_rows("problems.jsonl")}
    assert {row["group"] for row in problems.values()} == {"goal", "below", "in", "above"}

    # ---- the first attempts: blind, on the side the certificate allows, shared by the three arms
    rows = _attempts(store)
    firsts = [row for row in rows if row["loop"] == 0]
    assert {row["arm"] for row in firsts} == {FIRST} and all(row["how"] == "blind" and row["kept_lines"] == 0 for row in firsts)
    own = {(row["problem_id"], row["episode"]): row for row in firsts if not row["audit"]}
    assert len(own) == prepare["first_attempts"] == len([row for row in firsts if not row["audit"]])
    assert all(row["side"] == ("negation" if row["problem_id"] in KNOWN_FALSE else "statement") for row in own.values())
    assert all(row["arm"] != FIRST and not row["audit"] for row in rows if row["loop"] > 0)
    assert {row["status"] for row in firsts} >= {"verified", "lean_error", "capped_tokens"}

    # ---- every failed first attempt goes on three ways, each to its first verified attempt and at most four more
    later, beside = {}, {}
    for row in rows:
        if row["loop"] > 0:
            (beside if row["untrimmed"] else later).setdefault((row["arm"], row["problem_id"], row["episode"]), []).append(row)
    failed = {key for key, row in own.items() if row["status"] != "verified"}
    assert set(later) == {(arm, *key) for arm in ARMS for key in failed} and failed
    blind = {(row["problem_id"], row["episode"], row["loop"]): row for row in rows if row["arm"] == BLIND}
    trimmed_rows = []
    for (arm, problem_id, episode), chain in later.items():
        assert [row["loop"] for row in chain] == list(range(1, len(chain) + 1)) and len(chain) <= 4
        assert all(row["status"] != "verified" for row in chain[:-1]) and (chain[-1]["status"] == "verified" or len(chain) == 4)
        previous = own[problem_id, episode]
        for row in chain:
            if row["how"] == "trimmed":
                # Spec item 2a: no goal was left at the cut and the kept lines verified: the loop resolves the episode with nothing generated.
                assert arm in RESUME_ARMS and row is chain[-1] and previous["status"] == "lean_error"
                assert (row["status"], row["token_count"], row["prompt_tokens"], row["completion"], row["seed"], row["job"]) == ("verified", 0, 0, "", None, None)
                kept = rules.proof_lines(previous["proof"])[:row["kept_lines"]]
                assert row["proof"] == "".join(line + "\n" for line in kept) and kept[-1].strip() == "closes_goal" and row["sent_to_lean"] is True
                assert build_proof_source(ladder_l3a.statement_of(problems[problem_id], row["side"]), row["proof"]) in repair.pool.sources      # its one check
                trimmed_rows.append(row)
                continue
            # Every attempt of one loop of one episode is sampled with the same seed, whatever the arm.
            assert row["seed"] == ladder_l3a.attempt_seed(1030, problem_id, row["side"], episode, row["loop"])
            if row["how"] == "resume":
                assert arm in RESUME_ARMS and row["no_state"] is None and row["had_state"] == (arm == RESUME_WITH_STATE)
                kept = rules.proof_lines(previous["proof"])[:row["kept_lines"]]
                assert row["proof"] == "".join(line + "\n" for line in kept) + row["completion"] and "tactic state" not in row["proof"]
                assert previous["cut_line"] == (row["kept_lines"] + 1 if row["cut"] == rules.BODY else None)
            else:
                assert row["kept_lines"] == 0 and row["proof"] == row["completion"] and row["had_state"] is False
                assert (row["no_state"] in rules.NO_STATE_REASONS) if arm in RESUME_ARMS else row["no_state"] is None
            previous = row
    # A resume loop that got no state is a blind attempt: the blind arm's own sample at that loop, when it made one.
    fell_back = [row for row in rows if row["arm"] in RESUME_ARMS and row["how"] == "blind"]
    shared = [row for row in fell_back if (row["problem_id"], row["episode"], row["loop"]) in blind]
    assert shared and all(row["job"] == blind[row["problem_id"], row["episode"], row["loop"]]["job"]
                          and row["completion"] == blind[row["problem_id"], row["episode"], row["loop"]]["completion"] for row in shared)
    reasons = {row["no_state"] for row in fell_back if not row["untrimmed"]}
    assert {rules.ERROR_AFTER_THE_SORRY, rules.NO_ERROR_POSITION, rules.NO_GOALS_AT_THE_CUT} <= reasons     # a cut inside a structure; the cap or a
    # timeout; no goal left in an inner block, whose kept lines did not verify
    # The kinds of cut the scripted proofs give were all made.
    resumed = [row for row in rows if row["how"] == "resume"]
    assert {row["cut"] for row in resumed} == {rules.BODY, rules.WHOLE_PROOF} and {0, 1} <= {row["kept_lines"] for row in resumed}
    assert {row["repeats_failed_step"] for row in resumed} == {True, False, None}

    # ---- trimmed loops: in BOTH resuming arms from the shared first attempt, later in an arm's own; and beside each, the
    # loop as the blind fall-back it would have been, with what followed (the reading without trimming)
    at_first = {(row["problem_id"], row["episode"]) for row in trimmed_rows if row["loop"] == 1}
    assert at_first and all(sum(row["loop"] == 1 and (row["problem_id"], row["episode"]) == key for row in trimmed_rows) == 2 for key in at_first)
    assert all(own[key]["completion"] == "  step_one\n  closes_goal\n  bad_step\n" for key in at_first) and {row["loop"] for row in trimmed_rows} > {1}
    assert set(beside) == {(row["arm"], row["problem_id"], row["episode"]) for row in trimmed_rows}
    for row in trimmed_rows:
        chain = beside[row["arm"], row["problem_id"], row["episode"]]
        assert [other["loop"] for other in chain] == list(range(row["loop"], row["loop"] + len(chain))) and chain[-1]["loop"] <= 4
        assert all(other["status"] != "verified" for other in chain[:-1]) and (chain[-1]["status"] == "verified" or chain[-1]["loop"] == 4)
        assert (chain[0]["how"], chain[0]["no_state"], chain[0]["kept_lines"]) == ("blind", rules.NO_GOALS_AT_THE_CUT, 0)
        assert chain[0]["job"] == blind.get((row["problem_id"], row["episode"], row["loop"]), chain[0])["job"]       # the blind arm's own sample
    assert not [row for row in rows if row["untrimmed"] and row["arm"] not in RESUME_ARMS]

    # ---- the prompts: the state in the prover's comment format, and the other resuming arm the same without it
    with_state = [(prompt, seed) for prompt, seed in repair.engine.calls if "/- tactic state:" in prompt]
    assert with_state and all(prompt.endswith("-/\n") for prompt, _ in with_state)
    first_loop = {ladder_l3a.attempt_seed(1030, problem_id, own[problem_id, episode]["side"], episode, 1) for problem_id, episode in failed}
    for prompt, seed in with_state:
        if seed in first_loop:
            assert (COMMENT.sub("", prompt, count=1), seed) in repair.engine.calls
    assert "    /- tactic state:\n      kept : 1 lines\n" in "".join(prompt for prompt, _ in with_state)        # inside the nested block, at its indentation
    # Every goal open at the cut is in the comment, as Lean prints several goals (spec item 2).
    assert any("  /- tactic state:\n    case left\n    ⊢ the first goal of " in prompt and "\n    \n    case right\n    ⊢ the second goal of " in prompt
               for prompt, _ in with_state)
    plain = {build_prover_prompt(ladder_l3a.statement_of(problems[row["problem_id"]], row["side"])) for row in firsts}
    assert all(prompt in plain for prompt, seed in repair.engine.calls if prompt.count(":= by") == 1 and prompt.rstrip("\n").endswith(":= by"))
    assert len(repair.engine.calls) == len(set(repair.engine.calls)) == summaries["ladder_l3a_attempts"]["generations"]      # one generation a prompt and seed

    # ---- Lean: an attempt that reached the token cap is a failure and is never sent; a state file is the kept lines and `all_goals sorry`
    capped = [row for row in rows if row["status"] == "capped_tokens"]
    assert capped and all(row["finish_reason"] == "length" and row["sent_to_lean"] is False and row["first_error_line"] is None for row in capped)
    sent = set(repair.pool.sources)
    assert not any(build_proof_source(ladder_l3a.statement_of(problems[row["problem_id"]], row["side"]), row["proof"]) in sent for row in capped)
    step = summaries["ladder_l3a_attempts"]
    state_files = [source for source in repair.pool.sources if STATE_FILE.search(source)]
    assert len(repair.pool.sources) == step["lean_checks_sent"] and len(state_files) == step["state_files_sent_to_lean"] > 0
    assert not [source for source in repair.pool.sources if re.search(r"^ +sorry$", source, re.MULTILINE)]
    assert step["lean_checks_sent"] == sum(entry["attempts_sent_to_lean"] + entry["state_files_sent_to_lean"] + entry["kept_lines_checks_sent_to_lean"] for entry in step["loops"])
    errors = [row for row in rows if row["status"] == "lean_error" and row["errors"]]
    assert errors and all(row["first_error_line"] is not None and row["first_error"] for row in errors)
    assert {row["first_error_line"] for row in errors} >= {0, 1, 2}                 # the theorem's own line, and lines of the proof

    # ---- the state requests: one for every failed attempt that goes on in a resuming arm
    states = [row for loop in range(4) for row in store.read_rows(ladder_l3a.states_file(loop))]
    assert not store.path(ladder_l3a.states_file(4)).exists()
    assert {row["arm"] for row in states if row["after_loop"] == 0} == {FIRST} and {row["arm"] for row in states if row["after_loop"] > 0} == set(RESUME_ARMS)
    assert {(row["problem_id"], row["episode"]) for row in states if row["after_loop"] == 0} == failed
    assert {row["outcome"] for row in states} >= {"state", rules.ERROR_AFTER_THE_SORRY, rules.NO_ERROR_POSITION, rules.NO_GOALS_AT_THE_CUT, "trimmed"}
    assert all((row["state"] is not None) == (row["outcome"] == "state") and row["requested"] == (row["cut"] is not None) for row in states)
    assert all(row["cut_line"] == row["kept_lines"] + 1 and row["failed_step"] for row in states if row["cut"] == rules.BODY)
    # No goal left at the cut: the kept lines were checked on their own, and only those that verified are "trimmed".
    assert all((row["outcome"] == "trimmed") == (row["trim_status"] == "verified") for row in states)
    assert all(row["trim_status"] is None for row in states if row["outcome"] not in ("trimmed", rules.NO_GOALS_AT_THE_CUT))
    assert "lean_error" in {row["trim_status"] for row in states if row["outcome"] == rules.NO_GOALS_AT_THE_CUT}
    assert step["kept_lines_checks_sent_to_lean"] >= len({row["trim_file"] for row in states if row["trim_file"]}) > 0
    asked = {(row["arm"], row["problem_id"], row["episode"], row["after_loop"]): row for row in states}
    request_of = lambda row: asked[FIRST if row["loop"] == 1 else row["arm"], row["problem_id"], row["episode"], row["loop"] - 1]      # noqa: E731

    # ---- the episodes and the report
    episodes = store.read_rows("repair_episodes.jsonl")
    assert episodes == episode_rows(list(problems.values()), rows, states, 5) and len(episodes) == prepare["first_attempts"]
    assert step["episodes"] == len(episodes) and step["failed_first_attempts"] == len(failed) and step["attempts"] == len(rows)
    assert step["episodes_resolved"] == {arm: sum(row["arms"][arm]["resolved_at"] is not None for row in episodes) for arm in ARMS}
    assert step["episodes_resolved_by_a_trimmed_loop"] == {arm: sum(row["arm"] == arm for row in trimmed_rows) for arm in RESUME_ARMS}
    assert step["attempts_of_the_reading_without_trimming_only"] == sum(len(chain) for chain in beside.values())
    assert [entry["loop"] for entry in step["loops"]] == [0, 1, 2, 3, 4] and step["loops_done_in_this_run"] == [0, 1, 2, 3, 4]
    assert {row["problem_id"] for row in store.read_rows("repair_problems.jsonl")} == set(problems)
    by_key = {(row["problem_id"], row["episode"]): row for row in episodes}
    for row in trimmed_rows:        # the two readings of an episode with a trimmed loop
        entry = by_key[row["problem_id"], row["episode"]]["arms"][row["arm"]]
        assert (entry["trimmed_at"], entry["resolved_at"], entry["loops"][-1]["how"], entry["loops"][-1]["kept_lines_check"]) == (row["loop"], row["loop"] + 1, "trimmed", "verified")
        other = entry["without_trimming"]["loops"]
        assert (other[row["loop"] - 1]["how"], other[row["loop"] - 1]["no_state"]) == ("blind", rules.NO_GOALS_AT_THE_CUT) and other[:row["loop"] - 1] == [
            {**entry_step, "kept_lines_check": None} for entry_step in entry["loops"][:-1]]
        assert len(other) == row["loop"] - 1 + len(beside[row["arm"], row["problem_id"], row["episode"]])
    report = summaries["ladder_l3a_report"]
    assert report["fixture"] is True and report["stand_in_engine"] is True and report["branch"]["name"] in BRANCHES
    assert report["heldout"]["goal_set"] == 1 and report["primary"]["problems"] == 3 and report["primary"]["episodes"] == 6 + 2 * 4
    budget = report["budget"]["by_arm"]
    assert budget[BLIND]["state_requests"] == 0 and budget[BLIND]["lean_checks"] == budget[BLIND]["attempts_sent_to_lean"] and budget[BLIND]["of_which_trimmed"] == 0
    for arm in RESUME_ARMS:      # the spec's reading: a resume loop costs one more check, for the state, when its failed proof could be cut;
        # one more where the kept lines were checked on their own; a trimmed loop is an attempt with no tokens
        chains = [chain for (name, _, _), chain in later.items() if name == arm]
        requested = sum(request_of(row)["requested"] for chain in chains for row in chain)
        kept_lines = sum(request_of(row)["trim_status"] is not None for chain in chains for row in chain)
        assert budget[arm]["state_requests"] == requested and budget[arm]["kept_lines_checks"] == kept_lines > 0
        assert budget[arm]["attempts_sent_to_lean"] == sum(row["sent_to_lean"] and row["how"] != "trimmed" for chain in chains for row in chain)
        assert budget[arm]["lean_checks"] == budget[arm]["attempts_sent_to_lean"] + requested + kept_lines
        assert budget[arm]["attempts"] == sum(len(chain) for chain in chains) and budget[arm]["of_which_trimmed"] == sum(row["arm"] == arm for row in trimmed_rows)
        assert budget[arm]["generated_tokens"] == sum(row["token_count"] for chain in chains for row in chain)
        assert report["diagnostics"][arm]["without_a_state_by_reason"] and report["diagnostics"][arm]["repair_loops"] == budget[arm]["attempts"]
        assert report["trimmed"][arm]["resolutions"] == report["diagnostics"][arm]["trimmed"] == budget[arm]["of_which_trimmed"] > 0
        assert sum(report["trimmed"][arm]["by_set"][name] for name in ("goal", "below", "in", "above")) == report["trimmed"][arm]["resolutions"]
    # The primary, the single step and the goal set are given with the trimmed loops and without them.
    without = report["without_trimming"]
    assert without["primary"]["problems"] == report["primary"]["problems"] and without["primary"][BLIND] == report["primary"][BLIND]
    assert without["single_step"]["all"][RESUME_WITH_STATE]["of_which_trimmed"] == 0 < report["single_step"]["all"][RESUME_WITH_STATE]["of_which_trimmed"] == len(at_first)
    assert without["single_step"]["all"][BLIND] == report["single_step"]["all"][BLIND] and set(without["goal_set_by_problem"]) == set(report["goal_set_by_problem"]) - {"what"}
    assert "trimmed resolutions (kept lines that verified on their own, nothing generated)" in report["headline"] and "the primary without them" in report["headline"]
    assert json.loads(store.path("report_ladder_l3a.json").read_text())["headline"] == report["headline"]

    # ---- a rerun returns what is stored: nothing is sampled or checked again
    calls, sources = len(repair.engine.calls), len(repair.pool.sources)
    again = _run(config)
    assert (len(repair.engine.calls), len(repair.pool.sources)) == (calls, sources) and again == summaries
    assert not ladder_round._store(config).is_done("ladder_l3a_prepare") and _files(source) == before


def test_a_run_that_stopped_resumes_at_the_first_loop_not_done_and_reads_the_same(repair, monkeypatch, tmp_path):
    config = repair.config
    ladder_round.ladder_l1_prepare(config)
    ladder_l3a.ladder_l3a_prepare(config)
    whole = ladder_l3a.run_loop

    def stops(config, store, problems, own, loop, *rest):
        if loop == 2:
            raise RuntimeError("interrupted")
        return whole(config, store, problems, own, loop, *rest)

    monkeypatch.setattr(ladder_l3a, "run_loop", stops)
    with pytest.raises(RuntimeError, match="interrupted"):
        ladder_l3a.ladder_l3a_attempts(config)
    store = repair.store()
    assert [store.is_done(ladder_l3a.loop_marker(loop)) for loop in range(5)] == [True, True, False, False, False] and not store.is_done(ladder_l3a.ATTEMPTS)
    monkeypatch.setattr(ladder_l3a, "run_loop", whole)
    out = tmp_path / "rerun_out" / "steps"                               # the rerun is another task, with an output directory of its own
    monkeypatch.setenv("RLVR_LEAN_STEP_DIR", str(out))
    calls = len(repair.engine.calls)
    summary = ladder_l3a.ladder_l3a_attempts(config)
    assert summary["loops_done_in_this_run"] == [2, 3, 4] and len(repair.engine.calls) > calls
    assert len(repair.engine.calls) == len(set(repair.engine.calls))                                 # nothing of loops 0 and 1 was sampled again
    delivered = {path.name for path in out.iterdir()}
    assert {"repair_attempts_0.jsonl", "repair_states_0.jsonl", "repair_attempts_1.jsonl", "repair_attempts_4.jsonl", "repair_states_3.jsonl",
            "ladder_l3a_loop_0.done.json", "ladder_l3a_loop_4.done.json", "repair_episodes.jsonl", "repair_problems.jsonl", "ladder_l3a_attempts.done.json"} <= delivered
    resumed = store.read_rows("repair_episodes.jsonl")
    # The report step hands a rerun the per-episode and per-problem rows and the markers again, not the attempts.
    again = tmp_path / "report_out" / "steps"
    monkeypatch.setenv("RLVR_LEAN_STEP_DIR", str(again))
    ladder_l3a.ladder_l3a_report(config)
    handed = {path.name for path in again.iterdir()}
    assert {"report_ladder_l3a.json", "repair_episodes.jsonl", "repair_problems.jsonl", "problems.jsonl", "heldout_groups.jsonl", "ladder_l3a_prepare.done.json",
            "ladder_l3a_loop_3.done.json", "ladder_l3a_attempts.done.json", "ladder_l3a_report.done.json"} <= handed
    assert not [name for name in handed if name.startswith(("repair_attempts_", "repair_states_"))]
    # The same run made in one go reads the same.
    monkeypatch.delenv("RLVR_LEAN_STEP_DIR")
    shutil.rmtree(store.root)
    ladder_l3a.ladder_l3a_prepare(config)
    ladder_l3a.ladder_l3a_attempts(config)
    assert repair.store().read_rows("repair_episodes.jsonl") == resumed


def test_a_state_whose_prompt_would_not_fit_the_models_context_makes_the_loop_a_blind_attempt(repair):
    config = repair.config
    config["vllm"]["max_model_len"] = 1024                              # no prompt leaves room for 1,024 new tokens
    ladder_round.ladder_l1_prepare(config)
    report = _run(config)["ladder_l3a_report"]
    rows = _attempts(repair.store())
    assert not [row for row in rows if row["how"] == "resume"] and not [prompt for prompt, _ in repair.engine.calls if "tactic state" in prompt]
    reasons = {row["no_state"] for row in rows if row["arm"] in RESUME_ARMS}
    assert rules.PROMPT_TOO_LONG in reasons and report["diagnostics"][RESUME_WITH_STATE]["share_with_a_state"] == 0.0
    # With no state anywhere the three arms are the same attempts but for the trimmed loops, which come from the cut and
    # not from the prompt; and the run says it could not have seen a win.
    episodes = repair.store().read_rows("repair_episodes.jsonl")
    at = lambda row, arm: (row["arms"][arm]["without_trimming"] or row["arms"][arm])["resolved_at"]      # noqa: E731
    assert all(at(row, BLIND) == at(row, RESUME_WITH_STATE) == at(row, RESUME_WITHOUT_STATE) for row in episodes)
    assert [row for row in rows if row["how"] == "trimmed"] and report["without_trimming"]["primary"]["mean"] == 0.0
    assert report["branch"]["name"] == "INCONCLUSIVE" and "the cut is broken" in report["branch"]["reason"]


def test_a_proof_on_the_side_a_certificate_contradicts_stops_the_step_and_leaves_its_evidence(repair):
    config = repair.config
    config["ladder_loop"]["episode"]["contradicted_side"] = "all"       # every problem's ruled-out side is attempted and checked
    ladder_round.ladder_l1_prepare(config)
    known_true = next(row["problem_id"] for row in ladder_l3a._rows(ladder_l3a.source_directory(config) / "heldout_groups.jsonl") if row["side"] == "true")
    repair.pool.alarm = known_true
    ladder_l3a.ladder_l3a_prepare(config)
    with pytest.raises(SoundnessAlarm, match=f"{known_true} is known true"):
        ladder_l3a.ladder_l3a_attempts(config)
    store = repair.store()
    assert not store.is_done(ladder_l3a.loop_marker(0)) and not store.is_done(ladder_l3a.ATTEMPTS)
    audited = [row for row in store.read_rows(ladder_l3a.attempts_file(0)) if row["audit"]]
    assert any(row["status"] == "verified" and row["problem_id"] == known_true and row["side"] == "negation" for row in audited)
    # Without the alarm the ruled-out side's attempts are first attempts only: they are checked and never continued.
    repair.pool.alarm = None
    shutil.rmtree(store.root)
    _run(config)
    rows = _attempts(repair.store())
    assert [row for row in rows if row["audit"]] and all(row["loop"] == 0 and row["status"] != "verified" for row in rows if row["audit"])
    assert len(repair.store().read_rows("repair_episodes.jsonl")) == 6 + 5 * 4
    # ... and no state is asked for at them: every state file is of the side the certificate allows.
    ruled_out = {row["problem_id"] if row["side"] == "false" else f"negation_of_{row['problem_id']}" for row in repair.store().read_rows("problems.jsonl")}
    state_files = [source for source in repair.pool.sources if STATE_FILE.search(source)]
    assert state_files and not [source for source in state_files if re.search(r"^theorem (\S+)", source, re.MULTILINE).group(1) in ruled_out]


def test_the_reading_without_trimming_never_trims_and_asks_for_no_kept_lines_check(repair):
    """One loop, by hand. An arm whose episode a trimmed loop resolved goes on only in the reading WITHOUT
    trimming: there a cut with no goal left is a blind fall-back, whatever the kept lines would do, and no
    kept-lines check is made for it. An arm still in the spec's reading trims at the same request."""
    config = repair.config
    own = ladder_l3a.sizes(config)
    statement = "theorem demo (x : ℝ) : x = x := by\n"
    problem = {"problem_id": "demo", "group": "below", "side": "true", "statement": statement, "negation": None, "episodes": 1}
    row = lambda arm, loop, **more: {"arm": arm, "problem_id": "demo", "episode": 0, "loop": loop, "audit": False, "untrimmed": False, "side": "statement",      # noqa: E731
                                     "how": "blind", "status": "lean_error", "proof": "  step_one\n  closes_goal\n  bad_step\n", **more}
    attempts = {0: [row(FIRST, 0)],
                1: [row(BLIND, 1), row(RESUME_WITH_STATE, 1, how="trimmed", status="verified"), row(RESUME_WITH_STATE, 1, untrimmed=True),
                    row(RESUME_WITHOUT_STATE, 1, how="resume")]}
    request = {"problem_id": "demo", "episode": 0, "after_loop": 1, "outcome": "trimmed", "kept_lines": 2, "cut": rules.BODY, "trim_seconds": 0.1}
    states = [{**request, "arm": RESUME_WITH_STATE}, {**request, "arm": RESUME_WITHOUT_STATE}]
    jobs, trimmed = ladder_l3a.next_jobs(config, [problem], own, 2, attempts, states, lambda text: len(text) // 4)
    # Only the arm that had not trimmed yet gets a trimmed row; both then make the blind attempt, in the other reading.
    assert [(entry["arm"], entry["loop"], entry["how"], entry["proof"], entry["untrimmed"]) for entry in trimmed] == [
        (RESUME_WITHOUT_STATE, 2, "trimmed", "  step_one\n  closes_goal\n", False)]
    assert len(jobs) == 1 and jobs[0].prompt == build_prover_prompt(statement) and jobs[0].wants_state is True      # one generation: the blind arm's
    consumers = {consumer["arm"]: consumer for consumer in jobs[0].consumers}
    assert [consumers[arm]["untrimmed"] for arm in ARMS] == [False, True, True]
    assert all((consumers[arm]["how"], consumers[arm]["no_state"]) == ("blind", rules.NO_GOALS_AT_THE_CUT) for arm in RESUME_ARMS)
    assert consumers[BLIND]["no_state"] is None

    def settled(job):
        job.completion, job.finish_reason, job.token_count = "  step_one\n  closes_goal\n  bad_step\n", "stop", 9
        sent = len(repair.pool.sources)
        return ladder_l3a.settle_chunk(repair.pool, [job], config["ladder_loop"]["episode"]), repair.pool.sources[sent:]

    stats, sources = settled(jobs[0])
    # Its proof leaves no goal at its cut too, and nobody in the spec's reading goes on from it: the kept lines are not checked.
    assert stats == {"sent": 1, "state_files": 1, "kept_lines_checks": 0} and len(sources) == 2
    assert (jobs[0].request["outcome"], jobs[0].request["trim_status"], jobs[0].request["trim_file"]) == (rules.NO_GOALS_AT_THE_CUT, None, None)
    assert {entry["outcome"] for entry in ladder_l3a.state_rows(jobs[0], 2)} == {rules.NO_GOALS_AT_THE_CUT}
    # The same proof with an arm of the spec's reading going on from it: one more check, of the kept lines alone, and they verify.
    again, _ = ladder_l3a.next_jobs(config, [problem], own, 2, attempts, states, lambda text: len(text) // 4)
    again[0].consumers[1]["untrimmed"] = False
    stats, sources = settled(again[0])
    assert stats == {"sent": 1, "state_files": 1, "kept_lines_checks": 1} and sources[-1] == build_proof_source(statement, "  step_one\n  closes_goal\n")
    assert (again[0].request["outcome"], again[0].request["trim_status"]) == ("trimmed", "verified")
    assert [(entry["arm"], entry["outcome"], entry["trim_status"]) for entry in ladder_l3a.state_rows(again[0], 2)] == [
        (RESUME_WITH_STATE, "trimmed", "verified"), (RESUME_WITHOUT_STATE, "trimmed", "verified")]


def test_a_run_keeps_the_sizes_and_the_seed_it_was_prepared_with(repair, monkeypatch):
    config = repair.config
    ladder_round.ladder_l1_prepare(config)
    ladder_l3a.ladder_l3a_prepare(config)
    config["ladder_loop"]["repair"]["episodes_goal"] = 7
    with pytest.raises(RuntimeError, match="a run keeps the sizes it began with"):
        ladder_l3a.ladder_l3a_attempts(config)
    with pytest.raises(RuntimeError, match="a run keeps the sizes it began with"):
        ladder_l3a.ladder_l3a_prepare(config)
    config["ladder_loop"]["repair"]["episodes_goal"] = 6
    with pytest.raises(RuntimeError, match="needs the step ladder_l3a_attempts"):
        ladder_l3a.ladder_l3a_report(config)
    # Another seed is another run directory, reading that seed's L1 run, with sampling seeds of its own.
    monkeypatch.setenv("RLVR_LEAN_TRAINING_SEEDS", "2")
    assert ladder_l3a._store(config).root.name == "ladder_l3a_seed2" and ladder_l3a.source_directory(config).name == "ladder_l1_seed2"
    assert ladder_l3a.sizes(config)["sampling_seed"] == 1230
    monkeypatch.setenv(ladder_l3a.L3A_RUN_VARIABLE, "ladder_l3a_smoke")
    monkeypatch.setenv(ladder_l3a.L3A_SOURCE_VARIABLE, "ladder_l1_smoke")
    assert ladder_l3a._store(config).root.name == "ladder_l3a_smoke" and ladder_l3a.source_directory(config).name == "ladder_l1_smoke"


def test_the_sampling_seeds_are_the_stages_own_and_do_not_depend_on_the_arm(repair):
    config = repair.config
    base = ladder_l3a.sizes(config)["sampling_seed"]
    others = set(ladder_l2.sampling_seeds(config).values()) | {ladder_round.sampling_seed(config, name) for name in ladder_round.SAMPLED_SETS}
    assert base == 1000 + ladder_l3a.SEED_PLACE and base not in others
    seeds = {(problem, side, episode, loop): ladder_l3a.attempt_seed(base, problem, side, episode, loop)
             for problem in ("p", "q") for side in ("statement", "negation") for episode in range(4) for loop in range(5)}
    assert len(set(seeds.values())) == len(seeds) and min(seeds.values()) >= 2 ** 31 > max(others)          # never a seed another measurement uses
    assert ladder_l3a.attempt_seed(base, "p", "statement", 0, 1) == seeds["p", "statement", 0, 1] != ladder_l3a.attempt_seed(base + 100, "p", "statement", 0, 1)


def test_the_stage_is_registered_with_a_guard_before_each_gpu_step():
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l3a"]]
    assert [(environment, step) for environment, step, _ in steps] == [
        ("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_l3a_prepare"), ("guard", None), ("gpu", "ladder_l3a_attempts"),
        ("guard", None), ("gpu", "ladder_l3a_report")]
    assert set(ladder_l3a.STEPS) == {"ladder_l3a_prepare", "ladder_l3a_attempts", "ladder_l3a_report"} and all(options == {} for _, _, options in steps)
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l3a_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in steps]
    variables = entry.child_environment("gpu", "key", smoke[2][2])
    assert variables[ladder_l3a.L3A_SOURCE_VARIABLE] == "ladder_l1_smoke" and variables[ladder_l3a.L3A_RUN_VARIABLE] == "ladder_l3a_smoke"
