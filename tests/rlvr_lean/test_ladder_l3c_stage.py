"""The ladder loop's L3c, the stage end to end on what an L1 run stored: the first generations, the two arms that go on
from each failed one (blind; accumulate: the pool of verified lemmas, the fresh and the continuing generations, the pool
checks, the kept closers and the proofs they assemble), the known copies, the report; a rerun, a run resumed after an
interruption, the pilot, the refusals, the alarm, the registration. Spec: docs/spec/ladder-loop.spec.md, "L3c: an
episode that keeps what verified (no training)". The engine and Lean are scripted (Lean reads a proof's lemmas and its
closing step, and answers a pool's file with its goals); nothing touches a GPU or the network. The rules, the read and
the report are `test_ladder_l3c.py`."""

import contextlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l3a_stage import PLACEHOLDER, STATE_FILE, ScriptedPool  # noqa: E402 - L3a's scripted Lean: the pool's shape, the alarm
from test_ladder_round import KNOWN_FALSE, stage  # noqa: E402, F401 - the L1 fixture stage, with stand-ins

from rlvr_lean.domain.problem_pool import SoundnessAlarm  # noqa: E402
from rlvr_lean.domain.proving import build_prover_prompt  # noqa: E402
from rlvr_lean.domain.repair import BLIND, FIRST  # noqa: E402
from rlvr_lean.domain.repair import accumulate as pooling  # noqa: E402
from rlvr_lean.domain.repair import cut as rules  # noqa: E402
from rlvr_lean.domain.repair.accumulate import ACCUMULATE, ARMS, ASSEMBLED, BRANCHES, CONTINUE, FRESH, OWN_PROOF, STANDS  # noqa: E402
from rlvr_lean.domain.repair.alternate import KNOWN_COPY  # noqa: E402
from rlvr_lean.domain.verification.lean_source import build_proof_source  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import ladder_l2, ladder_l3a, ladder_l3a2, ladder_l3c, ladder_round  # noqa: E402
from rlvr_lean.gpu.stand_in_engine import StandInOutput, StandInSample, stand_in_parameters  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
SEED = 1032                                     # round.sampling_seed + 100 x the task's seed + the stage's place, one after L3a2's
GENERATIONS = 8
NOT_SENT_ANYWAY = ("capped_tokens", "rejected_lexical")
# Whole proofs (the plain prompt), chosen by the seed. A lemma is `have <name> : <fact> := by <tactic>`.
WHOLE_PROOFS = ("  have h₁ : fact_a := by good\n  exact_from h₁\n",                  # a lemma, then a closing step that names it and needs fact_b too
                "  have h₂ : fact_b := by\n    good\n  bad_step\n",                  # a lemma over two lines, then a step that never closes anything
                "  step_one\n  done\n",                                              # verified
                "  combine\n",                                                       # ONE closing step and no lemma: it needs fact_a and fact_b above it
                None,                                                                # the token cap
                "  have h₁ : fact_a := by good\n  have : fact_c := by good\n  have h₃ : fact_d := by bad_lemma\n  done\n",       # two lemmas before the error
                "  have k : fact_e := by needs_first\n  bad_step\n",                 # a lemma that stands only as the proof's first step
                "  have g : fact_g := by opens_goal\n  bad_step\n",                  # a `have` that leaves a goal of its own open
                "  slow_step\n  done\n",                                             # runs into the Lean limit: no error to cut by
                "  bad_step\n  done\n",                                              # an error on the first line: no lemma, and two steps are no closer
                "  have h₁ : fact_a := by good\n  have h₁ : fact_a := by good\n  have h₄ : fact_h := by good\n  bad_step\n")    # the same statement twice
# What the model writes after a pool (a continuing prompt), chosen by the seed.
CONTINUATIONS = ("  have h₂ : fact_b := by good\n  bad_step\n",                      # one more lemma: the pool grows, the next generation continues too
                 "  bad_step\n  done\n",                                             # nothing new: the next generation is fresh
                 "  combine\n",                                                      # closes when the pool holds fact_a and fact_b
                 "  done\n",                                                         # verified
                 "    bad_step\n",                                                   # not at the pool's indentation: nothing is harvested
                 "  have h₁ : fact_a := by good\n  have q : fact_q := by good\n  bad_step\n",       # a statement the pool may hold already, and a new one
                 "  bad_step\n")                                                     # ONE step, and the one many a proof's closer was
LEMMA = re.compile(r"\s*have(?: (\S+))? : (\S+) := by(?: (.+))?$")
UNPOOLED = re.compile(r"_g\d+'*")                # what a pooled name has more than the name the model wrote


class ScriptedEngine:
    """One sample for each prompt, chosen by its seed: a whole proof for the plain prompt, a continuation for a
    prompt that holds a pool. The same for the same prompt and seed."""

    def __init__(self):
        self.calls = []

    def generate(self, prompts, parameters, lora_request=None):
        assert isinstance(parameters, list) and len(parameters) == len(prompts) and all(one.n == 1 and one.max_tokens == 1024 for one in parameters)
        outputs = []
        for prompt, one in zip(prompts, parameters):
            self.calls.append((prompt, one.seed))
            plain = prompt.rstrip("\n").endswith(":= by") and prompt.count(":= by") == 1
            text = WHOLE_PROOFS[one.seed % len(WHOLE_PROOFS)] if plain else CONTINUATIONS[one.seed % len(CONTINUATIONS)]
            if text is None:
                outputs.append(StandInOutput([StandInSample("  step_one\n" * 40, list(range(1024)), "length")]))
            else:
                outputs.append(StandInOutput([StandInSample(text, list(range(max(1, len(text) // 4))), "stop")]))
        return outputs


class PoolingLean(ScriptedPool):
    """Lean by the text of a proof, step by step, answering as the pool does. A lemma (`have h : fact := by tactic`)
    puts its fact among the hypotheses when its tactic is `good` (or `needs_first` on the proof's first line, or
    `opens_goal`, which also leaves a second goal); `bad_lemma` is an error on its line and the proof goes on, as Lean
    goes on after a `have` whose own proof failed. A closing step proves the theorem (on the side that can be proved)
    when what it needs is above it: `combine` needs fact_a and fact_b, `exact_from h` needs `h` and fact_b, `done`
    nothing. A file that ends in `all_goals sorry` comes back with the goals open there: the hypotheses, then the goal."""

    def answer(self, source):
        lines = source.split("\n")
        theorem = next(index for index, line in enumerate(lines) if line.startswith("theorem "))
        name = lines[theorem].split()[1]
        last = lines.index(f"#print axioms {name}") - 1
        unsolved = {"severity": "error", "data": f"unsolved goals\n⊢ the goal of {name}", "pos": {"line": theorem + 1, "column": 40}, "endPos": {"line": last + 1, "column": 0}}
        proved = {"severity": "info", "data": f"'{name}' does not depend on any axioms", "pos": {"line": last + 2, "column": 0}}
        if self.alarm is not None and name == f"negation_of_{self.alarm}":      # a verifier gone wrong: it accepts a proof of the ruled-out side
            return {"response": {"messages": [proved]}}
        base = name[len("negation_of_"):] if name.startswith("negation_of_") else name
        allowed = name.startswith("negation_of_") == (base in KNOWN_FALSE)
        failed = lambda number, text: {"severity": "error", "data": f"{text.strip()} failed", "pos": {"line": number, "column": len(text) - len(text.lstrip())}}      # noqa: E731
        held, errors, goals, closed, waiting = {}, [], 1, False, None
        for number, text in ((index + 1, lines[index]) for index in range(theorem + 1, last)):
            step = text.strip()
            if step == PLACEHOLDER:
                column = len(text) - len(text.lstrip()) + len("all_goals ")
                state = "\n".join([*(f"{lemma or 'this'} : {fact}" for lemma, fact in held.items()), f"⊢ the goal of {name}"])
                asked = [] if closed else [state, *(f"⊢ the goal a lemma of {name} left open" for _ in range(goals - 1))]
                return {"response": {"messages": errors, "sorries": [{"pos": {"line": number, "column": column}, "goal": goal} for goal in asked]}}
            lemma = LEMMA.match(text) if waiting is None else None
            if lemma is not None and lemma.group(3) is None:
                waiting = (lemma.group(1) or "", lemma.group(2))                # its tactic is on the next line
                continue
            if lemma is not None or waiting is not None:
                bound, fact, tactic = (lemma.group(1) or "", lemma.group(2), lemma.group(3)) if lemma is not None else (*waiting, step)
                waiting = None
                if tactic == "bad_lemma" or (tactic == "needs_first" and number != theorem + 2):
                    errors.append(failed(number, text))
                else:
                    held[bound] = fact
                    goals += tactic == "opens_goal"
            elif "slow_step" in step:
                return {"error": "Lean REPL command timed out in 30 seconds"}
            elif step == "bad_step" or (step == "combine" and not {"fact_a", "fact_b"} <= set(held.values())) or (
                    step.startswith("exact_from ") and not (step.split()[1] in held and "fact_b" in held.values())):
                return {"response": {"messages": [unsolved, *errors, failed(number, text)]}}       # the tactic block stops at a step that fails
            elif step in ("done", "combine") or step.startswith("exact_from "):
                closed = allowed
        return {"response": {"messages": [proved] if closed and not errors and goals == 1 else [unsolved, *errors]}}


@pytest.fixture
def accumulating(stage, monkeypatch):  # noqa: F811
    """What L1's prepare step stores on the fixtures (one goal problem, five rung problems), a scripted engine and
    a scripted Lean; twelve first generations on the goal problem and on each below-band problem, six on the others."""
    for name in (ladder_l3c.L3C_RUN_VARIABLE, ladder_l3c.L3C_SOURCE_VARIABLE, ladder_l3c.L3C_PILOT_VARIABLE):
        monkeypatch.delenv(name, raising=False)
    stage.config["ladder_loop"]["accumulate"].update({"episodes_goal": 12, "episodes_below": 12, "episodes_rungs": 6})
    engine, pool = ScriptedEngine(), PoolingLean()
    monkeypatch.setattr(ladder_l3a, "engine_kit", lambda: lambda config: (engine, stand_in_parameters))      # the sampling is L3a's: so are the engine and
    monkeypatch.setattr(ladder_l3a, "lean_pool", lambda config: contextlib.nullcontext(pool))                # the Lean client it is given
    return SimpleNamespace(config=stage.config, engine=engine, pool=pool, store=lambda: ladder_l3c._store(stage.config))


def _run(config, stage_name="ladder_l3c"):
    return {step: ladder_l3c.STEPS[step](config) for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name])
            if environment == "gpu" and step in ladder_l3c.STEPS}


def _files(directory):
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir()) if path.is_file()}


def _rows(store, name, loops=GENERATIONS):
    return [row for loop in range(loops) for row in store.read_rows(name(loop))]


def test_the_whole_stage_runs_on_what_an_l1_run_stored_and_writes_a_run_of_its_own(accumulating):
    config = accumulating.config
    with pytest.raises(RuntimeError, match="L3c reads the run directory .*stage `ladder_l1` with --seeds 0"):         # nothing to read yet: refused
        ladder_l3c.ladder_l3c_prepare(config)
    assert not ladder_l3c.source_directory(config).exists()                               # and the refusal created nothing
    ladder_round.ladder_l1_prepare(config)
    source = ladder_l3c.source_directory(config)
    before = _files(source)
    summaries = _run(config)
    assert _files(source) == before                                                        # L1's run directory was only read
    store = accumulating.store()
    assert store.root.name == "ladder_l3c_seed0" and store.root.parent == source.parent
    prepare = summaries["ladder_l3c_prepare"]
    assert prepare["goal_set"] == 1 and prepare["rungs"] == {"below": 2, "in": 2, "above": 1} and prepare["first_attempts"] == 12 + 2 * 12 + 3 * 6
    assert prepare["sizes"] == {"episodes": {"goal": 12, "below": 12, "in": 6, "above": 6}, "pilot_problems": None, "generations": 8, "loops": 8,
                                "pool_blocks": 12, "kept_closers": 8, "max_new_tokens": 1024, "lean_seconds": 30, "sampling_seed": SEED}
    problems = {row["problem_id"]: row for row in store.read_rows("problems.jsonl")}
    statement = lambda row: ladder_l3a.statement_of(problems[row["problem_id"]], row["side"])      # noqa: E731

    # ---- generation 1: blind, on the side the certificate allows, shared by the two arms
    rows = _rows(store, ladder_l3a.attempts_file)
    firsts = [row for row in rows if row["loop"] == 0]
    assert {row["arm"] for row in firsts} == {FIRST} and all((row["how"], row["kind"], row["pool_blocks"], row["kept_lines"]) == ("blind", FRESH, 0, 0) for row in firsts)
    own = {(row["problem_id"], row["episode"]): row for row in firsts if not row["audit"]}
    assert len(own) == prepare["first_attempts"] and all(row["shared"] for row in own.values())
    assert all(row["side"] == ("negation" if row["problem_id"] in KNOWN_FALSE else "statement") for row in own.values())
    assert {row["arm"] for row in rows if row["loop"] > 0} == set(ARMS) and not [row for row in rows if row["loop"] > 0 and row["audit"]]
    pools, closers = _rows(store, ladder_l3c.pools_file), _rows(store, ladder_l3c.closers_file)
    of_pools, of_closers = {}, {}
    for row in pools:
        of_pools.setdefault((row["problem_id"], row["episode"]), []).append(row)
    for row in closers:
        of_closers.setdefault((row["problem_id"], row["episode"]), []).append(row)

    # ---- every failed first generation goes on in the two arms, each to the generation that resolves it and at most seven more
    later = {}
    for row in rows:
        if row["loop"] > 0:
            later.setdefault((row["arm"], row["problem_id"], row["episode"]), []).append(row)
    failed = {key for key, row in own.items() if row["status"] != "verified"}
    assembled_at = {key: min(row["loop"] for row in checks if row["status"] == "verified") for key, checks in of_closers.items()
                    if any(row["status"] == "verified" for row in checks)}
    for arm in ARMS:
        for key in failed:
            chain = later.get((arm, *key), [])
            assert [row["loop"] for row in chain] == list(range(1, len(chain) + 1)) and all(row["status"] != "verified" for row in chain[:-1])
            ended_by_a_closer = arm == ACCUMULATE and assembled_at.get(key) == len(chain)
            assert (chain and chain[-1]["status"] == "verified") or len(chain) == GENERATIONS - 1 or ended_by_a_closer
            assert not (chain and chain[-1]["status"] == "verified" and ended_by_a_closer)
    assert not [key for key in later if key[1:] not in failed]
    assert {len(chain) for (arm, *_), chain in later.items() if arm == BLIND} >= {1, 7} and assembled_at

    # ---- the kind of each generation of the accumulate arm, from what its episode held before it
    kinds = Counter()
    for key in failed:
        chain = [own[key], *later.get((ACCUMULATE, *key), [])]
        for index, row in enumerate(chain[1:], start=1):
            held = [pool for pool in of_pools.get(key, []) if pool["stands"] and pool["loop"] < row["loop"]]
            continued = [other["loop"] for other in chain[1:index] if other["kind"] == CONTINUE]
            changed = bool(held) and (not continued or held[-1]["loop"] >= continued[-1])
            assert row["kind"] == (CONTINUE if changed else FRESH)
            # Every generation of one loop of one episode is sampled with the same seed, whatever the arm; none is one of L3a's or L3a2's.
            assert row["seed"] == ladder_l3a.attempt_seed(SEED, key[0], row["side"], key[1], row["loop"])
            kinds[row["kind"], row["how"]] += 1
            if row["kind"] == FRESH:
                assert (row["how"], row["had_state"], row["no_state"], row["pool_blocks"], row["kept_lines"]) == ("blind", False, None, 0, 0) and row["proof"] == row["completion"]
            else:
                # It continues: the pool's lemmas are the proof so far, and the proof checked is the pool and what the model wrote.
                pool = held[-1]
                assert (row["how"], row["had_state"], row["pool_blocks"]) == ("resume", True, len(pool["blocks"])) and row["kept_lines"] == pool["text"].count("\n")
                assert row["proof"] == pool["text"] + row["completion"] and "tactic state" not in row["proof"]
                prompt = build_prover_prompt(statement(row)) + pool["text"] + rules.state_comment(pool["state"], "  ")
                assert (prompt, row["seed"]) in accumulating.engine.calls
    assert kinds[CONTINUE, "resume"] > 20 and kinds[FRESH, "blind"] > 20 and set(kinds) == {(CONTINUE, "resume"), (FRESH, "blind")}
    blind_rows = [row for row in rows if row["arm"] == BLIND]
    assert all((row["kind"], row["how"], row["pool_blocks"], row["harvested"], row["kept_closer"], row["pool_after"]) == (FRESH, "blind", 0, None, None, None)
               for row in blind_rows)

    # ---- a fresh generation is the blind arm's own sample while both arms have the episode open
    by_place = {}
    for row in rows:
        if row["loop"] > 0 and row["how"] == "blind":
            by_place.setdefault((row["problem_id"], row["episode"], row["loop"]), {})[row["arm"]] = row
    both = {place: pair for place, pair in by_place.items() if len(pair) == 2}
    assert both and all(pair[BLIND]["job"] == pair[ACCUMULATE]["job"] and pair[BLIND]["completion"] == pair[ACCUMULATE]["completion"] for pair in both.values())
    assert all(row["shared"] == ((row["problem_id"], row["episode"], row["loop"]) in both) for row in rows if row["loop"] > 0)
    assert [row for row in rows if row["arm"] == ACCUMULATE and row["kind"] == FRESH and not row["shared"]]      # the blind arm had resolved it already
    assert len(accumulating.engine.calls) == len(set(accumulating.engine.calls)) == summaries["ladder_l3c_attempts"]["generations"]

    # ---- the harvest: the lemmas before the first error join the pool under names of their generation; the pool only grows
    holders = [row for row in rows if row["arm"] in (FIRST, ACCUMULATE) and not row["audit"]]
    assert {row["no_harvest"] for row in holders} >= {None, KNOWN_COPY, rules.NO_ERROR_POSITION}
    assert {row["harvest_ended"] for row in holders if row["harvested"] is not None} >= {pooling.THE_ERROR, pooling.NOT_A_HAVE}
    for key, states in of_pools.items():
        blocks = []
        for state in states:
            holder = own[key] if state["loop"] == 0 else later[ACCUMULATE, *key][state["loop"] - 1]
            assert holder["harvested"] == state["harvested"] > 0 and state["generation"] == state["loop"] + 1 and state["blocks_before"] == len(blocks)
            assert state["blocks"][:len(blocks)] == blocks and len(state["blocks"]) == len(blocks) + state["harvested"] <= 12       # nothing leaves the front or the middle
            new = state["blocks"][len(blocks):]
            assert all(block["generation"] == state["generation"] and block["text"].startswith("  have") for block in new)
            assert all(block["name"].endswith(f"_g{state['generation']}") for block in new if block["name"])
            assert state["text"] == "".join(block["text"] + "\n" for block in state["blocks"])
            assert state["stands"] == (state["outcome"] == STANDS) == (state["state"] is not None)
            # The file that checked it: the pool, then `all_goals sorry`, sent once.
            assert accumulating.pool.sources.count(build_proof_source(statement(holder), state["text"] + "  all_goals sorry\n")) >= 1
            if state["stands"]:
                blocks = state["blocks"]
                assert all(f"{block['name'] or 'this'} : {block['statement'].lstrip(': ')}" in state["state"] for block in blocks)       # the pooled facts are hypotheses
            assert holder["pool_after"] == len(blocks)
        assert len({block["name"] for block in blocks if block["name"]}) == sum(bool(block["name"]) for block in blocks)                   # every pooled name is bound once
        assert len({(bool(block["name"]), block["statement"]) for block in blocks}) == len(blocks)                                        # and no statement is held twice
    outcomes = Counter(row["outcome"] for row in pools)
    assert outcomes[STANDS] > 20 and {rules.ERROR_BEFORE_THE_SORRY, pooling.SEVERAL_GOALS} <= set(outcomes)       # a lemma that stood only where it was; one that opened a goal
    assert any(row["harvest_duplicates"] for row in holders) and any(not block["name"] for row in pools for block in row["blocks"])
    assert [row for row in holders if row["kind"] == CONTINUE and row["harvested"]] and pooling.OFF_THE_POOL in {row["harvest_ended"] for row in holders}

    # ---- the closers: kept from a proof that was lemmas and one closing step, and tried again after the pool
    kept = [row for row in holders if row["kept_closer"]]
    assert kept and all(row["kept_closer"]["generation"] == row["loop"] + 1 for row in kept)
    assert {tuple(row["kept_closer"]["needs"]) for row in kept} >= {(), ("h₁_g1",)} and {row["kept_closer"]["text"] for row in kept} >= {"  combine", "  bad_step", "  exact_from h₁_g1"}
    for key, checks in of_closers.items():
        seen = set()
        for check in checks:
            holder = own[key] if check["loop"] == 0 else later[ACCUMULATE, *key][check["loop"] - 1]
            held = [pool for pool in of_pools.get(key, []) if pool["stands"] and pool["loop"] <= check["loop"]][-1]
            assert check["proof"] == held["text"] + check["closer"] + "\n" and check["pool_blocks"] == len(held["blocks"]) and check["pool_made_at_generation"] == held["generation"]
            assert set(check["needs"]) <= {block["name"] for block in held["blocks"]} and check["from_generation"] <= check["generation"] == holder["loop"] + 1
            assert (check["proof"], check["closer"]) not in seen                    # a (pool, closer) pair is checked once
            seen.add((check["proof"], check["closer"]))
            assert check["sent_to_lean"] == (check["status"] not in (OWN_PROOF, KNOWN_COPY)) == (check["file"] is not None)
            if check["sent_to_lean"]:
                assert accumulating.pool.sources.count(build_proof_source(statement(holder), check["proof"])) >= 1
            # A closer check Lean failed says where: the line of the assembled proof its first error is on (here the closing step's).
            assert (check["first_error_line"] is not None) == (check["status"] == "lean_error")
            if check["status"] == "lean_error":
                assert check["first_error_line"] == check["proof"].count("\n") - check["closer"].count("\n") and check["first_error"].endswith(" failed")
            if check["status"] == OWN_PROOF:
                # The pool right after a proof's own harvest, then that proof's own closer, is the proof Lean has just rejected under its
                # pooled names (the names apart, the same text): not sent.
                assert holder["status"] == "lean_error" and UNPOOLED.sub("", check["proof"]) == UNPOOLED.sub("", holder["proof"])
    statuses = Counter(row["status"] for row in closers)
    assert {"verified", "lean_error", OWN_PROOF} <= set(statuses) and {row["from_generation"] == row["generation"] for row in closers if row["status"] == OWN_PROOF} == {True, False}
    assembled = [row for row in closers if row["status"] == "verified"]
    assert {row["closer"] for row in assembled} >= {"  combine", "  exact_from h₁_g1"} and any(row["from_generation"] < row["generation"] for row in assembled)

    # ---- known copies: a proof Lean already rejected in the same arm of the same episode is a failed attempt that is not sent
    copies = [row for row in rows if row["status"] == KNOWN_COPY]
    assert {row["arm"] for row in copies} == set(ARMS) and all(not row["sent_to_lean"] and row["copy_of"] is not None and row["token_count"] > 0 for row in copies)
    assert all(row["no_harvest"] == KNOWN_COPY and row["harvested"] is None for row in copies if row["arm"] == ACCUMULATE)
    assert all((row["no_harvest"], row["harvested"], row["kept_closer"]) == (None, None, None) for row in holders if row["status"] == "verified")       # and nothing from a proof that verified
    # In the accumulate arm the rejected texts hold the proofs the pool and a kept closer make: a continuation that is that closer is not sent.
    assembled_copies = [row for row in copies if row["copy_of_an_assembled_proof"]]
    assert assembled_copies and all(row["arm"] == ACCUMULATE and row["how"] == "resume" and "the pool and a kept closer make" in row["first_error"] for row in assembled_copies)
    assert all("which Lean rejected: not sent again" in row["first_error"] and "kept closer" not in row["first_error"] for row in copies if not row["copy_of_an_assembled_proof"])
    by_job = {}
    for row in rows:
        by_job.setdefault(row["job"], []).append(row)
    sent_jobs = {job: shared[0] for job, shared in by_job.items() if any(row["sent_to_lean"] for row in shared)}
    assert all(row["status"] in (KNOWN_COPY, *NOT_SENT_ANYWAY) for job, shared in by_job.items() if job not in sent_jobs for row in shared)
    step = summaries["ladder_l3c_attempts"]
    pool_files = [source for source in accumulating.pool.sources if STATE_FILE.search(source)]
    # Each Lean file is sent once a chunk, whoever shares it: two episodes of one problem that hold the same pool ask one file.
    assert len(pools) >= len(pool_files) == step["pool_checks_sent_to_lean"] == len({(row["loop"], row["pool_file"]) for row in pools}) > 0
    assert sum(row["sent_to_lean"] for row in closers) >= step["closer_checks_sent_to_lean"] == len({(row["loop"], row["file"]) for row in closers if row["sent_to_lean"]}) > 0
    assert len(accumulating.pool.sources) == step["lean_checks_sent"] == len(sent_jobs) + len(pool_files) + step["closer_checks_sent_to_lean"]

    # ---- the episodes and the report
    episodes = store.read_rows("repair_episodes.jsonl")
    assert episodes == pooling.episode_rows(list(problems.values()), rows, pools, closers, GENERATIONS) and len(episodes) == prepare["first_attempts"]
    by_key = {(row["problem_id"], row["episode"]): row for row in episodes}
    for key, row in by_key.items():
        arm = row["arms"][ACCUMULATE]
        if key in assembled_at and not (later.get((ACCUMULATE, *key)) and later[ACCUMULATE, *key][-1]["status"] == "verified"):
            first = next(check for check in of_closers[key] if check["status"] == "verified")
            assert (arm["resolved_at"], arm["resolved_by"], arm["assembled_proof"]) == (assembled_at[key] + 1, ASSEMBLED, first["proof"])
            assert arm["proof_lines"] == len(first["proof"].strip("\n").split("\n")) and len(arm["loops"]) == assembled_at[key]
        else:
            assert arm["assembled_proof"] is None and arm["resolved_by"] in (None, FRESH, CONTINUE)
        assert arm["pool_blocks"] == ([pool for pool in of_pools.get(key, []) if pool["stands"]] or [{"blocks": []}])[-1]["blocks"].__len__()
        assert len(arm["pool_checks"]) == len(of_pools.get(key, [])) and len(arm["closer_checks"]) == len(of_closers.get(key, []))
    assert {row["arms"][ACCUMULATE]["resolved_by"] for row in episodes} == {None, FRESH, CONTINUE, ASSEMBLED}
    assert step["episodes"] == len(episodes) and step["failed_first_generations"] == len(failed) and step["attempts"] == len(rows)
    assert step["episodes_resolved"] == {arm: sum(row["arms"][arm]["resolved_at"] is not None for row in episodes) for arm in ARMS}
    assert step["episodes_resolved_in_the_accumulate_arm_by"][ASSEMBLED] == sum(row["arms"][ACCUMULATE]["resolved_by"] == ASSEMBLED for row in episodes) > 0
    assert [entry["loop"] for entry in step["loops"]] == list(range(8)) and step["loops_done_in_this_run"] == list(range(8)) and step["pilot"] is False
    assert {row["problem_id"] for row in store.read_rows("repair_problems.jsonl")} == set(problems)
    report = summaries["ladder_l3c_report"]
    assert report["fixture"] is True and report["stand_in_engine"] is True and report["pilot"] is False and report["branch"]["name"] in BRANCHES
    assert report["heldout"]["goal_set"] == 1 and report["primary"]["problems"] == 3 and report["primary"]["episodes"] == 3 * 12
    assert report["headline"].startswith("L3c seed 0: ") and "accumulate minus blind" in report["headline"]
    budget = report["budget"]["by_arm"]
    for arm in ARMS:
        chains = [chain for (name, _, _), chain in later.items() if name == arm]
        assert budget[arm]["generations"] == sum(len(chain) for chain in chains) and budget[arm]["generated_tokens"] == sum(row["token_count"] for chain in chains for row in chain)
        assert budget[arm]["known_copies"] == sum(row["status"] == KNOWN_COPY for chain in chains for row in chain) > 0
        assert budget[arm]["attempts_sent_to_lean"] == sum(row["sent_to_lean"] for chain in chains for row in chain)
    assert (budget[BLIND]["pool_checks"], budget[BLIND]["closer_checks"]) == (0, 0) and budget[BLIND]["lean_checks"] == budget[BLIND]["attempts_sent_to_lean"]
    # The accumulate arm's Lean checks: its attempts that were sent, one for every pool state a harvest made (the one after generation
    # 1 too) and one for every closer check sent. Each episode is counted in full, as each arm is.
    assert (budget[ACCUMULATE]["pool_checks"], budget[ACCUMULATE]["closer_checks"]) == (len(pools), sum(row["sent_to_lean"] for row in closers))
    assert budget[ACCUMULATE]["lean_checks"] == budget[ACCUMULATE]["attempts_sent_to_lean"] + len(pools) + sum(row["sent_to_lean"] for row in closers)
    stands = report["can_this_run_see_a_win"]["the_pool_stands"]
    assert (stands["pool_checks"], stands["stand"]) == (len(pools), outcomes[STANDS]) and report["pools"]["all"]["episodes_resolved_by_an_assembled_proof"] > 0
    states = report["can_this_run_see_a_win"]["a_state_for_the_continuing_generations"]
    assert (states["continuing_generations"], states["with_a_state"]) == (kinds[CONTINUE, "resume"], kinds[CONTINUE, "resume"]) and states["passes"] is True
    # The fixture's problems have no published proof in the lengths file: the report says so and does not guess.
    assert report["by_proof_length"]["hard_problems"] == {"not_known": 3} and report["reach"]["at_4_lines_or_more"]["problems"] == 0
    assert json.loads(store.path("report_ladder_l3c.json").read_text())["headline"] == report["headline"]

    # ---- a rerun returns what is stored: nothing is sampled or checked again
    calls, sources = len(accumulating.engine.calls), len(accumulating.pool.sources)
    again = _run(config)
    assert (len(accumulating.engine.calls), len(accumulating.pool.sources)) == (calls, sources) and again == summaries
    assert not ladder_round._store(config).is_done("ladder_l3c_prepare") and _files(source) == before


def test_a_run_that_stopped_resumes_at_the_first_loop_not_done_and_reads_the_same(accumulating, monkeypatch, tmp_path):
    config = accumulating.config
    ladder_round.ladder_l1_prepare(config)
    ladder_l3c.ladder_l3c_prepare(config)
    whole = ladder_l3c.run_loop

    def stops(config, store, problems, own, loop, *rest):
        if loop == 4:
            raise RuntimeError("interrupted")
        return whole(config, store, problems, own, loop, *rest)

    monkeypatch.setattr(ladder_l3c, "run_loop", stops)
    with pytest.raises(RuntimeError, match="interrupted"):
        ladder_l3c.ladder_l3c_attempts(config)
    store = accumulating.store()
    assert [store.is_done(ladder_l3a.loop_marker(loop, ladder_l3c.L3C)) for loop in range(8)] == [True] * 4 + [False] * 4
    assert not store.is_done(ladder_l3c.ATTEMPTS) and not [path for path in store.root.iterdir() if path.name.startswith(("ladder_l3a_", "ladder_l3a2_"))]
    monkeypatch.setattr(ladder_l3c, "run_loop", whole)
    out = tmp_path / "rerun_out" / "steps"                               # the rerun is another task, with an output directory of its own
    monkeypatch.setenv("RLVR_LEAN_STEP_DIR", str(out))
    calls = len(accumulating.engine.calls)
    summary = ladder_l3c.ladder_l3c_attempts(config)
    # What each episode holds at loop 4 (its pool, Lean's state after it, its kept closers, the proofs Lean rejected) is read from the rows loops 0 to 3 STORED.
    assert summary["loops_done_in_this_run"] == [4, 5, 6, 7] and len(accumulating.engine.calls) > calls
    assert len(accumulating.engine.calls) == len(set(accumulating.engine.calls))                         # nothing of loops 0 to 3 was sampled again
    delivered = {path.name for path in out.iterdir()}
    assert {"repair_attempts_0.jsonl", "repair_pools_0.jsonl", "repair_closers_3.jsonl", "repair_attempts_7.jsonl", "repair_pools_7.jsonl", "repair_closers_7.jsonl",
            "ladder_l3c_loop_0.done.json", "ladder_l3c_loop_7.done.json", "repair_episodes.jsonl", "repair_problems.jsonl", "ladder_l3c_attempts.done.json"} <= delivered
    resumed = store.read_rows("repair_episodes.jsonl")
    # The report step hands a rerun the per-episode and per-problem rows and the markers again, not the attempts.
    again = tmp_path / "report_out" / "steps"
    monkeypatch.setenv("RLVR_LEAN_STEP_DIR", str(again))
    ladder_l3c.ladder_l3c_report(config)
    handed = {path.name for path in again.iterdir()}
    assert {"report_ladder_l3c.json", "repair_episodes.jsonl", "repair_problems.jsonl", "problems.jsonl", "heldout_groups.jsonl", "ladder_l3c_prepare.done.json",
            "ladder_l3c_loop_5.done.json", "ladder_l3c_attempts.done.json", "ladder_l3c_report.done.json"} <= handed
    assert not [name for name in handed if name.startswith(("repair_attempts_", "repair_pools_", "repair_closers_"))]
    # The same run made in one go reads the same.
    monkeypatch.delenv("RLVR_LEAN_STEP_DIR")
    shutil.rmtree(store.root)
    ladder_l3c.ladder_l3c_prepare(config)
    ladder_l3c.ladder_l3c_attempts(config)
    assert accumulating.store().read_rows("repair_episodes.jsonl") == resumed


def _job(statement, completion, state, kept=(), how="blind", arm=ACCUMULATE, known=None, seed=7):
    """One generation of one accumulate episode, by hand: `state` is what the episode holds before it."""
    consumer = {"arm": arm, "problem_id": "demo", "group": "below", "side": "statement", "episode": 0, "audit": False, "untrimmed": False, "how": how,
                "had_state": how == "resume", "no_state": None, "kept_lines": len(kept), "cut": None, "failed_step": None,
                "kind": CONTINUE if how == "resume" else FRESH, "pool_blocks": len(state["pool"]), "accumulating": state, "rejected": dict(known or state["rejected"])}
    prompt = build_prover_prompt(statement) + "".join(line + "\n" for line in kept)
    job = ladder_l3a.Job(prompt, seed, statement, tuple(kept), True, False, [consumer])
    job.completion, job.finish_reason, job.token_count = completion, "stop", 9
    return job


def test_one_generation_by_hand_harvests_checks_the_pool_and_tries_the_closers(accumulating):
    """A fixture known true (so `done`, `combine` and `exact_from` can prove it), one episode, generation by generation:
    what `settle_chunk` sends to Lean and what it leaves on the episode."""
    episode_settings, own = accumulating.config["ladder_loop"]["episode"], ladder_l3c.sizes(accumulating.config)
    statement = "theorem demo (x : ℝ) : x = x := by\n"

    def settled(job, loop):
        sent = len(accumulating.pool.sources)
        stats = ladder_l3c.settle_chunk(accumulating.pool, [job], episode_settings, own, loop)
        return stats, accumulating.pool.sources[sent:], job.consumers[0]["after"], ladder_l3c.attempt_rows(job, loop)[0]

    # Generation 1, fresh: a lemma and a closer that names it. The lemma joins the pool as h₁_g1 (one more check: the pool, then
    # `all_goals sorry`); the closer is kept; the pool with that closer is the proof Lean has just rejected, and is not sent.
    stats, sources, after, row = settled(_job(statement, "  have h₁ : fact_a := by good\n  exact_from h₁\n", pooling.empty_state()), 0)
    assert stats == {"sent": 1, "pool_checks": 1, "closer_checks": 0} and sources[1] == build_proof_source(statement, "  have h₁_g1 : fact_a := by good\n  all_goals sorry\n")
    assert [block["text"] for block in after["pool_now"]] == ["  have h₁_g1 : fact_a := by good"] and after["pool"]["stands"] and after["pool"]["state"] == "h₁_g1 : fact_a\n⊢ the goal of demo"
    assert after["kept_closer"] == {"text": "  exact_from h₁_g1", "needs": ["h₁_g1"], "generation": 1} and after["made_at"] == 0 and after["assembled"] is False
    assert [(check["status"], check["sent"], check["proof"]) for check in after["closers"]] == [(OWN_PROOF, False, "  have h₁_g1 : fact_a := by good\n  exact_from h₁_g1\n")]
    assert (row["harvested"], row["harvest_ended"], row["kept_closer"]["text"], row["pool_after"], row["cut_line"], row["no_harvest"]) == (1, pooling.NOT_A_HAVE, "  exact_from h₁_g1", 1, 2, None)

    # Generation 2 continues: what the model wrote after the pool is one more lemma and a step that fails. The lemma was verified
    # with the pool above it and joins it; the pool has grown, so EVERY kept closer is tried after it: the first generation's
    # now has the fact it lacked, and the proof it makes verifies.
    first = {"loop": 0, "kind": FRESH, "proof": "  have h₁ : fact_a := by good\n  exact_from h₁\n", "status": "lean_error", "sent_to_lean": True, "kept_closer": after["kept_closer"]}
    pools = [{"loop": 0, "stands": True, "blocks": after["pool_now"], "state": after["pool"]["state"]}]
    closers = [{"loop": 0, "status": OWN_PROOF, "sent_to_lean": False, "proof": after["closers"][0]["proof"]}]
    state = pooling.episode_state([first], pools, closers)
    assert pooling.kind_of(state) == CONTINUE and state["closers"] == [after["kept_closer"]] and set(state["rejected"]) == {
        "  have h₁ : fact_a := by good\n  exact_from h₁", "  have h₁_g1 : fact_a := by good\n  exact_from h₁_g1"}
    kept = pooling.pool_lines(state["pool"])
    stats, sources, after, row = settled(_job(statement, "  have h₂ : fact_b := by good\n  bad_step\n", state, kept, "resume"), 1)
    assert stats == {"sent": 1, "pool_checks": 1, "closer_checks": 1} and sources[0] == build_proof_source(statement, "  have h₁_g1 : fact_a := by good\n  have h₂ : fact_b := by good\n  bad_step\n")
    assert sources[1] == build_proof_source(statement, "  have h₁_g1 : fact_a := by good\n  have h₂_g2 : fact_b := by good\n  all_goals sorry\n")
    assert [block["name"] for block in after["pool_now"]] == ["h₁_g1", "h₂_g2"] and after["kept_closer"] == {"text": "  bad_step", "needs": [], "generation": 2}
    assert [(check["closer"]["text"], check["status"], check["sent"]) for check in after["closers"]] == [("  exact_from h₁_g1", "verified", True), ("  bad_step", OWN_PROOF, False)]
    assert after["assembled"] is True and sources[2] == build_proof_source(statement, "  have h₁_g1 : fact_a := by good\n  have h₂_g2 : fact_b := by good\n  exact_from h₁_g1\n")
    assert (row["kind"], row["how"], row["pool_blocks"], row["harvested"], row["pool_after"], row["resolved_by_an_assembled_proof"], row["status"]) == (
        CONTINUE, "resume", 1, 1, 2, True, "lean_error")

    # A continuation that is ONE step: nothing joins the pool, the step is kept, and the pool with it is this very proof (the
    # same text): a known copy, not sent. A closer the episode holds already is not kept twice.
    stats, sources, after, row = settled(_job(statement, "  bad_step\n", {**state, "closers": [*state["closers"], {"text": "  bad_step", "needs": [], "generation": 1}]}, kept, "resume"), 2)
    assert stats == {"sent": 1, "pool_checks": 0, "closer_checks": 0} and after["kept_closer"] is None and after["closers"] == [] and after["pool"] is None
    stats, sources, after, row = settled(_job(statement, "  combine\n", state, kept, "resume"), 2)
    assert stats == {"sent": 1, "pool_checks": 0, "closer_checks": 0} and after["kept_closer"]["text"] == "  combine"
    assert [(check["status"], check["sent"]) for check in after["closers"]] == [(OWN_PROOF, False)] and row["harvested"] == 0

    # A fresh generation with a pool above it: its ONE closing step is new after the pool (another attempt had the facts), and is sent.
    grown = {**state, "pool": [*state["pool"], {"text": "  have h₂_g2 : fact_b := by good", "name": "h₂_g2", "statement": ": fact_b", "generation": 2}], "made_at": 1}
    stats, sources, after, row = settled(_job(statement, "  combine\n", {**grown, "closers": []}, known={}), 3)
    assert stats == {"sent": 1, "pool_checks": 0, "closer_checks": 1} and after["assembled"] is True and after["closers"][0]["status"] == "verified"
    assert sources == [build_proof_source(statement, "  combine\n"), build_proof_source(statement, "  have h₁_g1 : fact_a := by good\n  have h₂_g2 : fact_b := by good\n  combine\n")]
    # ... unless Lean already rejected that very text in the episode: a known copy, with the loop it was rejected at.
    text = "  have h₁_g1 : fact_a := by good\n  have h₂_g2 : fact_b := by good\n  combine"
    stats, sources, after, row = settled(_job(statement, "  combine\n", {**grown, "closers": []}, known={text: 2}), 3)
    assert stats["closer_checks"] == 0 and [(check["status"], check["copy_of"], check["sent"]) for check in after["closers"]] == [(KNOWN_COPY, 2, False)]

    # A lemma that does not stand with the pool is taken back out: the pool, its state and its closers' names stay as they were.
    stats, sources, after, row = settled(_job(statement, "  have k : fact_e := by needs_first\n  exact_from k\n", grown, known={}), 4)
    assert stats == {"sent": 1, "pool_checks": 1, "closer_checks": 0} and (after["pool"]["outcome"], after["pool"]["stands"], after["pool"]["state"]) == (rules.ERROR_BEFORE_THE_SORRY, False, None)
    assert after["pool_now"] == grown["pool"] and after["made_at"] == 1 and after["kept_closer"] == {"text": "  exact_from k_g5", "needs": ["k_g5"], "generation": 5}
    assert after["closers"] == [] and (row["harvested"], row["pool_after"]) == (1, 2)                # its closer names a lemma the pool does not hold: not tried
    # A `have` that left a goal of its own open is no lemma either: two goals after the pool.
    stats, sources, after, row = settled(_job(statement, "  have g : fact_g := by opens_goal\n  bad_step\n", grown, known={}), 4)
    assert (after["pool"]["outcome"], after["pool_now"]) == (pooling.SEVERAL_GOALS, grown["pool"])
    # A continuing proof whose lemma is taken back out: its closing step after the pool ALONE is not the proof Lean rejected (that
    # one had the lemma above the step), so it is tried, and here it closes the theorem.
    stats, sources, after, row = settled(_job(statement, "  have g : fact_g := by opens_goal\n  combine\n", {**grown, "closers": []}, pooling.pool_lines(grown["pool"]), "resume", known={}), 4)
    assert stats == {"sent": 1, "pool_checks": 1, "closer_checks": 1} and after["pool"]["stands"] is False and after["pool_now"] == grown["pool"]
    assert [(check["status"], check["sent"], check["proof"]) for check in after["closers"]] == [("verified", True, pooling.pool_text(grown["pool"]) + "  combine\n")]
    # The pool check allows no error at all: a pooled `have` left open is "unsolved goals" around the `sorry`, which L3a's own reading lets pass.
    raw = {"response": {"messages": [{"severity": "error", "data": "unsolved goals\n⊢ x = x", "pos": {"line": 8, "column": 30}, "endPos": {"line": 10, "column": 17}}],
                        "sorries": [{"pos": {"line": 10, "column": 12}, "goal": "⊢ fact_a"}]}}
    assert rules.read_state(raw, 10, 12) == ("⊢ fact_a", None) and pooling.read_pool(raw, 10, 12) == (None, pooling.ERROR_IN_THE_POOL)

    # Nothing is harvested from a proof Lean gave no error position for (the token cap, a timeout), from a known copy, or from a
    # continuing proof whose first error lies in the pool's own lines.
    capped = _job(statement, "  have h : fact_a := by good\n" * 3, pooling.empty_state())
    capped.finish_reason = "length"
    stats, sources, after, row = settled(capped, 0)
    assert stats == {"sent": 0, "pool_checks": 0, "closer_checks": 0} and (row["status"], row["no_harvest"], row["harvested"], row["pool_after"]) == ("capped_tokens", rules.NO_ERROR_POSITION, None, 0)
    stats, sources, after, row = settled(_job(statement, "  slow_step\n", pooling.empty_state()), 0)
    assert (row["status"], row["no_harvest"]) == ("timeout", rules.NO_ERROR_POSITION)
    stats, sources, after, row = settled(_job(statement, "  have h₃ : fact_c := by good\n  bad_step\n", grown, known={"  have h₃ : fact_c := by good\n  bad_step": 1}), 4)
    assert stats == {"sent": 0, "pool_checks": 0, "closer_checks": 0} and (row["status"], row["no_harvest"], row["copy_of"], row["pool_after"]) == (KNOWN_COPY, KNOWN_COPY, 1, 2)
    broken = [{"text": "  have h : fact_a := by bad_lemma", "name": "h", "statement": ": fact_a", "generation": 1}]
    stats, sources, after, row = settled(_job(statement, "  have h₂ : fact_b := by good\n  bad_step\n", {**pooling.empty_state(), "pool": broken, "made_at": 0},
                                              pooling.pool_lines(broken), "resume"), 1)
    assert (row["no_harvest"], row["harvested"], row["cut_line"], stats["pool_checks"]) == (pooling.ERROR_IN_THE_POOLS_LINES, None, 1, 0)


def test_a_pool_whose_prompt_would_not_fit_the_models_context_makes_the_generation_a_blind_attempt(accumulating):
    config = accumulating.config
    config["vllm"]["max_model_len"] = 1024                              # no prompt leaves room for 1,024 new tokens
    ladder_round.ladder_l1_prepare(config)
    report = _run(config)["ladder_l3c_report"]
    rows = _rows(accumulating.store(), ladder_l3a.attempts_file)
    assert not [row for row in rows if row["how"] == "resume"] and not [prompt for prompt, _ in accumulating.engine.calls if "tactic state" in prompt]
    continuing = [row for row in rows if row["kind"] == CONTINUE]
    assert continuing and all((row["how"], row["no_state"], row["had_state"], row["pool_blocks"]) == ("blind", rules.PROMPT_TOO_LONG, False, 0) for row in continuing)
    # Such a generation is the blind arm's own sample, and it used the change up: the next one is fresh until the pool grows again.
    states = report["can_this_run_see_a_win"]["a_state_for_the_continuing_generations"]
    assert (states["with_a_state"], states["passes"], states["without_a_state_by_reason"]) == (0, False, {rules.PROMPT_TOO_LONG: len(continuing)})
    assert report["branch"]["name"] == "INCONCLUSIVE" and "a state came back for 0.0 of the continuing generations" in report["branch"]["reason"]
    # The closers are still tried after the pool: the two arms differ by the proofs they assembled and by nothing else.
    episodes = accumulating.store().read_rows("repair_episodes.jsonl")
    assert all(row["arms"][ACCUMULATE]["resolved_by"] in (None, FRESH, ASSEMBLED) for row in episodes) and any(row["arms"][ACCUMULATE]["resolved_by"] == ASSEMBLED for row in episodes)
    assert all(row["arms"][ACCUMULATE]["resolved_at"] == row["arms"][BLIND]["resolved_at"] for row in episodes if row["arms"][ACCUMULATE]["resolved_by"] != ASSEMBLED
               and (row["arms"][BLIND]["resolved_at"] or 9) <= (row["arms"][ACCUMULATE]["resolved_at"] or 9))


def test_the_pool_holds_at_most_its_size_and_the_episode_at_most_its_closers(accumulating):
    config = accumulating.config
    config["ladder_loop"]["accumulate"].update({"pool_blocks": 2, "kept_closers": 1})
    ladder_round.ladder_l1_prepare(config)
    _run(config)
    store = accumulating.store()
    rows, pools = _rows(store, ladder_l3a.attempts_file), _rows(store, ladder_l3c.pools_file)
    assert pools and max(len(row["blocks"]) for row in pools) == 2 and any(row["harvest_over_the_size"] for row in rows if row["harvested"] is not None)
    # A full pool takes no more: no pool state is made, so nothing is checked, and the generation after it is fresh.
    full = [row for row in rows if row["harvest_over_the_size"] and row["harvested"] == 0]
    assert full and all(row["pool_after"] == 2 for row in full)
    by_episode = Counter((row["problem_id"], row["episode"]) for row in rows if row["kept_closer"])
    assert by_episode and max(by_episode.values()) == 1


def test_the_pilot_takes_the_first_hard_problems_in_a_run_and_with_a_seed_of_its_own(accumulating, monkeypatch):
    config = accumulating.config
    config["ladder_loop"]["accumulate"].update({"pilot_problems": 1, "pilot_episodes": 3})
    ladder_round.ladder_l1_prepare(config)
    real = ladder_l3c.sizes(config)
    monkeypatch.setenv(ladder_l3c.L3C_PILOT_VARIABLE, "1")
    own = ladder_l3c.sizes(config)
    assert own["episodes"] == {"goal": 3, "below": 3, "in": 0, "above": 0} and own["pilot_problems"] == 1 and real["pilot_problems"] is None
    assert own["sampling_seed"] == SEED + 1 == 1000 + ladder_l3c.PILOT_SEED_PLACE and real["sampling_seed"] == SEED       # no generation of the run is seen in the pilot
    assert ladder_l3c.the_check() is ladder_l3c.PILOT and ladder_l3c._store(config).root.name == "ladder_l3c_pilot_seed0"
    summaries = _run(config, "ladder_l3c_pilot")
    store = accumulating.store()
    problems = store.read_rows("problems.jsonl")
    source = ladder_l3a._rows(ladder_l3c.source_directory(config) / "problems.jsonl")
    groups = {row["problem_id"]: row["group"] for row in ladder_l3a._rows(ladder_l3c.source_directory(config) / "heldout_groups.jsonl")}
    wanted = [next(row["problem_id"] for row in source if groups.get(row["problem_id"]) == group and row["set"] in ("reach_base", "rungs_base")) for group in ("goal", "below")]
    assert [row["problem_id"] for row in problems] == wanted and {row["episodes"] for row in problems} == {3}       # the first of G and the first below the band
    assert summaries["ladder_l3c_prepare"]["first_attempts"] == 6 and summaries["ladder_l3c_prepare"]["rungs"] == {"below": 1, "in": 0, "above": 0}
    assert summaries["ladder_l3c_attempts"]["pilot"] is True and store.is_done("ladder_l3c_prepare") and store.is_done(ladder_l3a.loop_marker(7, ladder_l3c.L3C))
    report = summaries["ladder_l3c_report"]
    assert report["pilot"] is True and report["headline"].startswith("L3c PILOT seed 0: ") and "THE PILOT'S EPISODES ARE NOT READ AS A RESULT" in report["headline"]
    assert not (store.root.parent / "ladder_l3c_seed0").exists()                       # the pilot never marks the run's own directory
    # The run itself, afterwards, is another directory with every problem and no generation of the pilot's.
    monkeypatch.delenv(ladder_l3c.L3C_PILOT_VARIABLE)
    calls = set(accumulating.engine.calls)
    _run(config)
    assert ladder_l3c._store(config).root.name == "ladder_l3c_seed0" and len(ladder_l3c._store(config).read_rows("problems.jsonl")) == 6
    assert not {seed for _, seed in calls} & {seed for _, seed in set(accumulating.engine.calls) - calls}


def test_a_run_keeps_the_sizes_and_the_seed_it_was_prepared_with(accumulating, monkeypatch):
    config = accumulating.config
    ladder_round.ladder_l1_prepare(config)
    ladder_l3c.ladder_l3c_prepare(config)
    for name, value in (("episodes_below", 7), ("pool_blocks", 6), ("generations", 5), ("kept_closers", 2)):
        kept = config["ladder_loop"]["accumulate"][name]
        config["ladder_loop"]["accumulate"][name] = value
        with pytest.raises(RuntimeError, match="a run keeps the sizes it began with"):
            ladder_l3c.ladder_l3c_attempts(config)
        with pytest.raises(RuntimeError, match="a run keeps the sizes it began with"):
            ladder_l3c.ladder_l3c_prepare(config)
        config["ladder_loop"]["accumulate"][name] = kept
    with pytest.raises(RuntimeError, match="needs the step ladder_l3c_attempts of this run, which is not done: the stage `ladder_l3c`"):
        ladder_l3c.ladder_l3c_report(config)
    config["ladder_loop"]["accumulate"]["generations"] = 1
    with pytest.raises(ValueError, match="a first generation and at least one more"):
        ladder_l3c.sizes(config)
    config["ladder_loop"]["accumulate"]["generations"] = 8
    # Another seed is another run directory, reading that seed's L1 run, with sampling seeds of its own.
    monkeypatch.setenv("RLVR_LEAN_TRAINING_SEEDS", "2")
    assert ladder_l3c._store(config).root.name == "ladder_l3c_seed2" and ladder_l3c.source_directory(config).name == "ladder_l1_seed2"
    assert ladder_l3c.sizes(config)["sampling_seed"] == 1232
    monkeypatch.setenv(ladder_l3c.L3C_RUN_VARIABLE, "ladder_l3c_smoke")
    monkeypatch.setenv(ladder_l3c.L3C_SOURCE_VARIABLE, "ladder_l1_smoke")
    assert ladder_l3c._store(config).root.name == "ladder_l3c_smoke" and ladder_l3c.source_directory(config).name == "ladder_l1_smoke"
    # The other checks' variables do not move this stage, and this stage's do not move them.
    assert ladder_l3a._store(config).root.name == "ladder_l3a_seed2" and ladder_l3a2._store(config).root.name == "ladder_l3a2_seed2"
    # An episode's generations are this stage's own: `episode.loops`, which L3a and L3a2 read, is not.
    assert ladder_l3c.sizes(config)["generations"] == 8 != config["ladder_loop"]["episode"]["loops"] == ladder_l3a.sizes(config)["loops"]


def test_the_sampling_seed_is_the_stages_own_and_no_generation_is_one_of_l3as_or_l3a2s(accumulating):
    config = accumulating.config
    base = ladder_l3c.sizes(config)["sampling_seed"]
    others = set(ladder_l2.sampling_seeds(config).values()) | {ladder_round.sampling_seed(config, name) for name in ladder_round.SAMPLED_SETS}
    others |= {ladder_l3a.sizes(config)["sampling_seed"], ladder_l3a2.sizes(config)["sampling_seed"]}
    assert base == 1000 + ladder_l3c.SEED_PLACE == SEED and base not in others and base + 1 not in others
    assert len({ladder_l3a.SEED_PLACE, ladder_l3a2.SEED_PLACE, ladder_l3c.SEED_PLACE, ladder_l3c.PILOT_SEED_PLACE}) == 4
    places = [(problem, side, episode, loop) for problem in ("p", "q") for side in ("statement", "negation") for episode in range(6) for loop in range(8)]
    own = {ladder_l3a.attempt_seed(base, *place) for place in places}
    earlier = {ladder_l3a.attempt_seed(other, *place) for other in (ladder_l3a.sizes(config)["sampling_seed"], ladder_l3a2.sizes(config)["sampling_seed"], base + 1)
               for place in places}
    assert len(own) == len(places) and not own & earlier and min(own) >= 2 ** 31 > max(others)


def test_a_proof_on_the_side_a_certificate_contradicts_stops_the_step_and_leaves_its_evidence(accumulating):
    config = accumulating.config
    config["ladder_loop"]["episode"]["contradicted_side"] = "all"       # every problem's ruled-out side is attempted and checked
    ladder_round.ladder_l1_prepare(config)
    known_true = next(row["problem_id"] for row in ladder_l3a._rows(ladder_l3c.source_directory(config) / "heldout_groups.jsonl") if row["side"] == "true")
    accumulating.pool.alarm = known_true
    ladder_l3c.ladder_l3c_prepare(config)
    with pytest.raises(SoundnessAlarm, match=f"{known_true} is known true"):
        ladder_l3c.ladder_l3c_attempts(config)
    store = accumulating.store()
    assert not store.is_done(ladder_l3a.loop_marker(0, ladder_l3c.L3C)) and not store.is_done(ladder_l3c.ATTEMPTS)
    audited = [row for row in store.read_rows(ladder_l3a.attempts_file(0)) if row["audit"]]
    assert any(row["status"] == "verified" and row["problem_id"] == known_true and row["side"] == "negation" for row in audited)
    # Without the alarm the ruled-out side's attempts are first generations only: checked, never continued, and nothing is harvested from them.
    accumulating.pool.alarm = None
    shutil.rmtree(store.root)
    _run(config)
    store = accumulating.store()
    rows = _rows(store, ladder_l3a.attempts_file)
    audited = [row for row in rows if row["audit"]]
    assert audited and all((row["loop"], row["shared"], row["harvested"], row["no_harvest"], row["kept_closer"], row["pool_after"]) == (0, False, None, None, None, None)
                           and row["status"] != "verified" for row in audited)
    assert len(store.read_rows("repair_episodes.jsonl")) == 12 + 2 * 12 + 3 * 6
    ruled_out = {row["problem_id"] if row["side"] == "false" else f"negation_of_{row['problem_id']}" for row in store.read_rows("problems.jsonl")}
    later = [source for source in accumulating.pool.sources if STATE_FILE.search(source)] + [
        build_proof_source(ladder_l3a.statement_of({row["problem_id"]: row for row in store.read_rows("problems.jsonl")}[check["problem_id"]], check["side"]), check["proof"])
        for check in _rows(store, ladder_l3c.closers_file) if check["sent_to_lean"]]
    assert later and not [source for source in later if re.search(r"^theorem (\S+)", source, re.MULTILINE).group(1) in ruled_out]


def test_the_stages_are_registered_with_a_guard_before_each_gpu_step():
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l3c"]]
    assert [(environment, step) for environment, step, _ in steps] == [
        ("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_l3c_prepare"), ("guard", None), ("gpu", "ladder_l3c_attempts"),
        ("guard", None), ("gpu", "ladder_l3c_report")]
    assert set(ladder_l3c.STEPS) == {"ladder_l3c_prepare", "ladder_l3c_attempts", "ladder_l3c_report"} and all(options == {} for _, _, options in steps)
    assert not set(ladder_l3c.STEPS) & (set(ladder_l3a.STEPS) | set(ladder_l3a2.STEPS))
    assert [ladder_l3c.L3C.step(part) for part in ("prepare", "attempts", "report")] == [ladder_l3c.PREPARE, ladder_l3c.ATTEMPTS, ladder_l3c.REPORT]
    assert [ladder_l3c.PILOT.step(part) for part in ("prepare", "attempts", "report")] == [ladder_l3c.PREPARE, ladder_l3c.ATTEMPTS, ladder_l3c.REPORT]
    for name in ("ladder_l3c_smoke", "ladder_l3c_pilot"):               # the same code path
        other = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES[name]]
        assert [(environment, step) for environment, step, _ in other] == [(environment, step) for environment, step, _ in steps]
    smoke = entry.child_environment("gpu", "key", entry.step_fields(entry.STAGES["ladder_l3c_smoke"][2])[2])
    assert smoke[ladder_l3c.L3C_SOURCE_VARIABLE] == "ladder_l1_smoke" and smoke[ladder_l3c.L3C_RUN_VARIABLE] == "ladder_l3c_smoke" and ladder_l3c.L3C_PILOT_VARIABLE not in smoke
    pilot = entry.child_environment("gpu", "key", entry.step_fields(entry.STAGES["ladder_l3c_pilot"][2])[2])
    assert pilot[ladder_l3c.L3C_PILOT_VARIABLE] == "1" and ladder_l3c.L3C_RUN_VARIABLE not in pilot and ladder_l3c.L3C_SOURCE_VARIABLE not in pilot
    own = entry.child_environment("gpu", "key", entry.step_fields(entry.STAGES["ladder_l3c"][2])[2])
    assert not {ladder_l3c.L3C_PILOT_VARIABLE, ladder_l3c.L3C_RUN_VARIABLE, ladder_l3c.L3C_SOURCE_VARIABLE} & set(own)
    assert ladder_l3a.L3A_RUN_VARIABLE not in smoke and ladder_l3a2.L3A2_RUN_VARIABLE not in smoke and ladder_l3c.L3C_RUN_VARIABLE not in entry.child_environment(
        "gpu", "key", entry.step_fields(entry.STAGES["ladder_l3a2_smoke"][2])[2])


def test_the_steps_run_through_the_gpu_entry_point(accumulating, monkeypatch, tmp_path):
    from rlvr_lean.gpu import __main__ as gpu_main
    from rlvr_lean.gpu import milestone2, pipeline

    ladder_round.ladder_l1_prepare(accumulating.config)
    monkeypatch.setattr(milestone2, "load_config", lambda path: accumulating.config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    for name in (ladder_l3c.PREPARE, ladder_l3c.ATTEMPTS, ladder_l3c.REPORT):
        monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", name, "--out", str(tmp_path / f"{name}.json")])
        assert gpu_main.main() == 0 and json.loads((tmp_path / f"{name}.json").read_text())["ok"] is True
    assert accumulating.store().is_done(ladder_l3c.REPORT)
