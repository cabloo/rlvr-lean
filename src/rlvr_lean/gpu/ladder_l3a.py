"""The ladder loop's L3a: the repair check, no training. Spec: docs/spec/ladder-loop.spec.md, "L3a: the repair
check, no training". Each step is its own process (`rlvr_lean.runner.entry`, stage `ladder_l3a`).

  ladder_l3a_prepare    reads L1's run directory for this seed ON THE BOX (never writes there): G and the three rungs,
                        with their exact negations and their groups. Fixes the sampling seed. No GPU, no Lean.
  ladder_l3a_attempts   the base model, ONE process and one engine. Loop 0: the blind first attempts
                        (`repair.episodes_goal` on every problem of G, `repair.episodes_rungs` on every rung problem),
                        on the side the certificate allows. Loops 1 to `episode.loops` - 1: every episode whose first
                        attempt failed goes on in THREE arms from that same attempt, each to its first verified
                        attempt: blind (a whole proof from the plain prompt), resume with the state (the arm's latest
                        failed proof cut at its first error, Lean's proof state there as a comment), resume without
                        the state (the same cut and kept lines, no comment). Where no goal is left at a cut and the
                        kept lines verify on their own, the loop is TRIMMED: it resolves the episode in the resuming
                        arm with nothing generated (spec item 2a)
  ladder_l3a_report     the read fixed before the run, the two "can this run see a win" checks, the branch

The rules for one failed proof (the cut, the file that asks Lean for the state, the prompt) are
`domain/repair/cut.py`; the read is `domain/repair/read.py`.

ONE LOOP is one unit of work over every open episode: its prompts are generated in chunks, each chunk's proofs go to
Lean as they come, and for every proof that failed and goes on in a resuming arm the state file is sent as soon as its
check is back (one more Lean check). A loop's attempts and the state requests made from them are stored together and
the loop is then marked done: a rerun resumes at the first loop that is not done, and only whole loops are ever read.
Every check goes through ONE client (`LeanCheckPool`).

TRIMMED LOOPS, AND THE READING WITHOUT THEM. When the state file shows no goal left at the cut, the kept lines are
checked on their own (one more Lean check, the file assembled as an attempt's is). If they verify, the next loop of
each resuming arm that goes on from that proof is a `trimmed` row: verified, no prompt, no tokens. The report must also
read the run WITHOUT that rule ("those loops treated as the blind fall-back they would have been"), and what such a
fall-back would have led to cannot be known without running it. So the stage runs it: beside the trimmed row the arm
makes the blind attempt it would have made (reason `no_goals_at_the_cut`) and goes on from there to its first verified
attempt, in rows marked `untrimmed`. They belong to the reading without trimming only; the spec's reading ends at the
trimmed row and counts none of them. No kept-lines check is made for an `untrimmed` row.

RANDOM NUMBERS. Each attempt is one sample with a seed of its own, a hash of the stage's sampling seed, the problem,
the side, the episode and the loop, and NOT of the arm: at the same loop of the same episode the three arms sample
with the same random numbers, and two arms whose prompts are the same text (a resume loop that got no state is a
blind attempt) share one generation and one check. The stage's sampling seed is `round.sampling_seed` + 100 x the
task's seed + `SEED_PLACE`, a place no other measurement has; a hashed seed is never one of their small numbers.

The run directory is its own (`ladder_l3a_seed<seed>`). A verified proof on the side a certificate contradicts stops
the step with the soundness alarm's exit code, as everywhere: that side is attempted (first attempts only) for the
audited share of problems, as in L0 to L2.

TWO CHECKS RUN ON THESE LOOPS. L3a2 (`gpu/ladder_l3a2.py`; spec "L3a2: one repair step after each fresh failure, then
start over") is the same run, first attempts, loop, settling and files with other arms and sizes. What differs is a
`Check`: its name, its arms, which of an arm's loops are repair steps and whether they show the state, its sizes and
its sampling seed, and whether a proof already rejected in its episode is sent to Lean again. L3a's is `L3A`, and every
function here that is given no check runs L3a's.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterator, Sequence

from rlvr_lean.domain.ladder_round.read import GOAL
from rlvr_lean.domain.problem_pool.episodes import (
    CAPPED_TOKENS,
    CHECKED,
    NEGATION_SIDE,
    NOT_CHECKED,
    RUNGS,
    STATEMENT_SIDE,
    VERIFIED,
    contradicted_side,
    raise_on_contradiction,
    side_plan,
)
from rlvr_lean.domain.proving import build_prover_prompt, completion_from_output
from rlvr_lean.domain.repair import ARMS, BLIND, FIRST, RESUME_ARMS, RESUME_WITH_STATE
from rlvr_lean.domain.repair import cut as rules
from rlvr_lean.domain.repair.alternate import KNOWN_COPY, copy_of, rejected
from rlvr_lean.domain.repair.read import BLIND_ATTEMPT, RESUMED, TRIMMED, episode_rows, problem_rows, resolved
from rlvr_lean.domain.verification import VerificationStatus, build_proof_source, find_forbidden_token
from rlvr_lean.domain.verification.pin import lean_pin_from_config
from rlvr_lean.gpu import ladder_loop, pipeline
from rlvr_lean.gpu.ladder_round import BASE, Engines, _stand_in, _write_json, training_seed
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.infrastructure.verification_service import LeanCheckPool, pin_of

L3A_RUN_VARIABLE = "RLVR_LEAN_LADDER_L3A_RUN"            # another run directory than `ladder_l3a_seed<seed>` (a smoke run)
L3A_SOURCE_VARIABLE = "RLVR_LEAN_LADDER_L3A_SOURCE"      # another L1 run directory to read than `ladder_l1_seed<seed>`
PREPARE, ATTEMPTS, REPORT = "ladder_l3a_prepare", "ladder_l3a_attempts", "ladder_l3a_report"
SOURCE_FILES = ("problems.jsonl", "heldout_groups.jsonl", "ladder_l1_prepare.done.json")
# The stage's sampling seed is `round.sampling_seed` + 100 x the task's seed + this place. L1's kinds have places 0 to
# 3, L2's rounds 11 to 13 and its control 20 (`ladder_round.SAMPLING_KINDS`, `ladder_l2`): no measurement has 30.
SEED_PLACE = 30
STATE = "state"                                          # a state request's outcome when a state came back
REJECTED = VerificationStatus.REJECTED_LEXICAL.value
SETTLERS = 32                                            # threads that wait for chunks' checks (they only wait: every chunk of a
                                                         # loop is handed to Lean as soon as it is written)
CONTEXT_MARGIN = 2                                       # tokens kept free beside a prompt and its new tokens
STORED_ERRORS = 4                                        # errors kept on an attempt's row, the first by position first


# ----------------------------------------------------------------------------------------------- the check
@dataclasses.dataclass(frozen=True)
class Check:
    """What differs between the checks that run on this module's loops (this module's docstring): L3a's is `L3A`,
    below `sizes`; L3a2's is `ladder_l3a2.L3A2`."""
    name: str                                   # the stage: its steps, its loops' markers and its run directory carry the name
    title: str                                  # what the spec calls it, for a refusal's words
    run_variable: str                           # another run directory than `<name>_seed<seed>` (a smoke run)
    source_variable: str                        # another L1 run directory to read than `ladder_l1_seed<seed>`
    sizes: Callable[[dict], dict]               # what a run is made with: the episodes, the attempts of one, the caps, the seed
    episodes: Callable[[dict, str], int]        # the first attempts a problem gets, from those sizes and the problem's group
    arms: tuple[str, ...]                       # the arms that go on from a failed first attempt
    repair_arms: tuple[str, ...]                # those of them with repair steps: attempts that start from a failed proof's cut
    repairs: Callable[[str, int], bool]         # whether an arm's attempt at a loop is a repair step, from the arm's attempt before it
    shows_state: Callable[[str], bool]          # whether an arm's repair step is prompted with Lean's state at the cut
    known_copies: bool = False                  # a proof Lean already rejected in its episode is not sent again (`alternate.py`)

    def step(self, part: str) -> str:
        """The name of one of the stage's steps (`prepare`, `attempts`, `report`): its marker in the run too."""
        return f"{self.name}_{part}"


# ------------------------------------------------------------------------------------------------- the run
def _runs(config: dict) -> Path:
    return pipeline.STORE / lean_pin_from_config(ladder_loop.ladder_config(config)).runs_directory


def _store(config: dict, check: Check | None = None) -> ArtifactStore:
    check = check or L3A
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    run = os.environ.get(check.run_variable) or f"{check.name}_seed{training_seed(config)}"
    return ArtifactStore(_runs(config) / run, Path(mirror) if mirror else None)


def source_directory(config: dict, check: Check | None = None) -> Path:
    """L1's run directory for this seed. It is only ever READ, with plain file reads: nothing is created in it."""
    return _runs(config) / (os.environ.get((check or L3A).source_variable) or f"ladder_l1_seed{training_seed(config)}")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sizes(config: dict) -> dict:
    """What a run is made with: the episodes a problem gets, the attempts of an episode, the caps, the seed."""
    ladder = config["ladder_loop"]
    repair, episode = ladder["repair"], ladder["episode"]
    if episode["loops"] < 2:
        raise ValueError(f"ladder_loop.episode.loops is {episode['loops']}: an episode is a first attempt and at least one more")
    return {"episodes_goal": repair["episodes_goal"], "episodes_rungs": repair["episodes_rungs"], "loops": episode["loops"],
            "max_new_tokens": episode["max_new_tokens"], "lean_seconds": episode["lean_seconds"],
            "sampling_seed": ladder["round"]["sampling_seed"] + 100 * training_seed(config) + SEED_PLACE}


# L3a: three arms; the two resuming arms repair at EVERY loop, each from its own latest failed proof, one of them
# without the state; every proof is sent to Lean, a copy of a rejected one too.
L3A = Check(name="ladder_l3a", title="L3a", run_variable=L3A_RUN_VARIABLE, source_variable=L3A_SOURCE_VARIABLE, sizes=sizes,
            episodes=lambda own, group: own["episodes_goal"] if group == GOAL else own["episodes_rungs"],
            arms=ARMS, repair_arms=RESUME_ARMS, repairs=lambda arm, loop: arm in RESUME_ARMS, shows_state=lambda arm: arm == RESUME_WITH_STATE)


def attempt_seed(sampling_seed: int, problem_id: str, side: str, episode: int, loop: int) -> int:
    """The seed one attempt is sampled with. It does not depend on the arm: the three arms sample the same loop of
    the same episode with the same random numbers. At least 2^31, so never a seed another measurement uses."""
    digest = hashlib.sha256(f"{sampling_seed}:repair:{problem_id}:{side}:{episode}:{loop}".encode()).hexdigest()
    return (1 << 31) + int(digest[:8], 16) % (1 << 31)


def attempt_id(problem_id: str, arm: str, side: str, episode: int, loop: int) -> str:
    return f"{problem_id}#repair#{arm}#{side}#{episode}#{loop}"


def _need(store: ArtifactStore, marker: str, what: str, check: Check | None = None) -> None:
    if not store.is_done(marker):
        raise RuntimeError(f"{what} needs the step {marker} of this run, which is not done: the stage `{(check or L3A).name}` runs the steps in order")


def _same_sizes(config: dict, store: ArtifactStore, check: Check | None = None) -> dict:
    """The run's sizes, refused when they are not the ones it was prepared with: its loops are stored by their
    number, and a run resumed with other sizes or another seed would mix two measurements."""
    check = check or L3A
    now, prepared = check.sizes(config), store.done_summary(check.step("prepare"))["sizes"]
    if now != prepared:
        raise RuntimeError(f"{store.root} was prepared with {prepared} and the config now gives {now}: a run keeps the sizes it began with")
    return now


def loop_marker(loop: int, check: Check | None = None) -> str:
    return f"{(check or L3A).name}_loop_{loop}"


def attempts_file(loop: int) -> str:
    return f"repair_attempts_{loop}.jsonl"


def states_file(loop: int) -> str:
    return f"repair_states_{loop}.jsonl"


# ------------------------------------------------------------------------------------------------- prepare
def allowed_side(problem: dict, plan: dict) -> str | None:
    """The side of a problem its certificate allows, when the problem has it (a known-false problem whose exact
    negation was not built has none: it cannot be resolved and gets no episode)."""
    ruled_out = contradicted_side(problem["side"])
    return next((side for side in (STATEMENT_SIDE, NEGATION_SIDE) if side in plan and side != ruled_out), None)


def statement_of(problem: dict, side: str) -> str:
    return problem["statement"] if side == STATEMENT_SIDE else problem["negation"]


def prepare_run(config: dict, check: Check) -> dict:
    """The prepare step of a check: L1's run read and checked, the problems written with their groups and with the
    first attempts each gets, the sizes and the sampling seed fixed."""
    store, marker = _store(config, check), check.step("prepare")
    if store.is_done(marker):
        _same_sizes(config, store, check)
        return store.done_summary(marker)
    own, seed, source = check.sizes(config), training_seed(config), source_directory(config, check)
    missing = [name for name in SOURCE_FILES if not (source / name).exists()]
    if missing:
        raise RuntimeError(
            f"{source} does not hold {missing}. "
            f"{check.title} reads the run directory the L1 task for seed {seed} wrote on this box (stage `ladder_l1` with --seeds {seed}: "
            f"`python -m rlvr_lean.runner.entry --stage ladder_l1 --seeds {seed}`; for the smoke run, stage `ladder_l1_smoke`). "
            "Run that task to its end first; nothing was written.")
    groups = _rows(source / "heldout_groups.jsonl")
    group_of = {row["problem_id"]: row["group"] for row in groups}
    problems = _rows(source / "problems.jsonl")
    goal = [row for row in problems if row["set"] == f"reach_{BASE}"]
    rungs = [row for row in problems if row["set"] == f"rungs_{BASE}"]
    misplaced = [row["problem_id"] for row in goal if group_of.get(row["problem_id"]) != GOAL]
    misplaced += [row["problem_id"] for row in rungs if group_of.get(row["problem_id"]) not in RUNGS]
    if misplaced or not goal + rungs:
        raise RuntimeError(f"{source} does not hold G and the rungs as its held-out groups place them ({len(goal)} goal problems, {len(rungs)} rung "
                           f"problems, {len(misplaced)} in another group than their set; first: {misplaced[:1]}). Nothing was written.")
    episode_settings = config["ladder_loop"]["episode"]
    kept, without_a_side, audited = [], [], 0
    for row in [*goal, *rungs]:
        plan = side_plan(row, episode_settings, own["sampling_seed"])
        if allowed_side(row, plan) is None:
            without_a_side.append(row["problem_id"])
            continue
        audited += contradicted_side(row["side"]) in plan
        kept.append({**row, "set": "repair", "group": group_of[row["problem_id"]], "episodes": check.episodes(own, group_of[row["problem_id"]])})
    store.write_rows("problems.jsonl", kept)
    store.write_rows("heldout_groups.jsonl", groups)
    by_group = Counter(row["group"] for row in kept)
    summary = {"seed": seed, "source_run": str(source), "fixture": bool(json.loads((source / "ladder_l1_prepare.done.json").read_text()).get("fixture")),
               "stand_in_engine": _stand_in(), "lean_pin": config["ladder_loop"]["lean_pin"], "model": "the base model: nothing is trained",
               "sizes": own, "goal_set": by_group[GOAL], "rungs": {name: by_group[name] for name in RUNGS}, "problems": len(kept),
               "first_attempts": sum(row["episodes"] for row in kept),
               "problems_with_no_side_to_attempt": without_a_side,
               "contradicted_side_setting": episode_settings.get("contradicted_side", "all"),
               "problems_whose_contradicted_side_is_attempted_too": audited}
    store.mark_done(marker, summary)
    return summary


def ladder_l3a_prepare(config: dict) -> dict:
    return prepare_run(config, L3A)


# -------------------------------------------------------------------------------------------- Lean, engine
@contextlib.contextmanager
def lean_pool(config: dict) -> Iterator[LeanCheckPool]:
    """ONE client for all the checks of the step (attempts and state files): the requests in flight are
    `ladder_loop.lean_in_flight`, and the Lean limit is an episode's."""
    pool = LeanCheckPool(ladder_loop._lean_settings(config, lean_seconds=config["ladder_loop"]["episode"]["lean_seconds"]))
    try:
        yield pool
    finally:
        pool.close()


def engine_kit() -> Callable:
    """The base model's sampling engine, built at first use (a rerun of a finished step loads nothing)."""
    return Engines(enable_lora=False).kit()


def token_counter(engine) -> Callable[[str], int]:
    """How many tokens a prompt is to the engine. The stand-in engine has no tokenizer: four characters a token,
    as its own samples are counted."""
    get_tokenizer = getattr(engine, "get_tokenizer", None)
    counts: dict[str, int] = {}
    if get_tokenizer is None:
        count = lambda text: max(1, len(text) // 4)      # noqa: E731
    else:
        tokenizer = get_tokenizer()
        count = lambda text: len(tokenizer.encode(text))      # noqa: E731

    def cached(text: str) -> int:
        if text not in counts:
            counts[text] = count(text)
        return counts[text]

    return cached


def checked(pool, sources: dict[str, str], episode_settings: dict) -> dict[str, dict]:
    """The pool's raw answer for every Lean file (the answers carry positions, which a classified result does
    not). As in `ladder_loop.settle_block`: a file whose Lean HEADER timed out is asked once more, and a file the
    POOL did not take is asked again up to `episode.pool_refusal_rounds` times, then `LeanPoolRefused`: nothing
    of the loop is recorded and a rerun resumes there."""
    if not sources:
        return {}
    pin = pin_of(pool.settings)

    def ask(keys: list[str]) -> dict[str, dict]:
        return {raw["id"]: raw for raw in pool.submit_sources({key: sources[key] for key in keys}).result()}

    def those(test: Callable) -> list[str]:
        return [key for key, raw in answers.items() if test(pin.classify(key, raw))]

    answers = ask(list(sources))
    cold = those(ladder_loop._header_timed_out)
    if cold:
        answers.update(ask(cold))
    rounds = episode_settings.get("pool_refusal_rounds", 2)
    for _ in range(rounds):
        refused = those(ladder_loop._pool_refused)
        if not refused:
            break
        answers.update(ask(refused))
    refused = those(ladder_loop._pool_refused)
    if refused:
        raise ladder_loop.LeanPoolRefused(f"the Lean pool did not take {len(refused)} of {len(sources)} checks of a chunk, asked again {rounds} times "
                                          f"(first: {str(answers[refused[0]].get('error'))[:200]}): nothing was recorded for the loop, run the step again")
    return answers


# -------------------------------------------------------------------------------------------------- a loop
@dataclasses.dataclass
class Job:
    """One generation and its check, and every (arm, episode) it is an attempt of."""
    prompt: str
    seed: int
    statement: str                              # the theorem of the attempted side, up to `:= by`
    kept: tuple[str, ...]                       # the proof lines the prompt holds (none for a blind attempt)
    check: bool                                 # False: sampled and not sent (a side that is generated only)
    wants_state: bool                           # an attempt of an episode that goes on in a resuming arm
    consumers: list[dict] = dataclasses.field(default_factory=list)
    prompt_tokens: int = 0
    completion: str = ""
    token_count: int = 0
    finish_reason: str | None = None
    proof: str = ""
    status: str | None = None
    seconds: float | None = None
    sent: bool = False
    errors: list[dict] = dataclasses.field(default_factory=list)
    detail: str = ""
    request: dict | None = None                 # the state request made from this attempt, when it failed and goes on

    @functools.cached_property
    def key(self) -> str:
        return hashlib.sha256(f"{self.seed}\0{self.prompt}".encode()).hexdigest()[:24]


def _add(jobs: dict[tuple[str, int], Job], prompt: str, seed: int, statement: str, kept: tuple[str, ...], check: bool, wants_state: bool,
         consumer: dict) -> None:
    job = jobs.setdefault((prompt, seed), Job(prompt, seed, statement, kept, check, wants_state))
    job.consumers.append(consumer)
    job.wants_state = job.wants_state or wants_state


def first_jobs(config: dict, problems: list[dict], own: dict, check: Check | None = None) -> list[Job]:
    """Loop 0: every problem's blind first attempts on the side its certificate allows and, for the audited
    share of problems, on the side it rules out (checked, never continued: a verified proof there is the alarm)."""
    check = check or L3A
    episode_settings, jobs = config["ladder_loop"]["episode"], {}
    for problem in problems:
        plan = side_plan(problem, episode_settings, own["sampling_seed"])
        allowed = allowed_side(problem, plan)
        for side in (STATEMENT_SIDE, NEGATION_SIDE):
            if side not in plan:
                continue
            statement = statement_of(problem, side)
            for episode in range(problem["episodes"]):
                consumer = {"arm": FIRST, "problem_id": problem["problem_id"], "group": problem["group"], "side": side, "episode": episode,
                            "audit": side != allowed, "untrimmed": False, "how": BLIND_ATTEMPT, "had_state": False, "no_state": None,
                            "kept_lines": 0, "cut": None, "failed_step": None}
                if check.known_copies:
                    consumer["rejected"] = {}           # nothing was rejected before an episode's first attempt
                _add(jobs, build_prover_prompt(statement), attempt_seed(own["sampling_seed"], problem["problem_id"], side, episode, 0), statement, (),
                     plan[side] == CHECKED, own["loops"] > 1 and side == allowed and any(check.repairs(arm, 1) for arm in check.arms), consumer)
    return list(jobs.values())


def trimmed_row(problem: dict, side: str, episode: int, arm: str, loop: int, previous: dict, request: dict) -> dict:
    """Spec item 2a: the loop of a resuming arm whose kept lines verified on their own. It resolves the episode
    with nothing generated; its one check (the kept lines, assembled as an attempt's proof is) was made when the
    state file showed no goal left at the cut."""
    kept = rules.proof_lines(previous["proof"])[:request["kept_lines"]]
    return {"attempt_id": attempt_id(problem["problem_id"], arm, side, episode, loop) + "#trimmed", "arm": arm, "problem_id": problem["problem_id"],
            "group": problem["group"], "side": side, "episode": episode, "loop": loop, "audit": False, "untrimmed": False,
            "how": TRIMMED, "had_state": False, "no_state": None, "kept_lines": len(kept), "cut": request["cut"], "repeats_failed_step": None,
            "prompt_tokens": 0, "token_count": 0, "finish_reason": None, "completion": "", "proof": rules.trimmed_proof(kept),
            "status": VERIFIED, "seconds": request["trim_seconds"], "sent_to_lean": True,
            "first_error": "", "first_error_line": None, "first_error_column": None, "errors": [], "cut_line": None, "seed": None, "job": None}


def next_jobs(config: dict, problems: list[dict], own: dict, loop: int, attempts: dict[int, list[dict]], states: list[dict],
              count_tokens: Callable[[str], int], check: Check | None = None) -> tuple[list[Job], list[dict]]:
    """Loop `loop` (1 or later): (the generations, the trimmed rows). One attempt for every arm of every episode
    that arm has not resolved. A resuming arm cuts ITS latest failed proof (at loop 1, the shared first attempt):
    when the state request made from it gave a state and the prompt with that state fits the model's context, both
    resuming arms resume from the kept lines (one with the state comment, one without). When it showed no goal
    left and the kept lines verified, the loop is a trimmed row and nothing is generated for it. Otherwise the
    loop is a blind attempt, counted as "no state" with its reason.

    An arm whose episode a trimmed row resolved goes on in the reading WITHOUT trimming (rows marked `untrimmed`):
    its trimmed loop as the blind fall-back it would have been, then on from there (this module's docstring).

    That is L3a, whose resuming arms repair at every loop. Another check says which of an arm's loops are repair
    steps (`check.repairs`; any other loop of the arm is a whole proof from the plain prompt) and whether a step
    shows the state; with `check.known_copies` every attempt carries the proofs Lean has already rejected in its
    arm of the episode, and `settle_chunk` does not send one of those again."""
    check = check or L3A
    max_new, context = own["max_new_tokens"], config["vllm"]["max_model_len"]
    firsts = {(row["problem_id"], row["episode"]): row for row in attempts[0] if not row["audit"]}
    earlier = {(row["arm"], row["problem_id"], row["episode"], row["loop"]): row for index in range(1, loop) for row in attempts[index] if not row["untrimmed"]}
    beside = {(row["arm"], row["problem_id"], row["episode"], row["loop"]): row for index in range(1, loop) for row in attempts[index] if row["untrimmed"]}
    asked = {(row["arm"], row["problem_id"], row["episode"]): row for row in states}
    jobs: dict[tuple[str, int], Job] = {}
    trimmed: list[dict] = []
    for problem in problems:
        for episode in range(problem["episodes"]):
            first = firsts[problem["problem_id"], episode]
            if first["status"] == VERIFIED:
                continue
            side = first["side"]
            statement = statement_of(problem, side)
            plain = build_prover_prompt(statement)
            seed = attempt_seed(own["sampling_seed"], problem["problem_id"], side, episode, loop)
            for arm in check.arms:
                own_rows = [earlier[key] for index in range(1, loop) if (key := (arm, problem["problem_id"], episode, index)) in earlier]
                trimmed_at = next((row["loop"] for row in own_rows if row["how"] == TRIMMED), None)
                untrimmed = trimmed_at is not None
                if untrimmed:           # the episode is resolved; what goes on is the reading without trimming
                    own_rows = own_rows[:trimmed_at - 1] + [beside[key] for index in range(trimmed_at, loop)
                                                            if (key := (arm, problem["problem_id"], episode, index)) in beside]
                if any(row["status"] == VERIFIED for row in own_rows):
                    continue
                if len(own_rows) != loop - 1:
                    raise RuntimeError(f"{problem['problem_id']} episode {episode}: the {arm} arm has {len(own_rows)} stored attempts before loop "
                                       f"{loop} and none verified: a stored loop is not whole")
                consumer = {"arm": arm, "problem_id": problem["problem_id"], "group": problem["group"], "side": side, "episode": episode, "audit": False,
                            "untrimmed": untrimmed, "how": BLIND_ATTEMPT, "had_state": False, "no_state": None, "kept_lines": 0, "cut": None,
                            "failed_step": None}
                if check.known_copies:
                    consumer["rejected"] = rejected([first, *own_rows])
                prompt, kept = plain, ()
                if check.repairs(arm, loop):
                    previous = own_rows[-1] if own_rows else first
                    request = asked[FIRST if loop == 1 else arm, problem["problem_id"], episode]
                    outcome = request["outcome"]
                    if outcome == TRIMMED:
                        if not untrimmed:
                            trimmed.append(trimmed_row(problem, side, episode, arm, loop, previous, request))
                            if check.known_copies:
                                trimmed[-1]["copy_of"] = None       # nothing was generated: the row has the key every attempt's has
                            consumer["untrimmed"] = True
                        outcome = rules.NO_GOALS_AT_THE_CUT         # without trimming this loop is the blind fall-back
                    if outcome != STATE:
                        consumer["no_state"] = outcome
                    else:
                        lines = tuple(rules.proof_lines(previous["proof"])[:request["kept_lines"]])
                        with_state = rules.resume_prompt(statement, lines, request["indentation"], request["state"])
                        if count_tokens(with_state) + max_new + CONTEXT_MARGIN > context:
                            consumer["no_state"] = rules.PROMPT_TOO_LONG
                        else:
                            kept = lines
                            prompt = with_state if check.shows_state(arm) else rules.resume_prompt(statement, lines, request["indentation"], None)
                            consumer.update({"how": RESUMED, "had_state": check.shows_state(arm), "kept_lines": len(lines), "cut": request["cut"],
                                             "failed_step": request["failed_step"]})
                # A state is asked for the proof when the arm's NEXT loop is a repair step, which would start from it.
                _add(jobs, prompt, seed, statement, kept, True, loop + 1 < own["loops"] and check.repairs(arm, loop + 1), consumer)
    return list(jobs.values()), trimmed


def generate_each(engine, prompts: list[str], parameters: list) -> list:
    """One sample for each prompt, each with its own sampling parameters (its own seed): vLLM pairs a list of
    parameters with the prompts one by one."""
    outputs = engine.generate(prompts, parameters, lora_request=None)
    if len(outputs) != len(prompts) or any(len(output.outputs) != 1 for output in outputs):
        raise RuntimeError(f"the engine did not return one sample for each of {len(prompts)} prompts")
    return outputs


def _goes_on(job: Job, repair_arms: Sequence[str]) -> list[dict]:
    """The (arm, episode)s of a job whose next loop may start from its proof: the ones a state request is for."""
    return [consumer for consumer in job.consumers if not consumer["audit"] and consumer["arm"] in (FIRST, *repair_arms)]


def _is_known_copy(job: Job) -> bool:
    """A check with known copies (L3a2): "a proof whose text equals one already rejected in the same episode is not
    sent to Lean". Says, on each (arm, episode) the job is an attempt of, which attempt of THAT episode the proof
    repeats (`copy_of`: its loop, or None), and returns whether it repeats one in every one of them: only then is
    nothing sent. An attempt of L3a carries no rejected proofs and repeats none."""
    for consumer in job.consumers:
        consumer["copy_of"] = copy_of(job.proof, consumer.get("rejected"))
    return all(consumer["copy_of"] is not None for consumer in job.consumers)


def settle_chunk(pool, chunk: list[Job], episode_settings: dict, repair_arms: Sequence[str] = RESUME_ARMS) -> dict:
    """Check a generated chunk and, for every proof that failed and goes on in a resuming arm, ask Lean for the
    state at its cut; where that shows no goal left, check the kept lines on their own (spec item 2a). An attempt
    that reached the token cap is a failure and is not sent; nor is one the lexical filter rejects; nor, in a check
    with known copies, one whose proof Lean already rejected in its episode. Each Lean file is sent once a chunk,
    whoever shares it."""
    pin = pin_of(pool.settings)
    sources = {}
    for job in chunk:
        job.proof = rules.resumed_proof(job.kept, job.completion)
        if job.finish_reason == "length":
            job.status = CAPPED_TOKENS
        elif not job.check:
            job.status = NOT_CHECKED
        elif find_forbidden_token(job.proof) is not None:
            job.status, job.detail = REJECTED, f"the proof contains the forbidden token {find_forbidden_token(job.proof)!r}"
        elif _is_known_copy(job):
            job.status = KNOWN_COPY
        else:
            sources[job.key] = build_proof_source(job.statement, job.proof)
    answers = checked(pool, sources, episode_settings)
    for job in chunk:
        if job.key in answers and job.status is None:
            verdict = pin.classify(job.key, answers[job.key])
            job.status, job.seconds, job.sent = ladder_loop._status(verdict), verdict.verification_seconds, True
            job.errors, job.detail = rules.errors_of(answers[job.key]), verdict.detail
    requests = {}
    for job in chunk:
        if not job.wants_state:
            continue
        going_on = _goes_on(job, repair_arms)
        if going_on and all(consumer.get("copy_of") is not None for consumer in going_on):
            # A known copy for every episode that would repair it: it was not sent for them (whatever a check made for
            # another arm says), so Lean said nothing a cut could be made by. The repair step is a blind attempt.
            job.request = {"cut": None, "outcome": KNOWN_COPY, "state": None, "seconds": None, "file": None, "trim_file": None, "trim_status": None,
                           "trim_seconds": None}
            continue
        if job.status == VERIFIED:
            continue
        cut, why = rules.plan_cut(job.statement, job.proof, job.errors)
        job.request = {"cut": cut, "outcome": why, "state": None, "seconds": None, "file": None, "trim_file": None, "trim_status": None,
                       "trim_seconds": None}
        if cut is not None:
            source, line, column = rules.state_source(job.statement, cut)
            job.request.update({"file": hashlib.sha256(source.encode()).hexdigest()[:24], "line": line, "column": column})
            requests[job.request["file"]] = source
    answers = checked(pool, requests, episode_settings)
    for job in chunk:
        if job.request is not None and job.request["cut"] is not None:
            raw = answers[job.request["file"]]
            state, why = rules.read_state(raw, job.request["line"], job.request["column"])
            job.request.update({"state": state, "outcome": STATE if why is None else why, "seconds": raw.get("time")})
    # No goal left at the cut: the kept lines are a whole proof with something extra after it, IF they verify. They
    # are checked exactly as an attempt is. Not for a proof only the reading without trimming goes on from.
    trims = {}
    for job in chunk:
        if job.request is not None and job.request["outcome"] == rules.NO_GOALS_AT_THE_CUT and any(
                not consumer["untrimmed"] for consumer in _goes_on(job, repair_arms)):
            source = build_proof_source(job.statement, rules.trimmed_proof(job.request["cut"].kept))
            job.request["trim_file"] = hashlib.sha256(source.encode()).hexdigest()[:24]
            trims[job.request["trim_file"]] = source
    answers = checked(pool, trims, episode_settings)
    for job in chunk:
        if job.request is not None and job.request["trim_file"] is not None:
            verdict = pin.classify(job.request["trim_file"], answers[job.request["trim_file"]])
            job.request.update({"trim_status": ladder_loop._status(verdict), "trim_seconds": verdict.verification_seconds})
            if job.request["trim_status"] == VERIFIED:
                job.request["outcome"] = TRIMMED
    return {"sent": len(sources), "state_files": len(requests), "kept_lines_checks": len(trims)}


def attempt_rows(job: Job, loop: int) -> list[dict]:
    first_line = rules.proof_first_line(job.statement)
    _, error = rules.leading_error(job.statement, job.proof, job.errors)
    cut = job.request["cut"] if job.request else None
    rows = []
    for consumer in job.consumers:
        rows.append({
            "attempt_id": attempt_id(consumer["problem_id"], consumer["arm"], consumer["side"], consumer["episode"], loop),
            "arm": consumer["arm"], "problem_id": consumer["problem_id"], "group": consumer["group"], "side": consumer["side"],
            "episode": consumer["episode"], "loop": loop, "audit": consumer["audit"],
            # True: a row of the reading WITHOUT trimming only (the arm's episode was resolved by a trimmed row).
            "untrimmed": consumer["untrimmed"],
            # How the attempt was prompted: blind, or resumed from kept lines (with a state comment or without).
            "how": consumer["how"], "had_state": consumer["had_state"], "no_state": consumer["no_state"],
            "kept_lines": consumer["kept_lines"], "cut": consumer["cut"],
            "repeats_failed_step": rules.repeats_failed_step(job.completion, consumer["failed_step"]) if consumer["how"] == RESUMED else None,
            "prompt_tokens": job.prompt_tokens, "token_count": job.token_count, "finish_reason": job.finish_reason,
            "completion": job.completion, "proof": job.proof, "status": job.status, "seconds": job.seconds, "sent_to_lean": job.sent,
            # The error the proof is read by (`leading_error`: the first by position in the proof, when it has one); a
            # line is counted in the PROOF (1 is its first line, 0 the theorem's `:= by`).
            "first_error": (error["text"] if error else job.detail)[:300],
            "first_error_line": error["line"] - first_line + 1 if error else None, "first_error_column": error["column"] if error else None,
            "errors": [{**entry, "line": entry["line"] - first_line + 1, "end_line": None if entry["end_line"] is None else entry["end_line"] - first_line + 1,
                        "text": entry["text"][:300]} for entry in job.errors[:STORED_ERRORS]],
            # Where THIS proof is cut for the arm's next loop: the first proof line that is not kept.
            "cut_line": cut.kept_lines + 1 if cut is not None and cut.kind == rules.BODY else None,
            "seed": job.seed, "job": job.key})
        if "rejected" in consumer:
            # A check with known copies (L3a2): the loop of the attempt of this episode whose rejected proof this one
            # repeats, or None. A known copy is a failed attempt that was NOT sent for this arm and episode, whatever
            # a check of the same generation made for another arm says; its tokens stay counted.
            repeated = consumer.get("copy_of")
            rows[-1]["copy_of"] = repeated
            if repeated is not None:
                rows[-1].update({"status": KNOWN_COPY, "seconds": None, "sent_to_lean": False, "first_error_line": None, "first_error_column": None,
                                 "errors": [], "first_error": f"the proof of this episode's attempt {repeated + 1}, which Lean rejected: not sent again"})
    return rows


def state_rows(job: Job, loop: int, repair_arms: Sequence[str] = RESUME_ARMS) -> list[dict]:
    """The state request made from one failed attempt, once for every (arm, episode) that goes on from it."""
    if job.request is None:
        return []
    cut, first_line = job.request["cut"], rules.proof_first_line(job.statement)
    rows = []
    for consumer in _goes_on(job, repair_arms):
        rows.append({
            "arm": consumer["arm"], "problem_id": consumer["problem_id"], "episode": consumer["episode"], "after_loop": loop,
            "requested": cut is not None,           # a state file was sent to Lean: one more check for a resume loop
            # "state", why there is none, or "trimmed": no goal was left at the cut and the kept lines verified.
            "outcome": job.request["outcome"], "cut": cut.kind if cut else None, "kept_lines": cut.kept_lines if cut else None,
            # The kept lines checked on their own (no goal was left at the cut): one more check; None when none was made.
            "trim_status": job.request["trim_status"], "trim_seconds": job.request["trim_seconds"], "trim_file": job.request["trim_file"],
            "cut_line": cut.kept_lines + 1 if cut is not None and cut.kind == rules.BODY else None,
            "indentation": cut.indentation if cut else None, "failed_step": cut.failed_step if cut else None, "moved_back": cut.moved_back if cut else None,
            "error_line": cut.error["line"] - first_line + 1 if cut else None, "error_column": cut.error["column"] if cut else None,
            "error": cut.error["text"][:300] if cut else None,
            "state": job.request["state"], "seconds": job.request["seconds"], "state_file": job.request["file"]})
    return rows


def run_loop(config: dict, store: ArtifactStore, problems: list[dict], own: dict, loop: int, jobs: list[Job], trimmed: list[dict], engine,
             parameters_of: Callable, count_tokens: Callable[[str], int], pool, check: Check | None = None) -> dict:
    """Generate, check and store one loop. The calling thread samples chunk after chunk; each chunk is settled by
    a waiting thread as its checks come back, so Lean works while the model writes the next chunk. The loop's
    attempts (with its `trimmed` rows, for which nothing is generated) are written BEFORE they are judged (an
    alarm leaves its evidence), then its state requests, and only then is the loop marked done."""
    check = check or L3A
    episode_settings = config["ladder_loop"]["episode"]
    chunk_size = max(1, episode_settings.get("chunk_attempts", 512))
    started, generation_seconds, futures = time.monotonic(), 0.0, []
    with ThreadPoolExecutor(max_workers=SETTLERS, thread_name_prefix="repair-settle") as settlers:
        for start in range(0, len(jobs), chunk_size):
            failed = next((future for future in futures if future.done() and future.exception() is not None), None)
            if failed is not None:
                break
            chunk = jobs[start:start + chunk_size]
            for job in chunk:
                job.prompt_tokens = count_tokens(job.prompt)
            began = time.monotonic()
            outputs = generate_each(engine, [job.prompt for job in chunk], [parameters_of(1, job.seed, own["max_new_tokens"]) for job in chunk])
            generation_seconds += time.monotonic() - began
            for job, output in zip(chunk, outputs):
                sample = output.outputs[0]
                job.completion, job.token_count, job.finish_reason = completion_from_output(sample.text), len(sample.token_ids), sample.finish_reason
            futures.append(settlers.submit(settle_chunk, pool, chunk, episode_settings, check.repair_arms))
        generated = time.monotonic()
    settled = [future.result() for future in futures]       # a chunk that failed (a pool that refuses) fails the loop here, unrecorded
    rows = [*trimmed, *(row for job in jobs for row in attempt_rows(job, loop))]
    store.write_rows(attempts_file(loop), rows)
    by_problem: dict[str, list[dict]] = {}
    for row in rows:
        by_problem.setdefault(row["problem_id"], []).append(row)
    for problem in problems:
        raise_on_contradiction(problem, by_problem.get(problem["problem_id"], []))
    requests = [row for job in jobs for row in state_rows(job, loop, check.repair_arms)]
    if loop < own["loops"] - 1:
        store.write_rows(states_file(loop), requests)
    statuses = {arm: dict(Counter(row["status"] for row in rows if row["arm"] == arm)) for arm in (FIRST, *check.arms) if any(row["arm"] == arm for row in rows)}
    generated_tokens = sum(job.token_count for job in jobs)
    stats = {"loop": loop, "attempts": len(rows), "generations": len(jobs), "statuses": statuses,
             # Rows of this loop that are trimmed resolutions, and rows that belong to the reading without trimming only.
             "trimmed": len(trimmed), "untrimmed": sum(bool(row["untrimmed"]) for row in rows),
             "generated_tokens": generated_tokens, "prompt_tokens": sum(job.prompt_tokens for job in jobs),
             "generation_seconds": round(generation_seconds, 1),
             "tokens_per_second": round(generated_tokens / generation_seconds, 1) if generation_seconds else None,
             "attempts_sent_to_lean": sum(entry["sent"] for entry in settled), "state_files_sent_to_lean": sum(entry["state_files"] for entry in settled),
             "kept_lines_checks_sent_to_lean": sum(entry["kept_lines_checks"] for entry in settled),
             "state_requests": len(requests), "state_requests_by_outcome": dict(Counter(row["outcome"] for row in requests).most_common()),
             "wall_seconds": round(time.monotonic() - started, 1), "waited_on_lean_after_the_last_chunk_seconds": round(time.monotonic() - generated, 1)}
    store.mark_done(loop_marker(loop, check), stats)
    print(f"{loop_marker(loop, check)}: {stats['attempts']} attempts in {stats['generations']} generations ({stats['trimmed']} trimmed), {stats['statuses']}, "
          f"states {stats['state_requests_by_outcome']} ({stats['wall_seconds']} s)", flush=True)
    return stats


# ------------------------------------------------------------------------------------------------ attempts
def run_attempts(config: dict, check: Check) -> dict:
    """The whole sampling step of a check: the first attempts, then the loops, with one engine and one Lean client.
    A rerun returns what is stored; a rerun of a step that stopped resumes at the first loop that is not done."""
    store, marker = _store(config, check), check.step("attempts")
    delivered = ("repair_episodes.jsonl", "repair_problems.jsonl")
    if store.is_done(marker):
        for name in delivered:              # a rerun is another task: its out/ gets the results too
            store.mirror(name)
        return store.done_summary(marker)
    _need(store, check.step("prepare"), "the attempts", check)
    own, problems = _same_sizes(config, store, check), store.read_rows("problems.jsonl")
    kit, tools, written = engine_kit(), {}, set()
    attempts: dict[int, list[dict]] = {}
    with lean_pool(config) as pool:
        for loop in range(own["loops"]):
            if not store.is_done(loop_marker(loop, check)):
                if "engine" not in tools:
                    tools["engine"], tools["parameters_of"] = kit(config)
                    tools["count_tokens"] = token_counter(tools["engine"])
                jobs, trimmed = (first_jobs(config, problems, own, check), []) if loop == 0 else next_jobs(
                    config, problems, own, loop, attempts, store.read_rows(states_file(loop - 1)), tools["count_tokens"], check)
                run_loop(config, store, problems, own, loop, jobs, trimmed, tools["engine"], tools["parameters_of"], tools["count_tokens"], pool, check)
                written.add(loop)
            attempts[loop] = store.read_rows(attempts_file(loop))
    tools.clear()
    states = [row for loop in range(own["loops"] - 1) for row in store.read_rows(states_file(loop))]
    for loop in range(own["loops"]):        # what an earlier task's run of this step stored reaches this task's out/ too
        if loop not in written:
            store.mirror(attempts_file(loop))
            store.mirror(states_file(loop))
            store.mirror(f"{loop_marker(loop, check)}.done.json")
    every = [row for loop in range(own["loops"]) for row in attempts[loop]]
    episodes = episode_rows(problems, every, states, own["loops"], check.arms, check.repair_arms)
    store.write_rows(delivered[0], episodes)
    store.write_rows(delivered[1], problem_rows(episodes, own["loops"], check.arms, check.repair_arms))
    loops = [store.done_summary(loop_marker(loop, check)) for loop in range(own["loops"])]
    generated, seconds = sum(entry["generated_tokens"] for entry in loops), sum(entry["generation_seconds"] for entry in loops)
    failed = [row for row in episodes if row["failed_first"]]
    summary = {"seed": training_seed(config), "sizes": own, "problems": len(problems), "episodes": len(episodes), "failed_first_attempts": len(failed),
               "attempts": len(every), "attempts_on_the_contradicted_side": sum(bool(row["audit"]) for row in every),
               "generations": sum(entry["generations"] for entry in loops), "generated_tokens": generated, "generation_seconds": round(seconds, 1),
               "tokens_per_second": round(generated / seconds, 1) if seconds else None,
               "lean_checks_sent": sum(entry["attempts_sent_to_lean"] + entry["state_files_sent_to_lean"] + entry["kept_lines_checks_sent_to_lean"]
                                       for entry in loops),
               "state_files_sent_to_lean": sum(entry["state_files_sent_to_lean"] for entry in loops),
               "kept_lines_checks_sent_to_lean": sum(entry["kept_lines_checks_sent_to_lean"] for entry in loops),
               "episodes_resolved": {arm: sum(resolved(row, arm, own["loops"]) for row in episodes) for arm in check.arms},
               "episodes_resolved_by_a_trimmed_loop": {arm: sum(row["arms"][arm]["trimmed_at"] is not None for row in episodes) for arm in check.repair_arms},
               "attempts_of_the_reading_without_trimming_only": sum(bool(row["untrimmed"]) for row in every),
               # The spec's reading: the rows of the reading without trimming only are counted on the line above.
               "statuses_by_arm": {arm: dict(Counter(row["status"] for row in every if row["arm"] == arm and not row["untrimmed"])) for arm in (FIRST, *check.arms)},
               "loops": loops, "loops_done_in_this_run": sorted(written), "lean_in_flight": config["ladder_loop"].get("lean_in_flight"),
               "stand_in_engine": _stand_in()}
    store.mark_done(marker, summary)
    return summary


def ladder_l3a_attempts(config: dict) -> dict:
    return run_attempts(config, L3A)


# -------------------------------------------------------------------------------------------------- report
def write_report(config: dict, check: Check, build: Callable, name: str) -> dict:
    """The report step of a check: `build` (the check's own report, from the prepare step's summary, the per-episode
    rows and the attempts step's summary) written as `name`."""
    store = _store(config, check)
    _need(store, check.step("attempts"), "the report", check)
    report = build(store.done_summary(check.step("prepare")), store.read_rows("repair_episodes.jsonl"), store.done_summary(check.step("attempts")),
                   config["ladder_loop"], config["evaluation"])
    _write_json(store, name, report)
    # Everything a reader needs beside the report goes out again from here (a rerun is another task with an output
    # directory of its own, and a task that failed hands over no step files): the per-episode and per-problem rows,
    # the problems, the held-out groups, and what each step and loop recorded when it finished. Not the attempts and
    # not the state requests, which are large and which the step that wrote them delivered.
    for stored in ("repair_episodes.jsonl", "repair_problems.jsonl", "problems.jsonl", "heldout_groups.jsonl"):
        store.mirror(stored)
    for stored in sorted(path.name for path in store.root.glob("*.done.json")):
        store.mirror(stored)
    store.mark_done(check.step("report"), {"headline": report["headline"], "branch": report["branch"]})
    return report


def ladder_l3a_report(config: dict) -> dict:
    from rlvr_lean.reporting.ladder_l3a import build_l3a_report

    return write_report(config, L3A, build_l3a_report, "report_ladder_l3a.json")


STEPS = {
    PREPARE: ladder_l3a_prepare,
    ATTEMPTS: ladder_l3a_attempts,
    REPORT: ladder_l3a_report,
}
