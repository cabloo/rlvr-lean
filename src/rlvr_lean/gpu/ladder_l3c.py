"""The ladder loop's L3c: an episode that keeps what verified (no training). Spec: docs/spec/ladder-loop.spec.md,
"L3c: an episode that keeps what verified (no training)" and its "Made exact before the build (2026-10-07)". Each step
is its own process (`rlvr_lean.runner.entry`, stages `ladder_l3c`, `ladder_l3c_pilot`, `ladder_l3c_smoke`).

  ladder_l3c_prepare    reads L1's run directory for this seed ON THE BOX (never writes there): G and the three rungs,
                        with their exact negations and their groups. Fixes the sizes and the sampling seed. No GPU, no
                        Lean.
  ladder_l3c_attempts   the base model, ONE process and one engine. Loop 0 is generation 1: the first attempts
                        (`accumulate.episodes_goal` on every problem of G, `episodes_below` on every problem of the
                        below-band rung, `episodes_rungs` on every problem of the other two), on the side the
                        certificate allows. Loops 1 to `accumulate.generations` - 1: every episode whose first
                        generation failed goes on in TWO arms from that same generation, each to the generation that
                        resolves it: blind (a whole proof from the plain prompt at every loop) and accumulate (the
                        episode keeps a pool of verified lemmas: a generation is fresh, a whole proof from the plain
                        prompt, or it continues from the pool and Lean's state after it; after every failed proof its
                        leading lemmas are harvested, the pool is checked, and the kept closing steps are tried again
                        after it)
  ladder_l3c_report     the read fixed before the run, the three "can this run see a win" checks, the branch

THE RULES of the accumulate arm (the harvest, the pool check's reading, which closers are tried, the kind of a
generation, the episode's state from its stored rows) and the read are `domain/repair/accumulate.py`. The cut at a
proof's first error, the file that asks Lean for a state and the prover's comment format are L3a's
(`domain/repair/cut.py`); the known copies are L3a2's (`domain/repair/alternate.py`).

THE RUN, the prepare step, the first attempts, the sampling of a loop in chunks, the checking of a chunk's proofs, the
attempt rows and the report step are L3a's (`gpu/ladder_l3a.py`, given this stage's `Check`). What is this module's
own is ONE LOOP of an episode that carries a pool. After a chunk's proofs are checked, a waiting thread goes on for
every accumulate episode in it whose proof failed: the harvest, then ONE more Lean check where the pool would change
(the pool and `all_goals sorry`: it stands, and gives the state the next continuing generation is shown, or the harvest
is taken back out), then one for each kept closer that is tried after the pool. A loop's attempts
(`repair_attempts_<loop>.jsonl`), the pool states it made (`repair_pools_<loop>.jsonl`) and its closer checks
(`repair_closers_<loop>.jsonl`) are stored together and the loop is then marked done: a rerun resumes at the first loop
that is not done, and what an episode holds before a loop is read from the stored rows of the loops before it
(`accumulate.episode_state`), never from memory.

RANDOM NUMBERS, as in L3a and L3a2: a generation's seed is a hash of the stage's sampling seed, the problem, the side,
the episode and the loop, and NOT of the arm, and generations whose prompt and seed are the same are ONE sample and one
check. So a fresh generation of the accumulate arm is the blind arm's own sample at that position as long as both have
the episode open, and the arms differ only where the accumulate arm continued (and by what its closers assembled). The
stage's sampling seed is `round.sampling_seed` + 100 x the task's seed + `SEED_PLACE`, one place after L3a2's: no
generation is one of L3a's or of L3a2's.

THE PILOT (`ladder_l3c_pilot`: the variable `L3C_PILOT_VARIABLE`) is the same steps on the first
`accumulate.pilot_problems` problems of G and of the below-band rung, `accumulate.pilot_episodes` episodes each, in a
run directory of its own (`ladder_l3c_pilot_seed<seed>`) and with a sampling seed of its own (`PILOT_SEED_PLACE`), so
that no generation of the run has been seen before the run. It exists because the fixture's problems are too easy for
a smoke run to reach a continuing generation; its episodes are not read as a result.

The run directory is its own (`ladder_l3c_seed<seed>`). A verified proof on the side a certificate contradicts stops
the step with the soundness alarm's exit code, as everywhere: that side is attempted (first generations only) for the
audited share of problems, and nothing is harvested from it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from rlvr_lean.domain.ladder_round.read import GOAL
from rlvr_lean.domain.problem_pool.episodes import RUNGS, VERIFIED, raise_on_contradiction
from rlvr_lean.domain.proving import build_prover_prompt
from rlvr_lean.domain.repair import FIRST
from rlvr_lean.domain.repair import accumulate as pooling
from rlvr_lean.domain.repair import cut as rules
from rlvr_lean.domain.repair.accumulate import ACCUMULATE, ARMS, CONTINUE, FRESH, OWN_PROOF, STANDS
from rlvr_lean.domain.repair.alternate import KNOWN_COPY, checked_text, rejected
from rlvr_lean.domain.repair.read import BLIND_ATTEMPT, HARD, RESUMED, SETS, resolved
from rlvr_lean.domain.verification import build_proof_source, find_forbidden_token
from rlvr_lean.gpu import ladder_l3a, ladder_loop
from rlvr_lean.gpu.ladder_l3a import CONTEXT_MARGIN, REJECTED, Check, Job, attempt_seed, statement_of
from rlvr_lean.gpu.ladder_round import _stand_in, training_seed
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.infrastructure.verification_service import pin_of

L3C_RUN_VARIABLE = "RLVR_LEAN_LADDER_L3C_RUN"            # another run directory than `ladder_l3c_seed<seed>` (a smoke run)
L3C_SOURCE_VARIABLE = "RLVR_LEAN_LADDER_L3C_SOURCE"      # another L1 run directory to read than `ladder_l1_seed<seed>`
L3C_PILOT_VARIABLE = "RLVR_LEAN_LADDER_L3C_PILOT"        # set: the pilot (this module's docstring)
PREPARE, ATTEMPTS, REPORT = "ladder_l3c_prepare", "ladder_l3c_attempts", "ladder_l3c_report"
# The stage's sampling seed is `round.sampling_seed` + 100 x the task's seed + this place: L3a's is 30 and L3a2's 31
# (`ladder_l3a.SEED_PLACE`, where the other measurements' places are listed). The pilot has the place after it, so the
# run's generations are first seen in the run. No measurement has 32 or 33.
SEED_PLACE = 32
PILOT_SEED_PLACE = 33
PROOF_LINES = Path(__file__).resolve().parents[1] / "data" / "ladder_l0" / "heldout_proof_lines.jsonl"       # read by the report step alone


# ------------------------------------------------------------------------------------------------- the run
def is_pilot() -> bool:
    return bool(os.environ.get(L3C_PILOT_VARIABLE))


def sizes(config: dict) -> dict:
    """What a run is made with: the episodes a problem of each group gets, the generations of an episode, the pool's
    size and the closers kept, the caps, the seed. A group with no number of its own in `ladder_loop.accumulate`
    (`episodes_goal`, `episodes_below`) has the rungs'. The pilot: `pilot_episodes` on the hard problems it takes
    (`pilot_problems` of each hard group) and none on the others, with a sampling seed of its own."""
    ladder = config["ladder_loop"]
    own, episode, pilot = ladder["accumulate"], ladder["episode"], is_pilot()
    if own["generations"] < 2:
        raise ValueError(f"ladder_loop.accumulate.generations is {own['generations']}: an episode is a first generation and at least one more")
    episodes = {group: own.get(f"episodes_{group}", own["episodes_rungs"]) for group in (GOAL, *RUNGS)}
    if pilot:
        episodes = {group: own["pilot_episodes"] if group in SETS[HARD] else 0 for group in (GOAL, *RUNGS)}
    return {"episodes": episodes, "pilot_problems": own["pilot_problems"] if pilot else None,
            # `loops` is the name L3a's first attempts and its report step read the generations by: loop n is generation n + 1.
            "generations": own["generations"], "loops": own["generations"], "pool_blocks": own["pool_blocks"], "kept_closers": own["kept_closers"],
            "max_new_tokens": episode["max_new_tokens"], "lean_seconds": episode["lean_seconds"],
            "sampling_seed": ladder["round"]["sampling_seed"] + 100 * training_seed(config) + (PILOT_SEED_PLACE if pilot else SEED_PLACE)}


def chosen(own: dict, problems: list[dict]) -> list[dict]:
    """The problems a run takes: every one; the pilot, the first `pilot_problems` of G and the first of the below-band
    rung, in the order L1's run stored them."""
    if own["pilot_problems"] is None:
        return problems
    return [row for group in SETS[HARD] for row in [row for row in problems if row["group"] == group][:own["pilot_problems"]]]


# L3c: two arms. Neither asks for L3a's state request (the accumulate arm's state comes from its own pool check); a
# proof Lean already rejected in its episode is not sent again, in either arm.
L3C = Check(name="ladder_l3c", title="L3c", run_variable=L3C_RUN_VARIABLE, source_variable=L3C_SOURCE_VARIABLE, sizes=sizes,
            episodes=lambda own, group: own["episodes"][group], arms=ARMS, repair_arms=(), repairs=lambda arm, loop: False,
            shows_state=lambda arm: True, known_copies=True, chosen=chosen)
PILOT = dataclasses.replace(L3C, directory="ladder_l3c_pilot")       # the same steps and markers, in a run directory of its own


def the_check() -> Check:
    return PILOT if is_pilot() else L3C


def _store(config: dict) -> ArtifactStore:
    return ladder_l3a._store(config, the_check())


def source_directory(config: dict) -> Path:
    """L1's run directory for this seed. It is only ever READ, with plain file reads: nothing is created in it."""
    return ladder_l3a.source_directory(config, the_check())


def pools_file(loop: int) -> str:
    return f"repair_pools_{loop}.jsonl"


def closers_file(loop: int) -> str:
    return f"repair_closers_{loop}.jsonl"


def ladder_l3c_prepare(config: dict) -> dict:
    return ladder_l3a.prepare_run(config, the_check())


# -------------------------------------------------------------------------------------------------- a loop
def first_jobs(config: dict, problems: list[dict], own: dict) -> list[Job]:
    """Loop 0, generation 1: L3a's blind first attempts. Each one on the side the certificate allows is also the
    accumulate arm's first generation: a fresh one, from an empty pool. The audited attempts on the ruled-out side
    are checked for the alarm and hold no episode."""
    jobs = ladder_l3a.first_jobs(config, problems, own, the_check())
    for job in jobs:
        for consumer in job.consumers:
            consumer.update({"kind": FRESH, "pool_blocks": 0, "accumulating": None if consumer["audit"] else pooling.empty_state()})
    return jobs


def next_jobs(config: dict, problems: list[dict], own: dict, loop: int, attempts: dict[int, list[dict]], pools: dict[int, list[dict]],
              closers: dict[int, list[dict]], count_tokens: Callable[[str], int]) -> list[Job]:
    """Loop `loop` (1 or later): one generation for every arm of every episode that arm has not resolved. The blind
    arm: a whole proof from the plain prompt. The accumulate arm: what its episode holds is read from the stored rows
    of the loops before (`accumulate.episode_state`); the generation CONTINUES when the pool is not empty and has
    changed since the last continuing one (the plain prompt, the pool's lemmas, Lean's state after them as a comment;
    the proof checked is the pool and the continuation), and is FRESH otherwise: the plain prompt with the seed every
    arm has at this loop, so the blind arm's own sample where that arm is still open. A continuing generation whose
    prompt would not leave room for the new tokens is a blind attempt of that kind, counted with its reason. Every
    attempt carries the proofs Lean has already rejected in its arm of the episode: L3a's `settle_chunk` does not
    send one of those again."""
    max_new, context = own["max_new_tokens"], config["vllm"]["max_model_len"]
    firsts = {(row["problem_id"], row["episode"]): row for row in attempts[0] if not row["audit"]}
    earlier = {(row["arm"], row["problem_id"], row["episode"], row["loop"]): row for index in range(1, loop) for row in attempts[index]}
    of_pools: dict[tuple[str, int], list[dict]] = {}
    of_closers: dict[tuple[str, int], list[dict]] = {}
    for index in range(loop):
        for row in pools[index]:
            of_pools.setdefault((row["problem_id"], row["episode"]), []).append(row)
        for row in closers[index]:
            of_closers.setdefault((row["problem_id"], row["episode"]), []).append(row)
    jobs: dict[tuple[str, int], Job] = {}
    for problem in problems:
        for episode in range(problem["episodes"]):
            key = (problem["problem_id"], episode)
            first = firsts[key]
            if first["status"] == VERIFIED:
                continue
            side = first["side"]
            statement = statement_of(problem, side)
            plain = build_prover_prompt(statement)
            seed = attempt_seed(own["sampling_seed"], problem["problem_id"], side, episode, loop)
            for arm in ARMS:
                own_rows = [earlier[place] for index in range(1, loop) if (place := (arm, *key, index)) in earlier]
                if any(row["status"] == VERIFIED for row in own_rows):
                    continue
                if arm == ACCUMULATE and any(row["status"] == VERIFIED for row in of_closers.get(key, [])):
                    continue                # an assembled proof resolved the episode
                if len(own_rows) != loop - 1:
                    raise RuntimeError(f"{problem['problem_id']} episode {episode}: the {arm} arm has {len(own_rows)} stored generations before loop "
                                       f"{loop} and is not resolved: a stored loop is not whole")
                consumer = {"arm": arm, "problem_id": problem["problem_id"], "group": problem["group"], "side": side, "episode": episode, "audit": False,
                            "untrimmed": False, "how": BLIND_ATTEMPT, "had_state": False, "no_state": None, "kept_lines": 0, "cut": None,
                            "failed_step": None, "kind": FRESH, "pool_blocks": 0, "accumulating": None, "rejected": rejected([first, *own_rows])}
                prompt, kept = plain, ()
                if arm == ACCUMULATE:
                    state = pooling.episode_state([first, *own_rows], of_pools.get(key, []), of_closers.get(key, []))
                    consumer.update({"kind": pooling.kind_of(state), "accumulating": state, "rejected": state["rejected"]})
                    if consumer["kind"] == CONTINUE:
                        lines = tuple(pooling.pool_lines(state["pool"]))
                        continuing = rules.resume_prompt(statement, lines, pooling.POOL_INDENTATION, state["state"])
                        if count_tokens(continuing) + max_new + CONTEXT_MARGIN > context:
                            consumer["no_state"] = rules.PROMPT_TOO_LONG
                        else:
                            prompt, kept = continuing, lines
                            consumer.update({"how": RESUMED, "had_state": True, "kept_lines": len(lines), "pool_blocks": len(state["pool"])})
                ladder_l3a._add(jobs, prompt, seed, statement, kept, True, False, consumer)
    return list(jobs.values())


def _file(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:24]


def settle_chunk(pool, chunk: list[Job], episode_settings: dict, own: dict, loop: int) -> dict:
    """Check a generated chunk (L3a's `settle_chunk`: the token cap, the lexical filter, the known copies, Lean), then
    go on for every accumulate episode in it whose proof failed and was not a known copy:

      the harvest      its leading lemmas, cut where L3a's cut puts the first error, and its closing step
      the pool check   where lemmas would join the pool: the pool with them, then `all_goals sorry`. It stands (the
                       state is kept for the next continuing generation) or the harvest is taken back out
      the closers      the step is kept when the proof was lemmas and one closing step, the episode holds fewer than
                       `kept_closers` and does not hold that text already. Each kept closer that is tried now
                       (`accumulate.closers_to_check`) is checked after the pool, but for the proof Lean has just
                       rejected under its pooled names (`OWN_PROOF`) and a text Lean already rejected in the episode
                       (`KNOWN_COPY`), which are not sent

    What that left is put on the episode's consumer (`after`), for the loop's rows. Each Lean file is sent once a
    chunk, whoever shares it."""
    stats = ladder_l3a.settle_chunk(pool, chunk, episode_settings, ())
    pin = pin_of(pool.settings)
    holders, requests = [], {}
    for job in chunk:
        for consumer in job.consumers:
            state = consumer["accumulating"]
            if state is None:
                continue
            after = consumer["after"] = {"no_harvest": None, "harvest": None, "cut_line": None, "pool": None, "pool_now": state["pool"],
                                         "made_at": state["made_at"], "kept_closer": None, "closers": [], "assembled": False}
            if consumer.get("copy_of") is not None:
                after["no_harvest"] = KNOWN_COPY        # not sent for this episode: Lean said nothing a cut could be made by
                continue
            if job.status == VERIFIED:
                continue
            cut, why = rules.plan_cut(job.statement, job.proof, job.errors)
            if cut is None:
                after["no_harvest"] = why
                continue
            continues = consumer["how"] == RESUMED
            kept = cut.kept_lines - (len(job.kept) if continues else 0)
            after["cut_line"] = cut.kept_lines + 1 if cut.kind == rules.BODY else None
            if kept < 0:
                after["no_harvest"] = pooling.ERROR_IN_THE_POOLS_LINES
                continue
            found = after["harvest"] = pooling.harvest(job.completion if continues else job.proof, kept, state["pool"], loop + 1, own["pool_blocks"], continues)
            holders.append((job, consumer))
            if found.blocks:
                source, line, column = pooling.pool_source(job.statement, [*state["pool"], *found.blocks])
                after["pool"] = {"blocks": [*state["pool"], *found.blocks], "file": _file(source), "line": line, "column": column}
                requests[after["pool"]["file"]] = source
    answers = ladder_l3a.checked(pool, requests, episode_settings)
    sources = {}
    for job, consumer in holders:
        state, after, found = consumer["accumulating"], consumer["after"], consumer["after"]["harvest"]
        grew = False
        if after["pool"] is not None:
            raw = answers[after["pool"]["file"]]
            shown, outcome = pooling.read_pool(raw, after["pool"]["line"], after["pool"]["column"])
            grew = outcome == STANDS
            after["pool"].update({"outcome": outcome, "stands": grew, "state": shown, "seconds": raw.get("time")})
            if grew:
                after["pool_now"], after["made_at"] = after["pool"]["blocks"], loop
        written = found.closer
        if written is not None and len(state["closers"]) < own["kept_closers"] and written["text"] not in {closer["text"] for closer in state["closers"]}:
            after["kept_closer"] = written
        # The pool as it now is, then this proof's own closing step, is this proof under its pooled names (whether the step
        # was kept now or the episode held that text already): Lean has just rejected it. Not so where its lemmas were
        # taken back out: the step alone after the pool is another proof.
        own_proof = found.own_proof and (grew or not found.blocks)
        kept_closers = [*state["closers"], *([after["kept_closer"]] if after["kept_closer"] is not None else [])]
        for closer in pooling.closers_to_check(after["pool_now"], kept_closers, grew, after["kept_closer"]):
            proof = pooling.assembled_proof(after["pool_now"], closer)
            check = {"closer": closer, "proof": proof, "status": None, "sent": False, "seconds": None, "detail": "", "error": None, "file": None, "copy_of": None}
            if own_proof and closer["text"] == written["text"]:
                check["status"] = OWN_PROOF
            elif checked_text(proof) in consumer["rejected"]:
                check["status"], check["copy_of"] = KNOWN_COPY, consumer["rejected"][checked_text(proof)]
            elif find_forbidden_token(proof) is not None:
                check["status"], check["detail"] = REJECTED, f"the proof contains the forbidden token {find_forbidden_token(proof)!r}"
            else:
                source = build_proof_source(job.statement, proof)
                check["file"] = _file(source)
                sources[check["file"]] = source
            after["closers"].append(check)
    answers = ladder_l3a.checked(pool, sources, episode_settings)
    for job, consumer in holders:
        for check in consumer["after"]["closers"]:
            if check["file"] is not None:
                verdict = pin.classify(check["file"], answers[check["file"]])
                _, error = rules.leading_error(job.statement, check["proof"], rules.errors_of(answers[check["file"]]))       # as an attempt's is read
                check.update({"status": ladder_loop._status(verdict), "sent": True, "seconds": verdict.verification_seconds, "detail": verdict.detail,
                              "error": error})
        consumer["after"]["assembled"] = any(check["status"] == VERIFIED for check in consumer["after"]["closers"])
    return {"sent": stats["sent"], "pool_checks": len(requests), "closer_checks": len(sources)}


def attempt_rows(job: Job, loop: int) -> list[dict]:
    """L3a's attempt rows (L3a2's, with `copy_of`), each with what L3c adds: the generation's kind, the pool blocks in
    its prompt, whether its sample is shared with the other arm, and, on the rows that hold the accumulate arm's
    episode (its own, and the shared first generation), what the proof left: the lemmas harvested, the closer kept,
    the pool after it."""
    rows = ladder_l3a.attempt_rows(job, loop)
    for row, consumer in zip(rows, job.consumers):
        after = consumer.get("after")
        found = after["harvest"] if after else None
        row.update({
            "kind": consumer["kind"], "pool_blocks": consumer["pool_blocks"],
            # One sample for both arms: the first generation, and a later one that both arms asked for with the plain prompt.
            "shared": not consumer["audit"] and (loop == 0 or len(job.consumers) > 1),
            "cut_line": after["cut_line"] if after else None,
            # Why the proof left the pool nothing to try (a known copy, no error position, ...); None when it was harvested.
            "no_harvest": after["no_harvest"] if after else None,
            "harvested": len(found.blocks) if found else None, "harvest_duplicates": found.duplicates if found else None,
            "harvest_over_the_size": found.over_the_size if found else None, "harvest_ended": found.ended if found else None,
            "kept_closer": after["kept_closer"] if after else None,
            "pool_after": len(after["pool_now"]) if after else None,
            "resolved_by_an_assembled_proof": after["assembled"] if after else None,
            # A known copy of a proof that was no attempt's own text: the pool and a kept closer made it, and Lean rejected it
            # (when the closer was tried after the pool, or as the proof that left the closer, under its pooled names).
            "copy_of_an_assembled_proof": row["status"] == KNOWN_COPY and checked_text(job.proof) in (consumer["accumulating"] or {}).get("assembled", {})})
        if row["copy_of_an_assembled_proof"]:
            row["first_error"] = (f"the proof the pool and a kept closer make, which Lean rejected at this episode's generation {row['copy_of'] + 1}: "
                                  "not sent again")
    return rows


def pool_rows(job: Job, loop: int) -> list[dict]:
    """A row for every pool state a harvest of this generation made: its blocks and its text, what the harvest brought,
    and what Lean said of it (it stands, with the state; or why the harvest was taken back out)."""
    rows = []
    for consumer in job.consumers:
        made = (consumer.get("after") or {}).get("pool")
        if made is None:
            continue
        found = consumer["after"]["harvest"]
        rows.append({"problem_id": consumer["problem_id"], "group": consumer["group"], "side": consumer["side"], "episode": consumer["episode"],
                     "loop": loop, "generation": loop + 1, "kind": consumer["kind"], "how": consumer["how"],
                     "blocks_before": len(made["blocks"]) - len(found.blocks), "harvested": len(found.blocks), "duplicates": found.duplicates,
                     "over_the_size": found.over_the_size, "blocks": made["blocks"], "text": pooling.pool_text(made["blocks"]),
                     # "state": it stands; else why it does not, and the harvest was taken back out.
                     "outcome": made["outcome"], "stands": made["stands"], "state": made["state"], "seconds": made["seconds"], "pool_file": made["file"]})
    return rows


def closer_rows(job: Job, loop: int) -> list[dict]:
    """A row for every closer check made after this generation: the pool it was checked after (the generation whose
    harvest made that pool state), the closer and the generation it came from, the proof they make, the status."""
    rows, first_line = [], rules.proof_first_line(job.statement)
    for consumer in job.consumers:
        after = consumer.get("after") or {}
        for check in after.get("closers", []):
            rows.append({"problem_id": consumer["problem_id"], "group": consumer["group"], "side": consumer["side"], "episode": consumer["episode"],
                         "loop": loop, "generation": loop + 1, "pool_blocks": len(after["pool_now"]),
                         "pool_made_at_generation": None if after["made_at"] is None else after["made_at"] + 1,
                         "closer": check["closer"]["text"], "needs": check["closer"]["needs"], "from_generation": check["closer"]["generation"],
                         "proof": check["proof"], "status": check["status"], "sent_to_lean": check["sent"], "seconds": check["seconds"],
                         # The loop at which Lean rejected this text in the episode, when it is a known copy.
                         "copy_of": check["copy_of"],
                         # The error the assembled proof is read by, as an attempt's is (`cut.leading_error`; a line is counted in the
                         # PROOF: 1 is its first line, 0 the theorem's `:= by`), or why nothing was judged.
                         "first_error": (check["error"]["text"] if check["error"] else check["detail"])[:300],
                         "first_error_line": check["error"]["line"] - first_line + 1 if check["error"] else None, "file": check["file"]})
    return rows


def run_loop(config: dict, store: ArtifactStore, problems: list[dict], own: dict, loop: int, jobs: list[Job], engine, parameters_of: Callable,
             count_tokens: Callable[[str], int], pool) -> dict:
    """Generate, check and store one loop (the sampling in chunks is L3a's). The loop's attempts are written BEFORE
    they are judged (an alarm leaves its evidence), then its pool states and its closer checks, and only then is the
    loop marked done."""
    check, episode_settings = the_check(), config["ladder_loop"]["episode"]
    started = time.monotonic()
    settled, generation_seconds, generated = ladder_l3a.sampled(jobs, own, episode_settings, engine, parameters_of, count_tokens,
                                                                lambda chunk: settle_chunk(pool, chunk, episode_settings, own, loop))
    rows = [row for job in jobs for row in attempt_rows(job, loop)]
    store.write_rows(ladder_l3a.attempts_file(loop), rows)
    by_problem: dict[str, list[dict]] = {}
    for row in rows:
        by_problem.setdefault(row["problem_id"], []).append(row)
    for problem in problems:
        raise_on_contradiction(problem, by_problem.get(problem["problem_id"], []))
    made = [row for job in jobs for row in pool_rows(job, loop)]
    tried = [row for job in jobs for row in closer_rows(job, loop)]
    store.write_rows(pools_file(loop), made)
    store.write_rows(closers_file(loop), tried)
    statuses = {arm: dict(Counter(row["status"] for row in rows if row["arm"] == arm)) for arm in (FIRST, *ARMS) if any(row["arm"] == arm for row in rows)}
    own_rows = [row for row in rows if row["arm"] == ACCUMULATE]
    generated_tokens = sum(job.token_count for job in jobs)
    stats = {"loop": loop, "generation": loop + 1, "attempts": len(rows), "generations": len(jobs), "statuses": statuses,
             # The accumulate arm's generations of this loop: by kind, and how many had the pool in their prompt.
             "kinds": dict(Counter(row["kind"] for row in own_rows)), "continuing_with_the_pool": sum(row["how"] == RESUMED for row in own_rows),
             "shared_with_the_blind_arm": sum(bool(row["shared"]) for row in own_rows),
             "generated_tokens": generated_tokens, "prompt_tokens": sum(job.prompt_tokens for job in jobs),
             "generation_seconds": round(generation_seconds, 1),
             "tokens_per_second": round(generated_tokens / generation_seconds, 1) if generation_seconds else None,
             "attempts_sent_to_lean": sum(entry["sent"] for entry in settled), "pool_checks_sent_to_lean": sum(entry["pool_checks"] for entry in settled),
             "closer_checks_sent_to_lean": sum(entry["closer_checks"] for entry in settled),
             "pool_states": len(made), "pool_states_by_outcome": dict(Counter(row["outcome"] for row in made).most_common()),
             "lemmas_pooled": sum(row["harvested"] for row in made if row["stands"]),
             "closers_kept": sum(row["kept_closer"] is not None for row in rows),
             "closer_checks": len(tried), "closer_checks_by_status": dict(Counter(row["status"] for row in tried).most_common()),
             "episodes_resolved_by_an_assembled_proof": sum(bool(row["resolved_by_an_assembled_proof"]) for row in rows),
             "wall_seconds": round(time.monotonic() - started, 1), "waited_on_lean_after_the_last_chunk_seconds": round(time.monotonic() - generated, 1)}
    store.mark_done(ladder_l3a.loop_marker(loop, check), stats)
    print(f"{ladder_l3a.loop_marker(loop, check)}: {stats['attempts']} attempts in {stats['generations']} generations, {stats['statuses']}, kinds "
          f"{stats['kinds']}, pools {stats['pool_states_by_outcome']}, closers {stats['closer_checks_by_status']} ({stats['wall_seconds']} s)", flush=True)
    return stats


# ------------------------------------------------------------------------------------------------ attempts
def ladder_l3c_attempts(config: dict) -> dict:
    """The whole sampling step: the first generations, then the loops, with one engine and one Lean client. A rerun
    returns what is stored; a rerun of a step that stopped resumes at the first loop that is not done."""
    check = the_check()
    store = _store(config)
    delivered = ("repair_episodes.jsonl", "repair_problems.jsonl")
    if store.is_done(ATTEMPTS):
        for name in delivered:              # a rerun is another task: its out/ gets the results too
            store.mirror(name)
        return store.done_summary(ATTEMPTS)
    ladder_l3a._need(store, PREPARE, "the attempts", check)
    own, problems = ladder_l3a._same_sizes(config, store, check), store.read_rows("problems.jsonl")
    kit, tools, written = ladder_l3a.engine_kit(), {}, set()
    attempts: dict[int, list[dict]] = {}
    pools: dict[int, list[dict]] = {}
    closers: dict[int, list[dict]] = {}
    with ladder_l3a.lean_pool(config) as pool:
        for loop in range(own["loops"]):
            if not store.is_done(ladder_l3a.loop_marker(loop, check)):
                if "engine" not in tools:
                    tools["engine"], tools["parameters_of"] = kit(config)
                    tools["count_tokens"] = ladder_l3a.token_counter(tools["engine"])
                jobs = first_jobs(config, problems, own) if loop == 0 else next_jobs(config, problems, own, loop, attempts, pools, closers, tools["count_tokens"])
                run_loop(config, store, problems, own, loop, jobs, tools["engine"], tools["parameters_of"], tools["count_tokens"], pool)
                written.add(loop)
            attempts[loop] = store.read_rows(ladder_l3a.attempts_file(loop))
            pools[loop], closers[loop] = store.read_rows(pools_file(loop)), store.read_rows(closers_file(loop))
    tools.clear()
    for loop in range(own["loops"]):        # what an earlier task's run of this step stored reaches this task's out/ too
        if loop not in written:
            for name in (ladder_l3a.attempts_file(loop), pools_file(loop), closers_file(loop), f"{ladder_l3a.loop_marker(loop, check)}.done.json"):
                store.mirror(name)
    every = [row for loop in range(own["loops"]) for row in attempts[loop]]
    episodes = pooling.episode_rows(problems, every, [row for loop in range(own["loops"]) for row in pools[loop]],
                                    [row for loop in range(own["loops"]) for row in closers[loop]], own["loops"])
    store.write_rows(delivered[0], episodes)
    store.write_rows(delivered[1], pooling.problem_rows(episodes, own["loops"]))
    loops = [store.done_summary(ladder_l3a.loop_marker(loop, check)) for loop in range(own["loops"])]
    generated, seconds = sum(entry["generated_tokens"] for entry in loops), sum(entry["generation_seconds"] for entry in loops)
    summary = {"seed": training_seed(config), "pilot": is_pilot(), "sizes": own, "problems": len(problems), "episodes": len(episodes),
               "failed_first_generations": sum(row["failed_first"] for row in episodes),
               "attempts": len(every), "attempts_on_the_contradicted_side": sum(bool(row["audit"]) for row in every),
               "generations": sum(entry["generations"] for entry in loops), "generated_tokens": generated, "generation_seconds": round(seconds, 1),
               "tokens_per_second": round(generated / seconds, 1) if seconds else None,
               "lean_checks_sent": sum(entry["attempts_sent_to_lean"] + entry["pool_checks_sent_to_lean"] + entry["closer_checks_sent_to_lean"] for entry in loops),
               "pool_checks_sent_to_lean": sum(entry["pool_checks_sent_to_lean"] for entry in loops),
               "closer_checks_sent_to_lean": sum(entry["closer_checks_sent_to_lean"] for entry in loops),
               "episodes_resolved": {arm: sum(resolved(row, arm, own["loops"]) for row in episodes) for arm in ARMS},
               "episodes_resolved_in_the_accumulate_arm_by": dict(Counter(row["arms"][ACCUMULATE]["resolved_by"] for row in episodes
                                                                          if row["arms"][ACCUMULATE]["resolved_by"] is not None)),
               "episodes_with_a_pool_at_the_end": sum(row["arms"][ACCUMULATE]["pool_blocks"] > 0 for row in episodes),
               "statuses_by_arm": {arm: dict(Counter(row["status"] for row in every if row["arm"] == arm)) for arm in (FIRST, *ARMS)},
               "loops": loops, "loops_done_in_this_run": sorted(written), "lean_in_flight": config["ladder_loop"].get("lean_in_flight"),
               "stand_in_engine": _stand_in()}
    store.mark_done(ATTEMPTS, summary)
    return summary


# -------------------------------------------------------------------------------------------------- report
def proof_lengths(path: Path | None = None) -> dict[str, dict]:
    """The line count of each held-out problem's shortest published proof and its length group, by problem
    (`data/ladder_l0/heldout_proof_lines.jsonl`; spec "Proof lengths for the read by length"). The REPORT step reads
    it and nothing else does: no prompt and no rule sees a published proof's length."""
    with (path or PROOF_LINES).open() as handle:
        return {row["problem_id"]: row for row in (json.loads(line) for line in handle if line.strip())}


def ladder_l3c_report(config: dict) -> dict:
    from rlvr_lean.reporting.ladder_l3c import build_l3c_report

    lengths = proof_lengths()
    return ladder_l3a.write_report(config, the_check(), lambda *given: build_l3c_report(*given, lengths), "report_ladder_l3c.json")


STEPS = {
    PREPARE: ladder_l3c_prepare,
    ATTEMPTS: ladder_l3c_attempts,
    REPORT: ladder_l3c_report,
}
