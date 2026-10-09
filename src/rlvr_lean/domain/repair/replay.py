"""The accumulating episode's Lean-only part, replayed over attempts that are already stored (no generation). Spec:
docs/spec/ladder-loop.spec.md, "L3d: train on what the episode reaches" (the harvest H0; the goal set's replay), and
`docs/spec/ladder-l3c-RESULT.md`, the addendum of 2026-10-08. Pure: texts and Lean's answers in, texts out.

After a failed attempt an episode harvests its leading lemmas into the pool, checks the pool, and tries its kept
closing steps after it; a closer that verifies after the pool is an ASSEMBLED proof. That is what `gpu/ladder_l3c.py`
does after a failed generation of its accumulate arm (`settle_chunk`), with the generations left out: every attempt
here is a whole proof from the plain prompt, so no proof continues from a pool. The rules are `accumulate.py`'s and
`cut.py`'s; the sizes are L3c's (`accumulate.pool_blocks`, `accumulate.kept_closers`).

An episode is a dict the caller owns: `statement`, `pool` (the lemmas held), `closers` (the kept closing steps) and
`rejected` (the proof texts Lean has rejected in it). One attempt is replayed in three calls, with Lean between them:

  harvested            the attempt's text joins the rejected ones; its cut, its harvest, and the file that checks the
                       pool with the new lemmas (when there are any)
  closers_to_send      the pool check is read (it stands, or the harvest is taken back out); the closing step is kept;
                       the proofs the kept closers make after the pool, less the proof Lean has just rejected under its
                       pooled names, the texts Lean already rejected in the episode and any with a forbidden token
  first_verified       the closer checks are read: the first proof that verified, and the others join the rejected

`assemble` is those three over many episodes at once, attempt by attempt, with Lean handed in as a function.

and, for an assembled proof that is to be trained on:

  without_a_block, minimise    the pool blocks taken out one at a time, from the last to the first
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from rlvr_lean.domain.repair import accumulate as pooling
from rlvr_lean.domain.repair import cut as rules
from rlvr_lean.domain.repair.alternate import checked_text
from rlvr_lean.domain.verification.lean_source import build_proof_source, find_forbidden_token
from rlvr_lean.domain.verification.status import VerificationResult, classify_check_result


def name_of(source: str) -> str:
    """The name a Lean file is sent under: a hash of its text (as `gpu.ladder_l3c` names a pool's or a closer's file)."""
    return hashlib.sha256(source.encode()).hexdigest()[:24]


def new_episode(statement: str) -> dict:
    """What an episode holds before its first attempt."""
    return {"statement": statement, "pool": [], "closers": [], "rejected": set()}


def judged(name: str, answer: Mapping[str, Any]) -> VerificationResult:
    """One answer of the pool, read as a solver's attempt is under the ladder loop's pin (a warning that says "failed"
    is not an error there)."""
    return classify_check_result(name, answer, failed_warning_is_error=False)


def harvested(episode: MutableMapping, proof: str, errors: Sequence[Mapping], generation: int, pool_blocks: int, stats: Counter) -> dict | None:
    """One failed attempt (`proof`, with the errors Lean gave it): its text is one Lean has rejected in the episode;
    L3a's cut; the harvest of its leading lemmas. None when the proof cannot be cut (counted in `stats` with its
    reason). Else what the attempt leaves: `found` (the harvest), `pool_now` (the pool as it is before any check) and
    `pool`: None, or the pool with the new lemmas and the file that checks it (`file`, `source`, `line`, `column`)."""
    episode["rejected"].add(checked_text(proof))
    cut, why = rules.plan_cut(episode["statement"], proof, errors)
    if cut is None:
        stats["no_harvest:" + str(why)] += 1
        return None
    after = {"pool": None, "pool_now": episode["pool"]}
    found = after["found"] = pooling.harvest(proof, cut.kept_lines, episode["pool"], generation, pool_blocks, False)
    if found.blocks:
        source, line, column = pooling.pool_source(episode["statement"], [*episode["pool"], *found.blocks])
        after["pool"] = {"blocks": [*episode["pool"], *found.blocks], "file": name_of(source), "source": source, "line": line, "column": column}
    return after


def closers_to_send(episode: MutableMapping, after: MutableMapping, answers: Mapping[str, Mapping], kept_closers: int, stats: Counter) -> dict[str, str]:
    """After the pool checks are back: the episode's pool and kept closers as they now are, and the Lean files (name ->
    source) of the closer checks to make. `after["checks"]` holds each as (name, proof, closer)."""
    found, grew = after["found"], False
    if after["pool"] is not None:
        _, outcome = pooling.read_pool(answers[after["pool"]["file"]], after["pool"]["line"], after["pool"]["column"])
        grew = outcome == pooling.STANDS
        stats["pool:" + str(outcome)] += 1
        if grew:
            after["pool_now"] = after["pool"]["blocks"]
    written, kept_now = found.closer, None
    if written is not None and len(episode["closers"]) < kept_closers and written["text"] not in {closer["text"] for closer in episode["closers"]}:
        kept_now = written
    own_proof = found.own_proof and (grew or not found.blocks)
    closers = [*episode["closers"], *([kept_now] if kept_now is not None else [])]
    after["checks"], sources = [], {}
    for closer in pooling.closers_to_check(after["pool_now"], closers, grew, kept_now):
        proof = pooling.assembled_proof(after["pool_now"], closer)
        if own_proof and closer["text"] == written["text"]:
            stats["closer:own_proof"] += 1
            episode["rejected"].add(checked_text(proof))
        elif checked_text(proof) in episode["rejected"]:
            stats["closer:known_copy"] += 1
        elif find_forbidden_token(proof) is not None:
            stats["closer:forbidden"] += 1
        else:
            source = build_proof_source(episode["statement"], proof)
            after["checks"].append((name_of(source), proof, closer))
            sources[name_of(source)] = source
    episode["pool"], episode["closers"] = after["pool_now"], closers
    return sources


def first_verified(episode: MutableMapping, after: Mapping, answers: Mapping[str, Mapping], stats: Counter) -> tuple[str, Mapping] | None:
    """After the closer checks are back: (the proof, its closer) of the first check that verified, or None. A proof
    Lean did not verify joins the texts rejected in the episode."""
    won = None
    for name, proof, closer in after["checks"]:
        verdict = judged(name, answers[name])
        stats["closer:" + verdict.status.value] += 1
        if verdict.is_verified and won is None:
            won = (proof, closer)
        elif not verdict.is_verified:
            episode["rejected"].add(checked_text(proof))
    return won


def assemble(episodes: Sequence[MutableMapping], check: Callable[[dict[str, str]], Mapping[str, Mapping]], sizes: Mapping, attempts: int, stats: Counter,
             on_assembled: Callable[[Mapping, str], None] | None = None) -> dict:
    """The Lean-only part over every episode, attempt 1 to `attempts`. An episode holds what `new_episode` gives and
    `chain`: an attempt's place in the order drawn (from 1) -> `{completion, errors}` for each failed attempt that is
    replayed (`errors`: Lean's positioned errors for it; an attempt with none is passed over), and `resolved_at`:
    None. An episode a closer resolves after a pool gets `resolved_at` (the attempt it came after),
    `assembled_proof` and `closer`, and is replayed no further. `on_assembled(episode, proof)` is called before
    that is recorded (it may raise: the soundness alarm). `sizes`: `pool_blocks`, `kept_closers`. `check(name ->
    Lean file)` answers a batch. Returns how many pool checks and closer checks were made."""
    counts = {"pool_checks": 0, "closer_checks": 0}
    for generation in range(1, attempts + 1):
        holders, requests = [], {}
        for episode in episodes:
            row = episode["chain"].get(generation)
            if episode["resolved_at"] is not None or row is None or not row.get("errors"):
                continue
            after = harvested(episode, row["completion"], row["errors"], generation, sizes["pool_blocks"], stats)
            if after is None:
                continue
            holders.append((episode, after))
            if after["pool"] is not None:
                requests[after["pool"]["file"]] = after["pool"]["source"]
        if not holders:
            continue
        answers = check(requests) if requests else {}
        counts["pool_checks"] += len(requests)
        sources: dict[str, str] = {}
        for episode, after in holders:
            sources.update(closers_to_send(episode, after, answers, sizes["kept_closers"], stats))
        answers = check(sources) if sources else {}
        counts["closer_checks"] += len(sources)
        for episode, after in holders:
            won = first_verified(episode, after, answers, stats)
            if won is not None and episode["resolved_at"] is None:
                if on_assembled is not None:
                    on_assembled(episode, won[0])
                episode["resolved_at"], episode["assembled_proof"], episode["closer"] = generation, won[0], won[1]
    return counts


# ------------------------------------------------------------------------------------------- minimisation
def without_a_block(blocks: Sequence[Mapping], kept: Sequence[int], index: int, closer: Mapping) -> str:
    """The assembled proof with the pool blocks `kept` (places in `blocks`), less the one at `index`."""
    return pooling.assembled_proof([blocks[place] for place in kept if place != index], closer)


def minimise(proofs: Sequence[MutableMapping], check: Callable[[dict[str, str]], Mapping[str, Mapping]], stats: Counter) -> None:
    """Minimise every assembled proof before it is trained on (spec L3d, "A harvested proof is minimised"). An
    assembled proof holds every pooled lemma, those of unrelated failed attempts with it. Its pool blocks are taken
    out one at a time, from the last to the first, and a block stays out when the proof still verifies (one check
    each). What is left is checked once more as a solver's attempt is, and that text is the training text; if that
    check fails, the proof as it was assembled is used. Names are left as the pool made them.

    `proofs`: each `{statement, blocks, closer, assembled}` (the pool and the closer the proof was assembled from, and
    its text). Each gets `completion` (the training text), `kept` (the places of the blocks that stay), `minimised`
    (False: the last check failed and the proof is as assembled) and `minimise_checks`. `check(name -> Lean file)`
    answers a batch: the proofs go together, a block's place from the end at a time."""
    for proof in proofs:
        proof.update({"kept": list(range(len(proof["blocks"]))), "minimise_checks": 0})
    for back in range(max((len(proof["blocks"]) for proof in proofs), default=0)):
        sources, asked = {}, []
        for proof in proofs:
            index = len(proof["blocks"]) - 1 - back
            if index < 0:
                continue
            source = build_proof_source(proof["statement"], without_a_block(proof["blocks"], proof["kept"], index, proof["closer"]))
            sources[name_of(source)] = source
            asked.append((proof, index, name_of(source)))
        answers = check(sources)
        for proof, index, name in asked:
            verdict = judged(name, answers[name])
            proof["minimise_checks"] += 1
            stats["minimise:" + verdict.status.value] += 1
            if verdict.is_verified:
                proof["kept"].remove(index)
    sources, asked = {}, []
    for proof in proofs:
        proof["completion"] = pooling.assembled_proof([proof["blocks"][place] for place in proof["kept"]], proof["closer"])
        source = build_proof_source(proof["statement"], proof["completion"])
        sources[name_of(source)] = source
        asked.append((proof, name_of(source)))
    answers = check(sources)
    for proof, name in asked:
        verdict = judged(name, answers[name])
        proof["minimise_checks"] += 1
        stats["minimised_text:" + verdict.status.value] += 1
        proof["minimised"] = verdict.is_verified
        if not verdict.is_verified:         # the proof as it was assembled is used
            proof.update({"completion": proof["assembled"], "kept": list(range(len(proof["blocks"])))})
