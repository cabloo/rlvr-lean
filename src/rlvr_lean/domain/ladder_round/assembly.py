"""The round with assembly (L3d Step 2). Spec: docs/spec/ladder-loop.spec.md, "L3d: train on what the episode
reaches", "Step 2, made exact before it is built" ("What it is", "What counts", "The training set of a round") and
"Made exact by the build (Step 2)". Pure: rows in, rows out. The Lean-only part itself is `domain/repair/replay.py`.

After a batch's attempts, every side of every problem that none of its attempts resolved is put through the episode's
Lean-only part over its 8 attempts in the order drawn. A problem an assembled proof resolves is resolved, and for the
challenger it counts as k = 1; its minimised proof is its training example.

  what counts       `counted` (k as the challenger reads it), `with_assembly` (a batch's results, each row saying
                    whether assembly resolved it)
  what is replayed  `episodes_to_assemble`, `raise_on_the_ruled_out_side`, `assembled_rows`, `batch_counts`
  the training set  `round_rows` (a round's own rows, each with its origin), `h0_rows`, `training_set` (one proof a
                    problem: the round's own wins over H0's), `training_order` (a content hash, so that leaving rows
                    out moves no other), `attempts_alone` (the twin's rows)
"""

from __future__ import annotations

from collections import Counter
from typing import Collection, Mapping, Sequence

from rlvr_lean.domain.problem_pool.episodes import NEGATION_SIDE, contradicted_side
from rlvr_lean.domain.problem_pool.selection import SoundnessAlarm, rank
from rlvr_lean.domain.repair import replay
from rlvr_lean.domain.repair.accumulate import proof_line_count
from rlvr_lean.domain.verification.lean_source import find_forbidden_token

ATTEMPT, ASSEMBLED, H0 = "attempt", "assembled", "h0"      # where a training row is from
ORIGINS = (ATTEMPT, ASSEMBLED, H0)
FROM_ASSEMBLY = (ASSEMBLED, H0)                            # the rows `with` is trained on and `without` is not
ORDER_LABEL = "l3d2_order"
EXAMPLE_FIELDS = ("problem_id", "side", "theorem", "completion")
REJECTED = "lean_error"                                    # the status of an attempt Lean rejected: the only kind that is replayed


# ---------------------------------------------------------------------------------------------- what counts
def counted(rows: Sequence[Mapping]) -> list[Mapping]:
    """A batch's per-problem results AS THE CHALLENGER READS THEM: k is the number of verified attempts, and a
    problem with k = 0 that assembly resolved counts as k = 1. Every other row is the row itself (so results that
    know nothing of assembly are what they were)."""
    return [{**row, "resolved": 1} if row["resolved"] == 0 and row.get("resolved_by_assembly") else row for row in rows]


def with_assembly(results: Sequence[Mapping], assembled: Sequence[Mapping]) -> list[dict]:
    """A batch's per-problem results, each saying beside its counts whether an assembled proof resolved it (only a
    problem none of whose attempts verified can be)."""
    by_assembly = {row["problem_id"] for row in assembled}
    return [{**row, "resolved_by_assembly": row["resolved"] == 0 and row["problem_id"] in by_assembly} for row in results]


# ------------------------------------------------------------------------------------------ what is replayed
def episodes_to_assemble(problems: Sequence[Mapping], results: Sequence[Mapping], attempts: Sequence[Mapping],
                         errors: Mapping[str, Sequence[Mapping]]) -> tuple[list[dict], list[str]]:
    """(one episode for each side of each problem NO attempt resolved, in the order of the problems; the attempts
    whose errors are not in `errors`). An episode is that side's attempts in the order drawn: `chain` holds, at
    the attempt's place (from 1), each attempt Lean rejected (`lean_error`, with a completion and no forbidden
    token), with Lean's positioned errors for it. A capped attempt, a timeout and an attempt without a verdict have
    no place in the chain: there is nothing to cut them by. `problems`: the batch's, with `statement`, `negation`
    and the published `side`. `errors`: attempt id -> the errors Lean gave it."""
    unresolved = {row["problem_id"] for row in results if row["resolved"] == 0}
    by_side: dict[tuple[str, str], list[Mapping]] = {}
    for attempt in attempts:
        if attempt["problem_id"] in unresolved:
            by_side.setdefault((attempt["problem_id"], attempt["side"]), []).append(attempt)
    episodes, not_kept = [], []
    for problem in problems:
        for side in sorted(side for problem_id, side in by_side if problem_id == problem["problem_id"]):
            own, chain = sorted(by_side[problem["problem_id"], side], key=lambda attempt: attempt["episode"]), {}
            for attempt in own:
                if attempt["status"] != REJECTED or not attempt.get("completion") or find_forbidden_token(attempt["completion"]) is not None:
                    continue
                if attempt["attempt_id"] not in errors:
                    not_kept.append(attempt["attempt_id"])
                    continue
                chain[attempt["episode"] + 1] = {"attempt_id": attempt["attempt_id"], "completion": attempt["completion"], "errors": list(errors[attempt["attempt_id"]])}
            statement = problem["negation"] if side == NEGATION_SIDE else problem["statement"]
            episodes.append({**replay.new_episode(statement), "problem_id": problem["problem_id"], "side": side, "published_side": problem["side"],
                             "chain": chain, "attempts": len(own), "resolved_at": None, "assembled_proof": None, "closer": None})
    return episodes, not_kept


def raise_on_the_ruled_out_side(episode: Mapping, proof: str) -> None:
    """An assembled proof Lean verified on the side the problem's published answer rules out is a proof of both
    sides: the soundness alarm, as for an attempt."""
    if episode["side"] == contradicted_side(episode["published_side"]):
        raise SoundnessAlarm(f"{episode['problem_id']} is known {episode['published_side']} (a published proof our Lean verified) and an assembled proof "
                             f"verified its {episode['side']}: a proof of both sides. The proof: {proof[:600]!r}")


def to_minimise(episodes: Sequence[Mapping]) -> list[dict]:
    """What `replay.minimise` takes: each episode an assembled proof resolved, with the pool and the closer it was
    made of."""
    return [{"episode": episode, "statement": episode["statement"], "blocks": list(episode["pool"]), "closer": episode["closer"],
             "assembled": episode["assembled_proof"]} for episode in episodes if episode["resolved_at"] is not None]


def assembled_rows(proofs: Sequence[Mapping], number: int, batch: int) -> list[dict]:
    """One row for every assembled resolution of a batch, after `replay.minimise`: the problem and side, the
    attempt it came after, the proof as assembled and as minimised (`completion`: the training text), blocks and
    lines before and after, the checks. One row a problem: two would be a proof of both of its sides."""
    rows = []
    for proof in proofs:
        episode, blocks = proof["episode"], proof["blocks"]
        rows.append({"problem_id": episode["problem_id"], "side": episode["side"], "round": number, "batch": batch, "theorem": episode["statement"],
                     "resolved_after_attempt": episode["resolved_at"], "attempts": episode["attempts"], "attempts_replayed": len(episode["chain"]),
                     "pooled_attempts": sorted({block["generation"] for block in blocks} | {proof["closer"]["generation"]}),
                     "assembled_proof": proof["assembled"], "completion": proof["completion"], "minimised": proof["minimised"],
                     "blocks_before": len(blocks), "blocks_after": len(proof["kept"]), "lines_before": proof_line_count(proof["assembled"]),
                     "lines_after": proof_line_count(proof["completion"]), "minimise_checks": proof["minimise_checks"]})
    twice = sorted(problem for problem, count in Counter(row["problem_id"] for row in rows).items() if count > 1)
    if twice:
        raise SoundnessAlarm(f"{len(twice)} problems have an assembled proof on both sides (first: {twice[0]}): a proof of both sides")
    return rows


def batch_counts(results: Sequence[Mapping], episodes: Sequence[Mapping], rows: Sequence[Mapping], checks: Mapping[str, int], stats: Mapping[str, int]) -> dict:
    """What a batch's assembly did: the problems no attempt resolved, the sides and attempts replayed, the pools
    that stood, the problems an assembled proof resolved, and its Lean checks by kind with their timeouts."""
    unresolved = sum(row["resolved"] == 0 for row in results)
    minimise = sum(count for key, count in stats.items() if key.startswith(("minimise:", "minimised_text:")))
    by_kind = {"errors_from_the_rounds_own_checks": sum(len(episode["chain"]) for episode in episodes), "rechecks_for_error_positions": 0,
               "pool": checks.get("pool_checks", 0), "closer": checks.get("closer_checks", 0), "minimise": minimise}
    return {"problems": len(results), "unresolved_problems": unresolved, "sides_replayed": len(episodes),
            "attempts_replayed": by_kind["errors_from_the_rounds_own_checks"],
            "a_pool_stood_on": len({episode["problem_id"] for episode in episodes if episode["pool"]}),
            "resolved_by_assembly": len(rows), "share_of_the_unresolved": round(len(rows) / unresolved, 5) if unresolved else None,
            "lean_checks": by_kind["pool"] + by_kind["closer"] + by_kind["minimise"], "checks_by_kind": by_kind,
            "timeouts": {key: count for key, count in sorted(stats.items()) if "timeout" in key}, "statuses": dict(sorted(stats.items())),
            "proofs_made_shorter": sum(row["blocks_after"] < row["blocks_before"] for row in rows),
            "proofs_used_as_assembled": sum(not row["minimised"] for row in rows)}


# ------------------------------------------------------------------------------------------ the training set
def row_id(row: Mapping) -> str:
    """A training row's id: its attempt's; `<problem>#assembled#r<round>`; `<problem>#h0`."""
    if row["origin"] == ATTEMPT:
        return row["attempt_id"]
    return f"{row['problem_id']}#{ASSEMBLED}#r{row['round']}" if row["origin"] == ASSEMBLED else f"{row['problem_id']}#{H0}"


def round_rows(examples: Sequence[Mapping], assembled: Sequence[Mapping], number: int, batch_of: Mapping[str, int]) -> list[dict]:
    """A round's OWN training rows: its one-shot examples (one verified attempt of every problem with k >= 1:
    origin `attempt`), then the minimised proof of each problem only assembly resolved (origin `assembled`)."""
    rows = [{**example, "round": number, "batch": batch_of[example["problem_id"]], "origin": ATTEMPT} for example in examples]
    resolved = {example["problem_id"] for example in examples}
    rows += [{"problem_id": row["problem_id"], "side": row["side"], "attempt_id": None, "theorem": row["theorem"], "completion": row["completion"],
              "verified_attempts": 0, "round": number, "batch": row["batch"], "origin": ASSEMBLED, "resolved_after_attempt": row["resolved_after_attempt"],
              "minimised": row["minimised"]} for row in assembled if row["problem_id"] not in resolved]
    return [{"id": row_id(row), **row} for row in rows]


def check_harvest(harvest: Sequence[Mapping], heldout: Collection[str], base_map: Collection[str]) -> None:
    """What an H0 is refused for (ValueError) before any round: a row that is not an assembled proof with its
    statement, a problem twice, a held-out problem, a base-map problem."""
    lacking = [index for index, row in enumerate(harvest) if any(not row.get(field) for field in EXAMPLE_FIELDS) or row.get("assembled") is not True]
    if lacking:
        raise ValueError(f"{len(lacking)} rows of H0 are not assembled proofs with their statement (first: row {lacking[0]}). Refused")
    ids = [row["problem_id"] for row in harvest]
    twice = sorted(key for key, count in Counter(ids).items() if count > 1)
    if twice:
        raise ValueError(f"{len(twice)} problems stand twice in H0 (first: {twice[0]}): one proof a problem. Refused")
    for what, forbidden in (("held-out problems", heldout), ("problems of the base map", base_map)):
        found = sorted(set(ids) & set(forbidden))
        if found:
            raise ValueError(f"H0 holds {len(found)} {what} (first: {found[0]}). Refused")


def h0_rows(harvest: Sequence[Mapping], resolved: Collection[str], number: int) -> list[dict]:
    """H0's rows that stand in the training set after round `number`: those of problems the rounds so far have not
    resolved themselves (`resolved`: by an attempt or by assembly), in the file's order."""
    rows = [{**{field: row[field] for field in EXAMPLE_FIELDS}, "attempt_id": None, "verified_attempts": 0, "round": number, "batch": None, "origin": H0}
            for row in harvest if row["problem_id"] not in resolved]
    return [{"id": row_id(row), **row} for row in rows]


def training_set(by_round: Mapping[int, Sequence[Mapping]], number: int) -> list[Mapping]:
    """What M(`number`) is trained on, from the rounds' stored training examples: the OWN rows (origins `attempt`
    and `assembled`) of every round up to `number`, and the H0 rows that round `number`'s file holds. One proof a
    problem: the round's own where there is one (an H0 row of a problem a round resolved is refused here: it
    should have dropped out)."""
    own = [row for earlier in sorted(by_round) if earlier <= number for row in by_round[earlier] if row["origin"] != H0]
    rows = own + [row for row in by_round[number] if row["origin"] == H0]
    twice = sorted(problem for problem, count in Counter(row["problem_id"] for row in rows).items() if count > 1)
    if twice:
        raise ValueError(f"{len(twice)} problems have two proofs in the training set of round {number} (first: {twice[0]}): a problem has one. Refused")
    return rows


def training_order(rows: Sequence[Mapping], seed: int) -> list[Mapping]:
    """The order a model of this arm is trained in: the rows by a content hash of the seed and each row's id. A row's
    place among the others does not depend on which others there are, so a set with rows left out is trained in
    the same order with those rows left out."""
    return sorted(rows, key=lambda row: rank(seed, ORDER_LABEL, row["id"]))


def attempts_alone(ordered: Sequence[Mapping]) -> list[Mapping]:
    """The twin's rows: `ordered` (what `with` was trained on, in its order) with every row that is not a one-shot
    attempt's left out, none moved."""
    return [row for row in ordered if row["origin"] == ATTEMPT]


def by_origin(rows: Sequence[Mapping]) -> dict[str, int]:
    return {origin: sum(row["origin"] == origin for row in rows) for origin in ORIGINS}
