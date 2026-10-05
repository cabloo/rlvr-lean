"""What a round trains the solver on. Spec: "A round" step 4, fixtures 2, 7 and 8.

Every problem of the round with k >= 1 gives ONE example: one of its verified proofs, chosen at random with the
round's seed, on whichever side was proved. A proof here is always an attempt this run's solver wrote and Lean
verified. A published proof cannot get in: this function is given the solver's attempts and nothing else, and the
data that reaches the GPU box holds no published proof.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.problem_pool.episodes import NEGATION_SIDE, STATEMENT_SIDE, VERIFIED
from rlvr_lean.domain.problem_pool.selection import rank


def training_examples(problems: Sequence[Mapping], attempts: Sequence[Mapping], seed: int) -> list[dict]:
    """One (theorem, proof) per problem that has a verified attempt, in the problems' order. The theorem is the
    statement, or its exact negation when that is the side the chosen attempt proved. The choice is the
    verified attempt first in a seeded order of attempt ids: the same seed gives the same target, and no proof
    has standing for being short."""
    verified: dict[str, list[Mapping]] = {}
    for attempt in attempts:
        if attempt["status"] == VERIFIED:
            verified.setdefault(attempt["problem_id"], []).append(attempt)
    examples = []
    for problem in problems:
        candidates = verified.get(problem["problem_id"])
        if not candidates:
            continue
        chosen = min(candidates, key=lambda attempt: rank(seed, "training_target", attempt["attempt_id"]))
        theorem = {STATEMENT_SIDE: problem["statement"], NEGATION_SIDE: problem.get("negation")}[chosen["side"]]
        if theorem is None:
            raise ValueError(f"{chosen['attempt_id']} verified the negation of {problem['problem_id']}, which has no built negation")
        examples.append({"problem_id": problem["problem_id"], "side": chosen["side"], "attempt_id": chosen["attempt_id"],
                         "theorem": theorem, "completion": chosen["completion"], "verified_attempts": len(candidates)})
    return examples


def training_summary(examples: Sequence[Mapping]) -> dict:
    lengths = sorted(len(example["completion"]) for example in examples)
    return {"examples": len(examples),
            "on_the_negation": sum(example["side"] == NEGATION_SIDE for example in examples),
            "median_proof_chars": lengths[len(lengths) // 2] if lengths else None,
            "problems_with_one_verified_attempt": sum(example["verified_attempts"] == 1 for example in examples)}
