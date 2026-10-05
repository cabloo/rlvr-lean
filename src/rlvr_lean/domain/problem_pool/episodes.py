"""Episodes, the pass rate they give a problem, the challenger's reward for it, and the band and rungs cut from it.
Spec: docs/spec/ladder-loop.spec.md, "An episode", "The reward and the band", "Held-out sets". Pure.

Until L3 an EPISODE is one attempt at the statement and one at its exact negation. It RESOLVES the problem when
Lean verifies either. An attempt that reached the token cap is a failure whatever Lean would say, and is not
sent. A problem's pass rate is k of n: the episodes that resolved it, of those it was given.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Mapping

from rlvr_lean.domain.problem_pool.certificates import FALSE_SIDE, TRUE_SIDE
from rlvr_lean.domain.problem_pool.selection import SoundnessAlarm

STATEMENT_SIDE, NEGATION_SIDE = "statement", "negation"
VERIFIED = "verified"
CAPPED_TOKENS = "capped_tokens"         # the attempt reached the token cap: a failure, never sent to Lean
NO_ANSWER = "no_answer"                 # Lean gave no verdict on the proof (server error; the header timed out twice)

BELOW_BAND, IN_BAND, ABOVE_BAND = "below", "in", "above"
RUNGS = (BELOW_BAND, IN_BAND, ABOVE_BAND)

# Why a problem has ONE side only (its negation is not attempted).
NEGATION_NOT_BUILT = "negation_not_built"       # the statement cannot be split into binders and type
NEGATION_NOT_EXACT = "negation_not_exact"       # Lean did not confirm the built negation is the statement's negation
NEGATION_REJECTED = "negation_rejected_lexically"


# ------------------------------------------------------------------------------------------------ the reward
def reward(k: int, n: int, target_rate: float) -> float:
    """The challenger's reward for a problem k of n solvers resolved: `p (1 − p)^a / [t (1 − t)^a]` with
    p = k/n and a = 1/t − 1. It is 1 at p = t and 0 at k = 0 and at k = n."""
    if not 0 < target_rate < 1:
        raise ValueError(f"the target rate must be strictly between 0 and 1, got {target_rate}")
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"k of n needs 0 <= k <= n and n > 0, got {k} of {n}")
    return reward_at(k / n, target_rate)


def reward_at(pass_rate: float, target_rate: float) -> float:
    power = 1 / target_rate - 1
    return pass_rate * (1 - pass_rate) ** power / (target_rate * (1 - target_rate) ** power)


def band(target_rate: float, floor: float) -> tuple[float, float]:
    """The pass rates the reward scores at least `floor`: an interval around the target rate (the reward
    rises to 1 at t and falls after it). Found by bisection on each side."""
    if not 0 < floor < 1:
        raise ValueError(f"the band's floor must be strictly between 0 and 1, got {floor}")

    def edge(inside: float, outside: float) -> float:
        for _ in range(80):
            middle = (inside + outside) / 2
            if reward_at(middle, target_rate) >= floor:
                inside = middle
            else:
                outside = middle
        return inside

    return edge(target_rate, 0.0), edge(target_rate, 1.0)


def rung(k: int, n: int, target_rate: float, floor: float) -> str | None:
    """Where k of n places a problem against the band: below it, in it or above it. None for k = 0: a problem
    with no success is in no rung (a rung holds only problems the base resolved at least once)."""
    if k == 0:
        return None
    if reward(k, n, target_rate) >= floor:
        return IN_BAND
    return BELOW_BAND if k / n < target_rate else ABOVE_BAND


# ------------------------------------------------------------------------------------------------ episodes
def attempt_id(problem_id: str, variant: str, side: str, episode: int) -> str:
    return f"{problem_id}#{variant}#{side}#{episode}"


def contradicted_side(known_side: str) -> str:
    """The side of a problem that can never verify: its certificate proves the other."""
    return {TRUE_SIDE: NEGATION_SIDE, FALSE_SIDE: STATEMENT_SIDE}[known_side]


def raise_on_contradiction(problem: Mapping, attempts: Iterable[Mapping]) -> None:
    """Raise SoundnessAlarm if an attempt verified on the side the problem's certificate contradicts."""
    forbidden = contradicted_side(problem["side"])
    for attempt in attempts:
        if attempt["side"] == forbidden and attempt["status"] == VERIFIED:
            raise SoundnessAlarm(f"{problem['problem_id']} is known {problem['side']} (a published proof our Lean verified) and "
                                 f"attempt {attempt['attempt_id']} verified its {forbidden}: a proof of both sides")


# ----------------------------------------------------------------- the side the certificate rules out (a switch)
# A problem's certificate proves one side, so an attempt on the other can never verify: checking it buys nothing
# but the soundness alarm. `episode.contradicted_side` says what is done with it:
#   all     every attempt on it is sent to Lean (the default: both sides are checked)
#   audit   only for an audit share of the PROBLEMS, drawn by a seeded hash, is that side sent to Lean (in full,
#           and a verified proof there still raises the alarm); for the others it is not sent and resolves nothing
# and `episode.skip_generating_contradicted_side` (with `audit`) says whether, for those others, it is sampled at all.
CHECK_ALL, CHECK_AUDIT = "all", "audit"
CHECKED, GENERATED_ONLY = "checked", "generated_only"       # what is done with one side of one problem; absent: not generated
NOT_CHECKED = "contradicted_side_not_checked"               # the status of an attempt that was sampled and not sent


def audited(problem_id: str, seed: int, share: float) -> bool:
    """Whether a problem is in the audit share: a seeded hash of its id, so a rerun audits the same problems."""
    return int(hashlib.sha256(f"{seed}:contradicted_side_audit:{problem_id}".encode()).hexdigest()[:8], 16) / 0x100000000 < share


def side_plan(problem: Mapping, settings: Mapping, seed: int) -> dict[str, str]:
    """What is done with each side of one problem under the episode settings: side -> CHECKED or
    GENERATED_ONLY. A side that is absent is not sampled (the negation of a problem that has none, or the
    contradicted side when generating it is switched off and the problem is not audited)."""
    mode = settings.get("contradicted_side", CHECK_ALL)
    if mode not in (CHECK_ALL, CHECK_AUDIT):
        raise ValueError(f"ladder_loop.episode.contradicted_side is {mode!r}; it must be {CHECK_ALL!r} or {CHECK_AUDIT!r}")
    ruled_out = contradicted_side(problem["side"])
    plan = {}
    for side in (STATEMENT_SIDE, NEGATION_SIDE):
        if side == NEGATION_SIDE and not problem.get("negation"):
            continue
        if side != ruled_out or mode == CHECK_ALL or audited(problem["problem_id"], seed, settings.get("contradicted_side_audit_share", 0.02)):
            plan[side] = CHECKED
        elif not settings.get("skip_generating_contradicted_side", False):
            plan[side] = GENERATED_ONLY
    return plan


def problem_result(problem: Mapping, attempts: list[Mapping], episodes: int, plan: Mapping[str, str] | None = None) -> dict:
    """k of n for one problem from its attempts: episode i resolved it when its attempt at the statement or
    its attempt at the negation verified. Also who resolved it, and what kept attempts from a verdict.
    `plan` says which sides were sampled (default: the statement, and the negation when the problem has one)."""
    by_episode: dict[int, dict[str, str]] = {}
    for attempt in attempts:
        by_episode.setdefault(attempt["episode"], {})[attempt["side"]] = attempt["status"]
    if plan is None:
        plan = {side: CHECKED for side in ((STATEMENT_SIDE, NEGATION_SIDE) if problem.get("negation") else (STATEMENT_SIDE,))}
    sides = len(plan)
    missing = [index for index in range(episodes) if set(by_episode.get(index, {})) != set(plan)]
    if missing or len(by_episode) != (episodes if sides else 0):
        raise ValueError(f"{problem['problem_id']}: {episodes} episodes of {sides} side(s) were wanted, and episodes "
                         f"{missing[:5]} are incomplete ({len(by_episode)} present)")
    ruled_out = contradicted_side(problem["side"])
    has_ruled_out_side = ruled_out == STATEMENT_SIDE or bool(problem.get("negation"))
    by_statement = sum(statuses.get(STATEMENT_SIDE) == VERIFIED for statuses in by_episode.values())
    by_negation = sum(statuses.get(NEGATION_SIDE) == VERIFIED for statuses in by_episode.values())
    resolved = sum(VERIFIED in statuses.values() for statuses in by_episode.values())
    flat = [status for statuses in by_episode.values() for status in statuses.values()]
    return {"problem_id": problem["problem_id"], "kind": problem["kind"], "side": problem["side"],
            "heldout_part": problem.get("heldout_part"), "episodes": episodes, "resolved": resolved,
            "resolved_by_statement": by_statement, "resolved_by_negation": by_negation,
            "sides": sides, "one_side_reason": problem.get("one_side_reason"),
            # What was done with the side the certificate rules out: checked, generated_only, not_generated; None
            # when the problem has no such side to attempt (a known-true problem without an exact negation).
            "contradicted_side": plan.get(ruled_out, "not_generated") if has_ruled_out_side else None,
            "attempts_capped": flat.count(CAPPED_TOKENS), "attempts_timed_out": flat.count("timeout"),
            "attempts_without_an_answer": flat.count(NO_ANSWER), "attempts_not_checked": flat.count(NOT_CHECKED)}
