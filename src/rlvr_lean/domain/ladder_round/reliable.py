"""Reliability on the goal set, as a count. Spec: docs/spec/ladder-loop.spec.md, "L3d: train on what the episode
reaches", "The aim, as a count". Pure: stored rows in, counts out. No Lean.

An EPISODE is 8 one-shot attempts. A stored sampling of a goal problem (32 attempts, or 61) is cut into episodes of 8
in the order the attempts were drawn; the last few are left over. An episode is resolved BY ATTEMPTS ALONE when one of
its 8 attempts verified (on a problem attempted on both sides, when either side's did). A goal problem is solved
RELIABLY when it is resolved in at least half of its episodes.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.problem_pool.episodes import VERIFIED

EPISODE_ATTEMPTS = 8
QUARTER, HALF, NINE_TENTHS = 0.25, 0.5, 0.9


def attempt_flags(attempts: Sequence[Mapping]) -> dict[str, list[bool]]:
    """Of one stored sampling: problem -> one flag an attempt (did it verify, on either side), in the order drawn
    (`episode` is an attempt's place in its sampling)."""
    rows: dict[str, dict[int, bool]] = {}
    for row in attempts:
        found = rows.setdefault(row["problem_id"], {})
        found[row["episode"]] = found.get(row["episode"], False) or row["status"] == VERIFIED
    return {problem: [ok for _, ok in sorted(found.items())] for problem, found in rows.items()}


def episodes_of(flags: Sequence[bool], size: int = EPISODE_ATTEMPTS) -> list[bool]:
    """One sampling's attempts as episodes of `size` in the order drawn: each is resolved when one of its attempts
    verified. The attempts left over at the end are no episode."""
    return [any(flags[start:start + size]) for start in range(0, len(flags) - size + 1, size)]


def episodes_by_attempts(samplings: Sequence[Mapping[str, Sequence[bool]]], problem_ids: Sequence[str], size: int = EPISODE_ATTEMPTS) -> dict[str, list[bool]]:
    """Each problem's episodes over its samplings (`attempt_flags` of each), every sampling cut on its own."""
    return {problem: [resolved for flags in samplings for resolved in episodes_of(flags.get(problem, []), size)] for problem in problem_ids}


def reliability(episodes: Mapping[str, Sequence[bool]], problem_ids: Sequence[str]) -> dict:
    """Over `problem_ids`: how many are resolved in at least one, a quarter, half ("reliably") and nine tenths of
    their episodes. A problem with no episode is counted in none."""
    wins = {problem: sum(episodes.get(problem, ())) for problem in problem_ids}
    total = {problem: len(episodes.get(problem, ())) for problem in problem_ids}

    def at_least(share: float) -> int:
        return sum(total[problem] > 0 and wins[problem] >= share * total[problem] for problem in problem_ids)

    return {"problems": len(problem_ids), "episodes_a_problem": sorted(set(total.values())), "episodes": sum(total.values()),
            "resolved_episodes": sum(wins.values()), "solved_at_least_once": sum(count >= 1 for count in wins.values()),
            "in_a_quarter": at_least(QUARTER), "reliably": at_least(HALF), "in_nine_tenths": at_least(NINE_TENTHS)}
