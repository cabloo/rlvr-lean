"""The repair check's read, as the spec fixed it before the run. Spec: docs/spec/ladder-loop.spec.md, "L3a: the
repair check, no training", "The read, fixed before the run". Pure: rows in, numbers out.

An EPISODE is one blind first attempt and then, when it failed, up to `loops` - 1 more attempts in EACH of the three
arms: all three go on from the same failed first attempt. An episode whose first attempt verified is resolved in every
arm (its difference is zero). Every comparison is paired by episode, each problem's value is the mean over its
episodes, and the interval is a 95% bootstrap over problems.

TRIMMED loops (spec item 2a). Where no goal was left at a resuming arm's cut and the kept lines verified on their own,
that loop resolves the episode with nothing generated. The stage also runs the loop as the blind fall-back it would
have been without that rule, and goes on from there, so every episode with a trimmed loop holds TWO readings of that
arm: the spec's (`loops`, `resolved_at`; the last loop is the trimmed one) and `without_trimming` (the same loops
before it, then the fall-back and what followed). Every function here reads the spec's unless told `trimming=False`.

  episode_rows      one row an episode from the stored attempts and state requests: what each arm did, loop by loop
  problem_rows      one row a problem: its episodes resolved within the attempts of an episode, by arm
  repair_read       the primary and the three comparisons by set, the single step, the goal set as gained against
                    lost, the budget, what the model did with the state, the trimmed resolutions and the three main
                    reads without them, the two "can this run see a win" checks, and the branch
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Collection, Mapping, Sequence

from rlvr_lean.domain.evaluation.bootstrap import paired_bootstrap
from rlvr_lean.domain.ladder_round.read import GOAL
from rlvr_lean.domain.ladder_round.rounds import sign_test
from rlvr_lean.domain.problem_pool.episodes import ABOVE_BAND, BELOW_BAND, IN_BAND, VERIFIED
from rlvr_lean.domain.repair import ARMS, BLIND, FIRST, RESUME_ARMS, RESUME_WITH_STATE, RESUME_WITHOUT_STATE

HARD, EVERY = "hard", "all"
GROUPS = (GOAL, BELOW_BAND, IN_BAND, ABOVE_BAND)            # where L1's placing episodes put a measured problem of H
SETS = {GOAL: (GOAL,), BELOW_BAND: (BELOW_BAND,), IN_BAND: (IN_BAND,), ABOVE_BAND: (ABOVE_BAND,),
        HARD: (GOAL, BELOW_BAND),                           # the primary's problems: G and the below-band rung
        EVERY: GROUPS}
COMPARISONS = {"resume_with_state_minus_blind": (RESUME_WITH_STATE, BLIND),
               "resume_without_state_minus_blind": (RESUME_WITHOUT_STATE, BLIND),
               "resume_with_state_minus_resume_without_state": (RESUME_WITH_STATE, RESUME_WITHOUT_STATE)}
RESUMED, BLIND_ATTEMPT = "resume", "blind"                  # how one attempt was prompted
TRIMMED = "trimmed"                                         # ... or not prompted at all: the kept lines verified on their own
MORE_TOKENS_ALLOWED = 0.10                                  # the resume arm may generate this much more than the blind arm
HALF = 0.5                                                  # both "can this run see a win" checks are "at least half"

STRONGER = "REPAIR IS A STRONGER SEARCH STEP"               # primary above zero, interval clear of zero
NOT_SHOWN = "NOT SHOWN AT THIS SIZE"                        # the interval contains zero
WORSE_UNTRAINED = "RESUMING IS WORSE THAN RESAMPLING FOR THIS MODEL UNTRAINED"      # the interval lies below zero
INCONCLUSIVE = "INCONCLUSIVE"                               # the run could not have seen a win: fix and run again
BRANCHES = (STRONGER, NOT_SHOWN, WORSE_UNTRAINED, INCONCLUSIVE)


# ------------------------------------------------------------------------------------------------ episodes
def _step(row: Mapping, request: Mapping | None, reading_trims: bool) -> dict:
    """One loop of one arm as the read uses it. `request`: the state request made from the failed proof this loop
    went on from. `reading_trims`: the loop belongs to the spec's reading, in which a kept-lines check is made."""
    check = request.get("trim_status") if request and reading_trims else None
    return {"loop": row["loop"], "how": row["how"], "had_state": bool(row.get("had_state")), "no_state": row.get("no_state"),
            "status": row["status"], "token_count": row["token_count"], "prompt_tokens": row["prompt_tokens"],
            "sent_to_lean": bool(row["sent_to_lean"]), "kept_lines": row.get("kept_lines"), "cut": row.get("cut"),
            "repeats_failed_step": row.get("repeats_failed_step"),
            "state_requested": bool(request and request.get("requested")),
            # The kept lines checked on their own before this loop (no goal was left at the cut): "verified" on the
            # trimmed loop itself, "failed" on the blind attempt the loop then was; None when no such check was made.
            "kept_lines_check": None if check is None else "verified" if row["how"] == TRIMMED else "failed"}


def episode_rows(problems: Sequence[Mapping], attempts: Sequence[Mapping], states: Sequence[Mapping], loops: int,
                 arms: Sequence[str] = ARMS, resuming: Collection[str] = RESUME_ARMS) -> list[dict]:
    """One row for every episode on the side its problem's certificate allows. `attempts`: every stored attempt
    row (loop 0 is the first attempt, arm `first`; loops 1 to `loops` - 1 are the arms'; a row marked `untrimmed`
    belongs to the reading that does not trim). `states`: every stored state request. Refused unless each arm
    went on from every failed first attempt to its first verified attempt or to the last loop, in both readings:
    the read needs whole episodes. `arms`, and those of them a state is asked for (`resuming`): L3a's unless
    another check names its own (L3a2: `alternate.py`)."""
    firsts: dict[tuple[str, int], Mapping] = {}
    later: dict[tuple[str, str, int, int], Mapping] = {}
    beside: dict[tuple[str, str, int, int], Mapping] = {}
    for row in attempts:
        if row.get("audit"):
            continue
        if row["arm"] == FIRST:
            firsts[row["problem_id"], row["episode"]] = row
        else:
            (beside if row.get("untrimmed") else later)[row["arm"], row["problem_id"], row["episode"], row["loop"]] = row
    asked = {(row["arm"], row["problem_id"], row["episode"], row["after_loop"]): row for row in states}

    def follow(rows: Mapping, arm: str, problem_id: str, episode: int, start: int, steps: list[dict], reading_trims: bool) -> int | None:
        """Read one arm's loops from `start` to its first verified attempt: the attempt that resolved it, or None."""
        for loop in range(start, loops):
            row = rows.get((arm, problem_id, episode, loop))
            if row is None:
                raise ValueError(f"{problem_id} episode {episode}: the {arm} arm has no attempt at loop {loop} and had not verified "
                                 "before it: the read needs whole episodes")
            request = asked.get((FIRST if loop == 1 else arm, problem_id, episode, loop - 1)) if arm in resuming else None
            steps.append(_step(row, request, reading_trims))
            if row["status"] == VERIFIED:
                return loop + 1
        return None

    found = []
    for problem in problems:
        episodes = sorted(episode for problem_id, episode in firsts if problem_id == problem["problem_id"])
        for episode in episodes:
            first = firsts[problem["problem_id"], episode]
            entry = {"problem_id": problem["problem_id"], "group": problem["group"], "side": problem["side"], "episode": episode,
                     "failed_first": first["status"] != VERIFIED,
                     "first": {key: first[key] for key in ("status", "token_count", "prompt_tokens", "sent_to_lean")}, "arms": {}}
            for arm in arms:
                steps: list[dict] = []
                resolved_at = follow(later, arm, problem["problem_id"], episode, 1, steps, True) if entry["failed_first"] else 1
                trimmed_at = steps[-1]["loop"] if steps and steps[-1]["how"] == TRIMMED else None
                own = {"resolved_at": resolved_at, "loops": steps, "trimmed_at": trimmed_at, "without_trimming": None}
                if trimmed_at is not None:
                    # The other reading: the same loops before the trimmed one (a kept-lines check is no part of it),
                    # then that loop as the blind fall-back the stage also ran, and what followed it.
                    others = [{**step, "kept_lines_check": None} for step in steps[:-1]]
                    at = follow(beside, arm, problem["problem_id"], episode, trimmed_at, others, False)
                    own["without_trimming"] = {"resolved_at": at, "loops": others}
                entry["arms"][arm] = own
            found.append(entry)
    return found


def _reading(row: Mapping, arm: str, trimming: bool = True) -> Mapping:
    """One arm of one episode: the spec's reading, or (`trimming` False) the one in which a trimmed loop is the
    blind fall-back it would have been. The two are the same wherever no loop was trimmed."""
    own = row["arms"][arm]
    other = own.get("without_trimming")
    return own if trimming or other is None else other


def resolved(row: Mapping, arm: str, within: int, trimming: bool = True) -> bool:
    """Whether the episode was resolved in `arm` within `within` attempts (the first attempt is attempt 1)."""
    at = _reading(row, arm, trimming)["resolved_at"]
    return at is not None and at <= within


def problem_rows(rows: Sequence[Mapping], loops: int, arms: Sequence[str] = ARMS, resuming: Sequence[str] = RESUME_ARMS) -> list[dict]:
    """One row a problem: its episodes, how many failed at the first attempt, how many each arm resolved, and
    how many of a resuming arm's were resolved by a trimmed loop."""
    by_problem: dict[str, list[Mapping]] = {}
    for row in rows:
        by_problem.setdefault(row["problem_id"], []).append(row)
    return [{"problem_id": problem_id, "group": own[0]["group"], "side": own[0]["side"], "episodes": len(own),
             "failed_first": sum(row["failed_first"] for row in own),
             **{f"resolved_{arm}": sum(resolved(row, arm, loops) for row in own) for arm in arms},
             **{f"trimmed_{arm}": sum(row["arms"][arm].get("trimmed_at") is not None for row in own) for arm in resuming}}
            for problem_id, own in by_problem.items()]


def _of(rows: Sequence[Mapping], groups: Collection[str]) -> list[Mapping]:
    return [row for row in rows if row["group"] in groups]


def _share(count: int, total: int) -> float | None:
    return round(count / total, 5) if total else None


# --------------------------------------------------------------------------------------- paired comparisons
def _interval(differences: Sequence[float], resamples: int, seed: int) -> dict:
    if not differences:
        return {"problems": 0, "mean": None, "low": None, "high": None}
    interval = paired_bootstrap(differences, resamples=resamples, seed=seed)
    return {"problems": len(differences), "mean": round(interval.mean, 5), "low": round(interval.low, 5), "high": round(interval.high, 5)}


def paired(rows: Sequence[Mapping], arm: str, against: str, within: int, resamples: int, seed: int, within_against: int | None = None,
           trimming: bool = True) -> dict:
    """Episodes resolved within `within` attempts, `arm` minus `against`: paired by episode (both went on from
    the same first attempt), each problem's mean over its episodes, a 95% bootstrap over problems."""
    within_against = within if within_against is None else within_against
    by_problem: dict[str, list[tuple[bool, bool]]] = {}
    for row in rows:
        by_problem.setdefault(row["problem_id"], []).append((resolved(row, arm, within, trimming), resolved(row, against, within_against, trimming)))
    differences = [sum(first - second for first, second in own) / len(own) for own in by_problem.values()]
    result = _interval(differences, resamples, seed)
    result["episodes"] = len(rows)
    if by_problem:
        result[arm] = round(sum(sum(first for first, _ in own) / len(own) for own in by_problem.values()) / len(by_problem), 5)
        result[against] = round(sum(sum(second for _, second in own) / len(own) for own in by_problem.values()) / len(by_problem), 5)
    return result


def resolved_by_attempt(rows: Sequence[Mapping], loops: int, arms: Sequence[str] = ARMS) -> dict:
    """The share of episodes resolved within 1, 2, ... attempts, by arm (each problem's mean over its episodes,
    then the mean over problems)."""
    by_problem: dict[str, list[Mapping]] = {}
    for row in rows:
        by_problem.setdefault(row["problem_id"], []).append(row)
    if not by_problem:
        return {}
    return {arm: {str(within): round(sum(sum(resolved(row, arm, within) for row in own) / len(own) for own in by_problem.values()) / len(by_problem), 5)
                  for within in range(1, loops + 1)} for arm in arms}


# ------------------------------------------------------------------------------------------ the single step
def _next(row: Mapping, arm: str, trimming: bool = True) -> Mapping:
    return _reading(row, arm, trimming)["loops"][0]


def single_step(rows: Sequence[Mapping], trimming: bool = True) -> dict:
    """Given a failed first attempt, the share whose NEXT attempt verifies, by arm, and how many of those were
    trimmed loops (the kept lines verified; nothing was generated). For the arm that resumes with the state,
    also the next attempts whose prompt did hold a state, beside the other arms' on the same episodes."""
    failed = [row for row in rows if row["failed_first"]]
    result = {"failed_first_attempts": len(failed)}
    for arm in ARMS:
        verified = sum(_next(row, arm, trimming)["status"] == VERIFIED for row in failed)
        result[arm] = {"verified": verified, "share": _share(verified, len(failed)),
                       "of_which_trimmed": sum(_next(row, arm, trimming)["how"] == TRIMMED for row in failed)}
    held = [row for row in failed if _next(row, RESUME_WITH_STATE, trimming)["how"] == RESUMED]
    result["where_the_prompt_held_a_state"] = {
        "attempts": len(held),
        RESUME_WITH_STATE: _share(sum(_next(row, RESUME_WITH_STATE, trimming)["status"] == VERIFIED for row in held), len(held)),
        RESUME_WITHOUT_STATE: _share(sum(_next(row, RESUME_WITHOUT_STATE, trimming)["status"] == VERIFIED for row in held), len(held)),
        BLIND: _share(sum(_next(row, BLIND, trimming)["status"] == VERIFIED for row in held), len(held))}
    return result


# ---------------------------------------------------------------------------------- the goal set, by problem
def gained_against_lost(rows: Sequence[Mapping], arm: str, against: str, within: int, trimming: bool = True) -> dict:
    """Problems resolved in ANY episode: by `arm` and not by `against` (gained), the reverse (lost), and the
    two-sided sign test on those two counts."""
    by_problem: dict[str, list[Mapping]] = {}
    for row in rows:
        by_problem.setdefault(row["problem_id"], []).append(row)
    first = {problem_id for problem_id, own in by_problem.items() if any(resolved(row, arm, within, trimming) for row in own)}
    second = {problem_id for problem_id, own in by_problem.items() if any(resolved(row, against, within, trimming) for row in own)}
    gained, lost = len(first - second), len(second - first)
    return {"problems": len(by_problem), f"resolved_{arm}": len(first), f"resolved_{against}": len(second), "by_both": len(first & second),
            "gained": gained, "lost": lost, "sign_test_p": sign_test(gained, lost)}


# ------------------------------------------------------------------------------------------ trimmed loops
def trimmed_resolutions(rows: Sequence[Mapping], loops: int, resuming: Sequence[str] = RESUME_ARMS) -> dict:
    """Spec item 2a, reported apart: the episodes a resuming arm resolved by a trimmed loop (no goal was left at
    the cut and the kept lines verified on their own), by set and by loop, and how many of them the same arm
    resolved anyway where that loop was the blind fall-back instead."""
    result = {}
    for arm in resuming:
        trimmed = [row for row in rows if row["arms"][arm].get("trimmed_at") is not None]
        result[arm] = {"resolutions": len(trimmed),
                       "by_set": {set_name: sum(row["group"] in groups for row in trimmed) for set_name, groups in SETS.items()},
                       "by_loop": dict(sorted(Counter(str(row["arms"][arm]["trimmed_at"]) for row in trimmed).items())),
                       "resolved_without_trimming_too": sum(resolved(row, arm, loops, trimming=False) for row in trimmed)}
    return result


# -------------------------------------------------------------------------------------------------- budget
def arm_budget(rows: Sequence[Mapping], arm: str, through: int | None = None) -> dict:
    """What `arm` spent AFTER the first attempts (which the arms share): attempts, generated and prompt tokens,
    and Lean checks: an attempt that was sent is one check; a resume loop's state request is one more; and where
    no goal was left at the cut, the kept lines checked on their own are one more (it is the trimmed loop's only
    check besides the state request, and a trimmed loop generates nothing). `through`: only the loops up to that
    one."""
    steps = [step for row in rows for step in row["arms"][arm]["loops"] if through is None or step["loop"] <= through]
    sent = sum(step["sent_to_lean"] for step in steps if step["how"] != TRIMMED)
    requests, kept_lines = sum(step["state_requested"] for step in steps), sum(step.get("kept_lines_check") is not None for step in steps)
    return {"attempts": len(steps), "of_which_trimmed": sum(step["how"] == TRIMMED for step in steps),
            "generated_tokens": sum(step["token_count"] for step in steps),
            "prompt_tokens": sum(step["prompt_tokens"] for step in steps), "attempts_sent_to_lean": sent, "state_requests": requests,
            "kept_lines_checks": kept_lines, "lean_checks": sent + requests + kept_lines}


def budget(rows: Sequence[Mapping], primary_rows: Sequence[Mapping], loops: int, resamples: int, seed: int) -> dict:
    """Generated tokens, prompt tokens and Lean checks by arm, and the comparison at equal tokens. `rows`: every
    episode (the cost of the run); `primary_rows`: the primary's episodes, on which the equal-token read is made."""
    by_arm = {arm: arm_budget(rows, arm) for arm in ARMS}
    firsts = {"attempts": len(rows), "generated_tokens": sum(row["first"]["token_count"] for row in rows),
              "prompt_tokens": sum(row["first"]["prompt_tokens"] for row in rows), "lean_checks": sum(bool(row["first"]["sent_to_lean"]) for row in rows)}
    hard = {arm: arm_budget(primary_rows, arm) for arm in ARMS}
    blind_tokens, resume_tokens = hard[BLIND]["generated_tokens"], hard[RESUME_WITH_STATE]["generated_tokens"]
    ratio = round(resume_tokens / blind_tokens, 4) if blind_tokens else None
    result = {"what": "after the first attempts, which the three arms share: an arm's attempts, the tokens it generated, the tokens of its prompts, and "
                      "its Lean checks (each attempt sent, one more for each resume loop's state request, and one more where the kept lines "
                      "were checked on their own); a trimmed loop is an attempt with no tokens",
              "first_attempts": firsts, "by_arm": by_arm, "by_arm_on_the_primarys_problems": hard,
              "generated_tokens_resume_with_state_over_blind_on_the_primarys_problems": ratio,
              "resume_generates_more_than_10_percent_more": bool(ratio is not None and ratio > 1 + MORE_TOKENS_ALLOWED)}
    if not result["resume_generates_more_than_10_percent_more"]:
        result["equal_tokens"] = ("the arm that resumes with the state generated no more than 10% more tokens than the blind arm: the primary is "
                                  "read as it stands")
        return result
    # The spec: "the blind arm is read at the number of attempts that matches". The blind arm was sampled to
    # `loops` - 1 more attempts and the match needs more than that, so that read cannot be made from this run. What
    # can be read at equal tokens from what was sampled is its mirror: the resume arm at the loops whose tokens match.
    matching = round((loops - 1) * ratio, 2)
    kept_loops = 0
    for through in range(1, loops):
        if arm_budget(primary_rows, RESUME_WITH_STATE, through)["generated_tokens"] <= blind_tokens * (1 + MORE_TOKENS_ALLOWED):
            kept_loops = through
    result["equal_tokens"] = {
        "blind_attempts_after_the_first_that_match": matching, "blind_attempts_after_the_first_that_were_sampled": loops - 1,
        "the_blind_arm_at_the_matching_number_can_be_read": False,
        "why": f"the blind arm was sampled to {loops - 1} more attempts an episode and the resume arm's tokens match {matching}",
        "mirror": {"what": f"the resume arm read at its first {kept_loops} loops, whose generated tokens are within 10% of the blind arm's at {loops - 1}; "
                           "the blind arm at all of its attempts",
                   "resume_loops_read": kept_loops,
                   "resume_generated_tokens": arm_budget(primary_rows, RESUME_WITH_STATE, kept_loops)["generated_tokens"], "blind_generated_tokens": blind_tokens,
                   "primary": paired(primary_rows, RESUME_WITH_STATE, BLIND, 1 + kept_loops, resamples, seed, within_against=loops)}}
    return result


# ------------------------------------------------------------------------------- what the model did with it
def diagnostics(rows: Sequence[Mapping]) -> dict:
    """The share of repair loops with a state (and why the others had none); the share whose first step repeats
    the step that had just failed; how far along the proof the cut moves from loop to loop."""
    result = {}
    for arm in RESUME_ARMS:
        steps = [step for row in rows for step in row["arms"][arm]["loops"]]
        resumed = [step for step in steps if step["how"] == RESUMED]
        by_loop = {}
        for loop in sorted({step["loop"] for step in steps}):
            here = [step for step in steps if step["loop"] == loop]
            there = [step for step in here if step["how"] == RESUMED]
            by_loop[str(loop)] = {"loops": len(here), "resumed": len(there), "share_resumed": _share(len(there), len(here)),
                                  "mean_kept_lines": round(sum(step["kept_lines"] for step in there) / len(there), 3) if there else None}
        judged = [step for step in resumed if step["repeats_failed_step"] is not None]
        moves = []
        for row in rows:
            own = row["arms"][arm]["loops"]
            moves += [later["kept_lines"] - earlier["kept_lines"] for earlier, later in zip(own, own[1:])
                      if earlier["how"] == RESUMED and later["how"] == RESUMED]
        result[arm] = {
            "repair_loops": len(steps), "with_a_state": len(resumed), "share_with_a_state": _share(len(resumed), len(steps)),
            "trimmed": sum(step["how"] == TRIMMED for step in steps),
            "without_a_state_by_reason": dict(Counter(step["no_state"] for step in steps if step["how"] == BLIND_ATTEMPT).most_common()),
            "by_loop": by_loop,
            "nothing_kept": sum(step["kept_lines"] == 0 for step in resumed), "whole_proof_kept": sum(step["cut"] == "whole_proof" for step in resumed),
            "first_step_repeats_the_failed_step": {"loops_with_a_failed_step": len(judged), "repeats": sum(step["repeats_failed_step"] for step in judged),
                                                   "share": _share(sum(step["repeats_failed_step"] for step in judged), len(judged))},
            "cut_from_loop_to_loop": {"what": "kept proof lines at a loop minus kept lines at the loop before it, where both resumed",
                                      "pairs": len(moves), "mean_change_in_kept_lines": round(sum(moves) / len(moves), 3) if moves else None,
                                      "share_further_along": _share(sum(move > 0 for move in moves), len(moves)),
                                      "share_at_the_same_line": _share(sum(move == 0 for move in moves), len(moves)),
                                      "share_further_back": _share(sum(move < 0 for move in moves), len(moves))}}
    return result


# ------------------------------------------------------------------------------- can this run see a win
def can_see_a_win(rows: Sequence[Mapping]) -> dict:
    """The two checks of the spec. On the above-band rung the next attempt of the arm that resumes with the
    state must verify at least half as often as the blind arm's (less: the prompt or the cut is broken). A state
    must have come back for at least half of the failed first attempts (less: the cut is broken). A check that
    cannot be read (no failed first attempt there, or a blind arm that verified none) has not passed. A trimmed
    loop is a resolved loop, as the spec's letter has it; how many of the successes were trimmed is said beside."""
    above = single_step(_of(rows, SETS[ABOVE_BAND]))
    blind, resume = above[BLIND]["share"], above[RESUME_WITH_STATE]["share"]
    step_passes = bool(blind) and resume is not None and resume >= HALF * blind
    failed = [row for row in rows if row["failed_first"]]
    held = sum(_next(row, RESUME_WITH_STATE)["how"] == RESUMED for row in failed)
    state_share = _share(held, len(failed))
    states_pass = state_share is not None and state_share >= HALF
    where = above["where_the_prompt_held_a_state"]
    return {
        "the_resumed_next_attempt_on_the_above_band_rung": {
            "failed_first_attempts": above["failed_first_attempts"], "blind_share_verified": blind, "resume_with_state_share_verified": resume,
            "resume_with_state_verified": above[RESUME_WITH_STATE]["verified"], "of_which_trimmed": above[RESUME_WITH_STATE]["of_which_trimmed"],
            "needed": "at least half of the blind arm's share", "passes": step_passes,
            # Beside the spec's check, which counts the arm's blind fall-backs too: only the next attempts whose prompt held a state.
            "only_where_the_prompt_held_a_state": where,
            "only_where_the_prompt_held_a_state_at_least_half_of_blind": (
                None if not where["attempts"] or not where[BLIND] else where[RESUME_WITH_STATE] >= HALF * where[BLIND])},
        "a_state_for_the_failed_first_attempts": {"failed_first_attempts": len(failed), "with_a_state": held, "share": state_share,
                                                  "needed": "at least half", "passes": states_pass},
        "passes": step_passes and states_pass}


# -------------------------------------------------------------------------------------------------- branch
@dataclass(frozen=True)
class Branch:
    name: str
    reason: str


def repair_branch(primary: Mapping, checks: Mapping) -> Branch:
    """The branch the numbers select, in the spec's words. The two "can this run see a win" checks come first:
    when either fails the run is INCONCLUSIVE, not a verdict."""
    if not checks["passes"]:
        step, states = checks["the_resumed_next_attempt_on_the_above_band_rung"], checks["a_state_for_the_failed_first_attempts"]
        failed = []
        if not step["passes"]:
            failed.append(f"on the above-band rung the resumed next attempt verifies {step['resume_with_state_share_verified']} of the time against the "
                          f"blind arm's {step['blind_share_verified']} (at least half of it is needed): the prompt or the cut is broken")
        if not states["passes"]:
            failed.append(f"a state came back for {states['share']} of the failed first attempts (at least half is needed): the cut is broken")
        return Branch(INCONCLUSIVE, "this run could not have seen a win: " + "; ".join(failed) + ". Fix it and run again; this is not a verdict")
    if primary.get("mean") is None:
        return Branch(INCONCLUSIVE, "the primary could not be computed: no episode on G or on the below-band rung")
    interval = f"{primary['mean']} [{primary['low']}, {primary['high']}]"
    if primary["low"] > 0:
        return Branch(STRONGER, f"the primary is {interval}, above zero with an interval clear of zero: repair is a stronger search step; build it into "
                                "the round (episodes with loops) and train on failed-then-repaired proofs (L3b, its own read)")
    if primary["high"] < 0:
        return Branch(WORSE_UNTRAINED, f"the primary is {interval}, below zero: resuming is worse than resampling FOR THIS MODEL UNTRAINED; that is a "
                                       "finding about this format, not a verdict on repair")
    return Branch(NOT_SHOWN, f"the primary is {interval}, an interval that contains zero: not shown at this size; the two diagnostics (the share of "
                             "loops with a state, the share that repeat the failed step) say whether the model ignores the state or repeats itself, "
                             "which is what training on repair would be for, and that needs a decision")


# ------------------------------------------------------------------------------------------------ the read
def repair_read(rows: Sequence[Mapping], loops: int, resamples: int, seed: int) -> dict:
    """Everything the spec's read names, from the episode rows."""
    hard = _of(rows, SETS[HARD])
    primary = paired(hard, RESUME_WITH_STATE, BLIND, loops, resamples, seed)
    checks = can_see_a_win(rows)
    branch = repair_branch(primary, checks)
    goal = _of(rows, SETS[GOAL])
    return {
        "primary": primary,
        "comparisons": {name: {set_name: paired(_of(rows, groups), arm, against, loops, resamples, seed) for set_name, groups in SETS.items()}
                        for name, (arm, against) in COMPARISONS.items()},
        "resolved_within_attempts": {set_name: resolved_by_attempt(_of(rows, groups), loops) for set_name, groups in SETS.items()},
        "single_step": {set_name: single_step(_of(rows, groups)) for set_name, groups in SETS.items()},
        "goal_set_by_problem": {name: gained_against_lost(goal, arm, against, loops) for name, (arm, against) in COMPARISONS.items()},
        "trimmed": trimmed_resolutions(rows, loops),
        # The same three reads with every trimmed loop as the blind fall-back it would have been (and what followed).
        "without_trimming": {
            "primary": paired(hard, RESUME_WITH_STATE, BLIND, loops, resamples, seed, trimming=False),
            "single_step": {set_name: single_step(_of(rows, groups), trimming=False) for set_name, groups in SETS.items()},
            "goal_set_by_problem": {name: gained_against_lost(goal, arm, against, loops, trimming=False) for name, (arm, against) in COMPARISONS.items()}},
        "budget": budget(rows, hard, loops, resamples, seed),
        "diagnostics": diagnostics(rows),
        "can_this_run_see_a_win": checks,
        "branch": {"name": branch.name, "reason": branch.reason},
    }
