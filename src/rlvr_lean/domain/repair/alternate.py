"""The second repair check (L3a2): one repair step after each fresh failure, then start over. Spec:
docs/spec/ladder-loop.spec.md, "L3a2: one repair step after each fresh failure, then start over (no training)".
Pure: rows in, numbers out. What is done to ONE failed proof is L3a's (`cut.py`), and so are the per-episode rows
(`read.episode_rows`, given this check's arms).

THE EPISODE UNDER TEST ("alternate"). Up to `loops` attempts, stopping at the first verified one: a whole proof from
the plain prompt, ONE repair step from it, a whole proof from the plain prompt, one repair step from THAT, a whole
proof. In loops (attempt n is loop n - 1): the odd loops are the repair steps, and each starts from the attempt before
it, which is always a fresh one: never from a repair step's own output. A repair step is L3a's resume with the state
(the cut, the state by `all_goals sorry`, the prover's own comment format); where no goal is left at the cut and the
kept lines verify, it is TRIMMED and resolves the episode; where no state can be had it is a blind attempt, counted
with its reason. The other arm is BLIND: whole proofs from the plain prompt throughout. Both go on from the same
failed first attempt.

KNOWN COPIES. In BOTH arms a proof whose text equals one Lean has already rejected in the same episode is not sent to
Lean: it is a failed attempt, its tokens are counted, and its status is `KNOWN_COPY`. "Text" is the proof as it stands
in the file that is checked (`checked_text`); "rejected" is an attempt that was sent and that Lean failed (`is_rejected`:
not one that was never sent, and not one Lean gave no answer for). A known copy was not sent, so Lean said nothing a
cut could be made by: a repair step that would start from one is a blind attempt, with the reason `KNOWN_COPY`.

  rejected, copy_of     the proofs Lean has rejected so far in one arm of one episode; which of them a proof repeats
  by_position           the verified share by attempt position in each arm (positions 2 and 4 are the repair steps)
  budget                generated tokens, prompt tokens, Lean checks and known copies by arm; the read at equal tokens
  can_see_a_win         the two checks that are read before the branch
  alternate_read        the primary, the secondaries, the budget, the checks and the branch
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.read import GOAL
from rlvr_lean.domain.problem_pool.episodes import NO_ANSWER, VERIFIED
from rlvr_lean.domain.repair import BLIND
from rlvr_lean.domain.repair.read import (
    BLIND_ATTEMPT,
    HALF,
    HARD,
    INCONCLUSIVE,
    MORE_TOKENS_ALLOWED,
    NOT_SHOWN,
    RESUMED,
    SETS,
    TRIMMED,
    Branch,
    _of,
    _share,
    arm_budget,
    gained_against_lost,
    paired,
    resolved_by_attempt,
    trimmed_resolutions,
)

ALTERNATE = "alternate"             # the episode under test: blind, repair, blind, repair, blind
ARMS = (BLIND, ALTERNATE)
REPAIR_ARMS = (ALTERNATE,)          # the arms with repair steps: a state is asked for the failed proofs they start from
KNOWN_COPY = "known_copy"           # an attempt's status: its proof was already rejected in the episode and was not sent again;
                                    # and why a repair step that would start from such an attempt had no state

THE_EPISODE = "THIS IS THE EPISODE A ROUND SHOULD USE"                  # primary above zero, interval clear of zero
DID_NOT_REPEAT = "L3A'S SINGLE STEP DID NOT REPEAT ON FRESH ATTEMPTS"   # the interval lies below zero
BRANCHES = (THE_EPISODE, NOT_SHOWN, DID_NOT_REPEAT, INCONCLUSIVE)       # NOT_SHOWN: the interval contains zero


# --------------------------------------------------------------------------------------------- the episode
def is_repair_step(loop: int) -> bool:
    """Whether the alternate arm's attempt at `loop` is a repair step (loop 0 is the first attempt): attempts 2 and
    4 of 5. It starts from the attempt at `loop` - 1, a fresh one."""
    return loop % 2 == 1


def checked_text(proof: str) -> str:
    """A proof's text as it stands in the file that is checked (`build_proof_source` drops the white space at its
    end): two attempts at one statement with the same text are the same file to Lean."""
    return proof.rstrip()


def is_rejected(row: Mapping) -> bool:
    """Whether Lean rejected this attempt: it was sent, and Lean's answer was a failure (an error, the Lean limit,
    a `sorry`, an axiom). Not an attempt that was never sent (the token cap, the lexical filter, a known copy),
    and not one Lean gave no answer for: the same text is then sent again."""
    return bool(row["sent_to_lean"]) and row["status"] not in (VERIFIED, NO_ANSWER)


def rejected(chain: Sequence[Mapping]) -> dict[str, int]:
    """The proofs Lean has rejected so far in one arm of one episode: text -> the loop of the earliest attempt
    that wrote it. `chain`: the stored rows of the first attempt and of the arm's attempts after it."""
    found: dict[str, int] = {}
    for row in chain:
        if is_rejected(row):
            found.setdefault(checked_text(row["proof"]), row["loop"])
    return found


def copy_of(proof: str, already: Mapping[str, int] | None) -> int | None:
    """The loop of the rejected attempt this proof repeats, or None: it is then a proof Lean has not judged in the
    episode. `already`: `rejected` of the episode so far (None: a check that sends every proof, L3a)."""
    return (already or {}).get(checked_text(proof))


# ---------------------------------------------------------------------------------------------- positions
def _steps_at(rows: Sequence[Mapping], arm: str, loop: int) -> list[Mapping]:
    return [step for row in rows for step in row["arms"][arm]["loops"] if step["loop"] == loop]


def by_position(rows: Sequence[Mapping], loops: int) -> dict:
    """The verified share by attempt POSITION in each arm: of the attempts made at a position (the episodes the
    arm had not resolved before it), how many verified, and how many were known copies. Position 1 is the shared
    first attempt. For the alternate arm's repair steps (positions 2 and 4) also what they were: prompted with a
    state, trimmed (the kept lines verified; nothing generated), or a blind attempt for want of a state."""
    firsts = sum(not row["failed_first"] for row in rows)
    result: dict = {"first_attempts": {"attempts": len(rows), "verified": firsts, "share": _share(firsts, len(rows))}}
    for arm in ARMS:
        result[arm] = {}
        for loop in range(1, loops):
            steps = _steps_at(rows, arm, loop)
            verified = sum(step["status"] == VERIFIED for step in steps)
            entry = {"attempts": len(steps), "verified": verified, "share": _share(verified, len(steps)),
                     "known_copies": sum(step["status"] == KNOWN_COPY for step in steps)}
            if arm == ALTERNATE and is_repair_step(loop):
                held = [step for step in steps if step["how"] == RESUMED]
                entry.update({"repair_step": True, "with_a_state": len(held), "verified_with_a_state": sum(step["status"] == VERIFIED for step in held),
                              "of_which_trimmed": sum(step["how"] == TRIMMED for step in steps),
                              "blind_for_want_of_a_state": sum(step["how"] == BLIND_ATTEMPT for step in steps)})
            result[arm][str(loop + 1)] = entry
    return result


# -------------------------------------------------------------------------------------------------- budget
def arm_spend(rows: Sequence[Mapping], arm: str, through: int | None = None) -> dict:
    """What `arm` spent after the first attempts (`read.arm_budget`: attempts, generated and prompt tokens, Lean
    checks with one more for each state request and each kept-lines check), and its KNOWN COPIES: attempts whose
    proof Lean had already rejected in the episode, which were not sent (their tokens are in the count)."""
    steps = [step for row in rows for step in row["arms"][arm]["loops"] if through is None or step["loop"] <= through]
    return {**arm_budget(rows, arm, through), "known_copies": sum(step["status"] == KNOWN_COPY for step in steps)}


def budget(rows: Sequence[Mapping], primary_rows: Sequence[Mapping], loops: int, resamples: int, seed: int) -> dict:
    """Generated tokens, prompt tokens, Lean checks and known copies by arm, and the comparison at equal tokens.
    `rows`: every episode (the cost of the run); `primary_rows`: the primary's episodes, on which the equal-token
    read is made."""
    by_arm = {arm: arm_spend(rows, arm) for arm in ARMS}
    firsts = {"attempts": len(rows), "generated_tokens": sum(row["first"]["token_count"] for row in rows),
              "prompt_tokens": sum(row["first"]["prompt_tokens"] for row in rows), "lean_checks": sum(bool(row["first"]["sent_to_lean"]) for row in rows)}
    hard = {arm: arm_spend(primary_rows, arm) for arm in ARMS}
    blind_tokens, alternate_tokens = hard[BLIND]["generated_tokens"], hard[ALTERNATE]["generated_tokens"]
    ratio = round(alternate_tokens / blind_tokens, 4) if blind_tokens else None
    checks = round(hard[ALTERNATE]["lean_checks"] / hard[BLIND]["lean_checks"], 4) if hard[BLIND]["lean_checks"] else None
    result = {"what": "after the first attempts, which the two arms share: an arm's attempts, the tokens it generated, the tokens of its prompts, its Lean "
                      "checks (each attempt sent, one more for each repair step's state request, and one more where the kept lines were checked on "
                      "their own) and its known copies (attempts whose proof Lean had already rejected in the episode: not sent, tokens counted). Each "
                      "arm is counted in full: where the two arms' attempts are one generation and one check (the same prompt and seed), both count it",
              "first_attempts": firsts, "by_arm": by_arm, "by_arm_on_the_primarys_problems": hard,
              "generated_tokens_alternate_over_blind_on_the_primarys_problems": ratio,
              "lean_checks_alternate_over_blind_on_the_primarys_problems": checks,
              "alternate_generates_more_than_10_percent_more": bool(ratio is not None and ratio > 1 + MORE_TOKENS_ALLOWED)}
    if not result["alternate_generates_more_than_10_percent_more"]:
        result["equal_tokens"] = "the alternate arm generated no more than 10% more tokens than the blind arm: the primary is read as it stands"
        return result
    # The spec: "the blind arm is read at the number of attempts that matches". As in L3a, the blind arm was sampled to
    # `loops` - 1 more attempts and the match needs more than that, so that read cannot be made from this run. What can
    # be read at equal tokens from what was sampled is its mirror: the alternate arm at the loops whose tokens match.
    matching = round((loops - 1) * ratio, 2)
    kept_loops = 0
    for through in range(1, loops):
        if arm_spend(primary_rows, ALTERNATE, through)["generated_tokens"] <= blind_tokens * (1 + MORE_TOKENS_ALLOWED):
            kept_loops = through
    result["equal_tokens"] = {
        "blind_attempts_after_the_first_that_match": matching, "blind_attempts_after_the_first_that_were_sampled": loops - 1,
        "the_blind_arm_at_the_matching_number_can_be_read": False,
        "why": f"the blind arm was sampled to {loops - 1} more attempts an episode and the alternate arm's tokens match {matching}",
        "mirror": {"what": f"the alternate arm read at its first {kept_loops} attempts after the first, whose generated tokens are within 10% of the blind "
                           f"arm's at {loops - 1}; the blind arm at all of its attempts",
                   "alternate_loops_read": kept_loops,
                   "alternate_generated_tokens": arm_spend(primary_rows, ALTERNATE, kept_loops)["generated_tokens"], "blind_generated_tokens": blind_tokens,
                   "primary": paired(primary_rows, ALTERNATE, BLIND, 1 + kept_loops, resamples, seed, within_against=loops)}}
    return result


# ------------------------------------------------------------------------------- can this run see a win
def can_see_a_win(rows: Sequence[Mapping]) -> dict:
    """The two checks of the spec. A state for at least half of the failed attempts a repair step starts from
    (every repair step of the alternate arm starts from one; a step counts when its prompt held a state, so a
    trimmed step and a blind one for want of a state do not). And on the hard problems the repair step at
    position 2 verifies at least half as often as the blind arm's attempt at position 2. Less means this build
    broke the prompt or the cut. A check that cannot be read (no repair step; no failed first attempt on the
    hard problems, or a blind arm that verified none there) has not passed. Both are read on the counts: a share
    in the report is rounded, and a rounded share at exactly half would read as less."""
    steps = [step for row in rows for step in row["arms"][ALTERNATE]["loops"] if is_repair_step(step["loop"])]
    held = sum(step["how"] == RESUMED for step in steps)
    states_pass = bool(steps) and held >= HALF * len(steps)
    second = by_position(_of(rows, SETS[HARD]), 2)              # the attempts at position 2 alone
    blind, repair = second[BLIND]["2"], second[ALTERNATE]["2"]
    step_passes = blind["verified"] > 0 and repair["verified"] * blind["attempts"] >= HALF * blind["verified"] * repair["attempts"]
    return {
        "a_state_for_the_failed_attempts_a_repair_step_starts_from": {
            "repair_steps": len(steps), "with_a_state": held, "share": _share(held, len(steps)), "trimmed": sum(step["how"] == TRIMMED for step in steps),
            "without_a_state_by_reason": dict(Counter(step["no_state"] for step in steps if step["how"] == BLIND_ATTEMPT).most_common()),
            "needed": "at least half", "passes": states_pass},
        "the_repair_step_at_position_2_on_the_hard_problems": {
            "failed_first_attempts": blind["attempts"], "blind_verified": blind["verified"], "blind_share_verified": blind["share"],
            "repair_step_verified": repair["verified"], "repair_step_share_verified": repair["share"],
            # Beside the spec's check, which counts the step's trimmed loops and its blind attempts for want of a state too.
            "of_which_trimmed": repair["of_which_trimmed"], "with_a_state": repair["with_a_state"], "verified_with_a_state": repair["verified_with_a_state"],
            "needed": "at least half of the blind arm's share", "passes": step_passes},
        "passes": states_pass and step_passes}


# -------------------------------------------------------------------------------------------------- branch
def alternate_branch(primary: Mapping, checks: Mapping) -> Branch:
    """The branch the numbers select, in the spec's words. The two "can this run see a win" checks come first:
    when either fails the run is INCONCLUSIVE, not a verdict."""
    if not checks["passes"]:
        states, step = checks["a_state_for_the_failed_attempts_a_repair_step_starts_from"], checks["the_repair_step_at_position_2_on_the_hard_problems"]
        failed = []
        if not states["passes"]:
            failed.append(f"a state came back for {states['share']} of the failed attempts a repair step starts from (at least half is needed)")
        if not step["passes"]:
            failed.append(f"on the hard problems the repair step at position 2 verifies {step['repair_step_share_verified']} of the time against the "
                          f"blind arm's {step['blind_share_verified']} at position 2 (at least half of it is needed)")
        return Branch(INCONCLUSIVE, "this run could not have seen a win: " + "; ".join(failed) + ": this build broke the prompt or the cut. Fix it and run "
                                    "again; this is not a verdict")
    if primary.get("mean") is None:
        return Branch(INCONCLUSIVE, "the primary could not be computed: no episode on G or on the below-band rung")
    interval = f"{primary['mean']} [{primary['low']}, {primary['high']}]"
    if primary["low"] > 0:
        return Branch(THE_EPISODE, f"the primary is {interval}, above zero with an interval clear of zero: this is the episode a round should use; how it "
                                   "enters the round (and what training on repaired proofs adds, L3b) needs a decision, with the numbers")
    if primary["high"] < 0:
        return Branch(DID_NOT_REPEAT, f"the primary is {interval}, below zero with an interval clear of zero: L3a's single step did not repeat on fresh "
                                      "attempts; reported as that")
    return Branch(NOT_SHOWN, f"the primary is {interval}, an interval that contains zero: not shown at a size that resolves about ±0.008; a repair step "
                             "is then not worth its extra Lean check as a search step for the untrained model, and what is left of repair is training "
                             "on it, which needs a decision")


# ------------------------------------------------------------------------------------------------ the read
def alternate_read(rows: Sequence[Mapping], loops: int, resamples: int, seed: int) -> dict:
    """Everything the spec's read names, from the episode rows (`read.episode_rows` with this check's arms).
    Every comparison is alternate minus blind, paired by episode (the same first attempt), each problem's mean
    over its episodes, a 95% bootstrap over problems."""
    hard, goal = _of(rows, SETS[HARD]), _of(rows, SETS[GOAL])
    primary = paired(hard, ALTERNATE, BLIND, loops, resamples, seed)
    checks = can_see_a_win(rows)
    branch = alternate_branch(primary, checks)
    return {
        "primary": primary,
        # The same read within 2, 3, ... attempts, by set: the hard problems within fewer attempts, G alone, each rung alone.
        "within_attempts": {set_name: {str(within): paired(_of(rows, groups), ALTERNATE, BLIND, within, resamples, seed) for within in range(2, loops + 1)}
                            for set_name, groups in SETS.items()},
        "resolved_within_attempts": {set_name: resolved_by_attempt(_of(rows, groups), loops, ARMS) for set_name, groups in SETS.items()},
        "goal_set_by_problem": gained_against_lost(goal, ALTERNATE, BLIND, loops),
        "by_position": {set_name: by_position(_of(rows, groups), loops) for set_name, groups in SETS.items()},
        "trimmed": trimmed_resolutions(rows, loops, REPAIR_ARMS)[ALTERNATE],
        # The primary with every trimmed step as the blind attempt it would have been (and what followed it).
        "without_trimming": {"primary": paired(hard, ALTERNATE, BLIND, loops, resamples, seed, trimming=False)},
        "budget": budget(rows, hard, loops, resamples, seed),
        "can_this_run_see_a_win": checks,
        "branch": {"name": branch.name, "reason": branch.reason},
    }
