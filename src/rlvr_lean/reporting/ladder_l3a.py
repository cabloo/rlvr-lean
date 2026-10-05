"""L3a's report: the repair check, no training. Spec: docs/spec/ladder-loop.spec.md, "L3a: the repair check, no
training", "The read, fixed before the run". Pure: rows in, a report out. It answers the read exactly and names the
branch the numbers select; the two "can this run see a win" checks are read first, and a run that fails either is
INCONCLUSIVE, not a verdict.

Naming (`naming` in the report): an EPISODE is one blind first attempt and, when it failed, up to `loops` - 1 more
attempts in each of three arms that all go on from that same failed attempt.

  primary        on the hard problems (G and the below-band rung): episodes resolved within the attempts of an episode,
                 resume with the state minus blind; paired by episode, each problem's mean over its episodes, a 95%
                 bootstrap over problems
  comparisons    the same for resume without the state minus blind and for with the state minus without it, by set
                 (G, each rung, the hard problems, all)
  single step    given a failed first attempt, the share whose next attempt verifies, by arm and by set
  the goal set   problems of G resolved in any episode, gained against lost with a two-sided sign test
  trimmed        (spec item 2a) the episodes a resuming arm resolved by a trimmed loop (no goal was left at its cut and
                 the kept lines verified on their own: nothing was generated), by arm and set; and the primary, the
                 single step and the goal set read again WITHOUT them (`without_trimming`), each such loop being the
                 blind fall-back the stage also ran, and what followed it
  budget         generated tokens, prompt tokens and Lean checks by arm, and the read at equal tokens
  diagnostics    the share of repair loops with a state (and why the rest had none), the share whose first step
                 repeats the step that had just failed, how far the cut moves from loop to loop
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.problem_pool.episodes import ABOVE_BAND, BELOW_BAND, CAPPED_TOKENS, IN_BAND, NO_ANSWER, RUNGS
from rlvr_lean.domain.repair import ARMS, BLIND, FIRST, RESUME_WITH_STATE, RESUME_WITHOUT_STATE
from rlvr_lean.domain.repair.read import EVERY, GOAL, HARD, INCONCLUSIVE, repair_read

MAXIMUM_SHARE_WITHOUT_AN_ANSWER = 0.02      # above this share of attempts with no verdict an arm's counts are not to be read


def naming(loops: int) -> str:
    return (f"an episode is one blind first attempt and, when it failed, up to {loops - 1} more attempts in each of three arms, all going on from that "
            "same failed attempt: blind (whole proofs from the plain prompt), resume_with_state (the arm's latest failed proof cut at its first error, "
            "the kept lines and Lean's proof state there as a comment, the model continues), resume_without_state (the same cut and kept lines, no "
            "state comment). A resume loop for which no state can be had is a blind attempt of that arm, counted as 'no state'. A resume loop at "
            "whose cut no goal was left, and whose kept lines verify on their own, is 'trimmed': it resolves the episode with nothing generated, "
            "in both resuming arms; every read is the spec's (with those loops) unless it is under without_trimming")


def _health(statuses: Sequence[str]) -> dict:
    without = sum(status == NO_ANSWER for status in statuses)
    return {"attempts": len(statuses), "capped_at_the_token_limit": sum(status == CAPPED_TOKENS for status in statuses),
            "timed_out_in_lean": sum(status == "timeout" for status in statuses), "rejected_before_lean": sum(status == "rejected_lexical" for status in statuses),
            "without_an_answer": without, "share_without_an_answer": round(without / len(statuses), 5) if statuses else None}


def _interval(change: Mapping | None) -> str:
    return "not measured" if not change or change.get("mean") is None else f"{change['mean']} [{change['low']}, {change['high']}]"


def _split(counted: Mapping | None) -> str:
    return "not measured" if not counted else f"gained {counted['gained']}, lost {counted['lost']} (p = {counted['sign_test_p']})"


def build_l3a_report(prepare: Mapping, episodes: Sequence[Mapping], step: Mapping, settings: Mapping, evaluation: Mapping) -> dict:
    """`prepare`: the prepare step's summary. `episodes`: the per-episode rows (`domain.repair.read.episode_rows`).
    `step`: the attempts step's summary (its sizes, its loops, where the time went)."""
    loops = prepare["sizes"]["loops"]
    read = repair_read(episodes, loops, evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"])
    health = {FIRST: _health([row["first"]["status"] for row in episodes]),
              **{arm: _health([entry["status"] for row in episodes for entry in row["arms"][arm]["loops"]]) for arm in ARMS}}
    not_to_be_read = sorted(name for name, entry in health.items()
                            if entry["share_without_an_answer"] is not None and entry["share_without_an_answer"] > MAXIMUM_SHARE_WITHOUT_AN_ANSWER)
    primary, branch, checks = read["primary"], read["branch"], read["can_this_run_see_a_win"]
    comparisons, steps, diagnosed = read["comparisons"], read["single_step"], read["diagnostics"]
    hard_step, goal = steps[HARD], read["goal_set_by_problem"]["resume_with_state_minus_blind"]
    state_check, step_check = checks["a_state_for_the_failed_first_attempts"], checks["the_resumed_next_attempt_on_the_above_band_rung"]
    repeats = diagnosed[RESUME_WITH_STATE]["first_step_repeats_the_failed_step"]
    spent = read["budget"]
    sharper = step_check["only_where_the_prompt_held_a_state_at_least_half_of_blind"]
    trimmed, untrimmed = read["trimmed"], read["without_trimming"]
    headline = (
        f"L3a seed {prepare['seed']}: {branch['name']}. "
        f"Primary (G and the below-band rung, {primary['problems']} problems, {primary['episodes']} episodes resolved within {loops} attempts, resume with "
        f"the state minus blind): {_interval(primary)}"
        + (f" ({primary[RESUME_WITH_STATE]} against {primary[BLIND]})" if primary.get("mean") is not None else "")
        + f"; trimmed resolutions (kept lines that verified on their own, nothing generated): {trimmed[RESUME_WITH_STATE]['resolutions']} with the state "
        f"({trimmed[RESUME_WITH_STATE]['by_set'][HARD]} on these problems), {trimmed[RESUME_WITHOUT_STATE]['resolutions']} without it; the primary without "
        f"them: {_interval(untrimmed['primary'])}"
        + f"; without the state minus blind {_interval(comparisons['resume_without_state_minus_blind'][HARD])}; with the state minus without it "
        f"{_interval(comparisons['resume_with_state_minus_resume_without_state'][HARD])}; "
        f"the next attempt after a failed first one ({hard_step['failed_first_attempts']} on these problems) verifies: blind {hard_step[BLIND]['share']}, "
        f"with the state {hard_step[RESUME_WITH_STATE]['share']}, without it {hard_step[RESUME_WITHOUT_STATE]['share']}; "
        f"G by problem ({goal['problems']}), with the state against blind: {goal[f'resolved_{RESUME_WITH_STATE}']} to {goal[f'resolved_{BLIND}']}, {_split(goal)}; "
        f"a state for {state_check['share']} of the failed first attempts; the first step repeats the failed one in {repeats['share']} of the resumed loops; "
        f"generated tokens, resume with the state over blind: {spent['generated_tokens_resume_with_state_over_blind_on_the_primarys_problems']}; "
        f"above-band check: resumed {step_check['resume_with_state_share_verified']} against blind {step_check['blind_share_verified']} "
        f"({step_check['of_which_trimmed']} of its {step_check['resume_with_state_verified']} successes trimmed)"
        + ("; BESIDE THE SPEC'S CHECK: where the prompt held a state the resumed next attempt verifies less than half as often as blind on the "
           "above-band rung" if sharper is False and step_check["passes"] else "")
        + (f"; NOT TO BE READ: too many attempts without an answer in {', '.join(not_to_be_read)}" if not_to_be_read else ""))
    return {
        "spec": "docs/spec/ladder-loop.spec.md, L3a: the repair check, no training", "headline": headline, "ok": not not_to_be_read,
        "seed": prepare["seed"], "naming": naming(loops), "fixture": bool(prepare.get("fixture")),
        "stand_in_engine": bool(prepare.get("stand_in_engine") or step.get("stand_in_engine")), "model": "the base model: nothing is trained",
        "branch": branch, "inconclusive": branch["name"] == INCONCLUSIVE,
        "can_this_run_see_a_win": {"what": "the two checks the spec reads before the branch; when either fails the run is INCONCLUSIVE, not a verdict", **checks},
        "primary": {"what": f"on G and the below-band rung, the share of episodes resolved within {loops} attempts, resume with the state minus blind: paired "
                            "by episode (the same first attempt), each problem's mean over its episodes, a 95% bootstrap over problems", **primary},
        "comparisons": {"what": "the same read for each pair of arms, by set: goal (G), below, in, above (the rungs), hard (G and the below-band rung: the "
                                "primary's problems), all", **comparisons},
        "resolved_within_attempts": {"what": "the share of episodes resolved within 1, 2, ... attempts, by set and arm; attempt 1 is the shared first attempt",
                                     **read["resolved_within_attempts"]},
        "single_step": {"what": "given a failed first attempt, the share whose next attempt verifies, by set and arm (and how many of those were trimmed "
                                "loops); and, for the next attempts whose prompt held a state, the three arms on those same episodes", **steps},
        "goal_set_by_problem": {"what": "problems of G resolved in any episode: gained = by the first-named arm and not the second, lost = the reverse; "
                                        "two-sided sign test", **read["goal_set_by_problem"]},
        "trimmed": {"what": "spec item 2a, reported apart: episodes a resuming arm resolved by a trimmed loop (Lean reported no goal left at the cut and the "
                            "kept lines verified on their own: one more Lean check, nothing generated), by set and by loop; and how many of them the arm "
                            "resolved anyway in the reading without trimming", **trimmed},
        "without_trimming": {"what": "the primary, the single step and the goal set by problem read again WITHOUT the trimmed loops: each is the blind "
                                     "fall-back it would have been (reason no_goals_at_the_cut), which the stage also ran, with every loop that followed "
                                     "it to the arm's first verified attempt. Read from what is stored: nothing here is modelled",
                             "primary": untrimmed["primary"], "single_step": untrimmed["single_step"], "goal_set_by_problem": untrimmed["goal_set_by_problem"]},
        "budget": spent,
        "diagnostics": {"what": "what the model did with it, for each resuming arm: the share of repair loops with a state and why the others had none (those "
                                "are blind attempts), the share whose first step repeats the step that had just failed, and how far along the proof the cut "
                                "moves from loop to loop", **diagnosed},
        "heldout": {"goal_set": prepare["goal_set"], "rungs": {name: prepare["rungs"][name] for name in RUNGS},
                    "the_primarys_problems": prepare["goal_set"] + prepare["rungs"][BELOW_BAND],
                    "problems_with_no_side_to_attempt": prepare.get("problems_with_no_side_to_attempt", [])},
        "sizes": prepare["sizes"], "attempts": health, "not_to_be_read": not_to_be_read,
        "sets": {GOAL: "G", BELOW_BAND: "the below-band rung", IN_BAND: "the in-band rung", ABOVE_BAND: "the above-band rung", HARD: "G and the below-band rung",
                 EVERY: "G and the three rungs"},
        "step": {key: step[key] for key in ("problems", "episodes", "failed_first_attempts", "attempts", "attempts_on_the_contradicted_side", "generations",
                                            "generated_tokens", "generation_seconds", "tokens_per_second", "lean_checks_sent", "state_files_sent_to_lean",
                                            "kept_lines_checks_sent_to_lean", "episodes_resolved", "episodes_resolved_by_a_trimmed_loop",
                                            "attempts_of_the_reading_without_trimming_only", "statuses_by_arm", "loops", "lean_in_flight") if key in step},
        "not_measured": ["the model choosing its side (the spec: not part of this check; every attempt is on the side the certificate allows)",
                         f"the blind arm past {loops - 1} more attempts an episode (the equal-token read needs it only when the resuming arm generates more "
                         "than 10% more tokens: see budget.equal_tokens)",
                         "anything trained: this is the base model"],
    }
