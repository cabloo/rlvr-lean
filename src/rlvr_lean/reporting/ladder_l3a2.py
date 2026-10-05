"""L3a2's report: one repair step after each fresh failure, then start over (no training). Spec:
docs/spec/ladder-loop.spec.md, "L3a2: one repair step after each fresh failure, then start over (no training)",
"The read, fixed before the run". Pure: rows in, a report out. It answers the read exactly and names the branch the
numbers select; the two "can this run see a win" checks are read first, and a run that fails either is INCONCLUSIVE,
not a verdict.

Naming (`naming` in the report): an EPISODE is one blind first attempt and, when it failed, up to `loops` - 1 more
attempts in each of two arms that both go on from that same failed attempt. Every comparison is alternate minus blind.

  primary        on the hard problems (G and the below-band rung): episodes resolved within the attempts of an episode;
                 paired by episode, each problem's mean over its episodes, a 95% bootstrap over problems
  secondary      the same within 2, 3 and 4 attempts; G alone and the below-band rung alone; the in-band and the
                 above-band rung; G by problem (resolved in any of its episodes), gained against lost with a two-sided
                 sign test; the verified share by attempt position in each arm (positions 2 and 4 are the repair
                 steps); the trimmed resolutions (a repair step at whose cut no goal was left and whose kept lines
                 verified on their own: nothing generated), and the primary WITHOUT them (`without_trimming`), each such
                 step being the blind attempt the stage also ran, and what followed it
  budget         generated tokens, prompt tokens, Lean checks and known copies by arm, and the read at equal tokens
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.problem_pool.episodes import ABOVE_BAND, BELOW_BAND, IN_BAND, RUNGS
from rlvr_lean.domain.repair import BLIND, FIRST
from rlvr_lean.domain.repair.alternate import ALTERNATE, ARMS, alternate_read
from rlvr_lean.domain.repair.read import EVERY, GOAL, HARD, INCONCLUSIVE
from rlvr_lean.reporting.ladder_l3a import MAXIMUM_SHARE_WITHOUT_AN_ANSWER, _health, _interval, _split


def naming(loops: int) -> str:
    return (f"an episode is one blind first attempt and, when it failed, up to {loops - 1} more attempts in each of two arms, both going on from that "
            "same failed attempt: blind (whole proofs from the plain prompt) and alternate (attempts 2 and 4 are one repair step each, from the fresh "
            "attempt just before it: that proof cut at its first error, the kept lines and Lean's proof state there as a comment, the model "
            "continues; attempts 3 and 5 are whole proofs from the plain prompt). A repair step for which no state can be had is a blind attempt, "
            "counted with its reason. A repair step at whose cut no goal was left, and whose kept lines verify on their own, is 'trimmed': it "
            "resolves the episode with nothing generated; every read is the spec's (with those steps) unless it is under without_trimming. In both "
            "arms a proof whose text equals one Lean already rejected in the same episode is a 'known copy': a failed attempt that is not sent to "
            "Lean, its tokens counted. Positions count attempts: position 1 is the first attempt")


def _positions(shares: Mapping) -> str:
    return ", ".join(f"{position}: {entry['verified']}/{entry['attempts']}" for position, entry in shares.items())


def build_l3a2_report(prepare: Mapping, episodes: Sequence[Mapping], step: Mapping, settings: Mapping, evaluation: Mapping) -> dict:
    """`prepare`: the prepare step's summary. `episodes`: the per-episode rows (`domain.repair.read.episode_rows`
    with this check's arms). `step`: the attempts step's summary (its sizes, its loops, where the time went)."""
    loops = prepare["sizes"]["loops"]
    read = alternate_read(episodes, loops, evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"])
    health = {FIRST: _health([row["first"]["status"] for row in episodes]),
              **{arm: _health([entry["status"] for row in episodes for entry in row["arms"][arm]["loops"]]) for arm in ARMS}}
    not_to_be_read = sorted(name for name, entry in health.items()
                            if entry["share_without_an_answer"] is not None and entry["share_without_an_answer"] > MAXIMUM_SHARE_WITHOUT_AN_ANSWER)
    primary, branch, checks = read["primary"], read["branch"], read["can_this_run_see_a_win"]
    within, positions, goal, trimmed, spent = read["within_attempts"], read["by_position"][HARD], read["goal_set_by_problem"], read["trimmed"], read["budget"]
    state_check, step_check = checks["a_state_for_the_failed_attempts_a_repair_step_starts_from"], checks["the_repair_step_at_position_2_on_the_hard_problems"]
    on_the_primary, last = spent["by_arm_on_the_primarys_problems"], str(loops)
    headline = (
        f"L3a2 seed {prepare['seed']}: {branch['name']}. "
        f"Primary (G and the below-band rung, {primary['problems']} problems, {primary['episodes']} episodes resolved within {loops} attempts, alternate "
        f"minus blind): {_interval(primary)}"
        + (f" ({primary[ALTERNATE]} against {primary[BLIND]})" if primary.get("mean") is not None else "")
        + f"; trimmed resolutions (kept lines that verified on their own, nothing generated): {trimmed['resolutions']} ({trimmed['by_set'][HARD]} on these "
        f"problems); the primary without them: {_interval(read['without_trimming']['primary'])}; "
        + "; ".join(f"within {count} attempts {_interval(within[HARD][count])}" for count in within[HARD] if count != last)
        + f"; G alone {_interval(within[GOAL][last])}; the below-band rung alone {_interval(within[BELOW_BAND][last])}; the in-band rung "
        f"{_interval(within[IN_BAND][last])}; the above-band rung {_interval(within[ABOVE_BAND][last])}; "
        f"G by problem ({goal['problems']}), alternate against blind: {goal[f'resolved_{ALTERNATE}']} to {goal[f'resolved_{BLIND}']}, {_split(goal)}; "
        f"verified by attempt position on these problems (2 and 4 are the repair steps), alternate: {_positions(positions[ALTERNATE])}; blind: "
        f"{_positions(positions[BLIND])}; a state for {state_check['share']} of the failed attempts a repair step starts from; "
        f"known copies not sent to Lean: alternate {on_the_primary[ALTERNATE]['known_copies']}, blind {on_the_primary[BLIND]['known_copies']}; "
        f"generated tokens, alternate over blind: {spent['generated_tokens_alternate_over_blind_on_the_primarys_problems']}; Lean checks, alternate over "
        f"blind: {spent['lean_checks_alternate_over_blind_on_the_primarys_problems']}; "
        f"position-2 check: the repair step {step_check['repair_step_share_verified']} against blind {step_check['blind_share_verified']} "
        f"({step_check['of_which_trimmed']} of its {step_check['repair_step_verified']} successes trimmed)"
        + (f"; NOT TO BE READ: too many attempts without an answer in {', '.join(not_to_be_read)}" if not_to_be_read else ""))
    return {
        "spec": "docs/spec/ladder-loop.spec.md, L3a2: one repair step after each fresh failure, then start over (no training)", "headline": headline,
        "ok": not not_to_be_read, "seed": prepare["seed"], "naming": naming(loops), "fixture": bool(prepare.get("fixture")),
        "stand_in_engine": bool(prepare.get("stand_in_engine") or step.get("stand_in_engine")), "model": "the base model: nothing is trained",
        "branch": branch, "inconclusive": branch["name"] == INCONCLUSIVE,
        "can_this_run_see_a_win": {"what": "the two checks the spec reads before the branch; when either fails the run is INCONCLUSIVE, not a verdict", **checks},
        "primary": {"what": f"on G and the below-band rung, the share of episodes resolved within {loops} attempts, alternate minus blind: paired by "
                            "episode (the same first attempt), each problem's mean over its episodes, a 95% bootstrap over problems", **primary},
        "within_attempts": {"what": "the same read within 2, 3, ... attempts, by set: goal (G), below, in, above (the rungs), hard (G and the below-band "
                                    f"rung: the primary's problems; its entry at {loops} is the primary), all", **within},
        "resolved_within_attempts": {"what": "the share of episodes resolved within 1, 2, ... attempts, by set and arm; attempt 1 is the shared first attempt",
                                     **read["resolved_within_attempts"]},
        "goal_set_by_problem": {"what": "problems of G resolved in any of their episodes: gained = by the alternate arm and not the blind one, lost = the "
                                        "reverse; two-sided sign test", **goal},
        "by_position": {"what": "by set and arm, the attempts made at each position (the episodes the arm had not resolved before it), how many verified "
                                "and how many were known copies; position 1 is the shared first attempt. Positions 2 and 4 of the alternate arm are "
                                "the repair steps: how many were prompted with a state (and verified), were trimmed, or were blind attempts for want "
                                "of a state. At positions 3 and 5 both arms write a whole proof from the plain prompt with the same sampling seed: "
                                "in an episode both arms still have open it is the SAME sample (and so is a repair step that had no state), so the "
                                "two arms' numbers there differ only by which episodes were still open and by their known copies", **read["by_position"]},
        "trimmed": {"what": "spec L3a item 2a, reported apart: episodes the alternate arm resolved by a trimmed repair step (Lean reported no goal left at "
                            "the cut and the kept lines verified on their own: one more Lean check, nothing generated), by set and by loop (the attempt's "
                            "position less one); and how many of them the arm resolved anyway in the reading without trimming", **trimmed},
        "without_trimming": {"what": "the primary read again WITHOUT the trimmed steps: each is the blind attempt it would have been (reason "
                                     "no_goals_at_the_cut), which the stage also ran, with every attempt that followed it to the arm's first verified one. "
                                     "Read from what is stored: nothing here is modelled", **read["without_trimming"]},
        "budget": spent,
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
        "not_measured": ["the model choosing its side (every attempt is on the side the certificate allows)",
                         f"the blind arm past {loops - 1} more attempts an episode (the equal-token read needs it only when the alternate arm generates more "
                         "than 10% more tokens: see budget.equal_tokens)",
                         "a repair step from a repair step's own output (L3a measured that), and a repair step from a known copy (it is a blind attempt)",
                         "anything trained: this is the base model"],
    }
