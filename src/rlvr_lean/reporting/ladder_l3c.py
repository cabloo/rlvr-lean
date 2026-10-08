"""L3c's report: an episode that keeps what verified (no training). Spec: docs/spec/ladder-loop.spec.md, "L3c: an
episode that keeps what verified (no training)", "The read, fixed before the run". Pure: rows in, a report out. It
answers the read exactly and names the branch the numbers select; the three "can this run see a win" checks are read
first, and a run that fails any is INCONCLUSIVE, not a verdict.

Naming (`naming` in the report): an EPISODE is one first generation and, when it failed, up to `generations` - 1 more
in each of two arms that both go on from that same failed generation. Every comparison is accumulate minus blind, and
the arms are compared by RUNNING TOTALS only: episodes resolved within k generations (the correction of
2026-10-06). No number here compares the arms by the rate among the episodes each has left.

  primary        on the hard problems (G and the below-band rung): episodes resolved within the generations of an
                 episode; paired by episode, each problem's mean over its episodes, a 95% bootstrap over problems
  reach          the problems of G resolved in any of their episodes, gained against lost with a two-sided sign test;
                 and the same on the problems whose shortest published proof is 4 lines or more
  secondary      the running total at every number of generations from 2 (counts by arm, the paired difference and
                 its interval); the primary by the length of the shortest published proof; how the accumulate arm's
                 resolutions came (a fresh proof, a continuing generation, an assembled proof); the line counts of the
                 proofs each arm verified; the in-band and the above-band rung
  budget         generations, generated tokens, prompt tokens, Lean checks (attempts, pool checks, closer checks) and
                 known copies by arm, and the read at equal tokens

THE LENGTH OF A PUBLISHED PROOF is read HERE and nowhere else (`lengths`: the rows of
`data/ladder_l0/heldout_proof_lines.jsonl`, handed in by the report step): no prompt and no rule of the episode sees it.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from rlvr_lean.domain.problem_pool.episodes import ABOVE_BAND, BELOW_BAND, IN_BAND, RUNGS
from rlvr_lean.domain.repair import BLIND, FIRST
from rlvr_lean.domain.repair.accumulate import ACCUMULATE, ARMS, ASSEMBLED, CONTINUE, FRESH, accumulate_read
from rlvr_lean.domain.repair.read import EVERY, GOAL, HARD, INCONCLUSIVE, SETS, _of, gained_against_lost, paired
from rlvr_lean.reporting.ladder_l3a import MAXIMUM_SHARE_WITHOUT_AN_ANSWER, _health, _interval, _split

LENGTH_GROUPS = ("1", "2-3", "4-7", "8+")       # of a problem's shortest published proof, in lines (`data/heldout_proof_lines.py`)
FOUR_OR_MORE = ("4-7", "8+")                    # "the problems whose shortest published proof is 4 lines or more"
UNKNOWN = "not_known"                           # a problem the lengths do not hold (the fixture's)


def naming(generations: int) -> str:
    return (f"an episode is one first generation and, when it failed, up to {generations - 1} more in each of two arms, both going on from that same failed "
            "generation: blind (whole proofs from the plain prompt) and accumulate. The accumulate arm holds a POOL of verified lemmas: after each failed "
            "proof its leading top-level `have` steps that end before the first error join the pool (names made unique to the generation, a statement "
            "already pooled not added twice, at most the pool's size), and the pool is checked with `all_goals sorry` (a harvest that does not stand is "
            "taken back out). A generation of that arm is 'fresh' (a whole proof from the plain prompt: the blind arm's own sample at that generation "
            "while both arms are open) or it 'continues' (the pool's lemmas and Lean's state after them in the prompt; the proof checked is the pool "
            "and the continuation): it continues when the pool is not empty and has changed since the last continuing generation. A failed proof that "
            "was lemmas and then one closing step leaves that step as a kept closer; whenever the pool has grown each kept closer is checked after it, "
            "and one that verifies resolves the episode with nothing generated: 'assembled', counted within the generation after whose harvest it was "
            "found. In both arms a proof whose text Lean already rejected in the episode is a 'known copy': not sent, its tokens counted. Generations "
            "are counted from 1, the shared first one; 'within k' is a running total")


def _group(lengths: Mapping[str, Mapping], row: Mapping) -> str:
    return (lengths.get(row["problem_id"]) or {}).get("length_group") or UNKNOWN


def by_proof_length(episodes: Sequence[Mapping], lengths: Mapping[str, Mapping], generations: int, resamples: int, seed: int) -> dict:
    """The reads that need the published proofs' lengths: the primary by length group on the hard problems, and reach
    on G (the problems resolved in any of their episodes, gained against lost) by length group and on the problems
    whose shortest published proof is 4 lines or more."""
    hard, goal = _of(episodes, SETS[HARD]), _of(episodes, SETS[GOAL])
    groups = (*LENGTH_GROUPS, UNKNOWN) if any(_group(lengths, row) == UNKNOWN for row in hard) else LENGTH_GROUPS
    return {
        "hard_problems": dict(Counter(_group(lengths, row) for row in {row["problem_id"]: row for row in hard}.values())),
        "primary": {group: paired([row for row in hard if _group(lengths, row) == group], ACCUMULATE, BLIND, generations, resamples, seed) for group in groups},
        "goal_set_by_problem": {group: gained_against_lost([row for row in goal if _group(lengths, row) == group], ACCUMULATE, BLIND, generations) for group in groups},
        "goal_set_by_problem_at_4_lines_or_more": gained_against_lost([row for row in goal if _group(lengths, row) in FOUR_OR_MORE], ACCUMULATE, BLIND, generations)}


def _lines(group: str) -> str:
    return {"1": "1 line", UNKNOWN: "length not known"}.get(group, f"{group} lines")


def _totals(totals: Mapping) -> str:
    return ", ".join(f"{within}: {entry[ACCUMULATE]}/{entry[BLIND]}" for within, entry in totals.items() if within != "1")


def build_l3c_report(prepare: Mapping, episodes: Sequence[Mapping], step: Mapping, settings: Mapping, evaluation: Mapping,
                     lengths: Mapping[str, Mapping]) -> dict:
    """`prepare`: the prepare step's summary. `episodes`: the per-episode rows (`domain.repair.accumulate.episode_rows`).
    `step`: the attempts step's summary (its sizes, its loops, where the time went). `lengths`: by problem, the line
    count of its shortest published proof and its length group (this module's docstring)."""
    generations = prepare["sizes"]["generations"]
    resamples, seed = evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"]
    read = accumulate_read(episodes, generations, resamples, seed)
    by_length = by_proof_length(episodes, lengths, generations, resamples, seed)
    health = {FIRST: _health([row["first"]["status"] for row in episodes]),
              **{arm: _health([entry["status"] for row in episodes for entry in row["arms"][arm]["loops"]]) for arm in ARMS}}
    not_to_be_read = sorted(name for name, entry in health.items()
                            if entry["share_without_an_answer"] is not None and entry["share_without_an_answer"] > MAXIMUM_SHARE_WITHOUT_AN_ANSWER)
    primary, branch, checks, reach, spent = read["primary"], read["branch"], read["can_this_run_see_a_win"], read["reach"], read["budget"]
    within, totals, came, lines = read["within_generations"], read["running_totals"][HARD], read["resolutions"][HARD], read["verified_proof_lines"][HARD]
    longer = by_length["goal_set_by_problem_at_4_lines_or_more"]
    pools, states, stands = checks["a_pool_in_the_hard_episodes"], checks["a_state_for_the_continuing_generations"], checks["the_pool_stands"]
    on_the_primary, last, pilot = spent["by_arm_on_the_primarys_problems"], str(generations), prepare["sizes"].get("pilot_problems") is not None
    headline = (
        f"L3c {'PILOT ' if pilot else ''}seed {prepare['seed']}: {branch['name']}. "
        + ("THE PILOT'S EPISODES ARE NOT READ AS A RESULT. " if pilot else "")
        + f"Primary (G and the below-band rung, {primary['problems']} problems, {primary['episodes']} episodes resolved within {generations} generations, "
        f"accumulate minus blind): {_interval(primary)}"
        + (f" ({primary[ACCUMULATE]} against {primary[BLIND]})" if primary.get("mean") is not None else "")
        + f"; reach, G by problem ({reach['problems']}), accumulate against blind: {reach[f'resolved_{ACCUMULATE}']} to {reach[f'resolved_{BLIND}']}, {_split(reach)}; "
        f"where the shortest published proof is 4 lines or more ({longer['problems']}): {longer[f'resolved_{ACCUMULATE}']} to {longer[f'resolved_{BLIND}']}, "
        f"{_split(longer)}; hard episodes resolved within k generations, accumulate/blind: {_totals(totals)}; "
        + "; ".join(f"within {count} generations {_interval(within[HARD][count])}" for count in within[HARD] if count != last)
        + "; by the shortest published proof: " + ", ".join(f"{_lines(group)} {_interval(entry)}" for group, entry in by_length["primary"].items())
        + f"; the accumulate arm's resolutions on these problems: fresh {came[FRESH]} ({came['of_which_at_the_first_generation']} at the first generation), "
        f"continuing {came[CONTINUE]}, assembled {came[ASSEMBLED]}; lines of the verified proofs (median, share with 8 or more): accumulate "
        f"{lines[ACCUMULATE]['median']}, {lines[ACCUMULATE]['share_with_8_lines_or_more']}; blind {lines[BLIND]['median']}, {lines[BLIND]['share_with_8_lines_or_more']}; "
        f"the in-band rung {_interval(within[IN_BAND][last])}; the above-band rung {_interval(within[ABOVE_BAND][last])}; "
        f"a pool in {pools['share']} of the hard episodes by their last generation; a state for {states['share']} of the continuing generations; the pool "
        f"stands in {stands['share']} of its checks; known copies not sent to Lean: accumulate {on_the_primary[ACCUMULATE]['known_copies']}, blind "
        f"{on_the_primary[BLIND]['known_copies']}; generated tokens, accumulate over blind: {spent['generated_tokens_accumulate_over_blind_on_the_primarys_problems']}; "
        f"Lean checks, accumulate over blind: {spent['lean_checks_accumulate_over_blind_on_the_primarys_problems']}"
        + (f"; NOT TO BE READ: too many attempts without an answer in {', '.join(not_to_be_read)}" if not_to_be_read else ""))
    return {
        "spec": "docs/spec/ladder-loop.spec.md, L3c: an episode that keeps what verified (no training)", "headline": headline,
        "ok": not not_to_be_read, "seed": prepare["seed"], "pilot": pilot, "naming": naming(generations), "fixture": bool(prepare.get("fixture")),
        "stand_in_engine": bool(prepare.get("stand_in_engine") or step.get("stand_in_engine")), "model": "the base model: nothing is trained",
        "branch": branch, "inconclusive": branch["name"] == INCONCLUSIVE,
        "can_this_run_see_a_win": {"what": "the three checks the spec reads before the branch; when any fails the run is INCONCLUSIVE, not a verdict", **checks},
        "primary": {"what": f"on G and the below-band rung, the share of episodes resolved within {generations} generations, accumulate minus blind: paired "
                            "by episode (the same first generation), each problem's mean over its episodes, a 95% bootstrap over problems", **primary},
        "reach": {"what": "the second number that decides: problems of G resolved in any of their episodes: gained = by the accumulate arm and not the blind "
                          "one, lost = the reverse; two-sided sign test. `at_4_lines_or_more`: the same on the problems of G whose shortest published "
                          "proof is 4 lines or more. The branch's 'reach ahead' is read on all of G", **reach, "at_4_lines_or_more": longer},
        "running_totals": {"what": "episodes resolved WITHIN k generations (k = 1 is the shared first generation), as counts by set and arm, with the "
                                   "episodes only one arm resolved within k and the two-sided sign test on those two counts. This and "
                                   "within_generations are how the arms are compared: never by a rate among the episodes each arm has left",
                           **read["running_totals"]},
        "within_generations": {"what": "the running total as the primary is read (paired by episode, each problem's mean, 95% bootstrap over problems), "
                                       "within 2, 3, ... generations, by set: goal (G), below, in, above (the rungs), hard (G and the below-band rung: the "
                                       f"primary's problems; its entry at {generations} is the primary), all", **within},
        "resolved_within_generations": {"what": "the share of episodes resolved within 1, 2, ... generations, by set and arm (each problem's mean over its "
                                                "episodes, then the mean over problems)", **read["resolved_within_generations"]},
        "by_proof_length": {"what": "by the line count of a problem's shortest published proof (1, 2-3, 4-7, 8+ lines; the report alone reads it): the hard "
                                    "problems in each group, the primary on each group's episodes, and G by problem (resolved in any episode, gained "
                                    "against lost) on each group and on the problems of 4 lines or more", **by_length},
        "resolutions": {"what": "by set, how the accumulate arm's resolved episodes came: a fresh proof (and how many of those at the first generation, "
                                "which the arms share), a continuing generation, an assembled proof; and the blind arm's resolved episodes beside it",
                        **read["resolutions"]},
        "verified_proof_lines": {"what": "by set and arm, the line counts of the proofs that resolved an episode (lines that are neither empty nor a "
                                         "comment, as the published proofs are counted): how many, the median, the mean, the share with 8 lines or "
                                         "more, the longest; the same for the proofs found after the first generation; and the accumulate arm's by how "
                                         "it resolved (after the first generation)", **read["verified_proof_lines"]},
        "pools": {"what": "by set, what the accumulate arm's pools were: the episodes that held one at their end and how large, the pool checks by "
                          "outcome ('state': the pool stands), the harvests taken back out, the closers kept, the closer checks by status ('own_proof' "
                          "and 'known_copy' were not sent), the episodes an assembled proof resolved", **read["pools"]},
        "second_generation": {"what": "the first continuing generation, like for like: on the episodes whose first generation failed and whose second "
                                      "generation in the accumulate arm had the pool in its prompt, how many that generation verified against the blind "
                                      "arm's second generation on the SAME episodes (all open in both arms), with the episodes only one of them "
                                      "verified and the sign test", **read["second_generation"]},
        "by_generation": {"what": "NOT a comparison of the arms. By set, arm and generation: the generations made (the episodes THAT arm still had open "
                                  "there), how many verified and how many were known copies; for the accumulate arm their kinds, how many had the pool "
                                  "in their prompt, how many were one sample with the blind arm, and the episodes an assembled proof resolved after "
                                  "that generation's harvest. After generation 1 the two arms hold different open episodes, so a share of one arm's "
                                  "row is a share of the episodes it had left, and the two arms' rows are not to be set against each other",
                          **read["by_generation"]},
        "budget": spent,
        "heldout": {"goal_set": prepare["goal_set"], "rungs": {name: prepare["rungs"][name] for name in RUNGS},
                    "the_primarys_problems": prepare["goal_set"] + prepare["rungs"][BELOW_BAND],
                    "problems_with_no_side_to_attempt": prepare.get("problems_with_no_side_to_attempt", [])},
        "sizes": prepare["sizes"], "attempts": health, "not_to_be_read": not_to_be_read,
        "sets": {GOAL: "G", BELOW_BAND: "the below-band rung", IN_BAND: "the in-band rung", ABOVE_BAND: "the above-band rung", HARD: "G and the below-band rung",
                 EVERY: "G and the three rungs"},
        "step": {key: step[key] for key in ("problems", "episodes", "failed_first_generations", "attempts", "attempts_on_the_contradicted_side", "generations",
                                            "generated_tokens", "generation_seconds", "tokens_per_second", "lean_checks_sent", "pool_checks_sent_to_lean",
                                            "closer_checks_sent_to_lean", "episodes_resolved", "episodes_resolved_in_the_accumulate_arm_by",
                                            "episodes_with_a_pool_at_the_end", "statuses_by_arm", "loops", "lean_in_flight") if key in step},
        "not_measured": ["the model choosing its side (every attempt is on the side the certificate allows)",
                         f"the blind arm past {generations - 1} more generations an episode (the equal-token read needs it only when the accumulate arm "
                         "generates more than 10% more tokens: see budget.equal_tokens)",
                         "a pool larger than its size, a pool across the episodes of a problem, and any closing step the model did not write (no fixed "
                         "list of tactics is tried)",
                         "anything trained: this is the base model"],
    }
