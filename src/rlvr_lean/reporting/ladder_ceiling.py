"""The ceiling's report: a labelled diagnostic. Spec: docs/spec/ladder-loop.spec.md, "The ceiling: a labelled
diagnostic", "The read, fixed before the run". Pure: rows in, a report out. Every heading and every line it prints
says "ceiling": no number here is a result of the loop, and no model trained this way is kept.

One-shot attempts only. Every comparison is a model minus the BASE over the same problems and the same number of
attempts on each, paired by problem, with a 95% bootstrap interval over problems; problems solved are gained against
lost. No number compares models by a rate among the problems one of them has left unsolved.

  primary      the goal problems whose shortest published proof is 4 lines or more: successes per attempt, the model
               after the last checkpoint minus the base. The branch is read against the loop's own gain there, ONE
               number fixed before the run (`ceiling.loops_own_gain`: the three seeds pooled). Beside the primary
               stands this seed's three-round model, from its stored rows: information that decides nothing
  secondary    the same by length group and for the smaller dose; goal problems solved, gained against lost, by length
               group; the three rungs; the line counts of the proofs each model verifies; the share of distinct attempts
  the checks   the three "can this run see a win" checks, each with its number and PASS or FAIL. Any FAIL makes the
               report INCONCLUSIVE, and it then says nothing else about the model: what was measured is kept under
               `measured_and_not_read`, for whoever repairs the run
  the branch   in the spec's words

THE LENGTH OF A HELD-OUT PROBLEM'S PUBLISHED PROOF is read here and nowhere before (`lengths`: the rows of
`data/ladder_l0/heldout_proof_lines.jsonl`, handed in by the report step): no prompt and no step of the run sees it.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.ceiling import (
    CEILING,
    FULL,
    INCONCLUSIVE,
    LABEL,
    SMALL,
    can_this_run_see_a_win,
    ceiling_branch,
    interval_text,
    not_broken,
    still_writes_proofs,
)
from rlvr_lean.domain.ladder_round.read import GOAL, group_ids, paired_change
from rlvr_lean.domain.ladder_round.rounds import gained_lost, summed_budget
from rlvr_lean.domain.problem_pool.episodes import ABOVE_BAND, RUNGS
from rlvr_lean.domain.repair.accumulate import _lines as line_stats
from rlvr_lean.reporting.ladder_l2 import MAXIMUM_SHARE_WITHOUT_AN_ANSWER, _health, _split
from rlvr_lean.reporting.ladder_l3c import FOUR_OR_MORE, LENGTH_GROUPS, UNKNOWN
from rlvr_lean.reporting.ladder_l3c import _lines as lines_of

LOOP = "loop"                       # the three-round model that stands beside the ceiling's (`gpu.ladder_ceiling.LOOP`)
RUNG_PART = "rungs"
FOUR_PLUS, ALL = "4_or_more", "all"
ORDER = (FULL, SMALL, LOOP)         # the model the primary is read on, the smaller dose, the loop's own model


def lean_did_not_answer(rows: Sequence[Mapping]) -> bool:
    """A set is NOT TO BE READ when Lean gave no verdict on more than `MAXIMUM_SHARE_WITHOUT_AN_ANSWER` of its
    attempts: its counts say what the pool did, not what the model wrote. (Not the second check, which is about
    the model.) The report says so, and the step keeps the adapters and measures such a set of its own again."""
    share = _health(rows)["share_without_an_answer"]
    return share is not None and share > MAXIMUM_SHARE_WITHOUT_AN_ANSWER


def attempts_on_g(samplings: Sequence[Sequence[Mapping]]) -> list[dict]:
    """One model's results over its samplings of G, added problem by problem (`summed_budget`: its 32 and its 61
    are its 93). Every sampling must hold the same problems."""
    rows = [{"problem_id": row["problem_id"], "side": row.get("side"), "resolved": row["resolved"], "episodes": row["episodes"]} for row in samplings[0]]
    for more in samplings[1:]:
        rows = summed_budget(rows, more)
    return rows


def goal_sets(goal_ids: Sequence[str], lengths: Mapping[str, Mapping]) -> dict[str, list[str]]:
    """G by the length group of each problem's shortest published proof, the problems of 4 lines or more (the
    primary's), and all of it. A problem the lengths do not hold (the fixture's) is in a group of its own."""
    group_of = {problem_id: (lengths.get(problem_id) or {}).get("length_group") or UNKNOWN for problem_id in goal_ids}
    groups = (*LENGTH_GROUPS, UNKNOWN) if UNKNOWN in group_of.values() else LENGTH_GROUPS
    sets = {group: [problem_id for problem_id in goal_ids if group_of[problem_id] == group] for group in groups}
    return {**sets, FOUR_PLUS: [problem_id for problem_id in goal_ids if group_of[problem_id] in FOUR_OR_MORE], ALL: list(goal_ids)}


def per_attempt(model: Sequence[Mapping], base: Sequence[Mapping], problem_ids: Sequence[str], resamples: int, seed: int) -> dict:
    """Successes per attempt over `problem_ids`, the model minus the base: paired by problem with its 95% bootstrap
    interval over problems (`paired_change`, which refuses two sides with different numbers of attempts on a
    problem), and each side's totals over the same problems and attempts."""
    change, wanted = paired_change(model, base, problem_ids, resamples, seed), set(problem_ids)

    def totals(rows: Sequence[Mapping]) -> tuple[int, int]:
        own = [row for row in rows if row["problem_id"] in wanted]
        return sum(row["resolved"] for row in own), sum(row["episodes"] for row in own)

    (successes, attempts), (of_the_base, _) = totals(model), totals(base)
    return {**change, "successes": successes, "successes_of_the_base": of_the_base, "attempts_each": attempts,
            "per_1000": round(1000 * successes / attempts, 2) if attempts else None, "per_1000_of_the_base": round(1000 * of_the_base / attempts, 2) if attempts else None}


def solved(model: Sequence[Mapping], base: Sequence[Mapping], problem_ids: Sequence[str]) -> dict:
    """The problems of `problem_ids` each side solved at least once in its attempts, gained against lost."""
    wanted = set(problem_ids)
    return gained_lost([row for row in model if row["problem_id"] in wanted], [row for row in base if row["problem_id"] in wanted])


def verified_lines(counts: Mapping[str, int]) -> dict:
    """What `what_a_model_wrote` counted (proofs by their number of lines), as the read gives it: how many, the
    median, the mean, the shares with 4 lines or more and with 8 or more, the longest."""
    each = [int(lines) for lines, count in counts.items() for _ in range(count)]
    return {**line_stats(each), "share_with_4_lines_or_more": round(sum(lines >= 4 for lines in each) / len(each), 5) if each else None}


def _named(group: str) -> str:
    return {FOUR_PLUS: "4 lines or more", ALL: "all of G"}.get(group) or lines_of(group)


def _rate(entry: Mapping) -> str:
    if entry.get("mean") is None:
        return "not measured"
    return f"{interval_text(entry)}, {entry['per_1000']} against {entry['per_1000_of_the_base']} per 1,000"


def model_names(prepare: Mapping, models: Mapping[str, Mapping]) -> dict[str, str]:
    rate = prepare.get("loop_target_rate")
    at = f" at t = {Fraction(str(rate))}" if rate is not None else ""
    return {name: f"the three-round model{at}" if name == LOOP else f"the {entry['rows']:,}-proof model" for name, entry in models.items()}


def build_ceiling_report(prepare: Mapping, train: Mapping, losses: Mapping, groups: Sequence[Mapping], lengths: Mapping[str, Mapping], base: Mapping,
                         models: Mapping[str, Mapping], settings: Mapping, evaluation: Mapping) -> dict:
    """`prepare` and `train`: those steps' summaries. `losses`: the stored loss file (`row_losses`). `groups`: the
    held-out groups. `lengths`: by problem, its shortest published proof's line count and length group. `base` and
    each of `models` (`small`, `full`, and `loop` when a stored three-round model was read): `rungs` (per-problem
    rows on the three rungs), `goal` (one list of per-problem rows for each sampling of G), `verified_proof_lines`
    and `distinct_attempts` (by `rungs` and `goal`); a model of the ceiling's also has `rows`, what it had been
    trained on. `settings`: `ladder_loop.ceiling`."""
    resamples, seed = evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"]
    goal_ids, rung_ids = group_ids(groups, GOAL), {name: group_ids(groups, name) for name in RUNGS}
    sets = goal_sets(goal_ids, lengths)
    order = [name for name in ORDER if name in models]
    names = model_names(prepare, models)
    base_on_g, on_g = attempts_on_g(base[GOAL]), {name: attempts_on_g(models[name][GOAL]) for name in order}
    by_length = {name: {group: per_attempt(on_g[name], base_on_g, ids, resamples, seed) for group, ids in sets.items()} for name in order}
    solved_on_g = {name: {group: solved(on_g[name], base_on_g, ids) for group, ids in sets.items()} for name in order}
    rungs = {name: {rung: paired_change(models[name][RUNG_PART], base[RUNG_PART], rung_ids[rung], resamples, seed) for rung in RUNGS} for name in order}
    who = {"base": base, **{name: models[name] for name in order}}
    lines_verified = {name: {part: verified_lines(entry["verified_proof_lines"][part]) for part in (GOAL, RUNG_PART)} for name, entry in who.items()}
    distinct = {name: entry["distinct_attempts"] for name, entry in who.items()}

    # ---- the checks, read on the model the primary is read on; then the branch
    checks = can_this_run_see_a_win(losses["row_losses"], models[FULL][RUNG_PART], base[RUNG_PART], rung_ids[ABOVE_BAND], settings)
    primary, beside = by_length[FULL][FOUR_PLUS], by_length[LOOP][FOUR_PLUS] if LOOP in models else None
    own_gain, times = settings["loops_own_gain"], settings["times_the_loops_own_gain"]
    branch = ceiling_branch(checks, primary, own_gain, times)        # against the FIXED number: this seed's three-round model decides nothing
    inconclusive = branch["name"] == INCONCLUSIVE
    for_information = {"the_model_still_writes_proofs": still_writes_proofs(models[SMALL][RUNG_PART], settings["maximum_share_without_an_answer"]),
                       "training_has_not_broken_it": not_broken(models[SMALL][RUNG_PART], base[RUNG_PART], rung_ids[ABOVE_BAND],
                                                               settings["minimum_share_of_the_bases_pass_rate_above_the_band"])}

    # ---- whether Lean answered (the health of each set: not the second check, which is about the model)
    measured_sets = {"rungs_base": base[RUNG_PART], **{f"goal_base_{index}": rows for index, rows in enumerate(base[GOAL], start=1)}}
    for name in order:
        measured_sets[f"rungs_{name}"] = models[name][RUNG_PART]
        measured_sets.update({f"goal_{name}_{index}": rows for index, rows in enumerate(models[name][GOAL], start=1)})
    health = {name: _health(rows) for name, rows in measured_sets.items()}
    not_to_be_read = sorted(name for name, rows in measured_sets.items() if lean_did_not_answer(rows))

    attempts, full_rows = prepare["attempts_a_goal_problem"], models[FULL]["rows"]
    measured = {
        "primary": {"what": f"the goal problems whose shortest published proof is 4 lines or more: successes per attempt over {attempts} attempts a problem, "
                            f"{names[FULL]} minus the base, paired by problem, a 95% bootstrap interval over problems", "model": names[FULL], **primary},
        "beside_the_primary": None if beside is None else {"what": f"for information, deciding nothing: the same for THIS SEED's {names[LOOP][4:]} minus the base, "
                                                                   "from its stored rows (the branch is read against the_loops_own_gain)",
                                                           "model": names[LOOP], **beside},
        "secondary": {
            "by_length_group": {"what": "the primary's quantity (successes per attempt, the model minus the base, paired by problem) by the length group of each "
                                        "goal problem's shortest published proof (1, 2-3, 4-7, 8+ lines), on the problems of 4 lines or more and on all of G; "
                                        "for every model", "problems": {group: len(ids) for group, ids in sets.items()},
                                "models": {name: {"model": names[name], **by_length[name]} for name in order}},
            "goal_problems_solved": {"what": f"goal problems solved at least once in their {attempts} attempts: gained = by the model and not by the base, lost = "
                                             "the reverse; two-sided sign test; by length group", "models": {name: {"model": names[name], **solved_on_g[name]} for name in order}},
            "the_three_rungs": {"what": f"fresh pass rate on each held-out rung ({prepare['rung_episodes']} episodes a problem), the model minus the base, paired by problem",
                                "problems": {rung: len(rung_ids[rung]) for rung in RUNGS}, "models": {name: {"model": names[name], **rungs[name]} for name in order}},
            "verified_proof_lines": {"what": "the line counts of the proofs each model verifies (lines that are neither empty nor a comment, as the published "
                                             "proofs are counted), on G over all its attempts and on the three rungs: how many, the median, the mean, the "
                                             "shares with 4 lines or more and with 8 or more, the longest", **lines_verified},
            "distinct_attempts": {"what": "the share of a model's attempts at one problem and side that are distinct, averaged over problems and sides, on the "
                                          "three rungs and on G", **distinct}}}

    # ---- the lines, in the spec's order
    heading = (f"{CEILING}: THE CEILING, A LABELLED DIAGNOSTIC, seed {prepare['seed']}. One training from the base on {full_rows:,} published proofs of problems "
               f"the base cannot solve, in the file's order (a checkpoint after {models[SMALL]['rows']:,} rows and at the end), measured in one-shot attempts. "
               "An exception to the rule that published proofs are certificates and not training text: no model trained this way is kept or used in a round")
    lines = [heading]
    if prepare.get("stored_runs") is None:
        lines.append(f"{CEILING}: L2's stored attempts were not read (a smoke run): G was attempted {attempts} times a problem and no three-round model stands beside")
    if not inconclusive:
        lines.append(f"{CEILING}: PRIMARY. The goal problems whose shortest published proof is 4 lines or more ({primary['problems']} problems), successes per "
                     f"attempt over {attempts} attempts a problem, {names[FULL]} minus the base, paired by problem, 95% bootstrap over problems: {_rate(primary)}"
                     + (f" ({primary['successes']} successes against {primary['successes_of_the_base']} in {primary['attempts_each']:,} attempts each)"
                        if primary.get("mean") is not None else ""))
        lines.append(f"{CEILING}: the loop's own gain there, the ONE number the branch is read against: {own_gain:+.5f} per attempt (the three-round model at "
                     f"t = 1/10 minus the base, the three seeds pooled, fixed before the run); {times} times it is {times * own_gain:+.5f}")
        lines.append(f"{CEILING}: beside it, for information and deciding nothing, this seed's {names[LOOP][4:]} minus the base, from the stored rows: "
                     f"{_rate(beside)} ({beside['successes']} successes against {beside['successes_of_the_base']} in {beside['attempts_each']:,} attempts each)"
                     if beside is not None and beside.get("mean") is not None else f"{CEILING}: beside it: no stored three-round model was read")
        for name in order:
            lines.append(f"{CEILING}: SECONDARY, the same by the length of the shortest published proof, {names[name]} minus the base: "
                         + "; ".join(f"{_named(group)} ({len(ids)}): {_rate(by_length[name][group])}" for group, ids in sets.items()))
        for name in order:
            lines.append(f"{CEILING}: SECONDARY, goal problems solved at {attempts} attempts, {names[name]} against the base: "
                         + "; ".join(f"{_named(group)}: {solved_on_g[name][group]['resolved_after']} to {solved_on_g[name][group]['resolved_before']}, "
                                     f"{_split(solved_on_g[name][group])}" for group in (ALL, *(group for group in sets if group != ALL))))
        for name in order:
            lines.append(f"{CEILING}: SECONDARY, the three rungs ({prepare['rung_episodes']} episodes a problem), {names[name]} minus the base: "
                         + "; ".join(f"{rung} {interval_text(rungs[name][rung])}" for rung in RUNGS))
        for part, where in ((GOAL, "on G"), (RUNG_PART, "on the rungs")):
            lines.append(f"{CEILING}: SECONDARY, the lines of the proofs each model verifies {where} (proofs, median, mean, share with 4 lines or more, with 8 or "
                         "more, longest): " + "; ".join(
                             f"{'the base' if name == 'base' else names[name]} {entry[part]['proofs']}, {entry[part]['median']}, {entry[part]['mean']}, "
                             f"{entry[part]['share_with_4_lines_or_more']}, {entry[part]['share_with_8_lines_or_more']}, {entry[part]['longest']}"
                             for name, entry in lines_verified.items()))
        lines.append(f"{CEILING}: SECONDARY, the share of distinct attempts (on the rungs, on G): " + "; ".join(
            f"{'the base' if name == 'base' else names[name]} {entry[RUNG_PART]['mean_share_distinct']}, {entry[GOAL]['mean_share_distinct']}" for name, entry in distinct.items()))
    took, writes, broken = checks["the_training_took"], checks["the_model_still_writes_proofs"], checks["training_has_not_broken_it"]
    verdict = lambda check: "PASS" if check["passes"] else "FAIL"      # noqa: E731
    lines += [
        f"{CEILING}: CHECK 1, the training took: the mean training loss over the last {took['rows_compared']} rows is {took['last']}, over the first "
        f"{took['rows_compared']} it was {took['first']}: {verdict(took)}",
        f"{CEILING}: CHECK 2, the model still writes proofs: {writes['share']} of {names[FULL]}'s {writes['attempts']:,} attempts on the three rungs got no answer "
        f"({writes['capped_at_the_token_limit']} reached the token cap, {writes['without_a_verdict_from_lean']} had no verdict from Lean); under "
        f"{writes['maximum']} is asked: {verdict(writes)}",
        f"{CEILING}: CHECK 3, training on other provers' proofs has not broken it: {names[FULL]} passes {broken['pass_rate']} on the above-band rung against the "
        f"base's {broken['pass_rate_of_the_base']}; at least {broken['minimum']} is asked: {verdict(broken)}",
        f"{CEILING}: {'INCONCLUSIVE' if inconclusive else 'BRANCH: ' + branch['name']}. {branch['reason']}"]
    if not_to_be_read:
        lines.append(f"{CEILING}: NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(not_to_be_read)}")
    headline = (f"The ceiling (a labelled diagnostic; no model trained this way is kept) seed {prepare['seed']}: {branch['name']}. "
                + ("" if inconclusive else f"Primary (goal problems with a published proof of 4 lines or more, {primary['problems']} problems, {attempts} attempts each, "
                                           f"{names[FULL]} minus the base): {_rate(primary)}; the loop's own gain there, fixed before the run: {own_gain:+.5f}; "
                                           f"this seed's three-round model beside it, deciding nothing: {_rate(beside) if beside is not None else 'not read'}. ")
                + f"Checks: the training took {verdict(took)} ({took['last']} against {took['first']}), still writes proofs {verdict(writes)} ({writes['share']} "
                  f"without an answer), not broken {verdict(broken)} ({broken['pass_rate']} against the base's {broken['pass_rate_of_the_base']} above the band). "
                + branch["reason"]
                + (f" NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(not_to_be_read)}" if not_to_be_read else ""))
    return {
        "spec": "docs/spec/ladder-loop.spec.md, The ceiling: a labelled diagnostic", "diagnostic": CEILING, "label": LABEL, "headline": headline, "lines": lines,
        "ok": not not_to_be_read, "seed": prepare["seed"],
        "stand_in_engine": bool(prepare.get("stand_in_engine") or train.get("stand_in_engine") or any(models[name].get("stand_in_engine") for name in order)),
        "branch": branch, "inconclusive": inconclusive,
        "the_loops_own_gain": {"what": "the ONE number the branches are read against (ladder_loop.ceiling.loops_own_gain): on the goal problems whose shortest "
                                       "published proof is 4 lines or more, successes per attempt, the three-round model at t = 1/10 minus the base, the "
                                       "three seeds pooled (279 attempts a problem), paired by problem; computed from the stored rows before the run. No "
                                       "seed's own three-round model decides a branch",
                               "per_attempt": own_gain, "times": times, "times_it": round(times * own_gain, 5)},
        "can_this_run_see_a_win": {"what": f"the three checks the spec reads before the branch, on {names[FULL]}; when any fails the run is INCONCLUSIVE and "
                                           "nothing else is said about the model", **checks,
                                   "for_information_the_smaller_dose": {"model": names[SMALL], **for_information}},
        **({"primary": None, "beside_the_primary": None, "secondary": None,
            "measured_and_not_read": {"why": "a check failed: the run is INCONCLUSIVE, and these numbers say nothing about the model. They are kept for "
                                             "whoever repairs the run", **measured}} if inconclusive else measured),
        "models": {"base": "the base model, from stored attempts: nothing here measured it again", **names},
        "training": {"what": prepare["recipe"]["what"], **{key: value for key, value in train.items() if key != "checkpoints"}, "checkpoints": prepare["checkpoints"],
                     "doses": prepare["doses"], "recipe": prepare["recipe"], "training_file": prepare["training_file"],
                     "training_file_sha256": prepare["training_file_sha256"], "training_file_rows": prepare["training_file_rows"],
                     "rows_trained_on": prepare["rows_trained_on"], "order": prepare["order"], "longest_example_tokens": prepare["longest_example_tokens"],
                     "max_sequence_tokens": prepare["max_sequence_tokens"]},
        "heldout": {"goal_set": len(goal_ids), "goal_set_by_length_group": {group: len(ids) for group, ids in sets.items()},
                    "rungs": {rung: len(rung_ids[rung]) for rung in RUNGS}},
        "sizes": {key: prepare[key] for key in ("sampling_seeds", "rung_episodes", "goal_samplings", "attempts_a_goal_problem", "stored_runs", "loop_arm",
                                                "stored_measurements", "contradicted_side_setting")},
        "attempts": health, "not_to_be_read": not_to_be_read,
        "not_measured": ["the base and the three-round model (their rows are the stored ones of the runs named under sizes.stored_runs)",
                         "anything but one-shot attempts: no episode with more than one generation, no repair, no pool",
                         "the standard evaluation of Phase A (its statements are not built at v4.27)"],
    }
