"""L4b's report: the arm again, from the larger pretrained model. Spec: docs/spec/ladder-loop.spec.md, "L4b: the arm
again, from the larger pretrained model", "The read, fixed before the run". Pure: rows in, a report out.

It IS L4's report (`reporting/ladder_l4.build_l4_report`: the five checks first, the primary on G' over the second
sampling, the branch, the secondary reads), read with the start model in `pre`'s place, and with what L4b adds:

  beside the primary   on every branch, the condition "without overfitting": the goal problems solved at least once in all
                       the attempts, `with` against the start model (gained, lost, the two-sided sign test), and the
                       share of distinct attempts on G; NARROWER is written when it lost more than it gained with the
                       sign test under 0.05, whatever the primary says (`build_l4_report(breadth=True)`)
  the fifth check      the rehearsal rows of the rule `rehearse` ARE of the `pretrain` half and are named as such; a
                       round's row of that half, and any held-out problem, are barred as ever
  the rule             the rounds' table with the rule's kept share: of each round's rows, how many the last model was
                       trained on, in all and by k
  rank 16 beside       the rank-16 arm's stored figures beside each, when its report was on the box (never a reason
                       to refuse); and the goal problems nothing stored had solved that each model solves, when given

Everything here is labelled pretrained on published proofs.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.assembly import H0
from rlvr_lean.domain.ladder_round.ceiling import interval_text
from rlvr_lean.domain.ladder_round.l3d import WITH, WITHOUT
from rlvr_lean.domain.ladder_round.l4 import PRE
from rlvr_lean.domain.ladder_round.l4_rows import PRETRAINING, with_k
from rlvr_lean.domain.ladder_round.l4b import REHEARSE, by_origin, kept_share
from rlvr_lean.domain.ladder_round.read import GOAL
from rlvr_lean.domain.problem_pool.episodes import RUNGS
from rlvr_lean.reporting.ladder_ceiling import ALL, FOUR_PLUS, _rate, attempts_on_g
from rlvr_lean.reporting.ladder_l4 import PRIMARY, SAY, build_l4_report

L4B = "L4b"
SPEC = "docs/spec/ladder-loop.spec.md, L4b: the arm again, from the larger pretrained model"
RANK_16_SECTIONS = ("by_length_group", "the_three_rungs", "goal_problems_solved_by_attempts_alone", "goal_problems_solved", "with_minus_without", "by_round", "distinct_attempts")


def of_the_rank_16_arm(stored: Mapping) -> dict:
    """What is kept of the rank-16 arm's stored report (the arm from `pre`, `report_ladder_l4.json`), to stand beside:
    its headline, its branch, its primary and the secondary reads L4b reads too. An INCONCLUSIVE report keeps what it
    measured and did not read, and says so."""
    read = stored.get("measured_and_not_read") or stored
    return {"seed": stored.get("seed"), "arm": stored.get("arm"), "headline": stored.get("headline"), "branch": (stored.get("branch") or {}).get("name"),
            "inconclusive": bool(stored.get("inconclusive")), "ok": stored.get("ok", True), "primary": read.get("primary"),
            "secondary": {name: (read.get("secondary") or {}).get(name) for name in RANK_16_SECTIONS}}


def the_rule_by_round(rounds: Mapping[int, Mapping], last_set: Sequence[Mapping]) -> dict:
    """The rule's kept share, from the rounds' own rows and the last model's training set: of each round's rows (one-shot
    and assembled, each with its problem's k as the challenger read it), how many the last model was trained on; in all,
    and by k. `last_set`: the rows of the last model's set (`id`, `origin`)."""
    kept = {row["id"] for row in last_set}
    by_round, every = {}, []
    for number in sorted(rounds):
        own = with_k([row for row in rounds[number]["examples"] if row["origin"] != H0], rounds[number]["results"])
        by_round[str(number)] = kept_share(own, kept)
        every += own
    return {"what": "of the rounds' rows (one-shot and assembled), how many the last model was trained on: by round, in all and by k (k as the challenger reads it: an "
                    "assembled row is k = 1)", "by_round": by_round, **kept_share(every, kept), "set_by_origin": by_origin(last_set)}


def _beside(here: Mapping, there: Mapping, start: str) -> dict:
    """Each figure of this report with the rank-16 arm's stored one beside it. `here`, `there`: the two reports' read
    parts (`primary`, `secondary`); in the stored one the start model is `pre`."""
    def of(section: Mapping, own_start: str) -> dict:
        secondary = section.get("secondary") or {}
        pairs = (secondary.get("by_length_group") or {}).get("pairs") or {}
        rungs = (secondary.get("the_three_rungs") or {}).get("pairs") or {}
        alone = secondary.get("goal_problems_solved_by_attempts_alone") or {}
        solved = ((secondary.get("goal_problems_solved") or {}).get(PRIMARY) or {}).get(ALL) or {}
        distinct = secondary.get("distinct_attempts") or {}
        share = lambda name: ((distinct.get(name) or {}).get(GOAL) or {}).get("mean_share_distinct")      # noqa: E731
        reliable = lambda name: {key: ((alone.get(name) or {}).get(ALL) or {}).get(key) for key in ("solved_at_least_once", "reliably")}      # noqa: E731
        return {"primary": section.get("primary"),
                "all_of_g_with_minus_start": {group: (pairs.get(PRIMARY) or {}).get(group) for group in (ALL, FOUR_PLUS)},
                "all_of_g_with_minus_without": (pairs.get("with_minus_without") or {}).get(ALL),
                "the_three_rungs_with_minus_start": {rung: (rungs.get(PRIMARY) or {}).get(rung) for rung in RUNGS},
                "solved_at_least_once_and_reliably": {"start": reliable(own_start), WITH: reliable(WITH), WITHOUT: reliable(WITHOUT)},
                "goal_problems_solved_with_against_start": {key: solved.get(key) for key in ("resolved_after", "resolved_before", "gained", "lost", "sign_test_p")},
                "share_of_distinct_attempts_on_g": {"start": share(own_start), WITH: share(WITH), WITHOUT: share(WITHOUT)},
                "by_round": [{key: row.get(key) for key in ("round", "picks", "resolved_by_an_attempt", "only_assembly_resolved", "mean_pass_rate", "training_rows")}
                             for row in ((secondary.get("by_round") or {}).get("rows") or [])]}

    mine, theirs = of(here, start), of(there, PRE)
    return {name: {"here": mine[name], "at_rank_16": theirs[name]} for name in mine}


def _figure(entry: Mapping | None) -> str:
    """One paired difference as a line gives it: with each side's rate per 1,000 where the read has them (successes per
    attempt), the interval alone where it does not (a rung's pass rate)."""
    if not entry:
        return "not there"
    if entry.get("mean") is None:
        return "not measured"
    return _rate(entry) if "per_1000" in entry else interval_text(entry)


def build_l4b_report(read: Sequence, own: Mapping, trained_rows: Mapping[str, Sequence[Mapping]], last_set: Sequence[Mapping], rank_16: Mapping | None = None,
                     never: Sequence[str] | None = None) -> dict:
    """`read`: the arguments of `build_l4_report`, in its order (`gpu/ladder_l4.read_for_the_arms_report`), the start
    model's rows under its own name in the models. `own`: `ladder_l4b_prepare`'s summary (the start model, its rank,
    the rule, the five checks' first two as read on the start model). `trained_rows`: by training of the arm, the rows
    it was trained on as the training loop recorded them (`problem_id`, `origin`). `last_set`: the last model's
    training set. `rank_16`: what the prepare step kept of the rank-16 arm's stored report (`read`: whether it was on
    the box). `never`: the goal problems nothing stored had solved, when a file of them was given; None otherwise."""
    (prepare, _, arm_prepare, again, trains, row_losses, _, assembled_lines, rounds, groups, lengths, base, models, base_arm, settings, target_rate, evaluation) = read
    start, rule, recipe = own["start"], own["rule"], own["start_recipe"]
    rehearsing = rule == REHEARSE
    # The fifth check reads the rounds' rows; the rehearsal rows of the rule `rehearse` stand apart, named as what they are. Under any other rule EVERY row is a round's.
    of_the_rounds = {name: [row["problem_id"] for row in rows if not (rehearsing and row["origin"] == PRETRAINING)] for name, rows in trained_rows.items()}
    rehearsed = {name: [row["problem_id"] for row in rows if row["origin"] == PRETRAINING] for name, rows in trained_rows.items()} if rehearsing else None
    report = build_l4_report(prepare, own, arm_prepare, again, trains, row_losses, of_the_rounds, assembled_lines, rounds, groups, lengths, base, models, base_arm, settings,
                             target_rate, evaluation, start=start, breadth=True, rehearsed=rehearsed)
    inconclusive = report["inconclusive"]
    section = report["measured_and_not_read"] if inconclusive else report
    last = prepare["rounds"][-1]

    # ---- the rule's kept share, in the rounds' table
    of_the_rule = {"rule": rule, **the_rule_by_round(rounds, last_set), "of_each_training": {name: entry.get("rule") for name, entry in trains.items()}}
    for row in section["secondary"]["by_round"]["rows"]:
        row["kept_by_the_rule"] = of_the_rule["by_round"][str(row["round"])]
    section["secondary"]["the_rule"] = of_the_rule

    # ---- the rank-16 arm's stored figures, beside each
    beside = {"read": False, "why": (rank_16 or {}).get("why") or "the rank-16 arm's report was not on this box"}
    if rank_16 and rank_16.get("read"):
        beside = {"read": True, "what": f"the rank-16 arm (from `{PRE}`), as its own stored report read it, beside each figure of this arm; it decides nothing", "run": rank_16.get("run"),
                  "branch": rank_16.get("branch"), "inconclusive": rank_16.get("inconclusive"), "headline": rank_16.get("headline"), "figures": _beside(section, rank_16, start)}
    section["secondary"]["beside_the_rank_16_arm"] = beside

    # ---- the goal problems nothing stored had solved, when given
    never_solved = {"given": False, "what": "the goal problems nothing stored had solved are read from a file of ids in the run directory, and none was given"}
    if never is not None:
        wanted = set(never)
        goal_ids = {row["problem_id"] for row in models[WITH][GOAL][0]}
        solved_by = {name: sorted(row["problem_id"] for row in attempts_on_g(models[name][GOAL]) if row["problem_id"] in wanted and row["resolved"] > 0) for name in (start, WITH, WITHOUT)}
        never_solved = {"given": True, "what": "the goal problems nothing stored had solved (a file of ids given in the run directory), and those each model solves at least "
                                               "once in all its attempts; the start model's count checks the list (0 for a list made before it was measured is not expected: "
                                               "it was stored when the list was made, or after)",
                        "problems": len(never), "of_this_runs_goal_set": len(wanted & goal_ids), "solved": {name: len(found) for name, found in solved_by.items()},
                        "solved_problems": solved_by}
    section["secondary"]["never_solved"] = never_solved

    # ---- the lines: what this run is, first; the three reads L4b adds, after L4's own
    first = (f"{SAY}: {L4B}, THE ARM AGAIN, FROM THE LARGER PRETRAINED MODEL, seed {prepare['seed']}. The start model is `{start}` (rank {recipe['rank']}, alpha {recipe['alpha']}; the "
             f"config's own are {recipe.get('rank_of_the_config')} and {recipe.get('alpha_of_the_config')}): round 1 is attempted by it, the challenger starts from ITS OWN map, "
             f"and EVERY MODEL of the arm and its twin is trained from it AT ITS RANK. THE TRAINING RULE is `{rule}`. The read is L4's with `{start}` in `{PRE}`'s place; G' is "
             f"`{start}`'s ({len(again)} goal problems it does not solve in its first sampling)")
    lines = [first, *report["lines"]]
    if not inconclusive:
        share = of_the_rule
        origins = ", ".join(f"{origin} {count:,}" for origin, count in share["set_by_origin"].items())
        lines.append(f"{SAY}: SECONDARY, THE RULE `{rule}`: of the rounds' {share['rows']:,} rows M({last}) was trained on {share['kept']:,} ({share['share']}); by round: "
                     + "; ".join(f"round {number} {entry['kept']:,} of {entry['rows']:,}" for number, entry in share["by_round"].items())
                     + "; by k: " + "; ".join(f"k = {k} {entry['kept']:,} of {entry['rows']:,}" for k, entry in share["by_k"].items()) + f"; its set by origin: {origins}")
        if beside["read"]:
            figures = beside["figures"]
            solved_here, solved_there = (figures["goal_problems_solved_with_against_start"][side] for side in ("here", "at_rank_16"))
            distinct_here, distinct_there = (figures["share_of_distinct_attempts_on_g"][side] for side in ("here", "at_rank_16"))
            reliable_here, reliable_there = (figures["solved_at_least_once_and_reliably"][side] for side in ("here", "at_rank_16"))
            lines.append(f"{SAY}: SECONDARY, BESIDE EACH, THE RANK-16 ARM (from `{PRE}`, its stored report; it read {beside['branch']}): the primary {_figure(figures['primary']['here'])}, "
                         f"at rank 16 {_figure(figures['primary']['at_rank_16'])}; all of G, `with` minus its start model: {_figure(figures['all_of_g_with_minus_start']['here'][ALL])}, "
                         f"at rank 16 {_figure(figures['all_of_g_with_minus_start']['at_rank_16'][ALL])}; on the goal problems of 4 lines or more: "
                         f"{_figure(figures['all_of_g_with_minus_start']['here'][FOUR_PLUS])}, at rank 16 {_figure(figures['all_of_g_with_minus_start']['at_rank_16'][FOUR_PLUS])}")
            lines.append(f"{SAY}: SECONDARY, BESIDE EACH, THE RANK-16 ARM: goal problems solved in all the attempts, `with` against its start model: {solved_here['resolved_after']} to "
                         f"{solved_here['resolved_before']} (gained {solved_here['gained']}, lost {solved_here['lost']}, p = {solved_here['sign_test_p']}), at rank 16 "
                         f"{solved_there['resolved_after']} to {solved_there['resolved_before']} (gained {solved_there['gained']}, lost {solved_there['lost']}, p = "
                         f"{solved_there['sign_test_p']}); the share of distinct attempts on G, the start model, `with`, `without`: {distinct_here['start']}, {distinct_here[WITH]}, "
                         f"{distinct_here[WITHOUT]}, at rank 16 {distinct_there['start']}, {distinct_there[WITH]}, {distinct_there[WITHOUT]}; solved at least once / reliably, the start "
                         f"model, `with`, `without`: "
                         + ", ".join(f"{reliable_here[name]['solved_at_least_once']} / {reliable_here[name]['reliably']}" for name in ("start", WITH, WITHOUT)) + ", at rank 16 "
                         + ", ".join(f"{reliable_there[name]['solved_at_least_once']} / {reliable_there[name]['reliably']}" for name in ("start", WITH, WITHOUT))
                         + "; the three rungs, `with` minus its start model: "
                         + "; ".join(f"{rung} {_figure(figures['the_three_rungs_with_minus_start']['here'][rung])}, at rank 16 "
                                     f"{_figure(figures['the_three_rungs_with_minus_start']['at_rank_16'][rung])}" for rung in RUNGS)
                         + "; `with` minus `without` on all of G: " + f"{_figure(figures['all_of_g_with_minus_without']['here'])}, at rank 16 "
                         f"{_figure(figures['all_of_g_with_minus_without']['at_rank_16'])}")
        else:
            lines.append(f"{SAY}: SECONDARY, BESIDE EACH, THE RANK-16 ARM: NOT THERE ({beside['why']}). Nothing is refused for it")
        if never_solved["given"]:
            lines.append(f"{SAY}: SECONDARY, the goal problems nothing stored had solved ({never_solved['problems']} given, {never_solved['of_this_runs_goal_set']} of this run's goal "
                         "set): solved at least once in all the attempts by " + ", ".join(f"`{name}` {count}" for name, count in never_solved["solved"].items()))
        else:
            lines.append(f"{SAY}: SECONDARY, the goal problems nothing stored had solved: not given (no file of their ids in the run directory)")
    headline = report["headline"].replace("L4 (the loop from a model", f"{L4B} (the arm again from `{start}` at rank {recipe['rank']}, the rule `{rule}`; the loop from a model", 1)
    return {**report, "spec": SPEC, "check": L4B, "headline": headline, "lines": lines, "start": start, "rank": recipe["rank"], "alpha": recipe["alpha"], "rule": rule,
            "the_start_model": {key: own.get(key) for key in ("start", "start_adapter", "start_recipe", "pretraining_run", "start_run_read", "goal_set_again_made_here", "map",
                                                              "map_file")},
            "what_it_cannot_say": ["one seed: a positive read licenses two more seeds of this arm from the same start model, and nothing more",
                                   "the rule was named by L4t on the rank-16 rounds' rows; this arm is what confirms it, on rounds its own models made",
                                   "rank 64 is not the whole model, and one learning rate and one pass a training: a null here is not a null for a larger or longer training"]}
