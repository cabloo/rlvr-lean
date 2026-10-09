"""L4t's report: what should a round train on? Spec: docs/spec/ladder-loop.spec.md, "L4t: what should a round train
on? Three one-change checks on the rounds already made", "The read, fixed before the run". Pure: rows in, a report out.
Its pieces are the ceiling report's and L4's (the paired bootstrap over problems, gained against lost with the two-sided
sign test), read on this stage's models.

Everything here is labelled pretrained on published proofs, as L4's reports are: every line begins with L4's label.
One-shot attempts only.

  each trained model, BY ITSELF (`rehearse`, then `reward_rows`)
    the checks   FIRST, each PASS or FAIL: its training ran (the adapter it saved is not the start adapter's, the loss did
                 not rise by more than two standard errors); it still writes proofs (under 5% of its rung attempts without
                 an answer); Lean answered (at most 2% of each set read without an answer); no row it was trained on is a
                 held-out problem or, among its rounds' rows, a problem of the `pretrain` half. Any FAIL: THAT MODEL is
                 INCONCLUSIVE and nothing else is said of it (what was measured is kept under `measured_and_not_read`);
                 the other models are still read
    primary      breadth: the goal problems solved at least once in all the attempts, the model against `without`, paired
                 by problem: gained, lost, the two-sided sign test
    beside it    reliability kept: all of G, successes per attempt, the model minus `pre`, 95% bootstrap over problems;
                 `without`'s own, from its stored rows, stands beside
    the branch   in the spec's words (`l4_rows.rows_branch`)
    secondary    G' over the fresh attempts against `pre`; by proof length; solved reliably; the three rungs; the share of
                 distinct attempts; the lines of verified proofs; the goal problems nothing stored had solved, when their
                 ids were GIVEN; the rows it was trained on
  `hot`          read, not branched: for `with` and `pre` at the config's temperature (stored) and at the hot one, the
                 distinct goal problems solved, the successes per attempt and the share of distinct attempts, on G and on G'

A report in which Lean did not answer a set is NOT TO BE READ (`ok` is false, and the step fails): the task queued again
samples such a set of this run's own again from the kept adapter. THE LENGTH OF A HELD-OUT PROBLEM'S PUBLISHED PROOF is
read here and nowhere before.

FROM ANOTHER START THAN `pre` (spec, "L4r's first branch was taken: what L4t runs"): three trained models, `old_rule`
first. `old_rule` (the twin's rows, from the start adapter) is read BY ITSELF against the start model, which is the cap's
question: the goal problems solved in all the attempts (gained, lost, the sign test), the share of distinct attempts on G,
the share of its verified proofs with 8 lines or more, all of G per attempt, with the stored `without` against `pre`
beside each when those rows were on the box (`old_rule_figures`), and one of three named outcomes with the start adapter's
rank in it (`l4_rows.old_rule_branch`). The two others are read as above with `old_rule` in `without`'s place and the start
model in `pre`'s, and are NOT READ against an `old_rule` that is INCONCLUSIVE. `hot` is `old_rule` and the start model.
From `pre` the report is, key for key and line for line, what it was.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.ceiling import interval_text, still_writes_proofs
from rlvr_lean.domain.ladder_round.l3d import WITH, WITHOUT
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, PRE, the_training_ran
from rlvr_lean.domain.ladder_round.l4_rank import never_solved
from rlvr_lean.domain.ladder_round.l4_rows import (
    HOT,
    OLD_RULE,
    PRETRAINING,
    REHEARSE,
    REWARD_ROWS,
    SIGN_TEST_LEVEL,
    models_of,
    no_barred_row,
    old_rule_branch,
    reference_of,
    rows_branch,
)
from rlvr_lean.domain.ladder_round.read import GOAL, group_ids, paired_change
from rlvr_lean.domain.problem_pool.episodes import RUNGS
from rlvr_lean.reporting.ladder_ceiling import ALL, FOUR_PLUS, RUNG_PART, _named, _rate, attempts_on_g, goal_sets, lean_did_not_answer, per_attempt, solved, verified_lines
from rlvr_lean.reporting.ladder_l2 import MAXIMUM_SHARE_WITHOUT_AN_ANSWER, _health, _split
from rlvr_lean.reporting.ladder_l3d1 import by_attempts_alone
from rlvr_lean.reporting.ladder_l4 import LABEL, NOT_MEASURED, SAY, _ran, _verdict, adapter_against_the_start
from rlvr_lean.reporting.ladder_l4_rank import _named_problems

L4 = "l4"
QUESTION = "what should a round train on?"
STORED = (PRE, WITH, WITHOUT)
HOT_OF = (WITH, PRE)
IS = {REHEARSE: "the twin's one-shot rows and as many rows of the pretraining file (published proofs `pre` was trained on), one pass over both",
      REWARD_ROWS: "each row of the rounds kept with the probability of its problem's reward r(k), an assembled row at k = 1",
      OLD_RULE: "the twin's one-shot rows exactly, in the twin's order: what `without` is, trained from this run's start adapter (the old rule at its rank)"}
CANNOT_SAY = [
    "The rounds were attempted by models trained the old way, so this is the training rule alone and not the loop under it: a rule that wins here is confirmed by the "
    "next arm, not by this stage",
    "One seed",
    "`rehearse` trains again on published proofs `pre` has seen: a model that only gains from seeing them twice would show it against `pre`"]


def _lines(counts: Mapping[str, int]) -> str:
    stats = verified_lines(counts)
    return (f"{stats['proofs']:,} proofs, median {stats['median']}, mean {stats['mean']}, share with 4 lines or more {stats['share_with_4_lines_or_more']}, with 8 or more "
            f"{stats['share_with_8_lines_or_more']}, longest {stats['longest']}")


def _counted(rows: Sequence[Mapping], key: str) -> dict:
    found = [row[key] for row in rows if row.get(key) is not None]
    return {str(name): found.count(name) for name in sorted(set(found))}


def trained_on(rows: Sequence[Mapping]) -> dict:
    """What a model WAS trained on, from the rows its training loop recorded: how many, by origin, by k, and the lines
    of their proofs."""
    lines = _counted(rows, "lines")
    return {"what": "the rows the training loop recorded (never the prepared file alone): by origin, by their problem's k of n as the challenger read it (a rehearsal "
                    "row has none), and the lines of their proofs", "rows": len(rows), "rows_by_origin": _counted(rows, "origin"), "rows_by_k": _counted(rows, "k"),
            "rehearsal_rows": sum(row["origin"] == PRETRAINING for row in rows), "lines": verified_lines(lines), "rows_by_lines": lines}


def on_a_set(rows: Sequence[Mapping], problem_ids: Sequence[str]) -> dict:
    """One model's own figures over `problem_ids`: the problems it solves at least once, its successes and attempts."""
    wanted = set(problem_ids)
    own = [row for row in rows if row["problem_id"] in wanted]
    successes, attempts = sum(row["resolved"] for row in own), sum(row["episodes"] for row in own)
    return {"problems": len(own), "solved_at_least_once": sum(row["resolved"] > 0 for row in own), "successes": successes, "attempts": attempts,
            "per_1000": round(1000 * successes / attempts, 2) if attempts else None}


def _own(entry: Mapping) -> str:
    return f"{entry['solved_at_least_once']} of {entry['problems']} solved, {entry['per_1000']} per 1,000 ({entry['successes']:,} of {entry['attempts']:,})"


def old_rule_figures(on_g: Sequence[Mapping], of_the_start: Sequence[Mapping], wrote: Mapping, wrote_by_the_start: Mapping, goal_ids: Sequence[str], resamples: int,
                     seed: int) -> dict:
    """The four figures the old rule is read by against the model it was trained from (`old_rule` against the start
    model; beside it, the stored `without` against `pre`): the goal problems solved at least once in all the attempts
    (gained, lost, the two-sided sign test), the share of distinct attempts on G, the share of verified proofs with 8
    lines or more, and all of G per attempt (95% bootstrap over problems). `on_g`, `of_the_start`: each side's
    per-problem rows over all its attempts on G. `wrote`, `wrote_by_the_start`: what each wrote."""
    lines = [verified_lines(entry["verified_proof_lines"][GOAL]) for entry in (wrote, wrote_by_the_start)]
    return {"solved": solved(on_g, of_the_start, goal_ids), "per_attempt": per_attempt(on_g, of_the_start, goal_ids, resamples, seed),
            "share_of_distinct_attempts_on_g": {"of_the_old_rule": wrote["distinct_attempts"][GOAL]["mean_share_distinct"],
                                                "of_its_start": wrote_by_the_start["distinct_attempts"][GOAL]["mean_share_distinct"]},
            "share_of_verified_proofs_with_8_lines_or_more": {"of_the_old_rule": lines[0]["share_with_8_lines_or_more"], "of_its_start": lines[1]["share_with_8_lines_or_more"]},
            "verified_proofs": {"of_the_old_rule": lines[0]["proofs"], "of_its_start": lines[1]["proofs"]}}


def _figures(figures: Mapping, rule: str, start: str) -> str:
    """The four figures in one line: `rule` against `start`."""
    found, distinct, long = figures["solved"], figures["share_of_distinct_attempts_on_g"], figures["share_of_verified_proofs_with_8_lines_or_more"]
    return (f"goal problems solved at least once, `{rule}` against `{start}`: {found['resolved_after']} to {found['resolved_before']} of {found['problems']}, {_split(found)}; "
            f"the share of distinct attempts on G {distinct['of_the_old_rule']} against {distinct['of_its_start']}; the share of verified proofs with 8 lines or more "
            f"{long['of_the_old_rule']} against {long['of_its_start']}; all of G, successes per attempt, `{rule}` minus `{start}`: {_rate(figures['per_attempt'])}")


def build_rows_report(prepare: Mapping, trains: Mapping[str, Mapping], losses: Mapping[str, Sequence[float]], trained: Mapping[str, Sequence[Mapping]],
                      groups: Sequence[Mapping], lengths: Mapping[str, Mapping], stored: Mapping[str, Mapping], models: Mapping[str, Mapping],
                      hot: Mapping[str, Mapping], again: Sequence[str], settings: Mapping, evaluation: Mapping, never: Sequence[str] | None = None,
                      rank_16: Mapping[str, Mapping] | None = None) -> dict:
    """`prepare`: the prepare step's summary. `trains`, `losses`, `trained`: by trained model, its training step's
    summary (with what it recorded `against_the_start_adapter`), its rows' losses in the order trained, and the rows
    its training loop recorded (`problem_id`, `origin`, `k`, `lines`). `groups`: the held-out groups. `lengths`: by
    problem, its shortest published proof's length group. `stored` (`pre`, `with`, `without`) and `models` (the
    trained ones): each `rungs` (per-problem rows on the three rungs), `goal` (one list of per-problem rows for each
    sampling of G) and what it wrote (`verified_proof_lines`, `distinct_attempts`, `episodes_of_8`). `hot`: `with` and
    `pre` at the hot temperature: `goal` (one list a hot sampling) and what each wrote. `again`: G', the ids `pre`'s
    run stored. `settings`: `ladder_loop.l4`. `never`: the ids of the goal problems nothing stored had solved, as
    GIVEN in the run directory, or None when no file was given.

    FROM ANOTHER START THAN `pre` (`prepare["start"]`; spec, "L4r's first branch was taken: what L4t runs"): `stored`
    holds the START MODEL alone, `models` holds `old_rule` too, which stands in `without`'s place for the two other
    models (the start model in `pre`'s) and is read BY ITSELF first, against the start model (`l4_rows.old_rule_branch`);
    `hot` is `old_rule` and the start model; `again` is the start model's G'. `rank_16`: the stored `without` and `pre`
    (`goal`, and what each wrote), which stand beside `old_rule`'s own read, or None when they were not on the box."""
    resamples, seed = evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"]
    goal_ids, rung_ids = group_ids(groups, GOAL), {rung: group_ids(groups, rung) for rung in RUNGS}
    sets = goal_sets(goal_ids, lengths)
    start = prepare["start"]
    from_pre = start == PRE
    against = reference_of(start)                               # the model in `without`'s place: the stored `without`, or this run's `old_rule`
    trained_models = models_of(start)
    stored_names = STORED if from_pre else (start,)
    beside = stored_names if from_pre else (start, against)     # the models named beside one model in its secondary reads
    hot_models = HOT_OF if from_pre else (against, start)
    rank = (prepare.get("start_recipe") or {}).get("rank")      # the start adapter's, which every model of a run from another start has
    who = {**{name: stored[name] for name in stored_names}, **{name: models[name] for name in trained_models}}
    on_g = {name: attempts_on_g(entry[GOAL]) for name, entry in who.items()}
    samplings, attempts = prepare["goal_samplings"], prepare["attempts_a_goal_problem"]
    fresh = samplings[1] if len(samplings) > 1 else None
    in_again = set(again)
    stray = sorted(in_again - set(goal_ids))
    if stray:
        raise ValueError(f"{len(stray)} problems of G' are not goal problems of this run (first: {stray[0]}): the pretraining's report is of another held-out set")
    second = lambda entry: entry[GOAL][1]      # noqa: E731
    of_the_fresh = f"the {fresh['episodes']} attempts of the second sampling" if fresh else "a second sampling, which this run does not have"
    alone = by_attempts_alone(who, sets)
    heldout = {row["problem_id"] for row in groups}

    # ---- whether Lean answered, set by set (the stored models' sets were held to this at the prepare step too)
    measured_sets, sets_of = {}, {}
    for name, entry in who.items():
        sets_of[name] = [f"rungs_{name}", *(f"goal_{name}_{index}" for index in range(1, len(entry[GOAL]) + 1))]
        measured_sets[f"rungs_{name}"] = entry[RUNG_PART]
        measured_sets.update({f"goal_{name}_{index}": rows for index, rows in enumerate(entry[GOAL], start=1)})
    for name, entry in hot.items():
        measured_sets.update({f"goal_{HOT}_{name}_{index}": rows for index, rows in enumerate(entry[GOAL], start=1)})
    health = {name: _health(rows) for name, rows in measured_sets.items()}
    not_to_be_read = sorted(name for name, rows in measured_sets.items() if lean_did_not_answer(rows))

    # ---- what the stored models give: `without`'s own reliability (the figure the spec sets beside), and each one's breadth
    if from_pre:
        reference = {"what": f"from the stored rows, at {attempts} attempts a goal problem: each stored model's own figures on all of G, and `without` minus `pre` and "
                             "`with` minus `pre` (successes per attempt, paired by problem, 95% bootstrap over problems): what the old rule gave",
                     "on_all_of_g": {name: on_a_set(on_g[name], goal_ids) for name in STORED},
                     "without_minus_pre": per_attempt(on_g[WITHOUT], on_g[PRE], goal_ids, resamples, seed),
                     "with_minus_pre": per_attempt(on_g[WITH], on_g[PRE], goal_ids, resamples, seed),
                     "with_against_pre": solved(on_g[WITH], on_g[PRE], goal_ids), "without_against_pre": solved(on_g[WITHOUT], on_g[PRE], goal_ids)}
    elif rank_16:       # ... from another start: the old rule at rank 16, from the stored rows of `without` and `pre`, beside `old_rule`'s own read
        of_16 = {name: attempts_on_g(rank_16[name][GOAL]) for name in (PRE, WITHOUT)}
        reference = {"what": "the old rule at the rank the rounds were made at, from the stored rows of `without` and `pre`: the four figures `old_rule` is read by, "
                             "which stand beside its own and decide nothing", "read": True, "rank": (prepare.get("rows_made_by") or {}).get("rank"),
                     "on_all_of_g": {name: on_a_set(of_16[name], goal_ids) for name in (PRE, WITHOUT)},
                     **old_rule_figures(of_16[WITHOUT], of_16[PRE], rank_16[WITHOUT], rank_16[PRE], goal_ids, resamples, seed)}
    else:
        reference = {"what": "the old rule at the rank the rounds were made at, from the stored rows of `without` and `pre`: NOT THERE (their rows were not on the box, "
                             "or would not pair with this run's)", "read": False, "why": (prepare.get("rank_16") or {}).get("why")}

    # ---- each trained model, by itself
    read, lines_of, failed_of = {}, {}, {}
    for name in trained_models:
        model = models[name]
        versus = start if name == against else against          # `old_rule` itself is read against the model it was trained from
        by_itself = versus == start
        own_sets = [*sets_of[name], *sets_of[start], *([] if by_itself else sets_of[versus])]      # what this model's read stands on: its own sets, and the two it is read against
        unanswered = [key for key in own_sets if key in not_to_be_read]
        read_against = ("the stored ones of `pre` and `without`" if from_pre else f"the stored ones of `{start}`" if by_itself else
                        f"the stored ones of `{start}` and this run's own of `{versus}`")
        checks = {
            "its_training_ran": the_training_ran(losses[name], adapter_against_the_start(trains[name]), settings),
            "it_still_writes_proofs": still_writes_proofs(model[RUNG_PART], settings["maximum_share_without_an_answer"]),
            "lean_answered": {"what": f"Lean gave a verdict on all but at most {MAXIMUM_SHARE_WITHOUT_AN_ANSWER} of the attempts of each set read: the model's own, which "
                                      f"the task queued again samples again, and {read_against} it is read against",
                              "maximum": MAXIMUM_SHARE_WITHOUT_AN_ANSWER, "share_without_an_answer": {key: health[key]["share_without_an_answer"] for key in own_sets},
                              "sets_not_answered": unanswered, "passes": not unanswered},
            "no_barred_row": no_barred_row(trained[name], heldout, settings["half_seed"])}
        failed_of[name] = [check for check, entry in checks.items() if not entry["passes"]]
        breadth = solved(on_g[name], on_g[versus], goal_ids)
        kept = per_attempt(on_g[name], on_g[start], goal_ids, resamples, seed)
        branch = (old_rule_branch(checks, breadth, kept, rank, start) if by_itself else
                  rows_branch(checks, breadth, kept, against=versus, start=start, against_failed=None if from_pre else failed_of[versus]))
        inconclusive = branch["name"] == INCONCLUSIVE
        set_aside = inconclusive or "not_read_against" in branch       # nothing is read of it: a check of its own failed, or the model it is read against is INCONCLUSIVE
        on_again = per_attempt(second(model), second(who[start]), list(again), resamples, seed) if fresh else dict(NOT_MEASURED)
        unsolved = never_solved(never, on_g[name], on_g[start], goal_ids)
        counted_beside = (WITH, WITHOUT) if from_pre else () if by_itself else (versus,)
        if unsolved["given"]:
            unsolved = {**unsolved, "what": "the goal problems nothing stored had solved, as the file given in the run directory lists them: how many the model solves at "
                                            "least once in all its attempts on G" + ("; `with`'s and `without`'s own counts stand beside" if from_pre else
                                                                                     "" if by_itself else f"; `{versus}`'s own count stands beside"),
                        **{f"solved_by_{other}": never_solved(never, on_g[other], on_g[start], goal_ids)["solved"] for other in counted_beside}}
        seen_twice = bool(name == REHEARSE and on_again.get("low") is not None and on_again["low"] > 0)
        others = [other for other in beside if other != name]
        measured = {
            "primary": {"what": f"breadth: the goal problems solved at least once in their {attempts} attempts, `{name}` against `{versus}` ("
                                + ("the model it was trained from" if by_itself else "the same rounds' rows by the old rule")
                                + f"), paired by problem: gained = by `{name}` and not by `{versus}`, lost = the reverse; the two-sided sign test", **breadth},
            "beside_the_primary": {"what": f"reliability kept: all of G, successes per attempt over the {attempts} attempts a problem, `{name}` minus `{start}`, paired by "
                                           f"problem, a 95% bootstrap interval over problems. In its figures `of_the_base` is the side compared with: `{start}`",
                                   **kept, **({} if by_itself else {f"of_{versus}": reference["without_minus_pre"] if from_pre else
                                                                    per_attempt(on_g[versus], on_g[start], goal_ids, resamples, seed)})},
            "secondary": {
                "the_goal_set_again": {"what": f"G' (the goal problems `{start}` does not solve in its {samplings[0]['episodes']}-attempt sampling): successes per attempt over "
                                               f"{of_the_fresh}, `{name}` minus `{start}` (the arm's primary)", "goal_set_again": len(again), **on_again,
                                       **({"seen_twice": f"`rehearse` beats `{start}` on G' with an interval above zero. It trains again on published proofs `{start}` has seen: a "
                                                         "model that only gains from seeing them twice would show exactly this"} if seen_twice else {})},
                "by_length_group": {"what": "by the length group of each goal problem's shortest published proof, and on the problems of 4 lines or more: the reliability "
                                            f"(successes per attempt, `{name}` minus `{start}`) and the breadth (solved at least once, `{name}` against `{versus}`)",
                                    "problems": {group: len(ids) for group, ids in sets.items()},
                                    f"minus_{start}": {group: per_attempt(on_g[name], on_g[start], ids, resamples, seed) for group, ids in sets.items() if group != ALL},
                                    f"against_{versus}": {group: solved(on_g[name], on_g[versus], ids) for group, ids in sets.items() if group != ALL}},
                **({} if by_itself else {f"goal_problems_solved_against_{start}": {
                    "what": f"goal problems solved at least once in their {attempts} attempts, `{name}` against `{start}`: gained, lost, the two-sided sign test",
                    **solved(on_g[name], on_g[start], goal_ids)}}),
                "goal_problems_solved_by_attempts_alone": {
                    "what": "each model's one-shot attempts at a goal problem cut into episodes of 8 in the order drawn; goal problems resolved in at least one, a quarter, "
                            "half (RELIABLY) and nine tenths of their episodes", **{other: alone[other] for other in (*others, name)}},
                "the_three_rungs": {"what": f"fresh pass rate on each held-out rung ({prepare['rung_episodes']} episodes a problem), paired by problem: `{name}` minus `{start}`"
                                            + ("" if by_itself else f", and `{name}` minus `{versus}`"), "problems": {rung: len(rung_ids[rung]) for rung in RUNGS},
                                    f"minus_{start}": {rung: paired_change(model[RUNG_PART], who[start][RUNG_PART], rung_ids[rung], resamples, seed) for rung in RUNGS},
                                    **({} if by_itself else {f"minus_{versus}": {rung: paired_change(model[RUNG_PART], who[versus][RUNG_PART], rung_ids[rung], resamples, seed)
                                                                                 for rung in RUNGS}})},
                "distinct_attempts": {"what": "the share of a model's attempts at one problem and side that are distinct, on the three rungs and on G",
                                      **{other: who[other]["distinct_attempts"] for other in (*others, name)}},
                "verified_proof_lines": {"what": "the line counts of the proofs each model verifies on G over all its attempts",
                                         **{other: verified_lines(who[other]["verified_proof_lines"][GOAL]) for other in (*others, name)}},
                "never_solved_before": unsolved,
                "trained_on": trained_on(trained[name])}}
        if by_itself:       # the cap's question: the four figures the old rule is read by, and beside each the stored ones of the rank the rounds were made at
            measured = {"by_itself": {"what": f"`{name}` read BY ITSELF against `{start}`, the model it was trained from, which is the cap's question: the goal problems "
                                              f"solved at least once in the {attempts} attempts (gained, lost, the two-sided sign test), the share of distinct attempts on "
                                              "G, the share of its verified proofs with 8 lines or more, and all of G per attempt (95% bootstrap over problems)",
                                      **old_rule_figures(on_g[name], on_g[start], model, who[start], goal_ids, resamples, seed), "at_the_rank_of_the_rounds": reference},
                        **measured}
        read[name] = {"model": f"`{name}`: from `{prepare['start']}`, one pass over {trains[name]['rows']:,} rows: {IS[name]}", "branch": branch, "inconclusive": inconclusive,
                      "can_this_model_be_read": {"what": "the four checks the spec reads first; when any fails THIS MODEL is INCONCLUSIVE, by itself", **checks},
                      **({**{key: None for key in measured},
                          "measured_and_not_read": {"why": ("a check failed: this model is INCONCLUSIVE, and these numbers say nothing about its training rule. They are kept "
                                                            "for whoever repairs the run" if inconclusive else
                                                            f"`{versus}`, which this model is read against, is INCONCLUSIVE: these numbers are not read against it. They are "
                                                            "kept for whoever repairs the run"), **measured}} if set_aside else measured),
                      "training": {key: value for key, value in trains[name].items() if key != "checkpoints"}}

        # ---- its lines: the checks first, then the primary, what stands beside it, the branch and the secondary reads
        ran, writes, answered, barred = (checks[key] for key in ("its_training_ran", "it_still_writes_proofs", "lean_answered", "no_barred_row"))
        of = f"{SAY}: `{name}`"
        own = [
            f"{of} ({IS[name]}; {trains[name]['rows']:,} rows, {trains[name]['steps']:,} optimizer steps, from `{prepare['start']}`), CHECK 1, its training ran (the "
            "adapter saved is not the start adapter's; the mean loss over the last tenth of its rows is not above the first tenth's by more than "
            f"{settings['standard_errors_allowed']:g} standard errors of their difference): {_ran(f'`{name}`', ran)}: {_verdict(ran)}",
            f"{of}, CHECK 2, it still writes proofs: {writes['share']} of its {writes['attempts']:,} attempts on the three rungs got no answer; under {writes['maximum']} "
            f"is asked: {_verdict(writes)}",
            f"{of}, CHECK 3, Lean answered (at most {MAXIMUM_SHARE_WITHOUT_AN_ANSWER} of each set read without an answer): "
            + "; ".join(f"{key} {health[key]['share_without_an_answer']}" for key in own_sets) + f": {_verdict(answered)}",
            f"{of}, CHECK 4, no row it was trained on is a held-out problem or, among its rounds' rows, a problem of the `pretrain` half: {barred['held_out_problems']} "
            f"held-out problems and {barred['problems_of_the_pretrain_half_among_the_rounds_rows']} of the `pretrain` half among its {barred['rows_of_the_rounds']:,} "
            f"rounds' rows" + (f"; its {barred['rehearsal_rows_of_the_pretrain_half']:,} rehearsal rows ARE of the `pretrain` half (published proofs `pre` was trained "
                               "on), and are named as such" if barred["rehearsal_rows_of_the_pretrain_half"] else "") + f": {_verdict(barred)}"]
        if by_itself and not set_aside:
            own += [
                f"{of}, READ BY ITSELF against `{start}`, the model it was trained from (the cap's question; {attempts} attempts a goal problem): "
                + _figures(measured["by_itself"], name, start),
                f"{of}, BESIDE IT, the old rule at rank {reference.get('rank')}, where the rounds were made, from the stored rows: " + _figures(reference, WITHOUT, PRE)
                if reference["read"] else
                f"{of}, BESIDE IT, the old rule at the rank the rounds were made at (the stored `without` against `pre`): NOT THERE ({reference.get('why')})"]
        elif not set_aside:
            own += [
                f"{of}, PRIMARY, breadth. The goal problems solved at least once in the {attempts} attempts, `{name}` against `{versus}`, paired by problem: "
                f"{breadth['resolved_after']} to {breadth['resolved_before']} of {breadth['problems']}, {_split(breadth)}",
                f"{of}, BESIDE IT, reliability kept. All of G, successes per attempt over the {attempts} attempts, `{name}` minus `{start}`, 95% bootstrap over problems: "
                f"{_rate(kept)}; `{versus}`'s is {interval_text(measured['beside_the_primary'][f'of_{versus}'])}"]
        own.append(f"{of}: {'INCONCLUSIVE' if inconclusive else ('OUTCOME: ' if by_itself else 'BRANCH: ') + branch['name']}. {branch['reason']}")
        if not set_aside:
            secondary = measured["secondary"]
            by_length, rungs, record = secondary["by_length_group"], secondary["the_three_rungs"], secondary["trained_on"]
            own += [
                f"{of}, SECONDARY, G' ({len(again)} goal problems `{start}` does not solve in its first sampling), successes per attempt over {of_the_fresh}, `{name}` minus "
                f"`{start}` (the arm's primary): {_rate(on_again)}" + (f". NOTE: {secondary['the_goal_set_again']['seen_twice']}" if seen_twice else ""),
                f"{of}, SECONDARY, by the length of the shortest published proof, successes per attempt, `{name}` minus `{start}`: "
                + "; ".join(f"{_named(group)} ({len(sets[group])}): {_rate(entry)}" for group, entry in by_length[f"minus_{start}"].items()),
                f"{of}, SECONDARY, by the length of the shortest published proof, goal problems solved at least once, `{name}` against `{versus}`: "
                + "; ".join(f"{_named(group)}: {entry['resolved_after']} to {entry['resolved_before']}, {_split(entry)}" for group, entry in by_length[f"against_{versus}"].items()),
                *([] if by_itself else [
                    f"{of}, SECONDARY, goal problems solved at {attempts} attempts, `{name}` against `{start}`: "
                    f"{secondary[f'goal_problems_solved_against_{start}']['resolved_after']} to "
                    f"{secondary[f'goal_problems_solved_against_{start}']['resolved_before']}, {_split(secondary[f'goal_problems_solved_against_{start}'])}"]),
                f"{of}, SECONDARY, goal problems solved in at least one episode of 8 one-shot attempts / reliably (in at least half of their episodes): "
                + "; ".join(f"{_named(group)} ({len(sets[group])}): " + ", ".join(f"`{other}` {alone[other][group]['solved_at_least_once']} / {alone[other][group]['reliably']}"
                                                                                  for other in (*others, name)) for group in (ALL, FOUR_PLUS)),
                f"{of}, SECONDARY, the three rungs ({prepare['rung_episodes']} episodes a problem): `{name}` minus `{start}`: "
                + "; ".join(f"{rung} {interval_text(rungs[f'minus_{start}'][rung])}" for rung in RUNGS)
                + ("" if by_itself else f". `{name}` minus `{versus}`: " + "; ".join(f"{rung} {interval_text(rungs[f'minus_{versus}'][rung])}" for rung in RUNGS)),
                f"{of}, SECONDARY, the share of distinct attempts (on the rungs, on G): "
                + "; ".join(f"`{other}` {who[other]['distinct_attempts'][RUNG_PART]['mean_share_distinct']}, {who[other]['distinct_attempts'][GOAL]['mean_share_distinct']}"
                            for other in (*others, name)),
                f"{of}, SECONDARY, the lines of the proofs each model verifies on G: " + "; ".join(f"`{other}` {_lines(who[other]['verified_proof_lines'][GOAL])}"
                                                                                                 for other in (*others, name)),
                f"{of}, SECONDARY, the goal problems nothing stored had solved: "
                + (f"of the {unsolved['goal_problems']} listed, `{name}` solves {unsolved['solved']}" + _named_problems(unsolved["problem_ids"])
                   + "".join(("; " if index == 0 else " and ") + f"`{other}` " + ("solves " if index == 0 else "") + f"{unsolved[f'solved_by_{other}']}"
                             for index, other in enumerate(counted_beside))
                   + (f"; {unsolved['not_goal_problems_of_this_run']} more ids of the file are not goal problems of this run and are not counted"
                      if unsolved["not_goal_problems_of_this_run"] else "")
                   if unsolved["given"] else "not given (no file of their ids is in the run directory; the stage does not compute it)"),
                f"{of}, SECONDARY, the rows it was trained on: {record['rows']:,}, by origin " + ", ".join(f"{key} {count:,}" for key, count in record["rows_by_origin"].items())
                + "; by k " + (", ".join(f"{key}: {count:,}" for key, count in record["rows_by_k"].items()) or "none") + f"; their proofs: {_lines(record['rows_by_lines'])}"]
        lines_of[name] = own

    # ---- `hot`: read, not branched
    of_hot = prepare[HOT]
    hot_samplings, cold, warm = of_hot["samplings"], of_hot["temperature_of_every_other_measurement"], of_hot["temperature"]
    hot_attempts = of_hot["attempts_a_goal_problem"]
    t = lambda value: repr(float(value))      # noqa: E731 - a temperature as a line says it: 1.0, 1.2
    table, hot_lines = {}, []
    for name in hot_models:
        at = {"stored": {GOAL: who[name][GOAL], "distinct": who[name]["distinct_attempts"][GOAL], "temperature": cold},
              "hot": {GOAL: hot[name][GOAL], "distinct": hot[name]["distinct_attempts"][GOAL], "temperature": warm}}
        entry = {}
        for key, side in at.items():
            summed = attempts_on_g(side[GOAL])
            entry[key] = {"temperature": side["temperature"], "all_of_g": on_a_set(summed, goal_ids), "the_goal_set_again": on_a_set(summed, list(again)),
                          "the_goal_set_again_over_the_second_sampling": on_a_set(side[GOAL][1], list(again)) if len(side[GOAL]) > 1 else None,
                          "distinct_attempts_on_g": side["distinct"], "verified_proof_lines_on_g": verified_lines((hot[name] if key == "hot" else who[name])["verified_proof_lines"][GOAL])}
        same_attempts = hot_attempts == attempts and len(hot[name][GOAL]) == len(who[name][GOAL])
        entry["hot_against_stored"] = {
            "what": f"`{name}` at {t(warm)} against itself at {t(cold)}, on all of G, paired by problem: the problems solved at least once (gained, lost, the two-sided sign "
                    "test) and the successes per attempt (95% bootstrap over problems). The two share no sampling seed",
            "solved": solved(attempts_on_g(hot[name][GOAL]), on_g[name], goal_ids),
            "per_attempt": per_attempt(attempts_on_g(hot[name][GOAL]), on_g[name], goal_ids, resamples, seed) if same_attempts else dict(NOT_MEASURED)}
        entry["not_to_be_read"] = [key for key in not_to_be_read if key.startswith(f"goal_{HOT}_{name}_")]
        table[name] = entry
        stored_g, hot_g, changed = entry["stored"], entry["hot"], entry["hot_against_stored"]
        where = "stored" if name in stored_names else "measured here"
        hot_lines += [
            f"{SAY}: `{HOT}`, read and not branched. `{name}` on all of G ({len(goal_ids)} problems, {hot_attempts} attempts a problem at {t(warm)}, {attempts} at {t(cold)}): at "
            f"{t(cold)} ({where}) {_own(stored_g['all_of_g'])}, share of distinct attempts {stored_g['distinct_attempts_on_g']['mean_share_distinct']}; at {t(warm)} "
            f"{_own(hot_g['all_of_g'])}, share of distinct attempts {hot_g['distinct_attempts_on_g']['mean_share_distinct']}; {t(warm)} against {t(cold)}: "
            f"{_split(changed['solved'])}, successes per attempt {_rate(changed['per_attempt'])}"
            + (f". NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(entry['not_to_be_read'])}" if entry["not_to_be_read"] else ""),
            f"{SAY}: `{HOT}`, the same on G' ({len(again)} problems; at {t(cold)} `{start}`'s first sampling CHOSE them, so `{start}` has no success there in it): `{name}` at {t(cold)} "
            f"{_own(stored_g['the_goal_set_again'])}; at {t(warm)} {_own(hot_g['the_goal_set_again'])}"
            + (f"; over the second sampling alone, at {t(cold)} {_own(stored_g['the_goal_set_again_over_the_second_sampling'])}, at {t(warm)} "
               f"{_own(hot_g['the_goal_set_again_over_the_second_sampling'])}" if stored_g["the_goal_set_again_over_the_second_sampling"] and hot_g["the_goal_set_again_over_the_second_sampling"] else "")]

    sizes = prepare["training_sets"]
    read_models = tuple(name for name in trained_models if name != against)
    lines = [
        f"{SAY}: L4t, {QUESTION.upper()} Three one-change checks on the rounds already made, seed {prepare['seed']}. NO NEW ROUND: from the stored rounds of the arm "
        f"{prepare['arm']} ({prepare['arm_run']}) and from `{prepare['start']}`. Every model is trained FROM `{prepare['start']}`, one pass, the arm's recipe, in the "
        f"content-hash order, and measured as `with` was with `pre`'s sampling seeds. "
        + ("" if from_pre else f"EVERY MODEL IS AT RANK {rank}, the start adapter's. `{OLD_RULE}`, FIRST: the twin's {sizes[OLD_RULE]['rows']:,} one-shot rows, in the twin's "
                               f"order (what `without` is, from `{start}`). ")
        + f"`{REHEARSE}`: {sizes[REHEARSE]['rows']:,} rows ({sizes[REHEARSE]['twin_rows']:,} of the "
        f"twin and as many published proofs of the pretraining file). `{REWARD_ROWS}`: {sizes[REWARD_ROWS]['rows']:,} of the rounds' {sizes[REWARD_ROWS]['rows_of_the_rounds']:,} "
        f"rows, each kept with the probability of its problem's reward at t = {prepare['target_rate']:g}"
        + (f" ({sizes[REWARD_ROWS]['kept_by_the_runs_minimum']} of them by a smoke run's minimum and NOT by the rule)" if sizes[REWARD_ROWS]["kept_by_the_runs_minimum"] else "")
        + f". `{HOT}`: `{hot_models[0]}` and `{hot_models[1]}` sampled again on G at temperature {t(warm)}. "
        + ("Each trained model is read BY ITSELF against the stored `without` and `pre`" if from_pre else
           f"`{against}` is read BY ITSELF against the stored `{start}`, which is the cap's question; each of the two others BY ITSELF against `{against}`, which stands in "
           f"`without`'s place, and `{start}`, which stands in `pre`'s. THE ROWS ARE THE RANK-{(prepare.get('rows_made_by') or {}).get('rank')} ROUNDS': each proof was written "
           f"by a model built on `pre`, and each row's k is that model's")]
    if from_pre:
        lines.append(f"{SAY}: from the stored rows, what the old rule gave, on all of G over {attempts} attempts a problem: "
                     + "; ".join(f"`{name}` {_own(reference['on_all_of_g'][name])}" for name in STORED) + f"; `without` minus `pre` {_rate(reference['without_minus_pre'])}; `with` "
                     f"minus `pre` {_rate(reference['with_minus_pre'])}")
    if not fresh:
        lines.append(f"{SAY}: the stored models have ONE sampling of G here (a smoke run): G was attempted {attempts} times a problem, and G' has no fresh attempts to be read on")
    for name in trained_models:
        lines += lines_of[name]
    lines += hot_lines
    if not_to_be_read:
        lines.append(f"{SAY}: NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(not_to_be_read)}. The step fails; the task queued again "
                     "measures such a set of this run's own again from the kept adapter")

    def said(name: str) -> str:
        own, by_itself = read[name], name == against
        if own["primary"] is None:              # INCONCLUSIVE, or not read against an INCONCLUSIVE model
            return f"`{name}`: {own['branch']['name']}" + (f" (`{own['branch']['not_read_against']}` is INCONCLUSIVE)" if "not_read_against" in own["branch"] else "")
        if by_itself:
            return f"`{name}` by itself: {own['branch']['name']} (against `{start}`: {_split(own['primary'])}; all of G per attempt: {interval_text(own['beside_the_primary'])})"
        return (f"`{name}`: {own['branch']['name']}"
                + f" (against `{against}`: {_split(own['primary'])}; all of G against `{start}`: {interval_text(own['beside_the_primary'])})")

    headline = (f"L4t ({QUESTION[:-1]}; {LABEL}) seed {prepare['seed']}, from `{prepare['start']}`, a sign test under {SIGN_TEST_LEVEL:g} and an interval above zero asked. "
                + ". ".join(said(name) for name in trained_models) + f". `{HOT}` at {t(warm)}, read and not branched: "
                + "; ".join(f"`{name}` solves {table[name]['hot']['all_of_g']['solved_at_least_once']} goal problems against {table[name]['stored']['all_of_g']['solved_at_least_once']} "
                            f"at {t(cold)}, {table[name]['hot']['all_of_g']['per_1000']} against {table[name]['stored']['all_of_g']['per_1000']} per 1,000" for name in hot_models)
                + (f". NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(not_to_be_read)}" if not_to_be_read else ""))
    report = {
        "spec": "docs/spec/ladder-loop.spec.md, L4t: what should a round train on? Three one-change checks on the rounds already made", "stage": L4, "label": LABEL,
        "check": "L4t", "headline": headline, "lines": lines, "ok": not not_to_be_read, "seed": prepare["seed"],
        "stand_in_engine": bool(prepare.get("stand_in_engine") or any(entry.get("stand_in_engine") for entry in (*trains.values(), *models.values(), *hot.values()))),
        "branches": {name: read[name]["branch"]["name"] for name in trained_models}, "models": read, "what_the_old_rule_gave": reference,
        HOT: {"what": f"read, not branched: `{hot_models[0]}` and `{hot_models[1]}` at temperature {t(cold)} ("
                      + ("their stored rows" if from_pre else f"`{hot_models[0]}`'s rows measured by this run, `{hot_models[1]}`'s stored")
                      + f") and at {t(warm)} (sampled here on G alone, {hot_attempts} attempts a "
                      "problem, with sampling seeds of their own): the distinct goal problems solved, the successes per attempt and the share of distinct attempts, on all "
                      "of G and on G'. It says how much of the breadth is sampling, and which temperature later measurements of trained models should also report",
              "temperature": warm, "temperature_of_the_stored_rows": cold, "samplings": hot_samplings, **table},
        "what_it_cannot_say": list(CANNOT_SAY) if from_pre else [
            *CANNOT_SAY, f"The rows are the rounds the arm made from `pre`: the proofs were written by models built on `pre`, and each row's k is that model's. `{start}` is "
                         "stronger, so some of what was its limit for them is not for it: this is the training rule on given rows at this rank, not the rounds "
                         f"`{start}` would have made. What it settles is confirmed by the arm from `{start}`"],
        "the_stored_models": {PRE: f"`pre`: {LABEL}; measured by its own stage, its rows are that run's", WITH: "`with`: the arm's last model, `pre` trained one more pass on "
                              "the rounds' proofs by the old rule; its rows are the arm's run's", WITHOUT: "`without`: its twin, from `pre` on the rounds' one-shot rows alone; "
                              "its rows are the arm's run's", "runs": prepare["stored_models"]} if from_pre else {
            start: f"`{start}`: {LABEL}; the model of the check of the adapter's rank, measured by that stage; its rows are that run's",
            "runs": prepare["stored_models"], "rank_16": prepare.get("rank_16")},
        "training_sets": prepare["training_sets"],
        "heldout": {"goal_set": len(goal_ids), "goal_set_again": len(again), "goal_set_by_length_group": {group: len(ids) for group, ids in sets.items()},
                    "rungs": {rung: len(rung_ids[rung]) for rung in RUNGS}},
        "sizes": {key: prepare[key] for key in ("sampling_seeds", "rung_episodes", "goal_samplings", "attempts_a_goal_problem", "contradicted_side_setting", "start",
                                                "start_adapter", "arm", "arm_run", "pretraining_run", "target_rate")},
        "attempts": health, "not_to_be_read": not_to_be_read,
        "not_measured": ["`pre`, `with` and `without` at the config's temperature (their rows are the stored ones of the runs named under the_stored_models)",
                         "the rungs at the hot temperature: `hot` samples G alone", "a new round: the rounds are the arm's, attempted by models trained the old way",
                         "anything with assembly on the goal set"] if from_pre else [
            f"`{start}` at the config's temperature (its rows are the stored ones of the run named under the_stored_models)",
            "the rungs at the hot temperature: `hot` samples G alone", f"a new round: the rounds are the arm's, attempted by models built on `pre`, not on `{start}`",
            "anything with assembly on the goal set"],
    }
    if not from_pre:        # what a run from another start than `pre` says beside: from `pre` the report is what it always was
        in_g_again = [problem_id for problem_id in goal_ids if problem_id in in_again]
        report.update({
            "start": start, "reference": against, "rank": rank, "read_against_the_reference": list(read_models),
            "the_old_rule_by_itself": {"model": against, "outcome": read[against]["branch"], "what": "the cap's question: does the old rule narrow at the start adapter's "
                                       "rank too? Its figures are under models, `by_itself`; rank 16's stored ones are under what_the_old_rule_gave"},
            "the_goal_set_again_of_the_start": {"what": f"G' is `{start}`'s: the goal problems it does not solve in its first sampling of G, made by the prepare step from its "
                                                        "stored rows (the run that made it stored none); here by the length group of each one's shortest published proof",
                                                "problems": len(again), "by_length_group": {group: len(ids) for group, ids in goal_sets(in_g_again, lengths).items()}},
            "start_recipe": prepare.get("start_recipe"), "rows_made_by": prepare.get("rows_made_by")})
    return report
