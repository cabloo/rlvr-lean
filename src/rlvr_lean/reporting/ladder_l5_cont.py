"""L5c's report: the continuation. What lifted the published kind by 34 per 1,000 from the twice-pretrained model, and
does more of it go higher? Spec: docs/spec/ladder-loop.spec.md, "L5c: the continuation" ("The reads, fixed before
the run", "What follows"). Pure: rows in, a report out, on the rules of `domain/ladder_round/l5_cont.py`.

Everything here is labelled pretrained on published proofs, as L4's reports are: every line begins with L4's label.
One-shot attempts only. A model is ONE sampling of the goal set: no rung, no second sampling, no held-aside set.

  the checks        FIRST, each PASS or FAIL. This stage's: each set is its rule made again, `wide` is the set L5f's
                    run stores as `published` (as that run is when the report is built) with that run's four trainings
                    four orders of it, and `more` is L5s's rule counted again (`l5_cont.the_sets_are_the_rule`); L5f's
                    run of the seed carries this stage, as the first step held it and as its report reads now
                    (`l5_cont.of_l5f`); `second_full`'s adapter is there, at its rank, and not `pre_r64`'s; `single`'s
                    set was sampled from L5f's stored `published_1`; the stored first samplings of `second_full` and of
                    the bar are the setting's. Then L5d's own checks of the models BY L5d's OWN FUNCTIONS
                    (`reporting/ladder_l5_draws.py`: `checks_of_the_models`, `lines_of_the_checks`). ANY FAIL: THE RUN
                    IS INCONCLUSIVE and nothing else is said (what was measured is kept under `measured_and_not_read`).
                    L4p's OWN CHECKS OF `second_full` ARE PRINTED AND DECIDE NOTHING
  the reads         in the section's order, each against L5f's STORED `published` first sampling (the bar), paired by
                    problem: WHAT MADE IT (`l5_cont.what_made_it`); a wider average, a second step, more rows
                    (`the_read`); beside, naming nothing, each model against `second_full` and against `pre_r64`
                    (`against_the_start`, L5s's own), overfitting, the timeouts; what follows (`what_follows`)

A report in which Lean did not answer a set is NOT TO BE READ (`ok` is false, and the step fails): the task queued again
samples such a set again from its adapter.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE
from rlvr_lean.domain.ladder_round.l5 import timed_out
from rlvr_lean.domain.ladder_round.l5_cont import (
    AGAIN,
    ASKED,
    AVERAGED,
    BAR,
    MODELS,
    MORE,
    OF_THEIRS,
    SINGLE,
    START,
    THE_SINGLE,
    WIDE,
    against_the_start,
    of_l5f,
    the_read,
    the_sets_are_the_rule,
    what_follows,
    what_made_it,
)
from rlvr_lean.domain.ladder_round.l5_draws import AGAIN as G_AGAIN
from rlvr_lean.domain.ladder_round.l5_draws import ALL, SOLVED, on_the_parts, the_counts, the_parts, the_start_is_the_settings
from rlvr_lean.domain.ladder_round.l5_noise import figure
from rlvr_lean.domain.ladder_round.read import GOAL, group_ids
from rlvr_lean.reporting.ladder_ceiling import goal_sets
from rlvr_lean.reporting.ladder_l2 import _health
from rlvr_lean.reporting.ladder_l3c import LENGTH_GROUPS
from rlvr_lean.reporting.ladder_l4 import LABEL, SAY, _verdict
from rlvr_lean.reporting.ladder_l5_draws import PART_SAID, THE_RUN, THE_SAMPLING, _per_1000, checks_of_the_models, lines_of_the_checks
from rlvr_lean.reporting.ladder_l5_draws import THE_START as THE_STARTS_SAMPLING
from rlvr_lean.reporting.ladder_l5_full import THE_START_ADAPTER, the_start_adapter
from rlvr_lean.reporting.ladder_l5_noise import LEAN_ANSWERED

L4 = "l4"
QUESTION = "the continuation"
SPEC = "docs/spec/ladder-loop.spec.md, L5c: the continuation"
LONG = LENGTH_GROUPS[-1]                    # the goal problems whose shortest published proof has 8 lines or more: the "long problems" of the overfitting reads
THE_BAR = f"l5f_{BAR}"                      # L5f's stored `published` of the seed, as the counts and the timeouts name it
THE_SETS = "each_set_is_its_rule_made_again_and_wide_and_more_are_l5fs_and_l5ss"
L5FS_RUN = "l5fs_run_of_the_seed_carries_this_stage"
SINGLE_SERVED = "single_was_sampled_from_l5fs_stored_training"
THE_BARS_SAMPLING = "the_stored_first_sampling_of_the_bar_is_the_settings"
CANNOT_SAY = [
    "ONE SEED and one draw of rows: a scout, by the repo's seed rule. Each read names an escalation or none, and no effect",
    f"`{START}` is ONE training in one order: every model here, and the bar, inherits that order's luck",
    "Nothing here is about the model's own proofs: no row of this stage is one",
    f"The published rows are ones `{START}` has trained on twice: a gain on G from a third reading of them is a gain of the recipe, and whether it holds on the held-aside "
    "set is not measured here",
    "G alone and one sampling of it, 32 attempts a problem: nothing of the rungs, of the second sampling or of the held-aside set"]


def _signed(value: float | None) -> str:
    return "not read" if value is None else _per_1000(value, True)


def l5fs_run(of_the_first_step: Mapping, l5f: Mapping, seed: int, recipe: str, measurement: Mapping) -> dict:
    """The check of L5f's run of the seed: its adapters were held by the first step (as its own steps recorded them and,
    on a GPU box, from the files), and its report, read AGAIN now, still carries this stage (`l5_cont.of_l5f`)."""
    why = of_l5f(l5f, seed, recipe, measurement)
    held = of_the_first_step.get("held") or {}
    return {"what": f"L5f's run of this seed is there, finished and not {INCONCLUSIVE}, of this seed, start and recipe and measured as this run; its four trainings of "
                    f"`{BAR}` and its averaged one are at the rank, not the start adapter's, the averaged one their mean (as the first step held them)",
            "run": of_the_first_step.get("run"), "adapters_held_by_the_first_step": bool(held.get("passes")), "read_now_of_the_adapters": held.get("read_now"),
            "its_report_now": why is None, "why_not": why, "passes": bool(held.get("passes")) and why is None}


def single_served(measured: Mapping, rows: Sequence[Mapping], adapter: str | None) -> dict:
    """`single`'s set was sampled from L5f's stored `published_1` and from nothing else: its measuring step names that
    adapter, and every goal problem has attempts."""
    served = bool(adapter) and measured.get("adapter") == adapter and bool(rows) and all(row["episodes"] > 0 for row in rows)
    return {"what": f"`{SINGLE}`'s set was sampled from L5f's stored `{THE_SINGLE}`, where that run keeps it", "adapter": adapter, "served_from": measured.get("adapter"),
            "passes": served}


def build_l5_cont_report(prepare: Mapping, trains: Mapping[str, Mapping], trained: Mapping[str, Sequence[Mapping]], averaged: Mapping[str, Mapping],
                         measured: Mapping[str, Mapping], rows: Mapping[str, Sequence[Mapping]], sets: Mapping[str, Sequence[Mapping]], pretraining: Sequence[Mapping],
                         start_rows: Sequence[Mapping], first_start_rows: Sequence[Mapping], bar_rows: Sequence[Mapping], groups: Sequence[Mapping],
                         lengths: Mapping[str, Mapping], again: Sequence[str], aside: Sequence[str], settings: Mapping, base_map: Sequence[str], of_the_setting: Mapping,
                         of_theirs_sets: tuple[Mapping[str, Sequence[str] | None], Mapping[str, Sequence[str] | None]], l5f: Mapping, more_by_the_rule: int | None) -> dict:
    """`prepare`: the prepare step's summary. `trains`, `trained`, `averaged`, `measured`, `rows`, `sets`,
    `pretraining`, `groups`, `lengths`, `again`, `aside`, `settings`, `base_map`, `of_the_setting`: as L5d's report takes
    them (`reporting/ladder_l5_draws.build_l5_draws_report`); `sets` by averaged model and by trained model.
    `start_rows`: `second_full`'s stored first sampling, as this run copied it. `first_start_rows`: `pre_r64`'s.
    `bar_rows`: L5f's stored `published`'s. `of_theirs_sets`: (the ids L5f's run of this seed STORES as `published` and
    as each of its four orders; the ids each of those trainings' loops RECORDED), each None where that run holds none,
    read by the report step from that run as it is now. `l5f`: what the report of L5f's run of this seed says of its own
    seed (`of_this_seed`), or `{"there": False, "why": ...}`. `more_by_the_rule`: the rows L5s's rule gives `more`,
    counted again by the report step."""
    start, first, seed, models, sampling = prepare["start"], prepare["first_start"], prepare["seed"], list(prepare["models"]), prepare["the_sampling"]
    if start != START or models != list(MODELS) or sorted(rows) != sorted(models) or sorted(trains) != sorted(prepare["trained"]) or sorted(averaged) != sorted(prepare["averaged"]):
        raise ValueError(f"the report is given a run from `{start}` with the rows of {sorted(rows)}, the trainings of {sorted(trains)} and the averages of {sorted(averaged)}: it "
                         f"reads a run from `{START}` and its models ({list(MODELS)}), each measured")
    goal_ids = group_ids(groups, GOAL)
    if any(sorted(row["problem_id"] for row in own) != sorted(goal_ids) for own in rows.values()):
        raise ValueError("the models were not measured on the goal problems, each once: every model here is one sampling of the one goal set")
    rank, attempts = (prepare.get("start_recipe") or {}).get("rank"), sampling["episodes"]
    long_ids = goal_sets(goal_ids, lengths).get(LONG, [])
    stored, recorded = of_theirs_sets
    of_the_start_model, of_l5fs_run, recipe = prepare["the_start_model"], prepare["l5fs_run"], prepare["recipe"]["name"]
    the_measurement = {"sampling": sampling["name"], "episodes": attempts, "sampling_seed": sampling["sampling_seed"], "problems": len(goal_ids)}

    # ---- the checks, FIRST: this stage's of the run, then L5d's own of the models
    in_orders = {name: sets[name] for name in prepare["trained"]}
    checks = {THE_RUN: {THE_SETS: the_sets_are_the_rule({model: sets[model] for model in AVERAGED}, pretraining, seed, in_orders, stored, recorded, more_by_the_rule),
                        L5FS_RUN: l5fs_run(of_l5fs_run, l5f, seed, recipe, the_measurement),
                        THE_START_ADAPTER: the_start_adapter(of_the_start_model),
                        SINGLE_SERVED: single_served(measured[SINGLE], rows[SINGLE], (of_l5fs_run.get("adapters") or {}).get(THE_SINGLE)),
                        THE_STARTS_SAMPLING: the_start_is_the_settings(start_rows, goal_ids, sampling, of_the_setting["sampling"], of_the_setting["attempts"]),
                        THE_BARS_SAMPLING: the_start_is_the_settings(bar_rows, goal_ids, sampling, of_the_setting["sampling"], of_the_setting["attempts"])},
              **checks_of_the_models(prepare, trains, trained, averaged, measured, rows, sets, goal_ids, aside, [row["problem_id"] for row in groups], settings["half_seed"],
                                     base_map)}
    failed = [f"{'the run' if name == THE_RUN else f'`{name}`'}: {check}" for name, of in checks.items() for check, entry in of.items() if not entry["passes"]]
    inconclusive = bool(failed)
    not_to_be_read = [name for name in models if not checks[name][LEAN_ANSWERED]["passes"]]

    # ---- the figures; the parts of G are THE FIRST START's own first sampling's (the 200 it solves there, and G'), held to the G' the first step copied
    try:
        parts = the_parts(first_start_rows, again)
    except ValueError:      # the stored sampling is not the one G' was made from: no figure is given on a part
        parts = {ALL: list(goal_ids), SOLVED: [], G_AGAIN: []}
    share_distinct = lambda own: ((own.get("distinct_attempts") or {}).get(GOAL) or {}).get("mean_share_distinct")      # noqa: E731
    figures = {name: {part: {**own, **({"share_of_distinct_attempts": share_distinct(measured[name])} if part == ALL else {})}
                      for part, own in on_the_parts(rows[name], parts).items()} for name in models}
    of_the_start, of_pre, of_the_bar = figure(list(start_rows)), figure(list(first_start_rows)), on_the_parts(bar_rows, parts)
    counts = the_counts({THE_BAR: bar_rows, **{name: rows[name] for name in models}}, long_ids, against=THE_BAR)

    # ---- THE READS, in the section's order: what made it; the three against the bar; each model against the stored models; overfitting; timeouts; what follows
    on_g = {name: figures[name][ALL] for name in models}
    made = what_made_it(failed, on_g[SINGLE], of_the_bar[ALL])
    reads = {model: the_read(failed, model, on_g[model], of_the_bar[ALL]) for model in AVERAGED}
    single_minus_start = None if None in (on_g[SINGLE]["per_attempt"], of_the_start["per_attempt"]) else round(on_g[SINGLE]["per_attempt"] - of_the_start["per_attempt"], 6)
    against = {name: {THE_BAR: against_the_start(rows[name], bar_rows, parts, long_ids), start: against_the_start(rows[name], start_rows, parts, long_ids),
                      first: against_the_start(rows[name], first_start_rows, parts, long_ids)} for name in models}
    shares = {"share_of_distinct_attempts": {start: share_distinct(of_the_start_model), THE_BAR: share_distinct(of_l5fs_run.get("the_bar_wrote") or {})}}
    of_timeouts = timed_out({**{name: _health(rows[name]) for name in models}, THE_BAR: _health(bar_rows), start: _health(start_rows)},
                            {THE_SAMPLING: {name: name for name in (*models, THE_BAR, start)}})
    of_timeouts.update({"checked_again": False, "why": "no step checks the timed-out attempts again: where a pair is named here, it is to be done by hand (Lean only, no "
                                                       "measurement running) and the reads given both ways"})
    follows = what_follows(made, reads, on_g, of_the_bar[ALL], prepare["the_seeds"]["more_seeds"])
    of_this_seed = {"seed": seed, "start": start, "recipe": recipe, "ok": not not_to_be_read, "inconclusive": inconclusive, "failed_checks": failed,
                    "the_measurement": the_measurement, "figures": None if inconclusive else figures, "the_bar": None if inconclusive else of_the_bar,
                    "what_made_it": None if inconclusive else made.get("difference"),
                    "the_reads": None if inconclusive else {model: reads[model].get("difference") for model in AVERAGED}}

    of_the_sets, of_their_run, of_the_adapter, of_single, of_the_bars, smoke = (checks[THE_RUN][THE_SETS], checks[THE_RUN][L5FS_RUN], checks[THE_RUN][THE_START_ADAPTER],
                                                                               checks[THE_RUN][SINGLE_SERVED], checks[THE_RUN][THE_BARS_SAMPLING], prepare.get("smoke"))
    own_checks, search, the_more = of_the_start_model["l4ps_own_checks"], prepare["the_search"], prepare["the_more"]
    saved, now = (of_the_adapter["recorded_by_l4p"] or {}).get("adapter_saved"), of_the_adapter["read_now"]
    size = {model: of_the_sets["rows"][model] for model in AVERAGED}
    lines = [
        f"{SAY}: L5c, {QUESTION.upper()}. Seed {seed}: what made L5f's `{BAR}` (from `{start}`, {size[WIDE]:,} published rows in four orders, the averaged adapter) read what it "
        f"read, and does more of it go higher? NO ROW HERE IS ONE OF THE MODEL'S OWN PROOFS. Four models, at rank {rank}, one pass, the recipe as it is: `{SINGLE}` (L5f's "
        f"stored `{THE_SINGLE}`, measured: no training); `{WIDE}` (the mean of eight: L5f's four stored trainings and four more of the same rows, order seeds 5 to 8); "
        f"`{AGAIN}` (from L5f's stored averaged `{BAR}`: the next {size[AGAIN]:,} rows of the seed's order, four orders, averaged); `{MORE}` (from `{start}`: the first "
        f"{size[MORE]:,} rows, L5s's rule, four orders, averaged). L5f's run: {of_l5fs_run['run']}. EACH MODEL IS MEASURED on the goal set's first sampling alone: {attempts} "
        f"attempts at each of {len(goal_ids)} goal problems, sampling seed {sampling['sampling_seed']}; THE BAR of every read is L5f's stored `{BAR}` on that sampling, paired "
        "by problem"]
    if smoke:
        lines.append(f"{SAY}: A SMOKE RUN: L5's smoke run " + (f"reads INCONCLUSIVE ({', '.join(smoke['l5s_failed_checks'])}) and was read all the same; "
                                                              if smoke.get("l5_read_though_inconclusive") else "was read; ")
                     + "one deep solve is enough, and orders of a set's handful of rows that are alike are trained all the same. Nothing here is a read of the continuation")
    lines += [
        f"{SAY}: CHECK, each set is its rule made again (`{WIDE}` {size[WIDE]:,} rows, `{AGAIN}` the next {size[AGAIN]:,}, `{MORE}` {size[MORE]:,}; L5s's rule counted again: "
        f"{of_the_sets['rows_of_more_by_l5ss_rule']}), `{WIDE}` is the set L5f's run stores as `{BAR}` and that run's four trainings are four orders of it, and no row carries "
        "text: " + ("made again" if of_the_sets["refused"] is None else f"REFUSED ({of_the_sets['refused']})")
        + f"; sets that are not the rule: {of_the_sets['sets_that_are_not_the_rule']}; rows with text or not published: {of_the_sets['rows_with_text_or_not_published']}; sets "
        f"not in their order seed's order: {of_the_sets['sets_not_in_their_order_seeds_order']}; `{WIDE}` is L5f's stored set: {of_the_sets['wide_is_l5fs_stored_set']}; L5f's "
        f"trainings that are not orders of it: {of_the_sets['l5fs_trainings_that_are_not_orders_of_wide']}, not of these rows: "
        f"{of_the_sets['l5fs_trainings_not_of_these_rows']}: {_verdict(of_the_sets)}",
        f"{SAY}: CHECK, L5f's run of this seed carries this stage ({of_their_run['run']}): its four trainings of `{BAR}` and its averaged one held by the first step: "
        f"{of_their_run['adapters_held_by_the_first_step']}" + ("; the files themselves were read" if of_their_run["read_now_of_the_adapters"] else
                                                                "; its files were not read by the first step (no GPU box)")
        + "; its report now: " + ("read, of this seed, start and recipe, measured alike" if of_their_run["its_report_now"] else f"NOT ({of_their_run['why_not']})")
        + f": {_verdict(of_their_run)}",
        f"{SAY}: CHECK, the adapter of `{start}` is there, is at its rank ({of_the_adapter['rank']}, alpha {of_the_adapter['alpha']}) and is not `{first}`'s: "
        + (f"L4p's run saved it at rank {saved['rank']} and alpha {saved['alpha']}" if saved else "L4p's run recorded no read back of it")
        + (f"; read now at rank {now['read_back']['rank']} and alpha {now['read_back']['alpha']}, {now['against_the_first_start']['elements_changed']:,} of its "
           f"{now['against_the_first_start']['elements']:,} numbers not `{first}`'s" if now else "; its files were not read by the first step (no GPU box)")
        + f": {_verdict(of_the_adapter)}",
        f"{SAY}: CHECK, `{SINGLE}`'s set was sampled from L5f's stored `{THE_SINGLE}` ({of_single['adapter']}): served from {of_single['served_from']}: {_verdict(of_single)}",
        f"{SAY}: CHECK, the stored first sampling of the bar (L5f's `{BAR}`) is the setting's ({of_the_bars['problems']} problems of {of_the_bars['goal_problems']}, sampling "
        f"seed {of_the_bars['sampling_seed']}, attempts a problem that are not {attempts}: {of_the_bars['attempts_a_problem_that_are_not_the_settings']}): "
        f"{_verdict(of_the_bars)}",
        *lines_of_the_checks(prepare, checks, rank, attempts, not_to_be_read, mean_of="its members' (eight for `wide`, four for the two others)"),
        f"{SAY}: L4p's OWN CHECKS OF `{start}`, PRINTED AND DECIDING NOTHING (a second pass over rows the model has trained on has no falling loss to show; L6 treats it so): "
        + (f"its report reads the model {INCONCLUSIVE} for {own_checks['failed']}" if own_checks["failed"] else
           "its report failed none of them" if own_checks["report_there"] else "its report is not on the box")
        + f"; mean loss over its first {own_checks['rows_compared']} rows {own_checks['mean_loss_over_the_first_rows']}, over its last "
          f"{own_checks['mean_loss_over_the_last_rows']}"]
    if inconclusive:
        lines.append(f"{SAY}: THIS RUN IS {INCONCLUSIVE}: {'; '.join(failed)}. Nothing else is said of it")
    else:
        def figure_said(part: str) -> str:
            return ", ".join(f"`{name}` {_per_1000(figures[name][part]['per_attempt'])} ({_per_1000(figures[name][part]['sampling_noise'])})" for name in models)

        def against_said(name: str, other: str) -> str:
            own = against[name][other]
            return (f"`{name}` " + ", ".join(f"{_per_1000(own['per_attempt'][part]['model'])} against {_per_1000(own['per_attempt'][part]['start'])} "
                                              f"({_signed(own['per_attempt'][part]['difference'])}) on {PART_SAID[part].replace('the start model', f'`{first}`')} "
                                              f"({own['per_attempt'][part]['problems']})" for part in (ALL, SOLVED, G_AGAIN))
                    + f"; goal problems solved {own['solved']} against {own['solved_by_the_start']} (+{own['gained']}, -{own['lost']}); long ones {own['long']['solved']} "
                      f"against {own['long_solved_by_the_start']}")

        lines += [
            f"{SAY}: THE FIGURES, successes per 1,000 attempts over the {attempts} attempts, each with its binomial sampling noise: on all of G ({len(parts[ALL])} problems) "
            f"{figure_said(ALL)}; on the goal problems `{first}` solves in its own first sampling ({len(parts[SOLVED])}) {figure_said(SOLVED)}; on G' ({len(parts[G_AGAIN])}) "
            f"{figure_said(G_AGAIN)}. THE BAR, L5f's stored `{BAR}`, on all of G: {_per_1000(of_the_bar[ALL]['per_attempt'])}; `{start}`: "
            f"{_per_1000(of_the_start['per_attempt'])}; `{first}`: {_per_1000(of_pre['per_attempt'])}",
            f"{SAY}: WHAT MADE IT: {made['name']}. {made['reason']}. Beside it, `{SINGLE}` minus `{start}`'s stored first sampling (what the rows of one training add): "
            f"{_signed(single_minus_start)}",
            *(f"{SAY}: {ASKED[model]} (a scout's read): {reads[model]['name']}. {reads[model]['reason']}" for model in AVERAGED),
            f"{SAY}: BESIDE, NAMING NOTHING, paired by problem (per 1,000 attempts, the model against the stored model). Against `{start}`'s stored first sampling: "
            + "; ".join(against_said(name, start) for name in models) + f". Against `{first}`'s: " + "; ".join(against_said(name, first) for name in models)
            + f". Goal problems solved against the bar ({counts[THE_BAR]['solved']}): "
            + "; ".join(f"`{name}` {counts[name]['solved']} (+{counts[name]['gained']}, -{counts[name]['lost']})" for name in models),
            f"{SAY}: OVERFITTING, each model against `{start}`: goal problems lost beside gained: "
            + "; ".join(f"`{name}` -{against[name][start]['lost']} beside +{against[name][start]['gained']}" for name in models)
            + ". The share of distinct attempts on G: " + ", ".join(f"`{name}` {figures[name][ALL]['share_of_distinct_attempts']}" for name in models)
            + f", the bar {shares['share_of_distinct_attempts'][THE_BAR]}, `{start}` {shares['share_of_distinct_attempts'][start]} (as their own runs recorded them). The "
            f"{len(long_ids)} long goal problems (8 lines or more) solved: "
            + "; ".join(f"`{name}` {against[name][start]['long']['solved']} against {against[name][start]['long_solved_by_the_start']} "
                        f"(+{against[name][start]['long']['gained']}, -{against[name][start]['long']['lost']})" for name in models),
            f"{SAY}: TIMEOUTS, each set's share of attempts that timed out in Lean: "
            + "; ".join(f"`{name}` {of_timeouts['share'][name]}" for name in models) + f"; the bar {of_timeouts['share'][THE_BAR]}; `{start}` {of_timeouts['share'][start]}"
            + (f". MORE THAN A POINT APART: {'; '.join(' and '.join(pair['models']) for pair in of_timeouts['apart'])}: the timed-out attempts are to be checked again (Lean "
               "only, no measurement running) and the reads given both ways; no step does it" if of_timeouts["apart"] else ". No two are more than a point apart"),
            f"{SAY}: WHAT FOLLOWS: {follows['name']}. {follows['reason']}"]
    if not_to_be_read:
        lines.append(f"{SAY}: NOT TO BE READ: too many attempts without a verdict from Lean in the sets of {', '.join(f'`{name}`' for name in not_to_be_read)}. The step fails; "
                     "the task queued again measures such a set again from its adapter")

    headline = (f"L5c (the continuation; {LABEL}) seed {seed}, against L5f's stored `{BAR}` from `{start}`: "
                + (f"THIS RUN IS {INCONCLUSIVE} ({'; '.join(failed)})" if inconclusive else
                   "per 1,000 attempts on all of G " + ", ".join(f"`{name}` {_per_1000(figures[name][ALL]['per_attempt'])}" for name in models)
                   + f" (the bar {_per_1000(of_the_bar[ALL]['per_attempt'])}, `{start}` {_per_1000(of_the_start['per_attempt'])}, `{first}` {_per_1000(of_pre['per_attempt'])}); "
                   f"WHAT MADE IT: {made['name']} ({_signed(made.get('difference'))}); "
                   + "; ".join(f"`{model}`: {reads[model]['name']} ({_signed(reads[model].get('difference'))})" for model in AVERAGED))
                + f". What follows: {follows['name']}"
                + (f". NOT TO BE READ: too many attempts without a verdict from Lean in the sets of {', '.join(not_to_be_read)}" if not_to_be_read else ""))
    kept = lambda what: {"measured_and_not_read": what} if inconclusive else what      # noqa: E731
    return {
        "spec": SPEC, "stage": L4, "label": LABEL, "check": "L5c", "headline": headline, "lines": lines, "ok": not not_to_be_read, "seed": seed,
        "stand_in_engine": bool(prepare.get("stand_in_engine") or any(entry.get("stand_in_engine") for entry in (*trains.values(), *averaged.values(), *measured.values()))),
        "start": start, "first_start": first, "rank": rank, "models": models, "recipe": recipe,
        "branches": {"what_made_it": made["name"], **{model: reads[model]["name"] for model in AVERAGED}, "what_follows": follows["name"]},
        "inconclusive": inconclusive, "failed_checks": failed,
        "can_this_run_be_read": {"what": "the checks the spec reads first; when any fails the run is INCONCLUSIVE and nothing else is said of it", **checks},
        "l4ps_own_checks_of_the_start_model": own_checks,
        "of_this_seed": of_this_seed,
        "figures": {"what": f"each model's successes per attempt over the {attempts} attempts of the first sampling, with its binomial sampling noise, on all of G, on the goal "
                            f"problems `{first}` solves in its own first sampling and on G'", "problems": {part: len(own) for part, own in parts.items()},
                    "the_stored_models": {THE_BAR: of_the_bar[ALL], start: of_the_start, first: of_pre}, **kept(figures)},
        "counts": {"what": f"the goal problems each model solves at least once in its attempts, gained and lost against the bar (L5f's stored `{BAR}`); and of the long problems "
                           "(a shortest published proof of 8 lines or more)", "long_problems": len(long_ids), **kept(counts)},
        "the_bar": {"what": f"L5f's stored `{BAR}` of this seed's run, on the goal set's first sampling: what every read here is against, paired by problem",
                    "run": of_l5fs_run["run"], "file": sampling["stored_bar"], **kept({"figures": of_the_bar})},
        "what_made_it": made if inconclusive else {"what": f"`{SINGLE}` (one of the four trainings the bar is the mean of, measured alone) minus the bar, per attempt on all "
                                                           f"of G; beside it, `{SINGLE}` minus `{start}`'s stored first sampling", **made,
                                                   "single_minus_the_start": single_minus_start},
        "the_reads": reads if inconclusive else {"what": "each of the three averaged models minus the bar, per attempt on all of G: a scout's read at one seed", **reads},
        "against_the_stored_models": kept(against),
        "overfitting": kept({"what": f"for each model against `{start}`: the goal problems lost beside the ones gained, the share of distinct attempts on G, and the long goal "
                                     "problems (8 lines or more) solved, gained and lost. It names no branch of its own", "long_problems": len(long_ids),
                             "the_stored_models": shares,
                             **{name: {"lost": against[name][start]["lost"], "gained": against[name][start]["gained"],
                                       "share_of_distinct_attempts": figures[name][ALL]["share_of_distinct_attempts"], "long": against[name][start]["long"],
                                       "long_solved_by_the_start": against[name][start]["long_solved_by_the_start"]} for name in models}}),
        "timed_out_in_lean": of_timeouts, "what_follows": follows,
        "the_seeds": prepare["the_seeds"], "the_start_model": of_the_start_model, "the_search": search, "l5fs_run": of_l5fs_run, "the_models": prepare["the_models"],
        "the_sets": prepare["the_sets"], "the_more": the_more, "the_orders": prepare["the_orders"], "the_sampling": sampling, "training_sets": prepare["training_sets"],
        "training_recipe": prepare["recipe"],
        "models_made": {SINGLE: {"model": f"`{SINGLE}`: L5f's stored `{THE_SINGLE}`, only read: no training here", "adapter": (of_l5fs_run.get("adapters") or {}).get(THE_SINGLE)},
                        **{name: {"model": f"`{name}`: trained by nothing: the mean of {', '.join(f'`{of}`' for of in averaged[name].get('of') or [])}",
                                  "kept_by_l5f": [of for of in prepare["the_averaged"][name]["kept_by_l5f"] if of in OF_THEIRS],
                                  "trainings": {of: {key: value for key, value in trains[of].items() if key != "checkpoints"} for of in prepare["the_averaged"][name]["of"]},
                                  "averaging": averaged[name]} for name in AVERAGED}},
        "smoke": smoke, "what_it_cannot_say": list(CANNOT_SAY),
        "attempts": {name: _health(rows[name]) for name in models}, "not_to_be_read": not_to_be_read,
        "not_built": ["the further seeds of a model that reads ADDS: they need L5f's `published` trained at those seeds (its four trainings and their mean), which no run "
                      "holds and no job file of this stage or of L5f's is written for; and the read over three seeds",
                      "the check again in Lean of the timed-out attempts where two shares differ by more than a point: the pairs are named",
                      "the measurement in full (the rungs, the second sampling, the held-aside set) of the candidate `what_follows` names"],
        "not_measured": ["the three rungs, the goal set's second sampling and the held-aside set: this stage reads one sampling of the goal set",
                         f"`{start}`, `{first}` and L5f's `{BAR}`: their rows are the stored ones of their own runs", "anything at another temperature than the config's"]}
