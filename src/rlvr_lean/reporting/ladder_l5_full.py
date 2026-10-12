"""L5f's report: the three kinds again, from the twice-pretrained model. Do the model's own proofs still add after a
whole second pass over published ones? Spec: docs/spec/ladder-loop.spec.md, "L5f: the three kinds again, from the
twice-pretrained model" ("The reads, fixed before the run", "What follows, fixed now"). Pure: rows in, a report out, on
the rules of `domain/ladder_round/l5_full.py`.

Everything here is labelled pretrained on published proofs, as L4's reports are: every line begins with L4's label.
One-shot attempts only. A model is ONE sampling of the goal set: no rung, no second sampling, no held-aside set.

  the checks        FIRST, each PASS or FAIL: L5d's own for a seed, BY L5d's OWN FUNCTIONS (`reporting/ladder_l5_draws.py`:
                    `the_checks`, `lines_of_the_checks`), with the start model `second_full`: each set is L5's rule for
                    its seed made again and held; the stored first sampling of `second_full` is the setting's; each
                    trained adapter is not the start adapter's and is read back at its rank; a model was trained on its
                    set and on nothing else; an averaged adapter is the mean of its four, none of them, and the one
                    served; Lean answered. And this stage's two: the sets are the ones L5d's run of the seed stored and
                    trained (`l5_full.the_sets_are_theirs`, on that run as it is when the report is built); the start
                    adapter is there, at its rank, and not `pre_r64`'s (as the first step held it). ANY FAIL: THE RUN
                    IS INCONCLUSIVE and nothing else is said (what was measured is kept under `measured_and_not_read`).
                    L4p's OWN CHECKS OF `second_full` ARE PRINTED AND DECIDE NOTHING
  the reads         in the section's order: THE STACK (`l5_full.the_stack`); THE LEVEL (`the_level`); the solve rate
                    against the two start models, paired by problem (`against_the_start`, L5s's own); overfitting
                    against `second_full`; the timeouts; what follows (`what_follows`)

A report in which Lean did not answer a set is NOT TO BE READ (`ok` is false, and the step fails): the task queued again
samples such a set again from its adapter.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE
from rlvr_lean.domain.ladder_round.l5 import DEEP, MODELS, PUBLISHED, SHALLOW, timed_out
from rlvr_lean.domain.ladder_round.l5_draws import AGAIN, ALL, SOLVED, differences, on_the_parts, the_counts, the_parts
from rlvr_lean.domain.ladder_round.l5_full import START, against_the_start, the_level, the_sets_are_theirs, the_stack, what_follows
from rlvr_lean.domain.ladder_round.l5_noise import figure
from rlvr_lean.domain.ladder_round.l5_stock import of_l5d
from rlvr_lean.domain.ladder_round.read import GOAL, group_ids
from rlvr_lean.reporting import ladder_l5_draws
from rlvr_lean.reporting.ladder_ceiling import goal_sets
from rlvr_lean.reporting.ladder_l2 import _health
from rlvr_lean.reporting.ladder_l3c import LENGTH_GROUPS
from rlvr_lean.reporting.ladder_l4 import LABEL, SAY, _verdict
from rlvr_lean.reporting.ladder_l5_draws import PART_SAID, THE_RUN, THE_SAMPLING, THE_SETS, _per_1000, lines_of_the_checks
from rlvr_lean.reporting.ladder_l5_draws import THE_START as THE_STARTS_SAMPLING
from rlvr_lean.reporting.ladder_l5_noise import LEAN_ANSWERED

L4 = "l4"
QUESTION = "the three kinds again, from the twice-pretrained model"
SPEC = "docs/spec/ladder-loop.spec.md, L5f: the three kinds again, from the twice-pretrained model"
LONG = LENGTH_GROUPS[-1]                    # the goal problems whose shortest published proof has 8 lines or more: the "long problems" of the overfitting reads
THE_SETS_ARE_THEIRS = "the_sets_are_the_ones_l5ds_run_of_the_seed_trained"
THE_START_ADAPTER = "the_start_adapter_is_there_at_its_rank_and_is_not_the_first_starts"
CANNOT_SAY = [
    "ONE SEED: a scout, by the repo's seed rule. It names an escalation or none, and no effect",
    f"`{START}` is ONE training in one order: every model here inherits that order's luck, so the differences between the three kinds do not carry it, and what the second "
    "pass adds under them does (a single training's order is worth about 0.005 per attempt)",
    "The own proofs are of base-map problems and were found by the once-pretrained model, not by the model they are trained into here: a round of search BY the "
    "twice-pretrained model is another experiment",
    "The published rows of every set are rows the start model has trained on twice",
    "G alone and one sampling of it, 32 attempts a problem: nothing of the rungs, of the second sampling or of the held-aside set"]


def _signed(value: float | None) -> str:
    return "not read" if value is None else _per_1000(value, True)


def the_start_adapter(of_the_start: Mapping) -> dict:
    """The check of the start adapter, from what the first step held and recorded of it (`the_start_model`): it was
    there, at the first start's rank and alpha, and not the first start's adapter."""
    return {"what": f"the adapter of `{START}`, the start of every model here, is there, is at its rank and alpha, and is not the first start's (as the first step held it "
                    "before anything was written: on a GPU box from the files themselves, and from what L4p's run recorded when it saved it)",
            "run": of_the_start.get("run"), "adapter": of_the_start.get("adapter"), "rank": of_the_start.get("rank"), "alpha": of_the_start.get("alpha"),
            "recorded_by_l4p": of_the_start.get("recorded_by_l4p"), "read_now": of_the_start.get("read_now"), "passes": bool(of_the_start.get("passes"))}


def build_l5_full_report(prepare: Mapping, trains: Mapping[str, Mapping], trained: Mapping[str, Sequence[Mapping]], averaged: Mapping[str, Mapping],
                         measured: Mapping[str, Mapping], rows: Mapping[str, Sequence[Mapping]], sets: Mapping[str, Sequence[Mapping]], pretraining: Sequence[Mapping],
                         start_rows: Sequence[Mapping], first_start_rows: Sequence[Mapping], groups: Sequence[Mapping], lengths: Mapping[str, Mapping], again: Sequence[str],
                         aside: Sequence[str], settings: Mapping, base_map: Sequence[str], of_the_setting: Mapping,
                         of_theirs_sets: tuple[Mapping[str, Sequence[str] | None], Mapping[str, Sequence[str] | None]], l5d: Mapping) -> dict:
    """`prepare`: the prepare step's summary. `trains`, `trained`, `averaged`, `measured`, `rows`, `sets`,
    `pretraining`, `groups`, `lengths`, `again`, `aside`, `settings`, `base_map`, `of_the_setting`: as L5d's report takes
    them (`reporting/ladder_l5_draws.build_l5_draws_report`). `start_rows`: `second_full`'s stored first sampling, as
    this run copied it. `first_start_rows`: `pre_r64`'s. `of_theirs_sets`: (the ids of each set L5d's run of this seed
    STORED, in its order; the ids each of its trainings' loops RECORDED), each None where that run holds none, read by
    the report step from that run as it is now. `l5d`: what the report of L5d's run of this seed says of its own seed
    (`of_this_seed`), or `{"there": False, "why": ...}`."""
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
    of_the_start_model = prepare["the_start_model"]

    # ---- the checks, FIRST: L5d's own for a seed (the start's stored first sampling is `second_full`'s), and this stage's two
    of_l5ds = ladder_l5_draws.the_checks(prepare, trains, trained, averaged, measured, rows, sets, pretraining, start_rows, goal_ids, aside,
                                         [row["problem_id"] for row in groups], settings["half_seed"], base_map, of_the_setting)
    checks = {THE_RUN: {THE_SETS: of_l5ds[THE_RUN][THE_SETS], THE_SETS_ARE_THEIRS: the_sets_are_theirs(sets, stored, recorded),
                        THE_START_ADAPTER: the_start_adapter(of_the_start_model), THE_STARTS_SAMPLING: of_l5ds[THE_RUN][THE_STARTS_SAMPLING]},
              **{name: own for name, own in of_l5ds.items() if name != THE_RUN}}
    failed = [f"{'the run' if name == THE_RUN else f'`{name}`'}: {check}" for name, of in checks.items() for check, entry in of.items() if not entry["passes"]]
    inconclusive = bool(failed)
    not_to_be_read = [name for name in models if not checks[name][LEAN_ANSWERED]["passes"]]

    # ---- the figures; the parts of G are THE FIRST START's own first sampling's (the 200 it solves there, and G'), held to the G' the first step copied
    try:
        parts = the_parts(first_start_rows, again)
    except ValueError:      # the stored sampling is not the one G' was made from: no figure is given on a part
        parts = {ALL: list(goal_ids), SOLVED: [], AGAIN: []}
    share_distinct = lambda own: ((own.get("distinct_attempts") or {}).get(GOAL) or {}).get("mean_share_distinct")      # noqa: E731
    figures = {name: {part: {**own, **({"share_of_distinct_attempts": share_distinct(measured[name])} if part == ALL else {})}
                      for part, own in on_the_parts(rows[name], parts).items()} for name in models}
    of_the_start, of_pre = figure(list(start_rows)), figure(list(first_start_rows))
    the_measurement = {"sampling": sampling["name"], "episodes": attempts, "sampling_seed": sampling["sampling_seed"], "problems": len(goal_ids)}
    counts = the_counts(rows, long_ids)
    at_this_seed = differences(figures)

    # ---- THE READS, in the section's order: the stack; the level; each model against the two start models; overfitting against the start; timeouts; what follows
    of_theirs = of_l5d(l5d, seed, prepare["recipe"]["name"], the_measurement)
    stack = the_stack(failed, {name: figures[name][ALL] for name in models})
    level = the_level({name: figures[name][ALL] for name in models}, of_the_start, of_pre, of_theirs)
    against = {name: {start: against_the_start(rows[name], start_rows, parts, long_ids), first: against_the_start(rows[name], first_start_rows, parts, long_ids)} for name in models}
    shares = {"share_of_distinct_attempts": share_distinct(of_the_start_model)}
    of_timeouts = timed_out({**{name: _health(rows[name]) for name in models}, start: _health(start_rows)}, {THE_SAMPLING: {name: name for name in (*models, start)}})
    of_timeouts.update({"checked_again": False, "why": "no step checks the timed-out attempts again: where a pair is named here, it is to be done by hand (Lean only, no "
                                                       "measurement running) and the reads given both ways"})
    follows = what_follows(stack, prepare["the_seeds"]["more_seeds"])
    of_this_seed = {"seed": seed, "start": start, "recipe": prepare["recipe"]["name"], "ok": not not_to_be_read, "inconclusive": inconclusive, "failed_checks": failed,
                    "the_measurement": the_measurement, "figures": None if inconclusive else figures, "counts": None if inconclusive else counts,
                    "the_stack": None if inconclusive else stack.get("difference")}

    of_the_sets, of_their_sets, of_the_adapter, smoke = checks[THE_RUN][THE_SETS], checks[THE_RUN][THE_SETS_ARE_THEIRS], checks[THE_RUN][THE_START_ADAPTER], prepare.get("smoke")
    own_checks, search, of_l5ds_run = of_the_start_model["l4ps_own_checks"], prepare["the_search"], prepare["l5ds_run"]
    saved, now = (of_the_adapter["recorded_by_l4p"] or {}).get("adapter_saved"), of_the_adapter["read_now"]
    lines = [
        f"{SAY}: L5f, {QUESTION.upper()}. Seed {seed}: L5d's STAGE WITH ONE CHANGE, THE START ADAPTER. Its three sets are L5d's for this seed, by L5's own rule, from the ONE "
        f"search ({search['run']}: {search['deep_solves']:,} deep solves, D = {search['rows_of_each_kind']:,}), held to be the ones L5d's run of this seed "
        f"({of_l5ds_run['run']}) stored and trained. FROM `{start}` ({of_the_start_model['adapter']}: a whole second pass over the published rows from `{first}`) at rank {rank}, "
        "one pass: each set trained in four orders with the recipe as it is; the model measured is the adapter that is the mean of the four. EACH MODEL IS MEASURED on the goal "
        f"set's first sampling alone: {attempts} attempts at each of {len(goal_ids)} goal problems, sampling seed {sampling['sampling_seed']} (the one `{first}`'s run recorded, "
        f"which `{start}` was measured with): no rung, no second sampling, no held-aside set"]
    if smoke:
        lines.append(f"{SAY}: A SMOKE RUN: L5's smoke run " + (f"reads INCONCLUSIVE ({', '.join(smoke['l5s_failed_checks'])}) and was read all the same; "
                                                              if smoke.get("l5_read_though_inconclusive") else "was read; ")
                     + "one deep solve is enough, and orders of a set's handful of rows that are alike are trained all the same. Nothing here is a read of the stack")
    lines += [
        f"{SAY}: CHECK, each set is L5's rule for its seed, made again and held, and no published row carries text: "
        + (f"D = {of_the_sets['rows_of_each_kind']}" if of_the_sets["refused"] is None else f"REFUSED ({of_the_sets['refused']})")
        + f"; {of_the_sets['published_rows_with_text']} published rows with text; sets not in their order seed's order: {of_the_sets['sets_not_in_their_order_seeds_order']}: "
        f"{_verdict(of_the_sets)}",
        f"{SAY}: CHECK, the sets are the ones L5d's run of this seed trained ({of_l5ds_run['run']}), id for id and order for order, each of the four orders too, and that run's "
        f"trainings were of those rows, each once: {of_their_sets['sets']} sets held to its stored ones, not the stored ones: {of_their_sets['sets_that_are_not_the_stored_ones']}; "
        f"{of_their_sets['trainings_of_theirs']} of its trainings' records, not of these rows: {of_their_sets['sets_that_are_not_what_was_trained']}: {_verdict(of_their_sets)}",
        f"{SAY}: CHECK, the adapter of `{start}` is there, is at its rank ({of_the_adapter['rank']}, alpha {of_the_adapter['alpha']}) and is not `{first}`'s: "
        + (f"L4p's run saved it at rank {saved['rank']} and alpha {saved['alpha']}" if saved else "L4p's run recorded no read back of it")
        + (f"; read now at rank {now['read_back']['rank']} and alpha {now['read_back']['alpha']}, {now['against_the_first_start']['elements_changed']:,} of its "
           f"{now['against_the_first_start']['elements']:,} numbers not `{first}`'s" if now else "; its files were not read by the first step (no GPU box)")
        + f": {_verdict(of_the_adapter)}",
        *lines_of_the_checks(prepare, checks, rank, attempts, not_to_be_read),
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
                                              f"({own['per_attempt'][part]['problems']})" for part in (ALL, SOLVED, AGAIN))
                    + f"; goal problems solved {own['solved']} against {own['solved_by_the_start']} (+{own['gained']}, -{own['lost']})")

        highest, beside = level["highest"], level["beside"]
        lines += [
            f"{SAY}: THE FIGURES, successes per 1,000 attempts over the {attempts} attempts, each with its binomial sampling noise: on all of G ({len(parts[ALL])} problems) "
            f"{figure_said(ALL)}; on the goal problems `{first}` solves in its own first sampling ({len(parts[SOLVED])}) {figure_said(SOLVED)}; on G' ({len(parts[AGAIN])}) "
            f"{figure_said(AGAIN)}. The start model `{start}` on all of G: {_per_1000(of_the_start['per_attempt'])}; `{first}`: {_per_1000(of_pre['per_attempt'])}",
            f"{SAY}: THE STACK (the scout's read): {stack['name']}. {stack['reason']}",
            f"{SAY}: THE LEVEL, naming nothing at one seed, per 1,000 attempts on all of G. Each minus `{start}`'s stored first sampling, paired by problem: "
            + ", ".join(f"`{name}` {_signed(level['minus_the_start'][name])}" for name in models)
            + ". Each minus the same kind of L5d's run of this seed (the same rows from "
            + f"`{first}`: what the second pass adds under them): "
            + (", ".join(f"`{name}` {_signed(level['minus_the_same_kind_of_l5d'][name])}" for name in models) if level["l5d_read"] else f"NOT READ ({level['why']})")
            + f". The highest of the three, `{highest['model']}` {_per_1000(highest['per_attempt'])}, beside `{first}` {_per_1000(beside['pre_r64'])}, `{start}` "
            f"{_per_1000(beside[START])}" + (", L5d's " + ", ".join(f"`{name}` {_per_1000(beside['l5d'][name])}" for name in models) if beside["l5d"] else ""),
            f"{SAY}: THE SOLVE RATE AGAINST THE TWO START MODELS, paired by problem (per 1,000 attempts, the model against the start model). Against `{start}`'s stored first "
            f"sampling: " + "; ".join(against_said(name, start) for name in models) + f". Against `{first}`'s: " + "; ".join(against_said(name, first) for name in models),
            f"{SAY}: OVERFITTING, each model against `{start}`: goal problems lost beside gained: "
            + "; ".join(f"`{name}` -{against[name][start]['lost']} beside +{against[name][start]['gained']}" for name in models)
            + ". The share of distinct attempts on G: " + ", ".join(f"`{name}` {figures[name][ALL]['share_of_distinct_attempts']}" for name in models)
            + f", `{start}` {shares['share_of_distinct_attempts']} (as its own run recorded it, over all its attempts on G). The {len(long_ids)} long goal problems (8 lines or "
            "more) solved: " + "; ".join(f"`{name}` {against[name][start]['long']['solved']} against {against[name][start]['long_solved_by_the_start']} "
                                         f"(+{against[name][start]['long']['gained']}, -{against[name][start]['long']['lost']})" for name in models),
            f"{SAY}: TIMEOUTS, each set's share of attempts that timed out in Lean: " + "; ".join(f"`{name}` {of_timeouts['share'][name]}" for name in (*models, start))
            + (f". MORE THAN A POINT APART: {'; '.join(' and '.join(pair['models']) for pair in of_timeouts['apart'])}: the timed-out attempts are to be checked again (Lean "
               "only, no measurement running) and the reads given both ways; no step does it" if of_timeouts["apart"] else ". No two are more than a point apart"),
            f"{SAY}: WHAT FOLLOWS: {follows['name']}. {follows['reason']}"]
    if not_to_be_read:
        lines.append(f"{SAY}: NOT TO BE READ: too many attempts without a verdict from Lean in the sets of {', '.join(f'`{name}`' for name in not_to_be_read)}. The step fails; "
                     "the task queued again measures such a set again from its adapter")

    headline = (f"L5f (the stack; {LABEL}) seed {seed}, from `{start}`: "
                + (f"THIS RUN IS {INCONCLUSIVE} ({'; '.join(failed)})" if inconclusive else
                   "per 1,000 attempts on all of G " + ", ".join(f"`{name}` {_per_1000(figures[name][ALL]['per_attempt'])}" for name in models)
                   + f" (`{start}` {_per_1000(of_the_start['per_attempt'])}, `{first}` {_per_1000(of_pre['per_attempt'])}); THE STACK: {stack['name']}"
                   + (f" ({_signed(stack['difference'])})" if "difference" in stack else ""))
                + f". What follows: {follows['name']}"
                + (f". NOT TO BE READ: too many attempts without a verdict from Lean in the sets of {', '.join(not_to_be_read)}" if not_to_be_read else ""))
    kept = lambda what: {"measured_and_not_read": what} if inconclusive else what      # noqa: E731
    return {
        "spec": SPEC, "stage": L4, "label": LABEL, "check": "L5f", "headline": headline, "lines": lines, "ok": not not_to_be_read, "seed": seed,
        "stand_in_engine": bool(prepare.get("stand_in_engine") or any(entry.get("stand_in_engine") for entry in (*trains.values(), *averaged.values(), *measured.values()))),
        "start": start, "first_start": first, "rank": rank, "models": models, "recipe": prepare["recipe"]["name"],
        "branches": {"the_stack": stack["name"], "what_follows": follows["name"]}, "inconclusive": inconclusive, "failed_checks": failed,
        "can_this_run_be_read": {"what": "the checks the spec reads first; when any fails the run is INCONCLUSIVE and nothing else is said of it", **checks},
        "l4ps_own_checks_of_the_start_model": own_checks,
        "of_this_seed": of_this_seed,
        "figures": {"what": f"each model's successes per attempt over the {attempts} attempts of the first sampling, with its binomial sampling noise, on all of G, on the goal "
                            f"problems `{first}` solves in its own first sampling and on G'", "problems": {part: len(own) for part, own in parts.items()},
                    "the_start_models": {start: of_the_start, first: of_pre}, **kept(figures)},
        "counts": {"what": "the goal problems each model solves at least once in its attempts, gained and lost against `published`; and of the long problems (a shortest "
                           "published proof of 8 lines or more)", "long_problems": len(long_ids), **kept(counts)},
        "the_stack": stack if inconclusive else {"what": f"the scout's read: the mean of `{SHALLOW}` and `{DEEP}`, minus `{PUBLISHED}`, all three from `{start}`, per attempt on "
                                                         "all of G: L5d's primary", **stack},
        "the_primarys_difference_at_this_seed": kept({key: None if value is None else round(value, 6) for key, value in at_this_seed.items()}),
        "the_level": kept(level), "l5d_at_this_seed": of_theirs, "against_the_start_models": kept(against),
        "overfitting": kept({"what": f"for each model against `{start}`: the goal problems lost beside the ones gained, the share of distinct attempts on G, and the long goal "
                                     "problems (8 lines or more) solved, gained and lost. It names no branch of its own", "long_problems": len(long_ids), "the_start_model": shares,
                             **{name: {"lost": against[name][start]["lost"], "gained": against[name][start]["gained"],
                                       "share_of_distinct_attempts": figures[name][ALL]["share_of_distinct_attempts"], "long": against[name][start]["long"],
                                       "long_solved_by_the_start": against[name][start]["long_solved_by_the_start"]} for name in models}}),
        "timed_out_in_lean": of_timeouts, "what_follows": follows,
        "the_seeds": prepare["the_seeds"], "the_start_model": of_the_start_model, "the_search": search, "l5ds_run": of_l5ds_run, "the_sets": prepare["the_sets"],
        "the_orders": prepare["the_orders"], "the_sampling": sampling, "training_sets": prepare["training_sets"], "training_recipe": prepare["recipe"],
        "models_made": {name: {"model": f"`{name}`: trained by nothing: the mean of {', '.join(f'`{of}`' for of in averaged[name].get('of') or [])}",
                               "trainings": {of: {key: value for key, value in trains[of].items() if key != "checkpoints"} for of in prepare["the_averaged"][name]["of"]},
                               "averaging": averaged[name]} for name in models},
        "smoke": smoke, "what_it_cannot_say": list(CANNOT_SAY),
        "attempts": {name: _health(rows[name]) for name in models}, "not_to_be_read": not_to_be_read,
        "not_built": ["the read over three seeds (after an escalation: the primary with L5d's rule and names): each seed's report keeps what it needs under `of_this_seed`",
                      "the check again in Lean of the timed-out attempts where two shares differ by more than a point: the pairs are named",
                      "the measurement in full (the rungs, the second sampling, the held-aside set) of the averaged adapter that reads highest"],
        "not_measured": ["the three rungs, the goal set's second sampling and the held-aside set: this stage reads one sampling of the goal set",
                         f"`{start}`, `{first}` and L5d's models: their rows are the stored ones of their own runs", "anything at another temperature than the config's"]}
