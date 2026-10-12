"""L5s's report: the dose. Every own solve in stock that the model finds less than half the time, against the same
number of published proofs. Spec: docs/spec/ladder-loop.spec.md, "L5s: the dose. Every own solve in stock that the
model finds less than half the time, against the same number of published proofs" ("The reads, fixed before the run",
"What follows, fixed now"). Pure: rows in, a report out, on the rules of `domain/ladder_round/l5_stock.py`.

Everything here is labelled pretrained on published proofs, as L4's reports are: every line begins with L4's label.
One-shot attempts only. A model is ONE sampling of the goal set: no rung, no second sampling, no held-aside set.

  the checks        FIRST, each PASS or FAIL: each set is its rule made again and held, and no published row carries
                    text; then L5d's own for a seed, BY L5d's OWN FUNCTIONS (`reporting/ladder_l5_draws.py`:
                    `checks_of_the_models`, `lines_of_the_checks`): the start model's stored first sampling is the
                    setting's; each trained adapter is not the start adapter's and is read back at its rank; a model was
                    trained on its set and on nothing else, with no barred row and no problem of G or of the held-aside
                    set; an averaged adapter is the mean of its four, none of them, and the one served; Lean answered.
                    ANY FAIL: THE RUN IS INCONCLUSIVE and nothing else is said (what was measured is kept under
                    `measured_and_not_read`)
  the reads         in the section's order: THE DOSE (`l5_stock.the_dose`, against `deep` of L5d's run of this seed, NOT
                    READ and said when that run is not there); its own proofs at this dose and the steps without its own
                    proofs (`l5_stock.beside`); each model against the start model's stored first sampling, paired by
                    problem (`l5_stock.against_the_start`); the overfitting reads; the timeouts; what follows

A report in which Lean did not answer a set is NOT TO BE READ (`ok` is false, and the step fails): the task queued again
samples such a set again from its adapter.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE
from rlvr_lean.domain.ladder_round.l5 import DEEP, PUBLISHED, timed_out
from rlvr_lean.domain.ladder_round.l5_draws import AGAIN, ALL, SOLVED, on_the_parts, the_parts, the_start_is_the_settings
from rlvr_lean.domain.ladder_round.l5_stock import MODELS, PUBLISHED_STOCK, STOCK, against_the_start, beside, of_l5d, the_dose, the_sets_are_the_rule, what_follows
from rlvr_lean.domain.ladder_round.read import GOAL, group_ids
from rlvr_lean.reporting.ladder_ceiling import goal_sets
from rlvr_lean.reporting.ladder_l2 import _health
from rlvr_lean.reporting.ladder_l3c import LENGTH_GROUPS
from rlvr_lean.reporting.ladder_l4 import LABEL, SAY, _verdict
from rlvr_lean.reporting.ladder_l5_draws import PART_SAID, THE_RUN, THE_SAMPLING, THE_START, _per_1000, checks_of_the_models, lines_of_the_checks
from rlvr_lean.reporting.ladder_l5_noise import LEAN_ANSWERED

L4 = "l4"
QUESTION = "the dose: every own solve in stock that the model finds less than half the time, against the same number of published proofs"
SPEC = "docs/spec/ladder-loop.spec.md, L5s: the dose. Every own solve in stock that the model finds less than half the time, against the same number of published proofs"
LONG = LENGTH_GROUPS[-1]                    # the goal problems whose shortest published proof has 8 lines or more: the "long problems" of the overfitting reads
THE_SETS = "each_set_is_its_rule_made_again_and_held"
CANNOT_SAY = [
    "ONE SEED: a scout, by the repo's seed rule. It names an escalation or none, and no effect",
    "THE BAR IS A SECOND PASS over proofs the start model has trained on; the model's own proofs are of problems it never trained on. A difference says its own proofs of new "
    "problems are worth more or less than reading published proofs again, not than PUBLISHED proofs of new problems",
    "The dose is read against a model of another size of set (L5d's `deep` at this seed): the two differ by their steps too, which `published_stock` minus that seed's "
    "`published` stands beside",
    "G alone and one sampling of it, 32 attempts a problem: nothing of the rungs, of the second sampling or of the held-aside set"]


def the_checks(prepare: Mapping, trains: Mapping[str, Mapping], trained: Mapping[str, Sequence[Mapping]], averaged: Mapping[str, Mapping], measured: Mapping[str, Mapping],
               rows: Mapping[str, Sequence[Mapping]], sets: Mapping[str, Sequence[Mapping]], pretraining: Sequence[Mapping], start_rows: Sequence[Mapping],
               goal_ids: Sequence[str], aside: Sequence[str], heldout: Sequence[str], half_seed: int, base_map: Sequence[str], of_the_setting: Mapping,
               deep_solves: Sequence[str], ordinary_solves: Sequence[str]) -> dict[str, dict]:
    """The checks the spec reads FIRST: the run's two (`THE_RUN`: the sets, by this stage's own rule; the start model's
    stored first sampling, L5d's check), then L5d's own of each model (`checks_of_the_models`)."""
    in_orders = {name: sets[name] for name in prepare["trained"]}
    return {THE_RUN: {
        THE_SETS: the_sets_are_the_rule({model: sets[model] for model in MODELS}, pretraining, prepare["seed"], deep_solves, ordinary_solves, in_orders),
        THE_START: the_start_is_the_settings(start_rows, goal_ids, prepare["the_sampling"], of_the_setting["sampling"], of_the_setting["attempts"])},
        **checks_of_the_models(prepare, trains, trained, averaged, measured, rows, sets, goal_ids, aside, heldout, half_seed, base_map)}


def _signed(value: float | None) -> str:
    return "not read" if value is None else _per_1000(value, True)


def build_l5_stock_report(prepare: Mapping, trains: Mapping[str, Mapping], trained: Mapping[str, Sequence[Mapping]], averaged: Mapping[str, Mapping],
                          measured: Mapping[str, Mapping], rows: Mapping[str, Sequence[Mapping]], sets: Mapping[str, Sequence[Mapping]], pretraining: Sequence[Mapping],
                          start_rows: Sequence[Mapping], groups: Sequence[Mapping], lengths: Mapping[str, Mapping], again: Sequence[str], aside: Sequence[str],
                          settings: Mapping, base_map: Sequence[str], of_the_setting: Mapping, deep_solves: Sequence[str], ordinary_solves: Sequence[str],
                          l5d: Mapping) -> dict:
    """`prepare`: the prepare step's summary. `trains`, `trained`, `averaged`, `measured`, `rows`, `sets`,
    `pretraining`, `start_rows`, `groups`, `lengths`, `again`, `aside`, `settings`, `base_map`, `of_the_setting`: as
    L5d's report takes them (`reporting/ladder_l5_draws.build_l5_draws_report`). `deep_solves`, `ordinary_solves`: THE
    RULE MADE AGAIN by the report step, the problems the search solved and the ordinary solves L5's setting gives.
    `l5d`: what the report of L5d's run of this seed says of its own seed (`of_this_seed`), or `{"there": False,
    "why": ...}` when that run is not on the box and finished."""
    start, seed, models, sampling = prepare["start"], prepare["seed"], list(prepare["models"]), prepare["the_sampling"]
    if models != list(MODELS) or sorted(rows) != sorted(models) or sorted(trains) != sorted(prepare["trained"]) or sorted(averaged) != sorted(prepare["averaged"]):
        raise ValueError(f"the report is given the rows of {sorted(rows)}, the trainings of {sorted(trains)} and the averages of {sorted(averaged)}: it reads this run's models "
                         f"({list(MODELS)}), each measured")
    goal_ids = group_ids(groups, GOAL)
    if any(sorted(row["problem_id"] for row in own) != sorted(goal_ids) for own in rows.values()):
        raise ValueError("the models were not measured on the goal problems, each once: every model here is one sampling of the one goal set")
    rank, attempts = (prepare.get("start_recipe") or {}).get("rank"), sampling["episodes"]
    long_ids = goal_sets(goal_ids, lengths).get(LONG, [])

    # ---- the checks, FIRST
    checks = the_checks(prepare, trains, trained, averaged, measured, rows, sets, pretraining, start_rows, goal_ids, aside, [row["problem_id"] for row in groups],
                        settings["half_seed"], base_map, of_the_setting, deep_solves, ordinary_solves)
    failed = [f"{'the run' if name == THE_RUN else f'`{name}`'}: {check}" for name, of in checks.items() for check, entry in of.items() if not entry["passes"]]
    inconclusive = bool(failed)
    not_to_be_read = [name for name in models if not checks[name][LEAN_ANSWERED]["passes"]]

    # ---- the figures; the parts of G are the start model's own first sampling's, held to the G' the first step copied
    try:
        parts = the_parts(start_rows, again)
    except ValueError:      # the stored sampling is not the one G' was made from: no figure is given on a part
        parts = {ALL: list(goal_ids), SOLVED: [], AGAIN: []}
    share_distinct = lambda own: ((own.get("distinct_attempts") or {}).get(GOAL) or {}).get("mean_share_distinct")      # noqa: E731
    figures = {name: {part: {**own, **({"share_of_distinct_attempts": share_distinct(measured[name])} if part == ALL else {})}
                      for part, own in on_the_parts(rows[name], parts).items()} for name in models}
    the_measurement = {"sampling": sampling["name"], "episodes": attempts, "sampling_seed": sampling["sampling_seed"], "problems": len(goal_ids)}

    # ---- THE READS, in the section's order: the dose against L5d's run of this seed; two differences beside it; each model against the start model; overfitting; timeouts
    of_theirs = of_l5d(l5d, seed, prepare["recipe"]["name"], the_measurement)
    dose = the_dose(failed, figures[STOCK][ALL], of_theirs)
    of_beside = beside({name: figures[name][ALL] for name in models}, of_theirs)
    against = {name: against_the_start(rows[name], start_rows, parts, long_ids) for name in models}
    of_the_start = {"share_of_distinct_attempts": share_distinct(prepare.get("the_start_wrote") or {})}
    of_timeouts = timed_out({name: _health(rows[name]) for name in models}, {THE_SAMPLING: {name: name for name in models}})
    of_timeouts.update({"checked_again": False, "why": "no step checks the timed-out attempts again: where the pair is named here, it is to be done by hand (Lean only, no "
                                                       "measurement running) and the reads given both ways"})
    follows = what_follows(dose, prepare["the_seeds"]["more_seeds"])
    of_this_seed = {"seed": seed, "recipe": prepare["recipe"]["name"], "ok": not not_to_be_read, "inconclusive": inconclusive, "failed_checks": failed,
                    "the_measurement": the_measurement, "figures": None if inconclusive else figures,
                    "the_dose": None if inconclusive else dose.get("difference"),
                    "stock_minus_published_stock": None if inconclusive else of_beside["own_proofs_at_this_dose"]["difference"]}

    of_the_sets, smoke, stock = checks[THE_RUN][THE_SETS], prepare.get("smoke"), prepare["the_stock"]
    lines = [
        f"{SAY}: L5s, THE DOSE. Seed {seed}: `{STOCK}` is EVERY deep solve of the ONE search ({prepare['the_search']['run']}: {stock['deep_solves']:,}), EVERY ordinary solve "
        f"({stock['ordinary_solves']:,}: base-map problems solved in {stock['solved_at_least']} to {stock['solved_at_most']} of the map's attempts) and the first "
        f"{stock['own_rows']:,} rows of the seed's order of the pretraining file: {stock['rows'][STOCK]:,} rows; `{PUBLISHED_STOCK}` the first {stock['rows'][PUBLISHED_STOCK]:,} "
        f"rows of that order and nothing else. From `{start}` at rank {rank}, one pass: each set trained in four orders with the recipe as it is; the model measured is the "
        f"adapter that is the mean of the four. EACH MODEL IS MEASURED on the goal set's first sampling alone: {attempts} attempts at each of {len(goal_ids)} goal problems, "
        f"sampling seed {sampling['sampling_seed']} (the one `{start}`'s run recorded): no rung, no second sampling, no held-aside set"]
    if smoke:
        lines.append(f"{SAY}: A SMOKE RUN: L5's smoke run " + (f"reads INCONCLUSIVE ({', '.join(smoke['l5s_failed_checks'])}) and was read all the same; "
                                                              if smoke.get("l5_read_though_inconclusive") else "was read; ")
                     + f"one deep solve is enough; {smoke['ordinary_solves_taken_by_the_runs_minimum']} ordinary solves were taken by the run's minimum and not by the rule; "
                     "and orders of a set's handful of rows that are alike are trained all the same. Nothing here is a read of the dose")
    lines += [
        f"{SAY}: CHECK, each set is its rule made again and held (`{STOCK}`: every deep solve and every ordinary solve the rule gives, no other own row, and the first N "
        f"published rows of the seed's order; `{PUBLISHED_STOCK}`: the first 2N), and no published row carries text: "
        + (f"N = {of_the_sets['counted']['own_rows']:,} ({of_the_sets['counted']['deep_solves']:,} deep, {of_the_sets['counted']['ordinary_solves']:,} ordinary)"
           if of_the_sets["refused"] is None else f"REFUSED ({of_the_sets['refused']})")
        + f"; {of_the_sets['published_rows_with_text']} published rows with text; sets not in their order seed's order: {of_the_sets['sets_not_in_their_order_seeds_order']}: "
        f"{_verdict(of_the_sets)}", *lines_of_the_checks(prepare, checks, rank, attempts, not_to_be_read)]
    if inconclusive:
        lines.append(f"{SAY}: THIS RUN IS {INCONCLUSIVE}: {'; '.join(failed)}. Nothing else is said of it")
    else:
        def against_said(name: str) -> str:
            own = against[name]
            return (f"`{name}` " + ", ".join(f"{_per_1000(own['per_attempt'][part]['model'])} against {_per_1000(own['per_attempt'][part]['start'])} "
                                              f"({_signed(own['per_attempt'][part]['difference'])}) on {PART_SAID[part]} ({own['per_attempt'][part]['problems']})"
                                              for part in (ALL, SOLVED, AGAIN))
                    + f"; goal problems solved {own['solved']} against {own['solved_by_the_start']} (+{own['gained']}, -{own['lost']})")

        steps, own_proofs = of_beside["the_steps_without_its_own_proofs"], of_beside["own_proofs_at_this_dose"]
        lines += [
            f"{SAY}: THE FIGURES, successes per 1,000 attempts on all of G over the {attempts} attempts, each with its binomial sampling noise: "
            + ", ".join(f"`{name}` {_per_1000(figures[name][ALL]['per_attempt'])} ({_per_1000(figures[name][ALL]['sampling_noise'])})" for name in models)
            + ("; L5d's models at this seed: " + ", ".join(f"`{name}` {_per_1000(own['per_attempt'])}" for name, own in of_theirs["figures"].items()) if of_theirs["read"]
               else f"; L5d's run of this seed is NOT READ ({of_theirs['why']})"),
            f"{SAY}: THE DOSE (the scout's read): {dose['name']}. {dose['reason']}",
            f"{SAY}: ITS OWN PROOFS AT THIS DOSE ({stock['own_rows']:,} own rows against the {prepare['the_search']['rows_of_each_kind']:,} of L5d's sets), naming nothing: "
            f"`{STOCK}` minus `{PUBLISHED_STOCK}` {_signed(own_proofs['difference'])} per 1,000 attempts; L5d's primary at this seed "
            f"{_signed(own_proofs['l5ds_primary_at_this_seed'])}. One seed names nothing; three do, with L5d's rule and names",
            f"{SAY}: THE STEPS WITHOUT ITS OWN PROOFS, naming nothing: `{PUBLISHED_STOCK}` minus `{PUBLISHED}` of L5d's run of this seed "
            + (f"{_signed(steps['difference'])} per 1,000 attempts ({stock['rows'][PUBLISHED_STOCK]:,} rows against {2 * prepare['the_search']['rows_of_each_kind']:,} of the "
               f"same order). `{STOCK}` differs from that seed's models by its steps too: this is the same change of steps with no own proof" if steps["read"]
               else f"is NOT READ ({steps['why']})"),
            f"{SAY}: THE SOLVE RATE AGAINST THE START MODEL, each against `{start}`'s stored first sampling, paired by problem (per 1,000 attempts, the model against the "
            f"start model): " + "; ".join(against_said(name) for name in models),
            f"{SAY}: OVERFITTING, each model against `{start}`: goal problems lost beside gained: "
            + "; ".join(f"`{name}` -{against[name]['lost']} beside +{against[name]['gained']}" for name in models)
            + ". The share of distinct attempts on G: " + ", ".join(f"`{name}` {figures[name][ALL]['share_of_distinct_attempts']}" for name in models)
            + f", `{start}` {of_the_start['share_of_distinct_attempts']} (as its own run recorded it). The {len(long_ids)} long goal problems (8 lines or more) solved: "
            + "; ".join(f"`{name}` {against[name]['long']['solved']} against {against[name]['long_solved_by_the_start']} (+{against[name]['long']['gained']}, "
                        f"-{against[name]['long']['lost']})" for name in models),
            f"{SAY}: TIMEOUTS, each set's share of attempts that timed out in Lean: " + "; ".join(f"`{name}` {of_timeouts['share'][name]}" for name in models)
            + (". MORE THAN A POINT APART: the timed-out attempts are to be checked again (Lean only, no measurement running) and the reads given both ways; no step does it"
               if of_timeouts["apart"] else ". The two are not more than a point apart"),
            f"{SAY}: WHAT FOLLOWS: {follows['name']}. {follows['reason']}"]
    if not_to_be_read:
        lines.append(f"{SAY}: NOT TO BE READ: too many attempts without a verdict from Lean in the sets of {', '.join(f'`{name}`' for name in not_to_be_read)}. The step fails; "
                     "the task queued again measures such a set again from its adapter")

    headline = (f"L5s (the dose; {LABEL}) seed {seed}, from `{start}`: "
                + (f"THIS RUN IS {INCONCLUSIVE} ({'; '.join(failed)})" if inconclusive else
                   "per 1,000 attempts on all of G " + ", ".join(f"`{name}` {_per_1000(figures[name][ALL]['per_attempt'])}" for name in models)
                   + (f" (L5d's `{DEEP}` at this seed {_per_1000(of_theirs['figures'][DEEP]['per_attempt'])})" if of_theirs["read"] else "")
                   + f"; THE DOSE: {dose['name']}" + (f" ({_signed(dose['difference'])})" if "difference" in dose else ""))
                + f". What follows: {follows['name']}"
                + (f". NOT TO BE READ: too many attempts without a verdict from Lean in the sets of {', '.join(not_to_be_read)}" if not_to_be_read else ""))
    kept = lambda what: {"measured_and_not_read": what} if inconclusive else what      # noqa: E731
    return {
        "spec": SPEC, "stage": L4, "label": LABEL, "check": "L5s", "headline": headline, "lines": lines, "ok": not not_to_be_read, "seed": seed,
        "stand_in_engine": bool(prepare.get("stand_in_engine") or any(entry.get("stand_in_engine") for entry in (*trains.values(), *averaged.values(), *measured.values()))),
        "start": start, "rank": rank, "models": models, "recipe": prepare["recipe"]["name"],
        "branches": {"the_dose": dose["name"], "what_follows": follows["name"]}, "inconclusive": inconclusive, "failed_checks": failed,
        "can_this_run_be_read": {"what": "the checks the spec reads first; when any fails the run is INCONCLUSIVE and nothing else is said of it", **checks},
        "of_this_seed": of_this_seed,
        "figures": {"what": f"each model's successes per attempt over the {attempts} attempts of the first sampling, with its binomial sampling noise, on all of G, on the goal "
                            "problems the start model solves in its own first sampling and on G'", "problems": {part: len(own) for part, own in parts.items()}, **kept(figures)},
        "the_dose": dose if inconclusive else {"what": f"the scout's read: `{STOCK}` minus `{DEEP}` of L5d's run of this seed, per attempt on all of G", **dose},
        "l5d_at_this_seed": of_theirs, "beside_the_dose": kept(of_beside), "against_the_start_model": kept(against),
        "overfitting": kept({"what": "for each model against the start model: the goal problems lost beside the ones gained, the share of distinct attempts on G, and the long "
                                     "goal problems (8 lines or more) solved, gained and lost. It names no branch of its own", "long_problems": len(long_ids),
                             "the_start_model": of_the_start,
                             **{name: {"lost": against[name]["lost"], "gained": against[name]["gained"], "share_of_distinct_attempts": figures[name][ALL]["share_of_distinct_attempts"],
                                       "long": against[name]["long"], "long_solved_by_the_start": against[name]["long_solved_by_the_start"]} for name in models}}),
        "timed_out_in_lean": of_timeouts, "what_follows": follows,
        "the_seeds": prepare["the_seeds"], "the_search": prepare["the_search"], "the_stock": stock, "the_sets": prepare["the_sets"], "the_orders": prepare["the_orders"],
        "the_sampling": sampling, "training_sets": prepare["training_sets"], "training_recipe": prepare["recipe"],
        "models_made": {name: {"model": f"`{name}`: trained by nothing: the mean of {', '.join(f'`{of}`' for of in averaged[name].get('of') or [])}",
                               "trainings": {of: {key: value for key, value in trains[of].items() if key != "checkpoints"} for of in prepare["the_averaged"][name]["of"]},
                               "averaging": averaged[name]} for name in models},
        "smoke": smoke, "what_it_cannot_say": list(CANNOT_SAY),
        "attempts": {name: _health(rows[name]) for name in models}, "not_to_be_read": not_to_be_read,
        "not_built": ["the read over three seeds (after an escalation: `stock` minus `published_stock` with L5d's primary rule, and the dose with an interval over seeds): each "
                      "seed's report keeps what it needs under `of_this_seed`",
                      "the check again in Lean of the timed-out attempts where the two models' shares differ by more than a point: the pair is named"],
        "not_measured": ["the three rungs, the goal set's second sampling and the held-aside set: this stage reads one sampling of the goal set",
                         f"`{start}`, and L5d's models: their rows are the stored ones of their own runs", "anything at another temperature than the config's"]}
