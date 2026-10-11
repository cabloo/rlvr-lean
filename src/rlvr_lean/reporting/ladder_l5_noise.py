"""L5n's report: what is one training worth? The spread between trainings, and a recipe that lowers it. Spec:
docs/spec/ladder-loop.spec.md, "L5n: what is one training worth? The spread between trainings, and a recipe that
lowers it" ("The reads, fixed before the run", "Checks, read first", "What follows, fixed now"). Pure: rows in, a
report out, on the rules of `domain/ladder_round/l5_noise.py`.

Everything here is labelled pretrained on published proofs, as L4's reports are: every line begins with L4's label.
One-shot attempts only. A model is ONE sampling of the goal set: no rung, no second sampling, no held-aside set.

  the checks        FIRST, each PASS or FAIL: each trained adapter is not the start adapter's and is read back at its
                    rank; no row a model was trained on is anything but one of the fixed rows; each averaged adapter's
                    numbers ARE the mean of its models' (read back), it is none of them, and it is served; Lean answered
                    on each set. ANY FAIL: the run is INCONCLUSIVE, and nothing else is said (what was measured is kept
                    under `measured_and_not_read`)
  the figures       each model's successes per attempt on all of G over the first sampling's attempts, with its binomial
                    sampling noise (`l5_noise.figure`)
  the reads         THE SPREAD OF THE RECIPE AS IT IS; DOES DECAY LOWER IT; AND ITS LEVEL; AVERAGING, for each averaged
                    model; WHAT FOLLOWS, in one line
  beside            naming nothing: each model's goal problems solved and their spread, its share of distinct attempts,
                    its share of timeouts; and L5's own `published` of the search's seed (the same rows in that seed's
                    order, the recipe as it is) from its stored first sampling: one more draw of `constant` that is in
                    no standard deviation

A report in which Lean did not answer a set is NOT TO BE READ (`ok` is false, and the step fails): the task queued again
samples such a set again from its adapter.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, the_adapter_differs
from rlvr_lean.domain.ladder_round.l5_noise import (
    CONSTANT,
    DECAY,
    averaging,
    decay_lowers,
    figure,
    its_level,
    mean_of,
    model_of,
    only_the_fixed_rows,
    per_attempt,
    spread,
    spread_as_it_is,
    what_follows,
)
from rlvr_lean.domain.ladder_round.read import GOAL
from rlvr_lean.reporting.ladder_ceiling import lean_did_not_answer
from rlvr_lean.reporting.ladder_l2 import MAXIMUM_SHARE_WITHOUT_AN_ANSWER, _health
from rlvr_lean.reporting.ladder_l4 import LABEL, SAY, _verdict, adapter_against_the_start

L4 = "l4"
QUESTION = "what is one training worth? The spread between trainings, and a recipe that lowers it"
SPEC = "docs/spec/ladder-loop.spec.md, L5n: what is one training worth? The spread between trainings, and a recipe that lowers it"
IS = {CONSTANT: "the recipe as it is: five warm-up steps, then a constant learning rate",
      DECAY: "ONE change: after the warm-up the learning rate falls linearly to zero at the end of the pass"}
THE_ADAPTER = "its_adapter_is_not_the_start_adapters_and_is_read_back_at_its_rank"      # the checks, as a failed check names them
THE_ROWS = "no_row_is_anything_but_one_of_the_fixed_rows"
THE_MEAN = "its_numbers_are_the_mean_of_its_models_and_it_is_served"
LEAN_ANSWERED = "lean_answered"
CANNOT_SAY = [
    "One set of rows, published rows only: the spread of a training on a model's own proofs is not measured",
    "A handful of orders: a standard deviation of four figures is itself known to within about a factor of two",
    "G alone, and one sampling of it: nothing of the rungs, of the second sampling or of the held-aside set",
    "It measures the recipe's noise from this start model, not from another start",
    "The averaged adapter is the adapter of the mean model only to first order: its models start from one adapter and move little"]


def _per_1000(value: float | None) -> str:
    return "not measured" if value is None else f"{1000 * value:.2f}"


def _at_its_rank(saved: Mapping | None, rank: int | None, alpha: int | None) -> bool:
    """Whether an adapter read back from its two files has the start adapter's rank and alpha, in its config and in every matrix."""
    return bool(saved) and (saved.get("rank"), saved.get("alpha"), saved.get("ranks")) == (rank, alpha, [rank])


def the_checks(prepare: Mapping, trains: Mapping[str, Mapping], trained: Mapping[str, Sequence[Mapping]], averaged: Mapping[str, Mapping], measured: Mapping[str, Mapping],
               rows: Mapping[str, Sequence[Mapping]], fixed: Sequence[Mapping]) -> dict[str, dict]:
    """The checks the spec reads FIRST, by model: a trained model's adapter and its rows; an averaged model's numbers,
    and that it was served; and for each, whether Lean answered its set."""
    recipe = prepare.get("start_recipe") or {}
    rank, alpha = recipe.get("rank"), recipe.get("alpha")
    checks = {}
    for name in prepare["trained"]:
        differs, saved = the_adapter_differs(adapter_against_the_start(trains[name])), trains[name].get("adapter_saved")
        at_rank = _at_its_rank(saved, rank, alpha)
        checks[name] = {THE_ADAPTER: {**differs, "read_back": saved, "rank_of_the_start": rank, "alpha_of_the_start": alpha, "at_its_rank": at_rank,
                                      "passes": differs["passes"] and at_rank},
                        THE_ROWS: only_the_fixed_rows(trained[name], fixed)}
    for name in prepare["averaged"]:
        own, of_the_set = averaged[name], measured[name]
        held, saved = own.get("read_back") or {}, own.get("adapter_saved")
        at_rank = _at_its_rank(saved, rank, alpha)
        served = bool(own.get("adapter")) and of_the_set.get("adapter") == own.get("adapter") and bool(rows[name]) and all(row["episodes"] > 0 for row in rows[name])
        # A SMOKE RUN ALONE: its models may be one adapter (one optimizer step on the same rows), and their mean then is each of them: kept all the same, and said.
        one_of_them = bool(prepare.get("smoke")) and bool(held.get("is_the_mean")) and not held.get("is_none_of_them")
        checks[name] = {THE_MEAN: {"what": "read back from its file, every number is the mean of its models' numbers at that place and it is none of them; it has the start "
                                           "adapter's rank and alpha; and the model server took it: its set was sampled from it",
                                   "of": own.get("of"), "elements": held.get("elements"), "elements_not_the_mean": held.get("elements_not_the_mean"),
                                   "elements_not_those_of_each_model": held.get("elements_not_those_of_each_adapter"), "is_the_mean": bool(held.get("is_the_mean")),
                                   "is_none_of_them": bool(held.get("is_none_of_them")), "one_of_its_models_in_a_smoke_run": one_of_them, "read_back": saved,
                                   "at_its_rank": at_rank, "served": served, "passes": (bool(held.get("passes")) or one_of_them) and at_rank and served}}
    for name in prepare["models"]:
        health = _health(rows[name])
        checks[name][LEAN_ANSWERED] = {"what": f"Lean gave a verdict on all but at most {MAXIMUM_SHARE_WITHOUT_AN_ANSWER} of the attempts of its set; the task queued again "
                                               "samples such a set again", "maximum": MAXIMUM_SHARE_WITHOUT_AN_ANSWER, "share_without_an_answer": health["share_without_an_answer"],
                                       "passes": not lean_did_not_answer(rows[name])}
    return checks


def _adapter_said(name: str, entry: Mapping) -> str:
    changed = (f"{entry['elements_changed']:,} of the {entry['elements']:,} numbers of its adapter are not the start adapter's" if entry["compared"]
               else "its adapter was not compared with the start adapter's")
    saved = entry["read_back"]
    return f"`{name}`: {changed}; " + (f"read back at rank {saved['rank']} and alpha {saved['alpha']}, matrices of rank {saved['ranks']}" if saved else "not read back")


def _mean_said(name: str, entry: Mapping) -> str:
    if entry["elements"] is None:
        return f"`{name}`: no adapter was averaged; " + ("its set was sampled" if entry["served"] else "it was not served")
    return (f"`{name}` (the mean of {', '.join(f'`{model}`' for model in entry['of'])}): {entry['elements_not_the_mean']:,} of its {entry['elements']:,} numbers are not the "
            f"mean; the numbers that are not each model's: {', '.join(f'{count:,}' for count in entry['elements_not_those_of_each_model'])}"
            + (" (IT IS ONE OF ITS MODELS: a smoke run's trainings of one optimizer step on the same rows are one training; kept all the same)"
               if entry["one_of_its_models_in_a_smoke_run"] else "") + "; "
            + ("at the start adapter's rank; " if entry["at_its_rank"] else "NOT at the start adapter's rank; ") + ("its set was sampled from it" if entry["served"] else "it was NOT served"))


def build_l5_noise_report(prepare: Mapping, trains: Mapping[str, Mapping], trained: Mapping[str, Sequence[Mapping]], averaged: Mapping[str, Mapping],
                          measured: Mapping[str, Mapping], rows: Mapping[str, Sequence[Mapping]], fixed: Sequence[Mapping], published: Mapping[int, Mapping]) -> dict:
    """`prepare`: the prepare step's summary (`start`; `orders`, `recipes`; `trained`, `averaged`, `models`: this run's,
    in the order they are made; `the_rows`; `the_sampling`; `start_recipe`; `the_published_of_l5`). `trains`: by trained
    model, its training step's summary. `trained`: by trained model, the rows its training loop recorded (id, problem,
    place in the file). `averaged`: by averaged model, its averaging step's summary. `measured`: by model, its
    measuring step's summary. `rows`: by model, its per-problem rows on the goal set's first sampling. `fixed`: the
    fixed rows. `published`: L5's `published` by seed of L5, each `there` (with its per-problem `rows` on the same
    sampling) or not (with `why`): the seed that trained the fixed rows is the one more draw of `constant`."""
    start, orders, recipes, models = prepare["start"], list(prepare["orders"]), list(prepare["recipes"]), list(prepare["models"])
    of_the_rows, sampling, own_seed = prepare["the_rows"], prepare["the_sampling"], prepare["the_published_of_l5"]["seed"]
    if sorted(rows) != sorted(models) or sorted(trains) != sorted(prepare["trained"]) or sorted(averaged) != sorted(prepare["averaged"]):
        raise ValueError(f"the report is given the rows of {sorted(rows)}, the trainings of {sorted(trains)} and the averages of {sorted(averaged)}: it reads this run's models "
                         f"({models}), each measured")
    ids = {name: sorted(row["problem_id"] for row in own) for name, own in rows.items()}
    if any(own != ids[models[0]] or len(set(own)) != len(own) for own in ids.values()):
        raise ValueError("the models were not measured on the same problems: every model here is one sampling of the one goal set")
    rank = (prepare.get("start_recipe") or {}).get("rank")
    attempts, problems = sampling["episodes"], len(ids[models[0]])

    # ---- the checks, FIRST
    checks = the_checks(prepare, trains, trained, averaged, measured, rows, fixed)
    failed = [f"`{name}`: {check}" for name in models for check, entry in checks[name].items() if not entry["passes"]]
    inconclusive = bool(failed)
    not_to_be_read = [name for name in models if not checks[name][LEAN_ANSWERED]["passes"]]

    # ---- the figures, and the reads
    figures = {name: {**figure(rows[name]), "share_of_distinct_attempts": ((measured[name].get("distinct_attempts") or {}).get(GOAL) or {}).get("mean_share_distinct")}
               for name in models}
    rate = {name: per_attempt(rows[name]) for name in models}
    of = {recipe: [rate[model_of(recipe, order)] for order in orders] for recipe in recipes}
    there = {seed: entry for seed, entry in published.items() if entry.get("there")}
    over_the_seeds = [per_attempt(published[seed]["rows"]) for seed in published] if published and len(there) == len(published) else None
    missing = "; ".join(f"seed {seed}: {entry.get('why')}" for seed, entry in published.items() if not entry.get("there"))
    as_it_is = spread_as_it_is(failed, of[CONSTANT], [figures[model_of(CONSTANT, order)]["sampling_noise"] for order in orders], over_the_seeds, missing)
    lowers = decay_lowers(failed, of[CONSTANT], of[DECAY])
    level = its_level(failed, list(zip(of[DECAY], of[CONSTANT])))
    of_the_average = {recipe: averaging(failed, rate[mean_of(recipe)], figures[mean_of(recipe)]["sampling_noise"], of[recipe], recipe) for recipe in recipes}
    follows = what_follows(lowers, level, of_the_average[CONSTANT])
    solved = {recipe: [figures[model_of(recipe, order)]["goal_problems_solved"] for order in orders] for recipe in recipes}
    beside = {"goal_problems_solved": {recipe: {"by_order": dict(zip(map(str, orders), solved[recipe])), "standard_deviation": None if spread(solved[recipe]) is None
                                               else round(spread(solved[recipe]), 3)} for recipe in recipes},
              "the_published_of_l5": {str(seed): ({"there": True, "run": entry.get("run"), "inconclusive": entry.get("inconclusive"), **figure(entry["rows"]),
                                                  "share_of_distinct_attempts": entry.get("share_of_distinct_attempts")} if entry.get("there")
                                                 else {"there": False, "why": entry.get("why")}) for seed, entry in published.items()},
              "one_more_draw_of_constant": str(own_seed)}
    fifth = beside["the_published_of_l5"].get(str(own_seed)) or {"there": False, "why": "it was not given"}

    def said(name: str) -> str:
        own = figures[name]
        return f"`{name}` {_per_1000(rate[name])} ({_per_1000(own['sampling_noise'])})"

    def beside_said(name: str) -> str:
        own = figures[name]
        return f"`{name}` {own['goal_problems_solved']}, {own['share_of_distinct_attempts']}, {own['share_timed_out_in_lean']}"

    smoke = prepare.get("smoke")
    lines = [
        f"{SAY}: L5n, {QUESTION.upper()}. Seed {prepare['seed']}. ONE FIXED SET OF ROWS: the {of_the_rows['rows']:,} published rows L5's own `{of_the_rows['model']}` of seed "
        f"{own_seed} was trained on ({of_the_rows['run']}), by id and place. From `{start}` at rank {rank}, one pass each: the rows in the content-hash order of the order seeds "
        f"{orders}, each order trained as `{CONSTANT}` ({IS[CONSTANT]}) and as `{DECAY}` ({IS[DECAY]}); then, trained by nothing, "
        + " and ".join(f"`{mean_of(recipe)}`" for recipe in recipes) + f": the adapter whose every number is the mean of its {len(orders)} adapters'. EACH MODEL IS MEASURED on "
        f"the goal set's first sampling alone: {attempts} attempts at each of {problems} goal problems, sampling seed {sampling['sampling_seed']} (the one `{start}`'s run "
        "recorded): no rung, no second sampling, no held-aside set"]
    if smoke:
        lines.append(f"{SAY}: A SMOKE RUN: L5's smoke run " + (f"reads INCONCLUSIVE ({', '.join(smoke['l5s_failed_checks'])}) and was read all the same; "
                                                              if smoke.get("l5_read_though_inconclusive") else "was read; ")
                     + f"{len(orders)} of the setting's {len(smoke['orders_of_the_setting'])} orders are trained; the orders that are the one L5 trained in ({smoke['the_order_trained_before']}) "
                     f"or alike ({smoke['alike']}) are trained all the same. Nothing here is a read of the recipe")
    lines += [
        f"{SAY}: CHECK, each trained adapter is not the start adapter's and is read back at its rank ({rank}): "
        + "; ".join(_adapter_said(name, checks[name][THE_ADAPTER]) for name in prepare["trained"])
        + f": {_verdict({'passes': all(checks[name][THE_ADAPTER]['passes'] for name in prepare['trained'])})}",
        f"{SAY}: CHECK, no row a model was trained on is anything but one of the {of_the_rows['rows']:,} fixed rows (the training loop's own record; each of them once): "
        + "; ".join(f"`{name}` {checks[name][THE_ROWS]['rows']:,} rows, {checks[name][THE_ROWS]['rows_that_are_none_of_them']} none of them, {checks[name][THE_ROWS]['rows_twice']} "
                    f"twice, {checks[name][THE_ROWS]['fixed_rows_not_trained_on']} not trained on" for name in prepare["trained"])
        + f": {_verdict({'passes': all(checks[name][THE_ROWS]['passes'] for name in prepare['trained'])})}",
        f"{SAY}: CHECK, each averaged adapter's numbers ARE the mean of its models' (read back), it is none of them, and it is served: "
        + "; ".join(_mean_said(name, checks[name][THE_MEAN]) for name in prepare["averaged"])
        + f": {_verdict({'passes': all(checks[name][THE_MEAN]['passes'] for name in prepare['averaged'])})}",
        f"{SAY}: CHECK, Lean answered (at most {MAXIMUM_SHARE_WITHOUT_AN_ANSWER} of a set without an answer): "
        + "; ".join(f"`{name}` {checks[name][LEAN_ANSWERED]['share_without_an_answer']}" for name in models)
        + f": {_verdict({'passes': not not_to_be_read})}"]
    if inconclusive:
        lines.append(f"{SAY}: {INCONCLUSIVE}. {as_it_is['reason']}")
    else:
        lines += [
            f"{SAY}: THE FIGURES, successes per 1,000 attempts on all of G ({problems} problems) over the {attempts} attempts, each with its binomial sampling noise (from each "
            "problem's own rate): " + "; ".join(said(name) for name in models),
            f"{SAY}: THE SPREAD OF THE RECIPE AS IT IS: {as_it_is['name']}. {as_it_is['reason']}",
            f"{SAY}: DOES DECAY LOWER IT: {lowers['name']}. {lowers['reason']}",
            f"{SAY}: AND ITS LEVEL: {level['name']}. {level['reason']}",
            *(f"{SAY}: AVERAGING: {of_the_average[recipe]['name']}. {of_the_average[recipe]['reason']}" for recipe in recipes),
            f"{SAY}: WHAT FOLLOWS: {follows['name']}. {follows['reason']}",
            f"{SAY}: BESIDE, naming nothing: goal problems solved in the {attempts} attempts, the share of distinct attempts, the share of attempts that timed out in Lean: "
            + "; ".join(beside_said(name) for name in models) + ". The spread of the goal problems solved (standard deviation, n - 1): "
            + ", ".join(f"`{recipe}` {beside['goal_problems_solved'][recipe]['standard_deviation']}" for recipe in recipes),
            f"{SAY}: BESIDE, naming nothing: L5's own `{of_the_rows['model']}` of seed {own_seed} (the same rows in that seed's order, the recipe as it is), from its stored "
            "first sampling: one more draw of `constant`, in no standard deviation: "
            + (f"{fifth['per_1000']} per 1,000 ({_per_1000(fifth['sampling_noise'])}), {fifth['goal_problems_solved']} goal problems solved, share of distinct attempts "
               f"{fifth['share_of_distinct_attempts']}, share timed out in Lean {fifth['share_timed_out_in_lean']}" if fifth["there"] else f"NOT THERE ({fifth['why']})"),
            f"{SAY}: BESIDE, naming nothing: L5's `{of_the_rows['model']}` at each of its seeds (rows and order both drawn), on the same {attempts} attempts: "
            + "; ".join(f"seed {seed}: " + (f"{entry['per_1000']} per 1,000, {entry['goal_problems_solved']} solved" + (" (its own report reads INCONCLUSIVE)" if entry.get("inconclusive") else "")
                                            if entry["there"] else f"NOT THERE ({entry['why']})") for seed, entry in beside["the_published_of_l5"].items())]
    if not_to_be_read:
        lines.append(f"{SAY}: NOT TO BE READ: too many attempts without a verdict from Lean in the sets of {', '.join(f'`{name}`' for name in not_to_be_read)}. The step fails; "
                     "the task queued again measures such a set again from its adapter")

    headline = (f"L5n ({QUESTION.split('?')[0]}; {LABEL}) seed {prepare['seed']}, from `{start}`, {of_the_rows['rows']:,} fixed rows in {len(orders)} orders: "
                + (f"{INCONCLUSIVE}. {as_it_is['reason']}" if inconclusive else
                   f"the spread of the recipe as it is {_per_1000(as_it_is['standard_deviation'])} per 1,000 ({as_it_is['name']}); with the decay "
                   f"{_per_1000(lowers['of_decay'])} ({lowers['name']}; its level {level['name']}); averaging: "
                   + ", ".join(f"`{mean_of(recipe)}` {of_the_average[recipe]['name']}" for recipe in recipes) + f". What follows: {follows['name']}")
                + (f". NOT TO BE READ: too many attempts without a verdict from Lean in the sets of {', '.join(not_to_be_read)}" if not_to_be_read else ""))
    kept = lambda what: {"measured_and_not_read": what} if inconclusive else what      # noqa: E731
    return {
        "spec": SPEC, "stage": L4, "label": LABEL, "check": "L5n", "headline": headline, "lines": lines, "ok": not not_to_be_read, "seed": prepare["seed"],
        "stand_in_engine": bool(prepare.get("stand_in_engine") or any(entry.get("stand_in_engine") for entry in (*trains.values(), *averaged.values(), *measured.values()))),
        "start": start, "rank": rank, "models": models, "orders": orders, "recipes": recipes,
        "branches": {"the_spread_of_the_recipe_as_it_is": as_it_is["name"], "does_decay_lower_it": lowers["name"], "and_its_level": level["name"],
                     "averaging": {mean_of(recipe): of_the_average[recipe]["name"] for recipe in recipes}, "what_follows": follows["name"]},
        "inconclusive": inconclusive, "failed_checks": failed,
        "can_this_run_be_read": {"what": "the checks the spec reads first; when any fails the run is INCONCLUSIVE and nothing else is said", **checks},
        "figures": {"what": f"each model's successes per attempt on all of G over the {attempts} attempts of the first sampling, with its binomial sampling noise (from each "
                            "problem's own observed rate p over its n attempts: the square root of the sum of p(1 - p) / n, over the number of problems); beside, its goal "
                            "problems solved, its share of distinct attempts and its share of attempts that timed out in Lean", **kept(figures)},
        "the_spread_of_the_recipe_as_it_is": as_it_is, "does_decay_lower_it": lowers, "and_its_level": level,
        "averaging": {mean_of(recipe): of_the_average[recipe] for recipe in recipes}, "what_follows": follows,
        "beside": {"what": "naming nothing: each recipe's goal problems solved by order and their spread; L5's `published` at each of its seeds on the same sampling, the "
                           "seed that trained the fixed rows being one more draw of `constant` that is in no standard deviation", **kept(beside)},
        "the_rows": of_the_rows, "the_orders": prepare["the_orders"], "the_sampling": sampling, "training_sets": prepare["training_sets"], "recipe": prepare["recipe"],
        "models_made": {name: ({"model": f"`{name}`: from `{start}`, one pass over {trains[name]['rows']:,} rows in the order of order seed {prepare['training_sets'][name]['order_seed']}: "
                                         f"{IS[prepare['training_sets'][name]['recipe']]}",
                                "training": {key: value for key, value in trains[name].items() if key != "checkpoints"}} if name in trains else
                               {"model": f"`{name}`: trained by nothing: the mean of {', '.join(f'`{model}`' for model in averaged[name].get('of') or [])}", "averaging": averaged[name]})
                        for name in models},
        "smoke": smoke, "what_it_cannot_say": list(CANNOT_SAY),
        "attempts": {name: _health(rows[name]) for name in models}, "not_to_be_read": not_to_be_read,
        "not_measured": ["the three rungs, the goal set's second sampling and the held-aside set: this stage reads a spread, not a model",
                         f"`{start}`, and L5's models: their rows are the stored ones of their own runs", "anything at another temperature than the config's"]}
