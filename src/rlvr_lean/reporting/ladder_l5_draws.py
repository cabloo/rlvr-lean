"""L5d's report: the three kinds again, over fresh draws, measured on the goal set's first sampling. Spec:
docs/spec/ladder-loop.spec.md, "L5d: the three kinds again, over fresh draws, measured on the goal set's first
sampling" ("The reads, fixed before any fresh seed runs", "Checks, read first", "What follows, fixed now"). Pure: rows
in, a report out, on the rules of `domain/ladder_round/l5_draws.py`.

Everything here is labelled pretrained on published proofs, as L4's reports are: every line begins with L4's label.
One-shot attempts only. A model is ONE sampling of the goal set: no rung, no second sampling, no held-aside set.

  the checks        FIRST, each PASS or FAIL, of THIS seed: each set is L5's rule for its seed, made again and held, and
                    no published row carries text; the start model's stored first sampling is the setting's; each trained
                    adapter is not the start adapter's and is read back at its rank; a model was trained on its set and
                    on nothing else, with no barred row and no problem of G or of the held-aside set; an averaged adapter
                    is the mean of its four (read back), none of them, and the one served; Lean answered on each set. ANY
                    FAIL: this seed is INCONCLUSIVE, nothing else is said of it (what was measured is kept under
                    `measured_and_not_read`), and the read over the seeds is INCONCLUSIVE until the seed is run again
  this seed         each model's figure on all of G, on the goal problems the start model solves in its own first
                    sampling and on G'; the goal problems each solves, gained and lost against `published`, and of the
                    long ones; each set's share of attempts that timed out in Lean
  over the seeds    THE READ, from the fresh seeds that are there (`l5_draws.over_the_seeds`): the primary and its four
                    outcomes; beside it, naming nothing, three differences; where it sits; the spread; L5's own seeds
                    beside, EXPLORATORY; what follows. With a seed of the fixed number not there: NOT READ, naming it

`read_over_the_seeds` and `lines_over_the_seeds` are what `tools/ladder_l5_draws.py` makes that read with again, from
the pulled task directories.

A report in which Lean did not answer a set is NOT TO BE READ (`ok` is false, and the step fails): the task queued again
samples such a set again from its adapter.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ, the_adapter_differs
from rlvr_lean.domain.ladder_round.l5 import DEEP, MODELS, PUBLISHED, SHALLOW, no_barred_row, timed_out
from rlvr_lean.domain.ladder_round.l5_draws import (
    AGAIN,
    ALL,
    BESIDE,
    CONSTANT,
    DECAY,
    MEAN_CONSTANT,
    SOLVED,
    THE_PRIMARY,
    differences,
    its_trainings,
    no_problem_of,
    on_the_parts,
    over_the_seeds,
    the_counts,
    the_parts,
    the_sets_are_the_rules,
    the_start_is_the_settings,
    trained_on_its_set,
)
from rlvr_lean.domain.ladder_round.read import GOAL, group_ids
from rlvr_lean.reporting.ladder_ceiling import goal_sets, lean_did_not_answer
from rlvr_lean.reporting.ladder_l2 import MAXIMUM_SHARE_WITHOUT_AN_ANSWER, _health
from rlvr_lean.reporting.ladder_l3c import LENGTH_GROUPS
from rlvr_lean.reporting.ladder_l4 import LABEL, SAY, _verdict, adapter_against_the_start
from rlvr_lean.reporting.ladder_l5_noise import LEAN_ANSWERED, THE_ADAPTER, THE_MEAN, _at_its_rank, the_mean_check

L4 = "l4"
QUESTION = "the three kinds again, over fresh draws, measured on the goal set's first sampling"
SPEC = "docs/spec/ladder-loop.spec.md, L5d: the three kinds again, over fresh draws, measured on the goal set's first sampling"
LONG = LENGTH_GROUPS[-1]                    # the goal problems whose shortest published proof has 8 lines or more: the "long problems" of the counts
THE_RUN = "the_run"                         # the two checks that are of the run and of no one model, as a failed check names them
THE_SETS = "each_set_is_l5s_rule_for_its_seed_made_again_and_held"
THE_START = "the_stored_first_sampling_of_the_start_model_is_the_settings"
THE_ROWS = "it_was_trained_on_its_set_and_on_nothing_else"
NO_BARRED_ROW = "no_barred_row"
NO_PROBLEM = "no_problem_of_g_or_of_the_held_aside_set"
THE_SAMPLING = "the goal set's first sampling"      # the one kind of set a model here is measured on, as the timeouts name it
IS = {CONSTANT: "the recipe as it is, one training a model", DECAY: "one training a model, with L5n's decay",
      MEAN_CONSTANT: "each set trained in four orders with the recipe as it is; the model measured is the adapter that is the mean of the four"}
PART_SAID = {ALL: "all of G", SOLVED: "the goal problems the start model solves in its own first sampling", AGAIN: "G'"}
CANNOT_SAY = [
    "G alone and one sampling of it: nothing of the rungs, of the second sampling or of the held-aside set; reach on OTHER hard problems is not measured here",
    "THE BAR IS A SECOND PASS: `published` trains on rows the start model was already trained on, the model's own proofs are of problems it never trained on. The first "
    "outcome says its own proofs of new problems are worth more than reading published proofs again; it does not say they are worth more than PUBLISHED proofs of new problems",
    "ONE SEARCH: the deep solves and the ordinary ones come from the one search and the one map. The spread over seeds here is over draws and trainings, not over searches",
    "One start model, one size of set, one pass"]


def _per_1000(value: float | None, signed: bool = False) -> str:
    return "not measured" if value is None else f"{1000 * value:+.2f}" if signed else f"{1000 * value:.2f}"


def _interval_said(own: Mapping) -> str:
    return f"{_per_1000(own['mean'], True)} [{_per_1000(own['low'], True)}, {_per_1000(own['high'], True)}]"


# ------------------------------------------------------------------------------------------------ the checks
def the_checks(prepare: Mapping, trains: Mapping[str, Mapping], trained: Mapping[str, Sequence[Mapping]], averaged: Mapping[str, Mapping], measured: Mapping[str, Mapping],
               rows: Mapping[str, Sequence[Mapping]], sets: Mapping[str, Sequence[Mapping]], pretraining: Sequence[Mapping], start_rows: Sequence[Mapping],
               goal_ids: Sequence[str], aside: Sequence[str], heldout: Sequence[str], half_seed: int, base_map: Sequence[str], of_the_setting: Mapping) -> dict[str, dict]:
    """The checks the spec reads FIRST: the run's two (`THE_RUN`), then by trained model its adapter and its rows, and
    by measured model its averaged adapter (the averaged recipe) and whether Lean answered its set."""
    recipe = prepare.get("start_recipe") or {}
    rank, alpha = recipe.get("rank"), recipe.get("alpha")
    in_orders = {name: sets[name] for name in prepare["trained"] if name not in MODELS}
    checks: dict[str, dict] = {THE_RUN: {
        THE_SETS: the_sets_are_the_rules({model: sets[model] for model in MODELS}, pretraining, prepare["seed"], in_orders),
        THE_START: the_start_is_the_settings(start_rows, goal_ids, prepare["the_sampling"], of_the_setting["sampling"], of_the_setting["attempts"])}}
    for name in prepare["trained"]:
        differs, saved = the_adapter_differs(adapter_against_the_start(trains[name])), trains[name].get("adapter_saved")
        at_rank = _at_its_rank(saved, rank, alpha)
        checks[name] = {THE_ADAPTER: {**differs, "read_back": saved, "rank_of_the_start": rank, "alpha_of_the_start": alpha, "at_its_rank": at_rank,
                                      "passes": differs["passes"] and at_rank},
                        THE_ROWS: {"what": "the rows its training loop recorded are the rows of its stored set, each once", **trained_on_its_set([row["id"] for row in trained[name]], sets[name])},
                        NO_BARRED_ROW: no_barred_row(trained[name], set(heldout), half_seed, set(base_map)),
                        NO_PROBLEM: no_problem_of(trained[name], set(goal_ids), set(aside))}
    for name in prepare["models"]:
        own = checks.setdefault(name, {})
        if name in prepare["averaged"]:
            own[THE_MEAN] = the_mean_check(averaged[name], measured[name], rows[name], rank, alpha, bool(prepare.get("smoke")))
        health = _health(rows[name])
        own[LEAN_ANSWERED] = {"what": f"Lean gave a verdict on all but at most {MAXIMUM_SHARE_WITHOUT_AN_ANSWER} of the attempts of its set; the task queued again samples such "
                                      "a set again", "maximum": MAXIMUM_SHARE_WITHOUT_AN_ANSWER, "share_without_an_answer": health["share_without_an_answer"],
                              "passes": not lean_did_not_answer(rows[name])}
    return checks


# ------------------------------------------------------------------------------------- the read over the seeds
def read_over_the_seeds(by_seed: Mapping[int, Mapping], fixed: Sequence[int], of_l5: Mapping[int, Mapping] | None = None) -> dict:
    """THE READ OVER THE FRESH SEEDS (`l5_draws.over_the_seeds`), as a report and the tool hold it. Seeds of two recipes,
    or seeds that were not measured alike, are NOT POOLED: the read is then NOT READ, with the refusal as its reason."""
    try:
        return over_the_seeds(by_seed, fixed, of_l5)
    except ValueError as error:
        not_read = {"name": NOT_READ, "reason": f"{error}: nothing is read over the seeds"}
        return {"fixed_seeds": list(fixed), "seeds_there": sorted(seed for seed, entry in by_seed.items() if entry.get("there", True)), "read": False, "refused": str(error),
                "primary": not_read, "beside": None, "where_it_sits": dict(not_read), "the_spread": None, "exploratory": None, "what_follows": dict(not_read)}


def lines_over_the_seeds(read: Mapping, beside_every_read: str | None = None) -> list[str]:
    """The printed lines of the read over the seeds. `beside_every_read`: what is said beside every read when L5n did
    not name the recipe."""
    fixed, primary, sits, follows = read["fixed_seeds"], read["primary"], read["where_it_sits"], read["what_follows"]
    said = f" NOTE: {beside_every_read}" if beside_every_read else ""
    lines = [f"{SAY}: THE PRIMARY, over the fresh seeds {fixed} alone (every interval is over SEEDS: the mean of the per-seed differences, plus and minus Student's t times "
             f"their standard deviation over the root of the seeds): {primary['name']}. {primary['reason']}.{said}"]
    if read.get("seeds_not_of_the_fixed_number"):
        lines.append(f"{SAY}: the seeds {read['seeds_not_of_the_fixed_number']} are there and are not of the fixed number: a new replication, read on its own seeds alone, "
                     "and left out here")
    if not read["read"]:
        return lines
    spread, exploratory = read["the_spread"], read["exploratory"]
    lines += [
        f"{SAY}: BESIDE, each with its interval over the seeds, naming nothing (per 1,000 attempts): "
        + "; ".join(f"`{after}` minus `{before}` {_interval_said(read['beside'][key])}" for key, (after, before) in BESIDE.items())
        + f"; the primary's difference is positive at {primary['seeds_at_which_it_is_positive']} of the {primary['seeds']} fresh seeds",
        f"{SAY}: WHERE IT SITS: {sits['name']}. {sits['reason']}.{said}",
        f"{SAY}: THE SPREAD, each kind's figure per 1,000 attempts at each fresh seed, its standard deviation over the seeds (n - 1) and, beside, the sampling noise of one "
        "figure: " + "; ".join(f"`{model}` " + ", ".join(f"seed {seed} {_per_1000(rate)}" for seed, rate in spread[model]["by_seed"].items())
                               + f" (standard deviation {_per_1000(spread[model]['standard_deviation'])}, sampling noise {_per_1000(spread[model]['sampling_noise_of_one_figure'])})"
                               for model in MODELS),
        f"{SAY}: EXPLORATORY, naming nothing: " + (
            f"L5's own seeds {exploratory['seeds_that_found_the_sign']} FOUND THE SIGN (the primary's difference there: "
            + ", ".join(f"seed {seed} {_per_1000(value, True)}" for seed, value in exploratory["of_l5"].items()) + f"); pooled with the fresh seeds, {exploratory['pooled']['seeds']} "
            f"seeds: {_interval_said(exploratory['pooled'])} per 1,000. An interval that holds the seeds that found the sign confirms nothing"
            if exploratory["read"] else f"L5's seeds beside the fresh ones are NOT READ ({exploratory['why']})"),
        f"{SAY}: WHAT FOLLOWS: {follows['name']}. {follows['reason']}.{said}"]
    resolved = follows.get("seeds_would_resolve")
    if resolved:
        lines.append(f"{SAY}: what a number of seeds would resolve, with the standard deviation measured here (the interval's half-width per 1,000 attempts; no power is "
                     "computed): " + ", ".join(f"{count} seeds {_per_1000(width)}" for count, width in resolved["by_seeds"].items())
                     + f"; at most the effect found from {resolved['seeds_at_which_it_is_at_most_the_effect_found'] or 'more seeds than these'}")
    return lines


# ------------------------------------------------------------------------------------------------ the report
def build_l5_draws_report(prepare: Mapping, trains: Mapping[str, Mapping], trained: Mapping[str, Sequence[Mapping]], averaged: Mapping[str, Mapping],
                          measured: Mapping[str, Mapping], rows: Mapping[str, Sequence[Mapping]], sets: Mapping[str, Sequence[Mapping]], pretraining: Sequence[Mapping],
                          start_rows: Sequence[Mapping], groups: Sequence[Mapping], lengths: Mapping[str, Mapping], again: Sequence[str], aside: Sequence[str],
                          settings: Mapping, base_map: Sequence[str], of_the_setting: Mapping, others: Mapping[int, Mapping], fixed: Sequence[int],
                          of_l5: Mapping[int, Mapping]) -> dict:
    """`prepare`: the prepare step's summary (`seed`; `start`; `models`, `trained`, `averaged`; `the_recipe`;
    `the_sampling`; `start_recipe`). `trains`: by trained model, its training step's summary. `trained`: by trained
    model, the rows its training loop recorded (id, problem, origin, place in the file). `averaged`: by model of the
    averaged recipe, its averaging step's summary. `measured`: by model, its measuring step's summary. `rows`: by model,
    its per-problem rows on the goal set's first sampling. `sets`: by model and by trained model, its stored set.
    `pretraining`: the rows the start model was pretrained on, without their text. `start_rows`: the start model's
    stored first sampling, as this run copied it. `groups`: the held-out groups. `lengths`: each held-out problem's
    length group. `again`: G'. `aside`: the held-aside problems. `settings`: `ladder_loop.l4` (the half seed).
    `base_map`: the base map's problems. `of_the_setting`: the `sampling` and the `attempts` a problem the setting names
    NOW, which the start model's stored first sampling is held to. `others`: the OTHER fresh seeds, each what its report says of its own seed
    (`of_this_seed`) or not `there`, with `why`. `fixed`: the seeds the reads are over. `of_l5`: L5's own seeds, each
    `there` with its three models' per-problem `rows` on the same sampling, or not."""
    start, seed, models, recipe, sampling = prepare["start"], prepare["seed"], list(prepare["models"]), prepare["the_recipe"], prepare["the_sampling"]
    if sorted(rows) != sorted(models) or sorted(trains) != sorted(prepare["trained"]) or sorted(averaged) != sorted(prepare["averaged"]):
        raise ValueError(f"the report is given the rows of {sorted(rows)}, the trainings of {sorted(trains)} and the averages of {sorted(averaged)}: it reads this run's models "
                         f"({models}), each measured")
    goal_ids = group_ids(groups, GOAL)
    ids = {name: sorted(row["problem_id"] for row in own) for name, own in rows.items()}
    if any(own != sorted(goal_ids) for own in ids.values()):
        raise ValueError("the models were not measured on the goal problems, each once: every model here is one sampling of the one goal set")
    rank, attempts, name_of_the_recipe = (prepare.get("start_recipe") or {}).get("rank"), sampling["episodes"], recipe["recipe"]
    long_ids = goal_sets(goal_ids, lengths).get(LONG, [])

    # ---- the checks, FIRST
    checks = the_checks(prepare, trains, trained, averaged, measured, rows, sets, pretraining, start_rows, goal_ids, aside, [row["problem_id"] for row in groups],
                        settings["half_seed"], base_map, of_the_setting)
    failed = [f"{'the run' if name == THE_RUN else f'`{name}`'}: {check}" for name, of in checks.items() for check, entry in of.items() if not entry["passes"]]
    inconclusive = bool(failed)
    not_to_be_read = [name for name in models if not checks[name][LEAN_ANSWERED]["passes"]]

    # ---- this seed's figures and counts; the parts of G are the start model's own first sampling's, held to the G' the first step copied
    try:
        parts = the_parts(start_rows, again)
    except ValueError:      # the stored sampling is not the one G' was made from: no figure is given on a part, and where it sits is NOT READ
        parts = {ALL: list(goal_ids), SOLVED: [], AGAIN: []}
    figures = {name: {part: {**own, **({"share_of_distinct_attempts": ((measured[name].get("distinct_attempts") or {}).get(GOAL) or {}).get("mean_share_distinct")}
                                       if part == ALL else {})} for part, own in on_the_parts(rows[name], parts).items()} for name in models}
    counts = the_counts(rows, long_ids)
    at_this_seed = differences(figures)
    of_timeouts = timed_out({name: _health(rows[name]) for name in models}, {THE_SAMPLING: {name: name for name in models}})
    of_timeouts.update({"checked_again": False, "why": "no step checks a seed's timed-out attempts again: where a pair is named here, it is to be done by hand (Lean only, no "
                                                       "measurement running) and the reads given both ways"})
    of_this_seed = {"seed": seed, "recipe": name_of_the_recipe, "ok": not not_to_be_read, "inconclusive": inconclusive, "failed_checks": failed,
                    "the_measurement": {"sampling": sampling["name"], "episodes": attempts, "sampling_seed": sampling["sampling_seed"], "problems": len(goal_ids)},
                    "figures": None if inconclusive else figures, "counts": None if inconclusive else counts}

    # ---- THE READ OVER THE FRESH SEEDS, from the seeds that are there; L5's own seeds beside, on the same parts
    of_theirs = {key: ({"there": True, "run": entry.get("run"), "inconclusive": entry.get("inconclusive"),
                        "figures": {model: on_the_parts(entry["rows"][model], parts) for model in MODELS}} if entry.get("there") else {"there": False, "why": entry.get("why")})
                 for key, entry in of_l5.items()}
    read = read_over_the_seeds({seed: of_this_seed, **others}, fixed, of_theirs or None)
    beside_every_read = recipe.get("said_beside_every_read")

    def figure_said(part: str) -> str:
        return ", ".join(f"`{name}` {_per_1000(figures[name][part]['per_attempt'])} ({_per_1000(figures[name][part]['sampling_noise'])})" for name in models)

    def adapter_said(name: str) -> str:
        entry = checks[name][THE_ADAPTER]
        changed = (f"{entry['elements_changed']:,} of the {entry['elements']:,} numbers of its adapter are not the start adapter's" if entry["compared"]
                   else "its adapter was not compared with the start adapter's")
        saved = entry["read_back"]
        return f"`{name}`: {changed}; " + (f"read back at rank {saved['rank']} and alpha {saved['alpha']}" if saved else "not read back")

    of_the_sets, of_the_start, smoke = checks[THE_RUN][THE_SETS], checks[THE_RUN][THE_START], prepare.get("smoke")
    lines = [
        f"{SAY}: L5d, {QUESTION.upper()}. Seed {seed}, a further seed of L5: its three sets by L5's own rule for this seed, from the ONE search ({prepare['the_search']['run']}: "
        f"{prepare['the_search']['deep_solves']:,} deep solves, D = {prepare['the_search']['rows_of_each_kind']:,}). From `{start}` at rank {rank}, one pass: "
        f"{IS[name_of_the_recipe]}. EACH MODEL IS MEASURED on the goal set's first sampling alone: {attempts} attempts at each of {len(goal_ids)} goal problems, sampling seed "
        f"{sampling['sampling_seed']} (the one `{start}`'s run recorded): no rung, no second sampling, no held-aside set",
        f"{SAY}: THE RECIPE is `{name_of_the_recipe}`: {recipe['found']}" + (f". SAID BESIDE EVERY READ: {beside_every_read}" if beside_every_read else "")]
    if smoke:
        lines.append(f"{SAY}: A SMOKE RUN: L5's smoke run " + (f"reads INCONCLUSIVE ({', '.join(smoke['l5s_failed_checks'])}) and was read all the same; "
                                                              if smoke.get("l5_read_though_inconclusive") else "was read; ")
                     + "one deep solve is enough, and orders of a set's handful of rows that are alike are trained all the same. Nothing here is a read of the three kinds")
    lines += [
        f"{SAY}: CHECK, each set is L5's rule for its seed, made again and held, and no published row carries text: "
        + (f"D = {of_the_sets['rows_of_each_kind']}" if of_the_sets["refused"] is None else f"REFUSED ({of_the_sets['refused']})")
        + f"; {of_the_sets['published_rows_with_text']} published rows with text; sets not in their order seed's order: {of_the_sets['sets_not_in_their_order_seeds_order']}: "
        f"{_verdict(of_the_sets)}",
        f"{SAY}: CHECK, the stored first sampling of `{start}` is the setting's ({of_the_start['problems']} problems of {of_the_start['goal_problems']}, sampling seed "
        f"{of_the_start['sampling_seed']}, attempts a problem that are not {attempts}: {of_the_start['attempts_a_problem_that_are_not_the_settings']}): {_verdict(of_the_start)}",
        f"{SAY}: CHECK, each trained adapter is not the start adapter's and is read back at its rank ({rank}): " + "; ".join(adapter_said(name) for name in prepare["trained"])
        + f": {_verdict({'passes': all(checks[name][THE_ADAPTER]['passes'] for name in prepare['trained'])})}",
        f"{SAY}: CHECK, each model was trained on its set and on nothing else, with no barred row (no held-out problem; its published rows of the `pretrain` half; its own "
        "rows base-map problems) and no problem of G or of the held-aside set (the training loop's own record): "
        + "; ".join(f"`{name}` {checks[name][THE_ROWS]['rows']:,} rows of {checks[name][THE_ROWS]['rows_of_its_set']:,}, {checks[name][NO_BARRED_ROW]['held_out_problems']} held "
                    f"out, {checks[name][NO_PROBLEM]['problems_of_the_goal_set']} of G, {checks[name][NO_PROBLEM]['problems_of_the_held_aside_set']} held aside"
                    for name in prepare["trained"])
        + f": {_verdict({'passes': all(checks[name][check]['passes'] for name in prepare['trained'] for check in (THE_ROWS, NO_BARRED_ROW, NO_PROBLEM))})}"]
    if prepare["averaged"]:
        lines.append(f"{SAY}: CHECK, each averaged adapter's numbers ARE the mean of its four's (read back), it is none of them, and it is the one served: "
                     + "; ".join(f"`{name}` (the mean of {', '.join(f'`{of}`' for of in checks[name][THE_MEAN]['of'] or [])}): {checks[name][THE_MEAN]['elements_not_the_mean']} of "
                                 f"{checks[name][THE_MEAN]['elements']} numbers are not the mean" + ("" if checks[name][THE_MEAN]["served"] else "; it was NOT served")
                                 for name in prepare["averaged"])
                     + f": {_verdict({'passes': all(checks[name][THE_MEAN]['passes'] for name in prepare['averaged'])})}")
    lines.append(f"{SAY}: CHECK, Lean answered (at most {MAXIMUM_SHARE_WITHOUT_AN_ANSWER} of a set without an answer): "
                 + "; ".join(f"`{name}` {checks[name][LEAN_ANSWERED]['share_without_an_answer']}" for name in models) + f": {_verdict({'passes': not not_to_be_read})}")
    if inconclusive:
        lines.append(f"{SAY}: THIS SEED IS {INCONCLUSIVE}: {'; '.join(failed)}. Nothing else is said of it; it is left out of the read over the seeds, which is {INCONCLUSIVE} "
                     "until this seed is run again")
    else:
        lines += [
            f"{SAY}: THIS SEED'S FIGURES, successes per 1,000 attempts over the {attempts} attempts, each with its binomial sampling noise: on all of G ({len(parts[ALL])} "
            f"problems) {figure_said(ALL)}; on {PART_SAID[SOLVED]} ({len(parts[SOLVED])}) {figure_said(SOLVED)}; on {PART_SAID[AGAIN]} ({len(parts[AGAIN])}) {figure_said(AGAIN)}. "
            f"The mean of `{SHALLOW}` and `{DEEP}`, minus `{PUBLISHED}`, on all of G at this seed: {_per_1000(at_this_seed[THE_PRIMARY], True)} (one seed names nothing)",
            f"{SAY}: THIS SEED'S COUNTS, goal problems solved in the {attempts} attempts (gained, lost against `{PUBLISHED}`), and of the {len(long_ids)} long problems (8 lines "
            "or more): " + "; ".join(f"`{name}` {counts[name]['solved']} (+{counts[name]['gained']}, -{counts[name]['lost']}), long {counts[name]['long']['solved']} "
                                     f"(+{counts[name]['long']['gained']}, -{counts[name]['long']['lost']})" for name in models),
            f"{SAY}: TIMEOUTS, each set's share of attempts that timed out in Lean: " + "; ".join(f"`{name}` {of_timeouts['share'][name]}" for name in models)
            + (f". MORE THAN A POINT APART: {'; '.join(' and '.join(pair['models']) for pair in of_timeouts['apart'])}: this seed's timed-out attempts are to be checked again "
               "(Lean only, no measurement running) and the reads given both ways; no step does it" if of_timeouts["apart"] else ". No two models are more than a point apart")]
    lines += lines_over_the_seeds(read, beside_every_read)
    if not_to_be_read:
        lines.append(f"{SAY}: NOT TO BE READ: too many attempts without a verdict from Lean in the sets of {', '.join(f'`{name}`' for name in not_to_be_read)}. The step fails; "
                     "the task queued again measures such a set again from its adapter")

    headline = (f"L5d ({QUESTION.split(',')[0]}; {LABEL}) seed {seed}, from `{start}`, the recipe `{name_of_the_recipe}`: "
                + (f"THIS SEED IS {INCONCLUSIVE} ({'; '.join(failed)})" if inconclusive else
                   "per 1,000 attempts on all of G " + ", ".join(f"`{name}` {_per_1000(figures[name][ALL]['per_attempt'])}" for name in models)
                   + f"; the primary's difference at this seed {_per_1000(at_this_seed[THE_PRIMARY], True)}")
                + f". Over the fresh seeds {list(fixed)}: {read['primary']['name']}. What follows: {read['what_follows']['name']}"
                + (f". {beside_every_read}" if beside_every_read else "")
                + (f". NOT TO BE READ: too many attempts without a verdict from Lean in the sets of {', '.join(not_to_be_read)}" if not_to_be_read else ""))
    kept = lambda what: {"measured_and_not_read": what} if inconclusive else what      # noqa: E731
    return {
        "spec": SPEC, "stage": L4, "label": LABEL, "check": "L5d", "headline": headline, "lines": lines, "ok": not not_to_be_read, "seed": seed,
        "stand_in_engine": bool(prepare.get("stand_in_engine") or any(entry.get("stand_in_engine") for entry in (*trains.values(), *averaged.values(), *measured.values()))),
        "start": start, "rank": rank, "models": models, "recipe": name_of_the_recipe, "the_recipe": recipe,
        "branches": {"primary": read["primary"]["name"], "where_it_sits": read["where_it_sits"]["name"], "what_follows": read["what_follows"]["name"]},
        "inconclusive": inconclusive, "failed_checks": failed,
        "can_this_run_be_read": {"what": "the checks the spec reads first, of this seed; when any fails the seed is INCONCLUSIVE, nothing else is said of it, and it is left out "
                                         "of the read over the seeds until it is run again", **checks},
        "of_this_seed": of_this_seed,
        "figures": {"what": f"each model's successes per attempt over the {attempts} attempts of the first sampling, with its binomial sampling noise, on all of G, on the goal "
                            "problems the start model solves in its own first sampling and on G'", "problems": {part: len(own) for part, own in parts.items()}, **kept(figures)},
        "counts": {"what": "the goal problems each model solves at least once in its attempts, gained and lost against `published`; and of the long problems (a shortest "
                           "published proof of 8 lines or more)", "long_problems": len(long_ids), **kept(counts)},
        "the_primarys_difference_at_this_seed": kept({key: None if value is None else round(value, 6) for key, value in at_this_seed.items()}),
        "over_the_fresh_seeds": read, "the_seeds_of_l5": {str(key): own for key, own in of_theirs.items()}, "timed_out_in_lean": of_timeouts,
        "the_seeds": prepare["the_seeds"], "the_search": prepare["the_search"], "the_sets": prepare["the_sets"], "the_orders": prepare.get("the_orders"),
        "the_sampling": sampling, "training_sets": prepare["training_sets"], "training_recipe": prepare["recipe"],
        "models_made": {name: {"model": f"`{name}`: " + (f"trained by nothing: the mean of {', '.join(f'`{of}`' for of in averaged[name].get('of') or [])}" if name in averaged
                                                         else f"from `{start}`, one pass over {trains[name]['rows']:,} rows"),
                               "trainings": {of: {key: value for key, value in trains[of].items() if key != "checkpoints"} for of in its_trainings(name_of_the_recipe, name)},
                               **({"averaging": averaged[name]} if name in averaged else {})} for name in models},
        "smoke": smoke, "what_it_cannot_say": list(CANNOT_SAY),
        "attempts": {name: _health(rows[name]) for name in models}, "not_to_be_read": not_to_be_read,
        "not_measured": ["the three rungs, the goal set's second sampling and the held-aside set: this stage reads one difference on the goal set's first sampling",
                         f"`{start}`, and L5's models: their rows are the stored ones of their own runs", "anything at another temperature than the config's"]}
