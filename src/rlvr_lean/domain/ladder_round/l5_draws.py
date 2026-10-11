"""L5d: the three kinds again, over fresh draws, measured on the goal set's first sampling. Spec:
docs/spec/ladder-loop.spec.md, "L5d: the three kinds again, over fresh draws, measured on the goal set's first
sampling". Pure.

L5's three seeds left one sign that repeats: on the goal set the two models trained on the model's own proofs beside
published ones are above the model trained on published ones alone. It was found on those seeds, so it is read again
here on FRESH seeds, each a further seed of L5 with the measurement cut to the goal set's first sampling.

  the recipe        a SETTING, held to what L5n's report names (`the_recipe`): `constant` (the recipe as it is, one
                    training a model), `decay` (one training with L5n's decay), `mean_constant` (each set trained in
                    four orders with the recipe as it is; the model measured is the adapter whose every number is the
                    mean of the four)
  the seeds         the fresh ones the setting names, and more of them with explicit approval (`the_seed`,
                    `fixed_seeds`): the number the reads are over is fixed before any is read
  the checks        each set is L5's rule for its seed, made again and held (`the_sets_are_the_rules`); no row of a set
                    is a problem of G or of the held-aside set (`no_problem_of`); a model was trained on its set and on
                    nothing else (`trained_on_its_set`); the start model's stored first sampling is the setting's
                    (`the_start_is_the_settings`)
  the figure        of a model: L5n's (`l5_noise.figure`), on all of G, on the goal problems the start model solves in
                    its own first sampling, and on G' (`on_the_parts`); the goal problems it solves, gained and lost
                    against `published`, and of the long ones (`the_counts`)
  the reads         over SEEDS, never over problems (`over_the_seeds`): THE PRIMARY (the mean of `shallow` and `deep`,
                    minus `published`) and its four outcomes; three differences beside it, naming nothing; where it
                    sits; the spread; L5's own seeds beside the fresh ones, EXPLORATORY; and what follows

An interval here is the mean of the per-seed differences, plus and minus Student's t times their standard deviation
(n - 1) over the root of the seeds.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Collection, Mapping, Sequence

from rlvr_lean.domain.ladder_round.assembly import training_order
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ
from rlvr_lean.domain.ladder_round.l4_rows import PRETRAINING
from rlvr_lean.domain.ladder_round.l4_rows2 import TEXT_FIELDS
from rlvr_lean.domain.ladder_round.l5 import DEEP, MODELS, NOT_SHOWN, PUBLISHED, REACH_TOO, RELIABILITY, SHALLOW, THEY_COST, hold_the_sets
from rlvr_lean.domain.ladder_round.l5_noise import AVERAGE_OF_FOUR, CONSTANT, DECAY, ORDERS, THE_RECIPES_OWN, WITH_THE_DECAY, figure, spread, the_orders

MEAN_CONSTANT = "mean_constant"                              # each set trained in four orders with the recipe as it is; the model measured is their averaged adapter
RECIPES = (CONSTANT, DECAY, MEAN_CONSTANT)                   # what `ladder_loop.l5_draws.recipe` may name
ASKED_BY = {WITH_THE_DECAY: DECAY, AVERAGE_OF_FOUR: MEAN_CONSTANT, THE_RECIPES_OWN: CONSTANT}       # the recipe each case of L5n's "what follows" asks for
OF_L5 = CONSTANT                                             # the recipe L5's own seeds were trained with: they stand beside the fresh ones under it alone
BEAT = "THE MODEL'S OWN PROOFS BESIDE PUBLISHED ONES BEAT PUBLISHED ONES ALONE ON THE GOAL SET"
UNDETECTABLE = "UNDETECTABLE AT THIS NUMBER OF SEEDS"
PRIMARY = (BEAT, THEY_COST, UNDETECTABLE, NOT_SHOWN, INCONCLUSIVE, NOT_READ)
NAMED_NOTHING = "NAMED NOTHING"                              # where it sits, when the section names neither of its two cases
SITS = (RELIABILITY, REACH_TOO, NAMED_NOTHING, NOT_READ)
L6_RUNS = "THE BET'S FIRST CONDITION HOLDS ON THE GOAL SET AT THIS SIZE: L6 RUNS WITH THIS RECIPE"
NOT_CONFIRMED = "THE SIGN OF L5'S THREE SEEDS IS NOT CONFIRMED: L6 DOES NOT RUN ON IT"
STOP = "STOP: THE NINE SEEDS NEED A DECISION"
FOLLOWS = (L6_RUNS, NOT_CONFIRMED, STOP, NOT_READ)
THE_EFFECT_FOUND = 0.0165                                    # per attempt: the primary's mean over L5's three seeds (16.5 per 1,000), which a half-width is held against
# Two-sided 95% critical values of Student's t by degrees of freedom (one fewer than the seeds): the section's are 4.303 at three seeds and 2.571 at six; nine seeds
# (the six and L5's three, pooled) have 2.306. A number of seeds outside it is refused.
T_CRITICAL = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}
ALL, SOLVED, AGAIN = "all_of_g", "solved_by_the_start", "goal_set_again"       # the parts of G a figure is given on: all of it; (a) the problems the start model solves; (b) G'
PARTS = (ALL, SOLVED, AGAIN)
THE_PRIMARY = "the_mean_of_shallow_and_deep_minus_published"
BESIDE = {"deep_minus_published": (DEEP, PUBLISHED), "shallow_minus_published": (SHALLOW, PUBLISHED), "deep_minus_shallow": (DEEP, SHALLOW)}


# ------------------------------------------------------------------------------------------------ the recipe
def the_recipe(setting: str, of_l5n: Mapping) -> dict:
    """THE RECIPE IS A SETTING, HELD TO L5n's REPORT. `of_l5n`: what was found of L5n's run (`there`; when it is not,
    `why`; when it is, `ok`, `inconclusive` and the name of its "what follows"; `run`). Its report names a case: the
    setting must be the recipe that case asks for (`ASKED_BY`). L5n's run not there, unfinished, not to be read,
    INCONCLUSIVE or naming no case: the setting must be the recipe as it is, and that is said beside every read.
    Refused (ValueError), naming both, otherwise. Returns the recipe, what L5n asks for and what it was found to say."""
    if setting not in RECIPES:
        raise ValueError(f"the recipe `{setting}` is none of {list(RECIPES)}")
    name = of_l5n.get("what_follows")
    if not of_l5n.get("there"):
        found, asked, by_l5n = f"L5n's run is not there or not finished ({of_l5n.get('why')})", CONSTANT, False
    elif not of_l5n.get("ok", True) or of_l5n.get("inconclusive") or name not in ASKED_BY:
        found, asked, by_l5n = "L5n's report " + ("is not to be read" if not of_l5n.get("ok", True) else f"reads {INCONCLUSIVE}" if of_l5n.get("inconclusive") else
                                                  f"names no case for what follows ({name})"), CONSTANT, False
    else:
        found, asked, by_l5n = f"L5n's report names {name}", ASKED_BY[name], True
    if setting != asked:
        raise ValueError(f"the setting names the recipe `{setting}` and {found} ({of_l5n.get('run')}), which asks for `{asked}`: the recipe is the one L5n names, taken as "
                         "named. Change the setting, or finish L5n's run")
    return {"recipe": setting, "asked_for_by_l5n": asked, "found": found, "l5n_named_it": by_l5n,
            "said_beside_every_read": None if by_l5n else f"{found}: the recipe as it is, one training a model, is trained with, and L5n did not name it"}


def trained_of(model: str, order: int) -> str:
    """One of the four trainings of a kind under the averaged recipe: `published_1`."""
    return f"{model}_{order}"


def trained_models(recipe: str) -> tuple[str, ...]:
    """The models a run TRAINS, in the order it trains them: one a kind; under the averaged recipe four a kind, one an
    order seed."""
    return tuple(trained_of(model, order) for model in MODELS for order in ORDERS) if recipe == MEAN_CONSTANT else MODELS


def its_trainings(recipe: str, model: str) -> tuple[str, ...]:
    """The trained models a MEASURED model stands on: itself; under the averaged recipe the four its adapter is the mean of."""
    return tuple(trained_of(model, order) for order in ORDERS) if recipe == MEAN_CONSTANT else (model,)


def four_orders(rows: Sequence[Mapping], smoke: bool = False) -> tuple[dict[int, list[Mapping]], dict]:
    """One set's rows in the content-hash order of each ORDER SEED, as L5n orders rows (`l5_noise.the_orders`). `rows`:
    the set in the order L5's rule gives it for the task's seed. An order seed that IS the task's seed gives that
    order: it is trained all the same (no model here is trained in the seed's own order beside it), and named. Two
    orders that are one are refused (ValueError): their adapters would be one adapter. `smoke` (a SMOKE run alone,
    whose handful of rows has few orders): trained all the same. Returns (by order seed, the rows; what coincides)."""
    by_order, coincide = the_orders(rows, ORDERS, loosely=True)
    if coincide["alike"] and not smoke:
        raise ValueError(f"of the order seeds {list(ORDERS)}, the pairs {coincide['alike']} give one order of these {len(rows)} rows: the averaged adapter is the mean of "
                         "four trainings that differ by their order. Refused")
    return by_order, {"the_order_of_the_tasks_seed": coincide["the_order_trained_before"], "alike": coincide["alike"]}


# ------------------------------------------------------------------------------------------------- the seeds
def fixed_seeds(seeds: Sequence[int], more: Sequence[int], approved: bool) -> tuple[int, ...]:
    """The seeds the reads are over, fixed before any is read: the setting's; and the further ones too once their
    approval is recorded."""
    return (*seeds, *(more if approved else ()))


def the_seed(seed: int, seeds: Sequence[int], more: Sequence[int], approved: bool, of_l5: Collection[int]) -> None:
    """A task's seed is one of the fresh seeds. Refused (ValueError): a seed of L5's own (they found the sign); a seed of
    the further ones while their approval is not recorded; a seed that is in neither list."""
    if seed in of_l5:
        raise ValueError(f"seed {seed} is one of L5's own ({sorted(of_l5)}): those seeds found the sign, and this stage reads it on fresh seeds alone ({list(seeds)}; "
                         f"{list(more)} with explicit approval)")
    if seed in more and not approved:
        raise ValueError(f"seed {seed} is one of the further seeds {list(more)}, which run with explicit approval, and the approval is not recorded "
                         "(more_seeds_approved is false): more than three seeds needs it in advance")
    if seed not in fixed_seeds(seeds, more, approved):
        raise ValueError(f"seed {seed} is none of the fresh seeds {list(seeds)}" + (f" and {list(more)}" if approved else f" (nor of {list(more)}, which need explicit approval)")
                         + ": the number of seeds is fixed before the read, and a seed outside it is no seed of this stage")


# ------------------------------------------------------------------------------------------------ the checks
def the_sets_are_the_rules(sets: Mapping[str, Sequence[Mapping]], pretraining: Sequence[Mapping], seed: int, in_orders: Mapping[str, Sequence[Mapping]] | None = None) -> dict:
    """The first check: each set is L5's rule for its seed, MADE AGAIN AND HELD (`l5.hold_the_sets`, which a further seed
    of L5 holds its sets to), and no published row of one carries text. `sets`: the three sets as stored. `pretraining`:
    the rows the start model was pretrained on, in the file's order, without their text. `in_orders` (the averaged
    recipe): by trained model `<kind>_<order seed>`, its stored set, held to be its kind's rows in that order seed's
    order."""
    with_text = sorted(row["id"] for own in sets.values() for row in own if row["origin"] == PRETRAINING and any(field in row for field in TEXT_FIELDS))
    try:
        count, why = hold_the_sets(sets, pretraining, seed), None
    except (ValueError, KeyError) as error:
        count, why = None, str(error)
    out_of_order = []
    for name, own in (in_orders or {}).items():
        model, _, order = name.rpartition("_")
        if model not in sets or not order.isdigit() or [row["id"] for row in own] != [row["id"] for row in training_order(sets[model], int(order))]:
            out_of_order.append(name)
    return {"what": "each set, as stored, is the set L5's rule builds for this seed: D deep rows and D published rows, D ordinary solves and the same D published rows, 2D "
                    "published rows, the published rows the first of the hash order of this seed, each set in its content-hash order; no published row carries text; and "
                    "under the averaged recipe each of a kind's four sets is that kind's rows in its order seed's order",
            "rows_of_each_kind": count, "refused": why, "published_rows_with_text": len(with_text), "sets_not_in_their_order_seeds_order": out_of_order,
            "passes": count is not None and not with_text and not out_of_order}


def no_problem_of(rows: Sequence[Mapping], goal_ids: Collection[str], aside_ids: Collection[str]) -> dict:
    """No row of a set is of a problem of G or of the held-aside set (the goal set is what is measured; the held-aside
    set was L5's test)."""
    of_g = sorted({row["problem_id"] for row in rows if row["problem_id"] in goal_ids})
    of_the_set = sorted({row["problem_id"] for row in rows if row["problem_id"] in aside_ids})
    return {"what": "no row it was trained on is of a problem of G or of the held-aside set", "rows": len(rows), "problems_of_the_goal_set": len(of_g),
            "problems_of_the_held_aside_set": len(of_the_set), "first": [*of_g, *of_the_set][:5], "passes": bool(rows) and not of_g and not of_the_set}


def refuse_problems_of(rows: Sequence[Mapping], goal_ids: Collection[str], aside_ids: Collection[str], what: str) -> dict:
    """A set is refused (ValueError) BEFORE anything is trained when it holds a problem of G or of the held-aside set."""
    found = no_problem_of(rows, goal_ids, aside_ids)
    if not found["passes"]:
        raise ValueError(f"{found['problems_of_the_goal_set']} problems of {what} are goal problems and {found['problems_of_the_held_aside_set']} are held-aside problems "
                         f"(first: {(found['first'] or [None])[0]}): refused, nothing was written")
    return found


def trained_on_its_set(trained: Sequence[str], set_rows: Sequence[Mapping]) -> dict:
    """A training was of its stored set and of nothing else: the ids its TRAINING LOOP recorded are the set's, each once."""
    wanted, twice = {row["id"] for row in set_rows}, sorted(key for key, times in Counter(trained).items() if times > 1)
    strangers, missing = sorted(set(trained) - wanted), sorted(wanted - set(trained))
    return {"rows": len(trained), "rows_of_its_set": len(wanted), "rows_that_are_none_of_them": len(strangers), "rows_twice": len(twice), "rows_not_trained_on": len(missing),
            "first": [*strangers, *twice, *missing][:5], "passes": bool(trained) and not strangers and not twice and not missing}


def the_start_is_the_settings(rows: Sequence[Mapping], goal_ids: Sequence[str], recorded: Mapping, sampling: str, attempts: int) -> dict:
    """The last check: the start model's stored first sampling of G, as this run copied it, is the setting's: its
    problems are the goal problems, each has the setting's attempts, and it is the sampling the setting names, with the
    sampling seed every model here is measured with (`recorded`: the sampling as the prepare step recorded it)."""
    ids = [row["problem_id"] for row in rows]
    other = sorted({row["episodes"] for row in rows} - {attempts})
    return {"what": "the stored first sampling of the start model is the setting's: its problems, its sampling seed, its attempts a problem",
            "problems": len(ids), "goal_problems": len(goal_ids), "the_goal_problems": sorted(ids) == sorted(goal_ids) and len(set(ids)) == len(ids),
            "attempts_a_problem_that_are_not_the_settings": other, "sampling": recorded.get("name"), "sampling_seed": recorded.get("sampling_seed"),
            "passes": bool(ids) and sorted(ids) == sorted(goal_ids) and len(set(ids)) == len(ids) and not other
            and (recorded.get("name"), recorded.get("episodes")) == (sampling, attempts) and recorded.get("sampling_seed") is not None}


# ----------------------------------------------------------------------------------------------- the figures
def the_parts(start_rows: Sequence[Mapping], again: Collection[str]) -> dict[str, list[str]]:
    """The goal problems by part: all of G; (a) the ones the start model solves in its own first sampling; (b) G', the
    ones it does not. `again`: G' as the search's run stored it, which (b) is HELD to (ValueError)."""
    ids = [row["problem_id"] for row in start_rows]
    solved = [row["problem_id"] for row in start_rows if row["resolved"] > 0]
    unsolved = [row["problem_id"] for row in start_rows if row["resolved"] == 0]
    if sorted(unsolved) != sorted(again):
        raise ValueError(f"the start model's stored first sampling leaves {len(unsolved)} goal problems unsolved and G' holds {len(again)}: the two are not one sampling's")
    return {ALL: ids, SOLVED: solved, AGAIN: unsolved}


def on_the_parts(rows: Sequence[Mapping], parts: Mapping[str, Sequence[str]]) -> dict[str, dict]:
    """A model's figure (`l5_noise.figure`) on each part of G."""
    by_id = {row["problem_id"]: row for row in rows}
    return {part: figure([by_id[problem_id] for problem_id in ids if problem_id in by_id]) for part, ids in parts.items()}


def the_counts(rows: Mapping[str, Sequence[Mapping]], long_ids: Collection[str]) -> dict[str, dict]:
    """The counts at one seed: the goal problems each model solves at least once in its attempts, gained and lost
    against `published`; and the same of the long problems (`long_ids`: a shortest published proof of 8 lines or more)."""
    solves = {model: {row["problem_id"] for row in own if row["resolved"] > 0} for model, own in rows.items()}

    def of(model: str, among: Collection[str] | None) -> dict:
        own, base = (solves[name] if among is None else solves[name] & set(among) for name in (model, PUBLISHED))
        return {"solved": len(own), "gained": len(own - base), "lost": len(base - own)}

    return {model: {**of(model, None), "long": {"problems": len(long_ids), **of(model, long_ids)}} for model in rows}


def _rate(own: Mapping) -> float | None:
    return own["successes"] / own["attempts"] if own["attempts"] else None


def differences(figures: Mapping[str, Mapping[str, Mapping]], part: str = ALL) -> dict[str, float | None]:
    """One seed's differences on one part of G, per attempt, from its models' figures (their whole numbers of successes
    and attempts): the primary's (the mean of `shallow` and `deep`, minus `published`) and the three that stand beside
    it. None, each, on a part that holds no problem."""
    rate = {model: _rate(figures[model][part]) for model in MODELS}
    if None in rate.values():
        return {key: None for key in (THE_PRIMARY, *BESIDE)}
    return {THE_PRIMARY: (rate[SHALLOW] + rate[DEEP]) / 2 - rate[PUBLISHED], **{key: rate[after] - rate[before] for key, (after, before) in BESIDE.items()}}


# ------------------------------------------------------------------------------------------------- the reads
def interval(at_each: Mapping[int, float]) -> dict:
    """The interval over SEEDS of a per-seed difference: its mean, plus and minus Student's t at one degree fewer than
    the seeds times the standard deviation (n - 1) over the root of the seeds. Refused (ValueError) for a number of
    seeds the table does not hold."""
    values = [at_each[seed] for seed in sorted(at_each)]
    degrees = len(values) - 1
    if degrees not in T_CRITICAL:
        raise ValueError(f"the t interval over {len(values)} seeds is not given here: its critical values are those of {sorted(count + 1 for count in T_CRITICAL)} seeds")
    mean, deviation, critical = statistics.fmean(values), statistics.stdev(values), T_CRITICAL[degrees]
    half_width = critical * deviation / math.sqrt(len(values))
    return {"at_each_seed": {str(seed): round(at_each[seed], 6) for seed in sorted(at_each)}, "seeds": len(values), "mean": round(mean, 6),
            "standard_deviation": round(deviation, 6), "degrees_of_freedom": degrees, "critical_value": critical, "half_width": round(half_width, 6),
            "low": round(mean - half_width, 6), "high": round(mean + half_width, 6), "exact": {"mean": mean, "half_width": half_width}}


def named(low: float, high: float, half_width: float) -> str:
    """The primary's outcome from its interval. Above zero: THE MODEL'S OWN PROOFS BEAT. Below zero: THEY COST. It holds
    zero (one that ends at zero holds it): UNDETECTABLE AT THIS NUMBER OF SEEDS when its half-width is ABOVE the effect
    found (16.5 per 1,000), NOT SHOWN otherwise (a half-width that is exactly the effect found is not above it)."""
    if low > 0:
        return BEAT
    if high < 0:
        return THEY_COST
    return UNDETECTABLE if half_width > THE_EFFECT_FOUND else NOT_SHOWN


def _per_1000(value: float) -> str:
    return f"{1000 * value:+.2f}"


def _said(own: Mapping) -> str:
    return f"[{_per_1000(own['low'])}, {_per_1000(own['high'])}] per 1,000"


def primary(at_each: Mapping[int, float]) -> dict:
    """THE PRIMARY: the mean of `shallow` and `deep`, minus `published`, at each fresh seed; its interval over the seeds
    and its outcome (`named`). An interval that is UNDETECTABLE is given with no word about the effect."""
    own = interval(at_each)
    exact = own.pop("exact")
    low, high = exact["mean"] - exact["half_width"], exact["mean"] + exact["half_width"]
    name = named(low, high, exact["half_width"])
    each = ", ".join(f"seed {seed} {_per_1000(at_each[seed])}" for seed in sorted(at_each))
    said = {BEAT: f"mean {_per_1000(exact['mean'])}, interval over the {own['seeds']} seeds {_said(own)} ({each}): above zero",
            THEY_COST: f"mean {_per_1000(exact['mean'])}, interval over the {own['seeds']} seeds {_said(own)} ({each}): below zero",
            UNDETECTABLE: f"the interval over the {own['seeds']} seeds is {_said(own)}: it holds zero, and its half-width ({1000 * exact['half_width']:.2f}) is above the "
                          f"{1000 * THE_EFFECT_FOUND:g} per 1,000 found on L5's seeds: at this number of seeds the effect found could not have been told from none",
            NOT_SHOWN: f"mean {_per_1000(exact['mean'])}, interval over the {own['seeds']} seeds {_said(own)} ({each}): it holds zero, and its half-width "
                       f"({1000 * exact['half_width']:.2f}) is not above the {1000 * THE_EFFECT_FOUND:g} per 1,000 found on L5's seeds"}[name]
    return {"name": name, **own, "the_effect_found": THE_EFFECT_FOUND, "seeds_at_which_it_is_positive": sum(value > 0 for value in at_each.values()),
            "reason": f"the mean of `{SHALLOW}` and `{DEEP}`, minus `{PUBLISHED}`, on all of G over the first sampling's attempts: {said}"}


def where_it_sits(name: str, on_the_solved: Mapping[int, float], on_again: Mapping[int, float]) -> dict:
    """WHERE IT SITS, read only when the primary names the first outcome: the primary's difference (a) on the goal
    problems the start model solves in its own first sampling and (b) on G'. RELIABILITY when (a)'s interval is above
    zero and (b)'s holds zero; REACH TOO when (b)'s is above zero. Any other pair of intervals is given and NAMED
    NOTHING: the section names two cases."""
    if name != BEAT:
        return {"name": NOT_READ, "reason": f"the primary reads {name}: where it sits is read only when the model's own proofs beat"}
    if None in (*on_the_solved.values(), *on_again.values()):
        return {"name": NOT_READ, "reason": "a part of G holds no problem at a seed (the problems the start model solves in its first sampling, or G'): it has no difference"}
    solved, again = ({key: value for key, value in interval(at_each).items() if key != "exact"} for at_each in (on_the_solved, on_again))
    if again["low"] > 0:
        found = REACH_TOO
    elif solved["low"] > 0 and again["high"] >= 0:
        found = RELIABILITY
    else:
        found = NAMED_NOTHING
    return {"name": found, "on_the_problems_the_start_solves": solved, "on_the_goal_set_again": again,
            "reason": f"the primary's difference on the goal problems the start model solves in its own first sampling is {_said(solved)}, and on G' {_said(again)}"
                      + ("" if found != NAMED_NOTHING else ": neither of the section's two cases")}


def half_widths(deviation: float) -> dict:
    """What a number of seeds would resolve with this standard deviation: the interval's half-width at each number of
    seeds the table holds, and the smallest of them at which it is at most the effect found (None: more than the table
    holds). No power is computed."""
    by_seeds = {degrees + 1: critical * deviation / math.sqrt(degrees + 1) for degrees, critical in sorted(T_CRITICAL.items())}
    enough = [count for count, width in by_seeds.items() if width <= THE_EFFECT_FOUND]
    return {"what": "the half-width of the interval at each number of seeds, with the standard deviation measured here (Student's t at one degree fewer than the seeds); "
                    "no power is computed", "by_seeds": {str(count): round(width, 6) for count, width in by_seeds.items()},
            "seeds_at_which_it_is_at_most_the_effect_found": min(enough) if enough else None}


def what_follows(read: Mapping) -> dict:
    """What follows, by the section's three cases. THE MODEL'S OWN PROOFS BEAT: the bet's first condition holds on the
    goal set at this size, and L6 runs with this recipe. UNDETECTABLE or NOT SHOWN: the sign is not confirmed; the
    interval, the standard deviation and what a number of seeds would resolve need a decision, and L6 does not run on
    it. THEY COST: stop. Nothing is named while the primary is not read."""
    name = read["name"]
    if name in (INCONCLUSIVE, NOT_READ):
        return {"name": NOT_READ, "reason": f"the primary reads {name}: nothing is named for what follows"}
    resolved = half_widths(read["standard_deviation"])
    if name == BEAT:
        return {"name": L6_RUNS, "seeds_would_resolve": resolved,
                "reason": f"the primary reads {name}: L6 (a second round) runs with this recipe, with as many seeds an arm as the standard deviation measured here "
                          f"({1000 * read['standard_deviation']:.2f} per 1,000 over {read['seeds']} seeds) asks for its own difference"}
    if name == THEY_COST:
        return {"name": STOP, "reason": f"the primary reads {name}: stop. L5's three seeds and the fresh ones need a decision"}
    return {"name": NOT_CONFIRMED, "seeds_would_resolve": resolved,
            "reason": f"the primary reads {name}: the sign of L5's three seeds is not confirmed. The interval ({_said(read)}), the standard deviation "
                      f"({1000 * read['standard_deviation']:.2f} per 1,000) and what a number of seeds would resolve need a decision; L6 does not run on it"}


def _not_made(name: str, reason: str) -> dict:
    return {"name": name, "reason": reason}


def over_the_seeds(by_seed: Mapping[int, Mapping], fixed: Sequence[int], of_l5: Mapping[int, Mapping] | None = None) -> dict:
    """THE READ OVER THE FRESH SEEDS. `by_seed`: by seed, what its report holds (`recipe`; `ok`, `inconclusive`,
    `failed_checks`; `the_measurement`; `figures` by model and part; `counts`), or `{"there": False, "why": ...}` for a
    seed whose run is not there. `fixed`: the seeds the reads are over, fixed before any is read. `of_l5`: L5's own
    seeds on the same sampling, each `there` with its `figures`, or not: they stand beside the fresh ones, EXPLORATORY,
    when the recipe is the one L5 trained with.

    REFUSED (ValueError): seeds of two recipes, or seeds that were not measured alike: they are not pooled. A seed of
    the fixed number that is not there: every read is NOT READ, and it says which. A seed whose check failed, or whose
    report is not to be read: the run is INCONCLUSIVE until that seed is run again. A seed that is there and is not of
    the fixed number is a new replication: it is named, and left out."""
    there = {seed: entry for seed, entry in sorted(by_seed.items()) if entry.get("there", True)}
    recipes = {seed: entry["recipe"] for seed, entry in there.items()}
    if len(set(recipes.values())) > 1:
        raise ValueError("the seeds were trained with two recipes (" + ", ".join(f"seed {seed}: `{recipe}`" for seed, recipe in recipes.items()) + "): seeds of two recipes "
                         "are not pooled")
    measured = {seed: entry["the_measurement"] for seed, entry in there.items()}
    if len({repr(sorted(own.items())) for own in measured.values()}) > 1:
        raise ValueError(f"the seeds were not measured alike ({measured}): their figures are not pooled")
    recipe = next(iter(recipes.values()), None)
    missing = [seed for seed in fixed if seed not in there]
    failed = {seed: list(there[seed].get("failed_checks") or []) or ["its report is not to be read"] for seed in fixed
              if seed in there and (there[seed].get("inconclusive") or not there[seed].get("ok", True))}
    read = [seed for seed in fixed if seed in there and seed not in failed]
    found = {"fixed_seeds": list(fixed), "seeds_there": sorted(there), "seeds_missing": missing, "seeds_whose_checks_failed": {str(seed): own for seed, own in failed.items()},
             "seeds_not_of_the_fixed_number": [seed for seed in there if seed not in fixed], "recipe": recipe,
             "figures": {str(seed): {model: {part: {key: there[seed]["figures"][model][part][key] for key in ("successes", "attempts", "per_1000", "sampling_noise")}
                                             for part in PARTS} for model in MODELS} for seed in read},
             "counts": {str(seed): there[seed]["counts"] for seed in read}}
    if missing or failed:
        name = NOT_READ if missing else INCONCLUSIVE
        why = (f"the reads are fixed over the seeds {list(fixed)} and the runs of {missing} are not there" + (f" (and the checks of {sorted(failed)} failed)" if failed else "")
               + ": nothing is read over the seeds" if missing else
               f"a check failed at {', '.join(f'seed {seed} ({chr(59).join(own)})' for seed, own in failed.items())}: such a seed is left out, and the run is INCONCLUSIVE "
               "until it is run again")
        return {**found, "read": False, "primary": _not_made(name, why), "beside": None, "where_it_sits": _not_made(NOT_READ, why), "the_spread": None,
                "exploratory": None, "what_follows": what_follows({"name": name})}
    at = {seed: {part: differences(there[seed]["figures"], part) for part in PARTS} for seed in read}
    of_the_primary = primary({seed: at[seed][ALL][THE_PRIMARY] for seed in read})
    beside = {key: {name: value for name, value in interval({seed: at[seed][ALL][key] for seed in read}).items() if name != "exact"} for key in BESIDE}
    sits = where_it_sits(of_the_primary["name"], {seed: at[seed][SOLVED][THE_PRIMARY] for seed in read}, {seed: at[seed][AGAIN][THE_PRIMARY] for seed in read})
    the_spread = {}
    for model in MODELS:
        rates = [_rate(there[seed]["figures"][model][ALL]) for seed in read]
        noises = [there[seed]["figures"][model][ALL]["sampling_noise"] for seed in read]
        the_spread[model] = {"by_seed": {str(seed): round(rate, 6) for seed, rate in zip(read, rates)}, "mean": round(statistics.fmean(rates), 6),
                             "standard_deviation": round(spread(rates), 6), "sampling_noise_of_one_figure": round(math.sqrt(statistics.fmean(noise ** 2 for noise in noises)), 6)}
    return {**found, "read": True, "primary": of_the_primary, "beside": beside, "where_it_sits": sits, "the_spread": the_spread,
            "exploratory": pooled(recipe, {seed: at[seed][ALL][THE_PRIMARY] for seed in read}, of_l5), "what_follows": what_follows(of_the_primary)}


def pooled(recipe: str, at_each: Mapping[int, float], of_l5: Mapping[int, Mapping] | None) -> dict:
    """EXPLORATORY, and marked so: when the recipe is the one L5 trained with, L5's own seeds beside the fresh ones on
    the same first sampling, and all of them pooled, the seeds of L5 marked as the ones that found the sign. It names
    nothing. Not read under another recipe, or when a seed of L5 is not there."""
    marked = {"exploratory": True, "what": "EXPLORATORY: L5's own seeds found the sign, so an interval that holds them confirms nothing; it names nothing"}
    if recipe != OF_L5:
        return {**marked, "read": False, "why": f"the recipe is `{recipe}` and L5's seeds were trained with `{OF_L5}`: they do not stand beside these"}
    absent = {seed: entry.get("why") for seed, entry in (of_l5 or {}).items() if not entry.get("there")}
    if not of_l5 or absent:
        return {**marked, "read": False, "why": "L5's seeds are not there: " + ("; ".join(f"seed {seed}: {why}" for seed, why in absent.items()) or "none was given")}
    of_theirs = {seed: differences(entry["figures"])[THE_PRIMARY] for seed, entry in sorted(of_l5.items())}
    both = {**of_theirs, **at_each}
    if len(both) != len(of_theirs) + len(at_each) or len(both) - 1 not in T_CRITICAL:
        return {**marked, "read": False, "why": f"L5's seeds {sorted(of_theirs)} and the fresh seeds {sorted(at_each)} are not {len(of_theirs) + len(at_each)} seeds the "
                                                "interval is given for"}
    return {**marked, "read": True, "seeds_that_found_the_sign": sorted(of_theirs), "of_l5": {str(seed): round(value, 6) for seed, value in of_theirs.items()},
            "pooled": {key: value for key, value in interval(both).items() if key != "exact"}}
