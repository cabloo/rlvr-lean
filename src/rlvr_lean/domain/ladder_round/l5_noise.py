"""L5n: what is one training worth? The spread between trainings, and a recipe that lowers it. Spec:
docs/spec/ladder-loop.spec.md, "L5n: what is one training worth? The spread between trainings, and a recipe that
lowers it". Pure.

L5's second seed turned its first seed's order over: two trainings of one kind differ by far more than any difference
a stage since L4t has read between two single trained models. This measures that spread on ONE fixed set of rows, and
asks whether one change to the recipe, or an average of adapters, lowers it.

  the rows          ONE fixed set: the published rows L5's own task of its search's seed trained `published` on, by id
                    and place in the pretraining file, never with text (`the_rows`: held to that file, to the draw's
                    rule and to what that training's loop recorded)
  the orders        the rows in the content-hash order of each ORDER SEED (`the_orders`): none is the order that
                    training was made in, and no two are one order
  the models        each order trained twice from the start adapter: `constant_<k>` (the recipe as it is) and
                    `decay_<k>` (ONE change: after the warm-up the learning rate falls linearly to zero, `dose.decayed`);
                    and, trained by nothing, `mean_constant` and `mean_decay`: the adapter whose every number is the
                    mean of its recipe's adapters'
  the figure        of a model: its successes per attempt on all of G over the first sampling's attempts, with its
                    binomial sampling noise from each problem's own rate (`figure`)
  the reads         THE SPREAD OF THE RECIPE AS IT IS (`spread_as_it_is`); DOES DECAY LOWER IT (`decay_lowers`); AND ITS
                    LEVEL (`its_level`); AVERAGING (`averaging`); and one line for what follows (`what_follows`)

A check that failed makes every read INCONCLUSIVE, and nothing else is said. A standard deviation here is the sample's
(n - 1).
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from itertools import combinations
from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.assembly import training_order
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ
from rlvr_lean.domain.ladder_round.l4_rows import PRETRAINING, rehearsal_id, rehearsal_rows
from rlvr_lean.domain.ladder_round.l4_rows2 import _placed, refuse_text
from rlvr_lean.domain.ladder_round.l5 import NOT_SHOWN

CONSTANT, DECAY = "constant", "decay"                       # the two recipes: as it is; with ONE change, the learning rate falling linearly to zero after the warm-up
RECIPES = (CONSTANT, DECAY)
ORDERS = (1, 2, 3, 4)                                        # the order seeds: each is trained once with each recipe (the stage's steps are registered for them)
MEAN = "mean"                                                # an averaged model: `mean_<recipe>`

ORDER_ALONE, ROWS_DRAWN = "ORDER ALONE MAKES THE SPREAD", "THE ROWS DRAWN MAKE MOST OF IT"
AS_IT_IS = (ORDER_ALONE, ROWS_DRAWN, INCONCLUSIVE, NOT_READ)
DECAY_LOWERS = "DECAY LOWERS THE SPREAD"
LOWERS = (DECAY_LOWERS, NOT_SHOWN, INCONCLUSIVE)
DECAY_COSTS, NAMED_NOTHING = "DECAY COSTS", "NAMED NOTHING"   # its level: the interval below zero; otherwise the level is given and named nothing
LEVEL = (DECAY_COSTS, NAMED_NOTHING, INCONCLUSIVE)
AT_LEAST_THE_MEAN = "AVERAGING GIVES AT LEAST THE MEAN OF ITS MODELS"
LESS_THAN_THE_MEAN = "AVERAGING GIVES LESS THAN THE MEAN OF ITS MODELS"      # the outcome the section does not name: below that mean by more than twice its sampling noise
AVERAGING = (AT_LEAST_THE_MEAN, LESS_THAN_THE_MEAN, INCONCLUSIVE)
WITH_THE_DECAY = "EVERY LATER STAGE TRAINS WITH THE DECAY"
AVERAGE_OF_FOUR = "A LATER STAGE TRAINS EACH MODEL FOUR TIMES AND MEASURES THE AVERAGED ADAPTER"
THE_RECIPES_OWN = "THE SPREAD IS THE RECIPE'S OWN AT THIS SIZE: THE CHOICE NEEDS A DECISION"
FOLLOWS = (WITH_THE_DECAY, AVERAGE_OF_FOUR, THE_RECIPES_OWN, NOT_READ)
HALF = 0.5                                                   # "at least half of the second"; "at most half"
NOISES = 2                                                   # "by more than twice its sampling noise"
# Two-sided 95% critical values of Student's t by degrees of freedom (one fewer than the orders): the section's is 3.182, at 3. And the upper 5% point of F at
# equal degrees: what the ratio of two variances of that many orders each must pass to be beyond chance (the section's is 9.28, at 3 and 3).
T_CRITICAL = {1: 12.706, 2: 4.303, 3: 3.182}
F_CRITICAL = {1: 161.4, 2: 19.0, 3: 9.28}


# ------------------------------------------------------------------------------------------------ the models
def model_of(recipe: str, order: int) -> str:
    """A trained model: `constant_1`."""
    return f"{recipe}_{order}"


def mean_of(recipe: str) -> str:
    """An averaged model: `mean_constant`."""
    return f"{MEAN}_{recipe}"


TRAINED = tuple(model_of(recipe, order) for order in ORDERS for recipe in RECIPES)       # in the order they are trained and measured: each order with the recipe as it is, then with the decay
AVERAGED = tuple(mean_of(recipe) for recipe in RECIPES)
MODELS = (*TRAINED, *AVERAGED)


def parts_of(model: str) -> tuple[str, int]:
    """(the recipe, the order seed) of a trained model. Refused (ValueError) for a name that is none."""
    recipe, _, order = model.rpartition("_")
    if recipe not in RECIPES or not order.isdigit():
        raise ValueError(f"`{model}` is no trained model of L5n: one is `<recipe>_<order seed>`, a recipe of {list(RECIPES)}")
    return recipe, int(order)


def models_of(orders: Sequence[int], recipes: Sequence[str] = RECIPES) -> tuple[str, ...]:
    """The models of a run that trains `orders`: each order with each recipe in turn, then one averaged model a recipe."""
    return (*(model_of(recipe, order) for order in orders for recipe in recipes), *(mean_of(recipe) for recipe in recipes))


# -------------------------------------------------------------------------------------------------- the rows
def the_rows(stored: Sequence[Mapping], trained_ids: Sequence[str], pretraining: Sequence[Mapping], seed: int) -> list[dict]:
    """THE FIXED ROWS: the set L5's own task trained `published` on, as its run stored it (`stored`, in the order it
    was trained in), each row named by its id and its place in the pretraining file and NEVER with its text.
    `trained_ids`: the ids that training's loop recorded, in the order trained. `pretraining`: the rows the start model
    was pretrained on, in the file's order, without their text. `seed`: the seed of that task.

    HELD, each a refusal (ValueError): no row carries text; every row is a published row, and no id stands twice; each
    is the row of the file at its place, under its own id; they are the rows of the smallest hash that L5's rule draws
    for that seed (`l4_rows.rehearsal_rows`: the first as many of its order), standing in the content-hash order that
    task trained in; and the training's own record is these rows in this order. Returns the rows as stored, without
    their place in that set."""
    refuse_text(stored, "the stored set of L5's `published`")
    strangers = [row["id"] for row in stored if row["origin"] != PRETRAINING]
    twice = sorted(key for key, times in Counter(row["id"] for row in stored).items() if times > 1)
    if not stored or strangers or twice:
        raise ValueError(f"the stored set of L5's `published` holds {len(stored)} rows, {len(strangers)} that are no published rows and {len(twice)} ids twice"
                         + (f" (first: {(strangers or twice)[0]})" if strangers or twice else "") + ": the fixed rows are published rows, each once. Refused")
    misplaced = [row["id"] for row in stored if not 0 <= row["row_of_the_file"] < len(pretraining) or pretraining[row["row_of_the_file"]]["problem_id"] != row["problem_id"]
                 or row["id"] != rehearsal_id(row["problem_id"])]
    if misplaced:
        raise ValueError(f"{len(misplaced)} rows of the stored set of L5's `published` are not the rows of the pretraining file at their place (first: {misplaced[0]}): a "
                         "published row is named by its id and its place in the file. Refused")
    drawn = rehearsal_rows(pretraining, len(stored), seed)
    if sorted(map(_placed, stored)) != sorted(map(_placed, drawn)):
        other = sorted(set(map(_placed, stored)) ^ set(map(_placed, drawn)))
        raise ValueError(f"the {len(stored)} rows of the stored set of L5's `published` are not the first {len(drawn)} of the hash order of seed {seed} among the "
                         f"{len(pretraining)} rows the start model was pretrained on (first that differs: {other[0][0]}): that set was made otherwise. Refused")
    ids = [row["id"] for row in stored]
    if ids != [row["id"] for row in training_order(stored, seed)] or list(trained_ids) != ids:
        raise ValueError(f"the stored set of L5's `published` does not stand in the content-hash order of seed {seed}, or its training's own record ({len(trained_ids)} "
                         "rows) is not these rows in this order: the fixed rows are the rows that model was trained on, in the order it was. Refused")
    return [{key: value for key, value in row.items() if key != "row"} for row in stored]


def the_orders(rows: Sequence[Mapping], orders: Sequence[int], loosely: bool = False) -> tuple[dict[int, list[Mapping]], dict]:
    """The fixed rows in the content-hash order of each order seed (`assembly.training_order`: a hash of the ORDER SEED
    and each row's id). `rows`: the fixed rows, in the order L5's own task trained them in. HELD, each a refusal
    (ValueError): no order is the order that task trained in, and no two orders are one. `loosely` (a SMOKE run alone,
    whose handful of rows has few orders): such an order is trained all the same. Returns (by order seed, the rows in
    its order; which orders are the one trained before, and which pairs are alike: none, unless `loosely`)."""
    before = [row["id"] for row in rows]
    by_order = {order: list(training_order(rows, order)) for order in orders}
    ids = {order: [row["id"] for row in own] for order, own in by_order.items()}
    again = [order for order in orders if ids[order] == before]
    alike = [[first, second] for first, second in combinations(orders, 2) if ids[first] == ids[second]]
    if (again or alike) and not loosely:
        raise ValueError(f"of the order seeds {list(orders)}, {again} give the order L5's own task trained these {len(rows)} rows in, and the pairs {alike} give one order: "
                         "each order here is another draw of the order alone. Refused")
    return by_order, {"the_order_trained_before": again, "alike": alike}


def numbered(rows: Sequence[Mapping]) -> list[dict]:
    """A set's rows, each with its place in ITS set."""
    return [{"row": place, **{key: value for key, value in row.items() if key != "row"}} for place, row in enumerate(rows)]


def only_the_fixed_rows(trained: Sequence[Mapping], fixed: Sequence[Mapping]) -> dict:
    """The check of a trained model here, read on the rows its TRAINING LOOP recorded: no row is anything but one of
    the fixed rows (by id, problem and place in the file), and it holds each of them once: a model trained on fewer of
    them, or on one twice, is no training of the fixed set."""
    wanted, found = {_placed(row) for row in fixed}, [_placed(row) for row in trained]
    strangers = sorted({row for row in found if row not in wanted})
    twice = sorted(row for row, times in Counter(found).items() if times > 1)
    missing = sorted(wanted - set(found))
    return {"what": "no row it was trained on is anything but one of the fixed rows, by id and by place in the pretraining file, and it holds each of them once",
            "rows": len(found), "fixed_rows": len(wanted), "rows_that_are_none_of_them": len(strangers), "rows_twice": len(twice), "fixed_rows_not_trained_on": len(missing),
            "first": [row[0] for row in (*strangers, *twice, *missing)][:5], "passes": bool(found) and not strangers and not twice and not missing}


# ------------------------------------------------------------------------------------------------ the figure
def per_attempt(rows: Sequence[Mapping]) -> float | None:
    """A model's successes per attempt over the rows of one sampling (one row a problem: `resolved` of `episodes`)."""
    attempts = sum(row["episodes"] for row in rows)
    return sum(row["resolved"] for row in rows) / attempts if attempts else None


def sampling_noise(rows: Sequence[Mapping]) -> float | None:
    """The binomial sampling noise of that figure for a FIXED set of problems, from each problem's own observed rate p
    over its n attempts: the square root of the sum over problems of p(1 - p) / n, divided by the number of problems."""
    own = [row for row in rows if row["episodes"]]
    if not own:
        return None
    return math.sqrt(sum((row["resolved"] / row["episodes"]) * (1 - row["resolved"] / row["episodes"]) / row["episodes"] for row in own)) / len(own)


def figure(rows: Sequence[Mapping]) -> dict:
    """THE FIGURE OF A MODEL, from its per-problem rows on the goal set's first sampling: its successes per attempt on
    all of G with its sampling noise; and, beside, the goal problems it solves at least once and the share of its
    attempts that timed out in Lean (an audited problem is attempted on both sides)."""
    rate, noise, made = per_attempt(rows), sampling_noise(rows), sum(row["episodes"] * row["sides"] for row in rows)
    timed_out = sum(row["attempts_timed_out"] for row in rows)
    return {"problems": len(rows), "successes": sum(row["resolved"] for row in rows), "attempts": sum(row["episodes"] for row in rows),
            "per_attempt": None if rate is None else round(rate, 6), "per_1000": None if rate is None else round(1000 * rate, 2),
            "sampling_noise": None if noise is None else round(noise, 6), "goal_problems_solved": sum(row["resolved"] > 0 for row in rows),
            "timed_out_in_lean": timed_out, "share_timed_out_in_lean": round(timed_out / made, 5) if made else None}


def spread(values: Sequence[float]) -> float | None:
    """The standard deviation (n - 1) of the figures of several trainings; None for fewer than two."""
    return statistics.stdev(values) if len(values) > 1 else None


def _said(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.5f}"


# ------------------------------------------------------------------------------------------------- the reads
def _inconclusive(failed_checks: Sequence[str], what: str) -> dict:
    return {"name": INCONCLUSIVE, "failed_checks": list(failed_checks), "reason": f"the run could not have been read ({', '.join(failed_checks)}): {what} is not read"}


def spread_as_it_is(failed_checks: Sequence[str], constant: Sequence[float], noises: Sequence[float], over_the_seeds: Sequence[float] | None, not_there: str = "") -> dict:
    """THE SPREAD OF THE RECIPE AS IT IS: the standard deviation of the `constant` figures (one an order: the rows the
    same, the order drawn), with the sampling noise of one figure beside it (`noises`: each model's; the root of their
    mean square) and, from L5, the standard deviation of `published` over its seeds on the same attempts
    (`over_the_seeds`: its figure at each seed, rows AND order drawn). ORDER ALONE MAKES THE SPREAD when the first is at
    least half of the second; otherwise THE ROWS DRAWN MAKE MOST OF IT. `over_the_seeds` None (a seed's run is NOT
    THERE: `not_there` says which): the spread is given and the name is NOT READ."""
    if failed_checks:
        return _inconclusive(failed_checks, "the spread of the recipe as it is")
    own, noise = spread(constant), math.sqrt(statistics.fmean(noise ** 2 for noise in noises)) if noises else None
    found = {"failed_checks": [], "standard_deviation": None if own is None else round(own, 6), "trainings": len(constant),
             "sampling_noise_of_one_figure": None if noise is None else round(noise, 6)}
    given = f"the standard deviation of the {len(constant)} `{CONSTANT}` figures is {_said(own)} per attempt (the sampling noise of one figure: {_said(noise)})"
    of_l5 = spread(over_the_seeds) if over_the_seeds is not None else None
    if own is None or of_l5 is None:
        return {**found, "name": NOT_READ, "of_published_over_the_seeds": None,
                "reason": f"{given}; the standard deviation of L5's `published` over its seeds is NOT THERE ({not_there or 'fewer than two seeds, or fewer than two orders'}): "
                          "whether order alone makes the spread is not read"}
    alone = own >= HALF * of_l5
    return {**found, "name": ORDER_ALONE if alone else ROWS_DRAWN, "of_published_over_the_seeds": round(of_l5, 6), "seeds": len(over_the_seeds), "half_of_it": round(HALF * of_l5, 6),
            "reason": f"{given}; L5's `published` over its {len(over_the_seeds)} seeds, rows and order both drawn, has {_said(of_l5)} on the same attempts: the first is "
                      + ("at least" if alone else "less than") + " half of the second"}


def decay_lowers(failed_checks: Sequence[str], constant: Sequence[float], decay: Sequence[float]) -> dict:
    """DOES DECAY LOWER IT: the standard deviation of the `decay` figures against the `constant` one. DECAY LOWERS THE
    SPREAD when it is at most half; NOT SHOWN otherwise (and when the recipe as it is has no spread to lower). Said
    with it: what a ratio of two standard deviations of so few trainings must be to be beyond chance (the upper 5%
    point of F at as many degrees each: with four against four, 9.28, a ratio of about three)."""
    if failed_checks:
        return _inconclusive(failed_checks, "whether the decay lowers the spread")
    of_constant, of_decay = spread(constant), spread(decay)
    if of_constant is None or of_decay is None:
        return {"name": NOT_SHOWN, "failed_checks": [], "of_constant": None, "of_decay": None, "reason": "fewer than two trainings of a recipe: there is no spread to compare"}
    lowers = bool(of_constant > 0 and of_decay <= HALF * of_constant)
    degrees = len(constant) - 1
    beyond = F_CRITICAL.get(degrees) if len(decay) == len(constant) else None
    chance = (f"with {len(constant)} against {len(decay)} only a ratio of about {math.sqrt(beyond):.1f} between two standard deviations is beyond chance (F at {degrees} and "
              f"{degrees} degrees, {beyond:g})") if beyond else f"with {len(constant)} against {len(decay)} no ratio is said to be beyond chance here"
    return {"name": DECAY_LOWERS if lowers else NOT_SHOWN, "failed_checks": [], "of_constant": round(of_constant, 6), "of_decay": round(of_decay, 6),
            "of_decay_over_of_constant": round(of_decay / of_constant, 4) if of_constant else None, "half_of_constants": round(HALF * of_constant, 6),
            "f_at_equal_degrees": beyond, "ratio_beyond_chance": None if beyond is None else round(math.sqrt(beyond), 3),
            "reason": f"the standard deviation of the {len(decay)} `{DECAY}` figures is {_said(of_decay)} per attempt against {_said(of_constant)} of the `{CONSTANT}` ones: "
                      + ("at most" if lowers else "more than" if of_constant else "there is no spread to lower, so not at most") + f" half. {chance[0].upper()}{chance[1:]}"}


def its_level(failed_checks: Sequence[str], pairs: Sequence[tuple[float, float]]) -> dict:
    """AND ITS LEVEL: `decay_k` minus `constant_k` for each order (`pairs`: the two figures of one order, the decay's
    first), their mean and its t interval at one degree fewer than the orders (`T_CRITICAL`: 3.182 at 3). DECAY COSTS
    when the interval is below zero (one that ends at zero is not); otherwise its level is given and NAMED NOTHING."""
    if failed_checks:
        return _inconclusive(failed_checks, "the level of the decay")
    differences = [after - before for after, before in pairs]
    degrees = len(differences) - 1
    if degrees not in T_CRITICAL:
        raise ValueError(f"the t interval of {len(differences)} paired differences is not given here: its critical values are those of {sorted(count + 1 for count in T_CRITICAL)} orders")
    mean, critical = statistics.fmean(differences), T_CRITICAL[degrees]
    half_width = critical * statistics.stdev(differences) / math.sqrt(len(differences))
    low, high = mean - half_width, mean + half_width
    costs = bool(high < 0)
    return {"name": DECAY_COSTS if costs else NAMED_NOTHING, "failed_checks": [], "differences": [round(value, 6) for value in differences], "mean": round(mean, 6),
            "low": round(low, 6), "high": round(high, 6), "degrees_of_freedom": degrees, "critical_value": critical,
            "reason": f"`{DECAY}` minus `{CONSTANT}` over the {len(differences)} orders is {mean:+.5f} per attempt on average, t interval at {degrees} degrees "
                      f"[{low:+.5f}, {high:+.5f}]: " + ("it is below zero" if costs else "it is not below zero: the level is given and named nothing")}


def averaging(failed_checks: Sequence[str], of_the_average: float, noise: float, of_its_models: Sequence[float], recipe: str) -> dict:
    """AVERAGING, for one averaged model: its figure against the mean of its models' figures and against the best of
    them. AVERAGING GIVES AT LEAST THE MEAN OF ITS MODELS when it is not below that mean by more than twice its own
    sampling noise (`noise`); otherwise it gives less (the outcome the section does not name)."""
    if failed_checks:
        return _inconclusive(failed_checks, f"the averaged adapter `{mean_of(recipe)}`")
    of_the_mean, best = statistics.fmean(of_its_models), max(of_its_models)
    allowed = NOISES * noise
    gives = bool(of_the_mean - of_the_average <= allowed)
    return {"name": AT_LEAST_THE_MEAN if gives else LESS_THAN_THE_MEAN, "failed_checks": [], "model": mean_of(recipe), "per_attempt": round(of_the_average, 6),
            "mean_of_its_models": round(of_the_mean, 6), "best_of_its_models": round(best, 6), "minus_the_mean": round(of_the_average - of_the_mean, 6),
            "minus_the_best": round(of_the_average - best, 6), "sampling_noise": round(noise, 6), "allowed_below_the_mean": round(allowed, 6),
            "reason": f"`{mean_of(recipe)}` has {of_the_average:.5f} per attempt against {of_the_mean:.5f}, the mean of its {len(of_its_models)} models, and {best:.5f}, the "
                      f"best of them: {of_the_average - of_the_mean:+.5f} against the mean, where {NOISES} times its sampling noise ({noise:.5f}) is allowed below it"}


def what_follows(lowers: Mapping, level: Mapping, of_the_average: Mapping) -> dict:
    """What follows, by the section's three cases, in its order. DECAY LOWERS THE SPREAD and does not cost: every later
    stage trains with it. Not that, and AVERAGING GIVES AT LEAST THE MEAN (`of_the_average`: of the averaged adapter of
    the recipe as it is, which a later stage would then train with): a later stage trains each model four times and
    measures the averaged adapter. Neither: the spread is the recipe's own at this size, and the choice needs a
    decision. Nothing is named when a read it stands on is INCONCLUSIVE."""
    names = (lowers["name"], level["name"], of_the_average["name"])
    if INCONCLUSIVE in names:
        return {"name": NOT_READ, "reason": "the reads it stands on are INCONCLUSIVE: nothing is named for what follows"}
    read = f"the decay reads {names[0]}, its level {names[1]}, and the averaged adapter of the recipe as it is {names[2]}"
    if names[0] == DECAY_LOWERS and names[1] != DECAY_COSTS:
        return {"name": WITH_THE_DECAY, "reason": f"{read}: every later stage trains with the decay, and L5's three kinds are trained again with it before L6 runs"}
    if names[2] == AT_LEAST_THE_MEAN:
        return {"name": AVERAGE_OF_FOUR, "reason": f"{read}: a later stage trains each model four times and measures the averaged adapter"}
    return {"name": THE_RECIPES_OWN, "reason": f"{read}: the spread is the recipe's own at this size (its learning rate, its length), and the choice (a lower learning rate, "
                                               "or more seeds an arm) needs a decision, with these figures"}
