"""L5s: the dose. Every own solve in stock that the model finds less than half the time, against the same number of
published proofs. Spec: docs/spec/ladder-loop.spec.md, "L5s: the dose. Every own solve in stock that the model
finds less than half the time, against the same number of published proofs". Pure.

Every model trained on the model's own proofs so far held 400 of them. In stock, found and Lean-checked already, are
every deep solve of L5's one search and every ordinary solve L5's own setting names. One task trains on ALL of them
beside as many published rows, and a bar at equal steps on published rows alone.

  the seed          the one the setting names, and the further ones once the escalation is recorded (`the_seed`,
                    `fixed_seeds`)
  the sets          `stock`: every deep solve, every ordinary solve and the first N rows of the seed's hash order of the
                    pretraining file, N the own rows; `published_stock`: the first 2N rows of that order and nothing
                    else (`the_sets`, held by `hold_the_sets`); `stock` HOLDS the rows of L5d's three sets at that seed
                    (`holds_the_seeds_rows`)
  the check         each set is its rule made again and held, and no published row carries text
                    (`the_sets_are_the_rule`)
  the reads         THE DOSE, the scout's read: `stock` minus `deep` of L5d's run of that seed (`the_dose`: at or above
                    zero ESCALATE; below zero by more than `A_COST` MORE OF ITS OWN PROOFS COSTS AT THIS SEED; between,
                    NOT SHOWN AT ONE SEED); two differences beside it, naming nothing (`beside`); each model against the
                    start model's stored first sampling, paired by problem (`against_the_start`); what follows
                    (`what_follows`)

A model's figure is its successes per attempt on all of G over the first sampling's attempts (`l5_noise.figure`). A
branch is decided on the whole numbers of successes and attempts, never on a rounded figure.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from typing import Collection, Mapping, Sequence

from rlvr_lean.domain.ladder_round.assembly import training_order
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ
from rlvr_lean.domain.ladder_round.l4_rows import PRETRAINING, rehearsal_rows
from rlvr_lean.domain.ladder_round.l4_rows2 import TEXT_FIELDS, _placed
from rlvr_lean.domain.ladder_round.l4b import rehearsal_of
from rlvr_lean.domain.ladder_round.l5 import DEEP, PUBLISHED, SHALLOW
from rlvr_lean.domain.ladder_round.l5 import MODELS as OF_L5D
from rlvr_lean.domain.ladder_round.l5_draws import ALL, PARTS, THE_PRIMARY, differences, on_the_parts, the_counts

STOCK, PUBLISHED_STOCK = "stock", "published_stock"
MODELS = (STOCK, PUBLISHED_STOCK)                           # the two models, in the order they are trained, averaged and measured
OWN = {DEEP: "every deep solve of the search", SHALLOW: "every ordinary solve the rule gives"}        # where an own row of `stock` is from (L5's origins), and what it holds of each
ESCALATE = "THE SOLVE RATE DOES NOT FALL WITH THE DOSE: ESCALATE"
COSTS = "MORE OF ITS OWN PROOFS COSTS AT THIS SEED"
NOT_SHOWN_AT_ONE = "NOT SHOWN AT ONE SEED"
DOSE = (ESCALATE, COSTS, NOT_SHOWN_AT_ONE, INCONCLUSIVE, NOT_READ)
FURTHER_SEEDS = "ESCALATE: THE FURTHER SEEDS OF THIS STAGE RUN, EACH AFTER L5d's OWN RUN OF THAT SEED"
NO_FURTHER_SEED = "NO FURTHER SEED: THE DOSE IS NOT THE LEVER AT THIS SIZE"
FOLLOWS = (FURTHER_SEEDS, NO_FURTHER_SEED, NOT_READ)
A_COST = Fraction(1, 100)                                   # per attempt: "below zero by more than 0.010 (twice the standard deviation above)"
STANDARD_DEVIATION = 0.0048                                 # per attempt: what two averaged models of the same rows differ by (the section's: the order's part and the sampling noise, each twice)
START = "the_start_model"                                   # the start model, as the counts against it name it


# -------------------------------------------------------------------------------------------------- the seed
def fixed_seeds(seeds: Sequence[int], more: Sequence[int], escalated: bool) -> tuple[int, ...]:
    """The seeds of this stage: the setting's; and the further ones once the escalation is recorded."""
    return (*seeds, *(more if escalated else ()))


def the_seed(seed: int, seeds: Sequence[int], more: Sequence[int], escalated: bool) -> None:
    """A task's seed is one the setting lists. Refused (ValueError): a further seed while the escalation is not recorded
    (the scout's read names it, and it is recorded in the setting); a seed that is in neither list."""
    if seed in more and not escalated:
        raise ValueError(f"seed {seed} is one of the further seeds {list(more)}, which run once the scout's read at {list(seeds)} names the escalation and the setting "
                         "records it (escalated is false): one seed is the scout, and the further ones are not queued before it is read")
    if seed not in fixed_seeds(seeds, more, escalated):
        raise ValueError(f"seed {seed} is none of this stage's seeds {list(seeds)}" + (f" and {list(more)}" if escalated else f" (nor of {list(more)}, which wait for the "
                         "escalation)") + ": its sets hold the rows of L5d's own run of its seed, and a seed outside the setting is no seed of this stage")


# -------------------------------------------------------------------------------------------------- the sets
def ordinary_of_the_rule(map_rows: Sequence[Mapping], least: int, most: int, not_among: Collection[str] = ()) -> list[str]:
    """EVERY ordinary solve as L5 defines one: the map's problems the start model solved in `least` to `most` of its
    map attempts (L5's own setting), none of them among `not_among` (the problems the search attempted and the
    held-aside ones, as L5's draw leaves them out). Their ids, in the map's order."""
    return [row["problem_id"] for row in map_rows if least <= row["resolved"] <= most and row["problem_id"] not in not_among]


def the_sets(deep: Sequence[Mapping], ordinary: Sequence[Mapping], pretraining: Sequence[Mapping], seed: int) -> dict[str, list[Mapping]]:
    """The two training sets, each in the arm's content-hash order (`assembly.training_order`). `deep`: one row of EVERY
    deep solve (`l5.solve_rows`; no cap). `ordinary`: one row of EVERY ordinary solve. `pretraining`: the rows the start
    model was pretrained on, in the file's order, WITHOUT their text. `seed`: the task's.

      stock             the own rows, N of them, and the first N rows of the hash order L4t's rehearsal draw uses
                        (`l4_rows.rehearsal_rows`: the order L5's and L5d's published rows are the first of)
      published_stock   the first 2N rows of that order, and nothing else

    Refused (ValueError) as `hold_the_sets` refuses, which holds what is built here from the outside."""
    count = len(deep) + len(ordinary)
    if not deep or not ordinary:
        raise ValueError(f"`{STOCK}` has {len(deep)} deep solves and {len(ordinary)} ordinary solves: it holds every one of both, at least 1 of each. Refused")
    sets = {STOCK: training_order([*deep, *ordinary, *rehearsal_rows(pretraining, count, seed)], seed),
            PUBLISHED_STOCK: training_order(rehearsal_rows(pretraining, 2 * count, seed), seed)}
    hold_the_sets(sets, pretraining, seed)
    return sets


def hold_the_sets(sets: Mapping[str, Sequence[Mapping]], pretraining: Sequence[Mapping], seed: int, deep_solves: Collection[str] | None = None,
                  ordinary_solves: Collection[str] | None = None) -> dict:
    """What the two sets are HELD to, each a refusal (ValueError); returns the counts. `stock`: N rows of the model's
    own (origin `deep` or `shallow`, at least 1 of each, no problem twice) and N published rows, the FIRST N of the hash
    order of this seed by id and by place in the file; `published_stock`: 2N published rows, the first 2N of that order,
    and no other row. No id twice; no own row of a problem the pretraining file holds; each set in the content-hash
    order of this seed. `deep_solves`, `ordinary_solves` (THE RULE MADE AGAIN, by the report): the problems the search
    solved and the ordinary solves the rule gives: `stock`'s own rows are of EVERY one of them and of no other."""
    own = [row for row in sets[STOCK] if row["origin"] != PRETRAINING]
    of_the_file = {model: rehearsal_of(sets[model]) for model in MODELS}
    by_origin = Counter(row["origin"] for row in own)
    count = len(own)
    sizes = {STOCK: (count, len(of_the_file[STOCK])), PUBLISHED_STOCK: (len(sets[PUBLISHED_STOCK]) - len(of_the_file[PUBLISHED_STOCK]), len(of_the_file[PUBLISHED_STOCK]))}
    if not by_origin[DEEP] or not by_origin[SHALLOW] or set(by_origin) - set(OWN) or sizes != {STOCK: (count, count), PUBLISHED_STOCK: (0, 2 * count)}:
        raise ValueError(f"the two sets hold (rows of the model's own, published rows) {sizes}, the own rows by origin {dict(by_origin)}: `{STOCK}` is N own rows (deep and "
                         f"ordinary solves, at least 1 of each) and N published rows, `{PUBLISHED_STOCK}` 2N published rows and no other. Refused")
    for model in MODELS:
        twice = sorted(key for key, times in Counter(row["id"] for row in sets[model]).items() if times > 1)
        if twice:
            raise ValueError(f"`{model}` holds {len(twice)} ids twice (first: {twice[0]}). Refused")
        if [row["id"] for row in sets[model]] != [row["id"] for row in training_order(sets[model], seed)]:
            raise ValueError(f"`{model}` does not stand in the content-hash order of this seed: every model here is trained in orders of its own rows. Refused")
    problems = Counter(row["problem_id"] for row in own)
    twice = sorted(problem for problem, times in problems.items() if times > 1)
    seen = sorted(set(problems) & {row["problem_id"] for row in pretraining})
    if twice or seen:
        raise ValueError(f"{len(twice)} problems have two own rows in `{STOCK}`, and {len(seen)} problems of its own rows are rows of the pretraining file (first: "
                         f"{(twice or seen)[0]}): one verified attempt of each solve, and none of a problem whose published proof a model here has seen. Refused")
    for model, wanted in ((STOCK, rehearsal_rows(pretraining, count, seed)), (PUBLISHED_STOCK, rehearsal_rows(pretraining, 2 * count, seed))):
        if sorted(map(_placed, of_the_file[model])) != sorted(map(_placed, wanted)):
            other = sorted(set(map(_placed, of_the_file[model])) ^ set(map(_placed, wanted)))
            raise ValueError(f"the {len(of_the_file[model])} published rows of `{model}` are not the first {len(wanted)} of the hash order of this seed among the "
                             f"{len(pretraining)} rows the start model was pretrained on (first that differs: {other[0][0]}). Refused")
    for origin, wanted in ((DEEP, deep_solves), (SHALLOW, ordinary_solves)):
        of = {row["problem_id"] for row in own if row["origin"] == origin}
        if wanted is not None and of != set(wanted):
            other = sorted(of ^ set(wanted))
            raise ValueError(f"`{STOCK}` holds {len(of)} rows of origin `{origin}` and the rule gives {len(set(wanted))} ({OWN[origin]}; first that differs: {other[0]}): it "
                             "holds every one and no other own row. Refused")
    return {"own_rows": count, "deep_solves": by_origin[DEEP], "ordinary_solves": by_origin[SHALLOW], "rows": {model: len(sets[model]) for model in MODELS}}


def holds_the_seeds_rows(sets: Mapping[str, Sequence[Mapping]], theirs: Mapping[str, Sequence[Mapping]]) -> dict:
    """THE SETS HOLD THAT SEED'S ROWS: every row of L5d's three sets at this seed (`theirs`, by L5's rule for the seed:
    its deep rows, its ordinary solves, its published rows) is a row of `stock` or of `published_stock`, by id: the own
    rows and the first D published rows in `stock`, the first 2D in `published_stock`. Refused (ValueError) otherwise:
    the two models would not stand beside that seed's three. Returns what was counted."""
    ids = {model: {row["id"] for row in sets[model]} for model in MODELS}
    missing = {(model, mine): sorted({row["id"] for row in theirs[model]} - ids[mine]) for model, mine in ((DEEP, STOCK), (SHALLOW, STOCK), (PUBLISHED, PUBLISHED_STOCK))}
    if any(missing.values()):
        raise ValueError("; ".join(f"{len(own)} rows of that seed's `{model}` are no rows of `{mine}` (first: {own[0]})" for (model, mine), own in missing.items() if own)
                         + ": the sets of this stage hold the rows of L5d's three sets at its seed. Refused")
    return {"what": f"every row of that seed's `{DEEP}` and `{SHALLOW}` is a row of `{STOCK}`, and every row of its `{PUBLISHED}` a row of `{PUBLISHED_STOCK}`, by id",
            "rows": {model: len(theirs[model]) for model in OF_L5D}}


def the_sets_are_the_rule(sets: Mapping[str, Sequence[Mapping]], pretraining: Sequence[Mapping], seed: int, deep_solves: Collection[str], ordinary_solves: Collection[str],
                          in_orders: Mapping[str, Sequence[Mapping]]) -> dict:
    """The first check: each set is its rule MADE AGAIN AND HELD (`hold_the_sets`, with the problems the search solved
    and the ordinary solves the rule gives, read again by the report), no published row of one carries text, and each of
    a model's four sets (`in_orders`: by trained model `<model>_<order seed>`) is its rows in that order seed's order."""
    with_text = sorted(row["id"] for own in sets.values() for row in own if row["origin"] == PRETRAINING and any(field in row for field in TEXT_FIELDS))
    try:
        counted, why = hold_the_sets(sets, pretraining, seed, deep_solves, ordinary_solves), None
    except (ValueError, KeyError) as error:
        counted, why = None, str(error)
    out_of_order = []
    for name, own in in_orders.items():
        model, _, order = name.rpartition("_")
        if model not in sets or not order.isdigit() or [row["id"] for row in own] != [row["id"] for row in training_order(sets[model], int(order))]:
            out_of_order.append(name)
    return {"what": f"each set, as stored, is its rule made again: `{STOCK}` holds one row of every deep solve of the search and of every ordinary solve the rule gives, no "
                    f"other own row, and the first N rows of the hash order of this seed; `{PUBLISHED_STOCK}` the first 2N and nothing else; no published row carries "
                    "text; and each of a model's four sets is its rows in its order seed's order",
            "counted": counted, "refused": why, "published_rows_with_text": len(with_text), "sets_not_in_their_order_seeds_order": out_of_order,
            "passes": counted is not None and not with_text and not out_of_order}


# ------------------------------------------------------------------------------------------------- the reads
def _exact(own: Mapping) -> Fraction | None:
    return Fraction(own["successes"], own["attempts"]) if own["attempts"] else None


def of_l5d(entry: Mapping, seed: int, recipe: str, measurement: Mapping) -> dict:
    """What L5d's run of this seed hands the reads: its three models' figures on all of G and its own primary's
    difference, WHEN ITS RUN IS THERE AND READ ALIKE. `entry`: what its report says of its own seed (`of_this_seed`), or
    `{"there": False, "why": ...}`. NOT READ, with why, and never refused: a run that is not there, of another seed or
    recipe, INCONCLUSIVE or not to be read, or measured otherwise than this run (`measurement`: the sampling, its
    attempts, its sampling seed, the problems)."""
    if not entry.get("there", True):
        return {"read": False, "why": entry.get("why"), "run": entry.get("run")}
    why = (f"it is a report of seed {entry.get('seed')}" if entry.get("seed") != seed else
           f"its recipe is `{entry.get('recipe')}` and this run's `{recipe}`" if entry.get("recipe") != recipe else
           "its report is not to be read" if not entry.get("ok", True) else
           f"it reads {INCONCLUSIVE} ({'; '.join(entry.get('failed_checks') or [])})" if entry.get("inconclusive") or not entry.get("figures") else
           f"it was measured otherwise ({entry.get('the_measurement')}) than this run ({dict(measurement)})" if entry.get("the_measurement") != dict(measurement) else None)
    if why:
        return {"read": False, "why": f"L5d's run of seed {seed} does not stand beside this one: {why}", "run": entry.get("run")}
    figures = {model: {key: entry["figures"][model][ALL][key] for key in ("successes", "attempts", "per_attempt", "per_1000", "sampling_noise")} for model in OF_L5D}
    return {"read": True, "run": entry.get("run"), "figures": figures, "the_primarys_difference": round(differences(entry["figures"])[THE_PRIMARY], 6)}


def the_dose(failed_checks: Sequence[str], stock: Mapping | None, of_theirs: Mapping) -> dict:
    """THE DOSE, the scout's read: `stock` minus `deep` of L5d's run of this seed, per attempt on all of G. At or above
    zero: THE SOLVE RATE DOES NOT FALL WITH THE DOSE: ESCALATE. Below zero by MORE than `A_COST`: MORE OF ITS OWN PROOFS
    COSTS AT THIS SEED. Between (a difference of exactly minus `A_COST` is between): NOT SHOWN AT ONE SEED. `stock`: its
    figure on all of G (whole numbers of successes and attempts). `of_theirs`: `of_l5d`'s. INCONCLUSIVE when a check of
    this run failed; NOT READ, and said, when L5d's run of the seed is not there to be read."""
    if failed_checks:
        return {"name": INCONCLUSIVE, "reason": f"a check of this run failed ({'; '.join(failed_checks)}): the dose is not read"}
    if not of_theirs.get("read"):
        return {"name": NOT_READ, "reason": f"`{STOCK}` minus `{DEEP}` of L5d's run of this seed is NOT READ: {of_theirs.get('why')}"}
    deep = of_theirs["figures"][DEEP]
    difference = _exact(stock) - _exact(deep)
    name = ESCALATE if difference >= 0 else COSTS if difference < -A_COST else NOT_SHOWN_AT_ONE
    said = {ESCALATE: "at or above zero: seeds of this stage beyond the scout's run (three seeds conclude)",
            COSTS: f"below zero by more than {1000 * float(A_COST):g} per 1,000 (twice the standard deviation of two averaged models of the same rows, "
                   f"{1000 * STANDARD_DEVIATION:g}): no further seed",
            NOT_SHOWN_AT_ONE: f"below zero by at most {1000 * float(A_COST):g} per 1,000: no further seed is queued by this read"}[name]
    return {"name": name, "difference": round(float(difference), 6), "stock": {key: stock[key] for key in ("successes", "attempts")},
            "deep": {key: deep[key] for key in ("successes", "attempts")}, "a_cost": float(A_COST), "standard_deviation_of_two_averaged_models": STANDARD_DEVIATION,
            "reason": f"`{STOCK}` minus `{DEEP}` of L5d's run of this seed, per 1,000 attempts on all of G: {1000 * float(difference):+.2f} ({1000 * float(_exact(stock)):.2f} "
                      f"against {1000 * float(_exact(deep)):.2f}): {said}"}


def beside(figures: Mapping[str, Mapping], of_theirs: Mapping) -> dict:
    """The two differences that stand beside the dose and name nothing, per attempt on all of G. ITS OWN PROOFS AT FOUR
    TIMES THE DOSE: `stock` minus `published_stock`, L5d's primary at this dose, with that seed's own primary beside
    when its run is read. THE STEPS WITHOUT ITS OWN PROOFS: `published_stock` minus `published` of L5d's run of this
    seed (the same change of steps with no own proof), NOT READ when that run is not. `figures`: the two models' on all
    of G."""
    own = {model: _exact(figures[model]) for model in MODELS}
    read = bool(of_theirs.get("read"))
    theirs = _exact(of_theirs["figures"][PUBLISHED]) if read else None
    return {"own_proofs_at_this_dose": {"what": f"`{STOCK}` minus `{PUBLISHED_STOCK}`: L5d's primary at this dose. One seed names nothing",
                                        "difference": round(float(own[STOCK] - own[PUBLISHED_STOCK]), 6),
                                        "l5ds_primary_at_this_seed": of_theirs.get("the_primarys_difference") if read else None},
            "the_steps_without_its_own_proofs": {"what": f"`{PUBLISHED_STOCK}` minus `{PUBLISHED}` of L5d's run of this seed: the same order's rows, more of them, and no "
                                                         "own proof", "read": read, "why": None if read else of_theirs.get("why"),
                                                 "difference": round(float(own[PUBLISHED_STOCK] - theirs), 6) if read else None}}


def against_the_start(rows: Sequence[Mapping], start_rows: Sequence[Mapping], parts: Mapping[str, Sequence[str]], long_ids: Collection[str]) -> dict:
    """One model against the start model's stored first sampling, PAIRED BY PROBLEM: successes per attempt on each part
    of G (all of it; the goal problems the start model solves in that sampling; G'), the model's, the start model's and
    their difference; the goal problems solved in the attempts, gained and lost; and the same of the long ones
    (`long_ids`: a shortest published proof of 8 lines or more). `rows`, `start_rows`: per-problem rows."""
    own, of_the_start = on_the_parts(rows, parts), {row["problem_id"]: row for row in start_rows}
    per_attempt = {}
    for part in PARTS:
        theirs = [of_the_start[problem_id] for problem_id in parts[part] if problem_id in of_the_start]
        attempts = sum(row["episodes"] for row in theirs)
        start = sum(row["resolved"] for row in theirs) / attempts if attempts else None
        mine = own[part]["successes"] / own[part]["attempts"] if own[part]["attempts"] else None
        per_attempt[part] = {"problems": len(parts[part]), "model": None if mine is None else round(mine, 6), "start": None if start is None else round(start, 6),
                             "difference": None if None in (mine, start) else round(mine - start, 6)}
    counts = the_counts({START: start_rows, "model": rows}, long_ids, against=START)
    return {"per_attempt": per_attempt, **counts["model"], "solved_by_the_start": counts[START]["solved"], "long_solved_by_the_start": counts[START]["long"]["solved"]}


def what_follows(dose: Mapping, more: Sequence[int]) -> dict:
    """What follows, fixed before the run. ESCALATE: the further seeds of this stage, each after L5d's own run of that
    seed; over the three, `stock` minus `published_stock` is read with L5d's primary rule and the dose with an interval
    over seeds. COSTS, or NOT SHOWN: no further seed; the dose is not the lever at this size, and that goes into the
    decision on the full-size cycle. Nothing is named while the dose is not read."""
    name = dose["name"]
    if name in (INCONCLUSIVE, NOT_READ):
        return {"name": NOT_READ, "reason": f"the dose reads {name}: nothing is named for what follows"}
    if name == ESCALATE:
        return {"name": FURTHER_SEEDS, "seeds": list(more),
                "reason": f"the dose reads {name}: seeds {list(more)} of this stage, each after L5d's own run of that seed; over the three, `{STOCK}` minus "
                          f"`{PUBLISHED_STOCK}` is read with L5d's primary rule and the dose with an interval over seeds. The averaged adapter that reads highest on G's "
                          "first sampling is then the one measured in full (the rungs, the second sampling, the held-aside set), a step that is not built here"}
    return {"name": NO_FURTHER_SEED, "reason": f"the dose reads {name}: no further seed; the dose is not the lever at this size, and that goes into the decision on "
                                               "the full-size cycle with its interval"}
