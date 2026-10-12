"""L6b: a second round, from the best model. Does the improved model's own search yield proofs that add? Spec:
docs/spec/ladder-loop.spec.md, "L6b: a second round, from the best model". Pure.

B is L5f's stored averaged `published` of the seed (the best model on G's first sampling). B makes 64 attempts at each
problem of L6's second search; a problem it solves is a NEW SOLVE. Two models are trained from B with the averaged
recipe, each on as many rows as L5f's `published` set holds, and each is read against L5c's stored `again` (THE BAR:
from the same B, the next published rows of the same order, the same recipe).

  the setting       `hold_the_setting`: the start is L5f's `published`, the bar L5c's `again`, the seeds are L5c's
  the seed          `the_seed`: a seed the setting names
  K                 `the_dose`: the smaller of half the rows of a set, the old own rows L5d's `deep` set of the seed
                    holds, and the new solves (the section's "the smaller of 400 and N": half of 800 is 400, and L5d's
                    `deep` set holds 400 own rows)
  the sets          `published_next`: the NEXT rows of the seed's order of the pretraining file (rows 801 to 1,600:
                    L5c's `again` set, HELD to it by `hold_the_next`). `the_sets`: `new_own` is K new solves (a seeded
                    draw when there are more, `l5.capped`) and the first rows of those next ones; `old_own` is the first
                    K old own rows in their stored order and THE SAME published rows. HELD by `hold_the_sets`; made again
                    by the report (`the_sets_are_the_rule`)
  the searched set  `the_searched_is_l6s`: the search's stored problems are L6's search 2 made again (in its order, with
                    its parts), none held out, of the goal set, held aside or solved by round 1's search, and the search
                    sampled exactly those, each once, with the attempts it was prepared for
  L5c's run         what its report says of its seed, read for a reason it cannot carry this stage (`of_l5c`)
  the reads         THE PRIMARY (`the_primary`: `new_own` minus the bar) and WHOSE PROOFS (`whose_proofs`: `new_own`
                    minus `old_own`), each at or above plus `A_LINE`, below zero by more than it, or between; the
                    search's yield (`the_yield`); what follows (`what_follows`)

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
from rlvr_lean.domain.ladder_round.l4_rows2 import TEXT_FIELDS
from rlvr_lean.domain.ladder_round.l5 import DEEP, capped
from rlvr_lean.domain.ladder_round.l5_cont import A_LINE, AGAIN, BAR as L5FS_PUBLISHED
from rlvr_lean.domain.ladder_round.l5_cont import START as SECOND_FULL
from rlvr_lean.domain.ladder_round.l5_draws import four_orders, trained_of
from rlvr_lean.domain.ladder_round.l5_noise import ORDERS, numbered
from rlvr_lean.domain.ladder_round.l5_stock import NOT_SHOWN_AT_ONE
from rlvr_lean.domain.ladder_round.l6 import PART_A, PART_B

NEW_OWN, OLD_OWN = "new_own", "old_own"
MODELS = (NEW_OWN, OLD_OWN)                                 # the two trained models, in the order they are made and measured
START = L5FS_PUBLISHED                                      # B: L5f's stored averaged `published` of the seed's run; it makes the search and both models are trained from it
BAR = AGAIN                                                 # L5c's stored `again` of the seed's run: the bar of the primary read
NEW = "new"                                                 # where a row that is one of B's new solves is from (its origin)
OLD = DEEP                                                  # ... and one of `pre_r64`'s deep solves, as L5d's `deep` set holds it
OWN_OF = {NEW_OWN: NEW, OLD_OWN: OLD}
TRAINED = tuple(trained_of(model, order) for model in MODELS for order in ORDERS)       # the eight trainings, in the order they are made
KEYS = ("start", "bar", "seeds")                            # the setting's keys: one reader, and nothing else is read of it
ADD = "THE MODEL'S OWN NEW PROOFS ADD FROM THE BEST MODEL"
COST = "THEY COST THERE AT THIS SEED"
PRIMARY = (ADD, NOT_SHOWN_AT_ONE, COST, INCONCLUSIVE, NOT_READ)
TEACH_MORE = "PROOFS THE STRONGER MODEL FOUND TEACH MORE THAN THE WEAKER MODEL'S"
TEACH_LESS = "THEY TEACH LESS"
WHOSE = (TEACH_MORE, NOT_SHOWN_AT_ONE, TEACH_LESS, INCONCLUSIVE, NOT_READ)
THE_CYCLE_TURNS = "THE CYCLE TURNS FROM THE BEST MODEL: TWO FURTHER SEEDS OF `new_own` AND OF THE BAR BEFORE ANYTHING IS SAID OVER SEEDS"
SAID_AS_IT_IS = "NOT SHOWN: SAID AS IT IS, WITH THE SEARCH'S YIELD; A THIRD ROUND IS NOT RUN ON IT"
TO_A_DECISION = "THEY COST: TO A DECISION WITH L5f's READ; THE FULL-SIZE CYCLE IS NOT PROPOSED ON IT"
THE_SEARCH_IS_THE_FINDING = "TOO FEW NEW SOLVES: NOTHING IS TRAINED, AND THE SEARCH IS THE FINDING"
FOLLOWS = (THE_CYCLE_TURNS, SAID_AS_IT_IS, TO_A_DECISION, THE_SEARCH_IS_THE_FINDING, NOT_READ)


# ------------------------------------------------------------------------------------- the setting, the seed
def hold_the_setting(settings: Mapping, of_l5c: Collection[int]) -> tuple[int, ...]:
    """`ladder_loop.l6b` as it may be read; returns its seeds. Refused (ValueError): a key that is none of the
    setting's, or one missing; a start that is not L5f's `published`; a bar that is not L5c's `again`; no seed; seeds
    that are not distinct whole numbers of at least 0; a seed that is none of L5c's (`of_l5c`: the seeds L5c's own
    setting fixes, whose run of that seed holds the bar)."""
    if sorted(settings) != sorted(KEYS):
        raise ValueError(f"ladder_loop.l6b holds the keys {sorted(settings)}: it is read for {list(KEYS)} and for nothing else")
    start, bar, seeds = settings["start"], settings["bar"], settings["seeds"]
    if start != START:
        raise ValueError(f"ladder_loop.l6b.start is {start!r}: this stage is specified from L5f's stored averaged `{START}` of the seed alone (B: it makes the search, and "
                         "both models are trained from it)")
    if bar != BAR:
        raise ValueError(f"ladder_loop.l6b.bar is {bar!r}: the bar is L5c's stored `{BAR}` of the seed alone (from the same model, the next published rows of the same "
                         "order, the same recipe)")
    if (not isinstance(seeds, (list, tuple)) or not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds) or set(seeds) - set(of_l5c)):
        raise ValueError(f"ladder_loop.l6b.seeds is {seeds!r}: distinct whole numbers of at least 0, at least one, each a seed L5c's setting fixes ({sorted(of_l5c)}), whose "
                         "own run of that seed holds the bar")
    return tuple(seeds)


def the_seed(seed: int, seeds: Sequence[int]) -> None:
    """A task's seed is one the setting names; refused (ValueError) otherwise. A further seed needs L5f's `published`
    and L5c's `again` at that seed, which no run holds."""
    if seed not in seeds:
        raise ValueError(f"this task's seed is {seed} and ladder_loop.l6b.seeds is {list(seeds)}: a task of another seed is not run (it needs L5f's `{START}` and L5c's "
                         f"`{BAR}` of that seed, which no run holds)")


# ---------------------------------------------------------------------------------------------------- the sets
def the_dose(first: int, old: int, new: int) -> int:
    """K, the rows of the model's own in each set: the smaller of HALF the rows of a set (`first`: the rows of L5f's
    `published` set, 800), the old own rows L5d's `deep` set of the seed holds (`old`: 400) and the new solves (`new`:
    N). The section's "the smaller of 400 and N". Refused (ValueError) when it would be no row."""
    dose = min(first // 2, old, new)
    if dose < 1:
        raise ValueError(f"a set of {first} rows, {old} old own rows and {new} new solves give K = {dose}: each set holds at least one row of the model's own and as many "
                         "published rows at least. Refused")
    return dose


def published_next(pretraining: Sequence[Mapping], seed: int, first: int) -> list[dict]:
    """The NEXT `first` rows of the seed's order of the pretraining file (`l4_rows.rehearsal_rows`: rows `first` + 1 to
    2 x `first` of the order L5f's `published` set is the first rows of), in the draw's own order: the rows of L5c's
    `again` set. `pretraining`: the rows the first start was pretrained on, in the file's order, WITHOUT their text.
    Refused (ValueError): no row asked; a file that holds fewer rows."""
    if first < 1:
        raise ValueError(f"the next rows of the order are asked after {first} rows: L5f's `{START}` set holds at least one row. Refused")
    return rehearsal_rows(pretraining, 2 * first, seed)[first:]


def _differs(own: Sequence[str], theirs: Sequence[str] | None) -> dict | None:
    """Where two lists of ids first differ; None when they are one list."""
    if theirs is not None and list(own) == list(theirs):
        return None
    place = next((index for index, (mine, other) in enumerate(zip(own, theirs or [])) if mine != other), min(len(own), len(theirs or [])))
    return {"rows": len(own), "rows_of_theirs": None if theirs is None else len(theirs), "place": place,
            "here": own[place] if place < len(own) else None, "there": theirs[place] if theirs is not None and place < len(theirs) else None}


def _with_text(rows: Sequence[Mapping]) -> list[str]:
    """The ids of the rows that are no published row by id and place alone: of another origin, or carrying text."""
    return sorted(row["id"] for row in rows if row["origin"] != PRETRAINING or any(field in row for field in TEXT_FIELDS))


def hold_the_next(next_rows: Sequence[Mapping], seed: int, of_l5c: Sequence[str] | None) -> dict:
    """The next rows ARE L5c's `again` set: in the content-hash order of the seed they are, id for id and place for
    place, the set L5c's run of the seed stored as `again` (`of_l5c`: its ids, in its order); and every one is a
    published row, by id and place, never with text. Refused (ValueError) otherwise. Returns the counts."""
    differs = _differs([row["id"] for row in training_order(next_rows, seed)], of_l5c)
    if differs:
        raise ValueError(f"the next rows of the seed's order are not the set L5c's run of this seed stored as `{BAR}`, id for id ({differs['rows']} rows here and "
                         f"{differs['rows_of_theirs']} there; first that differs at place {differs['place']}: {differs['here']} here, {differs['there']} there). Refused")
    other = _with_text(next_rows)
    if other:
        raise ValueError(f"{len(other)} of the next rows of the seed's order are no published row or carry text (first: {other[0]}): a published row is stored by id and "
                         "place, never with text. Refused")
    return {"what": f"the next rows of the seed's order are the set L5c's run stored as `{BAR}`, id for id and place for place; every one a published row, by id and place and "
                    "without text", "rows": len(next_rows), "passes": True}


def the_sets(new: Sequence[Mapping], old: Sequence[Mapping], next_rows: Sequence[Mapping], seed: int, dose: int) -> dict[str, list[Mapping]]:
    """The two sets, each in the arm's content-hash order of the task's seed (`assembly.training_order`), each of as many
    rows as there are next rows (800). `new`: B's new solves, one row a problem solved, in the search's order. `old`:
    the own rows of L5d's `deep` set of the seed, in its stored order. `next_rows`: `published_next`. `dose`: K.

      new_own   K of the new solves (every one when there are K; otherwise a seeded draw of K, `l5.capped`) and the
                first rows of the next ones, as many as fill the set (rows 801 to 1,600 - K of the order)
      old_own   the FIRST K old own rows, in their stored order, and THE SAME published rows

    Refused (ValueError): a dose of no row, of more rows than either side holds, or of more than half a set."""
    size = len(next_rows)
    if dose < 1 or dose > len(new) or dose > len(old) or 2 * dose > size:
        raise ValueError(f"K is {dose} with {len(new)} new solves, {len(old)} old own rows and sets of {size} rows: K is at least 1, at most each of the two, and at most "
                         "half a set. Refused")
    published = list(next_rows[:size - dose])
    return {NEW_OWN: list(training_order([*capped(new, dose, seed), *published], seed)), OLD_OWN: list(training_order([*old[:dose], *published], seed))}


def _own(rows: Sequence[Mapping]) -> list[Mapping]:
    return [row for row in rows if row["origin"] != PRETRAINING]


def _published(rows: Sequence[Mapping]) -> list[Mapping]:
    return [row for row in rows if row["origin"] == PRETRAINING]


def hold_the_sets(sets: Mapping[str, Sequence[Mapping]], next_rows: Sequence[Mapping], seed: int, dose: int) -> dict:
    """What the two sets are HELD to, each a refusal (ValueError); returns the counts. Each set holds as many rows as
    there are next rows and no id twice, and stands in the content-hash order of the seed. `new_own`: K rows of origin
    `new` and the rest published; `old_own`: K rows of origin `deep` and the rest published. THE TWO HOLD THE SAME
    PUBLISHED ROWS, by id and place: the first rows of the next ones. No published row carries text. No problem is in
    both sets' own rows (a new solve is of a problem round 1 left unsolved or never drew; an old one is one it solved)."""
    size = len(next_rows)
    sizes = {model: (len(_own(sets[model])), len(_published(sets[model]))) for model in MODELS}
    if dose < 1 or sizes != {model: (dose, size - dose) for model in MODELS}:
        raise ValueError(f"the two sets hold (rows of the model's own, published rows) {sizes}: each is K = {dose} rows of the model's own and {size - dose} published "
                         "rows. Refused")
    for model in MODELS:
        strangers = [row["id"] for row in _own(sets[model]) if row["origin"] != OWN_OF[model]]
        twice = sorted(key for key, times in Counter(row["id"] for row in sets[model]).items() if times > 1)
        if strangers or twice:
            raise ValueError(f"`{model}` holds {len(strangers)} rows of another origin than `{OWN_OF[model]}` and {len(twice)} ids twice (first: {(strangers or twice)[0]}). "
                             "Refused")
        if [row["id"] for row in sets[model]] != [row["id"] for row in training_order(sets[model], seed)]:
            raise ValueError(f"`{model}` does not stand in the content-hash order of this seed: every model here is trained in an order of its set. Refused")
    placed = lambda rows: sorted((row["id"], row["problem_id"], row.get("row_of_the_file")) for row in rows)      # noqa: E731
    wanted = placed(next_rows[:size - dose])
    not_the_next = [model for model in MODELS if placed(_published(sets[model])) != wanted]
    with_text = sorted({key for model in MODELS for key in _with_text(_published(sets[model]))})
    both = sorted({row["problem_id"] for row in _own(sets[NEW_OWN])} & {row["problem_id"] for row in _own(sets[OLD_OWN])})
    if not_the_next or with_text or both:
        raise ValueError(f"the published rows of {not_the_next} are not the first {size - dose} of the next rows of the seed's order (the two sets hold THE SAME published "
                         f"rows); {len(with_text)} published rows carry text; {len(both)} problems are in both sets' own rows (first: {(with_text or both or [None])[0]}). "
                         "Refused")
    return {"what": f"each set is K rows of the model's own and the first rows of the next ones of the seed's order; the two hold the same published rows, by id and "
                    "place and without text; no problem is in both sets' own rows", "K": dose, "rows": size, "published_rows": size - dose, "passes": True}


def in_their_orders(sets: Mapping[str, Sequence[Mapping]], smoke: bool = False) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """Each set in the content-hash order of each ORDER SEED (`l5_draws.four_orders`, as L5d trains), numbered: by
    trained model `<model>_<order seed>`. Two orders of a set that are one are refused there (ValueError); `smoke` (a
    SMOKE run alone, whose handful of rows has few orders): trained all the same. Returns (the ordered sets; by model,
    what coincides)."""
    in_orders, coincide = {}, {}
    for model in MODELS:
        try:
            by_order, coincide[model] = four_orders(sets[model], smoke)
        except ValueError as error:
            raise ValueError(f"`{model}`: {error}") from error
        in_orders.update({trained_of(model, order): numbered(by_order[order]) for order in ORDERS})
    return in_orders, coincide


def the_sets_are_the_rule(sets: Mapping[str, Sequence[Mapping]], in_orders: Mapping[str, Sequence[Mapping]], new: Sequence[Mapping], old: Sequence[Mapping],
                          pretraining: Sequence[Mapping], seed: int, of_l5c: Sequence[str] | None) -> dict:
    """The check of the sets: each, as stored, is its rule MADE AGAIN. `sets`: the two stored sets. `in_orders`: by
    trained model `<model>_<order seed>`, its stored set. `new`: B's new solves made again from the stored search.
    `old`: the own rows of L5d's `deep` set, as that run stores it NOW. `pretraining`: the rows the first start was
    pretrained on. `of_l5c`: the ids L5c's run stores as `again` NOW (None where it holds none). Each set is `the_sets`
    on those, id for id and place for place; the two hold the same published rows, none with text; the next rows are
    L5c's `again` set; and each of a model's four sets is its rows in that order seed's order."""
    size = len(sets[NEW_OWN])
    placed = lambda own: [(row["id"], row["problem_id"], row.get("row_of_the_file")) for row in own]      # noqa: E731
    dose, not_again = None, None
    try:
        next_rows = published_next(pretraining, seed, size)
        not_again = _differs([row["id"] for row in training_order(next_rows, seed)], of_l5c)
        dose = the_dose(size, len(old), len(new))
        again, why = the_sets(new, old, next_rows, seed, dose), None
        hold_the_sets(sets, next_rows, seed, dose)
        not_the_rule = [model for model in MODELS if placed(sets[model]) != placed(again[model])]
    except (ValueError, KeyError) as error:
        why, not_the_rule = str(error), list(MODELS)
    with_text = sorted({key for model in MODELS for key in _with_text(_published(sets[model]))})
    shared = sorted(placed(_published(sets[NEW_OWN]))) == sorted(placed(_published(sets[OLD_OWN])))
    out_of_order = []
    for name, own in in_orders.items():
        model, _, order = name.rpartition("_")
        if model not in sets or not order.isdigit() or [row["id"] for row in own] != [row["id"] for row in training_order(sets[model], int(order))]:
            out_of_order.append(name)
    return {"what": f"each set, as stored, is its rule made again: K rows of the model's own (`{NEW_OWN}`: B's new solves, drawn by the seed when there are more than K; "
                    f"`{OLD_OWN}`: the first K own rows of L5d's `{DEEP}` set) and the first rows of the next ones of the seed's order, which are L5c's `{BAR}` set; the two "
                    "hold the same published rows; no published row carries text; and each of a model's four sets is its rows in its order seed's order",
            "rows": {model: len(sets[model]) for model in MODELS}, "K": dose, "new_solves": len(new), "old_own_rows": len(old), "refused": why,
            "sets_that_are_not_the_rule": not_the_rule, "published_rows_with_text": len(with_text), "the_two_hold_the_same_published_rows": shared,
            "the_next_rows_are_l5cs_again": not_again is None and why is None, "first_that_differs_from_l5cs": not_again, "sets_not_in_their_order_seeds_order": out_of_order,
            "passes": why is None and not not_the_rule and not with_text and shared and not_again is None and not out_of_order}


# ------------------------------------------------------------------------------------------ the searched set
def the_searched_is_l6s(stored: Sequence[Mapping], again: Sequence[Mapping] | None, refused: str | None, results: Sequence[Mapping], attempts: int,
                        heldout: Collection[str], goal_ids: Collection[str], aside: Collection[str], solved_one: Collection[str]) -> dict:
    """The check of the searched set: it is L6's SEARCH 2 set. `stored`: the search's problems as this run stored them
    (`problem_id`, `part`). `again`: L6's own rule made again on the stored runs as they are now (None when it refused,
    with `refused` its words). `results`: the search's per-problem rows. `attempts`: the attempts a problem the search
    was prepared for. Held: the stored problems are the rule's, in its order and with its parts; none is held out, a
    goal problem, a held-aside problem or one round 1's search solved (`solved_one`); and the search sampled exactly
    those problems, each once, with those attempts."""
    placed = lambda rows: [(row["problem_id"], row["part"]) for row in rows]      # noqa: E731
    ids = [row["problem_id"] for row in stored]
    of_the_rule = again is not None and placed(stored) == placed(again)
    barred = {"held_out": sorted(set(ids) & set(heldout)), "of_the_goal_set": sorted(set(ids) & set(goal_ids)), "of_the_held_aside_set": sorted(set(ids) & set(aside)),
              "solved_by_search_one": sorted(set(ids) & set(solved_one))}
    sampled = [row["problem_id"] for row in results]
    as_prepared = bool(ids) and sorted(sampled) == sorted(ids) and len(set(ids)) == len(ids)
    other = sorted({row["episodes"] for row in results} - {attempts})
    return {"what": "the searched set is L6's SEARCH 2 set, made again by L6's own rule on the stored runs: what round 1's search attempted and left unsolved, then the "
                    "candidates neither draw took; no problem of it is held out, a goal problem, a held-aside problem or one round 1's search solved; and the search "
                    "sampled exactly those problems, each once, with the attempts it was prepared for",
            "problems": len(ids), "by_part": {part: sum(row["part"] == part for row in stored) for part in (PART_A, PART_B)},
            "problems_of_the_rule": None if again is None else len(again), "is_the_rule_made_again": of_the_rule, "refused": refused,
            **{key: len(found) for key, found in barred.items()}, "first": [problem_id for found in barred.values() for problem_id in found][:5],
            "sampled": len(sampled), "sampled_as_prepared": as_prepared, "attempts_a_problem_that_are_not_the_settings": other,
            "passes": of_the_rule and not any(barred.values()) and as_prepared and not other}


# ------------------------------------------------------------------------------------------------ L5c's run
def of_l5c(entry: Mapping, seed: int, recipe: str, measurement: Mapping, though_inconclusive: bool = False) -> str | None:
    """Why L5c's run of this seed cannot carry this stage, or None. `entry`: what its report says of its own seed
    (`of_this_seed`), or `{"there": False, "why": ...}`. A run that is not there, of another seed, start or recipe, not
    to be read, INCONCLUSIVE, or measured otherwise than this run (`measurement`: the sampling, its attempts, its
    sampling seed, the problems). `though_inconclusive` (a SMOKE run alone): a report that reads INCONCLUSIVE is read
    all the same; one not to be read never is."""
    if not entry.get("there", True):
        return entry.get("why")
    return (f"it is a report of seed {entry.get('seed')}" if entry.get("seed") != seed else
            f"it is a run from `{entry.get('start')}` and L5c's models are from `{SECOND_FULL}`, whose `{START}` of L5f's run is this stage's start" if entry.get("start") != SECOND_FULL else
            f"its recipe is `{entry.get('recipe')}` and this run's `{recipe}`" if entry.get("recipe") != recipe else
            "its report is not to be read" if not entry.get("ok", True) else
            f"it reads {INCONCLUSIVE} ({'; '.join(entry.get('failed_checks') or [])})" if (entry.get("inconclusive") or not entry.get("figures")) and not though_inconclusive else
            f"it was measured otherwise ({entry.get('the_measurement')}) than this run ({dict(measurement)})" if entry.get("the_measurement") != dict(measurement) else None)


# ------------------------------------------------------------------------------------------------- the reads
THE_BARS_SIDE = f"L5c's stored {BAR}"       # the key of the bar's whole numbers in the primary read


def _exact(own: Mapping) -> Fraction | None:
    return Fraction(own["successes"], own["attempts"]) if own["attempts"] else None


def _whole(own: Mapping) -> dict:
    return {"successes": own["successes"], "attempts": own["attempts"]}


def _shown(key: str) -> str:
    """A side of a read, as a line says it: a model of this run by its name, the bar as L5c's stored model."""
    return f"L5c's stored `{BAR}`" if key == THE_BARS_SIDE else f"`{key}`"


def _read(failed_checks: Sequence[str], after: str, before: str, said: str, own: Mapping | None, other: Mapping | None, names: Sequence[str]) -> dict:
    """One read of two models' figures on all of G, on whole numbers: `after` minus `before`, per attempt. At or above
    plus `A_LINE`: `names[0]`. Below zero by MORE than `A_LINE`: `names[2]`. Between (a difference of exactly minus
    `A_LINE` is between): `names[1]`. INCONCLUSIVE when a check of this run failed; NOT READ without attempts."""
    if failed_checks:
        return {"name": INCONCLUSIVE, "reason": f"a check of this run failed ({'; '.join(failed_checks)}): {said} is not read"}
    if not own or not other or not own["attempts"] or not other["attempts"]:
        return {"name": NOT_READ, "reason": f"{_shown(after)} or {_shown(before)} has no attempt: {said} is NOT READ"}
    difference = _exact(own) - _exact(other)
    name = names[0] if difference >= A_LINE else names[2] if difference < -A_LINE else names[1]
    return {"name": name, "difference": round(float(difference), 6), "line": float(A_LINE), after: _whole(own), before: _whole(other),
            "reason": f"{_shown(after)} minus {_shown(before)}, per 1,000 attempts on all of G: {float(difference) * 1000:+.2f} ({own['successes']} of {own['attempts']} against "
                      f"{other['successes']} of {other['attempts']}); the lines are +{float(A_LINE) * 1000:g} and -{float(A_LINE) * 1000:g}"}


def the_primary(failed_checks: Sequence[str], new_own: Mapping | None, bar: Mapping | None) -> dict:
    """THE PRIMARY: `new_own` minus L5c's stored `again`, per attempt on all of G. At or above plus `A_LINE`: THE
    MODEL'S OWN NEW PROOFS ADD FROM THE BEST MODEL. Below zero by more than it: THEY COST THERE AT THIS SEED. Between:
    NOT SHOWN AT ONE SEED."""
    return _read(failed_checks, NEW_OWN, THE_BARS_SIDE, "the primary", new_own, bar, (ADD, NOT_SHOWN_AT_ONE, COST))


def whose_proofs(failed_checks: Sequence[str], new_own: Mapping | None, old_own: Mapping | None) -> dict:
    """WHOSE PROOFS: `new_own` minus `old_own`, per attempt on all of G. At or above plus `A_LINE`: PROOFS THE STRONGER
    MODEL FOUND TEACH MORE THAN THE WEAKER MODEL'S. Below zero by more than it: THEY TEACH LESS. Between: NOT SHOWN AT
    ONE SEED."""
    return _read(failed_checks, NEW_OWN, OLD_OWN, "whose proofs", new_own, old_own, (TEACH_MORE, NOT_SHOWN_AT_ONE, TEACH_LESS))


def under_half(solves: Sequence[Mapping]) -> int:
    """The solves whose problem the searching model solved in UNDER HALF of its attempts at it (`k` of `n`)."""
    return sum(2 * row["k"] < row["n"] for row in solves)


def the_yield(table: Mapping, solves: Sequence[Mapping], enough: Mapping, lines: Mapping) -> dict:
    """THE SEARCH, a finding of its own. `table`: `l6.second_search_table` of B's search (by part: the problems
    attempted and solved). `solves`: B's new solves, one row a problem (its `k` of `n`). `enough`: whether there are
    as many as are asked. `lines`: the line counts of the new solves' rows and, beside, of round 1's. N; how many of
    part (a), the problems round 1's search left unsolved, and of part (b), the candidates neither draw took; how many
    are solved in under half of B's attempts."""
    count = len(solves)
    return {"what": "B's own search: the problems it solved that neither search 1 nor the map's model solved (NEW SOLVES), by part, and how many of them it solved in under "
                    "half of its attempts; the lines of their rows beside round 1's", "new_solves": count, "minimum": enough["minimum"], "enough": bool(enough["passes"]),
            "problems": table["problems"], "share_solved": table["share_solved"],
            PART_A: {"what": "the problems search 1 attempted and left unsolved", **table[PART_A]}, PART_B: {"what": "the candidates neither draw took", **table[PART_B]},
            "solved_in_under_half_of_its_attempts": under_half(solves), "episodes_each": table["episodes_each"], "per_1000": table["per_1000"], "lines": dict(lines)}


def what_follows(primary: Mapping, whose: Mapping, enough: bool) -> dict:
    """What follows, fixed before the run. ADD: the cycle turns from the best model; two further seeds of `new_own` and
    of the bar (not built) before anything is said over seeds, and the highest model is a candidate for a measurement
    in full. NOT SHOWN: said as it is, with the search's yield; a third round is not run on it. COST: to a decision with
    L5f's read; the full-size cycle is not proposed on it. Whatever the primary reads, WHOSE PROOFS says whether the
    search by the stronger model is the part worth scaling. `enough` false (too few new solves): nothing was trained,
    and the search is the finding. Nothing is named while the primary is not read."""
    if not enough:
        return {"name": THE_SEARCH_IS_THE_FINDING, "reason": "the search found fewer new solves than are asked: no set was built and no model trained or measured; the "
                                                             "search's yield is what this run says"}
    if primary["name"] in (INCONCLUSIVE, NOT_READ):
        return {"name": NOT_READ, "reason": "the primary is not read: nothing is named for what follows"}
    name = {ADD: THE_CYCLE_TURNS, NOT_SHOWN_AT_ONE: SAID_AS_IT_IS, COST: TO_A_DECISION}[primary["name"]]
    said = {THE_CYCLE_TURNS: f"two further seeds of `{NEW_OWN}` and of the bar (L5c's `{BAR}` at those seeds, which no run holds: not built) before anything is said over "
                             "seeds; the highest model is a candidate for a measurement in full",
            SAID_AS_IT_IS: "it is said as it is, with the search's yield; a third round is not run on it",
            TO_A_DECISION: "it needs a decision, with L5f's read; the full-size cycle is not proposed on it"}[name]
    scaling = {TEACH_MORE: "the search by the stronger model is the part worth scaling", TEACH_LESS: "the search by the stronger model is not shown to be the part worth "
               "scaling: its proofs teach less than the weaker model's here", NOT_SHOWN_AT_ONE: "whose proofs is not shown at one seed"}.get(whose["name"], "whose proofs is not read")
    return {"name": name, "whose_proofs": whose["name"], "reason": f"the primary reads {primary['name']}: {said}. WHOSE PROOFS reads {whose['name']}: {scaling}"}
