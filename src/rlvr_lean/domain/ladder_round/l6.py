"""L6: a second round. Does the cycle compound? Spec: docs/spec/ladder-loop.spec.md, "L6: a second round. Does the
cycle compound?". Pure.

L5 asked whether ONE harvest of deep solves teaches what the same number of published proofs does not. The cycle is the
next question: the model trained on round 1's deep solves, M(1), searches again, and what it finds is trained on. Round 1
is L5's, read and never made again (its search, its held-aside set, its `deep` model).

  search 2          M(1) attempts every hard problem that is neither one of round 1's solves nor held aside
                    (`second_search`): (a) the problems search 1 attempted and left unsolved, (b) the candidates (map
                    problems the start model failed in every map attempt) that neither search 1's draw nor the held-aside
                    set took. HELD: with round 1's solves and the held-aside set they partition the candidates. A problem
                    it solves is a ROUND-2 solve; too few of them (`enough_round_two_solves`): nothing is trained
  four sets         each of 2N rows, N every deep solve of both rounds (no cap), the published rows the first of ONE hash
                    order of the pretraining file (`the_sets`, held by `hold_the_sets`): `published2` (the first 2N),
                    `one_round` (round 1's solves and the first 2N less as many), `two_rounds` (the N solves and the first
                    N: M(2)), `long_ordinary` (N ORDINARY solves drawn to have the LINE COUNTS of `two_rounds`' own N rows
                    bin by bin, `matched_in_length`, and the same first N)
  the reads         ON THE HELD-ASIDE SET, successes per attempt, paired by problem: `two_rounds` minus `published2`
                    (`primary_branch`), minus `one_round` (`compounding`), minus `long_ordinary` (`hard_or_long`); and on
                    the goal problems of 8 lines or more the count solved, `two_rounds` against `published2` (`reach`)
  what follows      by the scout's table, in one line (`what_follows`), with the spread L5's three seeds measured between
                    trainings of one kind on the held-aside set (`spread_of_l5`) when it is given

A check that failed makes the whole read INCONCLUSIVE, and nothing else is said. An interval that ends at zero holds zero.
"""

from __future__ import annotations

from collections import Counter
from typing import Collection, Mapping, Sequence

from rlvr_lean.domain.ladder_round.assembly import training_order
from rlvr_lean.domain.ladder_round.ceiling import interval_text
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ
from rlvr_lean.domain.ladder_round.l4_rows import PRETRAINING, SIGN_TEST_LEVEL, THE_MINIMUM, THE_RULE, rehearsal_rows
from rlvr_lean.domain.ladder_round.l4_rows2 import NOT_SEPARATED, _placed
from rlvr_lean.domain.ladder_round.l4b import rehearsal_of
from rlvr_lean.domain.ladder_round.l5 import NOT_SHOWN, bands, failed_every_attempt
from rlvr_lean.domain.ladder_round.l5 import no_barred_row as of_l5_no_barred_row
from rlvr_lean.domain.problem_pool.selection import rank

PUBLISHED2, ONE_ROUND, TWO_ROUNDS, LONG_ORDINARY = "published2", "one_round", "two_rounds", "long_ordinary"
MODELS = (PUBLISHED2, ONE_ROUND, TWO_ROUNDS, LONG_ORDINARY)        # the four trained models, in the order they are trained and measured
ROUND_1, ROUND_2, ORDINARY = "round_1", "round_2", "ordinary"      # where a row that is a model's own proof is from (its origin): a deep solve of round 1, of round 2, an ordinary solve
OWN = (ROUND_1, ROUND_2, ORDINARY)
ORIGINS = {PUBLISHED2: (), ONE_ROUND: (ROUND_1,), TWO_ROUNDS: (ROUND_1, ROUND_2), LONG_ORDINARY: (ORDINARY,)}       # the origins of each set's own rows
PART_A, PART_B = "a", "b"                                           # search 2's two parts: left unsolved by search 1; in neither draw
SEARCH_LABEL, LONG_LABEL = "l6_search", "l6_long_ordinary"          # the two draws' own labels in the content hash (the first is a SMOKE run's alone: the rule draws nothing)

CYCLE_ADDS = "THE CYCLE ADDS WHAT THE SAME NUMBER OF PUBLISHED PROOFS DOES NOT"     # the primary's interval above zero (holding zero: `NOT_SEPARATED`; below: `IT_COSTS`)
IT_COSTS = "IT COSTS"
PRIMARY = (CYCLE_ADDS, NOT_SEPARATED, IT_COSTS, INCONCLUSIVE, NOT_READ)
SECOND_ROUND_ADDS = "THE SECOND ROUND ADDS"                                         # compounding: `two_rounds` minus `one_round`
COMPOUNDING = (SECOND_ROUND_ADDS, NOT_SHOWN, IT_COSTS, INCONCLUSIVE, NOT_READ)
HARD_TO_FIND, NOT_FROM_LENGTH, LONG_DO_BETTER = "HARD TO FIND MATTERS BEYOND LENGTH", "NOT SEPARATED FROM LENGTH", "LONG ORDINARY SOLVES DO BETTER"
LENGTH = (HARD_TO_FIND, NOT_FROM_LENGTH, LONG_DO_BETTER, INCONCLUSIVE, NOT_READ)   # hard to find, or long: `two_rounds` minus `long_ordinary`
REACH_SHOWN = "REACH SHOWN AT THIS SEED"
REACH = (REACH_SHOWN, NOT_SHOWN, INCONCLUSIVE, NOT_READ)
TWO_FURTHER_SEEDS = "TWO FURTHER SEEDS OF THE FOUR MODELS ON THE SAME TWO SEARCHES"
DOES_NOT_CLEAR = "THE CYCLE AT THIS SIZE DOES NOT CLEAR THE BAR"
IN_BETWEEN = "IN BETWEEN: STATED AS IT IS, FOR A DECISION"
SPREAD_NOT_GIVEN = "THE SPREAD IS NOT GIVEN: NO KILL IS NAMED"
FOLLOWS = (TWO_FURTHER_SEEDS, DOES_NOT_CLEAR, IN_BETWEEN, SPREAD_NOT_GIVEN, NOT_READ)
SEEDS_OF_L5 = 3                                                     # the spread is the one L5's THREE seeds measured
SPREAD_FIGURE = "held_aside_per_1000"                               # ... of this figure of each kind of model, as the read over L5's seeds names it


# ------------------------------------------------------------------------------------------------ search 2
def second_search(map_rows: Sequence[Mapping], drawn: Sequence[Mapping], aside: Sequence[Mapping], solved: Collection[str], seed: int, at_least: int = 0) -> dict:
    """The problems of SEARCH 2: every hard problem that is neither one of round 1's solves nor held aside.
    `map_rows`: the start model's map, a row a problem. `drawn`: the problems search 1 attempted, in its draw's order
    (each with `taken_by`). `aside`: the held-aside problems. `solved`: the problems search 1 solved.

      (a)   the problems search 1 attempted and left UNSOLVED, in its draw's order
      (b)   the candidates (`l5.failed_every_attempt`) that neither search 1's draw nor the held-aside set took, in the
            map's order

    HELD, not assumed, each a refusal (ValueError): a problem search 1 solved is one it attempted; no problem is both
    searched and held aside, and none stands twice; a problem either draw took BY THE RULE is a candidate; and round
    1's solves, (a), (b) and the held-aside set PARTITION the candidates (with the problems a smoke run's looser draws
    took, which are no candidates). Returns the problems (`problem_id`, `part`, the map's k of n, `taken_by`) and the
    partition's counts: recorded, never numbers in the code.

    `at_least` (a SMOKE run alone; 0: the rule and nothing else): a fixture's map may leave search 2 no problem. When
    (a) and (b) hold fewer than `at_least`, the map's OTHER problems that are neither searched nor held aside join in
    the hash order of `seed` and the problem's id, each of part (b) and marked as taken by the run's minimum."""
    of_the_map = {row["problem_id"]: row for row in map_rows}
    candidates = [row["problem_id"] for row in failed_every_attempt(map_rows)]
    searched, held, found = [row["problem_id"] for row in drawn], [row["problem_id"] for row in aside], set(solved)
    by_the_rule = {row["problem_id"] for row in (*drawn, *aside) if row["taken_by"] == THE_RULE}
    strangers, unknown, both = sorted(by_the_rule - set(candidates)), sorted(found - set(searched)), sorted(set(searched) & set(held))
    twice = sorted(key for key, times in Counter([*searched, *held]).items() if times > 1)
    if strangers or unknown or both or twice:
        raise ValueError(f"round 1 does not agree with the map: {len(strangers)} problems a draw took by the rule are no candidates (the map's model did not fail them in "
                         f"every attempt), {len(unknown)} problems search 1 solved are none it attempted, {len(both)} are both searched and held aside, {len(twice)} stand "
                         f"twice (first: {(strangers or unknown or both or twice)[0]}). Refused")
    why = {row["problem_id"]: row["taken_by"] for row in drawn}
    named = lambda key, part, taken_by: {"problem_id": key, "part": part, "map_k": of_the_map[key]["resolved"], "map_n": of_the_map[key]["episodes"], "taken_by": taken_by}      # noqa: E731
    taken = {*searched, *held}
    left = [named(key, PART_A, why[key]) for key in searched if key not in found]
    neither = [named(key, PART_B, THE_RULE) for key in candidates if key not in taken]
    pool = {*candidates, *taken}                    # the candidates, and (a smoke run) what its looser draws took beside them
    parts = [found, {row["problem_id"] for row in left}, {row["problem_id"] for row in neither}, set(held)]
    if sum(map(len, parts)) != len(pool) or set().union(*parts) != pool:
        raise ValueError(f"round 1's solves ({len(found)}), the problems search 1 left unsolved ({len(left)}), the candidates in neither draw ({len(neither)}) and the "
                         f"held-aside set ({len(held)}) do not partition the {len(pool)} candidates. Refused")
    problems = [*left, *neither]
    if len(problems) < at_least:
        others = sorted((row["problem_id"] for row in map_rows if row["problem_id"] not in pool), key=lambda key: rank(seed, SEARCH_LABEL, key))
        problems += [named(key, PART_B, THE_MINIMUM) for key in others[:at_least - len(problems)]]
    return {"problems": problems,
            "partition": {"candidates": len(candidates), "attempted_by_search_1": len(searched), "solved_by_search_1": len(found), "a_left_unsolved_by_search_1": len(left),
                          "b_in_neither_draw": len(neither), "held_aside": len(held), "taken_by_the_runs_minimum": len(pool) - len(candidates)}}


def enough_round_two_solves(solved: int, minimum: int) -> dict:
    """The first check: search 2 found at least `minimum` round-2 solves. Fewer: nothing is trained, the later steps
    do not run, and the search is the finding."""
    return {"what": f"search 2 found at least {minimum} round-2 solves (fewer: nothing is trained, and the search is then the finding)",
            "round_two_solves": solved, "minimum": minimum, "passes": solved >= minimum}


def second_search_table(problems: Sequence[Mapping], results: Sequence[Mapping], solves: Sequence[Mapping], episodes: int, first_band: int) -> dict:
    """Search 2 itself. `problems`: the problems it attempted (`second_search`). `results`: its per-problem rows.
    `solves`: the round-2 rows, one a problem solved, each with `first_success`: the place of the problem's first
    verified attempt among M(1)'s OWN (from 1: M(1) made none at these problems before). `episodes`: its attempts a
    problem. `first_band`: the attempts a map makes at a problem (8): the first band is 1 to that many, and each later
    one ends at twice the one before (`l5.bands`: 9 to 16, 17 to 32, 33 to 64). For each PART, (a) and (b): the problems
    attempted and solved, by the attempt of the first success."""
    of_bands = [(1, min(first_band, episodes)), *bands(first_band, episodes)]
    part_of = {row["problem_id"]: row["part"] for row in problems}

    def of_a_part(part: str) -> dict:
        attempted, own = [row for row in problems if row["part"] == part], [row for row in solves if part_of[row["problem_id"]] == part]
        return {"problems": len(attempted), "solved": len(own), "share_solved": round(len(own) / len(attempted), 5) if attempted else None,
                "solved_by_first_success": {f"{low}_to_{high}": sum(low <= row["first_success"] <= high for row in own) for low, high in of_bands}}

    successes, made = sum(row["resolved"] for row in results), sum(row["episodes"] for row in results)
    return {"what": f"the problems search 2 attempted ({episodes} attempts each, by the model round 1 trained): (a) the problems search 1 left unsolved, (b) the candidates "
                    "neither draw took; the problems it solved, by the attempt of the first success among its own; its attempts (an episode is one attempt at the side "
                    "the published answer allows, and one at the other for the audited share); the episodes that resolved their problem, per 1,000",
            "problems": len(problems), "taken_by_the_rule": sum(row["taken_by"] == THE_RULE for row in problems), "taken_by_the_runs_minimum": sum(row["taken_by"] != THE_RULE for row in problems),
            "solved": len(solves), "share_solved": round(len(solves) / len(problems), 5) if problems else None, PART_A: of_a_part(PART_A), PART_B: of_a_part(PART_B),
            "episodes_each": episodes, "episodes": made, "attempts": sum(row["episodes"] * row["sides"] for row in results), "successes": successes,
            "per_1000": round(1000 * successes / made, 2) if made else None, "without_an_answer": sum(row["attempts_without_an_answer"] for row in results)}


# ------------------------------------------------------------------------------- ordinary solves matched in length
def line_bins(edges: Sequence[int]) -> list[tuple[str, int, int | None]]:
    """The bins of a proof's number of lines, from their lower edges (the setting's: 1, 2, 4, 8, 16), each (its name,
    its least, its most): `1`, `2-3`, `4-7`, `8-15`, `16+`. Refused (ValueError) unless the edges are whole numbers that
    begin at 1 and rise."""
    if not edges or edges[0] != 1 or any(isinstance(edge, bool) or not isinstance(edge, int) for edge in edges) or any(low >= high for low, high in zip(edges, edges[1:])):
        raise ValueError(f"the line bins' lower edges are {list(edges)}: whole numbers that begin at 1 and rise (a proof has at least one line)")
    tops = [*(high - 1 for high in edges[1:]), None]
    return [(f"{low}+" if high is None else str(low) if high == low else f"{low}-{high}", low, high) for low, high in zip(edges, tops)]


def bin_of(lines: int, edges: Sequence[int]) -> str:
    """The bin of a proof of `lines` lines (one of fewer than the first edge is of the first bin)."""
    of_bins = line_bins(edges)
    return next((name for name, low, high in reversed(of_bins) if lines >= low), of_bins[0][0])


def matched_in_length(own: Sequence[Mapping], offered: Sequence[Mapping], edges: Sequence[int], seed: int, loosely: bool = False) -> tuple[list[dict], list[dict]]:
    """The ORDINARY solves `long_ordinary` is trained on: as many of `offered` (one row an ordinary solve the map
    offers, with the `lines` of its proof) as `own` has rows (`two_rounds`' own N), drawn to have THEIR LINE COUNTS bin
    by bin (`line_bins`). Within a bin a seeded draw: the rows of the smallest content hash of the task's seed and the
    problem's id. A BIN THE MAP CANNOT FILL IS FILLED FROM THE NEXT SHORTER ONE: from the longest bin down, what a bin
    cannot give is asked of the next shorter one beside that bin's own. Returns (the rows taken, each with its `bin` and
    `taken_by`; the table, a row a bin from the shortest: `wanted`, what the map `offered`, what longer bins asked of it
    beside, `taken`, and its `shortfall`, which is asked of the next shorter bin).

    Refused (ValueError) when the SHORTEST bin cannot give what is asked of it: the section names no fill from a longer
    bin. `loosely` (a SMOKE run alone): the rows still missing are then taken among what is left, in the same hash
    order, each marked as taken by the run's minimum; refused when the map offers fewer rows than `own` has."""
    names = [name for name, _, _ in line_bins(edges)]
    wanted = Counter(bin_of(row["lines"], edges) for row in own)
    in_order = lambda rows: sorted(rows, key=lambda row: rank(seed, LONG_LABEL, row["problem_id"]))      # noqa: E731
    by_bin = {name: in_order(row for row in offered if bin_of(row["lines"], edges) == name) for name in names}
    taken, table, carried = [], [], 0
    for name in reversed(names):
        asked = wanted[name] + carried
        given = by_bin[name][:asked]
        table.append({"lines": name, "wanted": wanted[name], "offered": len(by_bin[name]), "asked_by_longer_bins": carried, "taken": len(given), "shortfall": asked - len(given)})
        taken += [{**row, "bin": name, "taken_by": THE_RULE} for row in given]
        carried = asked - len(given)
    if carried:
        have = {row["problem_id"] for row in taken}
        left = in_order(row for row in offered if row["problem_id"] not in have) if loosely else []
        if len(left) < carried:
            raise ValueError(f"the map offers {len(offered)} ordinary solves, {len(by_bin[names[0]])} of them in the shortest bin (`{names[0]}`), and {carried} of the "
                             f"{len(own)} rows asked for are still missing there: a bin the map cannot fill is filled from the next SHORTER one, and no bin is shorter. Refused")
        taken += [{**row, "bin": bin_of(row["lines"], edges), "taken_by": THE_MINIMUM} for row in left[:carried]]
    return taken, table[::-1]


# ------------------------------------------------------------------------------------------------- the sets
def hold_the_sets(sets: Mapping[str, Sequence[Mapping]], pretraining: Sequence[Mapping], seed: int, loosely: bool = False) -> int:
    """What the four sets are HELD to, each a refusal (ValueError); returns N. Each set holds 2N rows and no id twice.
    `two_rounds`: N rows of the model's own (round 1's solves and round 2's, at least one of each) and N published
    rows; `one_round`: round 1's solves, THE SAME ROWS, and as many published rows as make 2N; `long_ordinary`: N
    ordinary solves and THE SAME N published rows as `two_rounds`, by id and by place in the file; `published2`: 2N
    published rows and no other. The published rows of each are the FIRST of the ONE hash order L4t's rehearsal draw
    uses (`l4_rows.rehearsal_rows`), so each smaller draw is the first rows of each larger one. No row of a model's own
    is of a problem the pretraining file holds, and no problem is both a deep solve and an ordinary one (a deep solve
    was failed in every map attempt; `loosely`, a SMOKE run alone, whose looser draws may take one problem twice, does
    not hold that). Each set stands in the arm's content-hash order."""
    own = {model: [row for row in sets[model] if row["origin"] != PRETRAINING] for model in MODELS}
    of_the_file = {model: rehearsal_of(sets[model]) for model in MODELS}
    first, count = sum(row["origin"] == ROUND_1 for row in own[TWO_ROUNDS]), len(own[TWO_ROUNDS])
    sizes = {model: (len(own[model]), len(of_the_file[model])) for model in MODELS}
    published = {PUBLISHED2: 2 * count, ONE_ROUND: 2 * count - first, TWO_ROUNDS: count, LONG_ORDINARY: count}
    if not first or first == count or sizes != {PUBLISHED2: (0, 2 * count), ONE_ROUND: (first, 2 * count - first), TWO_ROUNDS: (count, count), LONG_ORDINARY: (count, count)}:
        raise ValueError(f"the four sets hold (rows of the model's own, published rows) {sizes}, and `{TWO_ROUNDS}` {first} rows of round 1: `{TWO_ROUNDS}` is the N solves of "
                         f"both rounds (at least one of each) and N published rows, `{ONE_ROUND}` round 1's solves and as many published rows as make 2N, `{LONG_ORDINARY}` N "
                         f"ordinary solves and N published rows, `{PUBLISHED2}` 2N published rows. Refused")
    for model in MODELS:
        strangers = [row["id"] for row in own[model] if row["origin"] not in ORIGINS[model]]
        twice = sorted(key for key, times in Counter(row["id"] for row in sets[model]).items() if times > 1)
        again = sorted(key for key, times in Counter(row["problem_id"] for row in own[model]).items() if times > 1)
        if strangers or twice or again:
            raise ValueError(f"`{model}` holds {len(strangers)} rows of another origin than its own ({list(ORIGINS[model])}), {len(twice)} ids twice and {len(again)} problems "
                             f"in two rows of its own (first: {(strangers or twice or again)[0]}). Refused")
        if [row["id"] for row in sets[model]] != [row["id"] for row in training_order(sets[model], seed)]:
            raise ValueError(f"`{model}` does not stand in the content-hash order of this seed: every model here is trained in that order. Refused")
    if sorted(row["id"] for row in own[ONE_ROUND]) != sorted(row["id"] for row in own[TWO_ROUNDS] if row["origin"] == ROUND_1):
        raise ValueError(f"`{ONE_ROUND}` does not hold the rows of round 1 that `{TWO_ROUNDS}` holds: round 1's solves are in both, the same rows. Refused")
    order = rehearsal_rows(pretraining, 2 * count, seed)            # the draw's own order, the smallest hash first: L4t's, L4t2's, L4p's and L5's draw, with this count
    for model in MODELS:
        wanted = rehearsal_rows(pretraining, published[model], seed)
        if list(map(_placed, wanted)) != list(map(_placed, order[:published[model]])):
            raise ValueError(f"the {published[model]} rows drawn for `{model}` are not the first {published[model]} of the hash order `{PUBLISHED2}`'s {2 * count} stand in. Refused")
        if sorted(map(_placed, of_the_file[model])) != sorted(map(_placed, wanted)):
            other = sorted(set(map(_placed, of_the_file[model])) ^ set(map(_placed, wanted)))
            raise ValueError(f"the {len(of_the_file[model])} published rows of `{model}` are not the first {len(wanted)} of the hash order among the {len(pretraining)} rows "
                             f"the start model was pretrained on (first that differs: {other[0][0]}): the four sets would not hold the same published rows. Refused")
    if list(map(_placed, of_the_file[TWO_ROUNDS])) != list(map(_placed, of_the_file[LONG_ORDINARY])):
        raise ValueError(f"`{TWO_ROUNDS}` and `{LONG_ORDINARY}` do not hold the same published rows in the same order. Refused")
    both = [] if loosely else sorted({row["problem_id"] for row in own[TWO_ROUNDS]} & {row["problem_id"] for row in own[LONG_ORDINARY]})
    seen = sorted({row["problem_id"] for model in MODELS for row in own[model]} & {row["problem_id"] for row in pretraining})
    if both or seen:
        raise ValueError(f"{len(both)} problems are both a deep solve of `{TWO_ROUNDS}` and an ordinary solve of `{LONG_ORDINARY}`, and {len(seen)} problems of the models' "
                         f"own rows are rows of the pretraining file (first: {(both or seen)[0]}): a deep solve is no ordinary solve, and neither is a problem whose "
                         "published proof a model here has seen. Refused")
    return count


def the_sets(round_1: Sequence[Mapping], round_2: Sequence[Mapping], ordinary: Sequence[Mapping], pretraining: Sequence[Mapping], seed: int,
             loosely: bool = False) -> dict[str, list[Mapping]]:
    """The four training sets, each in the arm's content-hash order (`assembly.training_order`). `round_1`, `round_2`:
    the deep rows of each round, one a problem solved (origins `round_1`, `round_2`); N is every one of them, with no
    cap. `ordinary`: N rows of ordinary solves (`matched_in_length`; origin `ordinary`). `pretraining`: the rows the
    start model was pretrained on, in the file's order, WITHOUT their text. `seed`: the task's.

      published2      the first 2N rows of the hash order L4t's rehearsal draw uses (`l4_rows.rehearsal_rows`): the bar
      one_round       round 1's solves and the first 2N less as many rows of that order
      two_rounds      the N solves of both rounds and the first N rows of that order: M(2)
      long_ordinary   the N ordinary solves and the SAME first N rows of that order

    Refused (ValueError) as `hold_the_sets` refuses, which holds what is built here from the outside."""
    count = len(round_1) + len(round_2)
    if not round_1 or not round_2 or len(ordinary) != count:
        raise ValueError(f"round 1 has {len(round_1)} deep rows, round 2 {len(round_2)}, and `{LONG_ORDINARY}` {len(ordinary)} ordinary solves: at least one of each round, "
                         "and as many ordinary solves as both rounds have deep rows. Refused")
    first = lambda rows: rehearsal_rows(pretraining, rows, seed)      # noqa: E731
    sets = {PUBLISHED2: training_order(first(2 * count), seed), ONE_ROUND: training_order([*round_1, *first(2 * count - len(round_1))], seed),
            TWO_ROUNDS: training_order([*round_1, *round_2, *first(count)], seed), LONG_ORDINARY: training_order([*ordinary, *first(count)], seed)}
    hold_the_sets(sets, pretraining, seed, loosely)
    return sets


def no_barred_row(trained: Sequence[Mapping], heldout: Collection[str], half_seed: int, base_map: Collection[str], aside: Collection[str]) -> dict:
    """The fourth check of a trained model here, read on the rows the TRAINING LOOP recorded: L5's own
    (`l5.no_barred_row`: no row is a held-out problem; its published rows are of the `pretrain` half; its own rows are
    base-map problems), AND no row, published or its own, is of a HELD-ASIDE problem (`aside`): that set is the test."""
    found = of_l5_no_barred_row(trained, heldout, half_seed, base_map)
    on_the_set = sorted({row["problem_id"] for row in trained} & set(aside))
    return {**found, "what": f"{found['what']}; and no row is of a held-aside problem, which no model here trains on", "held_aside_problems": len(on_the_set),
            "first": [*found["first"], *on_the_set][:5], "passes": found["passes"] and not on_the_set}


# --------------------------------------------------------------------------------------------- the branches
def _three_ways(failed_checks: Sequence[str], difference: Mapping, pair: tuple[str, str], names: tuple[str, str, str], said: tuple[str, str, str], what: str) -> dict:
    """One read on the held-aside set, successes per attempt, `pair[0]` minus `pair[1]`, paired by problem, with its
    interval: `names` (and what is `said` of each) when the interval is above zero, holds zero (one that ends at zero
    holds it) and is below zero. A failed check: INCONCLUSIVE, and nothing else is said. Nothing measured on both
    sides: NOT READ. `what`: the read, in a refusal's words."""
    after, before = pair
    if failed_checks:
        return {"name": INCONCLUSIVE, "failed_checks": list(failed_checks), "reason": f"the run could not have been read ({', '.join(failed_checks)}): {what} is not read"}
    if difference.get("mean") is None:
        return {"name": NOT_READ, "failed_checks": [], "reason": f"no held-aside problem was measured on both sides: {what} has nothing to be read on"}
    read, on = interval_text(difference), f"on the held-aside problems `{after}` minus `{before}`"
    if difference["low"] > 0:
        return {"name": names[0], "failed_checks": [], "reason": f"{on} is above zero ({read}): {said[0]}"}
    if difference["high"] < 0:
        return {"name": names[2], "failed_checks": [], "reason": f"{on} is below zero ({read}): {said[2]}"}
    resolves = round((difference["high"] - difference["low"]) / 2, 5)
    return {"name": names[1], "failed_checks": [], "resolves": resolves,
            "reason": f"{on} holds zero ({read}): {said[1]}. This difference resolves {resolves:.5f} per attempt (half the width of its interval)"}


def primary_branch(failed_checks: Sequence[str], difference: Mapping) -> dict:
    """THE PRIMARY, the bet at equal steps: on the held-aside set, `two_rounds` minus `published2`. Above zero: THE
    CYCLE ADDS WHAT THE SAME NUMBER OF PUBLISHED PROOFS DOES NOT. Holds zero: NOT SEPARATED AT THIS SIZE. Below: IT
    COSTS."""
    return _three_ways(failed_checks, difference, (TWO_ROUNDS, PUBLISHED2), (CYCLE_ADDS, NOT_SEPARATED, IT_COSTS),
                       ("the cycle adds what the same number of published proofs does not", "not separated at this size",
                        "in the place of the same number of published proofs the two rounds' solves cost"), "the primary")


def compounding(failed_checks: Sequence[str], difference: Mapping) -> dict:
    """COMPOUNDING: on the held-aside set, `two_rounds` minus `one_round` (round 2's solves against the same number of
    published rows, round 1's solves in both). THE SECOND ROUND ADDS / NOT SHOWN / IT COSTS."""
    return _three_ways(failed_checks, difference, (TWO_ROUNDS, ONE_ROUND), (SECOND_ROUND_ADDS, NOT_SHOWN, IT_COSTS),
                       ("round 2's solves add what the same number of published rows does not, beside round 1's solves", "that the second round adds is not shown",
                        "in the place of the same number of published rows round 2's solves cost"), "compounding")


def hard_or_long(failed_checks: Sequence[str], difference: Mapping) -> dict:
    """HARD TO FIND, OR LONG: on the held-aside set, `two_rounds` minus `long_ordinary` (ordinary solves matched in
    length). HARD TO FIND MATTERS BEYOND LENGTH / NOT SEPARATED FROM LENGTH / LONG ORDINARY SOLVES DO BETTER."""
    return _three_ways(failed_checks, difference, (TWO_ROUNDS, LONG_ORDINARY), (HARD_TO_FIND, NOT_FROM_LENGTH, LONG_DO_BETTER),
                       ("that a solve was hard to find matters beyond its length", "the deep solves are not separated from ordinary solves of their length",
                        "ordinary solves of the same length do better than the deep ones"), "hard to find, or long")


def reach(failed_checks: Sequence[str], solved: Mapping, level: float = SIGN_TEST_LEVEL) -> dict:
    """REACH, on the goal set: the goal problems whose shortest published proof has 8 lines or more that are solved at
    least once in all the attempts, `two_rounds` against `published2` (`solved`: `gained`, by `two_rounds` and not by
    `published2`; `lost`; the two-sided sign test). REACH SHOWN AT THIS SEED when MORE are gained than lost AND the
    sign test is under `level`; otherwise NOT SHOWN."""
    if failed_checks:
        return {"name": INCONCLUSIVE, "failed_checks": list(failed_checks), "reason": f"the run could not have been read ({', '.join(failed_checks)}): reach is not read"}
    if not solved.get("problems"):
        return {"name": NOT_READ, "failed_checks": [], "reason": "no long goal problem was measured on both sides: reach has nothing to be read on"}
    gained, lost, p = solved["gained"], solved["lost"], solved["sign_test_p"]
    more, under = bool(gained > lost), bool(p is not None and p < level)
    return {"name": REACH_SHOWN if more and under else NOT_SHOWN, "failed_checks": [], "gained": gained, "lost": lost, "sign_test_p": p, "level": level,
            "more_gained_than_lost": more, "sign_test_under_the_level": under,
            "reason": f"of the {solved['problems']} long goal problems `{TWO_ROUNDS}` solves {solved['resolved_after']} and `{PUBLISHED2}` {solved['resolved_before']}: gained "
                      f"{gained}, lost {lost}" + (f" (p = {p})" if p is not None else " (no problem changed)") + ": " + ("more" if more else "not more")
                      + f" are gained than lost, and the sign test is {'under' if under else 'not under'} {level:g}"}


def spread_of_l5(read: Mapping | None, seeds: int = SEEDS_OF_L5) -> dict:
    """The spread L5's three seeds measured between trainings of ONE kind on the held-aside set, from the read over
    L5's seeds as its command writes it (`reporting.ladder_l5.over_seeds`: `pooled.the_spread`, by kind of model the
    standard deviation of its seeds' successes per 1,000 attempts there). There is one for each kind of model, and THE
    LARGEST is taken: a kill then asks the most. NOT GIVEN (`given` false, with why): no file; a read that pooled
    nothing or is not over exactly `seeds` seeds; a kind without a standard deviation."""
    if read is None:
        return {"given": False, "why": "no file of the read over L5's seeds is in the run directory"}
    found, of_the_kinds = read.get("seeds_read") or [], ((read.get("pooled") or {}).get("the_spread") or {})
    by_kind = {kind: (figures.get(SPREAD_FIGURE) or {}).get("standard_deviation") for kind, figures in of_the_kinds.items()}
    if len(found) != seeds or not by_kind or any(value is None for value in by_kind.values()):
        return {"given": False, "seeds_read": list(found), "per_1000_by_kind": by_kind,
                "why": f"the file given holds the read of {len(found)} seeds of L5 ({', '.join(map(str, found)) or 'none'}), or no standard deviation for a kind of model: the "
                       f"spread is the one {seeds} seeds measured"}
    largest = max(by_kind, key=lambda kind: by_kind[kind])
    return {"given": True, "seeds_read": list(found), "per_1000_by_kind": by_kind, "of": largest, "per_attempt": round(by_kind[largest] / 1000, 6),
            "what": f"the standard deviation (n - 1) over L5's {seeds} seeds of a kind of model's successes per attempt on the held-aside set; the largest of the kinds' is "
                    f"taken (`{largest}`)"}


def what_follows(primary: Mapping, of_compounding: Mapping, of_reach: Mapping, difference: Mapping, spread: Mapping) -> dict:
    """What follows, by the scout's table, in the section's order. `difference`: the primary's. `spread`:
    `spread_of_l5`'s. The primary's POINT at or above zero, or REACH SHOWN, or THE SECOND ROUND ADDS: two further seeds
    of the four models on the same two searches. The point below zero by MORE than twice the spread: the cycle at this
    size does not clear the bar, it is written up, and the question of a larger model needs a decision. In between:
    stated as it is, for a decision. With the spread NOT GIVEN no kill is named: the line gives the point and
    says so. Nothing is named when the primary is INCONCLUSIVE or was not read."""
    name = primary["name"]
    if name in (INCONCLUSIVE, NOT_READ):
        return {"name": NOT_READ, "reason": f"the primary is {name}: nothing is named for what follows"}
    point = difference["mean"]
    read = f"the primary's point is {point:+.5f} ({name}), reach reads {of_reach['name']} and compounding {of_compounding['name']}"
    if point >= 0 or of_reach["name"] == REACH_SHOWN or of_compounding["name"] == SECOND_ROUND_ADDS:
        return {"name": TWO_FURTHER_SEEDS, "point": point, "reason": f"{read}: two further seeds of the four models on the same two searches"}
    if not spread.get("given"):
        return {"name": SPREAD_NOT_GIVEN, "point": point,
                "reason": f"{read}; the spread L5's three seeds measured between trainings of one kind on the held-aside set is NOT GIVEN ({spread.get('why')}): the point is "
                          "below zero, and whether by more than twice that spread cannot be said. No kill is named"}
    twice = 2 * spread["per_attempt"]
    said = f"{read}; twice the spread L5's three seeds measured between trainings of one kind on the held-aside set is {twice:.5f} per attempt"
    if point < -twice:
        return {"name": DOES_NOT_CLEAR, "point": point, "twice_the_spread": twice,
                "reason": f"{said}, and the point is below zero by more: the cycle at this size does not clear the bar; it is written up, and the question of a larger model "
                          "needs a decision"}
    return {"name": IN_BETWEEN, "point": point, "twice_the_spread": twice, "reason": f"{said}, and the point is below zero by no more: stated as it is, for a decision"}
