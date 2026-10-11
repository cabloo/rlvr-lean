"""L5: do DEEP solves teach what ordinary solves do not? One step-matched check. Spec:
docs/spec/ladder-loop.spec.md, "L5: do DEEP solves teach what ordinary solves do not? One step-matched check". Pure.

L4p and L4c read the same thing twice: the model's own proofs, found within 8 attempts, sharpen it and narrow it. A
proof found within 8 attempts is one the model could already write. This asks about the other kind, a problem the
start model had NOT solved, with the optimizer steps and the published rows held fixed. A scout: one seed, no round.

  the deep search   the problems of the start model's own map that it failed in EVERY map attempt (`failed_every_attempt`),
                    a seeded draw of them (`search_problems`), each attempted many more times. A problem solved there is
                    a DEEP solve: its row is ONE of its verified attempts, drawn by the seed as a round draws one
                    (`solve_rows`, on `training_set.training_examples`), with the place of its FIRST verified attempt
  three sets        each of 2D rows, D the number of deep solves under a cap (`capped`): `deep` (the D deep rows and the
                    first D rows of the hash order L4t's rehearsal draw uses), `shallow` (D rows of ORDINARY solves, map
                    problems solved in a few of the map's attempts, `ordinary_draw`, and the SAME D published rows),
                    `published` (the first 2D rows of that order). Each in the arm's content-hash order (`the_sets`,
                    held by `hold_the_sets`). A published row is named by its id and its place, never with its text

  the held-aside    HA (the section's amendment): the candidates the search's draw did NOT take, a seeded draw of its
                    own among them (`held_aside`). Problems the start model failed in every map attempt, of the same pool
                    and kind as the ones searched, that no model here trains on (`refuse_overlap`): the cycle's own test.
                    The start model and the three trained models each make the same fresh attempts at every one

  the primary       ON HA, successes per attempt, `deep` minus `published` (`primary_branch`): above zero, DEEP SOLVES
                    ADD WHAT THE SAME NUMBER OF PUBLISHED PROOFS DO NOT; holding zero, NOT SEPARATED AT THIS SIZE; below
                    zero, THEY COST. Beside it on every branch the breadth, the problems of HA solved at least once
                    (`l4b.breadth_beside`: NARROWER)
  the mechanism     `deep` minus `shallow`, the same two reads on HA (`mechanism`): DEPTH MATTERS, or NOT SHOWN
  the replication   `shallow` minus `published`, the same two reads on HA, named plainly (`replication`)
  what follows      the spec's branch for what follows, in one line (`what_follows`)

The same reads on G' and the goal problems stand BESIDE each (the held-out goal set's word) and name no branch. A check
that failed (the search found too few deep solves; a training did not run; a model writes no proofs; Lean did not answer;
a barred row) makes the whole read INCONCLUSIVE, and nothing else is said.
"""

from __future__ import annotations

from collections import Counter
from typing import Collection, Mapping, Sequence

from rlvr_lean.domain.ladder_round.assembly import training_order
from rlvr_lean.domain.ladder_round.ceiling import interval_text
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ
from rlvr_lean.domain.ladder_round.l4_dose import THEY_COST
from rlvr_lean.domain.ladder_round.l4_rows import PRETRAINING, SIGN_TEST_LEVEL, THE_MINIMUM, THE_RULE, barred, rehearsal_rows
from rlvr_lean.domain.ladder_round.l4_rows2 import NOT_SEPARATED, _placed
from rlvr_lean.domain.ladder_round.l4b import NARROWER, breadth_beside, rehearsal_of
from rlvr_lean.domain.ladder_round.training_set import training_examples
from rlvr_lean.domain.problem_pool.episodes import VERIFIED
from rlvr_lean.domain.problem_pool.selection import rank
from rlvr_lean.domain.repair.accumulate import proof_line_count

PUBLISHED, SHALLOW, DEEP = "published", "shallow", "deep"
MODELS = (PUBLISHED, SHALLOW, DEEP)                 # the three trained models, in the order they are trained and measured
OWN = (DEEP, SHALLOW)                               # where a row that is the start model's own proof is from (its origin): a deep solve, an ordinary solve
SEARCH_LABEL, CAP_LABEL, SHALLOW_LABEL, HELD_ASIDE_LABEL = "l5_deep_search", "l5_deep_rows", "l5_shallow", "l5_held_aside"      # the four draws' own labels in the content hash

DEEP_ADDS = "DEEP SOLVES ADD WHAT THE SAME NUMBER OF PUBLISHED PROOFS DO NOT"       # the primary's interval above zero (holding zero: `NOT_SEPARATED`; below: `THEY_COST`)
PRIMARY = (DEEP_ADDS, NOT_SEPARATED, THEY_COST, INCONCLUSIVE, NOT_READ)
DEPTH_MATTERS, NOT_SHOWN = "DEPTH MATTERS", "NOT SHOWN"                             # the mechanism: `deep` minus `shallow`
MECHANISM = (DEPTH_MATTERS, NOT_SHOWN, INCONCLUSIVE, NOT_READ)
ABOVE, HOLDS_ZERO, BELOW = "ABOVE ZERO", "HOLDS ZERO", "BELOW ZERO"                  # the replication, `shallow` minus `published`, named plainly
REPLICATION = (ABOVE, HOLDS_ZERO, BELOW, INCONCLUSIVE, NOT_READ)
MORE_SEEDS = "TWO MORE SEEDS, THEN A LOOP WHOSE ROUNDS ARE DEEP SEARCHES"           # what follows: deep solves add, or depth matters with the primary not below zero
NEEDS_A_DECISION = "THE MODEL'S OWN PROOFS, DEEP OR NOT, ARE NOT ROWS TO TRAIN ON AT THIS SIZE"     # ... they cost AND narrower: needs a decision, with the three results together
RAISE_THE_DEPTH = "NOT SEPARATED: THE DEPTH IS THE THING TO RAISE BEFORE THE SEEDS"            # ... otherwise
FOLLOWS = (MORE_SEEDS, NEEDS_A_DECISION, RAISE_THE_DEPTH, NOT_READ)


# ------------------------------------------------------------------------------------------ the deep search
def failed_every_attempt(map_rows: Sequence[Mapping]) -> list[Mapping]:
    """The candidates of the deep search: the problems of the start model's own map that NONE of its map attempts
    resolved (k = 0 of the map's n). `map_rows`: the map's per-problem rows."""
    return [row for row in map_rows if row["resolved"] == 0]


def search_problems(map_rows: Sequence[Mapping], count: int, seed: int, at_least: int = 0) -> list[dict]:
    """The problems the deep search attempts: a seeded draw of `count` of the candidates (`failed_every_attempt`), the
    `count` of the smallest content hash of the task's seed and the problem's id, in that order. Each says how many of
    the map's attempts resolved it (`map_k`, 0) and why it is there (`taken_by`). Refused (ValueError) when there are
    fewer candidates than `count`.

    `at_least` (a SMOKE run alone; 0: the rule and nothing else): a fixture's map may hold no problem its model failed
    every time. The draw is then what the rule gives, at most `count`, and when that is fewer than `at_least` the
    map's OTHER problems join in the same hash order until there are that many, each marked as taken by the run's
    minimum and NOT by the rule."""
    in_order = lambda rows: sorted(rows, key=lambda row: rank(seed, SEARCH_LABEL, row["problem_id"]))      # noqa: E731
    named = lambda row, why: {"problem_id": row["problem_id"], "map_k": row["resolved"], "map_n": row["episodes"], "taken_by": why}      # noqa: E731
    candidates = in_order(failed_every_attempt(map_rows))
    if not at_least and len(candidates) < count:
        raise ValueError(f"the map holds {len(candidates)} problems its model failed in every attempt and the deep search draws {count} of them: refused")
    drawn = [named(row, THE_RULE) for row in candidates[:count]]
    if len(drawn) < at_least:
        others = in_order([row for row in map_rows if row["resolved"] > 0])
        drawn += [named(row, THE_MINIMUM) for row in others[:at_least - len(drawn)]]
    return drawn


def held_aside(map_rows: Sequence[Mapping], searched: Collection[str], count: int, seed: int, at_least: int = 0) -> list[dict]:
    """HA, the held-aside set: the candidates of the deep search (`failed_every_attempt`) that its draw did NOT take
    (`searched`: the problems it did), a seeded draw of `count` of them: the `count` of the smallest content hash of the
    task's seed and the problem's id UNDER A LABEL OF ITS OWN, so it is a draw of its own and not the next rows of the
    search's order. Each says how many of the map's attempts resolved it (`map_k`, 0) and why it is there. Refused
    (ValueError) when fewer than `count` candidates are left.

    `at_least` (a SMOKE run alone; 0: the rule and nothing else): what the rule gives, at most `count`, and when that
    is fewer than `at_least` the map's other problems that the search did not take join in the same hash order, each
    marked as taken by the run's minimum and NOT by the rule."""
    in_order = lambda rows: sorted((row for row in rows if row["problem_id"] not in searched), key=lambda row: rank(seed, HELD_ASIDE_LABEL, row["problem_id"]))      # noqa: E731
    named = lambda row, why: {"problem_id": row["problem_id"], "map_k": row["resolved"], "map_n": row["episodes"], "taken_by": why}      # noqa: E731
    left = in_order(failed_every_attempt(map_rows))
    if not at_least and len(left) < count:
        raise ValueError(f"the deep search's draw left {len(left)} of the problems the map's model failed in every attempt and the held-aside set draws {count} of them: "
                         "refused")
    drawn = [named(row, THE_RULE) for row in left[:count]]
    if len(drawn) < at_least:
        others = in_order([row for row in map_rows if row["resolved"] > 0])
        drawn += [named(row, THE_MINIMUM) for row in others[:at_least - len(drawn)]]
    return drawn


def refuse_overlap(aside: Collection[str], searched: Collection[str], sets: Mapping[str, Sequence[Mapping]]) -> None:
    """The held-aside set is the cycle's own TEST: refused (ValueError) when a problem of it is one the deep search
    attempted, or is the problem of a row of any training set (a deep row, an ordinary solve's, a published row).
    `aside`, `searched`: problem ids. `sets`: the training sets, by model (none yet when the search is prepared)."""
    attempted = sorted(set(aside) & set(searched))
    trained = {model: sorted(set(aside) & {row["problem_id"] for row in rows}) for model, rows in sets.items()}
    if attempted or any(trained.values()):
        first = attempted[0] if attempted else next(own[0] for own in trained.values() if own)
        raise ValueError(f"{len(attempted)} problems of the held-aside set are problems the deep search attempted"
                         + "".join(f", {len(own)} are rows of `{model}`" for model, own in trained.items()) + f" (first: {first}): no model here trains on a held-aside "
                         "problem, and none of them was searched. Refused")


def solve_rows(problems: Sequence[Mapping], attempts: Sequence[Mapping], results: Sequence[Mapping], seed: int, origin: str, attempts_before: int = 0) -> list[dict]:
    """One training row for every problem of `problems` that has a verified attempt among `attempts`: ONE of its
    verified attempts, drawn by the seed AS A ROUND DRAWS ONE (`training_set.training_examples`, the same function: the
    first in a seeded order of attempt ids, on whichever side was proved), in the problems' order. `results`: the
    per-problem rows of that sampling. Each row has its id (the attempt's), its `origin` (`deep`, or `shallow`), its
    problem's k of n in that sampling, the lines of its proof, and `first_success`: the place of the problem's FIRST
    verified attempt, counting the `attempts_before` attempts made at it earlier (a deep solve: the map's, so 9 to 64
    after a map of 8 and a search of 56).

    Refused (ValueError) when the stored sampling does not agree with itself: a problem its results count as resolved
    has no verified attempt, or the reverse."""
    if origin not in OWN:
        raise ValueError(f"{origin!r} is not where a row of the model's own is from: it is one of {list(OWN)}")
    wanted = {problem["problem_id"] for problem in problems}
    first: dict[str, int] = {}
    for attempt in attempts:
        if attempt["status"] == VERIFIED and attempt["problem_id"] in wanted:
            first[attempt["problem_id"]] = min(first.get(attempt["problem_id"], attempt["episode"]), attempt["episode"])
    of = {row["problem_id"]: row for row in results if row["problem_id"] in wanted}
    resolved = {problem_id for problem_id, row in of.items() if row["resolved"] > 0}
    if resolved != set(first) or set(of) != wanted:
        other = sorted(resolved ^ set(first)) or sorted(set(of) ^ wanted)
        raise ValueError(f"the stored sampling does not agree with itself: {len(resolved)} of its {len(of)} problems are counted as resolved, {len(first)} of the "
                         f"{len(wanted)} asked for have a verified attempt (first that differs: {other[0]}). Refused")
    return [{"id": example["attempt_id"], **example, "origin": origin, "k": of[example["problem_id"]]["resolved"], "n": of[example["problem_id"]]["episodes"],
             "lines": proof_line_count(example["completion"]), "first_success": attempts_before + first[example["problem_id"]] + 1}
            for example in training_examples(problems, [attempt for attempt in attempts if attempt["problem_id"] in wanted], seed)]


def capped(rows: Sequence[Mapping], cap: int, seed: int) -> list[Mapping]:
    """The deep rows a set holds: every one when there are at most `cap`; otherwise a seeded draw of `cap` of them, the
    rows of the smallest content hash of the task's seed and the row's problem, none moved among the others."""
    if len(rows) <= cap:
        return list(rows)
    kept = {row["problem_id"] for row in sorted(rows, key=lambda row: rank(seed, CAP_LABEL, row["problem_id"]))[:cap]}
    return [row for row in rows if row["problem_id"] in kept]


def ordinary_draw(map_rows: Sequence[Mapping], least: int, most: int, count: int, seed: int, not_among: Collection[str] = (), loosely: bool = False) -> list[dict]:
    """The ORDINARY solves `shallow` is trained on: a seeded draw of `count` of the map's problems the start model
    solved in `least` to `most` of its map attempts (the `count` of the smallest content hash of the task's seed and the
    problem's id, in that order). Refused (ValueError) when there are fewer.

    `loosely` (a SMOKE run alone), `not_among` (the problems its deep search attempted, which a fixture's looser search
    may have taken from the solved ones): when the rule gives fewer than `count`, the map's other SOLVED problems join in
    the same hash order, each marked as taken by the run's minimum and NOT by the rule."""
    in_order = lambda rows: sorted((row for row in rows if row["problem_id"] not in not_among), key=lambda row: rank(seed, SHALLOW_LABEL, row["problem_id"]))      # noqa: E731
    named = lambda row, why: {"problem_id": row["problem_id"], "map_k": row["resolved"], "map_n": row["episodes"], "taken_by": why}      # noqa: E731
    of_the_rule = in_order([row for row in map_rows if least <= row["resolved"] <= most])
    drawn = [named(row, THE_RULE) for row in of_the_rule[:count]]
    if len(drawn) < count and loosely:
        others = in_order([row for row in map_rows if row["resolved"] > most or 0 < row["resolved"] < least])
        drawn += [named(row, THE_MINIMUM) for row in others[:count - len(drawn)]]
    if len(drawn) < count:
        raise ValueError(f"the map holds {len(of_the_rule)} problems its model solved in {least} to {most} of its attempts and `{SHALLOW}` takes {count} of them (as many "
                         f"as `{DEEP}` has deep rows): refused")
    return drawn


# ------------------------------------------------------------------------------------------------- the sets
def hold_the_sets(sets: Mapping[str, Sequence[Mapping]], pretraining: Sequence[Mapping], seed: int) -> int:
    """What the three sets are HELD to, each a refusal (ValueError); returns D. Each set holds 2D rows and no id twice.
    `deep`: D rows of origin `deep` and D published rows; `shallow`: D rows of origin `shallow` and THE SAME D published
    rows, by id and by place in the file; `published`: 2D published rows and no other. The D published rows are the
    FIRST D of the hash order L4t's rehearsal draw uses (`l4_rows.rehearsal_rows`: the draw L4p's `matched_1262` is the
    first rows of), and `published`'s are the first 2D, so the D are the first of `published`'s too. No problem is in
    both `deep` and `shallow` (a deep solve was failed in every map attempt, an ordinary one was not), and no row of the
    model's own is of a problem the pretraining file holds. Each set stands in the arm's content-hash order."""
    own = {model: [row for row in sets[model] if row["origin"] != PRETRAINING] for model in MODELS}
    of_the_file = {model: rehearsal_of(sets[model]) for model in MODELS}
    count = len(own[DEEP])
    sizes = {model: (len(own[model]), len(of_the_file[model])) for model in MODELS}
    if not count or sizes != {DEEP: (count, count), SHALLOW: (count, count), PUBLISHED: (0, 2 * count)}:
        raise ValueError(f"the three sets hold (rows of the model's own, published rows) {sizes}: `{DEEP}` is D deep rows and D published rows, `{SHALLOW}` D ordinary "
                         f"solves and the same D published rows, `{PUBLISHED}` 2D published rows, D at least 1. Refused")
    for model in MODELS:
        strangers = [row["id"] for row in own[model] if row["origin"] != model]
        twice = sorted(key for key, times in Counter(row["id"] for row in sets[model]).items() if times > 1)
        if strangers or twice:
            raise ValueError(f"`{model}` holds {len(strangers)} rows of another origin than its own and {len(twice)} ids twice (first: {(strangers or twice)[0]}). Refused")
        if [row["id"] for row in sets[model]] != [row["id"] for row in training_order(sets[model], seed)]:
            raise ValueError(f"`{model}` does not stand in the content-hash order of this seed: every model here is trained in that order. Refused")
    order = rehearsal_rows(pretraining, 2 * count, seed)            # the draw's own order, the smallest hash first: L4t's, L4t2's and L4p's draw, with this count
    first = rehearsal_rows(pretraining, count, seed)
    if list(map(_placed, first)) != list(map(_placed, order[:count])):
        raise ValueError(f"the {count} rows drawn for `{DEEP}` and `{SHALLOW}` are not the first {count} of the hash order `{PUBLISHED}`'s {2 * count} stand in. Refused")
    for model, wanted in ((DEEP, first), (SHALLOW, first), (PUBLISHED, order)):
        if sorted(map(_placed, of_the_file[model])) != sorted(map(_placed, wanted)):
            other = sorted(set(map(_placed, of_the_file[model])) ^ set(map(_placed, wanted)))
            raise ValueError(f"the {len(of_the_file[model])} published rows of `{model}` are not the first {len(wanted)} of the hash order among the {len(pretraining)} rows "
                             f"the start model was pretrained on (first that differs: {other[0][0]}): the three sets would not hold the same published rows. Refused")
    if list(map(_placed, of_the_file[DEEP])) != list(map(_placed, of_the_file[SHALLOW])):
        raise ValueError(f"`{DEEP}` and `{SHALLOW}` do not hold the same published rows in the same order. Refused")
    both = sorted({row["problem_id"] for row in own[DEEP]} & {row["problem_id"] for row in own[SHALLOW]})
    seen = sorted({row["problem_id"] for model in OWN for row in own[model]} & {row["problem_id"] for row in pretraining})
    if both or seen:
        raise ValueError(f"{len(both)} problems are in both `{DEEP}` and `{SHALLOW}`, and {len(seen)} problems of the model's own rows are rows of the pretraining file "
                         f"(first: {(both or seen)[0]}): a deep solve is no ordinary solve, and neither is a problem whose published proof a model here has seen. Refused")
    return count


def the_sets(deep: Sequence[Mapping], shallow: Sequence[Mapping], pretraining: Sequence[Mapping], seed: int) -> dict[str, list[Mapping]]:
    """The three training sets, each in the arm's content-hash order (`assembly.training_order`). `deep`: the D deep
    rows (`solve_rows`, under the cap: `capped`). `shallow`: the D rows of ordinary solves. `pretraining`: the rows the
    start model was pretrained on, in the file's order, WITHOUT their text. `seed`: the task's.

      published   the first 2D rows of the hash order L4t's rehearsal draw uses (`l4_rows.rehearsal_rows`), and nothing else
      shallow     the D ordinary solves and the first D rows of that order
      deep        the D deep rows and the SAME D rows of that order

    Refused (ValueError) as `hold_the_sets` refuses, which holds what is built here from the outside."""
    count = len(deep)
    if not count or len(shallow) != count:
        raise ValueError(f"`{DEEP}` has {count} deep rows and `{SHALLOW}` {len(shallow)} ordinary solves: the two hold the same number, at least 1. Refused")
    first = rehearsal_rows(pretraining, count, seed)
    sets = {PUBLISHED: training_order(rehearsal_rows(pretraining, 2 * count, seed), seed), SHALLOW: training_order([*shallow, *first], seed),
            DEEP: training_order([*deep, *first], seed)}
    hold_the_sets(sets, pretraining, seed)
    return sets


# --------------------------------------------------------------------------------------------- what is barred
def own_barred(rows: Sequence[Mapping], heldout: Collection[str], half_seed: int, base_map: Collection[str]) -> dict:
    """Of a set's rows (`problem_id`, `origin`): the held-out problems among them and the published rows that are NOT of
    the `pretrain` half (both L4t's own count, `l4_rows.barred`); and, among the rows of the model's own, the problems
    that are NOT base-map problems. (A base-map problem is in neither half's files: no model here has seen its
    published proof and no round draws it, whatever the hash of its id says of a half.)"""
    found = barred(rows, heldout, half_seed)
    own = [row for row in rows if row["origin"] != PRETRAINING]
    return {"rows": found["rows"], "rows_of_the_models_own": len(own), "published_rows": found["rehearsal_rows"], "held_out": found["held_out"],
            "published_rows_not_of_the_pretrain_half": found["rehearsal_rows_not_of_the_pretrain_half"],
            "own_rows_not_of_the_base_map": sorted({row["problem_id"] for row in own if row["problem_id"] not in base_map})}


def refuse_barred(rows: Sequence[Mapping], heldout: Collection[str], half_seed: int, base_map: Collection[str], what: str) -> dict:
    """A training set is refused (ValueError) BEFORE anything is trained when it holds a held-out problem, a published
    row that is not of the `pretrain` half, or a deep or shallow row that is not a base-map problem. Returns what was
    counted."""
    found = own_barred(rows, heldout, half_seed, base_map)
    for key, said in (("held_out", "are held-out problems"), ("published_rows_not_of_the_pretrain_half", "are published rows that are not of the `pretrain` half"),
                      ("own_rows_not_of_the_base_map", "are rows of the model's own that are not base-map problems")):
        if found[key]:
            raise ValueError(f"{len(found[key])} problems of {what} {said} (first: {found[key][0]}): refused, nothing was written")
    return found


def no_barred_row(trained: Sequence[Mapping], heldout: Collection[str], half_seed: int, base_map: Collection[str]) -> dict:
    """The fourth check of a trained model here, read on the rows the TRAINING LOOP recorded: no row it was trained on is
    a held-out problem; its published rows are of the `pretrain` half; its own rows (deep or ordinary solves) are
    base-map problems."""
    found = own_barred(trained, heldout, half_seed, base_map)
    first = [*found["held_out"], *found["published_rows_not_of_the_pretrain_half"], *found["own_rows_not_of_the_base_map"]]
    return {"what": "no row it was trained on is a held-out problem; its published rows are rows of the pretraining file, of the `pretrain` half; the rows of the "
                    "model's own (deep solves, ordinary solves) are base-map problems, whose published proofs no model here has seen",
            "rows": found["rows"], "rows_of_the_models_own": found["rows_of_the_models_own"], "published_rows_of_the_pretrain_half": found["published_rows"],
            "held_out_problems": len(found["held_out"]), "published_rows_not_of_the_pretrain_half": len(found["published_rows_not_of_the_pretrain_half"]),
            "own_rows_not_of_the_base_map": len(found["own_rows_not_of_the_base_map"]), "first": first[:5], "passes": bool(trained) and not first}


# ------------------------------------------------------------------------------------- the deep search's table
def bands(before: int, total: int) -> list[tuple[int, int]]:
    """The attempts after the map's, in bands that each end at twice the one before: 9 to 16, 17 to 32, 33 to 64 for a
    map of 8 attempts and 64 in all."""
    found, low, high = [], before + 1, 2 * before
    while low <= total and high >= low:
        found.append((low, min(high, total)))
        low, high = high + 1, 2 * high
    return found


def enough_deep_solves(solved: int, minimum: int) -> dict:
    """The first check: the deep search found at least `minimum` deep solves. Fewer: the three sets would be too small
    to read, the three trainings are not run, and the search is the finding."""
    return {"what": f"the deep search found at least {minimum} deep solves (fewer: the three sets would be too small to read; the search is then the finding)",
            "deep_solves": solved, "minimum": minimum, "passes": solved >= minimum}


def search_table(drawn: Sequence[Mapping], results: Sequence[Mapping], solves: Sequence[Mapping], before: int, episodes: int) -> dict:
    """The deep search itself. `drawn`: the problems it attempted (`search_problems`). `results`: its per-problem
    rows. `solves`: the deep rows (`solve_rows`), one a problem solved. `before`: the map's attempts a problem;
    `episodes`: the search's. The problems drawn and solved, the solved by the attempt of the first success (`bands`),
    the attempts, the successes per 1,000."""
    total = before + episodes
    by_band = {f"{low}_to_{high}": sum(low <= row["first_success"] <= high for row in solves) for low, high in bands(before, total)}
    successes, made = sum(row["resolved"] for row in results), sum(row["episodes"] for row in results)
    return {"what": f"the problems the deep search attempted ({episodes} more attempts each, {total} with the map's {before}); the problems it solved, by the attempt of "
                    "the first success counting the map's; its attempts (an episode is one attempt at the side the published answer allows, and one at the other for "
                    "the audited share); the episodes that resolved their problem, per 1,000",
            "problems": len(drawn), "taken_by_the_rule": sum(row["taken_by"] == THE_RULE for row in drawn), "taken_by_the_runs_minimum": sum(row["taken_by"] != THE_RULE for row in drawn),
            "solved": len(solves), "share_solved": round(len(solves) / len(drawn), 5) if drawn else None, "solved_by_first_success": by_band,
            "episodes_each": episodes, "attempts_before_each": before, "episodes": made, "attempts": sum(row["episodes"] * row["sides"] for row in results),
            "successes": successes, "per_1000": round(1000 * successes / made, 2) if made else None,
            "without_an_answer": sum(row["attempts_without_an_answer"] for row in results)}


# --------------------------------------------------------------------------------------------- the branches
def _beside(solved: Mapping, after: str, before: str, level: float) -> dict:
    """The breadth beside a difference: the problems of the held-aside set solved at least once in their attempts,
    `after` against `before` (gained = by `after` and not by `before`), with the two-sided sign test. NARROWER by L4b's
    own rule (`l4b.breadth_beside`: lost more than gained under `level`), here against `before`."""
    breadth = breadth_beside(solved, None, None, before, level)
    gained, lost, p = breadth["gained"], breadth["lost"], breadth["sign_test_p"]
    return {"what": f"the held-aside problems solved at least once in their attempts, `{after}` against `{before}`, paired by problem (gained = by `{after}` and not by "
                    f"`{before}`), with the two-sided sign test; lost more than gained under {level:g} is written {NARROWER}",
            **{key: breadth[key] for key in ("problems", "gained", "lost", "sign_test_p", "level", "counts", "narrower", "smallest_decisive_split")},
            f"solved_by_{after}": solved["resolved_after"], f"solved_by_{before}": solved["resolved_before"],
            "broader": bool(gained > lost and p is not None and p < level)}


def primary_branch(failed_checks: Sequence[str], difference: Mapping, solved: Mapping, level: float = SIGN_TEST_LEVEL) -> dict:
    """The primary's branch, fixed before the run (the section's amendment: ON THE HELD-ASIDE SET). `difference`: on
    HA, successes per attempt, `deep` minus `published`, paired by problem, with its interval. `solved`: the problems
    of HA solved at least once, `deep` against `published`. `failed_checks`: the checks that failed (the search's, a
    model's, Lean's answers on a held-aside set): INCONCLUSIVE, and nothing else is said. Nothing measured on both
    sides: NOT READ.

    The interval above zero: DEEP SOLVES ADD WHAT THE SAME NUMBER OF PUBLISHED PROOFS DO NOT. It holds zero (one that
    ends at zero holds it): NOT SEPARATED AT THIS SIZE, with what this difference resolves (half the width of its
    interval). Below zero: THEY COST. On every branch the breadth stands beside: NARROWER when `deep` lost more
    held-aside problems than it gained against `published` with the sign test under `level`."""
    if failed_checks:
        return {"name": INCONCLUSIVE, "failed_checks": list(failed_checks), "narrower": None,
                "reason": f"the run could not have been read ({', '.join(failed_checks)}): nothing is said about what deep solves teach"}
    if difference.get("mean") is None:
        return {"name": NOT_READ, "failed_checks": [], "narrower": None, "reason": "no held-aside problem was measured on both sides: the primary has nothing to be read on"}
    read, breadth = interval_text(difference), _beside(solved, DEEP, PUBLISHED, level)
    beside = (f"{NARROWER}" if breadth["narrower"] else f"not {NARROWER.lower()}") + f" against `{PUBLISHED}` ({breadth['counts']})"
    found = {"failed_checks": [], "narrower": breadth["narrower"], "breadth": breadth}
    if difference["low"] > 0:
        return {"name": DEEP_ADDS, **found,
                "reason": f"on the held-aside problems `{DEEP}` minus `{PUBLISHED}` is above zero ({read}): deep solves add what the same number of published "
                          f"proofs do not; {beside}"}
    if difference["high"] < 0:
        return {"name": THEY_COST, **found,
                "reason": f"on the held-aside problems `{DEEP}` minus `{PUBLISHED}` is below zero ({read}): in the place of the same number of published proofs the "
                          f"deep solves cost; {beside}"}
    resolves = round((difference["high"] - difference["low"]) / 2, 5)
    return {"name": NOT_SEPARATED, **found, "resolves": resolves,
            "reason": f"on the held-aside problems `{DEEP}` minus `{PUBLISHED}` holds zero ({read}): not separated at this size. This difference resolves "
                      f"{resolves:.5f} per attempt (half the width of its interval); {beside}"}


def mechanism(failed_checks: Sequence[str], difference: Mapping, solved: Mapping, level: float = SIGN_TEST_LEVEL) -> dict:
    """The claim under test: that a solve the model could not sample extends it where one it could does not. `deep`
    minus `shallow`, the same two reads ON THE HELD-ASIDE SET: `difference` the successes per attempt, `solved` the
    problems of HA solved at least once (gained = by `deep` and not by `shallow`). DEPTH MATTERS when the interval is
    above zero (one that ends at zero is not), OR more were gained than lost with the sign test under `level`.
    Otherwise NOT SHOWN. INCONCLUSIVE on a failed check; NOT READ when nothing was measured on both sides."""
    if failed_checks:
        return {"name": INCONCLUSIVE, "failed_checks": list(failed_checks), "reason": f"the run could not have been read ({', '.join(failed_checks)}): the mechanism is not read"}
    if difference.get("mean") is None:
        return {"name": NOT_READ, "failed_checks": [], "reason": "no held-aside problem was measured on both sides: the mechanism has nothing to be read on"}
    read, breadth = interval_text(difference), _beside(solved, DEEP, SHALLOW, level)
    above = bool(difference["low"] > 0)
    found = {"failed_checks": [], "above_zero_on_the_held_aside_set": above, "broader": breadth["broader"], "narrower": breadth["narrower"], "breadth": breadth}
    said = f"on the held-aside problems `{DEEP}` minus `{SHALLOW}` is {read}; held-aside problems solved, `{DEEP}` against `{SHALLOW}`: {breadth['counts']}"
    if above or breadth["broader"]:
        why = " and ".join([*(["the interval is above zero"] if above else []), *([f"more were gained than lost with the sign test under {level:g}"] if breadth["broader"] else [])])
        return {"name": DEPTH_MATTERS, **found, "reason": f"{said}: {why}: a solve the model could not sample extends it where one it could does not"}
    return {"name": NOT_SHOWN, **found,
            "reason": f"{said}: the interval is not above zero and not more were gained than lost with the sign test under {level:g}: not shown"
                      + (f"; `{DEEP}` is {NARROWER} against `{SHALLOW}`" if breadth["narrower"] else "")}


def replication(failed_checks: Sequence[str], difference: Mapping, solved: Mapping, level: float = SIGN_TEST_LEVEL) -> dict:
    """The replication L4c owes, at equal steps and equal published rows: `shallow` minus `published`, the same two
    reads ON THE HELD-ASIDE SET, NAMED PLAINLY: the interval of the successes per attempt is ABOVE ZERO, HOLDS ZERO
    (one that ends at zero holds it) or is BELOW ZERO; and `shallow` is NARROWER against `published` or is not."""
    if failed_checks:
        return {"name": INCONCLUSIVE, "failed_checks": list(failed_checks), "narrower": None,
                "reason": f"the run could not have been read ({', '.join(failed_checks)}): the replication is not read"}
    if difference.get("mean") is None:
        return {"name": NOT_READ, "failed_checks": [], "narrower": None, "reason": "no held-aside problem was measured on both sides: the replication has nothing to be read on"}
    read, breadth = interval_text(difference), _beside(solved, SHALLOW, PUBLISHED, level)
    name = ABOVE if difference["low"] > 0 else BELOW if difference["high"] < 0 else HOLDS_ZERO
    return {"name": name, "failed_checks": [], "narrower": breadth["narrower"], "breadth": breadth,
            **({"resolves": round((difference["high"] - difference["low"]) / 2, 5)} if name == HOLDS_ZERO else {}),
            "reason": f"on the held-aside problems `{SHALLOW}` minus `{PUBLISHED}` is {read}: {name}; " + (f"{NARROWER}" if breadth["narrower"] else f"not {NARROWER.lower()}")
                      + f" against `{PUBLISHED}` ({breadth['counts']})"}


def what_follows(primary: Mapping, of_the_mechanism: Mapping) -> dict:
    """The spec's branch for what follows, in its order. DEEP SOLVES ADD, or DEPTH MATTERS with the primary not below
    zero: two more seeds, then a loop whose rounds ARE deep searches. THEY COST and NARROWER: the model's own proofs,
    deep or not, are not rows to train on at this size, and that needs a decision, with the three results together.
    Otherwise: not separated, and the depth is the thing to raise before the seeds. Nothing is named when the primary
    is INCONCLUSIVE or was not read."""
    name, depth = primary["name"], of_the_mechanism["name"]
    if name in (INCONCLUSIVE, NOT_READ):
        return {"name": NOT_READ, "reason": f"the primary is {name}: no branch for what follows is named"}
    if name == DEEP_ADDS or (depth == DEPTH_MATTERS and name != THEY_COST):
        return {"name": MORE_SEEDS, "reason": f"the primary reads {name} and the mechanism {depth}: two more seeds, then a loop whose rounds ARE deep searches"}
    if name == THEY_COST and primary["narrower"]:
        return {"name": NEEDS_A_DECISION, "reason": f"the primary reads {name} and `{DEEP}` is {NARROWER} against `{PUBLISHED}`: the model's own proofs, deep or not, are not "
                                                "rows to train on at this size, and that needs a decision, with the three results together (L4p's, L4c's and this one)"}
    return {"name": RAISE_THE_DEPTH, "reason": f"the primary reads {name}" + (f" (not {NARROWER.lower()})" if name == THEY_COST else "") + f" and the mechanism {depth}: "
                                               "not separated, and the depth is the thing to raise before the seeds"}


# ------------------------------------------------------------------------------------------- further seeds
# Spec "L5, further seeds" and its amendment ("where the lead sits, and a read of reach"). The search's own seed found
# the goal set's lead, so it cannot confirm it: the lead, its split and the reach are read ON THE NEW SEEDS ALONE.
NEW_SEEDS = 2                                       # the further seeds the reads are fixed on; with the search's own, the three seeds of the primary
A_POINT = 0.01                                      # two models' shares of attempts that timed out in Lean "differ by more than a point"
NOT_SHOWN_AT_THREE = "NOT SHOWN AT THREE SEEDS"     # the primary over the three seeds, its interval holding zero (above: `DEEP_ADDS`; below: `THEY_COST`)
OVER_THE_SEEDS = (DEEP_ADDS, NOT_SHOWN_AT_THREE, THEY_COST, NOT_READ)
LEAD_CONFIRMED, NOT_CONFIRMED = "THE LEAD IS CONFIRMED", "NOT CONFIRMED"
LEAD = (LEAD_CONFIRMED, NOT_CONFIRMED, NOT_READ)
RELIABILITY, REACH_TOO = "RELIABILITY", "REACH TOO"                                 # what a CONFIRMED lead is named by its split (one that is not confirmed has no name)
REACH_CONFIRMED = "REACH IS CONFIRMED ON THE LONG PROBLEMS"
REACH = (REACH_CONFIRMED, NOT_CONFIRMED, NOT_READ)
A_LOOP_OF_DEEP_SEARCHES = "A LOOP WHOSE ROUNDS ARE DEEP SEARCHES, WITH ORDINARY SOLVES MATCHED IN LENGTH AS A CONTROL"
RELIABILITY_ALONE = "RELIABILITY, AND REACH NOT SHOWN: THE DEPTH OF THE SEARCH IS THE THING TO RAISE"
STOP = "STOP: THE THREE SEEDS NEED A DECISION"
OTHERWISE = "OTHERWISE: THE DEPTH OR THE DOSE IS THE THING TO RAISE"
AFTER_THE_SEEDS = (A_LOOP_OF_DEEP_SEARCHES, RELIABILITY_ALONE, STOP, OTHERWISE, NOT_READ)


def _each(at_each: Mapping[int, float]) -> str:
    return ", ".join(f"seed {seed} {value:+.5f}" for seed, value in sorted(at_each.items()))


def primary_over_the_seeds(difference: Mapping, at_each: Mapping[int, float], seeds: int = NEW_SEEDS + 1) -> dict:
    """THE PRIMARY OVER THE THREE SEEDS: on the held-aside set, successes per attempt, `deep` minus `published`, each
    problem's mean over the seeds of its difference (`difference`, with its interval over problems); `at_each`: each
    seed's own difference. Interval above zero: DEEP SOLVES ADD WHAT THE SAME NUMBER OF PUBLISHED PROOFS DO NOT. It
    holds zero (one that ends at zero holds it): NOT SHOWN AT THREE SEEDS. Below zero: THEY COST. NOT READ unless it is
    over exactly `seeds` seeds (the search's own and the two further) and was measured."""
    if len(at_each) != seeds or difference.get("mean") is None:
        return {"name": NOT_READ, "seeds": sorted(at_each),
                "reason": f"the primary is fixed over {seeds} seeds (the search's own and {seeds - 1} further) and {len(at_each)} are read here: no branch is named"}
    read = f"on the held-aside problems `{DEEP}` minus `{PUBLISHED}`, each problem's mean over the {seeds} seeds, is {interval_text(difference)} ({_each(at_each)})"
    signs = {"positive": sorted(seed for seed, value in at_each.items() if value > 0), "negative": sorted(seed for seed, value in at_each.items() if value < 0)}
    name = DEEP_ADDS if difference["low"] > 0 else THEY_COST if difference["high"] < 0 else NOT_SHOWN_AT_THREE
    said = {DEEP_ADDS: "above zero: deep solves add what the same number of published proofs do not",
            THEY_COST: "below zero: in the place of the same number of published proofs the deep solves cost", NOT_SHOWN_AT_THREE: "it holds zero: not shown at three seeds"}[name]
    return {"name": name, "seeds": sorted(at_each), "signs": signs, **({"resolves": round((difference["high"] - difference["low"]) / 2, 5)} if name == NOT_SHOWN_AT_THREE else {}),
            "reason": f"{read}: {said}"}


def the_lead(difference: Mapping, at_each: Mapping[int, float], seeds: int = NEW_SEEDS) -> dict:
    """THE GOAL SET'S LEAD, on the NEW seeds alone (the search's own seed found it, so it cannot confirm it): on all of
    G over all the attempts, `deep` minus `published`, each problem's mean over the new seeds (`difference`);
    `at_each`: each new seed's own difference. THE LEAD IS CONFIRMED when the interval is above zero AND the difference
    is positive at BOTH new seeds; otherwise NOT CONFIRMED. NOT READ unless exactly `seeds` new seeds are read."""
    if len(at_each) != seeds or difference.get("mean") is None:
        return {"name": NOT_READ, "seeds": sorted(at_each),
                "reason": f"the lead is confirmed or not on the {seeds} new seeds alone, and {len(at_each)} of them are read here: nothing is named"}
    above, at_both = bool(difference["low"] > 0), all(value > 0 for value in at_each.values())
    read = f"on all of G `{DEEP}` minus `{PUBLISHED}`, each problem's mean over the {seeds} new seeds, is {interval_text(difference)} ({_each(at_each)})"
    return {"name": LEAD_CONFIRMED if above and at_both else NOT_CONFIRMED, "seeds": sorted(at_each), "interval_above_zero": above, "positive_at_both_new_seeds": at_both,
            "reason": f"{read}: the interval is {'above zero' if above else 'not above zero'} and the difference is "
                      + ("positive at both new seeds" if at_both else "not positive at both new seeds")}


def the_split(lead: Mapping, on_again: Mapping) -> dict:
    """What a CONFIRMED lead is named by where it sits (the amendment): `on_again` is `deep` minus `published` per
    attempt on G' (the goal problems the start model fails in its first sampling), over all the attempts, each
    problem's mean over the new seeds. RELIABILITY when it holds zero (one that ends at zero holds it), REACH TOO when
    it is above zero. BELOW zero is RELIABILITY too, and its line says so: the lead then sits wholly on the problems
    the start model already solves. A lead that is not confirmed, or was not read, has NO name."""
    if lead["name"] != LEAD_CONFIRMED or on_again.get("mean") is None:
        return {"name": None, "reason": f"the lead reads {lead['name']}: its split names nothing"}
    below = bool(on_again["high"] < 0)
    name = REACH_TOO if on_again["low"] > 0 else RELIABILITY
    return {"name": name, "below_zero_on_the_goal_set_again": below,
            "reason": f"on G' `{DEEP}` minus `{PUBLISHED}` is {interval_text(on_again)}: "
                      + ("above zero: the confirmed lead is reach too" if name == REACH_TOO else
                         ("BELOW zero" if below else "it holds zero") + ": the confirmed lead is reliability, on the problems the start model already solves")}


def reach_on_the_long(difference: Mapping, counts: Mapping[int, Sequence[int]], seeds: int = NEW_SEEDS, after: str = DEEP, before: str = PUBLISHED) -> dict:
    """REACH ON THE LONG PROBLEMS, on the NEW seeds alone (the amendment). `difference`: over the goal problems whose
    shortest published proof has 8 lines or more, the share of the new seeds in which a model solves the problem at
    least once in all its attempts, `deep` minus `published`, paired, with its interval over those problems. `counts`:
    at each new seed, how many of them (`deep`, `published`) solve. REACH IS CONFIRMED ON THE LONG PROBLEMS when the
    interval is above zero AND `deep` solves MORE of them than `published` at BOTH new seeds; otherwise NOT CONFIRMED.
    NOT READ unless exactly `seeds` new seeds are read. `after`, `before`: the same read of another pair (`deep` minus
    `shallow` stands beside it, and names no branch)."""
    if len(counts) != seeds or difference.get("mean") is None:
        return {"name": NOT_READ, "seeds": sorted(counts),
                "reason": f"reach on the long problems is confirmed or not on the {seeds} new seeds alone, and {len(counts)} of them are read here: nothing is named"}
    above, at_both = bool(difference["low"] > 0), all(after > before for after, before in counts.values())
    said = ", ".join(f"seed {seed} {after} against {before}" for seed, (after, before) in sorted(counts.items()))
    return {"name": REACH_CONFIRMED if above and at_both else NOT_CONFIRMED, "seeds": sorted(counts), "interval_above_zero": above, "more_at_both_new_seeds": at_both,
            "reason": f"on the long problems the share of the {seeds} new seeds in which a problem is solved, `{after}` minus `{before}`, is {interval_text(difference)}; "
                      f"long problems solved, `{after}` against `{before}`: {said}: the interval is {'above zero' if above else 'not above zero'} and `{after}` "
                      + ("solves more at both new seeds" if at_both else "does not solve more at both new seeds")}


def after_the_seeds(primary: Mapping, lead: Mapping, split: Mapping, reach: Mapping) -> dict:
    """The section's branch for what follows the three seeds, in ITS order (the amended "Branches"). REACH CONFIRMED ON
    THE LONG PROBLEMS, or ADD on the held-aside set: a loop whose rounds are deep searches, with ordinary solves matched
    in length as a control. The lead confirmed as RELIABILITY and reach not: deep solves are the rows of its own a
    model can train on without narrowing; reach is not shown and the depth of the search is the thing to raise. THEY
    COST on the held-aside set: stop, and the three seeds need a decision. Otherwise: the depth or the dose is the
    thing to raise. Nothing is named while a read it stands on is NOT READ."""
    names = {"the primary": primary["name"], "the lead": lead["name"], "reach on the long problems": reach["name"]}
    read = "; ".join(f"{what} reads {name}" + (f" ({split['name']})" if what == "the lead" and split.get("name") else "") for what, name in names.items())
    if NOT_READ in names.values():
        return {"name": NOT_READ, "reason": f"{read}: no branch for what follows is named before the three seeds are read"}
    if reach["name"] == REACH_CONFIRMED or primary["name"] == DEEP_ADDS:
        return {"name": A_LOOP_OF_DEEP_SEARCHES,
                "reason": f"{read}: the next stage is a loop whose rounds are deep searches, with ordinary solves matched in length as a control (depth is not yet apart from "
                          "length), specified then"}
    if lead["name"] == LEAD_CONFIRMED and split.get("name") == RELIABILITY:
        return {"name": RELIABILITY_ALONE,
                "reason": f"{read}: deep solves are the rows of its own a model can train on without narrowing, which ordinary solves are not; reach is not shown and the "
                          "depth of the search is the thing to raise"}
    if primary["name"] == THEY_COST:
        return {"name": STOP, "reason": f"{read}: stop, and the three seeds need a decision"}
    return {"name": OTHERWISE,
            "reason": f"{read}: " + ("the lead is not confirmed, deep solves are no worse than published rows at equal steps (unlike ordinary ones in L4c), and the depth or the "
                                     "dose is the thing to raise" if lead["name"] != LEAD_CONFIRMED else
                                     "the section names no branch of its own for a lead confirmed as reach too whose reach on the long problems is not: it is read as its "
                                     "\"otherwise\", and the depth or the dose is the thing to raise")}


def timed_out(health: Mapping[str, Mapping], kinds: Mapping[str, Mapping[str, str]]) -> dict:
    """Each measured set's share of attempts that timed out in Lean, and where two models' shares differ by more than
    a point (`A_POINT`). `health`: by measured set, its `attempts` and how many `timed_out_in_lean`. `kinds`: by kind of
    set (the rungs, a sampling of G, the held-aside set), the set of each model that was measured on it. Such a pair's
    timed-out attempts are to be checked again (Lean only, no measurement running) and the reads given both ways:
    named here, done by no step."""
    shares = {key: (entry["timed_out_in_lean"] / entry["attempts"] if entry["attempts"] else None) for key, entry in health.items()}
    apart = []
    for kind, of in kinds.items():
        models = [model for model in of if shares.get(of[model]) is not None]
        for index, first in enumerate(models):
            for second in models[index + 1:]:
                if abs(shares[of[first]] - shares[of[second]]) > A_POINT:
                    apart.append({"kind": kind, "models": [first, second], "shares": [round(shares[of[first]], 5), round(shares[of[second]], 5)]})
    return {"what": "the share of each measured set's attempts that timed out in Lean; `apart`: the pairs of models whose shares on one kind of set differ by more than "
                    f"{A_POINT:g}: their timed-out attempts are to be checked again (Lean only, no measurement running) and the reads given both ways, which no step here does",
            "a_point": A_POINT, "share": {key: None if value is None else round(value, 5) for key, value in shares.items()}, "apart": apart}
