"""L5c: the continuation. What lifted the published kind by 34 per 1,000 from the twice-pretrained model, and does more
of it go higher? Spec: docs/spec/ladder-loop.spec.md, "L5c: the continuation". Pure.

L5f's `published` (800 published rows from `second_full`, four orders, the averaged adapter) read 144.77 per 1,000 on
G's first sampling against `second_full`'s 110.25. Two things changed at once: 800 more rows of training, and the
average of four trainings. One task measures four models against that stored `published` (THE BAR), paired by problem.
NO ROW OF THIS STAGE IS ONE OF THE MODEL'S OWN PROOFS.

  the seed          the one the setting names, and the further ones once the escalation is recorded (L5s's own
                    `the_seed` and `fixed_seeds`)
  the models        `single`: L5f's stored `published_1`, only read and measured. `wide`: the mean of EIGHT adapters,
                    L5f's stored `published_1` to `published_4` and four new trainings of the same rows in the orders
                    of order seeds 5 to 8. `again`: from L5f's stored averaged `published`, the NEXT rows of the seed's
                    order, four orders, averaged. `more`: from `second_full`, L5s's `published_stock` rows of that
                    order, four orders, averaged
  the sets          `the_sets`, on the seed's order of the pretraining file (`l4_rows.rehearsal_rows`: the order L5d's
                    `published` and L5s's `published_stock` are the first rows of); HELD to L5f's stored set and to
                    L5s's rule (`hold_the_sets`); made again by the report (`the_sets_are_the_rule`)
  L5f's run         what its report says of its seed, read for a reason it cannot carry this stage (`of_l5f`); its
                    stored adapters as its own steps recorded them (`theirs_are_held`)
  the reads         WHAT MADE IT (`what_made_it`: `single` minus the bar, below zero by more than `OF_ONE`: THE AVERAGE
                    IS PART OF THE GAIN; otherwise ONE CONTINUATION GIVES IT); `wide`, `again` and `more` each minus the
                    bar (`the_read`: at or above `A_LINE` ADDS: ESCALATE; below zero by more than it COSTS; between NOT
                    SHOWN AT ONE SEED); what follows (`what_follows`)

A model's figure is its successes per attempt on all of G over the first sampling's attempts (`l5_noise.figure`). A
branch is decided on the whole numbers of successes and attempts, never on a rounded figure.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.assembly import training_order
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ
from rlvr_lean.domain.ladder_round.l4_dose import SECOND_FULL
from rlvr_lean.domain.ladder_round.l4_rows import PRETRAINING, rehearsal_rows
from rlvr_lean.domain.ladder_round.l4_rows2 import TEXT_FIELDS
from rlvr_lean.domain.ladder_round.l5 import PUBLISHED
from rlvr_lean.domain.ladder_round.l5_draws import trained_of
from rlvr_lean.domain.ladder_round.l5_noise import ORDERS, numbered, the_orders
from rlvr_lean.domain.ladder_round.l5_stock import NOT_SHOWN_AT_ONE, against_the_start, fixed_seeds, the_seed  # noqa: F401 - the seed rule and the paired read are L5s's own

SINGLE, WIDE, AGAIN, MORE = "single", "wide", "again", "more"
MODELS = (SINGLE, WIDE, AGAIN, MORE)                        # the four models, in the order they are made and measured
AVERAGED = (WIDE, AGAIN, MORE)                              # the three that are the mean of trainings; `single` is one stored training, measured alone
START = SECOND_FULL                                         # the start of `wide` and `more`, and of L5f's models
BAR = PUBLISHED                                             # L5f's stored averaged `published` of the seed's run: the bar of every read, and the start of `again`
WIDE_ORDERS = (5, 6, 7, 8)                                  # the order seeds of `wide`'s four new trainings: L5f's four are 1 to 4
ORDERS_OF = {WIDE: WIDE_ORDERS, AGAIN: ORDERS, MORE: ORDERS}
TRAINED = tuple(trained_of(model, order) for model in AVERAGED for order in ORDERS_OF[model])       # the twelve trainings, in the order they are made
OF_THEIRS = tuple(trained_of(BAR, order) for order in ORDERS)       # L5f's four stored trainings of its `published` set: the first four members of `wide`
THE_SINGLE = OF_THEIRS[0]                                   # ... and the one `single` is
OF_ONE = Fraction(13, 1000)                                 # per attempt: twice the standard deviation of a difference between one training and an average of four
A_LINE = Fraction(85, 10000)                                # per attempt: L5f's own line for two averaged models
THE_AVERAGE_IS_PART = "THE AVERAGE IS PART OF THE GAIN"
ONE_CONTINUATION = "ONE CONTINUATION GIVES IT"
MADE_IT = (THE_AVERAGE_IS_PART, ONE_CONTINUATION, INCONCLUSIVE, NOT_READ)
ADDS = "ADDS: ESCALATE"
COSTS = "COSTS"
READS = (ADDS, COSTS, NOT_SHOWN_AT_ONE, INCONCLUSIVE, NOT_READ)
ASKED = {WIDE: "A WIDER AVERAGE", AGAIN: "A SECOND STEP", MORE: "MORE ROWS"}        # what each of the three reads is called
FURTHER_SEEDS = "ESCALATE: TWO FURTHER SEEDS OF EACH MODEL THAT ADDS AND OF THE BAR"
THE_RECIPE_STANDS = "NOTHING ADDS: L5f's `published` RECIPE STANDS AS THE BEST KNOWN"
FOLLOWS = (FURTHER_SEEDS, THE_RECIPE_STANDS, NOT_READ)


# -------------------------------------------------------------------------------------------------- the sets
def the_sets(pretraining: Sequence[Mapping], seed: int, first: int, more: int) -> dict[str, list[Mapping]]:
    """The three sets, each in the arm's content-hash order of the task's seed (`assembly.training_order`), all of rows
    of the seed's order of the pretraining file (`l4_rows.rehearsal_rows`). `pretraining`: the rows the first start was
    pretrained on, in the file's order, WITHOUT their text. `first`: the rows of L5f's `published` set (800). `more`:
    the rows of L5s's `published_stock` by its rule for the seed (3,268).

      wide      the first `first` rows of the order: L5f's `published` set
      again     the NEXT `first` rows of it (rows `first` + 1 to 2 x `first`)
      more      the first `more` rows of it

    Refused (ValueError): no row asked; `more` shorter than `first`; a file that holds fewer rows than are asked."""
    if first < 1 or more < first:
        raise ValueError(f"`{WIDE}` is asked {first} rows and `{MORE}` {more}: `{WIDE}` is L5f's `{BAR}` set, at least one row, and `{MORE}` begins with it. Refused")
    drawn = rehearsal_rows(pretraining, max(2 * first, more), seed)
    return {WIDE: list(training_order(drawn[:first], seed)), AGAIN: list(training_order(drawn[first:2 * first], seed)), MORE: list(training_order(drawn[:more], seed))}


def _differs(own: Sequence[str], theirs: Sequence[str] | None) -> dict | None:
    """Where two lists of ids first differ; None when they are one list."""
    if theirs is not None and list(own) == list(theirs):
        return None
    place = next((index for index, (mine, other) in enumerate(zip(own, theirs or [])) if mine != other), min(len(own), len(theirs or [])))
    return {"rows": len(own), "rows_of_theirs": None if theirs is None else len(theirs), "place": place,
            "here": own[place] if place < len(own) else None, "there": theirs[place] if theirs is not None and place < len(theirs) else None}


def hold_the_sets(sets: Mapping[str, Sequence[Mapping]], of_l5f: Sequence[str] | None, of_l5s: Sequence[str] | None) -> dict:
    """What the three sets are HELD to, each a refusal (ValueError); returns the counts. `wide` is, id for id and place
    for place, the set L5f's run of the seed stored as `published` (`of_l5f`: its ids, in its order); `more` is L5s's
    `published_stock` by its own rule for the seed (`of_l5s`: its ids, in its order); `again` shares no row with
    `wide`; every row is a published row and none carries text."""
    ids = {model: [row["id"] for row in sets[model]] for model in AVERAGED}
    for model, theirs, whose in ((WIDE, of_l5f, f"the set L5f's run of this seed stored as `{BAR}`"), (MORE, of_l5s, "L5s's `published_stock` by its rule for this seed")):
        differs = _differs(ids[model], theirs)
        if differs:
            raise ValueError(f"`{model}` is not {whose}, id for id ({differs['rows']} rows here and {differs['rows_of_theirs']} there; first that differs at place "
                             f"{differs['place']}: {differs['here']} here, {differs['there']} there). Refused")
    shared = sorted(set(ids[WIDE]) & set(ids[AGAIN]))
    other = sorted(row["id"] for model in AVERAGED for row in sets[model] if row["origin"] != PRETRAINING or any(field in row for field in TEXT_FIELDS))
    if shared or other or len(ids[AGAIN]) != len(ids[WIDE]) or not set(ids[WIDE]) <= set(ids[MORE]):
        raise ValueError(f"`{AGAIN}` holds {len(ids[AGAIN])} rows and `{WIDE}` {len(ids[WIDE])}, {len(shared)} of them shared; {len(other)} rows are no published row or "
                         f"carry text; `{MORE}` holds every row of `{WIDE}`: {set(ids[WIDE]) <= set(ids[MORE])}. `{AGAIN}` is the next rows of the order, as many as "
                         f"`{WIDE}`'s and none of them; every row is a published row by id and place, never with text; `{MORE}` begins with `{WIDE}`'s rows. Refused")
    return {"what": f"`{WIDE}` is L5f's stored `{BAR}` set and `{MORE}` L5s's `published_stock` by its rule, id for id and place for place; `{AGAIN}` is the next rows of "
                    "the seed's order, none of them a row of the first; every row is a published row, by id and place and without text",
            "rows": {model: len(ids[model]) for model in AVERAGED}, "passes": True}


def _members(wide: Sequence[Mapping], stored: Mapping[str, Sequence[str] | None], recorded: Mapping[str, Sequence[str] | None]) -> tuple[list[str], list[str]]:
    """(L5f's stored sets of its four trainings that are NOT `wide`'s rows in that training's order seed's order; its
    trainings whose loop did NOT record those rows, each once)."""
    ids = [row["id"] for row in wide]
    not_orders = [name for name in OF_THEIRS if stored.get(name) != [row["id"] for row in training_order(wide, int(name.rpartition("_")[2]))]]
    not_trained = [name for name in OF_THEIRS if recorded.get(name) is None or sorted(recorded[name]) != sorted(ids)]
    return not_orders, not_trained


def hold_the_members(wide: Sequence[Mapping], stored: Mapping[str, Sequence[str] | None], recorded: Mapping[str, Sequence[str] | None]) -> dict:
    """L5f's FOUR STORED TRAININGS ARE FOUR ORDERS OF `wide`'s ROWS, so that `wide` is the mean of eight trainings of
    one set: the set L5f's run stored for each (`stored`: by name, its ids in its order) is `wide`'s rows in the order
    of that training's order seed, and its training loop recorded those rows, each once (`recorded`). Refused
    (ValueError) otherwise. Returns what was counted."""
    not_orders, not_trained = _members(wide, stored, recorded)
    if not_orders or not_trained:
        raise ValueError(f"of L5f's four trainings of `{BAR}`, the stored sets of {not_orders} are not `{WIDE}`'s {len(wide)} rows in their order seed's order, and the "
                         f"trainings {not_trained} did not record those rows, each once: `{WIDE}` is the mean of eight trainings of ONE set. Refused")
    return {"what": f"L5f's four trainings of `{BAR}` are four orders of `{WIDE}`'s rows, as stored and as their training loops recorded", "trainings": list(OF_THEIRS),
            "rows": len(wide), "passes": True}


def in_their_orders(sets: Mapping[str, Sequence[Mapping]], smoke: bool = False) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """Each set in the content-hash order of each of ITS order seeds (`l5_noise.the_orders`), numbered: by trained
    model `<model>_<order seed>`. `wide`'s four new orders are held BESIDE L5f's four of the same rows (order seeds 1
    to 4): two of the eight that are one order are refused (ValueError), as two of a model's four are, since their
    adapters would be one adapter. `smoke` (a SMOKE run alone, whose handful of rows has few orders): trained all the
    same. An order seed that is the task's seed gives the seed's own order: trained all the same, and named. Returns
    (the ordered sets; by model, what coincides)."""
    in_orders, coincide = {}, {}
    for model in AVERAGED:
        every = (*ORDERS, *WIDE_ORDERS) if model == WIDE else ORDERS_OF[model]
        by_order, found = the_orders(sets[model], every, loosely=True)
        if found["alike"] and not smoke:
            raise ValueError(f"of the order seeds {list(every)} of `{model}`, the pairs {found['alike']} give one order of these {len(sets[model])} rows: an averaged adapter "
                             "is the mean of trainings that differ by their order. Refused")
        in_orders.update({trained_of(model, order): numbered(by_order[order]) for order in ORDERS_OF[model]})
        coincide[model] = {"the_order_of_the_tasks_seed": found["the_order_trained_before"], "alike": found["alike"]}
    return in_orders, coincide


def the_sets_are_the_rule(sets: Mapping[str, Sequence[Mapping]], pretraining: Sequence[Mapping], seed: int, in_orders: Mapping[str, Sequence[Mapping]],
                          stored: Mapping[str, Sequence[str] | None], recorded: Mapping[str, Sequence[str] | None], more_by_the_rule: int | None) -> dict:
    """The first check: each set is its rule MADE AGAIN (`the_sets`, on the rows the first start was pretrained on),
    id for id and place for place; `wide` is the set L5f's run stores as `published` NOW, and that run's four trainings
    are four orders of it (`stored`, `recorded`: L5f's sets as stored and as trained, None where that run holds none);
    `more` holds the rows L5s's rule gives (`more_by_the_rule`: twice the own solves in stock, counted again by the
    report); no row carries text or is no published row; and each of a model's four sets (`in_orders`: by trained model
    `<model>_<order seed>`) is its rows in that order seed's order."""
    of_l5f = stored.get(BAR)
    placed = lambda own: [(row["id"], row["problem_id"], row.get("row_of_the_file")) for row in own]      # noqa: E731
    try:
        again, why = the_sets(pretraining, seed, len(sets[WIDE]), len(sets[MORE])), None
        not_the_rule = [model for model in AVERAGED if placed(sets[model]) != placed(again[model])]
    except (ValueError, KeyError) as error:
        why, not_the_rule = str(error), list(AVERAGED)
    with_text = sorted(row["id"] for own in sets.values() for row in own if row["origin"] != PRETRAINING or any(field in row for field in TEXT_FIELDS))
    out_of_order = []
    for name, own in in_orders.items():
        model, _, order = name.rpartition("_")
        if model not in sets or not order.isdigit() or [row["id"] for row in own] != [row["id"] for row in training_order(sets[model], int(order))]:
            out_of_order.append(name)
    not_theirs = _differs([row["id"] for row in sets[WIDE]], of_l5f)
    not_orders, not_trained = _members(sets[WIDE], stored, recorded)
    more_is = len(sets[MORE]) == more_by_the_rule
    return {"what": f"each set, as stored, is its rule made again: `{WIDE}` the first rows of the seed's order of the pretraining file, and the set L5f's run stores as "
                    f"`{BAR}`; `{AGAIN}` the next as many; `{MORE}` the first rows L5s's rule gives (twice the own solves in stock, counted again); no row carries text; and "
                    "each of a model's four sets is its rows in its order seed's order",
            "rows": {model: len(sets[model]) for model in AVERAGED}, "refused": why, "sets_that_are_not_the_rule": not_the_rule,
            "rows_with_text_or_not_published": len(with_text), "sets_not_in_their_order_seeds_order": out_of_order, "wide_is_l5fs_stored_set": not_theirs is None,
            "first_that_differs_from_l5fs": not_theirs, "l5fs_trainings_that_are_not_orders_of_wide": not_orders, "l5fs_trainings_not_of_these_rows": not_trained,
            "rows_of_more_by_l5ss_rule": more_by_the_rule, "more_is_l5ss_rule": more_is,
            "passes": why is None and not not_the_rule and not with_text and not out_of_order and not_theirs is None and not not_orders and not not_trained and more_is}


# ------------------------------------------------------------------------------------------------ L5f's run
def of_l5f(entry: Mapping, seed: int, recipe: str, measurement: Mapping) -> str | None:
    """Why L5f's run of this seed cannot carry this stage, or None. `entry`: what its report says of its own seed
    (`of_this_seed`), or `{"there": False, "why": ...}`. A run that is not there, of another seed, start or recipe,
    INCONCLUSIVE, not to be read, or measured otherwise than this run (`measurement`: the sampling, its attempts, its
    sampling seed, the problems)."""
    if not entry.get("there", True):
        return entry.get("why")
    return (f"it is a report of seed {entry.get('seed')}" if entry.get("seed") != seed else
            f"it is a run from `{entry.get('start')}` and this stage's bar is from `{START}`" if entry.get("start") != START else
            f"its recipe is `{entry.get('recipe')}` and this run's `{recipe}`" if entry.get("recipe") != recipe else
            "its report is not to be read" if not entry.get("ok", True) else
            f"it reads {INCONCLUSIVE} ({'; '.join(entry.get('failed_checks') or [])})" if entry.get("inconclusive") or not entry.get("figures") else
            f"it was measured otherwise ({entry.get('the_measurement')}) than this run ({dict(measurement)})" if entry.get("the_measurement") != dict(measurement) else None)


def theirs_are_held(trainings: Mapping[str, Mapping], averaging: Mapping, rank: int | None, alpha: int | None, now: Mapping | None = None) -> dict:
    """L5f's five stored adapters of its `published` kind, as ITS OWN STEPS RECORDED THEM: each of the four trainings
    (`trainings`: by name, its `adapter`, what was read back of the one it saved, `adapter_saved`, and its bit-for-bit
    comparison with the start adapter's as `l4.the_adapter_differs` reads it, `differs`) saved an adapter read back at
    the rank and alpha, with numbers that are not the start adapter's; the averaged one (`averaging`: its step's
    summary) was read back as the mean of the four, none of them, at the rank. `now` (a GPU box): what the files
    themselves read as, by adapter, each with `passes`. Returns the check; nothing is refused here."""
    at_rank = lambda saved: bool(saved) and (saved.get("rank"), saved.get("alpha")) == (rank, alpha)      # noqa: E731
    of_each = {name: {"adapter": own.get("adapter"), "read_back": own.get("adapter_saved"), "at_its_rank": at_rank(own.get("adapter_saved")),
                      "numbers_not_the_start_adapters": (own.get("differs") or {}).get("elements_changed") or 0} for name, own in trainings.items()}
    held = averaging.get("read_back") or {}
    of_the_mean = {"adapter": averaging.get("adapter"), "of": averaging.get("of"), "read_back": averaging.get("adapter_saved"), "at_its_rank": at_rank(averaging.get("adapter_saved")),
                   "is_the_mean": bool(held.get("is_the_mean")), "is_none_of_them": bool(held.get("is_none_of_them")),
                   "one_of_its_models_in_a_smoke_run": bool(averaging.get("one_of_its_models_in_a_smoke_run")), "averaged": bool(averaging.get("averaged"))}
    recorded = (all(own["at_its_rank"] and own["numbers_not_the_start_adapters"] > 0 for own in of_each.values()) and len(of_each) == len(OF_THEIRS)
                and of_the_mean["averaged"] and of_the_mean["at_its_rank"] and of_the_mean["is_the_mean"]
                and (of_the_mean["is_none_of_them"] or of_the_mean["one_of_its_models_in_a_smoke_run"]) and list(of_the_mean["of"] or []) == list(OF_THEIRS))
    return {"what": f"L5f's four stored trainings of its `{BAR}` set and its averaged `{BAR}`, as its own steps recorded them: each training read back at the rank and alpha, "
                    "not the start adapter's; the averaged one the mean of the four, none of them, at the rank. On a GPU box the files themselves are read too",
            "rank": rank, "alpha": alpha, "trainings": of_each, "the_averaged": of_the_mean, "recorded": recorded, "read_now": None if now is None else dict(now),
            "passes": recorded and (now is None or all(own.get("passes") for own in now.values()))}


# ------------------------------------------------------------------------------------------------- the reads
def _exact(own: Mapping) -> Fraction | None:
    return Fraction(own["successes"], own["attempts"]) if own["attempts"] else None


def _whole(own: Mapping) -> dict:
    return {"successes": own["successes"], "attempts": own["attempts"]}


def what_made_it(failed_checks: Sequence[str], single: Mapping | None, bar: Mapping | None) -> dict:
    """WHAT MADE IT: `single` (one of the four trainings the bar is the mean of, measured alone) minus the bar, per
    attempt on all of G. Below zero by MORE than `OF_ONE`: THE AVERAGE IS PART OF THE GAIN. Otherwise (a difference of
    exactly minus `OF_ONE` is otherwise): ONE CONTINUATION GIVES IT. `single`, `bar`: their figures on all of G (whole
    numbers of successes and attempts). INCONCLUSIVE when a check of this run failed; NOT READ without attempts."""
    if failed_checks:
        return {"name": INCONCLUSIVE, "reason": f"a check of this run failed ({'; '.join(failed_checks)}): what made it is not read"}
    if not single or not bar or not single["attempts"] or not bar["attempts"]:
        return {"name": NOT_READ, "reason": f"`{SINGLE}` or the bar has no attempt: what made it is NOT READ"}
    difference = _exact(single) - _exact(bar)
    name = THE_AVERAGE_IS_PART if difference < -OF_ONE else ONE_CONTINUATION
    said = {THE_AVERAGE_IS_PART: f"it is below zero by more than {float(OF_ONE) * 1000:g} per 1,000: one of the four trainings alone does not read what their mean reads",
            ONE_CONTINUATION: f"it is not below zero by more than {float(OF_ONE) * 1000:g} per 1,000: one training of the same rows reads what the mean of four reads"}[name]
    return {"name": name, "difference": round(float(difference), 6), "line": float(OF_ONE), SINGLE: _whole(single), BAR: _whole(bar),
            "reason": f"`{SINGLE}` minus L5f's stored `{BAR}`, per 1,000 attempts on all of G: {float(difference) * 1000:+.2f} ({single['successes']} of {single['attempts']} "
                      f"against {bar['successes']} of {bar['attempts']}); {said}"}


def the_read(failed_checks: Sequence[str], model: str, own: Mapping | None, bar: Mapping | None) -> dict:
    """ONE OF THE THREE READS (`wide`, `again`, `more`): the model minus the bar, per attempt on all of G. At or above
    plus `A_LINE`: it ADDS: ESCALATE. Below zero by MORE than `A_LINE`: it COSTS. Between (a difference of exactly minus
    `A_LINE` is between): NOT SHOWN AT ONE SEED. INCONCLUSIVE when a check of this run failed; NOT READ without attempts."""
    if failed_checks:
        return {"name": INCONCLUSIVE, "reason": f"a check of this run failed ({'; '.join(failed_checks)}): `{model}` is not read"}
    if not own or not bar or not own["attempts"] or not bar["attempts"]:
        return {"name": NOT_READ, "reason": f"`{model}` or the bar has no attempt: it is NOT READ"}
    difference = _exact(own) - _exact(bar)
    name = ADDS if difference >= A_LINE else COSTS if difference < -A_LINE else NOT_SHOWN_AT_ONE
    return {"name": name, "difference": round(float(difference), 6), "line": float(A_LINE), model: _whole(own), BAR: _whole(bar),
            "reason": f"`{model}` minus L5f's stored `{BAR}`, per 1,000 attempts on all of G: {float(difference) * 1000:+.2f} ({own['successes']} of {own['attempts']} against "
                      f"{bar['successes']} of {bar['attempts']}); the lines are +{float(A_LINE) * 1000:g} and -{float(A_LINE) * 1000:g}"}


def what_follows(made: Mapping, reads: Mapping[str, Mapping], figures: Mapping[str, Mapping], bar: Mapping | None, more: Sequence[int]) -> dict:
    """What follows, fixed before the run. THE CANDIDATE for a measurement in full (not built) is the highest of this
    stage's four models and the bar, on whole numbers; the bar stands when none is above it. A model that ADDS gets its
    two further seeds, and the bar's, before anything is said of it. If nothing adds, L5f's `published` recipe stands as
    the best known. `figures`: by model, its figure on all of G. Nothing is named while the reads are not made."""
    names = [made["name"], *(reads[model]["name"] for model in AVERAGED)]
    if any(name in (INCONCLUSIVE, NOT_READ) for name in names) or not bar or not bar["attempts"]:
        return {"name": NOT_READ, "reason": "a read of this run is not made: nothing is named for what follows"}
    adds = [model for model in AVERAGED if reads[model]["name"] == ADDS]
    highest = max(MODELS, key=lambda model: _exact(figures[model]))        # the first of them, in the order they are made, when two read alike
    above = _exact(figures[highest]) > _exact(bar)
    candidate = {"model": highest if above else BAR, "of": "this stage" if above else "L5f's stored run of the seed",
                 "per_attempt": round(float(_exact(figures[highest]) if above else _exact(bar)), 6)}
    said = (f"the candidate for a measurement in full (both samplings of G, the held-aside set, the rungs; not built) is `{candidate['model']}` of {candidate['of']}, "
            f"{candidate['per_attempt'] * 1000:.2f} per 1,000 on G's first sampling")
    if adds:
        return {"name": FURTHER_SEEDS, "adds": adds, "seeds": list(more), "candidate": candidate,
                "reason": f"{', '.join(f'`{model}`' for model in adds)} read{'s' if len(adds) == 1 else ''} {ADDS}: seeds {list(more)} of each and of the bar (L5f's "
                          f"`{BAR}` at those seeds, which no run holds yet) before anything is said of it; {said}"}
    return {"name": THE_RECIPE_STANDS, "adds": [], "candidate": candidate,
            "reason": f"none of {', '.join(f'`{model}`' for model in AVERAGED)} reads {ADDS}: L5f's `{BAR}` recipe (its published rows from `{START}`, four orders, the "
                      f"averaged adapter) stands as the best known, and what made it is answered by `{SINGLE}` alone ({made['name']}); {said}"}
