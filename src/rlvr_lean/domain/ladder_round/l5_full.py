"""L5f: the three kinds again, from the twice-pretrained model. Do the model's own proofs still add after a whole
second pass over published ones? Spec: docs/spec/ladder-loop.spec.md, "L5f: the three kinds again, from the
twice-pretrained model". Pure.

L5d's stage with ONE change, the start adapter: `second_full`, L4p's whole second pass over the published rows. The
three sets are L5d's for the seed, HELD to be the ones L5d's own run of that seed stored and trained.

  the seed          the one the setting names, and the further ones once the escalation is recorded (`the_seed`,
                    `fixed_seeds`: L5s's own two, a scout's rule)
  the check         the sets are, id for id and order for order, the sets L5d's run of the seed stored, and what that
                    run's training loops recorded (`the_sets_are_theirs`; `hold_to_theirs` refuses)
  the reads         THE STACK, the scout's read: the mean of `shallow` and `deep`, minus `published`, all three from
                    `second_full` (`the_stack`: at or above zero ESCALATE; below zero by more than `A_COST` THEY COST
                    AFTER A SECOND PASS AT THIS SEED; between, NOT SHOWN AT ONE SEED); THE LEVEL, naming nothing
                    (`the_level`); each model against a start model's stored first sampling, paired by problem
                    (`against_the_start`, L5s's own); what follows (`what_follows`)

A model's figure is its successes per attempt on all of G over the first sampling's attempts (`l5_noise.figure`). A
branch is decided on the whole numbers of successes and attempts, never on a rounded figure.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ
from rlvr_lean.domain.ladder_round.l4_dose import SECOND_FULL
from rlvr_lean.domain.ladder_round.l5 import DEEP, MODELS, PUBLISHED, SHALLOW
from rlvr_lean.domain.ladder_round.l5_draws import trained_on_its_set
from rlvr_lean.domain.ladder_round.l5_stock import NOT_SHOWN_AT_ONE, against_the_start, fixed_seeds, the_seed  # noqa: F401 - the scout's seed rule and the read against a start model are L5s's own

START = SECOND_FULL                                         # the one start this stage is specified from: L4p's whole second pass
ESCALATE = "THE MODEL'S OWN PROOFS DO NOT FALL BEHIND PUBLISHED ONES AFTER A SECOND PASS: ESCALATE"
COSTS = "THEY COST AFTER A SECOND PASS AT THIS SEED"
STACK = (ESCALATE, COSTS, NOT_SHOWN_AT_ONE, INCONCLUSIVE, NOT_READ)
FURTHER_SEEDS = "ESCALATE: THE FURTHER SEEDS OF THIS STAGE RUN, EACH AFTER L5d's OWN RUN OF THAT SEED"
NO_FURTHER_SEED = "NO FURTHER SEED: IT NEEDS A DECISION"
FOLLOWS = (FURTHER_SEEDS, NO_FURTHER_SEED, NOT_READ)
A_COST = Fraction(85, 10000)                                # per attempt: "below zero by more than 0.0085 (twice the standard deviation above)"
OF_A_FIGURE = 0.0034                                        # per attempt: the standard deviation of an averaged model's figure (the section's)
OF_THE_PRIMARY = 0.0042                                     # per attempt: ... and of the primary's one-seed difference


# ------------------------------------------------------------------------------------------------ the check
def the_sets_are_theirs(sets: Mapping[str, Sequence[Mapping]], stored: Mapping[str, Sequence[str] | None], recorded: Mapping[str, Sequence[str] | None]) -> dict:
    """THE SETS ARE THE ONES L5d's RUN OF THE SEED TRAINED. `sets`: by model and by trained model (`<kind>_<order
    seed>`), this run's set. `stored`: by the same names, the ids of the set L5d's run STORED, in its order (None: that
    run holds no such set). `recorded`: by trained model, the ids L5d's TRAINING LOOP recorded (None: no record). Each
    set here is the stored one id for id and order for order, and each training of L5d's was of those rows, each once
    (`l5_draws.trained_on_its_set`, L5d's own check of a training)."""
    ids = {name: [row["id"] for row in own] for name, own in sets.items()}
    not_stored = [name for name in ids if stored.get(name) is None or ids[name] != list(stored[name])]
    not_trained = [name for name in recorded if name not in sets or recorded[name] is None or not trained_on_its_set(list(recorded[name]), sets[name])["passes"]]
    first = None
    for name in not_stored:
        theirs = list(stored.get(name) or [])
        place = next((index for index, (mine, other) in enumerate(zip(ids[name], theirs)) if mine != other), min(len(ids[name]), len(theirs)))
        first = {"set": name, "place": place, "here": ids[name][place] if place < len(ids[name]) else None, "theirs": theirs[place] if place < len(theirs) else None,
                 "rows_here": len(ids[name]), "rows_of_theirs": len(theirs)}
        break
    return {"what": "each set of this run (the three, and each in its four orders) is, id for id and order for order, the set L5d's run of this seed stored; and each of that "
                    "run's trainings was of those rows, each once, on its training loop's own record",
            "sets": len(ids), "trainings_of_theirs": len(recorded), "rows": {name: len(own) for name, own in ids.items()}, "sets_that_are_not_the_stored_ones": not_stored,
            "first_that_differs": first, "sets_that_are_not_what_was_trained": not_trained, "passes": bool(ids) and bool(recorded) and not not_stored and not not_trained}


def hold_to_theirs(sets: Mapping[str, Sequence[Mapping]], stored: Mapping[str, Sequence[str] | None], recorded: Mapping[str, Sequence[str] | None]) -> dict:
    """`the_sets_are_theirs`, REFUSED (ValueError) when it does not pass: the first step builds no run on other rows or
    another order than L5d's own run of the seed trained."""
    held = the_sets_are_theirs(sets, stored, recorded)
    if not held["passes"]:
        first = held["first_that_differs"]
        raise ValueError(f"the sets {held['sets_that_are_not_the_stored_ones']} are not the ones L5d's run of this seed stored"
                         + (f" (`{first['set']}`: {first['rows_here']} rows here and {first['rows_of_theirs']} there; first that differs at place {first['place']}: "
                            f"{first['here']} here, {first['theirs']} there)" if first else "")
                         + f", and L5d's trainings of {held['sets_that_are_not_what_was_trained']} were not of these rows, each once: this stage is L5d's with ONE change, the "
                           "start adapter, and its sets are held to be that run's, id for id and order for order. Refused")
    return held


# ------------------------------------------------------------------------------------------------- the reads
def _exact(own: Mapping) -> Fraction | None:
    return Fraction(own["successes"], own["attempts"]) if own["attempts"] else None


def the_stack(failed_checks: Sequence[str], figures: Mapping[str, Mapping]) -> dict:
    """THE STACK, the scout's read: L5d's primary from `second_full`, the mean of `shallow` and `deep`, minus
    `published`, per attempt on all of G. At or above zero: THE MODEL'S OWN PROOFS DO NOT FALL BEHIND PUBLISHED ONES
    AFTER A SECOND PASS: ESCALATE. Below zero by MORE than `A_COST`: THEY COST AFTER A SECOND PASS AT THIS SEED. Between
    (a difference of exactly minus `A_COST` is between): NOT SHOWN AT ONE SEED. `figures`: by model, its figure on all
    of G (whole numbers of successes and attempts). INCONCLUSIVE when a check of this run failed."""
    if failed_checks:
        return {"name": INCONCLUSIVE, "reason": f"a check of this run failed ({'; '.join(failed_checks)}): the stack is not read"}
    rate = {model: _exact(figures[model]) for model in MODELS}
    if None in rate.values():
        return {"name": NOT_READ, "reason": "a model has no attempt on the goal set: the stack is NOT READ"}
    difference = (rate[SHALLOW] + rate[DEEP]) / 2 - rate[PUBLISHED]
    name = ESCALATE if difference >= 0 else COSTS if difference < -A_COST else NOT_SHOWN_AT_ONE
    said = {ESCALATE: "at or above zero: the further seeds of this stage, each after L5d's own run of that seed (three seeds conclude, with L5d's primary rule and names)",
            COSTS: f"below zero by more than {1000 * float(A_COST):g} per 1,000 (twice the standard deviation of the primary's one-seed difference, "
                   f"{1000 * OF_THE_PRIMARY:g}): no further seed",
            NOT_SHOWN_AT_ONE: f"below zero by at most {1000 * float(A_COST):g} per 1,000: no further seed is queued by this read"}[name]
    return {"name": name, "difference": round(float(difference), 6), **{model: {key: figures[model][key] for key in ("successes", "attempts")} for model in MODELS},
            "a_cost": float(A_COST), "standard_deviation_of_the_one_seed_difference": OF_THE_PRIMARY,
            "reason": f"the mean of `{SHALLOW}` and `{DEEP}`, minus `{PUBLISHED}`, all three from `{START}`, per 1,000 attempts on all of G: {1000 * float(difference):+.2f} "
                      f"(`{SHALLOW}` {1000 * float(rate[SHALLOW]):.2f}, `{DEEP}` {1000 * float(rate[DEEP]):.2f}, `{PUBLISHED}` {1000 * float(rate[PUBLISHED]):.2f}): {said}"}


def the_level(figures: Mapping[str, Mapping], of_the_start: Mapping, of_pre: Mapping, of_theirs: Mapping) -> dict:
    """THE LEVEL, naming nothing at one seed, per attempt on all of G: each of the three minus `second_full`'s stored
    first sampling (the same problems, attempts and sampling seed: paired by problem); each minus THE SAME KIND of L5d's
    run of the seed (from `pre_r64`), NOT READ when that run is not; and the highest figure of the three beside the two
    start models' and L5d's three. `figures`: by model, on all of G. `of_the_start`, `of_pre`: `second_full`'s and
    `pre_r64`'s stored first samplings' figures on all of G. `of_theirs`: `l5_stock.of_l5d`'s."""
    own, start, read = {model: _exact(figures[model]) for model in MODELS}, _exact(of_the_start), bool(of_theirs.get("read"))
    theirs = {model: _exact(of_theirs["figures"][model]) for model in MODELS} if read else {}
    shown = lambda value: None if value is None else round(float(value), 6)      # noqa: E731
    highest = max(MODELS, key=lambda model: (own[model] if own[model] is not None else Fraction(-1), -MODELS.index(model)))
    return {"what": f"each model minus `{START}`'s stored first sampling, and minus the same kind of L5d's run of this seed, per attempt on all of G. One seed names nothing",
            "minus_the_start": {model: shown(None if None in (own[model], start) else own[model] - start) for model in MODELS},
            "minus_the_same_kind_of_l5d": {model: shown(own[model] - theirs[model]) if read and None not in (own[model], theirs.get(model)) else None for model in MODELS},
            "l5d_read": read, "why": None if read else of_theirs.get("why"),
            "highest": {"model": highest, "per_attempt": shown(own[highest])},
            "beside": {"pre_r64": shown(_exact(of_pre)), START: shown(start), "l5d": {model: shown(theirs[model]) for model in MODELS} if read else None}}


def what_follows(stack: Mapping, more: Sequence[int]) -> dict:
    """What follows, fixed before the run. ESCALATE: the further seeds of this stage, each after L5d's own run of that
    seed; three seeds conclude with L5d's primary rule and names; the highest averaged adapter of this stage and of L5s
    is then the one measured in full, a step that is not built here. COSTS, or NOT SHOWN: no further seed; it needs a
    decision. Nothing is named while the stack is not read."""
    name = stack["name"]
    if name in (INCONCLUSIVE, NOT_READ):
        return {"name": NOT_READ, "reason": f"the stack reads {name}: nothing is named for what follows"}
    if name == ESCALATE:
        return {"name": FURTHER_SEEDS, "seeds": list(more),
                "reason": f"the stack reads {name}: seeds {list(more)} of this stage, each after L5d's own run of that seed; three seeds conclude, with L5d's primary rule and "
                          "names. The averaged adapter that reads highest on G's first sampling, of this stage and of L5s, is then the one measured in full (the rungs, the "
                          "second sampling, the held-aside set), a step that is not built here"}
    return {"name": NO_FURTHER_SEED, "reason": f"the stack reads {name}: no further seed; the outcome needs a decision, with its figures"}
