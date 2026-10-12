"""L5m: the best model, measured in full. Does the gain hold on the held-aside set and on fresh attempts? Spec:
docs/spec/ladder-loop.spec.md, "L5m: the best model, measured in full". Pure.

L5f's `published` and L5c's four models are read on the goal set's FIRST sampling alone, the one set of episodes more
than a dozen models have been compared on: the highest of them carries the selection. Two measurements no stage has
selected on say whether the gain is the model's: G's SECOND sampling and the HELD-ASIDE set. NO TRAINING: one stored
averaged adapter (THE CANDIDATE, named in the settings) is measured in full, against `second_full` (THE BAR), the
model it was trained from.

  the candidate     `the_candidate`: the stage that made it (`l5_full` or `l5_cont`), its seed, its name there. An
                    averaged adapter of that stage and nothing else (`MODELS_OF`); the task's seed is its seed
  its run           what the report of the stage that made it says of its seed, read for a reason it cannot carry this
                    stage (`of_its_run`); the record of the mean (`is_the_one_measured`: the adapter that stage
                    averaged is the one it measured, and the one this run is handed)
  the checks        `sets_are_the_settings` (every set: its problems, its attempts, its sampling seed);
                    `adapters_are_held` (the two adapters are there, at the rank, and not each other's)
  the reads         `by_the_interval`: the candidate minus the bar, per attempt, paired by problem, with its 95%
                    bootstrap interval over problems. The interval above zero: the gain HOLDS; spanning zero (one that
                    ends at zero spans it): NOT SHOWN; below zero: it DOES NOT HOLD. On the held-aside set (the
                    primary) and on G's second sampling (fresh attempts). `what_follows`
  beside            `reliably` (the problems solved in half their attempts or more), `never_solved` (the ones a model
                    solved in none of its attempts)

These are reads about two stored models on problems, not about a recipe over trainings: the interval is over problems.
"""

from __future__ import annotations

from typing import Collection, Mapping, Sequence

from rlvr_lean.domain.ladder_round.ceiling import interval_text
from rlvr_lean.domain.ladder_round.l3d2 import NOT_SHOWN
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ
from rlvr_lean.domain.ladder_round.l4_dose import SECOND_FULL
from rlvr_lean.domain.ladder_round.l5 import DEEP, PUBLISHED, SHALLOW
from rlvr_lean.domain.ladder_round.l5_cont import AVERAGED as OF_L5C

L5F, L5C = "l5_full", "l5_cont"                             # the two stages a candidate may be of, as the settings name them
MODELS_OF = {L5F: (PUBLISHED, SHALLOW, DEEP), L5C: tuple(OF_L5C)}       # ... and the averaged adapters each keeps (`single` of L5c is one training, not an averaged adapter)
SAID = {L5F: "L5f", L5C: "L5c"}                             # who made it, in a printed line
BAR = SECOND_FULL                                           # the model the candidate was trained from: L4p's whole second pass
CANDIDATE = "candidate"                                     # the candidate in the names of THIS run's steps and sets (a step's name is registered without a config)
HOLDS_ON_THE_SET = "THE GAIN HOLDS ON THE HELD-ASIDE SET"
NOT_SHOWN_ON_THE_SET = "NOT SHOWN ON THE HELD-ASIDE SET"
DOES_NOT_HOLD_THERE = "IT DOES NOT HOLD THERE"
ON_THE_SET = (HOLDS_ON_THE_SET, NOT_SHOWN_ON_THE_SET, DOES_NOT_HOLD_THERE)      # the primary's three names: the interval above zero, spanning it, below it
HOLDS_ON_FRESH = "THE GAIN HOLDS ON FRESH ATTEMPTS"
DOES_NOT_HOLD = "IT DOES NOT HOLD"
ON_FRESH = (HOLDS_ON_FRESH, NOT_SHOWN, DOES_NOT_HOLD)       # ... and the second read's
THE_BEST_MEASURED = "THE CANDIDATE IS THE BEST MODEL MEASURED IN FULL: THE NEXT CYCLE IS PROPOSED FROM IT"
SELECTION_ON_G = "THE NIGHT'S FIRST-SAMPLING COMPARISONS ARE SELECTION ON G UNTIL A FURTHER SEED SHOWS OTHERWISE"
NOT_NAMED = "NOT NAMED BY THE SECTION: IT HOLDS ON THE HELD-ASIDE SET AND NOT ON FRESH ATTEMPTS"      # the one outcome "What follows" does not name
FOLLOWS = (THE_BEST_MEASURED, SELECTION_ON_G, NOT_NAMED, NOT_READ)
HALF = 2                                                    # "solved reliably (half the attempts or more)": `HALF` x its successes is at least its attempts


# ---------------------------------------------------------------------------------------------- the candidate
def the_candidate(of: str, seed: object, model: str, seeds_of: Mapping[str, Collection[int]]) -> None:
    """The candidate the settings name. Refused (ValueError): a stage that is neither `l5_full` nor `l5_cont`; a model
    that is not an averaged adapter of that stage; a seed that is not one of that stage's own (`seeds_of`: by stage,
    the seeds its settings list)."""
    if of not in MODELS_OF:
        raise ValueError(f"the candidate is named of the stage {of!r}: it is a stored averaged adapter of {' or '.join(f'`{name}`' for name in MODELS_OF)}, and of nothing else")
    if model not in MODELS_OF[of]:
        raise ValueError(f"the candidate is named {model!r} of `{of}`: that stage keeps the averaged adapters {', '.join(f'`{name}`' for name in MODELS_OF[of])}, and the "
                         "candidate is one of them")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in seeds_of[of]:
        raise ValueError(f"the candidate's seed is {seed!r}: `{of}` lists the seeds {sorted(seeds_of[of])}, and the candidate is the model of a run of one of them")


def the_seed(seed: int, of_the_candidate: int) -> None:
    """A task's seed is the candidate's (its run directory is named by it). Refused (ValueError) otherwise."""
    if seed != of_the_candidate:
        raise ValueError(f"this task's seed is {seed} and the candidate is of seed {of_the_candidate}: the task is queued with the candidate's seed, and there is one "
                         "candidate")


def of_its_run(entry: Mapping, seed: int, start: str, recipe: str, measurement: Mapping) -> str | None:
    """Why the run that made the candidate cannot carry this stage, or None. `entry`: what its report says of its own
    seed (`of_this_seed`), or `{"there": False, "why": ...}`. A run that is not there, of another seed, start or recipe,
    not to be read, INCONCLUSIVE, or whose models were measured otherwise than this run measures G's first sampling
    (`measurement`: the sampling, its attempts, its sampling seed, the problems)."""
    if not entry.get("there", True):
        return entry.get("why")
    return (f"it is a report of seed {entry.get('seed')}" if entry.get("seed") != seed else
            f"it is a run from `{entry.get('start')}` and this stage's bar is `{start}`, the model the candidate was trained from" if entry.get("start") != start else
            f"its recipe is `{entry.get('recipe')}` and the candidate is an averaged adapter (`{recipe}`)" if entry.get("recipe") != recipe else
            "its report is not to be read" if not entry.get("ok", True) else
            f"it reads {INCONCLUSIVE} ({'; '.join(entry.get('failed_checks') or [])})" if entry.get("inconclusive") or not entry.get("figures") else
            f"it was measured otherwise ({entry.get('the_measurement')}) than this run measures the first sampling ({dict(measurement)})"
            if entry.get("the_measurement") != dict(measurement) else None)


def is_the_one_measured(averaging: Mapping, measuring: Mapping, adapter: str, rank: int | None, alpha: int | None, served: Sequence[str | None] = (),
                        smoke: bool = False) -> dict:
    """THE CANDIDATE IS THE ADAPTER THE STAGE THAT MADE IT MEASURED, by that stage's own stored record of the mean.
    `averaging`: its averaging step's summary (the adapter it saved, read back as the mean of its members, none of
    them, at the rank). `measuring`: its measuring step's (the adapter its set was sampled from). `adapter`: where this
    run finds the candidate. `served`: the adapters this run's own measuring steps recorded (none before they ran).
    `smoke`: a SMOKE run of this stage, which reads a smoke run's candidate: an averaged adapter that is one of its
    members (that stage recorded it so) passes all the same there, and nowhere else."""
    held, saved = averaging.get("read_back") or {}, averaging.get("adapter_saved") or {}
    one_of_them = smoke and bool(averaging.get("one_of_its_models_in_a_smoke_run"))
    is_the_mean = bool(averaging.get("averaged")) and bool(held.get("is_the_mean")) and (bool(held.get("is_none_of_them")) or one_of_them)
    at_rank = (saved.get("rank"), saved.get("alpha")) == (rank, alpha) and rank is not None
    one_adapter = bool(adapter) and averaging.get("adapter") == adapter == measuring.get("adapter")
    here = all(own == adapter for own in served)
    return {"what": "the candidate is the adapter the stage that made it averaged, read back as the mean of its members and at the rank, the one that stage's own "
                    "measurement was sampled from, and the one every measurement of this run was sampled from",
            "adapter": adapter, "averaged_as": averaging.get("adapter"), "measured_there_from": measuring.get("adapter"), "served_here_from": list(served),
            "of": list(averaging.get("of") or []), "elements": held.get("elements"), "elements_not_the_mean": held.get("elements_not_the_mean"),
            "is_the_mean": is_the_mean, "one_of_its_models_in_a_smoke_run": one_of_them, "read_back": dict(saved) or None, "at_its_rank": at_rank,
            "one_adapter": one_adapter, "served_here": here, "passes": is_the_mean and at_rank and one_adapter and here}


def adapters_are_held(of_the_candidate: Mapping, of_the_bar: Mapping, rank: int | None, alpha: int | None) -> dict:
    """The two adapters are there, at the rank, and not each other's, as the first step held them before anything was
    written. `of_the_candidate`: what it held of the candidate (`recorded`: the read back its stage recorded, and the
    numbers of it that are not its start adapter's; `read_now`: on a GPU box, the files themselves, with `passes`).
    `of_the_bar`: what L5f's own reading held of `second_full` (`passes`)."""
    recorded, now = of_the_candidate.get("recorded") or {}, of_the_candidate.get("read_now")
    saved = recorded.get("adapter_saved") or {}
    by_its_record = (saved.get("rank"), saved.get("alpha")) == (rank, alpha) and rank is not None and (recorded.get("numbers_not_its_start_adapters") or 0) > 0
    candidate = by_its_record and (now is None or bool(now.get("passes")))
    return {"what": f"the candidate's adapter and `{BAR}`'s are there, at the rank and alpha, and are not each other's: as the first step held them before anything was "
                    "written (on a GPU box from the files themselves; from what the runs that made them recorded otherwise)",
            "rank": rank, "alpha": alpha, "the_candidate": {"adapter": of_the_candidate.get("adapter"), "recorded": dict(recorded), "read_now": now, "passes": candidate},
            "the_bar": {"adapter": of_the_bar.get("adapter"), "read_now": of_the_bar.get("read_now"), "passes": bool(of_the_bar.get("passes"))},
            "passes": candidate and bool(of_the_bar.get("passes"))}


# ---------------------------------------------------------------------------------------------- the sets
def sets_are_the_settings(sets: Mapping[str, Mapping]) -> dict:
    """EVERY SET IS THE SETTING'S: its problems (each once), its attempts a problem, its sampling seed. `sets`: by the
    name the report gives a set, its per-problem `rows`, the `problem_ids` the setting gives it, the setting's
    `attempts` and `sampling_seed`, and what its own step recorded (`recorded_attempts`, `recorded_seed`)."""
    of_each = {}
    for name, own in sets.items():
        ids, wanted = [row["problem_id"] for row in own["rows"]], list(own["problem_ids"])
        other = sorted({row["episodes"] for row in own["rows"]} - {own["attempts"]})
        the_problems = bool(ids) and sorted(ids) == sorted(wanted) and len(set(ids)) == len(ids)
        the_seed_ = own.get("recorded_seed") == own["sampling_seed"] and own.get("recorded_attempts") == own["attempts"]
        of_each[name] = {"problems": len(ids), "problems_of_the_setting": len(wanted), "the_problems": the_problems, "attempts": own["attempts"],
                         "attempts_a_problem_that_are_not_the_settings": other, "sampling_seed": own["sampling_seed"], "recorded_sampling_seed": own.get("recorded_seed"),
                         "recorded_attempts": own.get("recorded_attempts"), "passes": the_problems and not other and the_seed_}
    failing = [name for name, own in of_each.items() if not own["passes"]]
    return {"what": "every set read here is the setting's: its problems, each once; its attempts a problem; and the sampling seed its own step recorded",
            "sets": of_each, "not_the_settings": failing, "passes": bool(of_each) and not failing}


# ---------------------------------------------------------------------------------------------- the reads
def by_the_interval(failed_checks: Sequence[str], difference: Mapping, names: Sequence[str], where: str) -> dict:
    """ONE OF THE TWO READS, fixed before the run: the candidate minus the bar, successes per attempt, paired by
    problem, with its 95% bootstrap interval over problems (`difference`). `names`: what it is called with the interval
    above zero, spanning zero (one that ends at zero spans it) and below zero. `where`: the set, as a reason says it.
    INCONCLUSIVE when a check of this run failed; NOT READ when no problem was measured on both sides."""
    above, spanning, below = names
    if failed_checks:
        return {"name": INCONCLUSIVE, "failed_checks": list(failed_checks), "reason": f"a check of this run failed ({'; '.join(failed_checks)}): {where} is not read"}
    if difference.get("mean") is None:
        return {"name": NOT_READ, "failed_checks": [], "reason": f"no problem of {where} was measured on both sides: it is NOT READ"}
    read = f"{interval_text(difference)}, {difference.get('per_1000')} against {difference.get('per_1000_of_the_base')} per 1,000 over {difference.get('problems')} problems"
    name = above if difference["low"] > 0 else below if difference["high"] < 0 else spanning
    said = {above: "the interval is above zero", spanning: "the interval spans zero", below: "the interval is below zero"}[name]
    found = {"name": name, "failed_checks": [], "mean": difference["mean"], "low": difference["low"], "high": difference["high"]}
    if name == spanning:
        found["resolves"] = round((difference["high"] - difference["low"]) / 2, 5)
        said += f"; this difference resolves {found['resolves']:.5f} per attempt (half the width of its interval)"
    return {**found, "reason": f"on {where}, the candidate minus `{BAR}`, successes per attempt, paired by problem: {read}; {said}"}


def what_follows(on_the_set: Mapping, on_fresh: Mapping) -> dict:
    """What follows, fixed before the run. HOLDS ON BOTH: the candidate is the best model measured in full, said without
    the caution of selection, and the start the next cycle is proposed from. NOT SHOWN or NOT HOLDING on the held-aside
    set: the night's first-sampling comparisons are said to be selection on G until a further seed shows otherwise. It
    holds on the held-aside set and not on fresh attempts: THE SECTION NAMES NOTHING, which is said, and no claim is
    made. Nothing is named while a read is not made."""
    primary, second = on_the_set["name"], on_fresh["name"]
    if primary in (INCONCLUSIVE, NOT_READ) or second in (INCONCLUSIVE, NOT_READ):
        return {"name": NOT_READ, "reason": "a read of this run is not made: nothing is named for what follows"}
    if primary != HOLDS_ON_THE_SET:
        return {"name": SELECTION_ON_G,
                "reason": f"on the held-aside set the read is {primary} (on fresh attempts: {second}): the comparisons of the night on G's first sampling are said on the "
                          "page to be selection on G until a further seed shows otherwise"}
    if second == HOLDS_ON_FRESH:
        return {"name": THE_BEST_MEASURED,
                "reason": "the gain holds on the held-aside set and on fresh attempts: the candidate is the best model measured in full, said without the caution of "
                          "selection, and the start the next cycle is proposed from"}
    return {"name": NOT_NAMED,
            "reason": f"on the held-aside set the gain holds and on fresh attempts the read is {second}: the section names what follows for a gain that holds on both and "
                      "for one that does not hold on the held-aside set, and not this: nothing is claimed of it, and it needs a decision"}


# ---------------------------------------------------------------------------------------------- beside
def reliably(rows: Sequence[Mapping], problem_ids: Collection[str]) -> list[str]:
    """The problems of `problem_ids` a model solves RELIABLY: in half its attempts at them or more (and at least once)."""
    wanted = set(problem_ids)
    return sorted(row["problem_id"] for row in rows if row["problem_id"] in wanted and row["resolved"] > 0 and HALF * row["resolved"] >= row["episodes"])


def never_solved(rows: Sequence[Mapping], problem_ids: Collection[str]) -> list[str]:
    """The problems of `problem_ids` a model solved in NONE of its attempts, in the order of its rows."""
    wanted = set(problem_ids)
    return [row["problem_id"] for row in rows if row["problem_id"] in wanted and row["resolved"] == 0]
