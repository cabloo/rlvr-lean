"""L4: the loop from a model pretrained on published proofs. Spec: docs/spec/ladder-loop.spec.md, "L4: the loop
from a model pretrained on published proofs". Pure.

THE POOL IS CUT IN TWO, BY PROBLEM, BEFORE ANYTHING IS TRAINED. A model pretrained on a problem's published proof and
then asked to prove that problem is being asked to remember. So each pool candidate goes to one half by a hash of its
id: the pretraining file is made from the `pretrain` half alone, and the rounds of this arm draw from the `loop` half
alone. `half_of` is the ONE place that says which half a problem is in: the tool that writes the pretraining file and
the stage that runs the rounds both call it.

THE READ (spec, "The read, fixed before any run"). The goal set is drawn again for the stronger model: G' is the goal
problems `pre` does not solve in its 32-attempt sampling (`goal_set_again`); its other sampling, 61 attempts, is fresh
for those problems, and the primary is read there: the round-six model (`with`) minus `pre`.

THE MAP OF THE MODEL AN ARM STARTS FROM. The challenger starts from a map of pass rates on the base map's problems,
and the stored one is the BASE's. An arm whose solver is `pre` from its first attempt reads `pre`'s own map instead:
the same problems, sides, attempts and sampling seed (`map_rows`, `check_start_map`, `map_summary`).
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Collection, Iterable, Mapping, Sequence

from rlvr_lean.domain.ladder_round.ceiling import still_writes_proofs
from rlvr_lean.domain.problem_pool.selection import rank

PRETRAIN, LOOP = "pretrain", "loop"
HALVES = (PRETRAIN, LOOP)
HALF_LABEL = "l4_half"


def half_of(problem_id: str, seed: int) -> str:
    """The half a problem is in: SHA-256 of `<seed>:l4_half:<problem id>` read as an integer, even is `pretrain` and
    odd is `loop`. A content hash, so it is the same wherever and whenever it is asked, and a problem is never in both."""
    return PRETRAIN if int(rank(seed, HALF_LABEL, problem_id), 16) % 2 == 0 else LOOP


def halves(problem_ids: Iterable[str], seed: int) -> dict[str, int]:
    """How many of `problem_ids` fall in each half."""
    counts = Counter(half_of(problem_id, seed) for problem_id in problem_ids)
    return {half: counts[half] for half in HALVES}


def refuse_the_other_half(rows: Iterable[Mapping], half: str, seed: int, what: str) -> None:
    """Every row's problem must be of `half`: one of the other half stops what is being made (ValueError). `what`
    names it in the refusal ("the pretraining file", "a round's candidates")."""
    found = sorted(row["problem_id"] for row in rows if half_of(row["problem_id"], seed) != half)
    if found:
        raise ValueError(f"{len(found)} problems of {what} are not of the `{half}` half (first: {found[0]}): refused, nothing was written")


# ------------------------------------------------------------------------------------------------ the read
PRE = "pre"                                     # the pretrained model, and its adapter's name
ADDS = "THE LOOP ADDS ON TOP OF PRETRAINING"                        # the primary's interval clear of zero and above
NOT_SHOWN = "NOT SHOWN ON TOP OF PRETRAINING"                       # the interval holds zero
COSTS = "THE ROUNDS COST THE PRETRAINED MODEL"                      # the interval below zero
INCONCLUSIVE = "INCONCLUSIVE"                                       # a check failed: the run could not have seen a win
NOT_READ = "NOT READ"                                               # the primary has nothing to be read on (a smoke run)
BRANCHES = (ADDS, NOT_SHOWN, COSTS, INCONCLUSIVE, NOT_READ)


def goal_set_again(first_sampling: Sequence[Mapping], goal_ids: Sequence[str]) -> list[str]:
    """G' for a model: the goal problems, in `goal_ids`' order, that it does not solve in the FIRST sampling of G (its
    32 attempts). `first_sampling`: that sampling's per-problem rows (`resolved`). Its second sampling is then fresh
    for these problems: nothing of it chose them."""
    solved = {row["problem_id"] for row in first_sampling if row["resolved"] > 0}
    return [problem_id for problem_id in goal_ids if problem_id not in solved]


def pretraining_checks(on_g: Sequence[Mapping], again: Sequence[str], goal_ids: Sequence[str], minimum_solved: int, minimum_again: int) -> dict:
    """The first two "can this run see a win" checks, read on `pre`. `on_g`: its per-problem rows over ALL its attempts
    on G (93). The pretraining took: it solves at least `minimum_solved` goal problems. G' holds at least
    `minimum_again` problems."""
    wanted = set(goal_ids)
    solved = sum(row["resolved"] > 0 for row in on_g if row["problem_id"] in wanted)
    return {"the_pretraining_took": {"what": f"`pre` solves at least {minimum_solved} goal problems in all its attempts on G", "goal_problems_solved": solved,
                                     "goal_problems": len(goal_ids), "minimum": minimum_solved, "passes": solved >= minimum_solved},
            "the_goal_set_again_is_large_enough": {"what": f"G' (the goal problems `pre` does not solve in its first sampling) holds at least {minimum_again}",
                                                   "problems": len(again), "minimum": minimum_again, "passes": len(again) >= minimum_again}}


# ------------------------------------------------------------------- the map of the model an arm starts from
MAP_STEP = "ladder_l4_pretrain_map"             # the step of the pretraining stage that makes `pre`'s own map, and its marker
MAP_FILE = "l4_map_pre.jsonl"                   # the map, in the pretraining run's directory, in the shape the challenger reads the base's
MAP_SET = "base_map"                            # the set its rows name, as the stored map's do (`data.ladder_round_export.BASE_MAP_SET`)
MAP_FIELDS = ("problem_id", "episodes", "resolved", "resolved_by_statement", "resolved_by_negation", "sides")
START_MODEL = "start_model"                     # an arm's `map` setting: the map of the model it starts from


def map_rows(results: Sequence[Mapping], base_map_ids: Sequence[str]) -> list[dict]:
    """A model's map as the challenger reads the base's: one row for each of the base map's problems, in the base
    map's order, with the stored map's fields and set. `results`: the per-problem rows of its episodes on them."""
    by_problem = {row["problem_id"]: row for row in results}
    missing = [problem_id for problem_id in base_map_ids if problem_id not in by_problem]
    if missing or len(by_problem) != len(base_map_ids):
        raise ValueError(f"the map's episodes are of {len(by_problem)} problems and the base map holds {len(base_map_ids)}"
                         + (f" ({len(missing)} of its problems have no result; first: {missing[0]})" if missing else ""))
    return [{**{field: by_problem[problem_id][field] for field in MAP_FIELDS}, "set": MAP_SET} for problem_id in base_map_ids]


def check_start_map(rows: Sequence[Mapping], recorded: Mapping, base_map_ids: Sequence[str], episodes: int, sampling_seed: int) -> None:
    """A start model's map can stand in the base map's place only when it is the same map of another model: refused
    (ValueError) when it was made with another sampling seed or number of attempts than the stored map's (`recorded`:
    what its step recorded), or is not of exactly the base map's problems, each with that number of attempts."""
    if recorded.get("sampling_seed") != sampling_seed or recorded.get("episodes_each") != episodes:
        raise ValueError(f"it was made with sampling seed {recorded.get('sampling_seed')} and {recorded.get('episodes_each')} attempts a problem; the stored map's are "
                         f"{sampling_seed} and {episodes}")
    ids = [row["problem_id"] for row in rows]
    if sorted(ids) != sorted(base_map_ids):
        other = sorted(set(ids) ^ set(base_map_ids))
        raise ValueError(f"it is of {len(ids)} problems and the base map holds {len(base_map_ids)}; they are not the same problems"
                         + (f" (first that is in one alone: {other[0]})" if other else " (a problem stands twice)"))
    wrong = [row["problem_id"] for row in rows if row["episodes"] != episodes]
    if wrong:
        raise ValueError(f"{len(wrong)} of its problems do not have {episodes} attempts (first: {wrong[0]})")


def map_summary(rows: Sequence[Mapping]) -> dict:
    """A map as a report prints it: its problems by k (how many of a problem's attempts resolved it), the mean pass
    rate, and the problems resolved at least once."""
    counts = Counter(row["resolved"] for row in rows)
    episodes = max((row["episodes"] for row in rows), default=0)
    return {"problems": len(rows), "episodes_each": episodes, "problems_by_k": {str(k): counts.get(k, 0) for k in range(episodes + 1)},
            "mean_pass_rate": round(sum(row["resolved"] / row["episodes"] for row in rows) / len(rows), 5) if rows else None,
            "resolved_at_least_once": sum(row["resolved"] > 0 for row in rows)}


# ---------------------------------------------------------------------------------------------- the checks
def loss_did_not_rise(row_losses: Sequence[float], window: int, standard_errors: float) -> dict:
    """A training's loss did not rise: the mean over the last `window` rows is not above the mean over the first
    `window` by more than `standard_errors` standard errors of their difference. A row's loss is its mean loss per
    target token, read before the update of the step it was in. THE STANDARD ERROR is the two windows' own: each
    window's sample variance (n - 1) over its number of rows, the two added, the square root (the rows of the two
    windows are different rows, taken as independent). A window of one row has no variance: the allowance is then
    zero. A run of fewer than twice the window compares its first half with its second."""
    compared = min(window, len(row_losses) // 2)
    first, last = list(row_losses[:compared]), list(row_losses[len(row_losses) - compared:]) if compared else []
    if not compared:
        return {"rows": len(row_losses), "rows_compared": 0, "first": None, "last": None, "rise": None, "standard_error": None, "allowed": None, "passes": False}
    rise = statistics.fmean(last) - statistics.fmean(first)
    error = math.sqrt(statistics.variance(first) / compared + statistics.variance(last) / compared) if compared > 1 else 0.0
    return {"rows": len(row_losses), "rows_compared": compared, "first": round(statistics.fmean(first), 5), "last": round(statistics.fmean(last), 5),
            "rise": round(rise, 5), "standard_error": round(error, 5), "allowed": round(standard_errors * error, 5), "passes": rise <= standard_errors * error}


def the_adapter_differs(against: Mapping | None) -> dict:
    """A trained adapter is not the one it started from: at least one of its stored numbers is another. `against`:
    what the training recorded when it compared the file of the adapter it SAVED with the start adapter's, bit for
    bit (`infrastructure.adapter_files.compared`: `elements`, `elements_changed`); a count, so nothing here can pass
    or fail by rounding. A training that recorded no comparison (the stand-in trains nothing) does not pass."""
    if not against or against.get("elements_changed") is None:
        return {"compared": False, "elements": None, "elements_changed": None, "passes": False}
    return {"compared": True, "elements": against.get("elements"), "elements_changed": against["elements_changed"], "largest_change": against.get("largest_change"),
            "passes": against["elements_changed"] > 0}


def the_training_ran(row_losses: Sequence[float], against: Mapping | None, settings: Mapping) -> dict:
    """One training of the arm, as the check reads it: its adapter differs from the start adapter's, and its loss did
    not rise (the last tenth of its rows against the first, two standard errors allowed)."""
    differs = the_adapter_differs(against)
    loss = loss_did_not_rise(row_losses, max(1, int(len(row_losses) * settings["loss_share_of_rows"])), settings["standard_errors_allowed"])
    return {"the_adapter_differs_from_the_start": differs, "the_loss_did_not_rise": loss, "passes": differs["passes"] and loss["passes"]}


def arm_checks(of_the_pretraining: Mapping, measured_trainings: Mapping[str, Mapping], rung_rows: Mapping[str, Sequence[Mapping]],
               trained_on: Mapping[str, Sequence[str]], heldout: Collection[str], half_seed: int, settings: Mapping) -> dict:
    """The arm's "can this run see a win" checks (any failing: INCONCLUSIVE). `of_the_pretraining`: the two checks
    the pretraining stage's report read. `measured_trainings`: for the two trainings whose models are measured
    (`with`'s and `without`'s), its `row_losses` and what it recorded `against_the_start_adapter`. `rung_rows`: by
    measured model, its rows on the rungs. `trained_on`: by training of the arm (all seven), the problems of its rows."""
    ran = {name: the_training_ran(entry["row_losses"], entry.get("against_the_start_adapter"), settings) for name, entry in measured_trainings.items()}
    writes = {name: still_writes_proofs(rows, settings["maximum_share_without_an_answer"]) for name, rows in rung_rows.items()}
    barred = {name: sorted(key for key in set(problems) if key in heldout or half_of(key, half_seed) != LOOP) for name, problems in trained_on.items()}
    found = sorted({key for keys in barred.values() for key in keys})
    return {**{name: dict(check) for name, check in of_the_pretraining.items()},
            "the_two_measured_trainings_ran": {
                "what": "for each of the two trainings whose models are measured: the adapter it saved differs from the start adapter's, and the mean loss over the "
                        f"last {settings['loss_share_of_rows']:g} of its rows is not above the mean over the first by more than {settings['standard_errors_allowed']:g} "
                        "standard errors of their difference", **ran, "passes": len(ran) == 2 and all(entry["passes"] for entry in ran.values())},
            "each_measured_model_still_writes_proofs": {"what": "for each measured model, the share of its attempts on the three rungs without an answer is under "
                                                                f"{settings['maximum_share_without_an_answer']}", **writes,
                                                        "passes": bool(writes) and all(entry["passes"] for entry in writes.values())},
            "no_training_row_is_of_the_pretrain_half_or_held_out": {
                "what": "no row of any training of the arm is a problem of the `pretrain` half or a held-out problem",
                "rows": {name: len(problems) for name, problems in trained_on.items()}, "barred_problems": len(found), "first": found[:5], "passes": not found}}


def the_note(primary: Mapping, base_arm: Mapping | None) -> dict:
    """Beside an interval that holds zero: what the base arm's own gain (the same quantity, from its stored rows) would
    have looked like here: its size against half the width of this primary's interval."""
    resolves = round((primary["high"] - primary["low"]) / 2, 6)
    if not base_arm or base_arm.get("mean") is None:
        return {"what": "the base arm's stored rows were not read: its own gain is not set against this interval", "resolves": resolves, "base_arms_gain": None, "seen": None}
    return {"what": "the base arm's own gain on its own goal set drawn again, against half the width of this primary's interval",
            "resolves": resolves, "base_arms_gain": base_arm["mean"], "seen": abs(base_arm["mean"]) >= resolves}


def l4_branch(checks: Mapping, primary: Mapping, base_arm: Mapping | None = None) -> dict:
    """The branch, fixed before the run. `primary`: on G', successes per attempt over the 61 fresh attempts, `with`
    minus `pre`. A check failed: INCONCLUSIVE, and nothing else is said. Clear of zero and above: the loop adds on
    top of pretraining. It holds zero (one that ends at zero holds it): not shown, with the note. Below zero: the
    rounds cost the pretrained model."""
    failed = [name for name, check in checks.items() if not check["passes"]]
    if failed:
        return {"name": INCONCLUSIVE, "failed_checks": failed,
                "reason": f"this run could not have seen a win ({', '.join(failed)}): nothing is said about what the loop adds on top of pretraining"}
    if primary.get("mean") is None:
        return {"name": NOT_READ, "failed_checks": [], "reason": "G' holds no problem that was measured: the primary has nothing to be read on"}
    mean, low, high = primary["mean"], primary["low"], primary["high"]
    read = f"{mean:+.5f} [{low:+.5f}, {high:+.5f}]"
    if low > 0:
        return {"name": ADDS, "failed_checks": [],
                "reason": f"the interval is clear of zero and above ({read}): the loop adds on top of pretraining. Two more seeds of the arm (the pretraining is not "
                          "repeated: the same `pre`), and this is the loop from here"}
    if high < 0:
        return {"name": COSTS, "failed_checks": [],
                "reason": f"the interval lies below zero ({read}): the rounds cost the pretrained model on what it could not solve (the rounds' own proofs, easier "
                          "than what it was pretrained on, pull it back). The loop is not run on top of pretraining as it stands"}
    note = the_note(primary, base_arm)
    beside = ("the base arm's stored rows were not read" if note["seen"] is None else
              f"the base arm's own gain, {note['base_arms_gain']:+.5f}, {'would' if note['seen'] else 'would not'} have been seen here")
    return {"name": NOT_SHOWN, "failed_checks": [], "the_note": note,
            "reason": f"the interval holds zero ({read}): not shown on top of pretraining at this size. This run resolves {note['resolves']:.5f}; {beside}"}
