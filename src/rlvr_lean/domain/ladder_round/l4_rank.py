"""L4r: is the pretrained model capped by the size of its adapter? Spec: docs/spec/ladder-loop.spec.md, "L4r: is
the pretrained model capped by the size of its adapter?" and its "Made exact by the build". Pure.

A LABELLED CHECK OF THE PRETRAINING. `pre` is one adapter of rank 16, and it reaches what a third of its proofs reached.
One cause that fits is that the adapter is full. The check is `pre`'s pass again, from the base, with ONE change: the
adapter's rank and alpha (their ratio stays `pre`'s, so the adapter's scale is `pre`'s). Its model is `pre_r<rank>`; it
is measured as `pre` was, with `pre`'s sampling seeds, and read against `pre`'s stored rows.

  the check        which rank and alpha (`rank_and_alpha`), and the model's name
  the one change   what the check's prepare step recorded against what `pre`'s did: refused unless the rank and the
                   alpha are all that differ (`the_one_change`)
  an adapter       how many numbers a stored adapter holds, its rank, and how many one of another rank on the same
                   modules holds (`adapter_numbers`)
  the read         the loss by twentieth of the pass (`by_twentieth`); the first two checks (`training_checks`); the
                   branch, in the spec's words (`rank_branch`); the goal problems nothing stored has ever solved
                   (`never_solved`)
"""

from __future__ import annotations

import math
from typing import Collection, Mapping, Sequence

from rlvr_lean.domain.ladder_round.ceiling import still_writes_proofs, the_training_took
from rlvr_lean.domain.ladder_round.l3d import tenth
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ, PRE

RANK, FALLBACK = "rank", "fallback"                 # the two checks of `ladder_loop.l4.rank_check`: its own rank, and the one run when the box cannot hold it
CHECKS = (RANK, FALLBACK)
CAPPED = "THE ADAPTER'S SIZE CAPPED THE PRETRAINED MODEL"                       # the primary's interval clear of zero and above
NOT_THE_CAP = "NOT THE CAP AT THIS SIZE"                                        # the interval holds zero
WORSE = "THE LARGER ADAPTER IS WORSE AFTER ONE PASS AT THIS LEARNING RATE"      # the interval below zero
BRANCHES = (CAPPED, NOT_THE_CAP, WORSE, INCONCLUSIVE, NOT_READ)
TWENTIETHS = 20
# What the check's prepare step and `pre`'s must have recorded alike (beside the recipe, read key by key): the file, its
# rows and their order, the seed, the batch, the steps, the sequence limit.
THE_SAME = ("seed", "pretraining_file_sha256", "rows", "order", "effective_batch", "steps", "max_sequence_tokens")
CHANGED = ("rank", "alpha")                         # of the recipe's adapter: the ONE change


# ----------------------------------------------------------------------------------------------- the check
def rank_and_alpha(settings: Mapping, which: str) -> tuple[int, int]:
    """(the adapter's rank, its alpha) of the check `which` of `ladder_loop.l4.rank_check` (`settings`): `rank` is the
    setting's own pair; `fallback` is the pair run in its place when the box cannot hold that rank. Refused
    (ValueError): a check the setting does not have, a rank or alpha that is not a positive whole number, and a
    fallback that is not smaller than the rank it stands in for."""
    if which not in CHECKS:
        raise ValueError(f"{which!r} is not a check of ladder_loop.l4.rank_check: it is one of {list(CHECKS)}")
    pairs = {RANK: (settings["rank"], settings["alpha"]), FALLBACK: (settings["fallback_rank"], settings["fallback_alpha"])}
    for name, pair in pairs.items():
        if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in pair):
            raise ValueError(f"ladder_loop.l4.rank_check gives the check `{name}` rank {pair[0]!r} and alpha {pair[1]!r}: each is a positive whole number")
    if pairs[FALLBACK][0] >= pairs[RANK][0]:
        raise ValueError(f"ladder_loop.l4.rank_check.fallback_rank is {pairs[FALLBACK][0]} and its rank is {pairs[RANK][0]}: the fallback is the SMALLER adapter, "
                         "run when the box cannot hold the rank")
    return pairs[which]


def model_name(rank: int) -> str:
    """The check's model, its adapter's directory and its sets: `pre_r64`."""
    return f"{PRE}_r{rank}"


def _at(recorded: Mapping, key: str):
    value = recorded
    for part in key.split("."):
        value = value.get(part) if isinstance(value, Mapping) else None
    return value


def the_one_change(of_pre: Mapping, own: Mapping) -> dict:
    """The check is `pre`'s pass again with ONE change. `of_pre` and `own`: what the two prepare steps recorded (the
    pretraining's, and the check's). Refused (ValueError), naming everything that differs, unless the two recorded the
    same file (its SHA-256), rows, order, seed, batch, steps and sequence limit, and the same recipe but for the
    adapter's rank and alpha; refused too when the check's rank is not ABOVE `pre`'s (the question is whether a larger
    adapter reaches further) and when alpha / rank is not `pre`'s (the adapter's scale would be a second change).
    Returns the change, for the summary."""
    recipe_of_pre, recipe = of_pre["recipe"], own["recipe"]
    lora_of_pre, lora = recipe_of_pre["lora"], recipe["lora"]
    differs = [key for key in THE_SAME if of_pre.get(key) != own.get(key)]
    differs += [f"recipe.{key}" for key in sorted(set(recipe_of_pre) | set(recipe)) if key != "lora" and recipe_of_pre.get(key) != recipe.get(key)]
    differs += [f"recipe.lora.{key}" for key in sorted(set(lora_of_pre) | set(lora)) if key not in CHANGED and lora_of_pre.get(key) != lora.get(key)]
    if differs:
        raise ValueError(f"the check is `pre`'s pass again with ONE change, the adapter's rank and alpha, and {len(differs)} other "
                         f"thing{'s are' if len(differs) > 1 else ' is'} not `pre`'s: "
                         + "; ".join(f"{key} is {_at(own, key)!r} here and {_at(of_pre, key)!r} in `pre`'s run" for key in differs))
    if lora["rank"] <= lora_of_pre["rank"]:
        raise ValueError(f"the check's rank is {lora['rank']} and `pre`'s is {lora_of_pre['rank']}: the check asks whether a LARGER adapter reaches further")
    if lora["alpha"] * lora_of_pre["rank"] != lora_of_pre["alpha"] * lora["rank"]:
        raise ValueError(f"the check's alpha / rank is {lora['alpha']} / {lora['rank']} and `pre`'s is {lora_of_pre['alpha']} / {lora_of_pre['rank']}: the adapter's "
                         "scale would be a second change")
    return {"what": "the adapter's rank and alpha, and nothing else: the same file (its SHA-256), rows, order, seed, batch, steps, sequence limit, learning rate, "
                    "warm-up, dropout and target modules as `pre`'s prepare step recorded",
            "rank": {PRE: lora_of_pre["rank"], "check": lora["rank"]}, "alpha": {PRE: lora_of_pre["alpha"], "check": lora["alpha"]},
            "alpha_over_rank": lora["alpha"] / lora["rank"], "held_the_same": [*THE_SAME, *(f"recipe.{key}" for key in sorted(recipe) if key != "lora"),
                                                                               *(f"recipe.lora.{key}" for key in sorted(lora) if key not in CHANGED)]}


# ---------------------------------------------------------------------------------------------- an adapter
def adapter_numbers(tensors: Mapping[str, Mapping], rank: int | None = None) -> dict:
    """Of a stored LoRA adapter, from its file's header (`infrastructure.adapter_files.tensors_of`: by tensor name, its
    `shape`): its modules (one matrix A, rank x inputs, and one B, outputs x rank, each), the rank(s) its matrices
    have, and how many numbers it holds. With `rank`: how many an adapter of THAT rank on the same modules holds, rank x
    (inputs + outputs) summed over the modules. Nothing here needs a model."""
    a = [entry["shape"] for name, entry in tensors.items() if "lora_A" in name]
    b = [entry["shape"] for name, entry in tensors.items() if "lora_B" in name]
    a_rank = sum(shape[1] for shape in a) + sum(shape[0] for shape in b)        # what one more unit of rank adds: every A's inputs and every B's outputs
    counted = {"modules": len(a), "tensors": len(tensors), "ranks": sorted({shape[0] for shape in a} | {shape[1] for shape in b}),
               "numbers": sum(math.prod(entry["shape"]) for entry in tensors.values()), "numbers_a_unit_of_rank": a_rank}
    return counted if rank is None else {**counted, "rank": rank, "numbers_at_rank": rank * a_rank}


# ------------------------------------------------------------------------------------------------ the read
def by_twentieth(row_losses: Sequence[float], parts: int = TWENTIETHS) -> list[float | None]:
    """A training's loss by twentieth of its pass: the mean, over each of `parts` consecutive parts of its rows, of a
    row's loss (its mean loss per target token, read BEFORE the update of the step it was in: a loss on rows not yet
    trained on). Part i is the rows from `rows x i // parts` up to `rows x (i + 1) // parts`; a part with no row (a run
    of fewer rows than parts) is None."""
    count, means = len(row_losses), []
    for index in range(parts):
        part = row_losses[count * index // parts: count * (index + 1) // parts]
        means.append(round(sum(part) / len(part), 5) if part else None)
    return means


def training_checks(row_losses: Sequence[float], rung_rows: Sequence[Mapping], settings: Mapping) -> dict:
    """The first two "can this run see a win" checks, on the check's model. The training took: the mean loss over the
    last tenth of its rows is below the mean over the first tenth. It still writes proofs: under
    `maximum_share_without_an_answer` of its rung attempts without an answer. `settings`: `ladder_loop.l4`. (The third,
    Lean answered, is read by the report from each measured set.)"""
    return {"the_training_took": the_training_took(row_losses, tenth(len(row_losses), settings["loss_share_of_rows"])),
            "it_still_writes_proofs": still_writes_proofs(rung_rows, settings["maximum_share_without_an_answer"])}


def rank_branch(checks: Mapping, primary: Mapping, name: str, rank_of_pre: int) -> dict:
    """The branch, fixed before the run. `primary`: all of G, successes per attempt, the check's model minus `pre`,
    paired by problem, with its interval. `name`: the check's model. A check failed: INCONCLUSIVE, and nothing else is
    said. Clear of zero and above: the adapter's size capped the pretrained model. It holds zero (one that ends at zero
    holds it): not the cap at this size, with the interval's half-width as the size this run could have seen. Below
    zero: the larger adapter is worse after one pass at this learning rate."""
    failed = [check for check, read in checks.items() if not read["passes"]]
    if failed:
        return {"name": INCONCLUSIVE, "failed_checks": failed,
                "reason": f"this run could not have seen a win ({', '.join(failed)}): nothing is said about the size of the adapter"}
    if primary.get("mean") is None:
        return {"name": NOT_READ, "failed_checks": [], "reason": "no goal problem was measured on both sides: the primary has nothing to be read on"}
    mean, low, high = primary["mean"], primary["low"], primary["high"]
    read = f"{mean:+.5f} [{low:+.5f}, {high:+.5f}]"
    if low > 0:
        return {"name": CAPPED, "failed_checks": [],
                "reason": f"the interval is clear of zero and above ({read}): the adapter's size capped the pretrained model. The arm is then run again from `{name}` "
                          f"(its own map, the same halves, every model of the arm at that rank) before more seeds are bought at rank {rank_of_pre}, and every result of "
                          f"this project that reads \"not shown\" at rank {rank_of_pre} is marked as read under that cap"}
    if high < 0:
        return {"name": WORSE, "failed_checks": [],
                "reason": f"the interval lies below zero ({read}): the larger adapter is worse after one pass at this learning rate; rank {rank_of_pre} stays and "
                          "nothing is concluded about a larger adapter trained longer"}
    half_width = round((high - low) / 2, 6)
    return {"name": NOT_THE_CAP, "failed_checks": [], "half_width": half_width,
            "reason": f"the interval holds zero ({read}): not the cap at this size; rank {rank_of_pre} stays. The size this run could have seen is the interval's "
                      f"half-width, {half_width:.5f} per attempt"}


def never_solved(ids: Sequence[str] | None, on_g: Sequence[Mapping], of_pre: Sequence[Mapping], goal_ids: Collection[str]) -> dict:
    """How many of the goal problems that nothing stored has ever solved the check's model solves. `ids`: the list,
    GIVEN to the run as a file (it is not computed here: it is read over every stored run of the project), or None
    when no file was given. `on_g` and `of_pre`: the check's model's and `pre`'s per-problem rows over all their
    attempts on G. An id that is not a goal problem of this run is counted apart and reads nothing. `pre`'s own count
    stands beside: it is 0 for a list that is what it says."""
    if ids is None:
        return {"given": False, "what": "the goal problems nothing stored has ever solved: no file of their ids was given in the run directory, and the stage "
                                        "does not compute it"}
    goal, listed = set(goal_ids), list(dict.fromkeys(ids))
    wanted = {problem_id for problem_id in listed if problem_id in goal}
    solved_by = lambda rows: sorted(row["problem_id"] for row in rows if row["problem_id"] in wanted and row["resolved"] > 0)      # noqa: E731
    solved = solved_by(on_g)
    return {"given": True, "what": "the goal problems nothing stored has ever solved, as the file given in the run directory lists them: how many the check's model "
                                   "solves at least once in all its attempts on G",
            "ids": len(listed), "goal_problems": len(wanted), "not_goal_problems_of_this_run": len(listed) - len(wanted), "solved": len(solved), "problem_ids": solved,
            "solved_by_pre": len(solved_by(of_pre))}
