"""L1b, the dose curve: how long to train on one round's proofs. Spec: docs/spec/ladder-loop.spec.md, "L1b: the
dose curve". Pure: no model, no Lean.

  the schedule     steps in a pass, the step each checkpoint is saved at, the steps at which the loss is read
  the pairs        the fixed sample of training proofs, and the HELD-OUT proofs: verified proofs the base wrote for
                   held-out rung problems, one per problem, so many per rung, drawn with the seed
  the driver       the training loop as a schedule of calls (`run_dose`): the order of examples is `_train`'s own
                   (one generator, reshuffled each pass), so the first pass is L1's training step for step. The GPU
                   step supplies the calls; a test supplies stand-ins
  the read         the three questions of the spec's table and the branch, fixed before the run
"""

from __future__ import annotations

import math
import random
from typing import Callable, Mapping, Sequence

from rlvr_lean.domain.evaluation.loss_parts import loss_by_part
from rlvr_lean.domain.ladder_round.read import paired_change
from rlvr_lean.domain.problem_pool.episodes import ABOVE_BAND, BELOW_BAND, IN_BAND, NEGATION_SIDE, RUNGS, STATEMENT_SIDE, VERIFIED
from rlvr_lean.domain.problem_pool.selection import rank

ONE_PASS = 1.0
SERIES_KEYS = ("mean_loss", "first_token_nats", "body_nats_per_token", "newline_nats", "fence_nats")
VOID = "VOID"
ESCALATE = "ESCALATE"                               # a later checkpoint beats one pass on the below-band rung
ONE_PASS_STANDS = "ONE PASS STANDS"                 # no checkpoint differs from one pass
ONE_PASS_STANDS_LATER_WORSE = "ONE PASS STANDS, LATER CHECKPOINTS ARE WORSE"
BRANCHES = (VOID, ESCALATE, ONE_PASS_STANDS, ONE_PASS_STANDS_LATER_WORSE)


# ------------------------------------------------------------------------------------------------ schedule
def checkpoint_name(passes: float) -> str:
    """`p050` for half a pass, `p100` for one, `p300` for three."""
    return f"p{round(passes * 100):03d}"


def steps_per_pass(examples: int, batch: int) -> int:
    if examples < 1 or batch < 1:
        raise ValueError(f"a pass needs at least one example and a batch of at least one, got {examples} and {batch}")
    return math.ceil(examples / batch)


def checkpoint_steps(examples: int, batch: int, passes: Sequence[float]) -> dict[str, int]:
    """The optimizer step after which each checkpoint is saved: a whole number of passes ends at the last step of
    that pass; a fraction is rounded up to the next step."""
    per_pass = steps_per_pass(examples, batch)
    return {checkpoint_name(value): max(1, math.ceil(value * per_pass - 1e-9)) for value in passes}


def reading_steps(total_steps: int, every_step_until: int, every: int, checkpoints: Sequence[int]) -> list[int]:
    """The steps at which the loss on the fixed pairs is read: step 0 (the base), every step up to
    `every_step_until`, then every `every` steps, every checkpoint step, and the last step."""
    steps = set(range(0, min(every_step_until, total_steps) + 1)) | set(range(0, total_steps + 1, every))
    return sorted(steps | {step for step in checkpoints if step <= total_steps} | {total_steps})


def pass_of(step: int, per_pass: int) -> float:
    return round(step / per_pass, 4)


# --------------------------------------------------------------------------------------------------- pairs
def training_sample(examples: Sequence[Mapping], seed: int, size: int) -> list[int]:
    """The positions of a fixed seeded sample of the training examples, in their own order."""
    order = sorted(range(len(examples)), key=lambda index: rank(seed, "dose_training_sample", examples[index]["attempt_id"]))
    return sorted(order[:size])


def heldout_pairs(groups: Sequence[Mapping], problems: Sequence[Mapping], attempts: Sequence[Mapping], seed: int, per_rung: int) -> list[dict]:
    """The held-out proofs: for each rung, `per_rung` of its problems that have a verified attempt (a seeded
    draw), and for each ONE of its verified attempts (a seeded draw), on whichever side was proved. Fewer
    problems with a proof than `per_rung`: all of them. In the rungs' order, then the draw's."""
    rung_of = {row["problem_id"]: row["group"] for row in groups if row["group"] in RUNGS}
    by_id = {row["problem_id"]: row for row in problems}
    verified: dict[str, list[Mapping]] = {}
    for attempt in attempts:
        if attempt["status"] == VERIFIED and attempt["problem_id"] in rung_of:
            verified.setdefault(attempt["problem_id"], []).append(attempt)
    pairs = []
    for rung in RUNGS:
        members = sorted((problem_id for problem_id in verified if rung_of[problem_id] == rung),
                         key=lambda problem_id: rank(seed, "dose_heldout_problem", problem_id))
        for problem_id in members[:per_rung]:
            chosen = min(verified[problem_id], key=lambda attempt: rank(seed, "dose_heldout_proof", attempt["attempt_id"]))
            theorem = {STATEMENT_SIDE: by_id[problem_id]["statement"], NEGATION_SIDE: by_id[problem_id].get("negation")}[chosen["side"]]
            if theorem is None:
                raise ValueError(f"{chosen['attempt_id']} verified the negation of {problem_id}, which has no built negation")
            pairs.append({"problem_id": problem_id, "rung": rung, "side": chosen["side"], "attempt_id": chosen["attempt_id"],
                          "theorem": theorem, "completion": chosen["completion"]})
    return pairs


def check_nothing_held_out_is_trained_on(examples: Sequence[Mapping], pairs: Sequence[Mapping], groups: Sequence[Mapping]) -> None:
    """No held-out problem, and so no held-out proof, is a training example. Raises ValueError otherwise."""
    held = {row["problem_id"] for row in groups}
    trained = {example["problem_id"] for example in examples}
    shared = sorted(trained & (held | {pair["problem_id"] for pair in pairs}))
    if shared:
        raise ValueError(f"{len(shared)} training examples are held-out problems (first: {shared[0]}): refused")


# -------------------------------------------------------------------------------------------------- driver
def parts_row(per_pair_losses: Sequence[Sequence[float]]) -> dict:
    """One reading by part: the mean per-token loss over the pairs, each framing token's nats per proof, and the
    body's nats per body token pooled over the pairs."""
    parts = loss_by_part(per_pair_losses)
    return {"pairs": parts["pairs"], "mean_loss": round(parts["mean_loss"], 5), "first_token_nats": round(parts["nats_per_proof"]["first"], 5),
            "body_nats_per_token": round(parts["body_per_token"], 5), "newline_nats": round(parts["nats_per_proof"]["newline"], 5),
            "fence_nats": round(parts["nats_per_proof"]["fence"], 5)}


def epoch_orders(examples: int, passes: int, seed: int) -> list[list[int]]:
    """The order of the examples in each pass: ONE generator seeded with the round's seed, the list reshuffled at
    the start of every pass. This is `gpu.ladder_round._train`'s order, so the first pass repeats L1's."""
    generator, orders = random.Random(seed), []
    for _ in range(passes):
        order = list(range(examples))
        generator.shuffle(order)
        orders.append(order)
    return orders


def run_dose(examples: int, batch: int, passes: int, seed: int, readings: Sequence[int], checkpoints: Mapping[str, int],
             train_step: Callable[[list[int]], Sequence[Sequence[float]]], read: Callable[[int], dict], save: Callable[[str, int], None],
             orders: Sequence[Sequence[int]] | None = None, per_example: bool = False, positions: bool = False) -> dict:
    """The training loop as a schedule. `train_step(batch of example positions)` makes ONE optimizer step and
    returns each example's per-position losses as they were BEFORE the update; `read(step)` reads the fixed pairs
    (it is called at step 0, before any update, and after the update of every other reading step); `save(name,
    step)` saves the adapter after the update of a checkpoint's step.

    `orders`: the order of the examples in each pass when it is not `epoch_orders`' (the ceiling trains in its
    file's order). `per_example`: a step's row also holds each of its examples' own mean loss per target token
    (`example_losses`, in the step's order). `positions`: and the positions of the examples it was made on
    (`example_positions`, in the step's order): the record of what was trained on."""
    per_pass, at = steps_per_pass(examples, batch), set(readings)
    by_step: dict[int, list[str]] = {}
    for name, step in checkpoints.items():
        by_step.setdefault(step, []).append(name)
    step_rows, reading_rows, saved, step = [], [], [], 0
    if 0 in at:
        reading_rows.append({"step": 0, "pass": 0.0, **read(0)})
    for order in (epoch_orders(examples, passes, seed) if orders is None else orders):
        for start in range(0, len(order), batch):
            losses = train_step(order[start:start + batch])
            step += 1
            # The loss of step s is read BEFORE its update: it describes the model after s - 1 updates.
            step_rows.append({"step": step, "pass": pass_of(step, per_pass), "updates_before": step - 1, **parts_row(losses)})
            if per_example:
                step_rows[-1]["example_losses"] = [round(sum(one) / len(one), 5) for one in losses]
            if positions:
                step_rows[-1]["example_positions"] = list(order[start:start + batch])
            for name in by_step.get(step, ()):
                save(name, step)
                saved.append({"checkpoint": name, "step": step, "pass": pass_of(step, per_pass)})
            if step in at:
                reading_rows.append({"step": step, "pass": pass_of(step, per_pass), **read(step)})
    missing = sorted(set(checkpoints) - {row["checkpoint"] for row in saved})
    if missing:
        raise ValueError(f"the checkpoints {missing} lie beyond the last step ({step}): {dict(checkpoints)}")
    return {"steps": step, "steps_per_pass": per_pass, "training_steps": step_rows, "readings": reading_rows, "checkpoints": saved}


def series(rows: Sequence[Mapping], key: str | None = None) -> dict:
    """Rows as arrays a page can plot: `step`, `pass`, and one array per part. `key` picks a nested reading."""
    picked = [(row, row[key] if key else row) for row in rows if key is None or row.get(key) is not None]
    return {"step": [row["step"] for row, _ in picked], "pass": [row["pass"] for row, _ in picked],
            **{name: [values[name] for _, values in picked] for name in SERIES_KEYS}}


# ---------------------------------------------------------------------------------------------------- read
def rung_ids(groups: Sequence[Mapping]) -> dict[str, list[str]]:
    return {rung: [row["problem_id"] for row in groups if row["group"] == rung] for rung in RUNGS}


def overfitting_by_loss(readings: Sequence[Mapping], step: int, last: int = 4) -> dict:
    """The loss half of the overfitting rule at a checkpoint's step, on the body's nats per token: the held-out
    loss is above its lowest point so far by more than the spread (highest minus lowest) of its last four
    readings, while the training-sample loss is still below what it was where the held-out loss was lowest."""
    so_far = [row for row in readings if row["step"] <= step and row.get("heldout") is not None]
    if not so_far:
        return {"readings": 0, "by_loss": False}
    held = [row["heldout"]["body_nats_per_token"] for row in so_far]
    train = [row["training_sample"]["body_nats_per_token"] for row in so_far]
    lowest = min(range(len(held)), key=lambda index: held[index])
    spread = max(held[-last:]) - min(held[-last:])
    above = held[-1] - held[lowest]
    still_falling = train[-1] < train[lowest]
    return {"readings": len(so_far), "heldout_body_nats_per_token": held[-1], "heldout_lowest": held[lowest], "lowest_at_step": so_far[lowest]["step"],
            "above_its_lowest_by": round(above, 5), "spread_of_the_last_four_readings": round(spread, 5),
            "training_sample_body_nats_per_token": train[-1], "training_sample_at_the_heldout_lowest": train[lowest],
            "training_sample_still_falls": still_falling, "by_loss": bool(above > spread and still_falling)}


def _below_zero(change: Mapping) -> bool:
    return change.get("high") is not None and change["high"] < 0


def _above_zero(change: Mapping) -> bool:
    return change.get("low") is not None and change["low"] > 0


def dose_read(checkpoints: Mapping[str, Mapping], groups: Sequence[Mapping], rungs: Mapping[str, Sequence[Mapping]], base_rungs: Sequence[Mapping],
              readings: Sequence[Mapping], reference: Mapping, resamples: int, seed: int) -> dict:
    """The read of the spec's table. `checkpoints[name]` = {"step", "pass"}; `rungs[name]` = that checkpoint's
    per-problem results on the three rungs; `base_rungs` = the base's stored results on the same problems with
    the same sampling seed; `reference` = L1's reported change on each rung at one pass ({"in": {low, high, mean},
    "above": ...}). Every change is paired by problem with a 95% bootstrap interval over problems."""
    ids = rung_ids(groups)
    together = ids[BELOW_BAND] + ids[IN_BAND]
    one = next(name for name, entry in checkpoints.items() if entry["pass"] == ONE_PASS)
    table = {}
    for name, entry in sorted(checkpoints.items(), key=lambda item: item[1]["step"]):
        minus_base = {rung: paired_change(rungs[name], base_rungs, ids[rung], resamples, seed) for rung in RUNGS}
        minus_one = {rung: paired_change(rungs[name], rungs[one], ids[rung], resamples, seed) for rung in RUNGS} if name != one else None
        both = paired_change(rungs[name], rungs[one], together, resamples, seed) if name != one else None
        rate = paired_change(rungs[name], base_rungs, together, resamples, seed)
        later = entry["pass"] > ONE_PASS
        loss = overfitting_by_loss(readings, entry["step"])
        by_pass_rate = bool(later and any(_below_zero(minus_one[rung]) for rung in RUNGS))
        table[name] = {"step": entry["step"], "pass": entry["pass"], "minus_base": minus_base, "minus_one_pass": minus_one,
                       "below_and_in_together": {"problems": len(together), "pass_rate": rate.get("mean_after"), "minus_one_pass": both},
                       "overfitting": {"asked": later, "loss": loss, "a_rung_is_below_its_one_pass_value": by_pass_rate,
                                       "rungs_below": [rung for rung in RUNGS if later and _below_zero(minus_one[rung])],
                                       "yes": bool(later and loss["by_loss"] and by_pass_rate)}}
    later_names = [name for name, entry in table.items() if entry["pass"] > ONE_PASS]
    # ---- did L1 measure at the peak?
    best = max(table, key=lambda name: (table[name]["below_and_in_together"]["pass_rate"] or 0.0, -table[name]["step"]))
    beating = [name for name in table if name != one and _above_zero(table[name]["below_and_in_together"]["minus_one_pass"])]
    # ---- would training longer help?
    longer = {name: {rung: table[name]["minus_one_pass"][rung] for rung in (BELOW_BAND, IN_BAND)} for name in later_names if table[name]["pass"] >= 2}
    # ---- VOID: the one-pass checkpoint must reproduce L1's one pass
    reproduced = {}
    for rung in (IN_BAND, ABOVE_BAND):
        measured, wanted = table[one]["minus_base"][rung], reference.get(rung) or {}
        inside = measured.get("mean") is not None and wanted.get("low") is not None and wanted["low"] <= measured["mean"] <= wanted["high"]
        reproduced[rung] = {"measured": measured.get("mean"), "l1_reported": wanted.get("mean"), "l1_interval": [wanted.get("low"), wanted.get("high")],
                            "within": bool(inside)}
    void = [rung for rung in reproduced if not reproduced[rung]["within"]]
    # ---- the branch
    better = [name for name in later_names if _above_zero(table[name]["minus_one_pass"][BELOW_BAND])]
    worse = [name for name in later_names if any(_below_zero(table[name]["minus_one_pass"][rung]) for rung in (BELOW_BAND, IN_BAND))]
    overfit = [name for name in later_names if table[name]["overfitting"]["yes"]]
    if void:
        branch = {"name": VOID, "reason": f"the one-pass checkpoint does not reproduce L1 on the {' and the '.join(void)} rung: it is not the same training. Fix and run again"}
    elif better:
        chosen = max(better, key=lambda name: table[name]["minus_one_pass"][BELOW_BAND]["mean"])
        change = table[chosen]["minus_one_pass"][BELOW_BAND]
        branch = {"name": ESCALATE, "passes": table[chosen]["pass"],
                  "reason": f"{table[chosen]['pass']} passes beat one pass on the below-band rung: {change['mean']} [{change['low']}, {change['high']}]. "
                            "One seed is a scout: escalate that number of passes to three seeds"}
    elif worse:
        branch = {"name": ONE_PASS_STANDS_LATER_WORSE,
                  "reason": f"no later checkpoint beats one pass on the below-band rung, and {', '.join(worse)} fall below it on the below-band or in-band rung "
                            f"(interval below zero). Overfitting by the spec's rule at: {', '.join(overfit) if overfit else 'no checkpoint'}"}
    else:
        half = next((name for name, entry in table.items() if entry["pass"] < ONE_PASS), None)
        less = table[half]["minus_one_pass"][BELOW_BAND] if half else None
        branch = {"name": ONE_PASS_STANDS,
                  "reason": "no later checkpoint differs from one pass on the below-band or in-band rung (every interval contains zero)"
                            + (f"; half a pass minus one pass on the below-band rung: {less['mean']} [{less['low']}, {less['high']}]" if less else "")}
    return {
        "one_pass_checkpoint": one, "checkpoints": table,
        "are_we_overfitting": {"rule": "at a checkpoint later than one pass: the held-out body loss is above its lowest reading so far by more than the spread "
                                       "(highest minus lowest) of its last four readings while the training-sample body loss is below what it was at that "
                                       "lowest point, AND a held-out rung's pass rate is below its one-pass value with an interval below zero",
                               "yes_at": overfit, "by_loss_alone_at": [name for name in later_names if table[name]["overfitting"]["loss"]["by_loss"]],
                               "by_pass_rate_alone_at": [name for name in later_names if table[name]["overfitting"]["a_rung_is_below_its_one_pass_value"]]},
        "did_l1_measure_at_the_peak": {"highest_pass_rate_on_below_and_in_together": best, "checkpoints_that_beat_one_pass": beating,
                                       "at_the_peak": not beating},
        "would_training_longer_help": longer,
        "void_check": {"reproduces_l1_at_one_pass": not void, "by_rung": reproduced},
        "branch": branch,
    }
