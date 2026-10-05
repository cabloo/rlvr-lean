"""L1b's report: the dose curve. Spec: docs/spec/ladder-loop.spec.md, "L1b: the dose curve". Pure: rows in, a
report out. It holds the three loss series as arrays a page can plot, answers the three questions of the spec's
table, states the VOID check and names the branch.

Where the plottable series are:
  loss_curves.training_step       every optimizer step, read BEFORE its update (in the first pass each proof is new
                                  to the model when it is read; from the second pass on it is not)
  loss_curves.training_sample     the fixed sample of training proofs, at the reading points
  loss_curves.heldout             the held-out proofs, at the reading points
  loss_curves.heldout_by_rung     the same, for the proofs of each rung
each as {step, pass, mean_loss, first_token_nats, body_nats_per_token, newline_nats, fence_nats}, and
  pass_rate_curve                 per checkpoint and rung: the pass rate, the base's, and the paired changes
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.dose import dose_read, rung_ids, series
from rlvr_lean.domain.ladder_round.read import reach
from rlvr_lean.domain.problem_pool.episodes import BELOW_BAND, IN_BAND, RUNGS

MAXIMUM_SHARE_WITHOUT_AN_ANSWER = 0.02      # above this share of attempts with no verdict a set's counts are not to be read


def _health(rows: Sequence[Mapping]) -> dict:
    attempts = sum(row["episodes"] * row["sides"] for row in rows)
    without = sum(row["attempts_without_an_answer"] for row in rows)
    return {"attempts": attempts, "without_an_answer": without, "share_without_an_answer": round(without / attempts, 5) if attempts else None}


def build_l1b_report(prepare: Mapping, training: Mapping, curves: Mapping, groups: Sequence[Mapping], base_rungs: Sequence[Mapping],
                     base_reach: Sequence[Mapping], rungs: Mapping[str, Sequence[Mapping]], reach_rows: Mapping[str, Sequence[Mapping]],
                     measured: Mapping[str, Mapping], reference: Mapping, evaluation: Mapping) -> dict:
    checkpoints = {name: {"step": prepare["checkpoint_steps"][name], "pass": prepare["checkpoint_passes"][name]} for name in rungs}
    read = dose_read(checkpoints, groups, rungs, base_rungs, curves["readings"], reference["rungs_trained_minus_base"],
                     evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"])
    ids = rung_ids(groups)
    by_name = sorted(checkpoints, key=lambda name: checkpoints[name]["step"])
    pass_rates = {rung: {"problems": len(ids[rung]),
                         "base": read["checkpoints"][by_name[0]]["minus_base"][rung].get("mean_before"),
                         "pass": [checkpoints[name]["pass"] for name in by_name], "step": [checkpoints[name]["step"] for name in by_name],
                         "pass_rate": [read["checkpoints"][name]["minus_base"][rung].get("mean_after") for name in by_name],
                         "minus_base": [read["checkpoints"][name]["minus_base"][rung].get("mean") for name in by_name],
                         "minus_base_low": [read["checkpoints"][name]["minus_base"][rung].get("low") for name in by_name],
                         "minus_base_high": [read["checkpoints"][name]["minus_base"][rung].get("high") for name in by_name]}
                  for rung in RUNGS}
    readings = curves["readings"]
    rungs_read = sorted({rung for row in readings for rung in (row.get("heldout_by_rung") or {})}, key=RUNGS.index)
    loss_curves = {
        "x": "step = optimizer updates made; pass = step / steps_per_pass. A training_step value at step s was read before update s",
        "steps_per_pass": curves["steps_per_pass"], "steps": curves["steps"],
        "checkpoints": [{"checkpoint": name, **checkpoints[name]} for name in by_name],
        "training_step": {"step": [row["step"] for row in curves["training_steps"]], "pass": [row["pass"] for row in curves["training_steps"]],
                          **{key: [row[key] for row in curves["training_steps"]] for key in ("mean_loss", "first_token_nats", "body_nats_per_token",
                                                                                             "newline_nats", "fence_nats")}},
        "training_sample": series(readings, "training_sample"),
        "heldout": series(readings, "heldout"),
        "heldout_by_rung": {rung: series([{**row, "one": (row.get("heldout_by_rung") or {}).get(rung)} for row in readings], "one") for rung in rungs_read},
    }
    reach_report = {name: reach(reach_rows[name], base_reach) for name in reach_rows}
    distinct = {"base": reference.get("base_distinct_attempts_on_the_rungs"),
                **{name: measured[name]["distinct_attempts_on_the_rungs"] for name in by_name}}
    health = {"rungs_base": _health(base_rungs), "reach_base": _health(base_reach), **{f"rungs_{name}": _health(rungs[name]) for name in by_name},
              **{f"reach_{name}": _health(reach_rows[name]) for name in reach_rows}}
    not_to_be_read = sorted(name for name, entry in health.items()
                            if entry["share_without_an_answer"] is not None and entry["share_without_an_answer"] > MAXIMUM_SHARE_WITHOUT_AN_ANSWER)
    one = read["one_pass_checkpoint"]
    longer = read["would_training_longer_help"]
    held = loss_curves["heldout"]["body_nats_per_token"]
    lowest = min(range(len(held)), key=lambda index: held[index]) if held else None
    branch = read["branch"]
    headline = (f"L1b seed {prepare['seed']}: {branch['name']}. Held-out body loss lowest at step "
                f"{loss_curves['heldout']['step'][lowest] if lowest is not None else None} "
                f"(pass {loss_curves['heldout']['pass'][lowest] if lowest is not None else None}) of {curves['steps']}; "
                f"below-band rung, passes minus one pass: "
                + ", ".join(f"{read['checkpoints'][name]['pass']}: {longer[name][BELOW_BAND]['mean']} [{longer[name][BELOW_BAND]['low']}, {longer[name][BELOW_BAND]['high']}]"
                            for name in longer)
                + f"; overfitting by the rule at: {', '.join(read['are_we_overfitting']['yes_at']) or 'no checkpoint'}; "
                  f"L1 measured at the peak: {read['did_l1_measure_at_the_peak']['at_the_peak']}"
                + (f"; NOT TO BE READ: too many attempts without an answer in {', '.join(not_to_be_read)}" if not_to_be_read else ""))
    return {
        "spec": "docs/spec/ladder-loop.spec.md, L1b: the dose curve", "headline": headline, "ok": not not_to_be_read,
        "seed": prepare["seed"], "arm": prepare["arm"], "stand_in_engine": bool(prepare.get("stand_in_engine") or training.get("stand_in_engine")),
        "branch": branch, "void_check": read["void_check"],
        "are_we_overfitting": read["are_we_overfitting"], "did_l1_measure_at_the_peak": read["did_l1_measure_at_the_peak"],
        "would_training_longer_help": {name: {"pass": read["checkpoints"][name]["pass"], **longer[name]} for name in longer},
        "the_primary_and_the_in_band_rung": {"what": "passes minus one pass, paired by problem, 95% bootstrap interval over problems",
                                             "rungs": [BELOW_BAND, IN_BAND], "one_pass_checkpoint": one},
        "checkpoints": read["checkpoints"], "pass_rate_curve": pass_rates,
        "loss_curves": loss_curves,
        "reach_on_g": {"against": "L1's stored base afresh on G, the same sampling seed", "by_checkpoint": reach_report,
                       "l1_reported_at_one_pass": reference.get("reach_on_g")},
        "distinct_attempts_on_the_rungs": distinct,
        "training": {**{key: value for key, value in training.items() if key not in ("losses",)},
                     "training_examples": prepare["training_examples"], "training_examples_sha256": prepare["training_examples_sha256"]},
        "heldout_pairs": {"pairs": prepare["heldout_pairs"], "by_rung": prepare["heldout_pairs_by_rung"], "short_by": prepare["heldout_pairs_short_by"],
                          "on_the_negation": prepare["heldout_pairs_on_the_negation"]},
        "l1_reference": reference, "attempts": health, "not_to_be_read": not_to_be_read,
        "measurements": {name: {key: value for key, value in measured[name].items() if key in ("rungs", "reach")} for name in by_name},
        "schedule": {key: prepare[key] for key in ("passes", "effective_batch", "steps_per_pass", "steps", "checkpoint_steps", "checkpoint_passes",
                                                   "reading_steps", "sampling_seeds", "rung_episodes", "reach_episodes", "reach_at")},
    }
