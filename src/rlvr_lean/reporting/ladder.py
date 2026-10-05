"""The first rung of the 50% ladder (spec §13b): the report. Pure arithmetic over the run's JSONL artifacts.

Primary: the conjectures BELOW the band that the adapter never trained on (the base proved them 1 or 2 times
of 12), each sampled afresh by the adapter and by the base with one new sampling seed, in one session. The
measure is the mean pass rate, adapter minus base, paired over those conjectures. The secondaries use the
same fresh-sample pairing on statements the base never proved; the standard evaluation of spec §8 and the
census of false conjectures are reported beside them. Before any branch is read, the report states how the
Lean checks of the adapter's and of the base's fresh samples ended: they ran in the same task and should match.
"""

from __future__ import annotations

import json
import statistics

from rlvr_lean.domain.evaluation.bootstrap import BootstrapInterval, interval_label, paired_bootstrap
from rlvr_lean.domain.evaluation.ladder import CENSUS_MARKER, ladder_branch, trained_on
from rlvr_lean.domain.evaluation.pass_at_k import pass_at_k
from rlvr_lean.domain.evaluation.scout import distinct_share, share_starting_with
from rlvr_lean.domain.training.arms import FRESH_SETS, LADDER_ARM, ArmNames, FreshNames, base_loss_names
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.reporting.native_scout import _completions, _heldout_loss, _loss_rows, _status_shares, _training
from rlvr_lean.reporting.phase_a import _comparison, _counts
from rlvr_lean.reporting.phase_b import _arm_counts, _base_counts, _evaluation_ids, _subset

LADDER_REPORT_MARKER = "report_ladder"
LADDER_REPORT_FILE = "ladder_report.jsonl"
LADDER_REPORT_MARKDOWN = "ladder_report.md"
SECONDARIES = {"never_proved": "never-proved conjectures, proved at least once",
               "workbook_unsolved": "workbook-holdout problems the base never proved, proved at least once"}


def fresh_counts(store: ArtifactStore, who: str, set_name: str) -> dict[str, tuple[int, int]]:
    """statement id -> (fresh samples drawn, verified) for one model on one set; empty if not sampled yet."""
    fresh = FreshNames(who, set_name)
    if not store.is_done(fresh.marker):
        return {}
    verification = {row["attempt_id"]: row["status"] for row in store.read_rows(fresh.verification_file)}
    return _counts(store.read_rows(fresh.attempts_file), verification, fresh.variant)


def check_conditions(store: ArtifactStore, who: str, set_name: str) -> dict | None:
    """How one model's fresh samples on one set were checked: the share of timeouts and of server errors (a
    proof that was not judged counts as a failure) and the median check time."""
    fresh = FreshNames(who, set_name)
    if not store.is_done(fresh.marker):
        return None
    rows = store.read_rows(fresh.verification_file)
    seconds = [row["seconds"] for row in rows if row.get("seconds") is not None]
    statuses = [row["status"] for row in rows]
    return {"checks": len(rows), "timeout_share": round(statuses.count("timeout") / len(rows), 5),
            "server_error_share": round(statuses.count("server_error") / len(rows), 5),
            "median_check_seconds": round(statistics.median(seconds), 3) if seconds else None}


def paired_interval(base: dict, adapted: dict, k: int, resamples: int, seed: int) -> BootstrapInterval | None:
    """Adapter minus base on pass@k, paired over the statements both sampled (unrounded: what a branch reads)."""
    usable = sorted(i for i in base if base[i][0] >= k and i in adapted and adapted[i][0] >= k)
    if not usable:
        return None
    return paired_bootstrap([pass_at_k(*adapted[i], k) - pass_at_k(*base[i], k) for i in usable], resamples=resamples, seed=seed)


def mean_rate(counts: dict) -> float | None:
    return sum(verified / drawn for drawn, verified in counts.values()) / len(counts) if counts else None


def proved_once(counts: dict) -> int:
    return sum(verified > 0 for _, verified in counts.values())


def build_ladder_report(store: ArtifactStore, config: dict, profile: str, runs: list[ArmNames]) -> dict:
    evaluation, ladder = config["evaluation"], config["ladder"]
    resamples, bootstrap_seed = evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"]
    k = config["profiles"][profile]["minif2f_samples"]
    if len(runs) != 1 or runs[0].arm != LADDER_ARM or runs[0].seed != ladder["seed"]:
        raise ValueError(f"the rung is ONE run, {LADDER_ARM} at seed {ladder['seed']} (spec §13b); got {[names.key for names in runs]}")
    names = runs[0]
    fresh = {set_name: {"base": fresh_counts(store, "base", set_name), "adapter": fresh_counts(store, names.key, set_name)} for set_name in FRESH_SETS}
    if not fresh["below_band"]["base"] or not fresh["below_band"]["adapter"]:
        raise RuntimeError("the below-band conjectures have no fresh samples from both models: the primary cannot be read (a task that died reads VOID)")

    # --- the primary, and what it stands on
    below = fresh["below_band"]
    primary_interval = paired_interval(below["base"], below["adapter"], 1, resamples, bootstrap_seed)
    primary = _comparison(below["base"], [below["adapter"]], 1, resamples, bootstrap_seed)
    stored = _stored_counts(store)
    training_ids = _training_ids(store, names)
    below_ids = sorted(below["base"])
    base_fresh_rate = mean_rate(below["base"])
    low, high = ladder["void_base_fresh_rate"]

    # --- the secondaries: "proved at least once" is pass@k at k = the samples drawn
    secondary_intervals, secondaries = {}, {}
    for set_name, label in SECONDARIES.items():
        base_counts, adapter_counts = fresh[set_name]["base"], fresh[set_name]["adapter"]
        if not base_counts or not adapter_counts:
            secondaries[set_name] = None
            continue
        samples = next(iter(base_counts.values()))[0]
        interval = paired_interval(base_counts, adapter_counts, samples, resamples, bootstrap_seed)
        secondary_intervals[label] = interval
        secondaries[set_name] = {"statements": len(base_counts), "samples_each": samples, "base_proved_at_least_once": proved_once(base_counts),
                                 "adapter_proved_at_least_once": proved_once(adapter_counts),
                                 "proved_by_both": sum(base_counts[i][1] > 0 and adapter_counts.get(i, (0, 0))[1] > 0 for i in base_counts),
                                 "delta": round(interval.mean, 5), "interval": [round(interval.low, 5), round(interval.high, 5)],
                                 "label": interval_label(interval),
                                 "pass_rate": _comparison(base_counts, [adapter_counts], 1, resamples, bootstrap_seed)}
    branch = ladder_branch(primary_interval, base_fresh_rate, (low, high), secondary_intervals)
    conditions = {set_name: {"base": check_conditions(store, "base", set_name), "adapter": check_conditions(store, names.key, set_name)}
                  for set_name in FRESH_SETS}

    # --- the standard evaluation of spec §8, the adapter against the base's stored samples
    standard, solved = {}, {}
    if store.is_done(names.sampling_marker):
        ids = _evaluation_ids(store)
        base_holdouts, base_minif2f = _base_counts(store)
        base_sets = {"conjecture_holdout": _subset(base_holdouts, ids["conjecture_holdout"]),
                     "workbook_holdout": _subset(base_holdouts, ids["workbook_holdout"]), "minif2f": _subset(base_minif2f, ids["minif2f"])}
        adapter_counts = _arm_counts(store, names)
        for name, set_name, at in (("workbook_holdout_pass@1", "workbook_holdout", 1), ("conjecture_holdout_pass@1", "conjecture_holdout", 1),
                                   ("minif2f_pass@1", "minif2f", 1), (f"minif2f_pass@{k}", "minif2f", k)):
            standard[name] = _comparison(base_sets[set_name], [_subset(adapter_counts, ids[set_name])], at, resamples, bootstrap_seed)
        for set_name, members in ids.items():
            adapter_set = _subset(adapter_counts, members)
            solved[set_name] = {"base": proved_once(base_sets[set_name]), "adapter": proved_once(adapter_set),
                                "new": sum(base_sets[set_name][i][1] == 0 and adapter_set[i][1] > 0 for i in adapter_set if i in base_sets[set_name]),
                                "lost": sum(base_sets[set_name][i][1] > 0 and adapter_set[i][1] == 0 for i in adapter_set if i in base_sets[set_name])}
        attempts = store.read_rows(names.eval_attempts_file)
        distinct = {"base": round(distinct_share(_completions(store.read_rows("proof_attempts.jsonl"), "base", ids["conjecture_holdout"])), 4),
                    "adapter": round(distinct_share(_completions(attempts, names.variant, ids["conjecture_holdout"])), 4)}
        token_id = (_training(store, names) or {}).get("sequence_start_token_id")
        sequence_start = round(share_starting_with([row.get("first_token_id") for row in attempts], token_id), 5) if token_id is not None else None
    else:
        distinct, sequence_start = None, None
    base_names = base_loss_names(names.target_format)
    heldout = {"nf4": _heldout_loss(_loss_rows(store, base_names.nf4_file), _loss_rows(store, names.nf4_loss_file), resamples, bootstrap_seed),
               "fp8": _heldout_loss(_loss_rows(store, base_names.fp8_file), _loss_rows(store, names.fp8_loss_file), resamples, bootstrap_seed)}
    training = _training(store, names) or {}
    census = store.done_summary(CENSUS_MARKER) if store.is_done(CENSUS_MARKER) else None

    headline = (f"{profile}, seed {names.seed}: below-band pass rate ({len(below_ids)} untrained conjectures, {primary['problems']} paired), adapter minus base "
                f"{primary['delta']:+.4f} [{primary['interval'][0]:+.4f}, {primary['interval'][1]:+.4f}] {primary['label']} "
                f"(base {primary['base']}, adapter {primary['adapter']}; base must lie in [{low}, {high}]) -> {branch.name}: {branch.reason}")
    report = {"profile": profile, "arm": names.arm, "seed": names.seed, "headline": headline,
              "branch": {"name": branch.name, "reason": branch.reason, "action": branch.action},
              "primary": {**primary, "unrounded_delta": primary_interval.mean, "unrounded_interval": [primary_interval.low, primary_interval.high],
                          "below_band_conjectures": len(below_ids), "of_those_in_the_training_set": len(trained_on(below_ids, training_ids)),
                          "training_examples": len(training_ids),
                          "stored_verified_counts": {str(count): sum(stored[i][1] == count for i in below_ids) for count in sorted({stored[i][1] for i in below_ids})},
                          "base_stored_mean_rate": round(mean_rate({i: stored[i] for i in below_ids}), 4),
                          "base_fresh_mean_rate": round(base_fresh_rate, 4), "void_range": [low, high],
                          "base_fresh_rate_in_range": low <= base_fresh_rate <= high},
              "secondaries": secondaries, "never_proved_luck_baseline": {"expected": ladder["never_proved_luck_expected"], "range_99": ladder["never_proved_luck_range"]},
              "lean_checks_fresh": conditions,
              "standard_evaluation": standard, "solved_at_least_once": solved, "distinct_attempts_conjecture_holdout": distinct,
              "sequence_start_token_first_share": sequence_start, "lean_checks_standard": _status_shares(store, names.eval_verification_file),
              "heldout_loss": heldout, "heldout_loss_by_part_during_training": training.get("heldout_parts_curve"),
              "census": census, "training": {key: value for key, value in training.items() if key != "heldout_parts_curve"}}
    store.write_rows(LADDER_REPORT_FILE, [report])
    markdown = _markdown(report)
    store.path(LADDER_REPORT_MARKDOWN).write_text(markdown)
    if store.mirror_directory is not None:
        (store.mirror_directory / LADDER_REPORT_MARKDOWN).write_text(markdown)
    return report


def _stored_counts(store: ArtifactStore) -> dict[str, tuple[int, int]]:
    """statement id -> (samples drawn, verified) among the base's STORED samples."""
    verification = {row["attempt_id"]: row["status"] for row in store.read_rows("verification.jsonl")}
    return _counts(store.read_rows("proof_attempts.jsonl"), verification, "base")


def _training_ids(store: ArtifactStore, names: ArmNames) -> list[str]:
    """The conjectures the arm trained on: its selection file (written when it trained), or Phase A's examples."""
    name = f"selection_{names.examples_arm}.jsonl"
    if store.path(name).exists():
        return list(store.read_rows(name)[0]["conjecture_ids"])
    for row in store.read_rows("selections.jsonl"):
        if row["method"] == names.examples_arm:
            return list(row["conjecture_ids"])
    raise RuntimeError(f"no stored selection for {names.examples_arm}: the training set cannot be checked against the primary's conjectures")


def ladder_scalars(report: dict) -> dict[str, float]:
    """The headline figures as TensorBoard scalars (read while a long task is still running)."""
    primary = report["primary"]
    values = {"ladder/primary_delta": primary["delta"], "ladder/primary_low": primary["interval"][0], "ladder/primary_high": primary["interval"][1],
              "ladder/below_band_base_fresh_rate": primary["base_fresh_mean_rate"], "ladder/below_band_adapter_fresh_rate": primary["adapter"],
              "ladder/below_band_conjectures": float(primary["below_band_conjectures"]),
              "ladder/below_band_in_training_set": float(primary["of_those_in_the_training_set"]),
              f"ladder/branch/{report['branch']['name']}": 1.0}
    for set_name, result in report["secondaries"].items():
        if result is not None:
            values[f"ladder/{set_name}/base_proved"] = float(result["base_proved_at_least_once"])
            values[f"ladder/{set_name}/adapter_proved"] = float(result["adapter_proved_at_least_once"])
            values[f"ladder/{set_name}/delta_low"] = result["interval"][0]
            values[f"ladder/{set_name}/delta_high"] = result["interval"][1]
    for set_name, by_model in report["lean_checks_fresh"].items():
        for who, checks in by_model.items():
            if checks is not None:
                values[f"ladder/{set_name}/timeout_share_{who}"] = checks["timeout_share"]
                values[f"ladder/{set_name}/server_error_share_{who}"] = checks["server_error_share"]
    for name, result in report["standard_evaluation"].items():
        if result is not None:
            values[f"ladder/standard_{name}/delta"] = result["delta"]
            values[f"ladder/standard_{name}/low"] = result["interval"][0]
            values[f"ladder/standard_{name}/high"] = result["interval"][1]
    if report["census"] is not None:
        for key in ("never_proved", "sampled", "disproved", "disproved_among_exact", "negation_inexact", "negation_does_not_compile"):
            values[f"census/{key}"] = float(report["census"][key])
    return values


def _markdown(report: dict) -> str:
    branch, primary = report["branch"], report["primary"]
    lines = [f"# rlvr_lean ladder, first rung (spec 13b) — {report['profile']} run, seed {report['seed']}", "", f"**{report['headline']}**", "",
             f"Branch: **{branch['name']}** ({branch['reason']}). Next: {branch['action']}.", "",
             "## Before reading the branch: how the fresh samples were checked", "",
             "| Set | Model | Checks | Timeouts | Server errors | Median check seconds |", "|---|---|---|---|---|---|"]
    for set_name, by_model in report["lean_checks_fresh"].items():
        for who, checks in by_model.items():
            if checks is not None:
                lines.append(f"| {set_name} | {who} | {checks['checks']} | {checks['timeout_share']:.3%} | {checks['server_error_share']:.3%} | {checks['median_check_seconds']} |")
    lines += ["", "## Primary", "", f"`{json.dumps(primary)}`", "", "## Secondaries (fresh samples, proved at least once)", ""]
    for set_name, result in report["secondaries"].items():
        lines.append(f"- **{set_name}**: `{json.dumps(result)}`")
    lines += [f"- the base's luck on the never-proved conjectures, predicted from its stored counts: `{json.dumps(report['never_proved_luck_baseline'])}`", "",
              "## Standard evaluation (adapter against the base's stored samples)", "",
              "| Measure | Problems | Base | Adapter | Difference | 95% interval | Label |", "|---|---|---|---|---|---|---|"]
    for name, result in report["standard_evaluation"].items():
        if result is None:
            lines.append(f"| {name} | | | | | | not measurable |")
        else:
            lines.append(f"| {name} | {result['problems']} | {result['base']} | {result['adapter']} | {result['delta']:+.4f} | "
                         f"[{result['interval'][0]:+.4f}, {result['interval'][1]:+.4f}] | {result['label']} |")
    lines += ["", f"- Solved at least once: `{json.dumps(report['solved_at_least_once'])}`",
              f"- Distinct attempts on the conjecture holdout: `{json.dumps(report['distinct_attempts_conjecture_holdout'])}`",
              f"- Attempts that start with the sequence-start token: `{report['sequence_start_token_first_share']}`",
              f"- Lean checks of the standard evaluation: `{json.dumps(report['lean_checks_standard'])}`",
              f"- Held-out loss: `{json.dumps(report['heldout_loss'])}`", "", "## Held-out loss by part during training", "",
              "| Step | Mean per-token loss | Body nats per token | First token nats | Newline nats | Fence nats |", "|---|---|---|---|---|---|"]
    for point in report["heldout_loss_by_part_during_training"] or []:
        nats = point["nats_per_proof"]
        lines.append(f"| {point['step']} | {point['mean_loss']} | {point['body_per_token']} | {nats['first']} | {nats['newline']} | {nats['fence']} |")
    lines += ["", "## Census: never-proved conjectures whose negation the base proves", "", f"`{json.dumps(report['census'])}`", "",
              "## Training", "", f"`{json.dumps(report['training'])}`"]
    return "\n".join(lines) + "\n"
