"""The one-seed scout of spec §13a: does the pass@1 gain survive the format fix? Pure arithmetic over the
run's JSONL artifacts.

The arm `native_same_picks` trains on Phase A's own examples with the pairs in the model's native format
(spec §6 item 2a). Its control is Phase A's stored seed-0 adapter and evaluation, read here and never
re-run; the base model's stored samples are shared, since prompts and sampling do not change. Every figure
of the native adapter is reported with the legacy adapter's beside it.

This report has its own files and marker. Phase A's and Phase B's stay as they were written.
"""

from __future__ import annotations

import json

from rlvr_lean.domain.evaluation.bootstrap import interval_label, paired_bootstrap
from rlvr_lean.domain.evaluation.loss_parts import rows_have_parts, summarize_loss_rows
from rlvr_lean.domain.evaluation.pass_at_k import pass_at_k
from rlvr_lean.domain.evaluation.scout import control_reproduces, distinct_share, scout_branch, share_starting_with
from rlvr_lean.domain.training.arms import SCOUT_ARM, ArmNames, arm_names, base_loss_names
from rlvr_lean.domain.training.target_format import LEGACY
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.reporting.phase_a import _comparison, _loss_comparison
from rlvr_lean.reporting.phase_b import _arm_counts, _base_counts, _evaluation_ids, _subset

SCOUT_REPORT_MARKER = "report_native_scout"
SCOUT_REPORT_FILE = "native_scout_report.jsonl"
SCOUT_REPORT_MARKDOWN = "native_scout_report.md"
# The read over several seeds (the escalation the one-seed scout earned) is written beside the scout's own
# report, which stays as it was written.
SCOUT_SEEDS_REPORT_MARKER = "report_native_scout_seeds"
SCOUT_SEEDS_REPORT_FILE = "native_scout_seeds_report.jsonl"
SCOUT_SEEDS_REPORT_MARKDOWN = "native_scout_seeds_report.md"
PRIMARY = "workbook_holdout_pass@1"


def raw_difference(base: dict, adapted: dict, k: int) -> float | None:
    """Adapter minus base on pass@k, the unrounded mean over the problems both have: what the branch reads."""
    usable = sorted(i for i in base if base[i][0] >= k and i in adapted and adapted[i][0] >= k)
    if not usable:
        return None
    return sum(pass_at_k(*adapted[i], k) - pass_at_k(*base[i], k) for i in usable) / len(usable)


def raw_mean(counts: dict, k: int) -> float | None:
    usable = [value for value in counts.values() if value[0] >= k]
    return sum(pass_at_k(*value, k) for value in usable) / len(usable) if usable else None


def _between_arms(legacy: dict, native: dict, k: int, resamples: int, seed: int) -> dict | None:
    """Native minus legacy, paired over problems, with section 8's labels."""
    result = _comparison(legacy, [native], k, resamples, seed)
    if result is None:
        return None
    return {"problems": result["problems"], "legacy": result["base"], "native": result["adapter"], "delta": result["delta"],
            "interval": result["interval"], "label": result["label"]}


def _loss_rows(store: ArtifactStore, name: str) -> list[dict]:
    return store.read_rows(name) if store.path(name).exists() else []


def _rounded_parts(rows: list[dict]) -> dict | None:
    if not rows_have_parts(rows):
        return None
    parts = summarize_loss_rows(rows)
    return {"mean_loss": round(parts["mean_loss"], 4), "body_per_token": round(parts["body_per_token"], 4),
            "nats_per_proof": {part: round(value, 4) for part, value in parts["nats_per_proof"].items()},
            "contribution": {part: round(value, 4) for part, value in parts["contribution"].items()}}


def _heldout_loss(base_rows: list[dict], adapter_rows: list[dict], resamples: int, seed: int) -> dict | None:
    """Base against adapter on one stack and in one pair format: the paired comparison of the mean per-token
    loss, and each side by part where the stored rows hold the parts (rows written before the fix do not)."""
    if not base_rows or not adapter_rows:
        return None
    comparison = _loss_comparison({row["statement_id"]: row["loss"] for row in base_rows},
                                  [{row["statement_id"]: row["loss"] for row in adapter_rows}], resamples, seed)
    return {"comparison": comparison, "base_by_part": _rounded_parts(base_rows), "adapter_by_part": _rounded_parts(adapter_rows)}


def _completions(rows: list[dict], variant: str, ids: set[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        if row["variant"] == variant and row["statement_id"] in ids:
            grouped.setdefault(row["statement_id"], []).append(row["completion"])
    return grouped


def _status_shares(store: ArtifactStore, name: str) -> dict | None:
    """How the Lean checks of one evaluation ended: a timeout or a server error is a proof that was not judged."""
    if not store.path(name).exists():
        return None
    statuses = [row["status"] for row in store.read_rows(name)]
    return {"attempts": len(statuses), "timeout_share": round(statuses.count("timeout") / len(statuses), 5),
            "server_error_share": round(statuses.count("server_error") / len(statuses), 5)}


def _training(store: ArtifactStore, names: ArmNames) -> dict | None:
    if not store.is_done(names.train_marker):
        return None
    return {key: value for key, value in store.done_summary(names.train_marker).items() if key != "losses"}


def _between_arms_over_seeds(legacy_by_seed: list[dict], native_by_seed: list[dict], k: int, resamples: int, seed: int) -> dict | None:
    """Native minus legacy with a problem's score the MEAN of its per-seed estimates (spec §8 item 1), paired
    over the problems every seed of both arms has."""
    every = legacy_by_seed + native_by_seed
    usable = sorted(i for i in native_by_seed[0] if all(i in counts and counts[i][0] >= k for counts in every)) if every else []
    if not usable:
        return None

    def mean_over_seeds(by_seed: list[dict], problem: str) -> float:
        return sum(pass_at_k(*counts[problem], k) for counts in by_seed) / len(by_seed)

    legacy = [mean_over_seeds(legacy_by_seed, i) for i in usable]
    native = [mean_over_seeds(native_by_seed, i) for i in usable]
    interval = paired_bootstrap([a - b for a, b in zip(native, legacy)], resamples=resamples, seed=seed)
    return {"problems": len(usable), "seeds": len(native_by_seed), "legacy": round(sum(legacy) / len(usable), 4),
            "native": round(sum(native) / len(usable), 4), "delta": round(interval.mean, 4),
            "interval": [round(interval.low, 4), round(interval.high, 4)], "label": interval_label(interval)}


def _seeds_read(store: ArtifactStore, config: dict, profile: str, runs: list[ArmNames]) -> dict:
    """The same comparisons as the scout's, read over several seeds: a problem's score for an arm is the mean
    of its per-seed estimates, and the legacy control is Phase A's stored adapters at the same seeds."""
    evaluation = config["evaluation"]
    resamples, bootstrap_seed = evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"]
    k = config["profiles"][profile]["minif2f_samples"]
    reused = config["selection"]["phase_a_method"]
    seeds = [names.seed for names in runs]
    ids = _evaluation_ids(store)
    base_holdouts, base_minif2f = _base_counts(store)
    base = {"conjecture_holdout": _subset(base_holdouts, ids["conjecture_holdout"]),
            "workbook_holdout": _subset(base_holdouts, ids["workbook_holdout"]), "minif2f": _subset(base_minif2f, ids["minif2f"])}
    native = [_arm_counts(store, names) for names in runs]
    legacy = [_arm_counts(store, arm_names(reused, seed, reused)) for seed in seeds]
    if any(not counts for counts in legacy):
        raise RuntimeError(f"Phase A's stored evaluation is missing for one of the seeds {seeds}: the legacy control cannot be read")
    measures = ((PRIMARY, "workbook_holdout", 1), ("conjecture_holdout_pass@1", "conjecture_holdout", 1),
                ("minif2f_pass@1", "minif2f", 1), (f"minif2f_pass@{k}", "minif2f", k))
    results, per_seed = {}, {}
    for name, set_name, at in measures:
        native_sets = [_subset(counts, ids[set_name]) for counts in native]
        legacy_sets = [_subset(counts, ids[set_name]) for counts in legacy]
        results[name] = {"native_minus_base": _comparison(base[set_name], native_sets, at, resamples, bootstrap_seed),
                         "legacy_minus_base": _comparison(base[set_name], legacy_sets, at, resamples, bootstrap_seed),
                         "native_minus_legacy": _between_arms_over_seeds(legacy_sets, native_sets, at, resamples, bootstrap_seed)}
        per_seed[name] = {"native_minus_base": [round(raw_difference(base[set_name], counts, at), 4) for counts in native_sets],
                          "legacy_minus_base": [round(raw_difference(base[set_name], counts, at), 4) for counts in legacy_sets]}
    native_base = base_loss_names(runs[0].target_format)
    base_rows = _loss_rows(store, native_base.nf4_file)
    base_rows_fp8 = _loss_rows(store, native_base.fp8_file)
    heldout = {names.key: {"nf4": _heldout_loss(base_rows, _loss_rows(store, names.nf4_loss_file), resamples, bootstrap_seed),
                           "fp8": _heldout_loss(base_rows_fp8, _loss_rows(store, names.fp8_loss_file), resamples, bootstrap_seed)} for names in runs}
    holdout_ids = ids["conjecture_holdout"]
    distinct = {names.key: round(distinct_share(_completions(store.read_rows(names.eval_attempts_file), names.variant, holdout_ids)), 4) for names in runs}
    training = {names.key: {key: value for key, value in (_training(store, names) or {}).items() if key != "heldout_parts_curve"} for names in runs}
    curves = {names.key: (_training(store, names) or {}).get("heldout_parts_curve") for names in runs}
    return {"seeds": seeds, "results": results, "per_seed": per_seed, "heldout_loss": heldout,
            "distinct_attempts_conjecture_holdout": distinct,
            "lean_checks": {names.key: _status_shares(store, names.eval_verification_file) for names in runs},
            "training": training, "heldout_loss_by_part_during_training": curves}


def build_native_scout_report(store: ArtifactStore, config: dict, profile: str, runs: list[ArmNames]) -> dict:
    """With the scout's one run: the scout's report, its pre-registered branch included. With that run and
    more seeds: the same one-seed read, the read over all the seeds beside it, in files of their own."""
    scout = config["scout"]
    native_runs = sorted((names for names in runs if names.arm == SCOUT_ARM), key=lambda names: names.seed)
    if len(native_runs) != len(runs) or not native_runs or native_runs[0].seed != scout["seed"]:
        raise ValueError(f"the scout is {SCOUT_ARM} at seed {scout['seed']} (spec §13a), alone or with further seeds; got {[names.key for names in runs]}")
    for names in native_runs:
        if not (store.is_done(names.sampling_marker) and store.is_done(names.loss_marker)):
            raise RuntimeError(f"{names.key} is not evaluated yet: the scout cannot be read (a task that died reads VOID)")
    one_seed = _one_seed_report(store, config, profile, native_runs[0])
    if len(native_runs) == 1:
        _write(store, one_seed, SCOUT_REPORT_FILE, SCOUT_REPORT_MARKDOWN, _markdown(one_seed))
        return one_seed
    over_seeds = _seeds_read(store, config, profile, native_runs)
    primary = over_seeds["results"][PRIMARY]["native_minus_base"]
    control = over_seeds["results"][PRIMARY]["legacy_minus_base"]
    between = over_seeds["results"][PRIMARY]["native_minus_legacy"]
    headline = (f"{profile}, seeds {over_seeds['seeds']}: workbook-holdout pass@1, native minus base {primary['delta']:+.4f} "
                f"[{primary['interval'][0]:+.4f}, {primary['interval'][1]:+.4f}] {primary['label']} (base {primary['base']}, native {primary['adapter']}); "
                f"legacy {control['delta']:+.4f} [{control['interval'][0]:+.4f}, {control['interval'][1]:+.4f}] {control['label']}; "
                f"native minus legacy {between['delta']:+.4f} [{between['interval'][0]:+.4f}, {between['interval'][1]:+.4f}] {between['label']}. "
                f"One seed (the scout): {one_seed['headline']}")
    report = {"profile": profile, "arm": SCOUT_ARM, "seeds": over_seeds["seeds"], "headline": headline, "over_seeds": over_seeds, "one_seed": one_seed}
    _write(store, report, SCOUT_SEEDS_REPORT_FILE, SCOUT_SEEDS_REPORT_MARKDOWN, _seeds_markdown(report))
    return report


def _write(store: ArtifactStore, report: dict, file_name: str, markdown_name: str, markdown: str) -> None:
    store.write_rows(file_name, [report])
    store.path(markdown_name).write_text(markdown)
    if store.mirror_directory is not None:
        (store.mirror_directory / markdown_name).write_text(markdown)


def scout_scalars(report: dict) -> dict[str, float]:
    """The headline figures as TensorBoard scalars (read while a long task is still running)."""
    one_seed = report.get("one_seed", report)
    primary = one_seed["results"][PRIMARY]["native_minus_base"]
    values = {"scout/one_seed_primary_delta": primary["delta"], "scout/one_seed_primary_low": primary["interval"][0],
              "scout/one_seed_primary_high": primary["interval"][1], f"scout/one_seed_branch/{one_seed['branch']['name']}": 1.0}
    if "over_seeds" in report:
        for name, rows in report["over_seeds"]["results"].items():
            for comparison in ("native_minus_base", "legacy_minus_base", "native_minus_legacy"):
                result = rows[comparison]
                if result is not None:
                    values[f"scout/seeds_{name}/{comparison}_delta"] = result["delta"]
                    values[f"scout/seeds_{name}/{comparison}_low"] = result["interval"][0]
                    values[f"scout/seeds_{name}/{comparison}_high"] = result["interval"][1]
        for index, delta in enumerate(report["over_seeds"]["per_seed"][PRIMARY]["native_minus_base"]):
            values[f"scout/primary_native_minus_base_seed{report['over_seeds']['seeds'][index]}"] = delta
        for key, checks in report["over_seeds"]["lean_checks"].items():
            if checks:
                values[f"scout/timeout_share/{key}"] = checks["timeout_share"]
                values[f"scout/server_error_share/{key}"] = checks["server_error_share"]
    return values


def _one_seed_report(store: ArtifactStore, config: dict, profile: str, native: ArmNames) -> dict:
    evaluation, scout = config["evaluation"], config["scout"]
    resamples, bootstrap_seed = evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"]
    k = config["profiles"][profile]["minif2f_samples"]
    legacy = arm_names(config["selection"]["phase_a_method"], native.seed, config["selection"]["phase_a_method"])
    ids = _evaluation_ids(store)
    base_holdouts, base_minif2f = _base_counts(store)
    base = {"conjecture_holdout": _subset(base_holdouts, ids["conjecture_holdout"]),
            "workbook_holdout": _subset(base_holdouts, ids["workbook_holdout"]), "minif2f": _subset(base_minif2f, ids["minif2f"])}
    native_counts, legacy_counts = _arm_counts(store, native), _arm_counts(store, legacy)
    measures = ((PRIMARY, "workbook_holdout", 1), ("conjecture_holdout_pass@1", "conjecture_holdout", 1),
                ("minif2f_pass@1", "minif2f", 1), (f"minif2f_pass@{k}", "minif2f", k))
    results = {}
    for name, set_name, at in measures:
        native_set, legacy_set = _subset(native_counts, ids[set_name]), _subset(legacy_counts, ids[set_name])
        results[name] = {"native_minus_base": _comparison(base[set_name], [native_set], at, resamples, bootstrap_seed),
                         "legacy_minus_base": _comparison(base[set_name], [legacy_set], at, resamples, bootstrap_seed),
                         "native_minus_legacy": _between_arms(legacy_set, native_set, at, resamples, bootstrap_seed)}

    # The control (spec §13a): the stored base and legacy seed-0 figures must read what the pre-registration quotes.
    workbook_legacy = _subset(legacy_counts, ids["workbook_holdout"])
    measured = {"base": raw_mean(base["workbook_holdout"], 1), "legacy": raw_mean(workbook_legacy, 1)}
    expected = {"base": scout["control_base_workbook_pass_at_1"], "legacy": scout["control_legacy_workbook_pass_at_1"]}
    reproduced = all(measured[key] is not None and control_reproduces(measured[key], expected[key]) for key in expected)
    difference = raw_difference(base["workbook_holdout"], _subset(native_counts, ids["workbook_holdout"]), 1)
    if difference is None:
        raise RuntimeError("the native adapter has no workbook-holdout samples: the primary cannot be read")
    branch = scout_branch(difference, reproduced, scout["token_habit_below"])

    native_base = base_loss_names(native.target_format)
    legacy_base = base_loss_names(LEGACY)
    heldout = {
        "nf4": {"native": _heldout_loss(_loss_rows(store, native_base.nf4_file), _loss_rows(store, native.nf4_loss_file), resamples, bootstrap_seed),
                "legacy": _heldout_loss(_loss_rows(store, legacy_base.nf4_file), _loss_rows(store, legacy.nf4_loss_file), resamples, bootstrap_seed)},
        "fp8": {"native": _heldout_loss(_loss_rows(store, native_base.fp8_file), _loss_rows(store, native.fp8_loss_file), resamples, bootstrap_seed),
                "legacy": _heldout_loss(_loss_rows(store, legacy_base.fp8_file), _loss_rows(store, legacy.fp8_loss_file), resamples, bootstrap_seed)},
    }
    native_training, legacy_training = _training(store, native), _training(store, legacy)

    native_attempts = store.read_rows(native.eval_attempts_file)
    legacy_attempts = store.read_rows(legacy.eval_attempts_file) if store.path(legacy.eval_attempts_file).exists() else []
    holdout_ids = ids["conjecture_holdout"]
    distinct = {"base": distinct_share(_completions(store.read_rows("proof_attempts.jsonl"), "base", holdout_ids)),
                "legacy": distinct_share(_completions(legacy_attempts, legacy.variant, holdout_ids)) if legacy_attempts else None,
                "native": distinct_share(_completions(native_attempts, native.variant, holdout_ids))}
    distinct = {key: round(value, 4) if value is not None else None for key, value in distinct.items()}

    sequence_start = None
    token_id = (native_training or {}).get("sequence_start_token_id")
    if token_id is not None and native_attempts and all("first_token_id" in row for row in native_attempts):
        by_set = {}
        for set_name, members in ids.items():
            firsts = [row["first_token_id"] for row in native_attempts if row["statement_id"] in members]
            if firsts:
                by_set[set_name] = round(share_starting_with(firsts, token_id), 5)
        sequence_start = {"token_id": token_id, "attempts": len(native_attempts),
                          "native_share": round(share_starting_with([row["first_token_id"] for row in native_attempts], token_id), 5),
                          "native_share_by_set": by_set,
                          "legacy_and_base": "not recorded in the stored attempts; measured from their token counts in "
                                             "the held-out saturation and learning-progress score diagnosis (T2): the base writes it in "
                                             "99.98% of 34,000 attempts, the legacy seed-0 adapter in 0.01% of 19,464"}

    primary = results[PRIMARY]["native_minus_base"]
    control = results[PRIMARY]["legacy_minus_base"]
    headline = (f"{profile}, seed {native.seed}: workbook-holdout pass@1, native minus base {primary['delta']:+.4f} "
                f"[{primary['interval'][0]:+.4f}, {primary['interval'][1]:+.4f}] {primary['label']} "
                f"(base {primary['base']}, native {primary['adapter']}; legacy control {control['adapter'] if control else 'missing'}, "
                f"{'reproduced' if reproduced else 'NOT REPRODUCED'}) -> {branch.name}: {branch.reason}")
    report = {"profile": profile, "arm": native.arm, "seed": native.seed, "headline": headline,
              "branch": {"name": branch.name, "reason": branch.reason, "action": branch.action,
                         "difference": difference, "token_habit_below": scout["token_habit_below"]},
              "control": {"measured": measured, "expected": expected, "reproduced": reproduced},
              "results": results, "heldout_loss": heldout,
              "heldout_loss_by_part_during_training": (native_training or {}).get("heldout_parts_curve"),
              "distinct_attempts_conjecture_holdout": distinct, "sequence_start_token_when_sampling": sequence_start,
              "lean_checks": {"native": _status_shares(store, native.eval_verification_file),
                              "legacy": _status_shares(store, legacy.eval_verification_file)},
              "training": {"native": {key: value for key, value in (native_training or {}).items() if key != "heldout_parts_curve"},
                           "legacy": legacy_training}}
    return report


def _cell(result: dict | None, first: str, second: str) -> str:
    if result is None:
        return "| | | | | not measurable |"
    return (f"| {result[first]} | {result[second]} | {result['delta']:+.4f} | "
            f"[{result['interval'][0]:+.4f}, {result['interval'][1]:+.4f}] | {result['label']} |")


def _seeds_markdown(report: dict) -> str:
    over = report["over_seeds"]
    lines = [f"# rlvr_lean native format over seeds {over['seeds']} (the escalation of spec 13a's scout) — {report['profile']} run", "",
             f"**{report['headline']}**", "",
             "A problem's score for an arm is the mean of its per-seed estimates; the legacy control is Phase A's stored adapters at the same seeds.", "",
             "| Measure | Comparison | First | Second | Difference | 95% interval | Label |", "|---|---|---|---|---|---|---|"]
    for name, rows in over["results"].items():
        lines.append(f"| {name} | native − base (base, native) " + _cell(rows["native_minus_base"], "base", "adapter"))
        lines.append(f"| {name} | legacy − base (base, legacy) " + _cell(rows["legacy_minus_base"], "base", "adapter"))
        lines.append(f"| {name} | native − legacy (legacy, native) " + _cell(rows["native_minus_legacy"], "legacy", "native"))
    lines += ["", f"Per seed, each adapter minus base: `{json.dumps(over['per_seed'])}`", "",
              f"Held-out loss per run: `{json.dumps(over['heldout_loss'])}`", "",
              f"Distinct attempts on the conjecture holdout: `{json.dumps(over['distinct_attempts_conjecture_holdout'])}`", "",
              f"Lean checks: `{json.dumps(over['lean_checks'])}`", "", f"Training: `{json.dumps(over['training'])}`", "",
              "---", "", _markdown(report["one_seed"])]
    return "\n".join(lines) + "\n"


def _markdown(report: dict) -> str:
    branch = report["branch"]
    lines = [f"# rlvr_lean native-format scout (spec 13a) — {report['profile']} run, seed {report['seed']}", "",
             f"**{report['headline']}**", "",
             f"Branch: **{branch['name']}** ({branch['reason']}). Next: {branch['action']}. "
             f"Unrounded difference {branch['difference']:+.6f}; the token-habit line is {branch['token_habit_below']}.", "",
             f"Control: `{json.dumps(report['control'])}`", "",
             "## pass@k: each adapter minus base, and native minus legacy", "",
             "| Measure | Comparison | First | Second | Difference | 95% interval | Label |", "|---|---|---|---|---|---|---|"]
    for name, rows in report["results"].items():
        lines.append(f"| {name} | native − base (base, native) " + _cell(rows["native_minus_base"], "base", "adapter"))
        lines.append(f"| {name} | legacy − base (base, legacy) " + _cell(rows["legacy_minus_base"], "base", "adapter"))
        lines.append(f"| {name} | native − legacy (legacy, native) " + _cell(rows["native_minus_legacy"], "legacy", "native"))
    lines += ["", "## Held-out loss (each adapter against the base in its own pair format)", ""]
    for stack, by_format in report["heldout_loss"].items():
        for name, result in by_format.items():
            lines.append(f"- **{stack}, {name}**: `{json.dumps(result)}`")
    lines += ["", "## Held-out loss by part during training (native)", "",
              "| Step | Mean per-token loss | Body nats per token | First token nats | Newline nats | Fence nats |", "|---|---|---|---|---|---|"]
    for point in report["heldout_loss_by_part_during_training"] or []:
        nats = point["nats_per_proof"]
        lines.append(f"| {point['step']} | {point['mean_loss']} | {point['body_per_token']} | {nats['first']} | {nats['newline']} | {nats['fence']} |")
    lines += ["", "## Also read", "",
              f"- Distinct attempts on the conjecture holdout: `{json.dumps(report['distinct_attempts_conjecture_holdout'])}`",
              f"- Sequence-start token when sampling: `{json.dumps(report['sequence_start_token_when_sampling'])}`",
              f"- Lean checks: `{json.dumps(report['lean_checks'])}`", "", "## Training", ""]
    lines += [f"- **{name}**: `{json.dumps(summary)}`" for name, summary in report["training"].items()]
    return "\n".join(lines) + "\n"
