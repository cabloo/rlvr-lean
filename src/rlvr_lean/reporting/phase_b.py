"""Phase B report (spec §13) and the `eval/` values the TensorBoard runs draw. Pure arithmetic over the run's
JSONL artifacts.

Every arm-to-arm comparison is learning progress MINUS the other arm, paired over problems; a problem's score
for an arm is the mean of its per-seed pass@k estimates, taken before differencing. Only seeds evaluated for
every arm enter a comparison, so the arms are always compared at the same n.
"""

from __future__ import annotations

import json

from rlvr_lean.domain.evaluation.bootstrap import interval_label, paired_bootstrap, phase_b_label
from rlvr_lean.domain.evaluation.pass_at_k import pass_at_k
from rlvr_lean.domain.training.arms import ArmNames, base_loss_names
from rlvr_lean.domain.training.target_format import LEGACY
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.reporting.phase_a import _comparison, _counts, _loss_comparison, _losses

LP_LABELS = {"improved": "LP better", "degraded": "LP worse", "no measurable change": "no measurable difference"}


def _evaluation_ids(store: ArtifactStore) -> dict[str, set[str]]:
    return {"conjecture_holdout": {c["conjecture_id"] for c in store.read_rows("conjectures.jsonl") if c["holdout"]},
            "workbook_holdout": {r["statement_id"] for r in store.read_rows("statements_reward.jsonl") if r["half"] == "validation"},
            "minif2f": {r["statement_id"] for r in store.read_rows("minif2f_eval.jsonl")}}


def _base_counts(store: ArtifactStore) -> tuple[dict, dict]:
    """The base's holdout counts (the samples drawn in proof sampling) and its miniF2F counts (`eval_base`)."""
    attempts = store.read_rows("proof_attempts.jsonl") + store.read_rows("eval_attempts_base.jsonl")
    verification = {row["attempt_id"]: row["status"]
                    for row in store.read_rows("verification.jsonl") + store.read_rows("eval_verification_base.jsonl")}
    return _counts(attempts, verification, "base"), _counts(attempts, verification, "eval_base")


def _arm_counts(store: ArtifactStore, names: ArmNames) -> dict:
    if not store.path(names.eval_attempts_file).exists():
        return {}
    verification = {row["attempt_id"]: row["status"] for row in store.read_rows(names.eval_verification_file)}
    return _counts(store.read_rows(names.eval_attempts_file), verification, names.variant)


def _subset(counts: dict, ids: set[str]) -> dict:
    return {key: value for key, value in counts.items() if key in ids}


def evaluation_scalars(store: ArtifactStore, config: dict, profile: str, names: ArmNames | None,
                       base_format: str = LEGACY) -> dict[str, float]:
    """Spec §13's `eval/` tags for the base (`names` None) or one arm and seed. Means over the problems with
    enough samples; a measure whose inputs do not exist yet is left out. The base's held-out loss is the one
    measured in `base_format`, the pair format of the arm it is drawn beside (spec §6 item 2a)."""
    k = config["profiles"][profile]["minif2f_samples"]
    ids = _evaluation_ids(store)
    if names is None:
        holdout_counts, minif2f_counts = _base_counts(store)
        base_names = base_loss_names(base_format)
        loss_files = (base_names.fp8_file, base_names.nf4_file)
    else:
        holdout_counts = minif2f_counts = _arm_counts(store, names)
        loss_files = (names.fp8_loss_file, names.nf4_loss_file)
    values = {}
    for tag, counts, set_name, at in (("eval/pass@1_conjecture_holdout", holdout_counts, "conjecture_holdout", 1),
                                      ("eval/pass@1_workbook_holdout", holdout_counts, "workbook_holdout", 1),
                                      ("eval/pass@1_minif2f", minif2f_counts, "minif2f", 1),
                                      (f"eval/pass@{k}_minif2f", minif2f_counts, "minif2f", k)):
        usable = [counts[i] for i in ids[set_name] if i in counts and counts[i][0] >= at]
        if usable:
            values[tag] = sum(pass_at_k(n, c, at) for n, c in usable) / len(usable)
    for tag, name in zip(("eval/heldout_loss_fp8", "eval/heldout_loss_nf4"), loss_files):
        losses = _losses(store, name)
        if losses:
            values[tag] = sum(losses.values()) / len(losses)
    return values


def arm_difference(learning_progress_by_seed: list[dict], other_by_seed: list[dict], k: int, resamples: int, seed: int,
                   prompt_rule: bool = False) -> dict | None:
    """Learning progress minus another arm on pass@k, paired over the problems both have at every seed."""
    every = learning_progress_by_seed + other_by_seed
    if not learning_progress_by_seed or not other_by_seed:
        return None
    usable = sorted(i for i in learning_progress_by_seed[0] if all(i in counts and counts[i][0] >= k for counts in every))
    if not usable:
        return None

    def mean_over_seeds(by_seed: list[dict], problem: str) -> float:
        return sum(pass_at_k(*counts[problem], k) for counts in by_seed) / len(by_seed)

    first = [mean_over_seeds(learning_progress_by_seed, i) for i in usable]
    second = [mean_over_seeds(other_by_seed, i) for i in usable]
    interval = paired_bootstrap([a - b for a, b in zip(first, second)], resamples=resamples, seed=seed)
    label = phase_b_label(interval) if prompt_rule else LP_LABELS[interval_label(interval)]
    return {"problems": len(usable), "seeds": len(learning_progress_by_seed), "learning_progress": round(sum(first) / len(usable), 4),
            "other": round(sum(second) / len(usable), 4), "delta": round(interval.mean, 4),
            "interval": [round(interval.low, 4), round(interval.high, 4)], "label": label}


def loss_difference(base: dict[str, float], learning_progress_by_seed: list[dict[str, float]], other_by_seed: list[dict[str, float]],
                    resamples: int, seed: int) -> dict | None:
    """The scoring check (spec §8, §13): LP's held-out-loss reduction minus the other arm's. Per pair that is
    (base − LP) − (base − other) = other − LP, so a POSITIVE value means LP lowered the loss more."""
    every = learning_progress_by_seed + other_by_seed
    shared = sorted(i for i in base if every and all(i in losses for losses in every))
    if not shared or not learning_progress_by_seed or not other_by_seed:
        return None

    def mean_over_seeds(by_seed: list[dict[str, float]], pair: str) -> float:
        return sum(losses[pair] for losses in by_seed) / len(by_seed)

    differences = [mean_over_seeds(other_by_seed, i) - mean_over_seeds(learning_progress_by_seed, i) for i in shared]
    interval = paired_bootstrap(differences, resamples=resamples, seed=seed)
    return {"pairs": len(shared), "seeds": len(learning_progress_by_seed),
            "learning_progress_reduction": round(sum(base[i] - mean_over_seeds(learning_progress_by_seed, i) for i in shared) / len(shared), 4),
            "other_reduction": round(sum(base[i] - mean_over_seeds(other_by_seed, i) for i in shared) / len(shared), 4),
            "delta": round(interval.mean, 4), "interval": [round(interval.low, 4), round(interval.high, 4)],
            "label": LP_LABELS[interval_label(interval)]}


def _selection_profiles(store: ArtifactStore, arms: list[str], reused_arm: str) -> tuple[dict, dict]:
    """Per arm: the training set's mean base pass rate and canonical proof length; pairwise Jaccard overlaps."""
    pool = {row["conjecture_id"]: row for row in store.read_rows("pool.jsonl")}
    chosen = {row["method"]: row["conjecture_ids"] for row in store.read_rows("selections.jsonl")}
    for arm in arms:
        if store.path(f"selection_{arm}.jsonl").exists():
            chosen[arm] = store.read_rows(f"selection_{arm}.jsonl")[0]["conjecture_ids"]
    chosen[reused_arm] = [row["conjecture_id"] for row in store.read_rows("training_examples.jsonl")]
    profiles = {}
    for arm in arms:
        rows = [pool[i] for i in chosen.get(arm, [])]
        if rows:
            profiles[arm] = {"size": len(rows), "mean_base_pass_rate": round(sum(r["verified_count"] / r["sample_count"] for r in rows) / len(rows), 3),
                             "mean_proof_tokens": round(sum(r["canonical_proof_tokens"] for r in rows) / len(rows), 1)}
    overlaps = {}
    for index, first in enumerate(arms):
        for second in arms[index + 1:]:
            a, b = set(chosen.get(first, [])), set(chosen.get(second, []))
            if a | b:
                overlaps[f"{first}~{second}"] = round(len(a & b) / len(a | b), 3)
    return profiles, overlaps


def build_phase_b_report(store: ArtifactStore, config: dict, profile: str, runs: list[ArmNames]) -> dict:
    evaluation = config["evaluation"]
    resamples, bootstrap_seed = evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"]
    reused_arm = config["selection"]["phase_a_method"]
    k = config["profiles"][profile]["minif2f_samples"]
    arms = list(dict.fromkeys(names.arm for names in runs))
    others = [arm for arm in arms if arm != reused_arm]
    evaluated = {(names.arm, names.seed): names for names in runs
                 if store.is_done(names.sampling_marker) and store.is_done(names.loss_marker)}
    seeds = sorted({seed for _, seed in evaluated if all((arm, seed) in evaluated for arm in arms)})
    ids = _evaluation_ids(store)
    base_holdouts, base_minif2f = _base_counts(store)
    counts = {key: _arm_counts(store, names) for key, names in evaluated.items()}
    nf4 = {key: _losses(store, names.nf4_loss_file) for key, names in evaluated.items()}
    fp8 = {key: _losses(store, names.fp8_loss_file) for key, names in evaluated.items()}
    nf4_base, fp8_base = _losses(store, "heldout_loss_nf4_base.jsonl"), _losses(store, "heldout_loss_fp8_base.jsonl")
    measures = (("workbook_holdout_pass@1", "workbook_holdout", 1), ("conjecture_holdout_pass@1", "conjecture_holdout", 1),
                ("minif2f_pass@1", "minif2f", 1), (f"minif2f_pass@{k}", "minif2f", k))

    def by_seed(arm: str, set_name: str, chosen: list[int]) -> list[dict]:
        return [_subset(counts[(arm, seed)], ids[set_name]) for seed in chosen]

    comparisons = {}
    for other in others:
        rows = {name: arm_difference(by_seed(reused_arm, set_name, seeds), by_seed(other, set_name, seeds), at, resamples, bootstrap_seed)
                for name, set_name, at in measures}
        rows["heldout_loss_nf4_reduction"] = loss_difference(nf4_base, [nf4[(reused_arm, s)] for s in seeds], [nf4[(other, s)] for s in seeds], resamples, bootstrap_seed)
        rows["heldout_loss_fp8_reduction"] = loss_difference(fp8_base, [fp8[(reused_arm, s)] for s in seeds], [fp8[(other, s)] for s in seeds], resamples, bootstrap_seed)
        comparisons[other] = rows
    prompt_rule = None
    if "difficulty_heuristic" in others:
        prompt_rule = arm_difference(by_seed(reused_arm, "minif2f", seeds), by_seed("difficulty_heuristic", "minif2f", seeds), k,
                                     resamples, bootstrap_seed, prompt_rule=True)
    against_base = {}
    for arm in arms:
        base_sets = {"conjecture_holdout": _subset(base_holdouts, ids["conjecture_holdout"]),
                     "workbook_holdout": _subset(base_holdouts, ids["workbook_holdout"]), "minif2f": _subset(base_minif2f, ids["minif2f"])}
        rows = {name: _comparison(base_sets[set_name], by_seed(arm, set_name, seeds), at, resamples, bootstrap_seed) for name, set_name, at in measures}
        rows["heldout_loss_nf4"] = _loss_comparison(nf4_base, [nf4[(arm, s)] for s in seeds], resamples, bootstrap_seed)
        rows["heldout_loss_fp8"] = _loss_comparison(fp8_base, [fp8[(arm, s)] for s in seeds], resamples, bootstrap_seed)
        against_base[arm] = rows
    # Seed rule (spec §13): an arm whose seed reads DEGRADED against base on conjecture-holdout pass@1 stops.
    stops = []
    for (arm, seed) in sorted(evaluated):
        if arm == reused_arm:
            continue
        read = _comparison(_subset(base_holdouts, ids["conjecture_holdout"]), by_seed(arm, "conjecture_holdout", [seed]), 1, resamples, bootstrap_seed)
        if read and read["label"] == "degraded":
            stops.append({"arm": arm, "seed": seed, "delta": read["delta"], "interval": read["interval"]})
    profiles, overlaps = _selection_profiles(store, arms, reused_arm)
    training = {names.key: {key: value for key, value in store.done_summary(names.train_marker).items() if key != "losses"}
                for names in runs if store.is_done(names.train_marker)}
    deciding = {other: comparisons[other]["workbook_holdout_pass@1"] for other in others}
    verdict_note = "the verdict" if len(seeds) >= 3 else f"a {len(seeds)}-seed read; the verdict is read at three seeds"
    headline = (f"{profile}, seeds {seeds} ({verdict_note}): workbook-holdout pass@1, LP minus "
                + "; ".join(f"{other} {row['delta']:+.4f} [{row['interval'][0]:+.4f}, {row['interval'][1]:+.4f}] {row['label']}"
                            if row else f"{other} not measurable" for other, row in deciding.items()))
    if stops:
        headline += "; STOP for diagnosis: " + ", ".join(f"{s['arm']} seed {s['seed']} degraded" for s in stops)
    report = {"profile": profile, "seeds": seeds, "arms": arms, "headline": headline, "deciding": deciding,
              "comparisons": comparisons, "prompt_rule_pass@k_lp_minus_heuristic": prompt_rule, "against_base": against_base,
              "stops": stops, "selection_profiles": profiles, "overlaps": overlaps, "training": training}
    store.write_rows("phase_b_report.jsonl", [report])
    markdown = _markdown(report)
    store.path("phase_b_report.md").write_text(markdown)
    if store.mirror_directory is not None:
        (store.mirror_directory / "phase_b_report.md").write_text(markdown)
    return report


def _row(name: str, result: dict | None, first: str, second: str) -> str:
    if result is None:
        return f"| {name} | | | | | | not measurable |"
    size = result.get("problems", result.get("pairs"))
    return (f"| {name} | {size} | {result[first]} | {result[second]} | {result['delta']:+.4f} | "
            f"[{result['interval'][0]:+.4f}, {result['interval'][1]:+.4f}] | {result['label']} |")


def _markdown(report: dict) -> str:
    lines = [f"# rlvr_lean Phase B — {report['profile']} run", "", f"**{report['headline']}**", ""]
    for other, rows in report["comparisons"].items():
        lines += [f"## Learning progress minus {other}", "", "| Measure | Problems | LP | Other | Difference | 95% interval | Label |",
                  "|---|---|---|---|---|---|---|"]
        for name, result in rows.items():
            if name.startswith("heldout_loss"):
                lines.append(_row(name, result, "learning_progress_reduction", "other_reduction"))
            else:
                lines.append(_row(name, result, "learning_progress", "other"))
        lines.append("")
    lines += ["Loss rows compare loss REDUCTIONS against base: positive means LP lowered the held-out loss more.", ""]
    rule = report["prompt_rule_pass@k_lp_minus_heuristic"]
    lines += ["## The prompt's rule (miniF2F-test pass@k, LP minus heuristic)", "",
              f"`{json.dumps(rule)}`" if rule else "not measurable", ""]
    lines += ["## Every arm against base", "", "| Arm | Measure | Problems | Base | Adapter | Difference | 95% interval | Label |",
              "|---|---|---|---|---|---|---|---|"]
    for arm, rows in report["against_base"].items():
        for name, result in rows.items():
            if result is None:
                lines.append(f"| {arm} | {name} | | | | | | not measurable |")
                continue
            difference = result.get("delta", result.get("loss_reduction"))
            lines.append(f"| {arm} | {name} | {result.get('problems', result.get('pairs'))} | {result['base']} | {result['adapter']} | "
                         f"{difference:+.4f} | [{result['interval'][0]:+.4f}, {result['interval'][1]:+.4f}] | {result['label']} |")
    lines += ["", "## Selections", "", f"Profiles: `{json.dumps(report['selection_profiles'])}`", "",
              f"Overlaps (Jaccard): `{json.dumps(report['overlaps'])}`", "", "## Training", ""]
    lines += [f"- **{key}**: `{json.dumps(summary)}`" for key, summary in report["training"].items()]
    if report["stops"]:
        lines += ["", "## Stopped for diagnosis", "", f"`{json.dumps(report['stops'])}`"]
    return "\n".join(lines) + "\n"
