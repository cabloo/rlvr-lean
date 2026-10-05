"""Phase A report (spec §8 decision rules, §12 contents). Pure arithmetic over the run's JSONL artifacts.

Every comparison is paired (same problems, base vs trained adapter) and reported as the mean difference
with a percentile bootstrap interval over problems. Labels follow the pre-registered Phase A rule:
improved / degraded / no measurable change.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from rlvr_lean.domain.evaluation.bootstrap import interval_label, paired_bootstrap
from rlvr_lean.domain.evaluation.pass_at_k import pass_at_k
from rlvr_lean.infrastructure.artifact_store import ArtifactStore


def _counts(attempts: list[dict], verification: dict[str, str], variant: str) -> dict[str, tuple[int, int]]:
    """statement id -> (samples drawn, verified) for one variant."""
    counts: dict[str, list[int]] = {}
    for row in attempts:
        if row["variant"] == variant:
            entry = counts.setdefault(row["statement_id"], [0, 0])
            entry[0] += 1
            entry[1] += verification.get(row["attempt_id"]) == "verified"
    return {key: (value[0], value[1]) for key, value in counts.items()}


def _comparison(base: dict, adapted_by_seed: list[dict], k: int, resamples: int, seed: int) -> dict | None:
    """Paired over problems. With several training seeds, a problem's adapter score is the MEAN of its
    per-seed pass@k estimates (spec §8 item 1); the base is the same for every seed."""
    usable = sorted(i for i in base if base[i][0] >= k and all(i in seeds and seeds[i][0] >= k for seeds in adapted_by_seed))
    if not usable or not adapted_by_seed:
        return None
    base_values = [pass_at_k(*base[i], k) for i in usable]
    adapted_values = [sum(pass_at_k(*seeds[i], k) for seeds in adapted_by_seed) / len(adapted_by_seed) for i in usable]
    interval = paired_bootstrap([a - b for a, b in zip(adapted_values, base_values)], resamples=resamples, seed=seed)
    return {"problems": len(usable), "seeds": len(adapted_by_seed), "base": round(sum(base_values) / len(usable), 4),
            "adapter": round(sum(adapted_values) / len(usable), 4), "delta": round(interval.mean, 4),
            "interval": [round(interval.low, 4), round(interval.high, 4)], "label": interval_label(interval)}


def _loss_comparison(base: dict[str, float], adapted_by_seed: list[dict[str, float]], resamples: int, seed: int) -> dict | None:
    shared = sorted(i for i in base if all(i in seeds for seeds in adapted_by_seed))
    if not shared or not adapted_by_seed:
        return None
    adapted = {i: sum(seeds[i] for seeds in adapted_by_seed) / len(adapted_by_seed) for i in shared}
    # base - adapter, so a POSITIVE difference is a LOWER loss under the adapter: "improved" keeps its meaning.
    interval = paired_bootstrap([base[i] - adapted[i] for i in shared], resamples=resamples, seed=seed)
    return {"pairs": len(shared), "seeds": len(adapted_by_seed), "base": round(sum(base[i] for i in shared) / len(shared), 4),
            "adapter": round(sum(adapted.values()) / len(shared), 4), "loss_reduction": round(interval.mean, 4),
            "interval": [round(interval.low, 4), round(interval.high, 4)], "label": interval_label(interval)}


def _evaluated_seeds(store: ArtifactStore) -> list[int]:
    return sorted(int(path.name[len("evaluate_sampling_seed"):-len(".done.json")])
                  for path in store.root.glob("evaluate_sampling_seed*.done.json"))


def _losses(store: ArtifactStore, name: str) -> dict[str, float]:
    return {row["statement_id"]: row["loss"] for row in store.read_rows(name)} if store.path(name).exists() else {}


def _estimate_full_run(store: ArtifactStore, config: dict, timings: dict[str, float]) -> dict:
    """Scale each measured smoke rate to the full profile's sizes. Rough by construction: it assumes the
    full run's proofs look like the smoke run's (length, timeout share) and that sampling and verification
    overlap, so each sampling stage costs max(sampling, verification)."""
    full, sampling = config["profiles"]["full"], store.done_summary("sample_proofs")
    conjecturing = store.done_summary("generate_conjectures")
    attempts = sampling["conjecture_sampling"]["samples"] + sampling["reward_sampling"]["samples"]
    tokens = sampling["conjecture_sampling"]["generated_tokens"] + sampling["reward_sampling"]["generated_tokens"]
    tokens_per_sample = tokens / max(1, attempts)
    tokens_per_second = (sampling["conjecture_sampling"]["tokens_per_second"] or 1)
    proofs_per_second = sampling.get("verification_proofs_per_second") or 1
    full_attempts = full["conjecture_target"] * config["proving"]["samples_per_conjecture"] \
        + full["reward_statements"] * config["proving"]["samples_per_reward_statement"]
    full_eval = (full["conjecture_target"] * config["conjecturing"]["holdout_fraction"] * config["evaluation"]["conjecture_holdout_samples"]
                 + full["reward_statements"] / 2 * config["evaluation"]["workbook_holdout_samples"]
                 + 2 * full["minif2f_problems"] * full["minif2f_samples"])

    def stage_hours(samples: float) -> float:
        return max(samples * tokens_per_sample / tokens_per_second, samples / proofs_per_second) / 3600

    conjecture_rate = (timings.get("generate_conjectures", 0) / max(1, conjecturing["prompts"]))
    return {
        "assumptions": {"tokens_per_sample": round(tokens_per_sample, 1), "tokens_per_second": tokens_per_second,
                        "verification_proofs_per_second": proofs_per_second},
        "conjecture_generation_hours": round(conjecture_rate * full["seed_statements"] / 3600, 2),
        "proof_sampling_hours": round(stage_hours(full_attempts), 2),
        "evaluation_hours_per_seed": round(stage_hours(full_eval), 2),
        "scoring_and_training_hours": round((timings.get("score_selection", 0) + timings.get("train_adapter", 0))
                                            * full["conjecture_target"] / max(1, store.done_summary("score_selection")["pool"]) / 3600, 2),
    }


def build_report(store: ArtifactStore, config: dict, profile: str) -> dict:
    evaluation = config["evaluation"]
    resamples, bootstrap_seed = evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"]
    seeds = _evaluated_seeds(store)
    attempts = store.read_rows("proof_attempts.jsonl") + store.read_rows("eval_attempts_base.jsonl")
    verification_rows = store.read_rows("verification.jsonl") + store.read_rows("eval_verification_base.jsonl")
    for seed in seeds:
        attempts += store.read_rows(f"eval_attempts_seed{seed}.jsonl")
        verification_rows += store.read_rows(f"eval_verification_seed{seed}.jsonl")
    verification = {row["attempt_id"]: row["status"] for row in verification_rows}
    holdout_ids = {c["conjecture_id"] for c in store.read_rows("conjectures.jsonl") if c["holdout"]}
    workbook_ids = {r["statement_id"] for r in store.read_rows("statements_reward.jsonl") if r["half"] == "validation"}
    minif2f_ids = {r["statement_id"] for r in store.read_rows("minif2f_eval.jsonl")}
    samples = config["profiles"][profile]["minif2f_samples"]

    def subset(counts: dict, ids: set) -> dict:
        return {key: value for key, value in counts.items() if key in ids}

    base_samples = _counts(attempts, verification, "base")
    minif2f_base = subset(_counts(attempts, verification, "eval_base"), minif2f_ids)
    adapter_by_seed = {seed: _counts(attempts, verification, f"eval_adapter_s{seed}") for seed in seeds}
    nf4_base, fp8_base = _losses(store, "heldout_loss_nf4_base.jsonl"), _losses(store, "heldout_loss_fp8_base.jsonl")
    nf4_by_seed = {seed: _losses(store, f"heldout_loss_nf4_seed{seed}.jsonl") for seed in seeds}
    fp8_by_seed = {seed: _losses(store, f"heldout_loss_fp8_seed{seed}.jsonl") for seed in seeds}

    def measures(chosen: list[int]) -> dict:
        adapted = [adapter_by_seed[seed] for seed in chosen]
        return {
            "improvement_conjecture_holdout_pass@1": _comparison(subset(base_samples, holdout_ids), [subset(a, holdout_ids) for a in adapted], 1, resamples, bootstrap_seed),
            "transfer_workbook_holdout_pass@1": _comparison(subset(base_samples, workbook_ids), [subset(a, workbook_ids) for a in adapted], 1, resamples, bootstrap_seed),
            f"transfer_minif2f_pass@{samples}": _comparison(minif2f_base, [subset(a, minif2f_ids) for a in adapted], samples, resamples, bootstrap_seed),
            "transfer_minif2f_pass@1": _comparison(minif2f_base, [subset(a, minif2f_ids) for a in adapted], 1, resamples, bootstrap_seed),
            "heldout_loss_nf4": _loss_comparison(nf4_base, [nf4_by_seed[seed] for seed in chosen], resamples, bootstrap_seed),
            "heldout_loss_fp8": _loss_comparison(fp8_base, [fp8_by_seed[seed] for seed in chosen], resamples, bootstrap_seed),
        }

    results = measures(seeds)
    per_seed = {seed: {name: (value["label"], value["delta"] if "delta" in value else value["loss_reduction"]) if value else None
                       for name, value in measures([seed]).items()} for seed in seeds}
    step_dir = os.environ.get("RLVR_LEAN_STEP_DIR")
    partial = Path(step_dir).parent / "phase_a.partial.json" if step_dir else None
    timings = {}
    if partial is not None and partial.exists():
        timings = {step["step"]: step["seconds"] for step in json.loads(partial.read_text())["steps"]}
    phases = {phase: store.done_summary(phase) for phase in ("prepare_data", "generate_conjectures", "sample_proofs", "score_selection")}
    phases.update({f"train_adapter_seed{seed}": {k: v for k, v in store.done_summary(f"train_adapter_seed{seed}").items() if k != "losses"}
                   for seed in seeds if store.is_done(f"train_adapter_seed{seed}")})
    phases["evaluate_base"] = store.done_summary("evaluate_base") if store.is_done("evaluate_base") else None
    phases.update({f"evaluate_sampling_seed{seed}": store.done_summary(f"evaluate_sampling_seed{seed}") for seed in seeds})
    improvement = results["improvement_conjecture_holdout_pass@1"]
    transfer = results[f"transfer_minif2f_pass@{samples}"]
    headline = (f"{profile}, {len(seeds)} seed(s): improvement {improvement['label'] if improvement else 'not measurable'}, "
                f"miniF2F-{config['profiles'][profile]['minif2f_split']} pass@{samples} {transfer['label'] if transfer else 'not measurable'}, "
                f"held-out loss {results['heldout_loss_nf4']['label'] if results['heldout_loss_nf4'] else 'not measurable'}")
    estimate = _estimate_full_run(store, config, timings) if profile == "smoke" else None
    report = {"profile": profile, "seeds": seeds, "headline": headline, "results": results, "per_seed": per_seed,
              "phases": phases, "timings_seconds": timings, "full_run_estimate": estimate}
    store.write_rows("report.jsonl", [report])
    (store.path("report.md")).write_text(_markdown(report))
    if store.mirror_directory is not None:
        (store.mirror_directory / "report.md").write_text(_markdown(report))
    return report


def _markdown(report: dict) -> str:
    lines = [f"# rlvr_lean Phase A — {report['profile']} run", "", f"**{report['headline']}**", "",
             "| Measure | Problems | Base | Adapter | Difference | 95% interval | Label |", "|---|---|---|---|---|---|---|"]
    for name, result in report["results"].items():
        if result is None:
            lines.append(f"| {name} | 0 | | | | | not measurable |")
            continue
        size = result.get("problems", result.get("pairs"))
        difference = result.get("delta", result.get("loss_reduction"))
        lines.append(f"| {name} | {size} | {result['base']} | {result['adapter']} | {difference} | "
                     f"[{result['interval'][0]}, {result['interval'][1]}] | {result['label']} |")
    lines += ["", "Held-out loss rows report the loss REDUCTION (base minus adapter): positive is better.", ""]
    if len(report.get("per_seed", {})) > 1:
        lines += ["## Per seed", ""] + [f"- seed {seed}: `{json.dumps(labels)}`" for seed, labels in report["per_seed"].items()] + [""]
    lines += ["## Phases", ""]
    for phase, summary in report["phases"].items():
        lines.append(f"- **{phase}** ({report['timings_seconds'].get(phase, '?')} s): `{json.dumps(summary, default=str)[:900]}`")
    if report.get("full_run_estimate"):
        lines += ["", "## Full-run estimate (scaled from this run's measured rates)", "",
                  f"`{json.dumps(report['full_run_estimate'])}`"]
    return "\n".join(lines) + "\n"
