"""Phase A and Phase B pipeline stages (spec §2–§8, §13), each run as its own process by `rlvr_lean.runner.entry`.

Order: prepare_data → generate_conjectures → sample_proofs → score_selection → train_adapter →
evaluate_sampling → evaluate_loss → report. Each reads its inputs from the run's artifact store
(`<store>/runs/<profile>/`), writes JSONL keyed by content-derived ids, and marks itself done; a rerun
skips finished stages. The profile (`smoke` or `full`) comes from RLVR_LEAN_PROFILE; the arms trained and
evaluated from RLVR_LEAN_ARMS (default: Phase A's one); TensorBoard runs go under RLVR_LEAN_TB_DIR.
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path

from rlvr_lean.data.sources import (
    Statement,
    all_minif2f_normalized,
    is_conjecture_holdout,
    load_earlier_statements,
    load_lean_workbook,
    load_minif2f,
    ranked_workbook_candidates,
    reward_half,
)
from rlvr_lean.domain.conjecturing import build_compile_check_source, build_vacuity_source, parse_type_fingerprint
from rlvr_lean.domain.conjecturing.dedup import StatementDeduplicator
from rlvr_lean.domain.conjecturing.parsing import canonical_statement, parse_generated_statement
from rlvr_lean.domain.proving import PROMPT_PREFIX, build_prover_prompt, completion_from_output
from rlvr_lean.domain.training.arms import (
    FRESH_SETS,
    LADDER_ARM,
    LEGACY_ARMS,
    SCOUT_ARM,
    ArmNames,
    FreshNames,
    arm_names,
    base_loss_names,
    check_arm_list,
)
from rlvr_lean.domain.training.target_format import LEGACY
from rlvr_lean.domain.verification import LEAN_HEADER, VerificationStatus, find_forbidden_token
from rlvr_lean.domain.verification.pin import LEAN_PINS, LeanPin, lean_pin_from_config
from rlvr_lean.runner.heartbeat import ScalarEventWriter
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.infrastructure.verification_service import (
    ProofAttemptToVerify,
    VerificationService,
    check_lean_sources,
    kimina_settings_from_config,
)

STORE = Path(os.environ.get("RLVR_LEAN_STORE", "/root/rlvr_lean_store"))
FP8_DIR = STORE / "models" / "base-fp8"
NF4_DIR = STORE / "models" / "base-nf4"


def _profile() -> str:
    return os.environ.get("RLVR_LEAN_PROFILE", "smoke")


_LEAN_PIN: dict[str, LeanPin] = {}     # this process's Lean pin; `use_lean_pin` sets it before a step runs


def use_lean_pin(config: dict) -> LeanPin:
    """Fix this process's Lean pin from the config's ONE setting (the OEIS Open spec, O2a item 9a). `gpu.__main__`
    calls it before a step runs; `_store()` and `_adapter_dir()` then find the pin's own run directory."""
    _LEAN_PIN["pin"] = lean_pin_from_config(config)
    return _LEAN_PIN["pin"]


def _run_directory(pin: LeanPin | None = None) -> Path:
    """This profile's run directory under a pin (default: the one in use): `<store>/runs/<profile>/` at v4.9,
    the pin's own beside it otherwise, so no file, marker or adapter of one pin is read as another's or
    written over (item 9c). There is no default pin here: a caller that selected none is refused."""
    if pin is None and "pin" not in _LEAN_PIN:
        raise RuntimeError("no Lean pin is selected: call use_lean_pin(config) before a pipeline step, as gpu.__main__ does")
    return STORE / (pin or _LEAN_PIN["pin"]).runs_directory / _profile()


def _store() -> ArtifactStore:
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    return ArtifactStore(_run_directory(), Path(mirror) if mirror else None)


def _sizes(config: dict) -> dict:
    return config["profiles"][_profile()]


def _adapter_dir(names: ArmNames) -> Path:
    return _run_directory() / "adapters" / names.adapter_directory


def _training_seeds(config: dict) -> list[int]:
    """Seed 0 first, then 1 and 2 once it has read (spec §6 item 4); the entry passes the current list."""
    override = os.environ.get("RLVR_LEAN_TRAINING_SEEDS")
    return [int(seed) for seed in override.split(",")] if override else list(config["training"]["seeds"])


def _arm_runs(config: dict) -> list[ArmNames]:
    """Every (arm, seed) this task trains and evaluates, arm-major. Phase A's arm keeps Phase A's names."""
    override = os.environ.get("RLVR_LEAN_ARMS")
    reused = config["selection"]["phase_a_method"]
    arms = override.split(",") if override else [reused]
    check_arm_list(arms)
    return [arm_names(arm, seed, reused) for arm in arms for seed in _training_seeds(config)]


def _cap_torch_memory(config: dict) -> None:
    """A PyTorch stage may not take the desktop's reserve: past the cap it fails in its own process."""
    import torch

    from rlvr_lean.gpu.memory_budget import gpu_total_mib, process_memory_fraction

    fraction = process_memory_fraction(gpu_total_mib(), config["gpu"]["desktop_reserve_gb"], config["gpu"]["torch_overhead_gb"])
    torch.cuda.set_per_process_memory_fraction(fraction)
    print(f"PyTorch memory cap {fraction} of the card (reserve {config['gpu']['desktop_reserve_gb']} GiB for the desktop)", flush=True)


def _epochs(config: dict, names: ArmNames) -> int:
    """Passes over the training set: Phase A's three, except the ladder arm's one, fixed in spec §13b."""
    return config["ladder"]["epochs"] if names.arm == LADDER_ARM else config["training"]["epochs"]


def _is_phase_b(config: dict) -> bool:
    return any(names.arm != config["selection"]["phase_a_method"] for names in _arm_runs(config))


def _tensorboard(names: ArmNames) -> ScalarEventWriter | None:
    root = os.environ.get("RLVR_LEAN_TB_DIR")
    return ScalarEventWriter(Path(root) / names.tensorboard_run) if root else None


def _emit_once(names: ArmNames, what: str, values_by_step: dict[int, dict[str, float]]) -> None:
    """Write scalars that come from STORED results (a reused arm, a finished evaluation) once per task output,
    so a resumed task does not draw the same points twice. The marker sits beside the run's event files."""
    root = os.environ.get("RLVR_LEAN_TB_DIR")
    if not root or not values_by_step:
        return
    marker = Path(root) / names.tensorboard_run / f".emitted_{what}"
    if marker.exists():
        return
    writer = ScalarEventWriter(marker.parent)
    for step, values in sorted(values_by_step.items()):
        if values:
            writer.scalars(step, values)
    marker.write_text("")


def _lean_messages(raw: dict, severity: str) -> list[str]:
    return [str(message.get("data")) for message in (raw.get("response") or {}).get("messages", [])
            if message.get("severity") == severity]


def _compiles(raw: dict) -> bool:
    return not raw.get("error") and isinstance(raw.get("response"), dict) and "messages" in raw["response"] \
        and not _lean_messages(raw, "error")


def _sampling_parameters(config: dict, samples: int, **overrides):
    from vllm import SamplingParams

    values = dict(n=samples, temperature=config["sampling"]["temperature"], top_p=config["sampling"]["top_p"],
                  max_tokens=config["sampling"]["max_new_tokens"], stop=config["sampling"]["stop"], seed=0)
    values.update(overrides)
    return SamplingParams(**values)


def _engine(config: dict, enable_lora: bool):
    os.environ.update({key: str(value) for key, value in config["vllm"].get("environment", {}).items()})
    from vllm import LLM

    from rlvr_lean.gpu.memory_budget import gpu_total_mib, sampling_memory_fraction

    # Leave the desktop's reserve free: vLLM preallocates a share of the card's TOTAL memory.
    fraction = sampling_memory_fraction(gpu_total_mib(), config["gpu"]["desktop_reserve_gb"],
                                        config["gpu"]["vllm_overhead_gb"], config["vllm"]["gpu_memory_utilization"])
    print(f"vLLM gpu_memory_utilization {fraction} (reserve {config['gpu']['desktop_reserve_gb']} GiB for the desktop)", flush=True)
    return LLM(model=str(FP8_DIR), max_model_len=config["vllm"]["max_model_len"],
               gpu_memory_utilization=fraction, seed=0,
               enable_lora=enable_lora, max_lora_rank=config["vllm"]["max_lora_rank"], max_loras=1)


# ------------------------------------------------------------------------------------------- prepare_data
def prepare_data(config: dict) -> dict:
    store = _store()
    if store.is_done("prepare_data"):
        return store.done_summary("prepare_data")
    sizes, data_dir = _sizes(config), STORE / "data"
    split_seed = config["data_sources"]["split_seed"]
    workbook = load_lean_workbook(data_dir, config["data_sources"]["lean_workbook_revision"])
    excluded = all_minif2f_normalized(data_dir, config["data"]["deepseek_prover_commit"])
    ranked = ranked_workbook_candidates(workbook, split_seed, excluded)
    wanted = sizes["seed_statements"] + sizes["reward_statements"]
    candidates = ranked[: int(1.6 * wanted) + 20]
    settings = kimina_settings_from_config(config)
    # The OEIS Open spec, O2a item 9d: a pin that follows another filters THAT pin's seed and reward statements
    # again by compiling each under itself, instead of choosing its own from the ranked candidates.
    earlier_pin = settings.pin.statements_from_pin
    earlier = load_earlier_statements(_run_directory(LEAN_PINS[earlier_pin]), workbook) if earlier_pin else None
    if earlier is not None:
        candidates = earlier.candidates
    raw = check_lean_sources(settings, {s.statement_id: build_compile_check_source(s.statement) for s in candidates})
    compiled = [s for s in candidates if _compiles(raw[s.statement_id])]
    seeds, reward = compiled[: sizes["seed_statements"]], compiled[sizes["seed_statements"]: wanted]
    if earlier is not None:
        seeds, reward = earlier.survivors(compiled)

    def row(statement: Statement, **extra) -> dict:
        return {"statement_id": statement.statement_id, "statement": statement.statement, "source": statement.source,
                "type_fingerprint": parse_type_fingerprint(_lean_messages(raw[statement.statement_id], "info")), **extra}

    store.write_rows("statements_seed.jsonl", (row(s) for s in seeds))
    store.write_rows("statements_reward.jsonl", (row(s, half=reward_half(s, split_seed)) for s in reward))
    # miniF2F: the evaluation set of this profile (a seeded sample for the smoke test), plus fingerprints of
    # ALL 488 statements so generated conjectures can be deduplicated against every eval statement.
    commit = config["data"]["deepseek_prover_commit"]
    all_minif2f = load_minif2f(data_dir, commit, "valid") + load_minif2f(data_dir, commit, "test")
    minif2f_raw = check_lean_sources(settings, {s.statement_id: build_compile_check_source(s.statement) for s in all_minif2f})
    store.write_rows("minif2f_fingerprints.jsonl", (
        {"statement_id": s.statement_id, "statement": s.statement,
         "type_fingerprint": parse_type_fingerprint(_lean_messages(minif2f_raw[s.statement_id], "info"))} for s in all_minif2f))
    evaluation_split = load_minif2f(data_dir, commit, sizes["minif2f_split"])
    rng = random.Random(split_seed)
    chosen = sorted(rng.sample(evaluation_split, min(sizes["minif2f_problems"], len(evaluation_split))),
                    key=lambda s: s.statement_id)
    store.write_rows("minif2f_eval.jsonl", ({"statement_id": s.statement_id, "statement": s.statement} for s in chosen))
    summary = {"workbook_problems": len(workbook), "candidates_checked": len(candidates), "candidates_compiled": len(compiled),
               "seed_statements": len(seeds), "reward_statements": len(reward),
               "reward_gradient": sum(reward_half(s, split_seed) == "gradient" for s in reward),
               "minif2f_eval_split": sizes["minif2f_split"], "minif2f_eval_problems": len(chosen),
               "minif2f_fingerprinted": sum(parse_type_fingerprint(_lean_messages(r, "info")) is not None for r in minif2f_raw.values())}
    if earlier is not None:        # item 9d: how many of the earlier pin's statements survive under this one
        summary.update({"lean_pin": settings.pin.name,
                        "statements_from_pin": {"pin": earlier_pin, "seed_statements": len(earlier.seeds),
                                                "reward_statements": len(earlier.reward)}})
    store.mark_done("prepare_data", summary)
    return summary


# ------------------------------------------------------------------------------------ generate_conjectures
def _conjecture_prompt(focal: dict, others: list[dict]) -> str:
    """The model's native completion format: three workbook statements, each closed with `sorry`, then an open
    `theorem ` for the model to continue (spec §3 item 1)."""
    body = "".join(statement["statement"] + "  sorry\n\n" for statement in [focal, *others])
    return PROMPT_PREFIX + LEAN_HEADER + body + "theorem "


def generate_conjectures(config: dict) -> dict:
    store = _store()
    if store.is_done("generate_conjectures"):
        return store.done_summary("generate_conjectures")
    settings_conjecturing, sizes = config["conjecturing"], _sizes(config)
    seeds = store.read_rows("statements_seed.jsonl")
    rng = random.Random(config["data_sources"]["split_seed"])
    prompts, prompt_seeds = [], []
    for focal in seeds:
        others = rng.sample([s for s in seeds if s["statement_id"] != focal["statement_id"]], settings_conjecturing["seeds_per_prompt"] - 1)
        prompts.append(_conjecture_prompt(focal, others))
        prompt_seeds.append([focal["statement_id"]] + [s["statement_id"] for s in others])
    engine = _engine(config, enable_lora=False)
    parameters = _sampling_parameters(config, settings_conjecturing["samples_per_prompt"],
                                      max_tokens=settings_conjecturing["max_statement_tokens"], stop=[":=", "```"])
    started = time.monotonic()
    outputs = engine.generate(prompts, parameters)
    generation_seconds = time.monotonic() - started
    funnel = {"raw": 0, "unparsed": 0, "forbidden_token": 0, "duplicate_text": 0}
    deduplicator = StatementDeduplicator()
    for row in seeds + store.read_rows("statements_reward.jsonl") + store.read_rows("minif2f_fingerprints.jsonl"):
        deduplicator.add(row["statement"], row.get("type_fingerprint"))
    candidates, generated_tokens = [], 0
    for seed_ids, output in zip(prompt_seeds, outputs):
        for sample in output.outputs:
            funnel["raw"] += 1
            generated_tokens += len(sample.token_ids)
            statement = parse_generated_statement(sample.text, stopped_at_assignment=sample.stop_reason == ":=")
            if statement is None:
                funnel["unparsed"] += 1
                continue
            if find_forbidden_token(statement) is not None:
                funnel["forbidden_token"] += 1
                continue
            conjecture_id, renamed = canonical_statement(statement)
            if not deduplicator.add(renamed):
                funnel["duplicate_text"] += 1
                continue
            candidates.append({"conjecture_id": conjecture_id, "statement": renamed, "seed_statement_ids": seed_ids})
    del engine
    settings = kimina_settings_from_config(config)
    compiled = check_lean_sources(settings, {c["conjecture_id"]: build_compile_check_source(c["statement"]) for c in candidates})
    funnel.update({"compile_failed": 0, "duplicate_fingerprint": 0, "vacuous": 0})
    survivors = []
    for candidate in candidates:
        raw = compiled[candidate["conjecture_id"]]
        if not _compiles(raw):
            funnel["compile_failed"] += 1
            continue
        fingerprint = parse_type_fingerprint(_lean_messages(raw, "info"))
        if fingerprint is not None and not deduplicator.add("fingerprint-only:" + candidate["conjecture_id"], fingerprint):
            funnel["duplicate_fingerprint"] += 1
            continue
        survivors.append({**candidate, "type_fingerprint": fingerprint})
    vacuity = check_lean_sources(settings, {c["conjecture_id"]: build_vacuity_source(c["statement"]) for c in survivors})
    kept = []
    for candidate in survivors:
        if settings.pin.classify(candidate["conjecture_id"], vacuity[candidate["conjecture_id"]]).status is VerificationStatus.VERIFIED:
            funnel["vacuous"] += 1
            continue
        kept.append(candidate)
    survived = len(kept)
    kept = kept[: sizes["conjecture_target"]]
    split_seed, fraction = config["data_sources"]["split_seed"], settings_conjecturing["holdout_fraction"]
    store.write_rows("conjectures.jsonl", ({**c, "holdout": is_conjecture_holdout(c["conjecture_id"], split_seed, fraction)} for c in kept))
    # The yield gate (spec §3 item 7) is on SURVIVAL: raw generations that pass every filter, before the cap.
    summary = {"prompts": len(prompts), "funnel": funnel, "survived_filters": survived, "kept": len(kept),
               "survival_rate": round(survived / max(1, funnel["raw"]), 4),
               "holdout": sum(is_conjecture_holdout(c["conjecture_id"], split_seed, fraction) for c in kept),
               "generated_tokens": generated_tokens, "generation_seconds": round(generation_seconds, 1)}
    store.mark_done("generate_conjectures", summary)
    return summary


# ------------------------------------------------------------------------------------------ sample_proofs
def _sample_and_verify(engine, config: dict, items: list[dict], samples: int, variant: str, service: VerificationService,
                       lora_request=None, sampling_seed: int | None = None) -> tuple[list[dict], dict]:
    """Sample `samples` proofs for every item in chunks; each chunk goes to verification while the next samples.
    `sampling_seed` replaces the stored run's seed (0) for FRESH samples (spec §13b)."""
    rows, generated_tokens, generation_seconds = [], 0, 0.0
    chunk_size = config["proving"]["verification_chunk_prompts"]
    parameters = _sampling_parameters(config, samples) if sampling_seed is None else _sampling_parameters(config, samples, seed=sampling_seed)
    for start in range(0, len(items), chunk_size):
        chunk = items[start:start + chunk_size]
        began = time.monotonic()
        outputs = engine.generate([build_prover_prompt(item["statement"]) for item in chunk], parameters, lora_request=lora_request)
        generation_seconds += time.monotonic() - began
        to_verify = []
        for item, output in zip(chunk, outputs):
            for index, sample in enumerate(output.outputs):
                completion = completion_from_output(sample.text)
                attempt_id = f"{item['id']}#{variant}#{index}"
                generated_tokens += len(sample.token_ids)
                rows.append({"attempt_id": attempt_id, "statement_id": item["id"], "variant": variant, "sample_index": index,
                             "completion": completion, "token_count": len(sample.token_ids), "finish_reason": sample.finish_reason,
                             "first_token_id": int(sample.token_ids[0]) if len(sample.token_ids) else None})
                to_verify.append(ProofAttemptToVerify(attempt_id, item["statement"], completion))
        service.submit(to_verify)
    return rows, {"samples": len(rows), "generated_tokens": generated_tokens, "generation_seconds": round(generation_seconds, 1),
                  "tokens_per_second": round(generated_tokens / generation_seconds, 1) if generation_seconds else None}


def _verification_rows(results: dict) -> list[dict]:
    return [{"attempt_id": attempt_id, "status": result.status.value, "axioms": list(result.axioms),
             "seconds": result.verification_seconds,
             "first_error": next((m for m in result.messages if m.startswith("error")), result.detail)[:300]}
            for attempt_id, result in results.items()]


def sample_proofs(config: dict) -> dict:
    store = _store()
    if store.is_done("sample_proofs"):
        return store.done_summary("sample_proofs")
    conjectures = [{"id": c["conjecture_id"], "statement": c["statement"]} for c in store.read_rows("conjectures.jsonl")]
    reward = [{"id": r["statement_id"], "statement": r["statement"]} for r in store.read_rows("statements_reward.jsonl")]
    engine = _engine(config, enable_lora=False)
    service = VerificationService(kimina_settings_from_config(config))
    conjecture_rows, conjecture_stats = _sample_and_verify(engine, config, conjectures, config["proving"]["samples_per_conjecture"], "base", service)
    reward_rows, reward_stats = _sample_and_verify(engine, config, reward, config["proving"]["samples_per_reward_statement"], "base", service)
    del engine
    verification = _verification_rows(service.results())
    store.write_rows("proof_attempts.jsonl", conjecture_rows + reward_rows)
    store.write_rows("verification.jsonl", verification)
    statuses: dict[str, int] = {}
    for row in verification:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    verified = {row["attempt_id"] for row in verification if row["status"] == "verified"}
    solvable_conjectures = {row["statement_id"] for row in conjecture_rows if row["attempt_id"] in verified}
    solvable_reward = {row["statement_id"] for row in reward_rows if row["attempt_id"] in verified}
    summary = {"conjecture_sampling": conjecture_stats, "reward_sampling": reward_stats, "statuses": statuses,
               "solvable_conjectures": len(solvable_conjectures), "conjectures": len(conjectures),
               "solvable_reward_statements": len(solvable_reward), "reward_statements": len(reward),
               "verification_proofs_per_second": round(service.checked / service.check_seconds, 3) if service.check_seconds else None,
               "timeouts": statuses.get("timeout", 0), "server_errors": statuses.get("server_error", 0)}
    store.mark_done("sample_proofs", summary)
    return summary


# ---------------------------------------------------------------------------------------- score_selection
def _verified_by_statement(store: ArtifactStore, variant: str = "base") -> tuple[dict, dict]:
    """statement id -> list of (attempt id, token count) verified; statement id -> samples drawn."""
    verified_ids = {row["attempt_id"] for row in store.read_rows("verification.jsonl") if row["status"] == "verified"}
    verified, drawn = {}, {}
    for row in store.read_rows("proof_attempts.jsonl"):
        if row["variant"] != variant:
            continue
        drawn[row["statement_id"]] = drawn.get(row["statement_id"], 0) + 1
        if row["attempt_id"] in verified_ids:
            verified.setdefault(row["statement_id"], []).append((row["attempt_id"], row["token_count"]))
    return verified, drawn


def score_selection(config: dict) -> dict:
    import gc

    import torch

    from rlvr_lean.domain.evaluation.statistics import spearman
    from rlvr_lean.domain.selection.canonical_proof import choose_canonical_proof
    from rlvr_lean.domain.selection.pool import PoolConjecture, in_difficulty_band, training_set_size
    from rlvr_lean.domain.selection.strategies import (
        DifficultyHeuristicSelection,
        LearningProgressSelection,
        RandomSelection,
        selection_overlap,
    )
    from rlvr_lean.domain.training.equalization import build_training_examples
    from rlvr_lean.gpu.learning_progress import GradientScorer, dot, norm, score
    from rlvr_lean.gpu.model_utils import attach_lora, load_nf4_base, training_text

    store = _store()
    if store.is_done("score_selection"):
        return store.done_summary("score_selection")
    _cap_torch_memory(config)
    selection, lora = config["selection"], config["lora"]
    verified, drawn = _verified_by_statement(store)
    completions = {row["attempt_id"]: row["completion"] for row in store.read_rows("proof_attempts.jsonl")}
    conjectures = {c["conjecture_id"]: c for c in store.read_rows("conjectures.jsonl")}
    reward = {r["statement_id"]: r for r in store.read_rows("statements_reward.jsonl")}

    def canonical_pair(statement_id: str, statement: str) -> tuple[str, str, str, int]:
        attempt_id = choose_canonical_proof(verified[statement_id])
        prompt, target = training_text(statement, completions[attempt_id])
        return attempt_id, prompt, target, dict(verified[statement_id])[attempt_id]

    pool_rows = []
    for conjecture_id, conjecture in sorted(conjectures.items()):
        if conjecture["holdout"] or conjecture_id not in verified:
            continue
        attempt_id, prompt, target, tokens = canonical_pair(conjecture_id, conjecture["statement"])
        pool_rows.append({"conjecture_id": conjecture_id, "sample_count": drawn[conjecture_id], "verified_count": len(verified[conjecture_id]),
                          "canonical_proof_id": attempt_id, "canonical_proof_tokens": tokens, "prompt": prompt, "target": target})
    reward_pairs = []
    for statement_id, row in sorted(reward.items()):
        if row["half"] == "gradient" and statement_id in verified:
            _, prompt, target, _ = canonical_pair(statement_id, row["statement"])
            reward_pairs.append({"prompt": prompt, "target": target,
                                 "pass_rate": len(verified[statement_id]) / drawn[statement_id]})
    if not pool_rows or not reward_pairs:
        raise RuntimeError(f"nothing to score: {len(pool_rows)} pool conjectures, {len(reward_pairs)} reward-gradient pairs")

    def scores_with(seed: int, rows: list[dict]) -> tuple[dict, dict]:
        model, tokenizer = load_nf4_base(NF4_DIR)
        # LEGACY: this stage writes Phase A's files (`pool.jsonl`, `selections.jsonl`), whose scores are the
        # legacy ones and stay so (spec §6 item 2a: nothing is recomputed in place). A scoring pass in the
        # native format writes files of its own; it belongs to the comparison after the scout (spec §13a).
        scorer = GradientScorer(attach_lora(model, lora, seed=seed, trainable_a=False), tokenizer,
                                config["training"]["max_sequence_tokens"], LEGACY)
        reference = scorer.mean_gradient([(p["prompt"], p["target"]) for p in reward_pairs])
        band_pairs = [(p["prompt"], p["target"]) for p in reward_pairs if 0 < p["pass_rate"] <= selection["band_upper"]]
        frontier = scorer.mean_gradient(band_pairs) if band_pairs else None
        reference_norm = norm(reference)
        dots, extras = {}, {}
        for row in rows:
            gradient = scorer.gradient(row["prompt"], row["target"])
            value, cosine = score(gradient, reference, reference_norm)
            dots[row["conjecture_id"]] = value
            extras[row["conjecture_id"]] = {"cosine": cosine, "frontier": dot(gradient, frontier) if frontier else None}
        del scorer, model
        gc.collect()
        torch.cuda.empty_cache()
        return dots, extras

    started = time.monotonic()
    dots, extras = scores_with(selection["scoring_seed"], pool_rows)
    scoring_seconds = time.monotonic() - started
    rng = random.Random(selection["random_seed"])
    stability_rows = sorted(rng.sample(pool_rows, min(selection["stability_sample"], len(pool_rows))), key=lambda r: r["conjecture_id"])
    stability_dots, _ = scores_with(selection["stability_seed"], stability_rows)

    pool = [PoolConjecture(conjecture_id=r["conjecture_id"], sample_count=r["sample_count"], verified_count=r["verified_count"],
                           canonical_proof_id=r["canonical_proof_id"], canonical_proof_tokens=r["canonical_proof_tokens"],
                           learning_progress=dots[r["conjecture_id"]], learning_progress_cosine=extras[r["conjecture_id"]]["cosine"])
            for r in pool_rows]
    band_size = sum(in_difficulty_band(c, selection["band_upper"]) for c in pool)
    size = training_set_size(len(pool), band_size, maximum=selection["maximum_training_set"],
                             pool_fraction=selection["pool_fraction"], band_minimum=selection["band_minimum"])
    methods = [RandomSelection(), DifficultyHeuristicSelection(band_upper=selection["band_upper"]),
               LearningProgressSelection(), LearningProgressSelection(use_cosine=True)]
    selected = {method.name: method.select(pool, size.size, selection["random_seed"]) for method in methods}
    names = list(selected)
    overlaps = {f"{a}~{b}": round(selection_overlap(selected[a], selected[b]), 3)
                for index, a in enumerate(names) for b in names[index + 1:]}
    examples = build_training_examples(selected[selection["phase_a_method"]])
    store.write_rows("pool.jsonl", ({**{k: v for k, v in r.items() if k not in ("prompt", "target")},
                                     "learning_progress": dots[r["conjecture_id"]], **extras[r["conjecture_id"]]} for r in pool_rows))
    store.write_rows("selections.jsonl", ({"method": name, "conjecture_ids": [c.conjecture_id for c in chosen]} for name, chosen in selected.items()))
    store.write_rows("training_examples.jsonl", ({"conjecture_id": e.conjecture_id, "proof_attempt_id": e.proof_attempt_id} for e in examples))
    ids = [r["conjecture_id"] for r in pool_rows]
    stability_ids = [r["conjecture_id"] for r in stability_rows]
    frontier_ids = [i for i in ids if extras[i]["frontier"] is not None]
    summary = {"pool": len(pool), "band_size": band_size, "training_set_size": size.size, "heuristic_fills_upward": size.heuristic_fills_upward,
               "phase_a_method": selection["phase_a_method"],
               "reward_gradient_pairs": len(reward_pairs), "scoring_seconds": round(scoring_seconds, 1), "overlaps": overlaps,
               "stability_spearman": spearman([dots[i] for i in stability_ids], [stability_dots[i] for i in stability_ids]),
               "length_bias_spearman": spearman([dots[i] for i in ids], [r["canonical_proof_tokens"] for r in pool_rows]),
               "dot_vs_cosine_spearman": spearman([dots[i] for i in ids], [extras[i]["cosine"] for i in ids]),
               "dot_vs_frontier_spearman": spearman([dots[i] for i in frontier_ids], [extras[i]["frontier"] for i in frontier_ids]) if frontier_ids else None}
    store.mark_done("score_selection", summary)
    return summary


# ----------------------------------------------------------------------------------------- train_adapter
def _pool_conjectures(store: ArtifactStore) -> list:
    from rlvr_lean.domain.selection.pool import PoolConjecture

    return [PoolConjecture(conjecture_id=r["conjecture_id"], sample_count=r["sample_count"], verified_count=r["verified_count"],
                           canonical_proof_id=r["canonical_proof_id"], canonical_proof_tokens=r["canonical_proof_tokens"],
                           learning_progress=r["learning_progress"], learning_progress_cosine=r["cosine"])
            for r in store.read_rows("pool.jsonl")]


def _training_examples(store: ArtifactStore, config: dict, arm: str) -> list[dict]:
    """The arm's N (conjecture id, canonical proof id) pairs in selection order (spec §6 item 1, §13)."""
    from rlvr_lean.domain.selection.strategies import HalfPassRateSelection
    from rlvr_lean.domain.training.equalization import build_training_examples

    if arm not in LEGACY_ARMS:
        raise ValueError(f"{arm!r} names no selection; an arm in another format trains on a legacy arm's (ArmNames.examples_arm)")
    if arm == config["selection"]["phase_a_method"]:
        return store.read_rows("training_examples.jsonl")
    pool = _pool_conjectures(store)
    if arm == HalfPassRateSelection.name:
        name = f"selection_{arm}.jsonl"
        if not store.path(name).exists():
            size = store.done_summary("score_selection")["training_set_size"]
            chosen = HalfPassRateSelection().select(pool, size, config["selection"]["random_seed"])
            store.write_rows(name, [{"method": arm, "conjecture_ids": [c.conjecture_id for c in chosen]}])
        chosen_ids = store.read_rows(name)[0]["conjecture_ids"]
    else:
        chosen_ids = next(row["conjecture_ids"] for row in store.read_rows("selections.jsonl") if row["method"] == arm)
    by_id = {conjecture.conjecture_id: conjecture for conjecture in pool}
    return [{"conjecture_id": e.conjecture_id, "proof_attempt_id": e.proof_attempt_id}
            for e in build_training_examples([by_id[conjecture_id] for conjecture_id in chosen_ids])]


def _train_one_adapter(config: dict, names: ArmNames, examples: list[tuple[str, str]], heldout_pairs: list[dict]) -> dict:
    """Train and save one adapter. EVERY GPU object is a local of this function, so nothing can outlive it:
    the model, the optimizer and the last `loss`. That last one matters. With gradient checkpointing a
    loss tensor's graph keeps each layer's forward function, and through it the whole 4-bit base model
    (3.4 GB); left in the caller's scope it kept the PREVIOUS arm's model alive while the next one loaded
    (Phase A seed 2 peaked 3.1 GB over seed 1; Phase B's first queue ran out of memory, task 1106ae12;
    its re-run measured 3.36 GB still allocated after `del` + `gc.collect()`)."""
    import torch

    from rlvr_lean.domain.evaluation.loss_parts import loss_by_part
    from rlvr_lean.domain.training.target_format import is_heldout_read_step
    from rlvr_lean.gpu.model_utils import attach_lora, encode_pair, load_nf4_base, mean_proof_token_loss, position_losses

    training = config["training"]
    model, tokenizer = load_nf4_base(NF4_DIR)
    peft_model = attach_lora(model, config["lora"], seed=names.seed)
    optimizer = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=training["learning_rate"], weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / training["warmup_steps"]))
    # The arm's pair format (spec §6 item 2a), the same for its training pairs and its held-out pairs.
    encoded = [encode_pair(tokenizer, prompt, target, training["max_sequence_tokens"], names.target_format) for prompt, target in examples]
    heldout_encoded = [encode_pair(tokenizer, p["prompt"], p["target"], training["max_sequence_tokens"], names.target_format) for p in heldout_pairs]
    epochs = _epochs(config, names)
    total_steps = epochs * math.ceil(len(encoded) / training["effective_batch"])
    # Spec §13: step 0, every `heldout_loss_every` steps and the last step. §13a adds every step up to
    # `heldout_every_step_until` for an arm in the native format, where the first steps are what is in question.
    every_step_until = 0 if names.target_format == LEGACY else training["heldout_every_step_until"]
    tensorboard = _tensorboard(names)
    order_rng, losses, curve, parts_curve = random.Random(names.seed), [], [], []

    def heldout_by_part() -> dict:
        """Without gradients and in eval mode, then the previous mode back: the trajectory is unchanged."""
        was_training = peft_model.training
        peft_model.eval()
        with torch.no_grad():
            per_pair = [position_losses(peft_model, ids, mask)[0] for ids, mask in heldout_encoded]
        peft_model.train(was_training)
        return loss_by_part(per_pair)

    def record(step: int, values: dict[str, float]) -> None:
        if heldout_encoded and is_heldout_read_step(step, total_steps, training["heldout_loss_every"], every_step_until):
            parts = heldout_by_part()
            values = {**values, **_heldout_scalars(parts)}
            curve.append([step, round(parts["mean_loss"], 4)])
            parts_curve.append(_heldout_point(step, parts))
        if tensorboard is not None and values:
            tensorboard.scalars(step, values)

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    record(0, {})
    for _ in range(epochs):
        order = list(range(len(encoded)))
        order_rng.shuffle(order)
        for batch_start in range(0, len(order), training["effective_batch"]):
            batch = order[batch_start:batch_start + training["effective_batch"]]
            step_loss = 0.0
            for index in batch:
                loss = mean_proof_token_loss(peft_model, *encoded[index]) / len(batch)
                loss.backward()
                step_loss += float(loss.detach())
                del loss
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(round(step_loss, 4))
            record(len(losses), {"train/loss": step_loss})
    adapter_dir = _adapter_dir(names)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_dir))
    return {"arm": names.arm, "seed": names.seed, "examples": len(examples), "steps": len(losses), "epochs": epochs,
            "target_format": names.target_format, "sequence_start_token_id": int(tokenizer.bos_token_id),
            "first_loss": losses[0], "last_loss": losses[-1], "losses": losses, "heldout_loss_curve": curve,
            "heldout_parts_curve": parts_curve,
            "seconds": round(time.monotonic() - started, 1), "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}


HELDOUT_PART_TAGS = {"first": "heldout/first_token_nats", "newline": "heldout/newline_nats", "fence": "heldout/fence_nats"}


def _heldout_scalars(parts: dict) -> dict[str, float]:
    """The TensorBoard values of one held-out read: the mean per-token loss (spec §13's tag) and, so a cost
    paid once per proof cannot pass for learning, the body's nats per token and each framing token's nats."""
    return {"heldout/loss_nf4": parts["mean_loss"], "heldout/body_nats_per_token": parts["body_per_token"],
            **{tag: parts["nats_per_proof"][part] for part, tag in HELDOUT_PART_TAGS.items()}}


def _heldout_point(step: int, parts: dict) -> dict:
    """One held-out read as stored in the training summary (`heldout_parts_curve`)."""
    return {"step": step, "mean_loss": round(parts["mean_loss"], 4), "body_per_token": round(parts["body_per_token"], 4),
            "nats_per_proof": {part: round(value, 4) for part, value in parts["nats_per_proof"].items()},
            "contribution": {part: round(value, 4) for part, value in parts["contribution"].items()}}


def _stored_heldout_scalars(point: dict) -> dict[str, float]:
    """`_heldout_scalars` of a stored `_heldout_point`."""
    return {"heldout/loss_nf4": point["mean_loss"], "heldout/body_nats_per_token": point["body_per_token"],
            **{tag: point["nats_per_proof"][part] for part, tag in HELDOUT_PART_TAGS.items()}}


MAX_LEFTOVER_GB = 1.0      # after an adapter is trained and its function has returned, the card must be this empty


def train_adapter(config: dict) -> dict:
    """One adapter per arm and training seed. Each is marked done on its own, so adding seeds or arms later
    trains only the new ones. Live TensorBoard (spec §13): `train/loss` every optimizer step and the held-out
    NF4 loss every `heldout_loss_every` steps; a reused adapter's stored losses are written once."""
    import gc

    import torch

    from rlvr_lean.gpu.model_utils import training_text

    _cap_torch_memory(config)
    store = _store()
    completions = {row["attempt_id"]: row["completion"] for row in store.read_rows("proof_attempts.jsonl")}
    statements = {c["conjecture_id"]: c["statement"] for c in store.read_rows("conjectures.jsonl")}
    heldout_pairs = _heldout_pairs(store)
    per_run = {}
    for names in _arm_runs(config):
        if store.is_done(names.train_marker):
            per_run[names.key] = store.done_summary(names.train_marker)
            stored: dict[int, dict[str, float]] = {step: {"train/loss": loss}
                                                   for step, loss in enumerate(per_run[names.key]["losses"], start=1)}
            for step, value in per_run[names.key].get("heldout_loss_curve", []):
                stored.setdefault(step, {})["heldout/loss_nf4"] = value
            for point in per_run[names.key].get("heldout_parts_curve", []):
                stored.setdefault(point["step"], {}).update(_stored_heldout_scalars(point))
            _emit_once(names, "train_loss", stored)
            continue
        examples = [training_text(statements[e["conjecture_id"]], completions[e["proof_attempt_id"]])
                    for e in _training_examples(store, config, names.examples_arm)]
        if not examples:
            raise RuntimeError(f"the {names.examples_arm} selection is empty: the pool was too small for N >= 1")
        summary = _train_one_adapter(config, names, examples, heldout_pairs)
        gc.collect()
        torch.cuda.empty_cache()
        summary["allocated_after_cleanup_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
        store.mark_done(names.train_marker, summary)       # the adapter is saved: keep it whatever follows
        per_run[names.key] = summary
        if summary["allocated_after_cleanup_gb"] > MAX_LEFTOVER_GB:
            raise RuntimeError(f"{summary['allocated_after_cleanup_gb']} GB is still allocated on the GPU after {names.key} "
                               "was trained and released; the next adapter would load on top of it. Something still "
                               "references the model (see _train_one_adapter).")
    return {"per_run": {key: {k: v for k, v in summary.items() if k != "losses"} for key, summary in per_run.items()}}


# ------------------------------------------------------------------------------------- evaluate_sampling
def _heldout_pairs(store: ArtifactStore) -> list[dict]:
    """Reward-VALIDATION statements with a verified base proof: the held-out loss is measured on their
    canonical proofs (never used in scoring)."""
    from rlvr_lean.domain.selection.canonical_proof import choose_canonical_proof
    from rlvr_lean.gpu.model_utils import training_text

    verified, _ = _verified_by_statement(store)
    completions = {row["attempt_id"]: row["completion"] for row in store.read_rows("proof_attempts.jsonl")}
    pairs = []
    for row in store.read_rows("statements_reward.jsonl"):
        if row["half"] == "validation" and row["statement_id"] in verified:
            prompt, target = training_text(row["statement"], completions[choose_canonical_proof(verified[row["statement_id"]])])
            pairs.append({"statement_id": row["statement_id"], "prompt": prompt, "target": target})
    return pairs


def _fp8_losses(engine, tokenizer, pairs: list[dict], max_tokens: int, lora_request, cache_token_budget: int,
                target_format: str) -> list[dict]:
    """Each held-out pair's loss on the SERVING stack, from vLLM's prompt log-probabilities, as stored rows
    (`pair_loss_row`: the mean per-token NLL and each part's nats), with the pairs in `target_format`.

    Scored in groups that fit `cache_token_budget` tokens of the engine's cache, so no sequence is evicted
    half done (see `token_budget_batches`)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from rlvr_lean.domain.evaluation.loss_parts import pair_loss_row
    from rlvr_lean.gpu.memory_budget import token_budget_batches
    from rlvr_lean.gpu.model_utils import encode_pair

    encoded = [encode_pair(tokenizer, p["prompt"], p["target"], max_tokens, target_format) for p in pairs]
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids, _ in encoded]
    outputs = []
    for start, stop in token_budget_batches([len(ids) for ids, _ in encoded], cache_token_budget):
        outputs += engine.generate(prompts[start:stop], SamplingParams(max_tokens=1, prompt_logprobs=0), lora_request=lora_request)
    rows = []
    for pair, (ids, mask), output in zip(pairs, encoded, outputs):
        values = [-output.prompt_logprobs[position][token].logprob
                  for position, (token, keep) in enumerate(zip(ids, mask)) if keep and position > 0]
        rows.append(pair_loss_row(pair["statement_id"], values))
    return rows


NF4_TAG = "eval/heldout_loss_nf4"


def _emit_evaluations(store: ArtifactStore, config: dict, runs: list[ArmNames], stage: str) -> None:
    """The before-and-after `eval/` scalars of spec §13: base at step 0, the adapter at its last training step.
    `evaluate_sampling` writes all but the NF4 loss, `evaluate_loss` writes only that, each once per run."""
    from rlvr_lean.reporting.phase_b import evaluation_scalars

    if not os.environ.get("RLVR_LEAN_TB_DIR"):
        return
    marker_name = {"evaluate_sampling": "sampling_marker", "evaluate_loss": "loss_marker"}[stage]

    def keep(values: dict[str, float]) -> dict[str, float]:
        return {tag: value for tag, value in values.items() if (tag == NF4_TAG) == (stage == "evaluate_loss")}

    base: dict[str, dict[str, float]] = {}
    for names in runs:
        if not store.is_done(getattr(names, marker_name)) or not store.is_done(names.train_marker):
            continue
        if names.target_format not in base:
            base[names.target_format] = evaluation_scalars(store, config, _profile(), None, base_format=names.target_format)
        steps = store.done_summary(names.train_marker)["steps"]
        _emit_once(names, stage, {0: keep(base[names.target_format]), steps: keep(evaluation_scalars(store, config, _profile(), names))})


def evaluate_sampling(config: dict) -> dict:
    """The base model's miniF2F samples and FP8 held-out loss once, then each arm and seed's adapter on all
    three evaluation sets and its FP8 held-out loss. Every part is marked done on its own."""
    from transformers import AutoTokenizer
    from vllm.lora.request import LoRARequest

    store = _store()
    evaluation, sizes, max_tokens = config["evaluation"], _sizes(config), config["training"]["max_sequence_tokens"]
    holdout = [{"id": c["conjecture_id"], "statement": c["statement"]} for c in store.read_rows("conjectures.jsonl") if c["holdout"]]
    workbook_holdout = [{"id": r["statement_id"], "statement": r["statement"]} for r in store.read_rows("statements_reward.jsonl") if r["half"] == "validation"]
    minif2f = [{"id": r["statement_id"], "statement": r["statement"]} for r in store.read_rows("minif2f_eval.jsonl")]
    runs = _arm_runs(config)
    pending = [names for names in runs if not store.is_done(names.sampling_marker)]
    if store.is_done("evaluate_base") and not pending:
        _emit_evaluations(store, config, runs, "evaluate_sampling")
        return {"runs": [names.key for names in runs], "note": "already evaluated"}
    engine = _engine(config, enable_lora=True)
    tokenizer = AutoTokenizer.from_pretrained(str(FP8_DIR))
    pairs = _heldout_pairs(store)
    # The engine holds at least one max_model_len sequence in its cache or it does not start; half of that
    # leaves room for its free-block margin.
    loss_budget = config["vllm"]["max_model_len"] // 2
    summary = {}
    if not store.is_done("evaluate_base"):
        service = VerificationService(kimina_settings_from_config(config))
        # The loss takes seconds and the samples take most of an hour: the cheap part goes first, so a
        # failure in it does not throw the samples and their checks away.
        losses = _fp8_losses(engine, tokenizer, pairs, max_tokens, None, loss_budget, LEGACY) if pairs else []
        rows, stats = _sample_and_verify(engine, config, minif2f, sizes["minif2f_samples"], "eval_base", service)
        store.write_rows("eval_attempts_base.jsonl", rows)
        store.write_rows("eval_verification_base.jsonl", _verification_rows(service.results()))
        store.write_rows("heldout_loss_fp8_base.jsonl", losses)
        summary["base"] = {"minif2f": stats, "proofs_per_second": round(service.checked / service.check_seconds, 3) if service.check_seconds else None}
        store.mark_done("evaluate_base", summary["base"])
    # The base's held-out loss in every other format a pending arm uses, under that format's own names (spec §6
    # item 2a: Phase A's files keep the legacy figures). The base's SAMPLES are shared: prompts do not change.
    for target_format in sorted({names.target_format for names in pending} - {LEGACY}):
        base_names = base_loss_names(target_format)
        if pairs and not store.is_done(base_names.fp8_marker):
            store.write_rows(base_names.fp8_file, _fp8_losses(engine, tokenizer, pairs, max_tokens, None, loss_budget, target_format))
            store.mark_done(base_names.fp8_marker, {"pairs": len(pairs), "target_format": target_format})
    for lora_id, names in enumerate(runs, start=1):          # a LoRA id only has to be unique within this engine
        if store.is_done(names.sampling_marker):
            continue
        adapter = LoRARequest(names.adapter_directory, lora_id, str(_adapter_dir(names)))
        service = VerificationService(kimina_settings_from_config(config))
        losses = _fp8_losses(engine, tokenizer, pairs, max_tokens, adapter, loss_budget, names.target_format) if pairs else []
        rows, stats = [], {}
        for name, items, samples in (("conjecture_holdout", holdout, evaluation["conjecture_holdout_samples"]),
                                     ("workbook_holdout", workbook_holdout, evaluation["workbook_holdout_samples"]),
                                     ("minif2f", minif2f, sizes["minif2f_samples"])):
            if items:
                new_rows, stats[name] = _sample_and_verify(engine, config, items, samples, names.variant, service, lora_request=adapter)
                rows += new_rows
        store.write_rows(names.eval_attempts_file, rows)
        store.write_rows(names.eval_verification_file, _verification_rows(service.results()))
        store.write_rows(names.fp8_loss_file, losses)
        summary[names.key] = {"sampling": stats, "proofs_per_second": round(service.checked / service.check_seconds, 3) if service.check_seconds else None}
        store.mark_done(names.sampling_marker, summary[names.key])
        _emit_evaluations(store, config, [names], "evaluate_sampling")
    del engine
    _emit_evaluations(store, config, runs, "evaluate_sampling")    # the reused arm's stored evaluations
    return summary


# ----------------------------------------------------------------------------------------- evaluate_loss
def evaluate_loss(config: dict) -> dict:
    """Held-out loss on the TRAINING stack (NF4 + PEFT): base once, then each arm and seed's adapter."""
    import torch
    from peft import PeftModel

    from rlvr_lean.domain.evaluation.loss_parts import pair_loss_row
    from rlvr_lean.gpu.model_utils import encode_pair, load_nf4_base, position_losses

    store = _store()
    runs = _arm_runs(config)
    pending = [names for names in runs if not store.is_done(names.loss_marker)]
    # The base's loss is stored once per pair format: Phase A's names hold the legacy figures, every other
    # format has names of its own (spec §6 item 2a).
    base_formats = [target_format for target_format in sorted({LEGACY} | {names.target_format for names in runs})
                    if not store.is_done(base_loss_names(target_format).nf4_marker)]
    if not pending and not base_formats:
        _emit_evaluations(store, config, runs, "evaluate_loss")
        return {"note": "already evaluated"}
    _cap_torch_memory(config)
    pairs = _heldout_pairs(store)
    model, tokenizer = load_nf4_base(NF4_DIR)
    max_tokens = config["training"]["max_sequence_tokens"]
    encoded = {target_format: [encode_pair(tokenizer, p["prompt"], p["target"], max_tokens, target_format) for p in pairs]
               for target_format in set(base_formats) | {names.target_format for names in pending}}

    def loss_rows(current_model, target_format: str) -> list[dict]:
        """Each pair's loss under `current_model`, by part (`pair_loss_row`), with the pairs in `target_format`."""
        return [pair_loss_row(pair["statement_id"], position_losses(current_model, ids, mask)[0])
                for pair, (ids, mask) in zip(pairs, encoded[target_format])]

    summary = {}
    with torch.no_grad():
        for target_format in base_formats:
            base_names = base_loss_names(target_format)
            store.write_rows(base_names.nf4_file, loss_rows(model, target_format))
            store.mark_done(base_names.nf4_marker, {"pairs": len(pairs), "target_format": target_format})
        for names in pending:
            peft_model = PeftModel.from_pretrained(model, str(_adapter_dir(names)), adapter_name=names.key)
            peft_model.eval()
            store.write_rows(names.nf4_loss_file, loss_rows(peft_model, names.target_format))
            store.mark_done(names.loss_marker, {"pairs": len(pairs), "target_format": names.target_format})
            model = peft_model.unload()               # back to the bare base for the next adapter
            summary[names.key] = {"pairs": len(pairs)}
    _emit_evaluations(store, config, runs, "evaluate_loss")
    return summary


# ------------------------------------------------------------------------- the ladder's first rung (§13b)
def _status_counts(verification: list[dict]) -> dict[str, int]:
    statuses: dict[str, int] = {}
    for row in verification:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    return statuses


def _stored_base_counts(store: ArtifactStore) -> dict[str, int]:
    """statement id -> verified proofs among the base model's STORED samples (0 for a statement it never proved)."""
    verified, drawn = _verified_by_statement(store)
    return {statement_id: len(verified.get(statement_id, [])) for statement_id in drawn}


def _fresh_sets(store: ArtifactStore, config: dict) -> dict[str, tuple[list[dict], int]]:
    """The statements that get fresh samples and how many each (spec §13b), from the base's stored counts:
    the conjectures below the band, the workbook-holdout problems the base never proved, the never-proved
    conjectures. Each in a fixed order."""
    from rlvr_lean.domain.evaluation.ladder import below_band_ids, never_proved_ids

    ladder = config["ladder"]
    stored = _stored_base_counts(store)
    conjectures = {c["conjecture_id"]: c["statement"] for c in store.read_rows("conjectures.jsonl")}
    workbook = {r["statement_id"]: r["statement"] for r in store.read_rows("statements_reward.jsonl") if r["half"] == "validation"}
    conjecture_counts = {key: stored[key] for key in conjectures}
    workbook_counts = {key: stored[key] for key in workbook}
    low, high = ladder["below_band_verified"]

    def items(ids: list[str], statements: dict[str, str]) -> list[dict]:
        return [{"id": key, "statement": statements[key]} for key in ids]

    return {"below_band": (items(below_band_ids(conjecture_counts, low, high), conjectures), config["proving"]["samples_per_conjecture"]),
            "workbook_unsolved": (items(never_proved_ids(workbook_counts), workbook), config["evaluation"]["workbook_holdout_samples"]),
            "never_proved": (items(never_proved_ids(conjecture_counts), conjectures), config["proving"]["samples_per_conjecture"])}


def ladder_probes(config: dict) -> dict:
    """Fresh samples from the base and from the ladder adapter on the same statements, with one new sampling
    seed, in one engine session, the base then the adapter set by set (spec §13b: the primary and the two
    fresh-sample secondaries). Each (set, model) is written and marked done on its own."""
    from vllm.lora.request import LoRARequest

    from rlvr_lean.domain.evaluation.ladder import trained_on

    store = _store()
    runs = _arm_runs(config)
    if len(runs) != 1 or runs[0].arm != LADDER_ARM or runs[0].seed != config["ladder"]["seed"]:
        raise ValueError(f"the probes belong to ONE run, {LADDER_ARM} at seed {config['ladder']['seed']} (spec §13b); got {[n.key for n in runs]}")
    names = runs[0]
    sets = _fresh_sets(store, config)
    training_ids = [example["conjecture_id"] for example in _training_examples(store, config, names.examples_arm)]
    overlap = trained_on([item["id"] for item in sets["below_band"][0]], training_ids)
    if overlap:
        raise RuntimeError(f"{len(overlap)} below-band conjectures are in the training set: the primary would not measure untrained conjectures")
    wanted = [(set_name, who) for set_name in FRESH_SETS for who in ("base", names.key)]
    summary = {"sets": {set_name: {"statements": len(sets[set_name][0]), "samples_each": sets[set_name][1]} for set_name in FRESH_SETS},
               "below_band_in_training_set": len(overlap), "training_examples": len(training_ids),
               "sampling_seed": config["ladder"]["fresh_sampling_seed"], "parts": {}}
    if all(store.is_done(FreshNames(who, set_name).marker) for set_name, who in wanted):
        summary["note"] = "already sampled"
        return summary
    engine = _engine(config, enable_lora=True)
    adapter = LoRARequest(names.adapter_directory, 1, str(_adapter_dir(names)))
    for set_name, who in wanted:
        fresh = FreshNames(who, set_name)
        if store.is_done(fresh.marker):
            continue
        items, samples = sets[set_name]
        service = VerificationService(kimina_settings_from_config(config))
        rows, stats = _sample_and_verify(engine, config, items, samples, fresh.variant, service,
                                         lora_request=None if who == "base" else adapter, sampling_seed=config["ladder"]["fresh_sampling_seed"])
        verification = _verification_rows(service.results())
        store.write_rows(fresh.attempts_file, rows)
        store.write_rows(fresh.verification_file, verification)
        part = {"statements": len(items), "sampling": stats, "statuses": _status_counts(verification),
                "proofs_per_second": round(service.checked / service.check_seconds, 3) if service.check_seconds else None}
        store.mark_done(fresh.marker, part)
        summary["parts"][fresh.marker] = part
        print(f"{fresh.marker}: {part['statuses']}", flush=True)
    del engine
    return summary


def negation_census(config: dict) -> dict:
    """The training-data census of spec §13b, base model only: for every never-proved conjecture, samples at
    its NEGATION. A verified proof of the negation shows the conjecture false as stated. Every negation is
    compile-checked first (one that does not compile is counted, not sampled), and Lean is asked whether it
    is exactly `¬ (the conjecture's type)` (`build_negation_exactness_source`)."""
    from rlvr_lean.domain.conjecturing.negation import build_negation_exactness_source, negate_statement, negation_name
    from rlvr_lean.domain.evaluation.ladder import CENSUS_MARKER, census_summary, never_proved_ids

    store = _store()
    if store.is_done(CENSUS_MARKER):
        return store.done_summary(CENSUS_MARKER)
    conjectures = {c["conjecture_id"]: c["statement"] for c in store.read_rows("conjectures.jsonl")}
    stored = _stored_base_counts(store)
    never = never_proved_ids({key: stored[key] for key in conjectures})
    negated, statement_rows = {}, {}
    for conjecture_id in never:
        try:
            negated[conjecture_id] = negate_statement(conjectures[conjecture_id], negation_name(conjecture_id))
            if find_forbidden_token(negated[conjecture_id]) is not None:
                raise ValueError("the negated statement holds a forbidden token")
        except ValueError as error:
            negated.pop(conjecture_id, None)
            statement_rows[conjecture_id] = {"conjecture_id": conjecture_id, "built": False, "compiles": False, "exact": False,
                                             "statement": None, "error": str(error)[:300]}
    settings = kimina_settings_from_config(config)
    compiled = check_lean_sources(settings, {key: build_compile_check_source(statement) for key, statement in negated.items()})
    exactness = check_lean_sources(settings, {key: build_negation_exactness_source(conjectures[key]) for key in negated})
    for conjecture_id, statement in negated.items():
        raw = compiled[conjecture_id]
        statement_rows[conjecture_id] = {"conjecture_id": conjecture_id, "built": True, "compiles": _compiles(raw),
                                         "exact": _compiles(exactness[conjecture_id]), "statement": statement,
                                         "error": (_lean_messages(raw, "error") or [str(raw.get("error") or "")])[0][:300]}
    rows = [statement_rows[conjecture_id] for conjecture_id in never]
    store.write_rows("negation_statements.jsonl", rows)
    items = [{"id": row["conjecture_id"], "statement": row["statement"]} for row in rows if row["built"] and row["compiles"]]
    if not items:
        raise RuntimeError(f"none of the {len(never)} negated statements compiles: the builder or the server is wrong")
    engine = _engine(config, enable_lora=False)
    service = VerificationService(settings)
    attempts, stats = _sample_and_verify(engine, config, items, config["proving"]["samples_per_conjecture"], "negation_base", service,
                                         sampling_seed=config["ladder"]["fresh_sampling_seed"])
    del engine
    verification = _verification_rows(service.results())
    store.write_rows("negation_attempts_base.jsonl", attempts)
    store.write_rows("negation_verification_base.jsonl", verification)
    verified_ids = {row["attempt_id"] for row in verification if row["status"] == "verified"}
    verified_by_conjecture: dict[str, int] = {}
    for attempt in attempts:
        if attempt["attempt_id"] in verified_ids:
            verified_by_conjecture[attempt["statement_id"]] = verified_by_conjecture.get(attempt["statement_id"], 0) + 1
    summary = {**census_summary(rows, verified_by_conjecture), "samples_each": config["proving"]["samples_per_conjecture"],
               "sampling_seed": config["ladder"]["fresh_sampling_seed"], "sampling": stats, "statuses": _status_counts(verification),
               "proofs_per_second": round(service.checked / service.check_seconds, 3) if service.check_seconds else None}
    store.mark_done(CENSUS_MARKER, summary)
    return summary


REACH_MARKER = "base_reach_workbook"


def base_reach(config: dict) -> dict:
    """Last and optional (spec §13b): the BASE model's reach on the workbook-holdout problems it never proved
    in its stored samples, `ladder.reach_samples` samples each with a seed of their own. It defines "unsolved
    by the base model" for later rounds."""
    store = _store()
    if store.is_done(REACH_MARKER):
        return store.done_summary(REACH_MARKER)
    ladder = config["ladder"]
    items, _ = _fresh_sets(store, config)["workbook_unsolved"]
    engine = _engine(config, enable_lora=False)
    service = VerificationService(kimina_settings_from_config(config))
    attempts, stats = _sample_and_verify(engine, config, items, ladder["reach_samples"], "reach_base", service, sampling_seed=ladder["reach_sampling_seed"])
    del engine
    verification = _verification_rows(service.results())
    store.write_rows("reach_attempts_base.jsonl", attempts)
    store.write_rows("reach_verification_base.jsonl", verification)
    verified_ids = {row["attempt_id"] for row in verification if row["status"] == "verified"}
    proved = {attempt["statement_id"] for attempt in attempts if attempt["attempt_id"] in verified_ids}
    summary = {"problems": len(items), "samples_each": ladder["reach_samples"], "sampling_seed": ladder["reach_sampling_seed"],
               "proved_at_least_once": len(proved), "still_unproved": len(items) - len(proved), "sampling": stats,
               "statuses": _status_counts(verification),
               "proofs_per_second": round(service.checked / service.check_seconds, 3) if service.check_seconds else None}
    store.mark_done(REACH_MARKER, summary)
    return summary


def _emit_report_scalars(run: str, step: int, scalars: dict[str, float]) -> None:
    """A report's headline figures as TensorBoard scalars. Our job runner copies TensorBoard runs back while a
    task is still running and everything else only when it ends, so this is how a part's result can be read before
    a long task finishes."""
    root = os.environ.get("RLVR_LEAN_TB_DIR")
    if root and scalars:
        ScalarEventWriter(Path(root) / run).scalars(step, scalars)


# ---------------------------------------------------------------------------------------------- report
def report(config: dict) -> dict:
    """Phase A's report, or with more than Phase A's arm, Phase B's (Phase A's stays as it was written)."""
    from rlvr_lean.reporting.ladder import LADDER_REPORT_MARKER, build_ladder_report, ladder_scalars
    from rlvr_lean.reporting.native_scout import SCOUT_REPORT_MARKER, SCOUT_SEEDS_REPORT_MARKER, build_native_scout_report, scout_scalars
    from rlvr_lean.reporting.phase_a import build_report
    from rlvr_lean.reporting.phase_b import build_phase_b_report

    store = _store()
    runs = _arm_runs(config)
    if any(names.arm == SCOUT_ARM for names in runs):
        # Spec §13a. Its own files and marker: Phase A's and Phase B's reports stay as they were written, and
        # the read over several seeds has files of its own beside the one-seed scout's.
        summary = build_native_scout_report(store, config, _profile(), runs)
        store.mark_done(SCOUT_REPORT_MARKER if len(runs) == 1 else SCOUT_SEEDS_REPORT_MARKER, {"headline": summary["headline"]})
        _emit_report_scalars(f"report_{SCOUT_ARM}_{len(runs)}_seeds", len(runs), scout_scalars(summary))
        return summary
    if any(names.arm == LADDER_ARM for names in runs):
        # Spec §13b. Its own files and marker.
        summary = build_ladder_report(store, config, _profile(), runs)
        store.mark_done(LADDER_REPORT_MARKER, {"headline": summary["headline"]})
        _emit_report_scalars(f"report_{LADDER_ARM}", 1, ladder_scalars(summary))
        return summary
    if _is_phase_b(config):
        summary = build_phase_b_report(store, config, _profile(), _arm_runs(config))
        store.mark_done("report_phase_b", {"headline": summary["headline"]})
        return summary
    summary = build_report(store, config, _profile())
    store.mark_done("report", {"headline": summary["headline"]})
    return summary


STAGES = {
    "prepare_data": prepare_data,
    "generate_conjectures": generate_conjectures,
    "sample_proofs": sample_proofs,
    "score_selection": score_selection,
    "train_adapter": train_adapter,
    "evaluate_sampling": evaluate_sampling,
    "evaluate_loss": evaluate_loss,
    "report": report,
    "ladder_probes": ladder_probes,
    "negation_census": negation_census,
    "base_reach": base_reach,
}
