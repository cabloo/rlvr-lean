"""Milestone 2 steps: environments are built by the entry shim; these make the checkpoints and prove that
sampling, training and serving an adapter all work on the RTX 5080. Spec, Milestone 2 acceptance:
FP8 and NF4 checkpoints saved; vLLM samples from the FP8 base and from FP8 + a throwaway adapter (outputs
differ at temperature 0); one QLoRA step runs; tokens/s, peak memory and proof-length distribution reported.

Every step skips work whose output already exists in the store, so a rerun resumes.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

import yaml

from rlvr_lean.domain.proving import build_prover_prompt, completion_from_output
from rlvr_lean.domain.verification import (
    build_proof_source,
    find_forbidden_token,
    rejected_lexically,
)

STORE = Path(os.environ.get("RLVR_LEAN_STORE", "/root/rlvr_lean_store"))
BF16_DIR = STORE / "models" / "base-bf16"
FP8_DIR = STORE / "models" / "base-fp8"
NF4_DIR = STORE / "models" / "base-nf4"
ADAPTER_DIR = STORE / "adapters" / "m2-throwaway"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json")


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _directory_gb(directory: Path) -> float:
    return round(sum(f.stat().st_size for f in directory.rglob("*") if f.is_file()) / 1e9, 2)


def _fetch_cached(url: str, name: str) -> Path:
    target = STORE / "data" / name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, target)
    return target


def _minif2f_rows(config: dict, split: str) -> list[dict]:
    commit = config["data"]["deepseek_prover_commit"]
    path = _fetch_cached(f"https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Prover-V1.5/{commit}/datasets/minif2f.jsonl",
                         "minif2f.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted((row for row in rows if row["split"] == split), key=lambda row: row["name"])


def _few_shot_proofs(config: dict) -> list[dict]:
    commit = config["data"]["deepseek_prover_commit"]
    path = _fetch_cached(f"https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Prover-V1.5/{commit}/datasets/minif2f_valid_few_shot.jsonl",
                         "minif2f_valid_few_shot.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def download(config: dict) -> dict:
    from huggingface_hub import snapshot_download

    if not (BF16_DIR / "config.json").exists():
        snapshot_download(repo_id=config["model"]["hf_id"], revision=config["model"]["revision"], local_dir=str(BF16_DIR))
    return {"path": str(BF16_DIR), "size_gb": _directory_gb(BF16_DIR), "revision": config["model"]["revision"]}


def _safetensors_tensor_names(directory: Path) -> list[str]:
    """Tensor names from every shard's JSON header (8-byte little-endian length, then JSON): no torch needed."""
    names = []
    for shard in sorted(directory.glob("*.safetensors")):
        with shard.open("rb") as handle:
            header_length = int.from_bytes(handle.read(8), "little")
            names.extend(key for key in json.loads(handle.read(header_length)) if key != "__metadata__")
    return names


def quantize_fp8(config: dict) -> dict:
    """FP8 weights for vLLM, made once. Data-free (FP8_DYNAMIC: per-channel weights, per-token activations).

    Only the transformer's linear layers may be quantized: vLLM loads the token embedding and the output head
    in full precision, and refuses a checkpoint that carries scales for them (measured: the shard-by-shard
    exporter quantized `embed_tokens` when only `lm_head` was excluded). The recipe is recorded beside the
    checkpoint, so a changed recipe rebuilds it, and the result is checked for that mistake before use."""
    scheme = config["quantization"]["fp8_scheme"]
    ignore = list(config["quantization"]["fp8_ignore"])
    recipe = {"scheme": scheme, "ignore": ignore}
    recipe_file = FP8_DIR / "rlvr_lean_recipe.json"
    if FP8_DIR.exists() and (not recipe_file.exists() or json.loads(recipe_file.read_text()) != recipe):
        shutil.rmtree(FP8_DIR)
    method = "already present"
    if not (FP8_DIR / "config.json").exists():
        try:
            # Processes one safetensors shard at a time instead of loading the whole model into RAM.
            from llmcompressor import model_free_ptq
        except ImportError:
            model_free_ptq = None
        if model_free_ptq is not None:
            model_free_ptq(model_stub=str(BF16_DIR), save_directory=str(FP8_DIR), scheme=scheme,
                           ignore=ignore, device="cuda:0")
            method = "model_free_ptq"
        else:
            from llmcompressor import oneshot
            from llmcompressor.modifiers.quantization import QuantizationModifier
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(str(BF16_DIR), dtype="auto")
            oneshot(model=model, recipe=QuantizationModifier(targets="Linear", scheme=scheme, ignore=ignore))
            model.save_pretrained(str(FP8_DIR), save_compressed=True)
            method = "oneshot"
        recipe_file.write_text(json.dumps(recipe))
    for name in TOKENIZER_FILES:                       # vLLM loads the tokenizer from the model directory
        if (BF16_DIR / name).exists() and not (FP8_DIR / name).exists():
            shutil.copy2(BF16_DIR / name, FP8_DIR / name)
    quantization_config = json.loads((FP8_DIR / "config.json").read_text()).get("quantization_config", {})
    tensor_names = _safetensors_tensor_names(FP8_DIR)
    full_precision_violations = [name for name in tensor_names
                                 if ("embed_tokens" in name or "lm_head" in name) and not name.endswith(".weight")]
    scales = sum(name.endswith("weight_scale") for name in tensor_names)
    return {"path": str(FP8_DIR), "size_gb": _directory_gb(FP8_DIR), "method": method, "recipe": recipe,
            "quant_method": quantization_config.get("quant_method"), "format": quantization_config.get("format"),
            "quantized_weight_count": scales, "embedding_or_head_quantized": full_precision_violations[:5],
            "ok": bool(quantization_config) and scales > 0 and not full_precision_violations}


def quantize_nf4(config: dict) -> dict:
    """The 4-bit NF4 base QLoRA trains on, quantized once and saved so later runs load it directly."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if not (NF4_DIR / "config.json").exists():
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                          bnb_4bit_use_double_quant=config["quantization"]["nf4_double_quant"],
                                          bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(str(BF16_DIR), quantization_config=quantization,
                                                     device_map={"": 0}, dtype=torch.bfloat16)
        model.save_pretrained(str(NF4_DIR))
    # Tokenizer files are copied from the pristine download, never re-saved through transformers: 5.x
    # misloads this tokenizer, and saving it would write the broken version (see fix_tokenizers).
    for name in TOKENIZER_FILES:
        if (BF16_DIR / name).exists():
            shutil.copy2(BF16_DIR / name, NF4_DIR / name)
    return {"path": str(NF4_DIR), "size_gb": _directory_gb(NF4_DIR)}


def fix_tokenizers(config: dict) -> dict:
    """Give each checkpoint a tokenizer that transformers 5.x loads correctly, and prove it: the loaded
    tokenizer must reproduce [BOS] + the model's own tokenizer.json encoding and decode back exactly.
    Measured in Milestone 2 run 3: without this, every space and newline was dropped from every prompt."""
    from rlvr_lean.infrastructure.tokenizer_files import check_tokenizer, fix_tokenizer_files

    results = {}
    for directory in (FP8_DIR, NF4_DIR):
        for name in TOKENIZER_FILES:                       # start from the pristine download every time
            if (BF16_DIR / name).exists():
                shutil.copy2(BF16_DIR / name, directory / name)
        (directory / "tokenizer.original.json").unlink(missing_ok=True)
        fix_tokenizer_files(directory)
        results[directory.name] = check_tokenizer(directory)
    return {"checks": results, "ok": all(check["ok"] for check in results.values())}


def _vllm_engine(config: dict, enable_lora: bool):
    # vLLM reads these when the engine starts, and its engine process inherits them.
    os.environ.update({key: str(value) for key, value in config["vllm"].get("environment", {}).items()})
    from vllm import LLM

    from rlvr_lean.gpu.memory_budget import gpu_total_mib, sampling_memory_fraction

    fraction = sampling_memory_fraction(gpu_total_mib(), config["gpu"]["desktop_reserve_gb"],
                                        config["gpu"]["vllm_overhead_gb"], config["vllm"]["gpu_memory_utilization"])
    return LLM(model=str(FP8_DIR), max_model_len=config["vllm"]["max_model_len"],
               gpu_memory_utilization=fraction, seed=0,
               enable_lora=enable_lora, max_lora_rank=config["vllm"]["max_lora_rank"], max_loras=1)


def _percentiles(values: list[int]) -> dict:
    ordered = sorted(values)

    def at(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {"min": ordered[0], "p25": at(0.25), "median": at(0.5), "p75": at(0.75), "p95": at(0.95), "max": ordered[-1]}


def _verify(statements_and_completions: list[tuple[str, str, str]], config: dict) -> dict[str, str]:
    """attempt id -> status, through the live Kimina server (spec §4 order: lexical filter first)."""
    import asyncio

    from rlvr_lean.infrastructure.kimina_client import KiminaVerifier, LeanSnippet
    from rlvr_lean.infrastructure.verification_service import kimina_settings_from_config

    settings = kimina_settings_from_config(config)      # the endpoint and the rules of the config's Lean pin
    statuses, snippets = {}, []
    for attempt_id, statement, completion in statements_and_completions:
        token = find_forbidden_token(completion)
        if token is not None:
            statuses[attempt_id] = rejected_lexically(attempt_id, token).status.value
        else:
            snippets.append(LeanSnippet(attempt_id, settings.pin.source(build_proof_source(statement, completion))))

    async def run() -> list[dict]:
        async with KiminaVerifier(settings) as verifier:
            return await verifier.check(snippets)

    for raw in asyncio.run(run()):
        statuses[raw["id"]] = settings.pin.classify(raw["id"], raw).status.value
    return statuses


def sample_smoke(config: dict) -> dict:
    """Sample the FP8 base on miniF2F-VALID (the development set), measure throughput and proof lengths, and
    verify every sample so the pipeline's verification path runs end to end."""
    from vllm import SamplingParams

    settings = config["milestone2"]
    rows = _minif2f_rows(config, "valid")[: settings["smoke_problems"]]
    prompts = [build_prover_prompt(row["formal_statement"]) for row in rows]
    engine = _vllm_engine(config, enable_lora=False)
    parameters = SamplingParams(n=settings["smoke_samples_per_problem"], temperature=config["sampling"]["temperature"],
                                top_p=config["sampling"]["top_p"], max_tokens=config["sampling"]["max_new_tokens"],
                                stop=config["sampling"]["stop"], seed=0)
    started = time.monotonic()
    outputs = engine.generate(prompts, parameters)
    generation_seconds = time.monotonic() - started
    attempts, token_counts, hit_cap = [], [], 0
    for row, output in zip(rows, outputs):
        for index, sample in enumerate(output.outputs):
            token_counts.append(len(sample.token_ids))
            hit_cap += sample.finish_reason == "length"
            attempts.append((f"{row['name']}#{index}", row["formal_statement"], completion_from_output(sample.text)))
    del engine
    statuses = _verify(attempts, config)
    by_problem: dict[str, list[bool]] = {}
    for attempt_id, _, _ in attempts:
        by_problem.setdefault(attempt_id.split("#")[0], []).append(statuses[attempt_id] == "verified")
    samples_path = Path(os.environ.get("RLVR_LEAN_STEP_DIR", STORE / "outputs")) / "sample_smoke_samples.jsonl"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with samples_path.open("w") as handle:
        for (attempt_id, _, completion), tokens in zip(attempts, token_counts):
            handle.write(json.dumps({"attempt_id": attempt_id, "status": statuses[attempt_id], "tokens": tokens,
                                     "completion": completion}) + "\n")
    status_counts: dict[str, int] = {}
    for status in statuses.values():
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "problems": len(rows), "samples": len(attempts),
        "generated_tokens": sum(token_counts), "generation_seconds": round(generation_seconds, 1),
        "tokens_per_second": round(sum(token_counts) / generation_seconds, 1),
        "tokens_per_sample": _percentiles(token_counts), "samples_hitting_max_tokens": hit_cap,
        "verification_statuses": status_counts,
        "dev_pass_at_1": round(sum(sum(v) / len(v) for v in by_problem.values()) / len(by_problem), 4),
        "dev_pass_at_n": round(sum(any(v) for v in by_problem.values()) / len(by_problem), 4),
        "samples_file": str(samples_path),
    }


def qlora_step(config: dict) -> dict:
    """One optimizer step of QLoRA on the saved NF4 base, then save the adapter. The adapter is initialised
    with RANDOM B (not PEFT's zero default) on purpose: it is a throwaway whose only job is to change the
    model's outputs visibly, so the adapter check can show vLLM really applies it."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(NF4_DIR))
    model = AutoModelForCausalLM.from_pretrained(str(NF4_DIR), device_map={"": 0}, dtype=torch.bfloat16)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = config["lora"]
    model = get_peft_model(model, LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                             target_modules=lora["target_modules"], bias="none",
                                             task_type="CAUSAL_LM", init_lora_weights=False))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["training"]["learning_rate"])
    micro_batches = config["milestone2"]["qlora_micro_batches"]
    examples = _few_shot_proofs(config)[:micro_batches]
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    losses, sequence_lengths = [], []
    for example in examples:
        prompt = build_prover_prompt(example["formal_statement"])
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
        full_ids = tokenizer(prompt + example["formal_proof"] + "\n```", return_tensors="pt").input_ids
        full_ids = full_ids[:, : config["training"]["max_sequence_tokens"]]
        labels = full_ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100                  # loss on proof tokens only
        loss = model(input_ids=full_ids.to(0), labels=labels.to(0)).loss / len(examples)
        loss.backward()
        losses.append(float(loss) * len(examples))
        sequence_lengths.append(int(full_ids.shape[1]))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    step_seconds = time.monotonic() - started
    peak_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    return {"trainable_parameters": trainable, "micro_batches": len(examples), "sequence_lengths": sequence_lengths,
            "losses": [round(value, 4) for value in losses], "step_seconds": round(step_seconds, 1),
            "peak_torch_allocated_gb": round(peak_allocated_gb, 2), "adapter_path": str(ADAPTER_DIR),
            "adapter_files": sorted(p.name for p in ADAPTER_DIR.iterdir())}


def adapter_check(config: dict) -> dict:
    """Greedy outputs from the FP8 base with and without the throwaway adapter must differ: proof that vLLM
    loads a PEFT adapter trained on the NF4 base and applies it on the FP8 base."""
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    settings = config["milestone2"]
    rows = _minif2f_rows(config, "valid")[: settings["adapter_check_prompts"]]
    prompts = [build_prover_prompt(row["formal_statement"]) for row in rows]
    engine = _vllm_engine(config, enable_lora=True)
    parameters = SamplingParams(temperature=0.0, max_tokens=settings["adapter_check_max_tokens"], stop=config["sampling"]["stop"])
    base = engine.generate(prompts, parameters)
    adapted = engine.generate(prompts, parameters, lora_request=LoRARequest("m2_throwaway", 1, str(ADAPTER_DIR)))
    differing = sum(b.outputs[0].text != a.outputs[0].text for b, a in zip(base, adapted))
    return {"prompts": len(prompts), "outputs_differing": differing, "ok": differing >= 1,
            "example_base": base[0].outputs[0].text[:300], "example_adapted": adapted[0].outputs[0].text[:300]}


STEPS = {
    "download": download,
    "quantize_fp8": quantize_fp8,
    "quantize_nf4": quantize_nf4,
    "fix_tokenizers": fix_tokenizers,
    "sample_smoke": sample_smoke,
    "qlora_step": qlora_step,
    "adapter_check": adapter_check,
}
