"""Shared helpers for the PEFT side (scoring, training, held-out loss): tokenizing a (prompt, proof) pair with
a mask over the PROOF tokens, loading the NF4 base with a LoRA adapter, and the mean per-token loss."""

from __future__ import annotations

from pathlib import Path

from rlvr_lean.domain.proving import COMPLETION_STOP, build_prover_prompt


def training_text(statement: str, completion: str) -> tuple[str, str]:
    """(prompt, target): the target is the proof followed by the closing code fence, which is where the
    model's own generation stops, so training teaches it to stop there too."""
    return build_prover_prompt(statement), completion.rstrip() + "\n" + COMPLETION_STOP


def encode_with_proof_mask(tokenizer, prompt: str, target: str, max_tokens: int) -> tuple[list[int], list[bool]]:
    """Token ids of prompt + target and a mask that is True on tokens belonging to the target.

    Decided by character offsets, not by tokenizing the prompt separately: byte-level BPE can merge the
    prompt's final newline with the proof's leading spaces into one token, which shifts a length-based
    boundary by one."""
    ids, mask = _tokenize_as_one_text(tokenizer, prompt, target)
    return ids[:max_tokens], mask[:max_tokens]


def _tokenize_as_one_text(tokenizer, prompt: str, target: str) -> tuple[list[int], list[bool]]:
    text = prompt + target
    encoding = tokenizer(text, return_offsets_mapping=True)
    ids, offsets = encoding["input_ids"], encoding["offset_mapping"]
    return list(ids), [end > len(prompt) and end > start for start, end in offsets]


def encode_pair(tokenizer, prompt: str, target: str, max_tokens: int, target_format: str) -> tuple[list[int], list[bool]]:
    """The pair in `target_format` (spec §6 item 2a): the ONE encoding training, scoring and the held-out loss
    share. Native puts the tokenizer's sequence-start token between prompt and proof, as context (mask False);
    legacy is `encode_with_proof_mask`, kept for what is stored under legacy names."""
    from rlvr_lean.domain.training.target_format import NATIVE, in_target_format

    ids, mask = _tokenize_as_one_text(tokenizer, prompt, target)
    bos_token_id = tokenizer.bos_token_id if target_format == NATIVE else None
    ids, mask = in_target_format(ids, mask, target_format, bos_token_id)
    return ids[:max_tokens], mask[:max_tokens]


def load_nf4_base(nf4_directory: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(nf4_directory))
    model = AutoModelForCausalLM.from_pretrained(str(nf4_directory), device_map={"": 0}, dtype=torch.bfloat16)
    model.config.use_cache = False
    return model, tokenizer


def attach_lora(model, lora: dict, seed: int, trainable_a: bool = True):
    """PEFT's default initialisation (A Kaiming-uniform, B zero), drawn from `seed`."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    torch.manual_seed(seed)
    peft_model = get_peft_model(model, LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                                  target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM"))
    if not trainable_a:
        for name, parameter in peft_model.named_parameters():
            if "lora_A" in name:
                parameter.requires_grad_(False)
    return peft_model


def mean_proof_token_loss(model, ids: list[int], mask: list[bool]):
    """Mean negative log-likelihood over the target's tokens (the per-token normalisation of spec §5)."""
    import torch

    input_ids = torch.tensor([ids], device="cuda")
    labels = torch.tensor([[token if keep else -100 for token, keep in zip(ids, mask)]], device="cuda")
    return model(input_ids=input_ids, labels=labels).loss


def position_losses(model, ids: list[int], mask: list[bool], device: str = "cuda") -> tuple[list[float], list[int], list[float]]:
    """(loss, the model's own first choice, that choice's log-probability) at every target position, from one
    forward pass. The mean of the losses is `mean_proof_token_loss` for the same pair (measured: the two agree
    to four decimals on all 114 held-out pairs under the base and twelve adapters)."""
    import torch

    input_ids = torch.tensor([ids], device=device)
    logits = model(input_ids=input_ids).logits[0]
    positions = torch.tensor([position for position in range(1, len(ids)) if mask[position]], device=logits.device)
    log_probabilities = torch.log_softmax(logits[positions - 1].float(), dim=-1)
    targets = input_ids[0, positions]
    losses = -log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    top_value, top_id = log_probabilities.max(dim=1)
    return losses.tolist(), top_id.tolist(), top_value.tolist()
