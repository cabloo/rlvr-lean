"""Make a model directory's tokenizer load correctly under transformers 5.x, and prove it.

Measured 2026-10-03 (Milestone 2): DeepSeek-Prover-V1.5's `tokenizer_config.json` names
`LlamaTokenizerFast`, and transformers 5.17 rebuilds that class as a sentencepiece-style Llama tokenizer
that ignores the model's byte-level BPE: every space and newline vanished and `ℝ`/`₀` were lost. vLLM uses
the same loader, so the model was prompted with garbage. The model's own `tokenizer.json` is correct, and
the generic fast tokenizer uses it as-is, but its post-processor adds no beginning-of-sequence token, which
the model was trained with (`add_bos_token: true`).

The fix, applied to every model directory we load: declare `PreTrainedTokenizerFast`, and append a
template post-processor that prepends BOS. `check_tokenizer` then compares against the ground truth: the
`tokenizers` library loading the ORIGINAL tokenizer.json, plus BOS.
"""

from __future__ import annotations

import json
from pathlib import Path

FIX_MARKER = "rlvr_lean_tokenizer_fix"
REFERENCE_TEXT = ("Complete the following Lean 4 code:\n\n```lean4\nimport Mathlib\nimport Aesop\n\n"
                  "theorem t (x : ℝ) (h₀ : x + 1 = 2) : x = 1 := by\n  nlinarith [sq_nonneg (x - 1)]\n")


def _bos_token(tokenizer_json: dict, config: dict) -> tuple[str, int]:
    bos = config.get("bos_token")
    content = bos.get("content") if isinstance(bos, dict) else bos
    for token in tokenizer_json.get("added_tokens", []):
        if token["content"] == content:
            return content, token["id"]
    raise ValueError(f"BOS token {content!r} not found among tokenizer.json added_tokens")


def fix_tokenizer_files(model_directory: Path) -> dict:
    """Idempotent. Keeps the original tokenizer.json as tokenizer.original.json (the ground truth)."""
    tokenizer_path = model_directory / "tokenizer.json"
    original_path = model_directory / "tokenizer.original.json"
    config_path = model_directory / "tokenizer_config.json"
    if not original_path.exists():
        original_path.write_text(tokenizer_path.read_text())
    original = json.loads(original_path.read_text())
    config = json.loads(config_path.read_text())
    bos_content, bos_id = _bos_token(original, config)
    fixed = dict(original)
    fixed["post_processor"] = {"type": "Sequence", "processors": [
        original["post_processor"],
        {"type": "TemplateProcessing",
         "single": [{"SpecialToken": {"id": bos_content, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
         "pair": [{"SpecialToken": {"id": bos_content, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                  {"Sequence": {"id": "B", "type_id": 1}}],
         "special_tokens": {bos_content: {"id": bos_content, "ids": [bos_id], "tokens": [bos_content]}}},
    ]}
    tokenizer_path.write_text(json.dumps(fixed, ensure_ascii=False))
    config.update({"tokenizer_class": "PreTrainedTokenizerFast", FIX_MARKER: 1})
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    return {"directory": str(model_directory), "bos_token": bos_content, "bos_id": bos_id}


def check_tokenizer(model_directory: Path) -> dict:
    """transformers' load of the directory must equal [BOS] + the original tokenizer.json's encoding, and
    decode back to the text. Needs `transformers` and `tokenizers` (present in both GPU environments)."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    truth = Tokenizer.from_file(str(model_directory / "tokenizer.original.json"))
    truth_ids = truth.encode(REFERENCE_TEXT).ids
    loaded = AutoTokenizer.from_pretrained(str(model_directory))
    ids = loaded(REFERENCE_TEXT).input_ids
    bos_id = loaded.bos_token_id
    decoded = loaded.decode(ids, skip_special_tokens=True)
    ok = ids == [bos_id] + truth_ids and decoded == REFERENCE_TEXT
    return {"ok": ok, "class": type(loaded).__name__, "bos_id": bos_id, "token_count": len(ids),
            "first_ids": ids[:8], "expected_first_ids": ([bos_id] + truth_ids)[:8], "round_trip": decoded == REFERENCE_TEXT}
