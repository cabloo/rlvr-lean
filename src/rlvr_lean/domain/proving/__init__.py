"""Proving context: the prompt the prover model is given for one statement.

The model's non-CoT whole-proof prompt, exactly as DeepSeek-Prover-V1.5's `prover/utils.py` builds it
(read 2026-10-02), with no informal prefix (spec §4). The completion is everything after it, up to the
closing code fence.
"""

from __future__ import annotations

from rlvr_lean.domain.verification.lean_source import LEAN_HEADER

PROMPT_PREFIX = "Complete the following Lean 4 code:\n\n```lean4\n"
COMPLETION_STOP = "```"


def build_prover_prompt(statement: str) -> str:
    """`statement` is the theorem up to and including `:= by` (and its newline)."""
    statement_text = statement if statement.endswith("\n") else statement + "\n"
    return PROMPT_PREFIX + LEAN_HEADER + statement_text


def completion_from_output(generated_text: str) -> str:
    """The proof body: generated text up to the closing fence (vLLM's stop string normally removes it)."""
    return generated_text.split(COMPLETION_STOP, 1)[0]
