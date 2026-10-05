"""How a (prompt, proof) pair is laid out in tokens. Spec §6 item 2a.

The base model does not write a proof straight after the prompt: its next token there is the sequence-start
token (its first choice in 113 of 114 held-out proofs, p = 0.994), then the proof, the closing fence and
end-of-sequence. A pair tokenized as one piece of text holds no such token, so the proof's first token is one
the base gives a probability of about e^-12 (the held-out saturation and learning-progress score diagnosis).

  NATIVE  prompt, the sequence-start token, proof. The token is CONTEXT: it is never a target, so the
          target keeps the same tokens and the same count. Training, scoring and the held-out loss use it.
  LEGACY  prompt, proof. Phase A's and Phase B's format. It stays so that everything stored under a legacy
          name keeps meaning what it meant; nothing new is named after it (`domain.training.arms`).
"""

from __future__ import annotations

from typing import Sequence

LEGACY = "legacy"
NATIVE = "native"
TARGET_FORMATS = (LEGACY, NATIVE)


def native_format(ids: Sequence[int], mask: Sequence[bool], bos_token_id: int) -> tuple[list[int], list[bool]]:
    """The same (prompt, target) pair as the model itself writes it: a beginning-of-sequence token between
    the prompt and the proof. The inserted token is context, not a target (its mask is False), so the target
    keeps the same tokens and the same count."""
    flags = list(mask)
    if len(flags) != len(ids):
        raise ValueError(f"ids and mask differ in length: {len(ids)} and {len(flags)}")
    if True not in flags:
        raise ValueError("the pair has no target token")
    boundary = flags.index(True)
    return list(ids[:boundary]) + [bos_token_id] + list(ids[boundary:]), flags[:boundary] + [False] + flags[boundary:]


def in_target_format(ids: Sequence[int], mask: Sequence[bool], target_format: str, bos_token_id: int | None) -> tuple[list[int], list[bool]]:
    """A pair tokenized as one piece of text (the legacy layout), in `target_format`."""
    if target_format == LEGACY:
        return list(ids), list(mask)
    if target_format == NATIVE:
        if bos_token_id is None:
            raise ValueError("the native format needs the tokenizer's sequence-start token id")
        return native_format(ids, mask, bos_token_id)
    raise ValueError(f"unknown target format {target_format!r}; known: {', '.join(TARGET_FORMATS)}")


def is_heldout_read_step(step: int, total_steps: int, every: int, every_step_until: int) -> bool:
    """Whether the held-out loss is read after optimizer step `step` (0 = before training): at step 0, at
    every step up to `every_step_until`, then every `every` steps, and at the last step (spec §13, §13a)."""
    if every < 1 or every_step_until < 0 or total_steps < 0:
        raise ValueError("`every` must be at least 1 and the other two non-negative")
    if not 0 <= step <= total_steps:
        raise ValueError(f"step {step} is outside the run's 0..{total_steps}")
    return step <= every_step_until or step % every == 0 or step == total_steps
