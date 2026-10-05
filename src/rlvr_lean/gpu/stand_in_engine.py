"""A stand-in for the sampling engine: NOT a model. For a pre-flight on a box without a GPU.

`RLVR_LEAN_STAND_IN_ENGINE=1` makes the ladder loop's steps "sample" from the fixed proofs below instead of
loading vLLM. Everything else is the real step: the data files, the negations, the Lean checks through the
configured pin, the stored blocks and the resume, the bookkeeping, the alarm, the report. What it cannot show
is anything about vLLM (the load under the VRAM reserve, n samples of 1,024 tokens, the tokenizer). Every
result it produces is marked `stand_in_engine: true`; point `RLVR_LEAN_STORE` at a scratch directory.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

# Short proofs that close easy goals, so a pre-flight sees verified and failed attempts on both sides.
CANNED_PROOFS = (
    "\n  norm_num\n",
    "\n  simp\n",
    "\n  linarith\n",
    "\n  nlinarith [sq_nonneg (a - b), sq_nonneg (a + b)]\n",
    "\n  positivity\n",
    "\n  omega\n",
    "\n  decide\n",
    "\n  intro x\n  nlinarith [sq_nonneg x]\n",
    "\n  push_neg\n  exact ⟨0, by norm_num⟩\n",
    "\n  intro h\n  have := h 0\n  norm_num at this\n",
)
CAPPED_EVERY = 23            # one sample in this many "runs into the token cap"


@dataclass(frozen=True)
class StandInParameters:
    n: int
    seed: int
    max_tokens: int


@dataclass
class StandInSample:
    text: str
    token_ids: list[int]
    finish_reason: str
    stop_reason: str | None = None


@dataclass
class StandInOutput:
    outputs: list[StandInSample] = field(default_factory=list)


def stand_in_parameters(samples: int, seed: int, max_tokens: int) -> StandInParameters:
    return StandInParameters(n=samples, seed=seed, max_tokens=max_tokens)


class StandInEngine:
    """`generate` with vLLM's shape: one output per prompt, `n` samples each, the same for the same seed.
    `parameters` is one set for every prompt or, as vLLM also takes it, a list with one set for each prompt."""

    def generate(self, prompts: list[str], parameters: StandInParameters | list[StandInParameters], lora_request=None) -> list[StandInOutput]:
        each = list(parameters) if isinstance(parameters, (list, tuple)) else [parameters] * len(prompts)
        if len(each) != len(prompts):
            raise ValueError(f"{len(each)} sets of sampling parameters for {len(prompts)} prompts")
        outputs = []
        for prompt, parameters in zip(prompts, each):
            samples = []
            for index in range(parameters.n):
                draw = int(hashlib.sha256(f"{parameters.seed}:{index}:{prompt}".encode()).hexdigest()[:8], 16)
                if draw % CAPPED_EVERY == 0:
                    samples.append(StandInSample("\n  have h : True := trivial" * 40, list(range(parameters.max_tokens)), "length"))
                else:
                    text = CANNED_PROOFS[draw % len(CANNED_PROOFS)]
                    samples.append(StandInSample(text, list(range(max(1, len(text) // 4))), "stop"))
            outputs.append(StandInOutput(samples))
        return outputs
