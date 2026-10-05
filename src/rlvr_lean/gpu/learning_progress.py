"""The learning-progress score of spec §5, on the GPU.

LP(c) = <g(c), ḡ>, where g(c) is the gradient of conjecture c's mean per-token proof loss and ḡ the mean of
the same gradient over the reward-gradient pairs. One SGD step on c changes the held-out loss by about
-η·LP(c), so a higher score predicts more held-out progress.

The LoRA-initialisation fix: with B = 0 the gradient with respect to A is exactly zero, and the gradient with
respect to B is (α/r)·G·Aᵀ, where G is the gradient with respect to the full weight. PEFT draws each entry of
A (shape r × d_in) with variance 1/(3·d_in), so E[AᵀA] = (r/(3·d_in))·I and, per module,
    <∂B(c), ∂B(h)> ≈ (α/r)² · (r/(3·d_in)) · <G(c), G(h)>.
B-only scoring is therefore a random projection of the FULL-weight first-order influence at the base model.
Each module's term is multiplied by 3·d_in/r so modules of different widths are weighted as in the full
gradient (the remaining constant (α/r)² is the same for every module and cannot change a ranking).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GradientScorer:
    peft_model: object
    tokenizer: object
    max_tokens: int
    # The pair format the gradients are taken in (spec §6 item 2a), stated by every caller. In the legacy format
    # one position (the proof's first token, which the base expects to be its sequence-start token) holds
    # 99.8% of the reference gradient's squared norm, so the score ranks that token and not the proof.
    target_format: str

    def __post_init__(self) -> None:
        # (B parameter, the per-module rescale 3·d_in/r) for every LoRA module.
        named = dict(self.peft_model.named_parameters())
        self.b_parameters = []
        for name, parameter in named.items():
            if "lora_B" in name:
                a_weight = named[name.replace("lora_B", "lora_A")]
                rank, d_in = a_weight.shape
                self.b_parameters.append((parameter, 3.0 * d_in / rank))

    def gradient(self, prompt: str, target: str) -> list:
        """B-gradients of the mean proof-token loss, each multiplied by sqrt(rescale), so a plain dot product of
        two such lists is the rescaled score and a plain norm is the rescaled norm."""
        from rlvr_lean.gpu.model_utils import encode_pair, mean_proof_token_loss

        ids, mask = encode_pair(self.tokenizer, prompt, target, self.max_tokens, self.target_format)
        self.peft_model.zero_grad(set_to_none=True)
        mean_proof_token_loss(self.peft_model, ids, mask).backward()
        return [parameter.grad.detach().float().flatten() * math.sqrt(scale) for parameter, scale in self.b_parameters]

    def mean_gradient(self, pairs: list[tuple[str, str]]) -> list:
        total = None
        for prompt, target in pairs:
            gradient = self.gradient(prompt, target)
            total = gradient if total is None else [accumulated + part for accumulated, part in zip(total, gradient)]
        if total is None:
            raise ValueError("no reward-gradient pairs to average")
        return [part / len(pairs) for part in total]


def dot(first: list, second: list) -> float:
    return float(sum((a * b).sum() for a, b in zip(first, second)))


def norm(vector: list) -> float:
    return math.sqrt(dot(vector, vector))


def score(gradient: list, reference: list, reference_norm: float) -> tuple[float, float]:
    """(dot score, cosine score) of one conjecture against a cached mean held-out gradient."""
    value = dot(gradient, reference)
    gradient_norm = norm(gradient)
    cosine = value / (gradient_norm * reference_norm) if gradient_norm and reference_norm else 0.0
    return value, cosine
