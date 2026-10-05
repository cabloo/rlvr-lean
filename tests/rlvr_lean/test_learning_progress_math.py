"""Spec fixture 4: the B-only learning-progress score on a toy linear layer (spec §5).

With W' = W + (α/r)·B·A, B = 0 and A drawn as PEFT draws it (uniform, variance 1/(3·d_in)):
  1. the gradient with respect to A is exactly zero, and the gradient with respect to B is (α/r)·G·Aᵀ,
     so <∂B(c), ∂B(h)> = (α/r)²·tr(G(c)·AᵀA·G(h)ᵀ) exactly (G = the full-weight gradient);
  2. E[AᵀA] = (r / (3·d_in))·I, so the rescaled B-only score is an UNBIASED estimate of <G(c), G(h)>;
  3. at large rank one draw is already close: the error is a small fraction of ‖G(c)‖·‖G(h)‖ (random
     projection), which is the scale the tolerance must be stated in when the two gradients are nearly
     orthogonal and their dot product is small.
Runs where torch is installed (the repo's standard environment); skips elsewhere.
"""

import math

import pytest

torch = pytest.importorskip("torch")


def peft_like_a(rank, d_in, generator):
    bound = 1.0 / math.sqrt(d_in)                 # kaiming_uniform_(a=sqrt(5)) on a (rank, d_in) weight
    return (torch.rand(rank, d_in, generator=generator, dtype=torch.float64) * 2 - 1) * bound


def gradients(weight, a, alpha, inputs, direction):
    """Gradients with respect to A and B of sum(direction * (W' x)) at B = 0, and the full-weight gradient."""
    rank = a.shape[0]
    a = a.clone().requires_grad_(True)
    b = torch.zeros(weight.shape[0], rank, dtype=torch.float64, requires_grad=True)
    (direction * (inputs @ (weight + (alpha / rank) * b @ a).T)).sum().backward()
    return a.grad, b.grad, direction.T @ inputs


def toy_problem(seed, d_out=8, d_in=16):
    generator = torch.Generator().manual_seed(seed)

    def draw(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    return generator, draw(d_out, d_in), (draw(5, d_in), draw(5, d_out)), (draw(5, d_in), draw(5, d_out))


def test_exact_identities_at_initialisation():
    generator, weight, first, second = toy_problem(0)
    alpha, rank = 32.0, 16
    a = peft_like_a(rank, weight.shape[1], generator)
    first_a_grad, first_b_grad, first_full = gradients(weight, a, alpha, *first)
    _, second_b_grad, second_full = gradients(weight, a, alpha, *second)
    assert torch.count_nonzero(first_a_grad) == 0
    assert torch.allclose(first_b_grad, (alpha / rank) * first_full @ a.T)
    exact = (alpha / rank) ** 2 * torch.trace(first_full @ a.T @ a @ second_full.T)
    assert torch.allclose((first_b_grad * second_b_grad).sum(), exact)


def test_the_projection_is_unbiased():
    """Mean of AᵀA over many draws -> (r / (3·d_in))·I. Tolerance: six standard errors of the mean, from the
    uniform distribution's moments (diagonal entry sd per draw sqrt(4r/(45·d²)), off-diagonal sqrt(r)/(3d))."""
    generator = torch.Generator().manual_seed(1)
    rank, d_in, draws = 16, 12, 4000
    total = torch.zeros(d_in, d_in, dtype=torch.float64)
    for _ in range(draws):
        a = peft_like_a(rank, d_in, generator)
        total += a.T @ a
    mean = total / draws
    expected = rank / (3 * d_in)
    diagonal_tolerance = 6 * math.sqrt(4 * rank / (45 * d_in ** 2)) / math.sqrt(draws)
    off_diagonal_tolerance = 6 * (math.sqrt(rank) / (3 * d_in)) / math.sqrt(draws)
    assert torch.max(torch.abs(torch.diagonal(mean) - expected)) < diagonal_tolerance
    off_diagonal = mean - torch.diag(torch.diagonal(mean))
    assert torch.max(torch.abs(off_diagonal)) < off_diagonal_tolerance


def test_a_large_rank_single_draw_is_close_on_the_norm_scale():
    generator, weight, first, second = toy_problem(2)
    alpha, rank = 32.0, 8192
    d_in = weight.shape[1]
    a = peft_like_a(rank, d_in, generator)
    _, first_b_grad, first_full = gradients(weight, a, alpha, *first)
    _, second_b_grad, second_full = gradients(weight, a, alpha, *second)
    estimate = float((first_b_grad * second_b_grad).sum()) * (3 * d_in / rank) * (rank / alpha) ** 2
    truth = float((first_full * second_full).sum())
    norm_scale = float(first_full.norm() * second_full.norm())
    assert abs(estimate - truth) < 0.05 * norm_scale
