"""SimplexNorm (Noh et al., Eq. 1) — half of the paper's contribution.

    z_hat = [p_1, ..., p_L],   (p_i)_j = exp(z_ij / tau) / sum_k exp(z_ik / tau)

The representation of dimension d is cut into L partitions of V, and a softmax
at temperature tau is applied *within each partition independently*. Every
partition becomes a point on a probability simplex: non-negative and summing to
one. That induces sparsity without imposing a hard constraint — the paper calls
it a soft version of VQ-VAE's discrete codes.

The dimensions are scientific invariants (CLAUDE.md): d=128, L=16, V=8, tau=1.
These tests pin the mathematics, not an implementation, so they stay valid if
the module is rewritten.
"""

import pytest
import torch

from src.models.norms import SimplexNorm

D, V, L = 128, 8, 16
B = 6


def test_every_partition_is_a_probability_distribution():
    """The defining property: each of the 16 simplices sums to 1, nothing negative."""
    norm = SimplexNorm(D, per_partition=V)
    out = norm(torch.randn(B, D) * 5)

    assert out.shape == (B, D)
    assert torch.all(out >= 0.0)

    partitions = out.view(B, L, V)
    assert torch.allclose(partitions.sum(dim=-1), torch.ones(B, L), atol=1e-5)


def test_the_invariant_dimensions_give_sixteen_partitions_of_eight():
    """d=128, V=8 implies L=16. If this changes, the contribution changed."""
    norm = SimplexNorm(D, per_partition=V)
    assert norm.partitions == L
    assert norm.per_partition == V


def test_partitions_are_normalised_independently_of_each_other():
    """A softmax over the whole vector would couple them and destroy the point.

    The perturbation is to a *single element*, not a whole partition: softmax is
    shift-invariant, so adding a constant across a partition correctly changes
    nothing (pinned separately below).
    """
    norm = SimplexNorm(D, per_partition=V)
    x = torch.randn(1, D)
    before = norm(x).view(1, L, V)

    perturbed = x.clone()
    perturbed[0, 0] += 50.0  # one element of partition 0
    after = norm(perturbed).view(1, L, V)

    assert not torch.allclose(before[0, 0], after[0, 0]), "partition 0 should change"
    assert torch.allclose(before[0, 1:], after[0, 1:], atol=1e-6), (
        "every other partition must be untouched; a global softmax would couple them"
    )


def test_a_partition_is_invariant_to_a_constant_shift():
    """Softmax depends only on differences within a partition.

    Worth pinning: it means the LayerNorm immediately before this cannot change
    the representation through its shift (beta) parameter alone, only through
    its scale — which is part of why the paper's stated order matters.
    """
    norm = SimplexNorm(D, per_partition=V)
    x = torch.randn(1, D)
    shifted = x.clone()
    shifted[0, 0:V] += 50.0  # a constant added to every element of partition 0

    assert torch.allclose(norm(x), norm(shifted), atol=1e-5)


def test_a_lower_temperature_produces_a_sparser_representation():
    """tau is what makes SimplexNorm 'soft'; tau -> 0 approaches a one-hot code."""
    x = torch.randn(B, D)
    cold = SimplexNorm(D, per_partition=V, tau=0.1)(x).view(B, L, V)
    paper = SimplexNorm(D, per_partition=V, tau=1.0)(x).view(B, L, V)
    hot = SimplexNorm(D, per_partition=V, tau=5.0)(x).view(B, L, V)

    assert cold.max(dim=-1).values.mean() > paper.max(dim=-1).values.mean()
    assert paper.max(dim=-1).values.mean() > hot.max(dim=-1).values.mean()


def test_a_high_temperature_approaches_the_uniform_distribution():
    norm = SimplexNorm(D, per_partition=V, tau=1000.0)
    out = norm(torch.randn(B, D)).view(B, L, V)
    assert torch.allclose(out, torch.full_like(out, 1.0 / V), atol=1e-2)


def test_gradients_flow_through_the_normalisation():
    """It sits mid-network; a detached or zero-gradient norm would stop training."""
    norm = SimplexNorm(D, per_partition=V)
    x = torch.randn(B, D, requires_grad=True)
    norm(x).sum().backward()
    assert x.grad is not None
    assert torch.any(x.grad != 0)


def test_large_inputs_do_not_overflow():
    """Fallback ladder rung 4 blames this softmax first when NaNs appear."""
    norm = SimplexNorm(D, per_partition=V)
    out = norm(torch.full((B, D), 1e4))
    assert torch.isfinite(out).all()
    assert torch.allclose(out.view(B, L, V).sum(dim=-1), torch.ones(B, L), atol=1e-5)


def test_it_works_on_a_half_width_representation():
    """Each modality is normalised before the two halves are concatenated (B.1.3)."""
    norm = SimplexNorm(D // 2, per_partition=V)
    assert norm.partitions == L // 2
    out = norm(torch.randn(B, D // 2))
    assert torch.allclose(
        out.view(B, L // 2, V).sum(dim=-1), torch.ones(B, L // 2), atol=1e-5
    )


# -- construction guards ----------------------------------------------------


def test_a_dimension_not_divisible_by_the_partition_size_is_refused():
    """Silently dropping a remainder would mis-shape the representation."""
    with pytest.raises(ValueError, match="divisible"):
        SimplexNorm(130, per_partition=V)


def test_a_non_positive_temperature_is_refused():
    with pytest.raises(ValueError):
        SimplexNorm(D, per_partition=V, tau=0.0)
    with pytest.raises(ValueError):
        SimplexNorm(D, per_partition=V, tau=-1.0)


def test_the_module_reports_its_configuration():
    """The dimensions are invariants, so they must be visible in a repr and a log."""
    text = repr(SimplexNorm(D, per_partition=V))
    assert "128" in text and "16" in text and "8" in text
