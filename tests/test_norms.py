import pytest
import torch

from src.models.norms import SimplexNorm

D, V, L = 128, 8, 16
B = 6


def test_every_partition_is_a_probability_distribution():
    norm = SimplexNorm(D, per_partition=V)
    out = norm(torch.randn(B, D) * 5)

    assert out.shape == (B, D)
    assert torch.all(out >= 0.0)

    partitions = out.view(B, L, V)
    assert torch.allclose(partitions.sum(dim=-1), torch.ones(B, L), atol=1e-5)


def test_the_invariant_dimensions_give_sixteen_partitions_of_eight():
    norm = SimplexNorm(D, per_partition=V)
    assert norm.partitions == L
    assert norm.per_partition == V


def test_partitions_are_normalised_independently_of_each_other():
    norm = SimplexNorm(D, per_partition=V)
    x = torch.randn(1, D)
    before = norm(x).view(1, L, V)

    perturbed = x.clone()
    perturbed[0, 0] += 50.0
    after = norm(perturbed).view(1, L, V)

    assert not torch.allclose(before[0, 0], after[0, 0]), "partition 0 should change"
    assert torch.allclose(before[0, 1:], after[0, 1:], atol=1e-6), (
        "every other partition must be untouched; a global softmax would couple them"
    )


def test_a_partition_is_invariant_to_a_constant_shift():
    norm = SimplexNorm(D, per_partition=V)
    x = torch.randn(1, D)
    shifted = x.clone()
    shifted[0, 0:V] += 50.0

    assert torch.allclose(norm(x), norm(shifted), atol=1e-5)


def test_a_lower_temperature_produces_a_sparser_representation():
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
    norm = SimplexNorm(D, per_partition=V)
    x = torch.randn(B, D, requires_grad=True)
    norm(x).sum().backward()
    assert x.grad is not None
    assert torch.any(x.grad != 0)


def test_large_inputs_do_not_overflow():
    norm = SimplexNorm(D, per_partition=V)
    out = norm(torch.full((B, D), 1e4))
    assert torch.isfinite(out).all()
    assert torch.allclose(out.view(B, L, V).sum(dim=-1), torch.ones(B, L), atol=1e-5)


def test_it_works_on_a_half_width_representation():
    norm = SimplexNorm(D // 2, per_partition=V)
    assert norm.partitions == L // 2
    out = norm(torch.randn(B, D // 2))
    assert torch.allclose(
        out.view(B, L // 2, V).sum(dim=-1), torch.ones(B, L // 2), atol=1e-5
    )


def test_a_dimension_not_divisible_by_the_partition_size_is_refused():
    with pytest.raises(ValueError, match="divisible"):
        SimplexNorm(130, per_partition=V)


def test_a_non_positive_temperature_is_refused():
    with pytest.raises(ValueError):
        SimplexNorm(D, per_partition=V, tau=0.0)
    with pytest.raises(ValueError):
        SimplexNorm(D, per_partition=V, tau=-1.0)


def test_the_module_reports_its_configuration():
    text = repr(SimplexNorm(D, per_partition=V))
    assert "128" in text and "16" in text and "8" in text
