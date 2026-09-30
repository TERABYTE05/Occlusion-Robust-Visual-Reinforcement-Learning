"""Multimodal fusion and its dualization (Noh et al. §4.1.3, §4.2, Fig. 2, B.1.3).

The fusion module is where both contributions live. Each modality goes through a
single fully-connected layer, then LayerNorm, then SimplexNorm; the two outputs
are concatenated into one representation of dimension d = 128.

**Dualization** is the fact that there are two of these — `h_psi_actor` and
`h_psi_critic` — rather than one shared module (Fig. 2a vs 2b). Configs A and B
are the conventional shared design *without* the normalizations; Config C is the
full method. So B ↔ C differs in exactly these two switches and nothing else.

The failure these tests exist to catch is silent: two heads that accidentally
share weights reduce Config C to Config B, and the curves look plausible either
way.
"""

import pytest
import torch

from src.models.fusion import FusionHead, MultimodalFusion, build_fusion
from src.utils.config import load_config

IMG, PROP, D, V, L = 512, 128, 128, 8, 16
B = 4


def inputs(batch=B):
    return torch.randn(batch, IMG), torch.randn(batch, PROP)


# -- one fusion head --------------------------------------------------------


def test_the_head_produces_a_representation_of_dimension_d():
    head = FusionHead(IMG, PROP, feature_dim=D)
    assert head(*inputs()).shape == (B, D)
    assert head.feature_dim == D


def test_both_modalities_reach_the_output():
    """A head that ignored one branch would train and measure nothing."""
    head = FusionHead(IMG, PROP, feature_dim=D)
    z_image, z_prop = inputs(1)

    base = head(z_image, z_prop)
    moved_image = head(z_image + 10.0, z_prop)
    moved_prop = head(z_image, z_prop + 10.0)

    assert not torch.allclose(base, moved_image), "the image branch is not connected"
    assert not torch.allclose(base, moved_prop), "the proprioception branch is not connected"


def test_a_normalised_head_outputs_sixteen_simplices():
    """d=128 with V=8 gives L=16 across the concatenation, whichever half."""
    head = FusionHead(IMG, PROP, feature_dim=D, normalize=True, per_partition=V)
    out = head(*inputs())
    assert torch.all(out >= 0.0)
    assert torch.allclose(out.view(B, L, V).sum(dim=-1), torch.ones(B, L), atol=1e-5)


def test_the_normalisation_order_is_linear_then_layernorm_then_simplexnorm():
    """INVARIANT: not reordered, not fused, none of them dropped."""
    from src.models.norms import SimplexNorm

    head = FusionHead(IMG, PROP, feature_dim=D, normalize=True, per_partition=V)
    for branch in (head.image_branch, head.proprio_branch):
        kinds = [type(m) for m in branch]
        assert kinds == [torch.nn.Linear, torch.nn.LayerNorm, SimplexNorm], kinds


def test_an_unnormalised_head_has_neither_normalisation():
    """Configs A and B are the naive baseline: the projection only."""
    from src.models.norms import SimplexNorm

    head = FusionHead(IMG, PROP, feature_dim=D, normalize=False)
    modules = list(head.modules())
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in modules)
    assert not any(isinstance(m, SimplexNorm) for m in modules)
    assert sum(isinstance(m, torch.nn.Linear) for m in modules) == 2


def test_the_projection_is_present_even_without_normalisation():
    """A and B still project to d; they differ from C only in the norms (B.1.3)."""
    head = FusionHead(IMG, PROP, feature_dim=D, normalize=False)
    assert head(*inputs()).shape == (B, D)


def test_gradients_reach_both_projections():
    head = FusionHead(IMG, PROP, feature_dim=D, normalize=True, per_partition=V)
    head(*inputs()).sum().backward()
    for name, param in head.named_parameters():
        assert param.grad is not None, name


# -- dualization ------------------------------------------------------------


def test_dualized_heads_do_not_share_a_single_parameter():
    """INVARIANT. Sharing them silently reduces Config C to Config B."""
    fusion = MultimodalFusion(IMG, PROP, feature_dim=D, dualized=True, normalize=True,
                              per_partition=V)
    actor = {id(p) for p in fusion.actor_head.parameters()}
    critic = {id(p) for p in fusion.critic_head.parameters()}
    assert actor and critic
    assert actor.isdisjoint(critic)


def test_dualized_heads_produce_different_representations():
    fusion = MultimodalFusion(IMG, PROP, feature_dim=D, dualized=True, normalize=True,
                              per_partition=V)
    z_image, z_prop = inputs()
    assert not torch.allclose(
        fusion(z_image, z_prop, head="actor"),
        fusion(z_image, z_prop, head="critic"),
    )


def test_the_shared_design_really_is_one_module():
    """Fig. 2(b): the conventional baseline has a single fusion module."""
    fusion = MultimodalFusion(IMG, PROP, feature_dim=D, dualized=False, normalize=False)
    assert fusion.actor_head is fusion.critic_head
    z_image, z_prop = inputs()
    assert torch.equal(
        fusion(z_image, z_prop, head="actor"),
        fusion(z_image, z_prop, head="critic"),
    )


def test_the_shared_design_counts_its_parameters_once():
    """A duplicated module would double the optimizer's view of the same weights."""
    fusion = MultimodalFusion(IMG, PROP, feature_dim=D, dualized=False, normalize=False)
    assert len(list(fusion.parameters())) == len(list(fusion.critic_head.parameters()))


def test_dualization_roughly_doubles_the_fusion_parameters():
    shared = MultimodalFusion(IMG, PROP, feature_dim=D, dualized=False, normalize=True,
                              per_partition=V)
    dual = MultimodalFusion(IMG, PROP, feature_dim=D, dualized=True, normalize=True,
                            per_partition=V)
    n_shared = sum(p.numel() for p in shared.parameters())
    n_dual = sum(p.numel() for p in dual.parameters())
    assert n_dual == 2 * n_shared


def test_an_unknown_head_name_is_refused():
    fusion = MultimodalFusion(IMG, PROP, feature_dim=D, dualized=True, normalize=True,
                              per_partition=V)
    with pytest.raises(ValueError, match="head"):
        fusion(*inputs(), head="both")


# -- construction guards ----------------------------------------------------


def test_an_odd_feature_dim_is_refused():
    """The two halves are concatenated, so d must split evenly."""
    with pytest.raises(ValueError, match="even"):
        FusionHead(IMG, PROP, feature_dim=127)


def test_a_half_not_divisible_by_the_partition_size_is_refused():
    with pytest.raises(ValueError, match="divisible"):
        FusionHead(IMG, PROP, feature_dim=100, normalize=True, per_partition=V)


# -- wiring to the real configs ---------------------------------------------


def test_configs_a_and_b_build_the_naive_shared_fusion():
    from src.models.norms import SimplexNorm

    for name in ("config_a_concat_noocc", "config_b_concat_occ"):
        cfg = load_config(f"configs/{name}.yaml")
        fusion = build_fusion(cfg, image_dim=IMG, proprio_dim=PROP)
        assert fusion.actor_head is fusion.critic_head, f"{name} must share one module"
        assert not any(isinstance(m, SimplexNorm) for m in fusion.modules()), name


def test_config_c_builds_the_dualized_normalised_fusion():
    from src.models.norms import SimplexNorm

    cfg = load_config("configs/config_c_dual_occ.yaml")
    fusion = build_fusion(cfg, image_dim=IMG, proprio_dim=PROP)

    assert fusion.actor_head is not fusion.critic_head
    actor = {id(p) for p in fusion.actor_head.parameters()}
    critic = {id(p) for p in fusion.critic_head.parameters()}
    assert actor.isdisjoint(critic)
    assert any(isinstance(m, SimplexNorm) for m in fusion.modules())

    out = fusion(*inputs(), head="critic")
    assert out.shape == (B, D)
    assert torch.allclose(out.view(B, L, V).sum(dim=-1), torch.ones(B, L), atol=1e-5)


def test_config_c_uses_the_invariant_simplexnorm_dimensions():
    """d=128, L=16, V=8, tau=1 come from the config and must arrive intact."""
    from src.models.norms import SimplexNorm

    cfg = load_config("configs/config_c_dual_occ.yaml")
    fusion = build_fusion(cfg, image_dim=IMG, proprio_dim=PROP)
    norms = [m for m in fusion.modules() if isinstance(m, SimplexNorm)]

    assert norms, "Config C must contain SimplexNorm"
    for norm in norms:
        assert norm.per_partition == 8
        assert norm.tau == 1.0
    assert sum(n.partitions for n in norms) == 2 * L, "16 partitions per head, two heads"


def test_b_and_c_differ_only_in_fusion_and_normalization():
    """The experiment's whole meaning rests on this being the only difference."""
    b = load_config("configs/config_b_concat_occ.yaml")
    c = load_config("configs/config_c_dual_occ.yaml")
    differing = {k for k in b["agent"] if b["agent"][k] != c["agent"].get(k)}
    assert differing == {"fusion", "normalization"}
