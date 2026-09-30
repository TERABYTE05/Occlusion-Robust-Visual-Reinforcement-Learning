"""The full encoder stack, and the gradient routing dualization depends on.

``MultimodalEncoder`` is ``f_xi`` + ``g_zeta`` + ``h_psi``: what the agent treats
as "the encoder". It exposes two representations rather than one, because the
paper's two losses evaluate the critic at different points (D20):

* Eq. 2 scores the critic at ``z_mm_c``;
* Eq. 3 scores it at ``z_mm_a`` and optimises ``theta`` **and** ``psi_actor``.

The routing test below is the one that matters. "Detach before the actor" has
two readings, and only one of them is the paper's: the detach goes on the
*encoder outputs*, before ``h_psi_actor``, so the actor's fusion head still
learns. Detaching the fused vector instead leaves ``psi_actor`` with no gradient,
so it never trains and Config C silently becomes Config B carrying an extra
unused module.
"""

import pytest
import torch

from src.models.multimodal import MultimodalEncoder
from src.utils.config import load_config

B, IMG_SIZE, STACK, PROPRIO = 3, 84, 3, 13


def obs(batch=B):
    return {
        "pixels": torch.randint(0, 255, (batch, 3 * STACK, IMG_SIZE, IMG_SIZE), dtype=torch.uint8),
        "proprio": torch.randn(batch, PROPRIO),
    }


def build(name):
    cfg = load_config(f"configs/{name}.yaml")
    return MultimodalEncoder(cfg, image_size=IMG_SIZE, frame_stack=STACK)


# -- shape and wiring -------------------------------------------------------


def test_the_stack_produces_a_representation_of_dimension_d():
    encoder = build("config_c_dual_occ")
    assert encoder.repr_dim == 128
    assert encoder.critic_repr(obs()).shape == (B, 128)
    assert encoder.actor_repr(obs()).shape == (B, 128)


def test_forward_is_the_critic_representation():
    """The agent's default path; ``act`` asks for the actor's explicitly."""
    encoder = build("config_c_dual_occ")
    torch.manual_seed(0)
    sample = obs()
    assert torch.equal(encoder(sample), encoder.critic_repr(sample))


def test_config_c_gives_the_two_representations_different_values():
    encoder = build("config_c_dual_occ")
    sample = obs()
    assert not torch.allclose(encoder.critic_repr(sample), encoder.actor_repr(sample))


def test_the_naive_configs_give_one_representation_twice():
    """Configs A and B share a single fusion module (Fig. 2b)."""
    encoder = build("config_a_concat_noocc")
    sample = obs()
    assert torch.allclose(
        encoder.critic_repr(sample), encoder.actor_repr(sample).detach(), atol=1e-6
    )


# -- gradient routing (INVARIANT) -------------------------------------------


def test_the_critic_path_reaches_the_encoders_and_the_critic_head():
    encoder = build("config_c_dual_occ")
    encoder.critic_repr(obs()).sum().backward()

    assert any(p.grad is not None and torch.any(p.grad != 0)
               for p in encoder.image.parameters()), "f_xi must learn from the critic"
    assert any(p.grad is not None and torch.any(p.grad != 0)
               for p in encoder.proprio.parameters()), "g_zeta must learn from the critic"
    assert any(p.grad is not None and torch.any(p.grad != 0)
               for p in encoder.fusion.critic_head.parameters())


def test_the_actor_path_trains_psi_actor_but_not_the_encoders():
    """D20. Both halves of this matter and they pull in opposite directions."""
    encoder = build("config_c_dual_occ")
    encoder.actor_repr(obs()).sum().backward()

    assert all(p.grad is None or torch.all(p.grad == 0) for p in encoder.image.parameters()), (
        "the actor must not update f_xi"
    )
    assert all(p.grad is None or torch.all(p.grad == 0) for p in encoder.proprio.parameters()), (
        "the actor must not update g_zeta"
    )
    assert any(p.grad is not None and torch.any(p.grad != 0)
               for p in encoder.fusion.actor_head.parameters()), (
        "psi_actor MUST learn from the actor loss -- detaching the fused vector "
        "instead of the encoder outputs would leave it untrained and collapse "
        "Config C into Config B"
    )


def test_the_actor_path_does_not_train_the_critic_head():
    encoder = build("config_c_dual_occ")
    encoder.actor_repr(obs()).sum().backward()
    assert all(p.grad is None or torch.all(p.grad == 0)
               for p in encoder.fusion.critic_head.parameters())


# -- the invariant dimensions survive the whole stack -----------------------


def test_config_c_output_is_still_sixteen_simplices():
    encoder = build("config_c_dual_occ")
    out = encoder.critic_repr(obs())
    assert torch.all(out >= 0)
    assert torch.allclose(out.view(B, 16, 8).sum(dim=-1), torch.ones(B, 16), atol=1e-5)


def test_config_a_output_is_not_normalised():
    encoder = build("config_a_concat_noocc")
    out = encoder.critic_repr(obs())
    assert not torch.allclose(out.view(B, 16, 8).sum(dim=-1), torch.ones(B, 16), atol=1e-3)


def test_the_image_encoder_keeps_its_invariant_shape():
    encoder = build("config_c_dual_occ")
    assert encoder.image.repr_dim == 39200


def test_the_proprio_branch_refuses_a_widened_slice():
    """Landmine 3 must survive the extra layer of assembly."""
    encoder = build("config_c_dual_occ")
    bad = obs()
    bad["proprio"] = torch.randn(B, 25)
    with pytest.raises(Exception):
        encoder.critic_repr(bad)


# -- the conventional design owns nothing on the actor side -----------------


def test_the_naive_actor_representation_carries_no_gradient_at_all():
    """Section 4.2: conventionally "the actor is optimized solely with respect
    to theta". Letting it also train the shared fusion module would make Configs
    A and B a third, undeclared method rather than the paper's baseline.

    The representation is therefore fully detached, and the actor loss still
    trains theta because the policy network sits downstream of it.
    """
    encoder = build("config_a_concat_noocc")
    z = encoder.actor_repr(obs())
    assert not z.requires_grad, "the shared fusion module belongs to the critic loss"

    # theta still learns: the policy is downstream of the detached input.
    actor = torch.nn.Linear(encoder.repr_dim, 4)
    actor(z).sum().backward()
    assert actor.weight.grad is not None and torch.any(actor.weight.grad != 0)
    for name, param in encoder.named_parameters():
        assert param.grad is None or torch.all(param.grad == 0), name


def test_the_dualized_actor_representation_does_carry_a_gradient():
    """The mirror image, and the reason the two configs differ at all."""
    encoder = build("config_c_dual_occ")
    assert encoder.actor_repr(obs()).requires_grad


def test_parameter_ownership_splits_by_configuration():
    dual = build("config_c_dual_occ")
    naive = build("config_a_concat_noocc")

    assert dual.actor_parameters(), "Config C's actor owns psi_actor"
    assert not naive.actor_parameters(), "Configs A and B's actor owns only theta"

    # No parameter may be claimed by both optimizers.
    actor_ids = {id(p) for p in dual.actor_parameters()}
    critic_ids = {id(p) for p in dual.critic_parameters()}
    assert actor_ids.isdisjoint(critic_ids)

    # And together they must cover every parameter exactly once.
    owned = actor_ids | critic_ids
    everything = {id(p) for p in dual.parameters()}
    assert owned == everything, "some parameters would never be stepped"

    naive_owned = {id(p) for p in naive.critic_parameters()}
    assert naive_owned == {id(p) for p in naive.parameters()}
