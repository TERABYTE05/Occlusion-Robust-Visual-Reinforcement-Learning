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


def test_the_stack_produces_a_representation_of_dimension_d():
    encoder = build("config_c_dual_occ")
    assert encoder.repr_dim == 128
    assert encoder.critic_repr(obs()).shape == (B, 128)
    assert encoder.actor_repr(obs()).shape == (B, 128)


def test_forward_is_the_critic_representation():
    encoder = build("config_c_dual_occ")
    torch.manual_seed(0)
    sample = obs()
    assert torch.equal(encoder(sample), encoder.critic_repr(sample))


def test_config_c_gives_the_two_representations_different_values():
    encoder = build("config_c_dual_occ")
    sample = obs()
    assert not torch.allclose(encoder.critic_repr(sample), encoder.actor_repr(sample))


def test_the_naive_configs_give_one_representation_twice():
    encoder = build("config_a_concat_noocc")
    sample = obs()
    assert torch.allclose(
        encoder.critic_repr(sample), encoder.actor_repr(sample).detach(), atol=1e-6
    )


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
    encoder = build("config_c_dual_occ")
    bad = obs()
    bad["proprio"] = torch.randn(B, 25)
    with pytest.raises(Exception):
        encoder.critic_repr(bad)


def test_the_naive_actor_representation_carries_no_gradient_at_all():
    encoder = build("config_a_concat_noocc")
    z = encoder.actor_repr(obs())
    assert not z.requires_grad, "the shared fusion module belongs to the critic loss"

    actor = torch.nn.Linear(encoder.repr_dim, 4)
    actor(z).sum().backward()
    assert actor.weight.grad is not None and torch.any(actor.weight.grad != 0)
    for name, param in encoder.named_parameters():
        assert param.grad is None or torch.all(param.grad == 0), name


def test_the_dualized_actor_representation_does_carry_a_gradient():
    encoder = build("config_c_dual_occ")
    assert encoder.actor_repr(obs()).requires_grad


def test_parameter_ownership_splits_by_configuration():
    dual = build("config_c_dual_occ")
    naive = build("config_a_concat_noocc")

    assert dual.actor_parameters(), "Config C's actor owns psi_actor"
    assert not naive.actor_parameters(), "Configs A and B's actor owns only theta"

    actor_ids = {id(p) for p in dual.actor_parameters()}
    critic_ids = {id(p) for p in dual.critic_parameters()}
    assert actor_ids.isdisjoint(critic_ids)

    owned = actor_ids | critic_ids
    everything = {id(p) for p in dual.parameters()}
    assert owned == everything, "some parameters would never be stepped"

    naive_owned = {id(p) for p in naive.critic_parameters()}
    assert naive_owned == {id(p) for p in naive.parameters()}


def test_both_reprs_runs_the_trunk_once():
    encoder = build("config_c_dual_occ")
    calls = {"n": 0}
    inner = encoder.image.forward

    def counting(x):
        calls["n"] += 1
        return inner(x)

    encoder.image.forward = counting
    encoder.both_reprs(obs())
    assert calls["n"] == 1, f"the image encoder ran {calls['n']} times, expected 1"


def test_both_reprs_agrees_with_the_separate_paths():
    encoder = build("config_c_dual_occ")
    encoder.eval()
    sample = obs()
    z_c, z_a = encoder.both_reprs(sample)
    assert torch.allclose(z_c, encoder.critic_repr(sample), atol=1e-6)
    assert torch.allclose(z_a, encoder.actor_repr(sample), atol=1e-6)


def test_both_reprs_keeps_the_routing_for_each_configuration():
    dual = build("config_c_dual_occ")
    z_c, z_a = dual.both_reprs(obs())
    z_a.sum().backward(retain_graph=True)
    assert all(p.grad is None or torch.all(p.grad == 0) for p in dual.image.parameters()), (
        "the actor must still not reach f_xi"
    )
    assert any(p.grad is not None and torch.any(p.grad != 0)
               for p in dual.fusion.actor_head.parameters()), "psi_actor must still learn"

    naive = build("config_a_concat_noocc")
    _, z_a_naive = naive.both_reprs(obs())
    assert not z_a_naive.requires_grad, "the conventional actor still owns nothing here"
