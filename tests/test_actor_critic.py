import torch

from src.models.actor_critic import Actor, Critic

REPR, ACT, B = 28, 4, 5


def test_actor_output_is_inside_the_action_space():
    actor = Actor(REPR, ACT)
    action = actor(torch.randn(B, REPR) * 50)
    assert action.shape == (B, ACT)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_critic_returns_two_q_values():
    critic = Critic(REPR, ACT)
    q1, q2 = critic(torch.randn(B, REPR), torch.randn(B, ACT))
    assert q1.shape == (B, 1) and q2.shape == (B, 1)


def test_the_twin_critics_do_not_share_weights():
    critic = Critic(REPR, ACT)
    q1_params = {id(p) for p in critic.q1.parameters()}
    q2_params = {id(p) for p in critic.q2.parameters()}
    assert q1_params.isdisjoint(q2_params)

    q1, q2 = critic(torch.randn(B, REPR), torch.randn(B, ACT))
    assert not torch.allclose(q1, q2), "independently initialised critics should disagree"


def test_networks_are_three_layer_mlps_normalised_everywhere_but_the_last_layer():
    for module in (Actor(REPR, ACT), Critic(REPR, ACT).q1):
        linears = [m for m in module.modules() if isinstance(m, torch.nn.Linear)]
        norms = [m for m in module.modules() if isinstance(m, torch.nn.LayerNorm)]
        assert len(linears) == 3
        assert len(norms) == 2


def test_hidden_dim_is_honoured():
    actor = Actor(REPR, ACT, hidden_dim=64)
    first = [m for m in actor.modules() if isinstance(m, torch.nn.Linear)][0]
    assert first.out_features == 64


def test_the_critic_consumes_the_action_alongside_the_representation():
    critic = Critic(REPR, ACT)
    z = torch.randn(B, REPR)
    q_low, _ = critic(z, torch.full((B, ACT), -1.0))
    q_high, _ = critic(z, torch.full((B, ACT), 1.0))
    assert not torch.allclose(q_low, q_high)
