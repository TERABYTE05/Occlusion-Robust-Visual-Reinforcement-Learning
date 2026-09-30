"""The DDPG backbone (Noh et al. section 4.2, Eq. 2-3, Algorithm 1).

Most of these guard invariants rather than behaviour, because the failures they
catch are silent. An agent whose actor gradients reach the encoder still trains
and still produces a curve; it is just no longer the paper's method. An agent
that masks the bootstrap on truncation still trains, on FetchPush, where every
episode truncates, and quietly mislearns the value of every late state.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.agent.drqv2 import DDPGAgent
from src.models.encoders import StateEncoder

OBS, ACT = 28, 4

CFG = {
    "lr": 1e-4,
    "hidden_dim": 64,
    "discount": 0.99,
    "polyak": 0.01,
    "stddev_schedule": "linear(1.0,0.1,100)",
    "stddev_clip": 0.3,
}


class LearnedEncoder(nn.Module):
    """A stand-in with parameters, so gradient routing is observable."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(OBS, OBS)
        self.repr_dim = OBS

    def forward(self, x):
        return self.fc(x.float())


def make_agent(encoder=None, **overrides):
    cfg = dict(CFG)
    cfg.update(overrides)
    return DDPGAgent(
        encoder if encoder is not None else StateEncoder(OBS),
        action_dim=ACT,
        cfg=cfg,
        device="cpu",
    )


def make_batch(n=8, terminated=False, reward=1.0, discount=0.9):
    return {
        "obs": np.random.randn(n, OBS).astype(np.float32),
        "next_obs": np.random.randn(n, OBS).astype(np.float32),
        "action": np.random.uniform(-1, 1, (n, ACT)).astype(np.float32),
        "reward": np.full(n, reward, dtype=np.float32),
        "discount": np.full(n, discount, dtype=np.float32),
        "bootstrap": np.full(n, 0.0 if terminated else 1.0, dtype=np.float32),
    }


# -- acting -----------------------------------------------------------------


def test_actions_stay_inside_the_action_space_under_large_noise():
    """The tanh bounds the mean, not the mean plus noise; MuJoCo clips silently."""
    agent = make_agent(stddev_schedule="5.0")
    for _ in range(50):
        action = agent.act(np.random.randn(OBS).astype(np.float32), step=0)
        assert action.shape == (ACT,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_eval_mode_is_deterministic_and_noise_free():
    agent = make_agent()
    obs = np.random.randn(OBS).astype(np.float32)
    assert np.allclose(
        agent.act(obs, step=0, eval_mode=True),
        agent.act(obs, step=0, eval_mode=True),
    )


def test_training_mode_explores():
    agent = make_agent()
    obs = np.random.randn(OBS).astype(np.float32)
    actions = np.stack([agent.act(obs, step=0) for _ in range(10)])
    assert actions.std(axis=0).sum() > 0, "training actions must carry exploration noise"


# -- gradient routing (INVARIANT) -------------------------------------------


def test_the_critic_loss_reaches_the_encoder():
    agent = make_agent(LearnedEncoder())
    agent.update(make_batch(), step=0)
    grads = [p.grad for p in agent.encoder.parameters() if p.grad is not None]
    assert grads and any(torch.any(g != 0) for g in grads)


def test_the_actor_loss_does_not_reach_the_encoder():
    """INVARIANT: the actor is blocked from the encoders (CLAUDE.md, §4.2).

    ``update_actor`` takes the observation, not a representation, precisely so
    that the routing lives in the agent rather than in every call site.
    """
    agent = make_agent(LearnedEncoder())
    obs = torch.as_tensor(make_batch()["obs"])

    assert agent.encoder(obs).requires_grad, "the encoder must carry a graph to begin with"

    for p in agent.encoder.parameters():
        p.grad = None
    agent.update_actor(obs, step=0)

    for p in agent.encoder.parameters():
        assert p.grad is None or torch.all(p.grad == 0), (
            "the actor update must detach before the encoder; letting its "
            "gradients in silently changes the method"
        )


def test_the_actor_optimizer_owns_psi_actor_only_when_dualized():
    """Algorithm 1 steps psi_actor in UpdateActor, never in UpdateCritic (D20)."""

    class DualEncoder(nn.Module):
        """Minimal stand-in with the dualized ownership interface."""

        def __init__(self):
            super().__init__()
            self.trunk = nn.Linear(OBS, OBS)
            self.psi_actor = nn.Linear(OBS, OBS)
            self.psi_critic = nn.Linear(OBS, OBS)
            self.repr_dim = OBS

        def critic_repr(self, obs):
            return self.psi_critic(self.trunk(obs.float()))

        def actor_repr(self, obs):
            return self.psi_actor(self.trunk(obs.float()).detach())

        def actor_parameters(self):
            return list(self.psi_actor.parameters())

        def critic_parameters(self):
            return list(self.trunk.parameters()) + list(self.psi_critic.parameters())

        def forward(self, obs):
            return self.critic_repr(obs)

    encoder = DualEncoder()
    agent = make_agent(encoder)

    owned_by_actor = {id(p) for group in agent.actor_opt.param_groups for p in group["params"]}
    owned_by_encoder = {id(p) for group in agent.encoder_opt.param_groups for p in group["params"]}

    assert all(id(p) in owned_by_actor for p in encoder.psi_actor.parameters()), (
        "psi_actor must be stepped by the actor's optimizer or it never trains"
    )
    assert owned_by_actor.isdisjoint(owned_by_encoder)

    before = encoder.psi_actor.weight.detach().clone()
    agent.update(make_batch(), step=0)
    assert not torch.allclose(before, encoder.psi_actor.weight), "psi_actor did not move"


def test_the_trunk_is_not_moved_by_the_actor_in_a_dualized_agent():
    """The encoders stay the critic's, dualization or not."""

    class DualEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Linear(OBS, OBS)
            self.psi_actor = nn.Linear(OBS, OBS)
            self.psi_critic = nn.Linear(OBS, OBS)
            self.repr_dim = OBS

        def critic_repr(self, obs):
            return self.psi_critic(self.trunk(obs.float()))

        def actor_repr(self, obs):
            return self.psi_actor(self.trunk(obs.float()).detach())

        def actor_parameters(self):
            return list(self.psi_actor.parameters())

        def critic_parameters(self):
            return list(self.trunk.parameters()) + list(self.psi_critic.parameters())

        def forward(self, obs):
            return self.critic_repr(obs)

    encoder = DualEncoder()
    agent = make_agent(encoder)
    for p in encoder.parameters():
        p.grad = None

    agent.update_actor(torch.as_tensor(make_batch()["obs"]), step=0)

    for name, param in encoder.trunk.named_parameters():
        assert param.grad is None or torch.all(param.grad == 0), f"trunk.{name}"
    assert any(p.grad is not None and torch.any(p.grad != 0)
               for p in encoder.psi_actor.parameters())


# -- targets ----------------------------------------------------------------


def test_a_terminated_transition_drops_the_bootstrap_term():
    """Landmine 6 / D12, carried through into the TD target."""
    agent = make_agent()
    batch = make_batch(n=4, terminated=True, reward=2.0, discount=0.5)
    target = agent.td_target(batch)
    assert torch.allclose(target.squeeze(-1), torch.full((4,), 2.0))


def test_the_td_target_uses_the_per_sample_discount():
    """``discount`` is gamma**steps_taken, not a constant (D12)."""
    agent = make_agent(stddev_schedule="0.0")  # no target smoothing noise
    batch = make_batch(n=4, terminated=False, reward=0.0, discount=0.25)
    target = agent.td_target(batch)
    with torch.no_grad():
        z_next = agent.encoder(torch.as_tensor(batch["next_obs"]))
        q1, q2 = agent.critic_target(z_next, agent.actor(z_next))
    assert torch.allclose(target, 0.25 * torch.min(q1, q2), atol=1e-5)


def test_the_target_takes_the_minimum_of_the_twin_critics():
    agent = make_agent(stddev_schedule="0.0")
    batch = make_batch(n=6, terminated=False, reward=0.0, discount=1.0)
    target = agent.td_target(batch)
    with torch.no_grad():
        z_next = agent.encoder(torch.as_tensor(batch["next_obs"]))
        q1, q2 = agent.critic_target(z_next, agent.actor(z_next))
    assert torch.allclose(target, torch.min(q1, q2), atol=1e-5)
    assert not torch.allclose(q1, q2), "the two critics should not already agree"


def test_the_target_does_not_carry_a_gradient():
    agent = make_agent()
    assert not agent.td_target(make_batch()).requires_grad


# -- target networks --------------------------------------------------------


def test_polyak_moves_the_target_towards_the_online_critic():
    agent = make_agent(polyak=0.5)
    with torch.no_grad():
        for p in agent.critic.parameters():
            p.add_(1.0)

    before = [p.detach().clone() for p in agent.critic_target.parameters()]
    agent.update(make_batch(), step=0)
    after = list(agent.critic_target.parameters())

    assert any(not torch.allclose(b, a) for b, a in zip(before, after)), (
        "target networks must actually move"
    )


def test_the_target_critic_is_not_trained_directly():
    agent = make_agent()
    agent.update(make_batch(), step=0)
    assert all(p.grad is None for p in agent.critic_target.parameters())


def test_the_target_critic_starts_as_a_copy_of_the_online_critic():
    agent = make_agent()
    for online, target in zip(agent.critic.parameters(), agent.critic_target.parameters()):
        assert torch.allclose(online, target)


# -- metrics ----------------------------------------------------------------


def test_update_reports_the_metrics_the_run_log_needs():
    agent = make_agent()
    metrics = agent.update(make_batch(), step=0)
    for key in ("critic_loss", "actor_loss", "q_mean", "stddev"):
        assert key in metrics
        assert np.isfinite(metrics[key]), f"{key} was not finite"


def test_repeated_updates_reduce_the_critic_loss_on_a_fixed_batch():
    """The cheapest possible check that the optimiser is wired up at all."""
    agent = make_agent(lr=1e-2)
    batch = make_batch(n=32)
    first = agent.update(batch, step=0)["critic_loss"]
    for _ in range(50):
        last = agent.update(batch, step=0)["critic_loss"]
    assert last < first


def test_an_unknown_schedule_is_rejected_at_construction_not_mid_run():
    with pytest.raises(ValueError):
        make_agent(stddev_schedule="cosine(1,0,10)")
