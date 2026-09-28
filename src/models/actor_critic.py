"""Actor and twin critics (Noh et al., Appendix B.1.4).

Both are 3-layer MLPs with LayerNorm and ReLU after each layer except the last.
The paper's hidden dimension is 1024; ours is 512 for GPU memory and speed,
which is an adaptation recorded in DECISIONS.md, not an invariant.

These operate on an already-fused representation and know nothing about pixels,
proprioception or fusion. That is deliberate: the same two networks serve the
state-based sanity anchor today and the dualized pixel path once ``fusion.py``
and ``norms.py`` exist, so the DDPG core can be debugged with no vision in the
picture (ROADMAP section 2).

The actor is deterministic and ``tanh``-bounded. Exploration noise is added by
the agent, which re-clips afterwards -- ``tanh`` bounds the mean, not the mean
plus noise, and the environment would clip the difference silently.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _trunk(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """3 linear layers; LayerNorm + ReLU after the first two, nothing after the last."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class Actor(nn.Module):
    """``pi_theta``. Deterministic policy, bounded to the action space."""

    def __init__(self, repr_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = _trunk(repr_dim, hidden_dim, action_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(z))


class Critic(nn.Module):
    """``Q_phi1`` and ``Q_phi2``, for clipped double Q-learning.

    The two trunks are separate modules with disjoint parameters. Sharing them
    would make ``min(Q1, Q2)`` an expensive way of writing ``Q1`` and would
    remove the overestimation control the target depends on.
    """

    def __init__(self, repr_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.q1 = _trunk(repr_dim + action_dim, hidden_dim, 1)
        self.q2 = _trunk(repr_dim + action_dim, hidden_dim, 1)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z, action], dim=-1)
        return self.q1(x), self.q2(x)
