from __future__ import annotations

import torch
import torch.nn as nn


def _trunk(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
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

    def __init__(self, repr_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = _trunk(repr_dim, hidden_dim, action_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(z))


class Critic(nn.Module):

    def __init__(self, repr_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.q1 = _trunk(repr_dim + action_dim, hidden_dim, 1)
        self.q2 = _trunk(repr_dim + action_dim, hidden_dim, 1)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z, action], dim=-1)
        return self.q1(x), self.q2(x)
