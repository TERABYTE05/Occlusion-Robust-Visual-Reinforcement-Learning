from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.actor_critic import Actor, Critic
from ..utils.schedule import linear_schedule


def _critic_repr(encoder: nn.Module, obs):
    getter = getattr(encoder, "critic_repr", None)
    return getter(obs) if getter is not None else encoder(obs)


def _both_reprs(encoder: nn.Module, obs):
    getter = getattr(encoder, "both_reprs", None)
    if getter is not None:
        return getter(obs)
    return _critic_repr(encoder, obs), _actor_repr(encoder, obs)


def _actor_repr(encoder: nn.Module, obs):
    getter = getattr(encoder, "actor_repr", None)
    return getter(obs) if getter is not None else encoder(obs).detach()


class DDPGAgent:

    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        cfg: Mapping[str, Any],
        device: str | torch.device = "cpu",
        amp: bool = False,
    ):
        self.device = torch.device(device)
        self.cfg = dict(cfg)
        self.action_dim = int(action_dim)

        self.stddev_schedule = str(self.cfg.get("stddev_schedule", "linear(1.0,0.1,100000)"))
        linear_schedule(self.stddev_schedule, 0)

        self.stddev_clip = float(self.cfg.get("stddev_clip", 0.3))
        self.polyak = float(self.cfg.get("polyak", 0.01))
        hidden_dim = int(self.cfg.get("hidden_dim", 512))
        lr = float(self.cfg.get("lr", 1e-4))

        self.encoder = encoder.to(self.device)
        repr_dim = int(encoder.repr_dim)
        self._augment = getattr(self.encoder, "augment", None)

        self.amp = bool(amp) and self.device.type == "cuda"
        self._critic_scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self._actor_scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        self.actor = Actor(repr_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(repr_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(repr_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad_(False)

        owns_actor = getattr(self.encoder, "actor_parameters", None)
        owns_critic = getattr(self.encoder, "critic_parameters", None)
        encoder_actor_params = list(owns_actor()) if owns_actor else []
        encoder_critic_params = (
            list(owns_critic()) if owns_critic else list(self.encoder.parameters())
        )

        self.encoder_opt = (
            torch.optim.Adam(encoder_critic_params, lr=lr) if encoder_critic_params else None
        )
        self.actor_opt = torch.optim.Adam(
            list(self.actor.parameters()) + encoder_actor_params, lr=lr
        )
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def _tensor(self, value) -> torch.Tensor:
        return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=self.device)

    def _autocast(self):
        return torch.amp.autocast("cuda", enabled=self.amp)

    def _augmented(self, obs):
        return self._augment(obs) if self._augment is not None else obs

    def _obs(self, obs, batched: bool = True):
        if isinstance(obs, Mapping):
            return {k: self._obs(v, batched) for k, v in obs.items()}
        tensor = self._tensor(obs)
        return tensor if batched else tensor.unsqueeze(0)

    def _noise(self, like: torch.Tensor, step: int) -> torch.Tensor:
        stddev = linear_schedule(self.stddev_schedule, step)
        return (torch.randn_like(like) * stddev).clamp(-self.stddev_clip, self.stddev_clip)

    def train(self, mode: bool = True) -> None:
        for module in (self.encoder, self.actor, self.critic):
            module.train(mode)

    def eval(self) -> None:
        self.train(False)

    def act(self, obs, step: int, eval_mode: bool = False) -> np.ndarray:
        with torch.no_grad(), self._autocast():
            z = _actor_repr(self.encoder, self._obs(obs, batched=False))
            action = self.actor(z)
            if not eval_mode:
                action = action + self._noise(action, step)
            action = action.clamp(-1.0, 1.0)
        return action.squeeze(0).cpu().numpy()

    def td_target(self, batch: Mapping[str, Any], step: int = 0) -> torch.Tensor:
        reward = self._tensor(batch["reward"]).unsqueeze(-1)
        discount = self._tensor(batch["discount"]).unsqueeze(-1)
        bootstrap = self._tensor(batch["bootstrap"]).unsqueeze(-1)

        with torch.no_grad(), self._autocast():
            next_obs = self._augmented(self._obs(batch["next_obs"]))
            z_critic_next, z_actor_next = _both_reprs(self.encoder, next_obs)
            action_next = self.actor(z_actor_next)
            action_next = (action_next + self._noise(action_next, step)).clamp(-1.0, 1.0)
            q1, q2 = self.critic_target(z_critic_next, action_next)
            return reward + discount * bootstrap * torch.min(q1, q2).float()

    def update_critic(self, z: torch.Tensor, batch: Mapping[str, Any], step: int) -> dict[str, float]:
        target = self.td_target(batch, step)

        with self._autocast():
            q1, q2 = self.critic(z, self._tensor(batch["action"]))
            loss = F.mse_loss(q1, target.to(q1.dtype)) + F.mse_loss(q2, target.to(q2.dtype))

        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        self._critic_scaler.scale(loss).backward()
        self._critic_scaler.step(self.critic_opt)
        if self.encoder_opt is not None:
            self._critic_scaler.step(self.encoder_opt)
        self._critic_scaler.update()

        return {
            "critic_loss": float(loss.detach()),
            "q_mean": float(torch.min(q1, q2).mean().detach()),
        }

    def update_actor(self, obs, step: int) -> dict[str, float]:
        return self._actor_step(_actor_repr(self.encoder, obs), step)

    def _actor_step(self, z: torch.Tensor, step: int) -> dict[str, float]:
        with self._autocast():
            action = self.actor(z)
            action = (action + self._noise(action, step)).clamp(-1.0, 1.0)
            q1, q2 = self.critic(z, action)
            loss = -torch.min(q1, q2).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        self._actor_scaler.scale(loss).backward()
        self._actor_scaler.step(self.actor_opt)
        self._actor_scaler.update()

        return {"actor_loss": float(loss.detach())}

    def update_target(self) -> None:
        with torch.no_grad():
            for online, target in zip(self.critic.parameters(), self.critic_target.parameters()):
                target.mul_(1.0 - self.polyak).add_(self.polyak * online)

    def update(self, batch: Mapping[str, Any], step: int) -> dict[str, float]:
        obs = self._augmented(self._obs(batch["obs"]))
        with self._autocast():
            z_critic, z_actor = _both_reprs(self.encoder, obs)

        metrics = self.update_critic(z_critic, batch, step)
        metrics.update(self._actor_step(z_actor, step))
        self.update_target()
        metrics["stddev"] = linear_schedule(self.stddev_schedule, step)
        return metrics
