"""The DDPG backbone (Noh et al. section 4.2, Eq. 2-3, Algorithm 1).

Named for DrQ-v2 because that is the reference implementation the paper builds
on, and because the pixel path will add its augmentation and its encoder stack
here without the update rules changing.

Four things in this file are invariants, not tuning knobs:

* **Clipped double Q.** The target takes ``min`` over two independently
  initialised critics, which controls overestimation bias.
* **Polyak averaging** of the target critics at ``tau = 0.01``.
* **Gradient routing.** The critic loss updates the encoder; the actor loss
  does not. ``update_actor`` detaches the representation itself rather than
  trusting its caller, because this is the invariant that silently turns the
  paper's method back into the baseline if it slips.
* **No target actor.** Algorithm 1 computes the bootstrap action from the
  *online* actor and evaluates it under the *target* critics.

The TD target reads ``reward``, ``discount`` and ``bootstrap`` straight off the
batch instead of recomputing them. The buffer produced them, because it is the
only component that knows where episodes end (D12): ``discount`` is
``gamma ** steps_actually_taken`` for a window that may have been cut short at
an episode boundary, and ``bootstrap`` is ``1 - terminated`` -- never
``1 - (terminated or truncated)``, which on FetchPush would zero the target at
the end of every episode the agent ever sees (Landmine 6).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.actor_critic import Actor, Critic
from ..utils.schedule import linear_schedule


class DDPGAgent:
    """Actor-critic agent over an injected encoder.

    The encoder is a constructor argument so this class can be written once and
    used by both paths: ``StateEncoder`` for the sanity anchor, and the
    dualized image/proprioception stack for the real configurations. It must
    expose ``repr_dim``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        cfg: Mapping[str, Any],
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.cfg = dict(cfg)
        self.action_dim = int(action_dim)

        self.stddev_schedule = str(self.cfg.get("stddev_schedule", "linear(1.0,0.1,100000)"))
        # Fail at construction rather than 4000 steps into a run.
        linear_schedule(self.stddev_schedule, 0)

        self.stddev_clip = float(self.cfg.get("stddev_clip", 0.3))
        self.polyak = float(self.cfg.get("polyak", 0.01))
        hidden_dim = int(self.cfg.get("hidden_dim", 512))
        lr = float(self.cfg.get("lr", 1e-4))

        self.encoder = encoder.to(self.device)
        repr_dim = int(encoder.repr_dim)

        self.actor = Actor(repr_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(repr_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(repr_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad_(False)

        encoder_params = list(self.encoder.parameters())
        self.encoder_opt = torch.optim.Adam(encoder_params, lr=lr) if encoder_params else None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    # -- helpers ----------------------------------------------------------
    def _tensor(self, value) -> torch.Tensor:
        return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=self.device)

    def _noise(self, like: torch.Tensor, step: int) -> torch.Tensor:
        stddev = linear_schedule(self.stddev_schedule, step)
        return (torch.randn_like(like) * stddev).clamp(-self.stddev_clip, self.stddev_clip)

    def train(self, mode: bool = True) -> None:
        for module in (self.encoder, self.actor, self.critic):
            module.train(mode)

    def eval(self) -> None:
        self.train(False)

    # -- acting -----------------------------------------------------------
    def act(self, obs, step: int, eval_mode: bool = False) -> np.ndarray:
        """One action for one observation. ``obs`` is unbatched."""
        with torch.no_grad():
            z = self.encoder(self._tensor(obs).unsqueeze(0))
            action = self.actor(z)
            if not eval_mode:
                action = action + self._noise(action, step)
            # tanh bounds the mean, not the mean plus noise. Without this the
            # environment clips silently and the critic is trained on actions
            # that were never executed.
            action = action.clamp(-1.0, 1.0)
        return action.squeeze(0).cpu().numpy()

    # -- targets ----------------------------------------------------------
    def td_target(self, batch: Mapping[str, Any], step: int = 0) -> torch.Tensor:
        """Eq. 2's ``y_t``, with the bootstrap term masked by ``1 - terminated``."""
        reward = self._tensor(batch["reward"]).unsqueeze(-1)
        discount = self._tensor(batch["discount"]).unsqueeze(-1)
        bootstrap = self._tensor(batch["bootstrap"]).unsqueeze(-1)

        with torch.no_grad():
            z_next = self.encoder(self._tensor(batch["next_obs"]))
            action_next = self.actor(z_next)
            action_next = (action_next + self._noise(action_next, step)).clamp(-1.0, 1.0)
            q1, q2 = self.critic_target(z_next, action_next)
            return reward + discount * bootstrap * torch.min(q1, q2)

    # -- updates ----------------------------------------------------------
    def update_critic(self, z: torch.Tensor, batch: Mapping[str, Any], step: int) -> dict[str, float]:
        """Eq. 2. Updates the critics and, through ``z``, the encoder."""
        target = self.td_target(batch, step)
        q1, q2 = self.critic(z, self._tensor(batch["action"]))
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        return {
            "critic_loss": float(loss.detach()),
            "q_mean": float(torch.min(q1, q2).mean().detach()),
        }

    def update_actor(self, z: torch.Tensor, step: int) -> dict[str, float]:
        """Eq. 3. Updates only the actor.

        The detach happens *here*, not in the caller. Gradient routing is a
        scientific invariant, so it lives in the function that would violate it
        rather than depending on every call site remembering.
        """
        z = z.detach()

        action = self.actor(z)
        action = (action + self._noise(action, step)).clamp(-1.0, 1.0)
        q1, q2 = self.critic(z, action)
        loss = -torch.min(q1, q2).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_opt.step()

        return {"actor_loss": float(loss.detach())}

    def update_target(self) -> None:
        """Polyak averaging at ``tau``: ``phi_bar <- (1 - tau) phi_bar + tau phi``."""
        with torch.no_grad():
            for online, target in zip(self.critic.parameters(), self.critic_target.parameters()):
                target.mul_(1.0 - self.polyak).add_(self.polyak * online)

    def update(self, batch: Mapping[str, Any], step: int) -> dict[str, float]:
        """One training step: critic, then actor, then the target networks."""
        z = self.encoder(self._tensor(batch["obs"]))

        metrics = self.update_critic(z, batch, step)
        metrics.update(self.update_actor(z, step))
        self.update_target()
        metrics["stddev"] = linear_schedule(self.stddev_schedule, step)
        return metrics
