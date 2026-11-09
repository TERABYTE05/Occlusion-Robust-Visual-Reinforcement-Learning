from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from ..envs.proprio import PROPRIO_OBS_DIM
from .augment import RandomShiftAug
from .encoders import ImageEncoder, ProprioEncoder
from .fusion import build_fusion


class MultimodalEncoder(nn.Module):

    def __init__(self, cfg: Mapping[str, Any], image_size: int = 84, frame_stack: int = 3):
        super().__init__()
        agent_cfg = dict(cfg.get("agent", {}))

        self.image = ImageEncoder(in_channels=3 * frame_stack, image_size=image_size)
        self.proprio = ProprioEncoder(
            in_dim=PROPRIO_OBS_DIM,
            out_dim=int(agent_cfg.get("feature_dim", 128)),
        )
        self.fusion = build_fusion(cfg, self.image.repr_dim, self.proprio.repr_dim)
        self.repr_dim = self.fusion.feature_dim
        self.aug = RandomShiftAug(pad=int(agent_cfg.get("aug_pad", 4)))

    def augment(self, obs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {**obs, "pixels": self.aug(obs["pixels"].float())}

    def _encode(self, obs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.image(obs["pixels"]), self.proprio(obs["proprio"])

    def critic_repr(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        z_image, z_prop = self._encode(obs)
        return self.fusion(z_image, z_prop, head="critic")

    def actor_repr(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        z_image, z_prop = self._encode(obs)
        if self.fusion.dualized:
            return self.fusion(z_image.detach(), z_prop.detach(), head="actor")
        return self.fusion(z_image, z_prop, head="actor").detach()

    def both_reprs(self, obs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        z_image, z_prop = self._encode(obs)
        z_critic = self.fusion(z_image, z_prop, head="critic")
        if self.fusion.dualized:
            z_actor = self.fusion(z_image.detach(), z_prop.detach(), head="actor")
        else:
            z_actor = z_critic.detach()
        return z_critic, z_actor

    def forward(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.critic_repr(obs)

    def actor_parameters(self) -> list[nn.Parameter]:
        if not self.fusion.dualized:
            return []
        return list(self.fusion.actor_head.parameters())

    def critic_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.image.parameters())
            + list(self.proprio.parameters())
            + list(self.fusion.critic_head.parameters())
        )
