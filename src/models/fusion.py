from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from .norms import SimplexNorm

NORMALIZATIONS = ("none", "layernorm", "layernorm_simplexnorm", "layernorm_tanh")
NORMALIZED = "layernorm_simplexnorm"

DUALIZED = "dual"


class FusionHead(nn.Module):

    def __init__(
        self,
        image_dim: int,
        proprio_dim: int,
        feature_dim: int = 128,
        normalization: str = "none",
        per_partition: int = 8,
        tau: float = 1.0,
    ):
        super().__init__()
        if normalization not in NORMALIZATIONS:
            raise ValueError(
                f"unknown normalization {normalization!r}; expected one of {NORMALIZATIONS}"
            )
        feature_dim = int(feature_dim)
        if feature_dim % 2 != 0:
            raise ValueError(
                f"feature_dim must be even so the two halves concatenate evenly; "
                f"got {feature_dim}"
            )

        self.feature_dim = feature_dim
        self.normalization = normalization
        self.normalize = normalization != "none"
        half = feature_dim // 2

        if normalization == "layernorm_simplexnorm" and half % per_partition != 0:
            raise ValueError(
                f"each half of feature_dim ({half}) must be divisible by "
                f"per_partition ({per_partition})"
            )

        self.image_branch = self._branch(image_dim, half, per_partition, tau)
        self.proprio_branch = self._branch(proprio_dim, half, per_partition, tau)

    def _branch(self, in_dim: int, out_dim: int, per_partition: int, tau: float) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
        if self.normalization != "none":
            layers.append(nn.LayerNorm(out_dim))
        if self.normalization == "layernorm_simplexnorm":
            layers.append(SimplexNorm(out_dim, per_partition=per_partition, tau=tau))
        elif self.normalization == "layernorm_tanh":
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, z_image: torch.Tensor, z_prop: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.image_branch(z_image), self.proprio_branch(z_prop)], dim=-1)


class MultimodalFusion(nn.Module):

    def __init__(
        self,
        image_dim: int,
        proprio_dim: int,
        feature_dim: int = 128,
        dualized: bool = False,
        normalization: str = "none",
        per_partition: int = 8,
        tau: float = 1.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.dualized = bool(dualized)

        def head() -> FusionHead:
            return FusionHead(
                image_dim, proprio_dim, feature_dim=feature_dim,
                normalization=normalization, per_partition=per_partition, tau=tau,
            )

        self.critic_head = head()
        self.actor_head = head() if self.dualized else self.critic_head

    def forward(self, z_image: torch.Tensor, z_prop: torch.Tensor, head: str = "critic") -> torch.Tensor:
        if head == "critic":
            return self.critic_head(z_image, z_prop)
        if head == "actor":
            return self.actor_head(z_image, z_prop)
        raise ValueError(f"head must be 'actor' or 'critic'; got {head!r}")


def build_fusion(cfg: Mapping[str, Any], image_dim: int, proprio_dim: int) -> MultimodalFusion:
    agent = dict(cfg.get("agent", {}))
    simplex = dict(agent.get("simplexnorm", {}))

    return MultimodalFusion(
        image_dim=image_dim,
        proprio_dim=proprio_dim,
        feature_dim=int(agent.get("feature_dim", 128)),
        dualized=str(agent.get("fusion", "concat")) == DUALIZED,
        normalization=str(agent.get("normalization", "none")),
        per_partition=int(simplex.get("per_partition", 8)),
        tau=float(simplex.get("tau", 1.0)),
    )
