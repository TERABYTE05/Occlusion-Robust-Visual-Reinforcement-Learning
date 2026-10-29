"""Multimodal fusion and its dualization (Noh et al. §4.1.3, §4.2, Fig. 2, B.1.3).

This is where both of the paper's contributions live.

**The module.** Appendix B.1.3: "The multimodal fusion network is modeled as a
single fully-connected layer, followed by LayerNorm and SimplexNorm to control
the gradient scales. Lastly, the outputs are concatenated to form a single
multimodal representation vector of dimension d." So each modality gets its own
projection and its own normalization, and the two halves are concatenated into
``d = 128``. With ``V = 8`` that is 8 simplices per half and ``L = 16`` across
the concatenation, which is the invariant.

**The projection is not optional.** Configs A and B — the conventional baseline
of Fig. 2(b) — still project to ``d``; they differ from Config C by having no
LayerNorm and no SimplexNorm *after* the projection, not by having no projection.
Dropping it would leave the actor's first layer at 39200x512 = 20M parameters
and a single checkpoint at roughly 900 MB.

**Dualization.** The conventional design has one fusion module serving both
networks, so a single representation must satisfy two different objectives. The
paper's design gives the actor and the critic one each, optimized only by their
own loss: ``psi_critic`` by the critic loss alongside the encoders and the
critics, ``psi_actor`` by the actor loss alongside the policy.

The two heads **must not share weights**. If they do, Config C silently becomes
Config B and the curves look plausible either way — which is the worst kind of
failure this project can have, because it produces a number rather than an
error. ``tests/test_fusion.py`` asserts the parameter sets are disjoint.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from .norms import SimplexNorm

#: What may follow the fusion projection. One change per rung, so an ablation can
#: attribute an effect to a single component (the paper's Fig. 5 decomposition).
#:
#:   none                  Linear only -- unbounded. The naive baseline.
#:   layernorm             Linear -> LayerNorm
#:   layernorm_simplexnorm Linear -> LayerNorm -> SimplexNorm   (the paper's method)
#:   layernorm_tanh        Linear -> LayerNorm -> Tanh          (DrQ-v2's trunk)
#:
#: `layernorm_tanh` is diagnostic, not one of the paper's variants: DrQ-v2 -- the
#: implementation the paper builds on -- bounds its representation this way in
#: every variant. Our `none` rung grows to RMS 25 during training and goes
#: action-blind, which the paper's own naive baseline does not appear to do
#: (`RUNLOG.md` 2026-10-29). This rung tests whether the missing bound explains
#: that difference.
NORMALIZATIONS = ("none", "layernorm", "layernorm_simplexnorm", "layernorm_tanh")
NORMALIZED = "layernorm_simplexnorm"  # kept: the config value for the paper's method

#: Config value that turns dualization on.
DUALIZED = "dual"


class FusionHead(nn.Module):
    """One ``h_psi``: project each modality, normalise it, concatenate.

    Parameters
    ----------
    image_dim, proprio_dim:
        Widths of ``z_image`` and ``z_prop`` as they arrive from the encoders.
    feature_dim:
        ``d``, the width of the fused representation. Split evenly between the
        two modalities, so it must be even.
    normalize:
        Whether to apply ``LayerNorm -> SimplexNorm`` after each projection.
        False is the naive baseline of Configs A and B.
    """

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
        #: kept for callers that only need "is anything applied after the Linear"
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
        # INVARIANT for the paper's method: Linear -> LayerNorm -> SimplexNorm, in
        # this order. The intermediate rungs drop the tail, never reorder it.
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
    """The fusion stage: one shared head, or one per network.

    ``dualized=False`` reproduces Fig. 2(b) — a single module, genuinely shared,
    so ``actor_head is critic_head`` and its parameters are counted once.
    ``dualized=True`` is the paper's method.
    """

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
        # Deliberately the *same object* when not dualized, so nothing downstream
        # can treat the baseline as two heads that happen to start equal.
        self.actor_head = head() if self.dualized else self.critic_head

    def forward(self, z_image: torch.Tensor, z_prop: torch.Tensor, head: str = "critic") -> torch.Tensor:
        if head == "critic":
            return self.critic_head(z_image, z_prop)
        if head == "actor":
            return self.actor_head(z_image, z_prop)
        raise ValueError(f"head must be 'actor' or 'critic'; got {head!r}")


def build_fusion(cfg: Mapping[str, Any], image_dim: int, proprio_dim: int) -> MultimodalFusion:
    """Build the fusion stage a config asks for.

    Reads ``agent.fusion``, ``agent.normalization``, ``agent.feature_dim`` and
    ``agent.simplexnorm``. Configs A and B give a shared, unnormalised module;
    Config C gives two normalised ones, and that difference is the experiment.
    """
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
