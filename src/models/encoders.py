"""Encoders: ``f_xi`` over pixels, ``g_xi`` over proprioception.

``ImageEncoder`` is an invariant (CLAUDE.md): 4-layer CNN, 3x3 kernels, 32
channels, stride 2 on the first layer and stride 1 thereafter, LayerNorm + ReLU
after each convolution. Do not change it for speed or convergence.

One genuine ambiguity, flagged rather than silently resolved: "LayerNorm after
each convolution" does not say what it normalises over. ``norm="layer"``
normalises over (C, H, W) with a learnable parameter per position, which is the
literal reading and the default. ``norm="group"`` is GroupNorm(1, C), the same
statistics with per-channel affine and no dependence on the input resolution.
Confirm against the paper before the pilot; the choice is recorded in
DECISIONS.md as open.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..envs.proprio import PROPRIO_OBS_DIM


def conv_output_size(image_size: int) -> int:
    """Spatial extent after the four convolutions. 84 -> 35."""
    size = (image_size - 3) // 2 + 1  # stride-2 layer
    for _ in range(3):  # three stride-1 layers
        size = size - 2
    if size <= 0:
        raise ValueError(f"image_size={image_size} is too small for this encoder")
    return size


def _norm(kind: str, channels: int, spatial: int) -> nn.Module:
    if kind == "layer":
        return nn.LayerNorm([channels, spatial, spatial])
    if kind == "group":
        return nn.GroupNorm(1, channels)
    raise ValueError(f"unknown norm kind {kind!r}; expected 'layer' or 'group'")


class ImageEncoder(nn.Module):
    """``f_xi``. Invariant architecture -- see the module docstring."""

    def __init__(
        self,
        in_channels: int = 9,
        image_size: int = 84,
        channels: int = 32,
        norm: str = "layer",
    ):
        super().__init__()
        self.image_size = image_size

        sizes = [(image_size - 3) // 2 + 1]
        for _ in range(3):
            sizes.append(sizes[-1] - 2)

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=2),
            _norm(norm, channels, sizes[0]),
            nn.ReLU(inplace=True),
        ]
        for spatial in sizes[1:]:
            layers += [
                nn.Conv2d(channels, channels, kernel_size=3, stride=1),
                _norm(norm, channels, spatial),
                nn.ReLU(inplace=True),
            ]

        self.convnet = nn.Sequential(*layers)
        self.repr_dim = channels * sizes[-1] * sizes[-1]  # 39200 at 84x84

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """``pixels``: (B, C, H, W) uint8 or float in [0, 255]."""
        x = pixels.float() / 255.0 - 0.5
        return self.convnet(x).flatten(1)


class ProprioEncoder(nn.Module):
    """``g_xi``. Receives the 13-D gripper-only slice + goal (Landmine 3).

    The paper does not pin this MLP's width or depth the way it pins the CNN, so
    unlike ``ImageEncoder`` this is adaptable -- but any change still belongs in
    DECISIONS.md, because it changes the fusion input.
    """

    def __init__(
        self,
        in_dim: int = PROPRIO_OBS_DIM,
        hidden_dim: int = 512,
        out_dim: int = 128,
    ):
        super().__init__()
        if in_dim != PROPRIO_OBS_DIM:
            raise ValueError(
                f"g_xi input must be {PROPRIO_OBS_DIM}-D (Landmine 3); got {in_dim}. "
                "Widening it lets block state bypass the occlusion."
            )
        self.repr_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(proprio.float())
