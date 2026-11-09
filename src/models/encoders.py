from __future__ import annotations

import torch
import torch.nn as nn

from ..envs.proprio import PROPRIO_OBS_DIM


def conv_output_size(image_size: int) -> int:
    size = (image_size - 3) // 2 + 1
    for _ in range(3):
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
        self.repr_dim = channels * sizes[-1] * sizes[-1]

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = pixels.float() / 255.0 - 0.5
        return self.convnet(x).flatten(1)


class ProprioEncoder(nn.Module):

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


class StateEncoder(nn.Module):

    def __init__(self, obs_dim: int):
        super().__init__()
        self.repr_dim = int(obs_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float()
