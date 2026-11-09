from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_PAD = 4


class RandomShiftAug(nn.Module):

    def __init__(self, pad: int = DEFAULT_PAD):
        super().__init__()
        self.pad = int(pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got {tuple(x.shape)}")
        batch, _, height, width = x.shape
        if height != width:
            raise ValueError(f"expected square frames, got {height}x{width}")

        padded = F.pad(x, (self.pad,) * 4, mode="replicate")

        span = height + 2 * self.pad
        eps = 1.0 / span
        axis = torch.linspace(
            -1.0 + eps, 1.0 - eps, span, device=x.device, dtype=x.dtype
        )[:height]
        axis = axis.unsqueeze(0).repeat(height, 1).unsqueeze(2)
        base = torch.cat([axis, axis.transpose(1, 0)], dim=2)
        base = base.unsqueeze(0).repeat(batch, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(batch, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / span

        return F.grid_sample(
            padded, base + shift, padding_mode="zeros", align_corners=False
        )
