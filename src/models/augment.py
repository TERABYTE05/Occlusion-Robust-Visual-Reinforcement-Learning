"""Random-shift augmentation: +-4 px pad-and-crop. An invariant (CLAUDE.md).

This is DrQ-v2's augmentation and it is the reason DrQ-v2 is sample-efficient,
so the padding width and the way the shift is drawn are part of the method, not
knobs.

The property that matters: **one shift per sample, shared by every frame in its
stack**. The stack arrives as a single 9-channel image and ``grid_sample``
applies one grid across all channels, so the three frames move together. Shifting
them independently would inject apparent motion that the encoder would have to
learn to ignore, and would corrupt exactly the temporal signal the stack exists
to provide.

Padding is ``replicate``: edge pixels are extended rather than filled with zeros,
so the augmentation never introduces a black border that the encoder could use
to infer the shift.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

#: The paper's value. Do not tune it.
DEFAULT_PAD = 4


class RandomShiftAug(nn.Module):
    """Pad by ``pad`` pixels, then crop back at a random integer offset."""

    def __init__(self, pad: int = DEFAULT_PAD):
        super().__init__()
        self.pad = int(pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, C, H, W) float. Returns the same shape."""
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got {tuple(x.shape)}")
        batch, _, height, width = x.shape
        if height != width:
            raise ValueError(f"expected square frames, got {height}x{width}")

        padded = F.pad(x, (self.pad,) * 4, mode="replicate")

        # A grid over the padded image whose top-left cell is the original
        # image's top-left pixel; adding an integer offset crops elsewhere.
        span = height + 2 * self.pad
        eps = 1.0 / span
        axis = torch.linspace(
            -1.0 + eps, 1.0 - eps, span, device=x.device, dtype=x.dtype
        )[:height]
        axis = axis.unsqueeze(0).repeat(height, 1).unsqueeze(2)
        base = torch.cat([axis, axis.transpose(1, 0)], dim=2)
        base = base.unsqueeze(0).repeat(batch, 1, 1, 1)

        # One shift per sample -- shape (B, 1, 1, 2) broadcasts over every
        # pixel and, because grid_sample shares a grid across channels, over
        # every frame of the stack.
        shift = torch.randint(
            0, 2 * self.pad + 1, size=(batch, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / span

        return F.grid_sample(
            padded, base + shift, padding_mode="zeros", align_corners=False
        )
