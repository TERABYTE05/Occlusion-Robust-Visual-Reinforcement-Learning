from __future__ import annotations

import torch
import torch.nn as nn


class SimplexNorm(nn.Module):

    def __init__(self, dim: int, per_partition: int = 8, tau: float = 1.0):
        super().__init__()
        dim = int(dim)
        per_partition = int(per_partition)

        if per_partition <= 0:
            raise ValueError(f"per_partition must be positive; got {per_partition}")
        if dim % per_partition != 0:
            raise ValueError(
                f"dim={dim} is not divisible by per_partition={per_partition}; "
                "a remainder would silently mis-shape the representation"
            )
        if tau <= 0:
            raise ValueError(f"tau must be positive; got {tau}")

        self.dim = dim
        self.per_partition = per_partition
        self.partitions = dim // per_partition
        self.tau = float(tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.shape[-1]}")

        shape = x.shape
        partitioned = x.float().view(*shape[:-1], self.partitions, self.per_partition)
        normalised = torch.softmax(partitioned / self.tau, dim=-1)
        return normalised.view(shape).to(x.dtype)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, partitions={self.partitions}, "
            f"per_partition={self.per_partition}, tau={self.tau}"
        )
