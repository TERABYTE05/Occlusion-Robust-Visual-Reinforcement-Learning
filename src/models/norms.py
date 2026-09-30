"""Representation normalization (Noh et al. §4.1.3, Eq. 1).

Half of the paper's contribution. Two normalizations sit inside the fusion
module, in this order and no other:

    Linear -> LayerNorm -> SimplexNorm

``LayerNorm`` is ordinary and is what the paper credits with stabilising
training across layers and keeping gradient scales in hand. ``SimplexNorm`` is
the novel part and is implemented here.

Given a representation ``z`` of dimension ``d``, cut it into ``L`` partitions of
``V`` and apply a softmax at temperature ``tau`` *within each partition
independently*:

    z_hat = [p_1, ..., p_L],   (p_i)_j = exp(z_ij / tau) / sum_k exp(z_ik / tau)

Every partition is then a point on a probability simplex — non-negative and
summing to one — which makes the representation naturally sparse without a hard
constraint being imposed on it. The paper describes it as a "soft" version of
VQ-VAE's discrete codes: continuous value partitions rather than selection from
a codebook.

``d=128``, ``L=16``, ``V=8`` and ``tau=1`` are **scientific invariants**
(CLAUDE.md). Do not change them for convergence, for speed, or because a value
looks unusual.

A note on ``tau`` and mixed precision: the fallback ladder names this softmax as
the first suspect when NaNs appear (rung 4), because at ``tau=1`` with fp16 the
exponentials can underflow. The softmax below runs in float32 regardless of the
surrounding autocast context for exactly that reason, and casts back afterwards.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimplexNorm(nn.Module):
    """Project a representation onto ``L`` simplices of dimension ``V``.

    Parameters
    ----------
    dim:
        Width of the representation being normalised. Must be divisible by
        ``per_partition``.
    per_partition:
        ``V`` — the dimensionality of each simplex. The number of partitions
        ``L`` follows as ``dim // per_partition``.
    tau:
        Softmax temperature. Lower is sparser; the paper's value is 1.
    """

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
        """``x``: (..., dim). Returns the same shape, partition-wise normalised."""
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.shape[-1]}")

        shape = x.shape
        partitioned = x.float().view(*shape[:-1], self.partitions, self.per_partition)
        # float32 deliberately: at tau=1 under AMP the exponentials can underflow,
        # and a NaN here is indistinguishable from a broken learning rate.
        normalised = torch.softmax(partitioned / self.tau, dim=-1)
        return normalised.view(shape).to(x.dtype)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, partitions={self.partitions}, "
            f"per_partition={self.per_partition}, tau={self.tau}"
        )
