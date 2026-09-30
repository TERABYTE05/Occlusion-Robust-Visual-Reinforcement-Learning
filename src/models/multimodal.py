"""The full encoder stack: ``f_xi`` + ``g_zeta`` + ``h_psi``.

This is what the agent treats as "the encoder", and it is the piece that makes
dualization real rather than decorative.

It exposes **two** representations rather than one, because the paper's two
losses evaluate the critic at different points (D20):

* Eq. 2 — the critic loss — scores ``Q_phi`` at ``z_mm_c`` and updates
  ``phi_k``, ``psi_critic``, ``xi`` and ``zeta``;
* Eq. 3 — the actor loss — scores ``Q_phi`` at ``z_mm_a`` and updates ``theta``
  and ``psi_actor``, and nothing else.

The TD target mixes them: the bootstrap value is ``Q_bar(z_mm_c_{t+n}, a_{t+n})``
while the action is ``a_{t+n} = pi_theta(z_mm_a_{t+n}) + eps``.

**Where the detach goes.** §4.2: "we prevent the actor's gradients from updating
the image and proprioception encoders". That is ``xi`` and ``zeta`` -- *not*
``psi_actor``, which Eq. 3 explicitly optimises. So the detach sits on the
encoder outputs, before the actor's fusion head. Detaching the fused
representation instead would leave ``psi_actor`` with no gradient path at all;
it would never train, and Config C would become Config B carrying an extra
unused module while still reporting itself as dualized. That failure produces a
plausible curve rather than an error, which is why
``tests/test_multimodal_encoder.py`` asserts both halves of the routing.

For a single-headed encoder -- the state anchor, Configs A and B -- there is no
fusion head between the actor and the encoders, so detaching the output is the
same thing, and the behaviour is unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from ..envs.proprio import PROPRIO_OBS_DIM
from .augment import RandomShiftAug
from .encoders import ImageEncoder, ProprioEncoder
from .fusion import build_fusion


class MultimodalEncoder(nn.Module):
    """``f_xi``, ``g_zeta`` and the fusion stage, assembled from a config."""

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
        """``aug(o_image)`` -- random shift, applied to a training batch only.

        Every equation in the paper encodes ``aug(o^image)``, never the raw
        frame, and the augmentation is why DrQ-v2 is sample-efficient. It is
        applied in the update and **not** when acting: the policy must see the
        environment as it is.

        The shift is drawn once per sample and shared across the frames of its
        stack (`RandomShiftAug`), so the temporal signal the stack exists to
        carry is preserved.
        """
        return {**obs, "pixels": self.aug(obs["pixels"].float())}

    # -- the shared trunk -------------------------------------------------
    def _encode(self, obs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.image(obs["pixels"]), self.proprio(obs["proprio"])

    # -- the two representations -----------------------------------------
    def critic_repr(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``z_mm_c``. Gradients reach the encoders and ``psi_critic``."""
        z_image, z_prop = self._encode(obs)
        return self.fusion(z_image, z_prop, head="critic")

    def actor_repr(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``z_mm_a``, with the gradient path the configuration calls for.

        **Dualized (Config C).** Eq. 3 optimises ``theta`` *and* ``psi_actor``,
        so the detach sits on the encoder outputs and the actor's fusion head
        still learns. This one line is the difference between a real Config C
        and a Config B wearing its name.

        **Conventional (Configs A and B).** §4.2 is explicit that in the
        conventional methods "the actor is optimized solely with respect to
        theta". There is one shared fusion module and the critic loss owns it,
        so the whole fused vector is detached.
        """
        z_image, z_prop = self._encode(obs)
        if self.fusion.dualized:
            return self.fusion(z_image.detach(), z_prop.detach(), head="actor")
        return self.fusion(z_image, z_prop, head="actor").detach()

    def forward(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Default path is the critic's, so single-headed callers behave as before."""
        return self.critic_repr(obs)

    # -- who owns which parameters ----------------------------------------
    def actor_parameters(self) -> list[nn.Parameter]:
        """Encoder-side parameters the **actor loss** steps.

        Only ``psi_actor``, and only when dualized. Empty for the conventional
        design, where the actor owns nothing but ``theta``.
        """
        if not self.fusion.dualized:
            return []
        return list(self.fusion.actor_head.parameters())

    def critic_parameters(self) -> list[nn.Parameter]:
        """Encoder-side parameters the **critic loss** steps: xi, zeta, psi_critic."""
        return (
            list(self.image.parameters())
            + list(self.proprio.parameters())
            + list(self.fusion.critic_head.parameters())
        )
