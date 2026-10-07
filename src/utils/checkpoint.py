"""Run-state persistence.

Runs are four hours or more on a shared machine and the roadmap requires
checkpointing every 50K steps with working resume, verified by a real
kill-restart rather than a clean exit (G2).

Two things make resume *real* rather than nominal, and both are easy to omit:

* the **optimizer state**. Adam carries first and second moment estimates; a
  resume that reloads only ``state_dict()`` restarts them at zero, so every
  restart introduces a fresh warm-up transient into the middle of a run;
* the **random streams**, both NumPy's and torch's. Without them a restarted
  run re-draws the same exploration noise it drew before, so the post-restart
  trajectory is correlated with the pre-restart one in a way nothing records.

Only three files are ever written per run -- ``latest``, ``best`` and
``final``. One file per 50K interval across nine runs is what exhausted the
disk on the previous machine and forced the September platform migration; the
bound is deliberate, not incidental.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import torch

#: Every checkpoint a run may write. The set is closed on purpose.
CHECKPOINT_NAMES = ("latest.pt", "best.pt", "final.pt")

#: The replay snapshot that makes ``--resume`` valid. Not a checkpoint: it holds
#: no model state, it is written only beside ``latest.pt``, and it is removed
#: when a run finishes, because a finished run has nothing to resume and at
#: pixel size it is ~2.1 GB.
#:
#: It is a *directory* of ``.npy`` files, not a single pickle. Writing it with
#: ``torch.save`` raised RSS by 7.9 GB to persist 2.12 GB and was killed by
#: systemd-oomd mid-write, costing the first pilot run (`RUNLOG.md` 2026-10-01).
REPLAY_NAME = "replay"


def save_checkpoint(
    path: str | pathlib.Path,
    agent,
    step: int,
    rng: np.random.Generator,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write the complete run state to ``path``.

    The write goes to a temporary file and is then renamed, so a process killed
    mid-write leaves the previous good checkpoint intact rather than a
    half-written one. That matters because the kill-restart test is deliberately
    ungraceful.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "step": int(step),
        "encoder": agent.encoder.state_dict(),
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "actor_opt": agent.actor_opt.state_dict(),
        "critic_opt": agent.critic_opt.state_dict(),
        "encoder_opt": agent.encoder_opt.state_dict() if agent.encoder_opt is not None else None,
        "rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "extra": dict(extra or {}),
    }

    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | pathlib.Path, agent) -> dict[str, Any]:
    """Restore ``agent`` in place; return the run bookkeeping that is not the agent."""
    payload = torch.load(pathlib.Path(path), map_location=agent.device, weights_only=False)

    agent.encoder.load_state_dict(payload["encoder"])
    agent.actor.load_state_dict(payload["actor"])
    agent.critic.load_state_dict(payload["critic"])
    agent.critic_target.load_state_dict(payload["critic_target"])
    agent.actor_opt.load_state_dict(payload["actor_opt"])
    agent.critic_opt.load_state_dict(payload["critic_opt"])
    if agent.encoder_opt is not None and payload["encoder_opt"] is not None:
        agent.encoder_opt.load_state_dict(payload["encoder_opt"])

    rng = np.random.default_rng()
    rng.bit_generator.state = payload["rng"]

    torch_state = payload["torch_rng"]
    torch.set_rng_state(torch_state.cpu() if torch.is_tensor(torch_state) else torch_state)

    return {
        "step": int(payload["step"]),
        "rng": rng,
        "extra": dict(payload.get("extra", {})),
    }
