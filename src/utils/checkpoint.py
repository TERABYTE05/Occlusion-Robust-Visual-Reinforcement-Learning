from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import torch

CHECKPOINT_NAMES = ("latest.pt", "best.pt", "final.pt")

REPLAY_NAME = "replay"


def save_checkpoint(
    path: str | pathlib.Path,
    agent,
    step: int,
    rng: np.random.Generator,
    extra: dict[str, Any] | None = None,
) -> None:
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
