from __future__ import annotations

from typing import Any, Callable

import numpy as np


def evaluate(
    env,
    agent,
    episodes: int,
    obs_fn: Callable[[Any], np.ndarray],
    step: int = 0,
) -> dict[str, float]:
    returns: list[float] = []
    successes: list[float] = []
    lengths: list[int] = []

    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        total = 0.0
        length = 0
        success = float(info.get("is_success", 0.0))

        while not done:
            action = agent.act(obs_fn(obs), step=step, eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            length += 1
            success = max(success, float(info.get("is_success", 0.0)))
            done = bool(terminated or truncated)

        returns.append(total)
        successes.append(success)
        lengths.append(length)

    return {
        "success_rate": float(np.mean(successes)),
        "episode_return": float(np.mean(returns)),
        "episode_length": float(np.mean(lengths)),
        "episodes": float(len(returns)),
    }
