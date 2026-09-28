"""Deterministic evaluation.

Evaluation runs the policy with no exploration noise, on its own environment
instance so it cannot disturb the training environment's episode boundaries or
its random stream.

The headline metric is the **success rate**, not the return. FetchPushDense
pays a shaped distance reward, so a policy can improve its return substantially
while never once getting the block to the goal; the roadmap's gates and the
report's results table are both stated in success rate, and the return is kept
only as a secondary signal.
"""

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
    """Run ``episodes`` noise-free episodes and summarise them."""
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
            # FetchPush reports success per step and the block can be pushed
            # back out again, so the episode counts as a success if the goal
            # was ever reached, matching how gymnasium-robotics reports it.
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
