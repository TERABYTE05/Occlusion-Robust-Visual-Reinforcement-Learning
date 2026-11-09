from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import numpy as np

from .fetch_pixels import ENV_ID, PixelProprioWrapper, make_raw_env
from .occlusion import make_occluder


class FrameStack:

    def __init__(self, env, k: int = 3):
        self.env = env
        self.k = k
        self._frames: deque[np.ndarray] = deque(maxlen=k)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.__dict__["env"], name)

    def _stacked(self, obs):
        pixels = np.concatenate([f.transpose(2, 0, 1) for f in self._frames], axis=0)
        return {"pixels": pixels, "proprio": obs["proprio"]}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames.clear()
        for _ in range(self.k):
            self._frames.append(obs["pixels"])
        return self._stacked(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs["pixels"])
        return self._stacked(obs), reward, terminated, truncated, info

    def close(self):
        return self.env.close()


def make_env(cfg: Mapping[str, Any], seed: int | None = None):
    env_cfg = dict(cfg.get("env", {}))

    use_pixels = bool(env_cfg.get("pixels", True))
    image_size = int(env_cfg.get("image_size", 84))
    frame_stack = int(env_cfg.get("frame_stack", 3))

    env = make_raw_env(
        env_id=env_cfg.get("id", ENV_ID),
        camera=env_cfg.get("camera"),
        render_size=image_size,
        render=use_pixels,
    )

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    if not use_pixels:
        return env

    occluder = make_occluder(env_cfg.get("occlusion", "none"))
    env = PixelProprioWrapper(env, image_size=image_size, frame_transform=occluder)
    env = FrameStack(env, k=frame_stack)
    return env
