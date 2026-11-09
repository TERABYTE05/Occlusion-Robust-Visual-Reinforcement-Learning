from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .fetch_reset_fix import apply_reset_fix
from .proprio import PROPRIO_OBS_DIM, build_proprio

ENV_ID = "FetchPushDense-v3"

DEFAULT_FETCH_CAMERA = {
    "distance": 2.5,
    "azimuth": 132.0,
    "elevation": -14.0,
    "lookat": (1.3, 0.75, 0.55),
}

CAMERA_CONFIGS: dict[str, dict[str, Any]] = {
    "default": DEFAULT_FETCH_CAMERA,
    "v1": {
        "distance": 0.85,
        "azimuth": 180.0,
        "elevation": -45.0,
        "lookat": (1.3455, 0.749, 0.4247),
    },
}

CAMERA_CONFIG_VERSION = "v1"

DEFAULT_FOVY = 45.0


def camera_config(name: str | None = None) -> dict[str, Any]:
    name = name or CAMERA_CONFIG_VERSION
    if name not in CAMERA_CONFIGS:
        raise KeyError(f"unknown camera config {name!r}; have {sorted(CAMERA_CONFIGS)}")
    cfg = dict(CAMERA_CONFIGS[name])
    cfg["lookat"] = np.asarray(cfg["lookat"], dtype=np.float64)
    return cfg


def camera_frame(cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    az = math.radians(float(cfg["azimuth"]))
    el = math.radians(float(cfg["elevation"]))
    lookat = np.asarray(cfg["lookat"], dtype=np.float64)
    distance = float(cfg["distance"])

    forward = np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
        dtype=np.float64,
    )
    position = lookat - distance * forward

    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    return position, forward, right, up


def project_world_to_pixel(
    point: np.ndarray,
    cfg: Mapping[str, Any],
    width: int,
    height: int,
    fovy: float = DEFAULT_FOVY,
) -> tuple[float, float]:
    position, forward, right, up = camera_frame(cfg)
    delta = np.asarray(point, dtype=np.float64) - position

    depth = float(delta @ forward)
    if depth <= 1e-6:
        raise ValueError("point is behind the camera; cannot project")

    focal = (height / 2.0) / math.tan(math.radians(fovy) / 2.0)
    col = width / 2.0 + focal * float(delta @ right) / depth
    row = height / 2.0 - focal * float(delta @ up) / depth
    return col, row


def model_fovy(env) -> float:
    try:
        return float(env.unwrapped.model.vis.global_.fovy)
    except Exception:
        return DEFAULT_FOVY


def block_world_position(env) -> np.ndarray:
    unwrapped = env.unwrapped
    xpos = unwrapped._utils.get_site_xpos(unwrapped.model, unwrapped.data, "object0")
    return np.array(xpos, dtype=np.float64, copy=True)


def _resize(frame: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    if frame.shape[0] == size and frame.shape[1] == size:
        return frame
    return np.asarray(
        Image.fromarray(frame).resize((size, size), Image.BILINEAR), dtype=np.uint8
    )


def make_raw_env(
    env_id: str = ENV_ID,
    camera: str | None = None,
    render_size: int = 84,
    render: bool = True,
    max_episode_steps: int | None = None,
):
    import gymnasium as gym
    import gymnasium_robotics

    kwargs: dict[str, Any] = {}
    if render:
        kwargs["render_mode"] = "rgb_array"
        kwargs["default_camera_config"] = camera_config(camera)
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps

    env = gym.make(env_id, **kwargs)

    apply_reset_fix(env)

    if render:
        env.unwrapped.width = render_size
        env.unwrapped.height = render_size

    return env


class PixelProprioWrapper:

    def __init__(self, env, image_size: int = 84, frame_transform=None):
        self.env = env
        self.image_size = image_size
        self.frame_transform = frame_transform
        self._last_frame: np.ndarray | None = None

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.__dict__["env"], name)

    def _frame(self) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            raise RuntimeError(
                "env.render() returned None. The env was built without "
                "render_mode='rgb_array'."
            )
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"expected an (H,W,3) uint8 frame, got {frame.shape}")

        frame = _resize(frame, self.image_size)
        if self.frame_transform is not None:
            frame = self.frame_transform(frame, self.env)
        self._last_frame = frame
        return frame

    def _observation(self, obs_dict) -> dict[str, np.ndarray]:
        return {"pixels": self._frame(), "proprio": build_proprio(obs_dict)}

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        if self.frame_transform is not None and hasattr(self.frame_transform, "reset"):
            self.frame_transform.reset(self.env)
        return self._observation(obs_dict), info

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        return self._observation(obs_dict), reward, terminated, truncated, info

    @property
    def last_frame(self) -> np.ndarray | None:
        return self._last_frame

    def close(self):
        return self.env.close()


def observation_shapes(image_size: int = 84, frame_stack: int = 3):
    return {
        "pixels": (3 * frame_stack, image_size, image_size),
        "proprio": (PROPRIO_OBS_DIM,),
    }
