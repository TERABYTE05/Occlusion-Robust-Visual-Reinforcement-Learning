"""FetchPush as a pixels + proprioception environment.

Two Landmines live in this file.

Landmine 5 (silent). Fetch renders ``rgb_array`` from a wide external
viewpoint. Downsampled to 84x84 the block can occupy almost nothing, the image
encoder has no usable signal, and the whole vision branch contributes noise --
which reads as "the method doesn't work". The camera therefore has to be
retargeted, and confirmed *by eye* at true 84x84 with ``scripts/contact_sheet.py``.

Landmine 1. The env id is ``FetchPushDense-v3``. Never the sparse default.

Any change to ``CAMERA_CONFIGS["v1"]`` invalidates every run completed before
it. Bump ``CAMERA_CONFIG_VERSION``, add a new named config rather than editing
the old one in place, and record it in DECISIONS.md.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .fetch_reset_fix import apply_reset_fix
from .proprio import PROPRIO_OBS_DIM, build_proprio

#: The dense-reward push task. Landmine 1 -- the sparse default is unlearnable
#: for DDPG with n-step returns and no HER inside our step budget.
ENV_ID = "FetchPushDense-v3"

#: gymnasium-robotics' default Fetch viewpoint. Recorded for comparison only --
#: at 84x84 the block is a handful of pixels. Do not use it for runs.
DEFAULT_FETCH_CAMERA = {
    "distance": 2.5,
    "azimuth": 132.0,
    "elevation": -14.0,
    "lookat": (1.3, 0.75, 0.55),
}

#: Ours. Centred on the workspace (``initial_gripper_xpos`` at table height),
#: close enough that the block is legible at 84x84 and still wide enough that
#: the task region never leaves the frame.
#:
#: Chosen by sweeping distance and elevation against two measured quantities:
#: the block's apparent width, ``focal * 0.05 / depth``, and the fraction of the
#: reachable table region that stays inside the frame. At distance 0.85 the
#: block is ~6.0 px across (the default view gives ~2.0 px) and the spawn region
#: +- 0.22 m is fully framed with margin.
#:
#: Still to do on the lab machine, because this box has no GL context: confirm
#: by eye at true 84x84 with `scripts/contact_sheet.py` that the block is
#: actually distinguishable from the table, not merely large enough in principle.
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

#: MuJoCo's default vertical field of view, in degrees. Overridden per model by
#: ``model.vis.global_.fovy`` wherever we can read it.
DEFAULT_FOVY = 45.0


def camera_config(name: str | None = None) -> dict[str, Any]:
    """Return a named camera config as a plain dict (lookat as an array)."""
    name = name or CAMERA_CONFIG_VERSION
    if name not in CAMERA_CONFIGS:
        raise KeyError(f"unknown camera config {name!r}; have {sorted(CAMERA_CONFIGS)}")
    cfg = dict(CAMERA_CONFIGS[name])
    cfg["lookat"] = np.asarray(cfg["lookat"], dtype=np.float64)
    return cfg


# --------------------------------------------------------------------------
# world -> pixel projection
# --------------------------------------------------------------------------


def camera_frame(cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Camera position and orthonormal basis for a MuJoCo free camera.

    Returns ``(position, forward, right, up)`` in world coordinates, where
    ``forward`` points from the camera towards ``lookat``.
    """
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
    """Project a world xyz onto the rendered frame. Returns ``(col, row)``.

    This is a reimplementation of MuJoCo's free-camera projection, not a
    read-out from the renderer, so it is only as correct as the assumptions
    above. Confirm it once, by eye, with
    ``scripts/contact_sheet.py --annotate`` before trusting
    ``tests/test_occlusion.py``, which uses it to decide whether the mask
    actually covers the block.
    """
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
    """The model's vertical FOV in degrees, falling back to MuJoCo's default."""
    try:
        return float(env.unwrapped.model.vis.global_.fovy)
    except Exception:
        return DEFAULT_FOVY


def block_world_position(env) -> np.ndarray:
    """Ground-truth block xyz, as a copy.

    Privileged state. Legitimate for placing/validating occluders and for the
    state-based sanity anchor. It must never reach ``g_xi`` (Landmine 3).

    The copy is not optional. ``get_site_xpos`` hands back a live view into
    ``data.site_xpos``, so a caller that stores the result gets an array that
    silently changes on the next ``env.step()`` -- every sample in a collected
    list ends up showing the final state.
    """
    unwrapped = env.unwrapped
    xpos = unwrapped._utils.get_site_xpos(unwrapped.model, unwrapped.data, "object0")
    return np.array(xpos, dtype=np.float64, copy=True)


# --------------------------------------------------------------------------
# environment construction
# --------------------------------------------------------------------------


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
    """Build the underlying FetchPush env with our camera attached.

    ``render=False`` skips the ``rgb_array`` render mode entirely, which is what
    the state-based sanity anchor and stage 1 of the FPS benchmark want -- with
    a render mode set, MuJoCo still builds an offscreen context.
    """
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401  (registers the Fetch envs)

    kwargs: dict[str, Any] = {}
    if render:
        kwargs["render_mode"] = "rgb_array"
        kwargs["default_camera_config"] = camera_config(camera)
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps

    env = gym.make(env_id, **kwargs)

    # Landmine 7. Must happen before the first reset, while the post-_env_setup
    # state is still intact. Without it the block resets onto the floor and the
    # task is unsolvable.
    apply_reset_fix(env)

    if render:
        # The renderer is created lazily on the first render() call, so setting
        # the viewport here is early enough. Rendering straight at the training
        # resolution avoids paying for pixels we immediately throw away.
        env.unwrapped.width = render_size
        env.unwrapped.height = render_size

    return env


class PixelProprioWrapper:
    """Turns the FetchPush dict observation into ``{"pixels", "proprio"}``.

    Deliberately not a ``gym.ObservationWrapper``: an observation wrapper only
    sees the dict, and we need to call ``render()`` on every step, cache the
    frame, and let the occlusion wrapper act on the *rendered frame* rather than
    on the observation dict (CLAUDE.md, Conventions).
    """

    def __init__(self, env, image_size: int = 84, frame_transform=None):
        self.env = env
        self.image_size = image_size
        #: Applied to the raw rendered frame. This is where occlusion attaches.
        self.frame_transform = frame_transform
        self._last_frame: np.ndarray | None = None

    # -- plumbing ---------------------------------------------------------
    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.__dict__["env"], name)

    # -- core -------------------------------------------------------------
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
        # Landmine 6: terminated and truncated stay separate all the way into
        # the buffer. Never collapse them to a single `done`.
        return self._observation(obs_dict), reward, terminated, truncated, info

    @property
    def last_frame(self) -> np.ndarray | None:
        """The most recent post-transform frame, for contact sheets and video."""
        return self._last_frame

    def close(self):
        return self.env.close()


def observation_shapes(image_size: int = 84, frame_stack: int = 3):
    """Shapes the encoders should expect, in one place."""
    return {
        "pixels": (3 * frame_stack, image_size, image_size),
        "proprio": (PROPRIO_OBS_DIM,),
    }
