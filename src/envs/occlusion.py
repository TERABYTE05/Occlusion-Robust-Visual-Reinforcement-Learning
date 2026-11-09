from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

from .fetch_pixels import (
    block_world_position,
    camera_config,
    model_fovy,
    project_world_to_pixel,
)

DEFAULT_OCCLUDER_COLOR = (110, 110, 110)

DEFAULT_STATIC_RECT = (0.204, 0.307, 0.793, 0.731)


def _to_pixels(rect: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    c0 = int(round(x0 * width))
    c1 = int(round(x1 * width))
    r0 = int(round(y0 * height))
    r1 = int(round(y1 * height))
    c0, c1 = sorted((max(0, c0), min(width, c1)))
    r0, r1 = sorted((max(0, r0), min(height, r1)))
    return c0, r0, c1, r1


class Occluder:

    def reset(self, env) -> None:
        pass

    def rect_pixels(self, frame: np.ndarray, env) -> tuple[int, int, int, int] | None:
        raise NotImplementedError

    def __call__(self, frame: np.ndarray, env) -> np.ndarray:
        box = self.rect_pixels(frame, env)
        if box is None:
            return frame
        c0, r0, c1, r1 = box
        if c1 <= c0 or r1 <= r0:
            return frame
        frame = frame.copy()
        frame[r0:r1, c0:c1] = np.asarray(self.color, dtype=np.uint8)
        return frame

    def covers(self, col: float, row: float, frame_shape, env=None) -> bool:
        dummy = np.empty(frame_shape, dtype=np.uint8)
        box = self.rect_pixels(dummy, env)
        if box is None:
            return False
        c0, r0, c1, r1 = box
        return c0 <= col < c1 and r0 <= row < r1


class NoOcclusion(Occluder):

    color = DEFAULT_OCCLUDER_COLOR

    def rect_pixels(self, frame, env):
        return None

    def __call__(self, frame, env):
        return frame


class StaticOccluder(Occluder):

    def __init__(self, rect=DEFAULT_STATIC_RECT, color=DEFAULT_OCCLUDER_COLOR):
        self.rect = tuple(float(v) for v in rect)
        self.color = tuple(int(v) for v in color)

    def rect_pixels(self, frame, env):
        h, w = frame.shape[:2]
        return _to_pixels(self.rect, w, h)


class EpisodeStaticOccluder(Occluder):

    def __init__(
        self,
        size=(0.42, 0.42),
        center_range=((0.35, 0.65), (0.35, 0.65)),
        color=DEFAULT_OCCLUDER_COLOR,
        seed: int | None = None,
    ):
        self.size = tuple(float(v) for v in size)
        self.center_range = center_range
        self.color = tuple(int(v) for v in color)
        self._rng = np.random.default_rng(seed)
        self._rect = None
        self.reset(None)

    def reset(self, env) -> None:
        (cx_lo, cx_hi), (cy_lo, cy_hi) = self.center_range
        cx = float(self._rng.uniform(cx_lo, cx_hi))
        cy = float(self._rng.uniform(cy_lo, cy_hi))
        w, h = self.size
        self._rect = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def rect_pixels(self, frame, env):
        fh, fw = frame.shape[:2]
        return _to_pixels(self._rect, fw, fh)


class TrackingOccluder(Occluder):

    def __init__(
        self,
        size=(0.30, 0.30),
        color=DEFAULT_OCCLUDER_COLOR,
        camera: str | None = None,
        acknowledge_leak: bool = False,
    ):
        if not acknowledge_leak:
            raise ValueError(
                "TrackingOccluder leaks the block position through the mask's own "
                "location and invalidates the A/B/C comparison -- see "
                "src/envs/occlusion.py. Pass acknowledge_leak=True only for a "
                "deliberate ablation, never for a reported run."
            )
        warnings.warn(
            "TrackingOccluder is in use: the occlusion is not information-"
            "preserving and results from this run are not comparable to Config C.",
            RuntimeWarning,
            stacklevel=2,
        )
        self.size = tuple(float(v) for v in size)
        self.color = tuple(int(v) for v in color)
        self.camera = camera

    def rect_pixels(self, frame, env):
        if env is None:
            return None
        h, w = frame.shape[:2]
        col, row = project_world_to_pixel(
            block_world_position(env), camera_config(self.camera), w, h, model_fovy(env)
        )
        sw, sh = self.size
        return _to_pixels(
            (col / w - sw / 2, row / h - sh / 2, col / w + sw / 2, row / h + sh / 2), w, h
        )


_OCCLUDERS = {
    "none": NoOcclusion,
    "static": StaticOccluder,
    "episode_static": EpisodeStaticOccluder,
    "tracking": TrackingOccluder,
}


def make_occluder(spec) -> Occluder:
    if spec is None:
        return NoOcclusion()
    if isinstance(spec, str):
        spec = {"mode": spec}

    spec = dict(spec)
    mode = spec.pop("mode", "none")
    if mode not in _OCCLUDERS:
        raise KeyError(f"unknown occlusion mode {mode!r}; have {sorted(_OCCLUDERS)}")
    return _OCCLUDERS[mode](**spec)
