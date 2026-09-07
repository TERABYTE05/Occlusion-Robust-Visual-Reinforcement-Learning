"""Programmatic occlusion, applied to the rendered RGB frame.

The paper occludes with geometry in the ``dm_control`` scene. We approximate it
with a mask painted onto the rendered frame -- cheaper, installable, and it
never touches the observation dict. Masking the dict would defeat the entire
experiment (Landmine 3).

A SEVENTH LANDMINE, not in CLAUDE.md, lives here
--------------------------------------------------
An occluder that *tracks* the block -- a rectangle re-centred on the block's
projected position every frame -- is not an occlusion. The mask's own position
encodes the block's position exactly, in pixels the encoder can see, so the
agent recovers full observability by reading the rectangle's centroid. Config C
would then differ from Config B only cosmetically, all three curves would sit on
top of each other, and the failure would look like "the mechanism doesn't
transfer" rather than "the experiment measures nothing". It is Landmine 3 again,
in pixel space.

So the default occluder is *static in the frame*: a fixed region of the
workspace the camera cannot see behind, like a physical panel. The block is
hidden while it is in that region and visible when it leaves, which is genuine
intermittent occlusion, and the mask carries no information about where the
block is. ``TrackingOccluder`` exists only for a deliberate ablation and refuses
to be constructed without an explicit acknowledgement.
"""

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

#: Mid-grey. A neutral, opaque occluder; black would be indistinguishable from
#: unlit background regions in some camera poses.
DEFAULT_OCCLUDER_COLOR = (110, 110, 110)

#: Fixed occluded region, as (x0, y0, x1, y1) fractions of the frame.
#:
#: Measured against camera v1: the block's projected centre over 500 fresh
#: resets spans cols [22.1, 61.6] and rows [30.8, 56.4], and the goal region is
#: the same box. Padded by 5 px so the whole cube is hidden rather than just its
#: centre. Covers 25% of the frame -- the workspace goes, the arm and the
#: surroundings stay.
#:
#: Tied to CAMERA_CONFIG_VERSION: retargeting the camera moves the workspace in
#: the frame and this must be recomputed. `tests/test_occlusion.py` checks the
#: block falls inside it across 50 fresh resets, and
#: `scripts/contact_sheet.py --occlusion static --annotate` shows you why when
#: it does not.
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
    """Base class. Occluders receive the frame only -- never the obs dict."""

    def reset(self, env) -> None:  # noqa: D401 - hook, most occluders are stateless
        """Called on ``env.reset()``, before the first frame is produced."""

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
        """Whether a pixel coordinate falls inside the mask."""
        dummy = np.empty(frame_shape, dtype=np.uint8)
        box = self.rect_pixels(dummy, env)
        if box is None:
            return False
        c0, r0, c1, r1 = box
        return c0 <= col < c1 and r0 <= row < r1


class NoOcclusion(Occluder):
    """Configs A and the state anchor. Frames pass through untouched."""

    color = DEFAULT_OCCLUDER_COLOR

    def rect_pixels(self, frame, env):
        return None

    def __call__(self, frame, env):
        return frame


class StaticOccluder(Occluder):
    """A fixed region of the frame the agent cannot see into. The default.

    Carries no information about the block, because it does not move.
    """

    def __init__(self, rect=DEFAULT_STATIC_RECT, color=DEFAULT_OCCLUDER_COLOR):
        self.rect = tuple(float(v) for v in rect)
        self.color = tuple(int(v) for v in color)

    def rect_pixels(self, frame, env):
        h, w = frame.shape[:2]
        return _to_pixels(self.rect, w, h)


class EpisodeStaticOccluder(Occluder):
    """Position resampled once per episode, then frozen for the episode.

    Adds variety across episodes without leaking within one: the mask is
    constant while the block moves, so its position cannot encode the block's.
    """

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
    """Follows the block's projected position. LEAKS THE BLOCK POSITION.

    See the module docstring. Not a valid Config C occluder. Kept only so the
    leak can be demonstrated deliberately as an ablation, and it refuses to be
    built unless the caller says so in as many words.
    """

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
    """Build an occluder from a config value.

    Accepts ``None``/``"none"``, a mode name, or a dict with a ``mode`` key plus
    that occluder's keyword arguments.
    """
    if spec is None:
        return NoOcclusion()
    if isinstance(spec, str):
        spec = {"mode": spec}

    spec = dict(spec)
    mode = spec.pop("mode", "none")
    if mode not in _OCCLUDERS:
        raise KeyError(f"unknown occlusion mode {mode!r}; have {sorted(_OCCLUDERS)}")
    return _OCCLUDERS[mode](**spec)
