"""The one thing the suite never checked: that rendering actually works.

Blocker B3, open since 2026-09-07. Every other test is physics-only, which is
why the whole suite passed on the first Linux box -- a machine with no GL
context at all, where `env.render()` could not have worked. Rendering is the
single most platform-dependent thing in the project and the one that decides the
timeline (Landmine 4), so it should not be verified only by eye on contact
sheets.

These are deliberately cheap: a handful of frames, no training, no assertions
about content beyond what a broken renderer would violate.
"""

import numpy as np
import pytest

from src.envs.make_env import make_env
from src.utils.gl import resolved_backend


pytestmark = pytest.mark.skipif(
    resolved_backend() not in {"egl", "glfw"},
    reason=f"no GPU rendering backend (got {resolved_backend()!r}); see Landmine 4",
)


@pytest.fixture(scope="module")
def pixel_env():
    env = make_env({"env": {"pixels": True, "image_size": 84, "frame_stack": 3}}, seed=0)
    yield env
    env.close()


def test_the_active_backend_is_a_gpu_one():
    """osmesa is CPU rasterisation and would silently cost the schedule."""
    assert resolved_backend() in {"egl", "glfw"}, (
        f"backend is {resolved_backend()!r}; see Landmine 4"
    )


def test_reset_returns_a_stacked_uint8_observation(pixel_env):
    obs, _info = pixel_env.reset()
    assert set(obs) >= {"pixels", "proprio"}
    assert obs["pixels"].shape == (9, 84, 84), "3 frames x 3 channels, channel-first"
    assert obs["pixels"].dtype == np.uint8
    assert obs["proprio"].shape == (13,), "the 13-D slice (Landmine 3)"


def test_every_step_returns_a_well_formed_frame(pixel_env):
    pixel_env.reset()
    for _ in range(5):
        obs, _r, _term, _trunc, _info = pixel_env.step(pixel_env.action_space.sample())
        assert obs["pixels"].shape == (9, 84, 84)
        assert obs["pixels"].dtype == np.uint8


def test_the_frame_is_not_blank(pixel_env):
    """A context that fails quietly tends to hand back a constant buffer."""
    obs, _info = pixel_env.reset()
    assert obs["pixels"].std() > 1.0, "rendered frame is nearly constant"
    assert obs["pixels"].max() > obs["pixels"].min()


def test_the_scene_changes_as_the_arm_moves(pixel_env):
    """Guards against a renderer returning the same cached frame forever."""
    pixel_env.reset()
    first = pixel_env.step(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))[0]["pixels"].copy()
    for _ in range(10):
        later = pixel_env.step(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))[0]["pixels"]
    assert not np.array_equal(first, later), "frames are identical after moving the arm"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_occluder_actually_changes_the_frame():
    """Config B and C differ from A only here, so it must be visible in pixels.

    The GLFW teardown warning this suppresses is the documented harmless noise
    from MuJoCo tearing down a context at close (CLAUDE.md, Landmine 4). It is
    filtered rather than tolerated so the suite stays clean -- a run that prints
    a warning every time teaches people to ignore warnings.
    """
    clean = make_env({"env": {"pixels": True, "occlusion": "none"}}, seed=0)
    masked = make_env({"env": {"pixels": True, "occlusion": {"mode": "static"}}}, seed=0)
    try:
        a = clean.reset()[0]["pixels"]
        b = masked.reset()[0]["pixels"]
        assert not np.array_equal(a, b), "the occluder left the frame untouched"
    finally:
        clean.close()
        masked.close()
