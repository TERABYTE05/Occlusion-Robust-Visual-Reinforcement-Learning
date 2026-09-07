"""Occlusion is applied to the rendered frame, covers the block, and does not
encode where the block is.

The last of those is the seventh landmine (see src/envs/occlusion.py): a mask
that tracks the block hands its position back to the encoder, and Config C
stops measuring anything.
"""

import numpy as np
import pytest

from src.envs.occlusion import (
    DEFAULT_STATIC_RECT,
    EpisodeStaticOccluder,
    NoOcclusion,
    StaticOccluder,
    TrackingOccluder,
    make_occluder,
)

SHAPE = (84, 84, 3)


def _frame(value=200):
    return np.full(SHAPE, value, dtype=np.uint8)


def test_no_occlusion_is_a_passthrough():
    frame = _frame()
    assert np.array_equal(NoOcclusion()(frame, None), frame)


def test_static_occluder_masks_the_configured_rectangle():
    occ = StaticOccluder(rect=(0.25, 0.25, 0.75, 0.75), color=(0, 0, 0))
    out = occ(_frame(200), None)
    assert out[42, 42].tolist() == [0, 0, 0]
    assert out[2, 2].tolist() == [200, 200, 200]


def test_static_occluder_covers_the_expected_pixel_count():
    occ = StaticOccluder(rect=(0.0, 0.0, 0.5, 0.5), color=(0, 0, 0))
    out = occ(_frame(200), None)
    assert int((out == 0).all(axis=2).sum()) == 42 * 42


def test_occluder_does_not_mutate_the_frame_it_is_given():
    frame = _frame()
    original = frame.copy()
    StaticOccluder()(frame, None)
    assert np.array_equal(frame, original)


def test_covers_agrees_with_the_painted_pixels():
    occ = StaticOccluder(rect=(0.25, 0.25, 0.75, 0.75), color=(0, 0, 0))
    out = occ(_frame(200), None)
    for col, row in [(42, 42), (10, 10), (80, 5), (21, 21), (62, 62)]:
        painted = bool((out[row, col] == 0).all())
        assert occ.covers(col, row, SHAPE) == painted


def test_default_static_rect_is_inside_the_frame():
    x0, y0, x1, y1 = DEFAULT_STATIC_RECT
    assert 0.0 <= x0 < x1 <= 1.0
    assert 0.0 <= y0 < y1 <= 1.0


def test_episode_static_is_fixed_within_an_episode():
    """If it moved between steps it would track the block and leak its position."""
    occ = EpisodeStaticOccluder(seed=0)
    first = occ.rect_pixels(_frame(), None)
    for _ in range(10):
        assert occ.rect_pixels(_frame(), None) == first


def test_episode_static_moves_across_episodes():
    occ = EpisodeStaticOccluder(seed=0)
    rects = set()
    for _ in range(10):
        occ.reset(None)
        rects.add(occ.rect_pixels(_frame(), None))
    assert len(rects) > 1


def test_tracking_occluder_refuses_to_be_built_by_accident():
    with pytest.raises(ValueError, match="leaks"):
        TrackingOccluder()


def test_tracking_occluder_is_unreachable_from_a_config_file():
    """A config must not be able to select the leaking occluder silently."""
    with pytest.raises(ValueError, match="leaks"):
        make_occluder({"mode": "tracking"})


def test_make_occluder_dispatch():
    assert isinstance(make_occluder(None), NoOcclusion)
    assert isinstance(make_occluder("none"), NoOcclusion)
    assert isinstance(make_occluder("static"), StaticOccluder)
    assert isinstance(make_occluder({"mode": "static", "rect": (0, 0, 0.5, 0.5)}), StaticOccluder)
    with pytest.raises(KeyError):
        make_occluder("nonsense")


# ---------------------------------------------------------------------------
# Needs a real environment. Skips locally, runs on the lab machine.
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
def test_mask_covers_the_block_across_50_fresh_resets():
    """The block spawns somewhere new each reset -- one reset proves nothing.

    This test trusts the free-camera projection in fetch_pixels.py. Confirm that
    projection once by eye with `contact_sheet.py --annotate` before believing a
    green result here.
    """
    pytest.importorskip("gymnasium_robotics")

    from src.envs.fetch_pixels import (
        block_world_position,
        camera_config,
        make_raw_env,
        model_fovy,
        project_world_to_pixel,
    )

    size = 84
    env = make_raw_env(render_size=size)
    occ = make_occluder("static")
    cfg = camera_config()
    fovy = model_fovy(env)

    missed = []
    try:
        for seed in range(50):
            env.reset(seed=seed)
            occ.reset(env)
            col, row = project_world_to_pixel(block_world_position(env), cfg, size, size, fovy)
            if not occ.covers(col, row, (size, size, 3), env):
                missed.append((seed, round(col, 1), round(row, 1)))
    finally:
        env.close()

    assert not missed, (
        f"the block was outside the mask on {len(missed)}/50 resets: {missed[:5]} -- "
        "widen or recentre DEFAULT_STATIC_RECT, or retarget the camera"
    )
