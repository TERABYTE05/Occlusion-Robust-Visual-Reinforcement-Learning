"""The world->pixel projection, checked against MuJoCo's own camera.

tests/test_occlusion.py uses this projection to decide whether the occluder
covers the block, so if the projection is wrong that test lies. Rather than
trusting it, compare the camera basis we derive from (azimuth, elevation,
distance, lookat) against the one MuJoCo builds for the same free camera.

``mjv_updateScene`` is CPU-only, so this runs without a GL context.
"""

import numpy as np
import pytest

pytest.importorskip("gymnasium_robotics")

from src.envs.fetch_pixels import (  # noqa: E402
    CAMERA_CONFIGS,
    camera_config,
    camera_frame,
    make_raw_env,
    model_fovy,
    project_world_to_pixel,
)

pytestmark = pytest.mark.mujoco


@pytest.fixture(scope="module")
def sim():
    env = make_raw_env(render=False)
    env.reset(seed=0)
    yield env
    env.close()


def _mujoco_camera(env, cfg):
    """MuJoCo's own free-camera pose for this config."""
    import mujoco

    model, data = env.unwrapped.model, env.unwrapped.data
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = cfg["azimuth"]
    cam.elevation = cfg["elevation"]
    cam.distance = cfg["distance"]
    cam.lookat[:] = cfg["lookat"]

    scene = mujoco.MjvScene(model, 1000)
    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene
    )
    # camera[0] and camera[1] are the left and right stereo eyes, offset along
    # `right` by ipd/2 each. The mono camera the renderer uses is their mean.
    left, right = scene.camera[0], scene.camera[1]
    position = (np.array(left.pos) + np.array(right.pos)) / 2.0
    fovy = 2 * np.degrees(np.arctan(left.frustum_top / left.frustum_near))
    return position, np.array(left.forward), np.array(left.up), fovy


@pytest.mark.parametrize("name", sorted(CAMERA_CONFIGS))
def test_camera_basis_matches_mujoco(sim, name):
    cfg = camera_config(name)
    expected_pos, expected_fwd, expected_up, _ = _mujoco_camera(sim, cfg)
    position, forward, _, up = camera_frame(cfg)

    assert np.allclose(position, expected_pos, atol=1e-6)
    assert np.allclose(forward, expected_fwd, atol=1e-6)
    assert np.allclose(up, expected_up, atol=1e-6)


@pytest.mark.parametrize("name", sorted(CAMERA_CONFIGS))
def test_fovy_matches_the_render_frustum(sim, name):
    *_, expected_fovy = _mujoco_camera(sim, camera_config(name))
    assert model_fovy(sim) == pytest.approx(expected_fovy, abs=1e-3)


def test_basis_is_orthonormal(sim):
    _, forward, right, up = camera_frame(camera_config())
    for vector in (forward, right, up):
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-9)
    assert forward @ right == pytest.approx(0.0, abs=1e-9)
    assert forward @ up == pytest.approx(0.0, abs=1e-9)
    assert right @ up == pytest.approx(0.0, abs=1e-9)


def test_lookat_projects_to_the_frame_centre(sim):
    cfg = camera_config()
    col, row = project_world_to_pixel(cfg["lookat"], cfg, 84, 84, model_fovy(sim))
    assert col == pytest.approx(42.0, abs=1e-6)
    assert row == pytest.approx(42.0, abs=1e-6)


def test_moving_right_in_the_image_increases_the_column(sim):
    cfg = camera_config()
    _, _, right, up = camera_frame(cfg)
    base = np.asarray(cfg["lookat"])
    fovy = model_fovy(sim)

    col0, row0 = project_world_to_pixel(base, cfg, 84, 84, fovy)
    col1, row1 = project_world_to_pixel(base + 0.1 * right, cfg, 84, 84, fovy)
    col2, row2 = project_world_to_pixel(base + 0.1 * up, cfg, 84, 84, fovy)

    assert col1 > col0 and row1 == pytest.approx(row0, abs=1e-6)
    assert row2 < row0 and col2 == pytest.approx(col0, abs=1e-6)  # rows grow downwards


def test_rejects_a_point_behind_the_camera(sim):
    cfg = camera_config()
    position, forward, _, _ = camera_frame(cfg)
    with pytest.raises(ValueError):
        project_world_to_pixel(position - forward, cfg, 84, 84, model_fovy(sim))
