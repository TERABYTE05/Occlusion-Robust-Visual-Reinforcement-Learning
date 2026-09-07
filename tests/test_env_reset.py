"""Landmine 7. Without the repair in fetch_reset_fix.py the task is unsolvable.

These are the invariants a reader would assume hold for FetchPush and which do
not hold on a stock gymnasium-robotics 1.3.1 MuJoCo backend. If any of them
fails, no amount of RL is going to produce a success curve.

Built here with a bare ``gym.make`` rather than ``make_raw_env``, so the repair
is tested on its own rather than through our wrappers. ``achieved_goal`` is the
block's position, which keeps this independent of fetch_pixels.py too.
"""

import numpy as np
import pytest

pytest.importorskip("gymnasium_robotics")

import gymnasium as gym  # noqa: E402
import gymnasium_robotics  # noqa: E402,F401  (registers the Fetch envs)

from src.envs.fetch_reset_fix import apply_reset_fix  # noqa: E402

pytestmark = pytest.mark.mujoco

ENV_ID = "FetchPushDense-v3"


@pytest.fixture(scope="module")
def env():
    e = apply_reset_fix(gym.make(ENV_ID))
    yield e
    e.close()


def test_block_resets_onto_the_table_not_the_floor(env):
    height = float(env.unwrapped.height_offset)
    for seed in range(50):
        obs, _ = env.reset(seed=seed)
        z = float(obs["achieved_goal"][2])
        assert z == pytest.approx(height, abs=1e-3), (
            f"seed {seed}: block at z={z:.4f}, table is at {height:.4f} -- "
            "the reset repair is not in effect"
        )


def test_goal_and_block_share_the_table_height(env):
    """A push goal 0.4 m above the block can never be achieved."""
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        assert float(obs["desired_goal"][2]) == pytest.approx(
            float(obs["achieved_goal"][2]), abs=1e-3
        )


def test_gripper_resets_to_its_initial_pose(env):
    expected = np.asarray(env.unwrapped.initial_gripper_xpos)
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        assert np.allclose(obs["observation"][0:3], expected, atol=1e-3), (
            "the arm did not return to initial_gripper_xpos; on a stock 1.3.1 it "
            "parks at its retracted home pose, out of reach of the block"
        )


def test_block_spawns_within_reach_of_the_gripper(env):
    obj_range = float(env.unwrapped.obj_range)
    centre = np.asarray(env.unwrapped.initial_gripper_xpos)[:2]
    for seed in range(50):
        obs, _ = env.reset(seed=seed)
        assert np.all(np.abs(obs["achieved_goal"][:2] - centre) <= obj_range + 1e-6)


def test_block_position_varies_across_seeds(env):
    positions = set()
    for seed in range(50):
        obs, _ = env.reset(seed=seed)
        positions.add(tuple(np.round(obs["achieved_goal"], 5)))
    assert len(positions) > 40, "block spawn is not being randomised"


def test_block_is_at_rest_after_reset(env):
    """The block must not be falling or sliding when the episode starts."""
    obs, _ = env.reset(seed=0)
    start = obs["achieved_goal"].copy()
    zeros = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(50):
        obs, *_ = env.step(zeros)
    assert np.linalg.norm(obs["achieved_goal"] - start) < 1e-2


def test_gripper_holds_position_under_zero_actions(env):
    """If the mocap weld target is not restored, the arm jerks on step 1."""
    obs, _ = env.reset(seed=0)
    start = obs["observation"][0:3].copy()
    zeros = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(10):
        obs, *_ = env.step(zeros)
    assert np.linalg.norm(obs["observation"][0:3] - start) < 2e-2


def test_seeding_is_reproducible(env):
    """The repair must not change how many draws come off np_random."""
    first = [env.reset(seed=s)[0]["achieved_goal"].copy() for s in range(10)]
    second = [env.reset(seed=s)[0]["achieved_goal"].copy() for s in range(10)]
    for a, b in zip(first, second):
        assert np.allclose(a, b)


def test_dense_reward_tracks_distance_to_goal(env):
    """Landmine 1: the Dense variant must give a shaped, non-constant reward."""
    obs, _ = env.reset(seed=0)
    zeros = np.zeros(env.action_space.shape, dtype=np.float32)
    _, reward, *_ = env.step(zeros)
    distance = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
    assert reward == pytest.approx(-distance, abs=5e-2)
    assert reward != -1.0, "this looks like the sparse variant"


def test_applying_the_fix_twice_is_harmless(env):
    before = env.reset(seed=3)[0]["achieved_goal"].copy()
    apply_reset_fix(env)
    assert np.allclose(env.reset(seed=3)[0]["achieved_goal"], before)
