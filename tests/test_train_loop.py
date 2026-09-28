"""The training loop's own logic, separated from the four hours of training.

Everything here is the part of a run that can be wrong without crashing:
the observation the agent is handed, the batch adapter between the buffer and
the agent, and the guard that stops a run rather than letting it train on NaNs
until morning.
"""

import numpy as np
import pytest

from src.buffer.replay import ReplayBuffer
from src.train import assert_finite, flatten_state_obs, to_agent_batch


# -- observations -----------------------------------------------------------


def test_state_observation_is_flattened_with_the_goal_appended():
    """Landmine 2: an agent never told where to push cannot learn."""
    obs = {
        "observation": np.arange(25, dtype=np.float32),
        "achieved_goal": np.array([9, 9, 9], dtype=np.float32),
        "desired_goal": np.array([1, 2, 3], dtype=np.float32),
    }
    flat = flatten_state_obs(obs)

    assert flat.shape == (28,)
    assert flat.dtype == np.float32
    assert np.allclose(flat[:25], np.arange(25))
    assert np.allclose(flat[25:], [1, 2, 3])


def test_the_state_observation_excludes_achieved_goal():
    """achieved_goal is the block's position under another name."""
    obs = {
        "observation": np.zeros(25, dtype=np.float32),
        "achieved_goal": np.full(3, 7.0, dtype=np.float32),
        "desired_goal": np.zeros(3, dtype=np.float32),
    }
    assert not np.any(flatten_state_obs(obs) == 7.0)


# -- the buffer/agent adapter ----------------------------------------------


def test_the_agent_batch_carries_the_buffer_s_discount_and_bootstrap():
    """The agent must never recompute these -- only the buffer knows (D12)."""
    buffer_batch = {
        "proprio": np.zeros((4, 28), dtype=np.float32),
        "next_proprio": np.ones((4, 28), dtype=np.float32),
        "action": np.zeros((4, 4), dtype=np.float32),
        "reward": np.full(4, 2.0, dtype=np.float32),
        "discount": np.full(4, 0.5, dtype=np.float32),
        "bootstrap": np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32),
        "pixels": np.zeros((4, 3, 1, 1), dtype=np.uint8),
        "next_pixels": np.zeros((4, 3, 1, 1), dtype=np.uint8),
    }
    batch = to_agent_batch(buffer_batch, pixels=False)

    assert np.allclose(batch["obs"], 0.0)
    assert np.allclose(batch["next_obs"], 1.0)
    assert np.allclose(batch["discount"], 0.5)
    assert np.allclose(batch["bootstrap"], [1.0, 0.0, 1.0, 1.0])


# -- the NaN guard ----------------------------------------------------------


def test_a_non_finite_loss_fails_loudly():
    """Four hours of training on NaNs is the failure worth crashing to avoid."""
    with pytest.raises(RuntimeError, match="non-finite"):
        assert_finite({"critic_loss": float("nan"), "actor_loss": 1.0}, step=1234)
    with pytest.raises(RuntimeError, match="non-finite"):
        assert_finite({"critic_loss": float("inf"), "actor_loss": 1.0}, step=7)


def test_finite_metrics_pass_through_silently():
    assert assert_finite({"critic_loss": 0.5, "actor_loss": 1.0}, step=1234) is None


def test_the_guard_names_the_offending_metric_and_step():
    with pytest.raises(RuntimeError) as excinfo:
        assert_finite({"critic_loss": 1.0, "actor_loss": float("nan")}, step=99)
    message = str(excinfo.value)
    assert "actor_loss" in message and "99" in message


# -- episode bookkeeping ----------------------------------------------------


def test_every_episode_starts_with_add_first():
    """The buffer raises if a step is recorded without one, so the loop's
    reset handling is load-bearing rather than cosmetic."""
    buffer = ReplayBuffer(
        capacity=64, image_size=1, frame_stack=1, proprio_dim=28,
        action_dim=4, nstep=3, discount=0.99,
    )
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    state = np.zeros(28, dtype=np.float32)

    for _episode in range(3):
        buffer.add_first({"pixels": frame, "proprio": state})
        for step in range(4):
            buffer.add(
                np.zeros(4, dtype=np.float32), 1.0,
                {"pixels": frame, "proprio": state},
                terminated=False, truncated=(step == 3),
            )

    assert len(buffer) == 12
