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


# -- what gets stored vs what gets acted on ---------------------------------


def test_the_buffer_receives_a_single_frame_not_the_stack():
    """Storing stacks puts every frame in the buffer three times -- the 254 GB
    path CLAUDE.md forbids. The acting path and the storage path deliberately
    carry different shapes, and conflating them is a real bug that only shows up
    when a pixel run starts."""
    from src.train import _store_obs, pixel_obs

    stack = np.zeros((9, 84, 84), dtype=np.uint8)
    stack[6:9] = 200  # the newest frame, last three channels
    stored = _store_obs(pixel_obs, {"pixels": stack, "proprio": np.zeros(13, np.float32)},
                        uses_pixels=True)

    assert stored["pixels"].shape == (84, 84, 3), "the buffer stores single frames"
    assert stored["pixels"].dtype == np.uint8
    assert np.all(stored["pixels"] == 200), "it must be the newest frame, not the oldest"


def test_the_anchor_stores_the_dummy_frame():
    from src.train import _store_obs, flatten_state_obs

    obs = {"observation": np.zeros(25, np.float32),
           "achieved_goal": np.zeros(3, np.float32),
           "desired_goal": np.zeros(3, np.float32)}
    stored = _store_obs(flatten_state_obs, obs, uses_pixels=False)
    assert stored["pixels"].shape == (1, 1, 3)
    assert stored["proprio"].shape == (28,)


# -- the SOP's 100-episode rolling window -----------------------------------


def test_rolling_window_is_empty_safe_and_averages():
    """The SOP promises rolling curves over a 100-episode window."""
    from collections import deque

    from src.train import rolling

    assert rolling(deque()) == 0.0
    assert rolling(deque([1.0, 0.0, 1.0, 0.0])) == pytest.approx(0.5)
    assert rolling(deque([1.0])) == pytest.approx(1.0)


def test_rolling_window_keeps_only_the_last_hundred():
    from collections import deque

    from src.train import rolling

    window = deque(maxlen=100)
    for _ in range(100):
        window.append(0.0)
    assert rolling(window) == pytest.approx(0.0)
    for _ in range(100):
        window.append(1.0)          # pushes every zero out
    assert len(window) == 100
    assert rolling(window) == pytest.approx(1.0)


def test_rolling_window_reports_before_it_is_full():
    """A curve flat at zero for 100 episodes then jumping is harder to read."""
    from collections import deque

    from src.train import rolling

    window = deque(maxlen=100)
    window.append(1.0)
    assert rolling(window) == pytest.approx(1.0), "must not wait for 100 episodes"
