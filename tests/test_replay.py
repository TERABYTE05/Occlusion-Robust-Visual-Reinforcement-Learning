"""The buffer must rebuild exactly what naive stacking would have stored.

The failure this guards against is silent: a stack that reaches across an
episode boundary has the right shape and the wrong contents, and it poisons a
small, constant fraction of every batch for the whole run.
"""

import numpy as np
import pytest

from src.buffer.replay import ReplayBuffer

SIZE = 8
STACK = 3


def obs(value):
    """An observation whose frame is uniformly `value`, so stacks are readable."""
    return {
        "pixels": np.full((SIZE, SIZE, 3), value, dtype=np.uint8),
        "proprio": np.full(13, value, dtype=np.float32),
    }


def frame_values(stack):
    """The per-frame fill values of a (3k, H, W) stack."""
    return [int(stack[3 * i, 0, 0]) for i in range(stack.shape[0] // 3)]


def fill_episode(buffer, values, terminated=False, truncated=True):
    """Write one episode whose observations have the given fill values."""
    buffer.add_first(obs(values[0]))
    for step, value in enumerate(values[1:], start=1):
        last = step == len(values) - 1
        buffer.add(
            action=np.zeros(4, dtype=np.float32),
            reward=float(step),
            next_obs=obs(value),
            terminated=terminated and last,
            truncated=truncated and last,
        )


def make_buffer(capacity=100, seed=0):
    return ReplayBuffer(
        capacity=capacity, image_size=SIZE, frame_stack=STACK, action_dim=4, seed=seed
    )


# -- stack reconstruction ---------------------------------------------------


def test_stack_repeats_the_first_frame_at_the_start_of_an_episode():
    buffer = make_buffer()
    fill_episode(buffer, [10, 20, 30, 40])
    assert frame_values(buffer.stack_at(0)) == [10, 10, 10]
    assert frame_values(buffer.stack_at(1)) == [10, 10, 20]
    assert frame_values(buffer.stack_at(2)) == [10, 20, 30]
    assert frame_values(buffer.stack_at(3)) == [20, 30, 40]


def test_stack_never_reaches_into_the_previous_episode():
    buffer = make_buffer()
    fill_episode(buffer, [10, 20, 30])
    fill_episode(buffer, [70, 80, 90])
    # absolute index 3 is the first observation of the second episode
    assert frame_values(buffer.stack_at(3)) == [70, 70, 70]
    assert frame_values(buffer.stack_at(4)) == [70, 70, 80]
    assert frame_values(buffer.stack_at(5)) == [70, 80, 90]


def test_reconstructed_stack_equals_naive_stacking():
    """The property the buffer exists to preserve, checked against a naive log."""
    buffer = make_buffer(capacity=200)
    naive = []
    for episode in range(4):
        # kept well under 256: frames are uint8 and larger fill values wrap
        values = [1 + 10 * episode + step for step in range(6)]
        fill_episode(buffer, values)
        for step in range(len(values)):
            window = [values[max(0, step - offset)] for offset in reversed(range(STACK))]
            naive.append(window)
    for absolute, expected in enumerate(naive):
        assert frame_values(buffer.stack_at(absolute)) == expected


def test_stack_has_the_expected_shape_and_dtype():
    buffer = make_buffer()
    fill_episode(buffer, [1, 2, 3, 4])
    stack = buffer.stack_at(3)
    assert stack.shape == (3 * STACK, SIZE, SIZE)
    assert stack.dtype == np.uint8


# -- transition bookkeeping -------------------------------------------------


def test_only_complete_transitions_are_counted():
    buffer = make_buffer()
    buffer.add_first(obs(1))
    assert len(buffer) == 0, "the reset observation is not yet a transition"
    buffer.add(np.zeros(4), 1.0, obs(2), terminated=False, truncated=False)
    assert len(buffer) == 1


def test_terminated_and_truncated_are_stored_separately():
    """Landmine 6: FetchPush only ever truncates, and the mask needs to know."""
    buffer = make_buffer()
    fill_episode(buffer, [1, 2, 3], terminated=False, truncated=True)
    batch = buffer.sample(16)
    assert batch["truncated"].any()
    assert not batch["terminated"].any()


def test_a_terminated_episode_is_recorded_as_terminated():
    buffer = make_buffer()
    fill_episode(buffer, [1, 2, 3], terminated=True, truncated=False)
    indices = [i for i in range(3) if buffer._is_sampleable(i)]
    assert buffer._terminated[[buffer._slot(i) for i in indices]].any()


def test_add_without_add_first_is_rejected():
    buffer = make_buffer()
    with pytest.raises(RuntimeError):
        buffer.add(np.zeros(4), 0.0, obs(1), terminated=False, truncated=False)


def test_a_new_episode_must_start_with_add_first():
    buffer = make_buffer()
    fill_episode(buffer, [1, 2], truncated=True)
    with pytest.raises(RuntimeError):
        buffer.add(np.zeros(4), 0.0, obs(3), terminated=False, truncated=False)


# -- sampling ---------------------------------------------------------------


def test_sample_shapes():
    buffer = make_buffer()
    fill_episode(buffer, list(range(1, 8)))
    batch = buffer.sample(5)
    assert batch["pixels"].shape == (5, 3 * STACK, SIZE, SIZE)
    assert batch["next_pixels"].shape == (5, 3 * STACK, SIZE, SIZE)
    assert batch["proprio"].shape == (5, 13)
    assert batch["action"].shape == (5, 4)
    assert batch["reward"].shape == (5,)


def test_next_observation_is_the_successor_of_the_observation():
    buffer = make_buffer()
    fill_episode(buffer, [10, 20, 30, 40, 50])
    for absolute in range(4):
        assert frame_values(buffer.stack_at(absolute + 1))[-1] == frame_values(
            buffer.stack_at(absolute)
        )[-1] + 10


def test_sampling_an_empty_buffer_raises():
    with pytest.raises(ValueError):
        make_buffer().sample(4)


# -- the ring ---------------------------------------------------------------


def test_old_transitions_are_overwritten_and_never_resampled():
    buffer = make_buffer(capacity=10)
    for episode in range(6):
        fill_episode(buffer, [100 * episode + step for step in range(4)])
    assert len(buffer) <= 10
    for absolute in buffer.sample_indices(64):
        assert absolute >= buffer._start


def test_a_stack_whose_history_was_overwritten_is_not_sampled():
    buffer = make_buffer(capacity=6)
    fill_episode(buffer, list(range(1, 11)))  # longer than the ring
    for absolute in buffer.sample_indices(64):
        age = int(buffer._age[buffer._slot(absolute)])
        assert absolute - min(STACK - 1, age) >= buffer._start


def test_capacity_must_exceed_the_frame_stack():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=2, image_size=SIZE, frame_stack=STACK)


def test_reported_size_matches_the_documented_budget():
    """10^5 single 84x84x3 uint8 frames is the 2.12 GB figure in the plan.

    That figure is decimal GB: 10^5 * 84 * 84 * 3 = 2.117e9 bytes, which is
    1.97 GiB. Worth keeping straight -- the 16 GB RAM budget is quoted the
    same way, and confusing the two is a 7% error in the wrong direction.
    """
    buffer = ReplayBuffer(capacity=100_000, image_size=84, frame_stack=3, action_dim=4)
    assert 2.0 < buffer.nbytes() / 1e9 < 2.3
