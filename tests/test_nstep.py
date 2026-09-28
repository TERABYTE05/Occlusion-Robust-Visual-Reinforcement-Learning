"""The n-step return and the bootstrap mask (Landmine 6).

Two failures hide here, and both look like a hyperparameter problem rather than
a bug.

The first is masking on ``truncated``. FetchPush never terminates -- it runs a
fixed 50 steps and truncates -- so a mask of ``terminated or truncated`` would
zero the value target at the end of *every* episode the agent ever sees. The
critic would learn that the last states of an episode are worth only their
immediate reward, and that error propagates backwards through the whole
trajectory. Training runs, the curves move, and the numbers are wrong.

The second is a window that runs past the end of an episode and collects the
next episode's rewards. The block has respawned somewhere else by then, so
those rewards belong to a different trajectory entirely.

The discount is set to 0.5 throughout, not 0.99, so that every expected value
below is a short exact decimal and a wrong exponent cannot hide inside floating
point noise.
"""

import numpy as np
import pytest

from src.buffer.replay import ReplayBuffer

SIZE = 8
STACK = 3
GAMMA = 0.5


def obs(value):
    return {
        "pixels": np.full((SIZE, SIZE, 3), value, dtype=np.uint8),
        "proprio": np.full(13, value, dtype=np.float32),
    }


def make_buffer(nstep=3, discount=GAMMA, capacity=100, seed=0):
    return ReplayBuffer(
        capacity=capacity,
        image_size=SIZE,
        frame_stack=STACK,
        action_dim=4,
        nstep=nstep,
        discount=discount,
        seed=seed,
    )


def write_episode(buffer, values, rewards, terminated=False, truncated=True):
    """One episode with explicit per-transition rewards.

    ``values`` are the observation fill values; ``rewards[j]`` is earned by the
    transition out of observation ``j``, so there is one fewer reward than
    observation. ``terminated`` / ``truncated`` apply to the final transition.
    """
    assert len(rewards) == len(values) - 1
    buffer.add_first(obs(values[0]))
    for index, value in enumerate(values[1:]):
        last = index == len(values) - 2
        buffer.add(
            action=np.zeros(4, dtype=np.float32),
            reward=float(rewards[index]),
            next_obs=obs(value),
            terminated=terminated and last,
            truncated=truncated and last,
        )


def last_frame(stack):
    """The fill value of the most recent frame in a (3k, H, W) stack."""
    return int(stack[-3, 0, 0])


# -- the return itself ------------------------------------------------------


def test_three_step_return_matches_a_hand_computation():
    buffer = make_buffer(nstep=3)
    write_episode(buffer, [1, 2, 3, 4, 5], rewards=[1, 2, 3, 4])

    window = buffer.nstep_from(0)

    # 1 + 0.5*2 + 0.25*3
    assert window["reward"] == pytest.approx(2.75)
    assert window["steps"] == 3
    assert window["discount"] == pytest.approx(0.125)  # 0.5**3
    assert window["final"] == 3
    assert not window["terminated"]
    assert not window["truncated"]


def test_nstep_one_reduces_to_the_single_step_reward():
    buffer = make_buffer(nstep=1)
    write_episode(buffer, [1, 2, 3, 4], rewards=[7, 9, 11])

    window = buffer.nstep_from(1)

    assert window["reward"] == pytest.approx(9.0)
    assert window["steps"] == 1
    assert window["discount"] == pytest.approx(GAMMA)
    assert window["final"] == 2


def test_the_discount_exponent_equals_the_steps_actually_taken():
    """A fixed ``gamma**nstep`` would be wrong on every shortened window."""
    buffer = make_buffer(nstep=4)
    write_episode(buffer, [1, 2, 3], rewards=[1, 1])  # truncates after 2 steps

    window = buffer.nstep_from(0)

    assert window["steps"] == 2
    assert window["discount"] == pytest.approx(GAMMA**2)
    assert window["discount"] != pytest.approx(GAMMA**4)


def test_bootstrap_observation_is_the_one_the_window_ends_on():
    buffer = make_buffer(nstep=3)
    write_episode(buffer, [10, 20, 30, 40, 50], rewards=[0, 0, 0, 0])

    window = buffer.nstep_from(1)

    assert window["final"] == 4
    assert last_frame(buffer.stack_at(int(window["final"]))) == 50


# -- episode boundaries -----------------------------------------------------


def test_the_window_stops_at_a_truncation_but_still_bootstraps():
    """Landmine 6. Truncation is a time limit, not an absorbing state."""
    buffer = make_buffer(nstep=3)
    write_episode(buffer, [1, 2, 3, 4], rewards=[1, 2, 3], truncated=True)

    window = buffer.nstep_from(1)

    assert window["steps"] == 2, "the third step is past the end of the episode"
    assert window["reward"] == pytest.approx(2 + GAMMA * 3)
    assert window["truncated"]
    assert not window["terminated"], "FetchPush never terminates"


def test_a_truncated_window_keeps_its_bootstrap_mask_at_one():
    buffer = make_buffer(nstep=3)
    write_episode(buffer, [1, 2, 3], rewards=[1, 1], truncated=True)

    batch = buffer.sample(32)

    assert batch["truncated"].any(), "this episode ended by truncation"
    assert np.all(batch["bootstrap"] == 1.0), (
        "masking on truncation would zero the target at the end of every "
        "FetchPush episode -- Landmine 6"
    )


def test_termination_zeroes_the_bootstrap_mask():
    buffer = make_buffer(nstep=3)
    write_episode(buffer, [1, 2, 3], rewards=[5, 7], terminated=True, truncated=False)

    window = buffer.nstep_from(0)

    assert window["steps"] == 2
    assert window["reward"] == pytest.approx(5 + GAMMA * 7)
    assert window["terminated"]

    batch = buffer.sample(32)
    ending = batch["terminated"]
    assert ending.any()
    assert np.all(batch["bootstrap"][ending] == 0.0)


def test_the_window_never_collects_the_next_episodes_rewards():
    """The block respawns on reset, so those rewards are another trajectory."""
    buffer = make_buffer(nstep=3)
    write_episode(buffer, [1, 2, 3], rewards=[1, 1], truncated=True)
    write_episode(buffer, [70, 80, 90], rewards=[1000, 1000], truncated=True)

    window = buffer.nstep_from(1)  # the last transition of the first episode

    assert window["steps"] == 1
    assert window["reward"] == pytest.approx(1.0)
    assert window["reward"] < 1000, "the window leaked into the second episode"
    assert last_frame(buffer.stack_at(int(window["final"]))) == 3


def test_the_window_stops_at_the_write_frontier():
    """An episode still in progress has no successor recorded for its last obs."""
    buffer = make_buffer(nstep=3)
    write_episode(buffer, [1, 2, 3], rewards=[10, 20], terminated=False, truncated=False)

    window = buffer.nstep_from(1)

    assert window["steps"] == 1
    assert window["reward"] == pytest.approx(20.0)
    assert not window["terminated"] and not window["truncated"]
    assert window["discount"] == pytest.approx(GAMMA)


def test_an_index_with_no_recorded_transition_is_rejected():
    buffer = make_buffer(nstep=3)
    write_episode(buffer, [1, 2, 3], rewards=[1, 1], terminated=False, truncated=False)
    with pytest.raises(ValueError):
        buffer.nstep_from(2)  # the open frontier observation


# -- the batch --------------------------------------------------------------


def test_sample_carries_the_nstep_fields():
    buffer = make_buffer(nstep=3)
    write_episode(buffer, list(range(1, 9)), rewards=[1] * 7)

    batch = buffer.sample(6)

    for key in ("reward", "discount", "bootstrap"):
        assert batch[key].shape == (6,)
        assert batch[key].dtype == np.float32
    assert batch["steps"].shape == (6,)
    assert batch["next_pixels"].shape == (6, 3 * STACK, SIZE, SIZE)
    assert batch["next_proprio"].shape == (6, 13)


def test_every_sampled_discount_is_a_power_of_gamma_matching_its_steps():
    buffer = make_buffer(nstep=3)
    for episode in range(3):
        write_episode(buffer, [1 + 10 * episode + i for i in range(6)], rewards=[1] * 5)

    batch = buffer.sample(128)

    expected = GAMMA ** batch["steps"].astype(np.float64)
    assert np.allclose(batch["discount"], expected)
    assert np.all(batch["steps"] >= 1)
    assert np.all(batch["steps"] <= 3)


def test_sampled_returns_agree_with_a_naive_forward_walk():
    """The property the whole file exists to protect, checked end to end."""
    buffer = make_buffer(nstep=3, capacity=200)
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    write_episode(buffer, [1, 2, 3, 4, 5, 6], rewards=rewards, truncated=True)

    for start in range(len(rewards)):
        expected = sum(
            (GAMMA**offset) * rewards[start + offset]
            for offset in range(min(3, len(rewards) - start))
        )
        window = buffer.nstep_from(start)
        assert window["reward"] == pytest.approx(expected), f"window at {start}"


# -- construction -----------------------------------------------------------


def test_nstep_below_one_is_rejected():
    with pytest.raises(ValueError):
        make_buffer(nstep=0)


def test_a_discount_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        make_buffer(discount=0.0)
    with pytest.raises(ValueError):
        make_buffer(discount=1.5)


def test_the_default_buffer_is_single_step():
    """So the existing single-step callers keep their meaning until train.py sets nstep."""
    buffer = ReplayBuffer(capacity=50, image_size=SIZE, frame_stack=STACK, action_dim=4)
    assert buffer.nstep == 1
