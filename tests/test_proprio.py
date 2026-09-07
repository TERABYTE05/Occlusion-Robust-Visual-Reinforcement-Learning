"""Landmine 3. If any of these fail, occlusion leaks and the experiment is void.

The failure these guard against is silent: with object state in ``g_xi`` all
three configs score identically and the paper's claim becomes untestable rather
than refuted.
"""

import numpy as np
import pytest

from src.envs.proprio import (
    OBJECT_DIMS,
    OBSERVATION_DIM,
    PROPRIO_DIM,
    PROPRIO_OBS_DIM,
    build_proprio,
    proprio_indices,
    proprio_slice,
)


def test_dimensions_are_10_and_13():
    assert PROPRIO_DIM == 10
    assert PROPRIO_OBS_DIM == 13


def test_slice_keeps_exactly_ten_dims():
    assert len(proprio_indices()) == PROPRIO_DIM


def test_no_object_dimension_is_selected():
    selected = set(proprio_indices().tolist())
    assert selected.isdisjoint(OBJECT_DIMS), (
        f"object dims {sorted(selected & OBJECT_DIMS)} reach g_xi -- "
        "occlusion leaks through proprioception"
    )


def test_selected_indices_are_the_documented_ones():
    assert proprio_indices().tolist() == [0, 1, 2, 9, 10, 20, 21, 22, 23, 24]


def test_slice_picks_the_right_values():
    obs = np.arange(OBSERVATION_DIM, dtype=np.float32)
    assert proprio_slice(obs).tolist() == [0, 1, 2, 9, 10, 20, 21, 22, 23, 24]


def test_object_values_never_survive_the_slice():
    sentinel = 12345.0
    obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    obs[sorted(OBJECT_DIMS)] = sentinel
    assert not np.any(proprio_slice(obs) == sentinel)


def test_build_proprio_appends_the_goal():
    obs = {
        "observation": np.arange(OBSERVATION_DIM, dtype=np.float32),
        "achieved_goal": np.array([7.0, 7.0, 7.0], dtype=np.float32),
        "desired_goal": np.array([101.0, 102.0, 103.0], dtype=np.float32),
    }
    out = build_proprio(obs)
    assert out.shape == (PROPRIO_OBS_DIM,)
    assert out.dtype == np.float32
    assert out[-3:].tolist() == [101.0, 102.0, 103.0], "Landmine 2: the goal must reach g_xi"


def test_achieved_goal_is_not_included():
    """achieved_goal is the block's position -- another route for the leak."""
    obs = {
        "observation": np.zeros(OBSERVATION_DIM, dtype=np.float32),
        "achieved_goal": np.full(3, 999.0, dtype=np.float32),
        "desired_goal": np.zeros(3, dtype=np.float32),
    }
    assert not np.any(build_proprio(obs) == 999.0)


def test_rejects_an_unexpected_observation_width():
    with pytest.raises(ValueError):
        proprio_slice(np.zeros(28, dtype=np.float32))


def test_rejects_a_missing_goal():
    with pytest.raises(KeyError):
        build_proprio({"observation": np.zeros(OBSERVATION_DIM, dtype=np.float32)})
