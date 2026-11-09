from __future__ import annotations

from typing import Mapping

import numpy as np

OBSERVATION_DIM = 25

GRIP_POS = slice(0, 3)
OBJECT_POS = slice(3, 6)
OBJECT_REL_POS = slice(6, 9)
GRIPPER_STATE = slice(9, 11)
OBJECT_ROT = slice(11, 14)
OBJECT_VELP = slice(14, 17)
OBJECT_VELR = slice(17, 20)
GRIP_VELP = slice(20, 23)
GRIPPER_VEL = slice(23, 25)

PROPRIO_SLICES = (GRIP_POS, GRIPPER_STATE, slice(20, 25))

OBJECT_DIMS = frozenset(
    list(range(3, 9)) + list(range(11, 20))
)

PROPRIO_DIM = 10
GOAL_DIM = 3
PROPRIO_OBS_DIM = PROPRIO_DIM + GOAL_DIM


def proprio_indices() -> np.ndarray:
    return np.concatenate(
        [np.arange(s.start, s.stop, dtype=np.int64) for s in PROPRIO_SLICES]
    )


def proprio_slice(observation: np.ndarray) -> np.ndarray:
    observation = np.asarray(observation)
    if observation.shape[-1] != OBSERVATION_DIM:
        raise ValueError(
            f"expected a {OBSERVATION_DIM}-D FetchPush observation, "
            f"got shape {observation.shape}. Slicing the wrong layout would "
            "silently change which dims reach g_xi (Landmine 3)."
        )
    return np.concatenate(
        [observation[..., s] for s in PROPRIO_SLICES], axis=-1
    ).astype(np.float32, copy=False)


def build_proprio(obs: Mapping[str, np.ndarray]) -> np.ndarray:
    missing = {"observation", "desired_goal"} - set(obs)
    if missing:
        raise KeyError(f"observation dict is missing {sorted(missing)}")

    goal = np.asarray(obs["desired_goal"])
    if goal.shape[-1] != GOAL_DIM:
        raise ValueError(f"expected a {GOAL_DIM}-D desired_goal, got {goal.shape}")

    return np.concatenate(
        [proprio_slice(obs["observation"]), goal.astype(np.float32, copy=False)],
        axis=-1,
    ).astype(np.float32, copy=False)
