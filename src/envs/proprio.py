"""The proprioception slice -- Landmine 3, and a scientific invariant.

FetchPush's 25-D ``observation`` vector contains the block's full state. Feeding
all 25 dims to the proprioception encoder ``g_xi`` lets the agent read exact
block coordinates from proprioception, so masking the block in the RGB frame
costs it nothing: all three configs would score identically and the paper's
claim would become *untestable* rather than refuted. This cannot happen in the
original paper, where ``dm_control`` proprioception is joint and actuator state
only.

So the slice fed to ``g_xi`` is exactly the gripper-only dims, plus the goal
(Landmine 2 -- the task is goal-conditioned and unlearnable without it):

    [0:3] grip_pos + [9:11] gripper_state + [20:25] grip_velp,gripper_vel
        = 10-D, then || desired_goal (3-D) = 13-D

Do not widen this slice. ``tests/test_proprio.py`` asserts that no object
dimension can reach the encoder.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

#: Layout of the 25-D FetchPush ``observation`` vector.
OBSERVATION_DIM = 25

GRIP_POS = slice(0, 3)  # gripper xyz                       -> keep
OBJECT_POS = slice(3, 6)  # block xyz                       -> LEAK
OBJECT_REL_POS = slice(6, 9)  # block xyz relative to grip  -> LEAK
GRIPPER_STATE = slice(9, 11)  # finger joint positions      -> keep
OBJECT_ROT = slice(11, 14)  # block euler                   -> LEAK
OBJECT_VELP = slice(14, 17)  # block linear velocity        -> LEAK
OBJECT_VELR = slice(17, 20)  # block angular velocity       -> LEAK
GRIP_VELP = slice(20, 23)  # gripper linear velocity        -> keep
GRIPPER_VEL = slice(23, 25)  # finger joint velocities      -> keep

#: The kept slices, in order. [20:25] is grip_velp || gripper_vel, contiguous.
PROPRIO_SLICES = (GRIP_POS, GRIPPER_STATE, slice(20, 25))

#: Indices that carry block state. None of these may reach ``g_xi``.
OBJECT_DIMS = frozenset(
    list(range(3, 9)) + list(range(11, 20))
)  # object_pos, object_rel_pos, object_rot, object_velp, object_velr

PROPRIO_DIM = 10
GOAL_DIM = 3
PROPRIO_OBS_DIM = PROPRIO_DIM + GOAL_DIM  # 13


def proprio_indices() -> np.ndarray:
    """The concrete indices kept from the 25-D observation vector."""
    return np.concatenate(
        [np.arange(s.start, s.stop, dtype=np.int64) for s in PROPRIO_SLICES]
    )


def proprio_slice(observation: np.ndarray) -> np.ndarray:
    """Gripper-only dims of the raw 25-D observation. Returns (10,) float32."""
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
    """Build the 13-D proprioception vector from a FetchPush observation dict.

    ``o_prop = concat(proprio_slice(obs["observation"]), obs["desired_goal"])``

    The goal must be here: FetchPush is goal-conditioned and the agent is
    otherwise never told where to push (Landmine 2).
    """
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
