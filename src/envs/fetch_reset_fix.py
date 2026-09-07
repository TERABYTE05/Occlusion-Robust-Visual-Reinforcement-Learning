"""Repair for a gymnasium-robotics bug that makes FetchPush unsolvable.

Landmine 7, found while de-risking the environment in week 1. It fails *loudly*
once you look at the numbers and completely silently if you do not: training
runs, the reward moves, curves appear, and the task is impossible.

What is wrong
-------------
``MujocoRobotEnv.__init__`` runs ``_env_setup`` -- which drives the arm to its
starting pose and settles the block onto the table -- and then saves the
resulting state::

    self.initial_time = self.data.time            # robot_env.py:301
    self.initial_qpos = np.copy(self.data.qpos)   # robot_env.py:302
    self.initial_qvel = np.copy(self.data.qvel)   # robot_env.py:303

Nothing ever reads those three attributes back. Both ``_reset_sim``
implementations on the MuJoCo backend (``robot_env.py:305`` and
``fetch_env.py:375``) call ``mujoco.mj_resetData``, which restores the *model
defaults* from the XML, not the saved post-setup state. The MujocoPy backend
does it correctly (``fetch_env.py:252``: ``self.sim.set_state(self.initial_state)``),
so the bug is specific to the MuJoCo backend -- which is the one we use.

The consequences on ``FetchPushDense-v3``:

* the block resets onto the **floor** at z=0.025 instead of the table at
  z=0.4247 (``height_offset``);
* the goal is still sampled at table height, so it floats 0.4 m above the block
  and ``achieved_goal`` can never approach ``desired_goal``;
* the arm resets to its retracted home pose at y=0.264 while the block spawns at
  y in [0.6, 0.9], out of reach.

The task cannot be solved. Config A would never rise, the state-based sanity
anchor at G2 would never learn, and the natural reading would be "our DDPG is
broken" -- which is where the week goes.

Observed on gymnasium-robotics 1.3.1 / gymnasium 1.0.0 / mujoco 3.1.6.

The repair
----------
Restore what ``__init__`` already saved: qpos, qvel, time, and the mocap target
that the weld constraint drags the gripper towards. Without the mocap restore
the weld yanks the arm on the first step of every episode. Object randomisation
is then byte-for-byte upstream's, so seeding semantics are unchanged.

``tests/test_env_reset.py`` asserts the resulting invariants, so this stays
correct if a later gymnasium-robotics fixes the bug upstream.
"""

from __future__ import annotations

import types

import numpy as np


def _fixed_reset_sim(self) -> bool:
    """Upstream ``MujocoFetchEnv._reset_sim``, with the state actually restored."""
    import mujoco
    from gymnasium_robotics.utils.mujoco_utils import get_joint_qpos, set_joint_qpos

    mujoco.mj_resetData(self.model, self.data)

    # The three attributes __init__ saves and upstream never reads.
    self.data.time = self.initial_time
    self.data.qpos[:] = self._occl_init_qpos
    self.data.qvel[:] = self._occl_init_qvel

    # The mocap body is the weld target the gripper is dragged towards.
    # mj_resetData zeroes it; leaving it zeroed jerks the arm on step 1.
    if self.model.nmocap > 0:
        self.data.mocap_pos[:] = self._occl_init_mocap_pos
        self.data.mocap_quat[:] = self._occl_init_mocap_quat

    mujoco.mj_forward(self.model, self.data)

    # Below here is upstream fetch_env.py:379-393 unchanged, so that the number
    # and order of np_random draws -- and therefore seeding -- is identical.
    if self.has_object:
        object_xpos = self.initial_gripper_xpos[:2]
        while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                -self.obj_range, self.obj_range, size=2
            )
        object_qpos = get_joint_qpos(self.model, self.data, "object0:joint")
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        set_joint_qpos(self.model, self.data, "object0:joint", object_qpos)

    mujoco.mj_forward(self.model, self.data)
    return True


def apply_reset_fix(env):
    """Patch ``_reset_sim`` on this env instance. Call before the first reset.

    Must be called on a freshly constructed environment: the state it captures
    is the post-``_env_setup`` state, and a ``reset()`` would already have
    destroyed it. Patches the instance, not the class, so importing this module
    changes nothing on its own.
    """
    unwrapped = env.unwrapped

    if getattr(unwrapped, "_occl_reset_fix_applied", False):
        return env

    for attr in ("initial_qpos", "initial_qvel", "initial_time"):
        if not hasattr(unwrapped, attr):
            raise AttributeError(
                f"{type(unwrapped).__name__} has no {attr!r}; this gymnasium-robotics "
                "version does not look like the one this repair was written for. "
                "Re-check src/envs/fetch_reset_fix.py against the installed source."
            )

    unwrapped._occl_init_qpos = np.copy(unwrapped.initial_qpos)
    unwrapped._occl_init_qvel = np.copy(unwrapped.initial_qvel)
    if unwrapped.model.nmocap > 0:
        unwrapped._occl_init_mocap_pos = np.copy(unwrapped.data.mocap_pos)
        unwrapped._occl_init_mocap_quat = np.copy(unwrapped.data.mocap_quat)

    unwrapped._reset_sim = types.MethodType(_fixed_reset_sim, unwrapped)
    unwrapped._occl_reset_fix_applied = True
    return env
