from __future__ import annotations

import types

import numpy as np


def _fixed_reset_sim(self) -> bool:
    import mujoco
    from gymnasium_robotics.utils.mujoco_utils import get_joint_qpos, set_joint_qpos

    mujoco.mj_resetData(self.model, self.data)

    self.data.time = self.initial_time
    self.data.qpos[:] = self._occl_init_qpos
    self.data.qvel[:] = self._occl_init_qvel

    if self.model.nmocap > 0:
        self.data.mocap_pos[:] = self._occl_init_mocap_pos
        self.data.mocap_quat[:] = self._occl_init_mocap_quat

    mujoco.mj_forward(self.model, self.data)

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
