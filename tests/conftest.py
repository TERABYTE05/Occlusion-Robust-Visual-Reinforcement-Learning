import os
import pathlib
import sys

import pytest

# MuJoCo resolves its OpenGL backend at import time, and on this machine only
# `egl` is correct (Landmine 4). With the variable unset MuJoCo tries glfw and
# *aborts the process* rather than raising -- which takes the whole pytest run
# with it, including the physics-only tests. Set before anything imports mujoco
# so `python -m pytest tests/ -q` works as CLAUDE.md documents it.
os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    return ROOT


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "mujoco: needs gymnasium-robotics and a working MuJoCo renderer"
    )
