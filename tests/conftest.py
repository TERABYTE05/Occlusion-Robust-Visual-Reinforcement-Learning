import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    return ROOT


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "mujoco: needs gymnasium-robotics and a working MuJoCo renderer"
    )
