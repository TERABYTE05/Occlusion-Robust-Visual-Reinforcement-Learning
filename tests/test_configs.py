"""Guards on the config files themselves.

The A/B/C comparison is only valid if the configs are identical everywhere the
experiment does not intend them to differ, and the invariants are only
invariant if nothing quietly edits them.
"""

import pytest

from src.utils.config import load_config

CONFIG_DIR = "configs"
PIXEL_CONFIGS = [
    "config_a_concat_noocc.yaml",
    "config_b_concat_occ.yaml",
    "config_c_dual_occ.yaml",
]
ALL_CONFIGS = PIXEL_CONFIGS + ["state_ddpg.yaml"]


@pytest.fixture
def cfgs(repo_root):
    return {name: load_config(repo_root / CONFIG_DIR / name) for name in ALL_CONFIGS}


@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_env_id_is_the_dense_variant(repo_root, name):
    """Landmine 1: the sparse default is unlearnable inside our step budget."""
    cfg = load_config(repo_root / CONFIG_DIR / name)
    assert cfg["env"]["id"] == "FetchPushDense-v3"


@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_invariants_are_unchanged(repo_root, name):
    cfg = load_config(repo_root / CONFIG_DIR / name)
    agent = cfg["agent"]
    assert agent["feature_dim"] == 128, "d is the paper's contribution"
    assert agent["nstep"] == 3
    assert agent["polyak"] == 0.01
    assert agent["aug_pad"] == 4
    assert agent["simplexnorm"] == {"partitions": 16, "per_partition": 8, "tau": 1.0}


@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_simplexnorm_partitions_tile_the_feature_dim(repo_root, name):
    cfg = load_config(repo_root / CONFIG_DIR / name)
    sn = cfg["agent"]["simplexnorm"]
    assert sn["partitions"] * sn["per_partition"] == cfg["agent"]["feature_dim"]


def test_a_and_b_differ_only_in_occlusion(cfgs):
    """A - B is the occlusion penalty, so occlusion must be the only difference."""
    a = cfgs["config_a_concat_noocc.yaml"]
    b = cfgs["config_b_concat_occ.yaml"]
    assert a["agent"] == b["agent"]
    assert a["train"] == b["train"]
    assert a["buffer"] == b["buffer"]
    assert {k: v for k, v in a["env"].items() if k != "occlusion"} == {
        k: v for k, v in b["env"].items() if k != "occlusion"
    }
    assert a["env"]["occlusion"] != b["env"]["occlusion"]


def test_b_and_c_differ_only_in_fusion_and_normalization(cfgs):
    """C - B is what the mechanism recovers; nothing else may vary."""
    b = cfgs["config_b_concat_occ.yaml"]
    c = cfgs["config_c_dual_occ.yaml"]
    assert b["env"] == c["env"], "B and C must be occluded identically"
    assert b["train"] == c["train"]
    assert b["buffer"] == c["buffer"]

    changed = {k for k in b["agent"] if b["agent"][k] != c["agent"][k]}
    assert changed == {"fusion", "normalization"}, f"unexpected differences: {changed}"


def test_only_config_c_uses_the_papers_method(cfgs):
    assert cfgs["config_a_concat_noocc.yaml"]["agent"]["fusion"] == "concat"
    assert cfgs["config_b_concat_occ.yaml"]["agent"]["fusion"] == "concat"
    assert cfgs["config_c_dual_occ.yaml"]["agent"]["fusion"] == "dual"
    assert cfgs["config_c_dual_occ.yaml"]["agent"]["normalization"] == "layernorm_simplexnorm"


def test_state_anchor_has_no_pixels_and_no_occlusion(cfgs):
    env = cfgs["state_ddpg.yaml"]["env"]
    assert env["pixels"] is False
    assert env["occlusion"] == "none"
