"""Checkpoint and resume.

Runs are four hours or more on a shared machine, so resume has to be real. The
failure this guards against is not a crash: a checkpoint that reloads the
weights but not Adam's moments restarts the optimiser from zero at every 50K
boundary, and one that drops the random stream re-draws the same exploration
noise after every restart. Either silently turns one experiment into several,
and nothing in the logs would say so.
"""

import numpy as np
import torch

from src.agent.drqv2 import DDPGAgent
from src.models.encoders import StateEncoder
from src.utils.checkpoint import CHECKPOINT_NAMES, load_checkpoint, save_checkpoint

OBS, ACT = 28, 4

CFG = {
    "lr": 1e-4,
    "hidden_dim": 32,
    "discount": 0.99,
    "polyak": 0.01,
    "stddev_schedule": "linear(1.0,0.1,100)",
    "stddev_clip": 0.3,
}


def make_agent():
    return DDPGAgent(StateEncoder(OBS), action_dim=ACT, cfg=CFG, device="cpu")


def make_batch(n=8):
    return {
        "obs": np.random.randn(n, OBS).astype(np.float32),
        "next_obs": np.random.randn(n, OBS).astype(np.float32),
        "action": np.random.uniform(-1, 1, (n, ACT)).astype(np.float32),
        "reward": np.ones(n, dtype=np.float32),
        "discount": np.full(n, 0.9, dtype=np.float32),
        "bootstrap": np.ones(n, dtype=np.float32),
    }


def test_resume_restores_the_policy_exactly(tmp_path):
    agent = make_agent()
    for _ in range(3):
        agent.update(make_batch(), step=0)

    obs = np.random.randn(OBS).astype(np.float32)
    expected = agent.act(obs, step=0, eval_mode=True)

    path = tmp_path / "latest.pt"
    save_checkpoint(path, agent, step=1234, rng=np.random.default_rng(0))

    restored = make_agent()
    state = load_checkpoint(path, restored)

    assert state["step"] == 1234
    assert np.allclose(restored.act(obs, step=0, eval_mode=True), expected)


def test_resume_restores_the_target_critic_too(tmp_path):
    """A target reset to the online weights discards the averaging history."""
    agent = make_agent()
    for _ in range(5):
        agent.update(make_batch(), step=0)

    path = tmp_path / "latest.pt"
    save_checkpoint(path, agent, step=5, rng=np.random.default_rng(0))
    restored = make_agent()
    load_checkpoint(path, restored)

    for saved, loaded in zip(agent.critic_target.parameters(), restored.critic_target.parameters()):
        assert torch.allclose(saved, loaded)


def test_resume_restores_the_optimizer_moments(tmp_path):
    """Reloading weights alone silently restarts Adam at every 50K boundary."""
    agent = make_agent()
    for _ in range(3):
        agent.update(make_batch(), step=0)

    path = tmp_path / "latest.pt"
    save_checkpoint(path, agent, step=3, rng=np.random.default_rng(0))
    restored = make_agent()
    load_checkpoint(path, restored)

    saved = agent.critic_opt.state_dict()["state"]
    loaded = restored.critic_opt.state_dict()["state"]
    assert saved and saved.keys() == loaded.keys()
    for key in saved:
        assert torch.allclose(saved[key]["exp_avg"], loaded[key]["exp_avg"])
        assert torch.allclose(saved[key]["exp_avg_sq"], loaded[key]["exp_avg_sq"])


def test_resume_restores_the_random_stream(tmp_path):
    agent = make_agent()
    rng = np.random.default_rng(7)
    rng.random(5)  # advance it, so a fresh generator would not match

    path = tmp_path / "latest.pt"
    save_checkpoint(path, agent, step=10, rng=rng)
    expected = rng.random(3)

    state = load_checkpoint(path, make_agent())
    assert np.allclose(state["rng"].random(3), expected)


def test_extra_payload_round_trips(tmp_path):
    """The run carries its own bookkeeping -- episode count, best score."""
    agent = make_agent()
    path = tmp_path / "latest.pt"
    save_checkpoint(path, agent, step=1, rng=np.random.default_rng(0),
                    extra={"episodes": 42, "best_success": 0.3})
    state = load_checkpoint(path, make_agent())
    assert state["extra"]["episodes"] == 42
    assert state["extra"]["best_success"] == 0.3


def test_writing_a_checkpoint_leaves_no_temporary_file_behind(tmp_path):
    """The write is atomic: a kill mid-write must not corrupt the last good one."""
    agent = make_agent()
    path = tmp_path / "latest.pt"
    save_checkpoint(path, agent, step=1, rng=np.random.default_rng(0))
    assert path.exists()
    assert list(tmp_path.iterdir()) == [path]


def test_only_a_bounded_set_of_checkpoint_files_is_defined():
    """Nine runs x one file per 50K is what exhausted the disk in September."""
    assert set(CHECKPOINT_NAMES) == {"latest.pt", "best.pt", "final.pt"}
