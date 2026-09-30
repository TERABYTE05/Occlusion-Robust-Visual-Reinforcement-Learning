"""The training loop (Noh et al., Algorithm 1).

Usage::

    ./scripts/run.sh configs/state_ddpg.yaml --seed 1
    ./scripts/run.sh configs/state_ddpg.yaml --seed 1 --resume

A run is defined by a config file plus a seed, and nothing else (CLAUDE.md,
Conventions). There are no hyperparameter flags: a shorter run is a config
file, not a command-line override, so that every run in the report can be
reproduced from something that was committed.

Two paths share this loop. The **state anchor** (``env.pixels: false``) sees
the ground-truth state and exists to prove the task, the reward, goal
conditioning and the DDPG core before any vision enters. The **pixel path**
sees stacked frames plus the 13-D proprioception slice, fused by ``h_psi``
(shared for Configs A and B, dualized and normalized for Config C).

The anchor stores a 1x1 dummy frame in the replay buffer rather than using a
second, simpler buffer. That is deliberate: it means the anchor exercises the
same n-step accumulation and the same bootstrap masking as the real runs, so a
bug in D12's logic surfaces in the cheapest configuration rather than the most
expensive one. Three bytes per transition is not worth a second code path.
"""

from __future__ import annotations

import argparse
import pathlib
import time
from typing import Any, Mapping

import numpy as np
import torch

from .agent.drqv2 import DDPGAgent
from .buffer.replay import ReplayBuffer
from .envs.make_env import make_env
from .envs.proprio import PROPRIO_OBS_DIM
from .eval import evaluate
from .models.encoders import StateEncoder
from .models.multimodal import MultimodalEncoder
from .utils.checkpoint import REPLAY_NAME, load_checkpoint, save_checkpoint
from .utils.config import load_config
from .utils.gl import resolved_backend

#: A placeholder frame for the state anchor, which has no pixels at all.
_DUMMY_FRAME = np.zeros((1, 1, 3), dtype=np.uint8)


# -- observations -----------------------------------------------------------


def flatten_state_obs(obs: Mapping[str, Any]) -> np.ndarray:
    """Ground-truth state for the anchor: the full 25-D vector plus the goal.

    ``achieved_goal`` is deliberately excluded -- it is the block's position
    under another name, and including it would put the same quantity in the
    vector twice. ``desired_goal`` must be included or the agent is never told
    where to push (Landmine 2).

    The anchor is the one configuration allowed to see object state: with no
    rendering there is no occlusion to leak past, so Landmine 3 does not apply.
    """
    return np.concatenate(
        [np.asarray(obs["observation"], dtype=np.float32),
         np.asarray(obs["desired_goal"], dtype=np.float32)]
    ).astype(np.float32)


def pixel_obs(obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Pixel path: the stacked frames and the 13-D slice, kept separate."""
    return {"pixels": obs["pixels"], "proprio": obs["proprio"]}


# -- buffer/agent adapter ---------------------------------------------------

def to_agent_batch(batch: Mapping[str, Any], pixels: bool) -> dict[str, np.ndarray]:
    """Rename a buffer batch into what the agent consumes.

    ``reward``, ``discount`` and ``bootstrap`` pass straight through. The agent
    must never recompute them: the buffer is the only component that knows
    where episodes end, so it is the only one that can know whether a window
    was cut short and whether the bootstrap survives (D12).
    """
    if pixels:
        # The multimodal encoder consumes both streams, so the observation stays
        # a dict all the way through rather than being flattened here.
        obs = {"pixels": batch["pixels"], "proprio": batch["proprio"]}
        next_obs = {"pixels": batch["next_pixels"], "proprio": batch["next_proprio"]}
    else:
        obs = batch["proprio"]
        next_obs = batch["next_proprio"]
    return {
        "obs": obs,
        "next_obs": next_obs,
        "action": batch["action"],
        "reward": batch["reward"],
        "discount": batch["discount"],
        "bootstrap": batch["bootstrap"],
    }


# -- guards -----------------------------------------------------------------


def assert_finite(metrics: Mapping[str, float], step: int) -> None:
    """Stop the run the moment a loss stops being a number.

    The fallback ladder names the SimplexNorm softmax at tau=1 under mixed
    precision as the first suspect for NaNs (rung 4). Whatever the cause,
    training through them for four hours produces a checkpoint that is not a
    policy and a curve that looks like a learning-rate problem.
    """
    for key, value in metrics.items():
        if not np.isfinite(value):
            raise RuntimeError(
                f"non-finite {key}={value!r} at step {step}. "
                "Suspect mixed precision first (fallback ladder rung 4): run the "
                "normalization block in fp32 before changing anything else."
            )


# -- assembly ---------------------------------------------------------------


def build_agent(cfg: Mapping[str, Any], env, device: str) -> tuple[DDPGAgent, Any, bool]:
    """Return ``(agent, obs_fn, uses_pixels)`` for this config."""
    env_cfg = cfg.get("env", {})
    uses_pixels = bool(env_cfg.get("pixels", True))
    action_dim = int(env.action_space.shape[0])

    if uses_pixels:
        # f_xi + g_zeta + h_psi. Whether h_psi is shared or dualized, and whether
        # it normalizes, is read from the config -- that difference is the
        # experiment (Configs A/B vs C).
        encoder: torch.nn.Module = MultimodalEncoder(
            cfg,
            image_size=int(env_cfg.get("image_size", 84)),
            frame_stack=int(env_cfg.get("frame_stack", 3)),
        )
        obs_fn = pixel_obs
    else:
        sample, _ = env.reset()
        encoder = StateEncoder(flatten_state_obs(sample).shape[0])
        obs_fn = flatten_state_obs

    agent = DDPGAgent(encoder, action_dim, cfg.get("agent", {}), device=device)
    return agent, obs_fn, uses_pixels


def build_buffer(cfg: Mapping[str, Any], obs_dim: int, action_dim: int, uses_pixels: bool, seed: int):
    env_cfg = cfg.get("env", {})
    buffer_cfg = cfg.get("buffer", {})
    return ReplayBuffer(
        capacity=int(buffer_cfg.get("capacity", 100_000)),
        image_size=int(env_cfg.get("image_size", 84)) if uses_pixels else 1,
        frame_stack=int(env_cfg.get("frame_stack", 3)) if uses_pixels else 1,
        proprio_dim=PROPRIO_OBS_DIM if uses_pixels else obs_dim,
        action_dim=action_dim,
        nstep=int(buffer_cfg.get("nstep", 3)),
        discount=float(cfg.get("agent", {}).get("discount", 0.99)),
        seed=seed,
    )


def _store_obs(obs_fn, obs, uses_pixels: bool) -> dict[str, np.ndarray]:
    """What goes into the replay buffer, which is not what the policy acts on.

    The acting path carries a ``(3k, H, W)`` stack so the policy has temporal
    context. The buffer stores **single** frames in ``(H, W, 3)`` and rebuilds
    stacks at sample time -- storing stacks would put every frame in the buffer
    three times, which is the 254 GB path (CLAUDE.md, Conventions).

    The newest frame is the last three channels of the stack, because
    ``FrameStack`` concatenates oldest-first.
    """
    if uses_pixels:
        encoded = obs_fn(obs)
        newest = np.ascontiguousarray(encoded["pixels"][-3:].transpose(1, 2, 0))
        return {"pixels": newest, "proprio": encoded["proprio"]}
    return {"pixels": _DUMMY_FRAME, "proprio": obs_fn(obs)}


# -- the loop ---------------------------------------------------------------


def train(config_path: str, seed: int, resume: bool = False) -> dict[str, float]:
    cfg = load_config(config_path)
    train_cfg = cfg.get("train", {})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Never remove this print (Landmine 4): the backend decides the timeline,
    # and on this machine glfw silently renders on the Intel iGPU.
    print(f"[gl] active backend: {resolved_backend()}")
    print(f"[run] config={cfg['name']} seed={seed} device={device}")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    env = make_env(cfg, seed=seed)
    eval_env = make_env(cfg, seed=seed + 10_000)

    agent, obs_fn, uses_pixels = build_agent(cfg, env, device)
    action_dim = int(env.action_space.shape[0])
    obs_dim = int(agent.encoder.repr_dim) if not uses_pixels else PROPRIO_OBS_DIM
    buffer = build_buffer(cfg, obs_dim, action_dim, uses_pixels, seed)

    out_dir = pathlib.Path("results") / cfg["name"] / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    except Exception as exc:  # pragma: no cover - logging must never kill a run
        print(f"[warn] TensorBoard unavailable ({exc}); continuing without it")

    start_step = 0
    best_success = -1.0
    if resume and (out_dir / "latest.pt").exists():
        state = load_checkpoint(out_dir / "latest.pt", agent)
        start_step = state["step"]
        rng = state["rng"]
        best_success = float(state["extra"].get("best_success", -1.0))
        print(f"[resume] continuing from step {start_step}")

        # Without the replay history a resumed run trains on near-on-policy data
        # until the buffer refills, so it is not the same experiment as an
        # uninterrupted one -- and nothing in the curves would say so.
        replay_path = out_dir / REPLAY_NAME
        if replay_path.exists():
            buffer.load(replay_path)
            print(f"[resume] restored {len(buffer)} transitions of replay history")
        else:
            print(
                "[resume] WARNING: no replay snapshot found, so the buffer restarts "
                "EMPTY. This run is not equivalent to an uninterrupted one; record "
                "the restart in RUNLOG.md before using its results."
            )

    total_steps = int(train_cfg.get("steps", 500_000))
    seed_steps = int(train_cfg.get("seed_steps", 4000))
    batch_size = int(train_cfg.get("batch_size", 256))
    eval_every = int(train_cfg.get("eval_every", 10_000))
    eval_episodes = int(train_cfg.get("eval_episodes", 10))
    checkpoint_every = int(train_cfg.get("checkpoint_every", 50_000))

    obs, _info = env.reset()
    buffer.add_first(_store_obs(obs_fn, obs, uses_pixels))
    episode_return, episode_steps, episodes = 0.0, 0, 0
    last_time, last_step = time.time(), start_step
    metrics: dict[str, float] = {}
    # len(buffer) is O(capacity); once the batch is available it stays
    # available, so the check is latched rather than repeated every step.
    ready = False
    summary: dict[str, float] = {}

    for step in range(start_step, total_steps):
        if step < seed_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs_fn(obs), step=step)

        next_obs, reward, terminated, truncated, _info = env.step(action)
        buffer.add(action, float(reward), _store_obs(obs_fn, next_obs, uses_pixels),
                   terminated=bool(terminated), truncated=bool(truncated))
        episode_return += float(reward)
        episode_steps += 1
        obs = next_obs

        if terminated or truncated:
            episodes += 1
            if writer is not None:
                writer.add_scalar("train/episode_return", episode_return, step)
                writer.add_scalar("train/episode_length", episode_steps, step)
            obs, _info = env.reset()
            # The buffer raises if a step is recorded without this, so the
            # reset path is load-bearing rather than cosmetic.
            buffer.add_first(_store_obs(obs_fn, obs, uses_pixels))
            episode_return, episode_steps = 0.0, 0

        if not ready and step >= seed_steps and len(buffer) >= batch_size:
            ready = True
        if ready:
            metrics = agent.update(to_agent_batch(buffer.sample(batch_size), uses_pixels), step)
            assert_finite(metrics, step)
            if writer is not None and step % 100 == 0:
                for key, value in metrics.items():
                    writer.add_scalar(f"train/{key}", value, step)

        if eval_every and (step + 1) % eval_every == 0:
            summary = evaluate(eval_env, agent, eval_episodes, obs_fn, step=step)
            now = time.time()
            fps = (step - last_step) / max(1e-9, now - last_time)
            last_time, last_step = now, step
            summary["fps"] = fps
            print(
                f"[eval] step={step + 1} success={summary['success_rate']:.3f} "
                f"return={summary['episode_return']:.1f} fps={fps:.1f}"
            )
            if writer is not None:
                for key, value in summary.items():
                    writer.add_scalar(f"eval/{key}", value, step)
            if summary["success_rate"] > best_success:
                best_success = summary["success_rate"]
                save_checkpoint(out_dir / "best.pt", agent, step + 1, rng,
                                extra={"best_success": best_success, "episodes": episodes})

        if checkpoint_every and (step + 1) % checkpoint_every == 0:
            save_checkpoint(out_dir / "latest.pt", agent, step + 1, rng,
                            extra={"best_success": best_success, "episodes": episodes})
            buffer.save(out_dir / REPLAY_NAME)

    save_checkpoint(out_dir / "final.pt", agent, total_steps, rng,
                    extra={"best_success": best_success, "episodes": episodes})
    # A finished run has nothing to resume, and this file is ~2.1 GB on the
    # pixel path. Nine of them left behind is how the last disk filled up.
    (out_dir / REPLAY_NAME).unlink(missing_ok=True)
    if writer is not None:
        writer.close()
    env.close()
    eval_env.close()

    print(f"[done] {total_steps} steps, {episodes} episodes, best success {best_success:.3f}")
    return {"steps": float(total_steps), "episodes": float(episodes), "best_success": best_success}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one configuration, one seed.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", action="store_true",
                        help="continue from results/<name>/seed<N>/latest.pt")
    args = parser.parse_args()
    train(args.config, args.seed, resume=args.resume)


if __name__ == "__main__":
    main()
