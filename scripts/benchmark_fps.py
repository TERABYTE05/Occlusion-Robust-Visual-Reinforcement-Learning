#!/usr/bin/env python
"""Four-stage throughput measurement -- the number the whole schedule hangs off.

    stage 1  env.step alone            (no rendering at all)
    stage 2  + env.render()            (the Landmine 4 cost)
    stage 3  + encoder forward         (acting)
    stage 4  + a dummy gradient update (the closest thing to the real loop)

Stage 4 is the number that decides the run matrix at G4. It is deliberately
*optimistic*: it runs one encoder pair and one backward pass, where the real
DrQ-v2 step also runs twin critics, an actor, and two target networks. Treat it
as a ceiling and keep a margin.

    python scripts/benchmark_fps.py --steps 2000

Writes a table to stdout and appends it to BENCHMARK.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.envs.fetch_pixels import ENV_ID, PixelProprioWrapper, make_raw_env  # noqa: E402
from src.envs.make_env import FrameStack  # noqa: E402
from src.agent.drqv2 import DDPGAgent  # noqa: E402
from src.buffer.replay import ReplayBuffer  # noqa: E402
from src.models.multimodal import MultimodalEncoder  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.envs.occlusion import make_occluder  # noqa: E402
from src.envs.proprio import PROPRIO_OBS_DIM  # noqa: E402
from src.models.encoders import ImageEncoder, ProprioEncoder  # noqa: E402
from src.utils.gl import print_backend  # noqa: E402

STEP_BUDGET = 500_000


def _random_actions(env, n, rng):
    low, high = env.action_space.low, env.action_space.high
    return rng.uniform(low, high, size=(n, low.shape[0])).astype(np.float32)


def _run_env_loop(env, steps, rng, per_step=None):
    env.reset(seed=0)
    actions = _random_actions(env, steps, rng)
    start = time.perf_counter()
    for i in range(steps):
        obs, _, terminated, truncated, _ = env.step(actions[i])
        if per_step is not None:
            per_step(obs)
        if terminated or truncated:
            env.reset()
    return time.perf_counter() - start


def stage_env_only(args, rng):
    env = make_raw_env(env_id=args.env_id, render=False)
    try:
        return _run_env_loop(env, args.steps, rng)
    finally:
        env.close()


def _pixel_env(args):
    env = make_raw_env(env_id=args.env_id, camera=args.camera, render_size=args.image_size)
    env = PixelProprioWrapper(
        env, image_size=args.image_size, frame_transform=make_occluder(args.occlusion)
    )
    return FrameStack(env, k=args.frame_stack)


def stage_render(args, rng):
    env = _pixel_env(args)
    try:
        return _run_env_loop(env, args.steps, rng)
    finally:
        env.close()


def _encoders(args, device):
    image = ImageEncoder(
        in_channels=3 * args.frame_stack, image_size=args.image_size
    ).to(device)
    proprio = ProprioEncoder().to(device)
    return image, proprio


def stage_encoder(args, rng, device):
    env = _pixel_env(args)
    image_enc, proprio_enc = _encoders(args, device)
    image_enc.eval()
    proprio_enc.eval()

    def forward(obs):
        with torch.no_grad():
            pixels = torch.as_tensor(obs["pixels"], device=device).unsqueeze(0)
            prop = torch.as_tensor(obs["proprio"], device=device).unsqueeze(0)
            image_enc(pixels)
            proprio_enc(prop)

    try:
        return _run_env_loop(env, args.steps, rng, per_step=forward)
    finally:
        env.close()


def _prefilled_buffer(args):
    """A replay buffer filled to the capacity a production run reaches.

    Filled with zeros: the benchmark measures the cost of moving and sampling
    the data, which does not depend on its contents.
    """
    buffer = ReplayBuffer(
        capacity=args.buffer_capacity,
        image_size=args.image_size,
        frame_stack=args.frame_stack,
        proprio_dim=PROPRIO_OBS_DIM,
        action_dim=4,
        nstep=3,
        discount=0.99,
        seed=0,
    )
    frame = np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)
    proprio = np.zeros(PROPRIO_OBS_DIM, dtype=np.float32)
    action = np.zeros(4, dtype=np.float32)
    written = 0
    while written < args.buffer_capacity:
        buffer.add_first({"pixels": frame, "proprio": proprio})
        for step in range(50):
            buffer.add(action, 0.0, {"pixels": frame, "proprio": proprio},
                       terminated=False, truncated=(step == 49))
            written += 1
            if written >= args.buffer_capacity:
                break
    return buffer


def stage_full(args, rng, device):
    """The real training step: the actual agent, not a stand-in.

    Earlier versions of this stage drove a single Linear head over the encoders
    and reported ~35 FPS, then ~15 FPS once replay sampling was included. Both
    were upper bounds on a loop far lighter than training, because the stage
    omitted the fusion module, the actor, the twin critics, the target networks
    and the second backward pass -- while its printed verdict decided how many
    seeds the project could afford.

    Every proxy measurement of throughput on this project has been wrong: a
    failed GPU fan in September, then an O(buffer) sampling cost, then half the
    network. So this stage now builds the configuration's real agent and calls
    its real update. Config C is the default because it is the heaviest of the
    three, and the matrix has to fit the worst case.
    """
    env = _pixel_env(args)
    cfg = load_config(args.config)
    encoder = MultimodalEncoder(
        cfg, image_size=args.image_size, frame_stack=args.frame_stack
    )
    agent = DDPGAgent(
        encoder,
        action_dim=int(env.action_space.shape[0]),
        cfg=cfg.get("agent", {}),
        device=device,
    )
    buffer = _prefilled_buffer(args)

    def update(obs):
        batch = buffer.sample(args.batch_size)
        agent.update(
            {
                "obs": {"pixels": batch["pixels"], "proprio": batch["proprio"]},
                "next_obs": {"pixels": batch["next_pixels"], "proprio": batch["next_proprio"]},
                "action": batch["action"],
                "reward": batch["reward"],
                "discount": batch["discount"],
                "bootstrap": batch["bootstrap"],
            },
            step=0,
        )

    try:
        return _run_env_loop(env, args.steps, rng, per_step=update)
    finally:
        env.close()


def verdict(fps: float) -> str:
    if fps >= 15:
        return "3 seeds x 500K fits (Full matrix)"
    if fps >= 10:
        return "2 seeds x 500K (Reduced matrix)"
    return "BELOW GATE -- descend the fallback ladder (Roadmap §5)"


def main():
    # Windows: stdout defaults to the console codepage (cp1252), which cannot
    # encode the arrows and middots in the report. That is fine in a terminal
    # but raises the moment output is piped or redirected. Force UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):  # pragma: no cover - non-standard stream
            pass

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=2000, help="steps per stage")
    p.add_argument("--env-id", default=ENV_ID)
    p.add_argument("--camera", default=None, help="camera config name (default: v1)")
    p.add_argument("--image-size", type=int, default=84)
    p.add_argument("--frame-stack", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--config", default="configs/config_c_dual_occ.yaml",
                   help="the agent to benchmark; default is Config C, the "
                        "heaviest of the three")
    p.add_argument("--buffer-capacity", type=int, default=100_000,
                   help="replay fill for stage 4; the production value, "
                        "because sampling cost grew with it once before")
    p.add_argument("--occlusion", default="none")
    p.add_argument("--out", default="BENCHMARK.md")
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args()

    print_backend()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bench] device: {device}  steps/stage: {args.steps}")
    if device.type != "cuda":
        print("[bench] WARNING: no CUDA device. These numbers do not describe the lab machine.")

    rng = np.random.default_rng(0)
    stages = [
        ("env.step only", lambda: stage_env_only(args, rng)),
        ("+ env.render()", lambda: stage_render(args, rng)),
        ("+ encoder forward", lambda: stage_encoder(args, rng, device)),
        ("full loop with gradient update", lambda: stage_full(args, rng, device)),
    ]

    rows = []
    for name, fn in stages:
        print(f"[bench] {name} ...", flush=True)
        elapsed = fn()
        fps = args.steps / elapsed
        rows.append((name, fps, elapsed))
        print(f"[bench]   {fps:8.1f} FPS  ({elapsed:.1f}s)")

    full_fps = rows[-1][1]
    hours = STEP_BUDGET / full_fps / 3600

    lines = [
        "",
        f"## Run {dt.datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"`{args.env_id}` · camera `{args.camera or 'v1'}` · {args.image_size}x{args.image_size} · "
        f"stack {args.frame_stack} · batch {args.batch_size} · device `{device}` · "
        f"{args.steps} steps/stage",
        "",
        "| Stage | FPS | Wall-clock |",
        "|---|---:|---:|",
    ]
    for name, fps, elapsed in rows:
        lines.append(f"| {name} | {fps:.1f} | {elapsed:.1f}s |")
    lines += [
        "",
        f"**Full-loop FPS: {full_fps:.1f}** → {hours:.1f} h per 500K run "
        f"({hours * 6:.0f} h for 6 runs, {hours * 9:.0f} h for 9 runs)",
        "",
        f"**Verdict:** {verdict(full_fps)}",
        "",
        f"> Stage 4 drives the real agent from `{args.config}` -- fusion module, actor, "
        "twin critics, target networks and both backward passes. It is no longer an "
        "upper bound.",
    ]
    report = "\n".join(lines)
    print(report)

    if not args.no_write:
        out = pathlib.Path(args.out)
        header = "" if (out.exists() and out.stat().st_size > 0) else "# Benchmarks\n\nFPS measurements. The run matrix at G4 is chosen from the last row here.\n"
        with out.open("a", encoding="utf-8") as fh:
            fh.write(header + report + "\n")
        print(f"\n[bench] appended to {out}")


if __name__ == "__main__":
    main()
