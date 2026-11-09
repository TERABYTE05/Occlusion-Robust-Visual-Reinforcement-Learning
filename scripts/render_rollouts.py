from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.utils.gl import print_backend

print_backend()

import mujoco
from PIL import Image, ImageDraw, ImageFont

from src.agent.drqv2 import DDPGAgent
from src.envs.fetch_pixels import camera_config
from src.envs.make_env import make_env
from src.models.encoders import StateEncoder
from src.models.multimodal import MultimodalEncoder
from src.train import flatten_state_obs, pixel_obs
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config

EVAL_SEED = 11000
SUCCESS_RADIUS = 0.05
PANEL = 224

POLICIES = [
    ("Random actions", None, None),
    ("State DDPG anchor", "state_ddpg_500k", "final.pt"),
    ("B  shared, no norm", "config_b_concat_occ", "final.pt"),
    ("C  dual + LN + SN", "config_c_dual_occ", "final.pt"),
]

_FONT_DIR = pathlib.Path("/usr/share/fonts/truetype/dejavu")


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(_FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


def _camera() -> mujoco.MjvCamera:
    cfg = camera_config()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = cfg["lookat"]
    cam.distance = cfg["distance"] * 1.15
    cam.azimuth = cfg["azimuth"]
    cam.elevation = cfg["elevation"]
    return cam


def _sites(u) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block = np.array(u._utils.get_site_xpos(u.model, u.data, "object0"), copy=True)
    grip = np.array(u._utils.get_site_xpos(u.model, u.data, "robot0:grip"), copy=True)
    return block, grip, np.array(u.goal, copy=True)


def _load(config: str | None, seed: int):
    cfg = load_config(f"configs/{config or 'state_ddpg_500k'}.yaml")
    pixels = bool(cfg.get("env", {}).get("pixels", True)) and config is not None
    env = make_env(cfg, seed=EVAL_SEED)
    if config is None:
        rng = np.random.default_rng(EVAL_SEED)
        low, high = env.action_space.low, env.action_space.high
        return env, (lambda obs: rng.uniform(low, high).astype(np.float32)), False

    if pixels:
        encoder = MultimodalEncoder(cfg, 84, 3)
        obs_fn = pixel_obs
    else:
        sample, _ = env.reset()
        encoder = StateEncoder(flatten_state_obs(sample).shape[0])
        obs_fn = flatten_state_obs
        env.close()
        env = make_env(cfg, seed=EVAL_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DDPGAgent(encoder, int(env.action_space.shape[0]), cfg.get("agent", {}), device=device)
    load_checkpoint(pathlib.Path(f"results/{config}/seed{seed}/final.pt"), agent)
    agent.eval()
    return env, (lambda obs: agent.act(obs_fn(obs), step=500_000, eval_mode=True)), pixels


STRIP = 104


def _overlay(frame: np.ndarray, label: str, t: int, dist: float, moved: float,
             success: bool, agent_view: np.ndarray | None, note: str) -> Image.Image:
    img = Image.new("RGB", (PANEL, PANEL + STRIP), (22, 32, 42))
    img.paste(Image.fromarray(frame).convert("RGB"), (0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, PANEL, 22], fill=(22, 32, 42))
    d.text((6, 4), label, font=_font(12, bold=True), fill=(240, 244, 248))
    if success:
        d.rounded_rectangle([PANEL - 72, 3, PANEL - 4, 19], radius=4, fill=(44, 122, 75))
        d.text((PANEL - 66, 5), "SUCCESS", font=_font(10, bold=True), fill=(255, 255, 255))
    d.rectangle([0, PANEL - 36, PANEL, PANEL], fill=(22, 32, 42))
    d.text((6, PANEL - 33), f"t={t:02d}/50   block->goal {dist * 100:4.1f} cm", font=_font(11),
           fill=(230, 236, 240))
    d.text((6, PANEL - 17), f"block moved {moved * 100:4.1f} cm", font=_font(11), fill=(230, 236, 240))
    y0 = PANEL + 8
    if agent_view is not None:
        img.paste(Image.fromarray(agent_view).resize((88, 88), Image.NEAREST), (6, y0))
        d.text((102, y0 + 4), "what the policy sees:", font=_font(10), fill=(170, 182, 192))
        d.text((102, y0 + 20), "84x84, 3-frame stack", font=_font(10), fill=(230, 236, 240))
        d.text((102, y0 + 34), "(newest) + 13-D", font=_font(10), fill=(230, 236, 240))
        d.text((102, y0 + 48), "gripper/goal vector", font=_font(10), fill=(230, 236, 240))
    else:
        for i, line in enumerate(note.split("\n")):
            d.text((8, y0 + 4 + 15 * i), line, font=_font(10), fill=(230, 236, 240) if i else (170, 182, 192))
    return img


def rollout(label, config, seed, episodes):
    env, act, pixels = _load(config, seed)
    note = ("what the policy sees:\nnothing -- uniform random\nactions in [-1, 1]" if config is None else
            "what the policy sees:\nno camera: 28-D ground-truth\nstate incl. block position\n+ goal (privileged)")
    u = env.unwrapped
    renderer = mujoco.Renderer(u.model, PANEL, PANEL)
    cam = _camera()
    frames, stats, paths = [], [], []
    for _ep in range(episodes):
        obs, info = env.reset()
        b0, g0, goal = _sites(u)
        ep_frames, block_path, grip_path = [], [b0[:2]], [g0[:2]]
        success = float(info.get("is_success", 0.0)) > 0
        for t in range(51):
            block, grip, _ = _sites(u)
            dist = float(np.linalg.norm(block[:2] - goal[:2]))
            moved = float(np.linalg.norm(block - b0))
            renderer.update_scene(u.data, camera=cam)
            view = None
            if pixels:
                view = np.ascontiguousarray(obs["pixels"][-3:].transpose(1, 2, 0))
            ep_frames.append(_overlay(renderer.render(), label, t, dist, moved, success, view, note))
            if t == 50:
                break
            obs, _r, term, trunc, info = env.step(act(obs))
            success = success or float(info.get("is_success", 0.0)) > 0
            block, grip, _ = _sites(u)
            block_path.append(block[:2]); grip_path.append(grip[:2])
        start_dist = float(np.linalg.norm(b0[:2] - goal[:2]))
        stats.append({"start_dist_cm": start_dist * 100, "end_dist_cm": dist * 100,
                      "moved_cm": moved * 100, "success": bool(success)})
        paths.append({"block": np.array(block_path).tolist(), "grip": np.array(grip_path).tolist(),
                      "goal": goal[:2].tolist()})
        frames.append(ep_frames)
    renderer.close()
    env.close()
    return frames, stats, paths


def _trajectory_plot(results, out: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(3.4 * len(results), 3.7), sharex=True, sharey=True)
    for ax, (label, _stats, paths) in zip(axes, results):
        for p in paths:
            blk, grp, goal = np.array(p["block"]), np.array(p["grip"]), np.array(p["goal"])
            ax.plot(grp[:, 0], grp[:, 1], color="#1f5f8b", alpha=0.35, lw=0.9)
            ax.plot([blk[0, 0], goal[0]], [blk[0, 1], goal[1]], color="#bbbbbb", lw=0.6, ls=":")
            ax.plot(blk[:, 0], blk[:, 1], color="#16202a", lw=1.8)
            ax.scatter(*blk[0], s=14, color="#16202a", zorder=3)
            ax.scatter(*goal, s=18, color="#c4322b", marker="x", zorder=3)
        ax.set_title(label, fontsize=10)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("y (m)", fontsize=8)
    for ax in axes:
        ax.set_xlabel("x (m)", fontsize=8)
    fig.suptitle("Top-down paths: gripper (blue), block (black, dot = start), goal (red x)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4, help="episodes in the animation")
    ap.add_argument("--trace-episodes", type=int, default=20, help="episodes in the path plot")
    ap.add_argument("--seed", type=int, default=1, help="training seed of the checkpoints")
    ap.add_argument("--out", default="results/rollouts")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = max(args.episodes, args.trace_episodes)

    results, all_frames, summary = [], [], {}
    for label, config, _ckpt in POLICIES:
        print(f"[rollout] {label} ({config or 'uniform random'}) x {n} episodes")
        frames, stats, paths = rollout(label, config, args.seed, n)
        all_frames.append(frames[: args.episodes])
        results.append((label, stats, paths[: args.trace_episodes]))
        summary[label] = {
            "config": config, "seed": args.seed, "episodes": stats,
            "success_rate": float(np.mean([s["success"] for s in stats])),
            "mean_moved_cm": float(np.mean([s["moved_cm"] for s in stats])),
        }
        print(f"          success {summary[label]['success_rate']:.2f}  "
              f"moved {summary[label]['mean_moved_cm']:.2f} cm")

    gif_frames = []
    for ep in range(args.episodes):
        for t in range(51):
            row = Image.new("RGB", (PANEL * len(POLICIES) + 4 * (len(POLICIES) - 1), PANEL + STRIP), (246, 247, 248))
            for k, frames in enumerate(all_frames):
                row.paste(frames[ep][t], (k * (PANEL + 4), 0))
            canvas = Image.new("RGB", (row.width, PANEL + STRIP + 26), (246, 247, 248))
            ImageDraw.Draw(canvas).text((8, 6), f"Episode {ep + 1} of {args.episodes}  ·  same block spawn in every panel  ·  checkpoints seed {args.seed}, 500K steps",
                                        font=_font(12), fill=(22, 32, 42))
            canvas.paste(row, (0, 26))
            gif_frames.extend([canvas] * (6 if t in (0, 50) else 1))

    gif = out / "rollouts.gif"
    gif_frames[0].save(gif, save_all=True, append_images=gif_frames[1:], duration=80, loop=0, optimize=True)
    print(f"[out] {gif}  ({gif.stat().st_size / 1e6:.1f} MB, {len(gif_frames)} frames)")
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(gif), "-movflags", "faststart",
                        "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        str(out / "rollouts.mp4")], check=True)
        print(f"[out] {out / 'rollouts.mp4'}")
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[warn] no mp4 ({exc})")

    _trajectory_plot(results, out / "trajectories.png")
    print(f"[out] {out / 'trajectories.png'}")
    (out / "rollouts.json").write_text(json.dumps(summary, indent=2))
    print(f"[out] {out / 'rollouts.json'}")


if __name__ == "__main__":
    main()
