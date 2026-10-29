"""Mechanism-level metrics for a trained checkpoint.

Task success is a blunt instrument here: it is pinned to the block-spawn base
rate for any policy that does not move the block, so it cannot rank
configurations that all fail. These three can, and they ordered perfectly with
success on the first matrix (RUNLOG 2026-10-29):

  representation scale   RMS |z| of the fused vector entering the critic.
                         Unbounded growth swamps the 4-D action.
  action sensitivity     |Q(a=+1) - Q(a=-1)| / |Q|. How much the critic's value
                         depends on the action at all. If ~0 the actor has no
                         gradient and saturates.
  actor saturation       fraction of action components at |a| > 0.99.
  block displacement     cm the block actually moves. More sensitive than
                         success, which the spawn rate dominates.

    ./scripts/mechanism_probe.py config_c_dual_occ 1
    ./scripts/mechanism_probe.py --all
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.agent.drqv2 import DDPGAgent  # noqa: E402
from src.envs.make_env import make_env  # noqa: E402
from src.models.encoders import StateEncoder  # noqa: E402
from src.models.multimodal import MultimodalEncoder  # noqa: E402
from src.train import flatten_state_obs, pixel_obs  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402
from src.utils.config import load_config  # noqa: E402

EVAL_SEED = 11000


def measure(run: str, seed: int = 1, episodes: int = 20, ckpt: str = "final.pt") -> dict | None:
    path = pathlib.Path(f"results/{run}/seed{seed}/{ckpt}")
    if not path.exists():
        return None
    cfg = load_config(f"configs/{run}.yaml")
    pixels = bool(cfg.get("env", {}).get("pixels", True))
    env = make_env(cfg, seed=EVAL_SEED)

    if pixels:
        encoder = MultimodalEncoder(cfg, 84, 3)
    else:
        sample, _ = env.reset()
        encoder = StateEncoder(flatten_state_obs(sample).shape[0])
    agent = DDPGAgent(encoder, action_dim=int(env.action_space.shape[0]),
                      cfg=cfg.get("agent", {}), device="cuda")
    load_checkpoint(path, agent)

    Z, A, moved, succ = [], [], [], []
    for _ in range(episodes):
        obs, info = env.reset()
        u = env.unwrapped
        b0 = u._utils.get_site_xpos(u.model, u.data, "object0").copy()
        s = float(info.get("is_success", 0.0))
        done = False
        while not done:
            o = pixel_obs(obs) if pixels else flatten_state_obs(obs)
            if pixels:
                px = torch.as_tensor(o["pixels"], device="cuda").unsqueeze(0).float()
                pr = torch.as_tensor(o["proprio"], device="cuda").unsqueeze(0).float()
                with torch.no_grad():
                    z, _ = agent.encoder.both_reprs({"pixels": px, "proprio": pr})
            else:
                with torch.no_grad():
                    z = agent.encoder(torch.as_tensor(o, device="cuda").unsqueeze(0).float())
            a = agent.act(o, step=500_000, eval_mode=True)
            Z.append(z.squeeze(0)); A.append(a)
            obs, _r, term, trunc, info = env.step(a)
            s = max(s, float(info.get("is_success", 0.0)))
            done = term or trunc
        b1 = u._utils.get_site_xpos(u.model, u.data, "object0").copy()
        moved.append(np.linalg.norm(b1 - b0)); succ.append(s)
    env.close()

    Zt = torch.stack(Z)
    At = torch.as_tensor(np.array(A), device="cuda").float()
    with torch.no_grad():
        q_hi = torch.min(*agent.critic(Zt, torch.ones_like(At)))
        q_lo = torch.min(*agent.critic(Zt, -torch.ones_like(At)))
        q_at = torch.min(*agent.critic(Zt, At))
    zz = Zt.cpu().numpy()
    q_scale = float(q_at.abs().mean())
    return {
        "run": run, "seed": seed,
        "repr_rms": float(np.sqrt((zz ** 2).mean())),
        "repr_absmax": float(np.abs(zz).max()),
        "sensitivity": 100.0 * float((q_hi - q_lo).abs().mean()) / max(q_scale, 1e-9),
        "q_scale": q_scale,
        "saturation": float((np.abs(np.array(A)) > 0.99).mean()),
        "moved_cm": float(np.mean(moved) * 100),
        "success": float(np.mean(succ)),
    }


LADDER = [("config_b_concat_occ", "rung 1  shared, no norm"),
          ("config_d_ln_occ", "rung 2  shared, +LayerNorm"),
          ("config_e_ln_sn_occ", "rung 3  shared, +LN +SimplexNorm"),
          ("config_c_dual_occ", "rung 4  +dualization"),
          ("state_ddpg_500k", "state anchor (reference)"),
          ("diag_base", "diag  naive, 100K"),
          ("diag_tanh", "diag  +LayerNorm+Tanh, 100K")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", nargs="?")
    ap.add_argument("seed", nargs="?", type=int, default=1)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()

    todo = LADDER if args.all else [(args.run, args.run)]
    print(f"{'configuration':34}{'RMS|z|':>9}{'|z|max':>9}{'dQ/da %':>9}"
          f"{'sat':>7}{'moved cm':>10}{'success':>9}")
    print("-" * 87)
    for run, label in todo:
        try:
            m = measure(run, args.seed, args.episodes)
        except Exception as exc:                     # a missing run is not fatal
            print(f"{label:34}  ! {type(exc).__name__}: {str(exc)[:38]}")
            continue
        if m is None:
            print(f"{label:34}{'— not run —':>9}")
            continue
        print(f"{label:34}{m['repr_rms']:>9.3f}{m['repr_absmax']:>9.1f}"
              f"{m['sensitivity']:>9.2f}{m['saturation']:>7.0%}"
              f"{m['moved_cm']:>10.1f}{m['success']:>9.3f}")


if __name__ == "__main__":
    main()
