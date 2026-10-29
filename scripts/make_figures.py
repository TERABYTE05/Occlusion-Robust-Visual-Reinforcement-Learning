"""Figures and tables for the report, from the finished matrix.

Mirrors the structure the reference paper and our SOP use:
  Figure 1  learning curves            <- paper Fig. 3, SOP "rolling curves"
  Figure 2  the two gaps / ablation    <- paper Figs. 5-6
  Figure 3  representation diagnostic  <- ours; explains the null
  Table 1   adaptation blueprint       <- SOP Table I
  Table 2   hyperparameters            <- paper Table B.1
  Table 3   results                    <- rebuilt; the roadmap's spec does not survive the data

Colour follows the dataviz reference palette. The three configurations under
comparison take categorical slots 1-3; the state anchor is drawn as a dashed
neutral because it is a reference bound, not a fourth peer -- which also keeps
the categorical set at three, where it validates on all pairs.
"""
from __future__ import annotations

import glob
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT = pathlib.Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (1, 2, 3)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
# Line widths nest deliberately: A, B and C coincide to within seed noise, so a
# single width would hide two of them under the third and read as missing data.
# Drawn thick-to-thin they appear as concentric bands, which is the finding.
RUNS = [  # (dir, label, colour, linestyle, linewidth)
    ("config_a_concat_noocc", "A  concat, no occlusion",   "#2a78d6", "-",  4.0),
    ("config_b_concat_occ",   "B  concat, occlusion",      "#eb6834", "-",  2.4),
    ("config_c_dual_occ",     "C  dual + norm, occlusion", "#1baf7a", "-",  1.2),
    ("state_ddpg_500k",       "State DDPG (upper bound)",  INK,       "--", 1.8),
]

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 150,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})


def series(run: str, seed: int, tag: str):
    """(steps, values) for one tag, or None if that run never logged it.

    The state anchor's seed 1 ran on Sep 30, before the SOP's 100-episode
    rolling metric was implemented (D22), so it has no `*_100ep` tags. Rather
    than drop the series or silently average over a different seed count, the
    callers carry n and the figure states it.
    """
    files = sorted(glob.glob(f"results/{run}/seed{seed}/tb/events*"))
    acc = EventAccumulator(files[0], size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return None
    ev = acc.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def stacked(run: str, tag: str, grid: np.ndarray):
    """Mean, sd and seed count across whichever seeds logged this tag."""
    rows = []
    for seed in SEEDS:
        got = series(run, seed, tag)
        if got is None:
            continue
        steps, vals = got
        rows.append(np.interp(grid, steps, vals))
    rows = np.vstack(rows)
    sd = rows.std(0, ddof=1) if len(rows) > 1 else np.zeros_like(rows[0])
    return rows.mean(0), sd, len(rows)


def smooth(y: np.ndarray, k: int = 25):
    """Moving average that does not fabricate edges.

    `np.convolve(..., mode="same")` zero-pads, which drags the first and last
    k//2 points toward zero -- on a return curve sitting near -9 that produced a
    spurious dive at step 0 and a spike at 500K. Reflect-padding keeps the ends
    at the value the data actually has there.
    """
    if len(y) < k:
        return y
    pad = k // 2
    padded = np.pad(y, pad, mode="reflect")
    return np.convolve(padded, np.ones(k) / k, mode="valid")[: len(y)]


# ---------------------------------------------------------------- figure 1
grid = np.linspace(0, 499_000, 300)
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))

for ax, tag, ylab, title in (
    (axes[0], "train/success_rate_100ep", "Success rate", "Success rate (100-episode rolling)"),
    (axes[1], "train/episode_return_100ep", "Episode return", "Episode return (100-episode rolling)"),
):
    for run, label, colour, ls, lw in RUNS:
        mu, sd, n = stacked(run, tag, grid)
        mu, sd = smooth(mu), smooth(sd)
        ax.plot(grid / 1e3, mu, color=colour, lw=lw, ls=ls,
                label=f"{label}  (n={n})", zorder=3)
        ax.fill_between(grid / 1e3, mu - sd, mu + sd, color=colour,
                        alpha=0.13, lw=0, zorder=2)
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

axes[0].set_ylim(bottom=0)
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("Figure 1  Learning curves, mean ± sd over 3 seeds, 500K steps",
             x=0.005, ha="left", fontsize=11, weight="bold")
fig.tight_layout(rect=(0, 0.13, 1, 0.93))
fig.text(0.005, 0.015,
         "A, B and C are drawn thick-to-thin because they coincide to within seed noise — all three are present. "
         "Only the state-based upper\nbound separates. Bands are ±1 sd across seeds. The anchor shows n=2 here: "
         "its seed 1 predates the rolling-window logger (D22).",
         ha="left", va="bottom", fontsize=8, color=MUTED)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig1_learning_curves.{ext}")
plt.close(fig)
print("fig1_learning_curves")


# ---------------------------------------------------------------- figure 2
def final5(run, seed, tag):
    _, v = series(run, seed, tag)
    return float(np.mean(v[-5:]))

succ = {r: np.array([final5(r, s, "eval/success_rate") for s in SEEDS]) for r, *_ in RUNS}
ret = {r: np.array([final5(r, s, "eval/episode_return") for s in SEEDS]) for r, *_ in RUNS}

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))

ax = axes[0]
labels = ["A", "B", "C", "State\nDDPG"]
xs = np.arange(len(RUNS))
for i, (run, lbl, colour, _, _lw) in enumerate(RUNS):
    ax.bar(i, succ[run].mean(), yerr=succ[run].std(ddof=1), width=0.62,
           color=colour, edgecolor="white", linewidth=2, capsize=4, zorder=3)
    ax.scatter([i] * 3, succ[run], color="white", edgecolor=INK,
               s=18, zorder=4, linewidth=0.8)
    ax.text(i, succ[run].mean() + succ[run].std(ddof=1) + 0.03,
            f"{succ[run].mean():.3f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(xs); ax.set_xticklabels(labels)
ax.set_ylabel("Success rate"); ax.set_ylim(0, 1.05)
ax.set_title("Final success rate (final 5 evaluations)", loc="left")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)

ax = axes[1]
a, b, c = succ["config_a_concat_noocc"], succ["config_b_concat_occ"], succ["config_c_dual_occ"]

# A bar chart cannot show "exactly zero" -- a zero-height bar is invisible. A dot
# with whiskers against the configurations' own seed-noise floor shows both that
# the gaps are zero and that zero is small *relative to what would have counted*.
noise = float(np.mean([a.std(ddof=1), b.std(ddof=1), c.std(ddof=1)]))
ax.axhspan(-noise, noise, color=GRID, alpha=0.55, lw=0, zorder=1)
ax.text(-0.52, -noise + 0.002, f"±1 sd of seed noise ({noise:.3f})", va="bottom",
        ha="left", fontsize=7.5, color=MUTED, zorder=5)

gaps = [("A − B\nocclusion penalty", a - b, "#eb6834"),
        ("C − B\nmechanism recovers", c - b, "#1baf7a")]
for i, (lbl, d, colour) in enumerate(gaps):
    ax.errorbar(i, d.mean(), yerr=d.std(ddof=1), fmt="o", color=colour,
                markersize=11, markeredgecolor="white", markeredgewidth=2,
                capsize=6, elinewidth=2, zorder=4)
    ax.text(i, 0.020, f"{d.mean():+.4f}\n± {d.std(ddof=1):.4f}", ha="center",
            fontsize=8.5, color=INK, weight="bold", zorder=5)
ax.axhline(0, color=INK, lw=1, zorder=2)
ax.set_xticks(range(2)); ax.set_xticklabels([g[0] for g in gaps])
ax.set_xlim(-0.6, 1.6)
ax.set_ylabel("Δ success rate"); ax.set_ylim(-0.055, 0.055)
ax.set_title("The two gaps the design rests on", loc="left")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)

fig.suptitle("Figure 2  Ablation: both gaps are exactly zero",
             x=0.005, ha="left", fontsize=11, weight="bold")
fig.tight_layout(rect=(0, 0.13, 1, 0.91))
fig.text(0.005, 0.015,
         "Left: white dots are individual seeds. Right: (C−B)/(A−B) = 0.0000 / 0.0000 is undefined — the "
         "occlusion penalty the\nmechanism was designed to recover does not exist in our setup.",
         ha="left", va="bottom", fontsize=8, color=MUTED)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig2_ablation.{ext}")
plt.close(fig)
print("fig2_ablation")


# ---------------------------------------------------------------- figure 3
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))

ax = axes[0]
probe = [("Proprioception slice (13-D)", -0.005, "#eb6834"),
         ("Random encoder (untrained)", 0.954, GRID),
         ("Image encoder $z^{image}$ (39200-D)", 0.966, "#2a78d6"),
         ("Fused $z^{mm}$ critic (128-D)", 0.994, "#2a78d6"),
         ("Fused $z^{mm}$ actor — policy input", 0.994, "#1baf7a")]
ys = np.arange(len(probe))
ax.barh(ys, [p[1] for p in probe], color=[p[2] for p in probe],
        edgecolor="white", linewidth=2, height=0.62, zorder=3)
for y, (_, v, _) in zip(ys, probe):
    ax.text(v + 0.03 if v > 0 else 0.03, y, f"{v:+.3f}", va="center",
            fontsize=8, color=INK)
ax.set_yticks(ys); ax.set_yticklabels([p[0] for p in probe])
ax.invert_yaxis(); ax.set_xlim(-0.15, 1.18); ax.axvline(0, color=INK, lw=1)
ax.set_xlabel("$R^2$ decoding block (x, y)")
ax.set_title("Where the block's position survives", loc="left")
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)

ax = axes[1]
# Measured on MATCHED episodes (env re-seeded identically per mode). An earlier
# run of this ablation reused one env across modes, so each mode saw different
# episodes and appeared to show a ~2.8 return cost; that was episode variance.
abl = [("Real frames", 0.050, -8.66), ("Frames zeroed", 0.050, -9.45),
       ("Frames = noise", 0.050, -8.76)]
xs = np.arange(len(abl))
ax.bar(xs, [a[1] for a in abl], width=0.55, color="#2a78d6",
       edgecolor="white", linewidth=2, zorder=3)
ax.axhline(0.050, color=INK, lw=1, ls=":", zorder=4)
ax.text(2.42, 0.052, "block-spawn base rate", ha="right", va="bottom",
        fontsize=7.5, color=MUTED)
for x, (_, s, r) in zip(xs, abl):
    ax.text(x, s + 0.004, f"{s:.2f}", ha="center", fontsize=8, color=INK)
    ax.text(x, 0.004, f"return {r:.1f}", ha="center", fontsize=7.5, color="white")
ax.set_xticks(xs); ax.set_xticklabels([a[0] for a in abl])
ax.set_ylabel("Success rate"); ax.set_ylim(0, 0.105)
ax.set_title("Config A, frames corrupted (60 matched episodes)", loc="left")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)

fig.suptitle("Figure 3  Diagnostic: the information is present; the policy ignores it entirely",
             x=0.005, ha="left", fontsize=11, weight="bold")
fig.tight_layout(rect=(0, 0.15, 1, 0.91))
fig.text(0.005, 0.015,
         "Left: ridge probe, n=200, 70/30 split, Config A's final checkpoint. The policy's own input encodes "
         "the block at R²=0.994,\nso this is not a perception failure. Right: corrupting the frames changes "
         "nothing — success is the spawn base rate in all three cases.",
         ha="left", va="bottom", fontsize=8, color=MUTED)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig3_diagnostic.{ext}")
plt.close(fig)
print("fig3_diagnostic")

# save the numbers the tables need
np.save(OUT / "_succ.npy", np.array([succ[r] for r, *_ in RUNS]))
np.save(OUT / "_ret.npy", np.array([ret[r] for r, *_ in RUNS]))
print("figures written to", OUT)


# ---------------------------------------------------------------- tables
def auc(run, seed, tag="eval/success_rate"):
    steps, vals = series(run, seed, tag)
    return float(np.trapz(vals, steps) / (steps[-1] - steps[0]))

def steps_to(run, seed, thresh, tag="eval/success_rate"):
    steps, vals = series(run, seed, tag)
    hit = np.where(vals >= thresh)[0]
    return int(steps[hit[0]]) if len(hit) else None

rows = []
for run, label, *_ in RUNS:
    s = np.array([final5(run, sd_, "eval/success_rate") for sd_ in SEEDS])
    r = np.array([final5(run, sd_, "eval/episode_return") for sd_ in SEEDS])
    a_ = np.array([auc(run, sd_) for sd_ in SEEDS])
    t50 = [steps_to(run, sd_, 0.5) for sd_ in SEEDS]
    t50s = "—  never reached" if all(x is None for x in t50) else \
           f"{int(np.mean([x for x in t50 if x])/1000)}K (n={sum(x is not None for x in t50)}/3)"
    rows.append((label, s, r, a_, t50s))

lines = ["# Tables\n",
         "*Generated by `scripts/make_figures.py` from `results/`. 3 seeds × 500K steps.*\n",
         "\n## Table 1 — Results\n",
         "Replaces the roadmap's original specification. Two of its four requested columns do not",
         "survive the data: **recovery fraction** is 0/0 and undefined, and **steps-to-50 %** is",
         "undefined for every pixel configuration because none reaches 50 %. Both are reported as",
         "such rather than left blank.\n",
         "| Configuration | Success rate | Episode return | AUC (success) | Steps to 50 % |",
         "|---|---:|---:|---:|---:|"]
for label, s, r, a_, t50s in rows:
    lines.append(f"| {label} | {s.mean():.3f} ± {s.std(ddof=1):.3f} | "
                 f"{r.mean():.2f} ± {r.std(ddof=1):.2f} | {a_.mean():.3f} ± {a_.std(ddof=1):.3f} | {t50s} |")

A, B, C = (np.array([final5(r_, sd_, "eval/success_rate") for sd_ in SEEDS])
           for r_ in ("config_a_concat_noocc", "config_b_concat_occ", "config_c_dual_occ"))
Ar, Br, Cr = (np.array([final5(r_, sd_, "eval/episode_return") for sd_ in SEEDS])
              for r_ in ("config_a_concat_noocc", "config_b_concat_occ", "config_c_dual_occ"))
ST = np.array([final5("state_ddpg_500k", sd_, "eval/success_rate") for sd_ in SEEDS])

lines += ["\nMean ± sample sd over 3 seeds; success and return are the mean of the final five",
          "evaluations (50 episodes per configuration per seed).\n",
          "\n## Table 2 — The derived quantities the design rests on\n",
          "| Quantity | Success | Episode return |", "|---|---:|---:|",
          f"| A − B  occlusion penalty | {(A-B).mean():+.4f} ± {(A-B).std(ddof=1):.4f} | {(Ar-Br).mean():+.3f} ± {(Ar-Br).std(ddof=1):.3f} |",
          f"| C − B  recovered by the mechanism | {(C-B).mean():+.4f} ± {(C-B).std(ddof=1):.4f} | {(Cr-Br).mean():+.3f} ± {(Cr-Br).std(ddof=1):.3f} |",
          f"| **(C − B) / (A − B)** | **undefined (0 / 0)** | — |",
          f"| State anchor − pixel configurations | {(ST-A).mean():+.3f} ± {(ST-A).std(ddof=1):.3f} | {(np.array([final5('state_ddpg_500k',s_,'eval/episode_return') for s_ in SEEDS])-Ar).mean():+.2f} |",
          "\nThe zeros are exact, not rounded: per-seed success is identical across A, B and C,",
          f"seed for seed — {[round(v,2) for v in A]}.\n",
          "\n## Table 3 — Representation diagnostic\n",
          "Ridge probe decoding the block's ground-truth (x, y), n = 200, 70/30 split, Config A's",
          "final checkpoint.\n",
          "| Representation | R² |", "|---|---:|",
          "| Proprioception slice (13-D) | −0.005 |",
          "| Randomly initialised encoder, `z_image` | +0.954 |",
          "| Image encoder `z_image` (39200-D) | +0.966 |",
          "| Fused `z_mm` critic head (128-D) | +0.994 |",
          "| **Fused `z_mm` actor head — the policy's input** | **+0.994** |",
          "\n\n## Table 4 — Adaptation blueprint\n",
          "The SOP's Table I, with the *as-built* column added.\n",
          "| Component | Paper | Proposed (SOP) | As built |", "|---|---|---|---|",
          "| Replay buffer | 10⁶ transitions | 10⁵ | 10⁵ ✅ |",
          "| Buffer storage | float32 stacks | uint8, lazy stacking | uint8, lazy ✅ |",
          "| Actor/critic width | 1024 | 512 | 512 ✅ |",
          "| Precision | float32 | AMP (`torch.amp`) | AMP ✅ *(implemented Oct 5; absent until then, D21)* |",
          "| Environment | `dm_control` scenes | `FetchPush-v3` + mask | **`FetchPushDense-v3`** + mask ⚠️ *(D1)* |",
          "| Task scope | reach / stack / lift | push only | push only ✅ |",
          "| Training steps | 2–3M | 500K | 500K ✅ |",
          "| Seeds | 5 | two to three | 3 ✅ |",
          "| Image encoder, proprio encoder, fusion dualization, SimplexNorm (d=128, V=8, τ=1), augmentation, DDPG backbone | — | unchanged | **unchanged** ✅ |",
          "\n\n## Table 5 — Hyperparameters as run\n",
          "The paper's Table B.1, with our values.\n",
          "| Parameter | Paper | Ours |", "|---|---|---|",
          "| Replay buffer capacity | 10⁶ | 10⁵ |", "| Mini-batch size | 256 | 256 |",
          "| Frame stack | 3 | 3 |", "| Seed frames | 4000 | 4000 |",
          "| Action repeat | 1 | 1 |", "| Discount γ | 0.99 | 0.99 |",
          "| Optimizer | Adam | Adam |", "| Learning rate | 10⁻⁴ | 10⁻⁴ |",
          "| Soft-update rate τ | 0.01 | 0.01 |", "| n-step returns | 3 | 3 |",
          "| Actor–critic hidden dim | 1024 | 512 |",
          "| Multimodal representation dim d | 128 | 128 |",
          "| SimplexNorm dimension V | 8 | 8 |",
          "| SimplexNorm temperature τ | 1 | 1 |",
          "| Exploration clip c | *not stated* | 0.3 *(from DrQ-v2, D13)* |",
          "| Measured throughput | — | 11.4 FPS, 12.2 h per 500K run |"]

out = pathlib.Path("docs/tables.md")
out.write_text("\n".join(lines) + "\n")
print("tables written to", out)
