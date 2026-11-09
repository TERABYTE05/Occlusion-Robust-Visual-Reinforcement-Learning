from __future__ import annotations

import glob
import json
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
RUNS = [
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
    files = sorted(glob.glob(f"results/{run}/seed{seed}/tb/events*"))
    acc = EventAccumulator(files[0], size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return None
    ev = acc.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def stacked(run: str, tag: str, grid: np.ndarray):
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
    if len(y) < k:
        return y
    pad = k // 2
    padded = np.pad(y, pad, mode="reflect")
    return np.convolve(padded, np.ones(k) / k, mode="valid")[: len(y)]


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

np.save(OUT / "_succ.npy", np.array([succ[r] for r, *_ in RUNS]))
np.save(OUT / "_ret.npy", np.array([ret[r] for r, *_ in RUNS]))
print("figures written to", OUT)


LADDER = [("rung1 B  shared, none",    "B\nnone",      "#eb6834"),
          ("rung2 D  shared, +LN",     "D\n+LN",       "#7b5ea7"),
          ("rung3 E  shared, +LN+SN",  "E\n+LN+SN",    "#d4a017"),
          ("rung4 C  dual, +LN+SN",    "C\n+dual",     "#1baf7a")]
REF_A, REF_ANCHOR = "ref   A  no occ, none", "ref   state anchor"

RESULT_ROWS = [("config_a_concat_noocc", "A  concat, no occlusion"),
               ("config_b_concat_occ",   "B  concat, occlusion — rung 1"),
               ("config_d_ln_occ",       "D  + LayerNorm — rung 2"),
               ("config_e_ln_sn_occ",    "E  + LayerNorm + SimplexNorm — rung 3"),
               ("config_c_dual_occ",     "C  + dualization — rung 4"),
               ("state_ddpg_500k",       "State DDPG (upper bound)")]

PROBE = json.loads(pathlib.Path("results/mechanism.json").read_text())


def pm(label: str, key: str) -> np.ndarray:
    return np.array([r[key] for r in PROBE if r["label"] == label])


def seed_bars(ax, labels, colours, vals, fmt="{:.2f}", width=0.62):
    xs = np.arange(len(labels))
    for i, (v, c) in enumerate(zip(vals, colours)):
        ax.bar(i, v.mean(), yerr=v.std(ddof=1), width=width, color=c,
               edgecolor="white", linewidth=2, capsize=4, zorder=3)
        ax.scatter([i] * len(v), v, color="white", edgecolor=INK, s=18,
                   zorder=4, linewidth=0.8)
        top = max(v.mean() + v.std(ddof=1), v.max())
        ax.text(i, top + ax.get_ylim()[1] * 0.035,
                fmt.format(v.mean()), ha="center", fontsize=8, color=INK, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    return xs


def offchart(ax, text):
    ax.annotate(text, xy=(0.985, 0.955), xycoords="axes fraction", ha="right",
                va="top", fontsize=7.5, color=MUTED,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.6))


fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5))
labels = [l for _, l, _ in LADDER]
colours = [c for _, _, c in LADDER]

ax = axes[0]
sens = [pm(k, "sensitivity") for k, *_ in LADDER]
ax.set_ylim(0, 12)
seed_bars(ax, labels, colours, sens)
ax.set_ylabel("$|Q(a{=}{+}1) - Q(a{=}{-}1)|\\ /\\ |Q|$   (%)")
ax.set_title("Critic action-sensitivity", loc="left")
offchart(ax, f"state anchor {pm(REF_ANCHOR, 'sensitivity').mean():.1f} %  (off chart)")

ax = axes[1]
moved = [pm(k, "moved_cm") for k, *_ in LADDER]
ax.set_ylim(0, 4.6)
seed_bars(ax, labels, colours, moved)
ax.set_ylabel("Block displacement (cm)")
ax.set_title("Does the block actually move?", loc="left")
offchart(ax, f"state anchor {pm(REF_ANCHOR, 'moved_cm').mean():.1f} cm  ·  success needs 5 cm")

ax = axes[2]
succ_l = [np.array([final5(r, s, "eval/success_rate") for s in SEEDS])
          for r in ("config_b_concat_occ", "config_d_ln_occ",
                    "config_e_ln_sn_occ", "config_c_dual_occ")]
anchor_succ = np.array([final5("state_ddpg_500k", s, "eval/success_rate") for s in SEEDS])
ax.set_ylim(0, 1.0)
seed_bars(ax, labels + ["anchor"], colours + [INK],
          succ_l + [anchor_succ], fmt="{:.3f}")
ax.set_ylabel("Success rate")
ax.set_title("Task success (final 5 evaluations)", loc="left")

fig.suptitle("Figure 4  The ablation ladder: adding the paper's components changes nothing",
             x=0.005, ha="left", fontsize=11, weight="bold")
fig.tight_layout(rect=(0, 0.14, 1, 0.92))
sd = lambda v: v.std(ddof=1)
fig.text(0.005, 0.015,
         "All four rungs run on the occluded task, so each step adds exactly one component: B→D adds LayerNorm, "
         "D→E adds SimplexNorm, E→C adds dualization.\nWhite dots are individual seeds; bars are mean ± 1 sd, n=3. "
         f"Sensitivity does not climb — the two normalization steps move it {sens[0].mean():.2f} → {sens[1].mean():.2f} "
         f"→ {sens[2].mean():.2f}, both decreases inside the seed\nspread. Only dualization separates, and only on "
         f"displacement: {moved[3].mean():.2f} ± {sd(moved[3]):.2f} cm vs {moved[0].mean():.2f} ± {sd(moved[0]):.2f}, "
         f"with no seed overlap (min {moved[3].min():.2f} > max {moved[0].max():.2f}) and no gain in success.",
         ha="left", va="bottom", fontsize=8, color=MUTED)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig4_ladder.{ext}")
plt.close(fig)
print("fig4_ladder")


fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))

ax = axes[0]
for i, (key, lbl, colour) in enumerate([(REF_A, "A\nno occlusion", "#2a78d6"),
                                        ("rung1 B  shared, none", "B\nocclusion", "#eb6834")]):
    v = pm(key, "repr_rms")
    ax.bar(i, v.mean(), width=0.5, color=colour, edgecolor="white", linewidth=2, zorder=3)
    ax.scatter([i] * len(v), v, color="white", edgecolor=INK, s=22, zorder=4, linewidth=0.9)
    ax.text(i, v.max() * 1.6, f"{v.mean():.2f}", ha="center", fontsize=8.5,
            color=INK, weight="bold", zorder=5)
ax.set_yscale("log")
ax.set_ylim(0.1, 120)
ax.set_xticks([0, 1]); ax.set_xticklabels(["A\nno occlusion", "B\nocclusion"])
ax.set_ylabel("RMS $|z^{mm}|$   (log scale)")
ax.set_title(f"Same architecture, {pm(REF_A, 'repr_rms').mean() / pm('rung1 B  shared, none', 'repr_rms').mean():.0f}× apart",
             loc="left")
ax.axhline(1.0, color=MUTED, lw=0.8, ls=":", zorder=2)
ax.text(1.46, 1.08, "scale of the action, $[-1, 1]$", ha="right", va="bottom",
        fontsize=7.5, color=MUTED)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)

ax = axes[1]
sB, sC = pm("rung1 B  shared, none", "sensitivity"), pm("rung4 C  dual, +LN+SN", "sensitivity")
order = [(REF_A, "A", "#2a78d6")] + [(k, l.split("\n")[0], c) for k, l, c in LADDER]
vals = [pm(k, "sensitivity") for k, *_ in order]
ax.set_ylim(0, 18.5)
seed_bars(ax, [l for _, l, _ in order], [c for _, _, c in order], vals, width=0.58)
ax.set_ylabel("Critic action-sensitivity (%)")
ax.set_title("Confounded vs controlled", loc="left")
ax.annotate("", xy=(4, 15.0), xytext=(0, 15.0),
            arrowprops=dict(arrowstyle="->", color="#b3261e", lw=1.4,
                            connectionstyle="arc3,rad=0.0"))
ax.text(2.0, 15.9, f"confounded: A → C changes 3 things at once  "
                   f"({sC.mean() / pm(REF_A, 'sensitivity').mean():.1f}×)",
        ha="center", fontsize=7.5, color="#b3261e")
ax.annotate("", xy=(4, 11.6), xytext=(1, 11.6),
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.4,
                            connectionstyle="arc3,rad=0.0"))
ax.text(2.5, 12.3, f"controlled: B → C  ({sC.mean() / sB.mean():.1f}×, seeds overlap)",
        ha="center", fontsize=7.5, color=INK)

fig.suptitle("Figure 5  The pathology belongs to Config A, not to the missing normalization",
             x=0.005, ha="left", fontsize=11, weight="bold")
fig.tight_layout(rect=(0, 0.16, 1, 0.92))
rA, rB, sA_ = pm(REF_A, "repr_rms"), pm("rung1 B  shared, none", "repr_rms"), pm(REF_A, "sensitivity")
fig.text(0.005, 0.015,
         "A and B are the same network — `concat`, `normalization: none` — differing only in the occluder. A's fused "
         f"vector is unbounded (RMS {rA.mean():.2f} ± {rA.std(ddof=1):.2f},\nseeds at "
         f"{' / '.join(f'{v:.1f}' for v in sorted(rA))}, max {pm(REF_A, 'repr_absmax').max():.0f}) and "
         f"{sA_.mean():.2f} % action-sensitive; B's is well-scaled ({rB.mean():.2f} ± {rB.std(ddof=1):.2f}, max "
         f"{pm('rung1 B  shared, none', 'repr_absmax').max():.1f}) and already {sB.mean():.2f} % sensitive — as "
         "sensitive as the\nfull method. The Oct 29 audit's \"50× improvement from normalization\" compared A "
         "against C and so changed occlusion, normalization and dualization together.",
         ha="left", va="bottom", fontsize=8, color=MUTED)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig5_confound.{ext}")
plt.close(fig)
print("fig5_confound")


EP_PER_EVAL = 10
LADDER_RUNS = ["config_b_concat_occ", "config_d_ln_occ",
               "config_e_ln_sn_occ", "config_c_dual_occ"]

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))

ax = axes[0]
for run, (_, lbl, colour), lw in zip(LADDER_RUNS, LADDER, (4.2, 3.0, 1.9, 1.0)):
    steps, vals = series(run, 1, "eval/success_rate")
    ax.plot(steps / 1e3, vals, color=colour, lw=lw,
            label=lbl.replace("\n", "  "), zorder=3)
ax.set_xlabel("Environment steps (thousands)")
ax.set_ylabel("Evaluation success rate")
ax.set_title("All four rungs, seed 1 — one curve is visible", loc="left")
ax.legend(frameon=False, loc="upper left", ncol=2)
ax.set_ylim(0, 0.46)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)

ax = axes[1]
diffs = []
for seed in SEEDS:
    mat = np.vstack([series(r, seed, "eval/success_rate")[1] for r in LADDER_RUNS])
    n_ep = int(round(np.abs(mat - mat[0]).sum() * EP_PER_EVAL))
    diffs.append((n_ep, mat.shape[1] * EP_PER_EVAL))
xs = np.arange(len(SEEDS))
ax.bar(xs, [d[0] for d in diffs], width=0.5, color="#b3261e",
       edgecolor="white", linewidth=2, zorder=3)
for x, (n, tot) in zip(xs, diffs):
    ax.text(x, n + 0.12, f"{n} of {tot}", ha="center", fontsize=8.5,
            color=INK, weight="bold")
ax.set_xticks(xs); ax.set_xticklabels([f"seed {s}" for s in SEEDS])
ax.set_ylabel("Evaluation episodes that differ")
ax.set_ylim(0, max(d[0] for d in diffs) + 1.6)
ax.set_title("Episodes changed by swapping the entire fusion module", loc="left")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)

fig.suptitle("Figure 6  Task success cannot rank these configurations",
             x=0.005, ha="left", fontsize=11, weight="bold")
fig.tight_layout(rect=(0, 0.15, 1, 0.91))
fig.text(0.005, 0.015,
         "Left: the four rungs are drawn thick-to-thin and still coincide — the trajectories are bit-identical, not "
         "merely close. Right: across 500 evaluation\nepisodes per configuration, replacing the fusion module changes "
         "the outcome of one or two. No rung moves the block past ~1.7 cm and success needs\n5 cm, so the metric "
         "reports the block's spawn position and nothing about the policy. The ladder must be read on Figure 4's "
         "mechanism columns instead.",
         ha="left", va="bottom", fontsize=8, color=MUTED)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig6_metric_degeneracy.{ext}")
plt.close(fig)
print("fig6_metric_degeneracy")
DIFFS = diffs


def auc(run, seed, tag="eval/success_rate"):
    steps, vals = series(run, seed, tag)
    return float(np.trapz(vals, steps) / (steps[-1] - steps[0]))

def steps_to(run, seed, thresh, tag="eval/success_rate"):
    steps, vals = series(run, seed, tag)
    hit = np.where(vals >= thresh)[0]
    return int(steps[hit[0]]) if len(hit) else None

rows = []
for run, label in RESULT_ROWS:
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
          "| Measured throughput | — | 7.7 FPS, 18.1 h per 500K run (two concurrent) |"]


def pmrow(label):
    f = lambda k, p=2: f"{pm(label, k).mean():.{p}f} ± {pm(label, k).std(ddof=1):.{p}f}"
    return (f("repr_rms"), f("sensitivity"), f("q_scale"),
            f"{pm(label,'saturation').mean():.0%}", f("moved_cm"))

lines += ["\n\n## Table 6 — Mechanism ladder\n",
          "From `results/mechanism.json` (`scripts/probe_matrix.py`): 20 episodes per",
          "configuration per seed, final checkpoint, evaluation seed 11000. Mean ± sample sd",
          "over 3 seeds. **These columns, not success, are what ranks the configurations** —",
          "see Table 8.\n",
          "| | RMS \\|z\\| | dQ/da % | \\|Q\\| | Actor saturation | Block displacement |",
          "|---|---:|---:|---:|---:|---:|"]
for key, lbl in [("rung1 B  shared, none", "rung 1  B  shared, none"),
                 ("rung2 D  shared, +LN", "rung 2  D  + LayerNorm"),
                 ("rung3 E  shared, +LN+SN", "rung 3  E  + SimplexNorm"),
                 ("rung4 C  dual, +LN+SN", "rung 4  C  + dualization"),
                 (REF_A, "ref  A  no occlusion, none"),
                 (REF_ANCHOR, "ref  state anchor")]:
    rms, sens, q, sat, cm = pmrow(key)
    lines.append(f"| {lbl} | {rms} | {sens} | {q} | {sat} | {cm} cm |")

sB, sD, sE, sC = (pm(k, "sensitivity") for k, *_ in LADDER)
mB, mC = pm("rung1 B  shared, none", "moved_cm"), pm("rung4 C  dual, +LN+SN", "moved_cm")
lines += [
    "\n**Reading the ladder.** Each step adds exactly one component, all on the occluded task.",
    f"B→D adds LayerNorm: {sB.mean():.2f} → {sD.mean():.2f} %. D→E adds SimplexNorm:",
    f"{sD.mean():.2f} → {sE.mean():.2f} %. Both are *decreases*, both well inside the seed",
    f"spread. E→C adds dualization: {sE.mean():.2f} → {sC.mean():.2f} %, an increase whose seeds",
    f"overlap rung 1's ({sorted(round(v,2) for v in sC)} against a rung-1 maximum of {sB.max():.2f}).",
    "**No normalization step in the paper's method produces a measurable gain here.**",
    "",
    f"The one effect that survives is displacement: {mC.mean():.2f} ± {mC.std(ddof=1):.2f} cm for",
    f"rung 4 against {mB.mean():.2f} ± {mB.std(ddof=1):.2f} cm for rung 1, with the seeds **fully",
    f"separated** (min {mC.min():.2f} > max {mB.max():.2f}). It is real and it is not enough:",
    "success requires 5 cm and the state anchor moves 13.3 cm. Rung 4 is also the only rung with",
    "an inflated critic and a saturated actor, so the honest statement is that dualization yields a",
    "more confident, more saturated critic that pushes harder without solving the task.\n",
]

rmsA, rmsB = pm(REF_A, "repr_rms"), pm("rung1 B  shared, none", "repr_rms")
sA = pm(REF_A, "sensitivity")
lines += ["\n## Table 7 — The confound in the Oct 29 mechanism audit\n",
          "Configs A and B are the **same network** — `concat`, `normalization: none` — differing",
          "only in the occluder. The audit of Oct 29 10:15 reported normalization lifting",
          "action-sensitivity from 0.21 % to 10.48 %, a \"50× improvement\". That comparison was",
          "A against C, which changes occlusion, normalization *and* dualization simultaneously.\n",
          "| | RMS \\|z\\| | max \\|z\\| | dQ/da % | Actor saturation |",
          "|---|---:|---:|---:|---:|",
          f"| A  no occlusion | {rmsA.mean():.2f} ± {rmsA.std(ddof=1):.2f} | "
          f"{pm(REF_A,'repr_absmax').max():.0f} | {sA.mean():.2f} ± {sA.std(ddof=1):.2f} | "
          f"{pm(REF_A,'saturation').mean():.0%} |",
          f"| B  occlusion | {rmsB.mean():.2f} ± {rmsB.std(ddof=1):.2f} | "
          f"{pm('rung1 B  shared, none','repr_absmax').max():.1f} | {sB.mean():.2f} ± {sB.std(ddof=1):.2f} | "
          f"{pm('rung1 B  shared, none','saturation').mean():.0%} |",
          "",
          "| Comparison | Changes | Sensitivity ratio |", "|---|---|---:|",
          f"| A → C  *as reported Oct 29* | occlusion + normalization + dualization | "
          f"**{sC.mean()/sA.mean():.1f}×** |",
          f"| B → C  *controlled* | normalization + dualization | {sC.mean()/sB.mean():.2f}× |",
          f"| B → D → E  *controlled* | normalization only | "
          f"{sE.mean()/sB.mean():.2f}× (a decrease) |",
          "",
          f"**The pathology is Config A's, not the naive design's.** A's fused vector is unbounded",
          f"(seeds at {', '.join(f'{v:.1f}' for v in sorted(rmsA))}, max 171) against an action in",
          f"[−1, 1]; B's is well-scaled ({rmsB.mean():.2f} ± {rmsB.std(ddof=1):.2f}, max 2.8) and",
          f"already {sB.mean():.2f} % action-sensitive — as sensitive as the full method. Why the same",
          "architecture diverges under one condition and not the other is **not resolved by this",
          "project**; A's own seed spread (RMS 4.0–26.5, saturation 37–80 %) also means the Oct 29",
          "audit, run on A seed 1 alone, sampled its most extreme seed.\n"]

lines += ["\n## Table 8 — Why success rate is not used to rank the ladder\n",
          "`eval/success_rate` trajectories compared across all four rungs, seed by seed. One unit",
          "of 0.1 in a 10-episode evaluation is exactly one episode.\n",
          "| Seed | Evaluation episodes per configuration | Episodes that differ across all four rungs |",
          "|---|---:|---:|"]
for seed, (n, tot) in zip(SEEDS, DIFFS):
    lines.append(f"| {seed} | {tot} | **{n}** |")
lines += ["",
          "Replacing the entire fusion module — unnormalised shared head to dualized",
          "LayerNorm + SimplexNorm — changes the outcome of one or two evaluation episodes in 500.",
          "The trajectories are bit-identical, not merely close (Figure 6). No rung moves the block",
          "past ~1.7 cm and success needs 5 cm, so the metric reports where the block spawned and",
          "nothing about the policy. It remains valid for the state anchor, which does solve the",
          f"task ({anchor_succ.mean():.3f} ± {anchor_succ.std(ddof=1):.3f}), and it is reported for the pixel",
          "configurations only to establish",
          "the null.\n"]

out = pathlib.Path("docs/tables.md")
out.write_text("\n".join(lines) + "\n")
print("tables written to", out)
