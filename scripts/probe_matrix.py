"""Run mechanism_probe over the whole matrix and cache the result as JSON.

The probe rolls out 20 episodes per (config, seed) on the GPU, so the full
matrix costs ~30 minutes -- too slow to sit inside make_figures.py, which is
re-run every time a caption changes. This writes results/mechanism.json once;
the figures read it. Re-run it only when a checkpoint changes.

    MUJOCO_GL=egl python scripts/probe_matrix.py

Every number in Figures 4-5 and Tables 6-7 comes from this file, so nothing in
the plotting code is a hand-copied constant.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.mechanism_probe import measure  # noqa: E402

#: (results dir, label). The four ladder rungs first, then the two references.
#: Rung order is the ladder order and the figures rely on it.
MATRIX = [
    ("config_b_concat_occ",    "rung1 B  shared, none"),
    ("config_d_ln_occ",        "rung2 D  shared, +LN"),
    ("config_e_ln_sn_occ",     "rung3 E  shared, +LN+SN"),
    ("config_c_dual_occ",      "rung4 C  dual, +LN+SN"),
    ("config_a_concat_noocc",  "ref   A  no occ, none"),
    ("state_ddpg_500k",        "ref   state anchor"),
]
SEEDS = (1, 2, 3)
OUT = pathlib.Path("results/mechanism.json")


def main() -> None:
    rows = []
    for run, label in MATRIX:
        for seed in SEEDS:
            try:
                m = measure(run, seed, 20)
            except Exception as exc:              # a missing run is not fatal
                print(f"ERR  {run} seed{seed}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if m is None:
                print(f"--   {run} seed{seed} not run", flush=True)
                continue
            m["label"] = label
            rows.append(m)
            print(f"{label:26} s{seed}  rms={m['repr_rms']:7.3f} "
                  f"dQ%={m['sensitivity']:6.2f} |Q|={m['q_scale']:6.2f} "
                  f"sat={m['saturation']:.0%} cm={m['moved_cm']:5.2f} "
                  f"succ={m['success']:.3f}", flush=True)
    OUT.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"\n{len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
