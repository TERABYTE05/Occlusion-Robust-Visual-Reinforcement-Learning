# Occlusion-Robust Visual Reinforcement Learning

A constrained reproduction of Noh, Lee & Myung (2025), *"Sample-efficient and
occlusion-robust reinforcement learning for robotic manipulation via multimodal fusion
dualization and representation normalization"* (Neural Networks 185, 107202), on
`FetchPushDense-v3` with a programmatic RGB occluder.

The question is whether fusion dualization with LayerNorm → SimplexNorm still gives a
measurable occlusion-robustness gain at a much smaller step budget. It is not an attempt
to match the paper's numbers.

**Budget, stated first because it governs how everything below should be read.** The paper
solves its tasks within **2M environment steps** and shows every baseline failing by 3M
(§5.3.1); action repeat is 1 in both their setup and ours. **We ran 500K** — 25 % of the
budget at which their method first succeeds, on one of their three tasks.

**At that budget we measured no gain.** All five pixel configurations finish at the
block-spawn base rate and a controlled four-rung ablation finds no measurable contribution
from either normalization component. **This is not evidence against the paper** — at a
quarter of its budget, on the one task of three it identifies as least sensitive to the
contribution, our experiment was not in a position to detect the effect. What it can
report is where the effect vanishes, plus a correction to an earlier positive finding of
our own. See [`docs/limitations.md`](docs/limitations.md) and
[`docs/expected_vs_obtained.md`](docs/expected_vs_obtained.md).

## Result

Five pixel configurations and a state-based upper bound, 500K steps, 3 seeds each.

| Configuration | Fusion | Normalization | Occlusion | Final success |
|---|---|---|---|---:|
| A `concat_noocc` | shared | none | off | 0.073 ± 0.023 |
| B `concat_occ` — rung 1 | shared | none | **on** | 0.073 ± 0.023 |
| D `ln_occ` — rung 2 | shared | LayerNorm | on | 0.073 ± 0.023 |
| E `ln_sn_occ` — rung 3 | shared | LayerNorm + SimplexNorm | on | 0.073 ± 0.023 |
| C `dual_occ` — rung 4 | **dualized** | LayerNorm + SimplexNorm | on | 0.073 ± 0.023 |
| `state_ddpg_500k` | — | — | — | **0.787 ± 0.163** |

Mean ± sample sd over 3 seeds, mean of the final five evaluations.

**1. The task-level result is a complete null, at 25 % of the paper's budget.** No pixel configuration learns to push the
block. The state-based anchor, which shares the DDPG backbone and differs only in seeing
the 28-D world state instead of pixels, reaches 0.787 — so the backbone and the task are
sound and the failure is in the visual pathway.

**2. Success rate cannot rank these configurations, and is not used to.** The
`eval/success_rate` trajectories are *bit-identical* across the four occluded rungs within
a seed: replacing the entire fusion module changes the outcome of 1–2 evaluation episodes
out of 500. No rung moves the block past ~1.7 cm and success requires 5 cm, so the metric
reports where the block spawned and nothing about the policy. The ladder is therefore read
on mechanism-level measurements (`scripts/mechanism_probe.py`).

**3. The ablation ladder is flat.** Each rung adds exactly one component, all on the
occluded task:

| | Critic action-sensitivity | Block displacement |
|---|---:|---:|
| rung 1  B  shared, none | 6.10 ± 0.16 % | 0.08 ± 0.07 cm |
| rung 2  D  + LayerNorm | 5.55 ± 0.87 % | 0.26 ± 0.39 cm |
| rung 3  E  + SimplexNorm | 5.30 ± 1.43 % | 0.18 ± 0.26 cm |
| rung 4  C  + dualization | 7.50 ± 1.73 % | **1.69 ± 1.67 cm** |
| state anchor | 42.95 ± 6.15 % | 13.34 ± 0.85 cm |

Adding LayerNorm and then SimplexNorm *decreases* action-sensitivity, both moves well
inside the seed spread. The one effect that survives is dualization's: it displaces the
block ~20× further than rung 1, with the seeds fully separated (min 0.32 > max 0.14). It
buys no success — 1.7 cm against a 5 cm threshold — and rung 4 is also the only rung with
an inflated critic (|Q| 26.3 vs ~11.5) and a saturated actor (67 % vs 4–8 %). Read plainly:
dualization yields a more confident, more saturated critic that pushes harder and still
does not solve the task.

**4. A correction to our own earlier finding.** An intermediate analysis reported
normalization lifting action-sensitivity "50×" and treated it as a positive reproduction of
the paper's §4.1.3 claim. That comparison was **A against C**, which changes occlusion,
normalization *and* dualization at once. A and B are the same network — `concat`,
`normalization: none` — differing only in the occluder, and they are 34× apart in
representation scale (A: RMS 15.27 ± 11.27, max 171; B: 0.46 ± 0.05, max 2.8). The
unbounded representation is Config A's, not a consequence of omitting normalization: B is
already 6.10 % action-sensitive, as sensitive as the full method. Under control the effect
disappears. 

Why the same architecture diverges under one condition and not the other is **not resolved
by this project**. A's own seeds span RMS 4.0–26.5 and saturation 37–80 %, so the
unnormalised fusion head's representation scale is unstable across conditions; nothing here
pins the cause.

### Figures

Committed under [`results/figures/`](results/figures) as PNG and PDF.

| | |
|---|---|
| `fig1_learning_curves` | A/B/C and the anchor, 100-episode rolling |
| `fig2_ablation` | the two gaps the design rests on, both zero |
| `fig3_diagnostic` | the block decodes at R² 0.994 from the policy's own input |
| `fig4_ladder` | the four-rung ablation — the headline figure |
| `fig5_confound` | why the earlier 50× result does not survive control |
| `fig6_metric_degeneracy` | why success rate cannot rank the configurations |

## Platform

Ubuntu 24.04, Python 3.12, one NVIDIA GPU (developed on an RTX 3060 12 GB, driver 595,
CUDA 13). Pinned versions are in [`requirements.txt`](requirements.txt): gymnasium 1.0.0,
gymnasium-robotics 1.3.1, mujoco 3.1.6, torch 2.12.0. The PyPI wheel of torch is already
the CUDA 13 build, so plain `pip install` is enough and no PyTorch index is needed.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Rendering: always set `MUJOCO_GL=egl`

MuJoCo picks its OpenGL backend when it is imported. On this project only **`egl`** is
correct:

| `MUJOCO_GL` | What happens |
|---|---|
| `egl` | Headless rendering on the NVIDIA GPU. **Use this.** |
| `glfw` / unset | On a machine with an Intel iGPU driving the display, this renders on the **iGPU**. It works and prints `glfw`, but it is slow |
| `osmesa` | CPU software rendering. Refused by `src/utils/gl.py` |

Every script prints the active backend on its first line (`[gl] MUJOCO_GL=egl -> active
backend: egl`). Check that the GPU is really doing the rendering:

```bash
MUJOCO_GL=egl python -c "
import mujoco; from OpenGL import GL
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\".1\"/></worldbody></mujoco>')
r = mujoco.Renderer(m, 84, 84); r.render()
print(GL.glGetString(GL.GL_RENDERER).decode()); r.close()"
# expected: NVIDIA GeForce RTX 3060/PCIe/SSE2
```

After a render script exits you may see an `EGLError` traceback about `free`. That is
MuJoCo tearing down its EGL context at interpreter shutdown, and it does not affect
results.

## Verify the environment

Run these from the repo root with the venv active.

**1. Unit tests.** 228 tests covering physics, proprioception, the reset fix, projection,
occlusion, replay buffer, n-step returns, augmentation, encoders, fusion, the agent's
gradient routing and the configs. About 10 s.

```bash
python -m pytest -q
```

**2. Block visibility at 84×84.** Judge from `results/contact/none_84px_true.png`. The
dark block must be obvious at true size.

```bash
MUJOCO_GL=egl python scripts/contact_sheet.py --n 20 --out results/contact
```

**3. Occluder coverage across fresh resets.** The script must report
`mask covered the block on 20/20 fresh resets`. Check
`results/contact/static_84px_view.png`: the crosshair should sit on the block in each clean
frame and inside the grey mask in each occluded frame.

```bash
MUJOCO_GL=egl python scripts/contact_sheet.py --n 20 --occlusion static --compare --annotate --out results/contact
```

**4. Throughput.**

```bash
MUJOCO_GL=egl python scripts/benchmark_fps.py --steps 10000
nvidia-smi --query-gpu=temperature.gpu,fan.speed,power.draw,utilization.gpu,clocks_throttle_reasons.active --format=csv -l 5
```

Measured on the production runs: **7.7 FPS and 18.1 h per 500K run with two runs
concurrent**, against 11.4 FPS for a single run. Two concurrent runs finish a pair ~1.4×
sooner than running them back to back; three contend on VRAM and are not recommended.

## Training

A run is a config file plus a seed. No hyperparameters on the command line.

```bash
./scripts/run.sh configs/config_c_dual_occ.yaml --seed 1          # one run
./scripts/run.sh configs/config_c_dual_occ.yaml --seed 1 --resume # after a crash
```

`run.sh` exports `MUJOCO_GL=egl` and uses `.venv/bin/python` directly, so it does not
depend on the caller's shell. `--resume` restores weights, optimizer moments, both RNG
streams **and the replay history** — check the log says `restored N transitions`.

Whole configurations run unattended and resumably. `run_all.sh` is the A/B/C production
matrix; `run_ladder.sh` is one ladder rung across its seeds:

```bash
./scripts/run_all.sh --dry-run                              # plan and time estimate
./scripts/run_ladder.sh configs/config_d_ln_occ.yaml        # seeds 1 2 3
./scripts/run_ladder.sh configs/config_e_ln_sn_occ.yaml 2   # just seed 2
```

Both skip a seed that has `final.pt`, resume one that has `latest.pt`, and continue to the
next seed if one fails — so re-running either script after any interruption picks up where
it stopped. Long runs belong in `tmux` or `systemd`:

```bash
tmux new-session -d -s rung_d -c $PWD \
  "./scripts/run_ladder.sh configs/config_d_ln_occ.yaml 2>&1 | tee -a results/rung_d.log"
tmux attach -t rung_d    # Ctrl+B then D to detach; session gone = finished
```

The tmux server reparents to init, so exiting the terminal is safe. A full logout is not —
run `loginctl enable-linger $USER` first if that is a risk.

## Analysis

```bash
# mechanism metrics for one checkpoint, or the whole ladder
MUJOCO_GL=egl python scripts/mechanism_probe.py config_c_dual_occ 1
MUJOCO_GL=egl python scripts/mechanism_probe.py --all

# cache the full matrix (~30 min, 20 episodes per config per seed)
MUJOCO_GL=egl python scripts/probe_matrix.py      # -> results/mechanism.json

# all six figures and all eight tables
python scripts/make_figures.py                    # -> results/figures/, docs/tables.md
```

`mechanism_probe.py` reports representation scale, critic action-sensitivity, actor
saturation, block displacement and success. Those are the metrics the ladder is read on,
because task success is pinned to the spawn base rate for any policy that does not move the
block. `make_figures.py` reads `results/mechanism.json` rather than hard-coding probe
numbers, so regenerating the figures requires the cached probe (or a re-run of
`probe_matrix.py` against the checkpoints).

## Layout

```
configs/    base.yaml, configs A/B/C/D/E, the state-DDPG anchor, two diagnostics
scripts/    run.sh, run_all.sh, run_ladder.sh, benchmark_fps.py, contact_sheet.py,
            mechanism_probe.py, probe_matrix.py, make_figures.py
src/envs/   pixel wrapper, reset-bug fix, proprio slice, occluder, make_env
src/models/ image/proprio encoders, augmentation, fusion heads, SimplexNorm, actor-critic
src/buffer/ lazy uint8 replay buffer with n-step returns
src/agent/  DDPG with dualized updates and gradient routing
src/utils/  GL backend detection, config loader, checkpointing, schedules
tests/      228 tests
results/figures/   committed figures (PNG + PDF)
```

## What is in the repository

Source, configs, tests and the figures are tracked. **Run outputs are not**: checkpoints,
TensorBoard logs, `results/mechanism.json` and `docs/tables.md` are gitignored, as are the
working documents (`ROADMAP.md`, `DECISIONS.md`, `RUNLOG.md`, `PROGRESS.md`,
`BENCHMARK.md`), which are kept current locally and not pushed. A fresh clone can run the
tests and launch training, but cannot regenerate the figures without the run outputs.

## Team

Teesha Ramchandani · Suraj Kumar
