# Occlusion-Robust Visual Reinforcement Learning

A constrained reproduction of Noh, Lee & Myung (2025), *"Sample-efficient and
occlusion-robust reinforcement learning for robotic manipulation via multimodal fusion
dualization and representation normalization"* (Neural Networks 185, 107202), on
`FetchPushDense-v3` with a programmatic RGB occluder.

The question is whether fusion dualization with LayerNorm → SimplexNorm still gives a
measurable occlusion-robustness gain at a ~25× smaller step budget. It is not an attempt
to match the paper's numbers. See [`ROADMAP.md`](ROADMAP.md) for the design and schedule
and [`BENCHMARK.md`](BENCHMARK.md) for measured throughput.

| Config | Fusion | Normalization | Occlusion |
|---|---|---|---|
| A `concat_noocc` | concat | none | off |
| B `concat_occ` | concat | none | on |
| C `dual_occ` | dualized | LayerNorm + SimplexNorm | on |

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

## Verify the environment (Gate G1)

Run these from the repo root with the venv active.

**1. Unit tests.** These cover physics, proprioception, reset fix, projection, occlusion,
replay buffer, augmentation, encoders and configs. They take about 10 s.

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

**4. Throughput.** This benchmark sets the run matrix. It appends a table to
`BENCHMARK.md`; add `--no-write` for a dry run. Use `--steps 10000` (~10 min) to measure
*sustained* full-loop FPS, and watch the GPU in a second terminal so thermal throttling
shows up:

```bash
MUJOCO_GL=egl python scripts/benchmark_fps.py --steps 10000
nvidia-smi --query-gpu=temperature.gpu,fan.speed,power.draw,utilization.gpu,clocks_throttle_reasons.active --format=csv -l 5
```

Run-matrix thresholds on full-loop FPS: ≥ 15 → 3 seeds × 500K · 10–15 → 2 seeds × 500K ·
< 10 → 2 seeds × 300K. Stage 4 leaves out the actor, the twin critics and the target
networks, so treat it as an upper bound.

## Training

`src/train.py` lands in Phase 1. Once it exists, a single run is:

```bash
./scripts/run.sh configs/config_c_dual_occ.yaml --seed 1
```

`run.sh` exports `MUJOCO_GL=egl` itself. Long runs should go under `tmux` or `nohup`, so
that closing the terminal does not kill them.

## Layout

```
configs/    base.yaml + configs A/B/C and the state-DDPG anchor
scripts/    run.sh, benchmark_fps.py, contact_sheet.py
src/envs/   pixel wrapper, reset-bug fix, proprio slice, occluder
src/models/ image/proprio encoders, random-shift augmentation
src/buffer/ lazy uint8 replay buffer
tests/      pytest suite
```

## Team

Teesha Ramchandani · Suraj Kumar
