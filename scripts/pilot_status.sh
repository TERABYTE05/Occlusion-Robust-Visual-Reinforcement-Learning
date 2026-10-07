#!/usr/bin/env bash
# One-screen view of the pilot runs.
#
#     ./scripts/pilot_status.sh            once
#     watch -n 30 ./scripts/pilot_status.sh    live
#
# Reads TensorBoard under results/, not a log file: the event files are written
# by the runs themselves and live beside the checkpoints, so this keeps working
# after any shell, terminal or tool session that launched the runs has gone.
set -uo pipefail
cd "$(dirname "$0")/.."
# Same reason as run.sh: do not depend on the caller having activated anything.
PYTHON="$PWD/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON=python
exec "$PYTHON" - "$@" <<'PY'
import glob, os, pathlib, subprocess, sys

STEPS, RATE = 50_000, 11.4          # D4: measured steady-state pixel FPS
RUNS = ["config_a_pilot", "config_b_pilot", "config_c_pilot"]

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    sys.exit("tensorboard missing: .venv/bin/pip install -r requirements.txt")

print(f"{'CONFIG':<16} {'PROGRESS':<13} {'ETA':<7} {'SUCCESS (eval, every 5k)':<34} RETURN")
print("-" * 112)
remaining = 0.0
for name in RUNS:
    run = pathlib.Path("results") / name / "seed1"
    events = sorted(glob.glob(str(run / "tb" / "events*")))
    if not events:
        print(f"{name:<16} {'queued':<13} {'-':<7}")
        remaining += STEPS / RATE
        continue
    acc = EventAccumulator(events[0], size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags()["scalars"]
    succ = [e.value for e in acc.Scalars("eval/success_rate")] if "eval/success_rate" in tags else []
    ret = [e.value for e in acc.Scalars("eval/episode_return")] if "eval/episode_return" in tags else []
    step = acc.Scalars("eval/success_rate")[-1].step + 1 if succ else 0
    done = (run / "final.pt").exists()
    eta = "done" if done else f"~{(STEPS - step) / RATE / 60:.0f}m"
    if not done:
        remaining += (STEPS - step) / RATE
    print(f"{name:<16} {f'{step}/{STEPS}':<13} {eta:<7} "
          f"{str([round(v, 2) for v in succ[-7:]]):<34} {[round(v, 1) for v in ret[-5:]]}")

print()
live = subprocess.run(["pgrep", "-af", "src.train"], capture_output=True, text=True).stdout
active = next((w for line in live.splitlines() for w in line.split()
               if w.startswith("configs/config_")), "none")
print(f"running   : {active}")
print(f"all three : ~{remaining / 60:.0f} min remaining")
gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print(f"gpu       : {gpu}")
log = pathlib.Path("results/pilots.log")
if log.exists():
    tail = [l for l in log.read_text().splitlines() if l.startswith("[")][-1:]
    print(f"last line : {tail[0] if tail else '-'}")
PY
