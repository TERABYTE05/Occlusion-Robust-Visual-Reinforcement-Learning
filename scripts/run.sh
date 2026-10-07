#!/usr/bin/env bash
# Single run: ./scripts/run.sh configs/config_c_dual_occ.yaml --seed 1
#
# Landmine 4: MUJOCO_GL is set here, and only here. Never rely on it being set
# in a shell profile -- an unset variable lands on osmesa (CPU rendering) on a
# headless machine and quietly costs the project its schedule.
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <config.yaml> [--seed N] [extra args]" >&2
  exit 2
fi

CONFIG="$1"; shift

if [[ ! -f "$CONFIG" ]]; then
  echo "no such config: $CONFIG" >&2
  exit 2
fi

if [[ ! -f src/train.py ]]; then
  echo "src/train.py does not exist yet -- it lands in Phase 1 (G2, Sep 20)." >&2
  echo "Until then use scripts/benchmark_fps.py and scripts/contact_sheet.py." >&2
  exit 3
fi

# Use the project's own interpreter, never whatever `python` means in the
# caller's shell. A login shell here activates conda base, whose python has no
# numpy, and this script is launched from tmux, systemd and cron where nobody
# has activated anything. Relying on the caller to source the venv has now cost
# two launch attempts (RUNLOG 2026-10-01, 2026-10-06).
PYTHON="$PWD/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "no interpreter at $PYTHON" >&2
  echo "create it:  python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 4
fi

if ! "$PYTHON" -c "import numpy, torch, mujoco" 2>/dev/null; then
  echo "$PYTHON cannot import numpy/torch/mujoco -- the venv is incomplete" >&2
  echo "repair it:  .venv/bin/pip install -r requirements.txt" >&2
  exit 5
fi

# -m, not a path: src/train.py uses package-relative imports, and running it as
# a script puts src/ on sys.path instead of the repo root, which breaks them.
exec "$PYTHON" -m src.train --config "$CONFIG" "$@"
