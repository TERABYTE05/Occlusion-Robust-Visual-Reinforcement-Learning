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

exec python src/train.py --config "$CONFIG" "$@"
