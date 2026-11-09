#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <config.yaml> [seed ...]" >&2
  exit 2
fi

CONFIG="$1"; shift
SEEDS=("$@")
(( ${#SEEDS[@]} )) || SEEDS=(1 2 3)

[[ -f "$CONFIG" ]] || { echo "no such config: $CONFIG" >&2; exit 2; }

NAME=$(./.venv/bin/python -c "
from src.utils.config import load_config; print(load_config('$CONFIG')['name'])")

log() { printf '[%s %s] %s\n' "$NAME" "$(date '+%F %T')" "$*"; }

started=$(date +%s)
failed=()
log "rung starting: seeds ${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
  dir="results/$NAME/seed$seed"

  if [[ -f "$dir/final.pt" ]]; then
    log "SKIP  seed $seed -- already complete"
    continue
  fi

  flags=()
  if [[ -f "$dir/latest.pt" ]]; then
    flags+=(--resume)
    log "RESUME seed $seed -- watch for 'restored N transitions'"
  else
    log "START  seed $seed"
  fi

  began=$(date +%s)
  if ./scripts/run.sh "$CONFIG" --seed "$seed" "${flags[@]}"; then
    log "OK     seed $seed in $(( ($(date +%s) - began) / 60 )) min"
  else
    code=$?
    log "FAILED seed $seed (exit $code) -- continuing with the next seed"
    log "       resume it with: ./scripts/run_ladder.sh $CONFIG $seed"
    failed+=("$seed")
  fi
done

log "rung finished in $(( ($(date +%s) - started) / 3600 ))h"
if (( ${#failed[@]} )); then
  log "${#failed[@]} seed(s) FAILED: ${failed[*]}"
  log "re-running this script resumes them and skips everything complete."
else
  log "every seed completed."
fi
log "record the outcome in RUNLOG.md and the PROGRESS.md run ledger."
