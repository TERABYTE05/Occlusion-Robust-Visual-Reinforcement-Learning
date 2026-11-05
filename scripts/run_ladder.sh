#!/usr/bin/env bash
# One rung of the ablation ladder, every seed, unattended and resumable.
#
#     ./scripts/run_ladder.sh configs/config_d_ln_occ.yaml          seeds 1 2 3
#     ./scripts/run_ladder.sh configs/config_e_ln_sn_occ.yaml 1 2   seeds 1 and 2
#
# run_all.sh is the production matrix and its config list is fixed at A/B/C.
# The ladder's rungs D and E arrived after it, and they run as two concurrent
# streams rather than one serial matrix -- the diagnostics measured 7.9 FPS per
# run with two in flight, versus 11.4 for one, so a pair finishes 1.4x sooner
# than the same two runs back to back. Hence one script per rung, launched twice.
#
# It keeps run_all.sh's three properties, which are the ones that matter when a
# stream is interrupted at 3am:
#
#   * **Idempotent.** final.pt is skipped; latest.pt without final.pt is resumed,
#     replay history included (D18). Re-running this script after any
#     interruption picks up exactly where it stopped.
#   * **One failure does not stop the rung.** A crashed seed is logged and the
#     next seed starts.
#   * **Seeds in order**, so an early stop leaves seed 1 of both rungs complete
#     and the D->E comparison intact at fewer seeds.
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
