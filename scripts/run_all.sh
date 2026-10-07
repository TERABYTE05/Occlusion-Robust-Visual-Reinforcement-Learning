#!/usr/bin/env bash
# The production matrix: every configuration, every seed, unattended and resumable.
#
#     ./scripts/run_all.sh --dry-run      show the plan and the time estimate
#     ./scripts/run_all.sh                run it
#     SEEDS=2 ./scripts/run_all.sh        the Reduced matrix (fallback ladder rung 1)
#
# Launch it detached, because it runs for days:
#     tmux new -s matrix
#     ./scripts/run_all.sh 2>&1 | tee -a results/run_all.log
#     Ctrl+B then D
#
# Three properties that matter more than they look:
#
#   * **Seed-major order.** All three configs at seed 1, then all three at seed 2,
#     and so on -- never all seeds of A before touching B. If the Oct 18 freeze
#     arrives early, seed-major leaves a *complete ablation* at fewer seeds;
#     config-major would leave three seeds of Config A and no comparison at all.
#     The report needs A, B and C or it needs nothing.
#   * **Idempotent.** A run with final.pt is skipped; a run with latest.pt but no
#     final.pt is resumed, replay history included (D18). Re-running this script
#     after any interruption picks up exactly where it stopped, so it is safe to
#     run on a cron, after a reboot, or by a confused human at 2am.
#   * **One failure does not stop the matrix.** A crashed run is logged and the
#     next one starts. Losing one seed is bad; losing the eight that would have
#     run after it is worse.

set -uo pipefail
cd "$(dirname "$0")/.."

#: Seeds per configuration. 3 is the Full matrix (D4); the fallback ladder cuts
#: this to 2 before it cuts anything else, and cuts steps only after that.
SEEDS="${SEEDS:-3}"

#: The state-based DDPG baseline. The SOP commits to it as the stand-in for the
#: paper's SAC-State / TD3-State upper bounds (D19), so it is part of the matrix
#: and not an optional extra. It renders nothing and runs ~16x faster than a
#: pixel run, so it costs about 2% of the total.
STATE_CONFIG="configs/state_ddpg_500k.yaml"
PIXEL_CONFIGS=(
  configs/config_a_concat_noocc.yaml
  configs/config_b_concat_occ.yaml
  configs/config_c_dual_occ.yaml
)

PIXEL_FPS=11.4     # measured against the real agent (D4)
STATE_FPS=187      # the anchor, no rendering

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

log() { printf '[run_all %s] %s\n' "$(date '+%F %T')" "$*"; }

config_name() { ./.venv/bin/python -c "
from src.utils.config import load_config; print(load_config('$1')['name'])"; }

config_steps() { ./.venv/bin/python -c "
from src.utils.config import load_config; print(load_config('$1')['train']['steps'])"; }

# -- build the plan, seed-major -------------------------------------------
PLAN=()
for seed in $(seq 1 "$SEEDS"); do
  PLAN+=("$STATE_CONFIG|$seed")
  for cfg in "${PIXEL_CONFIGS[@]}"; do
    PLAN+=("$cfg|$seed")
  done
done

# -- what is already done --------------------------------------------------
total_hours=0
pending=0; done_already=0; resuming=0
echo
printf '%-26s %-5s %-9s %-10s %s\n' CONFIG SEED STEPS STATE 'EST'
printf '%-26s %-5s %-9s %-10s %s\n' -------------------------- ----- --------- ---------- -----
for item in "${PLAN[@]}"; do
  cfg="${item%%|*}"; seed="${item##*|}"
  name=$(config_name "$cfg"); steps=$(config_steps "$cfg")
  dir="results/$name/seed$seed"
  case "$cfg" in *state*) fps=$STATE_FPS ;; *) fps=$PIXEL_FPS ;; esac
  hours=$(./.venv/bin/python -c "print(f'{$steps/$fps/3600:.1f}')")

  if [[ -f "$dir/final.pt" ]]; then
    state="done"; done_already=$((done_already+1)); hours=0.0
  elif [[ -f "$dir/latest.pt" ]]; then
    state="resume"; resuming=$((resuming+1)); pending=$((pending+1))
  else
    state="queued"; pending=$((pending+1))
  fi
  total_hours=$(./.venv/bin/python -c "print(f'{$total_hours + $hours:.1f}')")
  printf '%-26s %-5s %-9s %-10s %sh\n' "$name" "$seed" "$steps" "$state" "$hours"
done
echo
log "$pending to run ($resuming resuming), $done_already already complete"
log "estimated ${total_hours}h = $(./.venv/bin/python -c "print(f'{$total_hours/24:.1f}')") days of GPU"

# -- preflight -------------------------------------------------------------
echo
if ! git diff --quiet || ! git diff --cached --quiet; then
  log "WARNING: working tree is dirty. The code freeze means what runs should"
  log "         match what is tagged -- commit before starting a production matrix."
fi
if ! git describe --tags --exact-match >/dev/null 2>&1; then
  log "WARNING: HEAD is not tagged. G4 requires the code tagged v1.0-experiments."
fi

free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
need_gb=$(( (pending * 160) / 1000 + 3 ))
log "disk: ${free_gb}G free, ~${need_gb}G needed (3 checkpoints/run + one 2.1G replay snapshot at a time)"
(( free_gb < need_gb )) && log "WARNING: that may not be enough."

if (( DRY_RUN )); then
  log "dry run -- nothing started."
  exit 0
fi

log "running the test suite before anything trains (CLAUDE.md)"
if ! ./.venv/bin/python -m pytest tests/ -q >/dev/null 2>&1; then
  log "TESTS FAILED -- refusing to start. Run: ./.venv/bin/python -m pytest tests/ -q"
  exit 1
fi
log "tests green"

# -- run -------------------------------------------------------------------
started=$(date +%s)
failed=()
for item in "${PLAN[@]}"; do
  cfg="${item%%|*}"; seed="${item##*|}"
  name=$(config_name "$cfg"); dir="results/$name/seed$seed"

  if [[ -f "$dir/final.pt" ]]; then
    log "SKIP  $name seed $seed -- already complete"
    continue
  fi

  flags=()
  if [[ -f "$dir/latest.pt" ]]; then
    flags+=(--resume)
    log "RESUME $name seed $seed -- watch for 'restored N transitions'"
  else
    log "START  $name seed $seed"
  fi

  run_began=$(date +%s)
  if ./scripts/run.sh "$cfg" --seed "$seed" "${flags[@]}"; then
    log "OK     $name seed $seed in $(( ($(date +%s) - run_began) / 60 )) min"
  else
    code=$?
    log "FAILED $name seed $seed (exit $code) -- continuing with the rest"
    log "       resume it later with: ./scripts/run.sh $cfg --seed $seed --resume"
    failed+=("$name seed $seed")
  fi
done

# -- summary ---------------------------------------------------------------
echo
log "matrix finished in $(( ($(date +%s) - started) / 3600 ))h"
if (( ${#failed[@]} )); then
  log "${#failed[@]} run(s) FAILED:"
  for f in "${failed[@]}"; do log "  - $f"; done
  log "re-running this script will resume them and skip everything complete."
else
  log "every run completed."
fi
log "record the outcome in RUNLOG.md and the PROGRESS.md run ledger."
