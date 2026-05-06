#!/usr/bin/env bash
# Run all 4 experiment groups sequentially with CPU-friendly caps and patch caching.
# Logs to experiments/logs/group_*.log. Writes JSON under results/.
set -u

cd "$(dirname "$0")/.."
PROJ="$PWD"

export PRNU_PATCH_CACHE_DIR="$PROJ/data/processed/residual_patches"
export PRNU_MAX_PATCHES_PER_IMAGE="${PRNU_MAX_PATCHES_PER_IMAGE:-8}"

mkdir -p "$PROJ/experiments/logs" "$PRNU_PATCH_CACHE_DIR"

CONFIG="${CONFIG:-configs/cpu.yaml}"
MAX_DEVICES_FLAG="${MAX_DEVICES:+--max-devices $MAX_DEVICES}"

echo "=== Config: $CONFIG ==="
echo "=== Patch cache dir: $PRNU_PATCH_CACHE_DIR ==="
echo "=== Max patches/image: $PRNU_MAX_PATCHES_PER_IMAGE ==="
echo "=== Max devices flag: ${MAX_DEVICES_FLAG:-<none>} ==="
echo

run_step () {
    local name="$1"; shift
    local log="$PROJ/experiments/logs/${name}.log"
    echo ">>> [$name] starting at $(date '+%F %T')"
    (
      echo "=== $name === $(date) ==="
      echo "cmd: python $*"
      echo
      python "$@" 2>&1
      rc=$?
      echo
      echo "=== $name exit=$rc at $(date) ==="
      exit $rc
    ) > "$log"
    local rc=$?
    echo "<<< [$name] exit=$rc at $(date '+%F %T')  (log: $log)"
    return $rc
}

run_step group_A experiments/run_group_A.py --config "$CONFIG" $MAX_DEVICES_FLAG --max-patches-per-image "$PRNU_MAX_PATCHES_PER_IMAGE" || echo "[group_A] FAILED, continuing to next group"
run_step group_B experiments/run_group_B.py --config "$CONFIG" $MAX_DEVICES_FLAG || echo "[group_B] FAILED, continuing to next group"
run_step group_C experiments/run_group_C.py --config "$CONFIG" $MAX_DEVICES_FLAG --skip-ieee || echo "[group_C] FAILED, continuing to next group"
run_step group_D experiments/run_group_D.py --config "$CONFIG" $MAX_DEVICES_FLAG || echo "[group_D] FAILED, continuing to next group"

echo
echo "=== ALL DONE at $(date) ==="
ls -la "$PROJ/results/"
