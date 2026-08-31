#!/usr/bin/env bash
# Finish the CVE-2017-2636 cross-CVE dataset collection.
#
# Sequence (each step serialized so QEMU/SSH never conflict):
#   1. wait for the in-progress 2636 attack collection (single_spray+combo,
#      launched separately) to exit;
#   2. top-up 2636 single_spray and combo to >= MIN_VALID VALID runs;
#   3. collect 2636 poc_cfh_baseline to >= MIN_VALID valid runs;
#   4. collect 2636 normal workloads (8 classes x NORMAL_RUNS).
#
# Resumable: each phase records a .done marker and is skipped on re-launch.
# Safe with re-runs: run dirs carry unique uuids (make_run_id), never clobbered.
#
# Launch detached (survives session close):
#   nohup scripts/collect/collect_2636_complete.sh \
#     > datasets/.m6/logs/collect_2636_complete.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."

ROOT="$PWD"
PY="$ROOT/.venv/bin/python3"
[[ -x "$PY" ]] || { echo "venv python missing: $PY"; exit 1; }

DATA="$ROOT/datasets"
RAW="$DATA/raw"
MARK_DIR="$DATA/.m6"
LOG_DIR="$DATA/.m6/logs"
mkdir -p "$MARK_DIR" "$LOG_DIR"

MIN_VALID="${MIN_VALID:-15}"
MAX_ATTEMPTS_PER_VARIANT="${MAX_ATTEMPTS_PER_VARIANT:-60}"
NORMAL_RUNS="${NORMAL_RUNS:-20}"
COLLECT_DURATION="${COLLECT_DURATION:-30}"
NORMAL_WORKLOADS="idle msg_msg keyctl net_busy fs_io fork_stress mem_pressure"
CVE="CVE-2017-2636"

valid_of() { # class variant
  local class="$1" variant="$2"
  local vdir="$RAW/$CVE/$class/$variant"
  if [ ! -d "$vdir" ]; then echo 0; return 0; fi
  grep -rl '"status": "valid"' "$vdir" 2>/dev/null | wc -l
  return 0
}

echo "[2636] $(date -Is) start (min_valid=$MIN_VALID, normal=$NORMAL_RUNS/workload)"

# ---- 1. wait for the running attack collection ------------------------------
echo "[2636] waiting for in-progress attack collection (single_spray+combo) to exit ..."
while pgrep -f "collect_attack_stable.py -c $CVE -v poc_cfh_single_spray poc_cfh_combo" >/dev/null 2>&1; do
  sleep 30
done
echo "[2636] attack collection exited at $(date -Is)"

# ---- 2. top-up single_spray + combo to MIN_VALID valid ----------------------
if [ ! -f "$MARK_DIR/collect_2636_attack_topup.done" ]; then
  for variant in poc_cfh_single_spray poc_cfh_combo; do
    attempted=0
    while :; do
      v=$(valid_of attack "$variant")
      [ "$v" -ge "$MIN_VALID" ] && { echo "[2636] $variant: $v valid OK"; break; }
      [ "$attempted" -ge "$MAX_ATTEMPTS_PER_VARIANT" ] && {
        echo "[2636] $variant: attempt cap reached, $v valid (shortfall)"; break; }
      echo "[2636] $variant top-up batch (valid=$v, attempted=$attempted) $(date -Is)"
      "$PY" scripts/collect/collect_attack_stable.py -c "$CVE" -v "$variant" \
          -n 6 --expect-crash "$CVE" --poc-timeout 90 -o "$RAW" \
          >> "$LOG_DIR/collect_2636_attack_topup.log" 2>&1 \
        || echo "[2636] WARN: batch for $variant rc=$? (may still bank valid runs)"
      attempted=$(( attempted + 6 ))
    done
  done
  touch "$MARK_DIR/collect_2636_attack_topup.done"
fi

# ---- 3. baseline (exploit path, no spray) to MIN_VALID valid ----------------
if [ ! -f "$MARK_DIR/collect_2636_baseline.done" ]; then
  attempted=0
  while :; do
    v=$(valid_of baseline poc_cfh_baseline)
    [ "$v" -ge "$MIN_VALID" ] && { echo "[2636] baseline: $v valid OK"; break; }
    [ "$attempted" -ge "$MAX_ATTEMPTS_PER_VARIANT" ] && {
      echo "[2636] baseline: attempt cap reached, $v valid (shortfall)"; break; }
    echo "[2636] baseline batch (valid=$v, attempted=$attempted) $(date -Is)"
    "$PY" scripts/collect/collect_attack_stable.py -c "$CVE" \
        -v poc_cfh_baseline -n 6 --expect-crash "$CVE" --poc-timeout 90 \
        -o "$RAW" >> "$LOG_DIR/collect_2636_baseline.log" 2>&1 \
      || echo "[2636] WARN: baseline batch rc=$? (may still bank valid runs)"
    attempted=$(( attempted + 6 ))
  done
  touch "$MARK_DIR/collect_2636_baseline.done"
fi

# ---- 4. normal workloads -----------------------------------------------------
if [ ! -f "$MARK_DIR/collect_2636_normal.done" ]; then
  echo "[2636] normal collection start $(date -Is) (duration=$COLLECT_DURATION)"
  "$PY" scripts/collect/collect_stable.py -c "$CVE" \
      -n "$NORMAL_RUNS" -d "$COLLECT_DURATION" \
      -w $NORMAL_WORKLOADS --msg-sizes 256 2048 -o "$RAW" \
      > "$LOG_DIR/collect_2636_normal.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    touch "$MARK_DIR/collect_2636_normal.done"
    echo "[2636] normal done $(date -Is)"
  else
    echo "[2636] normal FAILED rc=$rc (log: $LOG_DIR/collect_2636_normal.log)"
    tail -40 "$LOG_DIR/collect_2636_normal.log" >&2
    exit $rc
  fi
fi

echo "[2636] ALL DONE $(date -Is)"
echo "[2636] summary:"
for variant in poc_cfh_single_spray poc_cfh_combo; do
  echo "  attack/$variant: $(valid_of attack "$variant") valid"
done
echo "  baseline: $(valid_of baseline poc_cfh_baseline) valid"
