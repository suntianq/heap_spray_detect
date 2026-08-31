#!/usr/bin/env bash
# M6 final-v2 attack top-up: bring every attack variant up to >=15 VALID runs.
#
# Why: run_final_v2.sh passes -n 15 = 15 ATTEMPTS per (cve, variant). The plan
# 8.2 requires 15-20 VALID per variant, and the 11176 single_spray PoC panics
# the guest in ~50-70% of runs (those are invalid by design: an unexpected VM
# crash yields a partial trace with no clean spray window). Pilot needed 22
# attempts to bank 13 valid for that variant.
#
# Safety: this script ONLY acts after attack_collect.done exists, i.e. after the
# attack phase's QEMUs are all stopped (each run's finally clause calls
# stop_qemu). The pipeline (run_final_v2.sh) is killed by its process GROUP
# (setsid-launched => pgid == its pid); the collection QEMUs live in separate
# sessions (start_new_session=True) and are never touched by that kill.
#
# After top-up, the build/train/report markers are removed so the pipeline
# rebuilds processed/, re-trains all models, and re-compiles the report on the
# completed dataset. normal_collect.done and attack_collect.done are kept, so
# no raw collection is redone.
set -euo pipefail
cd "$(dirname "$0")/../.."

ROOT="$PWD"
PY="$ROOT/.venv/bin/python3"
DATA="$ROOT/datasets"
RAW="$DATA/raw"
MARK_DIR="$DATA/.m6"
LOG_DIR="$DATA/.m6/logs"
PIPE_PIDFILE="$DATA/.m6/pipeline.pid"

LOCK_FILE="$DATA/.m6/attack_topup.lock"
if [ -f "$LOCK_FILE" ]; then
  echo "[topup] lock present ($LOCK_FILE) — another top-up in flight, exit 0"
  exit 0
fi
trap 'rm -f "$LOCK_FILE"' EXIT
touch "$LOCK_FILE"

MIN_VALID="${MIN_VALID:-15}"
# Per-variant attempt cap: bound runtime if a variant keeps crashing (the 11176
# single_spray PoC panics the guest in ~50-70% of runs; ~45s/run => 60 attempts
# ~= 45 min for one variant).
MAX_ATTEMPTS_PER_VARIANT="${MAX_ATTEMPTS_PER_VARIANT:-60}"

echo "[topup] min_valid=$MIN_VALID"
if [ ! -f "$MARK_DIR/attack_collect.done" ]; then
  echo "[topup] attack_collect.done not present yet — nothing to do, exit 0"
  exit 0
fi

# --- find the pipeline pid ---
PIPE_PID=""
if [ -f "$PIPE_PIDFILE" ]; then
  PIPE_PID=$(cat "$PIPE_PIDFILE")
fi
if [ -z "$PIPE_PID" ] || ! kill -0 "$PIPE_PID" 2>/dev/null; then
  PIPE_PID=$(pgrep -f "bash scripts/collect/run_final_v2.sh" | head -1 || true)
fi
echo "[topup] pipeline pid: ${PIPE_PID:-none}"

# --- per-variant valid counts ---
valid_of() { # cve class variant
  local d="$RAW/$1/$2/$3"
  if [ ! -d "$d" ]; then
    echo 0; return 0
  fi
  grep -rl '"status": "valid"' "$d" 2>/dev/null | wc -l
  return 0   # grep exits 1 when no match; do not let that trip `set -e`
}

shortfalls=()
for cve in CVE-2017-11176 CVE-2017-7308; do
  for variant in poc_cfh_single_spray poc_cfh_combo; do
    v=$(valid_of "$cve" attack "$variant")
    if [ "$v" -lt "$MIN_VALID" ]; then
      short=$(( MIN_VALID - v ))
      echo "[topup] $cve/$variant: $v valid, short by $short"
      shortfalls+=("$cve|$variant|$short")
    else
      echo "[topup] $cve/$variant: $v valid OK"
    fi
  done
done

if [ ${#shortfalls[@]} -eq 0 ]; then
  echo "[topup] no shortfalls — nothing to collect"
  exit 0
fi

# --- stop the pipeline (safe: attack QEMUs all stopped, build/train run no QEMU) ---
if [ -n "$PIPE_PID" ]; then
  echo "[topup] stopping pipeline group -${PIPE_PID}"
  kill -TERM -- "-$PIPE_PID" 2>/dev/null || kill -TERM "$PIPE_PID" 2>/dev/null || true
  for i in $(seq 1 20); do
    kill -0 "$PIPE_PID" 2>/dev/null || break
    sleep 1
  done
  kill -9 -- "-$PIPE_PID" 2>/dev/null || kill -9 "$PIPE_PID" 2>/dev/null || true
  echo "[topup] pipeline stopped"
fi

# --- top-up collection (loop per short variant until >= MIN_VALID valid) ---
for entry in "${shortfalls[@]}"; do
  cve=${entry%%|*}; rest=${entry#*|}
  variant=${rest%%|*}; short=${rest##*|}
  attempted=0
  while :; do
    v=$(valid_of "$cve" attack "$variant")
    if [ "$v" -ge "$MIN_VALID" ]; then
      echo "[topup] $cve/$variant reached $v valid (after $attempted attempts)"
      break
    fi
    if [ "$attempted" -ge "$MAX_ATTEMPTS_PER_VARIANT" ]; then
      echo "[topup] $cve/$variant hit attempt cap ($MAX_ATTEMPTS_PER_VARIANT), "
            "$v valid — continuing with shortfall"
      break
    fi
    # collect a batch of 6 attempts; recheck each loop.
    # --expect-crash: the attack PoCs panic the guest in a large fraction of
    # runs; pilot collected every attack variant WITH --expect-crash so those
    # crash runs (with a partial-but-usable spray window) are banked as valid.
    # run_final_v2.sh omitted it for the attack variants — that bug is why
    # 11176/combo came back 0/15. Match pilot here.
    echo "[topup] collecting $cve/$variant batch (valid=$v, attempted=$attempted)"
    "$PY" scripts/collect/collect_attack_stable.py -c "$cve" -v "$variant" \
        -n 6 --expect-crash "$cve" --poc-timeout 90 -o "$RAW" \
        >> "$LOG_DIR/attack_topup.log" 2>&1 \
      || echo "[topup] WARN: batch for $cve/$variant rc=$? (may still have banked valid runs)"
    attempted=$(( attempted + 6 ))
  done
done

# --- re-verify all attack variants + baseline ---
echo "[topup] post-topup counts:"
for cve in CVE-2017-11176 CVE-2017-7308; do
  for variant in poc_cfh_single_spray poc_cfh_combo; do
    v=$(valid_of "$cve" attack "$variant")
    echo "  $cve/$variant: $v valid"
  done
  b=$(valid_of "$cve" baseline "poc_cfh_baseline")
  echo "  $cve/poc_cfh_baseline: $b valid"
done

# --- invalidate downstream markers so the pipeline rebuilds/re-trains/reports ---
echo "[topup] dropping build_gates / train_* / final_report markers"
rm -f "$MARK_DIR/build_gates.done" "$MARK_DIR"/train_*.done "$MARK_DIR/final_report.done"

# --- relaunch pipeline (marker-resumable: normal/attack collection skipped) ---
echo "[topup] relaunching run_final_v2.sh"
setsid nohup "$ROOT/scripts/collect/run_final_v2.sh" \
    > "$DATA/m6_run.log" 2>&1 < /dev/null &
PIPE_PID=$!
echo "$PIPE_PID" > "$PIPE_PIDFILE"
echo "[topup] relaunched with pid $PIPE_PID (log: $DATA/m6_run.log)"
