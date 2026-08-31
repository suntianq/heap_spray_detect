#!/usr/bin/env bash
# Generic cross-CVE dataset collection orchestrator.
#
# Collects, for ONE CVE, the three data families used by the M6 training/eval
# pipeline:
#   1. attack  -- exploit WITH heap-spray markers: single_spray + combo variants
#   2. baseline -- exploit trigger path, NO spray (control group)
#   3. normal  -- benign workloads (7 classes x NORMAL_RUNS)
#
# Steps are serialized so QEMU/SSH never conflict. Each phase records a .done
# marker and is skipped on re-launch (resumable). Run dirs carry unique uuids,
# so re-runs never clobber existing data.
#
# PoC variant names (poc_cfh_single_spray / combo / baseline) are universal
# across the KHeaps framework: every CVE under KHeaps/exploit_env/CVEs/<CVE>/poc
# ships the same set of .c files, so NO per-CVE code change is needed -- just
# set CVE below (and ensure the CVE is in config.py CVE_LIST).
#
# Layout (datasets restructure 2026-08-31): collectors write the class subdir
# themselves into datasets/raw/<CVE>/{attack,normal,baseline}/<variant|workload>/...
#
# Usage:
#   CVE=CVE-2017-7533 nohup scripts/collect/collect_cve_complete.sh \
#     > datasets/.m6/logs/collect_CVE-2017-7533_complete.log 2>&1 &
#   (MIN_VALID, NORMAL_RUNS, COLLECT_DURATION, DATA are overridable env vars)
#
# After collection, verify the target slab with the build/gates pipeline before
# large-scale training (see the note at the bottom).
set -uo pipefail
cd "$(dirname "$0")/../.."

ROOT="$PWD"
PY="$ROOT/.venv/bin/python3"
[[ -x "$PY" ]] || { echo "venv python missing: $PY"; exit 1; }

# ---- configurable (env-overridable) -----------------------------------------
CVE="${CVE:?set CVE=... (e.g. CVE-2017-7533)}"
# DATA is the datasets root (datasets restructure 2026-08-31): the collectors
# write raw/<CVE>/{attack,normal,baseline}/... themselves; build_pilot_dataset.py
# consumes --raw datasets/raw --out datasets.
DATA="${DATA:-$ROOT/datasets}"
RAW="$DATA/raw"
MARK_DIR="$DATA/.m6"
LOG_DIR="$DATA/.m6/logs"
mkdir -p "$MARK_DIR" "$LOG_DIR"

MIN_VALID="${MIN_VALID:-15}"                    # valid runs per attack/baseline variant
MAX_ATTEMPTS_PER_VARIANT="${MAX_ATTEMPTS_PER_VARIANT:-60}"
NORMAL_RUNS="${NORMAL_RUNS:-20}"                # per workload
COLLECT_DURATION="${COLLECT_DURATION:-30}"      # seconds per normal run
NORMAL_WORKLOADS="${NORMAL_WORKLOADS:-idle msg_msg keyctl net_busy fs_io fork_stress mem_pressure}"
ATTACK_VARIANTS="${ATTACK_VARIANTS:-poc_cfh_single_spray poc_cfh_combo}"
# CVE(s) whose crash the exploit is expected to trigger; VALID requires vm_crashed.
EXPECT_CRASH="${EXPECT_CRASH:-$CVE}"
POC_TIMEOUT="${POC_TIMEOUT:-90}"
# A short-hand slug used only in .done marker / log file names (defaults to CVE).
SLUG="${SLUG:-$CVE}"

valid_of() { # class variant
  local class="$1" variant="$2"
  local vdir="$RAW/$CVE/$class/$variant"
  if [ ! -d "$vdir" ]; then echo 0; return 0; fi
  grep -rl '"status": "valid"' "$vdir" 2>/dev/null | wc -l
  return 0
}

echo "[$SLUG] $(date -Is) start (cve=$CVE, min_valid=$MIN_VALID, normal=$NORMAL_RUNS/workload)"

# ---- 1. wait for any in-progress attack collection for this CVE -------------
echo "[$SLUG] waiting for in-progress attack collection to exit ..."
while pgrep -f "collect_attack_stable.py -c $CVE" >/dev/null 2>&1; do
  sleep 30
done
echo "[$SLUG] attack collection exited at $(date -Is)"

# ---- 2. attack top-up (single_spray + combo) to MIN_VALID valid -------------
if [ ! -f "$MARK_DIR/collect_${SLUG}_attack_topup.done" ]; then
  for variant in $ATTACK_VARIANTS; do
    attempted=0
    while :; do
      v=$(valid_of attack "$variant")
      [ "$v" -ge "$MIN_VALID" ] && { echo "[$SLUG] $variant: $v valid OK"; break; }
      [ "$attempted" -ge "$MAX_ATTEMPTS_PER_VARIANT" ] && {
        echo "[$SLUG] $variant: attempt cap reached, $v valid (shortfall)"; break; }
      echo "[$SLUG] $variant top-up batch (valid=$v, attempted=$attempted) $(date -Is)"
      "$PY" scripts/collect/collect_attack_stable.py -c "$CVE" -v "$variant" \
          -n 6 --expect-crash $EXPECT_CRASH --poc-timeout "$POC_TIMEOUT" -o "$RAW" \
          >> "$LOG_DIR/collect_${SLUG}_attack_topup.log" 2>&1 \
        || echo "[$SLUG] WARN: batch for $variant rc=$? (may still bank valid runs)"
      attempted=$(( attempted + 6 ))
    done
  done
  touch "$MARK_DIR/collect_${SLUG}_attack_topup.done"
fi

# ---- 3. baseline (exploit path, no spray) to MIN_VALID valid ----------------
if [ ! -f "$MARK_DIR/collect_${SLUG}_baseline.done" ]; then
  attempted=0
  while :; do
    v=$(valid_of baseline poc_cfh_baseline)
    [ "$v" -ge "$MIN_VALID" ] && { echo "[$SLUG] baseline: $v valid OK"; break; }
    [ "$attempted" -ge "$MAX_ATTEMPTS_PER_VARIANT" ] && {
      echo "[$SLUG] baseline: attempt cap reached, $v valid (shortfall)"; break; }
    echo "[$SLUG] baseline batch (valid=$v, attempted=$attempted) $(date -Is)"
    "$PY" scripts/collect/collect_attack_stable.py -c "$CVE" \
        -v poc_cfh_baseline -n 6 --expect-crash $EXPECT_CRASH --poc-timeout "$POC_TIMEOUT" \
        -o "$RAW" >> "$LOG_DIR/collect_${SLUG}_baseline.log" 2>&1 \
      || echo "[$SLUG] WARN: baseline batch rc=$? (may still bank valid runs)"
    attempted=$(( attempted + 6 ))
  done
  touch "$MARK_DIR/collect_${SLUG}_baseline.done"
fi

# ---- 4. normal workloads -----------------------------------------------------
if [ ! -f "$MARK_DIR/collect_${SLUG}_normal.done" ]; then
  echo "[$SLUG] normal collection start $(date -Is) (duration=$COLLECT_DURATION)"
  "$PY" scripts/collect/collect_stable.py -c "$CVE" \
      -n "$NORMAL_RUNS" -d "$COLLECT_DURATION" \
      -w $NORMAL_WORKLOADS --msg-sizes 256 2048 -o "$RAW" \
      > "$LOG_DIR/collect_${SLUG}_normal.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    touch "$MARK_DIR/collect_${SLUG}_normal.done"
    echo "[$SLUG] normal done $(date -Is)"
  else
    echo "[$SLUG] normal FAILED rc=$rc (log: $LOG_DIR/collect_${SLUG}_normal.log)"
    tail -40 "$LOG_DIR/collect_${SLUG}_normal.log" >&2
    exit $rc
  fi
fi

echo "[$SLUG] ALL DONE $(date -Is)"
echo "[$SLUG] summary:"
for variant in $ATTACK_VARIANTS; do
  echo "  attack/$variant: $(valid_of attack "$variant") valid"
done
echo "  baseline: $(valid_of baseline poc_cfh_baseline) valid"
echo
echo "[$SLUG] NOTE: after collection, verify the target slab before large-scale"
echo "[$SLUG] training -- run the build/gates pipeline and confirm the expected"
echo "[$SLUG] slab bucket is covered (see the CVE-2017-2636 lesson: planned 4096,"
echo "[$SLUG] actual reclaim was kmalloc-8192)."
