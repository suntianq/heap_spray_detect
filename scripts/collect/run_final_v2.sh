#!/usr/bin/env bash
# M6 unified-dataset formal collection + build + full train/eval orchestration.
#
# Runs the full pipeline and is resumable: each phase records a .done marker
# under datasets/.m6/ and is skipped on re-launch. Launch detached so it
# survives the session closing:
#     nohup scripts/collect/run_final_v2.sh > datasets/m6_run.log 2>&1 &
#
# Layout (datasets restructure 2026-08-31): raw is CVE-first, i.e.
#   datasets/raw/<CVE>/{attack,normal,baseline}/<variant|workload>/run_XXX_*/
# and processed is datasets/processed/{attack,normal}. Collectors write the
# class subdir themselves; build_pilot_dataset strips it into class-free run_ids.
#
# Phases:
#   1. normal collection  : 8 classes x CVE kernels (idle, msg_msg 256/2048,
#      keyctl, net_busy, fs_io, fork_stress, mem_pressure)
#   2. attack collection  : CVEs x {single_spray, combo} + no-spray baseline
#   3. build + data gates : trace2csv -> csv2features -> pilot_gates (G1-G6)
#   4. train/eval         : 4 models (base)
#   5. final report       : compile runs/ACCEPTANCE_M6.md from the runs
#
# Fail-closed: collection only ever boots its own QEMU (start_new_session=True)
# and stop_qemu kills only that process group. The orphan CVE-2016-0728 QEMU
# (PID 283853) is NOT ours and is deliberately left untouched.
set -euo pipefail
cd "$(dirname "$0")/../.."

ROOT="$PWD"
PY="$ROOT/.venv/bin/python3"
[[ -x "$PY" ]] || { echo "venv python missing: $PY"; exit 1; }

# Scale (env-overridable). 20/15 meet plan 8.1/8.2 lower bounds.
NORMAL_RUNS="${NORMAL_RUNS:-20}"
ATTACK_RUNS="${ATTACK_RUNS:-15}"
COLLECT_DURATION="${COLLECT_DURATION:-30}"

DATA="$ROOT/datasets"
RAW="$DATA/raw"
PROC_ATTACK="$DATA/processed/attack"
PROC_NORMAL="$DATA/processed/normal"
LOG_DIR="$DATA/.m6/logs"
MARK_DIR="$DATA/.m6"
mkdir -p "$LOG_DIR" "$MARK_DIR"

CVES="CVE-2017-11176 CVE-2017-7308"
NORMAL_WORKLOADS="idle msg_msg keyctl net_busy fs_io fork_stress mem_pressure"
# M6 model set (user decision, revised 2026-08-31): svm (ocsvm), lstm-ae,
# baselines mlp_ae and lstm_vae. ngram was removed (worst cross-CVE AUC 0.807).
# NO leave-one-CVE-out this round -- the final report is a straight N-model base
# comparison. TOLERANT models are ones whose G10 is known-unstable or unknown: a
# completed run whose only problem is a FAILed gate is a valid finding and must
# not halt the pipeline (a real crash -- no "SOME M5 GATES FAILED" in the log --
# still halts).
MODELS="ocsvm lstm_ae mlp_ae lstm_vae"
TOLERANT_MODELS="lstm_ae mlp_ae lstm_vae"

# Run a phase once; record a marker so re-launch resumes rather than redoes.
run_phase() {
  local name="$1"; shift
  local marker="$MARK_DIR/${name}.done"
  if [ -f "$marker" ]; then
    echo "[m6] phase $name already done, skip"
    return 0
  fi
  echo "[m6] === phase $name start $(date -Is) ==="
  "$@" > "$LOG_DIR/${name}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[m6] phase $name FAILED rc=$rc (log: $LOG_DIR/${name}.log) — halting"
    tail -40 "$LOG_DIR/${name}.log" >&2
    exit $rc
  fi
  touch "$marker"
  echo "[m6] phase $name done $(date -Is)"
}

# ---------------------------------------------------------------- normal ---
run_phase normal_collect bash -c "
for CVE in $CVES; do
  echo \"[m6] normal: \$CVE (duration=$COLLECT_DURATION, runs=$NORMAL_RUNS)\"
  $PY scripts/collect/collect_stable.py -c \$CVE -n $NORMAL_RUNS -d $COLLECT_DURATION \
      -w $NORMAL_WORKLOADS --msg-sizes 256 2048 -o $RAW
done
"

# ---------------------------------------------------------------- attack ---
# --expect-crash on the attack variants too: the PoCs panic the guest in a large
# fraction of runs, and pilot (run_m4_pilot.sh) collected every attack variant
# with it so those crash runs (partial-but-usable spray window) are banked valid.
# Without it, an unexpected crash invalidates the run (the final-v2 regression
# that shipped 11176/combo 0/15; see m6_attack_topup.sh).
# Both variants and baseline write under the same raw root; the collectors
# create the attack/ or baseline/ class subdir under each CVE themselves.
run_phase attack_collect bash -c "
$PY scripts/collect/collect_attack_stable.py -c $CVES \
    -v poc_cfh_single_spray poc_cfh_combo -n $ATTACK_RUNS \
    --expect-crash $CVES --poc-timeout 90 -o $RAW
$PY scripts/collect/collect_attack_stable.py -c $CVES \
    -v poc_cfh_baseline -n $ATTACK_RUNS --expect-crash $CVES \
    --poc-timeout 90 -o $RAW
"

# ----------------------------------------------------------------- build ---
run_phase build_gates bash -c "
$PY scripts/validate/build_pilot_dataset.py --raw $RAW --out $DATA
"

# ---------------------------------------------------------------- train ----
# Base aggregate eval for the 4-model set (no LOO this round, per user; ngram
# removed). The TOLERANT models above are gate-FAIL-only tolerated.
run_train() { # <name> <model>
  local name="$1" model="$2"
  local marker="$MARK_DIR/${name}.done"
  if [ -f "$marker" ]; then
    echo "[m6] phase $name already done, skip"; return 0
  fi
  local tolerant=""
  for t in $TOLERANT_MODELS; do [ "$t" = "$model" ] && tolerant=1; done
  echo "[m6] === phase $name start $(date -Is) ==="
  local rc=0
  "$PY" scripts/train/run_experiment.py --model "$model" \
      --attack-data "$PROC_ATTACK" --normal-data "$PROC_NORMAL" \
      --dataset-manifest "$DATA/dataset_manifest.json" --out "$ROOT/runs" \
      > "$LOG_DIR/${name}.log" 2>&1 || rc=$?
  if [ $rc -ne 0 ]; then
    if [ -n "$tolerant" ] && grep -q "SOME M5 GATES FAILED" "$LOG_DIR/${name}.log"; then
      echo "[m6] phase $name gate-FAIL rc=$rc (TOLERANT: continuing; log: $LOG_DIR/${name}.log)"
      tail -8 "$LOG_DIR/${name}.log"
    else
      echo "[m6] phase $name FAILED rc=$rc (log: $LOG_DIR/${name}.log) — halting"
      tail -40 "$LOG_DIR/${name}.log" >&2
      exit $rc
    fi
  fi
  touch "$marker"
  echo "[m6] phase $name done $(date -Is)"
}
for MODEL in $MODELS; do
  run_train "train_${MODEL}_base" "$MODEL"
done

# --------------------------------------------------------------- report ----
run_phase final_report bash -c "
$PY scripts/validate/final_v2_report.py --runs $ROOT/runs \
    --dataset $DATA --out $ROOT/runs/ACCEPTANCE_M6.md
"

echo "[m6] ALL PHASES DONE at $(date -Is)"
