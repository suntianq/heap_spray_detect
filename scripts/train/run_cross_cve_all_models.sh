#!/usr/bin/env bash
# Run the 4 cross-CVE scenarios for every frozen model (mlp_ae, lstm_ae,
# lstm_vae, ngram; ocsvm already done) on datasets/cross-cve.
# Four parallel workers (one per model), each running its 4 scenarios serially.
# Each run lands in its own runs/<timestamped>_cve*_v2_<model>_s42_*/ dir.
set -uo pipefail
cd "$(dirname "$0")/../.."

PY="$PWD/.venv/bin/python3"
ATT="datasets/cross-cve/processed/attack"
NRM="datasets/cross-cve/processed/normal"
MAN="datasets/cross-cve/dataset_manifest.json"
CSVN="datasets/cross-cve/csv/normal"
CSVA="datasets/cross-cve/csv/attack"
OUT="runs"
export OMP_NUM_THREADS=2

# Scenario report naming: deep models -> <date>_<tag>_v2_<model>_s42_*;
# ngram -> <date>_ngram_v2_ngram_s42_*_<trainSuffix>_test_<testSuffix>.
# Return 0 if an evaluation_report.json already exists for (model, tag).
scenario_done() { # model tag
  local model="$1" tag="$2"
  local glob
  case "$tag" in
    cveAB_testC)    glob="_111767308_test_2636";;
    cveABC_testABC) glob="_1117673082636_test_1117673082636";;
    cveABC_testC)   glob="_1117673082636_test_2636";;
    cveC_testAB)    glob="_2636_test_111767308";;
    *) echo "unknown tag $tag"; return 1;;
  esac
  if [ "$model" = "ngram" ]; then
    ls runs/*"$glob"/evaluation_report.json >/dev/null 2>&1
  else
    ls runs/*_${tag}_v2_${model}_s42_*/evaluation_report.json >/dev/null 2>&1
  fi
}

deep_scenario() { # model tag  train-cves...  --  test-cves...
  local model="$1" tag="$2"; shift 2
  local tr=() te=(); local sep=0
  for a in "$@"; do
    if [ "$a" = "--" ]; then sep=1; continue; fi
    [ "$sep" = 0 ] && tr+=("$a") || te+=("$a")
  done
  if scenario_done "$model" "$tag"; then
    echo "### [$model/$tag] SKIP (report exists)"
    return 0
  fi
  echo "### [$model/$tag] start $(date -Is) train=(${tr[*]}) test=(${te[*]})"
  "$PY" scripts/train/run_cve_split.py --model "$model" --seed 42 \
    --train-cves "${tr[@]}" --test-cves "${te[@]}" \
    --attack-data "$ATT" --normal-data "$NRM" --dataset-manifest "$MAN" \
    --out "$OUT" --name "$tag" || echo "[FAIL] $model/$tag rc=$?"
  echo "### [$model/$tag] done $(date -Is)"
}

ngram_scenario() { # tag  train-cves...  --  test-cves...
  local tag="$1"; shift
  local tr=() te=(); local sep=0
  for a in "$@"; do
    if [ "$a" = "--" ]; then sep=1; continue; fi
    [ "$sep" = 0 ] && tr+=("$a") || te+=("$a")
  done
  if scenario_done ngram "$tag"; then
    echo "### [ngram/$tag] SKIP (report exists)"
    return 0
  fi
  echo "### [ngram/$tag] start $(date -Is) train=(${tr[*]}) test=(${te[*]})"
  "$PY" scripts/train/run_ngram.py --seed 42 \
    --train-cves "${tr[@]}" --test-cves "${te[@]}" \
    --attack-data "$ATT" --normal-data "$NRM" \
    --attack-csv-root "$CSVA" --normal-csv-root "$CSVN" \
    --out "$OUT" || echo "[FAIL] ngram/$tag rc=$?"
  echo "### [ngram/$tag] done $(date -Is)"
}

ALL_A="CVE-2017-11176 CVE-2017-7308 CVE-2017-2636"
AB="CVE-2017-11176 CVE-2017-7308"
C="CVE-2017-2636"

worker_mlp() {
  deep_scenario mlp_ae cveAB_testC   $AB -- $C
  deep_scenario mlp_ae cveABC_testABC $ALL_A -- $ALL_A
  deep_scenario mlp_ae cveABC_testC  $ALL_A -- $C
  deep_scenario mlp_ae cveC_testAB   $C -- $AB
}
worker_lstm_ae() {
  deep_scenario lstm_ae cveAB_testC   $AB -- $C
  deep_scenario lstm_ae cveABC_testABC $ALL_A -- $ALL_A
  deep_scenario lstm_ae cveABC_testC  $ALL_A -- $C
  deep_scenario lstm_ae cveC_testAB   $C -- $AB
}
worker_lstm_vae() {
  deep_scenario lstm_vae cveAB_testC   $AB -- $C
  deep_scenario lstm_vae cveABC_testABC $ALL_A -- $ALL_A
  deep_scenario lstm_vae cveABC_testC  $ALL_A -- $C
  deep_scenario lstm_vae cveC_testAB   $C -- $AB
}
worker_ngram() {
  ngram_scenario cveAB_testC   $AB -- $C
  ngram_scenario cveABC_testABC $ALL_A -- $ALL_A
  ngram_scenario cveABC_testC  $ALL_A -- $C
  ngram_scenario cveC_testAB   $C -- $AB
}

echo "== cross-CVE ALL-MODELS run start $(date -Is) =="
# STRICTLY SEQUENTIAL: this 7GB WSL2 VM OOM-killed a 4-way parallel batch (two
# LSTMs each materializing ~2GB of sequence tensors at once). One model runs its
# 4 scenarios at a time; never run more than one training process at once.
# lstm_vae intentionally EXCLUDED: the user asked to finish after lstm_ae.
for entry in "worker_mlp model_mlp_ae" "worker_ngram model_ngram" \
             "worker_lstm_ae model_lstm_ae"; do
  read -r worker logname <<< "$entry"
  echo "-- $logname start $(date -Is) --"
  $worker > "datasets/cross-cve/.m6/logs/${logname}.log" 2>&1
  echo "-- $logname done rc=$? $(date -Is) --"
done
echo "== cross-CVE ALL-MODELS run done $(date -Is) =="
