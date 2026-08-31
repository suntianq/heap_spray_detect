#!/usr/bin/env bash
# Run the 4 cross-CVE comparison scenarios (task #21) on datasets/cross-cve.
# Sequential to avoid CPU contention; each writes its own runs/<exp> dir.
set -uo pipefail
cd "$(dirname "$0")/../.."

PY="$PWD/.venv/bin/python3"
ATT="datasets/cross-cve/processed/attack"
NRM="datasets/cross-cve/processed/normal"
MAN="datasets/cross-cve/dataset_manifest.json"
OUT="runs"

run() { # tag train... | test...
  local tag="$1"; shift
  local train="$1"; shift
  local test="$1"; shift
  echo "### [$tag] train=($train) test=($test) $(date -Is)"
  "$PY" scripts/train/run_cve_split.py \
    --model ocsvm --seed 42 \
    --train-cves $train --test-cves $test \
    --attack-data "$ATT" --normal-data "$NRM" \
    --dataset-manifest "$MAN" --out "$OUT" --name "$tag"
  local rc=$?
  echo "### [$tag] rc=$rc $(date -Is)"
  return $rc
}

echo "== cross-CVE 4-scenario run start $(date -Is) =="

# 1. train on the two original CVEs -> test the NEW CVE (zero-shot)
run cveAB_testC "CVE-2017-11176 CVE-2017-7308" "CVE-2017-2636" || echo "[FAIL] cveAB_testC"

# 2. train on ALL CVEs -> test ALL CVEs (standard, superset of M6)
run cveABC_testABC "CVE-2017-11176 CVE-2017-7308 CVE-2017-2636" "CVE-2017-11176 CVE-2017-7308 CVE-2017-2636" || echo "[FAIL] cveABC_testABC"

# 3. train on ALL CVEs -> test ONLY the new CVE
run cveABC_testC "CVE-2017-11176 CVE-2017-7308 CVE-2017-2636" "CVE-2017-2636" || echo "[FAIL] cveABC_testC"

# 4. train ONLY on the new CVE -> test the two original CVEs
run cveC_testAB "CVE-2017-2636" "CVE-2017-11176 CVE-2017-7308" || echo "[FAIL] cveC_testAB"

echo "== cross-CVE 4-scenario run done $(date -Is) =="
