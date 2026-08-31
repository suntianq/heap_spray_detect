#!/usr/bin/env bash
# Batch-collect the 8 CVE-ready KHeaps exploits (2026-08-31 marker status).
#
# These CVEs need NO PoC modification (libexp sprays are self-marked, or markers
# were added on 2026-08-31):
#   CVE-2016-0728  CVE-2016-8655  CVE-2017-6074  CVE-2017-8824  CVE-2018-6555  (libexp self-marked)
#   CVE-2010-2959  CVE-2017-7184  CVE-2017-8890                                  (custom spray, marker added)
#
# NOT included (need PoC changes first — see CVE_COLLECTION_GUIDE.md):
#   CVE-2016-10150 CVE-2016-4557 CVE-2017-10661 CVE-2017-15649 CVE-2017-7533 (loop sprays)
#   CVE-2016-6187 (no single_spray PoC)
#
# Each CVE runs its own collect_cve_complete.sh to a per-CVE log and .done marker,
# so the batch is resumable: re-running skips already-completed CVEs. Runs are
# SEQUENTIAL (one CVE at a time) because QEMU collection hammers KVM/CPU; do not
# parallelize on one host. A missing done marker + non-zero rc is reported at the end.
#
# Usage:
#   nohup scripts/collect/collect_all_cves.sh \
#     > datasets/.m6/logs/collect_all_cves.log 2>&1 &
#   (CVES, MIN_VALID, NORMAL_RUNS, COLLECT_DURATION are env-overridable)
set -uo pipefail
cd "$(dirname "$0")/../.."

ROOT="$PWD"
DATA="${DATA:-$ROOT/datasets}"
MARK_DIR="$DATA/.m6"
LOG_DIR="$DATA/.m6/logs"
mkdir -p "$MARK_DIR" "$LOG_DIR"

# 8 CVEs ready to collect now (marker verified / self-marked).
CVES="${CVES:-CVE-2010-2959 CVE-2016-0728 CVE-2016-8655 CVE-2017-6074 \
      CVE-2017-7184 CVE-2017-8824 CVE-2017-8890 CVE-2018-6555}"
export MIN_VALID NORMAL_RUNS COLLECT_DURATION  # pass through to collect_cve_complete.sh

echo "[batch] $(date -Is) start — ${CVES}"
failed=()
for CVE in $CVES; do
  done_marker="$MARK_DIR/collect_${CVE}_complete.done"
  if [ -f "$done_marker" ]; then
    echo "[batch] $CVE already done, skip"
    continue
  fi
  echo "[batch] === $CVE start $(date -Is) ==="
  CVE="$CVE" bash scripts/collect/collect_cve_complete.sh \
      > "$LOG_DIR/collect_${CVE}_complete.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] || grep -q 'ALL DONE' "$LOG_DIR/collect_${CVE}_complete.log" 2>/dev/null; then
    touch "$done_marker"
    echo "[batch] $CVE done rc=$rc $(date -Is)"
  else
    echo "[batch] $CVE FAILED rc=$rc (log: $LOG_DIR/collect_${CVE}_complete.log)"
    tail -20 "$LOG_DIR/collect_${CVE}_complete.log" >&2
    failed+=("$CVE")
  fi
done

echo "[batch] $(date -Is) finished"
echo "[batch] summary:"
for CVE in $CVES; do
  [ -f "$MARK_DIR/collect_${CVE}_complete.done" ] \
    && echo "  $CVE: DONE" \
    || echo "  $CVE: (missing done marker)"
done
if [ ${#failed[@]} -gt 0 ]; then
  echo "[batch] FAILED: ${failed[*]}"
  exit 1
fi
echo "[batch] ALL OK"
