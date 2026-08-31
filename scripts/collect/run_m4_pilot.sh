#!/usr/bin/env bash
# M4 pilot-v2 collection orchestration.
#
# Waits for the running attack batch (pid in $1, optional), then collects the
# remaining pilot data sequentially and builds/validates the processed set:
#   1. baseline PoCs per CVE (negative samples collected via the attack path)
#   2. matched normal controls per CVE (idle, msg_msg 256/2048, keyctl)
#   3. build_pilot_dataset.py -> csv/ + processed/ + acceptance gates
#
# Sequential execution keeps the attack batch's crash timing pristine (no CPU /
# memory contention) and lets each phase re-check its own report.
set -euo pipefail
cd "$(dirname "$0")/../.."

ATTACK_PID="${1:-}"
if [ -n "$ATTACK_PID" ] && kill -0 "$ATTACK_PID" 2>/dev/null; then
  echo "[m4] waiting for attack batch (pid $ATTACK_PID)..."
  while kill -0 "$ATTACK_PID" 2>/dev/null; do sleep 15; done
fi
echo "[m4] attack batch finished at $(date -Is)"

echo "[m4] === baseline collection ==="
python3 scripts/collect/collect_attack_stable.py \
  -c CVE-2017-11176 CVE-2017-7308 -v poc_cfh_baseline -n 5 \
  --expect-crash CVE-2017-11176 CVE-2017-7308 \
  -o datasets/pilot-v2/raw/baseline 2>&1 | tee /tmp/m4_baseline.log

echo "[m4] === normal matched controls ==="
for CVE in CVE-2017-11176 CVE-2017-7308; do
  python3 scripts/collect/collect_stable.py -c "$CVE" -n 4 -d 2 \
    -w idle msg_msg keyctl --msg-sizes 256 2048 \
    -o datasets/pilot-v2/raw/normal 2>&1 | tee "/tmp/m4_normal_${CVE}.log"
done

echo "[m4] === preprocess + gates ==="
python3 scripts/validate/build_pilot_dataset.py \
  --attack-raw datasets/pilot-v2/raw/attack \
  --normal-raw datasets/pilot-v2/raw/normal \
  --baseline-raw datasets/pilot-v2/raw/baseline \
  --out datasets/pilot-v2 2>&1 | tee /tmp/m4_build.log

echo "[m4] ALL DONE at $(date -Is)"
