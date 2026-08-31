#!/usr/bin/env bash
# M6 attack top-up watchdog: waits for attack_collect.done, then runs the
# top-up script once. Launched with setsid so it survives the Claude session
# closing (cron jobs die with the session; this does not).
#
# The top-up script is idempotent and lock-guarded, so an overlapping run from
# the cron (job 10b94e5a) just exits 0.
set -u
cd "$(dirname "$0")/../.."
DATA="$PWD/datasets/final-v2"
MARK="$DATA/.m6/attack_collect.done"
LOG="$DATA/.m6/logs/attack_watch.log"

echo "[watch] $(date -Is) waiting for $MARK" >> "$LOG"
while [ ! -f "$MARK" ]; do
  sleep 60
done
echo "[watch] $(date -Is) marker appeared — invoking top-up" >> "$LOG"
# brief grace so the pipeline's next phase (build_gates, QEMU-less) has started;
# the top-up kills the pipeline group anyway, so ordering here is just cosmetic.
sleep 5
bash "$PWD/scripts/collect/m6_attack_topup.sh" >> "$LOG" 2>&1
echo "[watch] $(date -Is) top-up finished rc=$?" >> "$LOG"
