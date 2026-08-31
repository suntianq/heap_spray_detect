#!/bin/bash
set -eu

TRACE_PID=${1:-$(pgrep -f "trace_pipe")}
TRACE_OUTPUT=${2:-/tmp/trace.log}
TRACE_STATS=${3:-/tmp/trace_stats.txt}
MODE=${4:-guest}          # guest = guest-side reader to kill, host-stream = no guest reader
TRACE_ROOT=/sys/kernel/debug/tracing

if [ "$MODE" = "guest" ] && [ -n "$TRACE_PID" ]; then
    kill "$TRACE_PID" 2>/dev/null || true
fi

echo 0 > "$TRACE_ROOT/tracing_on"
{
    for stats in "$TRACE_ROOT"/per_cpu/cpu*/stats; do
        test -f "$stats" || continue
        echo "[$stats]"
        cat "$stats"
    done
} > "$TRACE_STATS"
echo 0 > "$TRACE_ROOT/events/kmem/kmalloc/enable"
echo 0 > "$TRACE_ROOT/events/kmem/kfree/enable"
sync

# In host-stream mode the trace was streamed to the host (no guest-side file),
# so only assert on the file when the guest actually wrote one.
if [ "$MODE" = "guest" ]; then
    test -s "$TRACE_OUTPUT"
    echo "TRACE_LINES=$(wc -l < "$TRACE_OUTPUT")"
else
    echo "TRACE_LINES="
fi
