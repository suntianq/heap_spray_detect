#!/bin/bash
set -eu

BUFFER_SIZE_KB=${1:-16384}
TRACE_OUTPUT=${2:-/tmp/trace.log}
MODE=${3:-guest}          # guest = stream inside VM (legacy), host-stream = host reads trace_pipe
TRACE_ROOT=/sys/kernel/debug/tracing

test -d "$TRACE_ROOT"
test -e "$TRACE_ROOT/events/kmem/kmalloc/enable"
test -e "$TRACE_ROOT/events/kmem/kfree/enable"

echo 0 > "$TRACE_ROOT/tracing_on"

echo "$BUFFER_SIZE_KB" > "$TRACE_ROOT/buffer_size_kb"

echo 0 > "$TRACE_ROOT/events/enable"
echo 1 > "$TRACE_ROOT/events/kmem/kmalloc/enable"
echo 1 > "$TRACE_ROOT/events/kmem/kfree/enable"

echo 'sym-offset' > "$TRACE_ROOT/trace_options"
echo 'sym-addr' > "$TRACE_ROOT/trace_options"
echo 0 > "$TRACE_ROOT/options/stacktrace"

echo > "$TRACE_ROOT/trace"
rm -f "$TRACE_OUTPUT"

echo 1 > "$TRACE_ROOT/tracing_on"

if [ "$MODE" = "host-stream" ]; then
    # No guest-side reader: the host streams `cat trace_pipe` over SSH so events
    # before an expected VM crash are persisted to the host instead of being lost
    # when the guest dies (IMPLEMENTATION_PLAN.md 6.3).
    echo "TRACE_PIPE=$TRACE_ROOT/trace_pipe"
    echo "TRACE_PID="
else
    cat "$TRACE_ROOT/trace_pipe" > "$TRACE_OUTPUT" &
    TRACE_PID=$!
    kill -0 "$TRACE_PID"
    echo "TRACE_PID=$TRACE_PID"
fi
