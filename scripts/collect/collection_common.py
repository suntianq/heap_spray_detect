"""Shared, fail-closed helpers for QEMU/ftrace dataset collection."""

import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path


EVENT_RE = re.compile(r":\s+(?:kmalloc|kfree):")
MARKER_RE = re.compile(r":\s+tracing_mark_write\S*.*:\s+(SPRAY_START|SPRAY_END)\s*$")
# Extracts marker name + timestamp from a trace line, e.g.
#   bash-1883  [003] ....  13.843660: tracing_mark_write+0xb2/0x200 <...>: SPRAY_START
MARKER_TS_RE = re.compile(r"\[\d+\]\s+.*?\s+(\d+\.\d+):\s+tracing_mark_write\S*.*:\s+(SPRAY_START|SPRAY_END)\s*$")
# Matches the leading "<task>-<pid> [cpu] flags <sec.frac>:" of any trace line so
# the first/last event timestamps give the trace's time bounds.
TRACE_TS_RE = re.compile(r"\[\d+\]\s+.*?\s+(\d+\.\d+):\s+")


def make_run_id(index):
    return f"run_{index:03d}_{uuid.uuid4().hex[:12]}"


def sha256_file(path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_image(cve_folder):
    images = sorted((Path(cve_folder) / "img").glob("*.img"))
    return images[0] if images else None


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def ssh_args(key, port):
    return [
        "ssh", "-i", str(key), "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=5",
        "root@127.0.0.1",
    ]


def scp_args(key, port):
    return [
        "scp", "-i", str(key), "-P", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
    ]


def ssh_cmd(key, port, command, timeout=60):
    try:
        result = subprocess.run(ssh_args(key, port) + [command], capture_output=True,
                                text=True, timeout=timeout, check=False)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"


def scp_to_vm(key, port, local, remote, timeout=60):
    try:
        result = subprocess.run(scp_args(key, port) + [str(local), f"root@127.0.0.1:{remote}"],
                                capture_output=True, text=True, timeout=timeout, check=False)
        return result.returncode, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "timeout"


def scp_from_vm(key, port, remote, local, timeout=60):
    try:
        result = subprocess.run(scp_args(key, port) + [f"root@127.0.0.1:{remote}", str(local)],
                                capture_output=True, text=True, timeout=timeout, check=False)
        return result.returncode, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "timeout"


def launch_qemu(cve_folder, ssh_port, cores, mem_gb, qemu_log):
    """Launch one owned process group; callers must pass it to stop_qemu."""
    log_handle = open(qemu_log, "w")
    process = subprocess.Popen(
        ["bash", "./startvm", str(ssh_port), str(cores), f"{mem_gb}G"],
        cwd=str(cve_folder), stdin=subprocess.DEVNULL, stdout=log_handle,
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    process._heap_spray_log_handle = log_handle
    return process


def wait_for_ssh(process, key, port, timeout=100):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False, f"qemu exited with rc={process.returncode}"
        rc, output, error = ssh_cmd(key, port, "echo READY", timeout=10)
        if rc == 0 and output == "READY":
            return True, ""
        time.sleep(2)
    return False, "ssh readiness timeout"


def stop_qemu(process, grace_seconds=5):
    if process is None:
        return
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=grace_seconds)
        except ProcessLookupError:
            pass
    handle = getattr(process, "_heap_spray_log_handle", None)
    if handle:
        handle.close()


def write_manifest(path, payload):
    path = Path(path)
    payload = dict(payload)
    payload.setdefault("manifest_version", 2)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def validate_trace(path, require_markers=False, minimum_events=100):
    event_count = 0
    markers = []
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False, {"event_count": 0, "markers": [], "reason": "missing_or_empty_trace"}
    with path.open(errors="replace") as handle:
        for line in handle:
            if EVENT_RE.search(line):
                event_count += 1
            marker = MARKER_RE.search(line.strip())
            if marker:
                markers.append(marker.group(1))
    reason = None
    if event_count < minimum_events:
        reason = "too_few_events"
    elif require_markers and markers != ["SPRAY_START", "SPRAY_END"]:
        reason = "invalid_markers"
    return reason is None, {"event_count": event_count, "markers": markers, "reason": reason}


def parse_trace_overrun(path):
    total = 0
    path = Path(path)
    if not path.exists():
        return None
    pattern = re.compile(r"^overrun:\s+(\d+)")
    with path.open(errors="replace") as handle:
        for line in handle:
            match = pattern.match(line.strip())
            if match:
                total += int(match.group(1))
    return total


def trace_bounds(path):
    """Return (first_ts_ns, last_ts_ns) of the trace, or (None, None)."""
    first = last = None
    path = Path(path)
    if not path.exists():
        return None, None
    with path.open(errors="replace") as handle:
        for line in handle:
            match = TRACE_TS_RE.search(line.strip())
            if not match:
                continue
            ts_ns = int(float(match.group(1)) * 1e9)
            if first is None:
                first = ts_ns
            last = ts_ns
    return first, last


def extract_marker_timestamps(path):
    """Return [(name, timestamp_ns), ...] for SPRAY markers in trace order.

    Reads the trace log so the timestamps survive even when the guest itself
    streamed them (host-side streaming). Returns [] if the trace is missing.
    """
    result = []
    path = Path(path)
    if not path.exists():
        return result
    with path.open(errors="replace") as handle:
        for line in handle:
            match = MARKER_TS_RE.search(line.strip())
            if match:
                ts_ns = int(float(match.group(1)) * 1e9)
                result.append((match.group(2), ts_ns))
    return result


def validate_markers(markers):
    """Validate a list of (name, ts_ns) markers; returns (valid, info).

    Strict policy (IMPLEMENTATION_PLAN.md 6.3): markers must be balanced
    alternating SPRAY_START..SPRAY_END pairs, start with SPRAY_START, end with
    SPRAY_END, and no SPRAY_END may be earlier than its SPRAY_START.
    """
    names = [name for name, _ in markers]
    valid = bool(markers) and names[0] == "SPRAY_START" and names[-1] == "SPRAY_END"
    depth = 0
    for name in names:
        depth += 1 if name == "SPRAY_START" else -1
        if depth < 0:
            valid = False
    if depth != 0:
        valid = False
    if valid:
        start_ts = None
        for name, ts in markers:
            if name == "SPRAY_START":
                start_ts = ts
            elif start_ts is not None and ts <= start_ts:
                valid = False
    info = {
        "marker_count": len(markers),
        "marker_names": names,
        "spray_start_ns": markers[0][1] if markers else None,
        "spray_end_ns": markers[-1][1] if markers else None,
    }
    return valid, info


def resolve_spray_window(markers, expect_crash, vm_crashed, last_ts_ns):
    """Resolve the spray window for a run; returns (start, end, partial) or None.

    A balanced marker sequence gives the window directly. When a PoC expected to
    crash the VM crashes *during* the spray (the trace ends mid-spray, before the
    matching SPRAY_END), the crash is the natural end of the spray window, so the
    last event timestamp is used and the window is flagged partial.
    """
    valid, info = validate_markers(markers)
    if valid:
        return info["spray_start_ns"], info["spray_end_ns"], False
    if (expect_crash and vm_crashed and markers
            and markers[-1][0] == "SPRAY_START"):
        return markers[0][1], last_ts_ns, True
    return None
