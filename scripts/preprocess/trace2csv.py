import re
import csv
import sys
import os
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("trace2csv")

KMALLOC_RE = re.compile(
    r"^\s*(.+?)-(\d+)\s+\[(\d+)\]\s+.*?\s+"
    r"(\d+\.\d+):\s+kmalloc:\s+"
    r"call_site=([0-9a-fx]+)\s+"
    r"ptr=([0-9a-fx]+)\s+"
    r"bytes_req=(\d+)\s+"
    r"bytes_alloc=(\d+)\s+"
    r"gfp_flags=(\S+)"
)

KFREE_RE = re.compile(
    r"^\s*(.+?)-(\d+)\s+\[(\d+)\]\s+.*?\s+"
    r"(\d+\.\d+):\s+kfree:\s+"
    r"call_site=([0-9a-fx]+)\s+"
    r"ptr=([0-9a-fx]+)"
)

BPFTRACE_RE = re.compile(
    r"^(\d+),(\d+),(\d+),(.*?),(ALLOC|FREE),([0-9a-fx]+),(\d+),(\d+),([0-9a-fx]+)"
)


MARKER_RE = re.compile(r"^\s*.+?\s+\[\d+\]\s+.*?\s+(\d+\.\d+):\s+tracing_mark_write\S*\s+\S+:\s+(SPRAY_START|SPRAY_END)")


def parse_ftrace_line(line):
    m = KMALLOC_RE.match(line)
    if m:
        comm, pid, cpu, ts, call_site, ptr, bytes_req, bytes_alloc, gfp = m.groups()
        ts_ns = int(float(ts) * 1e9)
        return {
            "timestamp_ns": ts_ns,
            "pid": int(pid),
            "tid": int(pid),
            "comm": comm.strip(),
            "op": "ALLOC",
            "ptr": ptr,
            "bytes_req": int(bytes_req),
            "bytes_alloc": int(bytes_alloc),
            "call_site": call_site,
        }
    m = KFREE_RE.match(line)
    if m:
        comm, pid, cpu, ts, call_site, ptr = m.groups()
        ts_ns = int(float(ts) * 1e9)
        return {
            "timestamp_ns": ts_ns,
            "pid": int(pid),
            "tid": int(pid),
            "comm": comm.strip(),
            "op": "FREE",
            "ptr": ptr,
            "bytes_req": 0,
            "bytes_alloc": 0,
            "call_site": call_site,
        }
    return None


def detect_markers(input_path):
    markers = {}
    with open(input_path, "r", errors="replace") as f:
        for line in f:
            m = MARKER_RE.match(line.strip())
            if m:
                ts_ns = int(float(m.group(1)) * 1e9)
                markers[m.group(2)] = ts_ns
    return markers


def parse_bpftrace_line(line):
    m = BPFTRACE_RE.match(line)
    if m:
        ts, pid, tid, comm, op, ptr, breq, balloc, cs = m.groups()
        return {
            "timestamp_ns": int(ts),
            "pid": int(pid),
            "tid": int(tid),
            "comm": comm.strip(),
            "op": op,
            "ptr": ptr,
            "bytes_req": int(breq),
            "bytes_alloc": int(balloc),
            "call_site": cs,
        }
    return None


def detect_format(filepath):
    with open(filepath, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if BPFTRACE_RE.match(line):
                return "bpftrace"
            if KMALLOC_RE.match(line) or KFREE_RE.match(line):
                return "ftrace"
    return "unknown"


def convert_file(input_path, output_path):
    fmt = detect_format(input_path)
    if fmt == "unknown":
        log.warning("Skipping non-trace file (unknown format): %s", input_path)
        return None, {}

    parser = parse_ftrace_line if fmt == "ftrace" else parse_bpftrace_line
    count = 0
    markers = detect_markers(input_path) if fmt == "ftrace" else {}

    with open(input_path, "r", errors="replace") as fin, open(output_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["timestamp_ns", "pid", "tid", "comm", "op", "ptr",
                         "bytes_req", "bytes_alloc", "call_site"])
        for line in fin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = parser(line)
            if rec:
                writer.writerow([
                    rec["timestamp_ns"], rec["pid"], rec["tid"], rec["comm"],
                    rec["op"], rec["ptr"], rec["bytes_req"], rec["bytes_alloc"],
                    rec["call_site"],
                ])
                count += 1
    return count, markers


def manifest_status(dirpath):
    """Return the collector manifest status for the run in *dirpath*, or None."""
    manifest_path = os.path.join(dirpath, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    return manifest.get("status")


def manifest_spray_window(dirpath):
    """Read a collector manifest for the run in *dirpath*.

    Returns {SPRAY_START: ns, SPRAY_END: ns} when the run is marked valid and the
    collector recorded an explicit spray window. The collector is authoritative:
    for runs that crashed mid-spray it records the crash boundary as SPRAY_END
    (marker_partial), which raw trace-marker extraction cannot recover.
    """
    manifest_path = os.path.join(dirpath, "manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "valid":
        return {}
    start = manifest.get("spray_start_ns")
    end = manifest.get("spray_end_ns")
    if start is None or end is None or end <= start:
        return {}
    return {"SPRAY_START": start, "SPRAY_END": end}


def main():
    parser = argparse.ArgumentParser(description="Convert ftrace/bpftrace logs to CSV")
    parser.add_argument("-i", "--input", required=True, help="Input .log file or directory")
    parser.add_argument("-o", "--output", required=True, help="Output .csv file or directory")
    args = parser.parse_args()

    all_markers = {}

    if os.path.isfile(args.input):
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        n, markers = convert_file(args.input, args.output)
        if n is None:
            log.error("Input is not a trace file: %s", args.input)
            sys.exit(1)
        log.info("Converted %s: %d events -> %s", args.input, n, args.output)
        if markers:
            all_markers[os.path.basename(args.input)] = markers
    elif os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        total = 0
        file_count = 0
        for dirpath, _, filenames in os.walk(args.input):
            rel = os.path.relpath(dirpath, args.input)
            out_dir = os.path.join(args.output, rel) if rel != "." else args.output
            os.makedirs(out_dir, exist_ok=True)
            # Plan 6.1: preprocessing only reads runs the collector marked valid.
            status = manifest_status(dirpath)
            if status is not None and status != "valid":
                log.info("  skipping invalid run %s (status=%s)", dirpath, status)
                continue
            # The collector manifest (if any) is the authoritative spray window.
            manifest_window = manifest_spray_window(dirpath)
            for fname in sorted(filenames):
                if not fname.endswith(".log"):
                    continue
                in_path = os.path.join(dirpath, fname)
                out_path = os.path.join(out_dir, fname.replace(".log", ".csv"))
                n, markers = convert_file(in_path, out_path)
                if n is None:
                    continue  # non-trace file (e.g. qemu.log); no CSV produced
                total += n
                file_count += 1
                rel_path = os.path.relpath(in_path, args.input)
                log.info("  %s: %d events", rel_path, n)
                if manifest_window:
                    markers = dict(manifest_window)
                if markers:
                    all_markers[rel_path] = markers
        log.info("Total: %d events from %d files", total, file_count)
    else:
        log.error("Input not found: %s", args.input)
        sys.exit(1)

    if all_markers:
        marker_path = os.path.join(args.output if os.path.isdir(args.output) else os.path.dirname(args.output), "spray_markers.json")
        with open(marker_path, "w") as f:
            json.dump(all_markers, f, indent=2)
        log.info("Spray markers saved: %d files with markers", len(all_markers))


if __name__ == "__main__":
    main()
