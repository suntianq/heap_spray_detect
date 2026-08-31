"""Synthetic ftrace/bpftrace trace and CSV generators for tests.

Formats intentionally mirror the real parsers in scripts/preprocess/trace2csv.py
and the CSV layout it emits, so tests exercise the same regular expressions and
column semantics as production data.
"""

import csv
import os

DEFAULT_CALL_SITE_ALLOC = "ffffffff8139c7f1"
DEFAULT_CALL_SITE_FREE = "ffffffff8136ccd1"


def kmalloc_line(comm, pid, ts_ns, ptr, bytes_req, bytes_alloc,
                 call_site=DEFAULT_CALL_SITE_ALLOC, cpu=0):
    """A single kmalloc ftrace line."""
    return ("  {}-{:<5} [{:03d}] ...1 {:.6f}: kmalloc: "
            "call_site={} ptr={:016x} bytes_req={} bytes_alloc={} gfp_flags=GFP_KERNEL".format(
                comm, pid, cpu, ts_ns / 1e9, call_site, ptr, bytes_req, bytes_alloc))


def kfree_line(comm, pid, ts_ns, ptr, call_site=DEFAULT_CALL_SITE_FREE, cpu=0):
    """A single kfree ftrace line."""
    return "  {}-{:<5} [{:03d}] ...1 {:.6f}: kfree: call_site={} ptr={:016x}".format(
        comm, pid, cpu, ts_ns / 1e9, call_site, ptr)


def marker_line(ts_ns, marker):
    """A tracing_mark_write SPRAY_START/SPRAY_END line (matches real ftrace format)."""
    return ("  bash-100   [000] ....    {:.6f}: tracing_mark_write+0xb2/0x200 "
            "<ffffffff81167302>: {}".format(ts_ns / 1e9, marker))


def bpftrace_line(ts_ns, pid, tid, comm, op, ptr, bytes_req, bytes_alloc,
                  call_site=DEFAULT_CALL_SITE_ALLOC):
    """A bpftrace comma-separated line."""
    return "{},{},{},{},{},{:016x},{},{},{}".format(
        ts_ns, pid, tid, comm, op, ptr, bytes_req, bytes_alloc, call_site)


def write_trace_file(path, lines):
    """Write raw trace lines to *path* (with trailing newlines)."""
    with open(path, "w") as handle:
        for line in lines:
            handle.write(line + "\n")


def csv_row(ts_ns, pid, tid, comm, op, ptr, bytes_req, bytes_alloc, call_site):
    return [ts_ns, pid, tid, comm, op, ptr, bytes_req, bytes_alloc, call_site]


def write_csv_file(path, rows):
    """Write rows in the trace2csv.py CSV layout."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_ns", "pid", "tid", "comm", "op", "ptr",
                         "bytes_req", "bytes_alloc", "call_site"])
        writer.writerows(rows)


def write_spray_markers(path, markers):
    """Write a spray_markers.json mapping 'run.log' -> {SPRAY_START/END: ns}."""
    import json
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as handle:
        json.dump(markers, handle, indent=2)
