#!/usr/bin/env python3
"""Collect attack traces (schema v2) with fail-closed validation.

Per IMPLEMENTATION_PLAN.md section 6:

  * markers are written by the PoC itself (libexp.c write_trace_marker) inside
    the real spray functions -- the host never writes SPRAY_START/SPRAY_END;
  * trace_pipe is streamed to the host so pre-crash events survive an expected
    VM crash (no post-mortem scp for the main trace);
  * each run gets a unique run_uuid directory with a full manifest;
  * only the owned QEMU process group is terminated (no `killall`);
  * any validation failure marks the run invalid, keeps its files for triage,
    and never overwrites an existing run.
"""

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from collection_common import (extract_marker_timestamps, first_image, get_open_port,
                               launch_qemu, make_run_id, parse_trace_overrun,
                               resolve_spray_window, scp_from_vm, scp_to_vm,
                               sha256_file, ssh_args, ssh_cmd, stop_qemu,
                               trace_bounds, validate_markers, validate_trace,
                               wait_for_ssh, write_manifest)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("collect_attack_v2")

KEY = Path(config.KHEAPS_DIR) / "scripts" / "create-image" / "stretch.id_rsa"
TRACE_START = Path(__file__).parent / "trace_helpers" / "trace_start.sh"
TRACE_STOP = Path(__file__).parent / "trace_helpers" / "trace_stop.sh"
KEYUTILS_HEADER = Path(__file__).parent / "deps" / "keyutils.h"
KEYUTILS_STATIC_LIB = Path(__file__).parent / "deps" / "libkeyutils.a"
REMOTE_TRACE_STATS = "/tmp/trace_stats.txt"
REMOTE_POC_DIR = "/tmp/poc_src"

POC_VARIANTS = ["poc_cfh_single_spray", "poc_cfh_combo"]


def require_remote_step(port, command, name, timeout=60):
    rc, stdout, stderr = ssh_cmd(KEY, port, command, timeout=timeout)
    if rc != 0:
        raise RuntimeError(f"{name} failed rc={rc}: {stderr or stdout}")
    return stdout


def upload_required(port, local, remote):
    rc, error = scp_to_vm(KEY, port, local, remote)
    if rc != 0:
        raise RuntimeError(f"failed to upload {local}: {error}")


def build_pocs_in_vm(port, cve_folder, variants):
    """Upload and build the requested PoC variants for a CVE. Fail-closed.

    The stretch image ships gcc/make and the libkeyutils runtime but NOT the dev
    package (header + static lib). Those come from vendored copies so a flaky
    Debian mirror cannot fail a run. Only the requested variant targets are
    built, so unrelated PoC variants cannot break the run either.
    """
    require_remote_step(port, "rm -rf /tmp/poc_src && mkdir -p /tmp/poc_src && echo OK",
                        "prepare poc dir")
    poc_folder = cve_folder / "poc"
    files = sorted(p for p in poc_folder.iterdir() if p.is_file())
    if not files:
        raise RuntimeError(f"no files in {poc_folder}")
    for source in files:
        upload_required(port, source, f"{REMOTE_POC_DIR}/{source.name}")
    probe = require_remote_step(
        port,
        "command -v gcc >/dev/null && command -v make >/dev/null && "
        "test -f /usr/include/keyutils.h && test -f /usr/lib/x86_64-linux-gnu/libkeyutils.a "
        "&& echo TOOLCHAIN_OK || echo MISSING",
        "probe build toolchain", timeout=20)
    if probe.strip() != "TOOLCHAIN_OK":
        upload_required(port, KEYUTILS_HEADER, "/usr/include/keyutils.h")
        upload_required(port, KEYUTILS_STATIC_LIB, "/usr/lib/x86_64-linux-gnu/libkeyutils.a")
        require_remote_step(port, "test -f /usr/include/keyutils.h && "
                                  "test -f /usr/lib/x86_64-linux-gnu/libkeyutils.a && echo DEPS_OK",
                            "verify keyutils deps", timeout=20)
    targets = " ".join(variants)
    require_remote_step(port, f"cd {REMOTE_POC_DIR} && make {targets} && echo BUILD_OK",
                        "build PoCs", timeout=120)
    require_remote_step(port, f"ls {REMOTE_POC_DIR}/{variants[0]}", "verify PoC binary")


def start_trace_host_stream(port):
    """Configure tracing on the guest in host-stream mode; return remote trace_pipe."""
    upload_required(port, TRACE_START, "/tmp/trace_start.sh")
    upload_required(port, TRACE_STOP, "/tmp/trace_stop.sh")
    require_remote_step(port, "chmod +x /tmp/trace_start.sh /tmp/trace_stop.sh", "chmod trace helpers")
    output = require_remote_step(
        port,
        f"bash /tmp/trace_start.sh {config.TRACE_BUFFER_SIZE_KB} {config.REMOTE_TRACE_PATH} host-stream",
        "start trace (host-stream)", timeout=20)
    match = re.search(r"TRACE_PIPE=(\S+)", output)
    if not match:
        raise RuntimeError(f"trace helper did not return TRACE_PIPE: {output!r}")
    return match.group(1)


def start_host_reader(port, trace_pipe, trace_path):
    """Stream the guest's trace_pipe to the host so pre-crash events persist."""
    handle = trace_path.open("wb")
    process = subprocess.Popen(
        ssh_args(KEY, port) + [f"cat {trace_pipe}"],
        stdout=handle, stderr=subprocess.DEVNULL,
    )
    process._trace_handle = handle
    return process


def stop_reader(reader):
    if reader is None:
        return
    if reader.poll() is None:
        reader.send_signal(signal.SIGTERM)
        try:
            reader.wait(timeout=5)
        except subprocess.TimeoutExpired:
            reader.send_signal(signal.SIGKILL)
            reader.wait(timeout=5)
    handle = getattr(reader, "_trace_handle", None)
    if handle:
        handle.close()


def run_one_attack(cve, cve_folder, variant, index, output_root, pre_seconds,
                   post_seconds, poc_timeout, minimum_events, expect_crash):
    run_uuid = make_run_id(index)
    run_dir = output_root / cve / variant / run_uuid
    run_dir.mkdir(parents=True, exist_ok=False)
    trace_path = run_dir / "trace.log"
    stats_path = run_dir / "trace_stats.txt"
    qemu_log = run_dir / "qemu.log"
    poc_stdout = run_dir / "poc.stdout"
    poc_stderr = run_dir / "poc.stderr"
    manifest_path = run_dir / "manifest.json"
    ssh_port = get_open_port()
    process = None
    reader = None
    started = datetime.now(timezone.utc).isoformat()
    manifest = {
        "dataset_version": "v2",
        "dataset_schema_version": config.DATASET_SCHEMA_VERSION,
        "run_uuid": run_uuid,
        "class": "attack",
        "cve": cve,
        "variant": variant,
        "workload": None,
        "expected_crash": expect_crash,
        "status": "collecting",
        "started_at": started,
        "ssh_port": ssh_port,
        "pre_seconds": pre_seconds,
        "post_seconds": post_seconds,
        "kernel_hash": sha256_file(cve_folder / "kernel" / "arch" / "x86" / "boot" / "bzImage"),
        "image_hash": sha256_file(first_image(cve_folder)) if first_image(cve_folder) else None,
        "poc_hash": sha256_file(cve_folder / "poc" / f"{variant}.c"),
        "qemu_pid": None,
        "trace_start_ns": None,
        "trace_end_ns": None,
        "spray_start_ns": None,
        "spray_end_ns": None,
        "marker_count": None,
        "marker_names": [],
        "marker_partial": False,
        "poc_exit_code": None,
        "poc_timeout_hit": False,
        "vm_crashed": False,
        "event_count": None,
        "trace_overrun": None,
    }
    write_manifest(manifest_path, manifest)

    try:
        process = launch_qemu(cve_folder, ssh_port, config.QEMU_CORES, config.QEMU_MEM, qemu_log)
        manifest["qemu_pid"] = process.pid
        ready, reason = wait_for_ssh(process, KEY, ssh_port)
        if not ready:
            raise RuntimeError(reason)

        build_pocs_in_vm(ssh_port, cve_folder, [variant])
        trace_pipe = start_trace_host_stream(ssh_port)
        reader = start_host_reader(ssh_port, trace_pipe, trace_path)
        time.sleep(pre_seconds)

        # Launch the PoC with stdout/stderr redirected on the host, so whatever it
        # printed before an expected VM crash is persisted locally.
        poc_command = f"cd {REMOTE_POC_DIR} && ./{variant}"
        poc_proc = subprocess.Popen(
            ssh_args(KEY, ssh_port) + [poc_command],
            stdout=poc_stdout.open("wb"), stderr=poc_stderr.open("wb"),
        )
        poc_timeout_hit = False
        try:
            poc_proc.wait(timeout=poc_timeout)
            poc_exit = poc_proc.returncode
        except subprocess.TimeoutExpired:
            poc_timeout_hit = True
            poc_exit = None
            poc_proc.kill()
            poc_proc.wait(timeout=10)
        manifest["poc_exit_code"] = poc_exit
        manifest["poc_timeout_hit"] = poc_timeout_hit

        time.sleep(post_seconds)

        # Determine VM state before touching the guest again. A kernel panic does
        # not always make QEMU exit (the guest can sit in the panic loop), so
        # probe SSH instead of relying on process.poll() alone.
        if process.poll() is not None:
            manifest["vm_crashed"] = True
            log.info("[%s/%s] %s VM exited during run", cve, variant, run_uuid)
        else:
            probe_rc, _, _ = ssh_cmd(KEY, ssh_port, "echo ALIVE", timeout=8)
            if probe_rc != 0:
                manifest["vm_crashed"] = True
                log.info("[%s/%s] %s VM unreachable (crashed/hung)", cve, variant, run_uuid)
            else:
                manifest["vm_crashed"] = False

        overrun = None
        if not manifest["vm_crashed"]:
            # Stop tracing on the guest, then let the host reader drain the ring
            # buffer before terminating it.
            require_remote_step(
                ssh_port,
                f"bash /tmp/trace_stop.sh '' {config.REMOTE_TRACE_PATH} {REMOTE_TRACE_STATS} host-stream",
                "stop trace", timeout=20)
            time.sleep(2)
            rc, error = scp_from_vm(KEY, ssh_port, REMOTE_TRACE_STATS, stats_path)
            if rc != 0:
                raise RuntimeError(f"failed to download trace stats: {error}")
            overrun = parse_trace_overrun(stats_path)
            manifest["trace_overrun"] = overrun
            if overrun not in (0, None):
                raise RuntimeError(f"trace buffer overrun: {overrun}")

        stop_reader(reader)
        reader = None

        # Fail-closed validation.
        valid, validation = validate_trace(trace_path, require_markers=False,
                                           minimum_events=minimum_events)
        if not valid:
            raise RuntimeError(validation["reason"])
        markers = extract_marker_timestamps(trace_path)
        first_ts, last_ts = trace_bounds(trace_path)
        resolved = resolve_spray_window(markers, expect_crash,
                                        manifest["vm_crashed"], last_ts)
        if resolved is None:
            _, marker_info = validate_markers(markers)
            raise RuntimeError(
                f"invalid markers: {marker_info['marker_names']}")
        spray_start_ns, spray_end_ns, marker_partial = resolved
        marker_info = {
            "marker_count": len(markers),
            "marker_names": [name for name, _ in markers],
            "spray_start_ns": spray_start_ns,
            "spray_end_ns": spray_end_ns,
        }
        if marker_partial:
            log.info("[%s/%s] %s partial markers (crash mid-spray), "
                     "spray_end=%s", cve, variant, run_uuid, spray_end_ns)
        if manifest["vm_crashed"] and not expect_crash:
            raise RuntimeError("unexpected VM crash")
        if manifest["poc_timeout_hit"]:
            raise RuntimeError("poc timed out")
        if not manifest["kernel_hash"] or not manifest["image_hash"] or not manifest["poc_hash"]:
            raise RuntimeError("missing required artifact hash")

        manifest.update({
            "trace_start_ns": first_ts,
            "trace_end_ns": last_ts,
            "event_count": validation["event_count"],
            "marker_partial": marker_partial,
        })
        manifest.update(marker_info)
        manifest["status"] = "valid"
        log.info("[%s/%s] %s valid: %d events, %d markers, crashed=%s",
                 cve, variant, run_uuid, validation["event_count"],
                 marker_info["marker_count"], manifest["vm_crashed"])
        return True, manifest_path
    except Exception as error:
        manifest["status"] = "invalid"
        manifest["error"] = str(error)
        log.error("[%s/%s] %s invalid: %s", cve, variant, run_uuid, error)
        return False, manifest_path
    finally:
        stop_reader(reader)
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        if process is not None:
            manifest["qemu_returncode"] = process.poll()
        stop_qemu(process)
        write_manifest(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(description="Collect attack traces (schema v2, fail-closed)")
    parser.add_argument("-c", "--cves", nargs="+", default=config.CVE_LIST)
    parser.add_argument("-v", "--variants", nargs="+", default=POC_VARIANTS)
    parser.add_argument("-n", "--runs", type=int, default=config.ATTACK_RUNS)
    parser.add_argument("--pre-seconds", type=int, default=3)
    parser.add_argument("--post-seconds", type=int, default=3)
    parser.add_argument("--poc-timeout", type=int, default=60,
                        help="max seconds to wait for a PoC to exit")
    parser.add_argument("--min-events", type=int, default=100,
                        help="minimum kmalloc/kfree event count for a valid run")
    parser.add_argument("--expect-crash", nargs="*", default=[],
                        help="CVEs whose PoCs are expected to crash the VM")
    parser.add_argument("-o", "--output", default=os.path.join(config.DATA_DIR, "raw_v2", "attack"))
    args = parser.parse_args()
    if args.runs <= 0 or args.pre_seconds < 0 or args.post_seconds < 0 or args.poc_timeout <= 0:
        parser.error("invalid run/duration values")

    if not KEY.exists():
        parser.error(f"SSH key not found: {KEY}")
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    report = {
        "schema_version": 2,
        "class": "attack",
        "expect_crash": args.expect_crash,
        "cves": {},
    }
    for cve in args.cves:
        cve_folder = Path(config.CVE_DIR) / cve
        if not cve_folder.is_dir():
            log.warning("CVE folder not found: %s", cve)
            continue
        report["cves"][cve] = {}
        for variant in args.variants:
            if not (cve_folder / "poc" / f"{variant}.c").is_file():
                log.warning("PoC variant not found: %s/%s", cve, variant)
                continue
            log.info("=== %s / %s (%d runs) ===", cve, variant, args.runs)
            outcomes = []
            for index in range(args.runs):
                ok, manifest = run_one_attack(
                    cve, cve_folder, variant, index, output_root, args.pre_seconds,
                    args.post_seconds, args.poc_timeout, args.min_events,
                    cve in args.expect_crash)
                outcomes.append({"manifest": str(manifest.relative_to(output_root)), "valid": ok})
            report["cves"][cve][variant] = {
                "total": len(outcomes),
                "valid": sum(item["valid"] for item in outcomes),
                "runs": outcomes,
            }
            log.info("[%s/%s] completed: %d/%d valid",
                     cve, variant, sum(item["valid"] for item in outcomes), len(outcomes))

    with (output_root / "collection_report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    total_valid = sum(v["valid"] for c in report["cves"].values() for v in c.values())
    total_runs = sum(v["total"] for c in report["cves"].values() for v in c.values())
    log.info("Attack collection complete: %d/%d valid runs", total_valid, total_runs)


if __name__ == "__main__":
    main()
