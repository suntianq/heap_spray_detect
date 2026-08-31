#!/usr/bin/env python3
"""Collect normal traces with per-run manifests and owned QEMU processes."""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from collection_common import (first_image, get_open_port, launch_qemu, make_run_id,
                               parse_trace_overrun, scp_from_vm, scp_to_vm, sha256_file,
                               ssh_cmd, stop_qemu, validate_trace, wait_for_ssh,
                               write_manifest)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("collect_normal_v2")

KEY = Path(config.KHEAPS_DIR) / "scripts" / "create-image" / "stretch.id_rsa"
TRACE_START = Path(__file__).parent / "trace_helpers" / "trace_start.sh"
TRACE_STOP = Path(__file__).parent / "trace_helpers" / "trace_stop.sh"
WORKLOAD_MSG_SOURCE = Path(__file__).parent / "workloads" / "workload_msg.c"
WORKLOAD_KEY_SOURCE = Path(__file__).parent / "workloads" / "workload_key.c"
WORKLOAD_NET_SOURCE = Path(__file__).parent / "workloads" / "workload_net.c"
WORKLOAD_FS_SOURCE = Path(__file__).parent / "workloads" / "workload_fs.c"
WORKLOAD_FORK_SOURCE = Path(__file__).parent / "workloads" / "workload_fork.c"
WORKLOAD_MEM_SOURCE = Path(__file__).parent / "workloads" / "workload_mem.c"
# Vendored keyutils dev files (stretch image and host both lack them): the header
# and static lib come from libkeyutils-dev_1.6.3, uploaded offline (same as the
# attack collector). keyctl workload links them.
KEYUTILS_HEADER = Path(__file__).parent / "deps" / "keyutils.h"
KEYUTILS_STATIC_LIB = Path(__file__).parent / "deps" / "libkeyutils.a"
REMOTE_WORKLOAD_DIR = "/tmp/workloads"
REMOTE_TRACE_STATS = "/tmp/trace_stats.txt"


def build_workloads_in_vm(port, workloads):
    """Upload and compile workload sources on the guest.

    The stretch image ships gcc/make but its glibc is 2.24 while the host is
    2.34+, so a host-compiled binary fails at load on the guest with "version
    GLIBC_2.34 not found". Compiling in the guest (the same pattern the attack
    collector uses for PoCs) guarantees ABI compatibility. keyctl additionally
    needs the vendored keyutils dev files when the image lacks them.
    """
    require_remote_step(port, f"mkdir -p {REMOTE_WORKLOAD_DIR} && rm -f {REMOTE_WORKLOAD_DIR}/workload_*",
                        "prepare workload dir")
    steps = []
    if "msg_msg" in workloads:
        upload_required(port, WORKLOAD_MSG_SOURCE, f"{REMOTE_WORKLOAD_DIR}/workload_msg.c")
        steps.append("gcc -O2 -pthread -o workload_msg workload_msg.c")
    if "keyctl" in workloads:
        upload_required(port, WORKLOAD_KEY_SOURCE, f"{REMOTE_WORKLOAD_DIR}/workload_key.c")
        probe = require_remote_step(
            port,
            "test -f /usr/include/keyutils.h && test -f /usr/lib/x86_64-linux-gnu/libkeyutils.a "
            "&& echo OK || echo MISSING",
            "probe keyutils", timeout=20)
        if probe.strip() != "OK":
            upload_required(port, KEYUTILS_HEADER, "/usr/include/keyutils.h")
            upload_required(port, KEYUTILS_STATIC_LIB, "/usr/lib/x86_64-linux-gnu/libkeyutils.a")
        steps.append("gcc -O2 -pthread -I/usr/include -o workload_key workload_key.c "
                     "/usr/lib/x86_64-linux-gnu/libkeyutils.a")
    if "net_busy" in workloads:
        upload_required(port, WORKLOAD_NET_SOURCE, f"{REMOTE_WORKLOAD_DIR}/workload_net.c")
        steps.append("gcc -O2 -pthread -o workload_net workload_net.c")
    if "fs_io" in workloads:
        upload_required(port, WORKLOAD_FS_SOURCE, f"{REMOTE_WORKLOAD_DIR}/workload_fs.c")
        steps.append("gcc -O2 -o workload_fs workload_fs.c")
    if "fork_stress" in workloads:
        upload_required(port, WORKLOAD_FORK_SOURCE, f"{REMOTE_WORKLOAD_DIR}/workload_fork.c")
        steps.append("gcc -O2 -o workload_fork workload_fork.c")
    if "mem_pressure" in workloads:
        upload_required(port, WORKLOAD_MEM_SOURCE, f"{REMOTE_WORKLOAD_DIR}/workload_mem.c")
        steps.append("gcc -O2 -o workload_mem workload_mem.c")
    if not steps:
        return
    require_remote_step(port, f"cd {REMOTE_WORKLOAD_DIR} && {' && '.join(steps)} && echo BUILD_OK",
                        "build workloads", timeout=120)


def require_remote_step(port, command, name, timeout=60):
    rc, stdout, stderr = ssh_cmd(KEY, port, command, timeout=timeout)
    if rc != 0:
        raise RuntimeError(f"{name} failed rc={rc}: {stderr or stdout}")
    return stdout


def upload_required(port, local, remote):
    rc, error = scp_to_vm(KEY, port, local, remote)
    if rc != 0:
        raise RuntimeError(f"failed to upload {local}: {error}")


def run_one(cve, cve_folder, workload, index, duration, pre_seconds, post_seconds,
            output_root, workloads, msg_size=None):
    run_id = make_run_id(index)
    workload_label = f"{workload}_{msg_size}" if (workload == "msg_msg" and msg_size) else workload
    run_dir = output_root / cve / workload_label / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    trace_path = run_dir / "trace.log"
    stats_path = run_dir / "trace_stats.txt"
    qemu_log = run_dir / "qemu.log"
    manifest_path = run_dir / "manifest.json"
    ssh_port = get_open_port()
    process = None
    started = datetime.now(timezone.utc).isoformat()
    workload_sources = {
        "msg_msg": WORKLOAD_MSG_SOURCE,
        "keyctl": WORKLOAD_KEY_SOURCE,
        "net_busy": WORKLOAD_NET_SOURCE,
        "fs_io": WORKLOAD_FS_SOURCE,
        "fork_stress": WORKLOAD_FORK_SOURCE,
        "mem_pressure": WORKLOAD_MEM_SOURCE,
    }
    source = {w: workload_sources[w] for w in workloads if w in workload_sources}
    manifest = {
        "dataset_schema_version": config.DATASET_SCHEMA_VERSION,
        "run_id": f"{cve}/{workload_label}/{run_id}/trace",
        "class": "normal",
        "cve": cve,
        "workload": workload,
        "workload_label": workload_label,
        "msg_size": msg_size if workload == "msg_msg" else None,
        "cve_kernel": cve,
        "status": "collecting",
        "started_at": started,
        "ssh_port": ssh_port,
        "duration_seconds": duration,
        "pre_seconds": pre_seconds,
        "post_seconds": post_seconds,
        "kernel_sha256": sha256_file(cve_folder / "kernel" / "arch" / "x86" / "boot" / "bzImage"),
        "image_sha256": sha256_file(first_image(cve_folder)) if first_image(cve_folder) else None,
        "workload_sha256": sha256_file(source.get(workload)) if source.get(workload) else None,
    }
    write_manifest(manifest_path, manifest)

    try:
        process = launch_qemu(cve_folder, ssh_port, config.QEMU_CORES, config.QEMU_MEM, qemu_log)
        manifest["qemu_pid"] = process.pid
        ready, reason = wait_for_ssh(process, KEY, ssh_port)
        if not ready:
            raise RuntimeError(reason)

        # Compile workloads on the guest before tracing so the build is not part
        # of the recorded allocation activity (ABI: guest glibc is older).
        build_workloads_in_vm(ssh_port, workloads)
        upload_required(ssh_port, TRACE_START, "/tmp/trace_start.sh")
        upload_required(ssh_port, TRACE_STOP, "/tmp/trace_stop.sh")
        require_remote_step(ssh_port, "chmod +x /tmp/trace_start.sh /tmp/trace_stop.sh", "chmod trace helpers")
        output = require_remote_step(
            ssh_port,
            f"bash /tmp/trace_start.sh {config.TRACE_BUFFER_SIZE_KB} {config.REMOTE_TRACE_PATH}",
            "start trace", timeout=20)
        match = re.search(r"TRACE_PID=(\d+)", output)
        if not match:
            raise RuntimeError(f"trace helper did not return a PID: {output!r}")
        trace_pid = match.group(1)
        time.sleep(pre_seconds)

        if workload == "idle":
            require_remote_step(ssh_port, f"sleep {duration}", "idle workload", timeout=duration + 15)
        elif workload == "msg_msg":
            remote_binary = f"{REMOTE_WORKLOAD_DIR}/workload_msg"
            msg_args = str(msg_size) if msg_size else ""
            require_remote_step(ssh_port, f"{remote_binary} {duration} {msg_args}".rstrip(),
                                "msg_msg workload", timeout=duration + 20)
        elif workload == "keyctl":
            remote_binary = f"{REMOTE_WORKLOAD_DIR}/workload_key"
            require_remote_step(ssh_port, f"{remote_binary} {duration}",
                                "keyctl workload", timeout=duration + 20)
        elif workload == "net_busy":
            remote_binary = f"{REMOTE_WORKLOAD_DIR}/workload_net"
            require_remote_step(ssh_port, f"{remote_binary} {duration}",
                                "net_busy workload", timeout=duration + 20)
        elif workload == "fs_io":
            remote_binary = f"{REMOTE_WORKLOAD_DIR}/workload_fs"
            require_remote_step(ssh_port, f"{remote_binary} {duration}",
                                "fs_io workload", timeout=duration + 20)
        elif workload == "fork_stress":
            remote_binary = f"{REMOTE_WORKLOAD_DIR}/workload_fork"
            require_remote_step(ssh_port, f"{remote_binary} {duration}",
                                "fork_stress workload", timeout=duration + 20)
        elif workload == "mem_pressure":
            remote_binary = f"{REMOTE_WORKLOAD_DIR}/workload_mem"
            require_remote_step(ssh_port, f"{remote_binary} {duration}",
                                "mem_pressure workload", timeout=duration + 20)
        else:
            raise ValueError(f"unsupported workload: {workload}")
        time.sleep(post_seconds)

        require_remote_step(
            ssh_port,
            f"bash /tmp/trace_stop.sh {trace_pid} {config.REMOTE_TRACE_PATH} {REMOTE_TRACE_STATS}",
            "stop trace", timeout=20)
        rc, error = scp_from_vm(KEY, ssh_port, config.REMOTE_TRACE_PATH, trace_path)
        if rc != 0:
            raise RuntimeError(f"failed to download trace: {error}")
        rc, error = scp_from_vm(KEY, ssh_port, REMOTE_TRACE_STATS, stats_path)
        if rc != 0:
            raise RuntimeError(f"failed to download trace stats: {error}")

        valid, validation = validate_trace(trace_path, require_markers=False)
        overrun = parse_trace_overrun(stats_path)
        validation["trace_overrun"] = overrun
        if not valid:
            raise RuntimeError(validation["reason"])
        if overrun not in (0, None):
            raise RuntimeError(f"trace buffer overrun: {overrun}")
        manifest.update(validation)
        manifest["status"] = "valid"
        log.info("[%s] %s valid: %d events", workload, run_id, validation["event_count"])
        return True, manifest_path
    except Exception as error:
        manifest["status"] = "invalid"
        manifest["error"] = str(error)
        log.error("[%s] %s invalid: %s", workload, run_id, error)
        return False, manifest_path
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        if process is not None:
            manifest["qemu_returncode"] = process.poll()
        stop_qemu(process)
        write_manifest(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(description="Collect normal workload traces (schema v2)")
    parser.add_argument("-c", "--cve", default="CVE-2017-11176")
    parser.add_argument("-n", "--runs", type=int, default=20)
    parser.add_argument("-d", "--duration", type=int, default=30)
    parser.add_argument("--pre-seconds", type=int, default=3)
    parser.add_argument("--post-seconds", type=int, default=3)
    parser.add_argument("-o", "--output", default=os.path.join(config.DATA_DIR, "raw_v2", "normal"))
    parser.add_argument("-w", "--workloads", nargs="+", default=["idle", "msg_msg"],
                        choices=["idle", "msg_msg", "keyctl", "net_busy", "fs_io",
                                 "fork_stress", "mem_pressure"])
    parser.add_argument("--msg-sizes", nargs="+", type=int, default=[256, 2048],
                        help="msg_msg payload sizes to collect (bytes); one run set per size")
    args = parser.parse_args()
    if args.runs <= 0 or args.duration <= 0 or args.pre_seconds < 0 or args.post_seconds < 0:
        parser.error("invalid run/duration values")

    cve_folder = Path(config.CVE_DIR) / args.cve
    if not cve_folder.is_dir():
        parser.error(f"CVE folder not found: {cve_folder}")
    if not KEY.exists():
        parser.error(f"SSH key not found: {KEY}")
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    report = {"schema_version": 2, "cve_kernel": args.cve, "runs": []}
    for workload in args.workloads:
        if workload == "msg_msg":
            for msg_size in args.msg_sizes:
                log.info("=== %s / size=%d (%d runs) ===", workload, msg_size, args.runs)
                for index in range(args.runs):
                    ok, manifest = run_one(
                        args.cve, cve_folder, workload, index, args.duration,
                        args.pre_seconds, args.post_seconds, output_root,
                        args.workloads, msg_size=msg_size)
                    report["runs"].append({"manifest": str(manifest.relative_to(output_root)), "valid": ok})
        else:
            log.info("=== %s (%d runs) ===", workload, args.runs)
            for index in range(args.runs):
                ok, manifest = run_one(
                    args.cve, cve_folder, workload, index, args.duration,
                    args.pre_seconds, args.post_seconds, output_root,
                    args.workloads)
                report["runs"].append({"manifest": str(manifest.relative_to(output_root)), "valid": ok})

    with (output_root / "collection_report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    valid = sum(item["valid"] for item in report["runs"])
    log.info("Collection complete: %d/%d valid", valid, len(report["runs"]))


if __name__ == "__main__":
    main()
