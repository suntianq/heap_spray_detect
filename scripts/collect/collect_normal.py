import os
import sys
import time
import socket
import glob
import logging
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "KHeaps", "grader"))
from qemu_runner import QEMURunner
from pwn import ssh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("collect_normal")


def get_open_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


def resolve_key(cve_folder):
    glob_fmt = os.path.join(cve_folder, "img", "*.id_rsa")
    res = glob.glob(glob_fmt)
    if not res:
        raise FileNotFoundError(f"No SSH key found at {glob_fmt}")
    return res[0]


def run_one_normal(cve_folder, workload_name, run_id, output_dir, duration=30):
    ssh_port = get_open_port()
    key = resolve_key(cve_folder)

    qemu = QEMURunner(ssh_port, cve_folder, core_num=config.QEMU_CORES, mem_size=config.QEMU_MEM)
    qemu.launch()
    try:
        qemu.wait_ready(timeout=60)
    except RuntimeError:
        log.error("QEMU failed to start for %s run %d", workload_name, run_id)
        qemu.kill()
        return False

    try:
        conn = ssh(user=config.SSH_USER, host=config.SSH_HOST, port=ssh_port, keyfile=key)
    except Exception as e:
        log.error("SSH connect failed: %s", e)
        qemu.kill()
        return False

    trace_pid = None
    try:
        log.info("[%s] run %d: starting trace", workload_name, run_id)
        conn.upload(
            os.path.join(os.path.dirname(__file__), "trace_helpers", "trace_start.sh"),
            "/tmp/trace_start.sh",
        )
        conn.run("chmod +x /tmp/trace_start.sh")
        r = conn.run(f"bash /tmp/trace_start.sh {config.TRACE_BUFFER_SIZE_KB} {config.REMOTE_TRACE_PATH}")
        trace_pid_line = r.recvline(timeout=5)
        trace_pid = int(trace_pid_line.strip()) if trace_pid_line else None
        time.sleep(1)

        log.info("[%s] run %d: running workload", workload_name, run_id)
        if workload_name == "idle":
            conn.run(f"sleep {duration}")
        elif workload_name == "net_busy":
            r = conn.run(
                "phoronix-test-suite batch-benchmark apache-1.7.2",
                timeout=duration + 30,
            )
            time.sleep(min(duration, 20))
        elif workload_name == "msg_msg":
            conn.upload(
                os.path.join(os.path.dirname(__file__), "workloads", "workload_msg"),
                f"{config.REMOTE_WORKLOAD_DIR}/workload_msg",
            )
            conn.run(f"chmod +x {config.REMOTE_WORKLOAD_DIR}/workload_msg")
            conn.run(f"{config.REMOTE_WORKLOAD_DIR}/workload_msg {duration} &")
            time.sleep(duration)
        else:
            log.warning("Unknown workload: %s, doing idle", workload_name)
            conn.run(f"sleep {duration}")

        log.info("[%s] run %d: stopping trace", workload_name, run_id)
        conn.upload(
            os.path.join(os.path.dirname(__file__), "trace_helpers", "trace_stop.sh"),
            "/tmp/trace_stop.sh",
        )
        conn.run("chmod +x /tmp/trace_stop.sh")
        if trace_pid:
            conn.run(f"bash /tmp/trace_stop.sh {trace_pid} {config.REMOTE_TRACE_PATH}")
        else:
            conn.run(f"bash /tmp/trace_stop.sh '' {config.REMOTE_TRACE_PATH}")
        time.sleep(1)

        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"{workload_name}_run_{run_id:03d}.log")
        conn.download(config.REMOTE_TRACE_PATH, out_file)
        log.info("[%s] run %d: saved to %s", workload_name, run_id, out_file)
        return True

    except Exception as e:
        log.error("[%s] run %d: error: %s", workload_name, run_id, e)
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass
        qemu.kill()


def main():
    parser = argparse.ArgumentParser(description="Collect normal workload traces")
    parser.add_argument("-c", "--cve", default="CVE-2017-11176", help="CVE folder to use for kernel/image")
    parser.add_argument("-n", "--runs", type=int, default=config.NORMAL_RUNS)
    parser.add_argument("-d", "--duration", type=int, default=config.COLLECTION_DURATION)
    parser.add_argument("-o", "--output", default=config.NORMAL_RAW_DIR)
    parser.add_argument("-w", "--workloads", nargs="+", default=config.WORKLOADS)
    args = parser.parse_args()

    cve_folder = os.path.join(config.CVE_DIR, args.cve)
    if not os.path.isdir(cve_folder):
        log.error("CVE folder not found: %s", cve_folder)
        sys.exit(1)

    img_dir = os.path.join(cve_folder, "img")
    if not glob.glob(os.path.join(img_dir, "*.img")):
        log.info("No disk image found, running create-image.sh...")
        os.system(f"cd {img_dir} && bash create-image.sh")

    workload_bins = ["workload_msg"]
    local_workload_dir = os.path.join(os.path.dirname(__file__), "workloads")
    for wb in workload_bins:
        src = os.path.join(local_workload_dir, f"{wb}.c")
        dst = os.path.join(local_workload_dir, wb)
        if not os.path.exists(dst) and os.path.exists(src):
            log.info("Compiling %s...", wb)
            os.system(f"gcc -o {dst} {src} -lkeyutils -lpthread")

    results = {}
    for wl in args.workloads:
        success = 0
        for i in range(args.runs):
            ok = run_one_normal(cve_folder, wl, i, args.output, args.duration)
            if ok:
                success += 1
        results[wl] = {"total": args.runs, "success": success}
        log.info("%s: %d/%d successful", wl, success, args.runs)

    with open(os.path.join(args.output, "collection_report.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info("Done. Report saved.")


if __name__ == "__main__":
    main()
