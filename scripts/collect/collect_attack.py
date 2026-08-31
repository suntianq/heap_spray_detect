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
log = logging.getLogger("collect_attack")


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


def run_one_attack(cve, poc_name, cve_folder, run_id, output_dir, max_runtime=60):
    ssh_port = get_open_port()
    key = resolve_key(cve_folder)

    poc_folder = os.path.join(cve_folder, "poc")
    poc_path = os.path.join(poc_folder, poc_name)

    if not os.path.exists(poc_path):
        log.error("PoC not found: %s", poc_path)
        return False

    qemu = QEMURunner(ssh_port, cve_folder, core_num=config.QEMU_CORES, mem_size=config.QEMU_MEM)
    qemu.launch()
    try:
        qemu.wait_ready(timeout=60)
    except RuntimeError:
        log.error("QEMU failed to start for %s/%s run %d", cve, poc_name, run_id)
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
        conn.upload(
            os.path.join(os.path.dirname(__file__), "trace_helpers", "trace_start.sh"),
            "/tmp/trace_start.sh",
        )
        conn.run("chmod +x /tmp/trace_start.sh")
        r = conn.run(f"bash /tmp/trace_start.sh {config.TRACE_BUFFER_SIZE_KB} {config.REMOTE_TRACE_PATH}")
        trace_pid_line = r.recvline(timeout=5)
        trace_pid = int(trace_pid_line.strip()) if trace_pid_line else None
        time.sleep(1)

        conn.upload(poc_path, config.REMOTE_EXP_PATH)
        conn.run(f"chmod +x {config.REMOTE_EXP_PATH}")

        log.info("[%s/%s] run %d: launching exploit", cve, poc_name, run_id)
        exp_start_ns = None
        try:
            r = conn.run(f"date +%s%N", timeout=5)
            exp_start_line = r.recvline(timeout=3)
            if exp_start_line:
                exp_start_ns = int(exp_start_line.strip())
        except Exception:
            pass

        exp_r = conn.process([config.REMOTE_EXP_PATH, str(run_id)])

        start = time.time()
        while not qemu.crashed and time.time() - start < max_runtime:
            try:
                out = exp_r.recv(timeout=0.5)
            except EOFError:
                break

        exp_status = qemu.status

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

        cve_output_dir = os.path.join(output_dir, cve)
        os.makedirs(cve_output_dir, exist_ok=True)
        out_file = os.path.join(cve_output_dir, f"{poc_name}_run_{run_id:03d}.log")
        conn.download(config.REMOTE_TRACE_PATH, out_file)

        meta = {
            "cve": cve,
            "poc": poc_name,
            "run_id": run_id,
            "exp_status": exp_status,
            "exp_start_ns": exp_start_ns,
        }
        meta_file = os.path.join(cve_output_dir, f"{poc_name}_run_{run_id:03d}_meta.json")
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        log.info("[%s/%s] run %d: status=%s, saved to %s", cve, poc_name, run_id, exp_status, out_file)
        return True

    except Exception as e:
        log.error("[%s/%s] run %d: error: %s", cve, poc_name, run_id, e)
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass
        qemu.kill()


def main():
    parser = argparse.ArgumentParser(description="Collect attack traces")
    parser.add_argument("-c", "--cves", nargs="+", default=config.CVE_LIST)
    parser.add_argument("-v", "--variants", nargs="+", default=config.POC_VARIANTS)
    parser.add_argument("-n", "--runs", type=int, default=config.ATTACK_RUNS)
    parser.add_argument("-o", "--output", default=config.ATTACK_RAW_DIR)
    args = parser.parse_args()

    with open(os.path.join(config.KHEAPS_DIR, "exploit_env", "setup.conf")) as f:
        import json as _json
        setup_conf = _json.load(f)

    results = {}
    for cve in args.cves:
        cve_folder = os.path.join(config.CVE_DIR, cve)
        if not os.path.isdir(cve_folder):
            log.warning("CVE folder not found: %s, skipping", cve)
            continue

        poc_folder = os.path.join(cve_folder, "poc")
        if not os.path.isdir(poc_folder):
            log.warning("PoC folder not found: %s, skipping", cve)
            continue

        log.info("Building PoCs for %s...", cve)
        os.system(f"cd {poc_folder} && make clean >/dev/null 2>&1 && make >/dev/null 2>&1")

        max_runtime = setup_conf.get(cve, {}).get("max_runtime", 60)

        results[cve] = {}
        for variant in args.variants:
            poc_path = os.path.join(poc_folder, variant)
            if not os.path.exists(poc_path):
                log.warning("PoC variant not found: %s, skipping", poc_path)
                continue

            success = 0
            for i in range(args.runs):
                ok = run_one_attack(cve, variant, cve_folder, i, args.output, max_runtime)
                if ok:
                    success += 1
            results[cve][variant] = {"total": args.runs, "success": success}
            log.info("%s/%s: %d/%d successful", cve, variant, success, args.runs)

    with open(os.path.join(args.output, "collection_report.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info("Done. Report saved.")


if __name__ == "__main__":
    main()
