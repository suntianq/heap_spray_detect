#!/usr/bin/env python3
"""Build the pilot-v2 processed datasets and run the acceptance gates.

Pipeline (after collection):
  1. trace2csv on raw attack dir          -> csv/attack
  2. trace2csv on raw normal dir          -> csv/normal
  3. mark baseline PoCs as class=normal   -> runs_meta.json
  4. csv2features attack  -> processed/attack
  5. csv2features normal  -> processed/normal
  6. pilot_gates.py on both processed dirs

Baseline PoCs are collected by the attack collector (they run the exploit
path and may crash), but are NEGATIVE samples by design (7.2): their windows
are labelled 0 like any other normal workload. Path parsing would classify them
as attack (they live under a CVE dir), so a --run-meta override flips them back.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
TRACE2CSV = SCRIPTS / "preprocess" / "trace2csv.py"
CSV2FEATURES = SCRIPTS / "preprocess" / "csv2features.py"
PILOT_GATES = SCRIPTS / "validate" / "pilot_gates.py"

# csv2features v2 defaults (plan 5.5): 100ms window / 50ms stride / 32 seq.
WINDOW_MS = 100
STRIDE_MS = 50
SEQ_LEN = 32


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd))
    result = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True,
                            text=True, **kwargs)
    # Relay the step's output on success too: the build log is a deliverable
    # (final_v2_report parses the gate lines from it), and swallowing stdout on
    # success hid every [PASS]/[FAIL] line once gates started passing.
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip())
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {cmd[0]} rc={result.returncode}")
    return result.stdout


def clear_dir(path):
    """Remove derived artifacts so a rebuild cannot pick up stale files.

    trace2csv only writes new CSVs; a previous build's CSVs for runs that were
    later invalidated or moved would otherwise leak into processed/ and inflate
    the dataset with runs whose raw manifests are no longer valid.
    """
    if path.is_dir():
        for child in path.iterdir():
            if child.is_dir() and not child.is_symlink():
                import shutil
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)
    print(f"cleared {path}")


def build_run_meta(csv_normal):
    """Flip baseline PoCs (lives under a CVE dir) to class=normal."""
    overrides = {}
    for csv_path in sorted(csv_normal.rglob("*.csv")):
        rel = csv_path.relative_to(csv_normal).as_posix()
        if "/poc_cfh_baseline/" in "/" + rel:
            run_id = rel[:-4]
            overrides[run_id] = {"class": "normal", "variant": "poc_cfh_baseline",
                                 "workload": "poc_cfh_baseline"}
    meta_path = csv_normal / "runs_meta.json"
    meta_path.write_text(json.dumps(overrides, indent=2))
    print(f"run-meta overrides: {len(overrides)} baseline runs -> normal")
    return meta_path


def build_dataset_manifest(out, raw_dirs):
    """Record every run's manifest in datasets/<kind>/dataset_manifest.json.

    The template file describes the planned layout; this populates its run
    registry so any model metric can be traced to a concrete run (plan 4.2).
    Invalid runs are kept with their error reason, exactly as the collector
    left them.
    """
    manifest_path = out / "dataset_manifest.json"
    template = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    registry = {}
    for raw_dir in raw_dirs:
        for manifest_file in sorted(Path(raw_dir).rglob("manifest.json")):
            try:
                m = json.loads(manifest_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            # Normal collector stores a full run_id path; the attack collector
            # stores run_uuid + cve/variant. Build a unique key for either.
            rid = m.get("run_id") or f"{m.get('cve')}/{m.get('variant') or m.get('workload')}/{m.get('run_uuid')}/trace"
            if not rid or rid.endswith("None/trace"):
                continue
            registry[rid] = {
                "class": m.get("class"),
                "status": m.get("status"),
                "cve": m.get("cve"),
                "variant": m.get("variant"),
                "workload": m.get("workload"),
                "workload_label": m.get("workload_label"),
                "event_count": m.get("event_count"),
                "error": m.get("error"),
            }
    template["run_registry"] = dict(sorted(registry.items()))
    valid = sum(1 for r in registry.values() if r["status"] == "valid")
    template["status"] = "ready" if registry else template.get("status", "empty")
    template["updated"] = datetime.now(timezone.utc).isoformat()
    template["run_counts"] = {
        "total": len(registry),
        "valid": valid,
        "invalid": len(registry) - valid,
    }
    # Record the CVEs/variants actually collected. The pilot template carried a
    # `cve_variants` placeholder block (planned names replaced with the real
    # plan target slabs); the final-v2 template (schema v2, with sealed
    # dev/test CVE bookkeeping) does not. Build the block from the registry so
    # either template shape is populated correctly.
    collected = {}
    for cve in sorted({r.get("cve") for r in registry.values() if r.get("cve")}):
        variants = sorted({r.get("variant") for r in registry.values()
                           if r.get("cve") == cve and r.get("variant")})
        collected[cve] = variants
    try:
        from pilot_gates import TARGET_SLAB
    except ImportError:
        TARGET_SLAB = {}
    planned = [
        {"cve": cve, "variant": variant, "target_slab": TARGET_SLAB.get(cve)}
        for cve, variants in collected.items()
        for variant in variants
    ]
    template["cve_variants"] = {"planned": planned, "collected": collected}
    manifest_path.write_text(json.dumps(template, indent=2))
    print(f"dataset manifest: {len(registry)} runs ({valid} valid), status={template['status']}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Build pilot-v2 processed datasets + run gates")
    parser.add_argument("--attack-raw", required=True)
    parser.add_argument("--normal-raw", required=True,
                        help="raw dir of normal workloads (idle, msg_msg)")
    parser.add_argument("--baseline-raw", required=True,
                        help="raw dir of baseline PoCs (collected by collect_attack_stable)")
    parser.add_argument("--out", required=True, help="datasets/pilot-v2 dir")
    args = parser.parse_args()

    out = Path(args.out)
    csv_attack = out / "csv" / "attack"
    csv_normal = out / "csv" / "normal"
    proc_attack = out / "processed" / "attack"
    proc_normal = out / "processed" / "normal"
    # Clear derived dirs first: stale CSVs from earlier builds (moved/deleted
    # raw runs) must not leak into the rebuilt dataset.
    for d in (csv_attack, csv_normal, proc_attack, proc_normal):
        clear_dir(d)

    run(["python3", TRACE2CSV, "-i", args.attack_raw, "-o", csv_attack])
    # Baseline PoCs are negative samples; they merge into the normal CSV set.
    run(["python3", TRACE2CSV, "-i", args.normal_raw, "-o", csv_normal])
    run(["python3", TRACE2CSV, "-i", args.baseline_raw, "-o", csv_normal])

    meta_path = build_run_meta(csv_normal)

    run(["python3", CSV2FEATURES, "-i", csv_attack, "-o", proc_attack,
         "-w", WINDOW_MS, "-s", STRIDE_MS, "--seq-len", SEQ_LEN,
         "--is-attack", "--markers", csv_attack / "spray_markers.json",
         "--sequence-label", "any", "--force"])
    run(["python3", CSV2FEATURES, "-i", csv_normal, "-o", proc_normal,
         "-w", WINDOW_MS, "-s", STRIDE_MS, "--seq-len", SEQ_LEN,
         "--run-meta", meta_path, "--force"])

    run(["python3", PILOT_GATES, "--attack", proc_attack, "--normal", proc_normal])

    build_dataset_manifest(out, [args.attack_raw, args.normal_raw, args.baseline_raw])


if __name__ == "__main__":
    main()
