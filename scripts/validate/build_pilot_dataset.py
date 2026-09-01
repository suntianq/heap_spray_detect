#!/usr/bin/env python3
"""Build the pilot-v2 processed datasets and run the acceptance gates.

Pipeline (after collection), on the CVE-first raw layout
(datasets/raw/<CVE>/{attack,normal,baseline}/<variant|workload>/run_XXX_*/):
  1. trace2csv on the whole raw root         -> .tmp/csv_all
  2. _split_class_csv: drop the class segment into class-stripped staging
     (.tmp/attack, .tmp/normal); rewrite spray_markers.json keys accordingly
  3. mark baseline PoCs as class=normal       -> runs_meta.json
  4. csv2features attack  -> processed/attack
  5. csv2features normal  -> processed/normal
  6. pilot_gates.py on both processed dirs
  7. build_dataset_manifest at the out root
  8. remove .tmp (no csv/ intermediate storage survives)

Why the class segment is stripped at the staging step: the on-disk raw layout
nests the class between CVE and variant/workload, but run_ids MUST keep the form
CVE/<variant|workload>/run_000_*/trace (cve_of / run_stratum / parse_group /
parse_run_metadata / G10 baseline detection all assume no class segment). Moving
the class subdir up in the staging tree drops that segment so every downstream
run_id consumer is byte-for-byte unaffected.

Baseline PoCs are collected by the attack collector (they run the exploit path
and may crash), but are NEGATIVE samples by design (7.2): their windows are
labelled 0 like any other normal workload. Path parsing would classify them as
attack (they carry a poc_cfh_* segment), so a --run-meta override flips them
back (build_run_meta below).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
TRACE2CSV = SCRIPTS / "preprocess" / "trace2csv.py"
CSV2FEATURES = SCRIPTS / "preprocess" / "csv2features.py"
PILOT_GATES = SCRIPTS / "validate" / "pilot_gates.py"

# Use the same interpreter that launched this script (e.g. .venv/bin/python3)
# so subprocesses have access to the same installed libraries.
PYTHON = sys.executable

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
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)
    print(f"cleared {path}")


def _split_class_csv(all_csv_dir, attack_csv_dir, normal_csv_dir):
    """Split trace2csv output into two class-stripped staging trees.

    all_csv_dir mirrors the raw layout: <CVE>/{attack,normal,baseline}/...
    (each leaf being a run dir with trace.csv + spray markers keyed by
    input-relative *.log paths). The class segment must be dropped before
    csv2features derives run_ids, and spray_markers.json keys must be rewritten
    to match the class-free run_id (run_id + ".log").

    Returns None; mutates the staging dirs. attack-class runs go to
    attack_csv_dir, normal+baseline to normal_csv_dir (baseline is flipped to
    normal later by build_run_meta).
    """
    for cve_dir in sorted(all_csv_dir.iterdir()):
        if not cve_dir.is_dir():
            continue
        for cls in ("attack", "normal", "baseline"):
            class_dir = cve_dir / cls
            if not class_dir.is_dir():
                continue
            dest_cve = (attack_csv_dir if cls == "attack" else normal_csv_dir) / cve_dir.name
            dest_cve.mkdir(parents=True, exist_ok=True)
            # move every <variant|workload> subdir up, dropping the class segment
            for sub in sorted(class_dir.iterdir()):
                dst = dest_cve / sub.name
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(sub), str(dst))
            shutil.rmtree(class_dir)
            print(f"class-split: {cve_dir.name}/{cls} -> {dest_cve.relative_to(dest_cve.parents[1])}")
        # remove empty CVE dir (no run dirs left after moves)
        if not any(cve_dir.iterdir()):
            cve_dir.rmdir()

    # rewrite spray_markers.json: keys are input-relative *.log paths like
    # <CVE>/<class>/<variant|workload>/run_XXX_*/trace.log -> <CVE>/<...>/trace.log
    # routed to the matching staging root's spray_markers.json.
    marker_path = all_csv_dir / "spray_markers.json"
    if marker_path.is_file():
        markers = json.loads(marker_path.read_text())
    else:
        markers = {}
    attack_markers, normal_markers = {}, {}
    for key, value in markers.items():
        parts = str(key).split("/")
        if len(parts) >= 2 and parts[1] in ("attack", "normal", "baseline"):
            cls = parts[1]
            new_key = "/".join(parts[:1] + parts[2:])
        else:  # no class segment (legacy layout) — route by variant heuristic
            cls = "attack" if any("poc_cfh_" in p for p in parts) else "normal"
            new_key = str(key)
        target = attack_markers if cls == "attack" else normal_markers
        target[new_key] = value
    for cls, target, dest in (("attack", attack_markers, attack_csv_dir),
                              ("normal", normal_markers, normal_csv_dir)):
        if target:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "spray_markers.json").write_text(json.dumps(target, indent=2))
            print(f"spray_markers rewritten: {cls} ({len(target)} keys) -> {dest}")
    # remove the merged marker file (it is split into the staging roots)
    if marker_path.is_file():
        marker_path.unlink()


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
    """Record every run's manifest in datasets/dataset_manifest.json.

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
                "class": "normal" if m.get("variant") == "poc_cfh_baseline" else m.get("class"),
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
    parser = argparse.ArgumentParser(
        description="Build pilot-v2 processed datasets + run gates from the CVE-first raw root")
    parser.add_argument("--raw", required=True,
                        help="datasets/raw root: <CVE>/{attack,normal,baseline}/<variant|workload>/<run>/")
    parser.add_argument("--out", required=True,
                        help="datasets root: writes processed/{attack,normal} + dataset_manifest.json; .tmp is transient")
    parser.add_argument("--skip-gates", action="store_true",
                        help="skip pilot_gates (used by unit tests on synthetic data that "
                             "cannot satisfy the data-quality gates G3/G5/G9)")
    parser.add_argument("--workers", type=int,
                        default=max(1, int((os.cpu_count() or 1) * 0.8)),
                        help="number of parallel workers for trace2csv/csv2features")
    args = parser.parse_args()

    out = Path(args.out)
    tmp = out / ".tmp"
    csv_all = tmp / "csv_all"
    csv_attack = tmp / "attack"
    csv_normal = tmp / "normal"
    proc_attack = out / "processed" / "attack"
    proc_normal = out / "processed" / "normal"
    # Clear derived dirs first: stale CSVs from earlier builds (moved/deleted
    # raw runs) must not leak into the rebuilt dataset.
    for d in (csv_all, csv_attack, csv_normal, proc_attack, proc_normal):
        clear_dir(d)

    # trace2csv on the whole raw root mirrors <CVE>/{attack,normal,baseline}/...
    run([PYTHON, TRACE2CSV, "-i", args.raw, "-o", csv_all, "--workers", args.workers])
    _split_class_csv(csv_all, csv_attack, csv_normal)

    # Baseline PoCs are negative samples; they merge into the normal CSV set.
    meta_path = build_run_meta(csv_normal)

    run([PYTHON, CSV2FEATURES, "-i", csv_attack, "-o", proc_attack,
         "-w", WINDOW_MS, "-s", STRIDE_MS, "--seq-len", SEQ_LEN,
         "--is-attack", "--markers", csv_attack / "spray_markers.json",
         "--sequence-label", "any", "--force", "--workers", args.workers])
    run([PYTHON, CSV2FEATURES, "-i", csv_normal, "-o", proc_normal,
         "-w", WINDOW_MS, "-s", STRIDE_MS, "--seq-len", SEQ_LEN,
         "--run-meta", meta_path, "--force", "--workers", args.workers])

    if not args.skip_gates:
        run([PYTHON, PILOT_GATES, "--attack", proc_attack, "--normal", proc_normal])
    else:
        print("SKIP: pilot_gates (--skip-gates)")

    build_dataset_manifest(out, [Path(args.raw)])

    # No csv/ intermediate storage survives the build.
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"removed transient staging {tmp}")


if __name__ == "__main__":
    main()
