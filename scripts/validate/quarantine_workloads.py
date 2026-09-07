#!/usr/bin/env python3
"""Quarantine retired churn normal workloads inside the raw tree.

Design decision 2026-09-07: the msg_msg/keyctl churn workloads share the exact
call_site+size signatures that heap-spray exploits use (CVE-2017-8824 spray ==
keyctl churn 1:1, recall 0%), so a one-class model trained on them learns
spray as "normal". They are retired from the normal collection defaults and
moved into a per-CVE quarantine attic — a near-attack control family, never
training data, kept on disk for control experiments.

Layout-preserving, reversible move (quarantine lives INSIDE the raw tree, at
the CVE level, so datasets/raw stays one self-contained unit):

  raw/<CVE>/normal/<workload>/run_XXX_*/  ->  raw/<CVE>/quarantine/<workload>/run_XXX_*/

Downstream safety (both skippers key on the "quarantine" path segment):
  * trace2csv never converts quarantine traces (no rebuild input);
  * build_pilot_dataset's manifest registry never lists quarantine runs;
  * even without a rebuild, the training harness holds churn runs out by
    run_id (msg_msg_*/keyctl segments) and reports churn_near_attack (G11).

Behaviour:
  * dry-run by default; --apply performs the move;
  * appends every move to raw/<CVE>/quarantine/quarantine_manifest.json
    (merged by destination path, so re-runs never duplicate entries);
  * idempotent: missing sources and already-quarantined workloads are skipped.

Usage:
  python3 scripts/validate/quarantine_workloads.py                    # preview
  python3 scripts/validate/quarantine_workloads.py --apply            # move
  python3 scripts/validate/quarantine_workloads.py --workloads msg_msg_256   # subset
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

DEFAULT_WORKLOADS = ["msg_msg_256", "msg_msg_2048", "keyctl"]
REASON = ("retired churn normal workload: allocation signature (call_site+size) "
          "matches heap-spray exploits; near-attack control family, never "
          "training data (design decision 2026-09-07)")


def count_runs(workload_dir):
    return sum(1 for child in workload_dir.iterdir()
               if child.is_dir() and child.name.startswith("run_"))


def scan(raw_root, workloads):
    """Return the move plan: [(src_dir, dst_dir, cve, workload, runs)]."""
    plan = []
    if not raw_root.is_dir():
        return plan
    for cve_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        normal_dir = cve_dir / "normal"
        if not normal_dir.is_dir():
            continue
        for workload in workloads:
            src = normal_dir / workload
            if not src.is_dir():
                continue
            dst = cve_dir / "quarantine" / workload
            plan.append((src, dst, cve_dir.name, workload, count_runs(src)))
    return plan


def load_manifest(quarantine_dir):
    path = quarantine_dir / "quarantine_manifest.json"
    if path.is_file():
        try:
            return json.loads(path.read_text()), path
        except json.JSONDecodeError:
            print(f"[quarantine] WARN: unparseable manifest at {path}, starting fresh")
    return {"schema_version": 1, "reason": REASON, "moves": []}, path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default=config.DATASETS_RAW_DIR,
                        help="raw root: <CVE>/normal/<workload>/run_XXX_*/ "
                             f"(default {config.DATASETS_RAW_DIR})")
    parser.add_argument("--workloads", nargs="+", default=DEFAULT_WORKLOADS,
                        help="workload dir names to quarantine (default: %s)" % " ".join(DEFAULT_WORKLOADS))
    parser.add_argument("--apply", action="store_true",
                        help="perform the move (default: dry run)")
    args = parser.parse_args()

    raw_root = Path(args.raw)
    plan = scan(raw_root, args.workloads)
    if not plan:
        print("[quarantine] nothing to move: no matching workload dirs under "
              f"{raw_root}/<CVE>/normal/ for {args.workloads}")
        return 0

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[quarantine] {mode}: {len(plan)} workload dir(s) under {raw_root}")
    moved_by_cve = {}
    skipped = 0
    for src, dst, cve, workload, runs in plan:
        if dst.exists():
            print(f"[quarantine] SKIP (destination exists): {dst}")
            skipped += 1
            continue
        print(f"[quarantine] {src} -> {dst}  ({runs} runs)")
        if not args.apply:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved_by_cve.setdefault(cve, []).append({
            "cve": cve, "class": "normal", "workload": workload, "runs": runs,
            "from": str(src), "to": str(dst),
            "moved_at": datetime.now(timezone.utc).isoformat(),
        })

    if args.apply:
        for cve, entries in moved_by_cve.items():
            quarantine_dir = raw_root / cve / "quarantine"
            manifest, manifest_path = load_manifest(quarantine_dir)
            existing = {move["to"] for move in manifest["moves"]}
            for entry in entries:
                if entry["to"] not in existing:
                    manifest["moves"].append(entry)
            manifest["updated"] = datetime.now(timezone.utc).isoformat()
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"[quarantine] manifest -> {manifest_path}")
        print(f"[quarantine] moved {sum(len(v) for v in moved_by_cve.values())} "
              f"workload dir(s), skipped {skipped}")
    else:
        print(f"[quarantine] dry run only: {len(plan)} dir(s) would move "
              f"({skipped} skipped); re-run with --apply")

    print("\n[quarantine] next steps:")
    print("  1. rebuild processed so the npz files lose the churn sequences "
          "(trace2csv skips the quarantine segment automatically):")
    print("     python3 scripts/validate/build_pilot_dataset.py --raw datasets/raw --out datasets")
    print("     python3 scripts/preprocess/trace2tokens.py --raw datasets/raw --out datasets")
    print("  2. note: even BEFORE the rebuild, run_experiment/run_cve_split hold "
          "churn runs out of train/val/test by run_id and report them as the "
          "churn_near_attack control axis (G11)")
    print("  3. the moved data is a near-attack control family: score it for "
          "diagnostics, never fold it back into normal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
