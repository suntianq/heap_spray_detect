#!/usr/bin/env python3
"""Tokenize raw trace.log files into fixed-length event token sequences.

Phase 2 of the improvement plan: event-level predictive self-supervision.

Token = (op, size_bucket, behavior_type, frequency_rank, dt_bucket)
  op:             0=ALLOC, 1=FREE                      (2 values)
  size_bucket:    0-11 (32,64,...,8192,gt_8192)        (12 values)
  behavior_type:  0=mono, 1=narrow, 2=broad, 3=unknown (4 values)
  frequency_rank: 0-3 (top5%, p80, p50, rare/unseen)   (4 values)
  dt_bucket:      0:<2us, 1:2-50us, 2:50-1000us, 3:>1ms (4 values)

Vocab = 2*12*4*4*4 = 1536

Output: token_sequences.npz alongside features.npz in processed/{attack,normal}/
  token_seqs:          (N, SEQ_LEN) int32
  token_seq_run_ids:   (N,) object  (same run_id format as features.npz)
  token_seq_labels:    (N,) int8    (1=spray, 0=normal, -1=boundary-drop)
  schema_version:      scalar

Leak-safety: call_site profiles and dt thresholds are built from ALL normal
data (like csv2features computes per-window features from all data). The
train/val/test split is done later by run_experiment.py using run_ids, the
same logic as for feature sequences.
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from scripts.preprocess.trace2csv import (
    parse_ftrace_line, parse_bpftrace_line, detect_format, detect_markers,
)
from scripts.preprocess.csv2features import bucket_index, resolve_free_sizes
from scripts.common.io import write_json_atomic, snapshot_git_revision

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("trace2tokens")

VOCAB_SIZE = 2 * 12 * 4 * 4 * 4  # 1536
SEQ_LEN = 128
STRIDE = SEQ_LEN // 2  # 50% overlap

SIZE_BUCKETS = tuple(config.SIZE_BUCKETS)
OVERFLOW_BUCKET = len(SIZE_BUCKETS)
CVE_RE = re.compile(r"CVE-\d{4}-\d+")
POC_RE = re.compile(r"(poc_cfh_[a-z0-9_]+?)(?=_run_|$)")

# dt_bucket thresholds (microseconds), calibrated from normal data
DT_THRESHOLDS_US = [2.0, 50.0, 1000.0]


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------

def parse_trace_log(path):
    """Parse a trace.log into events + spray markers."""
    fmt = detect_format(path)
    if fmt == "unknown":
        return [], {}
    parser = parse_ftrace_line if fmt == "ftrace" else parse_bpftrace_line
    events = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = parser(line)
            if rec:
                events.append(rec)
    events.sort(key=lambda e: e["timestamp_ns"])
    markers = detect_markers(path) if fmt == "ftrace" else {}
    return events, markers


def manifest_spray_window(run_dir):
    """Read spray window from collector manifest (authoritative)."""
    manifest_path = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path) as f:
        m = json.load(f)
    if m.get("status") != "valid":
        return {}
    start = m.get("spray_start_ns")
    end = m.get("spray_end_ns")
    if start is None or end is None or end <= start:
        return {}
    return {"SPRAY_START": start, "SPRAY_END": end}


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def discover_runs(raw_dir):
    """Scan raw dir, return list of (run_id, class, trace_path, run_dir).

    Run_id format matches features.npz: CVE/<variant|workload>/run_xxx/trace
    (class segment stripped). class is attack/normal (baseline→normal).
    """
    runs = []
    for cve_dir in sorted(Path(raw_dir).iterdir()):
        if not cve_dir.is_dir() or not CVE_RE.fullmatch(cve_dir.name):
            continue
        for cls in ("attack", "normal", "baseline"):
            class_dir = cve_dir / cls
            if not class_dir.is_dir():
                continue
            for sub in sorted(class_dir.iterdir()):
                if not sub.is_dir():
                    continue
                for run_dir in sorted(sub.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    trace_path = run_dir / "trace.log"
                    if not trace_path.exists():
                        continue
                    # Skip invalid runs (manifest status check)
                    manifest_path = run_dir / "manifest.json"
                    if manifest_path.exists():
                        try:
                            m = json.loads(manifest_path.read_text())
                            if m.get("status") != "valid":
                                continue
                        except (json.JSONDecodeError, OSError):
                            pass
                    # Build run_id with class segment stripped
                    run_id = f"{cve_dir.name}/{sub.name}/{run_dir.name}/trace"
                    actual_class = "normal" if cls in ("normal", "baseline") else "attack"
                    runs.append((run_id, actual_class, str(trace_path), str(run_dir)))
    return runs


# ---------------------------------------------------------------------------
# Profile building (from normal data only)
# ---------------------------------------------------------------------------

def build_call_site_profiles(normal_events_all):
    """Build call_site → {behavior_type, frequency_rank} from normal events."""
    cs_sizes = defaultdict(Counter)
    cs_count = Counter()
    for events in normal_events_all:
        for event in events:
            if event["op"] == "ALLOC":
                cs = event.get("call_site", "")
                size = event.get("bytes_alloc", 0) or event.get("bytes_req", 0)
                idx = bucket_index(size) if size else 0
                cs_sizes[cs][idx] += 1
                cs_count[cs] += 1

    profiles = {}
    for cs, sizes in cs_sizes.items():
        total = sum(sizes.values())
        if total == 0:
            bt = 3
        else:
            n_buckets = len([v for v in sizes.values() if v > 0])
            top_frac = max(sizes.values()) / total
            if top_frac >= 0.9 and n_buckets <= 2:
                bt = 0  # mono
            elif n_buckets <= 3:
                bt = 1  # narrow
            else:
                bt = 2  # broad
        profiles[cs] = {"behavior_type": bt, "total_count": total}

    # Frequency rank: top 5% = rank 0, p80 = rank 1, p50 = rank 2, rest = rank 3
    counts = sorted(cs_count.values())
    if counts:
        p50 = float(np.percentile(counts, 50))
        p80 = float(np.percentile(counts, 80))
        p95 = float(np.percentile(counts, 95))
    else:
        p50 = p80 = p95 = 0.0

    for cs in profiles:
        cnt = profiles[cs]["total_count"]
        if cnt >= p95:
            profiles[cs]["frequency_rank"] = 0
        elif cnt >= p80:
            profiles[cs]["frequency_rank"] = 1
        elif cnt >= p50:
            profiles[cs]["frequency_rank"] = 2
        else:
            profiles[cs]["frequency_rank"] = 3

    return profiles, {"p50": p50, "p80": p80, "p95": p95}


def calibrate_dt_thresholds(normal_events_all):
    """Compute dt distribution from normal runs, return fixed thresholds."""
    dts_us = []
    for events in normal_events_all:
        prev_ts = None
        for event in events:
            if prev_ts is not None:
                dt_us = (event["timestamp_ns"] - prev_ts) / 1000.0
                if dt_us >= 0:
                    dts_us.append(dt_us)
            prev_ts = event["timestamp_ns"]
    if not dts_us:
        return list(DT_THRESHOLDS_US)
    d = np.array(dts_us)
    # Use fixed thresholds (2/50/1000 us) - validated stable across CVEs
    return list(DT_THRESHOLDS_US)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def dt_to_bucket(dt_ns):
    """Map time delta to bucket 0-3."""
    dt_us = dt_ns / 1000.0
    if dt_us < DT_THRESHOLDS_US[0]:
        return 0
    elif dt_us < DT_THRESHOLDS_US[1]:
        return 1
    elif dt_us < DT_THRESHOLDS_US[2]:
        return 2
    else:
        return 3


def encode_token(op, size_bucket, behavior_type, frequency_rank, dt_bucket):
    """Encode 5-tuple as single int token_id (0..1535)."""
    return (op * 12 * 4 * 4 * 4 +
            size_bucket * 4 * 4 * 4 +
            behavior_type * 4 * 4 +
            frequency_rank * 4 +
            dt_bucket)


def tokenize_run(events, profiles, spray_start=None, spray_end=None):
    """Tokenize a run's events into token_ids + per-event labels.

    Returns (tokens, event_labels) where tokens is list[int] and
    event_labels is list[int] (1 if in spray window, 0 otherwise).
    """
    resolve_free_sizes(events)
    tokens = []
    event_labels = []
    prev_ts = None

    for event in events:
        op = 0 if event["op"] == "ALLOC" else 1
        if event["op"] == "ALLOC":
            size = event.get("resolved_size") or event.get("bytes_alloc", 0) or event.get("bytes_req", 0)
        else:
            size = event.get("resolved_bytes_alloc") or 0
        sb = bucket_index(size) if size else 0

        cs = event.get("call_site", "")
        if cs in profiles:
            bt = profiles[cs]["behavior_type"]
            fr = profiles[cs]["frequency_rank"]
        else:
            bt = 3  # unknown
            fr = 3  # rare

        if prev_ts is not None:
            dt = dt_to_bucket(event["timestamp_ns"] - prev_ts)
        else:
            dt = 3
        prev_ts = event["timestamp_ns"]

        tokens.append(encode_token(op, sb, bt, fr, dt))

        if spray_start is not None and spray_end is not None:
            event_labels.append(1 if spray_start <= event["timestamp_ns"] <= spray_end else 0)
        else:
            event_labels.append(0)

    return tokens, event_labels


def cut_sequences(tokens, event_labels, run_id, seq_len=SEQ_LEN, stride=STRIDE):
    """Cut token list into fixed-length sequences with 50% overlap.

    Sequence label: 1 if any event in spray, 0 if none, -1 if boundary (mixed).
    """
    if len(tokens) < seq_len:
        return [], [], []

    seqs = []
    seq_labels = []
    seq_ids = []

    for i in range(0, len(tokens) - seq_len + 1, stride):
        seq = tokens[i:i + seq_len]
        lbls = event_labels[i:i + seq_len]

        if any(l == 1 for l in lbls) and not all(l == 1 for l in lbls):
            sl = -1  # boundary: partial overlap with spray
        elif any(l == 1 for l in lbls):
            sl = 1  # fully in spray
        else:
            sl = 0  # fully normal

        seqs.append(seq)
        seq_labels.append(sl)
        seq_ids.append(run_id)

    return seqs, seq_labels, seq_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tokenize raw trace.log into event token sequences (phase 2)")
    parser.add_argument("--raw", required=True,
                        help="raw directory (CVE-first: raw/<CVE>/{attack,normal,baseline}/...)")
    parser.add_argument("--out", required=True,
                        help="output root: writes processed/{attack,normal}/token_sequences.npz")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--force", action="store_true",
                        help="allow overwriting existing output")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    proc_attack = out_dir / "processed" / "attack"
    proc_normal = out_dir / "processed" / "normal"

    for d in (proc_attack, proc_normal):
        d.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(str(raw_dir))
    if not runs:
        parser.error(f"no valid runs found in {raw_dir}")
    log.info("discovered %d valid runs", len(runs))

    # ---- 1. Parse all trace.log files ----
    run_data = {}  # run_id -> (class, events, markers, spray_window)
    normal_events_all = []
    for run_id, cls, trace_path, run_dir in runs:
        events, markers = parse_trace_log(trace_path)
        if not events:
            log.warning("%s: no events", run_id)
            continue
        spray_window = markers or manifest_spray_window(run_dir)
        run_data[run_id] = (cls, events, spray_window)
        if cls == "normal":
            normal_events_all.append(events)

    log.info("parsed %d runs (%d normal, %d attack)",
             len(run_data),
             sum(1 for v in run_data.values() if v[0] == "normal"),
             sum(1 for v in run_data.values() if v[0] == "attack"))

    # ---- 2. Build call_site profiles from normal data ----
    profiles, freq_thresholds = build_call_site_profiles(normal_events_all)
    dt_thresholds = calibrate_dt_thresholds(normal_events_all)
    log.info("call_site profiles: %d entries (p50=%.0f p80=%.0f p95=%.0f)",
             len(profiles), freq_thresholds["p50"], freq_thresholds["p80"],
             freq_thresholds["p95"])

    # ---- 3. Tokenize + cut sequences for each run ----
    attack_seqs, attack_labels, attack_ids = [], [], []
    normal_seqs, normal_labels, normal_ids = [], [], []

    for run_id, (cls, events, spray_window) in sorted(run_data.items()):
        spray_start = spray_window.get("SPRAY_START")
        spray_end = spray_window.get("SPRAY_END")
        tokens, event_labels = tokenize_run(events, profiles, spray_start, spray_end)
        seqs, seq_lbls, seq_rids = cut_sequences(
            tokens, event_labels, run_id, args.seq_len, args.stride)
        if not seqs:
            log.warning("%s: too few events (%d) for seq_len=%d", run_id, len(tokens), args.seq_len)
            continue
        if cls == "attack":
            attack_seqs.extend(seqs)
            attack_labels.extend(seq_lbls)
            attack_ids.extend(seq_rids)
        else:
            normal_seqs.extend(seqs)
            normal_labels.extend(seq_lbls)
            normal_ids.extend(seq_rids)

    log.info("attack: %d sequences (spray=%d, normal=%d, boundary=%d)",
             len(attack_seqs),
             sum(1 for l in attack_labels if l == 1),
             sum(1 for l in attack_labels if l == 0),
             sum(1 for l in attack_labels if l == -1))
    log.info("normal: %d sequences", len(normal_seqs))

    # ---- 4. Save outputs ----
    def save_token_npz(path, seqs, labels, ids):
        if not seqs:
            log.warning("no sequences for %s, skipping", path)
            return
        np.savez_compressed(
            str(path),
            token_seqs=np.asarray(seqs, dtype=np.int32),
            token_seq_labels=np.asarray(labels, dtype=np.int8),
            token_seq_run_ids=np.asarray(ids, dtype=object),
            schema_version=np.asarray(config.DATASET_SCHEMA_VERSION),
            seq_len=np.asarray(args.seq_len),
            vocab_size=np.asarray(VOCAB_SIZE),
        )

    save_token_npz(proc_attack / "token_sequences.npz", attack_seqs, attack_labels, attack_ids)
    save_token_npz(proc_normal / "token_sequences.npz", normal_seqs, normal_labels, normal_ids)

    # Metadata
    meta = {
        "schema_version": config.DATASET_SCHEMA_VERSION,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "vocab_size": VOCAB_SIZE,
        "token_fields": ["op", "size_bucket", "behavior_type", "frequency_rank", "dt_bucket"],
        "token_field_sizes": [2, 12, 4, 4, 4],
        "call_site_profiles": {
            cs: {"behavior_type": p["behavior_type"],
                 "frequency_rank": p["frequency_rank"],
                 "total_count": p["total_count"]}
            for cs, p in profiles.items()
        },
        "dt_thresholds_us": dt_thresholds,
        "frequency_thresholds": freq_thresholds,
        "total_attack_sequences": len(attack_seqs),
        "total_normal_sequences": len(normal_seqs),
        "git_revision": snapshot_git_revision()[0],
    }
    write_json_atomic(str(out_dir / "processed" / "tokens_meta.json"), meta)

    log.info("done: token_sequences saved to %s/processed/{attack,normal}/", out_dir)


if __name__ == "__main__":
    main()
