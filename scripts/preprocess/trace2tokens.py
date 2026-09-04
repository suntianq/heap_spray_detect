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
Also writes event_fields.npz with the event-embedding field view used by
models/event_gru.py:
  event_fields:        (N, SEQ_LEN, 8) float32
    [op, size_bucket, call_site_hash, cpu_bucket, reclaim_flag,
     lifecycle, dt_bucket, dt_cont]
  event_field_labels / event_field_run_ids: same alignment as token_seq_*

Leak-safety: call_site profiles and dt thresholds are built from ALL normal
data (like csv2features computes per-window features from all data). The
train/val/test split is done later by run_experiment.py using run_ids, the
same logic as for feature sequences. The event-field view drops
behavior_type/frequency_rank entirely; its call_site channel is a stable hash
slot (crc32 mod 4096), so no global vocabulary of "unseen == rare" is built.
"""

import argparse
import json
import logging
import os
import re
import sys
import zlib
import multiprocessing
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

VOCAB_SIZE = 2 * 12 * 4 * 4 * 4 * 3 * 3  # 13824 (7-tuple: op, size, behavior, freq, dt, cpu, reclaim)
SEQ_LEN = 128
STRIDE = SEQ_LEN // 2  # 50% overlap

# Event-embedding field layout (see models/event_gru.py). The field matrix is
# (N, L, 8) float32: [op, size_bucket, call_site_hash, cpu_bucket, reclaim_flag,
# lifecycle, dt_bucket, dt_cont]. Channels 0-6 are integer indices; channel 7 is
# continuous log1p(dt_us)/log1p(1e9).
# behavior_type / frequency_rank are deliberately NOT included: they are computed
# from the global normal distribution and leak a "rare == anomalous" prior.
EVENT_FIELD_SIZE = 8
FIELD_CS_HASH_MOD = 4096  # call_site hash table size (stable across runs, no global vocab)

# Object lifecycle state (design section 5.2), recovered from the ptr tracking
# that csv2features.resolve_free_sizes already performs. This is the heap
# grooming / UAF fingerprint the op+size+dt fields cannot express: REUSE means
# "an ALLOC took back a ptr that was freed moments ago", which is exactly the
# ALLOC/ALLOC/FREE/FREE/REUSE cycle a groom drives.
LIFE_NEW = 0          # ALLOC of a ptr not recently freed
LIFE_REUSE = 1        # ALLOC reclaiming a recently-freed ptr
LIFE_FREE_SHORT = 2   # FREE of a short-lived object
LIFE_FREE_MEDIUM = 3  # FREE of a medium-lived object
LIFE_FREE_LONG = 4    # FREE of a long-lived object
LIFE_UNKNOWN = 5      # FREE whose allocation was never observed (unresolved)
LIFE_VOCAB = 6

# Lifetime bucket thresholds (microseconds) separating SHORT/MEDIUM/LONG.
# Fixed first-version values per design section 5.3; percentile calibration on
# normal data is the documented follow-up.
LIFETIME_THRESHOLDS_US = [100.0, 10_000.0]

SIZE_BUCKETS = tuple(config.SIZE_BUCKETS)
OVERFLOW_BUCKET = len(SIZE_BUCKETS)
CVE_RE = re.compile(r"CVE-\d{4}-\d+")
POC_RE = re.compile(r"(poc_cfh_[a-z0-9_]+?)(?=_run_|$)")

# dt_bucket thresholds (microseconds), calibrated from normal data
DT_THRESHOLDS_US = [2.0, 50.0, 1000.0]

# Inter-event delta buckets for the EVENT FIELD view (6 classes). Distinct from
# DT_THRESHOLDS_US, which stays at 4 classes because it feeds the legacy
# encode_token vocabulary. Boundaries fan out by decade from the sub-2us burst
# regime to >10ms, giving the dt head enough resolution to model the whole
# shape of the delta distribution. Measured on the CVE-first dataset, the
# separating band is NOT sub-2us (normal traffic is ~52% sub-2us too) but the
# 100us-1ms bucket: ~6% of normal deltas vs ~30% during spray.
DT_FIELD_THRESHOLDS_US = [2.0, 10.0, 100.0, 1000.0, 10_000.0]
DT_FIELD_VOCAB = len(DT_FIELD_THRESHOLDS_US) + 1  # 6

# dt continuation scale (must match models/event_gru.py): log1p(dt_us)/log1p(1e9).
# 1e9 us = ~16.7 min, a generous upper bound; dt=0 for a sequence's first event.
LOG_DT_NORM = float(np.log1p(1e9))


def hash_call_site(call_site):
    """Stable 0..FIELD_CS_HASH_MOD-1 slot for a call_site string.

    Uses zlib.crc32 (not Python's built-in hash()) so the slot is identical
    across processes, runs, and Python invocations. Unseen call_sites get a
    fresh slot instead of being flagged "rare" by a global vocabulary.
    """
    if not call_site:
        return 0
    return zlib.crc32(call_site.encode("utf-8")) % FIELD_CS_HASH_MOD


def dt_to_continuous(dt_ns):
    """Map an inter-event delta (ns) to a continuous scalar in ~[0, 1]."""
    dt_us = max(dt_ns, 0) / 1000.0
    return float(np.log1p(dt_us) / LOG_DT_NORM)


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
    """Build call_site → {behavior_type, frequency_rank} from normal events.

    Legacy interface: takes a list of event lists. Prefer
    build_call_site_profiles_from_stats for parallel workflows.
    """
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
    return build_call_site_profiles_from_stats(
        {cs: dict(sizes) for cs, sizes in cs_sizes.items()},
        dict(cs_count))


def build_call_site_profiles_from_stats(cs_sizes, cs_count):
    """Build profiles from pre-collected call_site size stats (parallel-friendly)."""
    profiles = {}
    for cs, sizes in cs_sizes.items():
        total = sum(sizes.values()) if isinstance(sizes, dict) else sum(sizes.values())
        if total == 0:
            bt = 3
        else:
            n_buckets = len([v for v in sizes.values() if v > 0]) if isinstance(sizes, dict) else 0
            top_frac = max(sizes.values()) / total if isinstance(sizes, dict) else 0
            if top_frac >= 0.9 and n_buckets <= 2:
                bt = 0  # mono
            elif n_buckets <= 3:
                bt = 1  # narrow
            else:
                bt = 2  # broad
        profiles[cs] = {"behavior_type": bt, "total_count": total}

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


def calibrate_dt_thresholds():
    """Return fixed dt bucket thresholds (microseconds).

    Validated stable across CVEs on kernel 4.15: 2/50/1000 us.
    """
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


def dt_to_field_bucket(dt_ns):
    """Map an inter-event delta to one of DT_FIELD_VOCAB classes (event view).

    Bucket 0 is the sub-2us spray burst regime; the remaining boundaries fan out
    by decade. Used as both an embedded input and a prediction target by
    models/event_gru.py, replacing the pure log1p regression whose gradient was
    dominated by the common millisecond-scale gaps.
    """
    dt_us = max(dt_ns, 0) / 1000.0
    for index, upper in enumerate(DT_FIELD_THRESHOLDS_US):
        if dt_us < upper:
            return index
    return DT_FIELD_VOCAB - 1


def lifecycle_state(event):
    """Object lifecycle class for one event (design section 5.2).

    Reads the annotations resolve_free_sizes() already writes onto each event:
    reclaim_from_free for ALLOC, object_lifetime_ns for FREE. An ALLOC that
    takes back a recently-freed ptr is REUSE; a FREE is classified by how long
    its object lived; a FREE whose ALLOC was never observed is UNKNOWN.
    """
    if event["op"] == "ALLOC":
        return LIFE_REUSE if event.get("reclaim_from_free", False) else LIFE_NEW
    lifetime_ns = event.get("object_lifetime_ns")
    if lifetime_ns is None:
        return LIFE_UNKNOWN
    lifetime_us = max(lifetime_ns, 0) / 1000.0
    if lifetime_us < LIFETIME_THRESHOLDS_US[0]:
        return LIFE_FREE_SHORT
    if lifetime_us < LIFETIME_THRESHOLDS_US[1]:
        return LIFE_FREE_MEDIUM
    return LIFE_FREE_LONG


def calibrate_lifetime_thresholds():
    """Return fixed lifetime bucket thresholds (microseconds).

    Mirrors calibrate_dt_thresholds(): the first version uses the fixed
    100us / 10ms split from design section 5.3. Percentile calibration on normal
    data is the documented follow-up once these prove workload-sensitive.
    """
    return list(LIFETIME_THRESHOLDS_US)


def encode_token(op, size_bucket, behavior_type, frequency_rank, dt_bucket, cpu_bucket, reclaim_flag):
    """Encode 7-tuple as single int token_id (0..3455).

    Fields:
      op:             0=ALLOC, 1=FREE                         (2)
      size_bucket:    0-11 (32,64,...,8192,gt)                (12)
      behavior_type:  0=mono, 1=narrow, 2=broad, 3=unknown    (4)
      frequency_rank: 0-3 (top5%, p80, p50, rare)            (4)
      dt_bucket:      0-3 (<2us, 2-50us, 50-1ms, >1ms)       (4)
      cpu_bucket:     0=dominant, 1=moderate, 2=spread       (3)
      reclaim_flag:   0=none, 1=same-site, 2=cross-site      (3)
    """
    return (op * 12 * 4 * 4 * 4 * 3 * 3 +
            size_bucket * 4 * 4 * 4 * 3 * 3 +
            behavior_type * 4 * 4 * 3 * 3 +
            frequency_rank * 4 * 3 * 3 +
            dt_bucket * 3 * 3 +
            cpu_bucket * 3 +
            reclaim_flag)


def _compute_cpu_buckets(events, window=32):
    """Compute per-event CPU concentration bucket.

    For each event, look at the last `window` events and compute what fraction
    are on the same CPU as the current event:
      0 = dominant (>70% same CPU — spray is CPU-pinned)
      1 = moderate (30-70%)
      2 = spread (<30% — normal load distributes across cores)

    Returns a list of cpu_bucket values, one per event.
    """
    n = len(events)
    buckets = [2] * n  # default: spread
    cpu_history = []  # list of CPU IDs in sliding window

    for i in range(n):
        cpu = events[i].get("cpu", 0)
        cpu_history.append(cpu)
        if len(cpu_history) > window:
            cpu_history = cpu_history[-window:]

        if len(cpu_history) <= 1:
            buckets[i] = 0  # first event: trivially dominant
            continue

        same_count = sum(1 for c in cpu_history if c == cpu)
        ratio = same_count / len(cpu_history)
        if ratio > 0.7:
            buckets[i] = 0  # dominant
        elif ratio > 0.3:
            buckets[i] = 1  # moderate
        else:
            buckets[i] = 2  # spread

    return buckets


def tokenize_run(events, profiles, spray_start=None, spray_end=None):
    """Tokenize a run's events into token_ids + per-event labels + fields.

    Returns (tokens, event_labels, event_fields) where:
      tokens:       list[int]      legacy 7-tuple token id (for token models)
      event_labels: list[int]      1 if in spray window, 0 otherwise
      event_fields: (N, 8) ndarray float32, the event-embedding field matrix
        [op, size_bucket, call_site_hash, cpu_bucket, reclaim_flag,
         lifecycle, dt_bucket, dt_cont].

    NOTE about the dt channels: channel 6 is the inter-event delta bucketed into
    DT_FIELD_VOCAB classes and channel 7 is the same delta as a continuous scalar
    in ~[0,1] (log1p(dt_us)/log1p(1e9)). The FIRST event of a run carries the
    longest dt bucket and dt_cont=0. dt is computed here as the delta to the
    previous event across the whole run, so a sequence's first position is reset
    by cut_event_sequences, not here.
    """
    resolve_free_sizes(events)
    cpu_buckets = _compute_cpu_buckets(events)
    tokens = []
    event_labels = []
    event_fields = []
    prev_ts = None

    for i, event in enumerate(events):
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
            delta_ns = event["timestamp_ns"] - prev_ts
            dt = dt_to_bucket(delta_ns)
            dt_field = dt_to_field_bucket(delta_ns)
            dt_cont = dt_to_continuous(delta_ns)
        else:
            dt = 3
            dt_field = DT_FIELD_VOCAB - 1
            dt_cont = 0.0
        prev_ts = event["timestamp_ns"]

        cpu_b = cpu_buckets[i]

        # reclaim_flag: from resolve_free_sizes annotations
        if event.get("reclaim_cross_site", False):
            reclaim_flag = 2  # cross-site reclaim (UAF fingerprint)
        elif event.get("reclaim_from_free", False):
            reclaim_flag = 1  # same-site reclaim
        else:
            reclaim_flag = 0  # not a reclaim

        tokens.append(encode_token(op, sb, bt, fr, dt, cpu_b, reclaim_flag))

        event_fields.append([op, sb, hash_call_site(cs), cpu_b, reclaim_flag,
                             lifecycle_state(event), dt_field, dt_cont])

        if spray_start is not None and spray_end is not None:
            event_labels.append(1 if spray_start <= event["timestamp_ns"] <= spray_end else 0)
        else:
            event_labels.append(0)

    return tokens, event_labels, np.asarray(event_fields, dtype=np.float32)


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


def cut_event_sequences(event_fields, event_labels, run_id, seq_len=SEQ_LEN, stride=STRIDE):
    """Cut the (N, 8) event field matrix into (n, seq_len, 8) chunks.

    Uses the same window/stride/labeling scheme as cut_sequences so the two
    views of the same run stay aligned. The first position of each sequence has
    its dt channels reset (there is no "previous event" across the boundary):
    dt_bucket becomes the longest bucket and dt_cont becomes 0, matching how
    tokenize_run treats the first event of a run.

    Returns (field_seqs, seq_labels, seq_ids).
    """
    n = event_fields.shape[0]
    if n < seq_len:
        return [], [], []

    field_seqs = []
    seq_labels = []
    seq_ids = []

    for i in range(0, n - seq_len + 1, stride):
        seq = event_fields[i:i + seq_len].copy()
        seq[0, 6] = DT_FIELD_VOCAB - 1  # no prior delta -> longest bucket
        seq[0, 7] = 0.0                 # no prior delta -> zero continuous dt
        lbls = event_labels[i:i + seq_len]

        if any(l == 1 for l in lbls) and not all(l == 1 for l in lbls):
            sl = -1
        elif any(l == 1 for l in lbls):
            sl = 1
        else:
            sl = 0

        field_seqs.append(seq)
        seq_labels.append(sl)
        seq_ids.append(run_id)

    return field_seqs, seq_labels, seq_ids


# ---------------------------------------------------------------------------
# Multiprocessing workers (top-level for picklability)
# ---------------------------------------------------------------------------

# Global profiles set once per worker process via Pool initializer
_worker_profiles = None


def _init_token_worker(profiles):
    """Initializer for tokenization workers — set global profiles once per process."""
    global _worker_profiles
    _worker_profiles = profiles


def _collect_stats_worker(trace_path):
    """Parse one trace.log and return call_site size stats (for profile building)."""
    events, _ = parse_trace_log(trace_path)
    cs_sizes = {}
    cs_count = {}
    for event in events:
        if event["op"] == "ALLOC":
            cs = event.get("call_site", "")
            size = event.get("bytes_alloc", 0) or event.get("bytes_req", 0)
            idx = bucket_index(size) if size else 0
            if cs not in cs_sizes:
                cs_sizes[cs] = {}
            cs_sizes[cs][idx] = cs_sizes[cs].get(idx, 0) + 1
            cs_count[cs] = cs_count.get(cs, 0) + 1
    return cs_sizes, cs_count


def _tokenize_run_worker(task):
    """Parse + tokenize one run. Returns dict with seqs/labels/ids or None.

    spray_window is resolved inside the worker (from manifest or trace markers)
    to avoid a sequential parsing bottleneck in the main process.
    """
    run_id, cls, trace_path, spray_window, run_dir, seq_len, stride = task
    events, markers = parse_trace_log(trace_path)
    if not events:
        return None
    # If manifest didn't provide a spray window, use markers from the trace
    if not spray_window and markers:
        spray_window = markers
    spray_start = spray_window.get("SPRAY_START") if spray_window else None
    spray_end = spray_window.get("SPRAY_END") if spray_window else None

    tokens, event_labels, event_fields = tokenize_run(events, _worker_profiles, spray_start, spray_end)
    seqs, seq_labels, seq_ids = cut_sequences(tokens, event_labels, run_id, seq_len, stride)
    field_seqs = field_seq_labels = field_seq_ids = None
    if event_fields.shape[0] >= seq_len:
        field_seqs, field_seq_labels, field_seq_ids = cut_event_sequences(
            event_fields, event_labels, run_id, seq_len, stride)

    return {
        "cls": cls,
        "seqs": seqs,
        "seq_labels": seq_labels,
        "seq_ids": seq_ids,
        "field_seqs": field_seqs,
        "field_seq_labels": field_seq_labels,
        "field_seq_ids": field_seq_ids,
        "has_seqs": len(seqs) > 0,
    }


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
    parser.add_argument("--workers", type=int,
                        default=max(1, int((os.cpu_count() or 1) * 0.8)),
                        help="number of parallel workers")
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

    normal_paths = [trace_path for _, cls, trace_path, _ in runs if cls == "normal"]

    # ---- 1. Collect call_site stats from normal runs (parallel) ----
    workers = min(args.workers, len(runs))
    if workers > 1 and len(normal_paths) > 1:
        log.info("Collecting call_site stats from %d normal runs with %d workers",
                 len(normal_paths), min(workers, len(normal_paths)))
        with multiprocessing.Pool(min(workers, len(normal_paths))) as pool:
            stats_results = pool.map(_collect_stats_worker, normal_paths)
    else:
        stats_results = [_collect_stats_worker(p) for p in normal_paths]

    # Merge stats
    merged_cs_sizes = defaultdict(Counter)
    merged_cs_count = Counter()
    for cs_sizes, cs_count in stats_results:
        for cs, sizes in cs_sizes.items():
            for idx, cnt in sizes.items():
                merged_cs_sizes[cs][idx] += cnt
        for cs, cnt in cs_count.items():
            merged_cs_count[cs] += cnt

    # ---- 2. Build call_site profiles (sequential, fast) ----
    profiles, freq_thresholds = build_call_site_profiles_from_stats(
        dict(merged_cs_sizes), dict(merged_cs_count))
    dt_thresholds = calibrate_dt_thresholds()
    lifetime_thresholds = calibrate_lifetime_thresholds()
    log.info("call_site profiles: %d entries (p50=%.0f p80=%.0f p95=%.0f)",
             len(profiles), freq_thresholds["p50"], freq_thresholds["p80"],
             freq_thresholds["p95"])

    # ---- 3. Tokenize + cut sequences for each run (parallel) ----
    # Prepare tasks: each worker parses + tokenizes one run.
    # spray_window extraction is done inside the worker (not here) to avoid
    # a sequential bottleneck of parsing 500+ normal trace.log files.
    tasks = []
    for run_id, cls, trace_path, run_dir in runs:
        spray_window = manifest_spray_window(run_dir)
        # If no manifest window, pass run_dir so the worker can extract markers
        # from the trace.log itself (parallelized, not sequential here).
        tasks.append((run_id, cls, trace_path, spray_window, run_dir,
                      args.seq_len, args.stride))

    if workers > 1 and len(tasks) > 1:
        log.info("Tokenizing %d runs with %d workers", len(tasks), workers)
        with multiprocessing.Pool(workers, initializer=_init_token_worker,
                                  initargs=(profiles,)) as pool:
            results = pool.map(_tokenize_run_worker, tasks)
    else:
        _init_token_worker(profiles)
        results = [_tokenize_run_worker(t) for t in tasks]

    attack_seqs, attack_labels, attack_ids = [], [], []
    normal_seqs, normal_labels, normal_ids = [], [], []
    attack_fields, attack_field_labels, attack_field_ids = [], [], []
    normal_fields, normal_field_labels, normal_field_ids = [], [], []
    for result in results:
        if result is None or not result["has_seqs"]:
            continue
        cls = result["cls"]
        seqs = result["seqs"]
        seq_lbls = result["seq_labels"]
        seq_rids = result["seq_ids"]
        if cls == "attack":
            attack_seqs.extend(seqs)
            attack_labels.extend(seq_lbls)
            attack_ids.extend(seq_rids)
        else:
            normal_seqs.extend(seqs)
            normal_labels.extend(seq_lbls)
            normal_ids.extend(seq_rids)

        fields = result.get("field_seqs")
        if fields is not None:
            f_lbls = result["field_seq_labels"]
            f_rids = result["field_seq_ids"]
            if cls == "attack":
                attack_fields.extend(fields)
                attack_field_labels.extend(f_lbls)
                attack_field_ids.extend(f_rids)
            else:
                normal_fields.extend(fields)
                normal_field_labels.extend(f_lbls)
                normal_field_ids.extend(f_rids)

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

    # ---- Event-embedding fields (for models/event_gru.py) ----
    def save_event_npz(path, field_seqs, labels, ids):
        if not field_seqs:
            log.warning("no event field sequences for %s, skipping", path)
            return
        np.savez_compressed(
            str(path),
            event_fields=np.asarray(field_seqs, dtype=np.float32),
            event_field_labels=np.asarray(labels, dtype=np.int8),
            event_field_run_ids=np.asarray(ids, dtype=object),
            schema_version=np.asarray(config.DATASET_SCHEMA_VERSION),
            seq_len=np.asarray(args.seq_len),
            field_size=np.asarray(EVENT_FIELD_SIZE),
            cs_hash_mod=np.asarray(FIELD_CS_HASH_MOD),
        )

    save_event_npz(proc_attack / "event_fields.npz", attack_fields, attack_field_labels, attack_field_ids)
    save_event_npz(proc_normal / "event_fields.npz", normal_fields, normal_field_labels, normal_field_ids)

    # Metadata
    meta = {
        "schema_version": config.DATASET_SCHEMA_VERSION,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "vocab_size": VOCAB_SIZE,
        "token_fields": ["op", "size_bucket", "behavior_type", "frequency_rank",
                         "dt_bucket", "cpu_bucket", "reclaim_flag"],
        "token_field_sizes": [2, 12, 4, 4, 4, 3, 3],
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
        "event_field_layout": {
            "channel": ["op", "size_bucket", "call_site_hash", "cpu_bucket",
                        "reclaim_flag", "lifecycle", "dt_bucket", "dt_cont"],
            "size": EVENT_FIELD_SIZE,
            "cs_hash_mod": FIELD_CS_HASH_MOD,
            "field_formats": ["int", "int", "int", "int", "int", "int", "int", "float"],
            "field_vocabs": [2, 12, FIELD_CS_HASH_MOD, 3, 3, LIFE_VOCAB,
                             DT_FIELD_VOCAB, None],
            "lifecycle_states": ["NEW", "REUSE", "FREE_SHORT", "FREE_MEDIUM",
                                 "FREE_LONG", "UNKNOWN"],
            "lifetime_thresholds_us": lifetime_thresholds,
            "dt_field_thresholds_us": list(DT_FIELD_THRESHOLDS_US),
        },
        "total_attack_event_sequences": len(attack_fields),
        "total_normal_event_sequences": len(normal_fields),
        "git_revision": snapshot_git_revision()[0],
    }
    write_json_atomic(str(out_dir / "processed" / "tokens_meta.json"), meta)

    log.info("done: token_sequences saved to %s/processed/{attack,normal}/", out_dir)


if __name__ == "__main__":
    main()
