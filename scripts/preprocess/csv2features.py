"""Build run-aware, fixed-cadence heap-allocation feature datasets.

Schema v2 stores raw features. Scaling (z-score, log1p) belongs in the training
pipeline after runs have been split; otherwise validation/test data leak into
normalisation.

Design notes (IMPLEMENTATION_PLAN.md section 5):
- Each trace is processed independently; sequences never cross runs.
- Fixed 100 ms windows on a 50 ms stride; empty windows are preserved.
- FREE sizes are resolved from a whole-trace ptr -> allocation state map
  (resolved_bytes_alloc, allocation_timestamp_ns, object_lifetime_ns).
- Window labels use the 50%-overlap policy (1 / 0 / -1 boundary).
- Sequence labels follow an endpoint or any policy; sequences containing a
  boundary window are explicitly ignored under boundary_policy="drop".
- Preprocessing output contains no NaN/Inf; identical input + config produces
  an identical output hash.
"""

import argparse
import csv
import json
import logging
import math
import multiprocessing
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.io import require_empty_dir, sha256_file, snapshot_git_revision, write_json_atomic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("csv2features")

SIZE_BUCKETS = tuple(config.SIZE_BUCKETS)
OVERFLOW_BUCKET = len(SIZE_BUCKETS)
BUCKET_LABELS = tuple(config.SIZE_BUCKET_LABELS)

# Default threshold: above this unresolved-FREE ratio a run is flagged as a data
# quality anomaly. Overridable via --free-unknown-threshold.
FREE_UNKNOWN_RATIO_THRESHOLD = 0.05

CVE_RE = re.compile(r"CVE-\d{4}-\d+")
POC_RE = re.compile(r"(poc_cfh_[a-z0-9_]+?)(?=_run_|$)")


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

def parse_run_metadata(run_id):
    """Derive structured run metadata from a run_id path.

    Recognises layouts such as:
      'CVE-2010-2959/poc_cfh_single_spray_run_001'  -> attack, cve, variant
      'CVE-2010-2959/poc_cfh_single_spray/run_000_x/trace'  -> attack, cve, variant
      'msg_msg_run_000'                              -> normal, workload
      'msg_msg_256/run_000_x/trace'                  -> normal, workload
      'CVE-2017-11176/idle/run_000_x/trace'          -> normal, cve, workload (idle)

    Returns a dict with keys class / cve / variant / workload.

    Class is decided by the presence of a ``poc_cfh_*`` variant segment, NOT by
    a CVE segment: the pilot collects matched normal controls under each CVE dir
    (idle / msg_msg / keyctl), so a CVE path alone must not imply an attack run.
    """
    parts = run_id.split("/")
    last = parts[-1]
    meta = {"class": None, "cve": None, "variant": None, "workload": None}
    cve = next((seg for seg in parts if CVE_RE.fullmatch(seg)), None)
    variant = None
    for seg in parts:
        match = POC_RE.search(seg)
        if match:
            variant = match.group(1)
            break
    if variant is not None:
        meta["class"] = "attack"
    else:
        meta["class"] = "normal"
    meta["cve"] = cve
    meta["variant"] = variant
    if meta["class"] == "normal":
        if len(parts) >= 2 and re.match(r"^run_\d+", parts[-2]):
            # v2 dir structure: <workload>/run_<idx>_<uuid>/trace
            meta["workload"] = parts[-3] if len(parts) >= 3 else parts[-2]
        else:
            match = re.match(r"^([a-z0-9_]+)_run_\d+", last)
            if match:
                meta["workload"] = match.group(1)
    return meta


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

def feature_names():
    names = []
    for prefix in ("alloc_count", "free_count", "alloc_rate", "free_rate"):
        names.extend(f"{prefix}_{label}" for label in BUCKET_LABELS)
    names.extend([
        "alloc_free_bucket_corr",
        "alloc_callsite_entropy",
        "free_callsite_entropy",
        "total_alloc",
        "total_free",
        "net_alloc_resolved",
        "free_unknown",
        "alloc_burst_1ms",
        "free_burst_1ms",
        "alloc_free_ratio",
        "n_active_tasks",
        "top_task_alloc_fraction",
        "task_alloc_entropy",
    ])
    names.extend(f"top_task_alloc_count_{label}" for label in BUCKET_LABELS)
    names.extend(f"top_task_free_count_{label}" for label in BUCKET_LABELS)
    names.extend([
        "top_task_alloc_burst_1ms",
        "top_task_alloc_free_ratio",
        "event_count",
        "is_empty",
        "time_since_previous_event_ms",
    ])
    # --- schema v3 extensions (11 dims) ---
    names.extend([
        "reclaim_count",
        "cross_site_reclaim_count",
        "reclaim_rate",
        "cpu_alloc_entropy",
        "top_cpu_alloc_fraction",
        "lifetime_median",
        "lifetime_p90",
        "short_lived_count",
        "long_lived_count",
        "burst_dominant_bucket_ratio",
        "burst_dominant_callsite_ratio",
    ])
    assert len(names) == config.FEAT_DIM, (len(names), config.FEAT_DIM)
    return names


def feature_groups():
    """Map feature names to semantic groups (plan 5.7 feature group)."""
    groups = {}
    for prefix in ("alloc_count", "free_count", "alloc_rate", "free_rate"):
        groups[prefix] = [f"{prefix}_{label}" for label in BUCKET_LABELS]
    groups["global"] = [
        "alloc_free_bucket_corr",
        "alloc_callsite_entropy",
        "free_callsite_entropy",
        "total_alloc",
        "total_free",
        "net_alloc_resolved",
        "free_unknown",
        "alloc_burst_1ms",
        "free_burst_1ms",
        "alloc_free_ratio",
        "n_active_tasks",
        "top_task_alloc_fraction",
        "task_alloc_entropy",
    ]
    groups["top_task_alloc"] = [f"top_task_alloc_count_{label}" for label in BUCKET_LABELS]
    groups["top_task_free"] = [f"top_task_free_count_{label}" for label in BUCKET_LABELS]
    groups["timing"] = [
        "top_task_alloc_burst_1ms",
        "top_task_alloc_free_ratio",
        "event_count",
        "is_empty",
        "time_since_previous_event_ms",
    ]
    groups["v3_reclaim"] = [
        "reclaim_count",
        "cross_site_reclaim_count",
        "reclaim_rate",
    ]
    groups["v3_cpu"] = [
        "cpu_alloc_entropy",
        "top_cpu_alloc_fraction",
    ]
    groups["v3_lifetime"] = [
        "lifetime_median",
        "lifetime_p90",
        "short_lived_count",
        "long_lived_count",
    ]
    groups["v3_burst"] = [
        "burst_dominant_bucket_ratio",
        "burst_dominant_callsite_ratio",
    ]
    flat = [name for group in groups.values() for name in group]
    assert sorted(flat) == sorted(feature_names()), "feature_groups must cover feature_names exactly"
    return groups


# ---------------------------------------------------------------------------
# Trace loading / FREE resolution
# ---------------------------------------------------------------------------

def load_events(csv_path):
    events = []
    with open(csv_path, "r", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                events.append({
                    "timestamp_ns": int(row["timestamp_ns"]),
                    # ftrace's common task id is a TID. We keep it as task_id and
                    # never rename it PID/TGID: TGID is not available from ftrace
                    # and is explicitly marked unknown (plan 5.5).
                    "task_id": int(row.get("tid", row["pid"])),
                    "pid": int(row["pid"]),
                    "cpu": int(row.get("cpu", 0)),
                    "comm": row.get("comm", ""),
                    "op": row["op"],
                    "ptr": row.get("ptr", ""),
                    "bytes_req": int(row.get("bytes_req", 0)),
                    "bytes_alloc": int(row.get("bytes_alloc", 0)),
                    "call_site": row.get("call_site", ""),
                })
            except (KeyError, TypeError, ValueError):
                continue
    events.sort(key=lambda event: event["timestamp_ns"])
    return events


def resolve_free_sizes(events, reclaim_window_ns=50_000_000):
    """Resolve FREE sizes once over the whole trace, preserving causality.

    Maintains a ptr -> (size, allocation_timestamp_ns) map. On FREE, annotates
    the event with:
      - size_resolved: bool
      - resolved_bytes_alloc: allocation size (bytes_alloc or bytes_req fallback)
      - allocation_timestamp_ns: when the ptr was allocated
      - object_lifetime_ns: free_ts - alloc_ts (>= 0)

    Additionally tracks reclaim events (schema v3): when an ALLOC reuses a ptr
    that was FREE'd within reclaim_window_ns, the ALLOC is annotated with
    reclaim_from_free=True and reclaim_cross_site (if the free and alloc
    call_sites differ). This captures the UAF free-then-reclaim fingerprint.

    Unresolvable frees (unknown ptr or double free) are counted and returned in
    a stats dict so the run can be flagged as a data-quality anomaly.
    """
    live = {}
    recent_free = {}
    stats = {"alloc_events": 0, "free_events": 0, "resolved": 0, "unresolved": 0,
             "reclaim_count": 0, "cross_site_reclaim_count": 0}
    for event in events:
        if event["op"] == "ALLOC":
            size = event["bytes_alloc"] or event["bytes_req"]
            event["size_resolved"] = True
            event["resolved_bytes_alloc"] = size
            event["resolved_size"] = size
            event["allocation_timestamp_ns"] = event["timestamp_ns"]
            event["object_lifetime_ns"] = None
            stats["alloc_events"] += 1
            ptr = event["ptr"]
            if ptr:
                entry = recent_free.pop(ptr, None)
                if entry is not None:
                    free_call_site, free_ts = entry
                    if event["timestamp_ns"] - free_ts <= reclaim_window_ns:
                        event["reclaim_from_free"] = True
                        stats["reclaim_count"] += 1
                        if free_call_site != event.get("call_site", ""):
                            event["reclaim_cross_site"] = True
                            stats["cross_site_reclaim_count"] += 1
                live[ptr] = (size, event["timestamp_ns"])
        elif event["op"] == "FREE":
            stats["free_events"] += 1
            ptr = event["ptr"]
            entry = live.pop(ptr, None)
            if entry is None:
                event["size_resolved"] = False
                event["resolved_bytes_alloc"] = None
                event["resolved_size"] = None
                event["allocation_timestamp_ns"] = None
                event["object_lifetime_ns"] = None
                stats["unresolved"] += 1
            else:
                size, alloc_ts = entry
                event["size_resolved"] = True
                event["resolved_bytes_alloc"] = size
                event["resolved_size"] = size
                event["allocation_timestamp_ns"] = alloc_ts
                event["object_lifetime_ns"] = max(0, event["timestamp_ns"] - alloc_ts)
                stats["resolved"] += 1
            if ptr:
                recent_free[ptr] = (event.get("call_site", ""), event["timestamp_ns"])
        else:
            event["size_resolved"] = False
    return stats


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def bucket_index(size):
    """Return a stable bucket index using the actual allocated size."""
    for index, upper in enumerate(SIZE_BUCKETS):
        if size <= upper:
            return index
    return OVERFLOW_BUCKET


def shannon_entropy(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = np.fromiter((count / total for count in counter.values() if count), dtype=np.float64)
    return float(-(probs * np.log2(probs)).sum())


def normalized_entropy(counter):
    if len(counter) <= 1:
        return 0.0
    return shannon_entropy(counter) / math.log2(len(counter))


def burst_1ms(timestamps, window_start_ns, window_end_ns):
    if not timestamps:
        return 0.0
    count = max(1, math.ceil((window_end_ns - window_start_ns) / 1_000_000))
    bins = np.zeros(count, dtype=np.int32)
    for timestamp in timestamps:
        index = min((timestamp - window_start_ns) // 1_000_000, len(bins) - 1)
        bins[int(max(index, 0))] += 1
    return float(bins.max())


def burst_subwindow_ratios(events, window_start_ns, window_end_ns):
    """Find the densest 1ms sub-window and return (dominant_bucket_ratio, dominant_callsite_ratio).

    Spray events are sub-millisecond but get diluted in a 100ms window. This
    finds the 1ms slice with the most events and computes the top1 size-bucket
    and top1 call_site concentration within that slice -- catching spray
    concentration that the full-window aggregates average away.
    """
    if not events:
        return 0.0, 0.0
    sub_ms = 1_000_000
    n_subs = max(1, int((window_end_ns - window_start_ns) // sub_ms))
    sub_events = defaultdict(list)
    for event in events:
        sub_idx = min(int((event["timestamp_ns"] - window_start_ns) // sub_ms), n_subs - 1)
        if sub_idx < 0:
            sub_idx = 0
        sub_events[sub_idx].append(event)
    if not sub_events:
        return 0.0, 0.0
    densest_idx = max(sub_events, key=lambda k: len(sub_events[k]))
    densest = sub_events[densest_idx]
    if not densest:
        return 0.0, 0.0
    bucket_counts = Counter()
    callsite_counts = Counter()
    for event in densest:
        if event["op"] == "ALLOC":
            bucket_counts[bucket_index(event["resolved_size"] or 0)] += 1
            callsite_counts[event["call_site"]] += 1
    total = sum(bucket_counts.values())
    if total == 0:
        return 0.0, 0.0
    dominant_bucket_ratio = bucket_counts.most_common(1)[0][1] / total if bucket_counts else 0.0
    dominant_callsite_ratio = callsite_counts.most_common(1)[0][1] / total if callsite_counts else 0.0
    return float(dominant_bucket_ratio), float(dominant_callsite_ratio)


def extract_features_from_events(events, window_ms, window_start_ns, previous_event_ns=None):
    bucket_count = len(BUCKET_LABELS)
    alloc_counts = np.zeros(bucket_count, dtype=np.float64)
    free_counts = np.zeros(bucket_count, dtype=np.float64)
    alloc_call_sites = Counter()
    free_call_sites = Counter()
    task_alloc_counts = defaultdict(lambda: np.zeros(bucket_count, dtype=np.float64))
    task_free_counts = defaultdict(lambda: np.zeros(bucket_count, dtype=np.float64))
    task_alloc_times = defaultdict(list)
    task_comms = defaultdict(Counter)
    alloc_times, free_times = [], []
    unknown_frees = 0
    reclaim_count = 0
    cross_site_reclaim_count = 0
    cpu_alloc_counts = Counter()
    lifetimes_us = []

    for event in events:
        task_id = event["task_id"]
        task_comms[task_id][event["comm"]] += 1
        if event["op"] == "ALLOC":
            index = bucket_index(event["resolved_size"] or 0)
            alloc_counts[index] += 1
            task_alloc_counts[task_id][index] += 1
            alloc_call_sites[event["call_site"]] += 1
            alloc_times.append(event["timestamp_ns"])
            task_alloc_times[task_id].append(event["timestamp_ns"])
            cpu_alloc_counts[event.get("cpu", 0)] += 1
            if event.get("reclaim_from_free", False):
                reclaim_count += 1
                if event.get("reclaim_cross_site", False):
                    cross_site_reclaim_count += 1
        elif event["op"] == "FREE":
            free_times.append(event["timestamp_ns"])
            free_call_sites[event["call_site"]] += 1
            if not event.get("size_resolved", False) or event["resolved_bytes_alloc"] is None:
                unknown_frees += 1
            else:
                index = bucket_index(event["resolved_bytes_alloc"])
                free_counts[index] += 1
                task_free_counts[task_id][index] += 1
            lt_ns = event.get("object_lifetime_ns")
            if lt_ns is not None and lt_ns >= 0:
                lifetimes_us.append(lt_ns / 1_000.0)

    total_alloc = len(alloc_times)
    total_free = len(free_times)
    resolved_free = int(free_counts.sum())
    duration_ms = float(window_ms)
    alloc_rate = alloc_counts / duration_ms
    free_rate = free_counts / duration_ms

    corr = 0.0
    if alloc_counts.std() > 0 and free_counts.std() > 0:
        corr = float(np.corrcoef(alloc_counts, free_counts)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0

    task_totals = Counter({task: int(counts.sum()) for task, counts in task_alloc_counts.items()})
    top_task = task_totals.most_common(1)[0][0] if task_totals else None
    top_comm = task_comms[top_task].most_common(1)[0][0] if top_task is not None else ""
    top_alloc = task_alloc_counts[top_task] if top_task is not None else np.zeros(bucket_count)
    top_free = task_free_counts[top_task] if top_task is not None else np.zeros(bucket_count)
    top_alloc_total = int(top_alloc.sum())
    top_free_total = int(top_free.sum())
    window_end_ns = window_start_ns + int(window_ms * 1_000_000)

    if previous_event_ns is None:
        since_previous_ms = duration_ms
    else:
        since_previous_ms = min(max((window_start_ns - previous_event_ns) / 1e6, 0.0), 60_000.0)

    cpu_entropy = normalized_entropy(cpu_alloc_counts)
    top_cpu_fraction = (cpu_alloc_counts.most_common(1)[0][1] / max(total_alloc, 1)
                        if cpu_alloc_counts else 0.0)

    if lifetimes_us:
        lt_arr = np.array(lifetimes_us, dtype=np.float64)
        lifetime_median = float(np.median(lt_arr)) / 1000.0
        lifetime_p90 = float(np.percentile(lt_arr, 90)) / 1000.0
        short_lived = int(np.sum(lt_arr < 1000.0))
    else:
        lifetime_median = 0.0
        lifetime_p90 = 0.0
        short_lived = 0
    long_lived = max(0, total_alloc - total_free)

    burst_bucket_ratio, burst_callsite_ratio = burst_subwindow_ratios(
        events, window_start_ns, window_end_ns)

    values = []
    values.extend(alloc_counts)
    values.extend(free_counts)
    values.extend(alloc_rate)
    values.extend(free_rate)
    values.extend([
        corr,
        normalized_entropy(alloc_call_sites),
        normalized_entropy(free_call_sites),
        float(total_alloc),
        float(total_free),
        float(total_alloc - resolved_free),
        float(unknown_frees),
        burst_1ms(alloc_times, window_start_ns, window_end_ns),
        burst_1ms(free_times, window_start_ns, window_end_ns),
        min(total_alloc / max(total_free, 1), 100.0),
        float(len(task_totals)),
        top_alloc_total / max(total_alloc, 1),
        normalized_entropy(task_totals),
    ])
    values.extend(top_alloc)
    values.extend(top_free)
    values.extend([
        burst_1ms(task_alloc_times.get(top_task, []), window_start_ns, window_end_ns),
        min(top_alloc_total / max(top_free_total, 1), 100.0),
        float(len(events)),
        float(not events),
        float(since_previous_ms),
    ])
    # --- schema v3 extensions (11 dims) ---
    values.extend([
        float(reclaim_count),
        float(cross_site_reclaim_count),
        reclaim_count / max(total_alloc, 1),
        cpu_entropy,
        top_cpu_fraction,
        lifetime_median,
        lifetime_p90,
        float(short_lived),
        float(long_lived),
        burst_bucket_ratio,
        burst_callsite_ratio,
    ])
    result = sanitize_feature_vector(values)
    assert len(result) == config.FEAT_DIM, (len(result), config.FEAT_DIM)
    return result, top_task, top_comm


def sanitize_feature_vector(values):
    """Guarantee the feature vector contains no NaN/Inf (plan 5.8)."""
    array = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(array, nan=0.0, posinf=1e6, neginf=-1e6)


# ---------------------------------------------------------------------------
# Window / sequence labelling
# ---------------------------------------------------------------------------

def window_label(window_start_ns, window_end_ns, is_attack, spray_start_ns, spray_end_ns):
    """Label a window against the spray interval (plan 5.6).

    A window is attack (1) if it overlaps the spray by at least half of EITHER
    the window or the spray span. The pilot PoCs spray in 0.1-175 ms against a
    100 ms window, so a single ratio breaks at one end: a window-length ratio
    leaves sub-50 ms sprays with no label-1 window (attack invisible to
    training), while a spray-length ratio leaves multi-window sprays with none
    (no window covers half a 500 ms spray). The union is duration-agnostic:
    short sprays are captured by the window that contains them, long sprays by
    every window they fill. Windows overlapping by less than half of both
    intervals stay -1 (boundary), so sequences can drop them under the policy.
    """
    if not is_attack:
        return 0
    if spray_start_ns is None or spray_end_ns is None or spray_end_ns <= spray_start_ns:
        return -1
    overlap = max(0, min(window_end_ns, spray_end_ns) - max(window_start_ns, spray_start_ns))
    if overlap == 0:
        return 0
    window_len = window_end_ns - window_start_ns
    spray_len = spray_end_ns - spray_start_ns
    if overlap / window_len >= 0.5 or overlap / spray_len >= 0.5:
        return 1
    return -1


def label_sequence(window_labels, policy, boundary_policy="drop"):
    """Sequence label under an endpoint/any policy.

    boundary_policy="drop" (v2 default): any boundary window (-1) in the sequence
    makes the whole sequence -1, so it is explicitly ignored downstream rather
    than automatically treated as normal or attack (plan 5.6).
    """
    if boundary_policy == "drop" and np.any(window_labels < 0):
        return -1
    if policy == "endpoint":
        return int(window_labels[-1])
    if policy == "any":
        if np.any(window_labels == 1):
            return 1
        return 0
    raise ValueError(f"unknown sequence label policy: {policy}")


# ---------------------------------------------------------------------------
# Per-trace processing
# ---------------------------------------------------------------------------

def process_csv(csv_path, window_ms, stride_ms, is_attack=False,
                spray_start_ns=None, spray_end_ns=None):
    events = load_events(csv_path)
    if not events:
        return (np.empty((0, config.FEAT_DIM), np.float32), np.empty(0, np.int8),
                np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, object),
                {"alloc_events": 0, "free_events": 0, "resolved": 0, "unresolved": 0}, 0.0)

    free_stats = resolve_free_sizes(events)
    window_ns = int(window_ms * 1e6)
    stride_ns = int(stride_ms * 1e6)
    first_ts, last_ts = events[0]["timestamp_ns"], events[-1]["timestamp_ns"]
    trace_start = (first_ts // stride_ns) * stride_ns
    trace_end = ((last_ts - trace_start) // stride_ns) * stride_ns + trace_start

    features, labels, starts, top_ids, top_comms = [], [], [], [], []
    left = right = previous_index = 0
    empty_windows = 0
    window_count = 0
    for start in range(trace_start, trace_end + 1, stride_ns):
        end = start + window_ns
        while left < len(events) and events[left]["timestamp_ns"] < start:
            left += 1
        right = max(right, left)
        while right < len(events) and events[right]["timestamp_ns"] < end:
            right += 1
        while previous_index < len(events) and events[previous_index]["timestamp_ns"] < start:
            previous_index += 1
        previous_ns = events[previous_index - 1]["timestamp_ns"] if previous_index else None
        window_events = events[left:right]
        if not window_events:
            empty_windows += 1
        values, top_task_id, top_comm = extract_features_from_events(
            window_events, window_ms, start, previous_ns)
        features.append(values)
        labels.append(window_label(start, end, is_attack, spray_start_ns, spray_end_ns))
        starts.append(start)
        top_ids.append(top_task_id if top_task_id is not None else -1)
        top_comms.append(top_comm)
        window_count += 1

    empty_ratio = empty_windows / max(window_count, 1)
    return (np.stack(features), np.asarray(labels, np.int8), np.asarray(starts, np.int64),
            np.asarray(top_ids, np.int64), np.asarray(top_comms, object),
            free_stats, empty_ratio)


def build_sequences(features, labels, starts, seq_len, run_id, label_policy="endpoint",
                    boundary_policy="drop", stride_ms=config.WINDOW_STRIDE_MS,
                    window_ms=config.WINDOW_SIZE_MS):
    """Build sequences within a single run; returns (seqs, seq_labels, groups, seq_starts, is_short)."""
    if len(features) < seq_len:
        empty = np.empty((0, seq_len, features.shape[1]), np.float32)
        return empty, np.empty(0, np.int8), np.empty(0, object), np.empty(0, np.int64), True

    # Every sequence must come from a single run (plan 5.1/5.8). This holds by
    # construction because features are per-file; assert the invariant.
    assert len({g for g in (run_id,)}) == 1
    # Start-to-start interval is (seq_len-1)*stride_ms; the covered time range is
    # window_ms + (seq_len-1)*stride_ms = 100 + 31*50 = 1650 ms (plan 5.2).
    expected_interval_ns = (seq_len - 1) * int(stride_ms * 1e6)
    assert int(starts[seq_len - 1] - starts[0]) == expected_interval_ns, (
        "sequence start interval mismatch", starts[seq_len - 1] - starts[0], expected_interval_ns)

    seqs, seq_labels, groups, seq_starts = [], [], [], []
    for index in range(len(features) - seq_len + 1):
        seqs.append(features[index:index + seq_len])
        seq_labels.append(label_sequence(labels[index:index + seq_len], label_policy, boundary_policy))
        groups.append(run_id)
        seq_starts.append(starts[index])
    return (np.stack(seqs), np.asarray(seq_labels, np.int8),
            np.asarray(groups, object), np.asarray(seq_starts, np.int64), False)


def load_markers(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def merge_run_meta(run_records, run_meta_path):
    """Merge an optional run-meta override file (run_id -> metadata dict) into records."""
    if not run_meta_path:
        return
    overrides = load_markers(run_meta_path)
    for record in run_records:
        if record["run_id"] not in overrides:
            continue
        record["metadata"].update(overrides[record["run_id"]])
        # Keep the top-level class/cve/variant/workload fields in sync with the
        # merged metadata (e.g. baseline runs parsed as attack must flip to normal).
        meta = record["metadata"]
        for key in ("class", "cve", "variant", "workload"):
            record[key] = meta.get(key)


# ---------------------------------------------------------------------------
# Multiprocessing worker (top-level for picklability)
# ---------------------------------------------------------------------------

def _process_run_worker(task):
    """Process a single CSV: extract features + build sequences.

    Returns a dict with all per-run arrays, or None if no parseable events.
    Must be top-level (not nested) to be picklable for multiprocessing.
    """
    (csv_path, window, stride, is_attack, spray_start, spray_end,
     seq_len, sequence_label, boundary_policy, short_run_policy, run_id) = task

    features, labels, starts, top_ids, top_comms, free_stats, empty_ratio = process_csv(
        csv_path, window, stride, is_attack, spray_start, spray_end)

    if is_attack and spray_start is None and short_run_policy == "__legacy__":
        labels[:] = 1

    if not len(features):
        return None

    seq, seq_label, seq_group, seq_start, is_short = build_sequences(
        features, labels, starts, seq_len, run_id,
        sequence_label, boundary_policy, stride, window)

    return {
        "features": features,
        "labels": labels,
        "starts": starts,
        "top_ids": top_ids,
        "top_comms": top_comms,
        "free_stats": free_stats,
        "empty_ratio": empty_ratio,
        "seq": seq,
        "seq_label": seq_label,
        "seq_group": seq_group,
        "seq_start": seq_start,
        "is_short": is_short,
        "run_id": run_id,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build run-aware feature dataset (schema v2)")
    parser.add_argument("-i", "--input", required=True, help="Input CSV directory")
    parser.add_argument("-o", "--output", required=True, help="Output dataset directory")
    parser.add_argument("-w", "--window", type=int, default=config.WINDOW_SIZE_MS)
    parser.add_argument("-s", "--stride", type=int, default=config.WINDOW_STRIDE_MS)
    parser.add_argument("--seq-len", type=int, default=config.SEQ_LEN)
    parser.add_argument("--is-attack", action="store_true")
    parser.add_argument("--markers", default=None,
                        help="spray_markers.json from trace2csv (defaults to <input>/spray_markers.json)")
    parser.add_argument("--sequence-label", choices=("endpoint", "any"), default="endpoint")
    parser.add_argument("--boundary-policy", choices=("drop", "keep"), default="drop",
                        help="drop: sequences containing a boundary window are labelled -1 and ignored")
    parser.add_argument("--short-run-policy", choices=("drop", "error"), default="drop",
                        help="drop: runs shorter than seq_len contribute windows only; error: abort")
    parser.add_argument("--free-unknown-threshold", type=float, default=FREE_UNKNOWN_RATIO_THRESHOLD,
                        help="unresolved-FREE ratio above which a run is flagged high_free_unknown")
    parser.add_argument("--run-meta", default=None,
                        help="optional JSON mapping run_id -> {cve,variant,workload,class} overrides")
    parser.add_argument("--legacy-all-attack", action="store_true",
                        help="Explicitly label marker-less attack traces as attack; never use for final data")
    parser.add_argument("--force", action="store_true",
                        help="Allow writing into a non-empty output directory")
    parser.add_argument("--normal-stats", default=None,
                        help="Deprecated compatibility option; schema v2 ignores this")
    parser.add_argument("--workers", type=int,
                        default=max(1, int((os.cpu_count() or 1) * 0.8)),
                        help="number of parallel workers for CSV processing")
    args = parser.parse_args()

    if args.normal_stats:
        log.warning("--normal-stats is ignored by schema v2; scaling is fit on training runs")
    if args.window <= 0 or args.stride <= 0 or args.seq_len <= 0:
        parser.error("window, stride and seq-len must be positive")

    input_root = Path(args.input).resolve()
    csv_files = sorted(input_root.rglob("*.csv"))
    if not csv_files:
        parser.error(f"no CSV files found in {input_root}")

    markers_path = args.markers or str(input_root / "spray_markers.json")
    markers = load_markers(markers_path)
    if args.is_attack and not markers and not args.legacy_all_attack:
        parser.error(f"attack schema v2 requires markers (--markers or <input>/spray_markers.json); "
                     "or explicit --legacy-all-attack for debugging only")

    output = Path(args.output)
    require_empty_dir(str(output), force=args.force)

    window_features, window_labels, window_groups, window_starts = [], [], [], []
    window_top_ids, window_top_comms = [], []
    sequences, sequence_labels, sequence_groups, sequence_starts = [], [], [], []
    run_records = []
    input_hashes = {}

    # Prepare tasks for parallel (or sequential) processing
    tasks = []
    for csv_path in csv_files:
        relative_csv = csv_path.relative_to(input_root).as_posix()
        run_id = relative_csv[:-4] if relative_csv.endswith(".csv") else relative_csv
        marker_key = run_id + ".log"
        run_markers = markers.get(marker_key, {})
        spray_start = run_markers.get("SPRAY_START")
        spray_end = run_markers.get("SPRAY_END")

        if args.is_attack:
            if (spray_start is None or spray_end is None) and not args.legacy_all_attack:
                raise ValueError(f"missing complete markers for attack run {run_id}")
            if (spray_start is not None and spray_end is not None
                    and spray_end <= spray_start) and not args.legacy_all_attack:
                raise ValueError(f"SPRAY_END <= SPRAY_START for attack run {run_id} (marker order error)")

        legacy_flag = "__legacy" if args.legacy_all_attack else "drop"
        tasks.append((str(csv_path), args.window, args.stride, args.is_attack,
                      spray_start, spray_end, args.seq_len, args.sequence_label,
                      args.boundary_policy, legacy_flag, run_id))

    # Process runs in parallel (or sequential fallback)
    workers = min(args.workers, len(tasks)) if tasks else 1
    if workers > 1 and len(tasks) > 1:
        log.info("Processing %d CSV files with %d workers", len(tasks), workers)
        with multiprocessing.Pool(workers) as pool:
            results = pool.map(_process_run_worker, tasks)
    else:
        results = [_process_run_worker(t) for t in tasks]

    # Collect results (metadata + hashing stays sequential — it's fast)
    for result, task in zip(results, tasks):
        if result is None:
            csv_path_str = task[0]
            relative_csv = Path(csv_path_str).relative_to(input_root).as_posix()
            log.warning("%s: no parseable events", relative_csv)
            continue
        features = result["features"]
        labels = result["labels"]
        starts = result["starts"]
        top_ids = result["top_ids"]
        top_comms = result["top_comms"]
        free_stats = result["free_stats"]
        empty_ratio = result["empty_ratio"]
        seq = result["seq"]
        seq_label = result["seq_label"]
        seq_group = result["seq_group"]
        seq_start = result["seq_start"]
        is_short = result["is_short"]
        run_id = result["run_id"]
        csv_path = Path(task[0])
        relative_csv = csv_path.relative_to(input_root).as_posix()
        spray_start = None
        spray_end = None
        # Recover spray markers for run record
        marker_key = run_id + ".log"
        run_markers = markers.get(marker_key, {})
        spray_start = run_markers.get("SPRAY_START")
        spray_end = run_markers.get("SPRAY_END")

        if args.is_attack and args.legacy_all_attack and spray_start is None:
            labels[:] = 1

        if is_short and args.short_run_policy == "error":
            raise RuntimeError(f"run {run_id} has {len(features)} windows < seq_len={args.seq_len}")

        window_features.append(features)
        window_labels.append(labels)
        window_groups.append(np.full(len(features), run_id, object))
        window_starts.append(starts)
        window_top_ids.append(top_ids)
        window_top_comms.append(top_comms)
        if len(seq):
            sequences.append(seq)
            sequence_labels.append(seq_label)
            sequence_groups.append(seq_group)
            sequence_starts.append(seq_start)

        metadata = parse_run_metadata(run_id)
        unresolved_ratio = free_stats["unresolved"] / max(free_stats["free_events"], 1)
        quality = "ok"
        if unresolved_ratio > args.free_unknown_threshold:
            quality = "high_free_unknown"
            log.warning("%s: unresolved free ratio %.1f%% exceeds threshold %s",
                        relative_csv, 100 * unresolved_ratio, args.free_unknown_threshold)
        run_records.append({
            "run_id": run_id,
            "metadata": metadata,
            "source_csv": relative_csv,
            "class": metadata["class"],
            "cve": metadata["cve"],
            "variant": metadata["variant"],
            "workload": metadata["workload"],
            "windows": int(len(features)),
            "sequences": int(len(seq)),
            "empty_window_ratio": float(empty_ratio),
            "free_stats": free_stats,
            "free_unknown_ratio": float(unresolved_ratio),
            "quality": quality,
            "marker_status": "complete" if spray_start is not None and spray_end is not None else "none",
            "label_counts": {str(k): int(v) for k, v in Counter(labels).items()},
        })
        input_hashes[relative_csv] = sha256_file(str(csv_path))
        log.info("%s: %d windows, %d sequences, labels=%s", relative_csv, len(features), len(seq), Counter(labels))

    merge_run_meta(run_records, args.run_meta)

    if not window_features:
        raise RuntimeError("no features extracted")

    all_windows = np.concatenate(window_features)
    all_window_labels = np.concatenate(window_labels)
    all_window_groups = np.concatenate(window_groups)
    all_window_starts = np.concatenate(window_starts)
    all_top_ids = np.concatenate(window_top_ids)
    all_top_comms = np.concatenate(window_top_comms)
    if sequences:
        all_sequences = np.concatenate(sequences)
        all_sequence_labels = np.concatenate(sequence_labels)
        all_sequence_groups = np.concatenate(sequence_groups)
        all_sequence_starts = np.concatenate(sequence_starts)
    else:
        all_sequences = np.empty((0, args.seq_len, config.FEAT_DIM), np.float32)
        all_sequence_labels = np.empty(0, np.int8)
        all_sequence_groups = np.empty(0, object)
        all_sequence_starts = np.empty(0, np.int64)

    # Plan 5.8: no NaN/Inf in preprocessing output.
    assert np.isfinite(all_windows).all(), "features contain NaN/Inf"
    assert np.isfinite(all_sequences).all(), "sequences contain NaN/Inf"

    np.savez_compressed(
        output / "features.npz",
        features=all_windows,
        labels=all_window_labels,
        window_run_ids=all_window_groups,
        window_start_ns=all_window_starts,
        top_task_ids=all_top_ids,
        top_task_comms=all_top_comms,
        sequences=all_sequences,
        seq_labels=all_sequence_labels,
        seq_run_ids=all_sequence_groups,
        seq_start_ns=all_sequence_starts,
        feature_names=np.asarray(feature_names(), object),
        schema_version=np.asarray(config.DATASET_SCHEMA_VERSION),
    )

    stats = {
        "schema_version": config.DATASET_SCHEMA_VERSION,
        "total_runs": len(run_records),
        "total_windows": int(len(all_windows)),
        "total_sequences": int(len(all_sequences)),
        "feat_dim": int(all_windows.shape[1]),
        "seq_len": args.seq_len,
        "window_ms": args.window,
        "stride_ms": args.stride,
        "is_attack": args.is_attack,
        "sequence_label_policy": args.sequence_label,
        "boundary_policy": args.boundary_policy,
        "short_run_policy": args.short_run_policy,
        "marker_required": bool(args.is_attack and not args.legacy_all_attack),
        "label_dist": {str(k): int(v) for k, v in Counter(all_window_labels).items()},
        "runs": run_records,
    }
    write_json_atomic(str(output / "stats.json"), stats)

    run_meta_out = {
        "runs": {
            record["run_id"]: {
                "class": record["class"],
                "cve": record["cve"],
                "variant": record["variant"],
                "workload": record["workload"],
                "windows": record["windows"],
                "sequences": record["sequences"],
                "empty_window_ratio": record["empty_window_ratio"],
                "free_unknown_ratio": record["free_unknown_ratio"],
                "quality": record["quality"],
                "marker_status": record["marker_status"],
                "label_counts": record["label_counts"],
            }
            for record in run_records
        }
    }
    write_json_atomic(str(output / "runs_meta.json"), run_meta_out)

    write_json_atomic(str(output / "feature_schema.json"), {
        "schema_version": config.DATASET_SCHEMA_VERSION,
        "features": feature_names(),
        "feature_groups": feature_groups(),
        "feat_dim": config.FEAT_DIM,
    })

    git_commit, git_dirty = snapshot_git_revision()
    output_hash = sha256_file(str(output / "features.npz"))
    dataset_manifest = {
        "dataset_version": "v2",
        "schema_version": config.DATASET_SCHEMA_VERSION,
        "dataset_kind": "pilot" if "pilot" in str(output) else ("final" if "final" in str(output) else "other"),
        "git_revision": git_commit,
        "git_dirty": git_dirty,
        "config_summary": {
            "window_ms": args.window,
            "stride_ms": args.stride,
            "seq_len": args.seq_len,
            "is_attack": args.is_attack,
            "sequence_label_policy": args.sequence_label,
            "boundary_policy": args.boundary_policy,
            "short_run_policy": args.short_run_policy,
            "legacy_all_attack": args.legacy_all_attack,
            "size_buckets": list(SIZE_BUCKETS),
            "overflow_bucket": "gt_8192",
            "feat_dim": config.FEAT_DIM,
            "free_unknown_ratio_threshold": args.free_unknown_threshold,
        },
        "input_file_count": len(csv_files),
        "input_hashes": input_hashes,
        "output_features_hash": output_hash,
    }
    write_json_atomic(str(output / "dataset_manifest.json"), dataset_manifest)

    log.info("Saved schema-v%d dataset to %s (output hash %s)",
             config.DATASET_SCHEMA_VERSION, output, output_hash)


if __name__ == "__main__":
    main()
