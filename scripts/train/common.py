"""Shared helpers for the schema-v2 training/evaluation harness (M5).

Leak-safety rules enforced here (IMPLEMENTATION_PLAN.md section 2/9):
  * the smallest unit of isolation is the run: sequences never cross a run, and
    a run never appears in more than one partition (G7);
  * scaler and model statistics are fit on training runs only;
  * the decision threshold is calibrated on validation normal scores at a target
    FPR (default p99), never by optimizing test-set Best F1;
  * test data is scored exactly once, at the frozen threshold.
"""

import json

import numpy as np

# Bucket labels in feature order (first 12 columns of the feature vector).
SIZE_BUCKET_LABELS = ["32", "64", "96", "128", "192", "256", "512",
                      "1024", "2048", "4096", "8192", "gt_8192"]
# Target slab per CVE (plan 7.1: 7308 = small kmalloc-256, 11176 = large
# kmalloc-2048). 2636 = kmalloc-8192: the n_hdlc double-free is reclaimed with
# 7872-byte UDP sk_buffs / 8144-byte add_key payloads (verified 100% run-level
# bucket-8192 coverage in spray windows on the cross-cve dataset).
TARGET_SLAB = {"CVE-2017-7308": "256", "CVE-2017-11176": "2048", "CVE-2017-2636": "8192"}
BUCKET_INDEX = {label: i for i, label in enumerate(SIZE_BUCKET_LABELS)}

DEFAULT_VAL_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.15
DEFAULT_TARGET_FPR = 0.01  # threshold = 99th percentile of validation-normal scores


def load_processed(proc_dir):
    """Load a processed dataset (features.npz + stats.json) into a dict."""
    data = np.load(str(proc_dir / "features.npz"), allow_pickle=True)
    if int(data.get("schema_version", 1)) < 3:
        raise ValueError(f"schema-v3 dataset required (has {data.get('schema_version', '?')}): {proc_dir}")
    with (proc_dir / "stats.json").open() as handle:
        stats = json.load(handle)
    return data, stats


def load_token_data(proc_dir):
    """Load token_sequences.npz for event-level models (GRU).

    Returns a dict-like npz with keys: token_seqs, token_seq_run_ids,
    token_seq_labels. Falls back to features.npz run_ids for splitting
    when token data is unavailable (e.g. legacy datasets).
    """
    token_path = proc_dir / "token_sequences.npz"
    if not token_path.exists():
        raise FileNotFoundError(f"token_sequences.npz not found in {proc_dir}")
    data = np.load(str(token_path), allow_pickle=True)
    return data


def run_stratum(run_id):
    """Stratum key for a run id: the workload/variant path minus the run index.

    'CVE-2017-11176/keyctl/run_000_abc123/trace' -> 'CVE-2017-11176/keyctl'
    """
    name = str(run_id)
    if "/run_" in name:
        return name.rsplit("/run_", 1)[0]
    return name


def split_run_groups(groups, seed, val_fraction=DEFAULT_VAL_FRACTION,
                     test_fraction=DEFAULT_TEST_FRACTION):
    """Split whole runs into train/val/test, stratified by workload/variant.

    Each stratum is shuffled independently and split 70/15/15 (with a floor of
    one run in each of val/test when the stratum is small). Returns lists of
    run ids. No run appears in more than one partition by construction (G7).
    """
    by_stratum = {}
    for group in sorted({str(g) for g in groups}):
        by_stratum.setdefault(run_stratum(group), []).append(group)
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for stratum in sorted(by_stratum):
        values = np.asarray(sorted(by_stratum[stratum]), dtype=object)
        rng.shuffle(values)
        count = len(values)
        if count < 3:
            raise ValueError(f"need at least 3 runs per stratum, got {count}: {values.tolist()}")
        n_val = max(1, int(round(count * val_fraction)))
        n_test = max(1, int(round(count * test_fraction)))
        if n_val + n_test >= count:  # too small to reserve both
            n_val = n_test = 1
        n_train = count - n_val - n_test
        train.extend(values[:n_train].tolist())
        val.extend(values[n_train:n_train + n_val].tolist())
        test.extend(values[n_train + n_val:].tolist())
    return train, val, test


def mask_for_groups(groups, selected):
    return np.isin(np.asarray(groups).astype(str), np.asarray(selected).astype(str))


def fit_scaler(train_windows):
    """Gaussian scaler fit on training windows only (plan 2: scaler fits train)."""
    mean = train_windows.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_windows.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def scale(values, mean, std):
    return ((values - mean) / std).astype(np.float32)


def aggregate_scores(per_window, aggregation):
    if aggregation == "last":
        return per_window[:, -1]
    if aggregation == "max":
        return per_window.max(axis=1)
    if aggregation == "p90":
        return np.percentile(per_window, 90, axis=1)
    if aggregation == "mean":
        return per_window.mean(axis=1)
    raise ValueError(f"unknown score aggregation: {aggregation}")


def score_windows(model, windows):
    """Score a (N, F) window matrix with any model exposing anomaly_score()."""
    return np.asarray(model.anomaly_score(np.asarray(windows, dtype=np.float32)), dtype=np.float64)


def score_sequences(model, sequences, aggregation):
    """Score sequences, aggregating per-sequence position scores.

    Works with both feature sequences (N, T, F) float32 and token sequences
    (N, L) int32. Sequence models expose sequence_anomaly_score which returns
    per-position scores; window models reuse score_windows via reshape.

    For token models (GRU), sequences is (N, L) int32 and
    sequence_anomaly_score returns (N, L) per-position violation indicators.
    """
    if hasattr(model, "sequence_anomaly_score"):
        per_position = np.asarray(model.sequence_anomaly_score(sequences),
                                  dtype=np.float64)
        return aggregate_scores(per_position, aggregation)
    # window-model fallback: reshape (N, T, F) -> (N*T, F)
    count, seq_len, feat_dim = sequences.shape
    per_window = score_windows(model, sequences.reshape(count * seq_len, feat_dim))
    per_window = per_window.reshape(count, seq_len)
    return aggregate_scores(per_window, aggregation)


def score_sequences_batched(model, sequences, aggregation, batch=4096):
    """Score a materialized (N,T,F) array in chunks to bound peak memory.

    sequence_anomaly_score / the window-reshape path materialize intermediate
    float32/float64 arrays proportional to N; for cross-CVE scenario 4 the eval
    slice alone is ~1.7GB on top of the 2.6GB base array (7GB VM). Chunking keeps
    at most batch*T*F resident per call; results are identical because each
    sequence is aggregated independently.
    """
    count = int(sequences.shape[0])
    if count <= batch:
        return score_sequences(model, sequences, aggregation)
    parts = [score_sequences(model, sequences[i:i + batch], aggregation)
             for i in range(0, count, batch)]
    return np.concatenate(parts)


def score_sequences_masked(model, base_sequences, mask, aggregation, batch=4096):
    """Score base_sequences[mask] without ever materializing the full slice.

    Equivalent to score_sequences(base_sequences[mask].astype(np.float32), ...)
    but reads at most `batch` sequences from the base array per chunk, so the
    peak is base + batch*T*F instead of base + slice + reshape. Used for the
    cross-CVE eval sets (scenario 4 can put all 349 AB runs = ~151k sequences
    into eval).
    """
    idx = np.flatnonzero(mask)
    count = int(len(idx))
    # token models use int32, feature models use float32
    dtype = np.int32 if base_sequences.dtype == np.int32 else np.float32
    if count <= batch:
        return score_sequences(model, base_sequences[idx].astype(dtype), aggregation)
    parts = [score_sequences(model, base_sequences[idx[i:i + batch]].astype(dtype),
                             aggregation)
             for i in range(0, count, batch)]
    return np.concatenate(parts)


def threshold_at_fpr(normal_scores, target_fpr=DEFAULT_TARGET_FPR):
    """Freeze threshold at the (1-target_fpr) percentile of validation-normal scores."""
    if target_fpr <= 0 or target_fpr >= 1:
        raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")
    return float(np.percentile(normal_scores, 100.0 * (1.0 - target_fpr)))


def run_max_scores(scores, groups):
    """Per-run max sequence score (run-level aggregation, plan 2 semantics)."""
    out, ids = [], []
    for group in sorted({str(g) for g in groups}):
        mask = np.asarray(groups).astype(str) == group
        out.append(float(np.asarray(scores)[mask].max()))
        ids.append(group)
    return np.asarray(out, dtype=np.float64), np.asarray(ids, dtype=object)


def exact_best_f1(labels, scores):
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(np.nanargmax(f1))
    thr = float(thresholds[index]) if index < len(thresholds) else float(np.max(scores))
    return float(f1[index]), thr


def classification_metrics(scores, labels, threshold):
    """Sequence/run-level metrics at a frozen threshold (plan 9.4)."""
    from sklearn.metrics import (average_precision_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if len(scores) != len(labels):
        raise ValueError(f"score/label length mismatch: {len(scores)} != {len(labels)}")
    if len(np.unique(labels)) < 2:
        return {"error": "requires both classes"}
    preds = (scores > threshold).astype(np.int8)
    normal = labels == 0
    oracle_f1, oracle_thr = exact_best_f1(labels, scores)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "f1_at_threshold": float(f1_score(labels, preds, zero_division=0)),
        "precision_at_threshold": float(precision_score(labels, preds, zero_division=0)),
        "recall_at_threshold": float(recall_score(labels, preds, zero_division=0)),
        "fpr_at_threshold": float(np.mean(preds[normal])) if normal.any() else 0.0,
        "threshold": float(threshold),
        "flagged": int(preds.sum()),
        "count": int(len(labels)),
        "oracle_best_f1_test_only": oracle_f1,
        "oracle_threshold_test_only": oracle_thr,
    }


def grouped_metrics(scores, labels, groups, threshold):
    """Metrics per group (workload / CVE / variant), run-level max aggregation."""
    out = {}
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    for group in sorted({str(g) for g in groups}):
        mask = np.asarray(groups).astype(str) == group
        if not mask.any():
            continue
        if len(np.unique(labels[mask])) < 2:
            out[group] = {
                "count": int(mask.sum()),
                "flagged": int((scores[mask] > threshold).sum()),
                "note": "single-class group",
            }
            continue
        out[group] = classification_metrics(scores[mask], labels[mask], threshold)
    return out


def bootstrap_run_ci(run_scores, run_labels, threshold, n_boot=2000, seed=0):
    """Bootstrap 95% CI over runs for run-level ROC-AUC / F1 (plan 9.4)."""
    from sklearn.metrics import f1_score, roc_auc_score
    run_scores = np.asarray(run_scores, dtype=np.float64)
    run_labels = np.asarray(run_labels, dtype=np.int8)
    if len(np.unique(run_labels)) < 2:
        return {"note": "requires both classes across runs"}
    rng = np.random.default_rng(seed)
    aucs, f1s = [], []
    n = len(run_scores)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(run_labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(run_labels[idx], run_scores[idx]))
        f1s.append(f1_score(run_labels[idx], run_scores[idx] > threshold, zero_division=0))
    if not aucs:
        return {"note": "bootstrap produced no two-class resample"}
    return {
        "roc_auc_ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        "n_boot": n_boot,
    }


def experiment_id(kind, model_name, seed):
    """Unique experiment id: <date>_<kind>_v2_<model>_s<seed>_<HHMMSS>.

    The time stamp makes every invocation land in a fresh directory (plan: data
    is never overwritten); run content stays deterministic for a fixed seed.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return f"{now.strftime('%Y-%m-%d')}_{kind}_v2_{model_name}_s{seed}_{now.strftime('%H%M%S')}"
