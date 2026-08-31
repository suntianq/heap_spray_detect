#!/usr/bin/env python3
"""Anomaly-score scatter for a frozen model, at run- or window-level.

--level run    (default) one point per collected run: x = the run's anomaly
               score (max-aggregated over its sequence scores, the same
               aggregation run_experiment uses for run-level metrics), y =
               uniform jitter purely for visual separation.
--level window one point per 500 ms window: x = seconds since that run's
               first window, y = the per-window anomaly score. This is the
               finest granularity the model can produce -- features and scores
               are per-window, and raw trace events are aggregated into them,
               so individual kmalloc/kfree events have no score of their own.
               Overlapping sequences (stride 0.5 s, seq_len 32) cover the same
               window many times; duplicates are collapsed by taking the max,
               one point per (run, half-second slot).

Normal runs blue, attack runs red. Scores are the raw model scores -- no
normalization, no edits. Test-normal is the held-out partition the frozen
model never trained on; attack boundary windows (label -1) are dropped.

Usage:
    python3 scripts/visualize/score_scatter.py \
        --run runs/2026-08-21_ocsvm_v2_ocsvm_s42_072736 \
        --attack-data datasets/processed/attack \
        --normal-data datasets/processed/normal \
        --level window \
        --out datasets/report/score_scatter_ocsvm_window.png
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.train import common  # noqa: E402


def per_window_scores(model, sequences):
    """(N, T, F) -> per-window anomaly scores (N, T).

    Sequence models expose sequence_anomaly_score((N,T,F)) -> (N,T); window
    models score each flattened window independently and are reshaped back.
    """
    count, seq_len, feat_dim = sequences.shape
    if hasattr(model, "sequence_anomaly_score"):
        return np.asarray(model.sequence_anomaly_score(sequences), dtype=np.float64)
    pw = common.score_windows(model, sequences.reshape(count * seq_len, feat_dim))
    return pw.reshape(count, seq_len)


def window_points(model, sequences, run_ids, seq_start_ns):
    """Per-window scatter points: (x_seconds_since_run_start, score), deduped.

    Each sequence row is anchored at seq_start_ns[i] and spans seq_len windows
    spaced by the observed inter-window step. Consecutive sequences overlap, so
    the same absolute window appears in many rows; collapse via per-(run,
    half-second-slot) max, consistent with the run-level "max" semantics.
    """
    n, t, f = sequences.shape
    pw = per_window_scores(model, sequences)                       # (n, t)
    starts = np.asarray(seq_start_ns, dtype=np.float64)
    run_ids = np.asarray(run_ids).astype(str)
    u, inv = np.unique(run_ids, return_inverse=True)
    first = np.full(len(u), np.inf)
    np.minimum.at(first, inv, starts)                              # run start (ns)
    # window step from within-run diffs only: all runs share the same relative
    # offsets, so a global diff collapses to 0 and would merge every run into
    # one point. stride is 50 ms in schema v2.
    step_ns = float(np.median(np.diff(starts)[run_ids[1:] == run_ids[:-1]]))
    step_s = step_ns / 1e9
    rel = starts - first[inv]                                      # (n,)
    time_s = rel[:, None] / 1e9 + step_s * np.arange(t)[None, :]   # (n,t)
    slot = np.round(time_s / step_s).astype(np.int64)              # (i-i0)+t, exact

    key = inv[:, None] * 1_000_000 + slot
    kf = key.ravel()
    vf = pw.ravel()
    uniq, inv2 = np.unique(kf, return_inverse=True)
    agg = np.full(len(uniq), -np.inf)
    np.maximum.at(agg, inv2, vf)
    xs = (uniq % 1_000_000) * step_s                               # seconds
    return xs, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="experiment dir of the frozen model")
    ap.add_argument("--attack-data", required=True)
    ap.add_argument("--normal-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", choices=["run", "window"], default="run",
                    help="scatter granularity: run (one point per run) or window "
                         "(one point per 500 ms window, x = time since run start)")
    ap.add_argument("--seed", type=int, default=0, help="jitter RNG seed (run level)")
    args = ap.parse_args()

    run_dir = Path(args.run)
    split = json.loads((run_dir / "split_manifest.json").read_text())
    model = pickle.load(open(run_dir / "model.pkl", "rb"))

    normal, _ = common.load_processed(Path(args.normal_data))
    attack, _ = common.load_processed(Path(args.attack_data))

    # ---- test-normal runs (the held-out partition this model never trained on) ----
    normal_groups = np.asarray(normal["seq_run_ids"]).astype(str)
    test_mask = common.mask_for_groups(normal_groups, split["test_groups"])
    test_seq = normal["sequences"][test_mask].astype(np.float32)

    # ---- attack runs (drop boundary sequences, plan 5.6) ----
    keep = np.asarray(attack["seq_labels"], dtype=np.int8) >= 0
    atk_seq = attack["sequences"][keep].astype(np.float32)

    if args.level == "window":
        nx, ny = window_points(model, test_seq,
                               normal_groups[test_mask],
                               normal["seq_start_ns"][test_mask])
        ax, ay = window_points(model, atk_seq,
                               np.asarray(attack["seq_run_ids"]).astype(str)[keep],
                               attack["seq_start_ns"][keep])
        print(f"normal windows={len(ny)}  attack windows={len(ay)}")
        print(f"normal score range [{ny.min():.4f}, {ny.max():.4f}]")
        print(f"attack score range [{ay.min():.4f}, {ay.max():.4f}]")
        print(f"run_threshold={split.get('run_threshold') or '(see train_report.json)'}")

        plt.figure(figsize=(9, 3.6))
        plt.scatter(nx, ny, color="#2878b5", s=2, alpha=0.35, label="Normal",
                    rasterized=True)
        plt.scatter(ax, ay, color="#d94b4b", s=2, alpha=0.6, label="Attack",
                    rasterized=True)
        plt.xlabel("Time since run start (s)")
        plt.ylabel("Per-window anomaly score")
    else:
        test_scores = common.score_sequences(model, test_seq, "max")
        test_run_scores, _ = common.run_max_scores(
            test_scores, np.char.add("normal:", normal_groups[test_mask]))
        atk_scores = common.score_sequences(model, atk_seq, "max")
        atk_groups = np.asarray(attack["seq_run_ids"]).astype(str)[keep]
        atk_run_scores, _ = common.run_max_scores(
            atk_scores, np.char.add("attack:", atk_groups))

        normal_scores = np.asarray(test_run_scores, dtype=np.float64)
        attack_scores = np.asarray(atk_run_scores, dtype=np.float64)
        print(f"normal runs={len(normal_scores)}  attack runs={len(attack_scores)}")
        print(f"normal score range [{normal_scores.min():.4f}, {normal_scores.max():.4f}]")
        print(f"attack score range [{attack_scores.min():.4f}, {attack_scores.max():.4f}]")
        print(f"run_threshold={split.get('run_threshold') or '(see train_report.json)'}")

        rng = np.random.default_rng(args.seed)
        normal_y = rng.uniform(-0.16, 0.16, len(normal_scores))
        attack_y = rng.uniform(-0.16, 0.16, len(attack_scores))

        plt.figure(figsize=(9, 2.6))
        plt.scatter(normal_scores, normal_y, color="#2878b5", label="Normal", s=22)
        plt.scatter(attack_scores, attack_y, color="#d94b4b", label="Attack", s=22)
        plt.xlabel("Anomaly score")
        plt.yticks([])

    plt.grid(axis="x", alpha=0.2)
    plt.legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
