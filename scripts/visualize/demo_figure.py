#!/usr/bin/env python3
"""Leadership demo figures: one 2x2 composite and/or four standalone panels.

All data from the frozen ocsvm model (seed 42, final-v2):
  P1 fingerprint  -- alloc-count heatmap by size bucket over time, one
        representative CVE-2017-11176 single-spray run. The spray fires a
        concentrated burst (512x kmalloc-2048) inside 100 ms.
  P2 normal       -- a normal msg_msg_2048 run on the same kernel: sustained
        allocations (mostly kmalloc-4096) across the whole run, no single-shot
        burst. "Volume alone is not the signature."
  P3 response     -- per-window ocsvm anomaly score of the SAME attack run;
        score jumps across the frozen run threshold exactly at the spray.
  P4 response     -- per-window ocsvm scores of the SAME normal msg_msg run as
        P2: stays well below the threshold for the whole 6.25 s; even the
        single biggest burst (16k allocs at ~3 s) only reaches -2.6, ~3300x
        below the frozen run threshold.

--mode composite: one 2x2 image. --mode split: four standalone PNGs (P1..P4).
--mode both (default): composite + the four standalone files.

Usage:
    python3 scripts/visualize/demo_figure.py \
        --run runs/2026-08-21_ocsvm_v2_ocsvm_s42_072736 \
        --attack-data datasets/final-v2/processed/attack \
        --normal-data datasets/final-v2/processed/normal \
        --mode both \
        --out datasets/final-v2/report/heapspray_demo.png
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.train import common  # noqa: E402

BUCKETS = [32, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096, 8192, ">8k"]
ATTACK_RUN = "CVE-2017-11176/poc_cfh_single_spray/run_000_c9e0986455b1/trace"
NORMAL_RUN = "CVE-2017-11176/msg_msg_2048/run_000_4c386c1762d2/trace"

NORMAL_C = "#2878b5"
ATTACK_C = "#d94b4b"

PANEL_TITLES = {
    "fingerprint": "P1  ·  Spray fingerprint — attack run (CVE-2017-11176)",
    "normal": "P2  ·  Normal workload, same kernel (msg_msg)",
    "response": "P3  ·  Detector response — same attack run (per-window score)",
    "normal_response": "P4  ·  Detector response — normal workload (msg_msg)",
}


def heatmap_ax(ax, alloc, start_ns, title, spray_idx=None, note=None):
    """alloc: (W, 12) counts per window; heatmap log10(1+count) over time."""
    w = alloc.shape[0]
    t = (start_ns - start_ns[0]) / 1e9
    img = np.log10(1.0 + np.asarray(alloc)).T          # (12, W)
    im = ax.imshow(img, aspect="auto", origin="lower", cmap="magma",
                   extent=[t[0] - 0.025, t[-1] + 0.025, -0.5, 11.5])
    ax.set_yticks(range(12), BUCKETS, fontsize=8)
    ax.set_xticks(np.linspace(t[0], t[-1], 5).round(1))
    ax.set_xlabel("time since run start (s)", fontsize=9)
    ax.set_ylabel("size class (bytes)", fontsize=9)
    ax.set_title(title, fontsize=11)
    for idx in (spray_idx or []):
        ax.axvspan(t[idx] - 0.025, t[idx] + 0.025, color="#ffd54f", alpha=0.35, lw=0)
    if note:
        ax.annotate(note, xy=(0.99, 0.96), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))
    return im


def draw_fingerprint(ax, awin, astart, spray_idx):
    return heatmap_ax(
        ax, awin[:, 0:12], astart, PANEL_TITLES["fingerprint"],
        spray_idx=spray_idx,
        note="SPRAY: 512x kmalloc-2048 inside 100 ms\n"
             "single process 98%  ·  callsite entropy 0.07")


def draw_normal(ax, nwin, nstart):
    return heatmap_ax(
        ax, nwin[:, 0:12], nstart, PANEL_TITLES["normal"],
        note="sustained allocations across the whole run\n"
             "(mostly kmalloc-4096), no single-shot burst")


def _score_axis(ax, s_min):
    """arcsinh score axis: keeps "higher = more anomalous" while resolving the
    huge dynamic range between quiet scores (~ -1000) and the near-zero
    threshold. Tick labels still show the original score values; only the
    spacing is compressed, so a normal peak (-2.6) sits visibly below the
    threshold (-0.0008) instead of vanishing into the axis top."""
    ax.set_yscale("function", functions=(np.arcsinh, np.sinh))
    ax.set_ylim(s_min * 1.05, 0.5)  # headroom above 0 so the line clears the spine
    ax.yaxis.set_major_locator(FixedLocator([-1000, -100, -10, -1, -0.1, 0.0]))


def draw_response(ax, tt, atraj, spray_idx, run_threshold):
    ax.plot(tt, atraj, color="#3b3b3b", lw=0.8)
    ax.axhline(run_threshold, color=ATTACK_C, ls="--", lw=1.4,
               label=f"run threshold = {run_threshold:.4f}")
    for idx in spray_idx:
        ax.axvspan(tt[idx] - 0.025, tt[idx] + 0.025, color="#ffd54f", alpha=0.35)
    ax.scatter(tt[spray_idx], atraj[spray_idx], color=ATTACK_C, s=45, zorder=5,
               label="spray windows")
    ax.set_title(PANEL_TITLES["response"])
    ax.set_xlabel("time since run start (s)", fontsize=9)
    ax.set_ylabel("anomaly score (higher = more anomalous)", fontsize=9)
    _score_axis(ax, atraj.min())
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


def draw_normal_response(ax, nt, ntraj, nalloc, run_threshold):
    """Per-window ocsvm scores of the normal msg_msg run (same run as P2).

    Stays well below the threshold for the whole 6.25 s; the single biggest
    burst (16k allocs at ~3 s) only pushes the score up to -2.6, about 3300x
    below the frozen run threshold -- volume alone is not an anomaly signal.
    """
    ax.plot(nt, ntraj, color=NORMAL_C, lw=0.9)
    ax.axhline(run_threshold, color=ATTACK_C, ls="--", lw=1.4,
               label=f"run threshold = {run_threshold:.4f}")
    peak = int(np.argmax(ntraj))
    ax.annotate(f"all {len(ntraj)} windows below the threshold\n"
                f"peak {ntraj[peak]:.2f} at {nt[peak]:.2f}s "
                f"(burst {int(nalloc[peak]):,} allocs)",
                xy=(0.02, 0.98), xycoords="axes fraction", ha="left", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))
    ax.set_title(PANEL_TITLES["normal_response"])
    ax.set_xlabel("time since run start (s)", fontsize=9)
    ax.set_ylabel("anomaly score (higher = more anomalous)", fontsize=9)
    _score_axis(ax, ntraj.min())
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--attack-data", required=True)
    ap.add_argument("--normal-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["composite", "split", "both"], default="both")
    args = ap.parse_args()

    run_dir = Path(args.run)
    train_report = json.loads((run_dir / "train_report.json").read_text())
    model = pickle.load(open(run_dir / "model.pkl", "rb"))
    run_threshold = train_report["run_threshold"]

    attack, _ = common.load_processed(Path(args.attack_data))
    normal, _ = common.load_processed(Path(args.normal_data))
    af = attack["features"].astype(np.float64)
    nf = normal["features"].astype(np.float64)
    arid = np.asarray(attack["window_run_ids"]).astype(str)
    nrid = np.asarray(normal["window_run_ids"]).astype(str)
    alab = np.asarray(attack["labels"], dtype=np.int8)

    # ---- representative runs ----------------------------------------------
    am = arid == ATTACK_RUN
    nm = nrid == NORMAL_RUN
    awin, albl = af[am], alab[am]
    nwin = nf[nm]
    spray_idx = list(np.where(albl == 1)[0])
    astart = np.asarray(attack["window_start_ns"])[am]
    nstart = np.asarray(normal["window_start_ns"])[nm]
    tt = (astart - astart[0]) / 1e9

    # ---- panel 3 data: detector response on the attack run ------------------
    atraj = np.asarray(model.anomaly_score(awin), dtype=np.float64)
    # ---- panel 4 data: detector response on the normal msg_msg run (same as P2)
    ntraj = np.asarray(model.anomaly_score(nwin), dtype=np.float64)
    nt = (nstart - nstart[0]) / 1e9

    print(f"attack run: windows={len(awin)} spray={len(spray_idx)} "
          f"score[{atraj.min():.2f},{atraj.max():.2f}] thr={run_threshold:.5f}")
    print(f"normal run: windows={len(ntraj)} score[{ntraj.min():.2f},{ntraj.max():.2f}] "
          f"peak_at={nt[int(np.argmax(ntraj))]:.2f}s  "
          f"above_thr={(ntraj > run_threshold).sum()}/{len(ntraj)}")

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": True,
                         "grid.alpha": 0.15})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode in ("split", "both"):
        # P1 fingerprint
        fig, ax = plt.subplots(figsize=(9.5, 5))
        im = draw_fingerprint(ax, awin, astart, spray_idx)
        fig.colorbar(im, ax=ax, label="log10(1 + alloc count)")
        fig.tight_layout()
        fig.savefig(out.parent / "heapspray_demo_p1_fingerprint.png", dpi=180)
        plt.close(fig)
        # P2 normal
        fig, ax = plt.subplots(figsize=(9.5, 5))
        im = draw_normal(ax, nwin, nstart)
        fig.colorbar(im, ax=ax, label="log10(1 + alloc count)")
        fig.tight_layout()
        fig.savefig(out.parent / "heapspray_demo_p2_normal.png", dpi=180)
        plt.close(fig)
        # P3 response
        fig, ax = plt.subplots(figsize=(9, 4.6))
        draw_response(ax, tt, atraj, spray_idx, run_threshold)
        fig.tight_layout()
        fig.savefig(out.parent / "heapspray_demo_p3_response.png", dpi=180)
        plt.close(fig)
        # P4 normal response
        fig, ax = plt.subplots(figsize=(9, 4.6))
        draw_normal_response(ax, nt, ntraj, nwin[:, 51], run_threshold)
        fig.tight_layout()
        fig.savefig(out.parent / "heapspray_demo_p4_normal_response.png", dpi=180)
        plt.close(fig)
        print("wrote", out.parent / "heapspray_demo_p{1..4}_*.png")

    if args.mode in ("composite", "both"):
        fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
        im = draw_fingerprint(axes[0, 0], awin, astart, spray_idx)
        fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04,
                     label="log10(1 + alloc count)")
        im2 = draw_normal(axes[0, 1], nwin, nstart)
        fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04,
                     label="log10(1 + alloc count)")
        draw_response(axes[1, 0], tt, atraj, spray_idx, run_threshold)
        draw_normal_response(axes[1, 1], nt, ntraj, nwin[:, 51], run_threshold)
        fig.suptitle("Heap-spray detection from kernel allocation telemetry",
                     fontsize=15, fontweight="bold", y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out, dpi=180)
        print("wrote", out)


if __name__ == "__main__":
    main()
