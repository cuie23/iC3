#!/usr/bin/env python3
"""
Plot EE defect, quaternion defect (deg), and object defect across iC3 iterations
for parallelized MSiC3 (MSiC3Parallel).

Usage:
  python3 examples/python/plot_defects.py
  python3 examples/python/plot_defects.py --watch
  python3 examples/python/plot_defects.py --show
  python3 examples/python/plot_defects.py --csv /path/to/defects.csv --out /path/to/defects.png
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def find_default_csv():
    candidates = [
        "defects.csv",
        "examples/resources/multifinger_hand/ic3_debug_data/defects.csv",
        "../examples/resources/multifinger_hand/ic3_debug_data/defects.csv",
        "/home/ericcui/c3_plus/c3/defects.csv",
        "/home/ericcui/c3_plus/c3/examples/resources/multifinger_hand/ic3_debug_data/defects.csv",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 0:
            return c
    return candidates[0]


def load_defect_data(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None

    data = defaultdict(lambda: {"iteration": [], "ee_defect_norm": [], "quat_defect_deg": [], "object_defect_norm": []})

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                seg = int(row["segment"])
                it = int(row["iteration"])
                ee = float(row["ee_defect_norm"])
                quat = float(row["quat_defect_deg"])
                obj = float(row["object_defect_norm"])
                data[seg]["iteration"].append(it)
                data[seg]["ee_defect_norm"].append(ee)
                data[seg]["quat_defect_deg"].append(quat)
                data[seg]["object_defect_norm"].append(obj)
            except (ValueError, KeyError):
                continue

    return data if data else None


def plot_defects(csv_path, out_path="defects.png", show=False, fig_ax=None):
    data = load_defect_data(csv_path)
    if data is None:
        print(f"[plot_defects] No valid defect data found in: {csv_path}")
        return None

    segments = sorted(data.keys())
    all_iterations = set()
    for seg in segments:
        all_iterations.update(data[seg]["iteration"])
    iterations = sorted(all_iterations)

    if fig_ax is None:
        fig, (ax_ee, ax_quat, ax_obj) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    else:
        fig, (ax_ee, ax_quat, ax_obj) = fig_ax
        ax_ee.clear()
        ax_quat.clear()
        ax_obj.clear()

    colors = plt.cm.tab10(range(max(len(segments), 1)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for idx, seg in enumerate(segments):
        seg_data = data[seg]
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        label = f"Segment {seg}"

        # 1. EE Defect
        ax_ee.plot(
            seg_data["iteration"],
            seg_data["ee_defect_norm"],
            marker=m,
            color=c,
            linewidth=2,
            markersize=6,
            label=label,
        )

        # 2. Quaternion Defect (degrees)
        ax_quat.plot(
            seg_data["iteration"],
            seg_data["quat_defect_deg"],
            marker=m,
            color=c,
            linewidth=2,
            markersize=6,
            label=label,
        )

        # 3. Object Defect
        ax_obj.plot(
            seg_data["iteration"],
            seg_data["object_defect_norm"],
            marker=m,
            color=c,
            linewidth=2,
            markersize=6,
            label=label,
        )

    # Styling: Top plot (EE Defect)
    ax_ee.set_ylabel("EE Defect (norm)", fontsize=12, fontweight="bold")
    ax_ee.set_title("MSiC3 Parallel Convergence - Segment Defects vs Iteration", fontsize=14, fontweight="bold", pad=12)
    ax_ee.grid(True, linestyle="--", alpha=0.6)
    ax_ee.legend(loc="upper right", framealpha=0.9)

    # Styling: Middle plot (Quaternion Defect)
    ax_quat.set_ylabel("Quat Defect (deg)", fontsize=12, fontweight="bold")
    ax_quat.grid(True, linestyle="--", alpha=0.6)
    ax_quat.legend(loc="upper right", framealpha=0.9)

    # Styling: Bottom plot (Object Defect)
    ax_obj.set_ylabel("Object Defect (norm)", fontsize=12, fontweight="bold")
    ax_obj.set_xlabel("iC3 Iteration", fontsize=12, fontweight="bold")
    ax_obj.grid(True, linestyle="--", alpha=0.6)
    ax_obj.legend(loc="upper right", framealpha=0.9)

    if len(iterations) > 0:
        ax_obj.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()

    # Save and overwrite plot
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[plot_defects] Saved defect plot to: {os.path.abspath(out_path)}")

    if show and fig_ax is None:
        plt.show()

    return fig, (ax_ee, ax_quat, ax_obj)


def watch_and_plot(csv_path, out_path="defects.png", show=False):
    print(f"[plot_defects] Watching '{csv_path}' for updates (Ctrl+C to stop)...")
    last_mtime = 0
    fig_ax = None

    if show:
        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
        fig_ax = (fig, (ax1, ax2, ax3))

    try:
        while True:
            if os.path.exists(csv_path):
                mtime = os.path.getmtime(csv_path)
                if mtime != last_mtime:
                    last_mtime = mtime
                    print(f"\n[plot_defects] Detected update at {time.strftime('%H:%M:%S')}")
                    time.sleep(0.05)  # brief wait for file write completion
                    fig_ax = plot_defects(csv_path, out_path, show=False, fig_ax=fig_ax)
                    if show and fig_ax is not None:
                        fig_ax[0].canvas.draw_idle()
                        fig_ax[0].canvas.flush_events()
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[plot_defects] Stopped watching.")


def main():
    parser = argparse.ArgumentParser(description="Plot segment defects for parallelized MSiC3.")
    parser.add_argument("--csv", type=str, default=None, help="Path to input defects.csv")
    parser.add_argument("--out", type=str, default="defects.png", help="Path to output plot image")
    parser.add_argument("--watch", action="store_true", help="Continuously watch CSV file and re-plot on update")
    parser.add_argument("--show", action="store_true", help="Display interactive GUI plot window")
    args = parser.parse_args()

    csv_path = args.csv if args.csv else find_default_csv()

    if args.watch:
        watch_and_plot(csv_path, args.out, show=args.show)
    else:
        plot_defects(csv_path, args.out, show=args.show)


if __name__ == "__main__":
    main()
