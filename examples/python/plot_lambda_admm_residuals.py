#!/usr/bin/env python3
"""
3D Plots of C3 Inner ADMM Convergence Metrics:
  1. Iterate Step Change: ||delta^k - delta^{k-1}|| (Stationarity)
  2. Complementarity Slackness: sum_i |lambda_i^T eta_i| (Complementarity Feasibility)
  3. Force Consensus Error: ||lambda - delta_lambda||_2 (Primal Consensus)

Axes for all plots:
  X: Plan Timestep (x)
  Y: C3 ADMM Iteration (y)
  Z: Metric Value (Average across C3 horizon N)

Usage:
  python3 examples/python/plot_lambda_admm_residuals.py
  python3 examples/python/plot_lambda_admm_residuals.py --metric iterate_step
  python3 examples/python/plot_lambda_admm_residuals.py --metric complementarity
  python3 examples/python/plot_lambda_admm_residuals.py --show
  python3 examples/python/plot_lambda_admm_residuals.py --watch
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

# Ensure user site-packages mpl_toolkits takes priority if present
for site_pkg in sys.path:
    cand = os.path.join(site_pkg, "mpl_toolkits")
    if os.path.isdir(cand):
        try:
            import mpl_toolkits
            if cand not in mpl_toolkits.__path__:
                mpl_toolkits.__path__.insert(0, cand)
        except Exception:
            pass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def find_default_csv():
    candidates = [
        "lambda_admm_residuals.csv",
        "examples/resources/multifinger_hand/ic3_debug_data/lambda_admm_residuals.csv",
        "../examples/resources/multifinger_hand/ic3_debug_data/lambda_admm_residuals.csv",
        "/home/ericcui/c3_plus/c3/lambda_admm_residuals.csv",
        "/home/ericcui/c3_plus/c3/examples/resources/multifinger_hand/ic3_debug_data/lambda_admm_residuals.csv",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 0:
            return c
    return candidates[0]


def load_all_metrics(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None, None

    # records: dict outer_iter -> dict of metric_name -> list of (step, admm_iter, val)
    records_by_iter = defaultdict(lambda: defaultdict(list))

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                outer_it = int(row["iteration"])
                step = int(row["plan_timestep"])
                admm_it = int(row["admm_iter"])
                lam_diff = float(row.get("lambda_diff_norm", 0.0))
                step_chg = float(row.get("iterate_step_change", 0.0))
                comp_slk = float(row.get("complementarity_slack", 0.0))

                records_by_iter[outer_it]["lambda_diff"].append((step, admm_it, lam_diff))
                records_by_iter[outer_it]["iterate_step"].append((step, admm_it, step_chg))
                records_by_iter[outer_it]["complementarity"].append((step, admm_it, comp_slk))
            except (ValueError, KeyError):
                continue

    if not records_by_iter:
        return None, None

    max_outer_iter = max(records_by_iter.keys())
    return records_by_iter, max_outer_iter


def render_surface_on_ax(ax, pts, z_label, title, cmap_name="viridis", elev=28, azim=-60):
    steps = np.array([p[0] for p in pts])
    admm_iters = np.array([p[1] for p in pts])
    z_vals = np.array([p[2] for p in pts])

    unique_steps = np.sort(np.unique(steps))
    unique_admm = np.sort(np.unique(admm_iters))

    grid_available = False
    if len(unique_steps) * len(unique_admm) == len(pts):
        try:
            step_to_i = {s: i for i, s in enumerate(unique_steps)}
            admm_to_j = {a: j for j, a in enumerate(unique_admm)}
            Z_grid = np.zeros((len(unique_admm), len(unique_steps)))
            for s, a, z in pts:
                Z_grid[admm_to_j[a], step_to_i[s]] = z
            X_grid, Y_grid = np.meshgrid(unique_steps, unique_admm)
            grid_available = True
        except Exception:
            grid_available = False

    if grid_available:
        surf = ax.plot_surface(
            X_grid, Y_grid, Z_grid,
            cmap=cmap_name,
            edgecolor='k',
            linewidth=0.2,
            alpha=0.9,
            antialiased=True,
            rstride=1,
            cstride=1
        )
    else:
        surf = ax.plot_trisurf(steps, admm_iters, z_vals, cmap=cmap_name, edgecolor='none', alpha=0.85)
        ax.scatter(steps, admm_iters, z_vals, c=z_vals, cmap=cmap_name, s=10, depthshade=True)

    ax.set_xlabel('Plan Timestep ($x$)', fontsize=11, labelpad=10, fontweight='bold')
    ax.set_ylabel('C3 ADMM Iteration ($y$)', fontsize=11, labelpad=10, fontweight='bold')
    ax.set_zlabel(z_label, fontsize=11, labelpad=10, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, linestyle='--', alpha=0.5)
    return surf


def plot_single_metric(pts, metric_key, outer_iter, output_path=None, show=False):
    metric_configs = {
        "iterate_step": {
            "title": f"C3 ADMM Iterate Step Change $\\|\delta^k - \delta^{{k-1}}\\|_2$\n(MS-iC3 Final Outer Iteration {outer_iter})",
            "z_label": r"$\|\delta^k - \delta^{k-1}\|_2$",
            "cbar_label": r"Mean $\|\delta^k - \delta^{k-1}\|_2$ over C3 $N$",
            "cmap": "plasma",
        },
        "complementarity": {
            "title": f"C3 Complementarity Slackness Gap $\\lambda^T \\eta$\n(MS-iC3 Final Outer Iteration {outer_iter})",
            "z_label": r"$\lambda^T \eta$",
            "cbar_label": r"Mean $\sum |\lambda_i \eta_i|$ over C3 $N$",
            "cmap": "magma",
        },
        "lambda_diff": {
            "title": f"C3 Inner ADMM Force Consensus Residual $\\|\\lambda - \\delta_\\lambda\\|_2$\n(MS-iC3 Final Outer Iteration {outer_iter})",
            "z_label": r"$\|\lambda - \delta_\lambda\|_2$",
            "cbar_label": r"Mean $\|\lambda - \delta_\lambda\|_2$ over C3 $N$ (N)",
            "cmap": "viridis",
        },
    }

    cfg = metric_configs.get(metric_key, metric_configs["lambda_diff"])
    fig = plt.figure(figsize=(14, 9), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    surf = render_surface_on_ax(ax, pts, cfg["z_label"], cfg["title"], cmap_name=cfg["cmap"])
    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, pad=0.08)
    cbar.set_label(cfg["cbar_label"], fontsize=12, labelpad=10)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved {metric_key} plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_combined_summary(iter_data, outer_iter, output_path="admm_convergence_summary_3d.png", show=False):
    fig = plt.figure(figsize=(26, 8), dpi=150)

    # 1. Iterate Step Change
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = render_surface_on_ax(
        ax1, iter_data["iterate_step"],
        r"$\|\delta^k - \delta^{k-1}\|_2$",
        "1. Iterate Step Change (Stationarity)\n$\\|\delta^k - \delta^{k-1}\\|_2$",
        cmap_name="plasma"
    )
    cb1 = fig.colorbar(surf1, ax=ax1, shrink=0.55, aspect=12, pad=0.08)
    cb1.set_label(r"Mean $\|\Delta \delta\|_2$", fontsize=10)

    # 2. Complementarity Slackness
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = render_surface_on_ax(
        ax2, iter_data["complementarity"],
        r"$\lambda^T \eta$",
        "2. Complementarity Slackness\n$\\lambda^T \\eta$",
        cmap_name="magma"
    )
    cb2 = fig.colorbar(surf2, ax=ax2, shrink=0.55, aspect=12, pad=0.08)
    cb2.set_label(r"Mean $\sum |\lambda \cdot \eta|$", fontsize=10)

    # 3. Force Consensus Error
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = render_surface_on_ax(
        ax3, iter_data["lambda_diff"],
        r"$\|\lambda - \delta_\lambda\|_2$",
        "3. Force Consensus Error\n$\\|\\lambda - \\delta_\\lambda\\|_2$",
        cmap_name="viridis"
    )
    cb3 = fig.colorbar(surf3, ax=ax3, shrink=0.55, aspect=12, pad=0.08)
    cb3.set_label(r"Mean $\|\lambda - \delta_\lambda\|_2$ (N)", fontsize=10)

    plt.suptitle(f"C3 ADMM Convergence Diagnostic Suite (Final Outer Iteration {outer_iter})", fontsize=17, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.88, wspace=0.15)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved combined summary plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot C3 ADMM convergence metrics in 3D.")
    parser.add_argument("--csv", type=str, default=None, help="Path to input lambda_admm_residuals.csv")
    parser.add_argument("--out", type=str, default=None, help="Path to save output 3D plot PNG")
    parser.add_argument("--metric", type=str, default="all", choices=["all", "iterate_step", "complementarity", "lambda_diff"], help="Metric to plot")
    parser.add_argument("--iter", type=int, default=None, help="Outer iteration to plot (default: final iteration)")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    parser.add_argument("--watch", action="store_true", help="Watch CSV file and re-plot on update")
    parser.add_argument("--interval", type=float, default=2.0, help="Watch polling interval (seconds)")
    args = parser.parse_args()

    csv_path = args.csv or find_default_csv()

    records_by_iter, max_outer = load_all_metrics(csv_path)
    if not records_by_iter:
        print(f"Error: Could not read data from {csv_path}")
        sys.exit(1)

    target_iter = args.iter if args.iter is not None else max_outer
    iter_data = records_by_iter[target_iter]

    if args.metric == "all":
        # Generate individual plots for each metric and the combined summary
        plot_single_metric(iter_data["iterate_step"], "iterate_step", target_iter, output_path="iterate_step_change_3d.png", show=False)
        plot_single_metric(iter_data["complementarity"], "complementarity", target_iter, output_path="complementarity_slack_3d.png", show=False)
        plot_single_metric(iter_data["lambda_diff"], "lambda_diff", target_iter, output_path="lambda_admm_residuals_3d.png", show=False)
        plot_combined_summary(iter_data, target_iter, output_path="admm_convergence_summary_3d.png", show=args.show)
    else:
        out_name = args.out or f"{args.metric}_3d.png"
        plot_single_metric(iter_data[args.metric], args.metric, target_iter, output_path=out_name, show=args.show)


if __name__ == "__main__":
    main()
