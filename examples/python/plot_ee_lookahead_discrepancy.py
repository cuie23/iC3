#!/usr/bin/env python3
"""
Plot C3 End-Effector Lookahead Discrepancy vs Actual Trajectory Step Change.

Computes and plots:
  x-axis: Plan Timestep (t)
  y-axis: ||(x_1(t) - x_0(t))_ee||_2 - ||(x_traj(t+1) - x_traj(t))_ee||_2
          where "_ee" denotes the end-effector state coordinates
          (first 9 elements for Point Hand 180/pivot, first 2 for plate).

Usage:
  python3 examples/python/plot_ee_lookahead_discrepancy.py
  python3 examples/python/plot_ee_lookahead_discrepancy.py --csv c3_x0_x1_plan.csv --out ee_lookahead_discrepancy.png
"""

import argparse
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def find_default_csv():
    candidates = [
        "c3_x0_x1_plan.csv",
        "c3_full_lookahead_plan.csv",
        "examples/resources/multifinger_hand/ic3_debug_data/c3_x0_x1_plan.csv",
        "examples/resources/multifinger_hand/ic3_debug_data/c3_full_lookahead_plan.csv",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 0:
            return c
    return candidates[0]


def load_plan_data(csv_path, target_outer_iter=None):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None, None, None, 0

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return None, None, None, 0

        is_full_csv = "lookahead_step" in header
        iter_idx = header.index("iteration")
        ts_idx = header.index("plan_timestep")

        rows_by_iter = {}

        if is_full_csv:
            k_idx = header.index("lookahead_step")
            x_cols = [i for i, h in enumerate(header) if h.startswith("x_")]
            for row in reader:
                if not row or len(row) < len(header):
                    continue
                try:
                    it = int(row[iter_idx])
                    ts = int(row[ts_idx])
                    k = int(row[k_idx])
                    if k > 1:
                        continue
                    x_vec = np.array([float(row[c]) for c in x_cols])
                    if it not in rows_by_iter:
                        rows_by_iter[it] = {}
                    if ts not in rows_by_iter[it]:
                        rows_by_iter[it][ts] = {}
                    rows_by_iter[it][ts][k] = x_vec
                except (ValueError, IndexError):
                    continue

            if not rows_by_iter:
                return None, None, None, 0

            chosen_it = target_outer_iter if target_outer_iter is not None else max(rows_by_iter.keys())
            if chosen_it not in rows_by_iter:
                chosen_it = max(rows_by_iter.keys())

            iter_dict = rows_by_iter[chosen_it]
            sorted_ts = sorted(iter_dict.keys())
            x0_list = []
            x1_list = []
            valid_ts = []
            for ts in sorted_ts:
                if 0 in iter_dict[ts] and 1 in iter_dict[ts]:
                    valid_ts.append(ts)
                    x0_list.append(iter_dict[ts][0])
                    x1_list.append(iter_dict[ts][1])

            return np.array(valid_ts), np.array(x0_list), np.array(x1_list), chosen_it

        else:
            x0_cols = [i for i, h in enumerate(header) if h.startswith("x0_")]
            x1_cols = [i for i, h in enumerate(header) if h.startswith("x1_")]
            for row in reader:
                if not row or len(row) < len(header):
                    continue
                try:
                    it = int(row[iter_idx])
                    ts = int(row[ts_idx])
                    x0_vec = np.array([float(row[c]) for c in x0_cols])
                    x1_vec = np.array([float(row[c]) for c in x1_cols])
                    if it not in rows_by_iter:
                        rows_by_iter[it] = []
                    rows_by_iter[it].append((ts, x0_vec, x1_vec))
                except (ValueError, IndexError):
                    continue

            if not rows_by_iter:
                return None, None, None, 0

            chosen_it = target_outer_iter if target_outer_iter is not None else max(rows_by_iter.keys())
            if chosen_it not in rows_by_iter:
                chosen_it = max(rows_by_iter.keys())

            records = sorted(rows_by_iter[chosen_it], key=lambda r: r[0])
            valid_ts = np.array([r[0] for r in records])
            x0_mat = np.array([r[1] for r in records])
            x1_mat = np.array([r[2] for r in records])
            return valid_ts, x0_mat, x1_mat, chosen_it


def plot_ee_discrepancy(timesteps, x0_mat, x1_mat, chosen_it, n_ee=None, out_path=None, show=False):
    T, n_x = x0_mat.shape

    if n_ee is None:
        if n_x >= 31:
            n_ee = 9  # Point Hand 180 / Pivot: 3 fingers x 3 joints
        elif n_x >= 23:
            n_ee = 2  # Plate: plate x, y coordinates
        else:
            n_ee = min(9, n_x)

    print(f"Plotting EE Lookahead Discrepancy for outer iteration {chosen_it}")
    print(f"Total timesteps: {T}, State dimension n_x: {n_x}, End-effector dimension n_ee: {n_ee}")

    # Extract end-effector states
    ee_x0 = x0_mat[:, :n_ee]
    ee_x1 = x1_mat[:, :n_ee]

    # 1. C3 internal 1-step lookahead jump norm at timestep t: ||x_1(t) - x_0(t)||_2
    c3_ee_diff = ee_x1 - ee_x0
    c3_ee_jump_norm = np.linalg.norm(c3_ee_diff, axis=1)

    # 2. Actual executed trajectory step norm from timestep t to t+1: ||x_traj(t+1) - x_traj(t)||_2
    # Since x0(t) is the trajectory state at timestep t:
    traj_ee_diff = ee_x0[1:] - ee_x0[:-1]
    traj_ee_step_norm = np.linalg.norm(traj_ee_diff, axis=1)

    # Align to length T - 1 (timesteps 0 to T - 2)
    t_eval = timesteps[:-1]
    c3_eval = c3_ee_jump_norm[:-1]
    traj_eval = traj_ee_step_norm
    discrepancy = c3_eval - traj_eval

    # Statistics
    mean_val = np.mean(discrepancy)
    std_val = np.std(discrepancy)
    max_val = np.max(discrepancy)
    min_val = np.min(discrepancy)
    mae_val = np.mean(np.abs(discrepancy))

    # Configure dark theme styling
    plt.style.use('dark_background')
    fig, (ax_main, ax_comp) = plt.subplots(2, 1, figsize=(14, 8.5), gridspec_kw={'height_ratios': [1.4, 1.0]}, sharex=True)
    fig.patch.set_facecolor('#0f111a')

    for ax in [ax_main, ax_comp]:
        ax.set_facecolor('#1a1c29')
        ax.tick_params(colors='white', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('#444b6e')
        ax.grid(True, linestyle='--', color='#2e3450', alpha=0.7)

    # -------------------------------------------------------------------------
    # Panel 1: Main Metric ||(x1 - x0)_ee|| - ||(x_{t+1} - x_t)_ee||
    # -------------------------------------------------------------------------
    ax_main.plot(t_eval, discrepancy, color='#ff79c6', lw=2.0, label=r'$\|(x_1(t) - x_0(t))_{\mathrm{ee}}\|_2 - \|(x_{\mathrm{traj}}(t+1) - x_{\mathrm{traj}}(t))_{\mathrm{ee}}\|_2$')
    ax_main.axhline(0.0, color='#8be9fd', linestyle='--', lw=1.5, alpha=0.85, label='Zero Discrepancy (Perfect Lookahead Match)')

    # Shaded positive vs negative mismatch regions
    ax_main.fill_between(t_eval, 0, discrepancy, where=(discrepancy >= 0), color='#ff5555', alpha=0.25, label='Lookahead Jump > Actual Step')
    ax_main.fill_between(t_eval, 0, discrepancy, where=(discrepancy < 0), color='#50fa7b', alpha=0.25, label='Lookahead Jump < Actual Step')

    # Mean line
    ax_main.axhline(mean_val, color='#f1fa8c', linestyle=':', lw=1.5, alpha=0.9, label=f'Mean Discrepancy: {mean_val:+.4f}')

    ax_main.set_title(
        f"C3 End-Effector Lookahead Discrepancy (First {n_ee} States) vs Plan Timestep\n"
        f"[Final Outer Iteration {chosen_it}]  |  MAE: {mae_val:.4f}  |  Std: {std_val:.4f}  |  Min/Max: [{min_val:+.4f}, {max_val:+.4f}]",
        color='white', fontsize=12, fontweight='bold', pad=12
    )
    ax_main.set_ylabel(r"$\Delta_{\mathrm{C3}} - \Delta_{\mathrm{traj}}$ (m)", color='white', fontsize=11, fontweight='bold')
    ax_main.legend(loc='upper right', facecolor='#1a1c29', edgecolor='#444b6e', labelcolor='white', fontsize=9.5)

    # -------------------------------------------------------------------------
    # Panel 2: Component Breakdown (C3 Jump vs Actual Trajectory Step)
    # -------------------------------------------------------------------------
    ax_comp.plot(t_eval, c3_eval, color='#ffb86c', lw=1.8, label=r'$\Delta_{\mathrm{C3}} = \|(x_1(t) - x_0(t))_{\mathrm{ee}}\|_2$ (C3 Planned Lookahead Jump)')
    ax_comp.plot(t_eval, traj_eval, color='#50fa7b', lw=1.8, linestyle='--', label=r'$\Delta_{\mathrm{traj}} = \|(x_{\mathrm{traj}}(t+1) - x_{\mathrm{traj}}(t))_{\mathrm{ee}}\|_2$ (Executed Trajectory Step)')
    ax_comp.set_title("Component Step Norm Comparison", color='white', fontsize=11, fontweight='bold', pad=8)
    ax_comp.set_xlabel("Plan Timestep (t)", color='white', fontsize=11, fontweight='bold')
    ax_comp.set_ylabel("Step Norm (m)", color='white', fontsize=11, fontweight='bold')
    ax_comp.legend(loc='upper right', facecolor='#1a1c29', edgecolor='#444b6e', labelcolor='white', fontsize=9.5)

    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
        print(f"Saved plot to {out_path}")

    if show:
        plt.show()
    elif not out_path:
        default_save = "ee_lookahead_discrepancy.png"
        plt.savefig(default_save, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
        print(f"Saved plot to {default_save}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot C3 End-Effector Lookahead Discrepancy vs Plan Timestep.")
    parser.add_argument("--csv", type=str, default=None, help="Path to c3_x0_x1_plan.csv or c3_full_lookahead_plan.csv")
    parser.add_argument("--iter", type=int, default=None, help="Outer iteration to plot (default: final iteration)")
    parser.add_argument("--nee", type=int, default=None, help="Number of end-effector elements (default: 9 for Point Hand, 2 for Plate)")
    parser.add_argument("--out", type=str, default="ee_lookahead_discrepancy.png", help="Path to save output plot PNG")
    parser.add_argument("--show", action="store_true", default=False, help="Display interactive plot window")
    args = parser.parse_args()

    csv_path = args.csv or find_default_csv()
    timesteps, x0_mat, x1_mat, chosen_it = load_plan_data(csv_path, target_outer_iter=args.iter)

    if timesteps is None or len(timesteps) < 2:
        print(f"Error: Could not load sufficient plan data from {csv_path}")
        print("Run an MS-iC3 simulation first to generate c3_x0_x1_plan.csv")
        sys.exit(1)

    plot_ee_discrepancy(
        timesteps, x0_mat, x1_mat, chosen_it,
        n_ee=args.nee, out_path=args.out, show=args.show
    )


if __name__ == "__main__":
    main()
