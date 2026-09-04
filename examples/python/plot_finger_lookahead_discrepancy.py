#!/usr/bin/env python3
"""
Plot Per-Finger C3 Lookahead Discrepancy and Contact Forces (Lambda / Eta) for the Last State in the C3 Trajectory.

Computes and plots per finger (Finger 1 = Red, Finger 2 = Green, Finger 3 = Blue):
  1. Lookahead Discrepancy (Left column):
     ||(x_1(t) - x_0(t))_finger||_2 - ||(x_traj(t+1) - x_traj(t))_finger||_2
     where:
       Finger 1: states x[0:3]
       Finger 2: states x[3:6]
       Finger 3: states x[6:9]
  2. Contact Reactions (Right column):
     Sum of the 4-component block for lambda (contact force) and eta (complementarity slack)
     of the last state in the C3 planned lookahead trajectory:
       Finger 1: sum(lambda_last[0:4]) and sum(eta_last[0:4])
       Finger 2: sum(lambda_last[4:8]) and sum(eta_last[4:8])
       Finger 3: sum(lambda_last[8:12]) and sum(eta_last[8:12])

Usage:
  python3 examples/python/plot_finger_lookahead_discrepancy.py
  python3 examples/python/plot_finger_lookahead_discrepancy.py --csv c3_x0_x1_plan.csv --show
  python3 examples/python/plot_finger_lookahead_discrepancy.py --state last --out finger_lookahead_discrepancy.png
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
    """
    Loads plan data including x0, x1, lambda, eta, lambda_last, and eta_last from CSV.
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None, None, None, None, None, None, None, 0

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return None, None, None, None, None, None, None, 0

        is_full_csv = "lookahead_step" in header
        iter_idx = header.index("iteration")
        ts_idx = header.index("plan_timestep")

        rows_by_iter = {}

        if is_full_csv:
            k_idx = header.index("lookahead_step")
            x_cols = [i for i, h in enumerate(header) if h.startswith("x_")]
            lam_cols = [i for i, h in enumerate(header) if h.startswith("lambda_") and not h.startswith("lambda_last_")]
            eta_cols = [i for i, h in enumerate(header) if h.startswith("eta_") and not h.startswith("eta_last_")]

            for row in reader:
                if not row or len(row) < len(header):
                    continue
                try:
                    it = int(row[iter_idx])
                    ts = int(row[ts_idx])
                    k = int(row[k_idx])
                    x_vec = np.array([float(row[c]) for c in x_cols])
                    lam_vec = np.array([float(row[c]) for c in lam_cols]) if lam_cols else np.zeros(12)
                    eta_vec = np.array([float(row[c]) for c in eta_cols]) if eta_cols else np.zeros(12)

                    if it not in rows_by_iter:
                        rows_by_iter[it] = {}
                    if ts not in rows_by_iter[it]:
                        rows_by_iter[it][ts] = {}
                    rows_by_iter[it][ts][k] = (x_vec, lam_vec, eta_vec)
                except (ValueError, IndexError):
                    continue

            if not rows_by_iter:
                return None, None, None, None, None, None, None, 0

            chosen_it = target_outer_iter if target_outer_iter is not None else max(rows_by_iter.keys())
            if chosen_it not in rows_by_iter:
                chosen_it = max(rows_by_iter.keys())

            iter_dict = rows_by_iter[chosen_it]
            sorted_ts = sorted(iter_dict.keys())
            x0_list, x1_list, lam_list, eta_list, lam_last_list, eta_last_list, valid_ts = [], [], [], [], [], [], []
            for ts in sorted_ts:
                k_keys = sorted(iter_dict[ts].keys())
                if 0 in iter_dict[ts] and len(k_keys) > 1:
                    valid_ts.append(ts)
                    x0_list.append(iter_dict[ts][0][0])
                    x1_list.append(iter_dict[ts][1][0])
                    lam_list.append(iter_dict[ts][0][1])
                    eta_list.append(iter_dict[ts][0][2])
                    max_k = max(k_keys)
                    lam_last_list.append(iter_dict[ts][max_k][1])
                    eta_last_list.append(iter_dict[ts][max_k][2])

            return (
                np.array(valid_ts),
                np.array(x0_list),
                np.array(x1_list),
                np.array(lam_list),
                np.array(eta_list),
                np.array(lam_last_list),
                np.array(eta_last_list),
                chosen_it,
            )

        else:
            x0_cols = [i for i, h in enumerate(header) if h.startswith("x0_")]
            x1_cols = [i for i, h in enumerate(header) if h.startswith("x1_")]
            lam_cols = [i for i, h in enumerate(header) if h.startswith("lambda_") and not h.startswith("lambda_last_")]
            eta_cols = [i for i, h in enumerate(header) if h.startswith("eta_") and not h.startswith("eta_last_")]
            lam_last_cols = [i for i, h in enumerate(header) if h.startswith("lambda_last_")]
            eta_last_cols = [i for i, h in enumerate(header) if h.startswith("eta_last_")]

            for row in reader:
                if not row or len(row) < len(header):
                    continue
                try:
                    it = int(row[iter_idx])
                    ts = int(row[ts_idx])
                    x0_vec = np.array([float(row[c]) for c in x0_cols])
                    x1_vec = np.array([float(row[c]) for c in x1_cols])
                    lam_vec = np.array([float(row[c]) for c in lam_cols]) if lam_cols else np.zeros(12)
                    eta_vec = np.array([float(row[c]) for c in eta_cols]) if eta_cols else np.zeros(12)
                    lam_last_vec = np.array([float(row[c]) for c in lam_last_cols]) if lam_last_cols else lam_vec
                    eta_last_vec = np.array([float(row[c]) for c in eta_last_cols]) if eta_last_cols else eta_vec

                    if it not in rows_by_iter:
                        rows_by_iter[it] = []
                    rows_by_iter[it].append((ts, x0_vec, x1_vec, lam_vec, eta_vec, lam_last_vec, eta_last_vec))
                except (ValueError, IndexError):
                    continue

            if not rows_by_iter:
                return None, None, None, None, None, None, None, 0

            chosen_it = target_outer_iter if target_outer_iter is not None else max(rows_by_iter.keys())
            if chosen_it not in rows_by_iter:
                chosen_it = max(rows_by_iter.keys())

            records = sorted(rows_by_iter[chosen_it], key=lambda r: r[0])
            valid_ts = np.array([r[0] for r in records])
            x0_mat = np.array([r[1] for r in records])
            x1_mat = np.array([r[2] for r in records])
            lam_mat = np.array([r[3] for r in records])
            eta_mat = np.array([r[4] for r in records])
            lam_last_mat = np.array([r[5] for r in records])
            eta_last_mat = np.array([r[6] for r in records])
            return valid_ts, x0_mat, x1_mat, lam_mat, eta_mat, lam_last_mat, eta_last_mat, chosen_it


def plot_finger_discrepancy_and_forces(
    timesteps, x0_mat, x1_mat, lam_mat, eta_mat, lam_last_mat, eta_last_mat,
    chosen_it, state_choice="last", out_path=None, show=False
):
    T, n_x = x0_mat.shape
    t_eval = timesteps[:-1]

    # Finger configuration: 3 fingers, 3 state dimensions each, 4 contact components each
    finger_configs = [
        {"name": "Finger 1", "state_slice": slice(0, 3), "force_slice": slice(0, 4), "color": "#ff5555", "accent": "#ff79c6"},
        {"name": "Finger 2", "state_slice": slice(3, 6), "force_slice": slice(4, 8), "color": "#50fa7b", "accent": "#8be9fd"},
        {"name": "Finger 3", "state_slice": slice(6, 9), "force_slice": slice(8, 12), "color": "#8be9fd", "accent": "#bd93f9"},
    ]

    # Select which lambda and eta to plot (last state vs first state vs both)
    use_last = (state_choice in ["last", "both"])
    use_first = (state_choice in ["first", "both"])

    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex=True)
    fig.patch.set_facecolor("#0f111a")

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("#1a1c29")
            ax.tick_params(colors="white", labelsize=9.5)
            for spine in ax.spines.values():
                spine.set_color("#444b6e")
            ax.grid(True, linestyle="--", color="#2e3450", alpha=0.7)

    for i, cfg in enumerate(finger_configs):
        ax_disc = axes[i, 0]
        ax_force = axes[i, 1]

        # 1. Lookahead Discrepancy for Finger i
        f_x0 = x0_mat[:, cfg["state_slice"]]
        f_x1 = x1_mat[:, cfg["state_slice"]]

        c3_jump = np.linalg.norm(f_x1 - f_x0, axis=1)[:-1]
        traj_step = np.linalg.norm(f_x0[1:] - f_x0[:-1], axis=1)
        discrepancy = c3_jump - traj_step

        mae = np.mean(np.abs(discrepancy))

        # Plot Discrepancy curve
        ax_disc.plot(
            t_eval,
            discrepancy,
            color=cfg["color"],
            lw=2.0,
            label=rf"$\Delta_{{C3}} - \Delta_{{traj}}$ (MAE={mae:.4f})",
        )
        ax_disc.axhline(0.0, color="#6272a4", linestyle="--", lw=1.2, alpha=0.8)
        ax_disc.fill_between(
            t_eval,
            0,
            discrepancy,
            where=(discrepancy >= 0),
            color=cfg["color"],
            alpha=0.2,
        )
        ax_disc.fill_between(
            t_eval,
            0,
            discrepancy,
            where=(discrepancy < 0),
            color="#6272a4",
            alpha=0.15,
        )

        # Plot C3 planned jump and Trajectory step for reference
        ax_disc.plot(
            t_eval,
            c3_jump,
            color="#f1fa8c",
            lw=1.2,
            linestyle=":",
            alpha=0.8,
            label=r"$\Delta_{C3} = \|(x_1 - x_0)_f\|_2$",
        )
        ax_disc.plot(
            t_eval,
            traj_step,
            color=cfg["accent"],
            lw=1.2,
            linestyle="--",
            alpha=0.8,
            label=r"$\Delta_{traj} = \|(x_{t+1} - x_t)_f\|_2$",
        )

        ax_disc.set_title(
            f"{cfg['name']} (Joints {cfg['state_slice'].start+1}-{cfg['state_slice'].stop}) Lookahead Discrepancy",
            color="white",
            fontsize=11,
            fontweight="bold",
            pad=6,
        )
        ax_disc.set_ylabel(r"$\Delta$ Norm (m / rad)", color="white", fontsize=9.5)
        ax_disc.legend(
            loc="upper right",
            facecolor="#1a1c29",
            edgecolor="#444b6e",
            labelcolor="white",
            fontsize=8.5,
        )

        # 2. Contact Force (Lambda) and Slack (Eta) Sums for Finger i
        # Last state in C3 lookahead trajectory
        if use_last:
            target_lam = lam_last_mat if lam_last_mat is not None else lam_mat
            target_eta = eta_last_mat if eta_last_mat is not None else eta_mat
            if target_lam is not None and target_lam.shape[1] >= cfg["force_slice"].stop:
                f_lam_last = target_lam[:, cfg["force_slice"]]
                lam_last_sum = np.sum(f_lam_last, axis=1)[:-1]
            else:
                lam_last_sum = np.zeros_like(t_eval)

            if target_eta is not None and target_eta.shape[1] >= cfg["force_slice"].stop:
                f_eta_last = target_eta[:, cfg["force_slice"]]
                eta_last_sum = np.sum(f_eta_last, axis=1)[:-1]
            else:
                eta_last_sum = np.zeros_like(t_eval)

            ax_force.plot(
                t_eval,
                lam_last_sum,
                color="#ffb86c",
                lw=1.8,
                label=rf"$\sum \lambda_{{last}}$ Contact Force (Last State, Comps {cfg['force_slice'].start+1}-{cfg['force_slice'].stop})",
            )
            ax_force.plot(
                t_eval,
                eta_last_sum,
                color="#bd93f9",
                lw=1.5,
                linestyle="--",
                label=rf"$\sum \eta_{{last}}$ Slack / Comp. (Last State, Comps {cfg['force_slice'].start+1}-{cfg['force_slice'].stop})",
            )

        if use_first and state_choice == "both":
            if lam_mat is not None and lam_mat.shape[1] >= cfg["force_slice"].stop:
                f_lam_first = lam_mat[:, cfg["force_slice"]]
                lam_first_sum = np.sum(f_lam_first, axis=1)[:-1]
            else:
                lam_first_sum = np.zeros_like(t_eval)

            if eta_mat is not None and eta_mat.shape[1] >= cfg["force_slice"].stop:
                f_eta_first = eta_mat[:, cfg["force_slice"]]
                eta_first_sum = np.sum(f_eta_first, axis=1)[:-1]
            else:
                eta_first_sum = np.zeros_like(t_eval)

            ax_force.plot(
                t_eval,
                lam_first_sum,
                color="#ff5555",
                lw=1.2,
                linestyle=":",
                alpha=0.75,
                label=r"$\sum \lambda_{0}$ (First State)",
            )
            ax_force.plot(
                t_eval,
                eta_first_sum,
                color="#50fa7b",
                lw=1.2,
                linestyle=":",
                alpha=0.75,
                label=r"$\sum \eta_{0}$ (First State)",
            )
        elif use_first and state_choice == "first":
            if lam_mat is not None and lam_mat.shape[1] >= cfg["force_slice"].stop:
                f_lam_first = lam_mat[:, cfg["force_slice"]]
                lam_first_sum = np.sum(f_lam_first, axis=1)[:-1]
            else:
                lam_first_sum = np.zeros_like(t_eval)

            if eta_mat is not None and eta_mat.shape[1] >= cfg["force_slice"].stop:
                f_eta_first = eta_mat[:, cfg["force_slice"]]
                eta_first_sum = np.sum(f_eta_first, axis=1)[:-1]
            else:
                eta_first_sum = np.zeros_like(t_eval)

            ax_force.plot(
                t_eval,
                lam_first_sum,
                color="#ffb86c",
                lw=1.8,
                label=rf"$\sum \lambda_{{0}}$ Contact Force (First State, Comps {cfg['force_slice'].start+1}-{cfg['force_slice'].stop})",
            )
            ax_force.plot(
                t_eval,
                eta_first_sum,
                color="#bd93f9",
                lw=1.5,
                linestyle="--",
                label=rf"$\sum \eta_{{0}}$ Slack / Comp. (First State, Comps {cfg['force_slice'].start+1}-{cfg['force_slice'].stop})",
            )

        title_suffix = "Last State in C3 Trajectory" if use_last else "First State in C3 Trajectory"
        ax_force.set_title(
            f"{cfg['name']} Contact Reactions: $\sum \lambda$ & $\sum \eta$ ({title_suffix})",
            color="white",
            fontsize=10.5,
            fontweight="bold",
            pad=6,
        )
        ax_force.set_ylabel("Force / Slack Sum", color="white", fontsize=9.5)
        ax_force.legend(
            loc="upper right",
            facecolor="#1a1c29",
            edgecolor="#444b6e",
            labelcolor="white",
            fontsize=8.0,
        )

    axes[2, 0].set_xlabel("Plan Timestep (t)", color="white", fontsize=10.5, fontweight="bold")
    axes[2, 1].set_xlabel("Plan Timestep (t)", color="white", fontsize=10.5, fontweight="bold")

    subtitle_state = "Last State in C3 Lookahead Plan" if state_choice == "last" else ("First State" if state_choice == "first" else "First & Last States")
    fig.suptitle(
        f"Per-Finger C3 Lookahead Discrepancy & Contact Reactions ({subtitle_state}) [Iteration {chosen_it}]\n"
        r"Discrepancy $= \|(x_1(t) - x_0(t))_f\|_2 - \|(x_{traj}(t+1) - x_{traj}(t))_f\|_2$",
        color="white",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"Saved finger lookahead discrepancy plot to {out_path}")

    if show:
        plt.show()
    elif not out_path:
        default_save = "finger_lookahead_discrepancy.png"
        plt.savefig(default_save, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"Saved finger lookahead discrepancy plot to {default_save}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-finger C3 lookahead discrepancy and contact reactions of the last/first state in C3 trajectory."
    )
    parser.add_argument("--csv", type=str, default=None, help="Path to c3_x0_x1_plan.csv")
    parser.add_argument("--iter", type=int, default=None, help="Outer iteration to plot (default: final iteration)")
    parser.add_argument(
        "--state",
        type=str,
        choices=["last", "first", "both"],
        default="last",
        help="Which state in the C3 lookahead trajectory to plot for lambda/eta: 'last' (default), 'first', or 'both'",
    )
    parser.add_argument("--out", type=str, default="finger_lookahead_discrepancy.png", help="Path to save output plot PNG")
    parser.add_argument("--show", action="store_true", default=False, help="Display interactive plot window")
    args = parser.parse_args()

    csv_path = args.csv or find_default_csv()
    timesteps, x0_mat, x1_mat, lam_mat, eta_mat, lam_last_mat, eta_last_mat, chosen_it = load_plan_data(
        csv_path, target_outer_iter=args.iter
    )

    if timesteps is None or len(timesteps) < 2:
        print(f"Error: Could not load sufficient plan data from {csv_path}")
        print("Run an MS-iC3 simulation first to generate c3_x0_x1_plan.csv")
        sys.exit(1)

    plot_finger_discrepancy_and_forces(
        timesteps,
        x0_mat,
        x1_mat,
        lam_mat,
        eta_mat,
        lam_last_mat,
        eta_last_mat,
        chosen_it,
        state_choice=args.state,
        out_path=args.out,
        show=args.show,
    )


if __name__ == "__main__":
    main()
