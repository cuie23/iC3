#!/usr/bin/env python3
"""
Plot Contact Reactions (Lambda / Eta) from the Final QP Step vs Final Projection Step.

Reads:
  - Final QP Step: z_sol.csv (or all_z_sols.bin exported to CSV)
  - Final Projection Step: delta_projections.csv (or all_delta_projections.bin exported to CSV)

Computes and plots per finger (Finger 1 = Red, Finger 2 = Green, Finger 3 = Blue):
  Left Column: Lambda (Contact Force) - Final QP Step vs Final Projection Step
  Right Column: Eta (Complementarity Slack) - Final QP Step vs Final Projection Step

Usage:
  python3 examples/python/plot_qp_vs_projection.py
  python3 examples/python/plot_qp_vs_projection.py --show
  python3 examples/python/plot_qp_vs_projection.py --out qp_vs_projection.png
"""

import argparse
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def find_file(filename, fallback_paths):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        return filename
    for p in fallback_paths:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


def load_csv_lambda_eta(csv_path):
    if not os.path.exists(csv_path):
        return None, None, None

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return None, None, None

        ts_idx = header.index("timestep") if "timestep" in header else 0
        lam_cols = [i for i, h in enumerate(header) if h.startswith("lambda_") and not h.startswith("lambda_last_")]
        eta_cols = [i for i, h in enumerate(header) if h.startswith("eta_") and not h.startswith("eta_last_")]

        timesteps = []
        lambdas = []
        etas = []

        for row in reader:
            if not row or len(row) < len(header):
                continue
            try:
                t = int(row[ts_idx])
                lam = [float(row[c]) for c in lam_cols]
                eta = [float(row[c]) for c in eta_cols]
                timesteps.append(t)
                lambdas.append(lam)
                etas.append(eta)
            except (ValueError, IndexError):
                continue

        return np.array(timesteps), np.array(lambdas), np.array(etas)


def plot_qp_vs_projection(
    t_qp, lam_qp, eta_qp, t_proj, lam_proj, eta_proj, mode="sum", out_path=None, show=False
):
    # Align timesteps
    min_len = min(len(t_qp), len(t_proj))
    t = t_qp[:min_len]
    lam_qp = lam_qp[:min_len]
    eta_qp = eta_qp[:min_len]
    lam_proj = lam_proj[:min_len]
    eta_proj = eta_proj[:min_len]

    finger_configs = [
        {"name": "Finger 1", "slice": slice(0, 4), "color": "#ff5555", "accent": "#ff79c6"},
        {"name": "Finger 2", "slice": slice(4, 8), "color": "#50fa7b", "accent": "#8be9fd"},
        {"name": "Finger 3", "slice": slice(8, 12), "color": "#8be9fd", "accent": "#bd93f9"},
    ]

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
        ax_lam = axes[i, 0]
        ax_eta = axes[i, 1]

        # Extract finger slices
        f_lam_qp = lam_qp[:, cfg["slice"]]
        f_lam_proj = lam_proj[:, cfg["slice"]]
        f_eta_qp = eta_qp[:, cfg["slice"]]
        f_eta_proj = eta_proj[:, cfg["slice"]]

        if mode == "sum":
            y_lam_qp = np.sum(f_lam_qp, axis=1)
            y_lam_proj = np.sum(f_lam_proj, axis=1)
            y_eta_qp = np.sum(f_eta_qp, axis=1)
            y_eta_proj = np.sum(f_eta_proj, axis=1)
            unit_str = "Sum"
        else:
            y_lam_qp = np.linalg.norm(f_lam_qp, axis=1)
            y_lam_proj = np.linalg.norm(f_lam_proj, axis=1)
            y_eta_qp = np.linalg.norm(f_eta_qp, axis=1)
            y_eta_proj = np.linalg.norm(f_eta_proj, axis=1)
            unit_str = "Norm"

        lam_diff = np.abs(y_lam_qp - y_lam_proj)
        eta_diff = np.abs(y_eta_qp - y_eta_proj)

        # ----------------- Left Column: Lambda (QP vs Proj) -----------------
        ax_lam.plot(
            t,
            y_lam_qp,
            color="#ffb86c",
            lw=1.8,
            label=rf"$\lambda_{{\mathrm{{QP}}}}$ (Final QP Step, MAE diff={np.mean(lam_diff):.4f})",
        )
        ax_lam.plot(
            t,
            y_lam_proj,
            color=cfg["color"],
            lw=1.6,
            linestyle="--",
            label=r"$\lambda_{{\mathrm{{Proj}}}}$ (Final Projection Step)",
        )
        ax_lam.fill_between(t, y_lam_qp, y_lam_proj, color=cfg["color"], alpha=0.18, label="QP vs Proj Gap")

        ax_lam.set_title(
            f"{cfg['name']} Contact Force $\lambda$ (Components {cfg['slice'].start+1}-{cfg['slice'].stop}): Final QP vs Final Proj",
            color="white",
            fontsize=10.5,
            fontweight="bold",
            pad=6,
        )
        ax_lam.set_ylabel(f"$\lambda$ {unit_str}", color="white", fontsize=9.5)
        ax_lam.legend(loc="upper right", facecolor="#1a1c29", edgecolor="#444b6e", labelcolor="white", fontsize=8.5)

        # ----------------- Right Column: Eta (QP vs Proj) -----------------
        ax_eta.plot(
            t,
            y_eta_qp,
            color="#bd93f9",
            lw=1.8,
            label=rf"$\eta_{{\mathrm{{QP}}}}$ (Final QP Step, MAE diff={np.mean(eta_diff):.4f})",
        )
        ax_eta.plot(
            t,
            y_eta_proj,
            color=cfg["accent"],
            lw=1.6,
            linestyle="--",
            label=r"$\eta_{{\mathrm{{Proj}}}}$ (Final Projection Step)",
        )
        ax_eta.fill_between(t, y_eta_qp, y_eta_proj, color=cfg["accent"], alpha=0.18, label="QP vs Proj Gap")

        ax_eta.set_title(
            f"{cfg['name']} Complementarity Slack $\eta$ (Components {cfg['slice'].start+1}-{cfg['slice'].stop}): Final QP vs Final Proj",
            color="white",
            fontsize=10.5,
            fontweight="bold",
            pad=6,
        )
        ax_eta.set_ylabel(f"$\eta$ {unit_str}", color="white", fontsize=9.5)
        ax_eta.legend(loc="upper right", facecolor="#1a1c29", edgecolor="#444b6e", labelcolor="white", fontsize=8.5)

    axes[2, 0].set_xlabel("Plan Timestep (t)", color="white", fontsize=10.5, fontweight="bold")
    axes[2, 1].set_xlabel("Plan Timestep (t)", color="white", fontsize=10.5, fontweight="bold")

    fig.suptitle(
        f"Per-Finger Contact Reactions: Final QP Step vs. Final Projection Step ({unit_str} of 4-component blocks)\n"
        r"$\lambda$: Contact Normal/Tangential Force  |  $\eta$: Complementarity / Penetration Slack",
        color="white",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"Saved QP vs Projection comparison plot to {out_path}")

    if show:
        plt.show()
    elif not out_path:
        default_save = "qp_vs_projection.png"
        plt.savefig(default_save, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"Saved QP vs Projection comparison plot to {default_save}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Lambdas and Etas: Final QP Step vs Final Projection Step.")
    parser.add_argument("--qp_csv", type=str, default=None, help="Path to z_sol.csv (Final QP step)")
    parser.add_argument("--proj_csv", type=str, default=None, help="Path to delta_projections.csv (Final Projection step)")
    parser.add_argument("--mode", type=str, choices=["sum", "norm"], default="sum", help="Plot sum or norm of 4-component blocks")
    parser.add_argument("--out", type=str, default="qp_vs_projection.png", help="Path to save output plot PNG")
    parser.add_argument("--show", action="store_true", default=False, help="Display interactive plot window")
    args = parser.parse_args()

    qp_candidates = [
        "examples/resources/multifinger_hand/ic3_debug_data/csv/z_sol.csv",
        "examples/resources/multifinger_hand/ic3_debug_data/csv/z_sol_180_separate_thresh.csv",
        "z_sol.csv",
    ]
    proj_candidates = [
        "examples/resources/multifinger_hand/ic3_debug_data/csv/delta_projections.csv",
        "examples/resources/multifinger_hand/ic3_debug_data/csv/delta_projections_180_separate_thresh.csv",
        "delta_projections.csv",
    ]

    qp_path = args.qp_csv or find_file("z_sol.csv", qp_candidates)
    proj_path = args.proj_csv or find_file("delta_projections.csv", proj_candidates)

    if not qp_path or not proj_path:
        print("Error: Could not find QP or Projection CSV files.")
        print("Run export_debug_to_csv to generate them from binary debug data.")
        sys.exit(1)

    print(f"Loading QP data from: {qp_path}")
    print(f"Loading Projection data from: {proj_path}")

    t_qp, lam_qp, eta_qp = load_csv_lambda_eta(qp_path)
    t_proj, lam_proj, eta_proj = load_csv_lambda_eta(proj_path)

    if t_qp is None or t_proj is None or len(t_qp) == 0 or len(t_proj) == 0:
        print("Error: Empty or invalid CSV files.")
        sys.exit(1)

    plot_qp_vs_projection(
        t_qp, lam_qp, eta_qp, t_proj, lam_proj, eta_proj,
        mode=args.mode, out_path=args.out, show=args.show
    )


if __name__ == "__main__":
    main()
