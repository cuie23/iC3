#!/usr/bin/env python3
"""
Animation of x0 and x1 from the C3 plan overlayed on top of each other
for every timestep in the MS-iC3 plan (final outer iteration).

Visualization features:
  - 3D Viewport:
      * x0 (current state): Solid fingertips (Blue, Red, Green) and solid/shaded 3D object cube.
      * x1 (C3 1-step planned lookahead): Distinct/translucent halo fingertips and wireframe 3D object cube.
      * Planned 1-step displacement vectors (x0 -> x1).
      * Historical trajectory path / trail of object and fingertips.
  - Side Dashboard:
      * State error ||x1 - x0|| (lookahead jump).
      * Object XYZ position and fingertip tracking over time with synchronized frame cursor.
  - Playback controls:
      * Interactive playback (Space to Play/Pause, Left/Right arrow keys for frame stepping).
      * Export to GIF / MP4 via --out.

Usage:
  python3 examples/python/animate_c3_x0_x1.py
  python3 examples/python/animate_c3_x0_x1.py --show
  python3 examples/python/animate_c3_x0_x1.py --out c3_x0_x1_animation.gif --fps 25
  python3 examples/python/animate_c3_x0_x1.py --csv /path/to/c3_x0_x1_plan.csv
"""

import argparse
import csv
import os
import sys
import numpy as np

# Ensure local user site-packages takes priority for mpl_toolkits
for site_pkg in sys.path:
    cand = os.path.join(site_pkg, "mpl_toolkits")
    if os.path.isdir(cand):
        try:
            import mpl_toolkits
            if cand not in mpl_toolkits.__path__:
                mpl_toolkits.__path__.insert(0, cand)
        except Exception:
            pass

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def find_default_csv():
    candidates = [
        "c3_x0_x1_plan.csv",
        "examples/resources/multifinger_hand/ic3_debug_data/c3_x0_x1_plan.csv",
        "../examples/resources/multifinger_hand/ic3_debug_data/c3_x0_x1_plan.csv",
        "/home/ericcui/c3_plus/c3/c3_x0_x1_plan.csv",
        "/home/ericcui/c3_plus/c3/examples/resources/multifinger_hand/ic3_debug_data/c3_x0_x1_plan.csv",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 0:
            return c
    return candidates[0]


def quat_to_rot_matrix(q):
    """Quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ])
    return R


def get_cube_vertices_and_faces(center, quat, half_size=0.0325):
    """Compute 8 vertices and 6 polygonal faces of a rotated 3D cube."""
    R = quat_to_rot_matrix(quat)
    # Unit cube offsets
    offsets = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1]
    ], dtype=float) * half_size

    verts = np.dot(offsets, R.T) + center

    # 6 quad faces
    faces = [
        [verts[0], verts[1], verts[2], verts[3]],  # Bottom
        [verts[4], verts[5], verts[6], verts[7]],  # Top
        [verts[0], verts[1], verts[5], verts[4]],  # Front
        [verts[2], verts[3], verts[7], verts[6]],  # Back
        [verts[1], verts[2], verts[6], verts[5]],  # Right
        [verts[0], verts[3], verts[7], verts[4]]   # Left
    ]
    return verts, faces


def load_c3_plan_data(csv_path, target_outer_iter=None):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None, None

    rows_by_iter = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return None, None

        # Parse header to determine n_x
        x0_cols = [i for i, h in enumerate(header) if h.startswith("x0_")]
        x1_cols = [i for i, h in enumerate(header) if h.startswith("x1_")]
        n_x = len(x0_cols)

        iter_idx = header.index("iteration")
        timestep_idx = header.index("plan_timestep")

        for row in reader:
            if not row or len(row) < len(header):
                continue
            try:
                it = int(row[iter_idx])
                ts = int(row[timestep_idx])
                x0_vec = np.array([float(row[c]) for c in x0_cols])
                x1_vec = np.array([float(row[c]) for c in x1_cols])

                if it not in rows_by_iter:
                    rows_by_iter[it] = []
                rows_by_iter[it].append((ts, x0_vec, x1_vec))
            except (ValueError, IndexError):
                continue

    if not rows_by_iter:
        return None, None

    max_it = max(rows_by_iter.keys())
    chosen_it = target_outer_iter if target_outer_iter is not None else max_it
    if chosen_it not in rows_by_iter:
        chosen_it = max_it

    # Sort records by timestep
    records = sorted(rows_by_iter[chosen_it], key=lambda r: r[0])
    return records, chosen_it


def create_animation(records, chosen_iter, output_path=None, show=False, fps=25):
    num_frames = len(records)
    if num_frames == 0:
        print("No plan records to animate.")
        return

    timesteps = np.array([r[0] for r in records])
    x0_mat = np.array([r[1] for r in records])  # (T, n_x)
    x1_mat = np.array([r[2] for r in records])  # (T, n_x)

    n_x = x0_mat.shape[1]
    is_point_hand = (n_x >= 31)

    # Compute step error ||x1 - x0||
    diff_norms = np.linalg.norm(x1_mat - x0_mat, axis=1)

    # Setup multi-panel figure
    fig = plt.figure(figsize=(18, 9), dpi=120)
    fig.patch.set_facecolor('#0f111a')

    # 3D Main animation viewport (left 60% of canvas)
    ax_3d = fig.add_axes([0.03, 0.08, 0.58, 0.84], projection='3d')
    ax_3d.set_facecolor('#0f111a')

    # 2D Side Dashboard (right 35% of canvas)
    ax_err = fig.add_axes([0.66, 0.58, 0.31, 0.32])
    ax_err.set_facecolor('#1a1c29')
    ax_pos = fig.add_axes([0.66, 0.12, 0.31, 0.38])
    ax_pos.set_facecolor('#1a1c29')

    # Style axes
    for ax in [ax_err, ax_pos]:
        ax.tick_params(colors='white', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#444b6e')
        ax.grid(True, linestyle='--', color='#2e3450', alpha=0.6)

    # Plot static background curves on 2D dashboard
    ax_err.plot(timesteps, diff_norms, color='#ff79c6', lw=1.8, label=r'$\|x_1 - x_0\|_2$ (C3 1-step lookahead jump)')
    ax_err.set_title(r"C3 1-Step Lookahead Discrepancy $\|x_1(t) - x_0(t)\|_2$", color='white', fontsize=11, pad=8, fontweight='bold')
    ax_err.set_xlabel("Plan Timestep (t)", color='white', fontsize=9)
    ax_err.set_ylabel(r"$\|x_1 - x_0\|_2$", color='white', fontsize=9)
    ax_err.legend(loc="upper right", facecolor='#1a1c29', edgecolor='#444b6e', labelcolor='white', fontsize=8)

    cursor_err = ax_err.axvline(x=timesteps[0], color='#50fa7b', linestyle='--', lw=2.0)
    dot_err, = ax_err.plot([timesteps[0]], [diff_norms[0]], marker='o', color='#50fa7b', markersize=6)

    if is_point_hand:
        cube_pos_x0 = x0_mat[:, 13:16]
        cube_pos_x1 = x1_mat[:, 13:16]
        ax_pos.plot(timesteps, cube_pos_x0[:, 0], color='#8be9fd', lw=1.5, label='Cube X (x0)')
        ax_pos.plot(timesteps, cube_pos_x0[:, 1], color='#50fa7b', lw=1.5, label='Cube Y (x0)')
        ax_pos.plot(timesteps, cube_pos_x0[:, 2], color='#f1fa8c', lw=1.5, label='Cube Z (x0)')
        ax_pos.plot(timesteps, cube_pos_x1[:, 0], color='#8be9fd', linestyle=':', lw=1.2, alpha=0.8, label='Cube X (x1)')
        ax_pos.plot(timesteps, cube_pos_x1[:, 1], color='#50fa7b', linestyle=':', lw=1.2, alpha=0.8, label='Cube Y (x1)')
        ax_pos.plot(timesteps, cube_pos_x1[:, 2], color='#f1fa8c', linestyle=':', lw=1.2, alpha=0.8, label='Cube Z (x1)')
        ax_pos.set_title("Object Cube Position (Solid: x0, Dotted: x1)", color='white', fontsize=11, pad=8, fontweight='bold')
        ax_pos.set_xlabel("Plan Timestep (t)", color='white', fontsize=9)
        ax_pos.set_ylabel("Position (m)", color='white', fontsize=9)
        ax_pos.legend(loc="upper right", facecolor='#1a1c29', edgecolor='#444b6e', labelcolor='white', fontsize=7.5, ncol=2)
    else:
        ax_pos.plot(timesteps, x0_mat[:, 0], color='#8be9fd', lw=1.5, label='State 0 (x0)')
        ax_pos.plot(timesteps, x1_mat[:, 0], color='#8be9fd', linestyle=':', lw=1.5, label='State 0 (x1)')
        ax_pos.set_title("Primary State Trajectory", color='white', fontsize=11, pad=8, fontweight='bold')
        ax_pos.legend(loc="upper right", facecolor='#1a1c29', edgecolor='#444b6e', labelcolor='white', fontsize=8)

    cursor_pos = ax_pos.axvline(x=timesteps[0], color='#50fa7b', linestyle='--', lw=2.0)

    # 3D viewport limits & background grid
    if is_point_hand:
        all_cube = np.vstack([x0_mat[:, 13:16], x1_mat[:, 13:16]])
        all_f1 = np.vstack([x0_mat[:, 0:3], x1_mat[:, 0:3]])
        all_f2 = np.vstack([x0_mat[:, 3:6], x1_mat[:, 3:6]])
        all_f3 = np.vstack([x0_mat[:, 6:9], x1_mat[:, 6:9]])
        all_pts = np.vstack([all_cube, all_f1, all_f2, all_f3])
        x_min, x_max = all_pts[:, 0].min() - 0.04, all_pts[:, 0].max() + 0.04
        y_min, y_max = all_pts[:, 1].min() - 0.04, all_pts[:, 1].max() + 0.04
        z_min, z_max = max(0.0, all_pts[:, 2].min() - 0.02), all_pts[:, 2].max() + 0.04
    else:
        x_min, x_max = -0.2, 0.2
        y_min, y_max = -0.2, 0.2
        z_min, z_max = 0.0, 0.3

    ax_3d.set_xlim([x_min, x_max])
    ax_3d.set_ylim([y_min, y_max])
    ax_3d.set_zlim([z_min, z_max])
    ax_3d.set_xlabel("X (m)", color='white', labelpad=8, fontweight='bold')
    ax_3d.set_ylabel("Y (m)", color='white', labelpad=8, fontweight='bold')
    ax_3d.set_zlabel("Z (m)", color='white', labelpad=8, fontweight='bold')
    ax_3d.tick_params(colors='white')
    ax_3d.grid(True, linestyle=':', color='#383c5a', alpha=0.4)

    # 3D elements containers
    cube_collection_x0 = None
    cube_collection_x1 = None
    scatter_f_x0 = None
    scatter_f_x1 = None
    quiver_arrows = []
    trail_cube_line, = ax_3d.plot([], [], [], color='#8be9fd', lw=1.2, alpha=0.6, linestyle='--')
    title_text = ax_3d.set_title(
        f"MS-iC3 Plan Animation: $x_0$ vs $x_1$ Overlay\n(Final Outer Iteration {chosen_iter} | Timestep 0 / {num_frames-1})",
        color='white', fontsize=13, fontweight='bold', pad=15
    )

    # Finger colors
    f_colors_x0 = ['#ff5555', '#50fa7b', '#8be9fd']
    f_colors_x1 = ['#ffb86c', '#f1fa8c', '#bd93f9']

    def update_frame(frame_idx):
        nonlocal cube_collection_x0, cube_collection_x1, scatter_f_x0, scatter_f_x1, quiver_arrows

        t_val = timesteps[frame_idx]
        x0_t = x0_mat[frame_idx]
        x1_t = x1_mat[frame_idx]

        # Update 2D dashboard cursors
        cursor_err.set_xdata([t_val, t_val])
        dot_err.set_data([t_val], [diff_norms[frame_idx]])
        cursor_pos.set_xdata([t_val, t_val])

        # Clear previous frame 3D mesh collections
        if cube_collection_x0 is not None:
            cube_collection_x0.remove()
            cube_collection_x0 = None
        if cube_collection_x1 is not None:
            cube_collection_x1.remove()
            cube_collection_x1 = None
        if scatter_f_x0 is not None:
            scatter_f_x0.remove()
            scatter_f_x0 = None
        if scatter_f_x1 is not None:
            scatter_f_x1.remove()
            scatter_f_x1 = None
        for q in quiver_arrows:
            try:
                q.remove()
            except Exception:
                pass
        quiver_arrows.clear()

        if is_point_hand:
            # 1. Cube x0 (Solid blue/teal box)
            q0 = x0_t[9:13]
            c0 = x0_t[13:16]
            verts0, faces0 = get_cube_vertices_and_faces(c0, q0)
            cube_collection_x0 = Poly3DCollection(
                faces0, facecolors='#6272a4', edgecolors='#8be9fd', linewidths=1.2, alpha=0.75
            )
            ax_3d.add_collection3d(cube_collection_x0)

            # 2. Cube x1 (Overlayed 1-step lookahead: orange dashed wireframe)
            q1 = x1_t[9:13]
            c1 = x1_t[13:16]
            verts1, faces1 = get_cube_vertices_and_faces(c1, q1)
            cube_collection_x1 = Poly3DCollection(
                faces1, facecolors='#ffb86c', edgecolors='#ff5555', linewidths=1.8, alpha=0.35, linestyle='--'
            )
            ax_3d.add_collection3d(cube_collection_x1)

            # 3. Fingertip points x0 (Solid spheres)
            p_f_x0 = np.array([x0_t[0:3], x0_t[3:6], x0_t[6:9]])
            scatter_f_x0 = ax_3d.scatter(
                p_f_x0[:, 0], p_f_x0[:, 1], p_f_x0[:, 2],
                c=f_colors_x0, s=90, depthshade=False, edgecolors='white', linewidths=1.2
            )

            # 4. Fingertip points x1 (Translucent halo / planned target spheres)
            p_f_x1 = np.array([x1_t[0:3], x1_t[3:6], x1_t[6:9]])
            scatter_f_x1 = ax_3d.scatter(
                p_f_x1[:, 0], p_f_x1[:, 1], p_f_x1[:, 2],
                c=f_colors_x1, s=140, marker='o', depthshade=False, edgecolors=f_colors_x1, linewidths=2.0, alpha=0.85
            )

            # 5. Motion displacement vectors x0 -> x1 (fingertips and cube center)
            for j in range(3):
                q_line = ax_3d.quiver(
                    p_f_x0[j, 0], p_f_x0[j, 1], p_f_x0[j, 2],
                    p_f_x1[j, 0] - p_f_x0[j, 0],
                    p_f_x1[j, 1] - p_f_x0[j, 1],
                    p_f_x1[j, 2] - p_f_x0[j, 2],
                    color='#f1fa8c', arrow_length_ratio=0.3, lw=1.5, alpha=0.9
                )
                quiver_arrows.append(q_line)

            # Cube center displacement
            q_cube = ax_3d.quiver(
                c0[0], c0[1], c0[2],
                c1[0] - c0[0], c1[1] - c0[1], c1[2] - c0[2],
                color='#ff79c6', arrow_length_ratio=0.3, lw=1.8, alpha=0.95
            )
            quiver_arrows.append(q_cube)

            # Historical trail
            trail_pts = x0_mat[:frame_idx + 1, 13:16]
            trail_cube_line.set_data(trail_pts[:, 0], trail_pts[:, 1])
            trail_cube_line.set_3d_properties(trail_pts[:, 2])

        if auto_orbit[0]:
            current_azim = (ax_3d.azim + 0.3) % 360
            ax_3d.view_init(elev=ax_3d.elev, azim=current_azim)

        if follow_object[0] and is_point_hand:
            c_curr = x0_t[13:16]
            span = zoom_level[0]
            ax_3d.set_xlim([c_curr[0] - span, c_curr[0] + span])
            ax_3d.set_ylim([c_curr[1] - span, c_curr[1] + span])
            ax_3d.set_zlim([max(0.0, c_curr[2] - span), c_curr[2] + span])

        title_text.set_text(
            f"MS-iC3 Plan Animation: $x_0$ vs $x_1$ Overlay (Outer Iter {chosen_iter})\n"
            f"Step {t_val} / {timesteps[-1]} | Jump $\\|x_1 - x_0\\|_2$: {diff_norms[frame_idx]:.4f} | Camera: Elev={ax_3d.elev:.0f}°, Azim={ax_3d.azim:.0f}°"
        )
        return [cursor_err, dot_err, cursor_pos, trail_cube_line, title_text]

    # Camera state variables
    auto_orbit = [False]
    follow_object = [False]
    zoom_level = [0.12]
    is_paused = [False]
    current_frame = [0]

    # Add HUD text for controls at bottom of 3D view
    fig.text(
        0.03, 0.015,
        "Camera & Playback: [Space] Pause/Play  [A/D or \u2190/\u2192] Rotate Azim  [W/S or \u2191/\u2193] Elev  [+/- or Scroll] Zoom  [1-4] Views (Iso/Top/Front/Side)  [C] Auto-Orbit  [F] Follow  [R] Reset",
        color='#6272a4', fontsize=8.5, fontweight='bold'
    )

    def on_key(event):
        step_angle = 5.0
        if event.key == ' ':
            is_paused[0] = not is_paused[0]
            if is_paused[0]:
                anim.pause()
            else:
                anim.resume()
        elif event.key in ['a', 'left']:
            ax_3d.view_init(elev=ax_3d.elev, azim=ax_3d.azim - step_angle)
            fig.canvas.draw_idle()
        elif event.key in ['d', 'right']:
            ax_3d.view_init(elev=ax_3d.elev, azim=ax_3d.azim + step_angle)
            fig.canvas.draw_idle()
        elif event.key in ['w', 'up']:
            ax_3d.view_init(elev=min(90.0, ax_3d.elev + step_angle), azim=ax_3d.azim)
            fig.canvas.draw_idle()
        elif event.key in ['s', 'down']:
            ax_3d.view_init(elev=max(-90.0, ax_3d.elev - step_angle), azim=ax_3d.azim)
            fig.canvas.draw_idle()
        elif event.key in ['+', '=', 'z']:
            # Zoom in
            zoom_level[0] = max(0.04, zoom_level[0] * 0.85)
            update_zoom()
        elif event.key in ['-', '_', 'x']:
            # Zoom out
            zoom_level[0] = min(0.4, zoom_level[0] * 1.15)
            update_zoom()
        elif event.key == '1':
            # Isometric view
            ax_3d.view_init(elev=25, azim=-60)
            fig.canvas.draw_idle()
        elif event.key == '2':
            # Top view
            ax_3d.view_init(elev=90, azim=-90)
            fig.canvas.draw_idle()
        elif event.key == '3':
            # Front view
            ax_3d.view_init(elev=0, azim=-90)
            fig.canvas.draw_idle()
        elif event.key == '4':
            # Side view
            ax_3d.view_init(elev=0, azim=0)
            fig.canvas.draw_idle()
        elif event.key in ['r', 'R']:
            # Reset view
            ax_3d.view_init(elev=30, azim=-60)
            zoom_level[0] = 0.12
            auto_orbit[0] = False
            follow_object[0] = False
            ax_3d.set_xlim([x_min, x_max])
            ax_3d.set_ylim([y_min, y_max])
            ax_3d.set_zlim([z_min, z_max])
            fig.canvas.draw_idle()
        elif event.key in ['c', 'C']:
            auto_orbit[0] = not auto_orbit[0]
            print(f"Auto-Orbiting camera: {'ENABLED' if auto_orbit[0] else 'DISABLED'}")
        elif event.key in ['f', 'F']:
            follow_object[0] = not follow_object[0]
            print(f"Follow Object camera: {'ENABLED' if follow_object[0] else 'DISABLED'}")

    def on_scroll(event):
        if event.inaxes == ax_3d:
            if event.button == 'up':
                zoom_level[0] = max(0.04, zoom_level[0] * 0.9)
            elif event.button == 'down':
                zoom_level[0] = min(0.4, zoom_level[0] * 1.1)
            update_zoom()

    def update_zoom():
        if is_point_hand:
            # Zoom around center
            mid_x = 0.5 * (ax_3d.get_xlim()[0] + ax_3d.get_xlim()[1])
            mid_y = 0.5 * (ax_3d.get_ylim()[0] + ax_3d.get_ylim()[1])
            mid_z = 0.5 * (ax_3d.get_zlim()[0] + ax_3d.get_zlim()[1])
            span = zoom_level[0]
            ax_3d.set_xlim([mid_x - span, mid_x + span])
            ax_3d.set_ylim([mid_y - span, mid_y + span])
            ax_3d.set_zlim([max(0.0, mid_z - span), mid_z + span])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    anim = FuncAnimation(fig, update_frame, frames=num_frames, interval=int(1000 / fps), blit=False)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        print(f"Rendering and saving animation to {output_path} ({num_frames} frames @ {fps} fps)...")
        if output_path.endswith(".gif"):
            writer = PillowWriter(fps=fps)
            anim.save(output_path, writer=writer)
        else:
            anim.save(output_path, fps=fps)
        print(f"Saved animation to {output_path}")

    if show:
        plt.show()
    elif not output_path:
        plt.close(fig)


def save_snapshot(records, chosen_iter, output_path="c3_x0_x1_snapshot.png", num_panels=4):
    """Save a static multi-panel filmstrip of x0 and x1 overlay at evenly spaced timesteps."""
    num_frames = len(records)
    if num_frames == 0:
        return

    sample_indices = np.linspace(0, num_frames - 1, num_panels, dtype=int)
    fig = plt.figure(figsize=(6 * num_panels, 6), dpi=140)
    fig.patch.set_facecolor('#0f111a')

    for p_idx, f_idx in enumerate(sample_indices):
        t_val, x0_t, x1_t = records[f_idx]
        diff_norm = np.linalg.norm(x1_t - x0_t)

        ax = fig.add_subplot(1, num_panels, p_idx + 1, projection='3d')
        ax.set_facecolor('#1a1c29')

        n_x = len(x0_t)
        if n_x >= 31:
            # 3D Point Hand & Object
            q0, c0 = x0_t[9:13], x0_t[13:16]
            q1, c1 = x1_t[9:13], x1_t[13:16]
            _, faces0 = get_cube_vertices_and_faces(c0, q0)
            _, faces1 = get_cube_vertices_and_faces(c1, q1)

            poly0 = Poly3DCollection(faces0, facecolors='#6272a4', edgecolors='#8be9fd', linewidths=1.2, alpha=0.7)
            poly1 = Poly3DCollection(faces1, facecolors='#ffb86c', edgecolors='#ff5555', linewidths=1.8, alpha=0.35, linestyle='--')
            ax.add_collection3d(poly0)
            ax.add_collection3d(poly1)

            p_f0 = np.array([x0_t[0:3], x0_t[3:6], x0_t[6:9]])
            p_f1 = np.array([x1_t[0:3], x1_t[3:6], x1_t[6:9]])
            f_cols0 = ['#ff5555', '#50fa7b', '#8be9fd']
            f_cols1 = ['#ffb86c', '#f1fa8c', '#bd93f9']
            ax.scatter(p_f0[:, 0], p_f0[:, 1], p_f0[:, 2], c=f_cols0, s=80, depthshade=False, edgecolors='white')
            ax.scatter(p_f1[:, 0], p_f1[:, 1], p_f1[:, 2], c=f_cols1, s=120, depthshade=False, edgecolors=f_cols1, alpha=0.85)

            for j in range(3):
                ax.quiver(p_f0[j, 0], p_f0[j, 1], p_f0[j, 2],
                          p_f1[j, 0] - p_f0[j, 0], p_f1[j, 1] - p_f0[j, 1], p_f1[j, 2] - p_f0[j, 2],
                          color='#f1fa8c', arrow_length_ratio=0.3, lw=1.5)

            ax.quiver(c0[0], c0[1], c0[2],
                      c1[0] - c0[0], c1[1] - c0[1], c1[2] - c0[2],
                      color='#ff79c6', arrow_length_ratio=0.3, lw=1.8)
        else:
            # General system (e.g. plate)
            ax.scatter([x0_t[0]], [x0_t[1] if n_x > 1 else 0], [x0_t[2] if n_x > 2 else 0], c='#8be9fd', s=100, label=r'$x_0$ (Current)')
            ax.scatter([x1_t[0]], [x1_t[1] if n_x > 1 else 0], [x1_t[2] if n_x > 2 else 0], c='#ff79c6', s=140, marker='^', label=r'$x_1$ (C3 Lookahead)')

        ax.set_title(f"Step {t_val} / {records[-1][0]}\n$\\|x_1 - x_0\\|_2 = {diff_norm:.4f}$", color='white', fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("X (m)", color='white', fontsize=8)
        ax.set_ylabel("Y (m)", color='white', fontsize=8)
        ax.set_zlabel("Z (m)", color='white', fontsize=8)
        ax.tick_params(colors='white', labelsize=8)
        ax.grid(True, linestyle=':', color='#383c5a', alpha=0.4)

    plt.suptitle(f"C3 Plan $x_0$ vs $x_1$ Overlay Keyframes (Final Outer Iteration {chosen_iter})\nSolid: $x_0$ Current State | Dotted/Translucent: $x_1$ C3 1-Step Lookahead",
                 color='white', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=140)
    plt.close(fig)
    print(f"Saved snapshot filmstrip to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create animation of x0 and x1 C3 plan overlay for MS-iC3.")
    parser.add_argument("--csv", type=str, default=None, help="Path to input c3_x0_x1_plan.csv")
    parser.add_argument("--out", type=str, default="c3_x0_x1_animation.gif", help="Path to save output animation (GIF/MP4)")
    parser.add_argument("--snapshot", type=str, default="c3_x0_x1_snapshot.png", help="Path to save keyframe snapshot filmstrip PNG")
    parser.add_argument("--iter", type=int, default=None, help="Outer iteration to animate (default: final iteration)")
    parser.add_argument("--fps", type=int, default=20, help="Animation frames per second")
    parser.add_argument("--show", action="store_true", help="Display interactive animation window")
    args = parser.parse_args()

    csv_path = args.csv or find_default_csv()
    records, chosen_it = load_c3_plan_data(csv_path, target_outer_iter=args.iter)

    if not records:
        print(f"Error: Could not read data from {csv_path}")
        print("Run an MS-iC3 simulation first to generate c3_x0_x1_plan.csv")
        sys.exit(1)

    print(f"Loaded {len(records)} plan timesteps from {csv_path} for outer iteration {chosen_it}")
    if args.snapshot:
        save_snapshot(records, chosen_it, output_path=args.snapshot)
    create_animation(records, chosen_it, output_path=args.out, show=args.show, fps=args.fps)


if __name__ == "__main__":
    main()
