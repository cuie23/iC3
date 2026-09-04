#!/usr/bin/env python3
"""
Drake Meshcat visualizer for overlaying x0 (current state) and x1 (C3 1-step lookahead)
using the MultibodyPlant models from the MS-iC3 examples.

Features:
  - Dual MultibodyPlant representation:
      * Model 0 (x0): Solid full-opacity robot and object representing the current state.
      * Model 1 (x1): Translucent / highlighted overlay representing the C3 1-step planned lookahead.
  - Meshcat Interactive Web Viewer:
      * Interactive 3D camera (Orbit, Pan, Zoom).
      * Full playback timeline scrubber (Play, Pause, Loop, Speed controls).
      * Export to standalone interactive HTML (--html).
  - Automatically handles both Point Hand (3 fingers + cube) and Plate manipulation examples.

Usage:
  # 1. Start interactive Meshcat server (open the printed URL in your browser):
  python3 examples/python/visualize_c3_overlay_drake.py

  # 2. Save standalone 3D interactive HTML file:
  python3 examples/python/visualize_c3_overlay_drake.py --html c3_x0_x1_meshcat.html

  # 3. Specify custom CSV path, playback speed, or outer iteration:
  python3 examples/python/visualize_c3_overlay_drake.py --csv c3_x0_x1_plan.csv --speed 1.0 --iter 5
"""

import argparse
import csv
import os
import sys
import time
import numpy as np

from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    RigidTransform,
    RotationMatrix,
    Rgba,
    Cylinder,
)


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


def load_c3_plan_data(csv_path, target_outer_iter=None):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None, None

    rows_by_iter = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return None, None

        x0_cols = [i for i, h in enumerate(header) if h.startswith("x0_")]
        x1_cols = [i for i, h in enumerate(header) if h.startswith("x1_")]

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

    records = sorted(rows_by_iter[chosen_it], key=lambda r: r[0])
    return records, chosen_it


def build_dual_point_hand_plant(builder):
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)

    hand_file = "examples/resources/multifinger_hand/urdf/simplified_hand_180.sdf"
    cube_file = "examples/resources/multifinger_hand/urdf/cube.sdf"
    ground_file = "examples/resources/multifinger_hand/ground.urdf"

    # Model 0: x0 (Solid Current State)
    hand_0 = parser.AddModels(hand_file)[0]
    cube_0 = parser.AddModels(cube_file)[0]
    plant.RenameModelInstance(hand_0, "hand_x0")
    plant.RenameModelInstance(cube_0, "cube_x0")

    # Model 1: x1 (Translucent C3 Lookahead)
    hand_1 = parser.AddModels(hand_file)[0]
    cube_1 = parser.AddModels(cube_file)[0]
    plant.RenameModelInstance(hand_1, "hand_x1_c3_lookahead")
    plant.RenameModelInstance(cube_1, "cube_x1_c3_lookahead")

    ground = parser.AddModels(ground_file)[0]

    # Weld base links to world frame
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link_1", hand_0), RigidTransform())
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link_2", hand_0), RigidTransform())
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link_3", hand_0), RigidTransform())

    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link_1", hand_1), RigidTransform())
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link_2", hand_1), RigidTransform())
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link_3", hand_1), RigidTransform())

    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("ground", ground), RigidTransform())

    plant.Finalize()
    return plant, scene_graph, {"system": "point_hand", "hand_0": hand_0, "cube_0": cube_0, "hand_1": hand_1, "cube_1": cube_1}


def build_dual_plate_plant(builder):
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)

    plate_file = "examples/resources/plate/plate.sdf"
    cube_file = "examples/resources/plate/cube.sdf"

    # Model 0: x0 (Solid Current State)
    plate_0 = parser.AddModels(plate_file)[0]
    cube_0 = parser.AddModels(cube_file)[0]
    plant.RenameModelInstance(plate_0, "plate_x0")
    plant.RenameModelInstance(cube_0, "cube_x0")

    # Model 1: x1 (Translucent C3 Lookahead)
    plate_1 = parser.AddModels(plate_file)[0]
    cube_1 = parser.AddModels(cube_file)[0]
    plant.RenameModelInstance(plate_1, "plate_x1_c3_lookahead")
    plant.RenameModelInstance(cube_1, "cube_x1_c3_lookahead")

    plant.Finalize()
    return plant, scene_graph, {"system": "plate", "plate_0": plate_0, "cube_0": cube_0, "plate_1": plate_1, "cube_1": cube_1}


def set_plant_state(plant, plant_context, model_info, x0, x1):
    actuator_0 = model_info.get("hand_0") or model_info.get("plate_0")
    cube_0 = model_info["cube_0"]

    actuator_1 = model_info.get("hand_1") or model_info.get("plate_1")
    cube_1 = model_info["cube_1"]

    n_act = plant.num_positions(actuator_0)
    n_cube = plant.num_positions(cube_0)

    # State 0 (Current)
    plant.SetPositions(plant_context, actuator_0, x0[0:n_act])
    plant.SetPositions(plant_context, cube_0, x0[n_act:n_act + n_cube])

    # State 1 (C3 1-step lookahead)
    plant.SetPositions(plant_context, actuator_1, x1[0:n_act])
    plant.SetPositions(plant_context, cube_1, x1[n_act:n_act + n_cube])


def main():
    parser = argparse.ArgumentParser(description="Drake Meshcat visualizer for C3 x0 and x1 plan overlay.")
    parser.add_argument("--csv", type=str, default=None, help="Path to c3_x0_x1_plan.csv")
    parser.add_argument("--iter", type=int, default=None, help="Outer iteration to visualize (default: final iteration)")
    parser.add_argument("--dt", type=float, default=0.05, help="Simulation timestep per plan step (default: 0.05s)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--html", type=str, default=None, help="Save standalone interactive Meshcat HTML file")
    parser.add_argument("--live", action="store_true", default=True, help="Keep Meshcat server running for interactive web browser viewing")
    parser.add_argument("--no-live", dest="live", action="store_false", help="Exit after generating HTML / recording without staying in live loop")
    args = parser.parse_args()

    csv_path = args.csv or find_default_csv()
    records, chosen_it = load_c3_plan_data(csv_path, target_outer_iter=args.iter)

    if not records:
        print(f"Error: Could not read data from {csv_path}")
        print("Run an MS-iC3 simulation first to generate c3_x0_x1_plan.csv")
        sys.exit(1)

    n_x = len(records[0][1])
    num_timesteps = len(records)
    print(f"Loaded {num_timesteps} timesteps (n_x = {n_x}) for outer iteration {chosen_it}")

    meshcat = Meshcat()
    print("\n" + "=" * 70)
    print(f" Meshcat Visualizer Running at: {meshcat.web_url()}")
    print("=" * 70 + "\n")

    builder = DiagramBuilder()
    if n_x >= 31:
        plant, scene_graph, model_info = build_dual_point_hand_plant(builder)
    else:
        plant, scene_graph, model_info = build_dual_plate_plant(builder)

    visualizer_params = MeshcatVisualizerParams()
    visualizer_params.publish_period = args.dt
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, visualizer_params)

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(diagram_context)

    # Initial forced publish
    diagram.ForcedPublish(diagram_context)

    # Style the overlay model (x1) in Meshcat with translucent opacity
    overlay_alpha = 0.40
    inspector = scene_graph.model_inspector()
    for gid in inspector.GetAllGeometryIds():
        geom_name = inspector.GetName(gid)
        frame_id = inspector.GetFrameId(gid)
        frame_name = inspector.GetName(frame_id)
        path = f"visualizer/{frame_name}/{geom_name}"

        # Maintain exact SDF finger colors: Finger 1 = Red, Finger 2 = Green, Finger 3 = Blue
        if "fingertip_1" in frame_name:
            meshcat.SetProperty(path, "color", [1.0, 0.0, 0.0, overlay_alpha if ("_1" in frame_name or "_1" in geom_name) else 1.0])
        elif "fingertip_2" in frame_name:
            meshcat.SetProperty(path, "color", [0.0, 1.0, 0.0, overlay_alpha if ("_1" in frame_name or "_1" in geom_name) else 1.0])
        elif "fingertip_3" in frame_name:
            meshcat.SetProperty(path, "color", [0.0, 0.0, 1.0, overlay_alpha if ("_1" in frame_name or "_1" in geom_name) else 1.0])

        if "_1" in frame_name or "_1" in geom_name:
            meshcat.SetProperty(path, "opacity", overlay_alpha)
            meshcat.SetProperty(f"visualizer/{frame_name}", "opacity", overlay_alpha)

    # Set initial camera pose and target for optimal framing
    meshcat.SetCameraPose(
        camera_in_world=np.array([0.38, -0.38, 0.28]),
        target_in_world=np.array([0.0, 0.0, 0.05])
    )

    # Record animation into Meshcat timeline
    print("Recording animation into Meshcat timeline scrubber...")
    visualizer.StartRecording(set_transforms_while_recording=True)

    step_dt = args.dt / max(args.speed, 0.01)
    sim_time = 0.0
    for idx, (t_val, x0_t, x1_t) in enumerate(records):
        diagram_context.SetTime(sim_time)
        set_plant_state(plant, plant_context, model_info, x0_t, x1_t)
        diagram.ForcedPublish(diagram_context)
        sim_time += step_dt

    visualizer.StopRecording()
    visualizer.PublishRecording()

    # Set animation directly on Meshcat so it is embedded in StaticHtml
    recording = visualizer.get_mutable_recording()
    meshcat.SetAnimation(recording)
    print("Meshcat animation published! You can scrub and play the timeline in your browser.")

    if args.html:
        raw_html = meshcat.StaticHtml()
        injection = """
  <script>
    (function() {
      function applyFingerStyles() {
        if (typeof viewer === 'undefined' || !viewer || !viewer.scene_tree || !viewer.scene_tree.object) return;
        viewer.scene_tree.object.traverse(function(child) {
          if (child.isMesh && child.material) {
            let pathName = child.name || "";
            let p = child.parent;
            while (p) {
              if (p.name) pathName = p.name + "/" + pathName;
              p = p.parent;
            }
            if (!child.material.__isConfigured) {
              child.material = child.material.clone();
              child.material.__isConfigured = true;
            }
            if (/fingertip_1/.test(pathName)) child.material.color.setRGB(1.0, 0.0, 0.0);
            else if (/fingertip_2/.test(pathName)) child.material.color.setRGB(0.0, 1.0, 0.0);
            else if (/fingertip_3/.test(pathName)) child.material.color.setRGB(0.0, 0.0, 1.0);
            
            if (/_1/.test(pathName)) {
              child.material.transparent = true;
              child.material.opacity = 0.40;
            }
            child.material.depthWrite = true;
            child.material.needsUpdate = true;
          }
        });
        if (typeof viewer.set_dirty === 'function') viewer.set_dirty();
      }
      const checkInterval = setInterval(applyFingerStyles, 300);
      setTimeout(function() { clearInterval(checkInterval); }, 15000);
    })();
  </script>
</body>
"""
        html_content = raw_html.replace("</body>", injection)
        os.makedirs(os.path.dirname(os.path.abspath(args.html)), exist_ok=True)
        with open(args.html, "w") as f:
            f.write(html_content)
        print(f"Saved standalone interactive HTML visualizer to {args.html}")

    # Live interactive loop
    if args.live:
        print(f"\nLive visualizer server ready at {meshcat.web_url()}")
        print("Use the Meshcat web interface at the URL above to Play (\u25B6), Pause (\u23F8), and scrub the timeline.")
        print("Press Ctrl+C to exit server.\n")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nVisualizer stopped.")


if __name__ == "__main__":
    main()
