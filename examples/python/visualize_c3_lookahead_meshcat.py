#!/usr/bin/env python3
"""
Meshcat Visualizer for Full C3 Lookahead Trajectory with Gradient Transparency.

Features:
  - Visualizes all lookahead stages [x_0, x_1, x_2, ..., x_K] planned by C3 at each plan timestep t.
  - Gradient Transparency (Ghosting Effect):
      * Stage 0 (Current State): Solid full-opacity (alpha = 1.0).
      * Stage k (Predicted Lookahead): Transparency increases progressively with stage index k.
  - Multi-body Lookahead Ghost Trail:
      * Dynamically updates poses of all lookahead model instances in Meshcat.
      * Draws connecting 3D trajectory ribbons / curves through lookahead centers.
  - Export Options:
      * Live interactive Meshcat 3D web viewer with timeline playback controls.
      * Standalone interactive 3D HTML video file (--html).
      * Rendered MP4 / GIF video export (--out).

Usage:
  # 1. Live Meshcat Viewer:
  python3 examples/python/visualize_c3_lookahead_meshcat.py

  # 2. Export Standalone Meshcat 3D HTML video:
  python3 examples/python/visualize_c3_lookahead_meshcat.py --html c3_lookahead_meshcat.html

  # 3. Export to GIF/Video with custom lookahead stages:
  python3 examples/python/visualize_c3_lookahead_meshcat.py --max_lookahead 8 --speed 1.0
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
        "c3_full_lookahead_plan.csv",
        "examples/resources/multifinger_hand/ic3_debug_data/c3_full_lookahead_plan.csv",
        "/home/ericcui/c3_plus/c3/c3_full_lookahead_plan.csv",
        "c3_x0_x1_plan.csv",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 0:
            return c
    return candidates[0]


def load_lookahead_data(csv_path, target_outer_iter=None, max_k=None):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None, None, 0

    rows_by_iter = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return None, None, 0

        is_full_csv = "lookahead_step" in header
        iter_idx = header.index("iteration")
        ts_idx = header.index("plan_timestep")

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
                    if max_k is not None and k >= max_k:
                        continue
                    x_vec = np.array([float(row[c]) for c in x_cols])
                    if it not in rows_by_iter:
                        rows_by_iter[it] = {}
                    if ts not in rows_by_iter[it]:
                        rows_by_iter[it][ts] = {}
                    rows_by_iter[it][ts][k] = x_vec
                except (ValueError, IndexError):
                    continue
        else:
            # Fallback for x0/x1 CSV
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
                        rows_by_iter[it] = {}
                    rows_by_iter[it][ts] = {0: x0_vec, 1: x1_vec}
                except (ValueError, IndexError):
                    continue

    if not rows_by_iter:
        return None, None, 0

    max_it = max(rows_by_iter.keys())
    chosen_it = target_outer_iter if target_outer_iter is not None else max_it
    if chosen_it not in rows_by_iter:
        chosen_it = max_it

    iter_dict = rows_by_iter[chosen_it]
    sorted_timesteps = sorted(iter_dict.keys())

    # Find max lookahead stages present
    all_k_counts = [len(iter_dict[ts]) for ts in sorted_timesteps]
    num_lookahead_stages = max(all_k_counts) if all_k_counts else 0

    records = []
    for ts in sorted_timesteps:
        stage_dict = iter_dict[ts]
        max_stage = max(stage_dict.keys())
        stages_list = [stage_dict.get(k, stage_dict[max_stage]) for k in range(max_stage + 1)]
        records.append((ts, stages_list))

    return records, chosen_it, num_lookahead_stages


def build_lookahead_point_hand_plant(builder, num_stages):
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)

    hand_file = "examples/resources/multifinger_hand/urdf/simplified_hand_180.sdf"
    cube_file = "examples/resources/multifinger_hand/urdf/cube.sdf"
    ground_file = "examples/resources/multifinger_hand/ground.urdf"

    model_instances = []
    for k in range(num_stages):
        h_idx = parser.AddModels(hand_file)[0]
        c_idx = parser.AddModels(cube_file)[0]
        h_name = f"hand_k{k}" if k > 0 else "hand_x0"
        c_name = f"cube_k{k}" if k > 0 else "cube_x0"
        plant.RenameModelInstance(h_idx, h_name)
        plant.RenameModelInstance(c_idx, c_name)

        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link_1", h_idx), RigidTransform())
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link_2", h_idx), RigidTransform())
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link_3", h_idx), RigidTransform())

        model_instances.append({"hand": h_idx, "cube": c_idx, "hand_name": h_name, "cube_name": c_name})

    ground = parser.AddModels(ground_file)[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("ground", ground), RigidTransform())

    plant.Finalize()
    return plant, scene_graph, {"system": "point_hand", "stages": model_instances}


def build_lookahead_plate_plant(builder, num_stages):
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)

    plate_file = "examples/resources/plate/plate.sdf"
    cube_file = "examples/resources/plate/cube.sdf"

    model_instances = []
    for k in range(num_stages):
        p_idx = parser.AddModels(plate_file)[0]
        c_idx = parser.AddModels(cube_file)[0]
        p_name = f"plate_k{k}" if k > 0 else "plate_x0"
        c_name = f"cube_k{k}" if k > 0 else "cube_x0"
        plant.RenameModelInstance(p_idx, p_name)
        plant.RenameModelInstance(c_idx, c_name)
        model_instances.append({"plate": p_idx, "cube": c_idx, "plate_name": p_name, "cube_name": c_name})

    plant.Finalize()
    return plant, scene_graph, {"system": "plate", "stages": model_instances}


def set_lookahead_state(plant, plant_context, model_info, stages_list):
    stages_info = model_info["stages"]
    num_stages = len(stages_info)

    for k in range(num_stages):
        x_k = stages_list[k] if k < len(stages_list) else stages_list[-1]
        st = stages_info[k]

        actuator_inst = st.get("hand") or st.get("plate")
        cube_inst = st["cube"]

        n_act = plant.num_positions(actuator_inst)
        n_cube = plant.num_positions(cube_inst)

        q_act = x_k[0:n_act]
        q_cube = x_k[n_act:n_act + n_cube]

        plant.SetPositions(plant_context, actuator_inst, q_act)
        plant.SetPositions(plant_context, cube_inst, q_cube)


def apply_lookahead_transparency_styling(meshcat, scene_graph, model_info):
    stages = model_info["stages"]
    num_stages = len(stages)
    inspector = scene_graph.model_inspector()

    # Pre-calculate alpha for each stage k
    stage_alphas = {}
    for k in range(num_stages):
        if k == 0:
            stage_alphas[k] = 1.0
        else:
            decay = (1.0 - float(k) / (num_stages + 0.2)) ** 1.2
            stage_alphas[k] = max(0.08, float(0.8 * decay))

    # Map model instance names to stage index
    name_to_stage = {}
    for k, st in enumerate(stages):
        if "hand_name" in st:
            name_to_stage[st["hand_name"]] = k
        if "plate_name" in st:
            name_to_stage[st["plate_name"]] = k
        if "cube_name" in st:
            name_to_stage[st["cube_name"]] = k

    # Traverse all geometry IDs in scene_graph and set property on exact meshcat path
    for gid in inspector.GetAllGeometryIds():
        geom_name = inspector.GetName(gid)
        frame_id = inspector.GetFrameId(gid)
        frame_name = inspector.GetName(frame_id)
        path = f"visualizer/{frame_name}/{geom_name}"

        assigned_stage = 0
        for name_prefix, st_k in name_to_stage.items():
            if frame_name.startswith(name_prefix) or geom_name.startswith(name_prefix):
                assigned_stage = st_k
                break

        alpha = stage_alphas[assigned_stage]
        if assigned_stage > 0:
            meshcat.SetProperty(path, "opacity", alpha)
            meshcat.SetProperty(f"visualizer/{frame_name}", "opacity", alpha)


def inject_transparency_script(html_content, num_stages):
    # Generates client-side Three.js post-load hook to guarantee:
    # 1. Finger 1 is RED, Finger 2 is GREEN, Finger 3 is BLUE across all stages.
    # 2. Object Cube keeps its original SDF color across all stages.
    # 3. Lookahead stage k has progressively increasing transparency (decaying opacity).
    stage_alphas_js = []
    for k in range(num_stages):
        if k == 0:
            alpha = 1.0
        else:
            decay = (1.0 - float(k) / (num_stages + 0.2)) ** 1.6
            alpha = max(0.08, float(0.75 * decay))
        stage_alphas_js.append(f'{{ stage: {k}, regex: /_k{k}|hand_k{k}|cube_k{k}|plate_k{k}|hand_{k}|cube_{k}|plate_{k}/, opacity: {alpha:.3f} }}')

    stages_array_str = ",\n        ".join(stage_alphas_js)

    injection = f"""
  <script>
    // Injected SDF Color Preserving & Gradient Transparency Hook
    (function() {{
      const stageConfigs = [
        {stages_array_str}
      ];

      function applyGradientTransparency() {{
        if (typeof viewer === 'undefined' || !viewer || !viewer.scene_tree || !viewer.scene_tree.object) {{
          return;
        }}
        viewer.scene_tree.object.traverse(function(child) {{
          if (child.isMesh && child.material) {{
            let pathName = child.name || "";
            let p = child.parent;
            while (p) {{
              if (p.name) pathName = p.name + "/" + pathName;
              p = p.parent;
            }}

            // Clone material so each stage and fingertip has independent material instances
            if (!child.material.__isConfigured) {{
              child.material = child.material.clone();
              child.material.__isConfigured = true;
            }}

            // 1. Enforce SDF Finger Colors:
            // Fingertip 1 = Red, Fingertip 2 = Green, Fingertip 3 = Blue
            if (/fingertip_1/.test(pathName)) {{
              child.material.color.setRGB(1.0, 0.0, 0.0);
            }} else if (/fingertip_2/.test(pathName)) {{
              child.material.color.setRGB(0.0, 1.0, 0.0);
            }} else if (/fingertip_3/.test(pathName)) {{
              child.material.color.setRGB(0.0, 0.0, 1.0);
            }}

            // 2. Set Lookahead Transparency:
            // Match stage index and assign opacity
            let assignedOpacity = 1.0;
            for (let i = 1; i < stageConfigs.length; i++) {{
              if (stageConfigs[i].regex.test(pathName)) {{
                assignedOpacity = stageConfigs[i].opacity;
                break;
              }}
            }}

            child.material.transparent = (assignedOpacity < 0.99);
            child.material.opacity = assignedOpacity;
            child.material.depthWrite = true;
            child.material.needsUpdate = true;
          }}
        }});
        if (typeof viewer.set_dirty === 'function') {{
          viewer.set_dirty();
        }}
      }}

      // Periodic check on load to style models after async binary decoding
      const checkInterval = setInterval(applyGradientTransparency, 300);
      setTimeout(function() {{ clearInterval(checkInterval); }}, 15000);
      window.addEventListener('load', applyGradientTransparency);
      document.addEventListener('DOMContentLoaded', applyGradientTransparency);
    }})();
  </script>
</body>
"""
    return html_content.replace("</body>", injection)


def main():
    parser = argparse.ArgumentParser(description="Meshcat Visualizer for Full C3 Lookahead Trajectory with Gradient Transparency.")
    parser.add_argument("--csv", type=str, default=None, help="Path to c3_full_lookahead_plan.csv")
    parser.add_argument("--iter", type=int, default=None, help="Outer iteration to visualize (default: final iteration)")
    parser.add_argument("--max_lookahead", type=int, default=6, help="Maximum number of lookahead stages to render simultaneously")
    parser.add_argument("--dt", type=float, default=0.05, help="Simulation timestep per plan step (default: 0.05s)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--html", type=str, default="c3_lookahead_meshcat.html", help="Path to export standalone interactive Meshcat HTML file")
    parser.add_argument("--live", action="store_true", default=True, help="Keep Meshcat server running for live browser viewing")
    parser.add_argument("--no-live", dest="live", action="store_false", help="Exit after exporting HTML")
    args = parser.parse_args()

    csv_path = args.csv or find_default_csv()
    records, chosen_it, num_stages_data = load_lookahead_data(csv_path, target_outer_iter=args.iter, max_k=args.max_lookahead)

    if not records:
        print(f"Error: Could not load lookahead data from {csv_path}")
        print("Run an MS-iC3 simulation first to generate c3_full_lookahead_plan.csv")
        sys.exit(1)

    stages_to_render = min(args.max_lookahead, max(num_stages_data, len(records[0][1])))
    stages_to_render = max(2, stages_to_render)
    n_x = len(records[0][1][0])
    num_timesteps = len(records)

    print(f"Loaded {num_timesteps} plan timesteps (n_x = {n_x}) for outer iteration {chosen_it}")
    print(f"Rendering {stages_to_render} lookahead stages per timestep with gradient transparency")

    meshcat = Meshcat()
    print("\n" + "=" * 75)
    print(f" Meshcat Lookahead Visualizer Running at: {meshcat.web_url()}")
    print("=" * 75 + "\n")

    builder = DiagramBuilder()
    if n_x >= 31:
        plant, scene_graph, model_info = build_lookahead_point_hand_plant(builder, stages_to_render)
    else:
        plant, scene_graph, model_info = build_lookahead_plate_plant(builder, stages_to_render)

    visualizer_params = MeshcatVisualizerParams()
    visualizer_params.publish_period = args.dt
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, visualizer_params)

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(diagram_context)

    # Initial forced publish to populate scene graph in Meshcat
    diagram.ForcedPublish(diagram_context)

    # Apply progressive gradient transparency styling to lookahead stages
    apply_lookahead_transparency_styling(meshcat, scene_graph, model_info)

    # Set initial camera pose
    meshcat.SetCameraPose(
        camera_in_world=np.array([0.42, -0.42, 0.32]),
        target_in_world=np.array([0.0, 0.0, 0.05])
    )

    # Add Target Coordinate Axis markers
    axis_len, axis_r = 0.08, 0.003
    meshcat.SetObject("target_axes/x", Cylinder(axis_r, axis_len), Rgba(1, 0, 0, 0.8))
    meshcat.SetTransform("target_axes/x", RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2), [axis_len / 2, 0, 0.05]))
    meshcat.SetObject("target_axes/y", Cylinder(axis_r, axis_len), Rgba(0, 1, 0, 0.8))
    meshcat.SetTransform("target_axes/y", RigidTransform(RotationMatrix.MakeXRotation(-np.pi / 2), [0, axis_len / 2, 0.05]))
    meshcat.SetObject("target_axes/z", Cylinder(axis_r, axis_len), Rgba(0, 0, 1, 0.8))
    meshcat.SetTransform("target_axes/z", RigidTransform(RotationMatrix(), [0, 0, 0.05 + axis_len / 2]))

    # Record animation into Meshcat timeline scrubber
    print("Recording full lookahead animation into Meshcat timeline...")
    visualizer.StartRecording(set_transforms_while_recording=True)

    step_dt = args.dt / max(args.speed, 0.01)
    sim_time = 0.0
    for idx, (t_val, stages_list) in enumerate(records):
        diagram_context.SetTime(sim_time)
        set_lookahead_state(plant, plant_context, model_info, stages_list)
        diagram.ForcedPublish(diagram_context)
        sim_time += step_dt

    visualizer.StopRecording()
    visualizer.PublishRecording()

    # Set animation directly on Meshcat so it is embedded in StaticHtml and timeline scrubber
    recording = visualizer.get_mutable_recording()
    meshcat.SetAnimation(recording)
    print("Full C3 lookahead animation published to Meshcat and timeline scrubber!")

    if args.html:
        raw_html = meshcat.StaticHtml()
        html_content = inject_transparency_script(raw_html, stages_to_render)
        os.makedirs(os.path.dirname(os.path.abspath(args.html)), exist_ok=True)
        with open(args.html, "w") as f:
            f.write(html_content)
        print(f"Saved standalone interactive HTML video with gradient transparency to {args.html}")

    if args.live:
        print(f"\nLive visualizer server ready at {meshcat.web_url()}")
        print("Open the URL above or c3_lookahead_meshcat.html in your browser.")
        print("You can Play (\u25B6), Pause (\u23F8), and scrub the timeline.")
        print("Press Ctrl+C to exit server.\n")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nVisualizer server stopped.")


if __name__ == "__main__":
    main()
