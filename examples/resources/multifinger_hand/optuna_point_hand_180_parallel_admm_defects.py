#!/usr/bin/env python3
import sys
import os
import re
import math
import subprocess
import platform
import yaml
import numpy as np
import optuna

# python3 examples/resources/multifinger_hand/optuna_point_hand_180_parallel_admm_defects.py

def get_quaternion_angle_diff(q1, q2):
    """Computes the angular difference (in degrees) between two quaternions."""
    dot_product = sum(a * b for a, b in zip(q1, q2))
    dot_product = max(min(dot_product, 1.0), -1.0)
    return 2 * math.acos(abs(dot_product)) * 180.0 / math.pi

if len(sys.argv) > 1:
    worker_id = int(sys.argv[1])
    print(f"Running worker number: {worker_id}")
else:
    worker_id = 0
    print("No integer passed, defaulting to 0")

# Base template parameter file paths
BASE_CONTROLLER_PARAMS = "examples/resources/multifinger_hand/ms_c3_tracking_options_point_hand_180.yaml"
BASE_MSIC3_PARAMS = "examples/resources/multifinger_hand/ms_ic3_options_point_hand_180.yaml"

# Parameter file paths for this worker
OUTPUT_DIR = "examples/resources/multifinger_hand/optuna_point_hand_180_parallel_admm"
CONTROLLER_PARAMS = f"examples/resources/multifinger_hand/optuna_point_hand_180/optuna_yamls/optuna_ms_c3_tracking_options_point_hand_180_{worker_id}.yaml"
MSIC3_PARAMS = f"examples/resources/multifinger_hand/optuna_point_hand_180/optuna_yamls/optuna_ms_ic3_options_point_hand_180_{worker_id}.yaml"

os.makedirs(os.path.dirname(CONTROLLER_PARAMS), exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def objective(trial):
    # =========================================================================
    # 1. C3 Tracking Controller Parameters
    # =========================================================================
    w_G = trial.suggest_int("w_G", 1, 500)
    g_lambda = trial.suggest_int("g_lambda", 1, 100)
    g_eta = trial.suggest_int("g_eta", 1, 100)

    u_ratio_finger = trial.suggest_int("u_ratio_finger", -200, 199)
    u_ratio_cube = trial.suggest_int("u_ratio_cube", -200, 199)

    admm_iter = trial.suggest_int("admm_iter", 3, 7)
    tracking_N = 4

    finger_position_weight = trial.suggest_int("finger_position_weight", 1000, 50000, step=1000)
    cube_position_weight = trial.suggest_int("cube_position_weight", 1000, 50000, step=1000)
    quat_weight = trial.suggest_int("quat_weight", 1000, 10000, step=500)

    w_G_final = 10
    x_change_weight = 1
    u_change_weight = 1

    finger_config = 1
    cube_model = 1
    n_contacts = 7

    # Load from clean base template
    with open(BASE_CONTROLLER_PARAMS, "r") as f:
        c3_options = yaml.safe_load(f)

    c3_options["c3_options"]["penalize_input_change"] = True
    c3_options["c3_options"]["penalize_x_change"] = True
    c3_options["c3_options"]["input_change_weight"] = (u_change_weight - 1) / 100.0
    c3_options["c3_options"]["x_change_weight"] = (x_change_weight - 1) / 100.0

    c3_options["c3_options"]["w_G"] = w_G / 100.0
    c3_options["c3_options"]["g_lambda"] = [g_lambda] * (4 * n_contacts)
    c3_options["c3_options"]["g_eta"] = [g_eta] * (4 * n_contacts)
    c3_options["c3_options"]["g_gamma"] = []
    c3_options["c3_options"]["g_lambda_n"] = []
    c3_options["c3_options"]["g_lambda_t"] = []
    c3_options["c3_options"]["g_eta_slack"] = []
    c3_options["c3_options"]["g_eta_n"] = []
    c3_options["c3_options"]["g_eta_t"] = []
    c3_options["c3_options"]["w_G_final"] = w_G_final
    c3_options["c3_options"]["admm_iter"] = admm_iter

    # Set U weights
    finger_ratio = abs(u_ratio_finger)
    if u_ratio_finger < 0:
        u_lambda_finger = 1.0
        u_eta_finger = finger_ratio
    else:
        finger_ratio += 1
        u_lambda_finger = finger_ratio
        u_eta_finger = 1.0

    cube_ratio = abs(u_ratio_cube)
    if u_ratio_cube < 0:
        u_lambda_cube = 1.0
        u_eta_cube = cube_ratio
    else:
        cube_ratio += 1
        u_lambda_cube = cube_ratio
        u_eta_cube = 1.0

    c3_options["c3_options"]["u_lambda"] = [u_lambda_finger] * (4 * 3) + [u_lambda_cube] * (4 * (n_contacts - 3))
    c3_options["c3_options"]["u_eta"] = [u_eta_finger] * (4 * 3) + [u_eta_cube] * (4 * (n_contacts - 3))
    c3_options["c3_options"]["u_gamma"] = []
    c3_options["c3_options"]["u_lambda_n"] = []
    c3_options["c3_options"]["u_lambda_t"] = []
    c3_options["c3_options"]["u_eta_slack"] = []
    c3_options["c3_options"]["u_eta_n"] = []
    c3_options["c3_options"]["u_eta_t"] = []

    c3_options["lcs_factory_options"]["N"] = tracking_N

    # Update q_vector weights
    q_vector = list(c3_options["c3_options"]["q_vector"])
    for i in range(9):
        q_vector[i] = finger_position_weight
    q_vector[13] = cube_position_weight
    q_vector[14] = cube_position_weight
    q_vector[15] = 1000
    for i in range(15):
        q_vector[16 + i] = 1
    c3_options["c3_options"]["q_vector"] = q_vector
    c3_options["Q_quaternion_weight"] = quat_weight

    with open(CONTROLLER_PARAMS, "w") as f:
        yaml.dump(c3_options, f, default_flow_style=True)

    # =========================================================================
    # 2. MS-iC3 ADMM Options & Hyperparameters
    # =========================================================================
    num_iters = trial.suggest_int("num_iters", 4, 16)
    num_segments = trial.suggest_categorical("num_segments", [10, 20, 30, 42, 60, 70])

    anchor_rho = trial.suggest_float("anchor_rho", 0.5, 50.0, step=0.5)

    accel_cost = 10
    use_pd = False

    with open(BASE_MSIC3_PARAMS, "r") as f:
        ic3_options = yaml.safe_load(f)

    ic3_options["N"] = 420
    ic3_options["num_segments"] = num_segments
    ic3_options["num_iters"] = num_iters
    ic3_options["num_warmup_iters"] = 0
    ic3_options["warm_start_alpha"] = 0.0

    ic3_options["anchor_rho"] = float(anchor_rho)

    ic3_options["acceleration_cost_weight"] = accel_cost

    kp = 200 if use_pd else 0
    kd = 20 if use_pd else 0

    ic3_options["rollout_Kp"] = [kp] * 9
    ic3_options["rollout_Kd"] = [kd] * 9
    ic3_options["rollout_dt_scaling"] = 10
    ic3_options["print_costs"] = False
    ic3_options["num_threads"] = 32

    ic3_options["use_drake_sim"] = True
    ic3_options["drake_sim_dt"] = 0.0001
    ic3_options["use_rollout_lambdas"] = True
    ic3_options["use_lambdas_for_lcs"] = False

    with open(MSIC3_PARAMS, "w") as f:
        yaml.dump(ic3_options, f, default_flow_style=True)

    # =========================================================================
    # 3. Execute Parallel MSiC3 ADMM
    # =========================================================================
    cmd = [
        "./bazel-bin/examples/lcs_factory_system_example",
        f"--optuna_instance={worker_id}",
        "--experiment_type=MSiC3_point_hand_180_optuna_parallel_admm",
        f"--ee_config={finger_config}",
        f"--cube_model={cube_model}"
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    full_output = []
    current_iter = 0

    # Defect tracking per iteration: dict mapping iter -> dict of segment defects
    # { iter: { seg_idx: { 'ee': float, 'quat': float, 'obj': float } } }
    iter_defects = {}
    angle_diff = None
    position_weight = None
    final_score_reported = None

    ee_defect_regex = re.compile(r"Segment\s+(\d+)\s+ee defect:\s*([^\n]+)")
    quat_defect_regex = re.compile(r"Segment\s+(\d+)\s+quaternion defect \(deg\):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    obj_defect_regex = re.compile(r"Segment\s+(\d+)\s+object defect:\s*([^\n]+)")

    try:
        for line in iter(process.stdout.readline, ''):
            full_output.append(line)

            # Check solver failures
            if "LCP failed: returning x_init" in line:
                raise optuna.TrialPruned("LCP solver failed")

            primal_match = re.search(r"Primal Res:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
            dual_match = re.search(r"Dual Res:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
            if primal_match and float(primal_match.group(1)) > 0.01:
                raise optuna.TrialPruned(f"Large Primal Residual ({float(primal_match.group(1))})")
            if dual_match and float(dual_match.group(1)) > 0.01:
                raise optuna.TrialPruned(f"Large Dual Residual ({float(dual_match.group(1))})")

            # Track iteration number
            if line.startswith("iC3 iteration"):
                try:
                    current_iter = int(line.split()[-1])
                    if current_iter not in iter_defects:
                        iter_defects[current_iter] = {}
                except ValueError:
                    pass

            # Parse EE defect
            m_ee = ee_defect_regex.search(line)
            if m_ee and current_iter > 0:
                seg_idx = int(m_ee.group(1))
                vals = [float(v) for v in m_ee.group(2).split() if v.strip()]
                norm = float(np.linalg.norm(vals))
                if current_iter not in iter_defects:
                    iter_defects[current_iter] = {}
                if seg_idx not in iter_defects[current_iter]:
                    iter_defects[current_iter][seg_idx] = {}
                iter_defects[current_iter][seg_idx]["ee"] = norm

            # Parse Quaternion defect (deg)
            m_quat = quat_defect_regex.search(line)
            if m_quat and current_iter > 0:
                seg_idx = int(m_quat.group(1))
                deg = float(m_quat.group(2))
                if current_iter not in iter_defects:
                    iter_defects[current_iter] = {}
                if seg_idx not in iter_defects[current_iter]:
                    iter_defects[current_iter][seg_idx] = {}
                iter_defects[current_iter][seg_idx]["quat"] = deg

            # Parse Object defect
            m_obj = obj_defect_regex.search(line)
            if m_obj and current_iter > 0:
                seg_idx = int(m_obj.group(1))
                vals = [float(v) for v in m_obj.group(2).split() if v.strip()]
                norm = float(np.linalg.norm(vals))
                if current_iter not in iter_defects:
                    iter_defects[current_iter] = {}
                if seg_idx not in iter_defects[current_iter]:
                    iter_defects[current_iter][seg_idx] = {}
                iter_defects[current_iter][seg_idx]["obj"] = norm

            # Intermediate pruning check on iteration end
            if "Iteration runtime:" in line and current_iter in iter_defects:
                current_seg_dict = iter_defects[current_iter]
                if len(current_seg_dict) > 0:
                    quats = [d.get("quat", 0.0) for d in current_seg_dict.values()]
                    mean_quat_defect = np.mean(quats) if quats else 0.0
                    # If quaternion defect is catastrophically high after a few iterations, prune
                    if current_iter >= 3 and mean_quat_defect > 80.0:
                        raise optuna.TrialPruned(f"Pruned at iteration {current_iter}: high mean quat defect ({mean_quat_defect:.2f} deg)")

            # Final metrics
            if "Angle diff:" in line:
                m = re.search(r"Angle diff:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                if m:
                    angle_diff = float(m.group(1))
            if "Position weight" in line:
                m = re.search(r"Position weight\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                if m:
                    position_weight = float(m.group(1))
            if "FINAL_METRIC:" in line:
                m = re.search(r"FINAL_METRIC:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                if m:
                    final_score_reported = float(m.group(1))

    except optuna.TrialPruned as e:
        process.terminate()
        process.wait()
        raise e
    finally:
        process.stdout.close()
        process.stderr.close()

    process.wait()
    if process.returncode != 0:
        print("".join(full_output))
        raise optuna.TrialPruned(f"Process failed with exit code {process.returncode}")

    # =========================================================================
    # 4. Compute Final Iteration Defects Objective
    # =========================================================================
    valid_iters = sorted([it for it in iter_defects.keys() if len(iter_defects[it]) > 0])
    if not valid_iters:
        print("".join(full_output))
        raise optuna.TrialPruned("No valid iteration defect logs found.")

    final_iter = valid_iters[-1]
    final_defects = iter_defects[final_iter]

    total_ee_defect = sum(d.get("ee", 0.0) for d in final_defects.values())
    total_quat_defect = sum(d.get("quat", 0.0) for d in final_defects.values())
    total_obj_defect = sum(d.get("obj", 0.0) for d in final_defects.values())
    max_quat_defect = max((d.get("quat", 0.0) for d in final_defects.values()), default=0.0)
    max_obj_defect = max((d.get("obj", 0.0) for d in final_defects.values()), default=0.0)

    # Goal tracking error
    target_angle_err = angle_diff if angle_diff is not None else 180.0
    pos_err = (position_weight / 15000.0) if position_weight is not None else 0.0

    # Compute individual metric contributions
    cost_total_quat = 5.0 * total_quat_defect
    cost_max_quat = 20.0 * max_quat_defect
    cost_total_obj = 50.0 * total_obj_defect
    cost_max_obj = 50.0 * max_obj_defect
    cost_total_ee = 3000.0 * total_ee_defect
    cost_target_angle = 3.0 * target_angle_err
    cost_pos = 50.0 * pos_err

    defect_metric = (
        cost_total_quat +
        cost_max_quat +
        cost_total_obj +
        cost_max_obj +
        cost_total_ee +
        cost_target_angle +
        cost_pos
    )

    # Record individual components and contributions for logging and analysis
    trial.set_user_attr("final_iter", int(final_iter))
    trial.set_user_attr("total_quat_defect_deg", float(total_quat_defect))
    trial.set_user_attr("cost_total_quat", float(cost_total_quat))
    trial.set_user_attr("max_quat_defect_deg", float(max_quat_defect))
    trial.set_user_attr("cost_max_quat", float(cost_max_quat))
    trial.set_user_attr("total_obj_defect", float(total_obj_defect))
    trial.set_user_attr("cost_total_obj", float(cost_total_obj))
    trial.set_user_attr("max_obj_defect", float(max_obj_defect))
    trial.set_user_attr("cost_max_obj", float(cost_max_obj))
    trial.set_user_attr("total_ee_defect", float(total_ee_defect))
    trial.set_user_attr("cost_total_ee", float(cost_total_ee))
    trial.set_user_attr("target_angle_err_deg", float(target_angle_err))
    trial.set_user_attr("cost_target_angle", float(cost_target_angle))
    trial.set_user_attr("pos_err", float(pos_err))
    trial.set_user_attr("cost_pos", float(cost_pos))

    print(f"\n[Trial #{trial.number} Result - Final Iter {final_iter}]")
    print(f"  Total Quat Defect:  {total_quat_defect:8.3f} deg   --> Contribution: {cost_total_quat:8.3f} ({cost_total_quat/max(defect_metric, 1e-6)*100:5.1f}%)")
    print(f"  Max Quat Defect:    {max_quat_defect:8.3f} deg   --> Contribution: {cost_max_quat:8.3f} ({cost_max_quat/max(defect_metric, 1e-6)*100:5.1f}%)")
    print(f"  Total Obj Defect:   {total_obj_defect:8.5f} m     --> Contribution: {cost_total_obj:8.3f} ({cost_total_obj/max(defect_metric, 1e-6)*100:5.1f}%)")
    print(f"  Max Obj Defect:     {max_obj_defect:8.5f} m     --> Contribution: {cost_max_obj:8.3f} ({cost_max_obj/max(defect_metric, 1e-6)*100:5.1f}%)")
    print(f"  Total EE Defect:    {total_ee_defect:8.5f} m     --> Contribution: {cost_total_ee:8.3f} ({cost_total_ee/max(defect_metric, 1e-6)*100:5.1f}%)")
    print(f"  Target Angle Error: {target_angle_err:8.3f} deg   --> Contribution: {cost_target_angle:8.3f} ({cost_target_angle/max(defect_metric, 1e-6)*100:5.1f}%)")
    print(f"  Position Error:     {pos_err:8.5f}       --> Contribution: {cost_pos:8.3f} ({cost_pos/max(defect_metric, 1e-6)*100:5.1f}%)")
    print(f"  -------------------------------------------------------------")
    print(f"  Composite Defect Score: {defect_metric:.4f}\n")

    return defect_metric

def format_component_scores(user_attrs):
    lines = ["Component Scores & Metric Contributions:"]
    lines.append(f"  Final Iteration:           {user_attrs.get('final_iter', 'N/A')}")
    
    t_q = user_attrs.get('total_quat_defect_deg')
    c_t_q = user_attrs.get('cost_total_quat')
    lines.append(f"  Total Quat Defect:         {t_q:.3f} deg (Contribution: {c_t_q:.3f})" if isinstance(t_q, (int, float)) and isinstance(c_t_q, (int, float)) else f"  Total Quat Defect:         {t_q}")
    
    m_q = user_attrs.get('max_quat_defect_deg')
    c_m_q = user_attrs.get('cost_max_quat')
    lines.append(f"  Max Quat Defect:           {m_q:.3f} deg (Contribution: {c_m_q:.3f})" if isinstance(m_q, (int, float)) and isinstance(c_m_q, (int, float)) else f"  Max Quat Defect:           {m_q}")
    
    t_o = user_attrs.get('total_obj_defect')
    c_t_o = user_attrs.get('cost_total_obj')
    lines.append(f"  Total Obj Defect:          {t_o:.5f} m (Contribution: {c_t_o:.3f})" if isinstance(t_o, (int, float)) and isinstance(c_t_o, (int, float)) else f"  Total Obj Defect:          {t_o}")
    
    m_o = user_attrs.get('max_obj_defect')
    c_m_o = user_attrs.get('cost_max_obj')
    lines.append(f"  Max Obj Defect:            {m_o:.5f} m (Contribution: {c_m_o:.3f})" if isinstance(m_o, (int, float)) and isinstance(c_m_o, (int, float)) else f"  Max Obj Defect:            {m_o}")
    
    t_e = user_attrs.get('total_ee_defect')
    c_t_e = user_attrs.get('cost_total_ee')
    lines.append(f"  Total EE Defect:           {t_e:.5f} m (Contribution: {c_t_e:.3f})" if isinstance(t_e, (int, float)) and isinstance(c_t_e, (int, float)) else f"  Total EE Defect:           {t_e}")
    
    t_a = user_attrs.get('target_angle_err_deg')
    c_t_a = user_attrs.get('cost_target_angle')
    lines.append(f"  Target Angle Error:        {t_a:.3f} deg (Contribution: {c_t_a:.3f})" if isinstance(t_a, (int, float)) and isinstance(c_t_a, (int, float)) else f"  Target Angle Error:        {t_a}")
    
    p_e = user_attrs.get('pos_err')
    c_p_e = user_attrs.get('cost_pos')
    lines.append(f"  Position Error:            {p_e:.5f} (Contribution: {c_p_e:.3f})" if isinstance(p_e, (int, float)) and isinstance(c_p_e, (int, float)) else f"  Position Error:            {p_e}")
    
    return "\n".join(lines)

def log_best_callback(study, trial):
    if trial.value is None:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    attrs = trial.user_attrs
    
    # Save sub-threshold trials
    if trial.value < 50:
        with open(f"{OUTPUT_DIR}/sub_50_defect_trials.txt", "a") as f:
            f.write(f"Trial #{trial.number} | Defect Score: {trial.value:.4f}\n")
            f.write(format_component_scores(attrs) + "\n")
            f.write("Parameters:\n")
            for key, value in trial.params.items():
                f.write(f"  {key}: {value}\n")
            f.write("-" * 50 + "\n")

    if study.best_trial.number == trial.number:
        print(f"\n--> [NEW BEST] Metric: {trial.value:.4f}. Saving to file...")
        with open(f"{OUTPUT_DIR}/best_params_180_parallel_admm_defects.txt", "w") as f:
            f.write("=========================================\n")
            f.write("       BEST HYPERPARAMETERS SO FAR       \n")
            f.write("=========================================\n")
            f.write(f"Best Trial Number: {trial.number}\n")
            f.write(f"Best Defect Metric: {trial.value:.4f}\n\n")
            f.write(format_component_scores(attrs) + "\n\n")
            f.write("Parameters:\n")
            for key, value in trial.params.items():
                f.write(f"  {key}: {value}\n")
            f.write("=========================================\n")

if __name__ == "__main__":
    print(f"OS Release: {platform.release().lower()}")
    if "microsoft" in platform.release().lower():
        STORAGE_URL = "sqlite:////home/ttesc255/optuna_data/optuna_results_180_parallel_admm_defects.db"
    else:
        STORAGE_URL = f"sqlite:///{OUTPUT_DIR}/optuna_results_180_parallel_admm_defects.db"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True)
    storage = optuna.storages.RDBStorage(
        url=STORAGE_URL,
        heartbeat_interval=60
    )

    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study = optuna.create_study(
        study_name="MSiC3_point_hand_180_parallel_admm_defects",
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        direction="minimize"
    )
    
    print(f"Starting Optuna study: MSiC3_point_hand_180_parallel_admm_defects (Worker {worker_id})")
    study.optimize(objective, n_trials=5000, callbacks=[log_best_callback])

    print("\n--- Optimization Complete ---")
    print(f"Best Trial Value: {study.best_value}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
