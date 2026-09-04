import yaml
import math
import re
import subprocess
import optuna
import sys
import platform

def get_quaternion_angle_diff(q1, q2):
    """Computes the angular difference (in degrees) between two quaternions."""
    # 1. Compute the dot product
    dot_product = sum(a * b for a, b in zip(q1, q2))
    
    # 2. Clamp the dot product to [-1, 1] to avoid math domain errors from floating point drift
    dot_product = max(min(dot_product, 1.0), -1.0)
    
    # 3. Calculate the angle in degrees
    angle_deg = 2 * math.acos(abs(dot_product)) * 180.0 / math.pi
    return angle_deg

if len(sys.argv) > 1:
    worker_id = int(sys.argv[1])
    print(f"Running worker number: {worker_id}")
else:
    # Fallback if you forget to pass the integer
    worker_id = 0 
    print("No integer passed, defaulting to 0")


# Define paths to your parameter files
CONTORLLER_PARAMS = f"examples/resources/multifinger_hand/optuna_point_hand_pivot/optuna_yamls/optuna_ms_c3_tracking_options_pivot_{worker_id}.yaml"
MSiC3_PARAMS = f"examples/resources/multifinger_hand/optuna_point_hand_pivot/optuna_yamls/optuna_ms_ic3_options_pivot_{worker_id}.yaml"

def objective(trial):
    # C3 parameters
    w_G = trial.suggest_int("w_G", 1, 5000)
    # g_x_fingers = trial.suggest_int("g_x_fingers", 10, 100, step=10)
    # g_x_cube = trial.suggest_int("g_x_cube", 10, 100, step=10)
    # g_u = trial.suggest_int("g_u", 10, 100, step=10)
    g_x_fingers = 50
    g_x_cube = 50
    g_u = 50

    g_lambda = trial.suggest_int("g_lambda", 2, 100, step=2)
    g_eta = trial.suggest_int("g_eta", 2, 100, step=2)

    w_G_final = trial.suggest_int("w_G_final", 1, 100)

    u_ratio_finger = trial.suggest_int("u_ratio_finger", -200, 199) # -100 = lambda/eta = 0.1 
    u_ratio_cube = trial.suggest_int("u_ratio_cube", -200, 199) # -100 = lambda/eta = 0.1

    # lambda_threshold = trial.suggest_int("lambda_threshold", 0, 60)
    # eta_threshold = trial.suggest_int("eta_threshold", 0, 10)
    lambda_threshold = 0
    eta_threshold = 0

    admm_iter = trial.suggest_int("admm_iter", 3, 6)
    tracking_N = trial.suggest_int("tracking_N", 3, 10)
    finger_position_weight = trial.suggest_int("finger_position_weight", 200, 10000, step=200)
    cube_position_weight = trial.suggest_int("cube_position_weight", 200, 10000, step=200)
    quat_weight = trial.suggest_int("quat_weight", 5000, 500000, step=5000)

    finger_config = trial.suggest_categorical("finger_config", [2, 3])
    # finger_config = 1
    
    x_change_weight = trial.suggest_int("x_change_weight", 1, 1001, log=True)
    u_change_weight = trial.suggest_int("u_change_weight", 1, 1001, log=True)

    cube_model = trial.suggest_int("cube_model", 1, 10)
    mu_finger_cube = trial.suggest_int("mu_finger_cube", 1, 100)

    towards_2_fingers = trial.suggest_categorical("towards_2_fingers", [True, False])

    with open(CONTORLLER_PARAMS, "r") as f:
        c3_options = yaml.safe_load(f)
        
    n_contacts = 11

    c3_options["c3_options"]["w_G"] = w_G / 100.0
    c3_options["c3_options"]["g_lambda"] = [g_lambda] * (4*n_contacts)
    c3_options["c3_options"]["g_eta_slack"] = []
    c3_options["c3_options"]["g_eta_n"] = []
    c3_options["c3_options"]["g_eta_t"] = []
    c3_options["c3_options"]["g_eta"] = [g_eta] * (4*n_contacts)

    c3_options["c3_options"]["w_G_final"] = w_G_final

    finger_ratio = abs(u_ratio_finger) 
    if (u_ratio_finger < 0):
        u_lambda_finger = 1.0
        u_eta_finger = finger_ratio
    else:
        finger_ratio += 1
        u_lambda_finger = finger_ratio
        u_eta_finger = 1.0

    cube_ratio = abs(u_ratio_cube)
    if (u_ratio_cube < 0):
        u_lambda_cube = 1.0
        u_eta_cube = cube_ratio
    else:
        cube_ratio += 1
        u_lambda_cube = cube_ratio
        u_eta_cube = 1.0

    c3_options["c3_options"]["u_lambda"] = [u_lambda_finger] * (4*3) + [u_lambda_cube] * (4*(n_contacts-3))
    c3_options["c3_options"]["u_eta"] = [u_eta_finger] * (4*3) + [u_eta_cube] * (4*(n_contacts-3))

    c3_options["c3_options"]["u_eta_slack"] = []
    c3_options["c3_options"]["u_eta_n"] = [] 
    c3_options["c3_options"]["u_eta_t"] = []

    c3_options["c3_options"]["lambda_threshold"] = [(lambda_threshold / 100.0)] * (4 * 3) + [0] * (4 * (n_contacts - 3))
    c3_options["c3_options"]["eta_threshold"] = [(eta_threshold / 10.0)] * (4 * 3) + [0] * (4 * (n_contacts - 3))

    c3_options["c3_options"]["admm_iter"] = admm_iter

    c3_options["Q_quaternion_weight"] = quat_weight

    c3_options["c3_options"]["penalize_input_change"] = True
    c3_options["c3_options"]["penalize_x_change"] = True
    c3_options["c3_options"]["input_change_weight"] = (u_change_weight-1) / 100.0
    c3_options["c3_options"]["x_change_weight"] = (x_change_weight-1) / 100.0

    for i in range(9):
        c3_options["c3_options"]["g_x"][i] = g_x_fingers
        c3_options["c3_options"]["u_x"][i] = 1

    c3_options["c3_options"]["g_u"] = [g_u] * 9
    c3_options["c3_options"]["u_u"] = [1] * 9

    for i in range(7):
        c3_options["c3_options"]["g_x"][9+i] = g_x_cube
        c3_options["c3_options"]["u_x"][9+i] = 1

    c3_options["c3_options"]["q_vector"][13] = cube_position_weight
    c3_options["c3_options"]["q_vector"][14] = cube_position_weight
    c3_options["c3_options"]["q_vector"][15] = 100

    for i in range(9): 
        c3_options["c3_options"]["q_vector"][i] = finger_position_weight

    c3_options["lcs_factory_options"]["num_contacts"] = 11

    mu_fc = mu_finger_cube / 100.0
    c3_options["lcs_factory_options"]["mu"] = [mu_fc, mu_fc, mu_fc, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    c3_options["lcs_factory_options"]["N"] = tracking_N
    c3_options["lcs_factory_options"]["dt"] = 0.02

    if (finger_config == 1):
        c3_options["x_init"] = [0.0, 0.07, 0.05,  # finger 1 
                                0.07, -0.055, 0.05,   # finger 2
                                -0.07, -0.055, 0.05,   # finger 3
                                1, 0, 0, 0, # cube orientation
                                0, 0, 0.052,  # cube position
                                0, 0, 0,     # finger 1 velo
                                0, 0, 0,     # finger 2 velo
                                0, 0, 0,     # finger 3 velo
                                0, 0, 0,     # cube ang velo
                                0, 0, 0]   	# cube velo
        
        c3_options["x_des"] = [0.0, 0.07, 0.05,  # finger 1 
                                0.07, -0.055, 0.05,   # finger 2
                                -0.07, -0.055, 0.05,   # finger 3
                                0, 1, 0, 0, # cube orientation
                                0, 0, 0.052,  # cube position
                                0, 0, 0,     # finger 1 velo
                                0, 0, 0,     # finger 2 velo
                                0, 0, 0,     # finger 3 velo
                                0, 0, 0,     # cube ang velo
                                0, 0, 0]   	# cube velo
    elif (finger_config == 2):
        c3_options["x_init"] = [0.0, 0.07, 0.05,  # finger 1 
                                0.07, -0.045, 0.05,   # finger 2
                                -0.07, -0.045, 0.05,   # finger 3
                                1, 0, 0, 0, # cube orientation
                                0, 0, 0.052,  # cube position
                                0, 0, 0,     # finger 1 velo
                                0, 0, 0,     # finger 2 velo
                                0, 0, 0,     # finger 3 velo
                                0, 0, 0,     # cube ang velo
                                0, 0, 0]   	# cube velo
        
        c3_options["x_des"] = [0.0, 0.07, 0.05,  # finger 1 
                                0.07, -0.045, 0.05,   # finger 2
                                -0.07, -0.045, 0.05,   # finger 3
                                0, 1, 0, 0, # cube orientation
                                0, 0, 0.052,  # cube position
                                0, 0, 0,     # finger 1 velo
                                0, 0, 0,     # finger 2 velo
                                0, 0, 0,     # finger 3 velo
                                0, 0, 0,     # cube ang velo
                                0, 0, 0]   	# cube velo
    elif (finger_config == 3):
        c3_options["x_init"] = [0.0, 0.07, 0.05,  # finger 1 
                                0.07, -0.03, 0.05,   # finger 2
                                -0.07, -0.03, 0.05,   # finger 3
                                1, 0, 0, 0, # cube orientation
                                0, 0, 0.052,  # cube position
                                0, 0, 0,     # finger 1 velo
                                0, 0, 0,     # finger 2 velo
                                0, 0, 0,     # finger 3 velo
                                0, 0, 0,     # cube ang velo
                                0, 0, 0]   	# cube velo
        
        c3_options["x_des"] = [0.0, 0.07, 0.05,  # finger 1 
                                0.07, -0.03, 0.05,   # finger 2
                                -0.07, -0.03, 0.05,   # finger 3
                                0, 1, 0, 0, # cube orientation
                                0, 0, 0.052,  # cube position
                                0, 0, 0,     # finger 1 velo
                                0, 0, 0,     # finger 2 velo
                                0, 0, 0,     # finger 3 velo
                                0, 0, 0,     # cube ang velo
                                0, 0, 0]   	# cube velo

    if (not towards_2_fingers):
        c3_options["x_des"][10] = -1

    c3_options["c3_options"]["w_Q"] = 5
    c3_options["c3_options"]["w_R"] = 50
    c3_options["c3_options"]["w_U"] = 1
    c3_options["c3_options"]["scale_lcs"] = True

    c3_options["c3_options"]["add_phi_buffer"] = False  
    c3_options["c3_options"]["epsilon"] = []
    c3_options["c3_options"]["phi_threshold"] = []
    c3_options["c3_options"]["gamma_threshold"] = []

    with open(CONTORLLER_PARAMS, "w") as f:
        yaml.dump(c3_options, f, default_flow_style=True)

# ======================================================================

    # iC3 parameters
    # num_warmup_iters = trial.suggest_int("num_warmup_iters", 0, 2)
    num_warmup_iters = 0
    # warm_start_alpha = trial.suggest_int("warm_start_alpha", 0, 100)
    warm_start_alpha = 0 

    num_segments = trial.suggest_categorical("num_segments", [5, 10, 15, 20, 30, 40, 60])

    num_iters = trial.suggest_int("num_iters", 2, 12)
    # alpha_ee = trial.suggest_int("alpha_ee", 0, 100)
    # alpha_object = trial.suggest_int("alpha_object", 0, 100)
    # alpha_ee_step = trial.suggest_int("alpha_ee_step", 0, 100)
    # alpha_object_step = trial.suggest_int("alpha_object_step", 0, 100)
    alpha_ee = 0
    alpha_object = 0
    alpha_ee_step = 0
    alpha_object_step = 0

    # use_pd = trial.suggest_categorical("use_pd", [True, False])
    use_pd = False
    rho = trial.suggest_int("rho", 1, 1000)

    # value_function_scaling = trial.suggest_int("value_function_scaling", 0, 100)
    # use_value_function = trial.suggest_categorical("use_value_function", [True, False])
    use_value_function = True

    value_function_scaling = 100 if use_value_function else 0
    vf_trust_region_weight = trial.suggest_int("vf_trust_region_weight", 0, 100)

    accel_cost = 5

    # use_rollout_lambdas = trial.suggest_categorical("use_rollout_lambdas", [True, False])
    use_rollout_lambdas = True

    with open(MSiC3_PARAMS, "r") as f:
        ic3_options = yaml.safe_load(f)
        
    ic3_options["N"] = 360
    ic3_options["num_segments"] = num_segments

    ic3_options["num_warmup_iters"] = num_warmup_iters

    warm_start_alpha_real = warm_start_alpha / 100.0
    ic3_options["warm_start_alpha"] = warm_start_alpha_real

    ic3_options["num_iters"] = num_iters

    alpha_ee_real = alpha_ee / 100.0
    ic3_options["alpha_ee"] = alpha_ee_real
    # ic3_options["alpha_ee_step"] = (1 - alpha_ee_real) / (num_iters - 1)
    ic3_options["alpha_ee_step"] = alpha_ee_step

    alpha_obj_real = alpha_object / 100.0
    ic3_options["alpha_object"] = alpha_obj_real
    # ic3_options["alpha_object_step"] = (1 - alpha_obj_real) / (num_iters - 1)
    ic3_options["alpha_object_step"] = alpha_object_step

    ic3_options["acceleration_cost_weight"] = accel_cost
    ic3_options["value_function_scaling"] = value_function_scaling / 100.0
    ic3_options["vf_trust_region_weight"] = vf_trust_region_weight

    Kp = 200 if use_pd else 0
    Kd = 20 if use_pd else 0
    ic3_options["rollout_Kp"] = [Kp] * 9
    ic3_options["rollout_Kd"] = [Kd] * 9
    ic3_options["rollout_dt_scaling"] = 10

    ic3_options["print_costs"] = False

    ic3_options["use_drake_sim"] = True
    ic3_options["drake_sim_dt"] = 0.0001

    ic3_options["num_threads"] = 32

    ic3_options["use_rollout_lambdas"] = use_rollout_lambdas
    ic3_options["anchor_rho"] = rho / 10.0

    if "p_vector" in ic3_options:
        del ic3_options["p_vector"]
    if "w_P" in ic3_options:
        del ic3_options["w_P"]

    ic3_options["add_terminal_constraint"] = False
    ic3_options["terminal_slack_vector"] = [1] * 31
    ic3_options["terminal_slack_quaternion_weight"] = 0

    with open(MSiC3_PARAMS, "w") as f:
        yaml.dump(ic3_options, f, default_flow_style=True)

    # Construct and execute the bazel command
    cmd = [
        "./bazel-bin/examples/lcs_factory_system_example", 
        f"--optuna_instance={worker_id}", 
        "--experiment_type=MSiC3_point_hand_optuna_parallel",
        f"--ee_config={finger_config}",
        f"--cube_model={cube_model}"
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)


    # --- State Variables for the iC3 Iteration Defect Tracking ---
    current_iter = 0
    iter_defects = {}
    angle_diff = None
    position_weight = None

    ee_defect_regex = re.compile(r"Segment\s+(\d+)\s+ee defect:\s*([^\n]+)")
    quat_defect_regex = re.compile(r"Segment\s+(\d+)\s+quaternion defect \(deg\):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    obj_defect_regex = re.compile(r"Segment\s+(\d+)\s+object defect:\s*([^\n]+)")

    try:
        for line in iter(process.stdout.readline, ''):
            full_output.append(line)

            # 1. Check for Early Solver Failures
            if "LCP failed: returning x_init" in line:
                raise optuna.TrialPruned("LCP solver failed")
            
            primal_match = re.search(r"Primal Res:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
            dual_match = re.search(r"Dual Res:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
            
            if primal_match and float(primal_match.group(1)) > 0.01:
                raise optuna.TrialPruned(f"Large Primal Residual ({float(primal_match.group(1))})")
            if dual_match and float(dual_match.group(1)) > 0.01:
                raise optuna.TrialPruned(f"Large Dual Residual ({float(dual_match.group(1))})")

            # 2. Track iC3 iteration
            if line.startswith("iC3 iteration"):
                try:
                    current_iter = int(line.split()[-1])
                    if current_iter not in iter_defects:
                        iter_defects[current_iter] = {}
                except ValueError:
                    pass

            # 3. Parse EE defect
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

            # 4. Parse Quaternion defect (deg)
            m_quat = quat_defect_regex.search(line)
            if m_quat and current_iter > 0:
                seg_idx = int(m_quat.group(1))
                deg = float(m_quat.group(2))
                if current_iter not in iter_defects:
                    iter_defects[current_iter] = {}
                if seg_idx not in iter_defects[current_iter]:
                    iter_defects[current_iter][seg_idx] = {}
                iter_defects[current_iter][seg_idx]["quat"] = deg

            # 5. Parse Object defect
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

            # 6. Parse terminal metrics
            if "Angle diff:" in line:
                m = re.search(r"Angle diff:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                if m:
                    angle_diff = float(m.group(1))
            if "Position weight" in line:
                m = re.search(r"Position weight\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                if m:
                    position_weight = float(m.group(1))

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

    target_angle_err = angle_diff if angle_diff is not None else 90.0
    pos_err = (position_weight / 10000.0) if position_weight is not None else 0.0

    defect_metric = (
        1.0 * total_quat_defect +
        2.0 * max_quat_defect +
        150.0 * total_obj_defect +
        300.0 * max_obj_defect +
        50.0 * total_ee_defect +
        0.5 * target_angle_err +
        100.0 * pos_err
    )

    # Record individual components for logging and analysis
    trial.set_user_attr("final_iter", int(final_iter))
    trial.set_user_attr("total_quat_defect_deg", float(total_quat_defect))
    trial.set_user_attr("max_quat_defect_deg", float(max_quat_defect))
    trial.set_user_attr("total_obj_defect", float(total_obj_defect))
    trial.set_user_attr("max_obj_defect", float(max_obj_defect))
    trial.set_user_attr("total_ee_defect", float(total_ee_defect))
    trial.set_user_attr("target_angle_err_deg", float(target_angle_err))
    trial.set_user_attr("pos_err", float(pos_err))

    print(f"\n[Trial #{trial.number} Result - Final Iter {final_iter}]")
    print(f"  Total Quat Defect (deg): {total_quat_defect:.3f} (max: {max_quat_defect:.3f})")
    print(f"  Total Obj Defect:        {total_obj_defect:.5f} (max: {max_obj_defect:.5f})")
    print(f"  Total EE Defect:         {total_ee_defect:.5f}")
    print(f"  Target Angle Diff (deg): {target_angle_err:.3f}")
    print(f"  Composite Defect Score:  {defect_metric:.4f}\n")

    return defect_metric

def format_component_scores(user_attrs):
    lines = ["Component Scores:"]
    lines.append(f"  Final Iteration:           {user_attrs.get('final_iter', 'N/A')}")
    t_q = user_attrs.get('total_quat_defect_deg')
    lines.append(f"  Total Quat Defect (deg):   {t_q:.3f}" if isinstance(t_q, (int, float)) else f"  Total Quat Defect (deg):   {t_q}")
    m_q = user_attrs.get('max_quat_defect_deg')
    lines.append(f"  Max Quat Defect (deg):     {m_q:.3f}" if isinstance(m_q, (int, float)) else f"  Max Quat Defect (deg):     {m_q}")
    t_o = user_attrs.get('total_obj_defect')
    lines.append(f"  Total Obj Defect (m):      {t_o:.5f}" if isinstance(t_o, (int, float)) else f"  Total Obj Defect (m):      {t_o}")
    m_o = user_attrs.get('max_obj_defect')
    lines.append(f"  Max Obj Defect (m):        {m_o:.5f}" if isinstance(m_o, (int, float)) else f"  Max Obj Defect (m):        {m_o}")
    t_e = user_attrs.get('total_ee_defect')
    lines.append(f"  Total EE Defect (m):       {t_e:.5f}" if isinstance(t_e, (int, float)) else f"  Total EE Defect (m):       {t_e}")
    t_a = user_attrs.get('target_angle_err_deg')
    lines.append(f"  Target Angle Error (deg):  {t_a:.3f}" if isinstance(t_a, (int, float)) else f"  Target Angle Error (deg):  {t_a}")
    p_e = user_attrs.get('pos_err')
    lines.append(f"  Position Error:            {p_e:.5f}" if isinstance(p_e, (int, float)) else f"  Position Error:            {p_e}")
    return "\n".join(lines)
    
def log_best_callback(study, trial):
    """
    Runs automatically after every trial.
    Logs the absolute best trial configuration AND appends any successful
    trials that achieved a metric score under 50.
    """
    if trial.value is None:
        return

    os.makedirs("examples/resources/multifinger_hand/optuna_point_hand_pivot_parallel", exist_ok=True)
    attrs = trial.user_attrs

    # 1. LOG EVERY TRIAL WITH METRIC < 50
    if trial.value < 50:
        print(f"--> Good trial found (Metric: {trial.value:.4f} < 50). Logging to historic file...")
        with open("examples/resources/multifinger_hand/optuna_point_hand_pivot_parallel/sub_50_trials_pivot_parallel.txt", "a") as f:
            f.write(f"Trial #{trial.number} | Metric Score: {trial.value:.4f}\n")
            f.write(format_component_scores(attrs) + "\n")
            f.write("Parameters:\n")
            for key, value in trial.params.items():
                f.write(f"  {key}: {value}\n")
            f.write("-" * 50 + "\n")
    
    # 2. TRACK THE ABSOLUTE BEST TRAJECTORY CONFIG
    if study.best_trial.number == trial.number:
        print(f"--> New absolute best metric found: {trial.value:.4f}. Saving to file...")
        with open("examples/resources/multifinger_hand/optuna_point_hand_pivot_parallel/best_params_pivot_new_anchor_updates.txt", "w") as f:
            f.write("=========================================\n")
            f.write("       BEST HYPERPARAMETERS SO FAR       \n")
            f.write("=========================================\n")
            f.write(f"Best Trial Number: {trial.number}\n")
            f.write(f"Best Metric Value: {trial.value:.4f}\n\n")
            f.write(format_component_scores(attrs) + "\n\n")
            f.write("Parameters:\n")
            for key, value in trial.params.items():
                f.write(f"  {key}: {value}\n")
            f.write("=========================================\n")

# sed -i 's/\t/  /g' examples/resources/multifinger_hand/ms_c3_tracking_options_point_hand_180.yaml
# sed -i 's/\t/  /g' examples/resources/multifinger_hand/ms_ic3_options_point_hand_180.yaml
# python3 examples/resources/multifinger_hand/optuna_point_hand_pivot_pruning_parallel.py
if __name__ == "__main__":

    print(platform.release().lower())
    if "microsoft" in platform.release().lower():
        STORAGE_URL = "sqlite:////home/ttesc255/optuna_data/optuna_results_pivot_new_anchor_updates.db"
    else:
        STORAGE_URL = "sqlite:///examples/resources/multifinger_hand/optuna_point_hand_pivot_parallel/optuna_results_pivot_new_anchor_updates.db"
        
    sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True)
    storage = optuna.storages.RDBStorage(
        url=STORAGE_URL,  
        heartbeat_interval=60            
    )
    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study = optuna.create_study(
        study_name="MSiC3_point_hand_pivot_new_anchor_updates",
        storage=storage,
        load_if_exists=True,  
        sampler=sampler,
        direction="minimize")
    study.optimize(objective, n_trials=5000, callbacks=[log_best_callback])

    print("\n--- Optimization Complete ---")
    print(f"Best Trial Value: {study.best_value}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
