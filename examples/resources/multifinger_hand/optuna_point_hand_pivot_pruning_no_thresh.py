import yaml
import math
import re
import subprocess
import optuna
import sys
import platform

def get_quaternion_angle_diff(q1, q2):
    """Computes the angular difference (in radians) between two quaternions."""
    # 1. Compute the dot product
    dot_product = sum(a * b for a, b in zip(q1, q2))
    
    # 2. Clamp the dot product to [-1, 1] to avoid math domain errors from floating point drift
    dot_product = max(min(dot_product, 1.0), -1.0)
    
    # 3. Calculate the angle. We use abs() because q and -q represent the same rotation.
    angle_rads = 2 * math.acos(abs(dot_product))
    return angle_rads

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
    w_G = trial.suggest_int("w_G", 5, 5000, step=5)
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
    tracking_N = trial.suggest_int("tracking_N", 3, 8)
    finger_position_weight = trial.suggest_int("finger_position_weight", 1000, 10000, step=100)
    cube_position_weight = trial.suggest_int("cube_position_weight", 100, 10000, step=100)
    quat_weight = trial.suggest_int("quat_weight", 5000, 500000, step=5000)

    # finger_config = trial.suggest_categorical("finger_config", [1, 2, 3])
    finger_config = 2

    cube_model = trial.suggest_int("cube_model", 1, 10)

    # towards_2_fingers = trial.suggest_categorical("towards_2_fingers", [True, False])
    towards_2_fingers = True

    large_dt = trial.suggest_categorical("large_dt", [True, False])


    mu_finger_cube = trial.suggest_int("mu_finger_cube", 1, 100)

    # scale_lcs = trial.suggest_categorical("scale_lcs", [True, False])
    scale_lcs = True

    # x_change_weight = trial.suggest_int("x_change_weight", 1, 1001, log=True)
    # u_change_weight = trial.suggest_int("u_change_weight", 1, 1001, log=True)
    x_change_weight = 1
    u_change_weight = 1

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

    c3_options["c3_options"]["penalize_input_change"] = True
    c3_options["c3_options"]["penalize_x_change"] = True
    c3_options["c3_options"]["input_change_weight"] = (u_change_weight-1) / 100.0
    c3_options["c3_options"]["x_change_weight"] = (x_change_weight-1) / 100.0

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

    mu_fc_real = mu_finger_cube / 100.0
    c3_options["lcs_factory_options"]["mu"] = [mu_fc_real, mu_fc_real, mu_fc_real, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    c3_options["lcs_factory_options"]["N"] = tracking_N
    c3_options["lcs_factory_options"]["dt"] = 0.04 if large_dt else 0.02

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

    # makes initial guess go the other direction
    if not towards_2_fingers:
        c3_options["x_des"][10] = -1

    c3_options["c3_options"]["w_Q"] = 5
    c3_options["c3_options"]["w_R"] = 50
    c3_options["c3_options"]["w_U"] = 1
    c3_options["c3_options"]["scale_lcs"] = scale_lcs

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
    num_segments = trial.suggest_categorical("num_segments", [2, 5, 10, 15, 30, 60])

    num_iters = trial.suggest_categorical("num_iters", [2, 3, 5, 6])
    alpha_ee = trial.suggest_int("alpha_ee", 0, 100)
    alpha_object = trial.suggest_int("alpha_object", 0, 100)

    vf_trust_region_weight = trial.suggest_int("vf_trust_region_weight", 0, 100)
    # value_function_scaling = trial.suggest_int("value_function_scaling", 0, 100)
    value_function_scaling = 100

    accel_cost = 10

    # use_rollout_lambdas = trial.suggest_categorical("use_rollout_lambdas", [True, False])
    use_rollout_lambdas = True

    with open(MSiC3_PARAMS, "r") as f:
        ic3_options = yaml.safe_load(f)
        
    ic3_options["N"] = 180 if large_dt else 360
    ic3_options["num_segments"] = num_segments

    ic3_options["num_warmup_iters"] = num_warmup_iters

    warm_start_alpha_real = warm_start_alpha / 100.0
    ic3_options["warm_start_alpha"] = warm_start_alpha_real

    ic3_options["num_iters"] = num_iters

    alpha_ee_real = alpha_ee / 100.0
    ic3_options["alpha_ee"] = alpha_ee_real
    ic3_options["alpha_ee_step"] = (1 - alpha_ee_real) / (num_iters - 1)

    alpha_obj_real = alpha_object / 100.0
    ic3_options["alpha_object"] = alpha_obj_real
    ic3_options["alpha_object_step"] = (1 - alpha_obj_real) / (num_iters - 1)

    ic3_options["acceleration_cost_weight"] = accel_cost
    ic3_options["value_function_scaling"] = value_function_scaling / 100.0
    ic3_options["vf_trust_region_weight"] = vf_trust_region_weight

    ic3_options["rollout_Kp"] = [0] * 9
    ic3_options["rollout_Kd"] = [0] * 9
    ic3_options["rollout_dt_scaling"] = 10

    ic3_options["print_costs"] = False

    ic3_options["use_drake_sim"] = True
    ic3_options["drake_sim_dt"] = 0.0001

    ic3_options["use_rollout_lambdas"] = use_rollout_lambdas
    
    ic3_options["num_threads"] = 32

    if "p_vector" in ic3_options:
        del ic3_options["p_vector"]
    if "w_P" in ic3_options:
        del ic3_options["w_P"]

    with open(MSiC3_PARAMS, "w") as f:
        yaml.dump(ic3_options, f, default_flow_style=True)

    # Construct and execute the bazel command
    cmd = [
        "./bazel-bin/examples/lcs_factory_system_example", 
        f"--optuna_instance={worker_id}", 
        "--experiment_type=MSiC3_point_hand_optuna",
        f"--ee_config={finger_config}",
        f"--cube_model={cube_model}"
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)


    final_score = None
    
    # --- State Variables for the iC3 Iteration ---
    current_iteration = None
    current_iteration_cost = 0.0
    current_anchor_q = None

    full_output = []

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

            # 2. Start a new iC3 iteration
            if line.startswith("iC3 iteration"):
                current_iteration = int(line.split()[-1])
                current_iteration_cost = 0.0  # Reset the sum for the new iteration
                
            # 3. Extract the target (anchor) quaternion
            elif "x anchor cube" in line:
                raw_numbers = line.split("cube")[1].split()
                current_anchor_q = [float(val) for val in raw_numbers[0:4]]
                
            # 4. Extract the actual (hat) quaternion and add to the running cost
            elif "x_hat[L] cube:" in line:
                raw_numbers = line.split("cube:")[1].split()
                hat_q = [float(val) for val in raw_numbers[0:4]]
                
                if current_anchor_q is not None:
                    angle_diff = get_quaternion_angle_diff(current_anchor_q, hat_q)
                    current_iteration_cost += angle_diff
                    current_anchor_q = None # Reset to prevent double-counting on malformed logs
                    
            # 5. End of iteration: REPORT AND PRUNE
            elif "Iteration runtime:" in line and current_iteration is not None:
                # Prune if angle sum is too small
                if current_iteration_cost < 0.15 and current_iteration != num_iters: 
                    print()
                    raise optuna.TrialPruned(f"Pruned at iC3 iteration {current_iteration} (Summed Angle Error: {current_iteration_cost:.4f})")

            # 6. Extract Final Metric
            final_match = re.search(r"FINAL_METRIC:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", line)
            if final_match:
                final_score = float(final_match.group(1))

    except optuna.TrialPruned as e:
        # Kill the C++ subprocess immediately to save compute time
        process.terminate() 
        process.wait() 
        raise e 
    
    finally:
        process.stdout.close()
        process.stderr.close()
        print("".join(full_output))  # Print all captured output for debugging

    process.wait()
    if process.returncode != 0:
        print(f"Trial failed with exit code {process.returncode}")
        raise optuna.TrialPruned("Process crashed or returned non-zero exit code")


    if final_score is not None:
        return final_score
    else:
        raise optuna.TrialPruned("Could not find FINAL_METRIC in output.")
    
def log_best_callback(study, trial):
    """
    Runs automatically after every trial.
    Logs the absolute best trial configuration AND appends any successful
    trials that achieved a metric score under 30.
    """
    if trial.value is None:
        return

    # 1. LOG EVERY TRIAL WITH METRIC < 30
    # Ensure the trial wasn't pruned, has a return value, and meets your condition
    if trial.value < 30:
        print(f"--> Good trial found (Metric: {trial.value} < 30). Logging to historic file...")
        
        # Open in "a" (append) mode so you accumulate all sub-30 trials in one place
        with open("examples/resources/multifinger_hand/optuna_point_hand_pivot/sub_30_trials_pivot_tighter_z_input_bounds.txt", "a") as f:
            f.write(f"Trial #{trial.number} | Metric Score: {trial.value}\n")
            f.write("Parameters:\n")
            for key, value in trial.params.items():
                f.write(f"  {key}: {value}\n")
            f.write("-" * 40 + "\n")
    
    # 2. TRACK THE ABSOLUTE BEST TRAJECTORY CONFIG (Overwrites with absolute best)
    if study.best_trial.number == trial.number:
        print(f"--> New absolute best metric found: {trial.value}. Saving to file...")
        
        with open("examples/resources/multifinger_hand/optuna_point_hand_pivot/best_params_pivot_tighter_z_input_bounds.txt", "w") as f:
            f.write("=========================================\n")
            f.write("       BEST HYPERPARAMETERS SO FAR       \n")
            f.write("=========================================\n")
            f.write(f"Best Trial Number: {trial.number}\n")
            f.write(f"Best Metric Value: {trial.value}\n\n")
            f.write("Parameters:\n")
            for key, value in trial.params.items():
                f.write(f"  {key}: {value}\n")
            f.write("=========================================\n")

# sed -i 's/\t/  /g' examples/resources/multifinger_hand/ms_c3_tracking_options_point_hand_180.yaml
# sed -i 's/\t/  /g' examples/resources/multifinger_hand/ms_ic3_options_point_hand_180.yaml
# python3 examples/resources/multifinger_hand/optuna_point_hand_pivot_pruning_no_thresh.py
if __name__ == "__main__":

    print(platform.release().lower())
    if "microsoft" in platform.release().lower():
        STORAGE_URL = "sqlite:////home/ttesc255/optuna_data/optuna_results_pivot_tighter_z_input_bounds.db"
    else:
        STORAGE_URL = "sqlite:///examples/resources/multifinger_hand/optuna_point_hand_pivot/optuna_results_pivot_tighter_z_input_bounds.db"
        
    sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True)
    storage = optuna.storages.RDBStorage(
        url=STORAGE_URL,  
        heartbeat_interval=60            
    )
    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study = optuna.create_study(
        study_name="MSiC3_point_hand_pivot_tighter_z_input_bounds",
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