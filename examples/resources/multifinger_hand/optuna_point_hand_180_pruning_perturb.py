
from py import process
import yaml
import re
import subprocess
import optuna
import optunahub
import sys
import math
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
CONTORLLER_PARAMS = f"examples/resources/multifinger_hand/optuna_point_hand_180/optuna_yamls/optuna_ms_c3_tracking_options_point_hand_180_{worker_id}.yaml"
MSiC3_PARAMS = f"examples/resources/multifinger_hand/optuna_point_hand_180/optuna_yamls/optuna_ms_ic3_options_point_hand_180_{worker_id}.yaml"
def get_yaw_quat(yaw_degrees):
    """Converts a yaw angle in degrees to a quaternion [w, x, y, z]."""
    half_yaw = math.radians(yaw_degrees) / 2.0
    return [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]

def objective(trial):
    # ======================================================================
    # 1. Sample all parameters ONCE for this trial
    # ======================================================================
    w_G = trial.suggest_int("w_G", 1, 100)
    g_x_fingers, g_x_cube, g_u = 50, 50, 50
    
    g_lambda = trial.suggest_int("g_lambda", 2, 100, step=2)
    g_eta = trial.suggest_int("g_eta", 2, 100, step=2)

    u_ratio_finger = trial.suggest_int("u_ratio_finger", -200, 199) 
    u_ratio_cube = trial.suggest_int("u_ratio_cube", -200, 199) 

    lambda_threshold, eta_threshold = 0, 0

    admm_iter = trial.suggest_int("admm_iter", 3, 6)
    finger_position_weight = trial.suggest_int("finger_position_weight", 100, 3000, step=100)
    cube_position_weight = trial.suggest_int("cube_position_weight", 2000, 10000, step=200)
    tracking_N = trial.suggest_int("tracking_N", 3, 6)
    quat_weight = trial.suggest_int("quat_weight", 500, 10000, step=500)

    cube_model = 1
    w_G_final = trial.suggest_int("w_G_final", 1, 100)

    x_change_weight = trial.suggest_int("x_change_weight", 1, 1001, log=True)
    u_change_weight = trial.suggest_int("u_change_weight", 1, 1001, log=True)

    finger_config = 1

    num_segments = trial.suggest_categorical("num_segments", [5, 10, 12, 15, 20, 30, 40, 60])
    num_warmup_iters = 0
    # warm_start_alpha = trial.suggest_int("warm_start_alpha", 0, 100)
    warm_start_alph = 0
    
    num_iters = trial.suggest_categorical("num_iters", [2, 3, 5, 6])
    alpha_ee = trial.suggest_int("alpha_ee", 0, 100)
    alpha_object = trial.suggest_int("alpha_object", 0, 100)
    
    value_function_scaling = 100
    vf_trust_region_weight = trial.suggest_int("vf_trust_region_weight", 0, 100)
    accel_cost = 5
    use_pd = False
    use_rollout_lambdas = True
    traj_N = trial.suggest_categorical("traj_N", [600, 720, 840])


    # ======================================================================
    # 2. Write MSiC3 parameters (These stay constant across perturbations)
    # ======================================================================
    with open(MSiC3_PARAMS, "r") as f:
        ic3_options = yaml.safe_load(f)
        
    ic3_options["N"] = traj_N
    ic3_options["num_segments"] = num_segments
    ic3_options["num_warmup_iters"] = num_warmup_iters
    ic3_options["warm_start_alpha"] = warm_start_alpha / 100.0
    ic3_options["num_iters"] = num_iters

    alpha_ee_real = alpha_ee / 100.0
    ic3_options["alpha_ee"] = alpha_ee_real
    ic3_options["alpha_ee_step"] = (1 - alpha_ee_real) / (num_iters - 1)

    alpha_object_real = alpha_object / 100.0
    ic3_options["alpha_object"] = alpha_object_real
    ic3_options["alpha_object_step"] = (1 - alpha_object_real) / (num_iters - 1)

    ic3_options["acceleration_cost_weight"] = accel_cost
    ic3_options["value_function_scaling"] = value_function_scaling / 100.0
    ic3_options["vf_trust_region_weight"] = vf_trust_region_weight

    kp = 200 if use_pd else 0
    kd = 20 if use_pd else 0
    ic3_options["rollout_Kp"] = [kp] * 9
    ic3_options["rollout_Kd"] = [kd] * 9
    ic3_options["rollout_dt_scaling"] = 10
    
    ic3_options["print_costs"] = False
    ic3_options["use_drake_sim"] = True
    ic3_options["drake_sim_dt"] = 0.0001
    ic3_options["use_rollout_lambdas"] = use_rollout_lambdas
    ic3_options["num_threads"] = 32

    for key in ["p_vector", "w_P"]:
        if key in ic3_options:
            del ic3_options[key]

    with open(MSiC3_PARAMS, "w") as f:
        yaml.dump(ic3_options, f, default_flow_style=True)


    # ======================================================================
    # 3. Evaluate the parameter set across multiple initial conditions
    # ======================================================================
    n_contacts = 7
    scores = {}

    # Setup C3 parameters
    with open(CONTORLLER_PARAMS, "r") as f:
        c3_options = yaml.safe_load(f)
        
    c3_options["c3_options"]["penalize_input_change"] = True
    c3_options["c3_options"]["penalize_x_change"] = True
    c3_options["c3_options"]["input_change_weight"] = (u_change_weight-1) / 100.0
    c3_options["c3_options"]["x_change_weight"] = (x_change_weight-1) / 100.0

    c3_options["c3_options"]["w_G"] = w_G / 100.0
    c3_options["c3_options"]["g_lambda"] = [g_lambda] * (4*n_contacts)
    c3_options["c3_options"]["g_eta"] = [g_eta] * (4*n_contacts)
    c3_options["c3_options"]["g_gamma"] = [] 
    c3_options["c3_options"]["g_lambda_n"] = []
    c3_options["c3_options"]["g_lambda_t"] = []
    c3_options["c3_options"]["g_eta_slack"] = []
    c3_options["c3_options"]["g_eta_n"] = []
    c3_options["c3_options"]["g_eta_t"] = []
    c3_options["c3_options"]["w_G_final"] = w_G_final

    finger_ratio = abs(u_ratio_finger) 
    if u_ratio_finger < 0:
        u_lambda_finger, u_eta_finger = 1.0, finger_ratio
    else:
        u_lambda_finger, u_eta_finger = finger_ratio + 1, 1.0

    cube_ratio = abs(u_ratio_cube)
    if u_ratio_cube < 0:
        u_lambda_cube, u_eta_cube = 1.0, cube_ratio
    else:
        u_lambda_cube, u_eta_cube = cube_ratio + 1, 1.0

    c3_options["c3_options"]["u_lambda"] = [u_lambda_finger]*(4*3) + [u_lambda_cube]*(4*(n_contacts-3))
    c3_options["c3_options"]["u_eta"] = [u_eta_finger]*(4*3) + [u_eta_cube]*(4*(n_contacts-3))
    c3_options["c3_options"]["u_gamma"] = []
    c3_options["c3_options"]["u_lambda_n"] = []
    c3_options["c3_options"]["u_lambda_t"] = []
    c3_options["c3_options"]["u_eta_slack"] = []
    c3_options["c3_options"]["u_eta_n"] = []
    c3_options["c3_options"]["u_eta_t"] = []

    c3_options["c3_options"]["lambda_threshold"] = [(lambda_threshold / 100.0)]*(4*3) + [0]*(4*(n_contacts - 3))
    c3_options["c3_options"]["eta_threshold"] = [(eta_threshold / 10.0)]*(4*3) + [0]*(4*(n_contacts - 3))
    c3_options["c3_options"]["phi_threshold"] = []
    c3_options["c3_options"]["gamma_threshold"] = []
    c3_options["c3_options"]["epsilon"] = []
    c3_options["c3_options"]["add_phi_buffer"] = False

    for i in range(9):
        c3_options["c3_options"]["g_x"][i] = g_x_fingers
    c3_options["c3_options"]["g_u"] = [g_u] * 9
    c3_options["c3_options"]["u_x"] = [1] * 31
    c3_options["c3_options"]["u_u"] = [1] * 9
    for i in range(7):
        c3_options["c3_options"]["g_x"][9+i] = g_x_cube

    c3_options["c3_options"]["admm_iter"] = admm_iter

    c3_options["lcs_factory_options"]["mu"] = [0.33, 0.33, 0.33, 0.3, 0.3, 0.3, 0.3]
    c3_options["lcs_factory_options"]["N"] = tracking_N
    c3_options["lcs_factory_options"]["dt"] = 0.02
    c3_options["lcs_factory_options"]["contact_model"] = "anitescu"

    if finger_config == 1:
        x_init_base = [0.0, 0.07, 0.05, 0.07, -0.055, 0.05, -0.07, -0.055, 0.05, 
                       1, 0, 0, 0, 0, 0, 0.052, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        x_des_base  = [0.0, 0.07, 0.05, 0.07, -0.055, 0.05, -0.07, -0.055, 0.05, 
                       0, 0, 0, 1, 0, 0, 0.052, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif finger_config == 2:
        x_init_base = [0.0, 0.07, 0.05, 0.06, -0.06, 0.05, -0.06, -0.06, 0.05, 
                       1, 0, 0, 0, 0, 0, 0.052, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        x_des_base  = [0.0, 0.07, 0.05, 0.06, -0.06, 0.05, -0.06, -0.06, 0.05, 
                       0, 0, 0, 1, 0, 0, 0.052, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif finger_config == 3:
        x_init_base = [0.0, 0.07, 0.05, 0.05, -0.07, 0.05, -0.05, -0.07, 0.05, 
                       1, 0, 0, 0, 0, 0, 0.052, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        x_des_base  = [0.0, 0.07, 0.05, 0.05, -0.07, 0.05, -0.05, -0.07, 0.05, 
                       0, 0, 0, 1, 0, 0, 0.052, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
    c3_options["x_init"] = x_init_base
    c3_options["x_des"] = x_des_base
        
    for i in range(9): 
        c3_options["c3_options"]["q_vector"][i] = finger_position_weight

    c3_options["Q_quaternion_weight"] = quat_weight
    c3_options["c3_options"]["q_vector"][13] = cube_position_weight
    c3_options["c3_options"]["q_vector"][14] = cube_position_weight

    c3_options["c3_options"]["scale_lcs"] = True
    c3_options["c3_options"]["w_Q"] = 5
    c3_options["c3_options"]["w_R"] = 50
    c3_options["c3_options"]["w_U"] = 1

    with open(CONTORLLER_PARAMS, "w") as f:
        yaml.dump(c3_options, f, default_flow_style=True)

    # Execute Subprocess once
    cmd = [
        "./bazel-bin/examples/lcs_factory_system_example", 
        f"--optuna_instance={worker_id}", 
        "--experiment_type=MSiC3_point_hand_180_optuna_robust",
        f"--ee_config={finger_config}",
        f"--cube_model={cube_model}"
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    current_iteration = None
    current_iteration_cost = 0.0
    current_anchor_q = None
    full_output = []

    try:
        for line in iter(process.stdout.readline, ''):
            full_output.append(line)
            
            if "LCP failed: returning x_init" in line:
                print("LCP solver failed during trajectory optimization or tracking.")
                return 99991, 99991
            
            primal_match = re.search(r"Primal Res:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
            dual_match = re.search(r"Dual Res:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
            
            if primal_match and float(primal_match.group(1)) > 0.01:
                print(f"Large Primal Residual ({float(primal_match.group(1))})")
                return 99992, 99992

            if dual_match and float(dual_match.group(1)) > 0.01:
                print(f"Large Dual Residual ({float(dual_match.group(1))})")
                return 99993, 99993
            
            if line.startswith("iC3 iteration"):
                current_iteration = int(line.split()[-1])
                current_iteration_cost = 0.0 
                
            elif "x anchor cube" in line:
                raw_numbers = line.split("cube")[1].split()
                current_anchor_q = [float(val) for val in raw_numbers[0:4]]
                
            elif "x_hat[L] cube:" in line:
                raw_numbers = line.split("cube:")[1].split()
                hat_q = [float(val) for val in raw_numbers[0:4]]
                
                if current_anchor_q is not None:
                    angle_diff = get_quaternion_angle_diff(current_anchor_q, hat_q)
                    current_iteration_cost += angle_diff
                    current_anchor_q = None 
                    
            elif "Iteration runtime:" in line and current_iteration is not None:
                # C++ only runs this for the nominal case now
                if current_iteration_cost < 0.5 and current_iteration != num_iters: 
                    print(f"Pruned at iC3 iter {current_iteration} (Angle Error: {current_iteration_cost:.4f})")
                    return 9994, 9994
                
            # Extract metrics 0 through 4
            final_match = re.search(r"FINAL METRIC (\d+):\s*([0-9.]+)", line)
            if final_match:
                metric_idx = int(final_match.group(1))
                metric_val = float(final_match.group(2))
                scores[metric_idx] = metric_val

    except optuna.TrialPruned as e:
        process.terminate() 
        process.wait() 
        raise e 
    
    finally:
        process.stdout.close()
        process.stderr.close()
        print("".join(full_output))  

    process.wait()
    if process.returncode != 0:
        raise optuna.TrialPruned(f"Process crashed with exit code {process.returncode}")

    if len(scores) < 5:
        raise optuna.TrialPruned(f"Could not find all 5 FINAL METRIC outputs. Found {len(scores)}.")

    # ======================================================================
    # 4. Multi-Objective Return
    # ======================================================================
    nominal_score = scores[2] 
    worst_score = max(scores.values())
    sensitivity = worst_score - nominal_score 

    return nominal_score, sensitivity

def log_best_callback(study, trial):
    """
    Runs automatically after every trial.
    Logs trials that achieved a nominal metric score under 15, 
    and updates the file containing the current Pareto front.
    """
    if trial.values is None:
        return

    nominal_score, sensitivity = trial.values

    # 1. LOG EVERY TRIAL WITH NOMINAL METRIC < 15
    # Ensure the trial wasn't pruned and meets your baseline criteria
    if nominal_score < 15:
        print(f"--> Good trial found (Nominal: {nominal_score:.3f} < 15, Sens: {sensitivity:.3f}). Logging to historic file...")
        
        with open("examples/resources/multifinger_hand/optuna_point_hand_180/sub_15_trials_180_hybrid_mpc_final_robust.txt", "a") as f:
            f.write(f"Trial #{trial.number} | Nominal Metric: {nominal_score:.3f} | Sensitivity: {sensitivity:.3f}\n")
            f.write("Parameters:\n")
            for key, value in trial.params.items():
                f.write(f"  {key}: {value}\n")
            f.write("-" * 40 + "\n")
    
    # 2. TRACK THE PARETO FRONT (Overwrites with all non-dominated trials)
    # Check if the current trial made it onto the Pareto front
    best_trial_numbers = [t.number for t in study.best_trials]
    if trial.number in best_trial_numbers:
        print(f"--> New Pareto optimal trial found (Trial {trial.number}). Updating best params file...")
        
        with open("examples/resources/multifinger_hand/optuna_point_hand_180/best_params_180_hybrid_mpc_final_robust.txt", "w") as f:
            f.write("=========================================\n")
            f.write("      PARETO OPTIMAL HYPERPARAMETERS     \n")
            f.write("=========================================\n")
            for t in study.best_trials:
                f.write(f"Trial #{t.number}\n")
                f.write(f"  Nominal Metric: {t.values[0]:.3f}\n")
                f.write(f"  Sensitivity:    {t.values[1]:.3f}\n\n")
                f.write("  Parameters:\n")
                for key, value in t.params.items():
                    f.write(f"    {key}: {value}\n")
                f.write("-----------------------------------------\n")


# sed -i 's/\t/  /g' examples/resources/multifinger_hand/ms_c3_tracking_options_point_hand_180.yaml
# sed -i 's/\t/  /g' examples/resources/multifinger_hand/ms_ic3_options_point_hand_180.yaml
# python3 examples/resources/multifinger_hand/optuna_point_hand_180_pruning_perturb.py

if __name__ == "__main__":

    print(platform.release().lower())
    if "microsoft" in platform.release().lower():
        STORAGE_URL = "sqlite:////home/ttesc255/optuna_data/optuna_results_180_hybrid_mpc_final_robust.db"
    else:
        STORAGE_URL = "sqlite:///examples/resources/multifinger_hand/optuna_point_hand_180/optuna_results_180_hybrid_mpc_final_robust.db"
        
    # TPE Sampler inherently supports multi-objective optimization (MOTPE)
    sampler = optuna.samplers.TPESampler(multivariate=True)
    storage = optuna.storages.RDBStorage(
        url=STORAGE_URL,  
        heartbeat_interval=60            
    )

    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    
    # NOTE: study_name changed to avoid crashing against your old single-objective database
    study = optuna.create_study(
        study_name="MSiC3_point_hand_180_hybrid_mpc_final_robust",
        storage=storage,
        load_if_exists=True, 
        sampler=sampler, 
        directions=["minimize", "minimize"] # MULTI-OBJECTIVE
    )
    
    study.optimize(objective, n_trials=5000, callbacks=[log_best_callback])

    print("\n--- Optimization Complete ---")
    print(f"Found {len(study.best_trials)} trials on the Pareto front:")
    for i, t in enumerate(study.best_trials):
        print(f"\nPareto Trial #{t.number}")
        print(f"  Nominal Metric: {t.values[0]:.3f}")
        print(f"  Sensitivity: {t.values[1]:.3f}")
        print("  Hyperparameters:")
        for key, value in t.params.items():
            print(f"    {key}: {value}")