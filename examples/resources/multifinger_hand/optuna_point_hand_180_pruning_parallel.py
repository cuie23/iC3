
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

def objective(trial):
    # C3 parameters
    w_G = trial.suggest_int("w_G", 1, 100)
    # g_x_fingers = trial.suggest_int("g_x_fingers", 2, 100, step=2)
    g_x_fingers = 50
    # g_x_cube = trial.suggest_int("g_x_cube", 2, 100, step=2)
    g_x_cube = 50
    # g_u = trial.suggest_int("g_u", 2, 100, step=2)
    g_u = 50

    g_lambda = trial.suggest_int("g_lambda", 2, 100, step=2)
    g_eta = trial.suggest_int("g_eta", 2, 100, step=2)

    # g_gamma = trial.suggest_int("g_gamma", 2, 100, step=2)
    # g_lambda_n = trial.suggest_int("g_lambda_n", 2, 100, step=2)
    # g_lambda_t = trial.suggest_int("g_lambda_t", 2, 100, step=2)
    # g_eta_slack = trial.suggest_int("g_eta_slack", 2, 100, step=2)
    # g_eta_n = trial.suggest_int("g_eta_n", 2, 100, step=2)
    # g_eta_t = trial.suggest_int("g_eta_t", 2, 100, step=2)

    u_ratio_finger = trial.suggest_int("u_ratio_finger", -200, 199) # -100 = lambda/eta = 0.1, 100 
    u_ratio_cube = trial.suggest_int("u_ratio_cube", -200, 199) # -100 = lambda/eta = 0.1, 100 

    # u_gamma = trial.suggest_int("u_gamma", 2, 100, step=2)
    # u_lambda_n = trial.suggest_int("u_lambda_n", 2, 100, step=2)
    # u_lambda_t = trial.suggest_int("u_lambda_t", 2, 100, step=2)
    # u_eta_slack = trial.suggest_int("u_eta_slack", 2, 100, step=2)
    # u_eta_n = trial.suggest_int("u_eta_n", 2, 100, step=2)
    # u_eta_t = trial.suggest_int("u_eta_t", 2, 100, step=2)

    # lambda_threshold = trial.suggest_int("lambda_threshold", 0, 60, step=5)
    # eta_threshold = trial.suggest_int("eta_threshold", 0, 10)
    lambda_threshold = 0
    eta_threshold = 0

    # gamma_threshold = trial.suggest_int("gamma_threshold", 0, 10)
    # phi_threshold = trial.suggest_int("phi_threshold", 0, 5)
    # add_phi_buffer = trial.suggest_int("add_phi_buffer", 0, 1)

    admm_iter = trial.suggest_int("admm_iter", 3, 6)
    finger_position_weight = trial.suggest_int("finger_position_weight", 100, 5000, step=100)
    cube_position_weight = trial.suggest_int("cube_position_weight", 200, 10000, step=200)
    tracking_N = trial.suggest_int("tracking_N", 3, 6)
    quat_weight = trial.suggest_int("quat_weight", 200, 10000, step=200)

    # finger_config = trial.suggest_int("finger_config", 1, 3)
    # cube_model = trial.suggest_int("cube_model", 1, 3)
    cube_model = 1
    
    w_G_final = trial.suggest_int("w_G_final", 1, 100)

    x_change_weight = trial.suggest_int("x_change_weight", 1, 1001, log=True)
    u_change_weight = trial.suggest_int("u_change_weight", 1, 1001, log=True)

    finger_config = 1

    with open(CONTORLLER_PARAMS, "r") as f:
        c3_options = yaml.safe_load(f)
        
    n_contacts = 7

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

    c3_options["c3_options"]["u_gamma"] = []
    c3_options["c3_options"]["u_lambda_n"] = []
    c3_options["c3_options"]["u_lambda_t"] = []
    c3_options["c3_options"]["u_eta_slack"] = []
    c3_options["c3_options"]["u_eta_n"] = []
    c3_options["c3_options"]["u_eta_t"] = []

    # multiplying by 50 is for dividing by dt=0.02 to get in signed distance units
    c3_options["c3_options"]["lambda_threshold"] = [(lambda_threshold / 100.0)] * (4 * 3) + [0] * (4 * (n_contacts - 3))
    c3_options["c3_options"]["eta_threshold"] = [(eta_threshold / 10.0)] * (4 * 3) + [0] * (4 * (n_contacts - 3))

    # c3_options["c3_options"]["gamma_threshold"] = [gamma_threshold / 100.0] * (n_contacts)
    # c3_options["c3_options"]["phi_threshold"] = [phi_threshold / 100.0] * (n_contacts)
    # c3_options["c3_options"]["add_phi_buffer"] = True if add_phi_buffer == 1 else False

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
                                0, 0, 0, 1, # cube orientation
                                0, 0, 0.052,  # cube position
                                0, 0, 0,     # finger 1 velo
                                0, 0, 0,     # finger 2 velo
                                0, 0, 0,     # finger 3 velo
                                0, 0, 0,     # cube ang velo
                                0, 0, 0]   	# cube velo
    elif (finger_config == 2):
        c3_options["x_init"] = [0.0, 0.07, 0.05,  # finger 1 
                                0.06, -0.06, 0.05,   # finger 2
                                -0.06, -0.06, 0.05,   # finger 3
                                1, 0, 0, 0, # cube orientation
                                0, 0, 0.052,  # cube position
                                0, 0, 0,     # finger 1 velo
                                0, 0, 0,     # finger 2 velo
                                0, 0, 0,     # finger 3 velo
                                0, 0, 0,     # cube ang velo
                                0, 0, 0]   	# cube velo

        c3_options["x_des"] = [0.0, 0.07, 0.05,  # finger 1 
                                0.06, -0.06, 0.05,   # finger 2
                                -0.06, -0.06, 0.05,   # finger 3
                                0, 0, 0, 1, # cube orientation
                                0, 0, 0.052,  # cube position
                                0, 0, 0,     # finger 1 velo
                                0, 0, 0,     # finger 2 velo
                                0, 0, 0,     # finger 3 velo
                                0, 0, 0,     # cube ang velo
                                0, 0, 0]   	# cube velo
    elif (finger_config == 3):
        c3_options["x_init"] = [0.0, 0.07, 0.05,  # finger 1 
                                0.05, -0.07, 0.05,   # finger 2
                                -0.05, -0.07, 0.05,   # finger 3
                                1, 0, 0, 0, # cube orientation
                                0, 0, 0.052,  # cube position
                                0, 0, 0,     # finger 1 velo
                                0, 0, 0,     # finger 2 velo
                                0, 0, 0,     # finger 3 velo
                                0, 0, 0,     # cube ang velo
                                0, 0, 0]   	# cube velo

        c3_options["x_des"] = [0.0, 0.07, 0.05,  # finger 1 
                                0.05, -0.07, 0.05,   # finger 2
                                -0.05, -0.07, 0.05,   # finger 3
                                0, 0, 0, 1, # cube orientation
                                0, 0, 0.052,  # cube position
                                0, 0, 0,     # finger 1 velo
                                0, 0, 0,     # finger 2 velo
                                0, 0, 0,     # finger 3 velo
                                0, 0, 0,     # cube ang velo
                                0, 0, 0]   	# cube velo
        
        
    for i in range(9): 
        c3_options["c3_options"]["q_vector"][i] = finger_position_weight

    c3_options["Q_quaternion_weight"] = quat_weight

    c3_options["c3_options"]["q_vector"][13] = cube_position_weight
    c3_options["c3_options"]["q_vector"][14] = cube_position_weight
    c3_options["c3_options"]["q_vector"][15] = 1000

    c3_options["c3_options"]["scale_lcs"] = True
    c3_options["c3_options"]["w_Q"] = 5
    c3_options["c3_options"]["w_R"] = 50
    c3_options["c3_options"]["w_U"] = 1

    with open(CONTORLLER_PARAMS, "w") as f:
        yaml.dump(c3_options, f, default_flow_style=True)

# ======================================================================

    # iC3 parameters
    num_segments = trial.suggest_categorical("num_segments", [5, 10, 12, 15, 20, 30, 40, 60])
    # num_warmup_iters = trial.suggest_int("num_warmup_iters", 0, 1)
    num_warmup_iters = 0
    # warm_start_alpha = trial.suggest_int("warm_start_alpha", 0, 100)

    num_iters = trial.suggest_int("num_iters", 2, 12)
    # alpha_ee = trial.suggest_int("alpha_ee", 0, 100)
    # alpha_object = trial.suggest_int("alpha_object", 0, 100)
    # alpha_ee_step = trial.suggest_int("alpha_ee_step", 0, 100)
    # alpha_object_step = trial.suggest_int("alpha_object_step", 0, 100)

    rho = trial.suggest_int("rho", 1, 1000)

    alpha_ee = 0
    alpha_object = 0
    alpha_ee_step = 0
    alpha_object_step = 0

    # value_function_scaling = trial.suggest_int("value_function_scaling", 0, 100)
    # use_value_function = trial.suggest_categorical("use_value_function", [True, False])
    use_value_function = True
    vf_trust_region_weight = trial.suggest_int("vf_trust_region_weight", 0, 100)

    # accel_cost = trial.suggest_int("accel_cost", 0, 50, step=10)
    accel_cost = 10

    # use_pd = trial.suggest_categorical("use_pd", [True, False])
    use_pd = False
    # use_rollout_lambdas = trial.suggest_categorical("use_rollout_lambdas", [True, False])
    use_rollout_lambdas = True

    use_lambdas_for_lcs = trial.suggest_categorical("use_lambdas_for_lcs", [True, False])

    # use_pd = trial.suggest_categorical("use_pd", [True, False])
    use_pd = False

    traj_N = trial.suggest_categorical("traj_N", [600, 660, 720])

    with open(MSiC3_PARAMS, "r") as f:
        ic3_options = yaml.safe_load(f)
        
    ic3_options["N"] = traj_N
    ic3_options["num_segments"] = num_segments

    ic3_options["num_warmup_iters"] = num_warmup_iters

    warm_start_alpha_real = 0 / 100.0
    ic3_options["warm_start_alpha"] = warm_start_alpha_real

    ic3_options["num_iters"] = num_iters

    alpha_ee_real = alpha_ee / 100.0
    ic3_options["alpha_ee"] = alpha_ee_real
    # ic3_options["alpha_ee_step"] = (1 - alpha_ee_real) / (num_iters - 1)
    ic3_options["alpha_ee_step"] = alpha_ee_step

    alpha_object_real = alpha_object / 100.0
    ic3_options["alpha_object"] = alpha_object_real
    # ic3_options["alpha_object_step"] = (1 - alpha_object_real) / (num_iters - 1)
    ic3_options["alpha_object_step"] = alpha_object_step

    ic3_options["acceleration_cost_weight"] = accel_cost

    value_function_scaling = 100 if use_value_function else 0
    ic3_options["value_function_scaling"] = value_function_scaling / 100.0
    ic3_options["vf_trust_region_weight"] = vf_trust_region_weight

    kp = 200 if use_pd else 0
    kd = 20 if use_pd else 0

    ic3_options["rollout_Kp"] = [kp] * 9
    ic3_options["rollout_Kd"] = [kd] * 9
    ic3_options["rollout_dt_scaling"] = 10
    
    ic3_options["print_costs"] = False

    ic3_options["num_threads"] = 32

    ic3_options["use_drake_sim"] = True
    ic3_options["drake_sim_dt"] = 0.0001

    ic3_options["use_rollout_lambdas"] = use_rollout_lambdas
    ic3_options["use_lambdas_for_lcs"] = use_lambdas_for_lcs

    ic3_options["anchor_rho"] = rho / 10.0

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
        "--experiment_type=MSiC3_point_hand_180_optuna_parallel",
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
            elif "x_anchor cube" in line:
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
                if current_iteration_cost < 0.35 and current_iteration != num_iters: # ~20 degrees
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
    trials that achieved a metric score under 15.
    """
    if trial.value is None:
        return

    # 1. LOG EVERY TRIAL WITH METRIC < 15
    # Ensure the trial wasn't pruned, has a return value, and meets your condition
    if trial.value < 15:
        print(f"--> Good trial found (Metric: {trial.value} < 15). Logging to historic file...")
        
        # Open in "a" (append) mode so you accumulate all sub-30 trials in one place
        with open("examples/resources/multifinger_hand/optuna_point_hand_180_parallel/sub_15_trials_180_sensitivty_anchor_update.txt", "a") as f:
            f.write(f"Trial #{trial.number} | Metric Score: {trial.value}\n")
            f.write("Parameters:\n")
            for key, value in trial.params.items():
                f.write(f"  {key}: {value}\n")
            f.write("-" * 40 + "\n")
    
    # 2. TRACK THE ABSOLUTE BEST TRAJECTORY CONFIG (Overwrites with absolute best)
    if study.best_trial.number == trial.number:
        print(f"--> New absolute best metric found: {trial.value}. Saving to file...")
        
        with open("examples/resources/multifinger_hand/optuna_point_hand_180_parallel/best_params_180_sensitivty_anchor_update.txt", "w") as f:
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
# python3 examples/resources/multifinger_hand/optuna_point_hand_180_pruning_parallel.py
if __name__ == "__main__":

    print(platform.release().lower())
    if "microsoft" in platform.release().lower():
        STORAGE_URL = "sqlite:////home/ttesc255/optuna_data/optuna_results_180_sensitivty_anchor_update.db"
    else:
      STORAGE_URL = "sqlite:///examples/resources/multifinger_hand/optuna_point_hand_180_parallel/optuna_results_180_sensitivty_anchor_update.db"
        
    # module = optunahub.load_module(package="samplers/catcmawm")
    # sampler = module.CatCmawmSampler()
    sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True)
    storage = optuna.storages.RDBStorage(
        url=STORAGE_URL,  
        heartbeat_interval=60            
    )

    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study = optuna.create_study(
        study_name="MSiC3_point_hand_180_sensitivty_anchor_update",
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