import yaml
import os
import re
import subprocess
import optuna
import sys

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
    g_x_fingers = trial.suggest_int("g_x_fingers", 2, 100, step=2)
    g_x_cube = trial.suggest_int("g_x_cube", 2, 100, step=2)
    g_u = trial.suggest_int("g_u", 2, 100, step=2)
    g_lambda = trial.suggest_int("g_lambda", 2, 100, step=2)
    g_eta = trial.suggest_int("g_eta", 2, 100, step=2)
    u_x_fingers = trial.suggest_int("u_x_fingers", 2, 100, step=2)
    u_x_cube = trial.suggest_int("u_x_cube", 2, 100, step=2)
    u_u = trial.suggest_int("u_u", 2, 100, step=2)
    u_lambda = trial.suggest_int("u_lambda", 2, 100, step=2)
    u_eta = trial.suggest_int("u_eta", 2, 100, step=2)
    finger_config = trial.suggest_int("finger_config", 1, 3)

    admm_iter = trial.suggest_int("admm_iter", 3, 8)
    finger_position_weight = trial.suggest_int("finger_position_weight", 10000, 300000, step=10000)
    cube_position_weight = trial.suggest_int("cube_position_weight", 200000, 1000000, step=20000)
    tracking_N = trial.suggest_int("tracking_N", 4, 8)
    quat_weight = trial.suggest_int("quat_weight", 200, 2000, step=200)

    with open(CONTORLLER_PARAMS, "r") as f:
        c3_options = yaml.safe_load(f)
        
    n_contacts = 7

    c3_options["c3_options"]["w_G"] = w_G
    c3_options["c3_options"]["g_lambda"] = [g_lambda] * (4*n_contacts)
    c3_options["c3_options"]["g_eta_slack"] = []
    c3_options["c3_options"]["g_eta_n"] = []
    c3_options["c3_options"]["g_eta_t"] = []
    c3_options["c3_options"]["g_eta"] = [g_eta] * (4*n_contacts)

    c3_options["c3_options"]["u_lambda"] = [u_lambda] * (4*n_contacts)
    c3_options["c3_options"]["u_eta_slack"] = []
    c3_options["c3_options"]["u_eta_n"] = []
    c3_options["c3_options"]["u_eta_t"] = []
    c3_options["c3_options"]["u_eta"] = [u_eta] * (4*n_contacts)
    
    for i in range(9):
        c3_options["c3_options"]["g_x"][i] = g_x_fingers
        c3_options["c3_options"]["u_x"][i] = u_x_fingers

    c3_options["c3_options"]["g_u"] = [g_u] * 9
    c3_options["c3_options"]["u_u"] = [u_u] * 9

    for i in range(7):
        c3_options["c3_options"]["g_x"][9+i] = g_x_cube
        c3_options["c3_options"]["u_x"][9+i] = u_x_cube

    c3_options["c3_options"]["admm_iter"] = admm_iter

    c3_options["lcs_factory_options"]["mu"] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    c3_options["lcs_factory_options"]["N"] = tracking_N
    c3_options["lcs_factory_options"]["dt"] = 0.02

    if (finger_config == 1):
        c3_options["x_init"] = [0.0, 0.07, 0.05,  # finger 1 
                                0.07, -0.055, 0.05,   # finger 2
                                -0.07, -0.055, 0.05,   # finger 3
                                1, 0, 0, 0, # cube orientation
                                0, 0, 0.051,  # cube position
                                0, 0, 0,     # finger 1 velo
                                0, 0, 0,     # finger 2 velo
                                0, 0, 0,     # finger 3 velo
                                0, 0, 0,     # cube ang velo
                                0, 0, 0]   	# cube velo

        c3_options["x_des"] = [0.0, 0.07, 0.05,  # finger 1 
                                0.07, -0.055, 0.05,   # finger 2
                                -0.07, -0.055, 0.05,   # finger 3
                                0, 0, 0, 1, # cube orientation
                                0, 0, 0.051,  # cube position
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
                                0, 0, 0.051,  # cube position
                                0, 0, 0,     # finger 1 velo
                                0, 0, 0,     # finger 2 velo
                                0, 0, 0,     # finger 3 velo
                                0, 0, 0,     # cube ang velo
                                0, 0, 0]   	# cube velo

        c3_options["x_des"] = [0.0, 0.07, 0.05,  # finger 1 
                                0.06, -0.06, 0.05,   # finger 2
                                -0.06, -0.06, 0.05,   # finger 3
                                0, 0, 0, 1, # cube orientation
                                0, 0, 0.051,  # cube position
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
                                0, 0, 0.051,  # cube position
                                0, 0, 0,     # finger 1 velo
                                0, 0, 0,     # finger 2 velo
                                0, 0, 0,     # finger 3 velo
                                0, 0, 0,     # cube ang velo
                                0, 0, 0]   	# cube velo

        c3_options["x_des"] = [0.0, 0.07, 0.05,  # finger 1 
                                0.05, -0.07, 0.05,   # finger 2
                                -0.05, -0.07, 0.05,   # finger 3
                                0, 0, 0, 1, # cube orientation
                                0, 0, 0.051,  # cube position
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

    with open(CONTORLLER_PARAMS, "w") as f:
        yaml.dump(c3_options, f, default_flow_style=True)

# ======================================================================

    # iC3 parameters
    num_segments = trial.suggest_categorical("num_segments", [5, 10, 15, 20, 25, 30, 40, 50])
    num_warmup_iters = trial.suggest_int("num_warmup_iters", 0, 2)
    warm_start_alpha = trial.suggest_float("warm_start_alpha", 0, 1, step=0.01)

    num_iters = trial.suggest_int("num_iters", 2, 3)
    alpha_ee = trial.suggest_float("alpha_ee", 0, 1, step=0.01)
    alpha_object = trial.suggest_float("alpha_object", 0, 1, step=0.01)

    accel_cost = trial.suggest_int("accel_cost", 0, 50, step=10)

    with open(MSiC3_PARAMS, "r") as f:
        ic3_options = yaml.safe_load(f)
        
    ic3_options["N"] = 600
    ic3_options["num_segments"] = num_segments

    ic3_options["num_warmup_iters"] = num_warmup_iters
    ic3_options["warm_start_alpha"] = warm_start_alpha

    ic3_options["num_iters"] = num_iters

    ic3_options["alpha_ee"] = alpha_ee
    ic3_options["alpha_ee_step"] = (1 - alpha_ee) / (num_iters - 1)

    ic3_options["alpha_object"] = alpha_object
    ic3_options["alpha_object_step"] = (1 - alpha_object) / (num_iters - 1)

    ic3_options["acceleration_cost_weight"] = accel_cost

    ic3_options["rollout_Kp"] = [0] * 9
    ic3_options["rollout_Kd"] = [0] * 9
    ic3_options["rollout_dt_scaling"] = 10
    
    ic3_options["print_costs"] = False

    with open(MSiC3_PARAMS, "w") as f:
        yaml.dump(ic3_options, f, default_flow_style=True)

    # Construct and execute the bazel command
    cmd = [
        "./bazel-bin/examples/lcs_factory_system_example", 
        f"--optuna_instance={worker_id}", 
        "--experiment_type=MSiC3_point_hand_180_optuna"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
            print(f"Trial failed with exit code {result.returncode}")
            print(f"Error log: {result.stderr}")
            raise optuna.TrialPruned()

    # Parse metric from std out
    print(result.stdout)
    match = re.search(r"FINAL_METRIC:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", result.stdout)
    
    solver_error_primal = re.search(r"Primal Res:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", result.stdout)
    solver_error_dual = re.search(r"Dual Res:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", result.stdout)

    if ("LCP failed: returning x_init" in result.stdout):
        raise optuna.TrialPruned("LCP solver failed")
    
    if (solver_error_primal is not None):
        if (float(solver_error_primal.group(1)) > 0.01):
            raise optuna.TrialPruned(f"Large Primal Residual ({float(solver_error_primal.group(1))})")
        if (float(solver_error_dual.group(1)) > 0.01):
            raise optuna.TrialPruned(f"Large Dual Residual ({float(solver_error_dual.group(1))})")

    if match:
        score = float(match.group(1))
        return score
    else:
        raise optuna.TrialPruned("Could not find metric in output.")
    
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
        with open("examples/resources/multifinger_hand/optuna_point_hand_180/sub_30_trials_with_finger_configs.txt", "a") as f:
            f.write(f"Trial #{trial.number} | Metric Score: {trial.value}\n")
            f.write("Parameters:\n")
            for key, value in trial.params.items():
                f.write(f"  {key}: {value}\n")
            f.write("-" * 40 + "\n")
    
    # 2. TRACK THE ABSOLUTE BEST TRAJECTORY CONFIG (Overwrites with absolute best)
    if study.best_trial.number == trial.number:
        print(f"--> New absolute best metric found: {trial.value}. Saving to file...")
        
        with open("examples/resources/multifinger_hand/optuna_point_hand_180/best_params_with_finger_configs.txt", "w") as f:
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
# python3 examples/resources/multifinger_hand/optuna_point_hand_180.py
if __name__ == "__main__":

    STORAGE_URL = "sqlite:///examples/resources/multifinger_hand/optuna_point_hand_180/optuna_results_with_finger_configs.db"

    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study = optuna.create_study(
        study_name="MSiC3_point_hand_180_with_finger_configs",
        storage=STORAGE_URL,
        load_if_exists=True,  
        direction="minimize")
    study.optimize(objective, n_trials=400, callbacks=[log_best_callback])

    print("\n--- Optimization Complete ---")
    print(f"Best Trial Value: {study.best_value}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")