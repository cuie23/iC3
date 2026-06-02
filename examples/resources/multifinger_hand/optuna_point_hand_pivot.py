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
CONTORLLER_PARAMS = f"examples/resources/multifinger_hand/optuna_point_hand_pivot/optuna_yamls/optuna_ms_c3_tracking_options_pivot_{worker_id}.yaml"
MSiC3_PARAMS = f"examples/resources/multifinger_hand/optuna_point_hand_pivot/optuna_yamls/optuna_ms_ic3_options_pivot_{worker_id}.yaml"

def objective(trial):
    # C3 parameters
    w_G = trial.suggest_int("w_G", 1, 200)
    g_lambda = trial.suggest_int("g_lambda", 5, 100, step=5)
    g_eta = trial.suggest_int("g_eta", 2, 50, step=2)
    u_lambda = trial.suggest_int("u_lambda", 5, 100, step=5)
    u_eta = trial.suggest_int("u_eta", 2, 50, step=2)
    admm_iter = trial.suggest_int("admm_iter", 4, 10)
    finger_position_weight = trial.suggest_int("finger_position_weight", 1000, 100000, step=1000)
    cube_position_weight = trial.suggest_int("cube_position_weight", 1000, 100000, step=1000)
    finger_config = trial.suggest_int("finger_config", 1, 3)


    quat_weight = trial.suggest_int("quat_weight", 500, 5000, step=500)

    with open(CONTORLLER_PARAMS, "r") as f:
        c3_options = yaml.safe_load(f)
        
    n_contacts = 11

    c3_options["c3_options"]["w_G"] = w_G
    c3_options["c3_options"]["g_lambda"] = [g_lambda] * (4*n_contacts)
    c3_options["c3_options"]["g_eta_slack"] = [g_eta] * n_contacts
    c3_options["c3_options"]["g_eta_n"] = [g_eta] * n_contacts
    c3_options["c3_options"]["g_eta_t"] = [g_eta] * n_contacts
    c3_options["c3_options"]["g_eta"] = [g_eta] * (4*n_contacts)

    c3_options["c3_options"]["u_lambda"] = [u_lambda] * (4*n_contacts)
    c3_options["c3_options"]["u_eta_slack"] = [u_eta] * n_contacts
    c3_options["c3_options"]["u_eta_n"] = [u_eta] * n_contacts
    c3_options["c3_options"]["u_eta_t"] = [u_eta] * n_contacts
    c3_options["c3_options"]["u_eta"] = [u_eta] * (4*n_contacts)
    
    c3_options["c3_options"]["admm_iter"] = admm_iter

    c3_options["Q_quaternion_weight"] = quat_weight

    
    for i in range(9): 
        c3_options["c3_options"]["q_vector"][i] = finger_position_weight

    c3_options["c3_options"]["q_vector"][13] = cube_position_weight
    c3_options["c3_options"]["q_vector"][14] = cube_position_weight
    c3_options["c3_options"]["q_vector"][15] = cube_position_weight


    c3_options["lcs_factory_options"]["num_contacts"] = 11
    c3_options["lcs_factory_options"]["mu"] = [0.6] * 11

    if (finger_config == 1):
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
                                0, 1, 0, 0, # cube orientation
                                0, 0, 0.051,  # cube position
                                0, 0, 0,     # finger 1 velo
                                0, 0, 0,     # finger 2 velo
                                0, 0, 0,     # finger 3 velo
                                0, 0, 0,     # cube ang velo
                                0, 0, 0]   	# cube velo
    elif (finger_config == 2):
        c3_options["x_init"] = [0.0, 0.07, 0.05,  # finger 1 
                                0.06, -0.07, 0.05,   # finger 2
                                -0.06, -0.07, 0.05,   # finger 3
                                1, 0, 0, 0, # cube orientation
                                0, 0, 0.051,  # cube position
                                0, 0, 0,     # finger 1 velo
                                0, 0, 0,     # finger 2 velo
                                0, 0, 0,     # finger 3 velo
                                0, 0, 0,     # cube ang velo
                                0, 0, 0]   	# cube velo
        
        c3_options["x_des"] = [0.0, 0.07, 0.05,  # finger 1 
                                0.06, -0.07, 0.05,   # finger 2
                                -0.06, -0.07, 0.05,   # finger 3
                                0, 1, 0, 0, # cube orientation
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
                                0, 1, 0, 0, # cube orientation
                                0, 0, 0.051,  # cube position
                                0, 0, 0,     # finger 1 velo
                                0, 0, 0,     # finger 2 velo
                                0, 0, 0,     # finger 3 velo
                                0, 0, 0,     # cube ang velo
                                0, 0, 0]   	# cube velo
        

    with open(CONTORLLER_PARAMS, "w") as f:
        yaml.dump(c3_options, f, default_flow_style=True)

# ======================================================================

    # iC3 parameters
    num_warmup_iters = trial.suggest_int("num_warmup_iters", 0, 3)
    warm_start_alpha = trial.suggest_float("warm_start_alpha", 0, 1, step=0.01)
    num_segments = trial.suggest_categorical("num_segments", [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30])

    num_iters = trial.suggest_int("num_iters", 2, 5)
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

    with open(MSiC3_PARAMS, "w") as f:
        yaml.dump(ic3_options, f, default_flow_style=True)

    # Construct and execute the bazel command
    cmd = [
        "bazel", 
        "run", 
        "//examples:lcs_factory_system_example", 
        "--",
        f"--optuna_instance={worker_id}", 
        "--experiment_type=MSiC3_point_hand_optuna"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
            print(f"Trial failed with exit code {result.returncode}")
            print(f"Error log: {result.stderr}")
            raise optuna.TrialPruned()

    # Parse metric from std out
    print(result.stdout)
    match = re.search(r"FINAL_METRIC:\s*([0-9.]+)", result.stdout)
    
    if match:
        score = float(match.group(1))
        return score
    else:
        raise optuna.TrialPruned("Could not find metric in output.")
def log_best_callback(study, trial):
    """
    Runs automatically after every trial.
    Logs the absolute best trial configuration AND appends any successful
    trials that achieved a metric score under 40.
    """
    # 1. LOG EVERY TRIAL WITH METRIC < 20
    # Ensure the trial wasn't pruned, has a return value, and meets your condition
    if trial.value is not None and trial.value < 40:
        print(f"--> Good trial found (Metric: {trial.value} < 40). Logging to historic file...")
        
        # Open in "a" (append) mode so you accumulate all sub-40 trials in one place
        with open("examples/resources/multifinger_hand/optuna_point_hand_pivot/sub_40_trials.txt", "a") as f:
            f.write(f"Trial #{trial.number} | Metric Score: {trial.value}\n")
            f.write("Parameters:\n")
            for key, value in trial.params.items():
                f.write(f"  {key}: {value}\n")
            f.write("-" * 40 + "\n")

    # 2. TRACK THE ABSOLUTE BEST TRAJECTORY CONFIG (Overwrites with absolute best)
    if study.best_trial.number == trial.number:
        print(f"--> New absolute best metric found: {study.best_value}. Saving to file...")
        
        with open("examples/resources/multifinger_hand/optuna_point_hand_pivot/best_params_pivot.txt", "w") as f:
            f.write("=========================================\n")
            f.write("       BEST HYPERPARAMETERS SO FAR       \n")
            f.write("=========================================\n")
            f.write(f"Best Trial Number: {study.best_trial.number}\n")
            f.write(f"Best Metric Value: {study.best_value}\n\n")
            f.write("Parameters:\n")
            for key, value in study.best_params.items():
                f.write(f"  {key}: {value}\n")
            f.write("=========================================\n")

# sed -i 's/\t/  /g' examples/resources/multifinger_hand/ms_c3_tracking_options_point_hand_180.yaml
# sed -i 's/\t/  /g' examples/resources/multifinger_hand/ms_ic3_options_point_hand_180.yaml
# python3 examples/resources/multifinger_hand/optuna_point_hand_pivot.py
if __name__ == "__main__":

    STORAGE_URL = "sqlite:///examples/resources/multifinger_hand/optuna_point_hand_pivot/optuna_results_pivot.db"

    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study = optuna.create_study(
        study_name="MSiC3_point_hand_pivot",
        storage=STORAGE_URL,
        load_if_exists=True,  
        direction="minimize")
    study.optimize(objective, n_trials=400, callbacks=[log_best_callback])

    print("\n--- Optimization Complete ---")
    print(f"Best Trial Value: {study.best_value}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")