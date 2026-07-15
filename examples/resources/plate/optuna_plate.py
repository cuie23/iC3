import yaml
import os
import re
import subprocess
import optuna
import optunahub

# Define paths to your parameter files
CONTORLLER_PARAMS = "examples/resources/plate/optuna_ms_c3_tracking_options.yaml"
MSiC3_PARAMS = "examples/resources/plate/optuna_ms_ic3_options.yaml"


def objective(trial):
    # C3 parameters
    admm_iter = trial.suggest_int("admm_iter", 3, 6)
    w_G = trial.suggest_int("w_G", 1, 200)
    plate_rot_cost = trial.suggest_int("plate_rot_cost", 100, 5000, step=100)
    tracking_N = trial.suggest_int("tracking_N", 3, 6)
    quat_weight = trial.suggest_int("quat_weight", 200, 5000, step=200)

    u_ratio = trial.suggest_int("u_ratio", -100, 99)

    init_x_offset = trial.suggest_int("init_x_offset", 13, 15)

    ratio = abs(u_ratio) / 10.0
    if (u_ratio < 0):
        u_lambda = 1.0
        u_eta = ratio
    else:
        ratio += 1
        u_lambda = ratio
        u_eta = 1.0

    with open(CONTORLLER_PARAMS, "r") as f:
        c3_options = yaml.safe_load(f)
        
    c3_options["c3_options"]["admm_iter"] = admm_iter
    c3_options["c3_options"]["w_G"] = w_G
    c3_options["c3_options"]["q_vector"][3] = plate_rot_cost
    c3_options["c3_options"]["q_vector"][4] = plate_rot_cost
    c3_options["Q_quaternion_weight"] = quat_weight
    c3_options["lcs_factory_options"]["N"] = tracking_N
    
    n_contacts = 8
    c3_options["c3_options"]["u_lambda"] = [u_lambda] * (4 * n_contacts)
    c3_options["c3_options"]["u_eta"] = [u_eta] * (4 * n_contacts)

    c3_options["c3_options"]["scale_lcs"] = False

    c3_options["x_init"][9] = init_x_offset / 100.0
    c3_options["x_des"][9] = init_x_offset / 100.0

    with open(CONTORLLER_PARAMS, "w") as f:
        yaml.dump(c3_options, f, default_flow_style=True)

# ======================================================================

    # iC3 parameters
    num_warmup_iters = trial.suggest_int("num_warmup_iters", 0, 1)
    warm_start_alpha = trial.suggest_int("warm_start_alpha", 0, 100)

    num_segments = trial.suggest_categorical("num_segments", [2, 5, 10])
    num_iters = trial.suggest_categorical("num_iters", [2, 3, 5, 6])
    alpha_ee = trial.suggest_int("alpha_ee", 0, 100)
    alpha_object = trial.suggest_int("alpha_object", 0, 100)

    accel_cost = trial.suggest_int("accel_cost", 0, 50, step=5)
    value_function_scaling = trial.suggest_int("value_function_scaling", 0, 100)

    with open(MSiC3_PARAMS, "r") as f:
        ic3_options = yaml.safe_load(f)
        
    ic3_options["num_warmup_iters"] = num_warmup_iters
    ic3_options["warm_start_alpha"] = warm_start_alpha / 100.0

    ic3_options["num_iters"] = num_iters
    ic3_options["num_segments"] = num_segments

    ic3_options["alpha_ee"] = alpha_ee / 100.0
    ic3_options["alpha_ee_step"] = (100 - alpha_ee) / (100 * (num_iters - 1))

    ic3_options["alpha_object"] = alpha_object / 100.0
    ic3_options["alpha_object_step"] = (100 - alpha_object) / (100 * (num_iters - 1))

    ic3_options["acceleration_cost_weight"] = accel_cost
    ic3_options["value_function_scaling"] = value_function_scaling / 100.0

    ic3_options["N"] = 50

    with open(MSiC3_PARAMS, "w") as f:
        yaml.dump(ic3_options, f, default_flow_style=True)

    # Construct and execute the bazel command
    cmd = [
        "./bazel-bin/examples/lcs_factory_system_example", 
        "--experiment_type=MSiC3_plate_optuna"
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

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
    This runs automatically after every trial. 
    It checks if the trial that just finished is the new best trial.
    """
    try: 
      # Check if the trial that just ended is the absolute best one so far
      if study.best_trial.number == trial.number:
          print(f"--> New best metric found: {study.best_value}. Saving to file...")
          
          # Open in "w" (write) mode to overwrite the file with the fresh best data
          with open("examples/resources/plate/optuna_best_params.txt", "w") as f:
              f.write("=========================================\n")
              f.write("       BEST HYPERPARAMETERS SO FAR       \n")
              f.write("=========================================\n")
              f.write(f"Best Trial Number: {study.best_trial.number}\n")
              f.write(f"Best Metric Value: {study.best_value}\n\n")
              f.write("Parameters:\n")
              
              for key, value in study.best_params.items():
                  f.write(f"  {key}: {value}\n")
              f.write("=========================================\n")
    except ValueError:
        print(f"Initial trial completed. Trial {trial.number} with value: {trial.value}")

# sed -i 's/\t/  /g' examples/resources/plate/optuna_ms_c3_tracking_options.yaml
# sed -i 's/\t/  /g' examples/resources/plate/optuna_ms_ic3_options.yaml
# python3 examples/resources/plate/optuna_plate.py
if __name__ == "__main__":

    STORAGE_URL = "sqlite:///examples/resources/plate/optuna_plate.db"

    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    sampler = optuna.samplers.TPESampler(multivariate=True)
    study = optuna.create_study(
        study_name="MSiC3_plate",
        storage=STORAGE_URL,
        load_if_exists=True,  
        sampler=sampler,
        direction="minimize")
    study.optimize(objective, n_trials=5000, callbacks=[log_best_callback])

    print("\n--- Optimization Complete ---")
    print(f"Best Trial Value: {study.best_value}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")