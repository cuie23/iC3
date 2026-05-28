import yaml
import os
import re
import subprocess
import optuna

# Define paths to your parameter files
CONTORLLER_PARAMS = "examples/resources/plate/optuna_ms_c3_tracking_options.yaml"
MSiC3_PARAMS = "examples/resources/plate/optuna_ms_ic3_options.yaml"


def objective(trial):
    # C3 parameters
    admm_iter = trial.suggest_int("admm_iter", 3, 8)
    w_G = trial.suggest_int("w_G", 1, 200)
    plate_rot_cost = trial.suggest_int("plate_rot_cost", 50, 1000, step=50)
    tracking_N = trial.suggest_int("tracking_N", 5, 10)
    quat_weight = trial.suggest_int("quat_weight", 500, 5000, step=500)

    with open(CONTORLLER_PARAMS, "r") as f:
        c3_options = yaml.safe_load(f)
        
    c3_options["c3_options"]["admm_iter"] = admm_iter
    c3_options["c3_options"]["w_G"] = w_G
    c3_options["c3_options"]["q_vector"][3] = plate_rot_cost
    c3_options["c3_options"]["q_vector"][4] = plate_rot_cost
    c3_options["Q_quaternion_weight"] = quat_weight
    c3_options["lcs_factory_options"]["N"] = tracking_N
    
    with open(CONTORLLER_PARAMS, "w") as f:
        yaml.dump(c3_options, f, default_flow_style=True)

# ======================================================================

    # iC3 parameters
    num_warmup_iters = trial.suggest_int("num_warmup_iters", 0, 2)
    warm_start_alpha = trial.suggest_float("warm_start_alpha", 0, 1)

    num_segments = trial.suggest_categorical("num_segments", [2, 3, 4, 6])
    num_iters = trial.suggest_int("num_iters", 2, 5)
    alpha_ee = trial.suggest_float("alpha_ee", 0, 1)
    alpha_object = trial.suggest_float("alpha_object", 0, 1)

    with open(MSiC3_PARAMS, "r") as f:
        ic3_options = yaml.safe_load(f)
        
    ic3_options["num_warmup_iters"] = num_warmup_iters
    ic3_options["warm_start_alpha"] = warm_start_alpha

    ic3_options["num_iters"] = num_iters
    ic3_options["num_segments"] = num_segments

    ic3_options["alpha_ee"] = alpha_ee
    ic3_options["alpha_ee_step"] = (1 - alpha_ee) / (num_iters - 1)

    ic3_options["alpha_object"] = alpha_object
    ic3_options["alpha_object_step"] = (1 - alpha_object) / (num_iters - 1)

    
    with open(MSiC3_PARAMS, "w") as f:
        yaml.dump(ic3_options, f, default_flow_style=True)

    # Construct and execute the bazel command
    cmd = [
        "bazel", 
        "run", 
        "//examples:lcs_factory_system_example", 
        "--", 
        "--experiment_type=MSiC3_plate_optuna"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
            print(f"Trial failed with exit code {result.returncode}")
            print(f"Error log: {result.stderr}")
            raise optuna.TrialPruned()

    # Parse metric from std out
    print(result.stdout)
    match = re.search(r"FINAL_METRIC:\s*([0-9.]+(?:[eE][+-]?\d+)?)", result.stdout)

    if match:
        metric_string = match.group(1)
        
        # 2. Correctly check the string value for "e+"
        if "e+" in metric_string:
            raise optuna.TrialPruned("Could not find metric in output (scientific notation detected).")
            
        # Otherwise, proceed to convert it to a float
        return float(metric_string)
    else:
        raise optuna.TrialPruned("Could not find metric in output (scientific notation detected).")


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
    study = optuna.create_study(
        study_name="MSiC3_plate",
        storage=STORAGE_URL,
        load_if_exists=True,  
        direction="minimize")
    study.optimize(objective, n_trials=500, callbacks=[log_best_callback])

    print("\n--- Optimization Complete ---")
    print(f"Best Trial Value: {study.best_value}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")