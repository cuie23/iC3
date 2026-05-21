import yaml
import os
import re
import subprocess
import optuna

# Define paths to your parameter files
CONTORLLER_PARAMS = "examples/resources/plate/optuna_ms_c3_tracking_options_point_hand_180.yaml"
MSiC3_PARAMS = "examples/resources/plate/optuna_ms_ic3_options_point_hand_180.yaml"


def objective(trial):
    # C3 parameters
    w_G = trial.suggest_int("w_G", 5, 100)
    g_lambda = trial.suggest_int("g_lambda", 1, 100)
    g_eta = trial.suggest_int("g_eta", 1, 20)
    u_lambda = trial.suggest_int("u_lambda", 1, 100)
    u_eta = trial.suggest_int("u_eta", 1, 20)

    with open(CONTORLLER_PARAMS, "r") as f:
        c3_options = yaml.safe_load(f)
        
    n_lambda = len(c3_options["c3_options"]["g_eta_slack"])

    c3_options["c3_options"]["w_G"] = w_G
    c3_options["c3_options"]["g_lambda"] = [g_lambda] * (4*n_lambda)
    c3_options["c3_options"]["g_eta_slack"] = [g_eta] * n_lambda
    c3_options["c3_options"]["g_eta_n"] = [g_eta] * n_lambda
    c3_options["c3_options"]["g_eta_t"] = [g_eta] * n_lambda
    c3_options["c3_options"]["g_eta"] = [g_eta] * (4*n_lambda)

    c3_options["c3_options"]["u_lambda"] = [u_lambda] * (4*n_lambda)
    c3_options["c3_options"]["u_eta_slack"] = [u_eta] * n_lambda
    c3_options["c3_options"]["u_eta_n"] = [u_eta] * n_lambda
    c3_options["c3_options"]["u_eta_t"] = [u_eta] * n_lambda
    c3_options["c3_options"]["u_eta"] = [u_eta] * (4*n_lambda)
    
    with open(CONTORLLER_PARAMS, "w") as f:
        yaml.dump(c3_options, f, default_flow_style=True)

# ======================================================================

    # iC3 parameters
    num_warmup_iters = trial.suggest_int("num_warmup_iters", 0, 4)
    warm_start_alpha = trial.suggest_float("warm_start_alpha", 0, 1)

    num_iters = trial.suggest_int("num_iters", 2, 6)
    alpha_ee = trial.suggest_float("alpha_ee", 0, 1)
    alpha_object = trial.suggest_float("alpha_object", 0, 1)

    with open(MSiC3_PARAMS, "r") as f:
        ic3_options = yaml.safe_load(f)
        
    ic3_options["num_warmup_iters"] = num_warmup_iters
    ic3_options["warm_start_alpha"] = warm_start_alpha

    ic3_options["num_iters"] = num_iters

    ic3_options["alpha_ee"] = warm_start_alpha
    ic3_options["alpha_ee_step"] = (1 - warm_start_alpha) / (num_iters - 1)

    ic3_options["alpha_object"] = warm_start_alpha
    ic3_options["alpha_object_step"] = (1 - warm_start_alpha) / (num_iters - 1)

    
    with open(MSiC3_PARAMS, "w") as f:
        yaml.dump(ic3_options, f, default_flow_style=True)

    # Construct and execute the bazel command
    cmd = [
        "bazel", 
        "run", 
        "//examples:lcs_factory_system_example", 
        "--", 
        "--experiment_type=MSiC3_point_hand_180_optuna"
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
    This runs automatically after every trial. 
    It checks if the trial that just finished is the new best trial.
    """
    # Check if the trial that just ended is the absolute best one so far
    if study.best_trial.number == trial.number:
        print(f"--> New best metric found: {study.best_value}. Saving to file...")
        
        # Open in "w" (write) mode to overwrite the file with the fresh best data
        with open("examples/resources/multifinger_hand/optuna_point_hand_180/best_params_15_segments.txt", "w") as f:
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
# python3 examples/resources/multifinger_hand/optuna_point_hand_180.py
if __name__ == "__main__":

    STORAGE_URL = "sqlite:///examples/resources/multifinger_hand/optuna_point_hand_180/optuna_results_15_segments.db"

    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study = optuna.create_study(
        study_name="MSiC3_point_hand_180",
        storage=STORAGE_URL,
        load_if_exists=True,  
        direction="minimize")
    study.optimize(objective, n_trials=100, callbacks=[log_best_callback])

    print("\n--- Optimization Complete ---")
    print(f"Best Trial Value: {study.best_value}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")