
from py import process
import yaml
import re
import subprocess
import optuna
import optunahub
import sys
import math
import platform


# Define paths to your parameter files
OSQP_PARAMS = f"core/configs/solver_options_default.yaml"

def objective(trial):

    polish = trial.suggest_categorical("polish", [0, 1])
    polish_refine_iter = trial.suggest_int("polish_refine_iter", 1, 10)
    check_termination = trial.suggest_int("check_termination", 50, 1000, step=50)
    scaling = trial.suggest_int("scaling", 2, 50, step=2)

    rho_mult = trial.suggest_int("rho_mult", 1, 9) 
    rho_exp = trial.suggest_int("rho_exp", -5, 2) # rho = rho_mult * 10^exp
    sigma_exp = trial.suggest_int("sigma_exp", -8, -2)
    alpha = trial.suggest_int("alpha", 1, 19) 
    delta = trial.suggest_int("delta", -8, -2)

    with open(OSQP_PARAMS, "r") as f:
        osqp_params = yaml.safe_load(f)

    osqp_params["int_options"]["polish"] = int(polish)
    osqp_params["int_options"]["polish_refine_iter"] = polish_refine_iter
    osqp_params["int_options"]["check_termination"] = check_termination
    osqp_params["int_options"]["scaling"] = scaling

    osqp_params["double_options"]["rho"] = rho_mult * 10**rho_exp
    osqp_params["double_options"]["sigma"] = 10**sigma_exp
    osqp_params["double_options"]["alpha"] = alpha / 10
    osqp_params["double_options"]["delta"] = 10**delta

    with open(OSQP_PARAMS, "w") as f:
        yaml.dump(osqp_params, f, default_flow_style=False)

    # Construct and execute the bazel command
    cmd_plate = [
        "./bazel-bin/examples/lcs_factory_system_example", 
        f"--optuna_instance={0}", 
        "--experiment_type=MSiC3_plate",
        f"--ee_config={1}",
    ]
    cmd_speed = [
        "./bazel-bin/examples/lcs_factory_system_example", 
        f"--optuna_instance={0}", 
        "--experiment_type=MSiC3_point_hand_180",
        f"--ee_config={1}",
        f"--cube_model={1}"
    ]
    cmd_pivot = [
        "./bazel-bin/examples/lcs_factory_system_example", 
        f"--optuna_instance={0}", 
        "--experiment_type=MSiC3_point_hand",
        f"--ee_config={3}",
        f"--cube_model={4}"
    ]

    commands = [cmd_plate, cmd_speed, cmd_pivot]
    num_runs_per_cmd = 3
    all_runtimes = []

    for cmd_idx, cmd in enumerate(commands):
        for run_idx in range(num_runs_per_cmd):
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            run_score = None
            print(f"--- Output for Command {cmd_idx + 1}/{len(commands)} | Run {run_idx + 1}/{num_runs_per_cmd} ---", flush=True)

            try:
                for line in iter(process.stdout.readline, ''):
                    print(line, end='', flush=True)
                    
                    primal_match = re.search(r"Primal Res:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                    dual_match = re.search(r"Dual Res:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                    
                    if primal_match and float(primal_match.group(1)) > 0.01:
                        print(f"Large Primal Residual ({float(primal_match.group(1))}) on cmd {cmd_idx + 1} run {run_idx + 1}")
                        process.terminate()
                        process.wait()
                        return 30000 * float(primal_match.group(1))
                    if dual_match and float(dual_match.group(1)) > 0.01:
                        print(f"Large Dual Residual ({float(dual_match.group(1))}) on cmd {cmd_idx + 1} run {run_idx + 1}")
                        process.terminate()
                        process.wait()
                        return 30000 * float(dual_match.group(1))

                    # Extract Final Metric
                    final_match = re.search(r"Total runtime:\s*([0-9.]+)", line)
                    if final_match:
                        run_score = float(final_match.group(1))

            except optuna.TrialPruned as e:
                # Kill the C++ subprocess immediately to save compute time
                process.terminate() 
                process.wait() 
                raise e 
            
            finally:
                process.stdout.close()
                process.stderr.close()

            process.wait()
            if process.returncode != 0:
                print(f"Trial failed with exit code {process.returncode} on cmd {cmd_idx + 1} run {run_idx + 1}")
                raise optuna.TrialPruned(f"Process crashed or returned non-zero exit code on cmd {cmd_idx + 1} run {run_idx + 1}")

            if run_score is not None:
                all_runtimes.append(run_score)
            else:
                raise optuna.TrialPruned(f"Could not find Total runtime in output on cmd {cmd_idx + 1} run {run_idx + 1}.")

    total_runs = len(commands) * num_runs_per_cmd
    if len(all_runtimes) == total_runs:
        # Sum of average runtime for each command: sum(all 9 runtimes) / 3
        metric = sum(all_runtimes) / num_runs_per_cmd
        print(f"Runtimes across all {total_runs} runs: {all_runtimes}, Metric (sum of averages): {metric:.4f}s")
        return metric
    else:
        raise optuna.TrialPruned(f"Did not complete all {total_runs} runs.")
    
def log_best_callback(study, trial):

    if trial.value is None:
        return
    
    # 2. TRACK THE ABSOLUTE BEST TRAJECTORY CONFIG (Overwrites with absolute best)
    if study.best_trial.number == trial.number:
        print(f"--> New absolute best metric found: {trial.value}. Saving to file...")
        
        with open("examples/resources/optuna_speed/best_params_speed.txt", "w") as f:
            f.write("=========================================\n")
            f.write("       BEST HYPERPARAMETERS SO FAR       \n")
            f.write("=========================================\n")
            f.write(f"Best Trial Number: {trial.number}\n")
            f.write(f"Best Metric Value: {trial.value}\n\n")
            f.write("Parameters:\n")
            for key, value in trial.params.items():
                f.write(f"  {key}: {value}\n")
            f.write("=========================================\n")

# sed -i 's/\t/  /g' examples/resources/multifinger_hand/ms_c3_tracking_options_point_hand_speed.yaml
# sed -i 's/\t/  /g' examples/resources/multifinger_hand/ms_ic3_options_point_hand_speed.yaml
# python3 examples/optuna_speed.py
if __name__ == "__main__":

    print(platform.release().lower())
    if "microsoft" in platform.release().lower():
        STORAGE_URL = "sqlite:////home/ttesc255/optuna_data/optuna_results_speed.db"
    else:
      STORAGE_URL = "sqlite:///examples/resources/optuna_speed/optuna_results_speed.db"
        
    # module = optunahub.load_module(package="samplers/catcmawm")
    # sampler = module.CatCmawmSampler()
    sampler = optuna.samplers.TPESampler(multivariate=True)
    storage = optuna.storages.RDBStorage(
        url=STORAGE_URL  
    )

    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study = optuna.create_study(
        study_name="MSiC3_point_hand_speed",
        storage=storage,
        load_if_exists=True, 
        sampler=sampler, 
        direction="minimize")
    study.optimize(objective, n_trials=20000, callbacks=[log_best_callback])

    print("\n--- Optimization Complete ---")
    print(f"Best Trial Value: {study.best_value}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")