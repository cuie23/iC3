import yaml
import os
import re
import subprocess
import optuna
import optunahub
import sys

if len(sys.argv) > 1:
    worker_id = int(sys.argv[1])
    print(f"Running worker number: {worker_id}")
else:
    # Fallback if you forget to pass the integer
    worker_id = 0 
    print("No integer passed, defaulting to 0")

CONTORLLER_PARAMS = f"examples/resources/plate/optuna_plate/optuna_yamls/optuna_ms_c3_tracking_options_{worker_id}.yaml"
MSiC3_PARAMS = f"examples/resources/plate/optuna_plate/optuna_yamls/optuna_ms_ic3_options_{worker_id}.yaml"

def objective(trial):
    # C3 parameters
    admm_iter = trial.suggest_int("admm_iter", 3, 6)
    w_G = trial.suggest_int("w_G", 1, 1000)
    plate_z_cost = trial.suggest_int("plate_z_cost", 50, 5000, step=50)
    plate_rot_cost = trial.suggest_int("plate_rot_cost", 100, 5000, step=100)
    tracking_N = trial.suggest_int("tracking_N", 3, 6)
    quat_weight = trial.suggest_int("quat_weight", 500, 50000, step=500)

    u_ratio = trial.suggest_int("u_ratio", -100, 99)

    # init_x_offset = trial.suggest_int("init_x_offset", 13, 15)
    init_x_offset = 13

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
    c3_options["c3_options"]["w_G"] = w_G / 10.0

    c3_options["c3_options"]["q_vector"][2] = plate_z_cost
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

    c3_options["x_init"][11] = 0.022
    c3_options["x_des"][11] = 0.022

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
    # value_function_scaling = trial.suggest_int("value_function_scaling", 0, 100)
    value_function_scaling = 100

    Kp_xy = trial.suggest_int("Kp_xy", 100, 1000, step=100)
    Kd_xy = trial.suggest_int("Kd_xy", 20, 200, step=20)
    Kp_z = trial.suggest_int("Kp_z", 100, 1000, step=100)
    Kd_z = trial.suggest_int("Kd_z", 20, 200, step=20)
    Kp_rot = trial.suggest_int("Kp_rot", 100, 1000, step=100)
    Kd_rot = trial.suggest_int("Kd_rot", 20, 200, step=20)

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

    ic3_options["N"] = 200

    ic3_options["rollout_Kp"] = [Kp_xy, Kp_xy, Kp_z, Kp_rot, Kp_rot]
    ic3_options["rollout_Kd"] = [Kd_xy, Kd_xy, Kd_z, Kd_rot, Kd_rot]


    with open(MSiC3_PARAMS, "w") as f:
        yaml.dump(ic3_options, f, default_flow_style=True)

    # Construct and execute the bazel command
    cmd = [
        "./bazel-bin/examples/lcs_factory_system_example", 
        "--experiment_type=MSiC3_plate_optuna",
        f"--optuna_instance={worker_id}"
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
            
            # 3. Extract the target z
            if "x anchor pancake" in line:
                raw_numbers = line.split("pancake")[1].split()
                current_anchor_z = float(raw_numbers[6])
                if (current_anchor_z < -1.5):
                    return 50000 * current_anchor_z * current_anchor_z
                
            # 4. Extract the actual (hat) quaternion and add to the running cost
            if "x_hat[L] pancake:" in line:
                raw_numbers = line.split("pancake:")[1].split()
                hat_z = float(raw_numbers[6])
                if (hat_z < -1.5):
                    return 50000 * hat_z * hat_z

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
          with open("examples/resources/plate/optuna_plate/optuna_plate_best_params_pd3.txt", "w") as f:
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
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    STORAGE_URL = "sqlite:///examples/resources/plate/optuna_plate/optuna_plate_pd3.db"
    storage = optuna.storages.RDBStorage(
        url=STORAGE_URL,  
        heartbeat_interval=60            
    )

    sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True)
    study = optuna.create_study(
        study_name="MSiC3_plate_pd3",
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