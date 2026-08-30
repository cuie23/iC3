import yaml
import os
import re
import subprocess
import optuna
import optunahub
import sys
import platform

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
    w_G = trial.suggest_int("w_G", 1, 5000)
    g_lambda = trial.suggest_int("g_lambda", 1, 100)
    g_eta = trial.suggest_int("g_eta", 1, 100)
    # w_G_final = trial.suggest_int("w_G_final", 1, 100)
    w_G_final = 10

    plate_z_cost = trial.suggest_int("plate_z_cost", 50, 20000, step=50)
    # plate_rot_cost = trial.suggest_int("plate_rot_cost", 100, 50000, step=100)
    plate_rot_cost = 16000
    # tracking_N = trial.suggest_int("tracking_N", 3, 8)
    tracking_N = 4
    quat_weight = trial.suggest_int("quat_weight", 100, 50000, step=100)

    u_ratio = trial.suggest_int("u_ratio", -200, 199)
    # dt = trial.suggest_categorical("dt", [1, 2, 4])
    # dt = trial.suggest_int("dt", 1, 2)
    dt = 2

    # mu = trial.suggest_int("mu", 1, 100)
    mu = 33

    # scale_lcs = trial.suggest_categorical("scale_lcs", [True, False])
    scale_lcs = True

    # init_x_offset = trial.suggest_int("init_x_offset", 13, 15)
    init_x_offset = 13

    plate_offset = False

    ratio = abs(u_ratio)
    if (u_ratio < 0):
        u_lambda = 1.0
        u_eta = ratio
    else:
        ratio += 1
        u_lambda = ratio
        u_eta = 1.0

    with open(CONTORLLER_PARAMS, "r") as f:
        c3_options = yaml.safe_load(f)

    n_contacts = 8

    c3_options["c3_options"]["admm_iter"] = admm_iter
    c3_options["c3_options"]["w_G"] = w_G / 10.0
    c3_options["c3_options"]["g_lambda"] = [g_lambda] * (4 * n_contacts)
    c3_options["c3_options"]["g_eta"] =  [g_eta] * (4 * n_contacts)
    c3_options["c3_options"]["w_G_final"] = w_G_final
    c3_options["c3_options"]["g_u"] = [10, 10, 10, 10, 10]


    c3_options["c3_options"]["q_vector"][2] = plate_z_cost
    c3_options["c3_options"]["q_vector"][3] = plate_rot_cost
    c3_options["c3_options"]["q_vector"][4] = plate_rot_cost

    c3_options["c3_options"]["q_vector"][11] = 400
    c3_options["c3_options"]["q_vector"][15] = 8
    c3_options["c3_options"]["q_vector"][16] = 8

    c3_options["c3_options"]["r_vector"] = [1, 1, 1, 1, 1]


    c3_options["Q_quaternion_weight"] = quat_weight
    c3_options["lcs_factory_options"]["N"] = tracking_N
    c3_options["lcs_factory_options"]["dt"] = dt / 100.0
    c3_options["lcs_factory_options"]["mu"] = [mu / 100] * n_contacts

    c3_options["c3_options"]["u_lambda"] = [u_lambda] * (4 * n_contacts)
    c3_options["c3_options"]["u_eta"] = [u_eta] * (4 * n_contacts)

    c3_options["c3_options"]["scale_lcs"] = scale_lcs

    c3_options["x_init"][2] = -0.107 if plate_offset else 0
    c3_options["x_des"][2] = -0.107 if plate_offset else 0

    c3_options["x_init"][9] = init_x_offset / 100.0
    c3_options["x_des"][9] = init_x_offset / 100.0
    # c3_options["x_des"][9] = 0

    c3_options["x_init"][11] = 0.022
    c3_options["x_des"][11] = 0.022

    c3_options["x_init"][4] = 0
    c3_options["x_init"][5] = 1
    c3_options["x_init"][6] = 0
    c3_options["x_init"][7] = 0
    c3_options["x_init"][8] = 0

    # c3_options["x_des"][22] = 1
    c3_options["x_des"][22] = 0

    # start pre tilted
    # c3_options["x_init"][4] = 0.2

    # c3_options["x_init"][5] = 0.995004
    # c3_options["x_init"][6] = 0
    # c3_options["x_init"][7] = 0.099833
    # c3_options["x_init"][8] = 0

    with open(CONTORLLER_PARAMS, "w") as f:
        yaml.dump(c3_options, f, default_flow_style=True)

# ======================================================================

    # iC3 parameters
    # num_warmup_iters = trial.suggest_int("num_warmup_iters", 0, 2)
    # warm_start_alpha = trial.suggest_int("warm_start_alpha", 0, 100)
    num_warmup_iters = 0
    warm_start_alpha = 0

    num_segments = trial.suggest_categorical("num_segments", [10, 25, 50])
    num_iters = trial.suggest_categorical("num_iters", [3, 5, 6])
    alpha_ee = trial.suggest_int("alpha_ee", 0, 100)
    alpha_object = trial.suggest_int("alpha_object", 0, 100)

    # accel_cost = trial.suggest_int("accel_cost", 0, 50, step=5)
    accel_cost = 25
    # value_function_scaling = trial.suggest_int("value_function_scaling", 0, 100)
    value_function_scaling = 100
    vf_trust_region_weight = trial.suggest_int("vf_trust_region_weight", 0, 100)

    # torque_bound = trial.suggest_int("torque_bound", 16, 24, step=2)
    torque_bound = -10

    # use_lambdas_for_lcs = trial.suggest_categorical("use_lambdas_for_lcs", [True, False])
    use_lambdas_for_lcs = False
    
    # Kp_xy = trial.suggest_int("Kp_xy", 100, 1000, step=100)
    # Kd_xy = trial.suggest_int("Kd_xy", 20, 200, step=20)
    # Kp_z = trial.suggest_int("Kp_z", 100, 1000, step=100)
    # Kd_z = trial.suggest_int("Kd_z", 20, 200, step=20)
    # Kp_rot = trial.suggest_int("Kp_rot", 100, 1000, step=100)
    # Kd_rot = trial.suggest_int("Kd_rot", 20, 200, step=20)

    Kp_xy = 0
    Kd_xy = 0
    Kp_z = 0
    Kd_z = 0
    Kp_rot = 0
    Kd_rot = 0

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

    ic3_options["N"] = int(200 / dt)

    ic3_options["rollout_Kp"] = [Kp_xy, Kp_xy, Kp_z, Kp_rot, Kp_rot]
    ic3_options["rollout_Kd"] = [Kd_xy, Kd_xy, Kd_z, Kd_rot, Kd_rot]

    ic3_options["vf_trust_region_weight"] = vf_trust_region_weight
    ic3_options["num_threads"] = 32

    ic3_options["use_lambdas_for_lcs"] = use_lambdas_for_lcs

    ic3_options["rollout_dt_scaling"] = 1

    with open(MSiC3_PARAMS, "w") as f:
        yaml.dump(ic3_options, f, default_flow_style=True)

    ee_config = 2 if plate_offset else 1
    # Construct and execute the bazel command
    cmd = [
        "./bazel-bin/examples/lcs_factory_system_example", 
        "--experiment_type=MSiC3_plate_optuna",
        f"--optuna_instance={worker_id}",
        f"--ee_config={ee_config}",
        f"--plate_u_torque_bound={torque_bound / 10.0}"
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
                if (current_anchor_z < -1):
                    return 50000 * current_anchor_z * current_anchor_z
                
            # 4. Extract the actual (hat) quaternion and add to the running cost
            if "x_hat[L] pancake:" in line:
                raw_numbers = line.split("pancake:")[1].split()
                hat_z = float(raw_numbers[6])
                if (hat_z < -1 or hat_z > 1):
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
          with open("examples/resources/plate/optuna_plate/optuna_plate_best_params_less_params.txt", "w") as f:
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

    print(platform.release().lower())
    if "microsoft" in platform.release().lower():
        STORAGE_URL = "sqlite:////home/ttesc255/optuna_data/optuna_plate_less_params.db"
    else:
        STORAGE_URL = "sqlite:///examples/resources/plate/optuna_plate/optuna_plate_less_params.db"
        
    storage = optuna.storages.RDBStorage(
        url=STORAGE_URL,  
        heartbeat_interval=60            
    )

    sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, n_startup_trials=400)
    study = optuna.create_study(
        study_name="MSiC3_plate_less_params",
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