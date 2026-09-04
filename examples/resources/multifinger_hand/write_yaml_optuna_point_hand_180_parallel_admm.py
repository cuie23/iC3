#!/usr/bin/env python3
import sys
import optuna
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

yaml = YAML()
yaml.preserve_quotes = True
yaml.width = 100000

def flow_seq(vals):
    seq = CommentedSeq(vals)
    seq.fa.set_flow_style()
    return seq

# python3 examples/resources/multifinger_hand/write_yaml_optuna_point_hand_180_parallel_admm.py

# Define paths to parameter files
CONTROLLER_PARAMS = "examples/resources/multifinger_hand/ms_c3_tracking_options_point_hand_180.yaml"
MSIC3_PARAMS = "examples/resources/multifinger_hand/ms_ic3_options_point_hand_180.yaml"

STORAGE_PATH = "sqlite:///examples/resources/multifinger_hand/optuna_point_hand_180_parallel_admm/optuna_results_180_parallel_admm_defects.db"
STUDY_NAME = "MSiC3_point_hand_180_parallel_admm_defects"

# Set specific trial number or None to use best trial
TRIAL_NUMBER = 267
if len(sys.argv) > 1:
    TRIAL_NUMBER = int(sys.argv[1])

study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_PATH)

if TRIAL_NUMBER is not None:
    trial = None
    for t in study.trials:
        if t.number == TRIAL_NUMBER:
            trial = t
            break
    if trial is None:
        raise ValueError(f"Could not find trial number {TRIAL_NUMBER}")
else:
    trial = study.best_trial
    print(f"Using best trial #{trial.number} with metric value: {trial.value}")

print(f"Loading parameters from Trial #{trial.number}")

# =========================================================================
# 1. C3 Tracking Controller Parameters
# =========================================================================
w_G = trial.params.get("w_G", 100)
g_lambda = trial.params.get("g_lambda", 50)
g_eta = trial.params.get("g_eta", 50)

u_ratio_finger = trial.params.get("u_ratio_finger", 0)
u_ratio_cube = trial.params.get("u_ratio_cube", 0)

admm_iter = trial.params.get("admm_iter", 3)
tracking_N = trial.params.get("tracking_N", 4)

finger_position_weight = trial.params.get("finger_position_weight", 3000)
cube_position_weight = trial.params.get("cube_position_weight", 8000)
quat_weight = trial.params.get("quat_weight", 4000)

w_G_final = 10
x_change_weight = 1
u_change_weight = 1
n_contacts = 7

with open(CONTROLLER_PARAMS, "r") as f:
    c3_options = yaml.load(f)

c3_options["c3_options"]["penalize_input_change"] = True
c3_options["c3_options"]["penalize_x_change"] = True
c3_options["c3_options"]["input_change_weight"] = (u_change_weight - 1) / 100.0
c3_options["c3_options"]["x_change_weight"] = (x_change_weight - 1) / 100.0

c3_options["c3_options"]["w_G"] = w_G / 100.0
c3_options["c3_options"]["g_lambda"] = flow_seq([g_lambda] * (4 * n_contacts))
c3_options["c3_options"]["g_eta"] = flow_seq([g_eta] * (4 * n_contacts))

c3_options["c3_options"]["g_gamma"] = flow_seq([])
c3_options["c3_options"]["g_lambda_n"] = flow_seq([])
c3_options["c3_options"]["g_lambda_t"] = flow_seq([])
c3_options["c3_options"]["g_eta_slack"] = flow_seq([])
c3_options["c3_options"]["g_eta_n"] = flow_seq([])
c3_options["c3_options"]["g_eta_t"] = flow_seq([])

c3_options["c3_options"]["w_G_final"] = w_G_final
c3_options["c3_options"]["admm_iter"] = admm_iter

# Set U weights
finger_ratio = abs(u_ratio_finger)
if u_ratio_finger < 0:
    u_lambda_finger = 1.0
    u_eta_finger = finger_ratio
else:
    finger_ratio += 1
    u_lambda_finger = finger_ratio
    u_eta_finger = 1.0

cube_ratio = abs(u_ratio_cube)
if u_ratio_cube < 0:
    u_lambda_cube = 1.0
    u_eta_cube = cube_ratio
else:
    cube_ratio += 1
    u_lambda_cube = cube_ratio
    u_eta_cube = 1.0

c3_options["c3_options"]["u_lambda"] = flow_seq([u_lambda_finger] * (4 * 3) + [u_lambda_cube] * (4 * (n_contacts - 3)))
c3_options["c3_options"]["u_eta"] = flow_seq([u_eta_finger] * (4 * 3) + [u_eta_cube] * (4 * (n_contacts - 3)))

c3_options["c3_options"]["u_gamma"] = flow_seq([])
c3_options["c3_options"]["u_lambda_n"] = flow_seq([])
c3_options["c3_options"]["u_lambda_t"] = flow_seq([])
c3_options["c3_options"]["u_eta_slack"] = flow_seq([])
c3_options["c3_options"]["u_eta_n"] = flow_seq([])
c3_options["c3_options"]["u_eta_t"] = flow_seq([])

c3_options["lcs_factory_options"]["N"] = tracking_N

# Update q_vector weights
q_vector = list(c3_options["c3_options"]["q_vector"])
for i in range(9):
    q_vector[i] = finger_position_weight
q_vector[13] = cube_position_weight
q_vector[14] = cube_position_weight
q_vector[15] = 1000
for i in range(15):
    q_vector[16 + i] = 1

c3_options["c3_options"]["q_vector"] = flow_seq(q_vector)
c3_options["Q_quaternion_weight"] = quat_weight

with open(CONTROLLER_PARAMS, "w") as f:
    yaml.dump(c3_options, f)

print(f"Updated {CONTROLLER_PARAMS}")

# =========================================================================
# 2. MS-iC3 ADMM Parameters
# =========================================================================
num_segments = trial.params.get("num_segments", 30)
num_iters = trial.params.get("num_iters", 10)

anchor_rho = trial.params.get("anchor_rho", 5.0)

accel_cost = 10
use_pd = trial.params.get("use_pd", False)

with open(MSIC3_PARAMS, "r") as f:
    ic3_options = yaml.load(f)

ic3_options["N"] = 420
ic3_options["num_segments"] = num_segments
ic3_options["num_iters"] = num_iters
ic3_options["num_warmup_iters"] = 0
ic3_options["warm_start_alpha"] = 0.0

ic3_options["anchor_rho"] = float(anchor_rho)

ic3_options["acceleration_cost_weight"] = accel_cost

kp = 200 if use_pd else 0
kd = 20 if use_pd else 0

ic3_options["rollout_Kp"] = flow_seq([kp] * 9)
ic3_options["rollout_Kd"] = flow_seq([kd] * 9)
ic3_options["rollout_dt_scaling"] = 10
ic3_options["print_costs"] = False
ic3_options["num_threads"] = 32

ic3_options["use_drake_sim"] = True
ic3_options["drake_sim_dt"] = 0.0001
ic3_options["use_rollout_lambdas"] = True
ic3_options["use_lambdas_for_lcs"] = False

ic3_options["add_position_constraints"] = True
ic3_options["add_input_constraints"] = True
ic3_options["early_termination"] = True
ic3_options["penalize_acceleration"] = True

with open(MSIC3_PARAMS, "w") as f:
    yaml.dump(ic3_options, f)

print(f"Updated {MSIC3_PARAMS}")
print("\nSummary of applied parameters:")
print(f"  num_segments:        {num_segments}")
print(f"  num_iters:           {num_iters}")
print(f"  anchor_rho:          {anchor_rho}")
