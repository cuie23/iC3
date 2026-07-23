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

# Define paths to your parameter files
CONTORLLER_PARAMS = "examples/resources/plate/ms_c3_tracking_options.yaml"
MSiC3_PARAMS = "examples/resources/plate/ms_ic3_options.yaml"

STORAGE_PATH = "sqlite:///examples/resources/plate/optuna_plate/optuna_plate_pd2.db"
STUDY_NAME = "MSiC3_plate_pd2"

TRIAL_NUMBER = 4275

study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_PATH)
trial = None
for t in study.trials:
    if t.number == TRIAL_NUMBER:
        trial = t
        break

if trial is None:
    raise ValueError(f"Could not find trial number {TRIAL_NUMBER}")

print(trial.params.keys())

admm_iter = trial.params["admm_iter"]
w_G = trial.params["w_G"] 
plate_rot_cost = trial.params["plate_rot_cost"]
tracking_N = trial.params["tracking_N"]
quat_weight = trial.params["quat_weight"]

try:
    plate_z_cost = trial.params["plate_z_cost"]
except KeyError:
    print("fallback plate z cost")
    plate_z_cost = 800

try:
    init_x_offset = trial.params["init_x_offset"]
except KeyError:
    print("fallback x offset")
    init_x_offset = 13


u_ratio = trial.params["u_ratio"]
ratio = abs(u_ratio) / 10.0
if (u_ratio < 0):
    u_lambda = 1.0
    u_eta = ratio
else:
    ratio += 1
    u_lambda = ratio
    u_eta = 1.0

with open(CONTORLLER_PARAMS, "r") as f:
    c3_options = yaml.load(f)   # <-- was yaml.safe_load

c3_options["c3_options"]["admm_iter"] = admm_iter
c3_options["c3_options"]["w_G"] = w_G / 10.0
c3_options["c3_options"]["q_vector"][2] = plate_z_cost
c3_options["c3_options"]["q_vector"][3] = plate_rot_cost
c3_options["c3_options"]["q_vector"][4] = plate_rot_cost
c3_options["Q_quaternion_weight"] = quat_weight
c3_options["lcs_factory_options"]["N"] = tracking_N

n_contacts = 8
c3_options["c3_options"]["u_lambda"] = flow_seq([u_lambda] * (4 * n_contacts))
c3_options["c3_options"]["u_eta"] = flow_seq([u_eta] * (4 * n_contacts))

c3_options["c3_options"]["scale_lcs"] = False

c3_options["x_init"][9] = init_x_offset / 100.0
c3_options["x_des"][9] = init_x_offset / 100.0

c3_options["x_init"][11] = 0.022
c3_options["x_des"][11] = 0.022

with open(CONTORLLER_PARAMS, "w") as f:
    yaml.dump(c3_options, f)

# ======================================================================

# iC3 parameters
num_warmup_iters = trial.params["num_warmup_iters"]
warm_start_alpha = trial.params["warm_start_alpha"]

num_segments = trial.params["num_segments"]
num_iters = trial.params["num_iters"]
alpha_ee = trial.params["alpha_ee"]
alpha_object = trial.params["alpha_object"]

accel_cost = trial.params["accel_cost"]

try:
    value_function_scaling = trial.params["value_function_scaling"]
except KeyError:
    value_function_scaling = 100

with open(MSiC3_PARAMS, "r") as f:
    ic3_options = yaml.load(f)

try:
    Kp_xy = trial.params["Kp_xy"]
    Kd_xy = trial.params["Kd_xy"]
    Kp_z = trial.params["Kp_z"]
    Kd_z = trial.params["Kd_z"]
    Kp_rot = trial.params["Kp_rot"]
    Kd_rot = trial.params["Kd_rot"]
except KeyError:
    print("fallback Kp Kd")
    Kp_xy = 0
    Kd_xy = 0
    Kp_z = 0
    Kd_z = 0
    Kp_rot = 0
    Kd_rot = 0

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

ic3_options["rollout_Kp"] = flow_seq([Kp_xy, Kp_xy, Kp_z, Kp_rot, Kp_rot])
ic3_options["rollout_Kd"] = flow_seq([Kd_xy, Kd_xy, Kd_z, Kd_rot, Kd_rot])

with open(MSiC3_PARAMS, "w") as f:
    yaml.dump(ic3_options, f)

# python3 examples/resources/plate/write_yaml_optuna_plate.py