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

STORAGE_PATH = "sqlite:///examples/resources/plate/optuna_plate/optuna_plate_obj_x_constraint_longer.db"
STUDY_NAME = "MSiC3_plate_obj_x_constraint_longer"

# STORAGE_PATH = "sqlite:///examples/resources/plate/optuna_plate/laptop_plate_db_2.db"
# STUDY_NAME = "MSiC3_plate_fixed_value_function"

TRIAL_NUMBER = 10384

# python3 examples/resources/plate/write_yaml_optuna_plate.py

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

plate_offset = False

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

try:
    g_lambda = trial.params["g_lambda"]
    g_eta = trial.params["g_eta"]
except KeyError:
    g_lambda = 2
    g_eta = 1

try:
    dt = trial.params["dt"] 
except KeyError:
    dt = 1

try:
    w_G_final = trial.params["w_G_final"]
except KeyError:
    w_G_final = 1

try:
    mu = trial.params["mu"]
except KeyError:
    mu = 33

u_ratio = trial.params["u_ratio"]
ratio = abs(u_ratio) 
if (u_ratio < 0):
    u_lambda = 1.0
    u_eta = ratio
else:
    ratio += 1
    u_lambda = ratio
    u_eta = 1.0

with open(CONTORLLER_PARAMS, "r") as f:
    c3_options = yaml.load(f)   # <-- was yaml.safe_load
    
n_contacts = 8

c3_options["c3_options"]["admm_iter"] = admm_iter
c3_options["c3_options"]["w_G"] = w_G / 10.0
c3_options["c3_options"]["g_lambda"] = flow_seq([g_lambda] * (4 * n_contacts))
c3_options["c3_options"]["g_eta"] = flow_seq([g_eta] * (4 * n_contacts))
c3_options["c3_options"]["g_u"] = flow_seq([10, 10, 10, 10, 10])
c3_options["c3_options"]["w_G_final"] = w_G_final

c3_options["c3_options"]["q_vector"][2] = plate_z_cost
c3_options["c3_options"]["q_vector"][3] = plate_rot_cost
c3_options["c3_options"]["q_vector"][4] = plate_rot_cost

c3_options["c3_options"]["q_vector"][11] = 400
c3_options["c3_options"]["q_vector"][15] = 8
c3_options["c3_options"]["q_vector"][16] = 8

c3_options["c3_options"]["r_vector"] = flow_seq([1, 1, 1, 1, 1])

c3_options["Q_quaternion_weight"] = quat_weight
c3_options["lcs_factory_options"]["N"] = tracking_N
c3_options["lcs_factory_options"]["dt"] = dt / 100.0
c3_options["lcs_factory_options"]["mu"] = flow_seq([mu / 100] * n_contacts)

c3_options["c3_options"]["u_lambda"] = flow_seq([u_lambda] * (4 * n_contacts))
c3_options["c3_options"]["u_eta"] = flow_seq([u_eta] * (4 * n_contacts))

c3_options["c3_options"]["scale_lcs"] = False

c3_options["x_init"][2] = -0.107 if plate_offset else 0
c3_options["x_des"][2] = -0.107 if plate_offset else 0

c3_options["x_init"][9] = init_x_offset / 100.0
c3_options["x_des"][9] = init_x_offset / 100.0
# c3_options["x_des"][9] = 0

c3_options["x_init"][11] = 0.022
c3_options["x_des"][11] = 0.022

if (STUDY_NAME == "MSiC3_plate_pitch_proj_z_velo_target"):
    c3_options["x_des"][22] = 1
else:
    c3_options["x_des"][22] = 0

c3_options["x_init"][4] = 0
c3_options["x_init"][5] = 1
c3_options["x_init"][6] = 0
c3_options["x_init"][7] = 0
c3_options["x_init"][8] = 0

# start pre tilted
# c3_options["x_init"][4] = 0.2

# c3_options["x_init"][5] = 0.995004
# c3_options["x_init"][6] = 0
# c3_options["x_init"][7] = 0.099833
# c3_options["x_init"][8] = 0

with open(CONTORLLER_PARAMS, "w") as f:
    yaml.dump(c3_options, f)

# ======================================================================

# iC3 parameters
try:
    num_warmup_iters = trial.params["num_warmup_iters"]
    warm_start_alpha = trial.params["warm_start_alpha"]
except:
    num_warmup_iters = 0
    warm_start_alpha = 0

num_segments = trial.params["num_segments"]
num_iters = trial.params["num_iters"]
alpha_ee = trial.params["alpha_ee"]
alpha_object = trial.params["alpha_object"]

try:
    accel_cost = trial.params["accel_cost"]
except KeyError:
    accel_cost = 0

try:
    value_function_scaling = trial.params["value_function_scaling"]
except KeyError:
    value_function_scaling = 100

try:
    use_lambdas_for_lcs = trial.params["use_lambdas_for_lcs"]
except KeyError:
    use_lambdas_for_lcs = False

try:
    vf_trust_region_weight = trial.params["vf_trust_region_weight"]
except KeyError:
    vf_trust_region_weight = 0

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
ic3_options["vf_trust_region_weight"] = vf_trust_region_weight
ic3_options["num_threads"] = 32
ic3_options["rollout_dt_scaling"] = 1

ic3_options["N"] = int(200 / dt)

ic3_options["rollout_Kp"] = flow_seq([Kp_xy, Kp_xy, Kp_z, Kp_rot, Kp_rot])
ic3_options["rollout_Kd"] = flow_seq([Kd_xy, Kd_xy, Kd_z, Kd_rot, Kd_rot])

ic3_options["use_lambdas_for_lcs"] = use_lambdas_for_lcs

with open(MSiC3_PARAMS, "w") as f:
    yaml.dump(ic3_options, f)

