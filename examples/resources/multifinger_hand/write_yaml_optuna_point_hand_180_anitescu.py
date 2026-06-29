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
CONTORLLER_PARAMS = "examples/resources/multifinger_hand/ms_c3_tracking_options_point_hand_180.yaml"
MSiC3_PARAMS = "examples/resources/multifinger_hand/ms_ic3_options_point_hand_180.yaml"

STORAGE_PATH = "sqlite:///examples/resources/multifinger_hand/optuna_point_hand_180/optuna_results_anitescu_drake_pd_config3.db"
STUDY_NAME = "MSiC3_point_hand_180_anitescu_drake_pd_config3"

TRIAL_NUMBER = 30

study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_PATH)
trial = None
for t in study.trials:
    if t.number == TRIAL_NUMBER:
        trial = t
        break

if trial is None:
    raise ValueError(f"Could not find trial number {TRIAL_NUMBER}")

w_G = trial.params["w_G"]
g_x_fingers = trial.params["g_x_fingers"]
g_x_cube = trial.params["g_x_cube"]
g_u = trial.params["g_u"]

g_lambda = trial.params["g_lambda"]
g_eta = trial.params["g_eta"]

# g_gamma = trial.params["g_gamma"]
# g_lambda_n = trial.params["g_lambda_n"]
# g_lambda_t = trial.params["g_lambda_t"]
# g_eta_slack = trial.params["g_eta_slack"]
# g_eta_n = trial.params["g_eta_n"]
# g_eta_t = trial.params["g_eta_t"]

u_lambda = trial.params["u_lambda"]
u_eta = trial.params["u_eta"]

# u_gamma = trial.params["u_gamma"]
# u_lambda_n = trial.params["u_lambda_n"]
# u_lambda_t = trial.params["u_lambda_t"]
# u_eta_slack = trial.params["u_eta_slack"]
# u_eta_n = trial.params["u_eta_n"]
# u_eta_t = trial.params["u_eta_t"]

lambda_threshold = trial.params["lambda_threshold"]
eta_threshold = trial.params["eta_threshold"]
# gamma_threshold = trial.params["gamma_threshold"]
# phi_threshold = trial.params["phi_threshold"]
# add_phi_buffer = trial.params["add_phi_buffer"]

admm_iter = trial.params["admm_iter"]
finger_position_weight = trial.params["finger_position_weight"]
cube_position_weight = trial.params["cube_position_weight"]
tracking_N = trial.params["tracking_N"]
quat_weight = trial.params["quat_weight"]

with open(CONTORLLER_PARAMS, "r") as f:
    c3_options = yaml.load(f)
    
n_contacts = 7

c3_options["c3_options"]["w_G"] = w_G
c3_options["c3_options"]["g_lambda"] = flow_seq([g_lambda] * (4*n_contacts))
c3_options["c3_options"]["g_eta"] = flow_seq([g_eta] * (4*n_contacts))

c3_options["c3_options"]["g_gamma"] = flow_seq([] )
c3_options["c3_options"]["g_lambda_n"] = flow_seq([])
c3_options["c3_options"]["g_lambda_t"] = flow_seq([])
c3_options["c3_options"]["g_eta_slack"] = flow_seq([])
c3_options["c3_options"]["g_eta_n"] = flow_seq([])
c3_options["c3_options"]["g_eta_t"] = flow_seq([])

c3_options["c3_options"]["u_lambda"] = flow_seq([u_lambda] * (4*n_contacts))
c3_options["c3_options"]["u_eta"] = flow_seq([u_eta] * (4*n_contacts))

c3_options["c3_options"]["u_gamma"] = flow_seq([])
c3_options["c3_options"]["u_lambda_n"] = flow_seq([])
c3_options["c3_options"]["u_lambda_t"] = flow_seq([])
c3_options["c3_options"]["u_eta_slack"] = flow_seq([])
c3_options["c3_options"]["u_eta_n"] = flow_seq([])
c3_options["c3_options"]["u_eta_t"] = flow_seq([])

c3_options["c3_options"]["lambda_threshold"] = flow_seq([(lambda_threshold / 100.0)] * (4 * n_contacts))
c3_options["c3_options"]["eta_threshold"] = flow_seq([(eta_threshold / 10.0)] * (4 * n_contacts))
c3_options["c3_options"]["gamma_threshold"] = flow_seq([])
c3_options["c3_options"]["phi_threshold"] = flow_seq([])
c3_options["c3_options"]["add_phi_buffer"] = False
c3_options["c3_options"]["epsilon"] = flow_seq([])

g_x = list(c3_options["c3_options"]["g_x"])

for i in range(9):
    g_x[i] = g_x_fingers
for i in range(7):
    g_x[9+i] = g_x_cube

c3_options["c3_options"]["g_x"] = flow_seq(g_x)
c3_options["c3_options"]["g_u"] = flow_seq([g_u] * 9)

c3_options["c3_options"]["u_x"] = flow_seq([1] * 31)
c3_options["c3_options"]["u_u"] = flow_seq([1] * 9)


c3_options["c3_options"]["admm_iter"] = admm_iter

c3_options["lcs_factory_options"]["mu"] = flow_seq([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
c3_options["lcs_factory_options"]["N"] = tracking_N
c3_options["lcs_factory_options"]["dt"] = 0.02
c3_options["lcs_factory_options"]["contact_model"] = "anitescu"

c3_options["x_init"] = flow_seq([0.0, 0.07, 0.05,  # finger 1 
                        0.05, -0.07, 0.05,   # finger 2
                        -0.05, -0.07, 0.05,   # finger 3
                        1, 0, 0, 0, # cube orientation
                        0, 0, 0.051,  # cube position
                        0, 0, 0,     # finger 1 velo
                        0, 0, 0,     # finger 2 velo
                        0, 0, 0,     # finger 3 velo
                        0, 0, 0,     # cube ang velo
                        0, 0, 0])   	# cube velo

c3_options["x_des"] = flow_seq([0.0, 0.07, 0.05,  # finger 1 
                        0.05, -0.07, 0.05,   # finger 2
                        -0.05, -0.07, 0.05,   # finger 3
                        0, 0, 0, 1, # cube orientation
                        0, 0, 0.051,  # cube position
                        0, 0, 0,     # finger 1 velo
                        0, 0, 0,     # finger 2 velo
                        0, 0, 0,     # finger 3 velo
                        0, 0, 0,     # cube ang velo
                        0, 0, 0])   	# cube velo

q_vector = list(c3_options["c3_options"]["q_vector"])

for i in range(9): 
    q_vector[i] = finger_position_weight
q_vector[13] = cube_position_weight
q_vector[14] = cube_position_weight

c3_options["c3_options"]["q_vector"] = flow_seq(q_vector)

c3_options["Q_quaternion_weight"] = quat_weight
c3_options["c3_options"]["scale_lcs"] = True

with open(CONTORLLER_PARAMS, "w") as f:
    yaml.dump(c3_options, f)

# ======================================================================

# iC3 parameters
num_segments = trial.params["num_segments"]
num_warmup_iters = trial.params["num_warmup_iters"]
warm_start_alpha = trial.params["warm_start_alpha"]

num_iters = trial.params["num_iters"]
alpha_ee = trial.params["alpha_ee"]
alpha_object = trial.params["alpha_object"]

accel_cost = trial.params["accel_cost"]

# use_pd = trial.params["use_pd"]
use_pd = 0

with open(MSiC3_PARAMS, "r") as f:
    ic3_options = yaml.load(f)

           
ic3_options["N"] = 600
ic3_options["num_segments"] = num_segments

ic3_options["num_warmup_iters"] = num_warmup_iters

warm_start_alpha_real = warm_start_alpha / 100.0
ic3_options["warm_start_alpha"] = warm_start_alpha_real

ic3_options["num_iters"] = num_iters

alpha_ee_real = alpha_ee / 100.0
ic3_options["alpha_ee"] = alpha_ee_real
ic3_options["alpha_ee_step"] = (1 - alpha_ee_real) / (num_iters - 1)

alpha_object_real = alpha_object / 100.0
ic3_options["alpha_object"] = alpha_object_real
ic3_options["alpha_object_step"] = (1 - alpha_object_real) / (num_iters - 1)

ic3_options["acceleration_cost_weight"] = accel_cost

kp = 200 if use_pd == 1 else 0
kd = 20 if use_pd == 1 else 0

ic3_options["rollout_Kp"] = flow_seq([kp] * 9)
ic3_options["rollout_Kd"] = flow_seq([kd] * 9)
ic3_options["rollout_dt_scaling"] = 10

ic3_options["print_costs"] = False

ic3_options["use_drake_sim"] = True
ic3_options["drake_sim_dt"] = 0.0001

with open(MSiC3_PARAMS, "w") as f:
    yaml.dump(ic3_options, f)

# python3 examples/resources/multifinger_hand/write_yaml_optuna_point_hand_180_anitescu.py