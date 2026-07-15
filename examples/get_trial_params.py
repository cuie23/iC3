import optuna

# STORAGE_PATH = "sqlite:///examples/resources/plate/optuna_plate.db"
# STUDY_NAME = "MSiC3_plate"

STORAGE_PATH = "sqlite:///examples/resources/multifinger_hand/optuna_point_hand_180/optuna_results_vf_scalar.db"
STUDY_NAME = "MSiC3_point_hand_180_vf_scalar"

# Load the existing study
study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_PATH)

trial_idx = 14

# Optuna uses 0-based indexing, so Trial 7 is at index 7
crashed_trial = study.trials[trial_idx]

print(f"Trial {crashed_trial.number} Status: {crashed_trial.state}")
for key, value in crashed_trial.params.items():
    print(f"  {key}: {value}")

# python3 examples/get_trial_params.py