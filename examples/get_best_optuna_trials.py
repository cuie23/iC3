import optuna

STORAGE_PATH = "sqlite:///examples/resources/plate/optuna_plate/optuna_plate_vert_proj_offset_x_target.db"
STUDY_NAME = "MSiC3_plate_vert_proj_offset_x_target"

# STORAGE_PATH = "sqlite:///examples/resources/multifinger_hand/optuna_point_hand_180/optuna_results_180_wG_final.db"
# STUDY_NAME = "MSiC3_point_hand_180_wG_final"

# STORAGE_PATH = "sqlite:///examples/resources/multifinger_hand/optuna_point_hand_180/optuna_results_lighter_cubes_no_thresh.db"
# STUDY_NAME = "MSiC3_point_hand_180_lighter_cubes_no_thresh"

# STORAGE_PATH = "sqlite:///examples/resources/multifinger_hand/optuna_point_hand_pivot/optuna_results_pivot_anitescu_drake_pruning.db"
# STUDY_NAME = "MSiC3_point_hand_pivot_anitescu_drake_pruning"

TOP_N = 5  # Change this to however many you want

study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_PATH)
print(len(study.trials))

# Filter to only completed trials and sort by value (ascending since direction="minimize")
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
top_trials = sorted(completed_trials, key=lambda t: t.value)[:TOP_N]

print(f"{'Rank':<6} {'Trial #':<10} {'Value':<20} Parameters")
print("=" * 80)
for rank, trial in enumerate(top_trials, start=1):
    print(f"{rank:<6} {trial.number:<10} {trial.value:<20.6f}")
    for key, value in trial.params.items():
        print(f"       {key}: {value}")
    print("-" * 80)

# python3 examples/get_best_optuna_trials.py