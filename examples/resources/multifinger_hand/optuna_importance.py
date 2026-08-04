import optuna
from optuna.importance import get_param_importances
# from optuna.visualization import plot_param_importances

# Define your storage URL and the specific study name
# STORAGE_URL = "sqlite:///examples/resources/multifinger_hand/optuna_point_hand_180/optuna_results_vf_scalar.db" 
# STUDY_NAME = "MSiC3_point_hand_180_vf_scalar" 

STORAGE_URL = "sqlite:///examples/resources/multifinger_hand/optuna_point_hand_pivot/optuna_results_pivot_choose_dt_direction_mu.db"
STUDY_NAME = "MSiC3_point_hand_pivot_choose_dt_direction_mu" 

# 1. Load the existing study from the database
study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)

importances = get_param_importances(study)
for param, weight in importances.items():
    print(f"{param}: {weight:.4f}")

# fig = plot_param_importances(study)
# fig.show()

# python3 examples/resources/multifinger_hand/optuna_importance.py