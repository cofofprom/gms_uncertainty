# This file describes parameters used for current experiment

import numpy as np
from utils import *
from solvers import *
import os

# Describes if previously generated models needs to be updated
UPDATE_MODELS = False

# --- Experiment config description ---
# 'name': name of folder for experiment data
# 'N': dimension of model
# 'n': number of samples
# 'dof': degrees of freedom for t-dist
# 'density': expected density of true graphical model
# 'algo_param': parameter that will be passed to estimator
# 'S_obs': number of generated datasets per graphical model
# 'S_sg':number of generated graphical models
# 'eps_list': mixture parameters set
# 'algorithm': utilized algorithm
# =====

experiment_config = {
    "name": "robust_selection_vs_robust_correlations", # name of folder for experiment data
    "N": 20, # dimension of model
    "n": 100, # number of samples
    "dof": 3, # degrees of freedom for t-dist
    "density": 0.2, # expected density of true graphical model
    'algo_param': 0.1, # parameter that will be passed to estimator
    'S_obs': 1, # number of generated datasets per graphical model
    'S_sg': 10, # number of generated graphical models
    'eps_list': np.linspace(0, 1, num=10), # mixture parameters set
    'algorithm': graphical_lasso_via_pearson # utilized algorithm for model selection
}

# Generate experiment data dir
exp_dir = os.path.join("data", experiment_config['name'])
os.makedirs(exp_dir, exist_ok=True)
experiment_config['datadir'] = exp_dir

# Construct models path
models_path = os.path.join(exp_dir, "models.data")
experiment_config['models'] = models_path