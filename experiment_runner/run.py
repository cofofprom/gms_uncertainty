from utils import *
from config import *
from solvers import graphical_lasso_via_pearson
import numpy as np
from robust_selection import RobustSelection
import multiprocessing as mp
from sklearn.covariance import graphical_lasso, empirical_covariance
import os
import pickle
import logging
import time

logging.getLogger().setLevel(logging.DEBUG)

def process_model(model, experiment_config):
    cov, prec, pcorr = model
    n_samples = experiment_config['n']
    dof = experiment_config['dof']
    n_rep = experiment_config['S_obs']
    mix_params = experiment_config['eps_list']
    algorithm_params = experiment_config['algo_param']
    algorithm_runner = experiment_config['algorithm']
    
    output = {}
    for mix_param in mix_params:
        output[mix_param] = []
        for it in range(n_rep):
            data = sample_from_mixed(n_samples, cov, mix_param, dof)
            result = algorithm_runner(data, algorithm_params)

            output[mix_param].append(result)

    return output

if __name__ == "__main__":
    require_generation = True
    if os.path.exists(models_path) and not UPDATE_MODELS:
        logging.warning("Experiment already has a set of pregenerated models. If you wish to update existing models, set UPDATE_MODELS=True.")
        require_generation = False
    elif os.path.exists(models_path) and UPDATE_MODELS:
        logging.warning("Experiment already has a set of pregenerated models. Updating them.")

    if require_generation:
        dim = experiment_config['N']
        density = experiment_config['density']
        num_models = experiment_config['S_sg']

        logging.info(f"Creating a set of S_sg={experiment_config['S_sg']}... Params: (N={dim}, d={density})")

        with open(models_path, "wb") as f:
            models = [generateDominantDiagonal(dim, density) for _ in range(num_models)]

            pickle.dump(models, f)

    logging.info("Reading model data...")
    with open(experiment_config['models'], "rb") as f:
        models = pickle.load(f)

        logging.info(f"Successfully loaded {len(models)} models")

    experiment_results = {eps: dict() for eps in experiment_config['eps_list']}
    def agg(x, i):
        for eps in x:
            if i not in experiment_results[eps]: experiment_results[eps][i] = list()
            experiment_results[eps][i].extend(x[eps])

    with mp.Pool() as p:
        waiters = []
        for i, model in enumerate(models):
            w = p.apply_async(process_model, args=(model, experiment_config))
            waiters.append(w)
        
        for i, w in enumerate(waiters):
            x = w.get()
            agg(x, i)

    assert(len(experiment_config['eps_list']) == len(experiment_results))
    logging.info(f"Successfully collected estimator output for {len(experiment_results)} models")
    logging.info("Saving experiment result...")

    result_path = os.path.join(experiment_config['datadir'], f"{int(time.time_ns())}")
    os.makedirs(result_path, exist_ok=True)

    with open(os.path.join(result_path, "output.data"), "wb") as f:
        pickle.dump(experiment_results, f)
        logging.info(f"Successfully saved result")

    with open(os.path.join(result_path, "config"), "w") as f:
        for k, v in experiment_config.items():
            f.write(f"{k}: {v}\n")

        logging.info(f"Successfully saved config")