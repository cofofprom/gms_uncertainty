from utils import *
import numpy as np
from sklearn.metrics import confusion_matrix
import multiprocessing as mp
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from scipy.stats import hmean
import time
import sys

def fixed_model_experiment(covar, true_edges, config):
    solver = MHTSolver(config['sig_alpha'], corr_pvalues, np.corrcoef)

    results = []

    for n_repl in range(config['S_obs']):
        res = []
        data = np.random.multivariate_normal(np.zeros(config['N']), covar, size=config['n'])
        solver.fit(data)

        for correction in config['algs']:
            pred_adj = solver.apply_correction(correction)
            pred_edges = pred_adj[np.triu_indices_from(pred_adj, k=1)]
            
            conf = confusion_matrix(true_edges, pred_edges, labels=[0,1])

            res.append(conf)
        
        results.append(res)

    results = np.stack(results)

    return results

def multiple_models_experiment(density, config, n_jobs=None):
    models = [generateDominantDiagonal(config['N'], density) for n_model in range(config['S_sg'])]

    results = []

    with mp.Pool(n_jobs) as pool:
        waiters = []

        for model in models:
            covar = model[1] # We control density in the inverse matrix

            true_adj = (covar != 0.).astype(int) - np.eye(config['N'])
            true_edges = true_adj[np.triu_indices_from(true_adj, k=1)]

            waiters.append(pool.apply_async(fixed_model_experiment, args=(covar, true_edges, config)))

        for waiter in waiters:
            results.append(waiter.get())

    results = np.stack(results)

    return results

def vary_density_experiment(config, output_file, n_jobs=None):
    results = []
    for density in config['densities']:
        print(f'Working on d={np.around(density, 2)}...')
        ts_start = time.perf_counter()
        results.append(multiple_models_experiment(density, config))
        ts_end = time.perf_counter()

        print(f"it took {np.around(ts_end - ts_start, 2)}s")

    results = np.stack(results)

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    return results

def draw_metrics(exp_data, config, output_file1, output_file2):
    densities = config['densities']
    algs = config['algs']

    tn = exp_data[:, :, :, :, 0, 0]
    fp = exp_data[:, :, :, :, 0, 1]
    fn = exp_data[:, :, :, :, 1, 0]
    tp = exp_data[:, :, :, :, 1, 1]

    fdr = np.nan_to_num(fp / (tp + fp), nan=0)
    fomr = np.nan_to_num(fn / (tn + fn), nan=0)
    tpr = np.nan_to_num(tp / (tp + fn), nan=1)
    tnr = np.nan_to_num(tn / (tn + fp), nan=1)

    ba = (tpr + tnr) / 2
    f1 = hmean([1 - fdr, tpr])
    mcc_first = tpr * tnr * (1 - fdr) * (1 - fomr)
    mcc_second = (1 - tpr) * (1 - tnr) * fomr * fdr
    mcc = np.sqrt(mcc_first) - np.sqrt(mcc_second)

    fig, axes = plt.subplots(2, 2)
    metrics = [tnr, fomr, fdr, tpr]
    mname = ['TNR', 'FOR', 'FDR', 'TPR']
    axes_ord = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for m, ax, mn in zip(metrics, axes_ord, mname):
        m = np.reshape(m, (len(densities), -1, len(algs)))
        for idx, alg in enumerate(algs):
            ys = m[:, :, idx].mean(axis=-1)

            ax.plot(densities, ys, label=alg)

        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0.1, 0.9)
        ax.set_title(mn)

    fig.set_size_inches(15, 15)
    fig.savefig(output_file1, pad_inches=0.5, bbox_inches='tight')

    fig, axes = plt.subplots(1, 3)
    metrics = [ba, f1, mcc]
    mname = ['BA', 'F1', 'MCC']
    axes_ord = axes

    for m, ax, mn in zip(metrics, axes_ord, mname):
        m = np.reshape(m, (len(densities), -1, len(algs)))
        for idx, alg in enumerate(algs):
            ys = m[:, :, idx].mean(axis=-1)

            ax.plot(densities, ys, label=alg)
        
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0.1, 0.9)
        ax.set_title(mn)

    fig.set_size_inches(15, 10)
    fig.savefig(output_file2, pad_inches=0.5, bbox_inches='tight')

def main():
    densities = np.linspace(0.1, 0.9, 20)

    config = {
        'N': 30,
        'n': int(sys.argv[1]),
        'S_sg': 500,
        'S_obs': 100,
        'sig_alpha': 0.05,
        'densities': densities,
        'algs': ('SI', 'B', 'H', 'BH', 'BY'),
    }

    exp_name = f'dim{config["N"]}_n{config["n"]}_Ssg{config["S_sg"]}_Sobs{config["S_obs"]}_pearson'
    exp_data = vary_density_experiment(config, f'data/{exp_name}.pickle')
    print(f'Got {exp_data.shape} shape')

    draw_metrics(exp_data, config,
                 f'plots/tpr_{exp_name}.png',
                 f'plots/mcc_{exp_name}.png')

if __name__ == '__main__':
    main()