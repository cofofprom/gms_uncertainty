import os
import pickle
import numpy as np
from listdir import print_config
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score
import argparse
from multiprocessing import Pool

def get_metrics(true, pred):
    true = (true[np.triu_indices_from(true, k=1)] != 0.).astype(int)
    pred = (pred[np.triu_indices_from(pred, k=1)] != 0.).astype(int)

    metrics = np.zeros(8)
    metrics[0] = recall_score(true, pred, pos_label=1, zero_division=0)
    metrics[1] = recall_score(true, pred, pos_label=0, zero_division=0)
    metrics[2] = 1 - precision_score(true, pred, pos_label=1, zero_division=0)
    metrics[3] = 1 - precision_score(true, pred, pos_label=0, zero_division=0)

    metrics[4] = (metrics[0] + metrics[1]) / 2
    metrics[5] = accuracy_score(true, pred)
    metrics[6] = f1_score(true, pred, zero_division=0)
    metrics[7] = matthews_corrcoef(true, pred)

    return metrics

def calculate_for_top_level(data, models):
    models_metrics = []
    for model, model_data in data.items():
        reps_metrics = []
        for rep_idx, rep_data in enumerate(model_data):
            if rep_data == None:
                #print(f"There was missing data for eps={eps}, model={model} {rep_idx}th repetition")
                reps_metrics.append(np.array([np.nan for _ in range(8)]))
                continue

            pred = rep_data[1]
            true = models[model][1]
            metrics = get_metrics(true, pred)

            reps_metrics.append(metrics)
        reps_metrics = np.stack(reps_metrics)
        models_metrics.append(reps_metrics)
    models_metrics = np.stack(models_metrics)

    return models_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", action='store', type=str)

    args = parser.parse_args()

    data_dir = os.path.join(args.run_dir, "..")
    models_path = os.path.join(data_dir, "models.data")

    with open(models_path, 'rb') as f: models = pickle.load(f)

    print(f"Loaded {len(models)} models")

    print_config(args.run_dir)
    with open(os.path.join(args.run_dir, "output.data"), 'rb') as f: run_data = pickle.load(f)

    print("Contents:")
    for k, v in run_data.items():
        print(f"{np.around(k, 2)}: {len(v)} models")

    print("Starting calculation of metrics. Calculated metrics array will be stored in run dir")
    eps_metrics = []
    with Pool() as p:
        waiters = []
        for eps, data in run_data.items():
            waiters.append(p.apply_async(calculate_for_top_level, args=(data, models)))
        
        for k, w in zip(run_data.keys(), waiters):
            eps_metrics.append(w.get())
            print(f"Eps={k} completed!")

    eps_metrics = np.stack(eps_metrics)

    with open(os.path.join(args.run_dir, "metrics.data"), "wb") as f:
        pickle.dump(eps_metrics, f)

    print(f"Output shape: {eps_metrics.shape}")

if __name__ == "__main__":
    main()