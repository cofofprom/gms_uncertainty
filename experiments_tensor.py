import numpy as np
from utils import generateDominantDiagonal, tlasso
import tensorly as tl
from sklearn.metrics import confusion_matrix
from time import perf_counter
import sys
from joblib import dump
import uuid

class TensorGMExperiment:
    def __init__(self, tensor_shape, densities, reg_params, n_samples, num_repl):
        self.dims = tensor_shape
        self.densities = densities
        self.reg_params = reg_params
        self.n_samples = n_samples
        self.n_repl = num_repl

        self.prepare_models()

    def prepare_models(self):
        self.covariances = []
        self.precisions = []
        self.adjacencies = []
        self.covariances_chol_cache = []

        for dim, density in zip(self.dims, self.densities):
            cov, prec, _ = generateDominantDiagonal(dim, density)
            self.covariances.append(cov)
            self.precisions.append(prec)
            self.adjacencies.append((prec != 0.).astype(int) - np.eye(dim))
            self.covariances_chol_cache.append(np.linalg.cholesky(cov))

    def generate_data(self):
        def sample():
            Z = tl.tensor(np.random.randn(*self.dims))
            Z = tl.tenalg.multi_mode_dot(Z, self.covariances_chol_cache)
            return Z

        samples = np.stack([sample() for _ in range(self.n_samples)])
        return samples
    
    def validate_models(self):
        pass

    def single_replication(self):
        data = self.generate_data()
        emp_precisions = tlasso(data, self.reg_params)

        emp_adjacencies = [(emp_precision != 0.).astype(int) \
                           - np.eye(emp_precision.shape[0]) \
                           for emp_precision in emp_precisions]
        
        confusions = [confusion_matrix(true_adj[np.triu_indices_from(true_adj, k=1)], 
                                       emp_adj[np.triu_indices_from(emp_adj, k=1)],
                                       labels=[0, 1]) 
                      for true_adj, emp_adj in zip(self.adjacencies, emp_adjacencies)]
        
        return np.stack(confusions)
    
    def run(self):
        self.validate_models()
        time_start = perf_counter()
        confusions = [self.single_replication() for repl in range(self.n_repl)]
        time_end = perf_counter()
        self.last_run_duration = time_end - time_start

        return np.stack(confusions)
        
def main():
    num_models = int(sys.argv[1])
    result_dir = sys.argv[2]

    model1_densities = np.linspace(0.1, 0.9, num=10)
    model2_density = 0.

    results = []
    for _ in range(num_models):
        density_exps = [TensorGMExperiment((20, 2), [d, model2_density], [0.1, 0.1], 100, 100) for d in model1_densities]
        confusions = np.stack([exp.run() for exp in density_exps])
        results.append(confusions)
    
    results = np.stack(results)
    with open(f'{result_dir}/{str(uuid.uuid4())}.bin', 'wb') as f:
        dump(results, f)

if __name__ == '__main__':
    main()