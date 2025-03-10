import networkx as nx
import numpy as np
from scipy.stats import t, kendalltau, norm
import tensorly as tl
from sklearn.covariance import graphical_lasso
from scipy.linalg import fractional_matrix_power

def generateDominantDiagonal(dim: int, density: float) -> tuple:
    # Generate random Erdos-Renyi graph with given probability of an edge (density)
    graph = nx.gnp_random_graph(dim, density)
    # Transform the graph to adjacency matrix
    adj = nx.adjacency_matrix(graph).toarray()
    
    # Generate random matrix A where elements are uniformly distributed
    A = np.random.uniform(0.5, 1, size=(dim, dim))

    # Generate random matrix B where elements are randomly selected from the set {-1, 1}
    B = np.random.choice([-1, 1], size=(dim, dim))

    # Find the matrix prec where elements are uniformly distributed on (-1, 0.5] AND [0.5, 1)
    # and if there's no edge between nodes i and j the value will be 0
    # multiplication here means element-wise multiplication
    prec = adj * A * B
    # Find the sum of absolute values of each row (at this point diagonal contains only zeros)
    rowsums = np.sum(np.abs(prec), axis=1)
    rowsums[rowsums == 0] = 0.0001 # handles the situation where row is completely empty

    # Scale each row by 1.5 times of corresponding row sum
    prec = prec / (1.5 * rowsums[:, None])
    # Make matrix symmetric and add ones on the diagonal
    prec = (prec + prec.T) / 2 + np.eye(dim)

    precision = prec
    # Find the covariance as matrix inverse of precision (since precision is SPD)
    covariance = np.linalg.inv(precision)

    # Calculate partial correlations (just in case)
    pD = np.diag(1 / np.sqrt(np.diag(precision))) 
    pcorr = -(pD @ precision @ pD)
    np.fill_diagonal(pcorr, 1)

    return covariance, precision, pcorr

def pcorr_pvalues(r, n, N):
    dof = n - N
    stat = r * np.sqrt(dof / (1 - (r ** 2)))
    pval = 2 * t.sf(np.abs(stat), dof)
    return pval

def corr_pvalues(r, n, N=0):
    dof = n - 2
    stat = r * np.sqrt(dof / (1 - (r ** 2)))
    pval = 2 * t.sf(np.abs(stat), dof)
    return pval

def tau_pvalues(r, n, N=0):
    return r

def pcorrcoef(X):
    cov = np.cov(X)
    prec = np.linalg.inv(cov)
    D = np.diag(1 / np.sqrt(np.diag(prec)))
    corr = -(D @ prec @ D)
    np.fill_diagonal(corr, 1)
    return corr

def tau(X):
    corr = np.zeros((X.shape[0], X.shape[0]))
    for idx1 in range(len(X)):
        for idx2 in range(idx1, len(X)):
            corr[idx1, idx2] = kendalltau(X[idx1], X[idx2]).pvalue

    corr += corr.T + np.eye(X.shape[0])

    return corr

def fechner_corr(x, y):
    ind = ((x - np.mean(x))*(y - np.mean(y)))
    res = (np.sum((ind >= 0)) - np.sum((ind < 0))) / len(x)
    return res

def fechnercoef(X):
    corr = np.eye(X.shape[0])
    for idx1 in range(len(X)):
        for idx2 in range(idx1 + 1, len(X)):
            corr[idx1, idx2] = fechner_corr(X[idx1], X[idx2])

    return corr + corr.T - np.eye(X.shape[0])

def fechner_pvalues(r, n, N=0):
    r = (r*n + n) / 2
    stat = (r - 0.5*n) / np.sqrt(0.5*n*0.5)

    pvalue = 2 * norm.sf(np.abs(stat))

    return pvalue


class MHTSolver:
    def __init__(self, alpha, p_val_fun, corr_fun=np.corrcoef):
        self.alpha = alpha
        self.p_val_fun = p_val_fun
        self.corr_fun = corr_fun

    def fit(self, X, y=None):
        self.n_obs, self.dim = X.shape
        self.n_tests = self.dim * (self.dim - 1) // 2
        self.corr_mat = self.corr_fun(X.T)
        np.fill_diagonal(self.corr_mat, 0)
        self.p_values = np.vectorize(self.p_val_fun)(self.corr_mat,
                                                     self.n_obs,
                                                     self.dim)

    def apply_correction(self, procedure):
        if procedure == 'SI':
            return (self.p_values < self.alpha).astype(int)

        if procedure == 'B':
            return (self.p_values < (self.alpha / self.n_tests)).astype(int)

        if procedure == 'H':
            triu_idx = np.triu_indices_from(self.p_values, k=1)

            triu_idx_list = np.array(triu_idx).T
            pvals_list = self.p_values[triu_idx]

            perm = np.argsort(pvals_list)

            idx_sorted = [triu_idx_list[idx] for idx in perm]
            pvals_sorted = [pvals_list[idx] for idx in perm]

            adj = np.zeros((self.dim, self.dim))

            for k in range(1, self.n_tests + 1):
                curve_val = self.alpha / (self.n_tests + 1 - k)
                if pvals_sorted[k - 1] > curve_val:
                    break
                adj[tuple(idx_sorted[k - 1])] = 1
                adj[tuple(idx_sorted[k - 1])[::-1]] = 1
            return adj

        if procedure == 'BH':
            triu_idx = np.triu_indices_from(self.p_values, k=1)

            triu_idx_list = np.array(triu_idx).T
            pvals_list = self.p_values[triu_idx]

            perm = np.argsort(pvals_list)

            idx_sorted = [triu_idx_list[idx] for idx in perm]
            pvals_sorted = [pvals_list[idx] for idx in perm]

            adj = np.ones((self.dim, self.dim)) - np.eye(self.dim)

            for k in range(self.n_tests, 0, -1):
                curve_val = self.alpha * k / self.n_tests
                if pvals_sorted[k - 1] <= curve_val:
                    break
                adj[tuple(idx_sorted[k - 1])] = 0
                adj[tuple(idx_sorted[k - 1])[::-1]] = 0
            return adj

        if procedure == 'BY':
            triu_idx = np.triu_indices_from(self.p_values, k=1)

            triu_idx_list = np.array(triu_idx).T
            pvals_list = self.p_values[triu_idx]

            perm = np.argsort(pvals_list)

            idx_sorted = [triu_idx_list[idx] for idx in perm]
            pvals_sorted = [pvals_list[idx] for idx in perm]

            adj = np.ones((self.dim, self.dim)) - np.eye(self.dim)
            harm = np.log(self.n_tests) + np.euler_gamma + 1 / (2*self.n_tests)

            for k in range(self.n_tests, 0, -1):
                curve_val = self.alpha * k / self.n_tests / harm
                if pvals_sorted[k - 1] <= curve_val:
                    break
                adj[tuple(idx_sorted[k - 1])] = 0
                adj[tuple(idx_sorted[k - 1])[::-1]] = 0
            return adj
        
def tensor_normal_sample(mean, covariances, size=1):
    chol_L = [np.linalg.cholesky(cov) for cov in covariances]
    
    def sample():
        Z = tl.tensor(np.random.randn(*mean.shape))
        Z = tl.tenalg.multi_mode_dot(Z, chol_L)
        return Z

    samples = np.stack([sample() for _ in range(size)])
    
    return samples

def emp_way_cov(data, way, precisions):
    corrected_precisions = [fractional_matrix_power(prec, 1/2) \
                            if (i+1) != way else np.eye(data.shape[way]) \
                            for i, prec in enumerate(precisions)]
    result_cov = np.zeros((data.shape[way], data.shape[way]))
    
    for T_i in data:
        V_i = tl.unfold(tl.tenalg.multi_mode_dot(T_i, corrected_precisions), way - 1)
        result_cov += V_i @ V_i.T

    return data.shape[way] / len(data) / np.prod(data.shape[1:]) * result_cov

def tlasso(data, reg_params, max_iters=100):
    solutions = [np.eye(dim) for dim in data.shape[1:]]
    diffs = np.array([np.inf for _ in solutions])

    for t in range(max_iters):
        for k in range(1, len(data.shape[1:])+1):
            S_k = emp_way_cov(data, k, solutions)
            _, emp_prec = graphical_lasso(S_k, reg_params[k-1])
            emp_prec /= np.linalg.norm(emp_prec)
            diffs[k-1] = np.linalg.norm(np.abs(solutions[k-1] - emp_prec))
            solutions[k-1] = emp_prec
        
        if np.all(diffs <= 1e-4): break
    
    return solutions