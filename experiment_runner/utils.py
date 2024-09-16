import numpy as np
import networkx as nx
from scipy.stats import kendalltau, multivariate_normal, multivariate_t
from sklearn.covariance import EmpiricalCovariance

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

def sample_from_mixed(n_samples: int,
                      cov_matrix: np.ndarray,
                      mix_param: float,
                      student_dof: float = 3.):
    dim = cov_matrix.shape[0]
    t_cov_scaler = (student_dof - 2) / student_dof
    norm_samples = multivariate_normal.rvs(np.zeros(dim), cov_matrix, size=n_samples)
    t_samples = multivariate_t.rvs(np.zeros(dim), t_cov_scaler * cov_matrix, df=student_dof, size=n_samples)
    selector = (np.random.uniform(0, 1, size=n_samples) < mix_param).astype(int)
    result = norm_samples.T * (1 - selector) + t_samples.T * selector

    return result.T

def fechner_corr(x, y):
    ind = ((x - np.mean(x))*(y - np.mean(y)))
    res = (np.sum((ind >= 0)) - np.sum((ind < 0))) / len(x)
    return res

def pearson_corr_mat(data):
    return np.corrcoef(data.T)

def kendall_corr_mat(data):
    dim = data.shape[1]
    matrix = np.array([[kendalltau(data[:, i], data[:, j]).statistic
                        for j in range(dim)] for i in range(dim)])

    return matrix

def fechner_corr_mat(data):
    dim = data.shape[1]
    matrix = np.array([[fechner_corr(data[:, i], data[:, j]) for j in range(dim)] for i in range(dim)])

    return matrix

def pearson_corr_via_kendall(data):
    kcorr_mat = kendall_corr_mat(data)
    pearson_corr = np.sin(np.pi / 2 * kcorr_mat)

    return pearson_corr

def pearson_corr_via_fechner(data):
    fcorr_mat = fechner_corr_mat(data)
    pearson_corr = np.sin(np.pi / 2 * fcorr_mat)

    return pearson_corr