from sklearn.covariance import graphical_lasso, empirical_covariance
from utils import pearson_corr_via_kendall, pearson_corr_via_fechner
from robust_selection import RobustSelection
import numpy as np

def graphical_lasso_via_pearson(data, reg_param):
    covar = np.corrcoef(data.T)
    try:
        result = graphical_lasso(covar, reg_param, return_costs=True, max_iter=500)
    except:
        result = None

    return result

def graphical_lasso_via_kendall(data, reg_param):
    covar = pearson_corr_via_kendall(data)
    try:
        result = graphical_lasso(covar, reg_param, return_costs=True, max_iter=500)
    except:
        result = None

    return result

def graphical_lasso_via_fechner(data, reg_param):
    covar = pearson_corr_via_fechner(data)
    try:
        result = graphical_lasso(covar, reg_param, return_costs=True, max_iter=500)
    except:
        result = None

    return result

def graphical_lasso_robsel(data, param):
    param = RobustSelection(data, param)

    return graphical_lasso_via_pearson(data, param)