from sklearn.covariance import graphical_lasso, empirical_covariance
from utils import pearson_corr_via_kendall, pearson_corr_via_fechner
from robust_selection import RobustSelection

def graphical_lasso_via_pearson(data, reg_param):
    covar = empirical_covariance(data, assume_centered=True)
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

def graphical_lasso_robsel(data, params):
    param = RobustSelection(data, params)

    return graphical_lasso_via_pearson(data, param)