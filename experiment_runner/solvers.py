from sklearn.covariance import graphical_lasso, empirical_covariance

def graphical_lasso_via_pearson(data, reg_param):
    covar = empirical_covariance(data, assume_centered=True)
    try:
        result = graphical_lasso(covar, reg_param, return_costs=True, max_iter=500)
    except:
        result = None

    return result