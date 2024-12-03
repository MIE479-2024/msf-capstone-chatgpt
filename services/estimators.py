import numpy as np


def OLS(returns, factRet):
    # basic OLS regression with all factors
    
    # number of observations and factors
    [T, p] = factRet.shape

    # data matrix
    X = np.concatenate([np.ones([T, 1]), factRet.values], axis=1)

    # regression coefficients
    B = np.linalg.solve(X.T @ X, X.T @ returns)

    # separate B into alpha and betas
    a = B[0, :]
    V = B[1:, :]

    # residual variance
    ep = returns - X @ B
    sigma_ep = 1 / (T - p - 1) * np.sum(ep.pow(2), axis=0)
    D = np.diag(sigma_ep)

    # factor expected returns and covariance matrix
    f_bar = np.expand_dims(factRet.mean(axis=0).values, 1)
    F = factRet.cov().values

    # calculate the asset expected returns and covariance matrix
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar
    Q = V.T @ F @ V + D

    # make the matrix perfectly symetric
    Q = (Q + Q.T) / 2

    return mu, Q
