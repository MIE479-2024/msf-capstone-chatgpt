import cvxpy as cp
import numpy as np
from scipy.stats import chi2

def MVO(mu, Q, short):
    # find the total number of assets
    n = len(mu)

    # set the target as the average expected return of all assets
    targetRet = np.mean(mu)

    # add the expected return constraint
    A = -1 * mu.T
    b = -1 * targetRet

    # constrain weights to sum to 1
    Aeq = np.ones([1, n])
    beq = 1

    # define the variable for asset weights
    x = cp.Variable(n)

    # set up the constraints list
    constraints = [
        A @ x <= b,
        Aeq @ x == beq
    ]

    # apply long/short constraints based on the `short` list
    for i in range(n):
        if short[i] == 0:
            # Long position constraint (non-negative weight)
            constraints.append(x[i] >= 0)
        else:
            # Short position constraint (non-positive weight)
            constraints.append(x[i] <= 0)

    # define and solve the optimization problem
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, Q)), constraints)
    
    prob.solve(verbose=False)
    return x.value

def Robust_MVO(mu, Q, lambda_value, alpha_value, N, short):
    # find the total number of assets
    n = len(mu)

    # uncertainty adjustment (theta and epsilon)
    theta = np.sqrt((1 / N) * np.diag(Q))
    epsilon = np.sqrt(chi2.ppf(alpha_value, n))

    # Define the portfolio weights variable
    x = cp.Variable(n)

    # Define the robust optimization objective
    robust_adjustment = epsilon * cp.norm(cp.multiply(theta, x), 2)
    objective = cp.Minimize(lambda_value * cp.quad_form(x, Q) - (mu.T @ x - robust_adjustment))

    # Constraints
    constraints = [
        cp.sum(x) == 1,  # Weights sum to 1
    ]

    # apply long/short constraints based on the `short` list
    for i in range(n):
        if short[i] == 0:
            # Long position constraint (non-negative weight)
            constraints.append(x[i] >= 0)
        else:
            # Short position constraint (non-positive weight)
            constraints.append(x[i] <= 0)

    # Set up and solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Return the optimized weights
    return x.value

def CVAR(mu, Q, alpha_value, returns, factRet, short):
    # number of assets and time periods
    n = mu.shape[0]
    N = returns.shape[0]
    returns_matrix = returns.values

    # calculate the minimum acceptable return R using the geometric mean of factor returns
    R = np.exp(np.mean(np.log(factRet + 1))) - 1

    # define portfolio weights variable
    x = cp.Variable(n)

    # define the auxiliary variable z (losses exceeding the CVaR threshold)
    z = cp.Variable(N)

    # define the CVaR (gamma)
    gamma = cp.Variable()

    # objective function (minimize CVaR + penalty term for exceeding losses)
    objective = cp.Minimize(
        gamma + (1 / ((1 - alpha_value) * N)) * cp.sum(z)
    )

    # constraints
    constraints = [
        cp.sum(x) == 1,  # portfolio weights sum to 1
       
        mu.T @ x >= R,    # portfolio expected return should be greater than or equal to the target return R
        
        # z_s >= 0 for all s
        z >= 0
    ]

    # CVaR constraints: z_s >= -r_s' x - gamma for each scenario s
    for i in range(N):
        constraints.append(z[i] >= -returns_matrix[i, :] @ x - gamma)

    # apply long/short constraints based on the `short` list
    for i in range(n):
        if short[i] == 0:
            # Long position constraint (non-negative weight)
            constraints.append(x[i] >= 0)
        else:
            # Short position constraint (non-positive weight)
            constraints.append(x[i] <= 0)

    # Set up and solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=True)

    # Return the optimized portfolio weights
    return x.value