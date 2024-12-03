import numpy as np
from services.estimators import *
from services.optimization import *

class OLS_MVO_ChatGPT:
    # uses historical returns to estimate the covariance matrix and expected return

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns, short):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :param short:
        :return:x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q = OLS(returns, factRet)
        x = MVO(mu, Q, short)
        return x
 
class ChatGPT_Weights:
    # uses chatgpt provided weights

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, weights):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param weights:
        :return:x
        """
        x = weights
        return x
    
class OLS_Robust_MVO_ChatGPT:
    # uses historical returns to estimate the covariance matrix and expected return

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns, short):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :param short:
        :return:x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q = OLS(returns, factRet)
        lambda_value = 20
        alpha_value = 0.9
        N = len(periodReturns)
        x = Robust_MVO(mu, Q, lambda_value, alpha_value, N, short)
        return x

class OLS_CVAR_ChatGPT:
    # uses historical returns to estimate the covariance matrix and expected return

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns, short):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :param short:
        :return:x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q = OLS(returns, factRet)
        alpha_value = 0.9
        N = len(periodReturns)
        x = CVAR(mu, Q, alpha_value, returns, factRet, short)
        return x