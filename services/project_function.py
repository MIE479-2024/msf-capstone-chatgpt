from services.strategies import *

def project_function(periodReturns, periodFactRet, x0, short, strategy, weights):
    # run the proper optimization based on strategy
    if strategy == "OLS_MVO_ChatGPT":
        Strategy = OLS_MVO_ChatGPT()
        x = Strategy.execute_strategy(periodReturns, periodFactRet, short)

    elif strategy == "ChatGPT_Weights":
        Strategy = ChatGPT_Weights()
        x = Strategy.execute_strategy(weights)

    elif strategy == "OLS_Robust_MVO_ChatGPT":
        Strategy = OLS_Robust_MVO_ChatGPT()
        x = Strategy.execute_strategy(periodReturns, periodFactRet, short)
        
    elif strategy == "OLS_CVAR_ChatGPT":
        Strategy = OLS_CVAR_ChatGPT()
        x = Strategy.execute_strategy(periodReturns, periodFactRet, short)
    
    return x
