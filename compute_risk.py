"""
    Compute VaR and CVaR for stock portfolio
"""
"""
    Import dependencies
"""
import datetime as dt 
from datetime import timezone
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from simulate_portfolio import portfolio

def MC_VaR(returns, alpha=5):
    """
        Input : pandas series of returns 
        Output: percentile on return distribution to a given confidence level alpha 
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else: 
        raise TypeError("Expected a pandas data series.")
    
def MC_CVaR(returns, alpha=5):
    """
        Input : pandas series of returns 
        Output: percentile on return distribution to a given confidence level alpha 
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= MC_VaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    else: 
        raise TypeError("Expected a pandas data series.")

portfolio_results = pd.Series(portfolio[-1, :])

VaR  = MC_VaR(portfolio_results, alpha=5)
CVaR = MC_CVaR(portfolio_results, alpha=5)

print('VaR ${}'.format(VaR))
print('CVaR ${}'.format(CVaR))
