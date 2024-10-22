"""
    Stock portfolio using Monte Carlo simulation
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
# import plotly.offline as plty
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# pd.options.plotting.backend = 'plotly'


def get_data(stocks, start, end):
    stock_data  = yf.download(stocklist, start=start, end=end)
    stock_close = stock_data.Close
    returns     = stock_close.pct_change(fill_method='pad')

    return returns.mean(), returns.cov()

end = dt.datetime.now(timezone.utc)
start = end - dt.timedelta(days=500)
print(start, end)

"""
    Stock tickers to analyze
"""
stocklist  = ['SPY', '^NSEI', 'NQ=F']
N          = len(stocklist)

initialValue = 10000

meanReturns, covReturns = get_data(stocklist, start, end)

L = np.linalg.cholesky(covReturns)

'''
    Convex weights for portfolio
'''
wts  = np.random.rand(N)
wts /= wts.sum()

"""
    Monte Carlo sampling
"""
nsamples = 200
time = 100

meanMC = np.full(shape=(time, N), fill_value=meanReturns).T
portfolio = np.zeros(shape=(time, nsamples))

for i in range(nsamples): 
    Z_t             = np.random.normal(size=(time, N))
    dailyReturns    = meanMC + np.inner(L, Z_t)
    portfolio[:, i] = np.cumprod(np.inner(wts, dailyReturns.T)+1)*initialValue

# plt.plot(portfolio)
# plt.ylabel('Portfolio value (in eur.)')
# plt.xlabel('Days')
# plt.title('Monte Carlo Simulation of a portfolio')
# plt.show()