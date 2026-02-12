import yfinance as yf
import numpy as np
import pandas as pd

def get_stock_data(ticker, start_date, end_date):
    # this line starts definition of the function for pulling out the needed data from yahoo finance data bank
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # this line extracts the closing price of stock, but also takes into consideration that the Stock might not have adjusted price for some period of time
    try:
        prices = data['Adj Close']
    except KeyError:
        prices = data['Close']
    

    # this calculates sum of daily returns of the stock and puts them in log function for numerical stability
    log_returns = np.log(1 + prices.pct_change())
    
    return prices, log_returns

# this is preperation for Normal distribution function, we calulate two needed elements: variance and the mean
# mean  represents  average direction of the stocks path over time (does it go up down or stays flat and by how much).
#variance measures how "spread out" the jumps are from the average, taking the absolute value
def get_stats(log_returns):
    # Calculates drift and variance
    u = log_returns.mean()
    var = log_returns.var()
    return u, var

