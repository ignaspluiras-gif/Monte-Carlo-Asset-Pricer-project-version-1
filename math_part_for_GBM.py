import numpy as np
import pandas as pd
from scipy.stats import norm
#we define the Geometric Brownian Motion short for (GBM)
# last_price - thats the starting point for our prediction
#mu - thats the average direction of the stocks path over time
# sigma - thats the root of our variance, it shows the absolute value of volatility
# n_sims - number of times simulation is done
# time horizon - the priod of time we are predicting

def simulate_gbm(last_price, mu, sigma, n_sims, time_horizon):
    # Z - the matrix with time horizon on Y axis and n_sims on X-axis, following Normal Distribution
    
    Z = np.random.normal(0, 1, (time_horizon, n_sims))


    
    
    # volatility_drag= -0.5 * sigma**2
    expected_CAGR = (mu - 0.5 * sigma**2)
    
    
    # daily returns are now defined as exponential of drift + schock to the normal Distribution
    daily_returns = np.exp(expected_CAGR + sigma * Z)
    
    # basically just a matrix of same shape as Z with 0 in every part
    price_paths = np.zeros_like(daily_returns)
    
    # Day 0 (the first row) for all simulations is set to today's actual stock price.
    price_paths[0] = last_price
    
    # Today's price = Yesterday's price * Today's simulated daily return multiplier.
    for t in range(1, time_horizon):
        price_paths[t] = price_paths[t-1] * daily_returns[t]
        
    return price_paths

#this is to finds the confidence level or more intuitevly P((-infinity, t]) = a, where a is confindnce level and t is the cutoff price 
def calculate_var(final_prices, confidence_level=0.05):
    cutoff = np.percentile(final_prices, confidence_level * 100)
    return cutoff
