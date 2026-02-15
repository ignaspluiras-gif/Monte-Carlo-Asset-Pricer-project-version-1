import matplotlib.pyplot as plt
from data_import import get_stock_data, get_stats
from math_part_for_GBM import simulate_gbm, calculate_var
import numpy as np
import scipy.stats as stats
import pylab


# Stock info
TICKER = 'AAPL'       
START = '2020-01-01'
END = '2023-12-31'
SIMULATIONS = 1000    
DAYS_AHEAD = 365      

# code run
print(f"Fetching data for {TICKER}...")
prices, log_returns = get_stock_data(TICKER, START, END)

u, var = get_stats(log_returns)
drift = float(u)
sigma = float(log_returns.std())

print(f"Simulating {SIMULATIONS} paths...")
last_price = prices.iloc[-1]
paths = simulate_gbm(last_price, drift, sigma, SIMULATIONS, DAYS_AHEAD)

var_price = calculate_var(paths[-1], 0.05)

# data plotting

plt.figure(figsize=(10, 6))
plt.plot(paths[:, :100], alpha=0.67) # Plot 100 random paths
plt.axhline(var_price, color='black', linestyle='--', label=f'Worst Case (5%): ${var_price:.2f}')

mean_path = paths.mean(axis=1)
plt.plot(mean_path, color='blue', linewidth=1, linestyle='--', label='Expected Price (Mean)')
worst_case_curve = np.percentile(paths, 5, axis=1)
plt.plot(worst_case_curve, color='black', linestyle='-', linewidth=2, label='Worst Case Scenario daily (5%)')
plt.title(f'Monte Carlo Simulation: {TICKER}')
plt.legend()


plt.figure(figsize=(10, 6))

plt.hist(paths[-1], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f'Distribution of FinalPrices for {TICKER}')
plt.xlabel('Price ($)')
plt.ylabel('Count')
worst_5 = np.percentile(paths[-1], 5)
plt.axvline(worst_5, color='green', linewidth=2, label=f'Worst case (5%): ${worst_5:.2f}')

expected_price = paths[-1].mean()
plt.axvline(expected_price, color='red', linewidth=2, label=f'Expected: ${expected_price:.2f}')
plt.legend()


plt.figure(figsize=(10, 6))
stats.probplot(log_returns.dropna().values.flatten(), dist='norm', plot=plt)
plt.title("Q-Q Plot of Log Returns")


plt.show()

print(f"Worst Case (VaR 95%): ${var_price:.2f}") 