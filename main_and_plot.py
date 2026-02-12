import matplotlib.pyplot as plt
from data_manager import get_stock_data, get_stats
from monte_carlo import simulate_gbm, calculate_var

# --- SETTINGS ---
TICKER = 'AAPL'       
START = '2020-01-01'
END = '2023-12-31'
SIMULATIONS = 1000    
DAYS_AHEAD = 252      

# --- EXECUTION ---
print(f"Fetching data for {TICKER}...")
prices, log_returns = get_stock_data(TICKER, START, END)

u, var = get_stats(log_returns)
drift = float(u)
sigma = float(log_returns.std())

print(f"Simulating {SIMULATIONS} paths...")
last_price = prices.iloc[-1]
paths = simulate_gbm(last_price, drift, sigma, SIMULATIONS, DAYS_AHEAD)

var_price = calculate_var(paths[-1], 0.05)

# --- PLOTTING ---
plt.figure(figsize=(10, 6))
plt.plot(paths[:, :100], color='black', alpha=0.67) # Plot 100 random paths
plt.axhline(var_price, color='red', linestyle='--', label=f'Worst Case (5%): ${var_price:.2f}')
plt.title(f'Monte Carlo Simulation: {TICKER}')
plt.legend()
plt.show()

print(f"Worst Case (VaR 95%): ${var_price:.2f}") 