import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# 1. Data Ingestion
# Download historical closing prices for Apple (AAPL) and the S&P 500 Index (^GSPC).
# Note: yfinance now returns adjusted close data in the 'Close' column by default.
tickers = ['AAPL', '^GSPC']
print(f"Downloading data for: {tickers}...")
data = yf.download(tickers, start='2024-01-01', end='2024-11-20')['Close']

# 2. Calculate Logarithmic Returns
# Formula: r = ln(P_t / P_{t-1})
log_returns = np.log(data / data.shift(1))

# 3. Calculate Annualized Volatility (Risk Assessment)
# Standard Deviation measures the dispersion of returns.
# We multiply daily volatility by sqrt(252) to annualize it (252 trading days/year).
daily_std = log_returns.std()
annualized_volatility = daily_std * np.sqrt(252)

print("\n--- Annualized Volatility (Risk) ---")
print(annualized_volatility)

# 4. Visualization: Return Distribution
# Plotting the histogram to visually assess the distribution of returns (Normality check).
plt.figure(figsize=(10, 6))
log_returns['AAPL'].hist(bins=50, alpha=0.6, color='#1f77b4', edgecolor='black')

plt.title("Distribution of Daily Log Returns - Apple (AAPL)")
plt.xlabel("Log Returns")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.5, linestyle='--')

plt.show()
