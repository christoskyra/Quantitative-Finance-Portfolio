import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. Data Collection
# Downloading Adjusted Close prices for Tesla (TSLA) and the Market Benchmark (S&P 500).
tickers = ['TSLA', '^GSPC']
print(f"Fetching data for {tickers}...")
data = yf.download(tickers, start='2023-01-01', end='2024-11-20')['Close']

# 2. Calculate Daily Returns (Percentage Change)
returns = data.pct_change().dropna()

# 3. Define Variables for Regression
# Independent Variable (X): Market Returns (S&P 500)
# Dependent Variable (Y): Stock Returns (Tesla)
x = returns['^GSPC']
y = returns['TSLA']

# 4. Linear Regression (CAPM Implementation)
# Equation: R_stock = alpha + beta * R_market + error
# Beta represents the sensitivity of the stock to market movements (Systematic Risk).
beta, alpha, r_value, p_value, std_err = stats.linregress(x, y)

print(f"\n--- CAPM Analysis Results (TSLA vs S&P 500) ---")
print(f"Beta:      {beta:.4f} (Market Sensitivity)")
print(f"Alpha:     {alpha:.5f} (Excess Return)")
print(f"R-squared: {r_value**2:.4f} (Model Fit)")

# 5. Visualization: Regression Line vs. Data Points
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5, label='Daily Returns', color='#1f77b4')

# Plot the regression line (The "Best Fit" line)
plt.plot(x, alpha + beta * x, color='red', linewidth=2, label=f'Regression Line (Beta={beta:.2f})')

plt.title('Capital Asset Pricing Model (CAPM): Tesla vs S&P 500')
plt.xlabel('Market Returns (S&P 500)')
plt.ylabel('Stock Returns (Tesla)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()
