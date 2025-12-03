import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Data Collection
# Define a diversified portfolio to analyze asset relationships:
# - AAPL: Technology (High Cap)
# - TSLA: Automotive/Tech (High Volatility)
# - GLD: Gold (Safe Haven / Crisis Hedge)
# - ^GSPC: S&P 500 (Market Benchmark)
tickers = ['AAPL', 'TSLA', 'GLD', '^GSPC']

print(f"Fetching data for: {tickers}...")
# Note: Using 'Close' prices adjusted for splits/dividends by default in yfinance
data = yf.download(tickers, start='2024-01-01')['Close']

# 2. Calculate Daily Returns
# Correlation is calculated based on returns (percentage change), not raw prices.
returns = data.pct_change().dropna()

# 3. Compute Correlation Matrix
# The Pearson correlation coefficient measures the linear relationship between assets.
# Range: -1 (Perfect Negative Correlation) to +1 (Perfect Positive Correlation).
correlation_matrix = returns.corr()

print("\n--- Correlation Matrix ---")
print(correlation_matrix)

# 4. Visualization: Correlation Heatmap
# Visualizing the matrix to identify diversification opportunities.
# Assets with low or negative correlation to the market (S&P 500) provide better hedging.
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, 
            annot=True,           # Display correlation coefficients
            cmap='coolwarm',      # Color map: Blue (Low) to Red (High)
            vmin=-1, vmax=1,      # Scale limits
            linewidths=0.5,       # Grid lines
            fmt=".2f")            # Format decimals

plt.title('Multi-Asset Portfolio Correlation Matrix (Diversification Analysis)')
plt.show()
