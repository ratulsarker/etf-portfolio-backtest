import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the ETF data
etf_list = ['SPY', 'VGK', 'EEM', 'GLD', 'TLT']
etf_data = {}

for etf in etf_list:
    etf_data[etf] = pd.read_csv(f"{etf}.csv", index_col='Date', parse_dates=True)

# Merge the ETF data into a single DataFrame
adj_close = pd.concat([etf_data[etf]['Adj Close'] for etf in etf_list], axis=1, join='inner')
adj_close.columns = etf_list

# Calculate daily returns
daily_returns = adj_close.pct_change().dropna()

# Determine monthly rebalancing dates
rebalance_dates = pd.date_range(adj_close.index[0], adj_close.index[-1], freq='M')
rebalance_dates = [date for date in rebalance_dates if date in adj_close.index]

# Initialize variables
allocation = None
rebalance_history = None
total_portfolio_returns = []

# Iterate through the rebalancing dates
for i, rebalance_date in enumerate(rebalance_dates):
    # Calculate the volatility for each ETF
    start_date = adj_close.index[0] if i == 0 else rebalance_dates[i - 1]
    volatility = daily_returns.loc[start_date:rebalance_date].std()

    # Calculate weights using the Inverse Volatility method
    weights = 1 / volatility
    weights = weights / np.sum(weights)

    # Store rebalancing information in a DataFrame
    rebalance_entry = pd.DataFrame({
        'Date': [rebalance_date],
        'SPY': [weights[0]],
        'VGK': [weights[1]],
        'EEM': [weights[2]],
        'GLD': [weights[3]],
        'TLT': [weights[4]]
    })

    # Append the rebalance_entry DataFrame to the rebalance_history DataFrame
    if rebalance_history is None:
        rebalance_history = rebalance_entry
    else:
        rebalance_history = pd.concat([rebalance_history, rebalance_entry], ignore_index=True)

    # Update the allocations for the next month
    allocation = adj_close.loc[rebalance_date] * weights

    # Calculate the daily returns of the portfolio after rebalancing
    portfolio_daily_returns = daily_returns.loc[rebalance_date:].multiply(allocation).sum(axis=1)
    total_portfolio_returns.extend(portfolio_daily_returns[:min(len(portfolio_daily_returns), 30)])

# Calculate portfolio performance metrics
total_portfolio_returns = np.array(total_portfolio_returns)
sharpe_ratio = total_portfolio_returns.mean() / total_portfolio_returns.std() * np.sqrt(252)
cagr = (adj_close.iloc[-1] / adj_close.iloc[0]) ** (252 / len(adj_close)) - 1

# Print the performance metrics
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"CAGR: {cagr.mean():.4f}")

# Save the rebalancing history to a CSV file
rebalance_history.to_csv('rebalance_history.csv', index=False)


# Plot the portfolio value over time
initial_value = 100
portfolio_value = initial_value * (1 + np.array(total_portfolio_returns)).cumprod()
portfolio_value = portfolio_value[:len(daily_returns.index[1:])]
plt.plot(pd.to_datetime(daily_returns.index[1:]), portfolio_value)
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value Over Time (Initial Value = 100)')
plt.grid()
plt.show()
