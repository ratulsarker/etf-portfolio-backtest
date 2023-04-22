import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt


etf_files = {
    'SPY': 'SPY.csv',
    'VGK': 'VGK.csv',
    'EEM': 'EEM.csv',
    'GLD': 'GLD.csv',
    'TLT': 'TLT.csv'
}

etf_data = {}
for etf, file in etf_files.items():
    etf_data[etf] = pd.read_csv(file, index_col='Date', parse_dates=True)

combined_data = pd.DataFrame({etf: df['Adj Close'] for etf, df in etf_data.items()})

daily_returns = combined_data.pct_change().dropna()

def inverse_volatility_weights(returns, lookback_period):
    volatilities = returns.rolling(window=lookback_period).std().iloc[-1]
    inv_volatilities = 1 / volatilities
    weights = inv_volatilities / np.sum(inv_volatilities)
    return weights


start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2022, 12, 31)
month_ends = pd.date_range(start_date, end_date, freq='BM')



initial_portfolio_value = 100
current_weights = None
rebalance_history = pd.DataFrame(columns=['Date', 'SPY', 'VGK', 'EEM', 'GLD', 'TLT'])


def rebalance_portfolio(date, daily_returns, lookback_period, current_weights, portfolio_value):
    weights = inverse_volatility_weights(daily_returns.loc[:date].tail(lookback_period), lookback_period)
    new_portfolio_value = portfolio_value * weights
    return new_portfolio_value, weights

portfolio_value = initial_portfolio_value

for date in month_ends:
    if date in daily_returns.index:
        # Perform rebalancing
        portfolio_value, current_weights = rebalance_portfolio(date, daily_returns, 21, current_weights, portfolio_value)
        
        # Record the rebalancing details
        rebalance_history = rebalance_history.append(
            {'Date': date, 'SPY': current_weights['SPY'], 'VGK': current_weights['VGK'],
             'EEM': current_weights['EEM'], 'GLD': current_weights['GLD'], 'TLT': current_weights['TLT']}, ignore_index=True)

        # Update the portfolio value for the next period
        next_day_returns = daily_returns.loc[date:].head(1)
        portfolio_value *= (1 + next_day_returns).values


# Calculate the total_portfolio_returns
total_portfolio_returns = (1 + daily_returns).cumprod().dot(rebalance_history.set_index('Date').reindex(daily_returns.index, method='ffill').T)
total_portfolio_returns = total_portfolio_returns.pct_change().dropna()


# Calculate the excess daily returns
excess_daily_returns = total_portfolio_returns

# Calculate the annualized average excess return and annualized standard deviation
average_excess_return = excess_daily_returns.mean() * 252
standard_deviation = excess_daily_returns.std() * np.sqrt(252)

# Calculate the annualized Sharpe Ratio
sharpe_ratio = average_excess_return / standard_deviation


# Calculate the total return of the portfolio
total_return = (portfolio_value[-1] / initial_portfolio_value) - 1

# Calculate the number of years
years = (end_date - start_date).days / 365

# Calculate the CAGR
cagr = (1 + total_return) ** (1 / years) - 1


cumulative_portfolio_value = (1 + total_portfolio_returns).cumprod() * initial_portfolio_value

cumulative_portfolio_value = (1 + total_portfolio_returns).cumprod() * initial_portfolio_value

def plot_portfolio_value(cumulative_portfolio_value):
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_portfolio_value.index, cumulative_portfolio_value, label='Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.legend()
    plt.grid()
    plt.show()

plot_portfolio_value(cumulative_portfolio_value)

rebalance_history.to_csv('rebalance_history.csv', index=False)
