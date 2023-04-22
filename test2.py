import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pandas.tseries.offsets import MonthEnd

def load_etf_data(file_name: str) -> pd.DataFrame:
    return pd.read_csv(file_name, parse_dates=['Date'], index_col='Date')

def calculate_daily_returns(price_data):
    return price_data['Adj Close'].pct_change().dropna()

START_DATE = datetime.datetime(2020, 1, 1)
END_DATE = datetime.datetime(2022, 12, 31)

def get_rebalance_dates(start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DatetimeIndex:
    return pd.date_range(start_date, end_date, freq='BM')

etf_data = {etf: load_etf_data(file_name=f'{etf}.csv') for etf in ['SPY', 'VGK', 'EEM', 'GLD', 'TLT']}
daily_returns = {etf: calculate_daily_returns(data) for etf, data in etf_data.items()}
daily_returns = pd.DataFrame(daily_returns)[START_DATE:]

rebalance_dates = get_rebalance_dates(START_DATE, END_DATE)
allocations_df = pd.DataFrame(columns=['SPY', 'VGK', 'EEM', 'GLD', 'TLT'])

for rebalance_date in rebalance_dates:
    past_month_returns = daily_returns.loc[rebalance_date - pd.DateOffset(months=1):rebalance_date - pd.DateOffset(days=1)]
    risks = past_month_returns.std()
    inv_vol_allocations = 1 / risks
    normalized_allocations = (inv_vol_allocations / inv_vol_allocations.sum()) * 100
    allocations_df = pd.concat([allocations_df, normalized_allocations.rename(rebalance_date).to_frame().T])

portfolio_values = [100]
portfolio_daily_returns = []

for i, date in enumerate(rebalance_dates[:-1]):
    period_returns = daily_returns.loc[date:rebalance_dates[i + 1] - pd.DateOffset(days=1)].multiply(allocations_df.iloc[i] / 100, axis=1).sum(axis=1)
    period_daily_returns = period_returns.pct_change().dropna()
    portfolio_daily_returns.extend(period_daily_returns)
    portfolio_values.append(portfolio_values[-1] * (1 + period_returns).prod())

portfolio_daily_returns = np.array(portfolio_daily_returns)
years = (END_DATE - START_DATE).days / 365.25
CAGR = (portfolio_values[-1] / portfolio_values[0]) ** (1 / years) - 1

annualized_return = np.mean(portfolio_daily_returns) * 252
annualized_volatility = np.std(portfolio_daily_returns) * np.sqrt(252)
sharpe_ratio = annualized_return / annualized_volatility

plt.plot([START_DATE] + list(rebalance_dates[:-1]), portfolio_values)
plt.xlabel('Time')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value over Time')
plt.savefig('portfolio_value_plot.png')

allocations_df.to_csv('allocations.csv', index_label='Date')

print(f"Compound Annual Growth Rate (CAGR): {CAGR:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")