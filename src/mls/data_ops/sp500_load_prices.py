from src.mls.assets.sp500_forecast_assets import *

import pandas as pd
import numpy as np
from datetime import datetime
import os

import yfinance as yf

# Live training dataset loading

# Function to generate historical training dataset
def get_historical_sp500_data(period_years=10, interval_days=1):
  """
  Returns historical S&P 500 data using a period in years and interval in days
  Parameters
  ----------
  period_years: Period in years
  interval_days: Interval in days
  """
  df_sp_500_price = yf.download(tickers='^GSPC', period=f'{period_years}y', interval=f'{interval_days}d')
  print(f"Head of S&P 500 prices: \n{df_sp_500_price.head()}")
  print(f"S&P 500 prices info: \n{df_sp_500_price.info()}")

  # We are only looking for the closing price for each record
  df_sp_500_price_closing = pd.DataFrame(df_sp_500_price['Close']).rename(columns={'Close': 'Price'})
  print(df_sp_500_price_closing.head())
  
  return df_sp_500_price_closing

# Fundtion to generate future dates for prediction data
def get_future_dates(start_date, into_future, offset=1):
  """
  Returns array of datetime values ranging from start_date to start_date +
  into_future (horizon).

  Parameters
  ----------
  start_date: Date to start range (np.datetime64).
  into_future: Number of days to add onto start date for range (int).
  offset: Number of days to offset start_date by (default = 1).
  """
  start_date = start_date + np.timedelta64(offset, "D")
  end_date = start_date + np.timedelta64(into_future, "D")

  return np.arange(start_date, end_date, dtype="datetime64[D]")