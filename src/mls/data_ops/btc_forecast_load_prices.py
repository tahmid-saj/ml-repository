from src.mls.assets.btc_forecast_assets import *

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf

import os
from datetime import datetime

# Live training dataset loading

# Function to generate historical training dataset
def get_historical_btc_data(period_years=10, interval_days=1):
  """
  Returns historical BTC data using a period in years and interval in days
  Parameters
  ----------
  period_years: Period in years
  interval_days: Interval in days
  """
  df_btc_price = yf.download(tickers='BTC-USD', period=f'{period_years}y', interval=f'{interval_days}d')
  print(f"Head of BTC prices: \n{df_btc_price.head()}")
  print(f"BTC prices info: \n{df_btc_price.info()}")

  # Only looking for the closing price for each record
  df_btc_price_closing = pd.DataFrame(df_btc_price['Close']).rename(columns={'Close': 'Price'})
  print(df_btc_price_closing.head())

  return df_btc_price_closing

# Function to generate future dates for prediction data
def get_future_dates(start_date, into_future, offset=1):
  """
  Returns array of datetime values ranging from start_date to start_date +
  into_future (horizon).

  Parameters
  ----------
  start_date: date to start range (np.datetime64).
  into_future: number of days to add onto start date for range (int).
  offset: number of days to offset start_date by (default = 1).
  """
  start_date = start_date + np.timedelta64(offset, "D")
  end_date = start_date + np.timedelta64(into_future, "D")

  return np.arange(start_date, end_date, dtype="datetime64[D]")