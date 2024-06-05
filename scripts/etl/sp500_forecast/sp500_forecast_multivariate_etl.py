from src.mls.assets.sp500_forecast_assets import *
from src.mls.data_ops.sp500_load_prices import *
from src.mls.etl.sp500_forecast_etl import *

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
from datetime import datetime
from sklearn.preprocessing import minmax_scale
import yfinance as yf

df_sp_500_price = get_historical_sp500_data()

# Including volume to the full stock closing price dataset
df_sp_500_price_volume = include_price_volume(df_sp_500_price=df_sp_500_price)

print(f"Head of S&P 500 price and volume dataset: \n {df_sp_500_price_volume.head()}")
print(f"Tail of S&P 500 price and volume dataset: \n {df_sp_500_price_volume.tail()}")
# Plotting the S&P 500 price and volume over time
# plot_volume_price_data(df_sp_500_price_volume=df_sp_500_price_volume)

# Obtain the full windows and labels with volume included
dataset_full_windows_vol, dataset_full_labels_vol = make_windows_labels_multivariate(df_sp_500_price_volume)

# Obtain the train/test sets of the full windows and labels with block reward included
train_windows, test_windows, train_labels, test_labels = make_train_test_splits(windows=dataset_full_windows_vol, labels=dataset_full_labels_vol)

# Obtain the train and test datasets
train_dataset, test_dataset = gen_train_test_datasets(X_train=train_windows, y_train=train_labels,
                                                      X_test=test_windows, y_test=test_labels)
print(f"Train dataset: {train_dataset}", f"Test dataset: {test_dataset}")