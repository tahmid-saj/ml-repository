from src.mls.assets.btc_forecast_assets import *
from src.mls.model.btc_forecast_ensemble_model import make_ensemble_preds

import numpy as np
import pandas as pd
import tensorflow as tf
import os
from datetime import datetime

def get_upper_lower_confidence(preds):
  """
  Returns the upper and lower bounds of the 95% confidence level using the
  following logic:

  95% confidence upper, lower bounds = mean of preds +/- (standard deviation
  of preds * 1.96)

  Parameters
  ----------
  preds: Predictions of ensemble of models.
  """
  # 1. Take the predictions of multiple randomly initialized deep learning neural networks using the preds parameter
  # 2. Measure the standard deviation of the predictions
  std = tf.math.reduce_std(preds, axis=0)

  # 3. Multiply the standard deviation by 1.96
  interval = 1.96 * std

  # 4. Get the prediction interval's upper and lower bounds
  preds_mean = tf.reduce_mean(preds, axis=0)
  lower, upper = preds_mean - interval, preds_mean + interval

  return lower, upper

# Function to make predictions into future (can include retraining the model with the predicted data appended, everytime the model makes a prediction)
# 1. Create function to make predictions into the future
def make_future_forecast(models, values, into_future, window_size=WINDOW_SIZE_WEEK) -> list:
  """
  Makes future forecasts into_future steps after values ends.

  Returns future forecasts as list of floats.

  Parameters
  ----------
  values: BTC price labels after windowing is in effect.
  models: List of ensemble models to predict forecast using.
  into_future: How many days into future to make forecast for.
  window_size: Window size.
  """
  # 2. Make an empty list for future forecasts/prepare data to forecast on
  future_forecast = []
  last_window = values[-window_size:] # Only want preds from the last window (this will get updated after every prediction)

  # 3. Make into_future number of predictions, altering the data which gets predicted on each time
  for day in range(into_future):
    # Predict on last window then append it again and again (model will start to make forecasts on its own forecasts)
    ensemble_preds = make_ensemble_preds(ensemble_models=models, input_data=tf.expand_dims(last_window, axis=0))
    future_pred = np.median(ensemble_preds, axis=0)
    print(f"Predicting on: \n {last_window} -> Prediction: {tf.squeeze(future_pred).numpy()}\n")

    # Append predictions on future_forecast
    future_forecast.append(tf.squeeze(future_pred).numpy())

    # Update last window with the new pred and get window_size of most recent preds (model was trained on window_size windows)
    last_window = np.append(last_window, future_pred)[-window_size:]
  
  return future_forecast