from src.mls.assets.btc_forecast_assets import *
from src.mls.data_ops.btc_forecast_load_prices import *
from src.mls.etl.btc_forecast_etl import *
from src.mls.model.btc_forecast_ensemble_model import *
from src.mls.evaluation.btc_forecast_evaluation import *
from src.mls.prediction.btc_forecast_prediction import *

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import os
from datetime import datetime
from sklearn.preprocessing import minmax_scale
import yfinance as yf

def btc_forecast_multivariate_training_prediction():
  # Create AR model
  AR_model = get_AR_model(len(train_windows), WINDOW_SIZE_WEEK + 1)

  # Create NBEATS model
  NBEATS_model = get_NBEATS_model()

  # Create LNRNN model
  LNRNN_model = get_LNRNN_model()

  # Create LSTM model
  LSTM_model_obj = LSTMModel()
  LSTM_model = LSTM_model_obj.get_model()

  # Create Dense model
  dense_model_obj = DenseModel()
  dense_model = dense_model_obj.get_model()

  train_models = [NBEATS_model, LSTM_model, dense_model]
  # train_models = [NBEATS_model, LNRNN_model, LSTM_model, dense_model]
  # train_models = [LNRNN_model]

  # Obtain list of trained ensemble models
  ensemble_models = get_ensemble_models(models=train_models, train_data=train_dataset, test_data=test_dataset,
                                        num_iter=ENSEMBLE_NUM_ITER, num_epochs=ENSEMBLE_NUM_EPOCHS)

  # Generate model summaries
  get_ensemble_models_summary(models=train_models)

  # Plot the ensemble models
  # plot_model(AR_model)

  # plot_model(NBEATS_model)

  # plot_model(LNRNN_model)

  # plot_model(LSTM_model)

  # plot_model(dense_model)

  # Generate ensemble predictions
  ensemble_preds = make_ensemble_preds(ensemble_models=ensemble_models, input_data=test_dataset)

  # Evaluate ensemble model predictions
  ensemble_results = evaluate_preds(y_true=test_labels, y_pred=np.median(ensemble_preds, axis=0))
  print(ensemble_results)

  # Obtain the upper and lower bounds of the 95% confidence levels
  lower, upper = get_upper_lower_confidence(preds=ensemble_preds)

  # Get the median values of the ensemble preds
  ensemble_median = np.median(ensemble_preds, axis=0)

  # Plot the confidence interval
  # plot_confidence_interval(test_windows, test_labels, ensemble_median, lower=lower, upper=upper, offset=300)

  # Make forecasts into future of the price of bitcoin
  future_forecast = make_future_forecast(models=ensemble_models, values=dataset_full_labels_br, 
                                        into_future=INTO_FUTURE_2_WEEK, window_size=WINDOW_SIZE_WEEK + 1)

  # plot_future_forecast(df_btc_price=df_btc_price_closing, future_forecast=future_forecast)
  print(f"Future forecast: {future_forecast}")