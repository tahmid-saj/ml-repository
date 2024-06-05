from src.mls.model.text_analyzer_ensemble_model import *

import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import os
import random
from datetime import datetime, timedelta
from tensorflow.keras import layers
from statistics import mean

from sklearn.metrics import accuracy_score, precision_recall_curve, precision_recall_fscore_support

def make_ensemble_preds(ensemble_models, input_data):
  """
  Returns predictions of ensemble models.

  Parameters
  ----------
  ensemble_models: Trained ensemble of models.
  input_data: Data to be predicted with.
  """
  ensemble_preds = []
  
  for model in ensemble_models:
    preds = model.predict(input_data) # Make predictions with current ensemble model
    ensemble_preds.append(preds)
  
  return tf.constant(tf.squeeze(ensemble_preds))

def get_ensemble_models_summary(models):
  """
  Generates model summaries of ensemble models.

  Parameters
  ----------
  models: Ensemble of models.
  """
  for model in models:
    print(model.summary())

def evaluate_preds(y_true, y_pred):
  accuracy = accuracy_score(y_true, y_pred) * 100
  precision, recall, f1_score, info = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  
  return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score, "info": info}