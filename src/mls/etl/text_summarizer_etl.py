import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import string

def get_lines(filename):
  """
  Reads filename and returns the lines of text as a list.
  """
  with open(filename, "r") as f:
    return f.readlines()

def preprocess_text_with_line_numbers(filename):
  """
  Returns a list of dictionaries of abstract line data.
  """
  input_lines = get_lines(filename)
  abstract_lines = ""
  abstract_samples = []

  for line in input_lines:
    if line.startswith("###"):
      abstract_id = line
      abstract_lines = ""
    elif line.isspace():
      abstract_line_split = abstract_lines.splitlines()

      for abstract_line_number, abstract_line in enumerate(abstract_line_split):
        line_data = {}
        target_text_split = abstract_line.split("\t")

        line_data["target"] = target_text_split[0]
        line_data["text"] = target_text_split[1].lower()
        line_data["line_number"] = abstract_line_number
        line_data["total_lines"] = len(abstract_line_split)
        abstract_samples.append(line_data)
    else:
      abstract_lines += line

  return abstract_samples