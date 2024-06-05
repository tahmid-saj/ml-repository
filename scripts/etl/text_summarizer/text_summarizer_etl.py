from src.mls.etl.text_summarizer_etl import *

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import string

DATA_DIR = "./data"

filenames = [DATA_DIR + filename for filename in os.listdir(DATA_DIR)]
print(filenames)

train_lines = get_lines(DATA_DIR + "/train.txt")
print(train_lines[:20])

train_samples = preprocess_text_with_line_numbers(DATA_DIR + "/train.txt")
val_samples = preprocess_text_with_line_numbers(DATA_DIR + "/dev.txt")
test_samples = preprocess_text_with_line_numbers(DATA_DIR + "/test.txt")

print([(i, "\n") for i in train_samples[:1]])

train_df = pd.DataFrame(train_samples)
val_df = pd.DataFrame(val_samples)
test_df = pd.DataFrame(test_samples)
print(train_df.head())

print(train_df.target.value_counts())

train_df.total_lines.plot.hist()

train_sentences = train_df["text"].tolist()
val_sentences = val_df["text"].tolist()
test_sentences = test_df["text"].tolist()

print(train_sentences[:10])

# Encoding with one-hot encoding

one_hot_encoder = OneHotEncoder(sparse=False)
train_labels_one_hot = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
val_labels_one_hot = one_hot_encoder.transform(val_df["target"].to_numpy().reshape(-1, 1))
test_labels_one_hot = one_hot_encoder.transform(test_df["target"].to_numpy().reshape(-1, 1))

print(train_labels_one_hot)

# Encoding with label encoding

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_df["target"].to_numpy())
val_labels_encoded = label_encoder.transform(val_df["target"].to_numpy())
test_labels_encoded = label_encoder.transform(test_df["target"].to_numpy())

print(train_labels_encoded)

num_classes = len(label_encoder.classes_)
class_names = label_encoder.classes_