import pandas as pd
import numpy as np
import tensorflow as tf
import random
import zipfile

import logging
import os
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split

import re
import string

BATCH_SIZE = 32
VOCAB_LENGTH = 15000

train_df = pd.read_csv("./data/train/train_dataset.csv")[["text", "target"]]
test_df = pd.read_csv("./data/test/test_dataset.csv")[["text"]]
print(train_df.head())

print(test_df.head())

print(f"Value counts of test dataset: {train_df.target.value_counts()}")

train_text, val_text, train_label, val_label = train_test_split(train_df["text"].to_numpy(), 
                                                                train_df["target"].to_numpy(), 
                                                                test_size=0.1, 
                                                                random_state=42)

print(train_text.shape, val_text.shape, train_label.shape, val_label.shape)

print(train_text[:5])

print(train_label[:5])

