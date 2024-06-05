from src.mls.model.text_analyzer_ensemble_model import *

import pandas as pd
import numpy as np
import tensorflow as tf

import logging
import os
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split

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

# Tokenization
mean_word_length = int(mean([len(w.split()) for w in train_text]))

tokenizer = layers.TextVectorization(max_tokens=VOCAB_LENGTH, standardize="lower_and_strip_punctuation", split="whitespace", ngrams=None, output_mode="int", output_sequence_length=mean_word_length)
tokenizer.adapt(train_text)

vocab = tokenizer.get_vocabulary()
frequent_words = vocab[:10]
unfrequent_words = vocab[-10:]
print(f"Frequent words: {frequent_words}\n Unfrequent words: {unfrequent_words}")

dense_model_obj = DenseModel(tokenizer=tokenizer)
dense_model = dense_model_obj.get_model()

lstm_model_obj = LSTMModel(tokenizer=tokenizer, vocab=vocab, mean_word_length=mean_word_length)
lstm_model = lstm_model_obj.get_model()

bidir_lstm_model_obj = BidirectionalLSTMModel(tokenizer=tokenizer, vocab=vocab, mean_word_length=mean_word_length)
bidir_lstm_model = bidir_lstm_model_obj.get_model()

ensemble_models = [dense_model, lstm_model, bidir_lstm_model]

ensemble_models = get_ensemble_models(models=ensemble_models, 
                                      train_data=train_text, 
                                      train_label=train_label, 
                                      val_data=(val_text, val_label), 
                                      num_iter=1, num_epochs=100)

ensemble_models[0].save("./models/ensemble_models/dense_model", save_format='tf')
ensemble_models[1].save("./models/ensemble_models/lstm_model", save_format='tf')
ensemble_models[2].save("./models/ensemble_models/bidir_lstm_model", save_format='tf')