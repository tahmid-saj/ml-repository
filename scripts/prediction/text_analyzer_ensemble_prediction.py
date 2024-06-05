from src.mls.model.text_analyzer_ensemble_model import *
from src.mls.evaluation.text_analyzer_evaluation import *

import pandas as pd
import numpy as np
import tensorflow as tf

import logging
import os
import random
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

dense_model_obj = DenseModel(tokenizer=tokenizer, vocab=vocab, mean_word_length=mean_word_length)
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
                                      num_iter=1, num_epochs=4)

lstm_model_pred_probs = ensemble_models[1].predict(val_text)
bidir_lstm_model_pred_probs = ensemble_models[2].predict(val_text)

comb_pred_probs = tf.squeeze(lstm_model_pred_probs, axis=1) + tf.squeeze(bidir_lstm_model_pred_probs, axis=1)
comb_pred_probs = tf.round(comb_pred_probs / 2)
print(comb_pred_probs, comb_pred_probs.shape)

ensemble_model_metrics = evaluate_preds(y_true=val_label, y_pred=comb_pred_probs)
print(f"Ensemble model metrics: {ensemble_model_metrics}")

test_text = test_df["text"].to_numpy()
test_samples = random.sample(list(test_text), 10)

for test_sample in test_samples:
    pred = make_ensemble_preds(ensemble_models, test_samples)
    print(f"Pred: {pred}")
    print(f"Text: \n{test_sample}\n\n")

print(f"Ensemble model 0: {ensemble_models[0]}")

ensemble_models[0].save("./models/ensemble_models/dense_model", save_format='tf')
ensemble_models[1].save("./models/ensemble_models/lstm_model", save_format='tf')
ensemble_models[2].save("./models/ensemble_models/bidir_lstm_model", save_format='tf')

loaded_ensemble_dense_model = tf.keras.models.load_model("./models/ensemble_models/dense_model")
loaded_ensemble_lstm_model = tf.keras.models.load_model("./models/ensemble_models/lstm_model")
loaded_ensemble_bidir_lstm_model = tf.keras.models.load_model("./models/ensemble_models/bidir_lstm_model")

print(loaded_ensemble_dense_model.summary(), "\n")
print(loaded_ensemble_lstm_model.summary(), "\n")
print(loaded_ensemble_bidir_lstm_model.summary(), "\n")

val_text = "This is going to be a friendly sentence"
val_label = 0

loaded_ensemble_models = [loaded_ensemble_dense_model, loaded_ensemble_lstm_model, loaded_ensemble_bidir_lstm_model]

loaded_ensemble_models.evaluate_preds(val_text, val_label)