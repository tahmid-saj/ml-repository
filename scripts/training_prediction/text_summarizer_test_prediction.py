from src.mls.etl.text_summarizer_etl import *
from src.mls.model.text_summarizer_tribid_embedding_model import *
from src.mls.evaluation.text_summarizer_evaluation import *

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import string

import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

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

# Positional embedding

print(train_df['line_number'].value_counts())

train_df.line_number.plot.hist()

# One-hot encoding of line numbers

train_line_numbers_one_hot = tf.one_hot(train_df['line_number'].to_numpy(), depth=15)
val_line_numbers_one_hot = tf.one_hot(val_df['line_number'].to_numpy(), depth=15)
test_line_numbers_one_hot = tf.one_hot(test_df['line_number'].to_numpy(), depth=15)

print(train_line_numbers_one_hot.shape, train_line_numbers_one_hot[:20])

print(train_df['total_lines'].value_counts())

train_df.total_lines.plot.hist()

# One-hot encoding of total lines
train_total_lines_one_hot = tf.one_hot(train_df["total_lines"].to_numpy(), depth=20)
val_total_lines_one_hot = tf.one_hot(val_df["total_lines"].to_numpy(), depth=20)
test_total_lines_one_hot = tf.one_hot(test_df["total_lines"].to_numpy(), depth=20)

print(train_total_lines_one_hot.shape, train_total_lines_one_hot[:10])

# Tribid embedding

tf_hub_embedding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        trainable=False,
                                        name="universal_sentence_encoder")

alphabet = string.ascii_lowercase + string.digits + string.punctuation
print(alphabet)

char_lens = [len(sentence) for sentence in train_sentences]
mean_char_len = np.mean(char_lens)
mean_char_len

output_seq_char_len = int(np.percentile(char_lens, 95))
output_seq_char_len

# Make function to split sentences into characters
def split_chars(text):
  return " ".join(list(text))

train_chars = [split_chars(sentence) for sentence in train_sentences]
val_chars = [split_chars(sentence) for sentence in val_sentences]
test_chars = [split_chars(sentence) for sentence in test_sentences]
print(train_chars[0])

NUM_CHAR_TOKENS = len(alphabet) + 2 # num characters in alphabet + space + OOV token
char_vectorizer = TextVectorization(max_tokens=NUM_CHAR_TOKENS,  
                                    output_sequence_length=output_seq_char_len,
                                    standardize="lower_and_strip_punctuation",
                                    name="char_vectorizer")

# Adapt character vectorizer to training characters
char_vectorizer.adapt(train_chars)

# Create char embedding layer
char_embed = layers.Embedding(input_dim=NUM_CHAR_TOKENS,
                              output_dim=25,
                              mask_zero=False,
                              name="char_embed")

# Token inputs
token_inputs = layers.Input(shape=[], dtype="string", name="token_inputs")
token_embeddings = tf_hub_embedding_layer(token_inputs)
token_outputs = layers.Dense(128, activation="relu")(token_embeddings)
token_model = tf.keras.Model(inputs=token_inputs, outputs=token_outputs)

# Character inputs
char_inputs = layers.Input(shape=(1, ), dtype="string", name="char_inputs")
char_vectors = char_vectorizer(char_inputs)
char_embeddings = char_embed(char_vectors)
char_bi_lstm = layers.Bidirectional(layers.LSTM(32))(char_embeddings)
char_model = tf.keras.Model(inputs=char_inputs, outputs=char_bi_lstm)

# Line numbers inputs
line_number_inputs = layers.Input(shape=(15, ), dtype=tf.int32, name="line_number_input")
x = layers.Dense(32, activation="relu")(line_number_inputs)
line_number_model = tf.keras.Model(inputs=line_number_inputs, outputs=x)

# Total lines inputs
total_lines_inputs = layers.Input(shape=(20, ), dtype=tf.int32, name="total_lines_input")
y = layers.Dense(32, activation="relu")(total_lines_inputs)
total_line_model = tf.keras.Model(inputs=total_lines_inputs, outputs=y)

combined_embeddings = layers.Concatenate(name="token_char_hybrid_embedding")([token_model.output, char_model.output])
z = layers.Dense(256, activation="relu")(combined_embeddings)
z = layers.Dropout(0.5)(z)

# Combining positional embeddings with combined token and character embeddings in a tribid embedding model
z = layers.Concatenate(name="token_char_positional_embedding")([line_number_model.output, total_line_model.output, z])
output_layer = layers.Dense(5, activation="softmax", name="output_layer")(z)
tribid_embedding_model = tf.keras.Model(inputs=[line_number_model.input, total_line_model.input, token_model.input, char_model.input], outputs=output_layer)

tribid_embedding_model.summary()

for layer in tribid_embedding_model.layers:
  print(layer, layer.trainable)

tribid_embedding_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                               optimizer=tf.keras.optimizers.Adam(),
                               metrics=["accuracy"])

train_pos_char_token_data = tf.data.Dataset.from_tensor_slices((train_line_numbers_one_hot, # line numbers
                                                                train_total_lines_one_hot, # total lines
                                                                train_sentences, # train tokens
                                                                train_chars)) # train chars
train_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(train_labels_one_hot) # train labels
train_pos_char_token_dataset = tf.data.Dataset.zip((train_pos_char_token_data, train_pos_char_token_labels)) # combine data and labels
train_pos_char_token_dataset = train_pos_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE) # turn into batches and prefetch appropriately

# Validation dataset
val_pos_char_token_data = tf.data.Dataset.from_tensor_slices((val_line_numbers_one_hot,
                                                              val_total_lines_one_hot,
                                                              val_sentences,
                                                              val_chars))
val_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(val_labels_one_hot)
val_pos_char_token_dataset = tf.data.Dataset.zip((val_pos_char_token_data, val_pos_char_token_labels))
val_pos_char_token_dataset = val_pos_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE) # turn into batches and prefetch appropriately

# Check input shapes
train_pos_char_token_dataset, val_pos_char_token_dataset

history_tribid_embedding_model = tribid_embedding_model.fit(train_pos_char_token_dataset,
                                                            steps_per_epoch=int(0.1 * len(train_pos_char_token_dataset)),
                                                            epochs=3,
                                                            validation_data=val_pos_char_token_dataset,
                                                            validation_steps=int(0.1 * len(val_pos_char_token_dataset)))

# Model predictions

tribid_embedding_model_probs = tribid_embedding_model.predict(val_pos_char_token_dataset, verbose=1)
tribid_embedding_model_probs

tribid_embedding_model_preds = tf.argmax(tribid_embedding_model_probs, axis=1)
tribid_embedding_model_preds

tribid_embedding_model_results = calculate_results(y_true=val_labels_encoded, y_pred=tribid_embedding_model_preds)
tribid_embedding_model_results

tribid_embedding_model.save("./models/tribid/tribid_embedding_model", save_format='tf')

LOADED_MODEL_PATH = "./models/tribid/tribid_embedding_model"

loaded_tribid_embedding_model = tf.keras.models.load_model(LOADED_MODEL_PATH)

# Testing and evaluating predictions

loaded_pred_probs = loaded_tribid_embedding_model.predict(val_pos_char_token_dataset, verbose=1)
loaded_preds = tf.argmax(loaded_pred_probs, axis=1)
loaded_preds[:10]

loaded_model_results = calculate_results(val_labels_encoded, loaded_preds)
loaded_model_results

loaded_tribid_embedding_model.summary()

# Create test dataset batch and prefetched
test_pos_char_token_data = tf.data.Dataset.from_tensor_slices((test_line_numbers_one_hot,
                                                               test_total_lines_one_hot,
                                                               test_sentences,
                                                               test_chars))
test_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(test_labels_one_hot)
test_pos_char_token_dataset = tf.data.Dataset.zip((test_pos_char_token_data, test_pos_char_token_labels))
test_pos_char_token_dataset = test_pos_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Check shapes
test_pos_char_token_dataset

test_pred_probs = loaded_tribid_embedding_model.predict(test_pos_char_token_dataset,
                                       verbose=1)
test_preds = tf.argmax(test_pred_probs, axis=1)
test_preds[:10]

loaded_model_test_results = calculate_results(y_true=test_labels_encoded,
                                              y_pred=test_preds)
loaded_model_test_results

# Model evaluation on wrong predictions
test_pred_classes = [label_encoder.classes_[pred] for pred in test_preds]
test_pred_classes

test_df["prediction"] = test_pred_classes
test_df["pred_prob"] = tf.reduce_max(test_pred_probs, axis=1).numpy()
test_df["correct"] = test_df["prediction"] == test_df["target"]
test_df.head(20)

top_100_wrong = test_df[test_df["correct"] == False].sort_values("pred_prob", ascending=False)[:100]
top_100_wrong