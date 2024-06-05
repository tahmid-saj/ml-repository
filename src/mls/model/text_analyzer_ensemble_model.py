from src.mls.evaluation.text_analyzer_evaluation import *

import pandas as pd
import numpy as np
import tensorflow as tf

import logging
import os
import random
from datetime import datetime, timedelta
from tensorflow.keras import layers
from statistics import mean

# import torch
# import torch.nn as nn
# import torchtext.data as ttd
# from torchtext.vocab import GloVe
# import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

IDX_CUS = 1
WORD_IDX_CUS = {'PAD': 0}

IDX_TECH = 1
WORD_IDX_TECH = {'PAD': 0}

BATCH_SIZE = 32
VOCAB_LENGTH = 15000

# TODO: predict with pytorch models

# # Tokenization
# mean_word_length = int(mean([len(w.split()) for w in train_text]))

# tokenizer = layers.TextVectorization(max_tokens=VOCAB_LENGTH, standardize="lower_and_strip_punctuation", split="whitespace", ngrams=None, output_mode="int", output_sequence_length=mean_word_length)
# tokenizer.adapt(train_text)

# vocab = tokenizer.get_vocabulary()
# frequent_words = vocab[:10]
# unfrequent_words = vocab[-10:]
# print(f"Frequent words: {frequent_words}\n Unfrequent words: {unfrequent_words}")

class DenseModel():
    def __init__(self, tokenizer, vocab, mean_word_length, n_units=1, activation="sigmoid"):
        self.tokenizer = tokenizer
        self.n_units = n_units
        self.activation = activation
        
        self.vocab = vocab
        self.mean_word_length = mean_word_length
    
    def get_model(self, model_name="dense_model"):
        dense_inputs = layers.Input(shape=(1, ), dtype="string")
        
        x = self.tokenizer(dense_inputs)
        dense_embedding = layers.Embedding(input_dim=len(self.vocab), 
                                           output_dim=128, 
                                           embeddings_initializer="uniform", 
                                           input_length=self.mean_word_length, 
                                           name="embedding")
        x = dense_embedding(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        dense_outputs = layers.Dense(units=self.n_units, activation=self.activation)(x)
        dense_model = tf.keras.Model(dense_inputs, dense_outputs, name=model_name)
        
        return dense_model

# LSTM model

class LSTMModel():
    def __init__(self, tokenizer, vocab, mean_word_length, embedding_output_dim=128, n_lstm_units=64, n_dense_units=1, activation="sigmoid"):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.mean_word_length = mean_word_length
        self.embedding_output_dim = embedding_output_dim
        self.n_lstm_units = n_lstm_units
        self.n_dense_units = n_dense_units
        self.activation = activation
        
        self.embedding_input_dim = len(vocab)
        
    def get_model(self, model_name="lstm_model"):
        tf.random.set_seed(42)
        
        lstm_embedding = layers.Embedding(input_dim=self.embedding_input_dim, 
                                          output_dim=self.embedding_output_dim, 
                                          embeddings_initializer="uniform",
                                          input_length=self.mean_word_length,
                                          name="embedding-lstm")
        
        lstm_inputs = layers.Input(shape=(1, ), dtype="string")
        x = self.tokenizer(lstm_inputs)
        x = lstm_embedding(x)
        x = layers.LSTM(units=self.n_lstm_units)(x)
        lstm_outputs = layers.Dense(units=self.n_dense_units, activation=self.activation)(x)
        lstm_model = tf.keras.Model(lstm_inputs, lstm_outputs, name=model_name)
        
        return lstm_model

# Bidirectional LSTM model

class BidirectionalLSTMModel():
    def __init__(self, tokenizer, vocab, mean_word_length, embedding_output_dim=128, n_lstm_units=64, n_dense_units=1, activation="sigmoid"):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.mean_word_length = mean_word_length
        self.embedding_output_dim = embedding_output_dim
        self.n_lstm_units = n_lstm_units
        self.n_dense_units = n_dense_units
        self.activation = activation
        
        self.embedding_input_dim = len(vocab)
        
    def get_model(self, model_name="bidirectional_lstm_model"):
        tf.random.set_seed(42)
        
        bidir_lstm_embedding = layers.Embedding(input_dim=self.embedding_input_dim,
                                               output_dim=self.embedding_output_dim,
                                               embeddings_initializer="uniform",
                                               input_length=self.mean_word_length,
                                               name="embedding-bidirectional-lstm")
        bidir_lstm_inputs = layers.Input(shape=(1, ), dtype="string")
        x = self.tokenizer(bidir_lstm_inputs)
        x = bidir_lstm_embedding(x)
        x = layers.Bidirectional(layers.LSTM(units=self.n_lstm_units))(x)
        bidir_lstm_outputs = layers.Dense(units=1, activation="sigmoid")(x)
        bidir_lstm_model = tf.keras.Model(bidir_lstm_inputs, bidir_lstm_outputs, name=model_name)
        
        return bidir_lstm_model

# Ensemble model

def get_ensemble_models(models, train_data, train_label, val_data, num_iter=10, num_epochs=100, loss_funcs=["binary_crossentropy"]):
    """
    Returns a list of num_iter models each trained on binary_crossentropy loss functions by default.
    
    For instance, if num_iter = 10, a list of 60 trained models will be returned.
    10 * len(loss_funcs) * len(models) = 60 
    
    Parameters
    ----------
    models: NLP models passed.
    train_data: Training text dataset before tokenization and embedding.
    train_label: Training label dataset.
    val_data: List of validation dataset before tokenization and embedding.
    """
    ensemble_models = []
    
    for n_iter in range(num_iter):
        for model in models:
            for loss_func in loss_funcs:
                print(f"Reducing: {loss_func} for epochs: {num_epochs}, num_iter: {n_iter}, model: {model.name}")
                
                model.compile(loss=loss_func, optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
                
                model.fit(train_data, train_label, epochs=num_epochs, verbose=2, validation_data=val_data,
                         callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True),
                                     tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1)])
                
                ensemble_models.append(model)
    
    return ensemble_models

# Tokenization

# def __text_tokenization(dataset_path, fmt='csv'):
#     """
#     Applying tokenization on preprocessed output usin PyTorch's torchtext.
#     """
    
#     text = ttd.Field(sequential=True,
#                     batch_first=True,
#                     lower=False,
#                     tokenize='spacy',
#                     pad_first=True)

#     label = ttd.Field(sequential=False, 
#                       use_vocab=False, 
#                       is_target=True)
    
#     dataset = ttd.TabularDataset(path=dataset_path,
#                                 format=fmt,
#                                 skip_header=True,
#                                 fields=[{'features', text}, {'labels', label}])

#     train_dataset, test_dataset = dataset.split()
    
#     text.build_vocab(train_dataset)
#     vocab = text.vocab
    
#     train_iter, test_iter = ttd.Iterator.splits((train_dataset, test_dataset), 
#                                                 sort_key=lambda x: len(x.data),
#                                                 batch_sizes=(32, 256), 
#                                                 device=device)
    
#     return train_iter, test_iter, train_dataset, test_dataset, vocab

def text_tokenization(df_train_text, df_train_label, df_val_text, df_val_label):
    """
    Applying tokenization on preprocessed output.
    """
    
    IDX_CUS = 1
    IDX_TECH = 1
    
    for i, doc in df_train_text.iterrows():
        tokens_cus = doc['clean_text_customer'].lower().split()
        tokens_tech = doc['clean_text_technician'].lower().split()
        
        for token in tokens_cus:
            if token not in WORD_IDX_CUS:
                WORD_IDX_CUS[token] = IDX_CUS
                IDX_CUS += 1
        
        for token in tokens_tech:
            if token not in WORD_IDX_TECH:
                WORD_IDX_TECH[token] = IDX_TECH
                IDX_TECH += 1
    
    print(f"Length of words in customer documents: {len(WORD_IDX_CUS)}")
    print(f"Length of words in technician documents: {len(WORD_IDX_TECH)}")
    
    train_sentences_as_int_cus = []
    train_sentences_as_int_tech = []
    
    for i, doc in df_train_text.iterrows():
        tokens_cus = doc['clean_text_customer'].lower().split()
        tokens_tech = doc['clean_text_technician'].lower().split()
        
        sentence_as_int_cus = [WORD_IDX_CUS[token] for token in tokens_cus]
        sentence_as_int_tech = [WORD_IDX_TECH[token] for token in tokens_tech]
        
        train_sentences_as_int_cus.append(sentence_as_int_cus)
        train_sentences_as_int_tech.append(sentence_as_int_tech)
        
    val_sentences_as_int_cus = []
    val_sentences_as_int_tech = []
    
    for i, doc in df_val_text.iterrows():
        tokens_cus = doc['clean_text_customer'].lower().split()
        tokens_tech = doc['clean_text_technician'].lower().split()
        
        sentence_as_int_cus = [WORD_IDX_CUS[token] for token in tokens_cus if token in WORD_IDX_CUS]
        sentence_as_int_tech = [WORD_IDX_TECH[token] for token in tokens_tech if token in WORD_IDX_TECH]
        
        val_sentences_as_int_cus.append(sentence_as_int_cus)
        val_sentences_as_int_tech.append(sentence_as_int_tech)
        
    return train_sentences_as_int_cus, train_sentences_as_int_tech, val_sentences_as_int_cus, val_sentences_as_int_tech

# def text_gen(features, labels, batch_size=32):
#     features, labels = shuffle(features, labels)
    
#     n_batches = int(np.ceil(len(labels)) / batch_size)
    
#     if n_batches == 0:
#         n_batches = 1
    
#     features_batch = None
#     labels_batch = None
    
#     for batch in range(n_batches):
#         #end = min((i + 1) * batch_size, len(y))
        
#         if (batch + 1) * batch_size < len(labels):
#             end = (batch + 1) * batch_size
#         else:
#             end = len(labels)
        
#         features_batch = features[batch * batch_size: end]
#         labels_batch = labels[batch * batch_size: end]
        
#         max_len = np.max([len(feature) for feature in features_batch])
        
#         for i in range(len(features_batch)):
#             feature = features_batch[i]
#             pad = [0] * (max_len - len(feature))
#             features_batch[i] = pad + feature
        
#         features_batch = torch.from_numpy(np.array(features_batch)).long()
#         labels_batch = torch.from_numpy(np.array(labels_batch)).long()
        
#     yield features_batch, labels_batch
    
# # Models

# class CNN_torch(nn.Module):
#     def __init__(self, n_vocab, embed_dim, n_outputs):
#         super(CNN_torch, self).__init__()
#         self.n_vocab = n_vocab
#         self.embed_dim = embed_dim
#         self.n_outputs = n_outputs

#         self.embed = nn.Embedding(self.n_vocab, self.embed_dim)

#         self.conv1 = nn.Conv1d(self.embed_dim, 32, 3, padding=1)
#         self.pool1 = nn.MaxPool1d(2)
#         self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
#         self.pool2 = nn.MaxPool1d(2)
#         self.conv3 = nn.Conv1d(64, 128, 3, padding=1)

#         self.fc = nn.Linear(128, self.n_outputs)
    
#     def forward(self, inputs):
#         out = self.embed(inputs)

#         out = out.permute(0, 2, 1)

#         out = self.conv1(out)
#         out = F.relu(out)
#         out = self.pool1(out)

#         out = self.conv2(out)
#         out = F.relu(out)
#         out = self.pool2(out)

#         out = self.conv3(out)
#         out = F.relu(out)

#         out = out.permute(0, 2, 1)

#         out, _ = torch.max(out, 1)

#         out = self.fc(out)

#         return out

# class RNN_torch(nn.Module):
#     def __init__(self, n_vocab, embed_dim, n_hidden, n_rnnlayers, n_outputs):
#         super(RNN_torch, self).__init__()
#         self.n_vocab = n_vocab
#         self.embed_dim = embed_dim
#         self.n_hidden = n_hidden
#         self.n_outputs = n_outputs
#         self.n_rnnlayers = n_rnnlayers

#         self.embed = nn.Embedding(self.n_vocab, self.embed_dim)
#         self.rnn = nn.LSTM(
#             input_size=self.embed_dim,
#             hidden_size=self.n_hidden,
#             num_layers=self.n_rnnlayers,
#             batch_first=True
#         )

#         self.fc = nn.Linear(self.n_hidden, self.n_outputs)
    
#     def forward(self, inputs):
#         hidden_state = torch.zeros(self.n_rnnlayers, inputs.size(0), self.n_hidden).to(device)
#         cell_state = torch.zeros(self.n_rnnlayers, inputs.size(0), self.n_hidden).to(device)

#         out = self.embed(inputs)

#         out, _ = self.rnn(out, (hidden_state, cell_state))

#         out, _ = torch.max(out, 1)

#         out = self.fc(out)

#         return out

# def BGD_training(model, criterion, optimizer, epochs=130):
#     train_losses = np.zeros(epochs)
#     val_losses = np.zeros(epochs)
    
#     for epoch in range(epochs):
#         train_loss = []
        
#         for inputs, targets in train_gen():
#             targets = targets.view(-1, 1).float()
#             inputs, targets = inputs.to(device), targets.to(device)
            
#             optimizer.zero_grad()
            
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
            
#             loss.backward()
#             optimizer.step()
            
#             train_loss.append(loss.item())
        
#         train_loss = np.mean(train_loss)
        
#         val_loss = []
        
#         for inputs, targets in val_gen():
#             inputs, targets = inputs.to(device), targets.to(device)
#             targets = targets.view(-1, 1).float()
            
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
            
#             val_loss.append(loss.item())
        
#         val_loss = np.mean(val_loss)
        
#         val_losses[epoch] = val_loss
#         train_losses[epoch] = train_loss
        
#         print(f"Epoch {epoch + 1} / {epochs}, train loss: {train_loss}, val loss: {val_loss}")

#     return train_losses, val_losses

# def accuracy_torch(model):
#     n_correct = 0
#     n_total = 0

#     for inputs, targets in train_gen():
#         targets = targets.view(-1, 1).float()
#         inputs, targets = inputs.to(device), targets.to(device)

#         outputs = model(inputs)

#         predictions = (outputs > 0)

#         n_correct += (predictions == targets).sum().item()
#         n_total += targets.shape[0]

#     train_acc = n_correct / n_total

#     n_correct = 0
#     n_total = 0

#     for inputs, targets in val_gen():
#         targets = targets.view(-1, 1).float()
#         inputs, targets = inputs.to(device), targets.to(device)

#         outputs = model(inputs)

#         predictions = (outputs > 0)

#         n_correct += (predictions == targets).sum().item()
#         n_total += targets.shape[0]

#     val_acc = n_correct / n_total
    
#     print(f"Train acc: {train_acc}, val acc: {val_acc}")
    
#     return train_acc, val_acc