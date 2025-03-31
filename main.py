import pandas as pd
import numpy as np

import src.preprocess as preprocess
from src.model import SentimentModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


from sklearn.model_selection import train_test_split

import time
import glob

import random

EMBED_SIZE = 300
READ_SIZE = 100

FEATURE_PATH = './data/features/*.npy'
LABEL_PATH = './data/labels/*.npy'

def pad_sequence(sample, max_seq_len):

    sample_len = sample.shape[0]

    if sample_len < max_seq_len:
        
        padded_sample = np.zeros((max_seq_len, sample.shape[1]))
        padded_sample[:sample_len] = sample

    else:
        padded_sample = sample[:max_seq_len]

    return padded_sample

def create_tensor_dataset(X, y):

    X_fixed = np.array([np.array(x, dtype=np.float32) for x in X.values], dtype=np.float32)

    y_tensor = torch.tensor(y.values, dtype=torch.long)
    X_tensor = torch.tensor(X_fixed, dtype=torch.float32)

    return TensorDataset(X_tensor, y_tensor)

def main():

    # Hyperparameters
    max_seq_len = 50
    batch_size = 32
    num_epochs = 1
    learning_rate = 0.001

    lstm_hidden_dim = 32
    lstm_num_layers = 1
    lstm_bidirectional = False
    lstm_dropout_rate = 0.5

    model = SentimentModel(
        input_dim=EMBED_SIZE,
        hidden_dim=lstm_hidden_dim,
        num_layers=lstm_num_layers,
        bidirectional=lstm_bidirectional,
        dropout_rate=lstm_dropout_rate,
        num_classes=3
    )

    loss = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=learning_rate)

 

    df = pd.DataFrame()


    feature_files = glob.glob(FEATURE_PATH)
    label_files = glob.glob(LABEL_PATH)

    for e in range(num_epochs):

        # At the start of each epoch, randomly shuffle the chunk order
        random.shuffle(feature_files)
        random.shuffle(label_files)

        for i in range(0, len(feature_files)):
            
            # Start timer
            print (f'Chunk {i}\n\tReading features and targets ...')
            start = time.time()

            df['X'] = np.load(feature_files[i], allow_pickle=True)[:READ_SIZE]
            df['y'] = np.load(label_files[i])[:READ_SIZE]

            # Preprocessing
            print (f'\tPreprocessing features')
            # Extract sequence lengths of all samples
            df['lengths'] = df['X'].apply(lambda x: x.shape[0] if x.shape[0] < max_seq_len else max_seq_len)
            lengths = torch.tensor(df['lengths'].values, dtype=torch.long)

            # Pad samples up to max_seq_len
            df['X'] = df['X'].apply(lambda x: pad_sequence(x, max_seq_len))

            # Splitting data
            X_train, X_temp, y_train, y_temp = train_test_split(
                df['X'],  df['y'], test_size=0.3, stratify=None, random_state=42)
            
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.66, stratify=None, random_state=42)
            
            print (f'\tCreating DataLoaders ...')

            # Convert to dataset
            train_dataset = create_tensor_dataset(X_train, y_train)
            val_dataset = create_tensor_dataset(X_val, y_val)
            test_dataset = create_tensor_dataset(X_test, y_test)
        
            # Convert to DataLoader
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            # Record time taken to load
            end = time.time()
            print (f'\tDone: {end-start:.2f}s')


if __name__ == "__main__":
    main()