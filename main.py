import pandas as pd
import numpy as np

import src.preprocess as preprocess
from src.model import SentimentModel, _train_subroutine
from src.dataset import YelpDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


from sklearn.model_selection import train_test_split

import time
import glob

import random

EMBED_SIZE = 300
READ_SIZE = 10000

FEATURE_PATH = './data/features/*.npy'
LABEL_PATH = './data/labels/*.npy'

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

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

 

    df = pd.DataFrame()


    feature_files = glob.glob(FEATURE_PATH)
    label_files = glob.glob(LABEL_PATH)

    for e in range(num_epochs):

        # At the start of each epoch, randomly shuffle the chunk order
        random.shuffle(feature_files)
        random.shuffle(label_files)

        model.train()

        epoch_history = {
            'loss': 0,
            'preds': [],
            'labels': [],
        }

        for i in range(0, len(feature_files)):
            
            # Start timer
            print (f'Chunk {i}\n\tReading features and targets ...')
            start = time.time()

            df['X'] = np.load(feature_files[i], allow_pickle=True)[:READ_SIZE]
            df['y'] = np.load(label_files[i])[:READ_SIZE]


            # Splitting data
            X_train, X_temp, y_train, y_temp = train_test_split(
                df['X'],  df['y'], test_size=0.3, stratify=None, random_state=42)
            
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.66, stratify=None, random_state=42)
            

            # Convert to dataset (features are padded here)
            train_dataset = YelpDataset(X_train, y_train, max_seq_len)
            val_dataset = YelpDataset(X_val, y_val, max_seq_len)
            test_dataset = YelpDataset(X_test, y_test, max_seq_len)
        
            # Convert to DataLoader
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            # Record time taken to load
            end = time.time()
            print (f'\t\tDone: {end-start:.2f}s')

            print (f'\tTraining ...')
            start = time.time()

            _train_subroutine(
                model, train_loader=train_loader, optimizer=optimizer, loss_fn=loss_fn, history=epoch_history)
            
            print (f'\n\tloss: {epoch_history["loss"]:.2f}')


if __name__ == "__main__":
    main()