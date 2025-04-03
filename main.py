import pandas as pd
import numpy as np

import src.preprocess as preprocess
from src.model import SentimentModel, train_model, evaluate_model
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
READ_SIZE = 70000

FEATURES_PATH = './data/features.npy'
LABELS_PATH = './data/labels.npy'

def cuda_check():
    print(torch.version.cuda)  # This will print the version of CUDA PyTorch is using

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # Hyperparameters
    max_seq_len = 100
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.01

    lstm_hidden_dim = 32
    lstm_num_layers = 2
    lstm_bidirectional = False
    lstm_dropout_rate = 0.5

    model = SentimentModel(
        input_dim=EMBED_SIZE,
        hidden_dim=lstm_hidden_dim,
        num_layers=lstm_num_layers,
        bidirectional=lstm_bidirectional,
        dropout_rate=lstm_dropout_rate,
        num_classes=3,
    )

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    df = pd.DataFrame()

    # Start timer
    print (f'Reading features and targets ...')
    start = time.time()

    df['X'] = np.load(FEATURES_PATH, allow_pickle=True)[:READ_SIZE]
    df['y'] = np.load(LABELS_PATH)[:READ_SIZE]


    # Splitting data
    X_train, X_temp, y_train, y_temp = train_test_split(
        df['X'],  df['y'], test_size=0.3, stratify=df['y'], random_state=42)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.66, stratify=y_temp, random_state=42)
    

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
    print (f'Done: {end-start:.2f}s')

    train_model(
        model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        loss_fn=loss_fn, 
        num_epochs=num_epochs,
        device=device)


if __name__ == "__main__":
    main()