import pandas as pd
import numpy as np

import src.preprocess as preprocess
from src.model import SentimentModel, train_model, evaluate_model
from src.dataset import YelpDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim



from sklearn.model_selection import train_test_split

import time
import optuna


EMBED_SIZE = 300
READ_SIZE = 70000

FEATURES_PATH = './data/features.npy'
LABELS_PATH = './data/labels.npy'


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")


    # Hyperparameters
    
    batch_size = 32
    learning_rate = 0.001

    # lstm_hidden_dim = 32
    # lstm_num_layers = 1
    # lstm_bidirectional = False
    # lstm_dropout_rate = 0.5

    max_seq_len = 100
    num_epochs = 200




    # Start timer
    print (f'Reading features and targets ...')
    start = time.time()

    df = pd.DataFrame()
    df['X'] = np.load(FEATURES_PATH, allow_pickle=True)
    df['y'] = np.load(LABELS_PATH)


    # Splitting data
    X_train, X_temp, y_train, y_temp = train_test_split(
        df['X'],  df['y'], test_size=0.3, stratify=df['y'], random_state=42)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.66, stratify=y_temp, random_state=42)
        
    
    model = SentimentModel(
        input_size=300,
        hidden_size=128,
        bidirectional=True,
        num_layers=2,
        num_classes=3
    )

    model = model.to(device)

    #loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
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

    print (f'Training ...')
    train_losses, val_losses, train_accs, val_accs = train_model(model, 
                                                                 train_loader=train_loader, 
                                                                 val_loader=val_loader, 
                                                                 optimizer=optimizer, 
                                                                 loss_fn=loss_fn, 
                                                                 num_epochs=num_epochs,device=device)
    
    # Getting val_losses at -1 will give you the best score
    score = val_losses[-1] 


if __name__ == "__main__":
    main()