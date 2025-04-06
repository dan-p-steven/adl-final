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
from functools import partial




EMBED_SIZE = 300
READ_SIZE = 70000

MAX_SEQ_LEN = 100
NUM_EPOCHS = 200

FEATURES_PATH = './data/features.npy'
LABELS_PATH = './data/labels.npy'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fixed_objective(trial, train_dataset, val_dataset):

    # Hyperparameters
    batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 10, log=True)
    hidden_size = trial.suggest_int('hidden_size', 16, 128, step=16)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    bidirectional = trial.suggest_categorical('bidirectional', [True, False])
    dropout_rate = trial.suggest_float('dropout_rate', 1e-2, 1, log=True)

    # Convert to DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = SentimentModel(
        input_size=EMBED_SIZE,
        hidden_size=hidden_size,
        bidirectional=bidirectional,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        num_classes=3
    )

    model = model.to(DEVICE)

    #loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    print (f'Training ...')
    _, val_losses, _, _ = train_model(model, 
                                                                 train_loader=train_loader, 
                                                                 val_loader=val_loader, 
                                                                 optimizer=optimizer, 
                                                                 loss_fn=loss_fn, 
                                                                 num_epochs=NUM_EPOCHS,device=DEVICE)
    
    # Getting val_losses at -1 will give you the best score
    score = val_losses[-1] 

    return score


def main():

    
    print(f"Using device: {DEVICE}\n")


    # Predefined Hyperparameters. These were chosen ahead of time due to computational
    # limits.



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

    # Convert to dataset (features are padded here)
    train_dataset = YelpDataset(X_train, y_train, MAX_SEQ_LEN)
    val_dataset = YelpDataset(X_val, y_val, MAX_SEQ_LEN)
    #test_dataset = YelpDataset(X_test, y_test, MAX_SEQ_LEN)


    # Record time taken to load
    end = time.time()
    print (f'Done: {end-start:.2f}s')

    study = optuna.load_study(study_name='sentiment_lstm_hpo',
                              storage='sqlite:///./models/sentiment_lstm_hpo.db')
    
    train_losses, val_losses, train_accs, val_accs = train_best_params(study, train_dataset=train_dataset, val_dataset=val_dataset)
    
    




    # # Partial function with fixed parameters
    # wrapped_objective = partial(fixed_objective, train_dataset=train_dataset, val_dataset=val_dataset)     
    
    # # Create the study and optimize
    # study = optuna.create_study(study_name="sentiment_lstm_hpo",
    #                             direction="minimize",
    #                             storage="sqlite:///./models/sentiment_lstm_hpo.db",
    #                             load_if_exists=True
    #                             )
    # study.optimize(wrapped_objective, n_trials=100)


def train_best_params(study: optuna.Study, train_dataset, val_dataset):
    '''
    Train a model using the best params and save its weights and biases.
    '''

    best = study.best_params

    # Convert to DataLoader
    train_loader = DataLoader(train_dataset, batch_size=best['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best['batch_size'])


    model = SentimentModel(
        input_size=EMBED_SIZE,
        hidden_size=best['hidden_size'],
        bidirectional=best['bidirectional'],
        num_layers=best['num_layers'],
        dropout_rate=best['dropout_rate'],
        num_classes=3
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best['learning_rate'])

    model = model.to(DEVICE)

    return train_model( model, 
                        train_loader=train_loader, 
                        val_loader=val_loader, 
                        optimizer=optimizer, 
                        loss_fn=loss_fn, 
                        num_epochs=NUM_EPOCHS,
                        device=DEVICE)







if __name__ == "__main__":
    study = optuna.load_study(study_name='sentiment_lstm_hpo',
                              storage='sqlite:///./models/sentiment_lstm_hpo.db')
    
    best_params = study.best_params
    print (best_params)