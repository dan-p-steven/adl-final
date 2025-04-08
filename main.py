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
import plotly.graph_objects as go





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

    model = SentimentModel(
        input_size=EMBED_SIZE,
        hidden_size=hidden_size,
        bidirectional=bidirectional,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        num_classes=3
    )

    model = model.to(DEVICE)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    print (f'Training ...')
    _, val_losses, _, _ = train_model(model, 
                                      train_loader=train_loader, 
                                      val_loader=val_loader, 
                                      optimizer=optimizer, 
                                      loss_fn=loss_fn, 
                                      num_epochs=NUM_EPOCHS,
                                      device=DEVICE)
    
    # Getting val_losses at -1 will give you the best score
    score = val_losses[-1] 
    return score

def load_split_data(features_path, labels_path):

    # Start timer
    print (f'Reading dataset...\n\tfeatures: {features_path}\n\tlabels: {labels_path}')

    start = time.time()

    df = pd.DataFrame()
    df['X'] = np.load(features_path, allow_pickle=True)
    df['y'] = np.load(labels_path)


    # Splitting data
    X_train, X_temp, y_train, y_temp = train_test_split(
        df['X'],  df['y'], test_size=0.3, stratify=df['y'], random_state=42)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.66, stratify=y_temp, random_state=42)

    # Convert to dataset (features are padded here)
    train_dataset = YelpDataset(X_train, y_train, MAX_SEQ_LEN)
    val_dataset = YelpDataset(X_val, y_val, MAX_SEQ_LEN)
    test_dataset = YelpDataset(X_test, y_test, MAX_SEQ_LEN)


    # Record time taken to load
    end = time.time()
    print (f'Done: {end-start:.2f}s')

    return train_dataset, val_dataset, test_dataset

def hpo(study_name, direction, storage):

    print(f"Using device: {DEVICE}\n")
    train_dataset, val_dataset, test_dataset = load_split_data(FEATURES_PATH, LABELS_PATH)

    # Partial function with fixed parameters
    wrapped_objective = partial(fixed_objective, train_dataset=train_dataset, val_dataset=val_dataset)     
    
    # Create the study and optimize
    study = optuna.create_study(study_name=study_name,
                                direction=direction,
                                storage=storage,
                                load_if_exists=True
                                )
    
    study.optimize(wrapped_objective, n_trials=100)


def main():
    pass






def final_model_train(study: optuna.Study, train_dataset, val_dataset):
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

def load_model_from_study(study: optuna.Study):

    # Get best params.
    best = study.best_params
    
    # Instantiate model using best params.
    model = SentimentModel(
        input_size=EMBED_SIZE,
        hidden_size=best['hidden_size'],
        bidirectional=best['bidirectional'],
        num_layers=best['num_layers'],
        dropout_rate=best['dropout_rate'],
        num_classes=3,
    )

    # Put model on GPU
    model = model.to(DEVICE)

    # Load best model weights and biases.
    model.load_state_dict(torch.load('./models/best_model.pth'))

    return model


def final_model_test(study: optuna.Study, test_dataset):

    model = load_model_from_study(study)

    test_loader = DataLoader(test_dataset, batch_size=study.best_params['batch_size'])

    loss_fn = nn.CrossEntropyLoss()

    # Evaluate test dataset
    y_pred, y_actual, _ = evaluate_model(model, test_loader, loss_fn, DEVICE)

    # Save y pred and y actual
    np.save('./models/y_pred.npy', np.array(y_pred))
    np.save('./models/y_actual.npy', np.array(y_actual))


def evaluate_sentences(model, sentences, nlp):

    disable = ['ner', 'parser', 'attribute_ruler', 'lemmatizer', 'tagger']
    batch_size = 10

    # List to store tokenized vectors.
    tokenized_vectors = []

    # Tokenize sentences.
    for doc in nlp.pipe(sentences, batch_size=batch_size, disable=disable, n_process=1):

        # Record the tokens that have a vector.
        s = []
        for token in doc:
            if token.has_vector:
                s.append(token.vector)

        # Append the vectorized sample.
        tokenized_vectors.append(s)

    df = pd.DataFrame()
    df['input'] = tokenized_vectors
    df['input'] = df['input'].apply(lambda v: np.vstack(v))
    df['dummy'] = np.zeros((len(sentences,)))

    input_dataset = YelpDataset(df['input'], df['dummy'], MAX_SEQ_LEN)
    #evaluate_model(model, data_loader=input_dataloader, 


def sequence_length_impact(study):

    # Read in an unseen dataset. 
    model = load_model_from_study(study)

    df = pd.read_csv('./data/yelp_review_100k.csv', skiprows=range(1, READ_SIZE+1))
    print (df.columns)
    df['length'] = df['text'].apply(lambda x: len(x.split()))
    
    print (df['length'].value_counts())
    
    


if __name__ == "__main__":

    print(f"Using device: {DEVICE}\n")
    


    # Optuna variables
    study_name="sentiment_lstm_hpo"
    direction="minimize"
    storage="sqlite:///./models/sentiment_lstm_hpo.db"

    # Load the optuna study
    study = optuna.load_study(study_name=study_name, storage=storage)

    sequence_length_impact(study)
    
    
