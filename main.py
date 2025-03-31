import pandas as pd
import numpy as np
import src.preprocess as preprocess
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

import time



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

    lstm_hidden_dim = 32
    lstm_num_layers = 1
    lstm_bidirectional = False
    lstm_dropout_rate = 0.5


    read_size = 1000

    df = pd.DataFrame()

    start = time.time()

    df['X'] = np.load(f'./data/features/X_1.npy', allow_pickle=True)[:read_size]
    df['y'] = np.load(f'./data/labels/y_1.npy')[:read_size]


    # Preprocessing
    embedding_size = df['X'].iloc[0].shape[1]
    print (embedding_size)

    # Extract sequence lengths of all samples
    df['lengths'] = df['X'].apply(lambda x: x.shape[0] if x.shape[0] < max_seq_len else max_seq_len)
    lengths = torch.tensor(df['lengths'].values, dtype=torch.long)

    # Pad samples up to max_seq_len
    df['X'] = df['X'].apply(lambda x: pad_sequence(x, max_seq_len))

    # Splitting data
    X_train, X_temp, y_train, y_temp = train_test_split(df['X'], 
                                                        df['y'], 
                                                        test_size=0.3, 
                                                        stratify=df['y'], 
                                                        random_state=42)
    
    X_val, X_test, y_val, y_test = train_test_split(X_temp, 
                                                    y_temp, 
                                                    test_size=0.66, 
                                                    stratify=y_temp, 
                                                    random_state=42)
    
    # Convert to dataset
    train_dataset = create_tensor_dataset(X_train, y_train)
    val_dataset = create_tensor_dataset(X_val, y_val)
    test_dataset = create_tensor_dataset(X_test, y_test)
   
     # Convert to DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    
    end = time.time()
    print (f'Loading data chunk 1: {end-start:.2f}s')




if __name__ == "__main__":
    main()
