import pandas as pd
import numpy as np
import src.preprocess as preprocess
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

import time



def pad_sequence(sample, seq_len):

    sample_len = sample.shape[0]

    if sample_len < seq_len:
        
        padded_sample = np.zeros((seq_len, sample.shape[1]))
        padded_sample[:sample_len] = sample

    else:
        padded_sample = sample[:seq_len]

    return padded_sample

def main():
    # Hyperparameters
    max_seq_len = 50
    batch_size = 32

    df = pd.DataFrame()

    start = time.time()

    df['X'] = np.load(f'./data/features/X_1.npy', allow_pickle=True)
    df['y'] = np.load(f'./data/labels/y_1.npy')


    df['X'] = df['X'].apply(lambda x: pad_sequence(x, max_seq_len))

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
    # train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    # val_dataset = TensorDataset(torch.tensor(X_val.values, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    # test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    # # Convert to DataLoader
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)

    end = time.time()

    print (f'Loading data chunk 1: {end-start:.2f}s')



if __name__ == "__main__":
    main()
