import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


def _pad_sequence(vector, max_sequence_len):

    if len(vector) < max_sequence_len:
        padded = np.zeros((max_sequence_len, vector.shape[1]))
        padded[:len(vector)] = vector
    else:
        padded = vector[:max_sequence_len]
    
    return padded


class YelpDataset(Dataset):
    def __init__(self, features: pd.Series, labels: pd.Series, max_sequence_len: int):

        
        # Preprocessing steps
        lengths = features.apply(lambda x: min(len(x), max_sequence_len))
        features = features.apply(lambda x: _pad_sequence(x, max_sequence_len))
        features = np.array(features.tolist())

        # Convert to numpy to convert to Tensor
        labels = np.array(labels.tolist())
        lengths = np.array(lengths.tolist())

        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        
        return {
           'feature': self.features[idx],
            'label': self.labels[idx],
            'length': self.lengths[idx]
        } 