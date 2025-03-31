import torch
from torch.utils.data import Dataset
import numpy as np



class YelpDataset(Dataset):
    def __init__(self, reviews_vectors, labels, max_seq_length):
        self.features = reviews_vectors.values
        self.labels = labels.values
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Get the pre-computed word vectors for this review
        review_vectors = self.features[idx]
        label = self.labels[idx]
        
        # Get actual sequence length before padding
        seq_length = min(len(review_vectors), self.max_seq_length)
        
        # Pad or truncate the sequence of vectors
        if len(review_vectors) < self.max_seq_length:
            # Pad with zeros
            padded_vectors = np.zeros((self.max_seq_length, review_vectors.shape[1]))
            padded_vectors[:len(review_vectors)] = review_vectors
        else:
            # Truncate
            padded_vectors = review_vectors[:self.max_seq_length]
        
        return {
            'feature': torch.tensor(padded_vectors, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'length': torch.tensor(seq_length, dtype=torch.long)
        }