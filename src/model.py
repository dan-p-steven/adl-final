import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SentimentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, bidirectional, dropout_rate, num_classes):
        
        super(SentimentModel, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, 
                           hidden_dim, 
                           num_layers=num_layers, 
                           bidirectional=bidirectional, 
                           batch_first=True, 
                           dropout=dropout_rate if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, embedded, text_lengths):

        # Pack sequence for LSTM
        embedded_packed = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Pass through lstm layer(s)
        out, (hidden, cell) = self.lstm(embedded_packed)

        # Re-pad sequence
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # If bidirectional, concatenate the final forward and backward hidden states
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]

        # Apply post-lstm dropout
        hidden = self.dropout(hidden)

        # Pass through output
        prediction = self.fc(hidden)
        return self.softmax(prediction)




