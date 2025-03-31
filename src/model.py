import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

def _train_subroutine(model, train_loader, optimizer, loss_fn, history):
    for b, batch in enumerate(train_loader):

        optimizer.zero_grad()

        X = batch['feature']
        y = batch['label']
        lengths = batch['length']

        print (f'\r\t\tbatch {b}:', end='', flush=True)

        # Forward pass
        predictions = model(X, lengths)

        # Compute loss
        loss = loss_fn(predictions, y)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Record predictions and loss for batch
        history['loss'] += loss.item()
        history['preds'].extend((predictions > 0.5).numpy())
        history['labels'].extend(y.numpy())


      



    # for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
    # vectors = batch['review_vectors'].to(device)
    # labels = batch['label'].to(device)
    # lengths = batch['length'].to(device)
    
    # # Zero gradients
    # optimizer.zero_grad()
    
    # # Forward pass
    # predictions = model(vectors, lengths).squeeze(1)
    
    # # Compute loss
    # loss = criterion(predictions, labels)
    
    # # Backward pass and optimize
    # loss.backward()
    # optimizer.step()
    
    # # Track loss and predictions
    # epoch_loss += loss.item()
    # epoch_preds.extend((predictions > 0.5).cpu().numpy())
    # epoch_labels.extend(labels.cpu().numpy())





