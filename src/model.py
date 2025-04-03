import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score

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



def evaluate_model(model, data_loader, loss_fn, device):

    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            X = batch['feature'].to(device)
            y = batch['label'].to(device)
            lengths = batch['length']
            
            # Forward pass
            predictions = model(X, lengths)
            
            # Compute loss
            loss = loss_fn(predictions, y)
            total_loss += loss.item()
            
            # Track predictions
            all_preds.extend(predictions.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    return all_preds, all_labels, avg_loss
        

def train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, device):

    #best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):

        # Training phase
        model.train()

        epoch_loss = 0
        epoch_preds = []
        epoch_labels = []

        print ('\n')

        for batch in tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', total=len(train_loader)):

            optimizer.zero_grad()

            X = batch['feature'].to(device)
            y = batch['label'].to(device)
            lengths = batch['length']

            # Forward pass
            predictions = model(X, lengths)

            # Compute loss
            loss = loss_fn(predictions, y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Record predictions and loss for batch
            # Track loss and predictions

            predictions = predictions.argmax(dim=1).cpu().numpy()
            epoch_loss += loss.item()
            epoch_preds.extend(predictions)
            epoch_labels.extend(y.cpu().numpy())

        # Calculate training metrics
        train_loss = epoch_loss / len(train_loader)
        train_acc = accuracy_score(epoch_labels, epoch_preds)


        print (f'\nValidation Phase')


        # Calculate validation metrics
        val_preds, val_labels, val_loss = evaluate_model(model, val_loader, loss_fn, device)
        val_acc = accuracy_score(val_labels, val_preds)

        # Keep track of validation and training metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)


        print(f'\tTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'\tVal Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return train_losses, val_losses, train_accuracies, val_accuracies