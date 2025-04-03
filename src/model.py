import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score

class SentimentModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        
        super(SentimentModel, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):

        batch_size = x.size(0)

        out, (h_n, c_n) = self.lstm(x)

        out = h_n[-1]
        out = self.fc(out)
        return out



def evaluate_model(model, data_loader, loss_fn, device):

    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            X = batch['feature'].to(device)
            y = batch['label'].to(device)
            
            # Forward pass
            predictions = model(X)
            
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

            # Forward pass
            predictions = model(X)

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