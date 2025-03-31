import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from gensim.models import KeyedVectors

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
EMBEDDING_DIM = 300  # Standard for word2vec
HIDDEN_DIM = 128
NUM_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT_RATE = 0.5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
MAX_SEQ_LENGTH = 100

# Custom dataset class for Yelp reviews with pre-computed embeddings
class YelpDataset(Dataset):
    def __init__(self, reviews_vectors, labels, max_seq_length):
        self.reviews_vectors = reviews_vectors
        self.labels = labels
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return len(self.reviews_vectors)
    
    def __getitem__(self, idx):
        # Get the pre-computed word vectors for this review
        review_vectors = self.reviews_vectors[idx]
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
            'review_vectors': torch.tensor(padded_vectors, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.float),
            'length': torch.tensor(seq_length, dtype=torch.long)
        }

# LSTM model for sentiment analysis (without embedding layer)
class LSTMSentiment(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, bidirectional, dropout_rate):
        super(LSTMSentiment, self).__init__()
        self.lstm = nn.LSTM(input_dim, 
                           hidden_dim, 
                           num_layers=num_layers, 
                           bidirectional=bidirectional, 
                           batch_first=True, 
                           dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, embedded, text_lengths):
        # embedded shape: [batch_size, seq_len, embedding_dim]
        
        # Pack sequence for LSTM
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack the sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # If bidirectional, concatenate the final forward and backward hidden states
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
            
        hidden = self.dropout(hidden)
        prediction = self.fc(hidden)
        return self.sigmoid(prediction)

# Function to load pre-computed word2vec embeddings for reviews
def load_preprocessed_data(reviews_vectors_path, labels_path):
    """
    Load pre-computed word vectors and labels
    """
    # Load the review vectors (list of numpy arrays, each representing a review)
    with open(reviews_vectors_path, 'rb') as f:
        reviews_vectors = pickle.load(f)
    
    # Load the sentiment labels
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    
    return reviews_vectors, labels

# Alternative function to prepare vectors from word2vec and raw tokens
def prepare_vectors_from_tokens(tokenized_reviews, word2vec_model_path, embedding_dim=300):
    """
    Convert tokenized reviews to sequences of word vectors using a pre-trained word2vec model
    """
    # Load word2vec model
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
    
    reviews_vectors = []
    for review in tqdm(tokenized_reviews, desc="Converting tokens to vectors"):
        # Get word vectors for each token in the review
        review_vectors = []
        for token in review:
            if token in word2vec_model:
                review_vectors.append(word2vec_model[token])
            else:
                # Use zero vector for OOV words
                review_vectors.append(np.zeros(embedding_dim))
        
        # Convert to numpy array
        if review_vectors:
            reviews_vectors.append(np.array(review_vectors))
        else:
            # Handle empty reviews with a single zero vector
            reviews_vectors.append(np.zeros((1, embedding_dim)))
    
    return reviews_vectors

# Training function
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device):
    best_val_loss = float('inf')
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
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            vectors = batch['review_vectors'].to(device)
            labels = batch['label'].to(device)
            lengths = batch['length'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(vectors, lengths).squeeze(1)
            
            # Compute loss
            loss = criterion(predictions, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss and predictions
            epoch_loss += loss.item()
            epoch_preds.extend((predictions > 0.5).cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_loss = epoch_loss / len(train_loader)
        train_acc = accuracy_score(epoch_labels, epoch_preds)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        val_loss, val_acc, _, _, _ = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_yelp_sentiment_model.pth')
            print('  Model saved!')
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Evaluation function
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            vectors = batch['review_vectors'].to(device)
            labels = batch['label'].to(device)
            lengths = batch['length'].to(device)
            
            # Forward pass
            predictions = model(vectors, lengths).squeeze(1)
            
            # Compute loss
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            
            # Track predictions
            all_preds.extend((predictions > 0.5).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    return avg_loss, accuracy, precision, recall, f1

# Function to plot training curves
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss curves
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracy curves
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

# Main function to run everything
def main():
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Method 1: Load pre-computed word vectors and labels
    print("Loading pre-computed word vectors and labels...")
    try:
        # Replace with your actual file paths
        reviews_vectors, labels = load_preprocessed_data(
            'yelp_reviews_vectors.pkl', 
            'yelp_reviews_labels.pkl'
        )
    except FileNotFoundError:
        print("Pre-computed vectors file not found. Please ensure data is preprocessed or use Method 2.")
        
        # Method 2: Alternatively, prepare vectors from raw tokens and word2vec
        print("Loading and preprocessing data using word2vec...")
        
        # This is a placeholder for loading raw token data - replace with your actual loading code
        # df = pd.read_csv('yelp_reviews_tokens.csv')
        # tokenized_reviews = df['tokens'].tolist()
        # sentiment_labels = df['sentiment'].tolist()
        
        # reviews_vectors = prepare_vectors_from_tokens(
        #     tokenized_reviews, 
        #     'word2vec-model.bin', 
        #     EMBEDDING_DIM
        # )
        # labels = sentiment_labels
        
        # Save the vectors and labels for future use
        # with open('yelp_reviews_vectors.pkl', 'wb') as f:
        #     pickle.dump(reviews_vectors, f)
        # with open('yelp_reviews_labels.pkl', 'wb') as f:
        #     pickle.dump(labels, f)
        return
    
    print(f"Loaded {len(reviews_vectors)} reviews with their vector representations")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        reviews_vectors, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")
    
    # Create datasets
    train_dataset = YelpDataset(X_train, y_train, MAX_SEQ_LENGTH)
    val_dataset = YelpDataset(X_val, y_val, MAX_SEQ_LENGTH)
    test_dataset = YelpDataset(X_test, y_test, MAX_SEQ_LENGTH)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model (using the dimensionality of the pre-computed embeddings)
    model = LSTMSentiment(
        input_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Print model summary
    print(model)
    
    # Train model
    print("Starting training...")
    train_model(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, device)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load('best_yelp_sentiment_model.pth'))
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, criterion, device)
    
    print("\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")

if __name__ == "__main__":
    main()
