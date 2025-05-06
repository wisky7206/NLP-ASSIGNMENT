import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

class StreamingQoSDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class SatisfactionClassifier(nn.Module):
    def __init__(self, input_size):
        super(SatisfactionClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class SatisfactionModel:
    def __init__(self, input_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SatisfactionClassifier(input_size).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def train(self, train_data, val_data, batch_size=32, epochs=50):
        # Convert DataFrame to numpy arrays and ensure correct shapes
        X_train = train_data.values
        y_train = val_data.values.reshape(-1, 1)  # Reshape to (n_samples, 1)
        
        train_dataset = StreamingQoSDataset(X_train, y_train)
        val_dataset = StreamingQoSDataset(val_data.values, val_data.values.reshape(-1, 1))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for features, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)
            
            print(f'Epoch {epoch + 1}:')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation Metrics: {val_metrics}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('models/best_satisfaction_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                predictions.extend((outputs > 0.5).float().cpu().numpy())
                actuals.extend(labels.cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        accuracy = np.mean(predictions == actuals)
        precision = np.sum((predictions == 1) & (actuals == 1)) / (np.sum(predictions == 1) + 1e-10)
        recall = np.sum((predictions == 1) & (actuals == 1)) / (np.sum(actuals == 1) + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return total_loss / len(data_loader), metrics
    
    def predict(self, features):
        self.model.eval()
        features = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(features)
        
        return (predictions > 0.5).float().cpu().numpy()
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path)) 