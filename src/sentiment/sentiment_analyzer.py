import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

class StreamingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentAnalyzer:
    def __init__(self, model_name='bert-base-uncased', num_labels=3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)
        
    def train(self, train_data, val_data, batch_size=16, epochs=3, learning_rate=2e-5):
        # Convert sentiment scores to labels (1-5 scale to 0-2 for 3 classes)
        train_labels = (train_data['user_sentiment'].values - 1) // 2
        val_labels = (val_data['user_sentiment'].values - 1) // 2
        
        train_dataset = StreamingDataset(
            train_data['text'].values,
            train_labels,
            self.tokenizer
        )
        val_dataset = StreamingDataset(
            val_data['text'].values,
            val_labels,
            self.tokenizer
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}')
            
            # Validation
            self.evaluate(val_loader)
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += len(labels)
        
        accuracy = correct_predictions / total_predictions
        avg_loss = total_loss / len(data_loader)
        print(f'Validation - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    def predict(self, texts):
        self.model.eval()
        predictions = []
        
        for text in texts:
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pred = torch.argmax(outputs.logits, dim=1)
                # Convert back to 1-5 scale
                predictions.append((pred.item() * 2 + 1))
        
        return np.array(predictions)
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path)) 