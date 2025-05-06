import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and preprocess the streaming QoS dataset."""
        # Load the dataset
        df = pd.read_csv(self.data_path)
        
        # Ensure all required columns exist
        required_columns = [
            'bitrate', 'resolution', 'frame_rate', 'latency', 'packet_loss',
            'viewer_count', 'follower_count', 'engagement_rate',
            'satisfaction_score', 'user_sentiment', 'text'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in dataset")
        
        # Prepare features for satisfaction prediction
        feature_columns = ['bitrate', 'resolution', 'frame_rate', 'latency', 'packet_loss']
        X = df[feature_columns].copy()
        
        # Scale numerical features
        X[feature_columns] = self.scaler.fit_transform(X[feature_columns])
        
        # Prepare sentiment data
        sentiment_data = df[['text', 'user_sentiment']].copy()
        
        # Split data for both tasks
        X_train, X_test, y_sat_train, y_sat_test = train_test_split(
            X, df['satisfaction_score'], test_size=0.2, random_state=42
        )
        
        # Split sentiment data
        sent_train, sent_test = train_test_split(
            sentiment_data, test_size=0.2, random_state=42
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_satisfaction_train': y_sat_train,
            'y_satisfaction_test': y_sat_test,
            'sentiment_train': sent_train,
            'sentiment_test': sent_test
        }

def main():
    # Create necessary directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Initialize data processor
    processor = DataProcessor('data/raw/streaming_qos_dataset.csv')
    
    try:
        # Process data
        processed_data = processor.load_data()
        
        # Save processed data
        for key, value in processed_data.items():
            value.to_csv(f'data/processed/{key}.csv', index=False)
        
        print("Data processing completed successfully!")
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        print("Please ensure the dataset is properly formatted with all required columns.")

if __name__ == "__main__":
    main() 