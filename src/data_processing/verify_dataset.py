import pandas as pd
import numpy as np
import os

def verify_dataset(file_path):
    """Verify the dataset format and required columns."""
    try:
        # Read the dataset
        df = pd.read_csv(file_path)
        
        # Required columns for Twitch dataset
        required_columns = [
            'stream_id', 'viewer_count', 'followers',
            'stream_duration', 'game', 'language'
        ]
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for missing values
        missing_values = df[required_columns].isnull().sum()
        if missing_values.any():
            print("\nMissing values in columns:")
            print(missing_values[missing_values > 0])
            return False
        
        # Check data types and ranges
        print("\nData type and range information:")
        for col in required_columns:
            print(f"\n{col}:")
            print(f"Type: {df[col].dtype}")
            if df[col].dtype in [np.int64, np.float64]:
                print(f"Range: {df[col].min()} to {df[col].max()}")
                print(f"Mean: {df[col].mean():.2f}")
                print(f"Standard Deviation: {df[col].std():.2f}")
        
        # Basic statistics
        print("\nDataset Statistics:")
        print(f"Total number of samples: {len(df)}")
        print(f"Number of unique streams: {df['stream_id'].nunique()}")
        print(f"Number of unique games: {df['game'].nunique()}")
        print(f"Number of languages: {df['language'].nunique()}")
        
        return True
        
    except Exception as e:
        print(f"Error verifying dataset: {str(e)}")
        return False

def prepare_dataset(file_path):
    """Prepare the dataset for processing."""
    try:
        # Read the dataset
        df = pd.read_csv(file_path)
        
        # Create derived metrics
        df['engagement_rate'] = df['viewer_count'] / df['followers']
        df['stream_duration_hours'] = df['stream_duration'] / 3600  # Convert to hours
        
        # Create satisfaction score based on engagement
        df['satisfaction_score'] = (df['engagement_rate'] * 5).clip(0, 5)
        
        # Create sentiment score based on viewer count and followers
        df['user_sentiment'] = ((df['viewer_count'] / df['followers'].max()) * 5).clip(0, 5)
        
        # Select and rename columns for our model
        processed_df = pd.DataFrame({
            'stream_id': df['stream_id'],
            'bitrate': df['viewer_count'] * 100,  # Simulated bitrate based on viewer count
            'resolution': 720,  # Default resolution
            'frame_rate': 30,   # Default frame rate
            'latency': np.random.normal(50, 10, len(df)),  # Simulated latency
            'packet_loss': np.random.normal(0.1, 0.05, len(df)),  # Simulated packet loss
            'user_sentiment': df['user_sentiment'],
            'satisfaction_score': df['satisfaction_score']
        })
        
        # Save prepared dataset
        output_path = 'data/raw/streaming_qos_dataset.csv'
        processed_df.to_csv(output_path, index=False)
        print(f"\nPrepared dataset saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        return False

def main():
    # Check if dataset exists
    file_path = 'data/raw/twitch_data.csv'  # Changed to match Twitch dataset filename
    if not os.path.exists(file_path):
        print(f"Dataset not found at: {file_path}")
        print("Please download the Twitch dataset from:")
        print("https://www.kaggle.com/datasets/austinreese/twitch-streamer-data")
        print("and place it in the data/raw directory as 'twitch_data.csv'")
        return
    
    # Verify dataset
    print("Verifying dataset...")
    if verify_dataset(file_path):
        print("\nDataset verification successful!")
        
        # Prepare dataset
        print("\nPreparing dataset...")
        if prepare_dataset(file_path):
            print("Dataset preparation successful!")
        else:
            print("Dataset preparation failed!")
    else:
        print("Dataset verification failed!")

if __name__ == "__main__":
    main() 