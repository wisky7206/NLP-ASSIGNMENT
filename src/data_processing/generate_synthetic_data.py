import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_streaming_text(satisfaction_score, sentiment_score):
    """Generate realistic streaming feedback text based on satisfaction and sentiment scores."""
    # Base phrases for different satisfaction levels
    base_phrases = {
        'high': [
            "Great stream quality!",
            "Perfect streaming experience",
            "Excellent video quality",
            "Smooth streaming with no issues",
            "Crystal clear video and audio"
        ],
        'medium': [
            "Decent stream quality",
            "Acceptable streaming experience",
            "Some minor quality issues",
            "Stream is watchable",
            "Average streaming quality"
        ],
        'low': [
            "Poor stream quality",
            "Frequent buffering issues",
            "Stream keeps freezing",
            "Very low quality video",
            "Unwatchable stream"
        ]
    }
    
    # Additional phrases based on sentiment
    sentiment_phrases = {
        'positive': [
            "Would definitely watch again",
            "Really enjoying the content",
            "Great community interaction",
            "Love the streamer's energy",
            "Amazing gameplay"
        ],
        'neutral': [
            "Content is okay",
            "Stream is fine",
            "Nothing special",
            "Average content",
            "Decent gameplay"
        ],
        'negative': [
            "Not worth watching",
            "Stream needs improvement",
            "Disappointing quality",
            "Technical issues are annoying",
            "Hard to enjoy the content"
        ]
    }
    
    # Select base phrase based on satisfaction
    if satisfaction_score >= 4:
        base = np.random.choice(base_phrases['high'])
    elif satisfaction_score >= 2:
        base = np.random.choice(base_phrases['medium'])
    else:
        base = np.random.choice(base_phrases['low'])
    
    # Select sentiment phrase
    if sentiment_score >= 4:
        sentiment = np.random.choice(sentiment_phrases['positive'])
    elif sentiment_score >= 2:
        sentiment = np.random.choice(sentiment_phrases['neutral'])
    else:
        sentiment = np.random.choice(sentiment_phrases['negative'])
    
    return f"{base} {sentiment}"

def generate_synthetic_dataset(num_samples=1000):
    """Generate a synthetic dataset for streaming QoS analysis."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate timestamps
    start_time = datetime.now() - timedelta(days=30)
    timestamps = [start_time + timedelta(minutes=i*30) for i in range(num_samples)]
    
    # Generate streaming quality metrics
    bitrates = np.random.normal(5000, 1000, num_samples).clip(1000, 10000)  # kbps
    resolutions = np.random.choice([720, 1080, 1440, 2160], num_samples)
    frame_rates = np.random.choice([30, 60, 120], num_samples)
    
    # Generate network conditions
    latencies = np.random.normal(50, 15, num_samples).clip(20, 200)  # ms
    packet_losses = np.random.normal(0.02, 0.01, num_samples).clip(0, 0.1)  # percentage
    
    # Generate viewer metrics
    viewer_counts = np.random.normal(1000, 500, num_samples).clip(0, 10000)
    follower_counts = np.random.normal(5000, 2000, num_samples).clip(100, 50000)
    engagement_rates = viewer_counts / follower_counts
    
    # Generate satisfaction scores based on quality metrics and engagement
    quality_score = (
        (bitrates / 10000) * 0.3 +
        (resolutions / 2160) * 0.3 +
        (frame_rates / 120) * 0.2 +
        (1 - latencies / 200) * 0.1 +
        (1 - packet_losses / 0.1) * 0.1
    ) * 5
    
    engagement_score = (engagement_rates / engagement_rates.max()) * 5
    satisfaction_scores = (quality_score * 0.7 + engagement_score * 0.3).clip(1, 5)
    
    # Generate sentiment scores based on satisfaction and engagement
    sentiment_scores = (satisfaction_scores * 0.8 + engagement_score * 0.2).clip(1, 5)
    
    # Generate text feedback
    texts = [
        generate_streaming_text(sat, sent)
        for sat, sent in zip(satisfaction_scores, sentiment_scores)
    ]
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'bitrate': bitrates,
        'resolution': resolutions,
        'frame_rate': frame_rates,
        'latency': latencies,
        'packet_loss': packet_losses,
        'viewer_count': viewer_counts,
        'follower_count': follower_counts,
        'engagement_rate': engagement_rates,
        'satisfaction_score': satisfaction_scores,
        'user_sentiment': sentiment_scores,
        'text': texts
    })
    
    return df

def main():
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    
    # Generate dataset
    print("Generating synthetic dataset...")
    df = generate_synthetic_dataset(num_samples=1000)
    
    # Save dataset
    output_path = 'data/raw/streaming_qos_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print("\nNumerical Features:")
    for col in ['bitrate', 'resolution', 'frame_rate', 'latency', 'packet_loss',
                'viewer_count', 'follower_count', 'engagement_rate',
                'satisfaction_score', 'user_sentiment']:
        print(f"\n{col}:")
        print(f"Mean: {df[col].mean():.2f}")
        print(f"Std: {df[col].std():.2f}")
        print(f"Min: {df[col].min():.2f}")
        print(f"Max: {df[col].max():.2f}")
    
    print("\nSample text entries:")
    print(df['text'].head())

if __name__ == "__main__":
    main() 