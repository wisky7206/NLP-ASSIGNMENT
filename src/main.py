import os
import pandas as pd
from data_processing.process_data import DataProcessor
from sentiment.sentiment_analyzer import SentimentAnalyzer
from classification.satisfaction_classifier import SatisfactionModel
from evaluation.qos_evaluator import QoSEvaluator

def main():
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    try:
        # Initialize components
        data_processor = DataProcessor('data/raw/streaming_qos_dataset.csv')
        
        # Process data
        print("Processing data...")
        processed_data = data_processor.load_data()
        
        # Initialize models
        sentiment_analyzer = SentimentAnalyzer()
        satisfaction_model = SatisfactionModel(input_size=5)  # 5 features: bitrate, resolution, frame_rate, latency, packet_loss
        qos_evaluator = QoSEvaluator()
        
        # Train sentiment analyzer
        print("\nTraining sentiment analyzer...")
        sentiment_analyzer.train(
            train_data=processed_data['sentiment_train'],
            val_data=processed_data['sentiment_test']
        )
        sentiment_analyzer.save_model('models/sentiment_model.pth')
        
        # Train satisfaction classifier
        print("\nTraining satisfaction classifier...")
        satisfaction_model.train(
            train_data=processed_data['X_train'],
            val_data=processed_data['y_satisfaction_train']
        )
        satisfaction_model.save_model('models/satisfaction_model.pth')
        
        # Generate predictions
        print("\nGenerating predictions...")
        sentiment_predictions = sentiment_analyzer.predict(processed_data['sentiment_test']['text'].values)
        satisfaction_predictions = satisfaction_model.predict(processed_data['X_test'])
        
        # Evaluate QoS
        print("\nEvaluating QoS...")
        qos_report = qos_evaluator.generate_qos_report(
            actual_qos=processed_data['y_satisfaction_test'],
            predicted_qos=satisfaction_predictions,
            sentiment_scores=sentiment_predictions,
            satisfaction_scores=satisfaction_predictions
        )
        
        # Save evaluation report
        qos_evaluator.save_report(qos_report)
        print("\nEvaluation complete! Check the reports directory for results.")
        
    except Exception as e:
        print(f"Error in main pipeline: {str(e)}")
        print("Please check the error message and ensure all components are properly configured.")

if __name__ == "__main__":
    main() 