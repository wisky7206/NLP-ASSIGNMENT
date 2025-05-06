import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class QoSEvaluator:
    def __init__(self):
        self.metrics = {}
        self.predictions = {}
    
    def evaluate_qos_metrics(self, actual_qos, predicted_qos):
        """Evaluate QoS metrics using various statistical measures."""
        metrics = {
            'mse': mean_squared_error(actual_qos, predicted_qos),
            'rmse': np.sqrt(mean_squared_error(actual_qos, predicted_qos)),
            'mae': mean_absolute_error(actual_qos, predicted_qos),
            'r2': r2_score(actual_qos, predicted_qos)
        }
        return metrics
    
    def evaluate_sentiment_impact(self, sentiment_scores, qos_metrics):
        """Analyze the impact of sentiment on QoS metrics."""
        correlation = np.corrcoef(sentiment_scores, qos_metrics)[0, 1]
        return correlation
    
    def evaluate_satisfaction_impact(self, satisfaction_scores, qos_metrics):
        """Analyze the impact of satisfaction on QoS metrics."""
        correlation = np.corrcoef(satisfaction_scores, qos_metrics)[0, 1]
        return correlation
    
    def generate_qos_report(self, actual_qos, predicted_qos, sentiment_scores, satisfaction_scores):
        """Generate a comprehensive QoS evaluation report."""
        # Calculate QoS metrics
        qos_metrics = self.evaluate_qos_metrics(actual_qos, predicted_qos)
        
        # Calculate impact correlations
        sentiment_correlation = self.evaluate_sentiment_impact(sentiment_scores, actual_qos)
        satisfaction_correlation = self.evaluate_satisfaction_impact(satisfaction_scores, actual_qos)
        
        # Generate visualizations
        self._plot_qos_comparison(actual_qos, predicted_qos)
        self._plot_sentiment_impact(sentiment_scores, actual_qos)
        self._plot_satisfaction_impact(satisfaction_scores, actual_qos)
        
        # Create report
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'qos_metrics': qos_metrics,
            'sentiment_correlation': sentiment_correlation,
            'satisfaction_correlation': satisfaction_correlation,
            'summary': self._generate_summary(qos_metrics, sentiment_correlation, satisfaction_correlation)
        }
        
        return report
    
    def _plot_qos_comparison(self, actual_qos, predicted_qos):
        """Plot actual vs predicted QoS metrics."""
        plt.figure(figsize=(10, 6))
        plt.scatter(actual_qos, predicted_qos, alpha=0.5)
        plt.plot([min(actual_qos), max(actual_qos)], [min(actual_qos), max(actual_qos)], 'r--')
        plt.xlabel('Actual QoS')
        plt.ylabel('Predicted QoS')
        plt.title('Actual vs Predicted QoS')
        plt.savefig('reports/qos_comparison.png')
        plt.close()
    
    def _plot_sentiment_impact(self, sentiment_scores, qos_metrics):
        """Plot sentiment impact on QoS metrics."""
        plt.figure(figsize=(10, 6))
        sns.regplot(x=sentiment_scores, y=qos_metrics)
        plt.xlabel('Sentiment Score')
        plt.ylabel('QoS Metric')
        plt.title('Sentiment Impact on QoS')
        plt.savefig('reports/sentiment_impact.png')
        plt.close()
    
    def _plot_satisfaction_impact(self, satisfaction_scores, qos_metrics):
        """Plot satisfaction impact on QoS metrics."""
        plt.figure(figsize=(10, 6))
        sns.regplot(x=satisfaction_scores, y=qos_metrics)
        plt.xlabel('Satisfaction Score')
        plt.ylabel('QoS Metric')
        plt.title('Satisfaction Impact on QoS')
        plt.savefig('reports/satisfaction_impact.png')
        plt.close()
    
    def _generate_summary(self, qos_metrics, sentiment_correlation, satisfaction_correlation):
        """Generate a human-readable summary of the evaluation results."""
        summary = f"""
QoS Evaluation Summary
---------------------
QoS Metrics:
- Mean Squared Error: {qos_metrics['mse']:.4f}
- Root Mean Squared Error: {qos_metrics['rmse']:.4f}
- Mean Absolute Error: {qos_metrics['mae']:.4f}
- R² Score: {qos_metrics['r2']:.4f}

Impact Analysis:
- Sentiment Correlation: {sentiment_correlation:.4f}
- Satisfaction Correlation: {satisfaction_correlation:.4f}

Interpretation:
- The model's prediction accuracy is {'good' if qos_metrics['r2'] > 0.7 else 'moderate' if qos_metrics['r2'] > 0.5 else 'poor'}
- Sentiment {'strongly' if abs(sentiment_correlation) > 0.7 else 'moderately' if abs(sentiment_correlation) > 0.5 else 'weakly'} correlates with QoS
- Satisfaction {'strongly' if abs(satisfaction_correlation) > 0.7 else 'moderately' if abs(satisfaction_correlation) > 0.5 else 'weakly'} correlates with QoS
"""
        return summary
    
    def save_report(self, report, filename='qos_evaluation_report.txt'):
        """Save the evaluation report to a file."""
        with open(f'reports/{filename}', 'w') as f:
            f.write(f"QoS Evaluation Report\n")
            f.write(f"Generated at: {report['timestamp']}\n")
            f.write("\nQoS Metrics:\n")
            for metric, value in report['qos_metrics'].items():
                f.write(f"- {metric.upper()}: {value:.4f}\n")
            f.write(f"\nSentiment Correlation: {report['sentiment_correlation']:.4f}\n")
            f.write(f"Satisfaction Correlation: {report['satisfaction_correlation']:.4f}\n")
            f.write("\nSummary:\n")
            f.write(report['summary']) 