# Streaming QoS Analysis with ML

This project implements a machine learning-based approach to analyze and optimize Quality of Service (QoS) for video game streaming platforms using benchmark datasets. The solution combines BERT-based sentiment analysis and Deep Neural Networks (DNN) for satisfaction classification to provide insights into streaming quality and user engagement.

## Features

- BERT-based sentiment analysis for user feedback
- DNN-based satisfaction classification
- QoS metrics evaluation
- Data visualization and analysis tools

## Project Structure

```
.
├── data/                  # Dataset directory
├── models/               # Saved model files
├── src/
│   ├── data_processing/  # Data preprocessing scripts
│   ├── sentiment/        # BERT sentiment analysis
│   ├── classification/   # DNN satisfaction classifier
│   └── evaluation/       # QoS metrics evaluation
├── notebooks/            # Jupyter notebooks for analysis
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your benchmark dataset in the `data/` directory
2. Run the data processing pipeline:
```bash
python src/data_processing/process_data.py
```

3. Train the models:
```bash
python src/sentiment/train_sentiment.py
python src/classification/train_classifier.py
```

4. Evaluate QoS metrics:
```bash
python src/evaluation/evaluate_qos.py
```

## Dataset

The project uses benchmark datasets for streaming QoS analysis. The dataset should include:
- Streaming quality metrics (bitrate, resolution, frame rate)
- Network conditions (latency, packet loss)
- User engagement metrics
- User feedback/sentiment data

## License

MIT License 