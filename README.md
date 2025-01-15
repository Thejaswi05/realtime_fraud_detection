# Real-time Fraud Detection System

A machine learning system for real-time credit card fraud detection using MLflow, FastAPI, Kafka, and Streamlit.

## Prerequisites
- Docker and Docker Compose
- Python 3.9+
- Git

## Installation

1. Clone the repository:


git clone <git@github.com:Thejaswi05/realtime_fraud_detection.git>

cd realtime-fraud-detection

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the System

1. Start all services:

```bash
# Clean start
docker-compose down -v
docker-compose up -d
```

2. Train the model:

```bash
docker-compose run --rm train
```

3. Access the services:
- MLflow UI: http://localhost:5001
- Model Playground: http://localhost:8501
- API Documentation: http://localhost:8000/docs

## Using the Streamlit Playground

The Streamlit playground provides an interactive interface to test the fraud detection model.

1. Access the playground at http://localhost:8501

2. Input transaction details:
   - **Amount**: Enter the transaction amount in dollars
   - **Time**: Select the transaction's position in the sequence
   
3. Click "Check for Fraud" to get predictions

4. Understand the results:
   - **Prediction**: Shows if the transaction is legitimate or fraudulent
   - **Confidence**: Shows the model's confidence in its prediction
   - **Feature Analysis**: Displays the top 10 influential features
     - Blue bars (positive values): Transaction patterns the model has learned
     - Red bars (negative values): Different transaction patterns
     - Bar length: Strength of the feature's influence
     - The combination of all features determines the final prediction

5. Feature Interpretation:
   - **Time**: Position in transaction sequence (helps detect patterns)
   - **Amount**: Transaction value
   - **V1-V28**: Anonymized transaction characteristics
   
## Development

To modify the system:

1. Update model training:

```

3. Update playground:

```bash
docker-compose restart playground
```

## Troubleshooting

1. Port conflicts:

```bash
# Stop all containers and remove volumes
docker-compose down -v
# Check for running containers
docker ps
# Kill specific ports if needed
sudo lsof -i :5001  # Check MLflow port
sudo lsof -i :8501  # Check Streamlit port
```

2. MLflow connection issues:

```bash
# Restart MLflow
docker-compose restart mlflow
# Check logs
docker-compose logs mlflow
```

3. Model loading errors:

```bash
# Ensure model is trained
docker-compose run --rm train
# Check MLflow UI for successful runs
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request


This README:
1. Provides clear installation steps
2. Explains how to run each component
3. Details the Streamlit playground usage
4. Includes troubleshooting guides
5. Explains project structure

Would you like me to:
1. Add more detailed configuration options?
2. Include example API calls?
3. Add environment variable documentation?
4. Expand the troubleshooting section?
```