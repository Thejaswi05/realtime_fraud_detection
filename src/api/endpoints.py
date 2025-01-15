from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import pandas as pd
import numpy as np
import time
import uuid
from datetime import datetime
import logging
from src.monitoring.monitoring import log_prediction
from .models import TransactionRequest, PredictionResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_feature_names():
    """Get feature names in correct order"""
    return ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# Load model on startup
@app.on_event("startup")
async def startup_event():
    global model
    try:
        mlflow.set_tracking_uri("http://mlflow:5001")
        model = mlflow.sklearn.load_model("models:/fraud_detection_model/latest")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    transaction: TransactionRequest,
):
    """Make fraud prediction on a transaction"""
    start_time = time.time()
    
    try:
        # Generate prediction ID
        prediction_id = str(uuid.uuid4())
        
        # Prepare features (similar to playground)
        feature_names = get_feature_names()
        input_data = {name: 0.0 for name in feature_names}
        
        # Update with actual values
        input_data['Amount'] = float(transaction.amount)
        # Convert timestamp to hours since midnight
        hours_since_midnight = transaction.timestamp.hour + transaction.timestamp.minute/60
        input_data['Time'] = float(hours_since_midnight * 3600)  # Convert to seconds
        
        # Generate random values for V1-V28 (same as playground)
        for i in range(1, 29):
            input_data[f'V{i}'] = float(np.random.normal(0, 1))
        
        # Create DataFrame with correct feature order
        features = pd.DataFrame([input_data])[feature_names]
        
        # Make prediction
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Create response
        response = PredictionResponse(
            prediction=prediction,
            fraud_probability=probability,
            prediction_id=prediction_id,
            processing_time_ms=processing_time
        )
        
        # Prepare transaction data for logging
        transaction_dict = {
            "amount": float(transaction.amount),
            "timestamp": transaction.timestamp.isoformat(),
            "time_seconds": input_data['Time']
        }
        
        # Log prediction for monitoring
        log_prediction(
            prediction_id=prediction_id,
            features=features,
            prediction=prediction,
            probability=probability,
            processing_time=processing_time,
            transaction=transaction_dict
        )
        
        logger.info(f"Prediction {prediction_id}: {'Fraud' if prediction == 1 else 'Legitimate'} "
                   f"(probability: {probability:.3f}, processing time: {processing_time:.2f}ms)")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
