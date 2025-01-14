from fastapi import FastAPI, HTTPException
import pandas as pd
from typing import Dict, Any
import mlflow

app = FastAPI()

# Load both model and feature processor
mlflow.set_tracking_uri("http://mlflow:5001")

# Get the latest versions
model = mlflow.sklearn.load_model("models:/fraud_detection_model/latest")
feature_processor = mlflow.pyfunc.load_model("models:/fraud_detection_feature_processor/latest")

@app.post("/predict")
async def predict(data: Dict[str, Any]):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data])
        
        # 1. Use feature processor to prepare features
        prepared_features = feature_processor.predict(None, df)  # predict() is our prepare_features() method
        
        # 2. Use model to make prediction
        prediction = model.predict(prepared_features)
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(model.predict_proba(prepared_features)[0][1])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
