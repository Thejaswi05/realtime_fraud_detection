from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

# Pydantic models are used for API request/response validation
# They are NOT machine learning models
# They help FastAPI validate data before it reaches our ML model

class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in dollars")
    timestamp: datetime = Field(default_factory=datetime.now, description="Transaction timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "amount": 125.0,
                "timestamp": "2024-02-20T10:30:00Z"
            }
        }

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 for legitimate, 1 for fraud")
    fraud_probability: float = Field(..., description="Probability of fraud")
    prediction_id: str = Field(..., description="Unique prediction identifier")
    processing_time_ms: float = Field(..., description="API processing time in milliseconds") 
