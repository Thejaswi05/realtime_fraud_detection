import pandas as pd
import numpy as np
from typing import List
import mlflow.pyfunc

class FeatureProcessor(mlflow.pyfunc.PythonModel):
    """Feature processor that follows MLflow's PythonModel interface"""
    
    def __init__(self):
        self.feature_columns: List[str] = None
        
    def fit(self, df: pd.DataFrame) -> 'FeatureProcessor':
        """Fit the processor by storing feature columns"""
        self.feature_columns = [col for col in df.columns if col != 'Class']
        return self
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training or inference"""
        if self.feature_columns is None:
            self.feature_columns = [col for col in df.columns if col != 'Class']
        
        # Ensure all required columns are present
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Return only the required columns in the correct order
        return df[self.feature_columns].copy()
    
    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """MLflow PythonModel interface for prediction"""
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        # Simply call prepare_features and return the result
        return self.prepare_features(model_input)