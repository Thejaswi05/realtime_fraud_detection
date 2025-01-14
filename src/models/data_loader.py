import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_path: str = "data/creditcard.csv"):
        self.data_path = data_path
        
    def load_data(self) -> pd.DataFrame:
        """Load the credit card fraud dataset"""
        try:
            logger.info(f"Loading data from {self.data_path}")
            if not Path(self.data_path).exists():
                raise FileNotFoundError(f"Data file not found at {self.data_path}")
                
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded dataset with shape: {df.shape}")
            
            # Data quality checks
            logger.info("Performing data quality checks...")
            logger.info(f"Missing values per column:\n{df.isnull().sum()}")
            logger.info(f"Data types:\n{df.dtypes}")
            
            # Basic data validation
            required_columns = ['Amount', 'Time'] + [f'V{i}' for i in range(1, 29)] + ['Class']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        try:
            # First split: train + validation, test
            train_val, test = train_test_split(
                df, 
                test_size=test_size,
                stratify=df['Class'],
                random_state=42
            )
            
            # Second split: train, validation
            train, val = train_test_split(
                train_val,
                test_size=0.2,
                stratify=train_val['Class'],
                random_state=42
            )
            
            logger.info(f"Train set shape: {train.shape}")
            logger.info(f"Validation set shape: {val.shape}")
            logger.info(f"Test set shape: {test.shape}")
            
            return {
                'train': train,
                'val': val,
                'test': test
            }
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise 