import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from src.models.features import FeatureProcessor
from src.models.data_loader import DataLoader
import time

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.feature_processor = FeatureProcessor()
        self.data_loader = DataLoader()
        self.model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        
    def wait_for_mlflow(self, timeout: int = 60, interval: int = 5):
        """Wait for MLflow server to be ready"""
        mlflow_url = "http://mlflow:5001"
        start_time = time.time()
        
        while True:
            try:
                response = requests.get(f"{mlflow_url}/health")
                if response.status_code == 200:
                    logger.info("MLflow server is ready")
                    break
            except requests.exceptions.RequestException:
                if time.time() - start_time > timeout:
                    raise TimeoutError("MLflow server did not become ready in time")
                logger.info("Waiting for MLflow server...")
                time.sleep(interval)

    def _save_artifacts(self) -> Dict[str, str]:
        """Save model and feature processor artifacts"""
        # Create temporary directory for artifacts
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = {}
            
            # Save model
            model_path = os.path.join(tmp_dir, "model.pkl")
            joblib.dump(self.model, model_path)
            artifacts['model_path'] = model_path
            
            # Save feature processor
            processor_path = os.path.join(tmp_dir, "feature_processor.pkl")
            joblib.dump(self.feature_processor, processor_path)
            artifacts['feature_processor_path'] = processor_path
            
            # Save classification reports
            val_report_path = os.path.join(tmp_dir, "validation_report.txt")
            with open(val_report_path, 'w') as f:
                f.write(self.val_classification_report)
            artifacts['validation_report_path'] = val_report_path
            
            test_report_path = os.path.join(tmp_dir, "test_report.txt")
            with open(test_report_path, 'w') as f:
                f.write(self.test_classification_report)
            artifacts['test_report_path'] = test_report_path
            
            return artifacts

    def _log_to_mlflow(self, metrics: Dict[str, float]):
        """Log metrics and model to MLflow using direct logging APIs"""
        try:
            self.wait_for_mlflow()
            
            mlflow.set_tracking_uri("http://mlflow:5001")
            mlflow.set_experiment("fraud_detection")
            
            # Create structured run name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"fraud_detection_v1_{timestamp}"
            
            with mlflow.start_run(run_name=run_name) as run:
                # Add structured tags
                mlflow.set_tags({
                    "version": "v1",
                    "model_type": "logistic_regression",
                    "data_version": "v1",
                    "features_version": "v1",
                    "environment": "development",
                    "run_type": "experiment",
                    "author": "data_scientist_1"
                })
                
                # Log parameters
                mlflow.log_params(self.model.get_params())
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(
                            metric_name, 
                            metric_value,
                            step=int(datetime.now().timestamp())
                        )
                
                # Log reports directly as text
                logger.info("Logging reports...")
                mlflow.log_text(
                    self.val_classification_report,
                    "reports/validation_report.txt"
                )
                mlflow.log_text(
                    self.test_classification_report,
                    "reports/test_report.txt"
                )
                
                # Log model with signature
                logger.info("Logging model with signature...")
                X_sample = self.feature_processor.prepare_features(
                    self.data_loader.load_data().head(1)
                )
                y_pred = self.model.predict(X_sample)
                signature = infer_signature(X_sample, y_pred)
                
                # Log model to registry
                mlflow.sklearn.log_model(
                    sk_model=self.model,
                    artifact_path="model",
                    signature=signature,
                    registered_model_name="fraud_detection_model"
                )
                
                # Log feature processor
                logger.info("Logging feature processor...")
                mlflow.pyfunc.log_model(
                    artifact_path="feature_processor",
                    python_model=self.feature_processor,
                    registered_model_name="fraud_detection_feature_processor"
                )
                
                # Get run ID
                run_id = run.info.run_id
                logger.info(f"MLflow run ID: {run_id}")
                logger.info(f"Artifact URI: {mlflow.get_artifact_uri()}")
                
                return run_id
                
        except Exception as e:
            logger.error(f"Error logging to MLflow: {str(e)}")
            logger.exception("Detailed traceback:")
            raise

    def train(self) -> Dict[str, Any]:
        """Main training pipeline with proper MLflow run context"""
        try:
            # Start MLflow run at the beginning of training
            self.wait_for_mlflow()
            mlflow.set_tracking_uri("http://mlflow:5001")
            mlflow.set_experiment("fraud_detection")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"fraud_detection_v1_{timestamp}"
            
            # Add active_run check to prevent duplicate runs
            if mlflow.active_run():
                logger.warning("Found active run. Ending it.")
                mlflow.end_run()
            
            with mlflow.start_run(run_name=run_name) as run:
                # Load and split data
                logger.info("Loading data...")
                df = self.data_loader.load_data()
                data_splits = self.data_loader.split_data(df)
                
                # Process features
                logger.info("Processing features...")
                X_train = self.feature_processor.prepare_features(data_splits['train'])
                X_val = self.feature_processor.prepare_features(data_splits['val'])
                X_test = self.feature_processor.prepare_features(data_splits['test'])
                
                y_train = data_splits['train']['Class']
                y_val = data_splits['val']['Class']
                y_test = data_splits['test']['Class']
                
                # Train model
                logger.info("Training model...")
                self.model.fit(X_train, y_train)
                
                # Evaluate model
                metrics = self._evaluate_model(X_val, y_val, X_test, y_test)
                
                # Log parameters
                mlflow.log_params(self.model.get_params())
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(
                            metric_name, 
                            metric_value,
                            step=int(datetime.now().timestamp())
                        )
                
                # Log reports
                logger.info("Logging reports...")
                mlflow.log_text(
                    self.val_classification_report,
                    "reports/validation_report.txt"
                )
                mlflow.log_text(
                    self.test_classification_report,
                    "reports/test_report.txt"
                )
                
                # Log model with signature
                logger.info("Logging model with signature...")
                X_sample = self.feature_processor.prepare_features(
                    self.data_loader.load_data().head(1)
                )
                y_pred = self.model.predict(X_sample)
                signature = infer_signature(X_sample, y_pred)
                
                # Log model to registry
                mlflow.sklearn.log_model(
                    sk_model=self.model,
                    artifact_path="model",
                    signature=signature,
                    registered_model_name="fraud_detection_model"
                )
                
                # Log feature processor
                logger.info("Logging feature processor...")
                mlflow.pyfunc.log_model(
                    artifact_path="feature_processor",
                    python_model=self.feature_processor,
                    registered_model_name="fraud_detection_feature_processor"
                )
                
                run_id = run.info.run_id
                logger.info(f"MLflow run ID: {run_id}")
                logger.info(f"Artifact URI: {mlflow.get_artifact_uri()}")
                
                return {
                    'metrics': metrics,
                    'run_id': run_id
                }
                
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            logger.exception("Detailed traceback:")
            if mlflow.active_run():
                mlflow.end_run()
            raise

    def _evaluate_model(self, X_val, y_val, X_test, y_test) -> Dict[str, float]:
        """Evaluate model performance"""
        metrics = {}
        
        try:
            # Validation metrics
            val_pred = self.model.predict(X_val)
            val_prob = self.model.predict_proba(X_val)[:, 1]
            
            metrics['val_roc_auc'] = float(roc_auc_score(y_val, val_prob))
            
            # Store classification report separately
            self.val_classification_report = classification_report(y_val, val_pred)
            
            # Test metrics
            test_pred = self.model.predict(X_test)
            test_prob = self.model.predict_proba(X_test)[:, 1]
            
            metrics['test_roc_auc'] = float(roc_auc_score(y_test, test_prob))
            
            # Store classification report separately
            self.test_classification_report = classification_report(y_test, test_pred)
            
            # Add numerical metrics only
            metrics['val_accuracy'] = float((y_val == val_pred).mean())
            metrics['test_accuracy'] = float((y_test == test_pred).mean())
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = ModelTrainer()
    results = trainer.train()
    logger.info(f"Training completed. Validation ROC-AUC: {results['metrics']['val_roc_auc']:.4f}")