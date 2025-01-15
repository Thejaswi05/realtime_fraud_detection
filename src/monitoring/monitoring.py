from pathlib import Path
import pandas as pd
from datetime import datetime
import json
import logging
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *
import mlflow

logger = logging.getLogger(__name__)

# Paths
MONITORING_PATH = Path("monitoring")
PREDICTIONS_PATH = MONITORING_PATH / "predictions"
REPORTS_PATH = MONITORING_PATH / "reports"

# Create directories
MONITORING_PATH.mkdir(exist_ok=True)
PREDICTIONS_PATH.mkdir(exist_ok=True)
REPORTS_PATH.mkdir(exist_ok=True)

def log_prediction(
    prediction_id: str,
    features: pd.DataFrame,
    prediction: int,
    probability: float,
    processing_time: float,
    transaction: dict
) -> None:
    """Log prediction details for monitoring"""
    timestamp = datetime.now().isoformat()
    
    # Prepare log entry
    log_entry = {
        "prediction_id": prediction_id,
        "timestamp": timestamp,
        "features": features.to_dict(orient="records")[0],
        "prediction": prediction,
        "probability": probability,
        "processing_time_ms": processing_time,
        "transaction": transaction
    }
    
    # Save to file
    log_file = PREDICTIONS_PATH / f"{timestamp[:10]}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def get_reference_data():
    """Get reference data from MLflow or local file"""
    try:
        # Try MLflow first
        client = mlflow.tracking.MlflowClient()
        model_name = "fraud_detection_model"
        
        # Get latest model version
        latest_version = client.get_latest_versions(model_name)[0]
        run_id = latest_version.run_id
        
        # Download reference data from artifacts
        artifact_path = client.download_artifacts(run_id, "reference_data/reference_data.csv")
        reference_data = pd.read_csv(artifact_path)
        print("Loaded reference data from MLflow artifacts")
        
    except Exception as e:
        print(f"Could not load from MLflow: {e}")
        print("Trying local file...")
        
        # Fallback to local file
        try:
            reference_data = pd.read_csv("data/reference_data.csv")
            print("Loaded reference data from local file")
        except Exception as e:
            print(f"Could not load reference data: {e}")
            return None
            
    return reference_data

def generate_monitoring_report() -> None:
    """Generate Evidently AI monitoring report"""
    try:
        # Get reference data
        reference_data = get_reference_data()
        if reference_data is None:
            raise Exception("No reference data available")
            
        # Load recent predictions
        prediction_files = list(PREDICTIONS_PATH.glob("*.jsonl"))
        if not prediction_files:
            logger.warning("No prediction data available for monitoring")
            return
            
        # Load and process prediction data
        predictions = []
        for file in prediction_files[-7:]:  # Last 7 days
            with open(file) as f:
                for line in f:
                    predictions.append(json.loads(line))
        
        if not predictions:
            logger.warning("No predictions found in files")
            return
            
        current_data = pd.DataFrame([p['features'] for p in predictions])
        current_data['prediction'] = [p['prediction'] for p in predictions]
        
        # Create report
        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
            ClassificationQualityMetric(),
            DatasetMissingValuesMetric(),
            ColumnDriftMetric(column_name="Amount")
        ])
        
        report.run(reference_data=reference_data, current_data=current_data)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_PATH / f"report_{timestamp}.html"
        report.save_html(str(report_path))
        logger.info(f"Generated monitoring report: {report_path}")
        
    except Exception as e:
        logger.error(f"Error generating monitoring report: {e}")
        raise
