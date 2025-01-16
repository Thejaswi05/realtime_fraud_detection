import pandas as pd
import numpy as np
import mlflow
from datetime import datetime, timedelta
from pathlib import Path
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    DatasetCorrelationsMetric,
    ClassificationQualityMetric
)
from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvidenceDemo:
    def __init__(self):
        """Initialize demo with model and data"""
        mlflow.set_tracking_uri("http://mlflow:5001")
        self.model = mlflow.sklearn.load_model("models:/fraud_detection_model/latest")
        self.data = pd.read_csv("data/creditcard.csv")
        
    def get_feature_names(self):
        """Get all feature names in correct order"""
        return ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

    def create_drift_scenarios(self):
        """Create multiple drift scenarios"""
        scenarios = {}
        
        # Use all features
        features = self.get_feature_names()
        base_data = self.data[features + ['Class']].copy()
        
        # Scenario 1: Amount Drift
        reference_data = base_data.copy()
        current_data = base_data.copy()
        current_data['Amount'] = current_data['Amount'] * 1.5
        scenarios['amount_drift'] = (reference_data, current_data)
        
        # Scenario 2: Feature Distribution Drift
        reference_data = base_data.copy()
        current_data = base_data.copy()
        current_data['V1'] = current_data['V1'] + np.random.normal(2, 0.5, len(current_data))
        current_data['V2'] = current_data['V2'] * 0.7
        scenarios['feature_drift'] = (reference_data, current_data)
        
        return scenarios

    def generate_drift_report(self, reference_data, current_data, scenario_name):
        """Generate comprehensive drift report"""
        logger.info(f"Generating drift report for scenario: {scenario_name}")
        
        try:
            # Get features in correct order
            features = self.get_feature_names()
            
            # Prepare data with correct feature order
            reference_features = reference_data[features].copy()
            current_features = current_data[features].copy()
            
            # Generate predictions
            reference_data['prediction'] = self.model.predict(reference_features)
            current_data['prediction'] = self.model.predict(current_features)
            
            column_mapping = ColumnMapping(
                target="Class",
                prediction="prediction",
                numerical_features=features  # Use all features
            )
            
            # Create report
            report = Report(metrics=[
                DataDriftPreset(),
                TargetDriftPreset(),
                ClassificationQualityMetric(),
                ColumnDriftMetric(column_name="Amount"),
                ColumnDriftMetric(column_name="V1"),
                ColumnDriftMetric(column_name="V2")
            ])
            
            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )
            
            report_path = f"reports/drift_report_{scenario_name}.html"
            report.save_html(report_path)
            logger.info(f"Report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating report for {scenario_name}: {str(e)}")
            raise

    def run_demo(self):
        """Run complete evidently demo"""
        try:
            # Create reports directory
            Path("reports").mkdir(exist_ok=True)
            
            # Generate scenarios
            scenarios = self.create_drift_scenarios()
            
            # Generate reports for each scenario
            for scenario_name, (reference_data, current_data) in scenarios.items():
                self.generate_drift_report(reference_data, current_data, scenario_name)
            
            logger.info("Demo completed successfully!")
            logger.info("Reports available in the reports directory:")
            logger.info("- reports/drift_report_amount_drift.html")
            logger.info("- reports/drift_report_feature_drift.html")
            logger.info("- reports/drift_report_target_drift.html")
            logger.info("- reports/test_results_*.html")
            
        except Exception as e:
            logger.error(f"Error running demo: {str(e)}")
            raise

if __name__ == "__main__":
    demo = EvidenceDemo()
    demo.run_demo() 