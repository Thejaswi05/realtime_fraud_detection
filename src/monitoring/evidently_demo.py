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
        logger.info(f"Loaded data with {len(self.data)} rows")
        
    def get_feature_names(self):
        """Get all feature names in correct order"""
        return ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

    def create_drift_scenarios(self):
        """Create meaningful drift scenarios"""
        scenarios = {}
        features = self.get_feature_names()
        
        # Scenario 1: Time-based split (first half vs second half)
        logger.info("Creating time-based drift scenario...")
        median_time = self.data['Time'].median()
        reference_data = self.data[self.data['Time'] <= median_time].copy()
        current_data = self.data[self.data['Time'] > median_time].copy()
        scenarios['time_drift'] = (reference_data, current_data)
        
        # Scenario 2: Amount distribution shift
        logger.info("Creating amount distribution drift scenario...")
        amount_threshold = self.data['Amount'].quantile(0.7)
        reference_data = self.data[self.data['Amount'] <= amount_threshold].copy()
        current_data = self.data[self.data['Amount'] > amount_threshold].copy()
        # Artificially increase amounts in current data
        current_data['Amount'] = current_data['Amount'] * 1.5
        scenarios['amount_distribution_drift'] = (reference_data, current_data)
        
        # Scenario 3: Target distribution drift (fraud ratio change)
        logger.info("Creating target distribution drift scenario...")
        reference_data = self.data.copy()
        current_data = self.data.copy()
        
        # Increase fraud ratio in current data
        fraud_indices = current_data[current_data['Class'] == 1].index
        non_fraud_indices = current_data[current_data['Class'] == 0].index
        
        # Randomly select some non-fraud transactions and make them fraud
        num_to_change = len(fraud_indices) * 2  # Double the fraud cases
        indices_to_change = np.random.choice(non_fraud_indices, num_to_change, replace=False)
        current_data.loc[indices_to_change, 'Class'] = 1
        
        scenarios['target_distribution_drift'] = (reference_data, current_data)
        
        # Scenario 4: Feature correlation drift
        logger.info("Creating feature correlation drift scenario...")
        reference_data = self.data.copy()
        current_data = self.data.copy()
        
        # Modify correlations between V1-V5
        for i in range(1, 6):
            current_data[f'V{i}'] = current_data[f'V{i}'] + (
                0.3 * current_data['Amount'] * np.random.normal(0, 1, len(current_data))
            )
        
        scenarios['feature_correlation_drift'] = (reference_data, current_data)
        
        # Log scenario statistics
        for name, (ref, curr) in scenarios.items():
            logger.info(f"\nScenario: {name}")
            logger.info(f"Reference data shape: {ref.shape}")
            logger.info(f"Current data shape: {curr.shape}")
            logger.info(f"Reference fraud ratio: {ref['Class'].mean():.4f}")
            logger.info(f"Current fraud ratio: {curr['Class'].mean():.4f}")
        
        return scenarios

    def generate_drift_report(self, reference_data, current_data, scenario_name):
        """Generate comprehensive drift report"""
        logger.info(f"Generating drift report for scenario: {scenario_name}")
        
        try:
            features = self.get_feature_names()
            
            # Add predictions
            reference_features = reference_data[features].copy()
            current_features = current_data[features].copy()
            
            reference_data['prediction'] = self.model.predict(reference_features)
            current_data['prediction'] = self.model.predict(current_features)
            
            # Calculate and log prediction statistics
            ref_fraud_rate = reference_data['prediction'].mean()
            curr_fraud_rate = current_data['prediction'].mean()
            logger.info(f"Reference prediction fraud rate: {ref_fraud_rate:.4f}")
            logger.info(f"Current prediction fraud rate: {curr_fraud_rate:.4f}")
            
            column_mapping = ColumnMapping(
                target="Class",
                prediction="prediction",
                numerical_features=features
            )
            
            report = Report(metrics=[
                DataDriftPreset(),
                TargetDriftPreset(),
                ClassificationQualityMetric(),
                DatasetDriftMetric(),
                DatasetCorrelationsMetric(),
                # Add specific column drift metrics
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