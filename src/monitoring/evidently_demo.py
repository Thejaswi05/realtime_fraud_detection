import pandas as pd
import numpy as np
import mlflow
from datetime import datetime
from pathlib import Path
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    TargetDriftPreset,
    DataQualityPreset,
    ClassificationPreset
)
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetCorrelationsMetric,
    ColumnDriftMetric
)
import shap
import matplotlib.pyplot as plt
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

    def create_meaningful_drift(self):
        """Create meaningful drift scenarios based on business-relevant amount thresholds"""
        logger.info("Creating meaningful drift scenarios...")
        
        # Analyze amount distribution
        amount_stats = {
            'min': self.data['Amount'].min(),
            'max': self.data['Amount'].max(),
            'mean': self.data['Amount'].mean(),
            'median': self.data['Amount'].median(),
            'p25': self.data['Amount'].quantile(0.25),
            'p75': self.data['Amount'].quantile(0.75)
        }
        
        logger.info("Transaction Amount Distribution:")
        for metric, value in amount_stats.items():
            logger.info(f"{metric}: ${value:.2f}")
        
        # Define business-meaningful thresholds
        SMALL_TRANSACTION_THRESHOLD = 100  # Transactions <= $100
        LARGE_TRANSACTION_THRESHOLD = 500  # Transactions > $500
        
        # Split data based on amount thresholds
        reference_data = self.data[self.data['Amount'] <= SMALL_TRANSACTION_THRESHOLD].copy()
        current_data = self.data[self.data['Amount'] > LARGE_TRANSACTION_THRESHOLD].copy()
        
        # Log insights about the splits
        logger.info("\nData Split Summary:")
        logger.info(f"Small Transactions (≤ ${SMALL_TRANSACTION_THRESHOLD}):")
        logger.info(f"- Count: {len(reference_data)}")
        logger.info(f"- Average Amount: ${reference_data['Amount'].mean():.2f}")
        logger.info(f"- Fraud Rate: {(reference_data['Class'] == 1).mean()*100:.2f}%")
        
        logger.info(f"\nLarge Transactions (> ${LARGE_TRANSACTION_THRESHOLD}):")
        logger.info(f"- Count: {len(current_data)}")
        logger.info(f"- Average Amount: ${current_data['Amount'].mean():.2f}")
        logger.info(f"- Fraud Rate: {(current_data['Class'] == 1).mean()*100:.2f}%")
        
        return reference_data, current_data

    def log_split_insights(self, reference_data, current_data):
        """Log meaningful insights about the data split"""
        logger.info("\n=== Data Split Insights ===")
        logger.info("Reference Data (Small Transactions):")
        logger.info(f"- Size: {len(reference_data)} transactions")
        logger.info(f"- Amount Range: ${reference_data['Amount'].min():.2f} - ${reference_data['Amount'].max():.2f}")
        logger.info(f"- Fraud Rate: {reference_data['Class'].mean()*100:.2f}%")
        
        logger.info("\nCurrent Data (Large Transactions):")
        logger.info(f"- Size: {len(current_data)} transactions")
        logger.info(f"- Amount Range: ${current_data['Amount'].min():.2f} - ${current_data['Amount'].max():.2f}")
        logger.info(f"- Fraud Rate: {current_data['Class'].mean()*100:.2f}%")

    def generate_drift_report(self, reference_data, current_data):
        """Generate comprehensive drift report with explanations"""
        try:
            features = self.get_feature_names()
            
            # Add predictions
            reference_features = reference_data[features].copy()
            current_features = current_data[features].copy()
            
            reference_data['prediction'] = self.model.predict_proba(reference_features)[:, 1]
            current_data['prediction'] = self.model.predict_proba(current_features)[:, 1]
            
            column_mapping = ColumnMapping(
                target="Class",
                prediction="prediction",
                numerical_features=features
            )
            
            # Data Quality and Drift Report
            data_report = Report(metrics=[
                DataQualityPreset(),
                DataDriftPreset(),
                DatasetDriftMetric(),
                DatasetCorrelationsMetric(),
                ColumnDriftMetric(column_name="Amount"),
                ColumnDriftMetric(column_name="V1"),
                ColumnDriftMetric(column_name="V2")
            ])
            
            data_report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )
            data_report.save_html("reports/data_quality_and_drift_report.html")
            
            # Target Drift Report
            target_report = Report(metrics=[
                TargetDriftPreset(),
                ClassificationPreset()
            ])
            
            target_report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )
            target_report.save_html("reports/target_and_performance_report.html")
            
            # Save SHAP analysis separately
            self.generate_shap_analysis(reference_data[features], current_data[features])
            
            # Create summary HTML
            self.create_summary_html()
            
        except Exception as e:
            logger.error(f"Error generating reports: {str(e)}")
            raise

    def create_summary_html(self):
        """Create a summary HTML file linking to all reports"""
        summary_html = """
        <html>
        <head>
            <title>Fraud Detection Model Monitoring</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .report-section { margin-bottom: 30px; }
                h1 { color: #2c3e50; }
                h2 { color: #34495e; }
                .description { color: #7f8c8d; margin-bottom: 10px; }
                a { color: #3498db; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>Fraud Detection Model Monitoring Reports</h1>
            
            <div class="report-section">
                <h2>1. Data Quality and Drift Analysis</h2>
                <div class="description">
                    This report shows how the data distribution has changed between small and large transactions:
                    <ul>
                        <li>Feature distribution changes</li>
                        <li>Data quality metrics</li>
                        <li>Correlation changes</li>
                    </ul>
                </div>
                <a href="data_quality_and_drift_report.html">View Data Quality and Drift Report</a>
            </div>
            
            <div class="report-section">
                <h2>2. Target and Model Performance Analysis</h2>
                <div class="description">
                    This report shows how the fraud patterns and model performance differ:
                    <ul>
                        <li>Target distribution changes</li>
                        <li>Model performance metrics</li>
                        <li>Classification quality analysis</li>
                    </ul>
                </div>
                <a href="target_and_performance_report.html">View Target and Performance Report</a>
            </div>
            
            <div class="report-section">
                <h2>3. SHAP Feature Importance Analysis</h2>
                <div class="description">
                    These plots show how different features impact the model's predictions:
                    <ul>
                        <li>Feature importance ranking</li>
                        <li>Impact magnitude and direction</li>
                        <li>Comparison between reference and current data</li>
                    </ul>
                </div>
                <img src="shap_summary_reference.png" alt="SHAP Summary - Reference Data" style="max-width: 100%; margin: 10px 0;">
                <img src="shap_summary_current.png" alt="SHAP Summary - Current Data" style="max-width: 100%; margin: 10px 0;">
            </div>
        </body>
        </html>
        """
        
        with open("reports/index.html", "w") as f:
            f.write(summary_html)
        
        logger.info("Created summary HTML at reports/index.html")

    def generate_shap_analysis(self, reference_features, current_data):
        """Generate SHAP analysis plots"""
        try:
            logger.info("Generating SHAP analysis...")
            
            # Create directory for plots if it doesn't exist
            Path("reports/plots").mkdir(parents=True, exist_ok=True)
            
            # Sample data if too large (for performance)
            n_samples = min(1000, len(reference_features))
            ref_sample = reference_features.sample(n_samples, random_state=42)
            curr_sample = current_data.sample(n_samples, random_state=42)
            
            # Create SHAP explainer for linear models
            background = shap.maskers.Independent(ref_sample, max_samples=100)
            explainer = shap.LinearExplainer(self.model, background)
            
            # Reference data SHAP analysis
            logger.info("Generating reference data SHAP plot...")
            plt.figure(figsize=(12, 8))
            shap_values_ref = explainer.shap_values(ref_sample)
            if isinstance(shap_values_ref, list):
                shap_values_ref = shap_values_ref[0]  # For binary classification
            shap.summary_plot(shap_values_ref, ref_sample, show=False)
            plt.title("SHAP Feature Importance - Reference Data (Small Transactions)")
            plt.tight_layout()
            plt.savefig("reports/plots/shap_summary_reference.png", bbox_inches='tight', dpi=300)
            plt.close()
            
            # Current data SHAP analysis
            logger.info("Generating current data SHAP plot...")
            plt.figure(figsize=(12, 8))
            shap_values_curr = explainer.shap_values(curr_sample)
            if isinstance(shap_values_curr, list):
                shap_values_curr = shap_values_curr[0]  # For binary classification
            shap.summary_plot(shap_values_curr, curr_sample, show=False)
            plt.title("SHAP Feature Importance - Current Data (Large Transactions)")
            plt.tight_layout()
            plt.savefig("reports/plots/shap_summary_current.png", bbox_inches='tight', dpi=300)
            plt.close()
            
            # Generate feature importance comparison
            logger.info("Generating feature importance comparison...")
            plt.figure(figsize=(12, 8))
            
            # Calculate mean absolute SHAP values for both datasets
            mean_shap_ref = np.abs(shap_values_ref).mean(axis=0)
            mean_shap_curr = np.abs(shap_values_curr).mean(axis=0)
            
            # Create comparison plot
            feature_names = reference_features.columns
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Reference': mean_shap_ref,
                'Current': mean_shap_curr
            })
            importance_df = importance_df.sort_values('Reference', ascending=True)
            
            # Plot
            plt.barh(range(len(feature_names)), importance_df['Reference'], alpha=0.6, label='Reference')
            plt.barh(range(len(feature_names)), importance_df['Current'], alpha=0.6, label='Current')
            plt.yticks(range(len(feature_names)), importance_df['Feature'])
            plt.xlabel('Mean |SHAP value|')
            plt.title('Feature Importance Comparison')
            plt.legend()
            plt.tight_layout()
            plt.savefig("reports/plots/feature_importance_comparison.png", bbox_inches='tight', dpi=300)
            plt.close()
            
            # Generate HTML report for SHAP analysis
            shap_html = f"""
            <html>
            <head>
                <title>SHAP Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .plot-container {{ margin: 20px 0; }}
                    .description {{ color: #666; margin: 10px 0; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #34495e; }}
                </style>
            </head>
            <body>
                <h1>SHAP Feature Importance Analysis</h1>
                
                <div class="plot-container">
                    <h2>Feature Importance Comparison</h2>
                    <div class="description">
                        This plot compares feature importance between small and large transactions:
                        <ul>
                            <li>Longer bars indicate more important features</li>
                            <li>Differences between reference and current show how importance shifts</li>
                            <li>Features are ordered by importance in reference data</li>
                        </ul>
                    </div>
                    <img src="plots/feature_importance_comparison.png" alt="Feature Importance Comparison" style="max-width: 100%;">
                </div>
                
                <div class="plot-container">
                    <h2>Small Transactions (Reference Data)</h2>
                    <div class="description">
                        SHAP values show how each feature impacts predictions for small transactions.
                    </div>
                    <img src="plots/shap_summary_reference.png" alt="SHAP Summary - Reference Data" style="max-width: 100%;">
                </div>
                
                <div class="plot-container">
                    <h2>Large Transactions (Current Data)</h2>
                    <div class="description">
                        SHAP values show how each feature impacts predictions for large transactions.
                    </div>
                    <img src="plots/shap_summary_current.png" alt="SHAP Summary - Current Data" style="max-width: 100%;">
                </div>
                
                <div class="description">
                    <h2>How to Interpret These Plots</h2>
                    <ul>
                        <li>Red colors indicate higher feature values, blue indicates lower values</li>
                        <li>Position on x-axis shows impact on prediction (negative = lower fraud probability, positive = higher)</li>
                        <li>Comparing reference vs current shows how feature importance changes with transaction size</li>
                        <li>The comparison plot directly shows importance shifts between datasets</li>
                    </ul>
                </div>
            </body>
            </html>
            """
            
            with open("reports/shap_analysis.html", "w") as f:
                f.write(shap_html)
            
            logger.info("SHAP analysis completed and saved")
            
        except Exception as e:
            logger.error(f"Error generating SHAP analysis: {str(e)}")
            raise

    def run_demo(self):
        """Run complete evidently demo"""
        try:
            reference_data, current_data = self.create_meaningful_drift()
            self.generate_drift_report(reference_data, current_data)
            
            logger.info("\n=== Reports Generated ===")
            logger.info("1. Data Quality and Drift Report:")
            logger.info("   - Shows data quality metrics")
            logger.info("   - Shows feature distribution changes")
            logger.info("   - Shows correlation changes")
            
            logger.info("\n2. Model Performance Report:")
            logger.info("   - Shows classification metrics")
            logger.info("   - Shows probability calibration")
            logger.info("   - Shows performance differences")
            
            logger.info("\n3. SHAP Analysis:")
            logger.info("   - Shows feature importance")
            logger.info("   - Compares feature impacts between datasets")
            
        except Exception as e:
            logger.error(f"Error running demo: {str(e)}")
            raise

if __name__ == "__main__":
    demo = EvidenceDemo()
    demo.run_demo() 