import streamlit as st
import pandas as pd
from datetime import datetime
from src.monitoring.metric_monitor import MetricMonitor
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.metrics import ClassificationPerformanceMetric
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset

def run_combined_monitoring():
    st.title("Model Monitoring Dashboard")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Select View",
        ["MLflow Metrics", "Data Drift", "Performance Monitoring"]
    )
    
    if page == "MLflow Metrics":
        run_mlflow_monitoring()
    elif page == "Data Drift":
        run_drift_monitoring()
    else:
        run_performance_monitoring()

def run_mlflow_monitoring():
    monitor = MetricMonitor()
    
    # Metric selection
    metrics = ["val_roc_auc", "test_roc_auc", "val_accuracy", "test_accuracy"]
    selected_metric = st.selectbox("Select Metric", metrics)
    
    # Display metric statistics
    stats = monitor.get_metric_statistics(selected_metric)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Value", f"{stats['last_value']:.4f}" if stats['last_value'] else "N/A")
    with col2:
        st.metric("Mean", f"{stats['mean']:.4f}" if pd.notnull(stats['mean']) else "N/A")
    with col3:
        st.metric("Std Dev", f"{stats['std']:.4f}" if pd.notnull(stats['std']) else "N/A")
    
    # Plot trend
    fig = monitor.plot_metric_trend(selected_metric)
    if fig:
        st.plotly_chart(fig)

def run_drift_monitoring():
    st.header("Data Drift Monitoring")
    
    # Create a report with data drift metrics
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        TargetDriftPreset()
    ])
    
    st.info("Data drift monitoring is ready. Upload reference and current datasets to analyze drift.")
    
    # File upload widgets
    reference_data = st.file_uploader("Upload reference dataset (CSV)", type="csv")
    current_data = st.file_uploader("Upload current dataset (CSV)", type="csv")
    
    if reference_data and current_data:
        try:
            # Load data
            reference_df = pd.read_csv(reference_data)
            current_df = pd.read_csv(current_data)
            
            # Run the report
            report.run(reference_data=reference_df, current_data=current_df)
            
            # Display results
            st.write(report)
        except Exception as e:
            st.error(f"Error analyzing drift: {str(e)}")

def run_performance_monitoring():
    st.header("Model Performance Monitoring")
    
    # Create a report with classification metrics
    report = Report(metrics=[
        ClassificationPerformanceMetric()
    ])
    
    st.info("Performance monitoring is ready. Upload predictions dataset to analyze model performance.")
    
    # File upload widget
    predictions_data = st.file_uploader("Upload predictions dataset (CSV)", type="csv")
    
    if predictions_data:
        try:
            # Load data
            predictions_df = pd.read_csv(predictions_data)
            
            # Run the report
            report.run(reference_data=predictions_df, current_data=predictions_df)
            
            # Display results
            st.write(report)
        except Exception as e:
            st.error(f"Error analyzing performance: {str(e)}")

if __name__ == "__main__":
    run_combined_monitoring() 