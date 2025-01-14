import mlflow
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import streamlit as st
import time
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class MetricMonitor:
    def __init__(self, experiment_name: str = "fraud_detection"):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri("http://mlflow:5001")
        
    def get_recent_metrics(self, days: int = 7) -> pd.DataFrame:
        """Get metrics from recent runs"""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        
        # Get runs from last N days
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"attributes.start_time >= {start_time}"
        )
        
        return runs
        
    def plot_metric_trend(self, metric_name: str = "val_roc_auc"):
        """Plot trend of a metric over time"""
        runs_df = self.get_recent_metrics()
        
        if f"metrics.{metric_name}" not in runs_df.columns:
            logger.warning(f"Metric {metric_name} not found in runs")
            return None
            
        fig = px.line(
            runs_df,
            x="start_time",
            y=f"metrics.{metric_name}",
            title=f"{metric_name} Trend Over Time"
        )
        
        return fig
        
    def get_metric_statistics(self, metric_name: str = "val_roc_auc") -> Dict:
        """Get statistical summary of a metric"""
        runs_df = self.get_recent_metrics()
        metric_values = runs_df[f"metrics.{metric_name}"]
        
        return {
            "mean": metric_values.mean(),
            "std": metric_values.std(),
            "min": metric_values.min(),
            "max": metric_values.max(),
            "last_value": metric_values.iloc[-1] if len(metric_values) > 0 else None
        } 