import streamlit as st
import pandas as pd
import mlflow
import numpy as np
import plotly.express as px
import time

def load_models():
    """Load models from MLflow registry"""
    mlflow.set_tracking_uri("http://mlflow:5001")
    
    try:
        model = mlflow.sklearn.load_model("models:/fraud_detection_model/latest")
        feature_processor = mlflow.pyfunc.load_model("models:/fraud_detection_feature_processor/latest")
        return model, feature_processor
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Make sure you've trained and registered the models first.")
        return None, None

def display_prediction_details(df: pd.DataFrame, prediction: int, probability: float):
    """Display prediction results and feature analysis"""
    # Show prediction with clear visual indicators
    if prediction == 1:
        st.error("⚠️ Potential Fraud Detected!")
        st.write(f"Confidence: {probability:.1%}")
    else:
        st.success("✅ Transaction Appears Legitimate")
        st.write(f"Confidence: {1-probability:.1%}")
    
    # Display feature analysis
    st.subheader("Transaction Analysis")
    
    # Show Amount prominently
    st.metric("Transaction Amount", f"${df['Amount'].iloc[0]:.2f}")
    
    # Display top influential features
    st.write("### Key Transaction Features")
    
    # Get top 10 features by absolute value (excluding Amount)
    feature_values = df.drop('Amount', axis=1).iloc[0]
    top_features = feature_values.abs().nlargest(10)
    
    # Create a more informative visualization using plotly
    fig = px.bar(
        x=top_features.values,
        y=top_features.index,
        orientation='h',
        title='Top 10 Influential Features',
        labels={'x': 'Feature Value', 'y': 'Feature Name'},
        color=top_features.values,
        color_continuous_scale='RdYlBu'
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)
    
    # Add explanation
    st.info("""
    📊 **Feature Analysis:**
    - The chart shows the most significant transaction characteristics
    - Larger absolute values (positive or negative) have more influence
    - These features are automatically computed from transaction details
    - V1-V28 are secure, anonymized transaction patterns
    """)

def main():
    st.title("Fraud Detection Model Playground")
    
    model, feature_processor = load_models()
    
    if model is None or feature_processor is None:
        st.stop()
        
    st.info("Models loaded successfully! Enter transaction details below.")
    
    # Simple input form
    st.subheader("Transaction Details")
    amount = st.number_input(
        "Transaction Amount ($)", 
        min_value=0.0, 
        value=100.0,
        help="Enter the transaction amount"
    )
    
    if st.button("Check for Fraud"):
        try:
            # Create input DataFrame with default values
            input_data = {
                "Time": 0.0,
                "Amount": float(amount),  # Ensure amount is float
                **{f"V{i}": float(np.random.normal(0, 1)) for i in range(1, 29)}  # Random values for demo
            }
            df = pd.DataFrame([input_data])
            
            # Make prediction
            prepared_features = feature_processor.predict(None, df)
            prediction = model.predict(prepared_features)[0]  # Get first element
            probability = model.predict_proba(prepared_features)[0][1]  # Get probability of fraud
            
            # Display results and analysis
            display_prediction_details(df, prediction, probability)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
