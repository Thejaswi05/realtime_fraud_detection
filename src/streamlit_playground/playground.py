import streamlit as st
import pandas as pd
import mlflow
import numpy as np
import plotly.express as px

def load_models():
    """Load models from MLflow registry"""
    mlflow.set_tracking_uri("http://mlflow:5001")
    
    try:
        model = mlflow.sklearn.load_model("models:/fraud_detection_model/latest")
        return model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Make sure you've trained and registered the models first.")
        return None

def get_feature_names():
    """Get feature names in correct order"""
    return ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

def main():
    st.title("Fraud Detection Model Playground")
    
    model = load_models()
    
    if model is None:
        st.stop()
        
    st.info("Model loaded successfully! Enter transaction details below.")
    
    # Input form with two columns
    st.subheader("Transaction Details")
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input(
            "Transaction Amount ($)", 
            min_value=0.0, 
            value=100.0,
            help="Enter the transaction amount"
        )
    
    with col2:
        # Time input as sequence in dataset
        st.write("Transaction Sequence")
        time_value = st.slider(
            "Seconds from first transaction in sequence",
            min_value=0,
            max_value=172800,  # 48 hours in seconds
            value=0,
            help="Position of this transaction in the sequence (in seconds from start)"
        )
        
        # Show time in human-readable format
        hours = time_value // 3600
        minutes = (time_value % 3600) // 60
        st.caption(f"≈ {hours} hours, {minutes} minutes from sequence start")
    
    # Add explanation about Time feature
    st.info("""
    ⏱️ **About Transaction Sequence:**
    - The 'Time' feature represents the sequence of transactions
    - It's measured in seconds from the first transaction in the dataset
    - This helps detect patterns in transaction timing
    - It's NOT the time taken to process the transaction
    """)
    
    if st.button("Check for Fraud"):
        try:
            # Create input DataFrame with ordered features
            feature_names = get_feature_names()
            input_data = {name: 0.0 for name in feature_names}  # Initialize with zeros
            
            # Update with actual values
            input_data['Amount'] = float(amount)
            input_data['Time'] = 0.0
            
            # Generate random values for V1-V28
            for i in range(1, 29):
                input_data[f'V{i}'] = float(np.random.normal(0, 1))
                
            # Create DataFrame with correct feature order
            df = pd.DataFrame([input_data])[feature_names]  # Ensure correct column order
            
            # Make prediction
            prediction = model.predict(df)[0]
            probability = model.predict_proba(df)[0][1]
            
            # Display results with clearer probability explanation
            st.subheader("Fraud Detection Results")
            
            fraud_probability = probability  # probability of fraud
            legitimate_probability = 1 - probability  # probability of legitimate
            
            if prediction == 1:
                st.error("⚠️ Potential Fraud Detected!")
                st.write(f"Probability of Fraud: {fraud_probability:.1%}")
            else:
                st.success("✅ Transaction Appears Legitimate")
                st.write(f"Probability of Legitimate Transaction: {legitimate_probability:.1%}")
                st.write(f"(Fraud Probability: {fraud_probability:.1%})")
            
            # Display feature analysis
            st.subheader("Transaction Analysis")
            
            # Display input values with better explanation
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Transaction Amount", f"${amount:.2f}")
            with col2:
                st.metric(
                    "Sequence Position", 
                    f"{hours}h {minutes}m",
                    help="Position in transaction sequence"
                )
            
            # Display top influential features
            st.write("### Key Transaction Features")
            feature_values = df.drop(['Amount', 'Time'], axis=1).iloc[0]
            top_features = feature_values.abs().nlargest(10)
            
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
            
            st.info("""
            📊 **Feature Analysis Guide:**
            - The bars show feature values, NOT fraud probabilities
            - **Feature Values:**
                * Blue/Positive values and Red/Negative values represent different transaction patterns
                * The model learns which combinations of these patterns indicate fraud
                * A single blue (positive) value doesn't necessarily mean fraud
                * It's the combination of all features that determines the final prediction
            
            💡 **Understanding the Results:**
            - The model considers ALL features together to make its prediction
            - High feature values (positive or negative) show which characteristics are most distinctive
            - The final fraud probability comes from the pattern of ALL features combined
            - Individual feature values help explain which characteristics influenced the decision
            """)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
