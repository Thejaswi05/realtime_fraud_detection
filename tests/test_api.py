import requests
import json
from datetime import datetime
import time

def test_api():
    # API endpoint
    url = "http://localhost:8000/predict"
    
    # Test cases
    test_cases = [
        {"amount": 100.0, "timestamp": datetime.now().isoformat()},
        {"amount": 1000.0, "timestamp": datetime.now().isoformat()},
        {"amount": 50000.0, "timestamp": datetime.now().isoformat()},
    ]
    
    print("\nTesting Fraud Detection API:")
    print("-" * 50)
    
    for case in test_cases:
        try:
            # Make prediction request
            response = requests.post(url, json=case)
            
            if response.status_code == 200:
                result = response.json()
                print(f"\nTest Case: Amount ${case['amount']}")
                print(f"Prediction: {'Fraud' if result['prediction'] == 1 else 'Legitimate'}")
                print(f"Fraud Probability: {result['fraud_probability']:.1%}")
                print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
                print(f"Prediction ID: {result['prediction_id']}")
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"Error: {str(e)}")
        
        time.sleep(1)  # Wait between requests

if __name__ == "__main__":
    test_api() 