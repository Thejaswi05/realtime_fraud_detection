import requests
import json
from datetime import datetime, timedelta
import time

def test_monitoring():
    """Test API predictions and check monitoring logs"""
    
    # API endpoint
    url = "http://localhost:8000/predict"
    
    # Make multiple predictions with different amounts
    test_cases = [
        {"amount": 100.0, "timestamp": datetime.now().isoformat()},
        {"amount": 1000.0, "timestamp": datetime.now().isoformat()},
        {"amount": 5000.0, "timestamp": datetime.now().isoformat()},
        {"amount": 50000.0, "timestamp": datetime.now().isoformat()},
    ]
    
    print("\nTesting Fraud Detection API and Monitoring:")
    print("-" * 50)
    
    for case in test_cases:
        try:
            # Make prediction request
            response = requests.post(url, json=case)
            
            if response.status_code == 200:
                result = response.json()
                print(f"\nTest Case: Amount ${case['amount']}")
                print(f"Prediction: {'Fraud' if result['prediction'] == 1 else 'Legitimate'}")
                print(f"Prediction ID: {result['prediction_id']}")
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"Error: {str(e)}")
        
        time.sleep(1)  # Wait between requests

def check_monitoring_logs():
    """Check the monitoring logs"""
    from pathlib import Path
    import json
    
    monitoring_path = Path("monitoring/predictions")
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = monitoring_path / f"{today}.jsonl"
    
    print("\nChecking Monitoring Logs:")
    print("-" * 50)
    
    if log_file.exists():
        with open(log_file) as f:
            predictions = [json.loads(line) for line in f]
            
        print(f"Found {len(predictions)} predictions logged today")
        
        # Show last 3 predictions
        print("\nLast 3 predictions:")
        for pred in predictions[-3:]:
            print(f"\nPrediction ID: {pred['prediction_id']}")
            print(f"Timestamp: {pred['timestamp']}")
            print(f"Amount: ${pred['transaction']['amount']}")
            print(f"Prediction: {'Fraud' if pred['prediction'] == 1 else 'Legitimate'}")
            print(f"Probability: {pred['probability']:.3f}")
    else:
        print(f"No log file found for today ({today})")

if __name__ == "__main__":
    # Run tests
    test_monitoring()
    
    # Wait a moment for logs to be written
    time.sleep(2)
    
    # Check logs
    check_monitoring_logs()
    
    # Generate report
    print("\nGenerating Monitoring Report...")
    from src.monitoring.monitoring import generate_monitoring_report
    generate_monitoring_report() 