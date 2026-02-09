import requests
import json

BASE_URL = "http://localhost:8000"

def test_api():
    """Test the API with sample requests"""
    
    print("Testing Electricity Supply Band Predictor API\n")
    print("=" * 50)
    
    # Test health check
    print("\n1. Health Check:")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.json()}")
    
    # Test predictions
    test_cases = [
        {
            "disco": "IKEDC",
            "zone": "Urban",
            "feeder_age": 5.0,
            "transformer_issue": False
        },
        {
            "disco": "EKEDC",
            "zone": "Rural",
            "feeder_age": 18.0,
            "transformer_issue": True
        },
        {
            "disco": "AEDC",
            "zone": "Suburban",
            "feeder_age": 10.5,
            "transformer_issue": False
        }
    ]
    
    print("\n2. Making Predictions:")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {test_case}")
        response = requests.post(f"{BASE_URL}/predict", json=test_case)
        prediction = response.json()
        print(f"Predicted Band: {prediction['supply_band']}")
        print(f"Confidence: {prediction['confidence']}")

if __name__ == "__main__":
    try:
        test_api()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the API is running on http://localhost:8000")
