# test_api.py
import requests
import json
import argparse

# Default API URL
DEFAULT_API_URL = "http://localhost:5000"

# Sample patient data
# High-risk patient (stroke case)
stroke_case = {
    'age': 67.0,
    'gender': 'Male',
    'hypertension': 1,  # Has hypertension
    'heart_disease': 1,  # Has heart disease
    'ever_married': 'Yes',
    'work_type': 'Private',
    'Residence_type': 'Urban',
    'avg_glucose_level': 228.69,  # High glucose (diabetic)
    'bmi': 36.6,  # Obese
    'smoking_status': 'formerly smoked'
}

# Low-risk patient (no stroke case)
no_stroke_case = {
    'age': 32.0,  # Young age
    'gender': 'Female',
    'hypertension': 0,  # No hypertension
    'heart_disease': 0,  # No heart disease
    'ever_married': 'Yes',
    'work_type': 'Private',
    'Residence_type': 'Rural',
    'avg_glucose_level': 85.12,  # Normal glucose
    'bmi': 22.4,  # Normal BMI
    'smoking_status': 'never smoked'
}

def test_health(api_url):
    """Test the health check endpoint"""
    url = f"{api_url}/health"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed with status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Health check failed with error: {str(e)}")

def test_prediction(api_url, patient_data, description):
    """Test the prediction endpoint with the provided patient data"""
    url = f"{api_url}/predict"
    headers = {'Content-Type': 'application/json'}
    
    print(f"\n===== TESTING {description} =====")
    print("\nInput Data:")
    print(json.dumps(patient_data, indent=2))
    
    try:
        response = requests.post(url, json=patient_data, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Prediction successful!")
            print("\nAPI Response:")
            print(json.dumps(result, indent=2))
            
            print("\nKey Results:")
            print(f"Stroke Probability: {result['stroke_probability']:.2%}")
            print(f"Risk Category: {result['risk_category']}")
            
            print("\nTop Contributing Factors:")
            for factor in result['top_factors']:
                print(f"- {factor['feature']}: {factor['contribution']:.4f} ({factor['direction']} risk)")
        else:
            print(f"❌ Prediction failed with status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Prediction failed with error: {str(e)}")
        print("\nPlease ensure the API is running at the correct URL")

def main():
    parser = argparse.ArgumentParser(description='Test the Stroke Prediction API')
    parser.add_argument('--url', default=DEFAULT_API_URL, help=f'API base URL (default: {DEFAULT_API_URL})')
    args = parser.parse_args()
    
    print("\n================ STROKE PREDICTION API TEST ================")
    print(f"API URL: {args.url}")
    
    # Test health check
    test_health(args.url)
    
    # Test predictions
    test_prediction(args.url, stroke_case, "HIGH RISK CASE")
    test_prediction(args.url, no_stroke_case, "LOW RISK CASE")
    
    print("\n==================== TEST COMPLETE ====================")

if __name__ == "__main__":
    main()
