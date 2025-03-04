# test_api_examples.py
import requests
import json

# Sample test cases
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

# API endpoint
url = 'http://localhost:5000/prediction/predict'
headers = {'Content-Type': 'application/json'}

# Model selection endpoint (optional)
model_url = 'http://localhost:5000/prediction/select-model'
model_data = {'model_name': 'Autoencoder'}  # Can be 'Autoencoder', 'FNN', or 'TabTransformer'

# Function to test the API with a case
def test_prediction(case, description):
    print(f"\n===== TESTING {description} =====")
    print("\nInput Data:")
    print(json.dumps(case, indent=2))
    
    try:
        # Optionally select the model first
        # requests.post(model_url, json=model_data, headers=headers)
        
        # Make the prediction
        response = requests.post(url, json=case, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print("\nAPI Response:")
            print(json.dumps(result, indent=2))
            
            print("\nKey Results:")
            print(f"Stroke Probability: {result['stroke_probability']:.2%}")
            print(f"Risk Category: {result['risk_category']}")
            print(f"Model Used: {result['model_used']}")
            
            print("\nTop Contributing Factors:")
            for factor in result['top_factors']:
                print(f"- {factor['feature']}: {factor['contribution']:.4f} ({factor['direction']} risk)")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Failed to connect to API: {str(e)}")
        print("\nPlease ensure the API is running at http://localhost:5000")

# Test both cases
print("\n================ STROKE PREDICTION API TEST ================")
test_prediction(stroke_case, "HIGH RISK CASE")
test_prediction(no_stroke_case, "LOW RISK CASE")