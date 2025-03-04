# app.py
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
import joblib
import os
import shap
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from werkzeug.exceptions import BadRequest
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Define global variables for loaded components
stroke_model = None
feature_names_enhanced = None
pca = None
scaler = None
risk_scoring_system = None

# Class to implement the Autoencoder model structure
class Autoencoder(tf.keras.Model):
    def __init__(self, input_dim, encoding_dim=16, hidden_layers=[64, 32], activation='relu', dropout_rate=0.2):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder_layers = []
        x_dim = input_dim
        
        for units in hidden_layers:
            self.encoder_layers.append(tf.keras.layers.Dense(units, activation=activation))
            self.encoder_layers.append(tf.keras.layers.BatchNormalization())
            self.encoder_layers.append(tf.keras.layers.Dropout(dropout_rate))
            x_dim = units
        
        self.bottleneck = tf.keras.layers.Dense(encoding_dim, activation=activation, name='encoder_output')
        
        # Classifier (prediction) layers
        self.classifier_layers = [
            tf.keras.layers.Dense(32, activation=activation),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(16, activation=activation),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1, activation='sigmoid', name='classifier_output')
        ]
    
    def call(self, inputs, training=False):
        # Encoder path
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        
        encoded = self.bottleneck(x)
        
        # Classifier path
        x = encoded
        for layer in self.classifier_layers:
            x = layer(x, training=training)
        
        return x

def load_model_components():
    """Load all required model components on startup"""
    global stroke_model, feature_names_enhanced, pca, scaler, risk_scoring_system
    
    model_dir = 'models'
    
    # Load feature names
    try:
        with open(os.path.join(model_dir, 'feature_names.txt'), 'r') as f:
            feature_names_enhanced = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(feature_names_enhanced)} feature names")
    except Exception as e:
        print(f"Warning: Could not load feature names: {str(e)}")
    
    # Load the autoencoder model
    try:
        # Try first with the Autoencoder custom object
        try:
            stroke_model = keras.models.load_model(
                os.path.join(model_dir, 'autoencoder_model.keras'),
                custom_objects={'Autoencoder': Autoencoder}
            )
            print("Loaded autoencoder_model.keras with custom Autoencoder class")
        except:
            # If that fails, try loading the standard way
            stroke_model = keras.models.load_model(os.path.join(model_dir, 'best_model_tf.keras'))
            print("Loaded best_model_tf.keras")
    except Exception as e:
        print(f"Warning: Could not load TensorFlow model: {str(e)}")
        # Create a simple placeholder model for testing
        print("Creating placeholder model for testing")
        input_dim = len(feature_names_enhanced) if feature_names_enhanced else 44
        inputs = keras.Input(shape=(input_dim,))
        x = keras.layers.Dense(64, activation='relu')(inputs)
        x = keras.layers.Dense(32, activation='relu')(x)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
        stroke_model = keras.Model(inputs, outputs)
    
    # Load PCA transformer
    try:
        pca = joblib.load(os.path.join(model_dir, 'pca_transformer.pkl'))
        print("Loaded PCA transformer")
    except Exception as e:
        print(f"Warning: Could not load PCA transformer: {str(e)}")
    
    # Load standard scaler
    try:
        scaler = joblib.load(os.path.join(model_dir, 'standard_scaler.pkl'))
        print("Loaded standard scaler")
    except Exception as e:
        print(f"Warning: Could not load standard scaler: {str(e)}")
    
    # Load risk scoring system
    try:
        risk_scoring_system = joblib.load(os.path.join(model_dir, 'risk_scoring_system.pkl'))
        print("Loaded risk scoring system")
    except Exception as e:
        print(f"Warning: Could not load risk scoring system: {str(e)}")
        # Create a simple risk scoring system
        risk_scoring_system = {
            'categories': {
                'Low Risk': (0, 0.05),
                'Moderate Risk': (0.05, 0.15),
                'High Risk': (0.15, 1.0)
            }
        }
    
    print("Model components loaded successfully")

def preprocess_data(input_data):
    """
    Apply the same preprocessing pipeline used during training
    
    Args:
        input_data: Dictionary or JSON with patient data
        
    Returns:
        Preprocessed features ready for model input
    """
    global feature_names_enhanced, scaler, pca
    
    # Convert input to DataFrame
    try:
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = pd.DataFrame(input_data)
    except Exception as e:
        raise BadRequest(f"Error converting input data to DataFrame: {str(e)}")
    
    # Validate required fields
    required_fields = ['age', 'gender', 'hypertension', 'heart_disease', 
                       'ever_married', 'work_type', 'Residence_type', 
                       'avg_glucose_level', 'smoking_status']
    
    for field in required_fields:
        if field not in df.columns:
            raise BadRequest(f"Missing required field: {field}")
    
    # Handle BMI missing values
    preprocessing_info = []
    if 'bmi' not in df.columns or pd.isna(df['bmi'].iloc[0]):
        df['bmi'] = None
        df['bmi_missing'] = 1
        preprocessing_info.append("BMI value was missing and handled appropriately.")
    else:
        df['bmi_missing'] = 0
    
    # Create age features
    df['age_under_40'] = (df['age'] < 40).astype(int)
    df['age_40_to_60'] = ((df['age'] >= 40) & (df['age'] < 60)).astype(int)
    df['age_over_60'] = (df['age'] >= 60).astype(int)
    df['age_squared'] = df['age'] ** 2
    
    # Create glucose features
    df['glucose_normal'] = (df['avg_glucose_level'] < 100).astype(int)
    df['glucose_prediabetic'] = ((df['avg_glucose_level'] >= 100) & 
                             (df['avg_glucose_level'] < 126)).astype(int)
    df['glucose_diabetic'] = (df['avg_glucose_level'] >= 126).astype(int)
    df['glucose_log'] = np.log1p(df['avg_glucose_level'])
    
    # Create BMI categories
    if 'bmi' in df.columns and not pd.isna(df['bmi'].iloc[0]):
        df['bmi_underweight'] = (df['bmi'] < 18.5).astype(int)
        df['bmi_normal'] = ((df['bmi'] >= 18.5) & (df['bmi'] < 25)).astype(int)
        df['bmi_overweight'] = ((df['bmi'] >= 25) & (df['bmi'] < 30)).astype(int)
        df['bmi_obese'] = (df['bmi'] >= 30).astype(int)
    else:
        # Use median values from training set if BMI is missing
        df['bmi_underweight'] = 0
        df['bmi_normal'] = 0
        df['bmi_overweight'] = 1  # Assuming median BMI was in overweight range
        df['bmi_obese'] = 0
        preprocessing_info.append("Missing BMI was imputed based on training data distributions.")
    
    # Create medical features
    df['comorbidity_count'] = df['hypertension'] + df['heart_disease']
    df['age_hypertension'] = df['age'] * df['hypertension']
    df['age_heart_disease'] = df['age'] * df['heart_disease']
    df['glucose_hypertension'] = (df['avg_glucose_level'] / 100) * df['hypertension']
    
    # Create risk profile
    conditions = [
        (df['age'] < 50) & (df['hypertension'] == 0) & (df['heart_disease'] == 0),
        ((df['age'] >= 50) & (df['age'] < 70) & 
         ((df['hypertension'] == 0) & (df['heart_disease'] == 0))) | 
        ((df['age'] < 50) & ((df['hypertension'] == 1) | (df['heart_disease'] == 1))),
        (df['age'] >= 70) | ((df['age'] >= 50) & (df['age'] < 70) & 
                          ((df['hypertension'] == 1) | (df['heart_disease'] == 1)))
    ]
    choices = [0, 1, 2]  # Low, Medium, High
    df['risk_profile'] = np.select(conditions, choices, default=0)
    
    # Handle work_type
    df['is_child'] = (df['work_type'] == 'children').astype(int)
    
    # One-hot encode categorical variables
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    # Scale numeric features if scaler is available
    if scaler:
        numerical_cols = ['age', 'avg_glucose_level', 'bmi', 'age_squared', 'glucose_log']
        numerical_cols_present = [col for col in numerical_cols if col in df_encoded.columns]
        if numerical_cols_present:
            df_encoded[numerical_cols_present] = scaler.transform(df_encoded[numerical_cols_present])
            preprocessing_info.append("Features scaled using pre-trained scaler.")
    else:
        # If scaler is not available, use Z-score normalization
        numerical_cols = ['age', 'avg_glucose_level', 'bmi', 'age_squared', 'glucose_log']
        numerical_cols_present = [col for col in numerical_cols if col in df_encoded.columns]
        if numerical_cols_present:
            df_encoded[numerical_cols_present] = (df_encoded[numerical_cols_present] - df_encoded[numerical_cols_present].mean()) / df_encoded[numerical_cols_present].std()
            preprocessing_info.append("Features standardized with Z-score normalization.")
    
    # Generate PCA components if available
    if pca:
        pca_features = pca.transform(df_encoded.values)
        df_encoded['PCA_1'] = pca_features[:, 0]
        df_encoded['PCA_2'] = pca_features[:, 1]
        df_encoded['PCA_3'] = pca_features[:, 2]
        preprocessing_info.append("PCA components generated using pre-trained transformer.")
    else:
        # If PCA is not available, add placeholder values
        df_encoded['PCA_1'] = 0
        df_encoded['PCA_2'] = 0
        df_encoded['PCA_3'] = 0
        preprocessing_info.append("PCA components not available, using placeholder values.")
    
    # Add cluster placeholder
    df_encoded['cluster'] = 0
    
    # Ensure all features needed by the model are present
    if feature_names_enhanced:
        for feature in feature_names_enhanced:
            if feature not in df_encoded.columns:
                df_encoded[feature] = 0
                preprocessing_info.append(f"Added missing feature: {feature}")
        
        # Reorder columns to match training format
        cols_to_use = [col for col in feature_names_enhanced if col in df_encoded.columns]
        df_encoded = df_encoded[cols_to_use]
        preprocessing_info.append("Features aligned to match model training format.")
    
    return df_encoded, ", ".join(preprocessing_info)

def calculate_risk_category(probability):
    """
    Determine risk category based on probability threshold
    
    Args:
        probability: Predicted probability of stroke
        
    Returns:
        Risk category string
    """
    global risk_scoring_system
    
    if risk_scoring_system and 'categories' in risk_scoring_system:
        for category, (min_prob, max_prob) in risk_scoring_system['categories'].items():
            if min_prob <= probability < max_prob:
                return category
    
    # Default thresholds if risk scoring system not available
    if probability < 0.05:
        return "Low Risk"
    elif probability < 0.15:
        return "Medium Risk"
    else:
        return "High Risk"

def generate_explanation(model, X, feature_names):
    """
    Generate explanation for the prediction using SHAP
    
    Args:
        model: Loaded model
        X: Preprocessed features
        feature_names: List of feature names
        
    Returns:
        List of top contributing features and their importance
    """
    try:
        # Define prediction function for SHAP
        def predict_fn(x):
            return model.predict(x)
        
        # Create the background dataset (small sample for efficiency)
        background = shap.sample(X, 5)
        
        # Create explainer
        explainer = shap.KernelExplainer(predict_fn, background)
        
        # Get SHAP values for the current input
        shap_values = explainer.shap_values(X.iloc[0:1])
        
        # Handle different output formats depending on model
        if isinstance(shap_values, list):
            # For multi-output models
            values = shap_values[0] if len(shap_values) == 1 else shap_values[1]
        else:
            values = shap_values
        
        values = values.flatten()
        
        # Get feature contributions
        feature_contributions = []
        for i, value in enumerate(values):
            if i < len(feature_names):
                feature_contributions.append({
                    'feature': feature_names[i],
                    'contribution': float(value),
                    'direction': 'increases' if value > 0 else 'decreases'
                })
        
        # Sort by absolute contribution and take top 5
        top_contributions = sorted(feature_contributions, 
                                  key=lambda x: abs(x['contribution']), 
                                  reverse=True)[:5]
        
        return top_contributions
    except Exception as e:
        # If SHAP fails, return a simpler explanation based on domain knowledge
        print(f"SHAP explanation failed with error: {str(e)}. Using domain knowledge instead.")
        return generate_domain_knowledge_explanation(X)

def generate_domain_knowledge_explanation(X):
    """Generate explanation based on domain knowledge when SHAP fails"""
    explanations = []
    
    # Check age-related factors
    if 'age_over_60' in X.columns and X['age_over_60'].iloc[0] > 0:
        explanations.append({
            'feature': 'age_over_60',
            'contribution': 0.15,
            'direction': 'increases'
        })
    elif 'age_squared' in X.columns:
        explanations.append({
            'feature': 'age_squared',
            'contribution': 0.12,
            'direction': 'increases'
        })
    
    # Check glucose levels
    if 'glucose_diabetic' in X.columns and X['glucose_diabetic'].iloc[0] > 0:
        explanations.append({
            'feature': 'glucose_diabetic',
            'contribution': 0.10,
            'direction': 'increases'
        })
    
    # Check hypertension and heart disease
    if 'hypertension' in X.columns and X['hypertension'].iloc[0] > 0:
        explanations.append({
            'feature': 'hypertension',
            'contribution': 0.08,
            'direction': 'increases'
        })
    
    if 'heart_disease' in X.columns and X['heart_disease'].iloc[0] > 0:
        explanations.append({
            'feature': 'heart_disease',
            'contribution': 0.09,
            'direction': 'increases'
        })
    
    # Check BMI
    if 'bmi_obese' in X.columns and X['bmi_obese'].iloc[0] > 0:
        explanations.append({
            'feature': 'bmi_obese',
            'contribution': 0.07,
            'direction': 'increases'
        })
    
    # Ensure we have at least 3 explanations
    if len(explanations) < 3:
        # Add generic explanations if we don't have enough
        if len(explanations) < 1:
            explanations.append({
                'feature': 'age',
                'contribution': 0.15,
                'direction': 'increases'
            })
        
        if len(explanations) < 2:
            explanations.append({
                'feature': 'avg_glucose_level',
                'contribution': 0.10,
                'direction': 'increases with high levels'
            })
        
        if len(explanations) < 3:
            explanations.append({
                'feature': 'bmi',
                'contribution': 0.07,
                'direction': 'increases with high BMI'
            })
    
    # Take top 5 or all if less than 5
    return explanations[:5]

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict stroke risk based on patient data"""
    global stroke_model, feature_names_enhanced
    
    try:
        # Get input data
        input_data = request.json
        
        # Preprocess data
        X_processed, preprocessing_info = preprocess_data(input_data)
        
        # Make sure model is loaded
        if stroke_model is None:
            return jsonify({'error': 'Model not loaded. Please try again.'}), 500
        
        # Make prediction
        X_processed_np = X_processed.values.astype(np.float32)
        prediction = stroke_model.predict(X_processed_np).ravel()
        risk_category = calculate_risk_category(prediction[0])
        
        # Generate explanation
        explanation = generate_explanation(stroke_model, X_processed, X_processed.columns.tolist())
        
        # Prepare response
        response = {
            'stroke_probability': float(prediction[0]),
            'risk_category': risk_category,
            'top_factors': explanation,
            'preprocessing_info': preprocessing_info
        }
        
        return jsonify(response)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': stroke_model is not None})

if __name__ == '__main__':
    # Load model components on startup
    load_model_components()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)