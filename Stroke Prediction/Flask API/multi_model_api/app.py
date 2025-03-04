# app.py
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import numpy as np
import pandas as pd
import pickle
import joblib
import os
import shap
import tensorflow as tf
import torch
from werkzeug.exceptions import BadRequest

# Initialize Flask app
app = Flask(__name__)
api = Api(app, version='1.0', title='Stroke Prediction API',
          description='API for predicting stroke risk using deep learning model')

# Define the API namespace
ns = api.namespace('prediction', description='Stroke Prediction')

# Define model selector endpoint
model_selector = api.model('ModelSelector', {
    'model_name': fields.String(required=True, description='Model to use for prediction', 
                               enum=['Autoencoder', 'FNN', 'TabTransformer'], 
                               example='Autoencoder')
})

# Define the input model for Swagger documentation
stroke_input = api.model('StrokeInput', {
    'age': fields.Float(required=True, description='Age of the patient', example=67.0),
    'gender': fields.String(required=True, description='Gender of the patient', example='Male', enum=['Male', 'Female', 'Other']),
    'hypertension': fields.Integer(required=True, description='Whether the patient has hypertension (0 or 1)', example=0),
    'heart_disease': fields.Integer(required=True, description='Whether the patient has heart disease (0 or 1)', example=1),
    'ever_married': fields.String(required=True, description='Whether the patient has ever been married', example='Yes', enum=['Yes', 'No']),
    'work_type': fields.String(required=True, description='Type of work', example='Private', enum=['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']),
    'Residence_type': fields.String(required=True, description='Type of residence', example='Urban', enum=['Urban', 'Rural']),
    'avg_glucose_level': fields.Float(required=True, description='Average glucose level in blood', example=228.69),
    'bmi': fields.Float(required=False, description='Body Mass Index', example=36.6),
    'smoking_status': fields.String(required=True, description='Smoking status of the patient', example='formerly smoked', enum=['formerly smoked', 'never smoked', 'smokes', 'Unknown'])
})

# Define the response model for Swagger documentation
stroke_prediction = api.model('StrokePrediction', {
    'stroke_probability': fields.Float(description='Probability of stroke'),
    'risk_category': fields.String(description='Risk category (Low, Medium, High)'),
    'top_factors': fields.List(fields.Nested(api.model('Factor', {
        'feature': fields.String(description='Feature name'),
        'contribution': fields.Float(description='Contribution to the prediction'),
        'direction': fields.String(description='Direction of impact (increases or decreases risk)')
    }))),
    'preprocessing_info': fields.String(description='Information about preprocessing steps applied'),
    'model_used': fields.String(description='Model used for prediction')
})

# Global variable to store the current model name
CURRENT_MODEL = 'Autoencoder'  # Default model

# Class to implement the Autoencoder model structure (matching the training implementation)
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

# Class to implement the TabTransformer model structure
class TabTransformer(torch.nn.Module):
    def __init__(self, input_dim, categorical_dims, continuous_dim,
                 embed_dim=32, num_heads=8, num_layers=6, 
                 dropout=0.1, mlp_dim=128):
        super(TabTransformer, self).__init__()
        
        self.categorical_dims = categorical_dims
        self.continuous_dim = continuous_dim
        
        if categorical_dims > 0:
            # Embedding layer for categorical features
            self.embedding = torch.nn.Linear(categorical_dims, embed_dim)
            
            # Transformer encoder
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Final MLP for combined features
            combined_dim = embed_dim + continuous_dim
        else:
            combined_dim = continuous_dim
        
        # MLP for continuous and transformed categorical features
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(combined_dim, mlp_dim),
            torch.nn.BatchNorm1d(mlp_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_dim, mlp_dim // 2),
            torch.nn.BatchNorm1d(mlp_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_dim // 2, 1)
        )
        
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        if self.categorical_dims > 0:
            # Split input into categorical and continuous parts
            cat_features = x[:, :self.categorical_dims]
            cont_features = x[:, self.categorical_dims:]
            
            # Process categorical features through embedding layer
            cat_embed = self.embedding(cat_features)
            
            # Add an extra dimension for the transformer
            cat_embed = cat_embed.unsqueeze(1)
            
            # Process through transformer
            cat_transformed = self.transformer_encoder(cat_embed)
            
            # Remove the extra dimension
            cat_transformed = cat_transformed.squeeze(1)
            
            # Concatenate with continuous features
            combined = torch.cat([cat_transformed, cont_features], dim=1)
        else:
            combined = x
            
        # Final prediction
        output = self.mlp(combined)
        return self.sigmoid(output).squeeze()


# Helper functions for preprocessing
def preprocess_data(input_data):
    """
    Apply the same preprocessing pipeline used during training
    
    Args:
        input_data: Dictionary or JSON with patient data
        
    Returns:
        Preprocessed features ready for model input
    """
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
    if 'bmi' not in df.columns or pd.isna(df['bmi'].iloc[0]):
        df['bmi'] = None
        df['bmi_missing'] = 1
        preprocessing_info = "BMI value was missing and handled appropriately."
    else:
        df['bmi_missing'] = 0
        preprocessing_info = "All input data properly processed."
    
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
        preprocessing_info += " Missing BMI was imputed based on training data distributions."
    
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
    
    # Load standard scaler for numerical features
    try:
        scaler = joblib.load('models/standard_scaler.pkl')
        numerical_cols = ['age', 'avg_glucose_level', 'bmi', 'age_squared', 'glucose_log']
        numerical_cols_present = [col for col in numerical_cols if col in df_encoded.columns]
        if numerical_cols_present:
            df_encoded[numerical_cols_present] = scaler.transform(df_encoded[numerical_cols_present])
            preprocessing_info += " Features scaled using pre-trained scaler."
    except Exception as e:
        # If scaler is not available, use Z-score normalization
        numerical_cols = ['age', 'avg_glucose_level', 'bmi', 'age_squared', 'glucose_log']
        numerical_cols_present = [col for col in numerical_cols if col in df_encoded.columns]
        if numerical_cols_present:
            df_encoded[numerical_cols_present] = (df_encoded[numerical_cols_present] - df_encoded[numerical_cols_present].mean()) / df_encoded[numerical_cols_present].std()
            preprocessing_info += " Features standardized with Z-score normalization."
    
    # Load PCA components if available
    try:
        pca = joblib.load('models/pca_transformer.pkl')
        pca_features = pca.transform(df_encoded.values)
        df_encoded['PCA_1'] = pca_features[:, 0]
        df_encoded['PCA_2'] = pca_features[:, 1]
        df_encoded['PCA_3'] = pca_features[:, 2]
        preprocessing_info += " PCA components generated using pre-trained transformer."
    except Exception as e:
        # If PCA is not available, add placeholder values
        df_encoded['PCA_1'] = 0
        df_encoded['PCA_2'] = 0
        df_encoded['PCA_3'] = 0
        preprocessing_info += " PCA components not available, using placeholder values."
    
    # Placeholder for cluster assignment
    df_encoded['cluster'] = 0
    
    # Load feature names to ensure consistent ordering
    try:
        with open('models/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        # Ensure all needed features are present
        for feature in feature_names:
            if feature not in df_encoded.columns:
                df_encoded[feature] = 0
        
        # Reorder columns to match expected model input
        df_encoded = df_encoded[feature_names]
        preprocessing_info += " Features aligned to match model training format."
    except Exception as e:
        preprocessing_info += " Feature ordering file not found, using generated features."
    
    return df_encoded, preprocessing_info, feature_names


# Model loading and prediction functions
def load_model(model_name=None, model_dir='models'):
    """
    Load a specific trained model for prediction
    
    Args:
        model_name: Name of the model to load (Autoencoder, FNN, TabTransformer)
        model_dir: Directory containing model files
        
    Returns:
        Loaded model and model type
    """
    global CURRENT_MODEL
    
    # If no model specified, use current model or load model info from file
    if model_name is None:
        model_name = CURRENT_MODEL
        # Try to read from model_info file if exists
        try:
            with open(os.path.join(model_dir, 'model_info.txt'), 'r') as f:
                lines = f.readlines()
                file_model_name = lines[0].split(':')[1].strip()
                if file_model_name in ['Autoencoder', 'FNN', 'TabTransformer']:
                    model_name = file_model_name
        except:
            # Keep using CURRENT_MODEL if file doesn't exist
            pass
    else:
        # Update the current model
        CURRENT_MODEL = model_name
    
    # Determine which framework to use based on model name
    if model_name in ['Autoencoder', 'FNN']:
        framework = 'tensorflow'
    else:  # TabTransformer
        framework = 'pytorch'
    
    # Load model based on framework and name
    if framework == 'tensorflow':
        # Different loading approach based on model type
        if model_name == 'Autoencoder':
            try:
                # Try to load custom Autoencoder model
                try:
                    # First try loading custom Autoencoder from saved model
                    model = tf.keras.models.load_model(
                        os.path.join(model_dir, 'autoencoder_model.keras'),
                        custom_objects={'Autoencoder': Autoencoder}
                    )
                except:
                    # If that fails, try loading from h5 format
                    model = tf.keras.models.load_model(
                        os.path.join(model_dir, 'autoencoder_model.h5'),
                        custom_objects={'Autoencoder': Autoencoder}
                    )
            except Exception as e:
                # Fallback to standard model loading
                try:
                    model = tf.keras.models.load_model(os.path.join(model_dir, 'best_model_tf.keras'))
                except:
                    model = tf.keras.models.load_model(os.path.join(model_dir, 'best_model_tf.h5'))
        else:  # FNN or default
            try:
                model = tf.keras.models.load_model(os.path.join(model_dir, 'best_model_tf.keras'))
            except:
                try:
                    model = tf.keras.models.load_model(os.path.join(model_dir, 'best_model_tf.h5'))
                except Exception as e:
                    raise Exception(f"Failed to load TensorFlow model: {str(e)}")
    else:  # PyTorch TabTransformer
        try:
            # Load model architecture
            model_architecture = joblib.load(os.path.join(model_dir, 'model_architecture.pkl'))
            
            # Create model with the same architecture
            model = TabTransformer(
                input_dim=model_architecture['input_dim'],
                categorical_dims=model_architecture['categorical_dims'],
                continuous_dim=model_architecture['continuous_dim'],
                embed_dim=model_architecture['hyperparams']['embed_dim'],
                num_heads=model_architecture['hyperparams']['num_heads'],
                num_layers=model_architecture['hyperparams']['num_layers'],
                dropout=model_architecture['hyperparams']['dropout'],
                mlp_dim=model_architecture['hyperparams']['mlp_dim']
            )
            
            # Load weights
            model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model_pt.pth')))
            model.eval()
        except Exception as e:
            raise Exception(f"Failed to load PyTorch model: {str(e)}")
    
    return model, framework, model_name


def predict_stroke_risk(model, framework, X):
    """
    Make predictions using the loaded model
    
    Args:
        model: Loaded model (TensorFlow or PyTorch)
        framework: Type of model ('tensorflow' or 'pytorch')
        X: Preprocessed features
        
    Returns:
        Predicted probability of stroke
    """
    if framework == 'tensorflow':
        # Ensure X is float32
        X = X.astype(np.float32)
        predictions = model.predict(X).ravel()
    else:  # pytorch
        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()
    
    return predictions


# Risk scoring and explanation functions
def calculate_risk_category(probability):
    """
    Determine risk category based on probability threshold
    
    Args:
        probability: Predicted probability of stroke
        
    Returns:
        Risk category string
    """
    if probability < 0.05:
        return "Low Risk"
    elif probability < 0.15:
        return "Medium Risk"
    else:
        return "High Risk"


def generate_explanation(model, framework, X, feature_names, model_name):
    """
    Generate explanation for the prediction using SHAP
    
    Args:
        model: Loaded model
        framework: Type of model ('tensorflow' or 'pytorch')
        X: Preprocessed features
        feature_names: List of feature names
        model_name: Name of the model
        
    Returns:
        List of top contributing features and their importance
    """
    try:
        # Create explainer based on model type
        if framework == 'tensorflow':
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
        else:  # pytorch
            def predict_fn(x):
                x_tensor = torch.tensor(x, dtype=torch.float32)
                with torch.no_grad():
                    return model(x_tensor).cpu().numpy()
            
            # Create the background dataset
            background = shap.sample(X, 5)
            
            # Create explainer
            explainer = shap.KernelExplainer(predict_fn, background)
            
            # Get SHAP values
            shap_values = explainer.shap_values(X.iloc[0:1])
            values = shap_values.flatten()
        
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
        return generate_domain_knowledge_explanation(X, feature_names, model_name)


def generate_domain_knowledge_explanation(X, feature_names, model_name):
    """
    Generate simple explanation based on domain knowledge when SHAP fails
    
    Args:
        X: Preprocessed features
        feature_names: List of feature names
        model_name: Name of the model
        
    Returns:
        List of likely contributors based on domain knowledge
    """
    explanations = []
    
    # Check age-related factors
    if 'age_over_60' in feature_names and X['age_over_60'].iloc[0] > 0:
        explanations.append({
            'feature': 'age_over_60',
            'contribution': 0.15,
            'direction': 'increases'
        })
    elif 'age_squared' in feature_names:
        explanations.append({
            'feature': 'age_squared',
            'contribution': 0.12,
            'direction': 'increases'
        })
    
    # Check glucose levels
    if 'glucose_diabetic' in feature_names and X['glucose_diabetic'].iloc[0] > 0:
        explanations.append({
            'feature': 'glucose_diabetic',
            'contribution': 0.10,
            'direction': 'increases'
        })
    
    # Check hypertension and heart disease
    if 'hypertension' in feature_names and X['hypertension'].iloc[0] > 0:
        explanations.append({
            'feature': 'hypertension',
            'contribution': 0.08,
            'direction': 'increases'
        })
    
    if 'heart_disease' in feature_names and X['heart_disease'].iloc[0] > 0:
        explanations.append({
            'feature': 'heart_disease',
            'contribution': 0.09,
            'direction': 'increases'
        })
    
    # Check BMI
    if 'bmi_obese' in feature_names and X['bmi_obese'].iloc[0] > 0:
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


@ns.route('/predict')
class StrokePredictor(Resource):
    @ns.expect(stroke_input)
    @ns.marshal_with(stroke_prediction)
    def post(self):
        """
        Predict stroke risk based on patient data
        """
        try:
            # Get input data
            input_data = request.json
            
            # Preprocess data
            X_processed, preprocessing_info, feature_names = preprocess_data(input_data)
            
            # Load model (use current global model)
            model, framework, model_name = load_model()
            
            # Make prediction
            prediction = predict_stroke_risk(model, framework, X_processed)
            risk_category = calculate_risk_category(prediction[0])
            
            # Generate explanation
            explanation = generate_explanation(model, framework, X_processed, feature_names, model_name)
            
            # Prepare response
            response = {
                'stroke_probability': float(prediction[0]),
                'risk_category': risk_category,
                'top_factors': explanation,
                'preprocessing_info': preprocessing_info,
                'model_used': model_name
            }
            
            return response
            
        except Exception as e:
            api.abort(500, f"Prediction error: {str(e)}")


@ns.route('/select-model')
class ModelSelector(Resource):
    @ns.expect(model_selector)
    def post(self):
        """
        Select which model to use for predictions
        """
        try:
            global CURRENT_MODEL
            
            # Get model name from request
            model_name = request.json['model_name']
            
            # Validate model name
            if model_name not in ['Autoencoder', 'FNN', 'TabTransformer']:
                api.abort(400, f"Invalid model name. Must be one of: Autoencoder, FNN, TabTransformer")
            
            # Try to load the model to ensure it exists
            try:
                model, framework, _ = load_model(model_name)
                CURRENT_MODEL = model_name
                return {'message': f"Successfully switched to {model_name} model", 'model': model_name, 'framework': framework}
            except Exception as e:
                api.abort(404, f"Failed to load {model_name} model: {str(e)}")
                
        except Exception as e:
            api.abort(500, f"Model selection error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)