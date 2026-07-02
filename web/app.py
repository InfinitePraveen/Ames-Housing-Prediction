"""Flask web application for Ames Housing Price Prediction."""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocess import create_features, encode_categorical
from src.models.predict import load_model

app = Flask(__name__)
CORS(app)

# Load model and preprocessor
MODEL_PATH = Path(__file__).parent.parent / 'models' / 'best_model.pkl'
SCALER_PATH = Path(__file__).parent.parent / 'models' / 'scaler.pkl'
FEATURES_PATH = Path(__file__).parent.parent / 'models' / 'selected_features.json'

model = None
scaler = None
selected_features = None

def load_artifacts():
    """Load model and preprocessing artifacts."""
    global model, scaler, selected_features
    
    try:
        model = load_model(MODEL_PATH)
        print('Model loaded successfully')
    except Exception as e:
        print(f'Error loading model: {e}')
        return False
    
    try:
        scaler = joblib.load(SCALER_PATH)
        print('Scaler loaded successfully')
    except Exception as e:
        print(f'Error loading scaler: {e}')
        # Continue without scaler
        scaler = None
    
    try:
        with open(FEATURES_PATH, 'r') as f:
            selected_features = json.load(f)
        print(f'Loaded {len(selected_features)} selected features')
    except Exception as e:
        print(f'Error loading features: {e}')
        selected_features = None
    
    return True


def preprocess_input(data):
    """
    Preprocess user input for prediction.
    
    Args:
        data: Dictionary of input features
        
    Returns:
        Preprocessed feature array
    """
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Create features
    df = create_features(df)
    
    # Encode categorical
    df_encoded, _ = encode_categorical(df)
    
    # Select features
    if selected_features:
        # Ensure all selected features exist
        available_features = [f for f in selected_features if f in df_encoded.columns]
        df_selected = df_encoded[available_features]
    else:
        df_selected = df_encoded
    
    # Scale features
    if scaler:
        X = scaler.transform(df_selected.values.reshape(1, -1))
    else:
        X = df_selected.values.reshape(1, -1)
    
    return X


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Make price prediction."""
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['Overall Qual', 'Gr Liv Area', 'Year Built', 'Total Bsmt SF', 'Garage Area']
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {missing_fields}'}), 400
        
        # Preprocess input
        X = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(X)
        price = float(prediction[0])
        
        # Calculate confidence (simplified)
        confidence = 0.95  # Placeholder
        
        return jsonify({
            'prediction': price,
            'formatted_price': f'${price:,.2f}',
            'confidence': confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })


if __name__ == '__main__':
    # Load artifacts
    if not load_artifacts():
        print('Warning: Could not load all artifacts. Some features may not work.')
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)