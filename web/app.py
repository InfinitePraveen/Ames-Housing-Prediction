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

from src.data.preprocess import create_features, encode_categorical_with_encoders
from src.models.predict import load_model

app = Flask(__name__)
CORS(app)

# Load model and preprocessor
MODEL_PATH = Path(__file__).parent.parent / 'models' / 'best_model.pkl'
SCALER_PATH = Path(__file__).parent.parent / 'models' / 'scaler.pkl'
FEATURES_PATH = Path(__file__).parent.parent / 'models' / 'selected_features.json'
ENCODERS_PATH = Path(__file__).parent.parent / 'models' / 'label_encoders.pkl'

model = None
scaler = None
selected_features = None
label_encoders = None

DEFAULT_PREDICTION_INPUTS = {
    'Yr Sold': 2010,
    'Year Remod/Add': 1993,
    'Overall Cond': 5,
    'MS SubClass': 50,
    'Lot Frontage': 68.0,
    'Lot Area': 9436.5,
    'Exterior 1st': 'VinylSd',
    'Exterior 2nd': 'VinylSd',
    'Mas Vnr Area': 0,
    'Exter Qual': 'TA',
    'Foundation': 'PConc',
    'Bsmt Qual': 'TA',
    'BsmtFin Type 1': 'Unf',
    'BsmtFin SF 1': 0,
    'Heating QC': 'TA',
    '1st Flr SF': 1084,
    '2nd Flr SF': 0,
    'Kitchen Qual': 'TA',
    'Fireplaces': 0,
    'Fireplace Qu': 'TA',
    'Garage Type': 'Attchd',
    'Garage Yr Blt': 1978,
    'Garage Finish': 'Unf',
    'Garage Cars': 2,
    'Open Porch SF': 27,
    'Bsmt Full Bath': 0,
    'Bsmt Half Bath': 0,
    'Full Bath': 2,
    'Half Bath': 0,
    'Bedroom AbvGr': 3,
    'Wood Deck SF': 0,
    'Enclosed Porch': 0,
    '3Ssn Porch': 0,
    'Screen Porch': 0,
    'Pool Area': 0,
    'Sale Condition': 'Normal',
    'Neighborhood': 'NAmes',
    'Year Built': 1973,
    'Year Remod/Add': 1993,
    'Yr Sold': 2010
}

def load_artifacts():
    """Load model and preprocessing artifacts."""
    global model, scaler, selected_features, label_encoders
    
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

    try:
        label_encoders = joblib.load(ENCODERS_PATH)
        print(f'Loaded {len(label_encoders)} label encoders')
    except Exception as e:
        print(f'Error loading label encoders: {e}')
        label_encoders = None
    
    return True


def preprocess_input(data):
    """
    Preprocess user input for prediction.
    
    Args:
        data: Dictionary of input features
        
    Returns:
        Preprocessed feature array
    """
    # Merge defaults with provided inputs so recruiter-friendly forms can stay compact
    merged_data = DEFAULT_PREDICTION_INPUTS.copy()
    merged_data.update(data)
    df = pd.DataFrame([merged_data])
    
    # Create features
    df = create_features(df)
    
    # Encode categorical
    if label_encoders is not None:
        df_encoded = encode_categorical_with_encoders(df, label_encoders)
    else:
        from src.data.preprocess import encode_categorical
        df_encoded, _ = encode_categorical(df)
    
    # Select features
    if selected_features:
        available_features = [f for f in selected_features if f in df_encoded.columns]
        df_selected = df_encoded[available_features]
    else:
        df_selected = df_encoded
    
    # Ensure all selected features exist and fill missing values
    for feature in selected_features or []:
        if feature not in df_selected.columns:
            df_selected[feature] = 0
    if selected_features:
        df_selected = df_selected[selected_features]
    
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
        required_fields = [
            'Overall Qual',
            'Overall Cond',
            'Gr Liv Area',
            'Year Built',
            'Year Remod/Add',
            'Yr Sold',
            'Total Bsmt SF',
            'Garage Area',
            'TotRms AbvGrd',
            'Full Bath',
            'Neighborhood',
            'Sale Condition'
        ]
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