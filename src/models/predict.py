"""Prediction module for Ames Housing Price Prediction."""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_model(model_path: Union[str, Path]) -> Any:
    """Load a trained model from disk."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def load_preprocessing_artifacts(artifacts_path: Union[str, Path]) -> Dict:
    """Load all preprocessing artifacts from a single file."""
    artifacts_path = Path(artifacts_path)
    
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Artifacts file not found: {artifacts_path}")
    
    try:
        artifacts = joblib.load(artifacts_path)
        logger.info(f"Artifacts loaded successfully from {artifacts_path}")
        return artifacts
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        raise


class HousingPricePredictor:
    """Class for making housing price predictions."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        scaler_path: Optional[Union[str, Path]] = None,
        features_path: Optional[Union[str, Path]] = None,
        artifacts_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the predictor with model and artifacts.
        
        Args:
            model_path: Path to the saved model
            scaler_path: Path to the saved scaler (optional)
            features_path: Path to the features JSON file (optional)
            artifacts_path: Path to combined artifacts file (preferred)
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.features_path = Path(features_path) if features_path else None
        self.artifacts_path = Path(artifacts_path) if artifacts_path else None
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Load all artifacts
        self.load_artifacts()
    
    def load_artifacts(self) -> None:
        """Load all required artifacts."""
        # Try loading combined artifacts first
        if self.artifacts_path and self.artifacts_path.exists():
            try:
                artifacts = load_preprocessing_artifacts(self.artifacts_path)
                self.model = artifacts.get('model')
                self.scaler = artifacts.get('scaler')
                self.feature_names = artifacts.get('feature_names', [])
                logger.info("All artifacts loaded from combined file")
                return
            except Exception as e:
                logger.warning(f"Could not load combined artifacts: {e}")
        
        # Load model separately
        self.model = load_model(self.model_path)
        
        # Load scaler
        if self.scaler_path:
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"Scaler loaded from {self.scaler_path}")
        else:
            # Try default location
            default_scaler = Path(__file__).parent.parent.parent / 'models' / 'scaler.pkl'
            if default_scaler.exists():
                self.scaler = joblib.load(default_scaler)
                logger.info(f"Scaler loaded from default location")
        
        # Load features
        if self.features_path:
            with open(self.features_path, 'r') as f:
                self.feature_names = json.load(f)
            logger.info(f"Features loaded from {self.features_path}")
        else:
            # Try default location
            default_features = Path(__file__).parent.parent.parent / 'models' / 'selected_features.json'
            if default_features.exists():
                with open(default_features, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info("Features loaded from default location")
    
    def prepare_features(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare input features for prediction.
        
        Args:
            data: Dictionary of feature values
            
        Returns:
            Feature array ready for prediction
        """
        # Create DataFrame from input
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input must be dict or DataFrame")
        
        # Select and order features
        if self.feature_names:
            # Check for missing features
            missing = [f for f in self.feature_names if f not in df.columns]
            if missing:
                logger.warning(f"Missing features: {missing}")
                # Fill missing with 0
                for f in missing:
                    df[f] = 0
            
            # Select features in correct order
            df = df[self.feature_names]
        
        # Convert to numpy
        features = df.values.astype(np.float64)
        
        # Check for NaN values
        if np.isnan(features).any():
            logger.warning("NaN values found in input, filling with 0")
            features = np.nan_to_num(features, 0)
        
        return features
    
    def predict(
        self,
        data: Dict[str, Any],
        with_confidence: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Make a prediction on a single input.
        
        Args:
            data: Dictionary of feature values
            with_confidence: Whether to return confidence interval
            
        Returns:
            Predicted price or dict with prediction and confidence
        """
        features = self.prepare_features(data)
        
        # Scale features
        if self.scaler:
            features = self.scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = float(self.model.predict(features)[0])
        
        if with_confidence:
            # Simple confidence estimation based on model type
            if hasattr(self.model, 'estimators_'):
                # For ensemble models, use std of predictions
                if hasattr(self.model, 'estimators_'):
                    predictions = [est.predict(features)[0] for est in self.model.estimators_]
                    std = np.std(predictions)
                    return {
                        'prediction': prediction,
                        'lower_bound': prediction - 1.96 * std,
                        'upper_bound': prediction + 1.96 * std,
                        'confidence': 0.95,
                        'std': std
                    }
        
        return prediction


# Convenience function for quick predictions
def quick_predict(
    data: Dict[str, Any],
    model_path: Optional[Union[str, Path]] = None,
    artifacts_path: Optional[Union[str, Path]] = None,
    with_confidence: bool = False
) -> Union[float, Dict[str, Any]]:
    """
    Quick prediction function for single inputs.
    
    Args:
        data: Dictionary of feature values
        model_path: Path to model file (uses default if None)
        artifacts_path: Path to artifacts file (optional)
        with_confidence: Whether to return confidence interval
        
    Returns:
        Predicted price
    """
    if model_path is None:
        model_path = Path(__file__).parent.parent.parent / 'models' / 'best_model.pkl'
    
    if artifacts_path is None:
        artifacts_path = Path(__file__).parent.parent.parent / 'models' / 'training_data.pkl'
    
    predictor = HousingPricePredictor(model_path, artifacts_path=artifacts_path)
    return predictor.predict(data, with_confidence)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test data
    sample_data = {
        'Overall Qual': 7,
        'Gr Liv Area': 1500,
        'Year Built': 2005,
        'Total Bsmt SF': 1000,
        'Garage Area': 400,
        'Bedroom AbvGr': 3,
        'Full Bath': 2,
        'Half Bath': 1,
        'Lot Area': 10000,
        'TotRms AbvGrd': 6,
        'Neighborhood': 'NAmes',
        'Sale Condition': 'Normal'
    }
    
    print("Example prediction:")
    print(f"Input: {sample_data}")
    
    # Try to load and predict
    try:
        model_path = '../../models/best_model.pkl'
        artifacts_path = '../../models/training_data.pkl'
        
        predictor = HousingPricePredictor(model_path, artifacts_path=artifacts_path)
        prediction = predictor.predict(sample_data, with_confidence=True)
        print(f"\nPrediction: ${prediction['prediction']:,.2f}")
        if isinstance(prediction, dict):
            print(f"95% CI: ${prediction['lower_bound']:,.2f} - ${prediction['upper_bound']:,.2f}")
    except FileNotFoundError as e:
        print(f"\n⚠️  Model artifacts not found: {e}")
        print("Please run notebooks/01_EDA_and_Data_Cleaning.ipynb first to generate the artifacts.")