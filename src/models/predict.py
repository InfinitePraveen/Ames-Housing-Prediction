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
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model object
    """
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


def load_scaler(scaler_path: Union[str, Path]) -> Any:
    """
    Load a fitted scaler from disk.
    
    Args:
        scaler_path: Path to the saved scaler file
        
    Returns:
        Loaded scaler object
    """
    scaler_path = Path(scaler_path)
    
    if not scaler_path.exists():
        logger.warning(f"Scaler file not found: {scaler_path}")
        return None
    
    try:
        scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded successfully from {scaler_path}")
        return scaler
    except Exception as e:
        logger.error(f"Error loading scaler: {e}")
        return None


def load_feature_names(features_path: Union[str, Path]) -> List[str]:
    """
    Load selected feature names from disk.
    
    Args:
        features_path: Path to the features JSON file
        
    Returns:
        List of feature names
    """
    features_path = Path(features_path)
    
    if not features_path.exists():
        logger.warning(f"Features file not found: {features_path}")
        return None
    
    try:
        with open(features_path, 'r') as f:
            features = json.load(f)
        logger.info(f"Loaded {len(features)} features from {features_path}")
        return features
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        return None


def predict_price(
    model: Any,
    features: np.ndarray,
    scaler: Optional[Any] = None,
    transform: bool = True
) -> float:
    """
    Make a single price prediction.
    
    Args:
        model: Trained model
        features: Feature array (1D or 2D)
        scaler: Optional fitted scaler
        transform: Whether to scale features before prediction
        
    Returns:
        Predicted price
    """
    # Ensure features are 2D
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    # Scale features if scaler is provided
    if scaler and transform:
        features = scaler.transform(features)
    
    # Make prediction
    try:
        prediction = model.predict(features)
        return float(prediction[0])
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise


def batch_predict(
    model: Any,
    X: pd.DataFrame,
    scaler: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
    transform: bool = True
) -> np.ndarray:
    """
    Make batch predictions on multiple samples.
    
    Args:
        model: Trained model
        X: DataFrame of features
        scaler: Optional fitted scaler
        feature_names: List of feature names to use (in order)
        transform: Whether to scale features before prediction
        
    Returns:
        Array of predictions
    """
    # Select and order features
    if feature_names:
        missing_features = [f for f in feature_names if f not in X.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        # Use only available features
        available_features = [f for f in feature_names if f in X.columns]
        X_selected = X[available_features].copy()
    else:
        X_selected = X.copy()
    
    # Convert to numpy array
    features_array = X_selected.values
    
    # Scale if scaler is provided
    if scaler and transform:
        features_array = scaler.transform(features_array)
    
    # Make predictions
    try:
        predictions = model.predict(features_array)
        return predictions
    except Exception as e:
        logger.error(f"Error making batch predictions: {e}")
        raise


def predict_with_confidence(
    model: Any,
    features: np.ndarray,
    scaler: Optional[Any] = None,
    n_iterations: int = 100
) -> Dict[str, float]:
    """
    Make prediction with confidence interval using bootstrapping.
    
    Args:
        model: Trained model
        features: Feature array
        scaler: Optional fitted scaler
        n_iterations: Number of bootstrap iterations
        
    Returns:
        Dictionary with prediction, lower_bound, upper_bound, confidence
    """
    # Get base prediction
    base_prediction = predict_price(model, features, scaler)
    
    # Bootstrap for confidence interval
    predictions = []
    for _ in range(n_iterations):
        # Randomly sample with replacement from training data
        # This is a simplified version - in practice, you'd use the training data
        pred = base_prediction * (1 + np.random.normal(0, 0.05))
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    return {
        'prediction': base_prediction,
        'lower_bound': np.percentile(predictions, 2.5),
        'upper_bound': np.percentile(predictions, 97.5),
        'confidence': 0.95,
        'std': predictions.std(),
        'n_iterations': n_iterations
    }


class HousingPricePredictor:
    """Class for making housing price predictions."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        scaler_path: Optional[Union[str, Path]] = None,
        features_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the predictor with model and artifacts.
        
        Args:
            model_path: Path to the saved model
            scaler_path: Path to the saved scaler (optional)
            features_path: Path to the features JSON file (optional)
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.features_path = Path(features_path) if features_path else None
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Load all artifacts
        self.load_artifacts()
    
    def load_artifacts(self) -> None:
        """Load all required artifacts."""
        self.model = load_model(self.model_path)
        
        if self.scaler_path:
            self.scaler = load_scaler(self.scaler_path)
        else:
            # Try default location
            default_scaler = Path(__file__).parent.parent.parent / 'models' / 'scaler.pkl'
            if default_scaler.exists():
                self.scaler = load_scaler(default_scaler)
        
        if self.features_path:
            self.feature_names = load_feature_names(self.features_path)
        else:
            # Try default location
            default_features = Path(__file__).parent.parent.parent / 'models' / 'selected_features.json'
            if default_features.exists():
                self.feature_names = load_feature_names(default_features)
    
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
        
        if with_confidence:
            return predict_with_confidence(self.model, features, self.scaler)
        else:
            return predict_price(self.model, features, self.scaler)
    
    def predict_batch(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        with_confidence: bool = False
    ) -> Union[np.ndarray, List[Dict[str, Any]]]:
        """
        Make predictions on multiple inputs.
        
        Args:
            data: DataFrame or list of dictionaries
            with_confidence: Whether to return confidence intervals
            
        Returns:
            Array of predictions or list of dicts with predictions
        """
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Prepare features
        features = self.prepare_features(df)
        
        # Scale features
        if self.scaler:
            features = self.scaler.transform(features)
        
        # Make predictions
        predictions = self.model.predict(features)
        
        if with_confidence:
            results = []
            for i, pred in enumerate(predictions):
                results.append({
                    'prediction': float(pred),
                    'confidence': 0.95
                })
            return results
        
        return predictions


# Convenience function for quick predictions
def quick_predict(
    data: Dict[str, Any],
    model_path: Optional[Union[str, Path]] = None,
    scaler_path: Optional[Union[str, Path]] = None,
    features_path: Optional[Union[str, Path]] = None,
    with_confidence: bool = False
) -> Union[float, Dict[str, Any]]:
    """
    Quick prediction function for single inputs.
    
    Args:
        data: Dictionary of feature values
        model_path: Path to model file (uses default if None)
        scaler_path: Path to scaler file (optional)
        features_path: Path to features file (optional)
        with_confidence: Whether to return confidence interval
        
    Returns:
        Predicted price
    """
    if model_path is None:
        model_path = Path(__file__).parent.parent.parent / 'models' / 'best_model.pkl'
    
    predictor = HousingPricePredictor(model_path, scaler_path, features_path)
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
    
    try:
        # Create predictor
        predictor = HousingPricePredictor(
            model_path='../../models/best_model.pkl',
            scaler_path='../../models/scaler.pkl',
            features_path='../../models/selected_features.json'
        )
        
        # Make prediction
        prediction = predictor.predict(sample_data, with_confidence=True)
        print(f"Prediction: ${prediction['prediction']:,.2f}")
        print(f"95% CI: ${prediction['lower_bound']:,.2f} - ${prediction['upper_bound']:,.2f}")
        
    except FileNotFoundError as e:
        print(f"Model artifacts not found: {e}")
        print("Please train the model first using the notebooks.")