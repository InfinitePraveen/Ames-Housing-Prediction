"""Tests for model training and evaluation."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load_data import load_ames_data
from src.data.preprocess import preprocess_pipeline
from src.models.train_model import get_models, evaluate_model, train_models


def test_get_models():
    """Test model dictionary creation."""
    models = get_models()
    
    assert len(models) > 0
    assert 'linear_regression' in models
    assert 'random_forest' in models
    assert 'xgboost' in models


def test_evaluate_model():
    """Test single model evaluation."""
    df = load_ames_data()
    X_train, X_test, y_train, y_test, _, _ = preprocess_pipeline(
        df, test_size=0.2, feature_selection=False
    )
    
    models = get_models()
    model = models['linear_regression']
    
    metrics, trained_model, predictions = evaluate_model(
        model, X_train, y_train, X_test, y_test
    )
    
    # Check metrics
    assert 'train_rmse' in metrics
    assert 'test_rmse' in metrics
    assert 'train_r2' in metrics
    assert 'test_r2' in metrics
    
    # Check predictions
    assert len(predictions) == len(y_test)


def test_train_models():
    """Test training multiple models."""
    df = load_ames_data()
    X_train, X_test, y_train, y_test, _, _ = preprocess_pipeline(
        df, test_size=0.2, feature_selection=False
    )
    
    # Use subset of models for speed
    models = {
        'linear_regression': get_models()['linear_regression'],
        'random_forest': get_models()['random_forest']
    }
    
    results, trained_models = train_models(
        X_train, y_train, X_test, y_test, models
    )
    
    # Check results
    assert len(results) == len(models)
    assert 'model' in results.columns
    assert 'test_rmse' in results.columns
    assert 'test_r2' in results.columns
    
    # Check trained models
    assert len(trained_models) == len(models)