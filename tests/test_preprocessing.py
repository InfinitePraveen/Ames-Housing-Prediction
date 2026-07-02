"""Tests for preprocessing module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load_data import load_ames_data
from src.data.preprocess import create_features, encode_categorical, preprocess_pipeline


def test_create_features():
    """Test feature creation."""
    df = load_ames_data()
    df_new = create_features(df)
    
    # Check new features were added
    assert df_new.shape[1] > df.shape[1]
    
    # Check specific features
    new_features = ['Age', 'Total_SF', 'Total_Bathrooms', 'Total_Porch_SF']
    for feature in new_features:
        assert feature in df_new.columns


def test_encode_categorical():
    """Test categorical encoding."""
    df = load_ames_data()
    df_encoded, encoders = encode_categorical(df)
    
    # Check no object columns remain
    object_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    assert len(object_cols) == 0
    
    # Check we got encoders for categorical columns
    original_categorical = df.select_dtypes(include=['object']).columns.tolist()
    for col in original_categorical:
        assert col in encoders


def test_preprocess_pipeline():
    """Test full preprocessing pipeline."""
    df = load_ames_data()
    X_train, X_test, y_train, y_test, features, scaler = preprocess_pipeline(
        df, test_size=0.2, feature_selection=False
    )
    
    # Check splits
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
    
    # Check features
    assert len(features) > 0
    
    # Check scaler
    assert scaler is not None