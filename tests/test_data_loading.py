"""Tests for data loading module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load_data import load_ames_data, get_data_info, get_feature_types


def test_load_data():
    """Test loading the Ames Housing dataset."""
    df = load_ames_data()
    
    # Check shape
    assert df.shape[0] > 0, "Dataset should not be empty"
    assert df.shape[1] > 0, "Dataset should have columns"
    
    # Check required columns
    required_cols = ['SalePrice', 'MS SubClass', 'MS Zoning', 'Lot Area']
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_get_data_info():
    """Test data info function."""
    df = load_ames_data()
    info = get_data_info(df)
    
    assert 'shape' in info
    assert 'columns' in info
    assert 'missing_values' in info
    assert 'target_stats' in info
    
    # Check target stats
    assert 'SalePrice' in info['target_stats']
    assert info['target_stats']['count'] == len(df)


def test_get_feature_types():
    """Test feature type classification."""
    df = load_ames_data()
    types = get_feature_types(df)
    
    assert 'id_columns' in types
    assert 'numeric_features' in types
    assert 'categorical_features' in types
    assert 'target' in types
    
    assert types['target'] == 'SalePrice'
    assert len(types['numeric_features']) > 0
    assert len(types['categorical_features']) > 0


def test_handle_missing_values():
    """Test handling of missing values."""
    from src.data.clean_data import clean_ames_data
    
    df = load_ames_data()
    df_clean = clean_ames_data(df)
    
    # Check that missing values are handled
    total_missing = df_clean.isnull().sum().sum()
    assert total_missing == 0, f"Missing values remain: {total_missing}"