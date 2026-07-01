"""Data loading module for Ames Housing dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def load_ames_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the Ames Housing dataset.
    
    Args:
        file_path: Path to the CSV file. If None, uses default path.
        
    Returns:
        DataFrame with the Ames Housing data
    """
    if file_path is None:
        file_path = Path(__file__).parent.parent.parent / "data" / "raw" / "AmesHousing.csv"
    
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Loaded {len(data)} records with {len(data.columns)} columns")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def get_data_info(data: pd.DataFrame) -> dict:
    """
    Get comprehensive information about the dataset.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Dictionary with data information
    """
    return {
        "shape": data.shape,
        "columns": list(data.columns),
        "dtypes": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "missing_percentage": (data.isnull().sum() / len(data) * 100).to_dict(),
        "numeric_columns": data.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": data.select_dtypes(include=['object']).columns.tolist(),
        "memory_usage": data.memory_usage(deep=True).sum() / (1024 ** 2),  # MB
        "target_stats": data['SalePrice'].describe().to_dict() if 'SalePrice' in data else {}
    }


def get_feature_types(data: pd.DataFrame) -> dict:
    """
    Classify columns into different feature types.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Dictionary with feature types
    """
    # Identify ID columns
    id_cols = [col for col in data.columns if 'id' in col.lower() or 'pid' in col.lower()]
    
    # Identify numeric features
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Identify categorical features
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target and ID from feature lists
    if 'SalePrice' in numeric_cols:
        numeric_cols.remove('SalePrice')
    
    return {
        'id_columns': id_cols,
        'numeric_features': numeric_cols,
        'categorical_features': categorical_cols,
        'target': 'SalePrice' if 'SalePrice' in data else None
    }


if __name__ == "__main__":
    # Test the loading functionality
    logging.basicConfig(level=logging.INFO)
    df = load_ames_data()
    info = get_data_info(df)
    print(f"Dataset Shape: {info['shape']}")
    print(f"Numeric Columns: {len(info['numeric_columns'])}")
    print(f"Categorical Columns: {len(info['categorical_columns'])}")
    print(f"Missing Values: {sum(v > 0 for v in info['missing_percentage'].values())} columns")