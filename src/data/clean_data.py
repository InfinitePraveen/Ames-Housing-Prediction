"""Data cleaning module for Ames Housing dataset."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def clean_ames_data(data: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Clean the Ames Housing dataset.
    
    Args:
        data: Raw DataFrame
        config: Configuration dictionary
        
    Returns:
        Cleaned DataFrame
    """
    df = data.copy()
    
    # Remove ID columns
    id_cols = [col for col in df.columns if 'id' in col.lower() or 'pid' in col.lower()]
    if id_cols:
        df = df.drop(columns=id_cols)
        logger.info(f"Removed ID columns: {id_cols}")
    
    # Handle missing values for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        # Fill 'NA' as a category for missing values
        if df[col].isnull().any():
            df[col] = df[col].fillna('NA')
            logger.info(f"Filled missing values in {col} with 'NA'")
    
    # Handle missing values for numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'SalePrice']
    
    threshold = config.get('missing_threshold', 0.5) if config else 0.5
    
    for col in numeric_cols:
        missing_pct = df[col].isnull().mean()
        if missing_pct > threshold:
            # Drop columns with too many missing values
            df = df.drop(columns=[col])
            logger.info(f"Dropped {col} (missing: {missing_pct:.1%})")
        elif missing_pct > 0:
            # Fill missing with median for remaining columns
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Filled missing values in {col} with median ({median_val})")
    
    return df


def detect_outliers_iqr(data: pd.DataFrame, features: list) -> Dict:
    """
    Detect outliers using IQR method.
    
    Args:
        data: DataFrame
        features: List of feature names
        
    Returns:
        Dictionary with outlier info per feature
    """
    outlier_info = {}
    
    for feature in features:
        if feature in data.columns:
            Q1 = data[feature].quantile(0.25)
            Q3 = data[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
            outlier_info[feature] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(data) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'min': data[feature].min(),
                'max': data[feature].max()
            }
    
    return outlier_info


def handle_outliers(data: pd.DataFrame, method: str = 'cap', features: list = None) -> pd.DataFrame:
    """
    Handle outliers in the dataset.
    
    Args:
        data: Input DataFrame
        method: 'cap', 'remove', or 'none'
        features: List of features to process (if None, process all numeric except target)
        
    Returns:
        DataFrame with outliers handled
    """
    df = data.copy()
    
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'SalePrice' in features:
            features.remove('SalePrice')
    
    if method == 'cap':
        for feature in features:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)
            logger.info(f"Capped outliers in {feature}")
    
    elif method == 'remove':
        outlier_mask = pd.Series(False, index=df.index)
        for feature in features:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            feature_outliers = (df[feature] < lower_bound) | (df[feature] > upper_bound)
            outlier_mask = outlier_mask | feature_outliers
        
        df = df[~outlier_mask]
        logger.info(f"Removed {outlier_mask.sum()} rows with outliers")
    
    return df


if __name__ == "__main__":
    from load_data import load_ames_data
    
    logging.basicConfig(level=logging.INFO)
    df = load_ames_data()
    df_clean = clean_ames_data(df)
    print(f"Clean shape: {df_clean.shape}")
    
    # Check outliers on SalePrice
    outliers = detect_outliers_iqr(df_clean, ['SalePrice'])
    print(f"SalePrice outliers: {outliers['SalePrice']['count']} ({outliers['SalePrice']['percentage']:.1f}%)")