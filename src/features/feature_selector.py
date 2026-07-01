"""Feature selection module for Ames Housing."""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)


def select_features(X_train, y_train, X_test=None, k=50, method='mutual_info'):
    """
    Select top k features using specified method.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features (optional)
        k: Number of features to select
        method: 'mutual_info', 'f_regression', or 'random_forest'
        
    Returns:
        Selected features, transformed data
    """
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
    elif method == 'f_regression':
        selector = SelectKBest(score_func=f_regression, k=k)
    elif method == 'random_forest':
        # Use random forest for feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_
        feature_names = X_train.columns if hasattr(X_train, 'columns') else range(X_train.shape[1])
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        selected_features = feature_importance.head(k)['feature'].tolist()
        X_train_selected = X_train[selected_features] if hasattr(X_train, 'columns') else X_train[:, selected_features]
        X_test_selected = X_test[selected_features] if X_test is not None and hasattr(X_test, 'columns') else None
        
        return selected_features, X_train_selected, X_test_selected
    else:
        raise ValueError(f"Unknown method: {method}")
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test) if X_test is not None else None
    
    # Get selected feature names
    if hasattr(X_train, 'columns'):
        selected_mask = selector.get_support()
        selected_features = X_train.columns[selected_mask].tolist()
    else:
        selected_features = list(range(k))
    
    logger.info(f'Selected {len(selected_features)} features using {method}')
    
    return selected_features, X_train_selected, X_test_selected


def get_feature_importance(model, X, feature_names=None):
    """
    Extract feature importance from trained model.
    
    Returns:
        DataFrame with feature importance
    """
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        logger.warning('Model does not have feature_importances_ or coef_ attribute')
        return None
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Normalize importance
    importance_df['importance_normalized'] = importance_df['importance'] / importance_df['importance'].sum()
    
    return importance_df


def correlation_analysis(X, y, threshold=0.3):
    """
    Analyze correlation between features and target.
    
    Returns:
        DataFrame with correlation scores
    """
    if hasattr(X, 'columns'):
        feature_names = X.columns
    else:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    correlations = []
    for i, feature in enumerate(feature_names):
        corr = np.corrcoef(X[:, i] if hasattr(X, 'shape') else X[feature], y)[0, 1]
        correlations.append({
            'feature': feature,
            'correlation': corr,
            'abs_correlation': abs(corr)
        })
    
    corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
    
    # Filter by threshold if specified
    if threshold:
        corr_df = corr_df[corr_df['abs_correlation'] > threshold]
    
    logger.info(f'Found {len(corr_df)} features with |correlation| > {threshold}')
    
    return corr_df


def remove_highly_correlated(X, threshold=0.9, feature_names=None):
    """
    Remove highly correlated features to reduce multicollinearity.
    
    Returns:
        DataFrame with reduced features
    """
    if feature_names is None:
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    corr_matrix = np.corrcoef(X.T if hasattr(X, 'T') else X.T)
    
    # Find highly correlated features
    to_drop = set()
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(corr_matrix[i, j]) > threshold:
                to_drop.add(feature_names[j])
    
    # Keep features not in to_drop
    keep_features = [f for f in feature_names if f not in to_drop]
    
    if hasattr(X, 'columns'):
        X_reduced = X[keep_features]
    else:
        # Convert indices to list if X is numpy array
        keep_indices = [feature_names.index(f) for f in keep_features if f in feature_names]
        X_reduced = X[:, keep_indices]
    
    logger.info(f'Removed {len(to_drop)} highly correlated features, kept {len(keep_features)}')
    
    return X_reduced, keep_features