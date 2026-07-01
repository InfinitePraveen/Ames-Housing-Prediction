"""Model training module for Ames Housing prediction."""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import yaml

logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from yaml file."""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_models():
    """Return dictionary of models to evaluate."""
    return {
        'linear_regression': LinearRegression(),
        'ridge': Ridge(random_state=42),
        'lasso': Lasso(random_state=42),
        'elastic_net': ElasticNet(random_state=42),
        'random_forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(random_state=42),
        'xgboost': XGBRegressor(random_state=42, n_jobs=-1),
        'lightgbm': LGBMRegressor(random_state=42, n_jobs=-1),
        'catboost': CatBoostRegressor(random_state=42, verbose=0),
        'svr': SVR()
    }


def evaluate_model(model, X_train, y_train, X_test, y_test, cv_folds=5):
    """
    Evaluate a single model.
    
    Returns dictionary of metrics.
    """
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, 
                                scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'cv_rmse_mean': cv_rmse.mean(),
        'cv_rmse_std': cv_rmse.std()
    }
    
    return metrics, model, y_pred_test


def train_models(X_train, y_train, X_test, y_test, models=None):
    """
    Train and evaluate multiple models.
    
    Returns DataFrame with results and trained models.
    """
    if models is None:
        models = get_models()
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        try:
            metrics, trained_model, predictions = evaluate_model(
                model, X_train, y_train, X_test, y_test
            )
            metrics['model'] = name
            results.append(metrics)
            trained_models[name] = trained_model
            logger.info(f"{name} - Test RMSE: {metrics['test_rmse']:.2f}")
        except Exception as e:
            logger.error(f"Error training {name}: {e}")
    
    return pd.DataFrame(results), trained_models


def tune_best_model(X_train, y_train, model_type='xgboost'):
    """
    Hyperparameter tuning for the best performing model.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to tune
        
    Returns:
        Best model and parameter grid results
    """
    param_grids = {
        'xgboost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        },
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, 25],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'lightgbm': {
            'n_estimators': [100, 200, 300],
            'num_leaves': [31, 50, 80],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [5, 10, 15]
        },
        'catboost': {
            'iterations': [100, 200, 300],
            'depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5, 7]
        }
    }
    
    if model_type not in param_grids:
        raise ValueError(f"Model type {model_type} not supported")
    
    models = {
        'xgboost': XGBRegressor(random_state=42, n_jobs=-1),
        'random_forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'lightgbm': LGBMRegressor(random_state=42, n_jobs=-1),
        'catboost': CatBoostRegressor(random_state=42, verbose=0)
    }
    
    model = models[model_type]
    param_grid = param_grids[model_type]
    
    grid_search = GridSearchCV(
        model, param_grid, cv=5, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1, verbose=1
    )
    
    logger.info(f"Starting hyperparameter tuning for {model_type}...")
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best score: {-grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_


if __name__ == "__main__":
    # This is a placeholder - the actual training will be in notebooks
    logging.basicConfig(level=logging.INFO)
    print("Model training module loaded successfully")