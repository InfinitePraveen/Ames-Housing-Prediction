"""Model evaluation module."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive model metrics.
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    
    # Additional metrics
    errors = y_true - y_pred
    metrics['mean_error'] = errors.mean()
    metrics['std_error'] = errors.std()
    metrics['max_error'] = errors.max()
    metrics['min_error'] = errors.min()
    metrics['error_95_ci'] = 1.96 * errors.std()
    
    return metrics


def evaluate_model_performance(model, X_test, y_test, feature_names=None):
    """
    Comprehensive model evaluation.
    
    Returns:
        metrics dict, predictions, residual analysis
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_test)  # Use test for both
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Residual analysis
    residuals = y_test - y_pred
    residual_metrics = {
        'mean': residuals.mean(),
        'std': residuals.std(),
        'skew': pd.Series(residuals).skew(),
        'kurtosis': pd.Series(residuals).kurtosis()
    }
    
    logger.info(f'Model Evaluation: RMSE={metrics["rmse"]:.2f}, R²={metrics["r2"]:.4f}')
    
    return metrics, y_pred, residuals, residual_metrics


def plot_residuals(y_true, y_pred, save_path=None):
    """
    Create residual plots for model diagnostics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save the figure
    """
    residuals = y_true - y_pred
    standardized_residuals = residuals / residuals.std()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Q-Q Plot
    from scipy import stats
    stats.probplot(standardized_residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals Histogram
    axes[1, 0].hist(standardized_residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Standardized Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Standardized Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Actual vs Predicted
    axes[1, 1].scatter(y_true, y_pred, alpha=0.5)
    axes[1, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                    'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 1].set_xlabel('Actual Values')
    axes[1, 1].set_ylabel('Predicted Values')
    axes[1, 1].set_title('Actual vs Predicted')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f'Residual plots saved to {save_path}')
    
    plt.show()
    
    return fig


def cross_validation_scores(model, X, y, cv=5):
    """
    Perform cross-validation and return scores.
    
    Returns:
        Dictionary with CV results
    """
    from sklearn.model_selection import cross_val_score, cross_val_predict
    
    # RMSE
    rmse_scores = cross_val_score(model, X, y, cv=cv, 
                                  scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-rmse_scores)
    
    # R²
    r2_scores = cross_val_score(model, X, y, cv=cv, 
                                scoring='r2')
    
    # Predictions
    y_pred_cv = cross_val_predict(model, X, y, cv=cv)
    
    results = {
        'rmse_scores': rmse_scores,
        'rmse_mean': rmse_scores.mean(),
        'rmse_std': rmse_scores.std(),
        'r2_scores': r2_scores,
        'r2_mean': r2_scores.mean(),
        'r2_std': r2_scores.std(),
        'cv_predictions': y_pred_cv
    }
    
    logger.info(f'CV Results - RMSE: {results["rmse_mean"]:.2f} ± {results["rmse_std"]:.2f}')
    logger.info(f'CV Results - R²: {results["r2_mean"]:.4f} ± {results["r2_std"]:.4f}')
    
    return results