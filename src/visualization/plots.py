"""Plot generation for model evaluation."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


def create_model_comparison_plot(results_df, save_path=None):
    """
    Create model comparison bar plots.
    
    Args:
        results_df: DataFrame with model results
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sort by RMSE
    results_df = results_df.sort_values('test_rmse')
    
    # RMSE comparison
    axes[0].barh(results_df['model'], results_df['test_rmse'], 
                 color='skyblue', edgecolor='black')
    axes[0].set_title('Model RMSE Comparison')
    axes[0].set_xlabel('RMSE')
    axes[0].axvline(results_df['test_rmse'].min(), color='green', linestyle='--', 
                   linewidth=2, label=f'Best: {results_df["test_rmse"].min():.2f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(results_df.iterrows()):
        axes[0].text(row['test_rmse'] + 0.05, i, f'{row["test_rmse"]:.2f}', va='center')
    
    # R² comparison
    axes[1].barh(results_df['model'], results_df['test_r2'], 
                 color='lightgreen', edgecolor='black')
    axes[1].set_title('Model R² Comparison')
    axes[1].set_xlabel('R² Score')
    axes[1].axvline(results_df['test_r2'].max(), color='green', linestyle='--', 
                   linewidth=2, label=f'Best: {results_df["test_r2"].max():.4f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(results_df.iterrows()):
        axes[1].text(row['test_r2'] + 0.002, i, f'{row["test_r2"]:.4f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f'Model comparison plot saved to {save_path}')
    
    plt.show()


def create_feature_importance_plot(importance_df, n_top=20, save_path=None):
    """
    Create feature importance bar plot.
    
    Args:
        importance_df: DataFrame with feature importance
        n_top: Number of top features to show
        save_path: Path to save the figure
    """
    top_features = importance_df.head(n_top)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    bars = ax.barh(top_features['feature'], top_features['importance'], 
                   color='coral', edgecolor='black', alpha=0.8)
    
    # Color the top bar differently
    bars[0].set_color('darkred')
    
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {n_top} Feature Importances')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    # Add value labels
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row['importance'] + 0.002, i, f'{row["importance"]:.3f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f'Feature importance plot saved to {save_path}')
    
    plt.show()


def create_residual_plot(y_true, y_pred, save_path=None):
    """
    Create residual analysis plots.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save the figure
    """
    residuals = y_true - y_pred
    standardized_residuals = residuals / residuals.std()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=30)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
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
    axes[1, 0].hist(standardized_residuals, bins=30, edgecolor='black', 
                    color='skyblue', alpha=0.7, density=True)
    x = np.linspace(-4, 4, 100)
    axes[1, 0].plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='Normal Distribution')
    axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set_xlabel('Standardized Residuals')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Actual vs Predicted
    axes[1, 1].scatter(y_true, y_pred, alpha=0.5, s=30)
    axes[1, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                    'red', linestyle='--', linewidth=2, label='Perfect Prediction')
    axes[1, 1].set_xlabel('Actual Values')
    axes[1, 1].set_ylabel('Predicted Values')
    axes[1, 1].set_title('Actual vs Predicted')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Model Residual Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f'Residual plot saved to {save_path}')
    
    plt.show()