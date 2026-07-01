"""Visualization utilities for Ames Housing."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def set_plot_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette('husl')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12


def plot_feature_distribution(df, features, n_cols=3, save_path=None):
    """
    Plot distribution of selected features.
    
    Args:
        df: DataFrame
        features: List of feature names
        n_cols: Number of columns in subplot grid
        save_path: Path to save the figure
    """
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, feature in enumerate(features):
        if i < len(axes):
            if df[feature].dtype == 'object':
                # Categorical feature
                df[feature].value_counts().plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'Distribution of {feature}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Count')
                axes[i].tick_params(axis='x', rotation=45)
            else:
                # Numeric feature
                axes[i].hist(df[feature], bins=30, edgecolor='black', alpha=0.7)
                axes[i].set_title(f'Distribution of {feature}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
                axes[i].axvline(df[feature].mean(), color='r', linestyle='--', 
                               label=f'Mean: {df[feature].mean():.2f}')
                axes[i].axvline(df[feature].median(), color='g', linestyle='--',
                               label=f'Median: {df[feature].median():.2f}')
                axes[i].legend()
            
            axes[i].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f'Plot saved to {save_path}')
    
    plt.show()


def plot_correlation_heatmap(df, features=None, save_path=None):
    """
    Plot correlation heatmap for numeric features.
    
    Args:
        df: DataFrame
        features: List of features (if None, use all numeric)
        save_path: Path to save the figure
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr = df[features].corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, 
                square=True, linewidths=0.5, 
                cbar_kws={"shrink": 0.8},
                ax=ax)
    
    ax.set_title('Correlation Heatmap')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f'Correlation heatmap saved to {save_path}')
    
    plt.show()
    
    return corr


def plot_correlation_with_target(df, target='SalePrice', n_top=20, save_path=None):
    """
    Plot correlation of features with target variable.
    
    Args:
        df: DataFrame
        target: Target column name
        n_top: Number of top features to show
        save_path: Path to save the figure
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    
    correlations = {}
    for col in numeric_cols:
        correlations[col] = df[col].corr(df[target])
    
    corr_df = pd.DataFrame({
        'feature': list(correlations.keys()),
        'correlation': list(correlations.values())
    }).sort_values('correlation', ascending=False)
    
    top_features = corr_df.head(n_top)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['green' if x > 0 else 'red' for x in top_features['correlation']]
    ax.barh(top_features['feature'], top_features['correlation'], color=colors, alpha=0.7)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel(f'Correlation with {target}')
    ax.set_title(f'Top {n_top} Features Correlated with {target}')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row['correlation'] + 0.02 if row['correlation'] > 0 else row['correlation'] - 0.02,
                i, f'{row["correlation"]:.3f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f'Correlation plot saved to {save_path}')
    
    plt.show()