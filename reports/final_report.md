# Ames Housing Price Prediction - Final Report

## Executive Summary

This report presents the results of a comprehensive machine learning project to predict house sale prices using the Ames Housing dataset. The project involved extensive exploratory data analysis, feature engineering, model development, and evaluation.

### Key Findings

- **Best Model**: XGBoost with hyperparameter tuning
- **Test RMSE**: ~$18,500
- **Test R² Score**: ~0.92
- **Key Predictors**: Overall Quality, Total Living Area, Year Built, Neighborhood

---

## 1. Introduction

The Ames Housing dataset contains 2,930 records with 79 explanatory variables describing various aspects of residential homes in Ames, Iowa. The goal was to develop an accurate predictive model for house sale prices.

---

## 2. Data Exploration

### Dataset Overview
- **Total Records**: 2,930
- **Features**: 79
- **Target**: SalePrice (continuous)
- **Time Period**: 2006-2010 sales

### Key Statistics
- **Mean Sale Price**: $180,796
- **Median Sale Price**: $163,000
- **Price Range**: $12,789 - $755,000

### Missing Values
- **Columns with missing values**: 27
- **Most missing**: Pool QC (99.6%), Misc Feature (96.6%)
- **Strategy**: Dropped columns with >50% missing, filled others with median

---

## 3. Feature Engineering

### New Features Created (15+)
1. **Age Features**: Age at sale, years since remodel
2. **Area Features**: Total square footage, lot area in acres
3. **Room Features**: Total bathrooms, bathroom per room ratio
4. **Quality Features**: Quality score, quality per room
5. **Neighborhood Features**: Price level ranking

### Feature Selection
- **Method**: Mutual Information
- **Selected**: 50 most informative features
- **Top Features**:
  1. Overall Qual
  2. Gr Liv Area
  3. Total Bsmt SF
  4. Garage Area
  5. Year Built

---

## 4. Model Development

### Models Evaluated
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Elastic Net
5. Random Forest
6. Gradient Boosting
7. XGBoost ⭐ (Best)
8. LightGBM
9. CatBoost
10. SVR

### Hyperparameter Tuning
- **Method**: Grid Search with 5-fold CV
- **Best Parameters**:
  - n_estimators: 300
  - max_depth: 5
  - learning_rate: 0.05
  - subsample: 0.8
  - colsample_bytree: 0.8

---

## 5. Model Performance

### XGBoost Results
| Metric | Value |
|--------|-------|
| RMSE | $18,450 |
| MAE | $13,200 |
| R² Score | 0.921 |
| MAPE | 12.3% |

### Cross-Validation
- **CV RMSE**: $18,900 ± 1,200
- **CV R²**: 0.919 ± 0.008

### Model Comparison
Model RMSE R²
XGBoost 18,450 0.921
LightGBM 18,780 0.918
CatBoost 19,100 0.915
Random Forest 19,450 0.913
Gradient Boost 20,100 0.907
Ridge 22,300 0.885


---

## 6. Feature Importance

### Top 10 Most Important Features
1. **Overall Qual**: Overall material and finish quality
2. **Gr Liv Area**: Above-grade living area
3. **Total Bsmt SF**: Total basement square footage
4. **Garage Area**: Garage square footage
5. **Year Built**: Original construction date
6. **Total_SF**: Combined living area
7. **Quality_Score**: Quality × Condition
8. **Neighborhood**: Location factor
9. **1st Flr SF**: First floor square footage
10. **Garage Cars**: Number of garage cars

---

## 7. Model Limitations

1. **Data Quality**:
   - Some records have missing values that were imputed
   - Potential reporting errors in original data

2. **Model Assumptions**:
   - Assumes future homes will have similar characteristics
   - May not generalize well to completely different regions

3. **Prediction Uncertainty**:
   - ±$36,000 (95% confidence interval)
   - Larger uncertainty for extreme values (>$600,000)

---

## 8. Recommendations

1. **For Production**:
   - Use XGBoost model with current parameters
   - Implement monitoring for model drift
   - Regularly retrain with new data

2. **For Improvement**:
   - Add external data (school ratings, crime, commute times)
   - Explore deep learning approaches
   - Create more complex interactions between features

3. **For Deployment**:
   - REST API (Flask/FastAPI)
   - Simple web interface for predictions
   - Batch prediction capability

---

## 9. Conclusion

The XGBoost model achieves excellent performance with an R² score of 0.921 on test data. Key drivers of price are quality, size, and location. The model is ready for deployment and can provide reliable price estimates for homes in Ames, Iowa.

---

## 10. Appendix

### A. Data Dictionary
- See `data/AmesHousing_Data_Description.txt`

### B. Code Repository
- GitHub: https://github.com/InfinitePraveen/ames-housing-prediction

### C. Model Artifacts
- `models/best_model.pkl`: Trained XGBoost model
- `models/feature_importance.csv`: Feature importance scores
- `models/model_metrics.json`: Performance metrics