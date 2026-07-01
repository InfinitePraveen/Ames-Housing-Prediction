"""Preprocessing pipeline for Ames Housing dataset."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_features(df):
    """Create new features from existing data."""
    df_new = df.copy()
    
    # 1. Age features
    df_new['Age'] = df_new['Yr Sold'] - df_new['Year Built']
    df_new['Age_At_Remodel'] = df_new['Year Remod/Add'] - df_new['Year Built']
    df_new['Is_Remodeled'] = (df_new['Age_At_Remodel'] > 0).astype(int)
    
    # 2. Area features
    df_new['Total_SF'] = df_new['1st Flr SF'] + df_new['2nd Flr SF']
    df_new['Total_Basement_SF'] = df_new['Total Bsmt SF']
    df_new['Total_Area'] = df_new['Total_SF'] + df_new['Total_Basement_SF']
    df_new['SF_per_Lot'] = df_new['Total_Area'] / (df_new['Lot Area'] + 1)
    
    # 3. Room features
    df_new['Total_Bathrooms'] = (df_new['Bsmt Full Bath'] + 
                                df_new['Bsmt Half Bath'] * 0.5 + 
                                df_new['Full Bath'] + 
                                df_new['Half Bath'] * 0.5)
    df_new['Total_Rooms'] = df_new['TotRms AbvGrd'] + df_new['Bedroom AbvGr']
    df_new['Bathroom_Per_Room'] = df_new['Total_Bathrooms'] / (df_new['TotRms AbvGrd'] + 1)
    
    # 4. Porch features
    df_new['Total_Porch_SF'] = (df_new['Wood Deck SF'] + 
                               df_new['Open Porch SF'] + 
                               df_new['Enclosed Porch'] + 
                               df_new['3Ssn Porch'] + 
                               df_new['Screen Porch'])
    df_new['Has_Porch'] = (df_new['Total_Porch_SF'] > 0).astype(int)
    df_new['Has_Pool'] = (df_new['Pool Area'] > 0).astype(int)
    df_new['Has_Fireplace'] = (df_new['Fireplaces'] > 0).astype(int)
    
    # 5. Garage features
    df_new['Has_Garage'] = (df_new['Garage Area'] > 0).astype(int)
    df_new['Garage_Cars_Per_Area'] = df_new['Garage Cars'] / (df_new['Garage Area'] + 1) * 100
    
    # 6. Quality features
    df_new['Quality_Score'] = df_new['Overall Qual'] * df_new['Overall Cond']
    df_new['Quality_Per_Room'] = df_new['Overall Qual'] / (df_new['TotRms AbvGrd'] + 1)
    
    # 7. Sale features
    df_new['Is_New_Construction'] = (df_new['Sale Condition'] == 'New').astype(int)
    df_new['Is_Abnormal_Sale'] = (df_new['Sale Condition'].isin(['Abnorml', 'AdjLand', 'Alloca'])).astype(int)
    
    # 8. Neighborhood features
    neighborhood_price = df_new.groupby('Neighborhood')['SalePrice'].median()
    df_new['Neighborhood_Price_Level'] = df_new['Neighborhood'].map(neighborhood_price)
    df_new['Neighborhood_Price_Level_Rank'] = df_new['Neighborhood_Price_Level'].rank(pct=True)
    
    return df_new


def encode_categorical(df, method='label'):
    """Encode categorical variables."""
    df_encoded = df.copy()
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
    
    logger.info(f'Encoded {len(categorical_cols)} categorical variables')
    return df_encoded, label_encoders


def preprocess_pipeline(df, test_size=0.2, random_state=42, feature_selection=True):
    """
    Complete preprocessing pipeline.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    # Create features
    df_featured = create_features(df)
    logger.info(f'Created features, shape: {df_featured.shape}')
    
    # Encode categorical
    df_encoded, label_encoders = encode_categorical(df_featured)
    
    # Separate target
    X = df_encoded.drop('SalePrice', axis=1)
    y = df_encoded['SalePrice']
    
    # Transform skewed features
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    skewness = X[numeric_cols].skew()
    highly_skewed = skewness[skewness > 1].index.tolist()
    
    for col in highly_skewed:
        if (X[col] > 0).all():
            X[col] = np.log1p(X[col])
        else:
            pt = PowerTransformer(method='yeo-johnson')
            X[col] = pt.fit_transform(X[[col]])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Feature selection
    if feature_selection:
        selector = SelectKBest(score_func=mutual_info_regression, k=50)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
        selected_features = X.columns[selector.get_support()].tolist()
        logger.info(f'Selected {len(selected_features)} features')
    else:
        selected_features = X.columns.tolist()
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    logger.info(f'Preprocessing complete. Train: {X_train.shape}, Test: {X_test.shape}')
    
    return X_train, X_test, y_train, y_test, selected_features, scaler


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from load_data import load_ames_data
    
    df = load_ames_data()
    X_train, X_test, y_train, y_test, features, scaler = preprocess_pipeline(df)
    
    # Save processed data
    df_processed = pd.DataFrame(X_train, columns=features)
    df_processed['SalePrice'] = y_train.values
    df_processed.to_csv('../data/processed/ames_processed_train.csv', index=False)
    
    df_test_processed = pd.DataFrame(X_test, columns=features)
    df_test_processed['SalePrice'] = y_test.values
    df_test_processed.to_csv('../data/processed/ames_processed_test.csv', index=False)
    
    print('Preprocessing complete. Processed data saved.')