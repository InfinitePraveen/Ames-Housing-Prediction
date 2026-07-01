"""Feature engineering module for Ames Housing dataset."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def create_new_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features based on domain knowledge.
    
    Args:
        data: Input DataFrame
        
    Returns:
        DataFrame with new features
    """
    df = data.copy()
    
    # 1. Age features
    df['Age_at_Sale'] = df['Yr Sold'] - df['Year Built']
    df['Years_Since_Remodel'] = df['Yr Sold'] - df['Year Remod/Add']
    df['Has_Remodel'] = (df['Year Remod/Add'] > df['Year Built']).astype(int)
    
    # 2. Total SF features
    df['Total_SF'] = df['1st Flr SF'] + df['2nd Flr SF']
    df['Total_Basement_SF'] = df['Total Bsmt SF']
    df['Total_Area'] = df['Total_SF'] + df['Total_Basement_SF']
    
    # 3. Room features
    df['Total_Bathrooms'] = (df['Bsmt Full Bath'] + df['Bsmt Half Bath'] * 0.5 + 
                             df['Full Bath'] + df['Half Bath'] * 0.5)
    df['Total_Rooms'] = df['TotRms AbvGrd'] + df['Bedroom AbvGr']
    df['Bedroom_Ratio'] = df['Bedroom AbvGr'] / df['TotRms AbvGrd']
    
    # 4. Porch features
    df['Total_Porch_SF'] = (df['Wood Deck SF'] + df['Open Porch SF'] + 
                           df['Enclosed Porch'] + df['3Ssn Porch'] + df['Screen Porch'])
    df['Has_Porch'] = (df['Total_Porch_SF'] > 0).astype(int)
    
    # 5. Garage features
    df['Garage_Cars_Per_Area'] = df['Garage Cars'] / (df['Garage Area'] + 1)
    df['Has_Garage'] = (df['Garage Area'] > 0).astype(int)
    
    # 6. Lot features
    df['Lot_Frontage_to_Area'] = df['Lot Frontage'] / (df['Lot Area'] + 1)
    df['Lot_Shape_Score'] = df['Lot Shape'].map({'Reg': 1, 'IR1': 2, 'IR2': 3, 'IR3': 4})
    
    # 7. Quality features
    df['Total_Quality'] = df['Overall Qual'] * df['Overall Cond']
    df['Avg_Quality_Per_Room'] = df['Overall Qual'] / (df['TotRms AbvGrd'] + 1)
    
    # 8. Miscellaneous
    df['Is_New_Construction'] = (df['Sale Condition'] == 'New').astype(int)
    df['Is_Family_Sale'] = (df['Sale Condition'] == 'Family').astype(int)
    df['Sale_Year_Month'] = df['Mo Sold'] + (df['Yr Sold'] - 2000) * 12
    
    return df


def encode_categorical(data: pd.DataFrame, method: str = 'label') -> pd.DataFrame:
    """
    Encode categorical variables.
    
    Args:
        data: Input DataFrame
        method: 'label' or 'onehot'
        
    Returns:
        DataFrame with encoded categorical variables
    """
    df = data.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if method == 'label':
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
        logger.info(f"Label encoded {len(categorical_cols)} categorical variables")
    
    elif method == 'onehot':
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        logger.info(f"One-hot encoded {len(categorical_cols)} categorical variables")
    
    return df


def preprocess_pipeline(data: pd.DataFrame, 
                       test_size: float = 0.2,
                       scale: bool = True,
                       encode: bool = True) -> tuple:
    """
    Complete preprocessing pipeline.
    
    Args:
        data: Input DataFrame
        test_size: Proportion for test set
        scale: Whether to scale numeric features
        encode: Whether to encode categorical features
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor)
    """
    df = data.copy()
    
    # Separate target
    y = df['SalePrice']
    X = df.drop(columns=['SalePrice'])
    
    # Create new features
    X = create_new_features(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Encode categorical features
    if encode:
        X_train = encode_categorical(X_train)
        X_test = encode_categorical(X_test)
    
    # Scale numeric features
    if scale:
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    return X_train, X_test, y_train, y_test