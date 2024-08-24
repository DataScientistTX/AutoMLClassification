import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame, target_column: str = 'target') -> pd.DataFrame:
    """
    Preprocess the data by handling missing values, encoding categorical variables,
    scaling numerical variables, and creating new features.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target column. Default is 'target'.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    logger.info("Starting data preprocessing")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    logger.info(f"Numeric features: {numeric_features}")
    logger.info(f"Categorical features: {categorical_features}")
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after preprocessing
    onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = onehot_encoder.get_feature_names(categorical_features)
    feature_names = list(numeric_features) + list(cat_feature_names)
    
    # Convert to DataFrame
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
    
    # Combine processed features with target
    df_processed = pd.concat([X_processed_df, y], axis=1)
    
    logger.info(f"Data preprocessing completed. Shape: {df_processed.shape}")
    
    return df_processed

def create_interaction_features(df: pd.DataFrame, feature_pairs: list) -> pd.DataFrame:
    """
    Create interaction features by multiplying specified pairs of numerical features.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_pairs (list): List of tuples containing pairs of feature names to interact.

    Returns:
        pd.DataFrame: DataFrame with added interaction features.
    """
    df_copy = df.copy()
    for pair in feature_pairs:
        if pair[0] in df.columns and pair[1] in df.columns:
            new_feature_name = f"{pair[0]}_{pair[1]}_interaction"
            df_copy[new_feature_name] = df[pair[0]] * df[pair[1]]
            logger.info(f"Created interaction feature: {new_feature_name}")
        else:
            logger.warning(f"Could not create interaction feature for {pair}. One or both features not found in DataFrame.")
    return df_copy

def create_polynomial_features(df: pd.DataFrame, features: list, degree: int = 2) -> pd.DataFrame:
    """
    Create polynomial features for specified numerical features.

    Args:
        df (pd.DataFrame): Input DataFrame.
        features (list): List of feature names to create polynomial features for.
        degree (int): Degree of the polynomial. Default is 2.

    Returns:
        pd.DataFrame: DataFrame with added polynomial features.
    """
    df_copy = df.copy()
    for feature in features:
        if feature in df.columns:
            for d in range(2, degree + 1):
                new_feature_name = f"{feature}_degree_{d}"
                df_copy[new_feature_name] = df[feature] ** d
                logger.info(f"Created polynomial feature: {new_feature_name}")
        else:
            logger.warning(f"Could not create polynomial features for {feature}. Feature not found in DataFrame.")
    return df_copy

def feature_selection(df: pd.DataFrame, target_column: str, top_n: int = 10) -> pd.DataFrame:
    """
    Perform simple feature selection based on correlation with the target variable.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target column.
        top_n (int): Number of top features to select. Default is 10.

    Returns:
        pd.DataFrame: DataFrame with selected features and target column.
    """
    # Calculate correlation with target
    correlations = df.corr()[target_column].abs().sort_values(ascending=False)
    
    # Select top N features (excluding the target itself)
    top_features = correlations[1:top_n+1].index.tolist()
    
    logger.info(f"Selected top {len(top_features)} features based on correlation: {top_features}")
    
    # Return DataFrame with selected features and target
    return df[top_features + [target_column]]

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'numeric1': [1, 2, 3, 4, 5],
        'numeric2': [5, 4, 3, 2, 1],
        'categorical1': ['A', 'B', 'A', 'C', 'B'],
        'categorical2': ['X', 'Y', 'Z', 'X', 'Y'],
        'target': [0, 1, 0, 1, 1]
    })
    
    processed_data = preprocess_data(sample_data)
    print(processed_data)
    
    interaction_data = create_interaction_features(sample_data, [('numeric1', 'numeric2')])
    print(interaction_data)
    
    poly_data = create_polynomial_features(sample_data, ['numeric1', 'numeric2'])
    print(poly_data)
    
    selected_data = feature_selection(processed_data, 'target', top_n=3)
    print(selected_data)