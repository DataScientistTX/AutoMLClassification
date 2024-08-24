import pandas as pd
import numpy as np
from typing import Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Load data from a CSV file or pandas DataFrame.

    Args:
        file_path (Union[str, pd.DataFrame]): Path to the CSV file or a pandas DataFrame.

    Returns:
        pd.DataFrame: Loaded and initially processed data.

    Raises:
        ValueError: If the file format is not supported or if there are issues with the data.
    """
    try:
        if isinstance(file_path, str):
            logger.info(f"Loading data from file: {file_path}")
            df = pd.read_csv(file_path)
        elif isinstance(file_path, pd.DataFrame):
            logger.info("Using provided pandas DataFrame")
            df = file_path.copy()
        else:
            raise ValueError("Unsupported data format. Please provide a file path or pandas DataFrame.")

        # Perform initial data validation and cleaning
        df = _initial_data_cleaning(df)

        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def _initial_data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial data cleaning and validation.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove duplicates
    df_cleaned = df.drop_duplicates()
    if len(df_cleaned) < len(df):
        logger.warning(f"Removed {len(df) - len(df_cleaned)} duplicate rows")

    # Handle missing values
    missing_values = df_cleaned.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Missing values detected:\n{missing_values[missing_values > 0]}")
        # For numeric columns, fill with median
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].median())
        # For categorical columns, fill with mode
        categorical_columns = df_cleaned.select_dtypes(exclude=[np.number]).columns
        df_cleaned[categorical_columns] = df_cleaned[categorical_columns].fillna(df_cleaned[categorical_columns].mode().iloc[0])

    # Check for and remove constant columns
    constant_columns = [col for col in df_cleaned.columns if df_cleaned[col].nunique() == 1]
    if constant_columns:
        logger.warning(f"Removing constant columns: {constant_columns}")
        df_cleaned = df_cleaned.drop(columns=constant_columns)

    # Ensure proper data types
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            try:
                df_cleaned[col] = pd.to_datetime(df_cleaned[col])
                logger.info(f"Converted column '{col}' to datetime")
            except ValueError:
                pass  # Keep as object type if not convertible to datetime

    return df_cleaned

def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        dict: Dictionary containing data information.
    """
    info = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "column_types": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "unique_values": {col: df[col].nunique() for col in df.columns}
    }
    return info

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5],
        'D': pd.date_range(start='2021-01-01', periods=5)
    })
    
    loaded_data = load_data(sample_data)
    print(get_data_info(loaded_data))