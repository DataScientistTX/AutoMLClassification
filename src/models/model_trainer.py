import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from typing import Dict, Tuple, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_models(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Train multiple classification models.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.

    Returns:
        Dict: Dictionary of trained models.
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Neural Network': MLPClassifier(random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        logger.info(f"Training {name}")
        model.fit(X, y)
        trained_models[name] = model

    return trained_models

def tune_hyperparameters(model, param_grid: Dict, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Tuple[object, Dict]:
    """
    Perform hyperparameter tuning using GridSearchCV.

    Args:
        model: Base model to tune.
        param_grid (Dict): Dictionary of hyperparameters to tune.
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        cv (int): Number of cross-validation folds. Default is 5.

    Returns:
        Tuple: Best model and dictionary of best parameters.
    """
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def train_best_model(X: pd.DataFrame, y: pd.Series, model_type: str = 'Random Forest') -> object:
    """
    Train the best model with hyperparameter tuning.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        model_type (str): Type of model to train. Default is 'Random Forest'.

    Returns:
        object: Trained best model.
    """
    if model_type == 'Logistic Regression':
        base_model = LogisticRegression(random_state=42)
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    elif model_type == 'Random Forest':
        base_model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'Gradient Boosting':
        base_model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5, 10]
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logger.info(f"Training best {model_type} model with hyperparameter tuning")
    best_model, best_params = tune_hyperparameters(base_model, param_grid, X, y)
    
    return best_model

def train_and_evaluate(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train models and split data for evaluation.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        Tuple: Dictionary of trained models, training features, test features, training target, test target.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    models = train_models(X_train, y_train)
    
    return models, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_classes=2, n_features=20, n_informative=10, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)

    # Train models
    models, X_train, X_test, y_train, y_test = train_and_evaluate(X, y)
    
    print("Trained models:", list(models.keys()))
    
    # Train best Random Forest model
    best_rf_model = train_best_model(X_train, y_train, model_type='Random Forest')
    print("Best Random Forest model:", best_rf_model)