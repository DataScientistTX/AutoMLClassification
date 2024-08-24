import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_models(models: Dict, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> pd.DataFrame:
    """
    Evaluate multiple models using cross-validation.

    Args:
        models (Dict): Dictionary of models to evaluate.
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        cv (int): Number of cross-validation folds. Default is 5.

    Returns:
        pd.DataFrame: DataFrame with evaluation metrics for each model.
    """
    results = []
    for name, model in models.items():
        logger.info(f"Evaluating {name}")
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        results.append({
            'Model': name,
            'Mean CV Accuracy': cv_scores.mean(),
            'Std CV Accuracy': cv_scores.std()
        })
    return pd.DataFrame(results)

def detailed_model_evaluation(model, X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Perform a detailed evaluation of a single model.

    Args:
        model: Trained model object.
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.

    Returns:
        Dict: Dictionary containing various evaluation metrics.
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='weighted'),
        'recall': recall_score(y, y_pred, average='weighted'),
        'f1_score': f1_score(y, y_pred, average='weighted')
    }

    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y, y_prob)

    return metrics

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        classes (List[str]): List of class names.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    return fig

def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, classes: List[str]) -> plt.Figure:
    """
    Plot ROC curve for multi-class classification.

    Args:
        y_true (np.ndarray): True labels.
        y_scores (np.ndarray): Predicted probabilities for each class.
        classes (List[str]): List of class names.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle

    n_classes = len(classes)
    y_bin = label_binarize(y_true, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'yellow', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    return fig

def feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Get feature importance from the model.

    Args:
        model: Trained model object.
        feature_names (List[str]): List of feature names.

    Returns:
        pd.DataFrame: DataFrame with feature importances.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        logger.warning("Model does not have feature_importances_ or coef_ attribute")
        return pd.DataFrame()

    feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    return feature_imp.sort_values('importance', ascending=False)

def generate_evaluation_report(model, X: pd.DataFrame, y: pd.Series, classes: List[str]) -> Tuple[Dict, plt.Figure, plt.Figure, pd.DataFrame]:
    """
    Generate a comprehensive evaluation report for a model.

    Args:
        model: Trained model object.
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        classes (List[str]): List of class names.

    Returns:
        Tuple: Dictionary of metrics, confusion matrix plot, ROC curve plot, feature importance DataFrame.
    """
    metrics = detailed_model_evaluation(model, X, y)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    cm_plot = plot_confusion_matrix(y, y_pred, classes)
    roc_plot = plot_roc_curve(y, y_prob, classes)
    feature_imp = feature_importance(model, X.columns)

    return metrics, cm_plot, roc_plot, feature_imp

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # Generate sample data
    X, y = make_classification(n_samples=1000, n_classes=3, n_features=20, n_informative=10, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    metrics, cm_plot, roc_plot, feature_imp = generate_evaluation_report(model, X_test, y_test, classes=['Class 0', 'Class 1', 'Class 2'])

    print("Metrics:", metrics)
    print("\nTop 5 Important Features:")
    print(feature_imp.head())

    # Display plots
    cm_plot.show()
    roc_plot.show()