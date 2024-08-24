import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import learning_curve
from typing import List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_feature_importance(importance: np.ndarray, names: List[str], model_type: str) -> plt.Figure:
    """
    Create a bar plot of feature importances.

    Args:
        importance (np.ndarray): Array of feature importance values.
        names (List[str]): List of feature names.
        model_type (str): Type of the model (e.g., 'Random Forest', 'Logistic Regression').

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'], ax=ax)

    ax.set_title(f'{model_type} Feature Importance')
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Feature Name')

    return fig

def plot_learning_curve(estimator, X: np.ndarray, y: np.ndarray, cv: int = 5) -> plt.Figure:
    """
    Plot the learning curve of a model.

    Args:
        estimator: The model/estimator.
        X (np.ndarray): Features.
        y (np.ndarray): Target variable.
        cv (int): Number of cross-validation folds.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10), scoring="accuracy"
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, label='Training score')
    ax.plot(train_sizes, test_mean, label='Cross-validation score')

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)

    ax.set_xlabel('Number of training examples')
    ax.set_ylabel('Accuracy score')
    ax.set_title('Learning Curve')
    ax.legend(loc='best')

    return fig

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
    n_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, n_classes))

    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:0.2f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")

    return fig

def plot_correlation_matrix(df: pd.DataFrame) -> plt.Figure:
    """
    Plot correlation matrix of features.

    Args:
        df (pd.DataFrame): DataFrame containing features.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Feature Correlation Matrix')
    return fig

def plot_feature_distributions(df: pd.DataFrame, n_cols: int = 3) -> plt.Figure:
    """
    Plot distributions of all features in the dataset.

    Args:
        df (pd.DataFrame): DataFrame containing features.
        n_cols (int): Number of columns in the subplot grid.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    n_features = df.shape[1]
    n_rows = (n_features - 1) // n_cols + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(col)

    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Generate sample data
    X, y = make_classification(n_samples=1000, n_classes=3, n_features=20, n_informative=10, random_state=42)
    feature_names = [f'feature_{i}' for i in range(20)]
    X = pd.DataFrame(X, columns=feature_names)
    classes = ['Class 0', 'Class 1', 'Class 2']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Generate plots
    feature_importance_plot = plot_feature_importance(model.feature_importances_, feature_names, 'Random Forest')
    learning_curve_plot = plot_learning_curve(model, X, y)
    confusion_matrix_plot = plot_confusion_matrix(y_test, model.predict(X_test), classes)
    roc_curve_plot = plot_roc_curve(y_test, model.predict_proba(X_test), classes)
    correlation_matrix_plot = plot_correlation_matrix(X)
    feature_distributions_plot = plot_feature_distributions(X)

    # Display plots
    feature_importance_plot.show()
    learning_curve_plot.show()
    confusion_matrix_plot.show()
    roc_curve_plot.show()
    correlation_matrix_plot.show()
    feature_distributions_plot.show()