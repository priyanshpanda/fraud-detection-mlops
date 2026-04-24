"""
Evaluate fraud detection model and generate evaluation metrics and plots.

Loads the trained BentoML model, computes confusion matrix, classification report,
AUC-ROC, AUC-PR, and generates visualization plots.

Why AUC-PR is the correct metric for imbalanced fraud data:
- With imbalanced datasets (2% fraud rate), AUC-ROC can be misleading because
  it treats all thresholds equally, including high thresholds that barely change
  the True Positive Rate on the skewed minority class.
- AUC-PR (Precision-Recall) focuses specifically on the minority class by plotting
  precision vs recall, making it much more sensitive to model performance on
  fraudulent transactions, which is our primary concern.
- Precision-Recall curves better represent real-world performance where we care
  about how many of our fraud predictions are actually correct (Precision) and
  how many fraud cases we catch (Recall).
"""

import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, auc,
    roc_curve, precision_recall_curve, average_precision_score
)
import bentoml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = os.getenv('DATA_PATH', 'data/creditcard.csv')
MODEL_NAME = os.getenv('MODEL_NAME', 'fraud_detector')
PLOTS_DIR = 'plots'
RANDOM_STATE = 42


def create_plots_directory() -> None:
    """Create plots directory if it doesn't exist."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    logger.info(f"Plots directory: {PLOTS_DIR}")


def load_data_and_model() -> Tuple[pd.DataFrame, pd.Series, object]:
    """
    Load test data and trained BentoML model.

    Returns:
        Tuple of (test features, test target, loaded model)
    """
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Train/test split (same as training)
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )
    
    logger.info(f"Loading model '{MODEL_NAME}' from BentoML...")
    model = bentoml.xgboost.load_model(MODEL_NAME)
    
    return X_test, y_test, model


def print_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> None:
    """
    Print confusion matrix and classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    cm = confusion_matrix(y_true, y_pred)
    logger.info("\n" + "="*50)
    logger.info("CONFUSION MATRIX")
    logger.info("="*50)
    logger.info(f"\n{cm}")
    logger.info(f"\nTrue Negatives: {cm[0, 0]}")
    logger.info(f"False Positives: {cm[0, 1]}")
    logger.info(f"False Negatives: {cm[1, 0]}")
    logger.info(f"True Positives: {cm[1, 1]}")
    
    logger.info("\n" + "="*50)
    logger.info("CLASSIFICATION REPORT")
    logger.info("="*50)
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraud'])}")


def print_auc_scores(
    y_true: pd.Series,
    y_pred_proba: np.ndarray
) -> Tuple[float, float]:
    """
    Print AUC-ROC and AUC-PR scores.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities

    Returns:
        Tuple of (AUC-ROC, AUC-PR) scores
    """
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    auc_pr = average_precision_score(y_true, y_pred_proba)
    
    logger.info("\n" + "="*50)
    logger.info("AUC SCORES")
    logger.info("="*50)
    logger.info(f"AUC-ROC: {auc_roc:.4f}")
    logger.info(f"AUC-PR: {auc_pr:.4f}")
    logger.info("\nNote: AUC-PR is the primary metric for imbalanced fraud detection")
    
    return auc_roc, auc_pr


def plot_roc_curve(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    auc_roc: float
) -> None:
    """
    Plot and save ROC curve.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        auc_roc: AUC-ROC score
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plot_path = os.path.join(PLOTS_DIR, 'roc_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"ROC curve saved: {plot_path}")
    plt.close()


def plot_precision_recall_curve(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    auc_pr: float
) -> None:
    """
    Plot and save Precision-Recall curve.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        auc_pr: AUC-PR score
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='darkblue', lw=2, label=f'PR curve (AUC = {auc_pr:.4f})')
    plt.axhline(y=y_true.mean(), color='red', linestyle='--', lw=2, label=f'Baseline (fraud rate = {y_true.mean():.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Primary Metric for Imbalanced Data)')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    
    plot_path = os.path.join(PLOTS_DIR, 'pr_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Precision-Recall curve saved: {plot_path}")
    plt.close()


def plot_feature_importance(model: object) -> None:
    """
    Plot and save feature importance chart.

    Args:
        model: Trained XGBoost model
    """
    # Get feature importances
    importances = model.feature_importances_
    feature_names = [f'V{i}' if i > 0 else 'Amount' for i in range(len(importances))]
    
    # Sort by importance
    indices = np.argsort(importances)[-20:]  # Top 20 features
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_DIR, 'feature_importance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Feature importance plot saved: {plot_path}")
    plt.close()


def main() -> None:
    """Main entry point for model evaluation."""
    try:
        create_plots_directory()
        
        # Load data and model
        X_test, y_test, model = load_data_and_model()
        
        # Generate predictions
        logger.info("Generating predictions...")
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Print evaluation metrics
        print_confusion_matrix(y_test, y_pred)
        auc_roc, auc_pr = print_auc_scores(y_test, y_pred_proba)
        
        # Generate plots
        plot_roc_curve(y_test, y_pred_proba, auc_roc)
        plot_precision_recall_curve(y_test, y_pred_proba, auc_pr)
        plot_feature_importance(model)
        
        logger.info("\n✓ Model evaluation completed successfully")
        
    except FileNotFoundError:
        logger.error(f"Data file not found: {DATA_PATH}")
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()