"""
Train XGBoost fraud detection model with SMOTE for imbalanced data.

Loads creditcard.csv from Kaggle, applies SMOTE to training set only,
trains XGBoost with optimized hyperparameters, and saves to BentoML registry.
"""

import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
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
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_and_prepare_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load creditcard.csv and prepare for modeling.

    Args:
        data_path: Path to creditcard.csv file

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Class distribution:\n{df['Class'].value_counts()}")
    logger.info(f"Fraud rate: {df['Class'].mean():.4f}")
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    return X, y


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> xgb.XGBClassifier:
    """
    Train XGBoost model with optimal hyperparameters for fraud detection.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target

    Returns:
        Trained XGBClassifier model
    """
    logger.info("Training XGBoost model...")
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=100,  # Handle class imbalance
        eval_metric='aucpr',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    logger.info("Model training completed")
    return model


def evaluate_with_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model: xgb.XGBClassifier
) -> None:
    """
    Perform 5-fold cross-validation using AUC-PR metric.

    Args:
        X: Features
        y: Target
        model: Trained XGBoost model
    """
    logger.info("Performing 5-fold cross-validation...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Cross-validation with AUC-PR scoring
    scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring='average_precision',
        n_jobs=-1
    )
    
    logger.info(f"Cross-validation AUC-PR scores: {scores}")
    logger.info(f"Mean AUC-PR: {scores.mean():.4f} (+/- {scores.std():.4f})")


def save_model_to_bentoml(model: xgb.XGBClassifier, model_name: str) -> str:
    """
    Save trained model to BentoML registry.

    Args:
        model: Trained XGBoost model
        model_name: Name for the model in BentoML registry

    Returns:
        Model tag string
    """
    logger.info(f"Saving model to BentoML registry as '{model_name}'...")
    
    tag = bentoml.xgboost.save_model(model_name, model)
    
    logger.info(f"Model saved successfully: {tag}")
    return str(tag)


def main() -> None:
    """Main entry point for model training."""
    try:
        # Load and prepare data
        X, y = load_and_prepare_data(DATA_PATH)
        
        # Train/test split (stratified)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        logger.info(f"Training fraud rate: {y_train.mean():.4f}")
        logger.info(f"Test fraud rate: {y_test.mean():.4f}")
        
        # Apply SMOTE only to training set (never to test set)
        logger.info("Applying SMOTE to training set only...")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        logger.info(f"After SMOTE training set size: {X_train_smote.shape[0]}")
        logger.info(f"After SMOTE training fraud rate: {y_train_smote.mean():.4f}")
        
        # Train model
        model = train_xgboost_model(X_train_smote, y_train_smote, X_test, y_test)
        
        # Cross-validation
        evaluate_with_cross_validation(X_train_smote, y_train_smote, model)
        
        # Save model
        model_tag = save_model_to_bentoml(model, MODEL_NAME)
        
        logger.info(f"\n✓ Model training pipeline completed successfully")
        logger.info(f"Model tag: {model_tag}")
        
    except FileNotFoundError:
        logger.error(f"Data file not found: {DATA_PATH}")
        logger.error("Please download creditcard.csv from Kaggle and place it in data/ directory")
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()