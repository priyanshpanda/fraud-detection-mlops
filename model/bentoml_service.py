"""
BentoML service for real-time fraud detection scoring.

Exposes POST /predict endpoint that accepts transaction data,
performs fraud scoring, and returns predictions with probabilities.
"""

import time
from typing import List, Dict, Any

import bentoml
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """Pydantic model for transaction input validation."""
    txn_id: str = Field(..., description="Transaction ID")
    user_id: str = Field(..., description="User ID")
    amount: float = Field(..., description="Transaction amount", ge=0)
    merchant: str = Field(..., description="Merchant name")
    category: str = Field(..., description="Transaction category")
    hour: int = Field(..., description="Hour of day", ge=0, le=23)
    country_code: str = Field(..., description="Country code")
    timestamp: str = Field(..., description="ISO timestamp")
    is_fraud: int = Field(0, description="Fraud label (0 or 1)", ge=0, le=1)
    rolling_avg_spend: float = Field(0, description="Rolling average spend")
    txn_velocity: int = Field(0, description="Transaction velocity")
    amount_zscore: float = Field(0, description="Amount z-score")


class PredictionOutput(BaseModel):
    """Pydantic model for prediction output."""
    txn_id: str = Field(..., description="Transaction ID")
    fraud_probability: float = Field(..., description="Fraud probability")
    is_fraud_flag: bool = Field(..., description="Fraud flag (threshold=0.5)")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")


# Load model from BentoML registry
model_ref = bentoml.xgboost.get("fraud_detector:latest")
model = model_ref.to_runner()

# Create BentoML service
svc = bentoml.Service("fraud_detection_service", runners=[model])


@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def predict(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Score transactions for fraud probability.

    Args:
        transactions: List of transaction dictionaries

    Returns:
        List of predictions with fraud scores
    """
    start_time = time.time()
    predictions = []
    
    try:
        # Validate inputs
        validated_transactions = []
        for txn in transactions:
            try:
                validated_txn = TransactionInput(**txn)
                validated_transactions.append(validated_txn.dict())
            except Exception as e:
                # Log validation error but continue
                print(f"Validation error for transaction: {str(e)}")
                continue
        
        if not validated_transactions:
            return predictions
        
        # Extract features for prediction (order must match training)
        features = [
            [
                txn['amount'],
                txn['hour'],
                txn['rolling_avg_spend'],
                txn['txn_velocity'],
                txn['amount_zscore']
            ]
            for txn in validated_transactions
        ]
        
        # Get predictions from model
        fraud_probabilities = model.predict_proba.run(features)
        
        # Format predictions
        for txn, fraud_prob_array in zip(validated_transactions, fraud_probabilities):
            fraud_prob = float(fraud_prob_array[1])  # Probability of fraud class
            is_fraud_flag = fraud_prob >= 0.5
            
            prediction = {
                'txn_id': txn['txn_id'],
                'fraud_probability': fraud_prob,
                'is_fraud_flag': is_fraud_flag,
                'latency_ms': (time.time() - start_time) * 1000
            }
            
            # Validate output
            try:
                validated_output = PredictionOutput(**prediction)
                predictions.append(validated_output.dict())
            except Exception as e:
                print(f"Output validation error: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
    
    return predictions


if __name__ == "__main__":
    # Run BentoML service
    bentoml.serve(svc, port=5000)