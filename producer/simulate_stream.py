"""
Producer service for simulating real-time bank transactions to Kafka.

Generates synthetic transactions with realistic fraud patterns and publishes
to the 'transactions' Kafka topic at 10 transactions per second.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

import numpy as np
from faker import Faker
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'transactions')
TXN_PER_SECOND = int(os.getenv('TXN_PER_SECOND', '10'))
FRAUD_PROBABILITY = float(os.getenv('FRAUD_PROBABILITY', '0.02'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '100'))

# Initialize Faker for synthetic data
faker = Faker()


def generate_transaction() -> Dict[str, Any]:
    """
    Generate a single synthetic transaction with realistic features.

    Returns:
        Dictionary containing transaction data with fields:
        - txn_id: UUID identifier
        - user_id: User identifier
        - amount: Transaction amount (exponential distribution)
        - merchant: Merchant name
        - category: Transaction category
        - hour: Hour of day (0-23)
        - country_code: Country code (ISO 3166-1 alpha-3)
        - timestamp: ISO formatted timestamp
        - is_fraud: Binary fraud label (0 or 1)
    """
    # Generate base transaction data
    txn_id = str(uuid.uuid4())
    user_id = f"USER_{faker.random_int(min=1000, max=999999)}"
    
    # Amount follows exponential distribution with lambda=1/200
    amount = float(np.random.exponential(scale=200))
    
    merchant = faker.company()
    category = faker.random_element(['grocery', 'travel', 'online', 'ATM'])
    hour = faker.random_int(min=0, max=23)
    country_code = faker.country_code(representation='alpha-3')
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Determine fraud label with realistic patterns
    is_fraud = 0
    if np.random.random() < FRAUD_PROBABILITY:
        is_fraud = 1
        # Inject fraud patterns when is_fraud=1
        if np.random.random() < 0.5:
            amount = np.random.uniform(2000, 5000)  # High amount
        if np.random.random() < 0.3:
            hour = faker.random_element(list(range(0, 5)) + list(range(23, 24)))  # Odd hours
        if np.random.random() < 0.8:
            country_code = faker.random_element(['CHN', 'RUS', 'NG', 'IN', 'BR'])  # Foreign country
    
    transaction = {
        'txn_id': txn_id,
        'user_id': user_id,
        'amount': round(amount, 2),
        'merchant': merchant,
        'category': category,
        'hour': hour,
        'country_code': country_code,
        'timestamp': timestamp,
        'is_fraud': is_fraud
    }
    
    return transaction


def connect_kafka(max_retries: int = 5) -> KafkaProducer:
    """
    Establish connection to Kafka with exponential backoff retry logic.

    Args:
        max_retries: Maximum number of connection attempts

    Returns:
        Connected KafkaProducer instance

    Raises:
        Exception: If unable to connect after max_retries attempts
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempting to connect to Kafka (attempt {attempt}/{max_retries})")
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3,
                request_timeout_ms=10000
            )
            logger.info("Successfully connected to Kafka")
            return producer
        except Exception as e:
            backoff_time = 2 ** (attempt - 1)
            logger.warning(
                f"Connection attempt {attempt} failed: {str(e)}. "
                f"Retrying in {backoff_time}s..."
            )
            if attempt < max_retries:
                time.sleep(backoff_time)
            else:
                logger.error("Failed to connect to Kafka after maximum retries")
                raise


def produce_transactions(num_transactions: int = 1000) -> None:
    """
    Produce synthetic transactions to Kafka topic at controlled rate.

    Args:
        num_transactions: Total number of transactions to generate
    """
    producer = connect_kafka()
    
    try:
        interval = 1.0 / TXN_PER_SECOND
        transactions_sent = 0
        
        logger.info(
            f"Starting transaction producer: {TXN_PER_SECOND} txn/sec, "
            f"Target: {num_transactions} total transactions"
        )
        
        start_time = time.time()
        
        for i in range(num_transactions):
            try:
                transaction = generate_transaction()
                
                # Send to Kafka
                producer.send(
                    KAFKA_TOPIC,
                    value=transaction,
                    key=transaction['user_id'].encode('utf-8')
                ).get(timeout=5)
                
                transactions_sent += 1
                
                # Log every 100 transactions
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = transactions_sent / elapsed
                    logger.info(
                        f"Produced {transactions_sent}/{num_transactions} transactions "
                        f"({rate:.2f} txn/sec)"
                    )
                
                # Rate limiting
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error producing transaction {i}: {str(e)}")
                continue
        
        producer.flush(timeout=10)
        elapsed = time.time() - start_time
        logger.info(
            f"Successfully produced {transactions_sent} transactions in {elapsed:.2f}s "
            f"({transactions_sent/elapsed:.2f} txn/sec average)"
        )
        
    except KeyboardInterrupt:
        logger.info("Producer interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in producer: {str(e)}")
    finally:
        producer.close(timeout=10)
        logger.info("Producer closed")


if __name__ == "__main__":
    produce_transactions(num_transactions=10000)