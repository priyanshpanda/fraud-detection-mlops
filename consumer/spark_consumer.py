"""
Spark Structured Streaming consumer for real-time fraud detection.

Reads transactions from Kafka, applies feature engineering, calls BentoML
scoring service, and writes flagged transactions to PostgreSQL.
Exposes Prometheus metrics for monitoring.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List

import requests
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, from_json, window, count, avg, stddev,
    when, current_timestamp, lit
)
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from prometheus_client import Counter, Histogram, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'transactions')
BENTOML_ENDPOINT = os.getenv('BENTOML_ENDPOINT', 'http://localhost:5000/predict')
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'frauddb')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'admin')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'secret')
PROMETHEUS_PORT = int(os.getenv('PROMETHEUS_PORT', '8000'))
FRAUD_THRESHOLD = float(os.getenv('FRAUD_THRESHOLD', '0.5'))

# Define Prometheus metrics
transactions_processed_total = Counter(
    'transactions_processed_total',
    'Total number of transactions processed'
)
fraud_detected_total = Counter(
    'fraud_detected_total',
    'Total number of fraudulent transactions detected'
)
prediction_latency_seconds = Histogram(
    'prediction_latency_seconds',
    'Latency of fraud prediction in seconds',
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0)
)

# Define transaction schema
transaction_schema = StructType([
    StructField('txn_id', StringType()),
    StructField('user_id', StringType()),
    StructField('amount', DoubleType()),
    StructField('merchant', StringType()),
    StructField('category', StringType()),
    StructField('hour', IntegerType()),
    StructField('country_code', StringType()),
    StructField('timestamp', StringType()),
    StructField('is_fraud', IntegerType())
])


def create_spark_session() -> SparkSession:
    """
    Create and configure Spark Session for streaming.

    Returns:
        Configured SparkSession instance
    """
    return SparkSession.builder \
        .appName("fraud-detection-consumer") \
        .master("local[*]") \
        .config("spark.sql.streaming.schemaInference", "true") \
        .config("spark.streaming.kafka.maxRatePerPartition", "10000") \
        .config("spark.streaming.stopGracefullyOnShutdown", "true") \
        .getOrCreate()


def read_from_kafka(spark: SparkSession) -> DataFrame:
    """
    Read transactions from Kafka topic.

    Args:
        spark: SparkSession instance

    Returns:
        Streaming DataFrame with transaction data
    """
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    # Parse JSON payload
    df = df.select(
        from_json(col("value").cast("string"), transaction_schema).alias("data")
    ).select("data.*")

    return df


def apply_feature_engineering(df: DataFrame) -> DataFrame:
    """
    Apply feature engineering to transaction stream.

    Features:
    - rolling_avg_spend: 5-minute rolling average spend per user
    - txn_velocity: transaction count per user in last 10 minutes
    - amount_zscore: z-score of amount relative to rolling mean

    Args:
        df: Input DataFrame with transactions

    Returns:
        DataFrame with engineered features
    """
    # Define windows for feature computation
    five_min_window = Window.partitionBy("user_id") \
        .orderBy(col("timestamp").cast("timestamp").cast("long")) \
        .rangeBetween(-300, 0)  # 5 minutes in seconds

    ten_min_window = Window.partitionBy("user_id") \
        .orderBy(col("timestamp").cast("timestamp").cast("long")) \
        .rangeBetween(-600, 0)  # 10 minutes in seconds

    # Convert timestamp to Unix timestamp for windowing
    df = df.withColumn("timestamp_ts", col("timestamp").cast("timestamp"))

    # Calculate rolling average spend (5-minute window)
    df = df.withColumn(
        "rolling_avg_spend",
        avg("amount").over(five_min_window)
    )

    # Calculate transaction velocity (10-minute window)
    df = df.withColumn(
        "txn_velocity",
        count("*").over(ten_min_window)
    )

    # Calculate z-score (amount deviation from rolling mean)
    df = df.withColumn(
        "rolling_stddev",
        stddev("amount").over(five_min_window)
    )
    df = df.withColumn(
        "amount_zscore",
        when(
            col("rolling_stddev") > 0,
            (col("amount") - col("rolling_avg_spend")) / col("rolling_stddev")
        ).otherwise(0)
    )

    # Drop temporary columns
    df = df.drop("timestamp_ts", "rolling_stddev")

    return df


def score_with_bentoml(transaction_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Call BentoML endpoint to score a batch of transactions.

    Args:
        transaction_batch: List of transaction dictionaries

    Returns:
        List of transactions with fraud predictions
    """
    start_time = time.time()

    try:
        response = requests.post(
            BENTOML_ENDPOINT,
            json=transaction_batch,
            timeout=30
        )
        response.raise_for_status()
        predictions = response.json()

        latency = time.time() - start_time
        prediction_latency_seconds.observe(latency)

        return predictions

    except Exception as e:
        logger.error(f"Error calling BentoML endpoint: {str(e)}")
        # Return default predictions on error
        return [
            {
                'txn_id': txn['txn_id'],
                'fraud_probability': 0.0,
                'is_fraud_flag': False
            }
            for txn in transaction_batch
        ]


def foreach_batch_function(batch_df: DataFrame, batch_id: int) -> None:
    """
    Process each micro-batch: call BentoML and write to PostgreSQL.

    Args:
        batch_df: Micro-batch DataFrame
        batch_id: Unique batch identifier
    """
    if batch_df.count() == 0:
        logger.info(f"Batch {batch_id}: Empty batch, skipping")
        return

    try:
        # Convert to list of dictionaries for API call
        rows = batch_df.collect()
        batch_data = [row.asDict() for row in rows]

        # Score predictions with BentoML
        predictions = score_with_bentoml(batch_data)

        # Update transactions processed metric
        transactions_processed_total.inc(len(rows))

        # Enrich batch with predictions
        fraud_records = []
        for prediction in predictions:
            txn_id = prediction.get('txn_id')
            fraud_prob = prediction.get('fraud_probability', 0.0)
            is_fraud_flag = prediction.get('is_fraud_flag', False)

            # Find original transaction
            txn = next((t for t in batch_data if t['txn_id'] == txn_id), None)
            if not txn:
                continue

            # Track fraud detections
            if is_fraud_flag:
                fraud_detected_total.inc()
                
                fraud_record = {
                    'txn_id': str(uuid.uuid4()),
                    'user_id': txn.get('user_id'),
                    'amount': float(txn.get('amount', 0)),
                    'merchant': txn.get('merchant'),
                    'category': txn.get('category'),
                    'fraud_probability': float(fraud_prob),
                    'is_fraud_flag': bool(is_fraud_flag),
                    'country_code': txn.get('country_code'),
                    'hour_of_day': txn.get('hour'),
                    'rolling_avg_spend': float(txn.get('rolling_avg_spend', 0)),
                    'txn_velocity': int(txn.get('txn_velocity', 0)),
                    'amount_zscore': float(txn.get('amount_zscore', 0)),
                    'timestamp': txn.get('timestamp'),
                    'processed_at': datetime.now(timezone.utc).isoformat()
                }
                fraud_records.append(fraud_record)

        # Write flagged transactions to PostgreSQL
        if fraud_records:
            write_to_postgres(fraud_records)
            logger.info(f"Batch {batch_id}: Wrote {len(fraud_records)} fraud records to PostgreSQL")

    except Exception as e:
        logger.error(f"Error processing batch {batch_id}: {str(e)}")


def write_to_postgres(records: List[Dict[str, Any]]) -> None:
    """
    Write fraud-flagged records to PostgreSQL.

    Args:
        records: List of fraud records to write
    """
    import psycopg2
    from psycopg2.extras import execute_values

    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        cur = conn.cursor()

        insert_query = """
            INSERT INTO flagged_transactions 
            (txn_id, user_id, amount, merchant, category, fraud_probability, 
             is_fraud_flag, country_code, hour_of_day, rolling_avg_spend, 
             txn_velocity, amount_zscore, timestamp, processed_at)
            VALUES %s
            ON CONFLICT (txn_id) DO NOTHING
        """

        values = [
            (
                record['txn_id'],
                record['user_id'],
                record['amount'],
                record['merchant'],
                record['category'],
                record['fraud_probability'],
                record['is_fraud_flag'],
                record['country_code'],
                record['hour_of_day'],
                record['rolling_avg_spend'],
                record['txn_velocity'],
                record['amount_zscore'],
                record['timestamp'],
                record['processed_at']
            )
            for record in records
        ]

        execute_values(cur, insert_query, values)
        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        logger.error(f"Error writing to PostgreSQL: {str(e)}")


def main() -> None:
    """Main entry point for Spark Structured Streaming consumer."""
    # Start Prometheus metrics server
    start_http_server(PROMETHEUS_PORT)
    logger.info(f"Prometheus metrics server started on port {PROMETHEUS_PORT}")

    # Create Spark session
    spark = create_spark_session()
    logger.info("Spark session created")

    try:
        # Read from Kafka
        df = read_from_kafka(spark)
        logger.info(f"Reading from Kafka topic: {KAFKA_TOPIC}")

        # Apply feature engineering
        df = apply_feature_engineering(df)
        logger.info("Feature engineering applied")

        # Write stream with foreachBatch for custom processing
        query = df.writeStream \
            .foreachBatch(foreach_batch_function) \
            .option("checkpointLocation", "/tmp/fraud_detection_checkpoint") \
            .trigger(processingTime="10s") \
            .start()

        logger.info("Streaming query started")
        query.awaitTermination()

    except Exception as e:
        logger.error(f"Error in main streaming loop: {str(e)}")
    finally:
        spark.stop()
        logger.info("Spark session stopped")


if __name__ == "__main__":
    main()