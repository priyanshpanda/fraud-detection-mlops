"""
Alert service for fraud detection monitoring.

Polls PostgreSQL database for flagged transactions and sends alerts
with structured logging and optional Slack notifications.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Set, Dict, Any

import psycopg2

# Configure logging with JSON formatter
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'frauddb')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'admin')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'secret')
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', '')
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '5'))
FRAUD_THRESHOLD = float(os.getenv('FRAUD_THRESHOLD', '0.5'))


def connect_postgres() -> psycopg2.extensions.connection:
    """
    Establish connection to PostgreSQL database.

    Returns:
        PostgreSQL connection object

    Raises:
        psycopg2.Error: If connection fails
    """
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        logger.info("Connected to PostgreSQL")
        return conn
    except psycopg2.Error as e:
        logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
        raise


def fetch_new_fraud_records(conn: psycopg2.extensions.connection, processed_ids: Set[str]) -> list:
    """
    Fetch new flagged transactions from PostgreSQL.

    Args:
        conn: PostgreSQL connection
        processed_ids: Set of already-processed transaction IDs

    Returns:
        List of new fraud records
    """
    try:
        cur = conn.cursor()
        query = """
            SELECT txn_id, user_id, amount, merchant, category, 
                   fraud_probability, timestamp, created_at
            FROM flagged_transactions
            WHERE is_fraud_flag = TRUE
            ORDER BY created_at DESC
            LIMIT 100
        """
        cur.execute(query)
        records = cur.fetchall()
        cur.close()
        
        # Filter for new records
        new_records = []
        for record in records:
            txn_id = str(record[0])
            if txn_id not in processed_ids:
                new_records.append({
                    'txn_id': txn_id,
                    'user_id': record[1],
                    'amount': record[2],
                    'merchant': record[3],
                    'category': record[4],
                    'fraud_probability': record[5],
                    'timestamp': record[6],
                    'created_at': record[7]
                })
                processed_ids.add(txn_id)
        
        return new_records
    
    except psycopg2.Error as e:
        logger.error(f"Error fetching records from PostgreSQL: {str(e)}")
        return []


def log_structured_alert(record: Dict[str, Any]) -> None:
    """
    Log fraud alert with structured JSON and plain text formats.

    Args:
        record: Fraud transaction record
    """
    # Structured JSON log
    alert_dict = {
        'alert_type': 'fraud_detected',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'txn_id': record['txn_id'],
        'user_id': record['user_id'],
        'amount': float(record['amount']),
        'merchant': record['merchant'],
        'category': record['category'],
        'fraud_probability': float(record['fraud_probability']),
        'transaction_timestamp': record['timestamp'].isoformat() if hasattr(record['timestamp'], 'isoformat') else str(record['timestamp'])
    }
    
    logger.info(f"FRAUD_ALERT: {json.dumps(alert_dict)}")
    
    # Plain text log
    logger.warning(
        f"\n{'='*60}\n"
        f"🚨 FRAUD ALERT 🚨\n"
        f"{'='*60}\n"
        f"Transaction ID: {record['txn_id']}\n"
        f"User ID: {record['user_id']}\n"
        f"Amount: ${record['amount']:.2f}\n"
        f"Merchant: {record['merchant']}\n"
        f"Category: {record['category']}\n"
        f"Fraud Probability: {record['fraud_probability']:.4f}\n"
        f"Timestamp: {record['timestamp']}\n"
        f"{'='*60}\n"
    )


def send_slack_alert(record: Dict[str, Any]) -> None:
    """
    Send fraud alert to Slack channel (stub implementation).

    Args:
        record: Fraud transaction record

    To integrate with real Slack webhook:
    1. Create a Slack webhook: https://api.slack.com/messaging/webhooks
    2. Set SLACK_WEBHOOK_URL environment variable
    3. Uncomment the code below:

    import requests
    
    if not SLACK_WEBHOOK_URL:
        return
    
    message = {
        "text": "🚨 Fraud Detected",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Fraud Alert*\\n"
                            f"TXN ID: {record['txn_id']}\\n"
                            f"User: {record['user_id']}\\n"
                            f"Amount: ${record['amount']:.2f}\\n"
                            f"Fraud Prob: {record['fraud_probability']:.4f}"
                }
            }
        ]
    }
    
    try:
        requests.post(SLACK_WEBHOOK_URL, json=message, timeout=10)
        logger.info(f"Slack alert sent for transaction {record['txn_id']}")
    except Exception as e:
        logger.error(f"Failed to send Slack alert: {str(e)}")
    """
    logger.info(f"Slack alert stub called for transaction {record['txn_id']}")


def main() -> None:
    """Main entry point for alert service."""
    logger.info("Starting fraud alert service...")
    
    processed_ids: Set[str] = set()
    
    while True:
        try:
            # Connect to PostgreSQL
            conn = connect_postgres()
            
            # Fetch new fraud records
            new_records = fetch_new_fraud_records(conn, processed_ids)
            
            # Process each new record
            for record in new_records:
                try:
                    # Log structured alert
                    log_structured_alert(record)
                    
                    # Send to Slack (stub)
                    send_slack_alert(record)
                
                except Exception as e:
                    logger.error(f"Error processing alert for {record['txn_id']}: {str(e)}")
            
            if new_records:
                logger.info(f"Processed {len(new_records)} new fraud alerts")
            
            conn.close()
            
            # Wait before next poll
            time.sleep(POLL_INTERVAL)
        
        except psycopg2.Error:
            logger.error("Database connection error, retrying in 10 seconds...")
            time.sleep(10)
        except KeyboardInterrupt:
            logger.info("Alert service interrupted by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in alert service: {str(e)}", exc_info=True)
            time.sleep(10)


if __name__ == "__main__":
    main()