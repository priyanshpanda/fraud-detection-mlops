"""PostgreSQL initialization script for fraud detection system."""

CREATE TABLE IF NOT EXISTS flagged_transactions (
    id SERIAL PRIMARY KEY,
    txn_id UUID NOT NULL UNIQUE,
    user_id VARCHAR(50) NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    merchant VARCHAR(100),
    category VARCHAR(50),
    fraud_probability FLOAT NOT NULL,
    is_fraud_flag BOOLEAN NOT NULL,
    country_code VARCHAR(3),
    hour_of_day INT,
    rolling_avg_spend FLOAT,
    txn_velocity INT,
    amount_zscore FLOAT,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP
);

CREATE INDEX idx_flagged_transactions_user_id ON flagged_transactions(user_id);
CREATE INDEX idx_flagged_transactions_timestamp ON flagged_transactions(timestamp);
CREATE INDEX idx_flagged_transactions_is_fraud ON flagged_transactions(is_fraud_flag);