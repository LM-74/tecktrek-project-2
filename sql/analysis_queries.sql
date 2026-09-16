-- 1. Total transactions
SELECT COUNT(*) AS total_transactions
FROM transactions;


-- 2. Transactions by dataset
SELECT
    dataset_split,
    COUNT(*) AS transaction_count
FROM transactions
GROUP BY dataset_split;


-- 3. Fraud transactions by dataset
SELECT
    dataset_split,
    SUM(actual_fraud) AS fraud_count
FROM transactions
GROUP BY dataset_split;


-- 4. Fraud rate by dataset
SELECT
    dataset_split,
    COUNT(*) AS total_transactions,
    SUM(actual_fraud) AS fraud_transactions,
    ROUND(
        100.0 * SUM(actual_fraud) / COUNT(*),
        4
    ) AS fraud_rate_percentage
FROM transactions
GROUP BY dataset_split;


-- 5. Average transaction amount
SELECT
    dataset_split,
    AVG(amount) AS average_amount
FROM transactions
GROUP BY dataset_split;


-- 6. Registered models
SELECT
    model_id,
    model_name,
    model_version,
    threshold,
    pr_auc
FROM model_versions
ORDER BY model_id;


-- 7. Prediction summary
SELECT
    model_id,
    predicted_class,
    COUNT(*) AS prediction_count
FROM predictions
GROUP BY model_id, predicted_class
ORDER BY model_id, predicted_class;


-- 8. Alert summary
SELECT
    alert_status,
    COUNT(*) AS alert_count
FROM alerts
GROUP BY alert_status;


-- 9. Average risk score
SELECT
    AVG(risk_score) AS average_risk_score
FROM alerts;