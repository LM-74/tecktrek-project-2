import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "fraud_detection.db"

connection = sqlite3.connect(DATABASE_PATH)

print("========== DATABASE CHECK ==========\n")

# 1. Tables
print("Tables:")
tables = connection.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()

for table in tables:
    print("-", table[0])


# 2. Transactions
print("\nTransaction count:")

count = connection.execute(
    "SELECT COUNT(*) FROM transactions"
).fetchone()[0]

print(count)


# 3. Dataset splits
print("\nTransactions by dataset:")

rows = connection.execute(
    """
    SELECT dataset_split, COUNT(*)
    FROM transactions
    GROUP BY dataset_split
    ORDER BY dataset_split
    """
).fetchall()

for row in rows:
    print(row)


# 4. Fraud count
print("\nFraud count by dataset:")

rows = connection.execute(
    """
    SELECT
        dataset_split,
        actual_fraud,
        COUNT(*)
    FROM transactions
    GROUP BY dataset_split, actual_fraud
    ORDER BY dataset_split, actual_fraud
    """
).fetchall()

for row in rows:
    print(row)


# 5. Registered models
print("\nRegistered Models:")

models = connection.execute(
    """
    SELECT
        model_id,
        model_name,
        model_version,
        threshold,
        pr_auc
    FROM model_versions
    ORDER BY model_id
    """
).fetchall()

for model in models:
    print(model)


# 6. Predictions
print("\nPredictions by model:")

predictions = connection.execute(
    """
    SELECT
        model_id,
        predicted_class,
        COUNT(*)
    FROM predictions
    GROUP BY model_id, predicted_class
    ORDER BY model_id, predicted_class
    """
).fetchall()

for row in predictions:
    print(row)


# 7. Alerts
print("\nAlerts:")

alert_count = connection.execute(
    "SELECT COUNT(*) FROM alerts"
).fetchone()[0]

print("Total alerts:", alert_count)


alert_status = connection.execute(
    """
    SELECT alert_status, COUNT(*)
    FROM alerts
    GROUP BY alert_status
    """
).fetchall()

for row in alert_status:
    print(row)


# 8. Average risk score
print("\nAverage alert risk score:")

average_risk = connection.execute(
    "SELECT AVG(risk_score) FROM alerts"
).fetchone()[0]

print(average_risk)


connection.close()

print("\n CHECK COMPLETED ")