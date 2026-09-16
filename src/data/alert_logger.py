import sqlite3
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "fraud_detection.db"


def create_alert(
    transaction_id,
    prediction_id,
    risk_score,
    alert_status="new"
):
    connection = sqlite3.connect(DATABASE_PATH)

    try:
        created_at = datetime.now().isoformat(timespec="seconds")

        connection.execute(
            """
            INSERT INTO alerts
            (
                transaction_id,
                prediction_id,
                risk_score,
                alert_status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                transaction_id,
                prediction_id,
                risk_score,
                alert_status,
                created_at
            )
        )

        connection.commit()

    finally:
        connection.close()
        