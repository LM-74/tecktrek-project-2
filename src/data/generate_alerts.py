import sqlite3
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "fraud_detection.db"


def generate_alerts(model_id):

    connection = sqlite3.connect(DATABASE_PATH)

    try:

        predictions = connection.execute(
            """
            SELECT
                prediction_id,
                transaction_id,
                fraud_probability
            FROM predictions
            WHERE model_id = ?
              AND predicted_class = 1
            ORDER BY prediction_id
            """,
            (model_id,)
        ).fetchall()

        print(f"Fraud predictions found: {len(predictions)}")

        for prediction_id, transaction_id, risk_score in predictions:

            created_at = datetime.now().isoformat(
                timespec="seconds"
            )

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
                    "new",
                    created_at
                )
            )

        connection.commit()

        print(
            f"{len(predictions)} alerts created successfully."
        )

    finally:
        connection.close()


if __name__ == "__main__":

    # Use the selected deployed model
    generate_alerts(model_id=1)