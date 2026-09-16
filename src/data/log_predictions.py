import sqlite3
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "fraud_detection.db"


def log_predictions(
    predictions,
    model_id,
    threshold=0.5
):
    connection = sqlite3.connect(DATABASE_PATH)

    try:
        for index, probability in enumerate(predictions):

            transaction_id = 241168 + index

            predicted_class = int(
                probability >= threshold
            )

            prediction_time = datetime.now().isoformat(
                timespec="seconds"
            )

            connection.execute(
                """
                INSERT INTO predictions
                (
                    transaction_id,
                    model_id,
                    fraud_probability,
                    predicted_class,
                    threshold,
                    prediction_time
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    transaction_id,
                    model_id,
                    float(probability),
                    predicted_class,
                    threshold,
                    prediction_time
                )
            )

        connection.commit()

        print(
            f"Logged {len(predictions)} predictions "
            f"for model ID {model_id}."
        )

    finally:
        connection.close()