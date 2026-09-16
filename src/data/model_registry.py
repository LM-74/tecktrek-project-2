import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "fraud_detection.db"


def register_model(
    model_name,
    model_version,
    training_date,
    threshold,
    pr_auc
):
    connection = sqlite3.connect(DATABASE_PATH)

    try:
        connection.execute(
            """
            INSERT INTO model_versions
            (
                model_name,
                model_version,
                training_date,
                threshold,
                pr_auc
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                model_name,
                model_version,
                training_date,
                threshold,
                pr_auc
            )
        )

        connection.commit()

        print(f"{model_name} registered successfully.")

    finally:
        connection.close()


def get_registered_models():

    connection = sqlite3.connect(DATABASE_PATH)

    try:
        return connection.execute(
            """
            SELECT
                model_id,
                model_name,
                model_version,
                training_date,
                threshold,
                pr_auc
            FROM model_versions
            ORDER BY model_id
            """
        ).fetchall()

    finally:
        connection.close()

