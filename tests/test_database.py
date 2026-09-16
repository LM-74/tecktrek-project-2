import sqlite3

from src.data.database import DATABASE_PATH


def test_database_exists():
    assert DATABASE_PATH.exists()


def test_required_tables():
    connection = sqlite3.connect(DATABASE_PATH)

    tables = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        """
    ).fetchall()

    connection.close()

    table_names = {
        table[0]
        for table in tables
    }

    required_tables = {
        "transactions",
        "model_versions",
        "predictions",
        "alerts"
    }

    assert required_tables.issubset(table_names)


def test_transactions_count():
    connection = sqlite3.connect(DATABASE_PATH)

    count = connection.execute(
        "SELECT COUNT(*) FROM transactions"
    ).fetchone()[0]

    connection.close()

    assert count == 283726


def test_dataset_splits():
    connection = sqlite3.connect(DATABASE_PATH)

    rows = connection.execute(
        """
        SELECT dataset_split, COUNT(*)
        FROM transactions
        GROUP BY dataset_split
        """
    ).fetchall()

    connection.close()

    split_counts = dict(rows)

    assert split_counts["train"] == 198608
    assert split_counts["validation"] == 42559
    assert split_counts["test"] == 42559


def test_fraud_values():
    connection = sqlite3.connect(DATABASE_PATH)

    values = connection.execute(
        """
        SELECT DISTINCT actual_fraud
        FROM transactions
        """
    ).fetchall()

    connection.close()

    fraud_values = {
        row[0]
        for row in values
    }

    assert fraud_values.issubset({0, 1})


def test_models_registered():
    connection = sqlite3.connect(DATABASE_PATH)

    count = connection.execute(
        "SELECT COUNT(*) FROM model_versions"
    ).fetchone()[0]

    connection.close()

    assert count == 3


def test_predictions_logged():
    connection = sqlite3.connect(DATABASE_PATH)

    count = connection.execute(
        "SELECT COUNT(*) FROM predictions"
    ).fetchone()[0]

    connection.close()

    assert count == 127677


def test_prediction_models():
    connection = sqlite3.connect(DATABASE_PATH)

    rows = connection.execute(
        """
        SELECT model_id, COUNT(*)
        FROM predictions
        GROUP BY model_id
        """
    ).fetchall()

    connection.close()

    counts = dict(rows)

    assert counts[1] == 42559
    assert counts[2] == 42559
    assert counts[3] == 42559


def test_alerts_have_valid_risk_scores():
    connection = sqlite3.connect(DATABASE_PATH)

    invalid_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM alerts
        WHERE risk_score < 0
           OR risk_score > 1
        """
    ).fetchone()[0]

    connection.close()

    assert invalid_count == 0