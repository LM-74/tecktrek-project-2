import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATABASE_DIR / "fraud_detection.db"


def get_connection():
    """
    Create and return a connection to the SQLite database.
    """

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)

    # Enable foreign key constraints
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def create_database():
    """
    Create all database tables using schema.sql.
    """

    schema_path = PROJECT_ROOT / "sql" / "schema.sql"

    with open(schema_path, "r", encoding="utf-8") as file:
        schema = file.read()

    connection = get_connection()

    try:
        connection.executescript(schema)
        connection.commit()

        print("Database created successfully.")
        print(f"Database path: {DATABASE_PATH}")

    finally:
        connection.close()


if __name__ == "__main__":
    create_database()
    