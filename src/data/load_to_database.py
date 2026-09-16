import sqlite3
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATABASE_PATH = PROJECT_ROOT / "data" / "fraud_detection.db"


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def load_dataset(file_name, dataset_split):
    file_path = DATA_DIR / file_name

    print(f"\nLoading {dataset_split} data...")
    print(f"File: {file_path}")

    df = pd.read_csv(file_path)

    df = df.rename(
        columns={
            "Time": "transaction_time",
            "Amount": "amount",
            "Class": "actual_fraud"
        }
    )

    df["dataset_split"] = dataset_split

    return df


def insert_dataset(df):
    connection = get_connection()

    try:
        df.to_sql(
            "transactions",
            connection,
            if_exists="append",
            index=False,
            chunksize=5000
        )

        connection.commit()

    finally:
        connection.close()


def load_all_datasets():

    datasets = [
        ("train.csv", "train"),
        ("val.csv", "validation"),
        ("test.csv", "test")
    ]

    for file_name, dataset_split in datasets:

        df = load_dataset(
            file_name,
            dataset_split
        )

        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")

        insert_dataset(df)

        print(f"{dataset_split} data inserted successfully.")


if __name__ == "__main__":
    load_all_datasets()

    print("\nAll datasets inserted successfully.")