import pandas as pd
from pathlib import Path


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"


# =========================
# Load Data
# =========================

def load_processed_data():
    """
    Load train, validation, and test datasets.
    """

    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "val.csv"
    test_path = DATA_DIR / "test.csv"

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    return train, val, test


# =========================
# Validate Data
# =========================

def validate_data(train, val, test):
    """
    Validate the structure and basic quality
    of train, validation, and test datasets.
    """

    required_columns = [
        "Time",
        "Amount",
        "Class"
    ]

    datasets = {
        "train": train,
        "validation": val,
        "test": test
    }

    for name, df in datasets.items():

        # Check that dataset is not empty
        assert not df.empty, f"{name} dataset is empty."

        # Check required columns
        for column in required_columns:
            assert column in df.columns, (
                f"{column} is missing from {name} dataset."
            )

        # Check target values
        assert set(df["Class"].unique()).issubset({0, 1}), (
            f"Unexpected target values in {name} dataset."
        )

        # Check missing values
        assert df.isnull().sum().sum() == 0, (
            f"Missing values found in {name} dataset."
        )

    return True


# =========================
# Preprocess Data
# =========================

def preprocess_data(train, val, test):
    """
    Prepare datasets for machine learning.

    The target column is separated from the features.
    """

    target = "Class"

    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_val = val.drop(columns=[target])
    y_val = val[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]

    return X_train, X_val, X_test, y_train, y_val, y_test


# =========================
# Save Data
# =========================

def save_processed_data(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):
    """
    Save processed features and target values.
    """

    output_dir = DATA_DIR / "pipeline_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_val.to_csv(output_dir / "X_val.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)

    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_val.to_csv(output_dir / "y_val.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print("Processed datasets saved successfully.")
    print(f"Output directory: {output_dir}")


# =========================
# Main Pipeline
# =========================

def run_pipeline():

    print("Loading data...")

    train, val, test = load_processed_data()

    print("Data loaded successfully.")
    print(f"Train shape: {train.shape}")
    print(f"Validation shape: {val.shape}")
    print(f"Test shape: {test.shape}")

    print("\nValidating data...")

    validate_data(train, val, test)

    print("Data validation passed.")

    print("\nPreprocessing data...")

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        train,
        val,
        test
    )

    print("Preprocessing completed.")

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")

    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"y_test shape: {y_test.shape}")

    print("\nSaving processed data...")

    save_processed_data(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()