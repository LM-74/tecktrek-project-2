"""
src/features/feature_engineering.py

The single source of truth for this project's features.

The logic is a direct copy of notebooks/02_feature_engineering.ipynb. It
lives here so the notebooks, the training scripts and the Streamlit app
all build features identically — if a feature changes, it changes here
and everything downstream follows.
"""

import numpy as np
import pandas as pd

from src.data.validation import RAW_FEATURES, TARGET, V_COLUMNS, validate

# The 36 columns every saved model and preprocessor expects, in order.
ENGINEERED_FEATURES = ["Hour", "Day", "Hour_sin", "Hour_cos", "Time_of_day", "amount_log"]
MODEL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

# Excluded from the autoencoder: the chronological split makes the time
# columns shift between train and test, which the autoencoder would
# otherwise read as "anomalous" for every late transaction.
EXCLUDE_FROM_AUTOENCODER = ["Time", "Hour", "Day", "Time_of_day", "Amount"]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the engineered time and amount features.

    Identical to notebook 02, and idempotent — running it twice on the
    same frame produces the same result.
    """
    df = df.copy()

    df["Hour"] = (df["Time"] // 3600).astype(int) % 24
    df["Day"] = (df["Time"] // (3600 * 24)).astype(int) % 7

    # Cyclical encoding so 23:00 and 00:00 are close together.
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    conditions = [
        (df["Hour"] >= 6) & (df["Hour"] < 12),
        (df["Hour"] >= 12) & (df["Hour"] < 18),
        (df["Hour"] >= 18) & (df["Hour"] < 24),
    ]
    choices = ["Morning", "Afternoon", "Evening"]
    df["Time_of_day"] = np.select(conditions, choices, default="Night")

    # Amount is heavily right-skewed; the log makes it usable by the
    # linear models and better behaved for the scaler.
    df["amount_log"] = np.log1p(df["Amount"])

    return df


def prepare_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take raw (or already-engineered) transactions and return exactly
    MODEL_FEATURES in the right order, ready for any saved preprocessor.

    Validates first, then engineers, then drops everything else —
    including Class, which must never reach a model as an input.
    """
    clean, _warnings = validate(df, strict=True)
    return create_features(clean)[MODEL_FEATURES]


def split_feature_types(df: pd.DataFrame, exclude: list[str] | None = None):
    """
    Split columns into numeric and categorical lists for a ColumnTransformer.
    Mirrors how the saved preprocessors were built.
    """
    exclude = (exclude or []) + [TARGET]
    candidates = df.drop(columns=[c for c in exclude if c in df.columns])

    numeric = candidates.select_dtypes(include=[np.number]).columns.tolist()
    categorical = candidates.select_dtypes(include=[object]).columns.tolist()
    return numeric, categorical


def blank_transaction() -> dict:
    """
    A neutral transaction to seed the app's manual entry form: every PCA
    component at 0 (the dataset mean), a midday timestamp, a small amount.
    """
    values = {c: 0.0 for c in V_COLUMNS}
    values["Time"] = 43200.0
    values["Amount"] = 100.0
    return values
