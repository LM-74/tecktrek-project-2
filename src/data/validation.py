"""
src/data/validation.py

Guards against bad input before it ever reaches a model.

The Streamlit app lets anyone upload a CSV, so "does this file actually
look like our transaction data?" needs a real answer with a readable
error message, not a stack trace from deep inside sklearn.
"""

import numpy as np
import pandas as pd

# The 30 columns that come straight out of the raw Kaggle dataset.
V_COLUMNS = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = ["Time"] + V_COLUMNS + ["Amount"]
TARGET = "Class"


class SchemaError(ValueError):
    """Raised when input data doesn't match the expected transaction schema."""


def check_required_columns(df: pd.DataFrame) -> None:
    """Every raw feature must be present. Extra columns are fine."""
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise SchemaError(
            "The data is missing required columns: "
            + ", ".join(missing)
            + ". Expected the raw creditcard.csv layout (Time, V1-V28, Amount)."
        )


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force the raw features to numeric. Anything unparseable becomes NaN
    and is reported by check_no_missing() below.
    """
    out = df.copy()
    out[RAW_FEATURES] = out[RAW_FEATURES].apply(pd.to_numeric, errors="coerce")
    return out


def check_no_missing(df: pd.DataFrame) -> None:
    """No NaNs allowed in the raw features — the models can't handle them."""
    bad = df[RAW_FEATURES].isna().any()
    if bad.any():
        raise SchemaError(
            "Non-numeric or missing values found in: " + ", ".join(bad[bad].index.tolist())
        )


def check_value_ranges(df: pd.DataFrame) -> list[str]:
    """
    Soft checks: return a list of human-readable warnings rather than
    raising. Odd values are worth surfacing but shouldn't block scoring.
    """
    warnings = []

    if (df["Amount"] < 0).any():
        warnings.append(f"{int((df['Amount'] < 0).sum())} row(s) have a negative Amount.")

    if (df["Time"] < 0).any():
        warnings.append(f"{int((df['Time'] < 0).sum())} row(s) have a negative Time.")

    # The PCA components are standardised; anything past ~|50| is extreme.
    extreme = (df[V_COLUMNS].abs() > 50).any(axis=1).sum()
    if extreme:
        warnings.append(f"{int(extreme)} row(s) have unusually large PCA component values.")

    return warnings


def validate(df: pd.DataFrame, strict: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """
    Full validation pass. Returns the cleaned frame plus any soft warnings.

    strict=True raises SchemaError on missing columns or unparseable
    values. Set it to False only if you intend to handle NaNs yourself.
    """
    check_required_columns(df)
    out = coerce_numeric(df)

    if strict:
        check_no_missing(out)

    return out, check_value_ranges(out)


def describe_target(df: pd.DataFrame) -> dict:
    """Class balance summary — used in EDA and to sanity check the splits."""
    if TARGET not in df.columns:
        return {}

    y = df[TARGET].to_numpy()
    n_fraud = int((y == 1).sum())
    return {
        "rows": len(y),
        "fraud": n_fraud,
        "legit": int((y == 0).sum()),
        "fraud_rate": float(np.mean(y == 1)) if len(y) else 0.0,
    }
