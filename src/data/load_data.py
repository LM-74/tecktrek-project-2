"""
src/data/load_data.py

One place that knows how to read data off disk.

The training scripts use load_splits(); the Streamlit app uses
load_uploaded_csv() for files a user drops in. Both end up validated the
same way.
"""

import pandas as pd

from src.data.validation import TARGET, validate
from src.utils.config import RAW_CSV, SPLITS


class DataNotFoundError(FileNotFoundError):
    """Raised when an expected data file hasn't been generated yet."""


def _read(path, what: str) -> pd.DataFrame:
    if not path.exists():
        raise DataNotFoundError(
            f"Could not find the {what} file at {path}. "
            "Run notebooks/01_loading_and_eda.ipynb and "
            "notebooks/02_feature_engineering.ipynb once to generate it."
        )
    return pd.read_csv(path)


def load_raw() -> pd.DataFrame:
    """The original Kaggle creditcard.csv."""
    return _read(RAW_CSV, "raw dataset")


def load_split(name: str) -> pd.DataFrame:
    """Load one processed split: 'train', 'val' or 'test'."""
    if name not in SPLITS:
        raise KeyError(f"Unknown split {name!r}. Expected one of {list(SPLITS)}.")
    return _read(SPLITS[name], f"{name} split")


def load_splits(names=("train", "val", "test")) -> tuple[pd.DataFrame, ...]:
    """Load several splits at once: train_df, val_df, test_df = load_splits()."""
    return tuple(load_split(n) for n in names)


def split_xy(df: pd.DataFrame):
    """Separate features from the target."""
    if TARGET not in df.columns:
        raise KeyError(f"Column {TARGET!r} not found — this split has no labels.")
    return df.drop(columns=[TARGET]), df[TARGET]


def load_uploaded_csv(file_like) -> tuple[pd.DataFrame, list[str]]:
    """
    Read a user-uploaded CSV and validate it.

    Returns the cleaned frame and any soft warnings. Raises SchemaError
    with a readable message if the file isn't usable at all.
    """
    df = pd.read_csv(file_like)
    return validate(df, strict=True)
