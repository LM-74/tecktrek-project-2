"""
src/evaluation/error_analysis.py

Required by the ML track: analyse the false negatives and false positives
rather than stopping at a single aggregate score.

The question worth answering is *which* frauds we miss. Missing 20 small
frauds is a very different problem from missing 2 large ones.
"""

import numpy as np
import pandas as pd

AMOUNT_BINS = [-np.inf, 10, 50, 200, 1000, np.inf]
AMOUNT_LABELS = ["0-10", "10-50", "50-200", "200-1000", "1000+"]


def label_errors(y_true, y_pred) -> np.ndarray:
    """Tag every row as TP, FP, FN or TN."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    out = np.full(len(y_true), "TN", dtype=object)
    out[(y_true == 1) & (y_pred == 1)] = "TP"
    out[(y_true == 1) & (y_pred == 0)] = "FN"
    out[(y_true == 0) & (y_pred == 1)] = "FP"
    return out


def build_error_frame(df: pd.DataFrame, y_true, y_pred, y_score) -> pd.DataFrame:
    """
    Attach predictions and error labels back onto the original features,
    so errors can be sliced by Amount, Hour, Time_of_day, etc.
    """
    out = df.copy().reset_index(drop=True)
    out["actual"] = np.asarray(y_true)
    out["predicted"] = np.asarray(y_pred)
    out["score"] = np.asarray(y_score)
    out["error_type"] = label_errors(y_true, y_pred)

    if "Amount" in out.columns:
        out["amount_band"] = pd.cut(out["Amount"], bins=AMOUNT_BINS, labels=AMOUNT_LABELS)

    return out


def errors_by_segment(error_df: pd.DataFrame, by: str) -> pd.DataFrame:
    """
    Recall and error counts broken down by any column — amount_band,
    Time_of_day, Hour. Shows where the model is weakest.
    """
    if by not in error_df.columns:
        raise KeyError(f"Column {by!r} not found in the error frame.")

    grouped = (
        error_df.groupby(by, observed=False)["error_type"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=["TP", "FN", "FP", "TN"], fill_value=0)
    )

    frauds = grouped["TP"] + grouped["FN"]
    grouped["frauds"] = frauds
    grouped["recall"] = np.where(frauds > 0, grouped["TP"] / frauds.replace(0, np.nan), np.nan)

    return grouped.reset_index()


def missed_frauds(error_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    The false negatives, highest amount first — the most expensive
    mistakes the model made, and the best place to start improving it.
    """
    cols = [c for c in ["Amount", "Hour", "Time_of_day", "score", "actual"] if c in error_df.columns]
    return (
        error_df[error_df["error_type"] == "FN"]
        .sort_values("Amount", ascending=False)
        .head(n)[cols]
    )


def worst_false_positives(error_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    The legitimate transactions the model was most confident about. These
    are what annoy real customers, so they belong in the report too.
    """
    cols = [c for c in ["Amount", "Hour", "Time_of_day", "score", "actual"] if c in error_df.columns]
    return (
        error_df[error_df["error_type"] == "FP"]
        .sort_values("score", ascending=False)
        .head(n)[cols]
    )


def summarize(error_df: pd.DataFrame) -> dict:
    """Headline error numbers, including the money attached to missed fraud."""
    counts = error_df["error_type"].value_counts().to_dict()
    fn_amount = float(error_df.loc[error_df["error_type"] == "FN", "Amount"].sum()) \
        if "Amount" in error_df.columns else float("nan")

    tp, fn = counts.get("TP", 0), counts.get("FN", 0)
    return {
        "TP": tp,
        "FN": fn,
        "FP": counts.get("FP", 0),
        "TN": counts.get("TN", 0),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "missed_fraud_amount": fn_amount,
    }
