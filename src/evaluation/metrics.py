"""
src/evaluation/metrics.py

The shared metric vocabulary for the whole project, so every member's
comparison table has the same columns computed the same way.

PR-AUC is the headline metric. With 492 frauds in 284,807 transactions,
accuracy is meaningless here — a model predicting "never fraud" scores
99.8%.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE

METRIC_COLUMNS = ["Precision", "Recall", "F1 Score", "ROC AUC", "PR AUC"]


def binary_metrics(y_true, y_pred, y_score) -> dict:
    """
    The standard five. y_pred is the thresholded decision, y_score the
    continuous probability or anomaly score.
    """
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, y_score),
        "PR AUC": average_precision_score(y_true, y_score),
    }


def confusion_counts(y_true, y_pred) -> dict:
    """Confusion matrix as named counts, which read better than a 2x2 array."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def business_cost(
    y_true,
    y_pred,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
) -> float:
    """
    Total cost under our stated fraud-loss assumptions. Missing a fraud is
    far more expensive than reviewing a legitimate transaction, which is
    exactly why the threshold shouldn't be left at 0.5.
    """
    counts = confusion_counts(y_true, y_pred)
    return counts["fn"] * cost_fn + counts["fp"] * cost_fp


def evaluate(y_true, y_pred, y_score, name: str | None = None) -> dict:
    """Metrics, confusion counts and cost in a single row, ready for a table."""
    row = binary_metrics(y_true, y_pred, y_score)
    row.update(confusion_counts(y_true, y_pred))
    row["Business Cost"] = business_cost(y_true, y_pred)
    if name is not None:
        row["Model"] = name
    return row


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    """Turn evaluate() rows into a tidy, consistently ordered table."""
    df = pd.DataFrame(rows)
    lead = [c for c in ["Model", "Strategy"] if c in df.columns]
    ordered = lead + [c for c in METRIC_COLUMNS if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    return df[ordered + rest]


# ---------------------------------------------------------------
# Operational view — used by the Streamlit app
# ---------------------------------------------------------------
def alert_summary(results: pd.DataFrame) -> dict:
    """
    How many alerts does this threshold actually generate? A model that
    flags 8% of all traffic is unusable no matter how good its recall is.

    Expects the scored table produced by the prediction layer.
    """
    n = len(results)
    flagged = int(results["flagged"].sum())
    return {
        "transactions": n,
        "alerts": flagged,
        "alert_rate": flagged / n if n else 0.0,
        "amount_flagged": float(results.loc[results["flagged"] == 1, "Amount"].sum()),
    }


def cost_summary(results: pd.DataFrame) -> dict | None:
    """
    Confusion counts and cost for a scored batch — but only when the file
    carried real labels. Returns None otherwise; we never invent ground
    truth to fill a dashboard.
    """
    if "actual" not in results.columns:
        return None

    y = results["actual"].to_numpy()
    p = results["flagged"].to_numpy()

    counts = confusion_counts(y, p)
    tp, fn, fp = counts["tp"], counts["fn"], counts["fp"]

    counts["recall"] = tp / (tp + fn) if (tp + fn) else float("nan")
    counts["precision"] = tp / (tp + fp) if (tp + fp) else float("nan")
    counts["cost"] = business_cost(y, p)
    return counts


def recall_at_precision(y_true, y_score, min_precision: float = 0.9) -> float:
    """
    Best recall achievable while holding precision at or above a floor.
    This is the number a fraud team actually cares about: "how much fraud
    can we catch if analysts only tolerate 1 false alarm in 10?"
    """
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    ok = precisions >= min_precision
    return float(np.max(recalls[ok])) if ok.any() else 0.0
