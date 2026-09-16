"""
src/evaluation/threshold.py

Choosing the decision threshold, which for an imbalanced problem matters
as much as choosing the model.

Two rules we follow everywhere:
  - tune the threshold on validation, never on test;
  - report what the threshold costs, not just the F1 it produces.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.utils.config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE


def best_f1_threshold(y_true, y_score) -> dict:
    """
    Sweep the precision-recall curve and take the point with the highest
    F1. This is how the MLP and autoencoder thresholds were chosen.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    # The final PR point has no matching threshold, hence the [:-1].
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best = int(np.argmax(f1s[:-1]))

    return {
        "threshold": float(thresholds[best]),
        "precision": float(precisions[best]),
        "recall": float(recalls[best]),
        "f1": float(f1s[best]),
    }


def best_cost_threshold(
    y_true,
    y_score,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
    n_steps: int = 200,
) -> dict:
    """
    Pick the threshold that minimises total business cost rather than
    maximising F1. Because a missed fraud costs far more than a false
    alarm, this usually lands lower than the F1-optimal threshold — it
    accepts more false alarms to catch more fraud.
    """
    y_true = np.asarray(y_true)
    candidates = np.quantile(y_score, np.linspace(0.5, 0.9999, n_steps))
    candidates = np.unique(candidates)

    best = None
    for t in candidates:
        pred = (y_score >= t).astype(int)
        fn = int(((y_true == 1) & (pred == 0)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        cost = fn * cost_fn + fp * cost_fp

        if best is None or cost < best["cost"]:
            best = {"threshold": float(t), "cost": float(cost), "fn": fn, "fp": fp}

    return best


def threshold_sweep(
    y_true,
    y_score,
    thresholds=None,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
) -> pd.DataFrame:
    """
    A table of precision/recall/F1/alerts/cost across candidate thresholds.

    Useful for the report and for the app: it shows the trade-off curve
    instead of asserting one magic number.
    """
    y_true = np.asarray(y_true)

    if thresholds is None:
        thresholds = np.quantile(y_score, np.linspace(0.5, 0.999, 25))
        thresholds = np.unique(thresholds)

    rows = []
    for t in thresholds:
        pred = (y_score >= t).astype(int)

        tp = int(((y_true == 1) & (pred == 1)).sum())
        fn = int(((y_true == 1) & (pred == 0)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        rows.append(
            {
                "threshold": float(t),
                "alerts": int(pred.sum()),
                "alert_rate": float(pred.mean()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "frauds_missed": fn,
                "cost": fn * cost_fn + fp * cost_fp,
            }
        )

    return pd.DataFrame(rows)


def save_threshold(value: float, path) -> None:
    """Persist a chosen threshold as plain text next to the model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(float(value)))


def load_threshold(path, default: float) -> float:
    """
    Read a saved threshold. Handles both formats in this repo: a plain
    .txt number (MLP) and a {'threshold': ...} joblib (autoencoder).
    Falls back to the supplied default rather than failing.
    """
    try:
        if path is None or not path.exists():
            return float(default)

        if str(path).endswith(".txt"):
            return float(path.read_text().strip())

        import joblib

        obj = joblib.load(path)
        if isinstance(obj, dict):
            return float(obj["threshold"])
        return float(obj)
    except Exception:
        return float(default)
