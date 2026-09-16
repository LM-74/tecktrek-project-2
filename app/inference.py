"""
app/inference.py

The bridge between the UI and src/.

Every function here is a thin pass-through to the shared modules, kept
separate so app.py deals only with widgets and layout. Scoring done here
and scoring done from the command line follow exactly the same path.
"""

import config  # noqa: F401  — sets up sys.path for the src imports below
import pandas as pd

from src.data.validation import SchemaError, validate
from src.evaluation.metrics import alert_summary, cost_summary
from src.evaluation.threshold import threshold_sweep
from src.features.feature_engineering import blank_transaction
from src.prediction.batch_predict import score_dataframe
from src.prediction.predictor import FraudPredictor
from src.utils.logger import read_prediction_log

__all__ = [
    "SchemaError",
    "validate",
    "alert_summary",
    "cost_summary",
    "blank_transaction",
    "read_prediction_log",
    "score_batch",
    "score_one",
    "sweep_thresholds",
]


def score_batch(predictor: FraudPredictor, df: pd.DataFrame, threshold: float, source: str = "app"):
    """
    Score an uploaded file. Returns (results, summary) and writes the run
    to the prediction log.
    """
    return score_dataframe(
        df,
        model_name=predictor.name,
        threshold=threshold,
        predictor=predictor,
        source=source,
        log=True,
    )


def score_one(predictor: FraudPredictor, values: dict, threshold: float):
    """
    Score a single manually entered transaction. Returns (score, flagged).
    Logged like any other request so the audit trail stays complete.
    """
    results, _summary = score_batch(predictor, pd.DataFrame([values]), threshold, source="single")
    return float(results.loc[0, "score"]), bool(results.loc[0, "flagged"])


def sweep_thresholds(results: pd.DataFrame) -> pd.DataFrame | None:
    """
    The precision/recall/cost trade-off across thresholds, but only when
    the scored file carried labels to compare against.
    """
    if "actual" not in results.columns:
        return None
    return threshold_sweep(results["actual"].to_numpy(), results["score"].to_numpy())
