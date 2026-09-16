"""
src/utils/logger.py

Two separate jobs that both count as "logging":

1. get_logger()   — ordinary console logging for the training scripts.
2. log_prediction() — the MLOps requirement: an audit trail of every
   scoring request, written to logs/predictions.csv.

We log one row per *request* rather than per transaction, so scoring
50,000 rows doesn't produce a 50,000-line log file.
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from src.utils.config import INPUT_SCHEMA_VERSION, LOGS_DIR, PREDICTION_LOG

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


# ---------------------------------------------------------------
# Console logging
# ---------------------------------------------------------------
def get_logger(name: str = "fraud", level: int = logging.INFO) -> logging.Logger:
    """A configured logger. Safe to call repeatedly; handlers aren't duplicated."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False

    return logger


# ---------------------------------------------------------------
# Prediction logging
# ---------------------------------------------------------------
def log_prediction(
    model_name: str,
    threshold: float,
    n_transactions: int,
    n_alerts: int,
    source: str = "app",
) -> None:
    """
    Append one row describing a scoring request.

    Never raises: a logging failure must not take down the app or a batch
    job. Records the timestamp, model, threshold and schema version so a
    past prediction can be traced back to the artifact that produced it.
    """
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        row = pd.DataFrame(
            [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "model": model_name,
                    "threshold": round(float(threshold), 6),
                    "source": source,
                    "n_transactions": int(n_transactions),
                    "n_alerts": int(n_alerts),
                    "alert_rate": round(n_alerts / n_transactions, 6) if n_transactions else 0.0,
                    "input_schema_version": INPUT_SCHEMA_VERSION,
                }
            ]
        )
        row.to_csv(
            PREDICTION_LOG,
            mode="a",
            header=not PREDICTION_LOG.exists(),
            index=False,
        )
    except Exception:  # pragma: no cover - logging must never break callers
        pass


def read_prediction_log(n: int = 20) -> pd.DataFrame:
    """The most recent log entries, newest first. Empty frame if no log yet."""
    if not PREDICTION_LOG.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(PREDICTION_LOG).tail(n).iloc[::-1]
    except Exception:
        return pd.DataFrame()
