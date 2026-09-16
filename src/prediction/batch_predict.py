"""
src/prediction/batch_predict.py

Batch scoring — required deliverable: "batch scoring script that accepts
CSV files".

Used two ways:
  - the Streamlit app calls score_dataframe() for uploaded files;
  - the command line scores a file and writes the results out:

        python -m src.prediction.batch_predict \
            --input data/processed/model_test.csv \
            --output reports/scored_test.csv \
            --model "XGBoost (tuned)"
"""

import argparse
import sys

import pandas as pd

from src.data.validation import validate
from src.evaluation.metrics import alert_summary, cost_summary
from src.prediction.predictor import FraudPredictor, available_models
from src.utils.config import MODEL_REGISTRY
from src.utils.logger import get_logger, log_prediction

logger = get_logger("batch_predict")


def score_dataframe(
    df: pd.DataFrame,
    model_name: str,
    threshold: float | None = None,
    predictor: FraudPredictor | None = None,
    source: str = "batch",
    log: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Validate, score and summarise a batch of transactions.

    Returns the scored table and a summary dict. Pass an already-loaded
    predictor to avoid re-reading the artifact (the app does this).
    """
    clean, warnings = validate(df, strict=True)

    predictor = predictor or FraudPredictor.load(model_name)
    threshold = predictor.tuned_threshold if threshold is None else float(threshold)

    results = predictor.score_table(clean, threshold)

    summary = alert_summary(results)
    summary["warnings"] = warnings
    summary["threshold"] = threshold
    summary["model"] = predictor.name
    summary["costs"] = cost_summary(results)

    if log:
        log_prediction(
            model_name=predictor.name,
            threshold=threshold,
            n_transactions=summary["transactions"],
            n_alerts=summary["alerts"],
            source=source,
        )

    return results, summary


def score_csv(
    input_path: str,
    output_path: str,
    model_name: str,
    threshold: float | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Read a CSV, score it, write the results, return them."""
    logger.info("Reading %s", input_path)
    df = pd.read_csv(input_path)

    if limit:
        df = df.head(limit)

    results, summary = score_dataframe(df, model_name, threshold, source="cli")

    for warning in summary["warnings"]:
        logger.warning(warning)

    logger.info(
        "Scored %s transactions with %s at threshold %.4f -> %s alerts (%.3f%%)",
        f"{summary['transactions']:,}",
        summary["model"],
        summary["threshold"],
        f"{summary['alerts']:,}",
        summary["alert_rate"] * 100,
    )

    costs = summary["costs"]
    if costs:
        logger.info(
            "Against the labels in the file: caught %s, missed %s, recall %.3f, cost %.0f",
            costs["tp"],
            costs["fn"],
            costs["recall"],
            costs["cost"],
        )

    results.to_csv(output_path, index=False)
    logger.info("Wrote %s", output_path)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score a CSV of transactions for fraud.")
    parser.add_argument("--input", required=True, help="CSV with Time, V1-V28, Amount")
    parser.add_argument("--output", required=True, help="Where to write the scored CSV")
    parser.add_argument(
        "--model",
        default="XGBoost (tuned)",
        choices=list(MODEL_REGISTRY),
        help="Which saved model to score with",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold. Defaults to the one tuned during training.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N rows")
    args = parser.parse_args(argv)

    usable = available_models()
    if args.model not in usable:
        logger.error("%r is not available. Loadable models: %s", args.model, usable)
        return 1

    try:
        score_csv(args.input, args.output, args.model, args.threshold, args.limit)
    except Exception as exc:
        logger.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
