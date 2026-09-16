"""
Inference: load saved models and score transactions.

batch_predict is deliberately not imported here. It runs as a script
(`python -m src.prediction.batch_predict`), and importing it at package
level makes Python load the module twice and emit a RuntimeWarning.
Import it directly instead:

    from src.prediction.batch_predict import score_dataframe
"""

from src.prediction.predictor import (
    FraudPredictor,
    ModelUnavailableError,
    available_models,
    missing_reason,
    module_available,
    tensorflow_available,
)

__all__ = [
    "FraudPredictor",
    "ModelUnavailableError",
    "available_models",
    "missing_reason",
    "module_available",
    "tensorflow_available",
]