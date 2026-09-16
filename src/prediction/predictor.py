"""
src/prediction/predictor.py

The inference layer: load a saved artifact, score transactions with it.

Deliberately free of Streamlit. The app wraps these functions in its own
caching, and batch_predict.py calls them from the command line — both get
identical predictions because both go through this file.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.evaluation.threshold import load_threshold
from src.features.feature_engineering import prepare_for_model
from src.utils.config import INSTALL_HINTS, MODEL_REGISTRY


class ModelUnavailableError(RuntimeError):
    """
    Raised when a model can't be loaded — a missing artifact file or a
    missing package. Carries a message that says how to fix it.
    """


@lru_cache(maxsize=None)
def module_available(module: str) -> bool:
    """
    Can this optional dependency actually be imported?

    Checked with importlib rather than a try/import so a heavy package
    like TensorFlow isn't loaded into memory just to answer the question.
    Cached because the answer can't change inside one process.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def tensorflow_available() -> bool:
    """Whether the deep-learning models can be used. Kept as a named helper."""
    return module_available("tensorflow")


def missing_reason(name: str) -> str | None:
    """
    None if the model is usable right now, otherwise a short reason.

    This must catch missing *packages*, not just missing files. A pickled
    XGBoost pipeline looks fine on disk but raises ModuleNotFoundError on
    load if xgboost isn't installed — so the model has to be filtered out
    of the menu before anyone can select it.
    """
    cfg = MODEL_REGISTRY[name]

    if not cfg["model_path"].exists():
        return f"missing file {cfg['model_path'].name}"

    pre = cfg.get("preprocessor_path")
    if pre is not None and not pre.exists():
        return f"missing file {pre.name}"

    required = cfg.get("requires")
    if required and not module_available(required):
        hint = INSTALL_HINTS.get(required, f"pip install {required}")
        return f"{required} not installed — run `{hint}`"

    return None


def available_models() -> list[str]:
    """Names of every model that can be loaded in this environment."""
    return [n for n in MODEL_REGISTRY if missing_reason(n) is None]


@dataclass
class FraudPredictor:
    """A loaded model plus everything needed to interpret its output."""

    name: str
    kind: str
    score_type: str
    model: object
    preprocessor: object | None
    tuned_threshold: float

    # -----------------------------------------------------------
    # Loading
    # -----------------------------------------------------------
    @classmethod
    def load(cls, name: str) -> "FraudPredictor":
        if name not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model {name!r}. Known: {list(MODEL_REGISTRY)}")

        reason = missing_reason(name)
        if reason:
            raise ModelUnavailableError(f"Cannot load {name!r}: {reason}.")

        cfg = MODEL_REGISTRY[name]
        threshold = load_threshold(cfg.get("threshold_path"), cfg["default_threshold"])

        try:
            if cfg["kind"] == "classical":
                # These pickles are complete sklearn Pipelines: preprocessing
                # and model travel together, so there's nothing else to load.
                model, preprocessor = joblib.load(cfg["model_path"]), None
            else:
                from tensorflow import keras

                preprocessor = joblib.load(cfg["preprocessor_path"])
                model = keras.models.load_model(cfg["model_path"])
        except ModuleNotFoundError as exc:
            # Belt and braces: missing_reason() should have caught this
            # already, but a pickle can pull in a package we didn't list.
            package = exc.name or "a required package"
            hint = INSTALL_HINTS.get(package, f"pip install {package}")
            raise ModelUnavailableError(
                f"Cannot load {name!r}: the saved model needs the {package!r} "
                f"package, which isn't installed. Run `{hint}`, or "
                "`pip install -r requirements.txt` to get everything at once."
            ) from exc

        return cls(
            name=name,
            kind=cfg["kind"],
            score_type=cfg["score_type"],
            model=model,
            preprocessor=preprocessor,
            tuned_threshold=threshold,
        )

    # -----------------------------------------------------------
    # Scoring
    # -----------------------------------------------------------
    def _reconstruction_error(self, X) -> np.ndarray:
        """Per-row mean squared error between input and reconstruction."""
        recon = self.model.predict(X, verbose=0)
        return np.mean(np.square(X - recon), axis=1)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Score raw transactions.

        Returns a fraud probability for the supervised models, or a
        reconstruction error for the autoencoder. Check score_type before
        presenting the number to anyone — they are not comparable.
        """
        X = prepare_for_model(df)

        if self.kind == "classical":
            return self.model.predict_proba(X)[:, 1]

        X_proc = self.preprocessor.transform(X).astype(np.float32)

        if self.kind == "keras":
            return self.model.predict(X_proc, verbose=0).ravel()

        return self._reconstruction_error(X_proc)

    def score_table(self, df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
        """
        Score a batch and return a tidy result table: enough context to
        read each row, the score, and the resulting decision.

        Keeps the true label as 'actual' when the input happens to carry
        one, which is what lets the app evaluate a scored file.
        """
        threshold = self.tuned_threshold if threshold is None else float(threshold)

        scores = self.score(df)
        flags = (scores >= threshold).astype(int)

        out = pd.DataFrame(
            {
                "row": np.arange(len(df)),
                "Time": df["Time"].to_numpy(),
                "Amount": df["Amount"].to_numpy(),
                "score": scores,
                "flagged": flags,
                "decision": np.where(flags == 1, "REVIEW", "Approve"),
            }
        )

        if "Class" in df.columns:
            out["actual"] = df["Class"].to_numpy()

        return out