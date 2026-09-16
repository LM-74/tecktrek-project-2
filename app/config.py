"""
src/utils/config.py

Every path, model location and tunable constant the project uses lives
here. Notebooks, training scripts, the prediction layer and the Streamlit
app all import from this one file, so nothing is hard-coded twice.

Imports nothing from the rest of the project, so it is always safe to
import first.
"""

from pathlib import Path

# ---------------------------------------------------------------
# Paths (resolved relative to this file, so the working directory
# you launch from never matters)
# ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
CLASSICAL_DIR = MODELS_DIR / "classical"
DEEP_LEARNING_DIR = MODELS_DIR / "deep_learning"
ANOMALY_DIR = MODELS_DIR / "anomaly"

REPORTS_DIR = PROJECT_ROOT / "reports" / "model_results"
LOGS_DIR = PROJECT_ROOT / "logs"
PREDICTION_LOG = LOGS_DIR / "predictions.csv"

RAW_CSV = RAW_DATA_DIR / "creditcard.csv"

# Split files written by the notebooks.
SPLITS = {
    "train": PROCESSED_DATA_DIR / "model_train.csv",
    "val": PROCESSED_DATA_DIR / "model_val.csv",
    "test": PROCESSED_DATA_DIR / "model_test.csv",
}

# ---------------------------------------------------------------
# Reproducibility / training defaults
# ---------------------------------------------------------------
RANDOM_STATE = 42
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 8
LEARNING_RATE = 1e-3

# ---------------------------------------------------------------
# Business cost assumptions
#
# Project 2 asks for an explicit fraud-loss matrix. These are our own
# stated assumptions for the cost simulation, not real bank figures.
# ---------------------------------------------------------------
COST_FALSE_NEGATIVE = 500.0   # money lost when a fraud slips through
COST_FALSE_POSITIVE = 5.0     # analyst time spent on a false alarm

# ---------------------------------------------------------------
# Model registry
#
# "kind" tells the predictor how to load and read each artifact:
#   classical   -> sklearn Pipeline, exposes predict_proba
#   keras       -> preprocessor + keras model, outputs a probability
#   autoencoder -> preprocessor + keras model, outputs reconstruction error
#
# "requires" is the third-party package needed to *unpickle and run* the
# artifact, or None for plain scikit-learn. A pickled XGBClassifier can't
# be loaded without xgboost installed, and a .keras file can't be loaded
# without tensorflow — so the app checks this before offering the model
# rather than crashing when someone selects it.
# ---------------------------------------------------------------
MODEL_REGISTRY = {
    "Logistic Regression": {
        "kind": "classical",
        "requires": None,
        "model_path": CLASSICAL_DIR / "logistic_regression.pkl",
        "score_type": "probability",
        "default_threshold": 0.5,
        "note": "Baseline. High recall, very low precision — lots of false alarms.",
    },
    "Random Forest": {
        "kind": "classical",
        "requires": None,
        "model_path": CLASSICAL_DIR / "random_forest.pkl",
        "score_type": "probability",
        "default_threshold": 0.5,
        "note": "Class-weighted ensemble baseline.",
    },
    "XGBoost (tuned)": {
        "kind": "classical",
        "requires": "xgboost",
        "model_path": CLASSICAL_DIR / "xgboost.pkl",
        "score_type": "probability",
        "default_threshold": 0.5,
        "note": "Best classical model, tuned with RandomizedSearchCV on PR-AUC.",
    },
    "MLP (deep learning)": {
        "kind": "keras",
        "requires": "tensorflow",
        "model_path": DEEP_LEARNING_DIR / "mlp_model.keras",
        "preprocessor_path": DEEP_LEARNING_DIR / "mlp_preprocessor.pkl",
        "threshold_path": REPORTS_DIR / "mlp_threshold.txt",
        "score_type": "probability",
        "default_threshold": 0.9865,  # replaced by threshold_path when present
        "note": "Supervised neural net. Threshold picked by max F1 on validation.",
    },
    "Autoencoder (anomaly)": {
        "kind": "autoencoder",
        "requires": "tensorflow",
        "model_path": ANOMALY_DIR / "autoencoder.keras",
        "preprocessor_path": ANOMALY_DIR / "autoencoder_preprocessor.joblib",
        "threshold_path": ANOMALY_DIR / "autoencoder_threshold.joblib",
        "score_type": "anomaly_score",
        "default_threshold": 7.586,  # replaced by threshold_path when present
        "note": "Unsupervised. Trained on normal transactions only; the score is "
        "reconstruction error, not a probability.",
    },
}

# How to install each optional dependency, shown in the app when a model
# can't be loaded so the fix is obvious without digging through docs.
INSTALL_HINTS = {
    "xgboost": "pip install xgboost",
    "tensorflow": "pip install tensorflow-cpu",
}

# Slider bounds per score type, so the app's threshold control makes
# sense for probabilities and reconstruction errors alike.
THRESHOLD_RANGES = {
    "probability": (0.0, 1.0, 0.005),      # min, max, step
    "anomaly_score": (0.0, 50.0, 0.1),
}

# Bumped whenever the feature set changes, and written to the prediction log.
INPUT_SCHEMA_VERSION = "36-features-v1"

APP_TITLE = "Fraud Detection & Anomaly Intelligence"


def ensure_dirs() -> None:
    """Create the output directories training scripts write into."""
    for path in (CLASSICAL_DIR, DEEP_LEARNING_DIR, ANOMALY_DIR, REPORTS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
