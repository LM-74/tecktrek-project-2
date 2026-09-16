"""
src/models/train_autoencoder.py

Deep Learning track — unsupervised anomaly detection (Member 3).

The script version of notebooks/05_anomaly_detection.ipynb. Run it from
the project root:

    python -m src.models.train_autoencoder

The autoencoder only ever sees normal transactions during training, so it
learns to reconstruct legitimate behaviour well and unusual behaviour
badly. Reconstruction error becomes the anomaly score. This is what lets
it flag fraud patterns nobody labelled.
"""

import joblib
import numpy as np
import pandas as pd

from src.data.load_data import load_splits, split_xy
from src.evaluation.metrics import evaluate
from src.evaluation.threshold import best_f1_threshold
from src.features.feature_engineering import (
    EXCLUDE_FROM_AUTOENCODER,
    MODEL_FEATURES,
    split_feature_types,
)
from src.utils.config import (
    ANOMALY_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    RANDOM_STATE,
    REPORTS_DIR,
    ensure_dirs,
)
from src.utils.logger import get_logger

logger = get_logger("train_autoencoder")


def build_preprocessor(train_df: pd.DataFrame):
    """
    Same scaling as elsewhere, but the time and raw amount columns are
    excluded — the chronological split shifts them between train and
    test, and the autoencoder would read that drift as anomalous for
    every late transaction.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric, categorical = split_feature_types(train_df, exclude=EXCLUDE_FROM_AUTOENCODER)
    logger.info("%d numeric, %d categorical features", len(numeric), len(categorical))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric),
            (
                "cat",
                Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                categorical,
            ),
        ]
    )


def build_autoencoder(input_dim: int, bottleneck: int = 8):
    """Symmetric encoder/decoder squeezing the input through a bottleneck."""
    from tensorflow import keras
    from tensorflow.keras import layers

    encoder = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(16, activation="relu"),
            layers.Dense(bottleneck, activation="relu"),
        ],
        name="encoder",
    )

    decoder = keras.Sequential(
        [
            layers.Input(shape=(bottleneck,)),
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(input_dim, activation="linear"),
        ],
        name="decoder",
    )

    return keras.Sequential([encoder, decoder], name="autoencoder")


def reconstruction_error(model, X) -> np.ndarray:
    """Per-row mean squared error — the anomaly score."""
    recon = model.predict(X, verbose=0)
    return np.mean(np.square(X - recon), axis=1)


def train(max_epochs: int = MAX_EPOCHS):
    import tensorflow as tf
    from tensorflow import keras

    ensure_dirs()
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    logger.info("Loading splits...")
    train_df, val_df, test_df = load_splits()

    preprocessor = build_preprocessor(train_df)

    X_train = preprocessor.fit_transform(train_df[MODEL_FEATURES])
    X_val = preprocessor.transform(val_df[MODEL_FEATURES])
    X_test = preprocessor.transform(test_df[MODEL_FEATURES])

    y_train = split_xy(train_df)[1].to_numpy()
    y_val = split_xy(val_df)[1].to_numpy()
    y_test = split_xy(test_df)[1].to_numpy()

    # This is what makes the method unsupervised: fraud rows are removed
    # from training and from the early-stopping validation set.
    X_train_normal = X_train[y_train == 0]
    X_val_normal = X_val[y_val == 0]
    logger.info(
        "Training on %s normal transactions (excluded %d fraud rows)",
        f"{len(X_train_normal):,}",
        int(y_train.sum()),
    )

    autoencoder = build_autoencoder(input_dim=X_train.shape[1])
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    history = autoencoder.fit(
        X_train_normal,
        X_train_normal,
        validation_data=(X_val_normal, X_val_normal),
        epochs=max_epochs,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    # Errors are computed on all rows — the model simply never trained on
    # the fraud ones.
    val_errors = reconstruction_error(autoencoder, X_val)
    test_errors = reconstruction_error(autoencoder, X_test)

    chosen = best_f1_threshold(y_val, val_errors)
    logger.info("Chosen threshold (max F1 on val): %.5f", chosen["threshold"])

    test_preds = (test_errors >= chosen["threshold"]).astype(int)
    row = evaluate(y_test, test_preds, test_errors, name="Autoencoder")
    results = pd.DataFrame([row])
    logger.info("\n%s", results.to_string(index=False))

    autoencoder.save(ANOMALY_DIR / "autoencoder.keras")
    joblib.dump(preprocessor, ANOMALY_DIR / "autoencoder_preprocessor.joblib")
    joblib.dump({"threshold": chosen["threshold"]}, ANOMALY_DIR / "autoencoder_threshold.joblib")
    results.to_csv(REPORTS_DIR / "autoencoder_results.csv", index=False)

    save_training_curve(history)
    logger.info("Saved autoencoder, preprocessor, threshold and results.")
    return autoencoder, results


def save_training_curve(history) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 4))
        plt.plot(history.history["loss"], label="train loss")
        plt.plot(history.history["val_loss"], label="val loss (normal only)")
        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction MSE")
        plt.title("Autoencoder Training Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "autoencoder_training_curve.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        logger.warning("Could not save training curve: %s", exc)


if __name__ == "__main__":
    train()
