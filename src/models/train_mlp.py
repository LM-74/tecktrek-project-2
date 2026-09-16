"""
src/models/train_mlp.py

Deep Learning track — supervised MLP for tabular fraud classification
(Member 2).

The script version of notebooks/04_deep_learning.ipynb. Run it from the
project root:

    python -m src.models.train_mlp

It fits the preprocessor on train only, trains with class weighting and
early stopping, picks the decision threshold on validation, and applies
that fixed threshold once to test.
"""

import joblib
import numpy as np
import pandas as pd

from src.data.load_data import load_splits, split_xy
from src.evaluation.metrics import evaluate
from src.evaluation.threshold import best_f1_threshold, save_threshold
from src.features.feature_engineering import MODEL_FEATURES, split_feature_types
from src.utils.config import (
    BATCH_SIZE,
    DEEP_LEARNING_DIR,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    RANDOM_STATE,
    REPORTS_DIR,
    ensure_dirs,
)
from src.utils.logger import get_logger

logger = get_logger("train_mlp")


def build_preprocessor(train_df: pd.DataFrame):
    """Scale the numeric columns, one-hot the categorical ones."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric, categorical = split_feature_types(train_df)

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


def build_model(input_dim: int, hidden_dims=(64, 32, 16), dropout: float = 0.3):
    """A small dense network — the dataset is 36 features, not images."""
    from tensorflow import keras

    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for units in hidden_dims:
        x = keras.layers.Dense(units, activation="relu")(x)
        x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs)


def train(max_epochs: int = MAX_EPOCHS):
    import tensorflow as tf
    from tensorflow import keras

    ensure_dirs()
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    logger.info("Loading splits...")
    train_df, val_df, test_df = load_splits()

    preprocessor = build_preprocessor(train_df)

    # Fit on train only — transforming val/test with statistics learned
    # from them would leak information into the evaluation.
    X_train = preprocessor.fit_transform(train_df[MODEL_FEATURES]).astype(np.float32)
    X_val = preprocessor.transform(val_df[MODEL_FEATURES]).astype(np.float32)
    X_test = preprocessor.transform(test_df[MODEL_FEATURES]).astype(np.float32)

    y_train = split_xy(train_df)[1].to_numpy().astype(np.float32)
    y_val = split_xy(val_df)[1].to_numpy().astype(np.float32)
    y_test = split_xy(test_df)[1].to_numpy().astype(np.float32)

    joblib.dump(preprocessor, DEEP_LEARNING_DIR / "mlp_preprocessor.pkl")
    logger.info("Feature matrix: %s", X_train.shape)

    # Weight the fraud class inversely to its frequency.
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    class_weight = {0: 1.0, 1: float(n_neg / n_pos)}
    logger.info("class_weight: %s", class_weight)

    model = build_model(input_dim=X_train.shape[1])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max", patience=PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc", mode="max", factor=0.5, patience=3
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    # Threshold chosen on validation, then applied once to test.
    val_probs = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0).ravel()
    chosen = best_f1_threshold(y_val, val_probs)
    logger.info("Chosen threshold (best val F1): %.4f", chosen["threshold"])

    test_probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).ravel()
    test_preds = (test_probs >= chosen["threshold"]).astype(int)

    row = evaluate(y_test, test_preds, test_probs, name="MLP")
    results = pd.DataFrame([row])
    logger.info("\n%s", results.to_string(index=False))

    model.save(DEEP_LEARNING_DIR / "mlp_model.keras")
    save_threshold(chosen["threshold"], REPORTS_DIR / "mlp_threshold.txt")
    results.to_csv(REPORTS_DIR / "mlp_results.csv", index=False)

    save_training_curves(history)
    logger.info("Saved model, threshold and results.")
    return model, results


def save_training_curves(history) -> None:
    """Loss and PR-AUC per epoch — the DL track asks for training curves."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history.history["loss"], label="Train Loss")
        axes[0].plot(history.history["val_loss"], label="Val Loss")
        axes[0].set_title("Loss over Epochs")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Binary Cross-Entropy")
        axes[0].legend()

        axes[1].plot(history.history["pr_auc"], label="Train PR-AUC")
        axes[1].plot(history.history["val_pr_auc"], label="Val PR-AUC")
        axes[1].set_title("PR-AUC over Epochs")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "mlp_training_curves.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # plotting must not fail the training run
        logger.warning("Could not save training curves: %s", exc)


if __name__ == "__main__":
    train()
