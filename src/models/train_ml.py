"""
train_ml.py

Classical ML training for the Fraud Detection project (Member 1).

What this script does:
1. Loads the processed train/val/test csv files.
2. Trains Logistic Regression, Random Forest, and XGBoost baselines.
3. Runs a class-imbalance experiment (no handling vs class weighting vs under-sampling).
4. Tunes XGBoost's hyperparameters with RandomizedSearchCV.
5. Saves the final models and results tables so the rest of the team can use them.

This is the "clean, reusable" version of what we explored in
notebooks/03_ml_models.ipynb. Run it with:

    python src/models/train_ml.py
"""

import os

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

DATA_DIR = "data/processed"
MODEL_DIR = "models/classical"
RESULTS_DIR = "reports/model_results"


# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------
def load_data():
    train_df = pd.read_csv(f"{DATA_DIR}/model_train.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/model_test.csv")
    return train_df, test_df


# ---------------------------------------------------------------
# Preprocessing (same idea as the notebook: scale numeric columns,
# one-hot encode the categorical ones)
# ---------------------------------------------------------------
def build_preprocessor(df):
    numeric_features = df.drop(columns=["Class"]).select_dtypes(include="number").columns.tolist()
    categorical_features = df.drop(columns=["Class"]).select_dtypes(include="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return preprocessor


# ---------------------------------------------------------------
# Simple metric helper (Member 4 will build the "official" shared
# evaluation module later; this is just enough for us to compare models)
# ---------------------------------------------------------------
def evaluate(y_true, y_pred, y_proba):
    return {
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_proba),
        "PR AUC": average_precision_score(y_true, y_proba),
    }


# ---------------------------------------------------------------
# Step 1: baseline models
# ---------------------------------------------------------------
def train_baseline_models(preprocessor, X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=8, scale_pos_weight=4, random_state=RANDOM_STATE
        ),
    }

    rows = []
    fitted_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        row = evaluate(y_test, y_pred, y_proba)
        row["Model"] = name
        rows.append(row)
        fitted_pipelines[name] = pipe

    results_df = pd.DataFrame(rows)[["Model", "Precision", "Recall", "F1 Score", "ROC AUC", "PR AUC"]]
    return results_df, fitted_pipelines


# ---------------------------------------------------------------
# Step 2: class imbalance experiment (tested on XGBoost)
# ---------------------------------------------------------------
def imbalance_experiment(preprocessor, train_df, X_train, y_train, X_test, y_test):
    rows = []

    # Strategy 1: no handling at all
    pipe_none = Pipeline(
        [("preprocessor", preprocessor), ("model", xgb.XGBClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE))]
    )
    pipe_none.fit(X_train, y_train)
    pred, proba = pipe_none.predict(X_test), pipe_none.predict_proba(X_test)[:, 1]
    rows.append({"Strategy": "No handling", **evaluate(y_test, pred, proba)})

    # Strategy 2: class weighting via scale_pos_weight
    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    pipe_weighted = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", xgb.XGBClassifier(n_estimators=300, max_depth=8, scale_pos_weight=fraud_ratio, random_state=RANDOM_STATE)),
        ]
    )
    pipe_weighted.fit(X_train, y_train)
    pred, proba = pipe_weighted.predict(X_test), pipe_weighted.predict_proba(X_test)[:, 1]
    rows.append({"Strategy": "Class weighting", **evaluate(y_test, pred, proba)})

    # Strategy 3: random under-sampling of the majority (legit) class
    fraud_train = train_df[train_df["Class"] == 1]
    legit_train = train_df[train_df["Class"] == 0].sample(n=len(fraud_train), random_state=RANDOM_STATE)
    train_under = pd.concat([fraud_train, legit_train]).sample(frac=1, random_state=RANDOM_STATE)

    X_train_under = train_under.drop(columns=["Class"])
    y_train_under = train_under["Class"]

    pipe_under = Pipeline(
        [("preprocessor", preprocessor), ("model", xgb.XGBClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE))]
    )
    pipe_under.fit(X_train_under, y_train_under)
    pred, proba = pipe_under.predict(X_test), pipe_under.predict_proba(X_test)[:, 1]
    rows.append({"Strategy": "Random under-sampling", **evaluate(y_test, pred, proba)})

    return pd.DataFrame(rows)[["Strategy", "Precision", "Recall", "F1 Score", "ROC AUC", "PR AUC"]]


# ---------------------------------------------------------------
# Step 3: hyperparameter tuning for XGBoost (our strongest baseline)
# ---------------------------------------------------------------
def tune_xgboost(preprocessor, X_train, y_train, X_test, y_test):
    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", xgb.XGBClassifier(scale_pos_weight=fraud_ratio, random_state=RANDOM_STATE)),
        ]
    )

    param_grid = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [4, 6, 8, 10],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_grid,
        n_iter=20,
        scoring="average_precision",  # PR-AUC, our main metric
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_pipe = search.best_estimator_
    pred, proba = best_pipe.predict(X_test), best_pipe.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, pred, proba)

    tuning_results = pd.DataFrame(search.cv_results_).sort_values("mean_test_score", ascending=False)

    print("Best parameters:", search.best_params_)
    print("Tuned XGBoost test metrics:", test_metrics)

    return best_pipe, tuning_results


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading data...")
    train_df, test_df = load_data()

    X_train, y_train = train_df.drop(columns=["Class"]), train_df["Class"]
    X_test, y_test = test_df.drop(columns=["Class"]), test_df["Class"]

    preprocessor = build_preprocessor(train_df)

    print("\nTraining baseline models (Logistic Regression, Random Forest, XGBoost)...")
    baseline_results, fitted_pipelines = train_baseline_models(preprocessor, X_train, y_train, X_test, y_test)
    print(baseline_results)

    print("\nRunning class imbalance experiment...")
    imbalance_results = imbalance_experiment(preprocessor, train_df, X_train, y_train, X_test, y_test)
    print(imbalance_results)

    print("\nTuning XGBoost hyperparameters (this can take a few minutes)...")
    best_xgb_pipe, tuning_results = tune_xgboost(preprocessor, X_train, y_train, X_test, y_test)

    print("\nSaving models to", MODEL_DIR)
    joblib.dump(fitted_pipelines["Logistic Regression"], f"{MODEL_DIR}/logistic_regression.pkl")
    joblib.dump(fitted_pipelines["Random Forest"], f"{MODEL_DIR}/random_forest.pkl")
    joblib.dump(best_xgb_pipe, f"{MODEL_DIR}/xgboost.pkl")  # tuned version is our final XGBoost

    print("Saving results to", RESULTS_DIR)
    baseline_results.to_csv(f"{RESULTS_DIR}/classical_models.csv", index=False)
    imbalance_results.to_csv(f"{RESULTS_DIR}/imbalance_experiment.csv", index=False)
    tuning_results.to_csv(f"{RESULTS_DIR}/xgboost_tuning.csv", index=False)

    print("\nDone! Everything is saved and ready to hand off to Member 4.")


if __name__ == "__main__":
    main()
