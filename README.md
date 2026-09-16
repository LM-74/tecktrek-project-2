# Real-Time Credit Card Fraud Detection

A fraud-scoring system built on the ULB / Kaggle credit card dataset. It
compares classical machine learning, a supervised neural network and an
unsupervised autoencoder, and exposes all three through a Streamlit
interface that scores single transactions or CSV batches.

TechTrek Advanced Data Science & AI — Graduation Project 2.

> **Status key.** Sections marked **Built** are implemented and runnable.
> Sections marked **Planned** are part of the project specification but
> not finished yet. This distinction is deliberate: the README should not
> claim more than the repository delivers.

---

## 1. Problem

Detect fraudulent card transactions in a stream where fraud is rare and
the two kinds of mistake cost wildly different amounts.

Two facts drive every design decision:

- **Fraud is a rare event.** 492 frauds in 284,807 transactions, about
  0.17%. A model that predicts "never fraud" is 99.83% accurate and
  completely worthless, so accuracy is never used as a headline metric.
- **False negatives cost far more than false positives.** A missed fraud
  is a real financial loss; a false alarm costs a few minutes of an
  analyst's time. The decision threshold has to reflect that asymmetry
  rather than sitting at the default 0.5.

The system is decision support for a review queue. It ranks transactions
for human attention — it does not block cards and does not accuse anyone
of fraud.

---

## 2. Dataset

**Credit Card Fraud Detection** — ULB Machine Learning Group, via Kaggle:
https://www.kaggle.com/mlg-ulb/creditcardfraud

| Property | Value |
|---|---|
| Transactions | 284,807 |
| Frauds | 492 (0.172%) |
| Period | Two days, September 2013, European cardholders |
| Features | `Time`, `V1`–`V28`, `Amount`, `Class` |

`V1`–`V28` are the output of a PCA transformation applied by the dataset
authors to protect confidentiality. They have no business meaning, and
the project never pretends otherwise — "V14 is low" cannot be explained
to a customer. `Time` (seconds since the first transaction) and `Amount`
are the only two raw, interpretable columns.

The dataset is licensed for academic use. It is not redistributed in this
repository; see [Dataset Setup](#22-dataset-setup).

---

## 3. Business Context

A fraud team cannot review everything. The model's job is to fill a
review queue that is small enough to work through and rich enough in real
fraud to be worth working through.

That makes two numbers matter more than any leaderboard metric:

- **Alert volume.** A model flagging 8% of traffic generates roughly
  22,000 reviews over this dataset and would be switched off within a
  week, whatever its recall.
- **Cost at the chosen threshold.** Assumptions used throughout, defined
  in `src/utils/config.py`:

  ```python
  COST_FALSE_NEGATIVE = 500.0   # money lost when a fraud slips through
  COST_FALSE_POSITIVE = 5.0     # analyst time spent on a false alarm
  ```

  These are our own stated assumptions for the simulation, not figures
  from a real institution. They are adjustable in one place, and every
  cost number in the app and the reports flows from them.

---

## 4. Architecture

```
  Kaggle CSV
      │
      ▼
  ┌─────────────────┐     ┌──────────────────┐
  │ src/data        │────▶│ src/features     │
  │ load + validate │     │ engineer 36 cols │
  └─────────────────┘     └────────┬─────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
  ┌───────────────┐      ┌──────────────────┐     ┌──────────────────┐
  │ train_ml.py   │      │ train_mlp.py     │     │ train_autoenc.py │
  │ LR / RF / XGB │      │ supervised MLP   │     │ unsupervised AE  │
  └───────┬───────┘      └────────┬─────────┘     └────────┬─────────┘
          └───────────────────────┼────────────────────────┘
                                  ▼
                        models/  +  reports/model_results/
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ src/prediction         │
                     │ FraudPredictor         │
                     └───────────┬────────────┘
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
      ┌──────────────────┐             ┌────────────────────┐
      │ app/  Streamlit  │             │ batch_predict CLI  │
      └────────┬─────────┘             └─────────┬──────────┘
               └────────► logs/predictions.csv ◄─┘
```

The key property: **the app and the CLI score through the same
`FraudPredictor`**. There is no second copy of the preprocessing logic, so
a prediction made in the dashboard is identical to one made in a batch
job. Nothing under `src/` imports Streamlit.

---

## 5. Repository Structure

```
├── app/                          # Streamlit UI (widgets and layout only)
│   ├── app.py                    # entry point
│   ├── config.py                 # path bootstrap + re-exports
│   ├── model_loader.py           # Streamlit caching over the predictor
│   └── inference.py              # thin bridge to src/
│
├── src/
│   ├── data/
│   │   ├── load_data.py          # raw / processed / uploaded CSV loading
│   │   └── validation.py         # schema checks, readable errors
│   ├── features/
│   │   └── feature_engineering.py  # single source of truth for features
│   ├── models/
│   │   ├── train_ml.py           # LR / RF / XGBoost + tuning
│   │   ├── train_mlp.py          # supervised neural network
│   │   └── train_autoencoder.py  # unsupervised anomaly detection
│   ├── evaluation/
│   │   ├── metrics.py            # shared metric vocabulary
│   │   ├── threshold.py          # F1- and cost-optimal thresholds
│   │   └── error_analysis.py     # which frauds we miss, and why
│   ├── prediction/
│   │   ├── predictor.py          # load artifacts, score transactions
│   │   └── batch_predict.py      # CSV scoring + CLI
│   └── utils/
│       ├── config.py             # every path and constant
│       └── logger.py             # console + prediction logging
│
├── notebooks/                    # 01 EDA → 05 anomaly detection
├── models/                       # saved artifacts (committed, small)
├── reports/model_results/        # metric tables and figures
├── data/                         # gitignored, see Dataset Setup
├── requirements.txt
├── Dockerfile
└── README.md
```

`src/models/__init__.py` is intentionally free of imports, so touching
anything under `src.models` does not pull TensorFlow into memory for code
that only needs the classical models.

---

## 6. Data Pipeline

**Built.** Raw → cleaned → feature-ready, reproducible end to end.

1. **Load** (`notebooks/01_loading_and_eda.ipynb`) — read the raw CSV,
   inspect class balance, distributions and missingness.
2. **Split chronologically.** `Time` is ordered, so a random split would
   let the model learn from future transactions to predict past ones.
   Splitting by time is the only honest option for a system that claims
   streaming behaviour.
3. **Engineer features** (`notebooks/02_feature_engineering.ipynb`) —
   writes `model_train.csv`, `model_val.csv`, `model_test.csv`.
4. **Train** — three scripts, each writing to `models/` and
   `reports/model_results/`.

Preprocessors are **fit on train only** and applied to validation and
test. Fitting a scaler on the full dataset leaks test statistics into
training and inflates every downstream number.

---

## 7. Feature Engineering

**Built.** `src/features/feature_engineering.py`.

The 30 raw columns become 36. Everything added is derived from `Time` and
`Amount`, the only two interpretable fields:

| Feature | Rationale |
|---|---|
| `Hour` | Hour of day, 0–23 |
| `Day` | Day index (this dataset covers only two days) |
| `Hour_sin`, `Hour_cos` | Cyclical encoding, so 23:00 and 00:00 are adjacent rather than maximally far apart |
| `Time_of_day` | Morning / Afternoon / Evening / Night, one-hot encoded |
| `amount_log` | `log1p(Amount)` — the raw amount is heavily right-skewed |

This module is the **single source of truth**. Notebooks, training
scripts and the live app all call `create_features()`, so training-serving
skew is structurally impossible: there is only one implementation.

`prepare_for_model()` validates, engineers, and returns exactly the 36
columns in the exact order the saved preprocessors expect — dropping
everything else, including `Class`, which must never reach a model as an
input.

---

## 8. Classical ML

**Built.** `src/models/train_ml.py`, `notebooks/03_ml_models.ipynb`.

Logistic Regression as the baseline, then Random Forest and XGBoost.
Stratified cross-validation, a full sklearn `Pipeline` so preprocessing
travels with the model, and `RandomizedSearchCV` on XGBoost scored by
average precision (PR-AUC), not accuracy.

**Class imbalance experiment** — three strategies on XGBoost:

| Strategy | Recall | Precision | F1 | PR-AUC |
|---|---|---|---|---|
| No handling | 0.5192 | 0.7500 | 0.6136 | 0.5295 |
| Class weighting | 0.7500 | 0.8667 | 0.8041 | **0.7574** |
| Random under-sampling | **0.9038** | 0.0177 | 0.0348 | 0.6858 |

Under-sampling wins on recall and is unusable: 1.8% precision means about
55 false alarms for every real fraud found. Class weighting was adopted.
This is exactly why recall alone never decides anything here.

---

## 9. Deep Learning

**Built.** `src/models/train_mlp.py`, `notebooks/04_deep_learning.ipynb`.

A dense network (64 → 32 → 16, ReLU, dropout 0.3, sigmoid output) on the
36 engineered features. Adam at 1e-3, class weighting for the imbalance,
early stopping on validation PR-AUC with best-weight restoration, and
`ReduceLROnPlateau` for learning-rate scheduling.

The threshold is chosen by sweeping the precision-recall curve **on
validation** and taking maximum F1, then applied once to test. Tuning a
threshold on test is a form of leakage that quietly inflates results.

```bash
python -m src.models.train_mlp
```

---

## 10. Anomaly Detection

**Built.** `src/models/train_autoencoder.py`,
`notebooks/05_anomaly_detection.ipynb`.

A symmetric autoencoder (32 → 16 → 8 bottleneck → 16 → 32) trained
**only on normal transactions** — fraud rows are excluded from both
training and the early-stopping validation set. It learns to reconstruct
legitimate behaviour well and anything unusual badly, so reconstruction
error becomes the anomaly score.

Two design notes:

- `Time`, `Hour`, `Day`, `Time_of_day` and `Amount` are excluded. The
  chronological split shifts them between train and test, and the model
  would read that drift as "anomalous" for every late transaction.
- The output is **not a probability**. A score of 12 does not mean 12%
  anything. The app labels it separately and gives it its own threshold
  scale so the two are never confused.

The value of this model is that it needs no labels, so it can flag
patterns nobody has labelled yet.

---

## 11. Evaluation

**Built.** `src/evaluation/`.

**PR-AUC is the headline metric.** With 0.17% positives, ROC-AUC is
flattered by the enormous true-negative count; precision-recall is the
honest view.

Test-set results:

| Model | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8846 | 0.0278 | 0.0538 | 0.9234 | 0.7233 |
| Random Forest | 0.7500 | 0.8478 | 0.7959 | 0.8749 | **0.7787** |
| XGBoost (tuned) | 0.7308 | 0.9048 | 0.8085 | 0.8653 | 0.7595 |
| MLP | 0.7308 | 0.9268 | **0.8172** | — | 0.7414 |

Logistic Regression illustrates the trap: the best recall and the best
ROC-AUC in the table, and 2.8% precision — roughly 35 false alarms per
fraud caught. The deep learning model earns its place by producing the
best F1 and precision, but it does **not** beat Random Forest on PR-AUC.
That is a real finding and it is reported as one rather than buried.

`error_analysis.py` goes past the aggregates: it labels every row TP / FP
/ FN / TN, breaks recall down by amount band and time of day, and lists
the largest missed frauds. Missing twenty small frauds and missing two
large ones are different problems.

---

## 12. Cost-Sensitive Decision Making

**Built.** `src/evaluation/threshold.py`.

Three ways to pick a threshold, all available:

- `best_f1_threshold()` — maximum F1 on validation. Used for the saved
  MLP and autoencoder thresholds.
- `best_cost_threshold()` — minimises `FN × 500 + FP × 5`. Because a
  missed fraud costs 100× a false alarm, this lands lower than the
  F1-optimal point: it deliberately accepts more false alarms to catch
  more fraud.
- `threshold_sweep()` — the full trade-off table (precision, recall,
  alert rate, frauds missed, cost) across candidate thresholds. Shown in
  the dashboard, because a trade-off curve is more honest than asserting
  one magic number.

The threshold is a business decision, not a hyperparameter. The app makes
it adjustable and shows the consequences immediately.

---

## 13. Streaming Simulation

**Planned.** Not yet implemented.

The intended design: replay the test set in timestamp order in batches,
score each batch through `FraudPredictor`, and record alert rate and
recall over time to demonstrate concept drift. The temporal split and the
`FraudPredictor` interface are already in place, so this is a
presentation layer over existing components rather than new modelling
work.

An optional Spark Structured Streaming version is listed as a distinction
feature in the specification and is out of scope for the current build.

---

## 14. Streamlit Dashboard

**Built.** `app/`.

```bash
streamlit run app/app.py
```

Three tabs:

1. **Single transaction** — enter `Time`, `Amount` and the 28 PCA
   components; get a score and a decision.
2. **Batch scoring** — upload a CSV, get scored rows, alert-rate metrics
   and a download. If the file contains a `Class` column it is used to
   evaluate the run (confusion counts, recall, precision, cost) rather
   than scored as an input.
3. **Model results** — the saved comparison tables, training curves and
   the recent prediction log.

The sidebar switches between all five models and adjusts the threshold.
The slider automatically changes scale — 0–1 for probabilities, 0–50 for
the autoencoder's reconstruction error — and any model that cannot be
loaded is listed with the reason and the command that fixes it.

---

## 15. Monitoring

**Partially built.**

Built: every scoring request is logged to `logs/predictions.csv` with
timestamp, model, threshold, transaction count, alert count, alert rate
and input schema version — enough to trace any past prediction back to
the artifact that produced it. One row per *request*, not per
transaction, so scoring 50,000 rows does not produce a 50,000-line file.
The recent log is visible in the dashboard.

Planned: a drift-detection page comparing live feature distributions
against the training baseline, and alert-rate trending over time.

---

## 16. Batch Prediction

**Built.** `src/prediction/batch_predict.py`.

```bash
python -m src.prediction.batch_predict \
    --input data/processed/model_test.csv \
    --output reports/scored_test.csv \
    --model "XGBoost (tuned)"
```

Options: `--model` (any registry entry), `--threshold` (defaults to the
one tuned during training), `--limit` (first N rows).

Output columns: `row`, `Time`, `Amount`, `score`, `flagged`, `decision`,
and `actual` when the input carried labels. Every run is written to the
prediction log, exactly like a dashboard request.

---

## 17. SQL

**Planned.** Not yet implemented.

The universal requirements ask for at least 10 analytical queries using
JOIN, CTE and window functions. The intended schema: a `transactions`
table, a `predictions` table (model, threshold, score, timestamp) and a
`model_registry` table, with queries covering fraud rate by hour, rolling
alert volume via window functions, and model-versus-model agreement.

The prediction log is already structured as the seed of the `predictions`
table; the remaining work is a schema plus a loader in `src/data/`.

---

## 18. Model Versioning

**Partially built.**

Built: `MODEL_REGISTRY` in `src/utils/config.py` is the single place
defining every model — its artifact paths, score type, tuned threshold
and required package. Preprocessing and model are saved together as one
sklearn `Pipeline` for the classical models, so they cannot drift apart.
Thresholds are persisted next to their models. `INPUT_SCHEMA_VERSION`
(currently `36-features-v1`) is written into every log row.

Planned: a registry table capturing the training-data hash and the full
metric set per version, and champion/challenger comparison.

---

## 19. Testing

**Planned.** Not yet implemented.

The specification asks for basic data, model and UI tests. The structure
is deliberately test-friendly — `src/` has no Streamlit dependency, so
everything is importable from pytest. Priority cases:

- `validation.py` — missing columns and non-numeric values raise
  `SchemaError` with readable messages.
- `feature_engineering.py` — `create_features()` is idempotent and
  `prepare_for_model()` returns the 36 expected columns in order.
- `predictor.py` — each model loads and returns scores in the expected
  range; `missing_reason()` correctly reports absent packages.
- UI — Streamlit ships `streamlit.testing.v1.AppTest`, which runs the app
  headlessly and asserts no exceptions.

---

## 20. Docker

**Built.**

```bash
docker build -t fraud-detection .
docker run -p 8501:8501 fraud-detection
```

Then open http://localhost:8501.

To keep the prediction log between runs, mount it:

```bash
docker run -p 8501:8501 -v "$(pwd)/logs:/app/logs" fraud-detection
```

The image installs `xgboost` and `tensorflow-cpu`, so all five models work
inside the container. It copies `app/`, `src/`, `models/` and `reports/`,
but not `data/` — the raw dataset is large and not redistributable.

---

## 21. Installation

Requires Python 3.9–3.12.

```bash
git clone https://github.com/LM-74/tecktrek-project-2.git
cd tecktrek-project-2

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app/app.py
```

**`scikit-learn` is pinned to 1.6.1 deliberately.** The saved `.pkl`
pipelines were created with a 1.6.x `ColumnTransformer` and fail to
unpickle on 1.8 with `Can't get attribute '_RemainderColsList'`. Retrain
the models before bumping that pin.

**If a model is missing from the dropdown**, the sidebar's "Unavailable
models" panel names the reason. The usual causes:

| Message | Fix |
|---|---|
| `xgboost not installed` | `pip install xgboost` |
| `tensorflow not installed` | `pip install tensorflow-cpu` |
| `missing file ...` | Run the matching training script |

The app degrades rather than crashing: if TensorFlow is absent, the
classical models still work.

---

## 22. Dataset Setup

The dataset is not committed. Download it, then generate the splits.

1. Get `creditcard.csv` from
   https://www.kaggle.com/mlg-ulb/creditcardfraud (a Kaggle account is
   required) and place it at `data/raw/creditcard.csv`.
2. Run `notebooks/01_loading_and_eda.ipynb`, then
   `notebooks/02_feature_engineering.ipynb`. These write
   `model_train.csv`, `model_val.csv` and `model_test.csv` into
   `data/processed/`.
3. Optionally retrain:

   ```bash
   python -m src.models.train_ml
   python -m src.models.train_mlp
   python -m src.models.train_autoencoder
   ```

   The repository already ships trained artifacts in `models/`, so the
   app runs without this step.

Read the dataset licence before redistributing anything, and cite the
original authors (Dal Pozzolo et al., ULB Machine Learning Group) in the
final report.

---

## 23. Limitations

Stated plainly, because a technical defence is easier with them on the
table than discovered under questioning.

- **This is not a production fraud system** and not a deployed approval
  mechanism. It ranks transactions for human review.
- **The features are not interpretable.** `V1`–`V28` are anonymised PCA
  components. We can say a transaction scored high; we cannot tell a
  customer why in business terms. Only `Time` and `Amount` are real.
- **Two days of data from 2013, one region.** Fraud tactics have moved on
  considerably. Nothing here generalises to current traffic without
  retraining.
- **492 positives in total**, so only a few dozen frauds land in the test
  split. Metrics computed on that few positives carry wide confidence
  intervals, and a handful of transactions moves recall by several
  points.
- **The cost figures are assumptions**, set by us in `config.py`, not
  measured from a real institution. The framework is sound; the specific
  numbers are illustrative.
- **No fairness analysis.** The dataset carries no demographic
  attributes, so proxy-group analysis is not possible here.
- **Streaming, SQL and automated tests are unfinished**, as marked above.

---

## 24. Team

TechTrek Advanced Data Science & AI, Level 2.

| Member | Area |
|---|---|
| Heba | Data pipeline, EDA, classical ML (`train_ml.py`) |
| Lina | Deep learning, supervised MLP (`train_mlp.py`) |
| Baher | Anomaly detection, autoencoder (`train_autoencoder.py`) |
| Loay | Evaluation, deployment, Streamlit app |
| Ghram | SQL, SQL, Data Pipeline & MLOps |
---
