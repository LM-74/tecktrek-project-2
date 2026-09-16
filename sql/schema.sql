CREATE TABLE IF NOT EXISTS transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_time REAL NOT NULL,

    V1 REAL,
    V2 REAL,
    V3 REAL,
    V4 REAL,
    V5 REAL,
    V6 REAL,
    V7 REAL,
    V8 REAL,
    V9 REAL,
    V10 REAL,
    V11 REAL,
    V12 REAL,
    V13 REAL,
    V14 REAL,
    V15 REAL,
    V16 REAL,
    V17 REAL,
    V18 REAL,
    V19 REAL,
    V20 REAL,
    V21 REAL,
    V22 REAL,
    V23 REAL,
    V24 REAL,
    V25 REAL,
    V26 REAL,
    V27 REAL,
    V28 REAL,

    amount REAL NOT NULL,
    actual_fraud INTEGER NOT NULL,
    dataset_split TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS model_versions (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL UNIQUE,
    training_date TEXT,
    threshold REAL,
    pr_auc REAL
);


CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    fraud_probability REAL NOT NULL,
    predicted_class INTEGER NOT NULL,
    threshold REAL,
    prediction_time TEXT NOT NULL,

    FOREIGN KEY (transaction_id)
        REFERENCES transactions(transaction_id),

    FOREIGN KEY (model_id)
        REFERENCES model_versions(model_id)
);


CREATE TABLE IF NOT EXISTS alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id INTEGER NOT NULL,
    prediction_id INTEGER NOT NULL,
    risk_score REAL NOT NULL,
    alert_status TEXT NOT NULL,
    created_at TEXT NOT NULL,

    FOREIGN KEY (transaction_id)
        REFERENCES transactions(transaction_id),

    FOREIGN KEY (prediction_id)
        REFERENCES predictions(prediction_id)
);