import pandas as pd

from src.data.data_pipeline import (
    load_processed_data,
    validate_data,
    preprocess_data
)


def test_load_processed_data():

    train, val, test = load_processed_data()

    assert not train.empty
    assert not val.empty
    assert not test.empty


def test_required_columns():

    train, val, test = load_processed_data()

    required_columns = ["Time", "Amount", "Class"]

    for df in [train, val, test]:

        for column in required_columns:
            assert column in df.columns


def test_validate_data():

    train, val, test = load_processed_data()

    result = validate_data(train, val, test)

    assert result is True


def test_preprocess_data():

    train, val, test = load_processed_data()

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        train,
        val,
        test
    )

    assert "Class" not in X_train.columns
    assert "Class" not in X_val.columns
    assert "Class" not in X_test.columns

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)


def test_target_values():

    train, val, test = load_processed_data()

    for df in [train, val, test]:

        assert set(df["Class"].unique()).issubset({0, 1})