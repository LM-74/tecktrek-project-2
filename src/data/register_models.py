from model_registry import register_model, get_registered_models


# Logistic Regression
register_model(
    model_name="Logistic Regression",
    model_version="logreg_v1",
    training_date=None,
    threshold=0.5,
    pr_auc=0.723276
)


# Random Forest
register_model(
    model_name="Random Forest",
    model_version="rf_v1",
    training_date=None,
    threshold=0.5,
    pr_auc=0.778682
)


# XGBoost
register_model(
    model_name="XGBoost",
    model_version="xgb_v1",
    training_date=None,
    threshold=0.5,
    pr_auc=0.759471
)


print("\nRegistered Models:")

models = get_registered_models()

for model in models:
    print(model)