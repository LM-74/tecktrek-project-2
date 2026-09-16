"""
app/app.py

Streamlit front-end for the fraud detection project.

Run from the project root:

    streamlit run app/app.py

All the logic lives in src/ — this file is layout and widgets only.
"""

import pandas as pd
import streamlit as st

from config import (
    APP_TITLE,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    MODEL_REGISTRY,
    REPORTS_DIR,
    THRESHOLD_RANGES,
)
from inference import (
    SchemaError,
    alert_summary,
    blank_transaction,
    cost_summary,
    read_prediction_log,
    score_batch,
    score_one,
    sweep_thresholds,
)
from model_loader import available_models, blocked_models, load_predictor
from src.prediction.predictor import ModelUnavailableError
from src.data.validation import V_COLUMNS

st.set_page_config(page_title=APP_TITLE, page_icon="🛡️", layout="wide")


# ---------------------------------------------------------------
# Sidebar: pick a model and a decision threshold
# ---------------------------------------------------------------
def sidebar():
    st.sidebar.title("Model")

    usable = available_models()
    if not usable:
        st.sidebar.error("No models found. Run the training scripts or notebooks first.")
        st.stop()

    name = st.sidebar.selectbox("Scoring model", usable)
    cfg = MODEL_REGISTRY[name]

    try:
        predictor = load_predictor(name)
    except ModelUnavailableError as exc:
        # Shouldn't happen — available_models() filters these out — but if a
        # pickle needs a package we didn't anticipate, show the fix rather
        # than a traceback.
        st.sidebar.error(str(exc))
        st.stop()

    st.sidebar.caption(cfg["note"])

    lo, hi, step = THRESHOLD_RANGES[predictor.score_type]
    default = (
        cfg["default_threshold"] if predictor.kind == "classical" else predictor.tuned_threshold
    )
    default = float(min(max(default, lo), hi))

    label = (
        "Decision threshold (probability)"
        if predictor.score_type == "probability"
        else "Decision threshold (reconstruction error)"
    )
    threshold = st.sidebar.slider(
        label, min_value=lo, max_value=hi, value=default, step=step, key=f"thr_{name}"
    )

    if predictor.kind != "classical":
        st.sidebar.caption(f"Tuned threshold from training: {predictor.tuned_threshold:.4f}")

    blocked = blocked_models()
    if blocked:
        with st.sidebar.expander(f"Unavailable models ({len(blocked)})"):
            for n, why in blocked.items():
                st.write(f"**{n}** — {why}")
            st.caption("`pip install -r requirements.txt` installs everything.")

    return predictor, threshold


# ---------------------------------------------------------------
# Shared rendering
# ---------------------------------------------------------------
def show_results(results: pd.DataFrame, score_type: str):
    summary = alert_summary(results)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{summary['transactions']:,}")
    c2.metric("Alerts raised", f"{summary['alerts']:,}")
    c3.metric("Alert rate", f"{summary['alert_rate']:.2%}")
    c4.metric("Amount flagged", f"{summary['amount_flagged']:,.2f}")

    costs = cost_summary(results)
    if costs:
        st.subheader("Against the labels in your file")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Frauds caught", costs["tp"])
        d2.metric("Frauds missed", costs["fn"])
        d3.metric("Recall", f"{costs['recall']:.3f}")
        d4.metric("Precision", f"{costs['precision']:.3f}")
        st.caption(
            f"Estimated cost at this threshold: **{costs['cost']:,.0f}** "
            f"(missed fraud {COST_FALSE_NEGATIVE:,.0f} each, "
            f"false alarm {COST_FALSE_POSITIVE:,.0f} each). "
            "These are our own assumptions, not real figures."
        )

        sweep = sweep_thresholds(results)
        if sweep is not None:
            with st.expander("Threshold trade-off"):
                st.dataframe(sweep, width="stretch", hide_index=True)

    st.subheader("Highest scoring transactions")
    st.dataframe(
        results.sort_values("score", ascending=False).head(50),
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "Download all scored rows (CSV)",
        results.to_csv(index=False).encode("utf-8"),
        file_name="scored_transactions.csv",
        mime="text/csv",
    )

    if score_type == "anomaly_score":
        st.caption(
            "The autoencoder returns a reconstruction error, not a probability. "
            "A higher score only means the transaction looks unusual."
        )


# ---------------------------------------------------------------
# Tab 1 — one transaction
# ---------------------------------------------------------------
def tab_single(predictor, threshold):
    st.subheader("Score a single transaction")
    st.caption(
        "V1-V28 are anonymised PCA components from the source dataset, so they "
        "carry no business meaning. Defaults are 0 (an average transaction)."
    )

    defaults = blank_transaction()
    values = {}

    c1, c2 = st.columns(2)
    values["Time"] = c1.number_input(
        "Time (seconds since the first transaction)", value=defaults["Time"], step=60.0
    )
    values["Amount"] = c2.number_input("Amount", value=defaults["Amount"], step=10.0)

    with st.expander("PCA components (V1-V28)"):
        cols = st.columns(4)
        for i, v in enumerate(V_COLUMNS):
            values[v] = cols[i % 4].number_input(v, value=defaults[v], step=0.1, format="%.4f")

    if st.button("Score transaction", type="primary"):
        score, flagged = score_one(predictor, values, threshold)

        label = "Fraud probability" if predictor.score_type == "probability" else "Anomaly score"
        m1, m2 = st.columns(2)
        m1.metric(label, f"{score:.4f}")
        m2.metric("Decision", "Flag for review" if flagged else "Approve")

        if flagged:
            st.error("Above the threshold — send to a human reviewer.")
        else:
            st.success("Below the threshold — no alert.")

        st.caption(
            "Decision support for review triage. Not an automated block, and "
            "not an accusation of fraud."
        )


# ---------------------------------------------------------------
# Tab 2 — CSV batch
# ---------------------------------------------------------------
def tab_batch(predictor, threshold):
    st.subheader("Score a CSV file")
    st.caption(
        "Upload transactions in the raw creditcard.csv layout (Time, V1-V28, "
        "Amount). A Class column, if present, is used to evaluate the run "
        "rather than scored as an input."
    )

    uploaded = st.file_uploader("Transactions CSV", type=["csv"])
    if uploaded is None:
        st.info("Waiting for a file. data/processed/model_test.csv works well for a demo.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read that file: {exc}")
        return

    st.write(f"Loaded **{len(df):,}** rows.")
    max_rows = st.number_input(
        "Rows to score (keeps the demo fast)",
        min_value=1,
        max_value=len(df),
        value=min(5000, len(df)),
        step=500,
    )

    if st.button("Run batch scoring", type="primary"):
        try:
            with st.spinner("Scoring..."):
                results, summary = score_batch(
                    predictor, df.head(int(max_rows)), threshold, source="batch"
                )
        except SchemaError as exc:
            st.error(str(exc))
            return

        for warning in summary["warnings"]:
            st.warning(warning)

        show_results(results, predictor.score_type)


# ---------------------------------------------------------------
# Tab 3 — training results
# ---------------------------------------------------------------
def _show_csv(filename: str, caption: str):
    path = REPORTS_DIR / filename

    if not path.exists():
        st.warning(f"Results file not found: `{path}`")
        return

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        st.error(f"Could not read `{filename}`: {exc}")
        return

    st.markdown(f"**{caption}**")
    st.dataframe(df, width="stretch", hide_index=True)


def tab_results():
    st.subheader("Model comparison")
    st.caption("Test-set results saved during training. PR-AUC is the headline metric.")

    _show_csv("classical_models.csv", "Classical ML baselines")
    _show_csv("mlp_results.csv", "Deep learning (MLP)")
    _show_csv("autoencoder_results.csv", "Autoencoder (anomaly detection)")
    _show_csv("imbalance_experiment.csv", "Class imbalance experiment (XGBoost)")

    c1, c2 = st.columns(2)
    for col, img, cap in [
        (c1, "mlp_training_curves.png", "MLP training curves"),
        (c2, "mlp_confusion_matrix.png", "MLP confusion matrix"),
    ]:
        path = REPORTS_DIR / img
        if path.exists():
            col.image(str(path), caption=cap, width="stretch")
    
    s1, s2 = st.columns(2)
    for col, img, cap in [
        (s1, "autoencoder_training_curve.png", "Autoencoder training curves"),
        (s2, "autoencoder_reconstruction_errors.png", "Autoencoder reconstruction errors"),
    ]:
        path = REPORTS_DIR / img
        if path.exists():
            col.image(str(path), caption=cap, width="stretch")

    st.subheader("Recent prediction log")
    log = read_prediction_log()
    if log.empty:
        st.caption("No predictions logged yet.")
    else:
        st.dataframe(log, width="stretch", hide_index=True)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    st.title(APP_TITLE)
    st.caption(
        "Credit card fraud scoring — classical ML, a supervised neural net and "
        "an unsupervised autoencoder, over the ULB / Kaggle dataset."
    )

    predictor, threshold = sidebar()

    t1, t2, t3 = st.tabs(["Single transaction", "Batch scoring", "Model results"])
    with t1:
        tab_single(predictor, threshold)
    with t2:
        tab_batch(predictor, threshold)
    with t3:
        tab_results()


if __name__ == "__main__":
    main()
