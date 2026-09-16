"""
app/model_loader.py

Streamlit's caching layer over src.prediction.predictor.

The loading logic itself lives in src/ so the command line and the app
share it. All this file adds is @st.cache_resource, so a 3 MB pickle
isn't re-read on every click and switching between models is instant.
"""

import config  # noqa: F401  — sets up sys.path for the src imports below
import streamlit as st

from src.prediction.predictor import (
    FraudPredictor,
    available_models as _available_models,
    missing_reason as _missing_reason,
)


@st.cache_resource(show_spinner="Loading model...")
def load_predictor(name: str) -> FraudPredictor:
    """Load and cache one model by name."""
    return FraudPredictor.load(name)


@st.cache_data(show_spinner=False)
def available_models() -> list[str]:
    """Models that can actually be loaded in this environment."""
    return _available_models()


@st.cache_data(show_spinner=False)
def blocked_models() -> dict[str, str]:
    """Models that can't be loaded, mapped to the reason why."""
    from src.utils.config import MODEL_REGISTRY

    return {
        name: reason
        for name in MODEL_REGISTRY
        if (reason := _missing_reason(name)) is not None
    }
