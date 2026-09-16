# ---------------------------------------------------------------
# Fraud Detection — Streamlit application
#
# Build from the project root:
#     docker build -t fraud-detection .
#
# Run:
#     docker run -p 8501:8501 fraud-detection
#
# Persist the prediction log on the host:
#     docker run -p 8501:8501 -v "$(pwd)/logs:/app/logs" fraud-detection
# ---------------------------------------------------------------
FROM python:3.11-slim

# Unbuffered output so container logs appear immediately, and no .pyc
# files cluttering the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONPATH=/app

WORKDIR /app

# curl is only needed for the healthcheck below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so this layer is cached when only code changes.
# This installs xgboost and tensorflow-cpu, so all five models work.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code, the trained artifacts and the saved result tables.
# The raw dataset is deliberately not copied — it is large and gitignored.
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY reports/ ./reports/

# Fail the build here, with a clear message, if the layout is ever wrong —
# rather than failing later at `docker run` with a bare ModuleNotFoundError.
RUN python -c "import src.prediction.predictor; print('src package import OK')"

# Prediction log lives here; mount a volume to keep it between runs.
RUN mkdir -p logs

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]