# ------------------------------
# BUILDER STAGE
# ------------------------------
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install system libraries needed for Python packages (e.g., OpenCV, NumPy, Ultralytics)
# and basic build tools (in case a package compiles from source)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgomp1 \
        curl \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv (fast package installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first → layer caching
COPY requirements.txt pyproject.toml ./

# Create virtual environment and install dependencies
# Retry once on failure to handle transient network errors
RUN uv venv /app/.venv && \
    (uv pip install --python /app/.venv/bin/python --no-cache -r requirements.txt || \
     (sleep 5 && uv pip install --python /app/.venv/bin/python --no-cache -r requirements.txt))

# ------------------------------
# RUNNER STAGE
# ------------------------------
FROM python:3.11-slim-bookworm AS runner

WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GIT_PYTHON_REFRESH=quiet \
    MLFLOW_TRACKING_URI=file:///app/mlruns \
    MLFLOW_ALLOW_FILE_STORE=true \
    PATH="/app/.venv/bin:$PATH"

# Install runtime system dependencies
# --fix-missing helps when a package is temporarily unavailable
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        curl \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Create non‑root user
RUN useradd --create-home --uid 1000 appuser

# Create writable directories for Ultralytics and MLflow (file store)
RUN mkdir -p /home/appuser/.config/Ultralytics /app/mlruns && \
    chown -R appuser:appuser /home/appuser/.config /app/mlruns

# Copy application code with correct ownership
COPY --chown=appuser:appuser . .

# Switch to non‑root user
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--timeout-keep-alive", "65", \
     "--log-level", "warning"]