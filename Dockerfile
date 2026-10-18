# ─────────────────────────────────────────────────────────────
# Multi-Stage Production Dockerfile for Fraud Detection System
# ─────────────────────────────────────────────────────────────

# Stage 1: Dependency Builder
FROM python:3.11-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --user -r requirements.txt


# Stage 2: Minimal Runtime
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH=/home/appuser/.local/bin:$PATH \
    API_URL="http://127.0.0.1:8000"

RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Security: Create and run as non-root user
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -m -s /bin/bash appuser

# Copy installed Python packages from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application code and configurations
COPY --chown=appuser:appuser . .

# Setup logging & state directories with proper ownership
RUN mkdir -p /tmp /app/logs /app/models/registry /app/docs/figures && \
    chown -R appuser:appuser /tmp /app

USER appuser

# Expose Streamlit frontend (Hugging Face default) and FastAPI backend
EXPOSE 7860 8000

# Healthcheck probe verifying internal backend readiness
HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

# Launch both FastAPI and Streamlit concurrently via Supervisord
CMD ["supervisord", "-c", "/app/app/supervisord.conf"]
