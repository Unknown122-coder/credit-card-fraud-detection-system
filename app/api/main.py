"""FastAPI serving application for Credit Card Fraud Detection.

Endpoints:
- GET  /health: Readiness and model status probe
- POST /predict: Low-latency single-transaction scoring (<25ms SLO)
- POST /predict_batch: High-throughput JSON array batch scoring
- POST /predict_batch_csv: Multipart CSV file batch scoring (up to 10MB)
- POST /explain: Local TreeSHAP explanation with risk driver narratives
"""

from __future__ import annotations

import io
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    ExplainRequest,
    ExplainResponse,
    FeatureAttribution,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    TransactionFeatures,
)
from src.artifacts.save import ModelArtifacts, load_model_artifacts
from src.config import Settings, get_settings
from src.explain.shap_utils import compute_local_explanation
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Feature column ordering expected by model
FEATURE_COLUMNS: list[str] = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model artifacts into application state during startup."""
    settings = get_settings()
    logger.info("Initializing fraud detection service lifespan...")

    model_dir = settings.model_dir
    app.state.settings = settings
    app.state.artifacts = None

    try:
        if (model_dir / settings.artifacts.pipeline_filename).exists():
            artifacts = load_model_artifacts(model_dir)
            app.state.artifacts = artifacts
            logger.info(
                "Model artifacts successfully loaded. Champion version: %s, threshold: %.4f",
                artifacts.metadata.get("version", "unknown"),
                artifacts.threshold,
            )

            # Warmup inference
            warmup_df = pd.DataFrame([dict.fromkeys(FEATURE_COLUMNS, 0.0)])
            _ = score_dataframe(warmup_df, artifacts, threshold=0.5)
            logger.info("Warmup inference completed successfully.")
        else:
            logger.warning(
                "Model artifacts not found in %s. Prediction endpoints will return 503 until artifacts exist.",
                model_dir,
            )
    except Exception as exc:
        logger.error("Failed to load model artifacts: %s", exc, exc_info=True)

    yield

    logger.info("Shutting down fraud detection service.")


app = FastAPI(
    title="Credit Card Fraud Detection Service",
    description="High-throughput, explainable real-time risk scoring engine.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.serving.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def audit_logging_middleware(request: Request, call_next: Any) -> Any:
    """Log HTTP requests and execution duration without leaking sensitive request payloads."""
    req_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()

    # Attach request ID to state
    request.state.request_id = req_id

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        "[%s] %s %s - Status: %d - Latency: %.2f ms",
        req_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
    return response


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all unhandled exception handler returning RFC-7807 structured envelope."""
    req_id = getattr(request.state, "request_id", "unknown")
    logger.error("[%s] Unhandled exception: %s", req_id, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            message="An unexpected system error occurred during transaction evaluation.",
            reference_id=req_id,
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Scoring Helper Functions
# ---------------------------------------------------------------------------


def score_dataframe(
    df: pd.DataFrame,
    artifacts: ModelArtifacts,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score a DataFrame of transactions through pipeline and calibrator.

    Parameters
    ----------
    df:
        Feature DataFrame ordered by FEATURE_COLUMNS.
    artifacts:
        Loaded model artifacts.
    threshold:
        Decision boundary threshold.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (probabilities, decisions, expected_costs)
    """
    ordered_df = df[FEATURE_COLUMNS]

    # Model probability
    pipeline = artifacts.pipeline
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(ordered_df)[:, 1]
    else:
        probs = pipeline.predict(ordered_df).astype(float)

    # Apply calibrator if available
    if artifacts.calibrator is not None:
        try:
            # CalibratedClassifierCV expects features or scaled array
            probs = artifacts.calibrator.predict_proba(ordered_df)[:, 1]
        except Exception:
            pass

    decisions = (probs >= threshold).astype(int)

    # Compute expected financial cost per transaction
    # FP: $10.00; FN: Amount + $25.00
    amounts = ordered_df["Amount"].to_numpy()
    fn_cost = amounts + 25.0
    fp_cost = 10.0

    # Risk-weighted expected loss:
    # If decision == 1 (declined): expected cost is (1 - p) * fp_cost
    # If decision == 0 (approved): expected cost is p * fn_cost
    expected_costs = np.where(
        decisions == 1,
        (1.0 - probs) * fp_cost,
        probs * fn_cost,
    )

    return probs, decisions, expected_costs


def features_to_dataframe(features: TransactionFeatures) -> pd.DataFrame:
    """Convert Pydantic TransactionFeatures model to a single-row DataFrame."""
    return pd.DataFrame([features.model_dump()])[FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """Liveness probe reporting model status and active configuration."""
    artifacts: Optional[ModelArtifacts] = getattr(app.state, "artifacts", None)
    is_loaded = artifacts is not None
    explainer_loaded = is_loaded and (artifacts.explainer is not None)

    version = artifacts.metadata.get("version", "none") if is_loaded else "none"
    thresh = artifacts.threshold if is_loaded else 0.5

    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_version=version,
        model_loaded=is_loaded,
        explainer_loaded=explainer_loaded,
        operating_threshold=thresh,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_single(payload: PredictRequest) -> PredictResponse:
    """Evaluate real-time fraud probability and operational action for a single transaction."""
    artifacts: Optional[ModelArtifacts] = getattr(app.state, "artifacts", None)
    if artifacts is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded on this server.",
        )

    thresh = payload.override_threshold if payload.override_threshold is not None else artifacts.threshold

    df = features_to_dataframe(payload.transaction)
    probs, decisions, costs = score_dataframe(df, artifacts, threshold=thresh)

    return PredictResponse(
        fraud_probability=float(probs[0]),
        decision=int(decisions[0]),
        operating_threshold=float(thresh),
        model_version=artifacts.metadata.get("version", "1.0.0"),
        expected_cost=round(float(costs[0]), 2),
    )


@app.post("/predict_batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    """Evaluate batch of up to 10,000 JSON transactions with high vectorized throughput."""
    artifacts: Optional[ModelArtifacts] = getattr(app.state, "artifacts", None)
    if artifacts is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded on this server.",
        )

    thresh = payload.override_threshold if payload.override_threshold is not None else artifacts.threshold

    data_dicts = [t.model_dump() for t in payload.transactions]
    df = pd.DataFrame(data_dicts)[FEATURE_COLUMNS]

    probs, decisions, costs = score_dataframe(df, artifacts, threshold=thresh)
    version = artifacts.metadata.get("version", "1.0.0")

    results: list[PredictResponse] = []
    for p, d, c in zip(probs, decisions, costs):
        results.append(
            PredictResponse(
                fraud_probability=float(p),
                decision=int(d),
                operating_threshold=float(thresh),
                model_version=version,
                expected_cost=round(float(c), 2),
            )
        )

    total = len(results)
    flagged = int(np.sum(decisions))

    return BatchPredictResponse(
        total_processed=total,
        flagged_count=flagged,
        flagged_ratio=round(flagged / max(total, 1), 4),
        operating_threshold=float(thresh),
        model_version=version,
        results=results,
    )


@app.post("/predict_batch_csv", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch_csv(
    file: UploadFile = File(...),
    override_threshold: Optional[float] = None,
) -> BatchPredictResponse:
    """Upload and score a CSV file of transactions (enforcing 10MB upload limit)."""
    settings: Settings = app.state.settings
    artifacts: Optional[ModelArtifacts] = getattr(app.state, "artifacts", None)

    if artifacts is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded on this server.",
        )

    # 1. Validate file size
    contents = await file.read()
    if len(contents) > settings.serving.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded CSV exceeds maximum allowed payload of {settings.serving.max_upload_bytes // (1024*1024)} MB",
        )

    # 2. Parse CSV
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Malformed CSV file: {exc}",
        )

    # 3. Validate columns
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"CSV is missing required feature columns: {missing[:5]} (and {len(missing)-5} more)",
        )

    thresh = override_threshold if override_threshold is not None else artifacts.threshold

    probs, decisions, costs = score_dataframe(df[FEATURE_COLUMNS], artifacts, threshold=thresh)
    version = artifacts.metadata.get("version", "1.0.0")

    results: list[PredictResponse] = []
    for p, d, c in zip(probs, decisions, costs):
        results.append(
            PredictResponse(
                fraud_probability=float(p),
                decision=int(d),
                operating_threshold=float(thresh),
                model_version=version,
                expected_cost=round(float(c), 2),
            )
        )

    total = len(results)
    flagged = int(np.sum(decisions))

    return BatchPredictResponse(
        total_processed=total,
        flagged_count=flagged,
        flagged_ratio=round(flagged / max(total, 1), 4),
        operating_threshold=float(thresh),
        model_version=version,
        results=results,
    )


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
async def explain_transaction(payload: ExplainRequest) -> ExplainResponse:
    """Generate instant local TreeSHAP attribution and audit narrative for a transaction."""
    artifacts: Optional[ModelArtifacts] = getattr(app.state, "artifacts", None)
    if artifacts is None or artifacts.explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explainer artifact is not loaded on this server.",
        )

    df = features_to_dataframe(payload.transaction)
    probs, decisions, _ = score_dataframe(df, artifacts, threshold=artifacts.threshold)

    explanation = compute_local_explanation(artifacts.explainer, df, top_n=10)

    top_attributions = [
        FeatureAttribution(
            feature_name=a["feature_name"],
            feature_value=a["feature_value"],
            shap_value=a["shap_value"],
            feature_group=a["feature_group"],
        )
        for a in explanation["top_attributions"]
    ]

    return ExplainResponse(
        base_value=explanation["base_value"],
        prediction_score=explanation["prediction_score"],
        fraud_probability=float(probs[0]),
        decision=int(decisions[0]),
        operating_threshold=artifacts.threshold,
        top_attributions=top_attributions,
        summary_text=explanation["summary_text"],
    )
