"""Pydantic schemas and interface contracts for the Fraud Detection API.

Provides:
- Transaction feature vector input with physical range and finite-number validations
- Single & batch prediction requests/responses
- Local SHAP explainability responses
- Structured health and error formats
"""

from __future__ import annotations

import math
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class TransactionFeatures(BaseModel):
    """30-dimensional transaction feature payload."""

    Time: float = Field(
        ..., ge=0.0, description="Elapsed seconds since reference transaction"
    )
    Amount: float = Field(
        ..., ge=0.0, le=500000.0, description="Transaction magnitude in USD"
    )
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    @field_validator(
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Time",
        "Amount",
        mode="after",
    )
    @classmethod
    def check_finite(cls, v: float) -> float:
        """Enforce that no NaN or Infinite values enter the ML pipeline."""
        if math.isnan(v) or math.isinf(v):
            raise ValueError(
                "Feature value must be a finite numerical float (NaN and Inf forbidden)."
            )
        return v


class PredictRequest(BaseModel):
    """Payload for single-event real-time scoring."""

    transaction: TransactionFeatures
    override_threshold: Optional[float] = Field(
        None,
        ge=0.01,
        le=0.99,
        description="Optional operational threshold override. If omitted, uses model optimal threshold.",
    )


class PredictResponse(BaseModel):
    """Response returned for a transaction risk evaluation."""

    fraud_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated risk probability"
    )
    decision: int = Field(
        ..., description="1 = Flagged as Fraud, 0 = Approved Legitimate"
    )
    operating_threshold: float = Field(
        ..., description="Decision boundary threshold applied"
    )
    model_version: str = Field(..., description="Active champion model version")
    expected_cost: float = Field(
        ..., description="Estimated expected financial cost of this decision"
    )


class BatchPredictRequest(BaseModel):
    """Batch scoring request for up to 10,000 transactions."""

    transactions: List[TransactionFeatures] = Field(..., max_length=10000)
    override_threshold: Optional[float] = Field(None, ge=0.01, le=0.99)


class BatchPredictResponse(BaseModel):
    """Batch scoring summary and detailed item records."""

    total_processed: int
    flagged_count: int
    flagged_ratio: float
    operating_threshold: float
    model_version: str
    results: List[PredictResponse]


class FeatureAttribution(BaseModel):
    """Per-feature SHAP impact on the prediction score."""

    feature_name: str
    feature_value: float
    shap_value: float
    feature_group: str


class ExplainRequest(BaseModel):
    """Request payload for on-demand local explainability."""

    transaction: TransactionFeatures


class ExplainResponse(BaseModel):
    """Detailed attribution and plain-language audit trail."""

    base_value: float
    prediction_score: float
    fraud_probability: float
    decision: int
    operating_threshold: float
    top_attributions: List[FeatureAttribution]
    summary_text: str


class HealthResponse(BaseModel):
    """Liveness and readiness health probe output."""

    status: str
    model_version: str
    model_loaded: bool
    explainer_loaded: bool
    operating_threshold: float
    timestamp: str


class ErrorResponse(BaseModel):
    """Uniform RFC-7807 compliant error envelope."""

    error_code: str
    message: str
    reference_id: str
