"""Unit tests for FastAPI endpoints, request validation, and error responses."""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from imblearn.pipeline import Pipeline as ImbPipeline

from app.api.main import app, FEATURE_COLUMNS
from src.artifacts.save import ModelArtifacts


@pytest.fixture
def dummy_artifacts() -> ModelArtifacts:
    """Create lightweight dummy fitted pipeline for fast API testing."""
    rng = np.random.default_rng(42)
    n = 50
    X = rng.normal(0, 1, size=(n, len(FEATURE_COLUMNS)))
    y = rng.integers(0, 2, size=n)

    pipe = ImbPipeline([
        ("scaler", RobustScaler()),
        ("classifier", LogisticRegression()),
    ])
    pipe.fit(X, y)

    return ModelArtifacts(
        pipeline=pipe,
        calibrator=None,
        threshold=0.35,
        metadata={"version": "test-1.0.0"},
        feature_schema={"columns": FEATURE_COLUMNS},
        explainer=None,
    )


@pytest.fixture
def client(dummy_artifacts: ModelArtifacts) -> TestClient:
    """Initialize FastAPI TestClient with dummy artifacts injected into state."""
    app.state.artifacts = dummy_artifacts
    app.state.settings = type("DummySettings", (), {
        "serving": type("Serving", (), {"max_upload_bytes": 10485760})()
    })()
    return TestClient(app)


@pytest.fixture
def valid_transaction() -> dict[str, float]:
    """Sample valid transaction dictionary."""
    txn = {f"V{i}": 0.05 for i in range(1, 29)}
    txn["Time"] = 12000.0
    txn["Amount"] = 99.50
    return txn


def test_health_endpoint(client: TestClient) -> None:
    """Verify /health reports healthy status and model version."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_version"] == "test-1.0.0"
    assert data["operating_threshold"] == 0.35


def test_predict_single_endpoint(client: TestClient, valid_transaction: dict[str, float]) -> None:
    """Verify /predict returns valid probability and decision."""
    payload = {"transaction": valid_transaction}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "fraud_probability" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["decision"] in (0, 1)
    assert data["operating_threshold"] == 0.35


def test_predict_threshold_override(client: TestClient, valid_transaction: dict[str, float]) -> None:
    """Verify /predict respects client override_threshold."""
    payload = {"transaction": valid_transaction, "override_threshold": 0.85}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["operating_threshold"] == 0.85


def test_predict_validation_rejects_negative_amount(client: TestClient, valid_transaction: dict[str, float]) -> None:
    """Ensure negative amounts trigger 422 Unprocessable Entity."""
    bad_txn = valid_transaction.copy()
    bad_txn["Amount"] = -25.0
    resp = client.post("/predict", json={"transaction": bad_txn})
    assert resp.status_code == 422


def test_predict_validation_rejects_nan_values(client: TestClient, valid_transaction: dict[str, float]) -> None:
    """Ensure NaN float values are explicitly rejected by validator."""
    bad_txn = valid_transaction.copy()
    bad_txn["V4"] = float("nan")
    resp = client.post("/predict", json={"transaction": bad_txn})
    assert resp.status_code == 422


def test_predict_batch_endpoint(client: TestClient, valid_transaction: dict[str, float]) -> None:
    """Verify /predict_batch handles multiple transaction records."""
    payload = {
        "transactions": [valid_transaction, valid_transaction],
        "override_threshold": 0.40,
    }
    resp = client.post("/predict_batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_processed"] == 2
    assert len(data["results"]) == 2


def test_predict_batch_csv_endpoint(client: TestClient, valid_transaction: dict[str, float]) -> None:
    """Verify /predict_batch_csv parses multipart CSV files correctly."""
    df = pd.DataFrame([valid_transaction, valid_transaction])
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    files = {"file": ("test_transactions.csv", csv_bytes, "text/csv")}
    resp = client.post("/predict_batch_csv", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_processed"] == 2
