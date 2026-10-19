"""Contract and edge-case unit tests for the FastAPI fraud detection service."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from app.api.main import app, FEATURE_COLUMNS
from src.artifacts.save import ModelArtifacts


@pytest.fixture
def mock_client() -> TestClient:
    """Fixture providing TestClient with active dummy model artifacts."""
    rng = np.random.default_rng(42)
    n = 40
    X = rng.normal(0, 1, size=(n, len(FEATURE_COLUMNS)))
    y = rng.integers(0, 2, size=n)

    pipe = ImbPipeline(
        [
            ("scaler", RobustScaler()),
            ("classifier", LogisticRegression()),
        ]
    )
    pipe.fit(X, y)

    artifacts = ModelArtifacts(
        pipeline=pipe,
        calibrator=None,
        threshold=0.25,
        metadata={"version": "1.0.0-audit"},
        feature_schema={"columns": FEATURE_COLUMNS},
        explainer=None,
    )

    app.state.artifacts = artifacts
    app.state.settings = type(
        "MockSettings",
        (),
        {
            "serving": type(
                "Serving",
                (),
                {
                    "max_upload_bytes": 10485760,
                    "max_batch_rows": 10000,
                },
            )()
        },
    )()

    return TestClient(app)


def test_empty_csv_upload_returns_422(mock_client: TestClient) -> None:
    """Uploading completely empty CSV file must return 422 Unprocessable Entity."""
    empty_bytes = b""
    files = {"file": ("empty.csv", empty_bytes, "text/csv")}
    resp = mock_client.post("/predict_batch_csv", files=files)
    assert resp.status_code == 422


def test_invalid_mime_type_returns_415(mock_client: TestClient) -> None:
    """Uploading executable or binary file must return 415 Unsupported Media Type."""
    files = {"file": ("malware.exe", b"MZ...", "application/octet-stream")}
    resp = mock_client.post("/predict_batch_csv", files=files)
    assert resp.status_code == 415


def test_csv_with_missing_columns_returns_422(mock_client: TestClient) -> None:
    """Uploading CSV missing required PCA components must return 422."""
    df = pd.DataFrame({"Time": [10.0], "Amount": [50.0], "V1": [0.1]})
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("bad_schema.csv", csv_bytes, "text/csv")}
    resp = mock_client.post("/predict_batch_csv", files=files)
    assert resp.status_code == 422


def test_csv_with_negative_amount_returns_422(mock_client: TestClient) -> None:
    """Uploading CSV with negative transaction amounts must return 422."""
    data = {c: [0.1] for c in FEATURE_COLUMNS}
    data["Amount"] = [-150.0]
    df = pd.DataFrame(data)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("negative_amount.csv", csv_bytes, "text/csv")}
    resp = mock_client.post("/predict_batch_csv", files=files)
    assert resp.status_code == 422


def test_csv_with_nan_values_returns_422(mock_client: TestClient) -> None:
    """Uploading CSV with NaN or string values in numeric columns must return 422."""
    data = {c: [0.1] for c in FEATURE_COLUMNS}
    data["V14"] = [float("nan")]
    df = pd.DataFrame(data)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("nan_values.csv", csv_bytes, "text/csv")}
    resp = mock_client.post("/predict_batch_csv", files=files)
    assert resp.status_code == 422


def test_single_row_batch_prediction(mock_client: TestClient) -> None:
    """Single-item batch request must return valid BatchPredictResponse with total_processed=1."""
    txn = {c: 0.05 for c in FEATURE_COLUMNS}
    payload = {"transactions": [txn], "override_threshold": 0.50}
    resp = mock_client.post("/predict_batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_processed"] == 1
    assert len(data["results"]) == 1


def test_boundary_thresholds(mock_client: TestClient) -> None:
    """Verify system accepts boundary thresholds 0.01 and 0.99."""
    txn = {c: 0.05 for c in FEATURE_COLUMNS}
    for t in (0.01, 0.99):
        resp = mock_client.post(
            "/predict", json={"transaction": txn, "override_threshold": t}
        )
        assert resp.status_code == 200
        assert resp.json()["operating_threshold"] == t
