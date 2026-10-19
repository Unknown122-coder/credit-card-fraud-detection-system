"""Security tests verifying cryptographic SHA-256 integrity verification for serialized model artifacts."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.artifacts.save import (
    load_model_artifacts,
    save_model_artifacts,
)


def test_sha256_checksum_verification_detects_tampering(tmp_path: Path) -> None:
    """Verify load_model_artifacts raises PermissionError if artifact bytes are tampered with."""
    clf = LogisticRegression()
    clf.fit(np.array([[1.0], [2.0], [3.0]]), np.array([0, 0, 1]))

    dummy_df = pd.DataFrame({"feat_1": [1.0, 2.0, 3.0]})

    # Save artifact with SHA-256 integrity hash
    _metadata = save_model_artifacts(
        pipeline=clf,
        calibrator=None,
        threshold=0.30,
        metrics={"pr_auc": 0.85},
        X_train=dummy_df,
        output_dir=tmp_path,
    )

    # Clean load must succeed
    artifacts = load_model_artifacts(tmp_path, verify_checksums=True)
    assert artifacts is not None
    assert artifacts.threshold == 0.30

    # Tamper with the pipeline artifact by modifying its byte content
    pipeline_file = tmp_path / "champion_pipeline.joblib"
    with open(pipeline_file, "ab") as f:
        f.write(b"\x00\x00MALICIOUS_PAYLOAD_TAMPER\x00\x00")

    # Tampered load must fail with PermissionError
    with pytest.raises(PermissionError) as exc_info:
        load_model_artifacts(tmp_path, verify_checksums=True)

    assert "tampering detected" in str(exc_info.value).lower()
