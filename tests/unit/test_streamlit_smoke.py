"""Smoke and component tests for Streamlit UI helpers and security sanitizers."""

from __future__ import annotations

import pandas as pd

from app.ui.streamlit_app import (
    make_attribution_chart,
    make_gauge_chart,
    sanitize_csv_injection,
)


def test_make_gauge_chart_generates_valid_plotly_figure() -> None:
    """Verify gauge chart builds a valid Plotly Figure without exceptions."""
    fig = make_gauge_chart(probability=0.74, threshold=0.28)
    assert fig is not None
    assert "data" in fig
    assert len(fig.data) > 0


def test_make_attribution_chart_generates_valid_bar_chart() -> None:
    """Verify attribution bar chart builds cleanly from mock SHAP dictionary."""
    mock_attributions = [
        {
            "feature_name": "V4",
            "feature_value": 3.2,
            "shap_value": 1.45,
            "feature_group": "Velocity Proxy",
        },
        {
            "feature_name": "V14",
            "feature_value": -4.1,
            "shap_value": -1.25,
            "feature_group": "Terminal Proxy",
        },
        {
            "feature_name": "Amount",
            "feature_value": 482.0,
            "shap_value": 0.85,
            "feature_group": "Magnitude",
        },
    ]
    fig = make_attribution_chart(mock_attributions)
    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].orientation == "h"


def test_sanitize_csv_injection_neutralizes_formulas() -> None:
    """Verify formula injection triggers (=, +, -, @) are prefixed with single quote."""
    dirty_df = pd.DataFrame(
        {
            "Transaction #": [1, 2, 3, 4, 5],
            "Decision": [
                "=SUM(A1:A10)",
                "+cmd|' /C calc'!A0",
                "-2+3",
                "@SUM(1+1)",
                "NORMAL_TEXT",
            ],
            "Loss": ["$10.00", "$20.00", "$30.00", "$40.00", "$50.00"],
        }
    )

    cleaned_df = sanitize_csv_injection(dirty_df)

    # Every formula trigger character must now be escaped with a leading single quote
    assert cleaned_df["Decision"].iloc[0].startswith("'=")
    assert cleaned_df["Decision"].iloc[1].startswith("'+")
    assert cleaned_df["Decision"].iloc[2].startswith("'-")
    assert cleaned_df["Decision"].iloc[3].startswith("'@")
    # Normal text must remain unaltered
    assert cleaned_df["Decision"].iloc[4] == "NORMAL_TEXT"
