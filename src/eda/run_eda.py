"""Exploratory Data Analysis module for Credit Card Fraud Detection.

Performs:
- Class distribution auditing (raw counts, fraud ratio)
- Transaction Amount analysis (descriptives, log-scaling, outliers)
- Time & temporal cyclicality analysis (diurnal patterns, hour-of-day fraud rates)
- Correlation analysis (Spearman rank correlations of V1-V28 with Class)
- Duplicate transaction check
- Generation of publication-quality plots saved to docs/figures/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # Headless backend for automation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import Settings, get_settings
from src.data.load import load_dataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_full_eda(
    df: Optional[pd.DataFrame] = None,
    config: Optional[Settings] = None,
    output_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Execute complete EDA audit and output statistical summaries and figures.

    Parameters
    ----------
    df:
        DataFrame containing ULB credit card transactions. Loaded if None.
    config:
        Settings configuration. Loaded if None.
    output_dir:
        Directory to save generated charts. Defaults to docs/figures.

    Returns
    -------
    dict[str, Any]
        Dictionary of EDA summary statistics.
    """
    if config is None:
        config = get_settings()

    if df is None:
        logger.info("Loading dataset for EDA...")
        df = load_dataset(config)

    save_path = Path(output_dir) if output_dir else config.figures_dir
    save_path.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}

    # 1. Class Distribution
    total_txns = len(df)
    fraud_txns = int(df["Class"].sum())
    legit_txns = total_txns - fraud_txns
    fraud_rate = fraud_txns / total_txns

    summary["class_distribution"] = {
        "total": total_txns,
        "legitimate": legit_txns,
        "fraud": fraud_txns,
        "fraud_rate_percent": round(fraud_rate * 100, 4),
        "imbalance_ratio": f"1:{int(legit_txns / max(fraud_txns, 1))}",
    }
    logger.info("Class distribution: %s", summary["class_distribution"])

    # 2. Duplicate Check
    # Check duplicates ignoring 'Time' as per blueprint
    dup_cols = [c for c in df.columns if c != "Time"]
    duplicates_count = int(df.duplicated(subset=dup_cols).sum())
    summary["duplicate_count_excluding_time"] = duplicates_count
    logger.info("Duplicate transactions (excluding Time): %d", duplicates_count)

    # 3. Amount Statistics
    amount_legit = df.loc[df["Class"] == 0, "Amount"]
    amount_fraud = df.loc[df["Class"] == 1, "Amount"]

    summary["amount_statistics"] = {
        "legit_mean": float(amount_legit.mean()),
        "legit_median": float(amount_legit.median()),
        "legit_std": float(amount_legit.std()),
        "legit_iqr": float(amount_legit.quantile(0.75) - amount_legit.quantile(0.25)),
        "fraud_mean": float(amount_fraud.mean()),
        "fraud_median": float(amount_fraud.median()),
        "fraud_std": float(amount_fraud.std()),
        "fraud_iqr": float(amount_fraud.quantile(0.75) - amount_fraud.quantile(0.25)),
    }

    # 4. Generate Visualizations
    _plot_class_imbalance(
        legit_txns, fraud_txns, save_path / "eda_class_distribution.png"
    )
    _plot_amount_distributions(
        amount_legit, amount_fraud, save_path / "eda_amount_distributions.png"
    )
    _plot_temporal_patterns(df, save_path / "eda_temporal_patterns.png")
    _plot_top_correlations(df, save_path / "eda_correlations_top_features.png")

    logger.info("EDA completed successfully. Figures saved to %s", save_path)
    return summary


def _plot_class_imbalance(legit: int, fraud: int, save_path: Path) -> None:
    """Plot bar chart of class imbalance on log-scale."""
    fig, ax = plt.subplots(figsize=(6, 4))
    classes = ["Legitimate (0)", "Fraud (1)"]
    counts = [legit, fraud]
    colors = ["#2b5c8f", "#d9534f"]

    bars = ax.bar(classes, counts, color=colors, width=0.5)
    ax.set_yscale("log")
    ax.set_title("Transaction Class Distribution (Log Scale)", fontsize=12, pad=12)
    ax.set_ylabel("Transaction Count (log10)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, count in zip(bars, counts):
        pct = (count / (legit + fraud)) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.2,
            f"{count:,}\n({pct:.3f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _plot_amount_distributions(
    legit_amt: pd.Series, fraud_amt: pd.Series, save_path: Path
) -> None:
    """Plot distribution and boxplots of Amount for both classes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Log1p distribution
    sns.kdeplot(
        np.log1p(legit_amt),
        label="Legitimate",
        color="#2b5c8f",
        ax=axes[0],
        fill=True,
        alpha=0.3,
    )
    sns.kdeplot(
        np.log1p(fraud_amt),
        label="Fraud",
        color="#d9534f",
        ax=axes[0],
        fill=True,
        alpha=0.3,
    )
    axes[0].set_title("Log(Amount + 1) Kernel Density Estimate")
    axes[0].set_xlabel("log(Amount + 1)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Boxplot of Amount (clipped for visibility)
    plot_df = pd.DataFrame(
        {
            "Amount": pd.concat([legit_amt, fraud_amt]),
            "Class": ["Legitimate"] * len(legit_amt) + ["Fraud"] * len(fraud_amt),
        }
    )
    sns.boxplot(
        data=plot_df,
        x="Class",
        y="Amount",
        palette=["#2b5c8f", "#d9534f"],
        ax=axes[1],
        showfliers=False,
    )
    axes[1].set_title("Transaction Amount Distribution (Excl. Outliers)")
    axes[1].set_ylabel("Amount (USD)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _plot_temporal_patterns(df: pd.DataFrame, save_path: Path) -> None:
    """Plot fraud occurrence rate by Hour of Day."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Calculate hour of day
    hours = ((df["Time"] % 86400) / 3600).astype(int)
    temp_df = pd.DataFrame({"Hour": hours, "Class": df["Class"]})

    # Total volume by hour
    volume_by_hour = temp_df.groupby("Hour")["Class"].count()
    axes[0].plot(
        volume_by_hour.index, volume_by_hour.values, marker="o", color="#2b5c8f", lw=2
    )
    axes[0].set_title("Hourly Total Transaction Volume")
    axes[0].set_ylabel("Total Transactions")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Fraud rate by hour
    fraud_rate_by_hour = temp_df.groupby("Hour")["Class"].mean() * 100
    axes[1].plot(
        fraud_rate_by_hour.index,
        fraud_rate_by_hour.values,
        marker="s",
        color="#d9534f",
        lw=2,
    )
    axes[1].set_title("Hourly Fraud Rate (%)")
    axes[1].set_xlabel("Hour of Day (0-23)")
    axes[1].set_ylabel("Fraud Percentage (%)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _plot_top_correlations(df: pd.DataFrame, save_path: Path) -> None:
    """Plot Spearman correlation coefficients of features with Class."""
    features = [c for c in df.columns if c != "Class"]
    corrs = df[features].apply(lambda s: s.corr(df["Class"], method="spearman"))
    corrs_sorted = corrs.sort_values()

    # Select top 5 negative and top 5 positive
    top_corrs = pd.concat([corrs_sorted.head(5), corrs_sorted.tail(5)])

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2b5c8f" if v < 0 else "#d9534f" for v in top_corrs.values]
    bars = ax.barh(top_corrs.index, top_corrs.values, color=colors)
    ax.set_title("Top 10 Features Correlated with Fraud (Spearman Rank)", fontsize=11)
    ax.set_xlabel("Correlation Coefficient")
    ax.axvline(0, color="black", linestyle="--", lw=0.8)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, top_corrs.values):
        ax.text(
            val + (0.01 if val >= 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    run_full_eda()
