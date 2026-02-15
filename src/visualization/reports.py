from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class ReportCharts:
    def plot_model_comparison(
        self,
        evaluation_report: dict[str, Any],
        output_path: Path,
    ) -> Path:
        models = evaluation_report.get("models", {})
        if not models:
            return output_path

        model_names = list(models.keys())
        metrics_to_plot = ["roc_auc", "pr_auc", "precision", "recall"]

        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(16, 5))

        for i, metric in enumerate(metrics_to_plot):
            values = [models[m].get("val_metrics", {}).get(metric, 0) for m in model_names]
            axes[i].bar(model_names, values, color=["#3498db", "#2ecc71", "#e74c3c"])
            axes[i].set_title(metric.upper())
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis="x", rotation=30)

        plt.suptitle("Model Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()

        logger.info("model_comparison_plotted", path=str(output_path))
        return output_path

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        output_path: Path,
        model_name: str = "Model",
    ) -> Path:
        from sklearn.metrics import average_precision_score, precision_recall_curve

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})", linewidth=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()

        logger.info("pr_curve_plotted", path=str(output_path))
        return output_path

    def plot_feature_importance(
        self,
        feature_names: list[str],
        importances: np.ndarray,
        output_path: Path,
        top_n: int = 20,
    ) -> Path:
        indices = np.argsort(importances)[-top_n:]
        top_names = [feature_names[i] for i in indices]
        top_values = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top_names, top_values, color="#3498db")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importances")
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()

        logger.info("feature_importance_plotted", path=str(output_path))
        return output_path

    def plot_transaction_timeline(
        self,
        transactions: pl.DataFrame,
        account_id: str,
        output_path: Path,
    ) -> Path:
        account_txns = transactions.filter(
            (pl.col("sender_id") == account_id) | (pl.col("receiver_id") == account_id)
        ).sort("timestamp")

        if account_txns.height == 0:
            logger.warning("no_transactions_for_account", account_id=account_id)
            return output_path

        df = account_txns.to_pandas()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        sent = df[df["sender_id"] == account_id]
        received = df[df["receiver_id"] == account_id]

        ax1.scatter(sent["timestamp"], sent["amount"], c="red", alpha=0.6, s=20, label="Sent")
        ax1.scatter(
            received["timestamp"],
            received["amount"],
            c="green",
            alpha=0.6,
            s=20,
            label="Received",
        )
        ax1.set_ylabel("Amount")
        ax1.set_title(f"Transaction Timeline: {account_id[:12]}...")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        daily = df.set_index("timestamp").resample("D").size()
        ax2.bar(daily.index, daily.values, width=1, alpha=0.7, color="#3498db")
        ax2.set_ylabel("Daily Transaction Count")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()

        logger.info("timeline_plotted", account_id=account_id, path=str(output_path))
        return output_path
