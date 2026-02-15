from __future__ import annotations

from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class ModelEvaluator:
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        from sklearn.metrics import (
            average_precision_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_pred = (y_prob >= threshold).astype(int)

        metrics: dict[str, float] = {}

        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 6)
        except ValueError:
            metrics["roc_auc"] = 0.0

        try:
            metrics["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 6)
        except ValueError:
            metrics["pr_auc"] = 0.0

        metrics["precision"] = round(float(precision_score(y_true, y_pred, zero_division=0)), 6)
        metrics["recall"] = round(float(recall_score(y_true, y_pred, zero_division=0)), 6)

        for k in [50, 100, 200, 500]:
            p_at_k, r_at_k = self._precision_recall_at_k(y_true, y_prob, k)
            metrics[f"precision@{k}"] = round(p_at_k, 6)
            metrics[f"recall@{k}"] = round(r_at_k, 6)

        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        metrics["false_positive_rate"] = round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 6)

        return metrics

    def _precision_recall_at_k(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        k: int,
    ) -> tuple[float, float]:
        if k > len(y_true):
            k = len(y_true)

        top_k_indices = np.argsort(y_prob)[-k:]
        top_k_true = y_true[top_k_indices]

        precision = float(top_k_true.sum() / k) if k > 0 else 0.0

        total_positives = y_true.sum()
        recall = float(top_k_true.sum() / total_positives) if total_positives > 0 else 0.0

        return precision, recall

    def generate_report(
        self,
        results: dict[str, Any],
        test_data: np.ndarray | None = None,
        test_labels: np.ndarray | None = None,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {"models": {}}

        for model_name, model_data in results.items():
            model_report: dict[str, Any] = {
                "val_metrics": model_data.get("val_metrics", {}),
            }

            if test_data is not None and test_labels is not None:
                model = model_data.get("model")
                if model and hasattr(model, "predict_proba"):
                    test_prob = model.predict_proba(test_data)[:, 1]
                    model_report["test_metrics"] = self.compute_metrics(test_labels, test_prob)

            report["models"][model_name] = model_report

        best_model = max(
            report["models"].items(),
            key=lambda x: x[1].get("val_metrics", {}).get("pr_auc", 0),
        )
        report["best_model"] = best_model[0]
        report["best_pr_auc"] = best_model[1].get("val_metrics", {}).get("pr_auc", 0)

        logger.info("evaluation_report_generated", best_model=report["best_model"])
        return report
