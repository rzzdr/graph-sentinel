from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class RiskDecomposer:
    def decompose_account(
        self,
        account_id: str,
        features: pl.DataFrame,
        risk_scores: pl.DataFrame,
        model: Any | None = None,
        feature_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        account_features = features.filter(pl.col("account_id") == account_id)
        account_scores = risk_scores.filter(pl.col("account_id") == account_id)

        if account_features.height == 0:
            return {"error": f"Account {account_id} not found"}

        feature_row = account_features.row(0, named=True)
        score_row = account_scores.row(0, named=True) if account_scores.height > 0 else {}

        decomposition: dict[str, Any] = {
            "account_id": account_id,
            "risk_score": score_row.get("risk_score", 0.0),
            "risk_level": score_row.get("risk_level", "unknown"),
            "behavioral_risk_score": score_row.get("behavioral_risk_score", 0.0),
            "structural_risk_score": score_row.get("structural_risk_score", 0.0),
            "network_propagation_score": score_row.get("network_propagation_score", 0.0),
        }

        top_features = self._get_top_contributing_features(feature_row, feature_columns)
        decomposition["top_contributing_features"] = top_features

        if model is not None:
            shap_values = self._compute_shap_values(model, account_features, feature_columns)
            if shap_values:
                decomposition["shap_values"] = shap_values

        return decomposition

    def batch_decompose(
        self,
        features: pl.DataFrame,
        risk_scores: pl.DataFrame,
        top_n: int = 50,
    ) -> list[dict[str, Any]]:
        top_accounts = (
            risk_scores.sort("risk_score", descending=True)
            .head(top_n)
            .select("account_id")
            .to_series()
            .to_list()
        )

        results = []
        for account_id in top_accounts:
            result = self.decompose_account(account_id, features, risk_scores)
            results.append(result)

        return results

    def _get_top_contributing_features(
        self,
        feature_row: dict[str, Any],
        feature_columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        exclude = {"account_id", "is_fraud", "fraud_role", "community_id"}
        candidates = {}

        for key, value in feature_row.items():
            if key in exclude:
                continue
            if feature_columns and key not in feature_columns:
                continue
            if isinstance(value, int | float) and not np.isnan(value):
                candidates[key] = abs(float(value))

        sorted_features = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        return [
            {"feature": name, "value": feature_row[name], "abs_magnitude": magnitude}
            for name, magnitude in sorted_features[:10]
        ]

    def _compute_shap_values(
        self,
        model: Any,
        account_features: pl.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> dict[str, float] | None:
        try:
            import shap

            cols = feature_columns or [
                c
                for c in account_features.columns
                if c not in {"account_id", "is_fraud", "fraud_role"}
                and account_features[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
            ]

            X = account_features.select(cols).to_numpy()
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X)

            if isinstance(sv, list):
                sv = sv[1]

            return {cols[i]: round(float(sv[0][i]), 6) for i in range(len(cols))}
        except Exception as e:
            logger.warning("shap_computation_failed", error=str(e))
            return None
