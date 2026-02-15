from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class UnsupervisedDetector:
    def __init__(self) -> None:
        self.models: dict[str, Any] = {}

    def detect(
        self,
        features: pl.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> pl.DataFrame:
        logger.info("unsupervised_detection_start", accounts=features.height)

        cols = feature_columns or self._resolve_numeric_columns(features)
        X = features.select(cols).to_numpy()

        from sklearn.preprocessing import RobustScaler

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        iso_scores = self._isolation_forest(X_scaled)
        lof_scores = self._local_outlier_factor(X_scaled)
        density_scores = self._density_deviation(X_scaled)

        combined = (
            self._normalize(iso_scores) * 0.4
            + self._normalize(lof_scores) * 0.3
            + self._normalize(density_scores) * 0.3
        )

        result = features.select("account_id").with_columns(
            pl.Series("anomaly_score_iforest", iso_scores),
            pl.Series("anomaly_score_lof", lof_scores),
            pl.Series("anomaly_score_density", density_scores),
            pl.Series("anomaly_score_combined", combined),
        )

        result = result.sort("anomaly_score_combined", descending=True)
        logger.info("unsupervised_detection_complete", accounts=result.height)
        return result

    def _isolation_forest(self, X: np.ndarray) -> np.ndarray:
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(
            n_estimators=200,
            contamination=0.02,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X)
        self.models["isolation_forest"] = model

        raw_scores = -model.score_samples(X)
        return raw_scores

    def _local_outlier_factor(self, X: np.ndarray) -> np.ndarray:
        from sklearn.neighbors import LocalOutlierFactor

        model = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.02,
            novelty=False,
            n_jobs=-1,
        )
        model.fit_predict(X)
        self.models["lof"] = model

        return -model.negative_outlier_factor_

    def _density_deviation(self, X: np.ndarray) -> np.ndarray:
        from sklearn.cluster import DBSCAN

        clustering = DBSCAN(eps=2.0, min_samples=5)
        labels = clustering.fit_predict(X)

        noise_mask = labels == -1
        scores = np.zeros(len(X))
        scores[noise_mask] = 1.0

        for label in set(labels):
            if label == -1:
                continue
            cluster_mask = labels == label
            cluster_points = X[cluster_mask]
            centroid = cluster_points.mean(axis=0)

            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            median_dist = np.median(distances)

            if median_dist > 0:
                scores[cluster_mask] = distances / (median_dist * 3)

        return scores

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        min_val, max_val = scores.min(), scores.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(scores)
        result: np.ndarray = (scores - min_val) / (max_val - min_val)
        return result

    def _resolve_numeric_columns(self, df: pl.DataFrame) -> list[str]:
        exclude = {"account_id", "is_fraud", "fraud_role"}
        return [
            c
            for c in df.columns
            if c not in exclude
            and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt32)
        ]
