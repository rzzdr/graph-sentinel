from __future__ import annotations

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class DriftDetector:
    def __init__(self, significance_threshold: float = 0.05) -> None:
        self.significance_threshold = significance_threshold
        self.baseline_stats: dict[str, dict[str, float]] | None = None
        self.baseline_distributions: dict[str, np.ndarray] | None = None

    def set_baseline(self, features: pl.DataFrame) -> None:
        self.baseline_stats = {}
        self.baseline_distributions = {}

        numeric_cols = [
            c
            for c in features.columns
            if features[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        ]

        for col in numeric_cols:
            values = features[col].drop_nulls().to_numpy()
            if len(values) == 0:
                continue
            self.baseline_stats[col] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
            }
            self.baseline_distributions[col] = values[:5000].copy()

        logger.info("baseline_set", features=len(self.baseline_stats))

    def detect_drift(self, current_features: pl.DataFrame) -> dict[str, dict[str, object]]:
        if self.baseline_stats is None or self.baseline_distributions is None:
            msg = "Set baseline before detecting drift"
            raise RuntimeError(msg)

        from scipy import stats

        drift_results: dict[str, dict[str, object]] = {}

        for col, baseline in self.baseline_stats.items():
            if col not in current_features.columns:
                continue

            current_values = current_features[col].drop_nulls().to_numpy()
            if len(current_values) == 0:
                continue

            current_mean = float(np.mean(current_values))

            mean_shift = abs(current_mean - baseline["mean"])
            relative_shift = mean_shift / (baseline["std"] + 1e-8)

            baseline_dist = self.baseline_distributions[col]

            try:
                ks_stat, p_ks = stats.ks_2samp(baseline_dist, current_values[:5000])
            except Exception:
                ks_stat, p_ks = 0.0, 1.0

            try:
                psi = self._compute_psi(baseline_dist, current_values[:5000])
            except Exception:
                psi = 0.0

            try:
                wasserstein = float(
                    stats.wasserstein_distance(baseline_dist, current_values[:5000])
                )
            except Exception:
                wasserstein = 0.0

            drifted = p_ks < self.significance_threshold or relative_shift > 2.0 or psi > 0.2

            drift_results[col] = {
                "drifted": drifted,
                "baseline_mean": round(baseline["mean"], 4),
                "current_mean": round(current_mean, 4),
                "relative_shift": round(relative_shift, 4),
                "ks_statistic": round(float(ks_stat), 6),
                "ks_p_value": round(float(p_ks), 6),
                "psi": round(psi, 6),
                "wasserstein": round(wasserstein, 6),
            }

        n_drifted = sum(1 for r in drift_results.values() if r["drifted"])
        logger.info(
            "drift_detection_complete", drifted_features=n_drifted, total=len(drift_results)
        )

        return drift_results

    def get_drift_summary(self, drift_results: dict[str, dict[str, object]]) -> dict[str, object]:
        drifted = [k for k, v in drift_results.items() if v["drifted"]]

        return {
            "total_features": len(drift_results),
            "features_drifted": len(drifted),
            "drift_ratio": round(len(drifted) / max(len(drift_results), 1), 4),
            "drifted_features": drifted,
            "needs_retraining": len(drifted) > len(drift_results) * 0.3,
        }

    def _compute_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        n_bins: int = 20,
    ) -> float:
        """Population Stability Index — measures distribution shift magnitude."""
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())

        if max_val - min_val < 1e-10:
            return 0.0

        bins = np.linspace(min_val, max_val, n_bins + 1)

        baseline_counts = np.histogram(baseline, bins=bins)[0].astype(float)
        current_counts = np.histogram(current, bins=bins)[0].astype(float)

        baseline_pct = (baseline_counts + 1) / (len(baseline) + n_bins)
        current_pct = (current_counts + 1) / (len(current) + n_bins)

        psi = float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))
        return psi
