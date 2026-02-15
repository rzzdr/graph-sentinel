from __future__ import annotations

from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class PredictionDriftDetector:
    """Detects drift in model prediction distributions over time.

    Monitors both the score distribution and the label distribution,
    flagging when either shifts beyond expected bounds. Supports
    batch retraining triggers.
    """

    def __init__(self, significance: float = 0.05) -> None:
        self.significance = significance
        self.baseline_scores: np.ndarray | None = None
        self.baseline_stats: dict[str, float] | None = None

    def set_baseline(self, scores: np.ndarray) -> None:
        self.baseline_scores = scores.copy()
        self.baseline_stats = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "median": float(np.median(scores)),
            "q90": float(np.percentile(scores, 90)),
            "q99": float(np.percentile(scores, 99)),
            "alert_rate": float(np.mean(scores > 0.75)),
        }
        logger.info("prediction_baseline_set", samples=len(scores))

    def detect(self, current_scores: np.ndarray) -> dict[str, Any]:
        if self.baseline_scores is None or self.baseline_stats is None:
            raise RuntimeError("Set baseline before detecting prediction drift")

        from scipy import stats

        ks_stat, ks_p = stats.ks_2samp(self.baseline_scores[:5000], current_scores[:5000])
        ks_stat = float(ks_stat)  # pyright: ignore[reportArgumentType]
        ks_p = float(ks_p)  # pyright: ignore[reportArgumentType]

        wasserstein = float(
            stats.wasserstein_distance(self.baseline_scores[:5000], current_scores[:5000])
        )

        current_stats = {
            "mean": float(np.mean(current_scores)),
            "std": float(np.std(current_scores)),
            "median": float(np.median(current_scores)),
            "q90": float(np.percentile(current_scores, 90)),
            "q99": float(np.percentile(current_scores, 99)),
            "alert_rate": float(np.mean(current_scores > 0.75)),
        }

        mean_shift = abs(current_stats["mean"] - self.baseline_stats["mean"])
        relative_shift = mean_shift / (self.baseline_stats["std"] + 1e-8)

        alert_rate_delta = abs(current_stats["alert_rate"] - self.baseline_stats["alert_rate"])

        drifted = ks_p < self.significance or relative_shift > 2.0 or alert_rate_delta > 0.05

        return {
            "drifted": drifted,
            "ks_statistic": round(ks_stat, 6),
            "ks_p_value": round(ks_p, 6),
            "wasserstein_distance": round(wasserstein, 6),
            "mean_shift": round(mean_shift, 6),
            "relative_shift": round(relative_shift, 4),
            "alert_rate_delta": round(alert_rate_delta, 4),
            "baseline_stats": {k: round(v, 4) for k, v in self.baseline_stats.items()},
            "current_stats": {k: round(v, 4) for k, v in current_stats.items()},
        }
