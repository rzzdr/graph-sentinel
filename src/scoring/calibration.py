from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger(__name__)


class ScoreCalibrator:
    """Post-hoc probability calibration for fraud risk scores.

    Applies isotonic regression calibration to raw model probabilities
    to produce well-calibrated score distributions. This ensures that
    a score of 0.8 actually corresponds to ~80% observed fraud rate
    within that score band, making threshold selection reliable.
    """

    def __init__(self) -> None:
        self.calibrator: Any = None
        self._fitted = False

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        """Fit the calibrator on raw scores and true labels."""
        if len(np.unique(labels)) < 2:
            logger.warning("calibrator_skip", reason="single class")
            return {"status": "skipped"}

        from sklearn.isotonic import IsotonicRegression

        self.calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )
        self.calibrator.fit(raw_scores, labels)
        self._fitted = True

        calibrated = self.calibrate(raw_scores)
        reliability = self._compute_reliability(calibrated, labels)

        logger.info("calibrator_fitted", samples=len(labels), **reliability)
        return reliability

    def calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        """Transform raw scores into calibrated probabilities."""
        if not self._fitted or self.calibrator is None:
            return raw_scores

        return self.calibrator.predict(raw_scores)

    def to_risk_score(self, calibrated_probs: np.ndarray, scale: float = 100.0) -> np.ndarray:
        """Convert calibrated probabilities to 0-100 risk scores."""
        scores = calibrated_probs * scale
        return np.clip(scores, 0, scale)

    def _compute_reliability(
        self,
        calibrated: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> dict[str, float]:
        """Compute Expected Calibration Error (ECE) and Brier score."""
        brier = float(np.mean((calibrated - labels) ** 2))

        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (calibrated >= bin_edges[i]) & (calibrated < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            avg_confidence = calibrated[mask].mean()
            avg_accuracy = labels[mask].mean()
            ece += mask.sum() / len(labels) * abs(avg_confidence - avg_accuracy)

        return {
            "ece": round(ece, 6),
            "brier_score": round(brier, 6),
        }

    def save(self, path: Path) -> None:
        """Persist the fitted calibrator to disk."""
        if not self._fitted or self.calibrator is None:
            logger.warning("calibrator_save_skip", reason="not fitted")
            return

        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "calibrator": self.calibrator,
            "_fitted": self._fitted,
        }
        joblib.dump(state, path)
        logger.info("calibrator_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> ScoreCalibrator:
        """Load a persisted calibrator from disk."""
        import joblib

        state = joblib.load(path)
        instance = cls()
        instance.calibrator = state["calibrator"]
        instance._fitted = state["_fitted"]
        logger.info("calibrator_loaded", path=str(path))
        return instance
