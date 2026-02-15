from __future__ import annotations

import numpy as np
import polars as pl

from src.scoring.calibration import ScoreCalibrator
from src.scoring.meta_model import MetaScorer


class TestMetaScorer:
    def _make_signals(self, n: int = 200) -> tuple[pl.DataFrame, np.ndarray]:
        rng = np.random.default_rng(42)
        fraud = rng.random(n) > 0.9
        labels = fraud.astype(int)

        data = {
            "account_id": [f"acc_{i}" for i in range(n)],
            "sig_behavioral": rng.random(n).tolist(),
            "sig_structural": rng.random(n).tolist(),
            "sig_propagation": (fraud * 0.7 + rng.random(n) * 0.3).tolist(),
            "sig_graph": (fraud * 0.6 + rng.random(n) * 0.4).tolist(),
            "sig_sequence": rng.random(n).tolist(),
        }
        return pl.DataFrame(data), labels

    def test_fit_and_predict(self) -> None:
        signals, labels = self._make_signals()
        scorer = MetaScorer()
        result = scorer.fit(signals, labels)

        assert "cv_pr_auc" in result
        assert scorer._fitted

        probs = scorer.predict(signals)
        assert len(probs) == signals.height
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_feature_importance(self) -> None:
        signals, labels = self._make_signals()
        scorer = MetaScorer()
        scorer.fit(signals, labels)

        importance = scorer.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_predict_before_fit_returns_zeros(self) -> None:
        signals, _ = self._make_signals(50)
        scorer = MetaScorer()
        probs = scorer.predict(signals)
        assert np.all(probs == 0.0)

    def test_single_class_skips(self) -> None:
        signals, _ = self._make_signals(50)
        labels = np.zeros(50, dtype=int)
        scorer = MetaScorer()
        result = scorer.fit(signals, labels)
        assert result.get("status") == "skipped"  # type: ignore[comparison-overlap]

    def test_missing_columns_handled(self) -> None:
        signals, labels = self._make_signals()
        scorer = MetaScorer()
        scorer.fit(signals, labels)

        partial = signals.select(["account_id", "sig_behavioral", "sig_structural"])
        probs = scorer.predict(partial)
        assert len(probs) == partial.height


class TestScoreCalibrator:
    def test_fit_and_calibrate(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        labels = (rng.random(n) > 0.85).astype(int)
        raw_scores = labels * 0.7 + rng.random(n) * 0.3

        calibrator = ScoreCalibrator()
        result = calibrator.fit(raw_scores, labels)

        assert "ece" in result
        assert "brier_score" in result
        assert calibrator._fitted

    def test_calibrated_in_range(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        labels = (rng.random(n) > 0.85).astype(int)
        raw_scores = labels * 0.7 + rng.random(n) * 0.3

        calibrator = ScoreCalibrator()
        calibrator.fit(raw_scores, labels)

        calibrated = calibrator.calibrate(raw_scores)
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)

    def test_to_risk_score(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        labels = (rng.random(n) > 0.85).astype(int)
        raw_scores = labels * 0.7 + rng.random(n) * 0.3

        calibrator = ScoreCalibrator()
        calibrator.fit(raw_scores, labels)
        calibrated = calibrator.calibrate(raw_scores)
        risk = calibrator.to_risk_score(calibrated)

        assert np.all(risk >= 0.0)
        assert np.all(risk <= 100.0)

    def test_calibrate_without_fit_returns_input(self) -> None:
        calibrator = ScoreCalibrator()
        raw = np.array([0.1, 0.5, 0.9])
        result = calibrator.calibrate(raw)
        np.testing.assert_array_equal(result, raw)

    def test_single_class_skips(self) -> None:
        calibrator = ScoreCalibrator()
        raw = np.array([0.1, 0.2, 0.3])
        labels = np.array([0, 0, 0])
        result = calibrator.fit(raw, labels)
        assert result.get("status") == "skipped"
