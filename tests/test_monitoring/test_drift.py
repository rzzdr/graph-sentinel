from __future__ import annotations

import networkx as nx
import numpy as np
import polars as pl
import pytest

from src.monitoring.drift import DriftDetector
from src.monitoring.graph_drift import GraphDriftDetector
from src.monitoring.prediction_drift import PredictionDriftDetector


class TestDriftDetector:
    def _make_features(self, n: int = 100, seed: int = 42) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        return pl.DataFrame(
            {
                "account_id": [f"acc_{i}" for i in range(n)],
                "feature_a": rng.normal(0, 1, n).tolist(),
                "feature_b": rng.normal(5, 2, n).tolist(),
                "feature_c": rng.exponential(1, n).tolist(),
            }
        )

    def test_set_baseline_and_detect(self) -> None:
        baseline = self._make_features(200, seed=42)
        current = self._make_features(200, seed=43)

        detector = DriftDetector()
        detector.set_baseline(baseline)
        results = detector.detect_drift(current)

        assert isinstance(results, dict)
        for _, info in results.items():
            assert "drifted" in info
            assert "ks_p_value" in info
            assert "psi" in info
            assert "wasserstein" in info

    def test_identical_data_no_drift(self) -> None:
        data = self._make_features(200, seed=42)

        detector = DriftDetector()
        detector.set_baseline(data)
        results = detector.detect_drift(data)

        for col, info in results.items():
            assert not info["drifted"], f"Feature {col} falsely detected drift"

    def test_shifted_data_detects_drift(self) -> None:
        baseline = self._make_features(200, seed=42)
        shifted = baseline.with_columns(
            (pl.col("feature_a") + 10).alias("feature_a"),
        )

        detector = DriftDetector()
        detector.set_baseline(baseline)
        results = detector.detect_drift(shifted)

        assert results["feature_a"]["drifted"]

    def test_summary(self) -> None:
        baseline = self._make_features(200, seed=42)
        current = self._make_features(200, seed=42)

        detector = DriftDetector()
        detector.set_baseline(baseline)
        results = detector.detect_drift(current)
        summary = detector.get_drift_summary(results)

        assert "total_features" in summary
        assert "features_drifted" in summary
        assert "needs_retraining" in summary

    def test_detect_before_baseline_raises(self) -> None:
        detector = DriftDetector()
        with pytest.raises(RuntimeError):
            detector.detect_drift(self._make_features())


class TestPredictionDriftDetector:
    def test_set_baseline_and_detect(self) -> None:
        rng = np.random.default_rng(42)
        baseline = rng.random(500)
        current = rng.random(500)

        detector = PredictionDriftDetector()
        detector.set_baseline(baseline)
        result = detector.detect(current)

        assert "drifted" in result
        assert "ks_statistic" in result
        assert "wasserstein_distance" in result
        assert "alert_rate_delta" in result

    def test_identical_scores_no_drift(self) -> None:
        scores = np.random.default_rng(42).random(500)

        detector = PredictionDriftDetector()
        detector.set_baseline(scores)
        result = detector.detect(scores)

        assert not result["drifted"]

    def test_shifted_scores_detect_drift(self) -> None:
        rng = np.random.default_rng(42)
        baseline = rng.random(500) * 0.3
        current = rng.random(500) * 0.3 + 0.5

        detector = PredictionDriftDetector()
        detector.set_baseline(baseline)
        result = detector.detect(current)

        assert result["drifted"]

    def test_detect_before_baseline_raises(self) -> None:
        detector = PredictionDriftDetector()
        with pytest.raises(RuntimeError):
            detector.detect(np.array([0.5]))


class TestGraphDriftDetector:
    def _make_graph(self, n_nodes: int = 50, n_edges: int = 100, seed: int = 42) -> nx.DiGraph:
        rng = np.random.default_rng(seed)
        graph = nx.DiGraph()
        nodes = [f"n_{i}" for i in range(n_nodes)]
        graph.add_nodes_from(nodes)
        for _ in range(n_edges):
            u = nodes[rng.integers(0, n_nodes)]
            v = nodes[rng.integers(0, n_nodes)]
            if u != v:
                graph.add_edge(u, v, weight=float(rng.random()), frequency=int(rng.integers(1, 10)))
        return graph

    def test_set_baseline_and_detect(self) -> None:
        graph = self._make_graph()

        detector = GraphDriftDetector()
        detector.set_baseline(graph)
        result = detector.detect(graph)

        assert "drifted" in result
        assert "metrics_drifted" in result
        assert "details" in result

    def test_same_graph_no_drift(self) -> None:
        graph = self._make_graph()

        detector = GraphDriftDetector()
        detector.set_baseline(graph)
        result = detector.detect(graph)

        assert not result["drifted"]

    def test_very_different_graph_detects_drift(self) -> None:
        small = self._make_graph(n_nodes=20, n_edges=30)
        large = self._make_graph(n_nodes=200, n_edges=2000, seed=99)

        detector = GraphDriftDetector()
        detector.set_baseline(small)
        result = detector.detect(large)

        assert result["drifted"]

    def test_detect_before_baseline_raises(self) -> None:
        detector = GraphDriftDetector()
        with pytest.raises(RuntimeError):
            detector.detect(self._make_graph())
