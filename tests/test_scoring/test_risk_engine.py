from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

    from src.config.settings import ScoringConfig

import numpy as np

from src.features.pipeline import FeatureEngineeringPipeline
from src.graph.builder import GraphBuilder
from src.scoring.risk_engine import RiskScoringEngine


class TestRiskScoringEngine:
    def _build_features(
        self,
        sample_accounts: pl.DataFrame,
        sample_transactions: pl.DataFrame,
    ) -> pl.DataFrame:
        builder = GraphBuilder()
        graph = builder.build_from_transactions(sample_transactions, sample_accounts)
        pipeline = FeatureEngineeringPipeline()
        return pipeline.run(graph, sample_transactions, sample_accounts)

    def test_scoring_produces_scores(
        self,
        scoring_config: ScoringConfig,
        sample_accounts: pl.DataFrame,
        sample_transactions: pl.DataFrame,
    ) -> None:
        features = self._build_features(sample_accounts, sample_transactions)
        engine = RiskScoringEngine(scoring_config)
        scores = engine.score(features)

        assert scores.height > 0
        assert "risk_score" in scores.columns
        assert "risk_level" in scores.columns

    def test_scores_in_range(
        self,
        scoring_config: ScoringConfig,
        sample_accounts: pl.DataFrame,
        sample_transactions: pl.DataFrame,
    ) -> None:
        features = self._build_features(sample_accounts, sample_transactions)
        engine = RiskScoringEngine(scoring_config)
        scores = engine.score(features)

        min_score = scores["risk_score"].min()
        max_score = scores["risk_score"].max()
        assert min_score >= 0
        assert max_score <= 100

    def test_valid_risk_levels(
        self,
        scoring_config: ScoringConfig,
        sample_accounts: pl.DataFrame,
        sample_transactions: pl.DataFrame,
    ) -> None:
        features = self._build_features(sample_accounts, sample_transactions)
        engine = RiskScoringEngine(scoring_config)
        scores = engine.score(features)

        valid_levels = {"minimal", "low", "medium", "high", "critical"}
        actual = set(scores["risk_level"].unique().to_list())
        assert actual.issubset(valid_levels)

    def test_score_with_signals_fallback(
        self,
        scoring_config: ScoringConfig,
        sample_accounts: pl.DataFrame,
        sample_transactions: pl.DataFrame,
    ) -> None:
        features = self._build_features(sample_accounts, sample_transactions)
        engine = RiskScoringEngine(scoring_config)
        scores = engine.score_with_signals(features)

        assert scores.height > 0
        assert "risk_score" in scores.columns
        min_score = scores["risk_score"].min()
        max_score = scores["risk_score"].max()
        assert min_score >= 0
        assert max_score <= 100

    def test_score_with_signals_and_extras(
        self,
        scoring_config: ScoringConfig,
        sample_accounts: pl.DataFrame,
        sample_transactions: pl.DataFrame,
    ) -> None:
        features = self._build_features(sample_accounts, sample_transactions)
        account_ids = features["account_id"].to_list()

        propagation = {aid: np.random.random() for aid in account_ids}
        graph_scores = {aid: np.random.random() for aid in account_ids}
        seq_scores = {aid: np.random.random() for aid in account_ids}

        engine = RiskScoringEngine(scoring_config)
        scores = engine.score_with_signals(
            features,
            propagation_scores=propagation,
            graph_scores=graph_scores,
            sequence_anomaly_scores=seq_scores,
        )

        assert scores.height > 0
        assert "risk_score" in scores.columns
        assert "advanced_propagation_score" in scores.columns
