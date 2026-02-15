from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

from src.features.pipeline import FeatureEngineeringPipeline
from src.graph.builder import GraphBuilder


class TestFeatureEngineering:
    def test_pipeline_produces_features(
        self,
        sample_accounts: pl.DataFrame,
        sample_transactions: pl.DataFrame,
    ) -> None:
        builder = GraphBuilder()
        graph = builder.build_from_transactions(sample_transactions, sample_accounts)

        pipeline = FeatureEngineeringPipeline()
        features = pipeline.run(graph, sample_transactions, sample_accounts)

        assert features.height > 0
        assert "account_id" in features.columns
        assert features.width > 10

    def test_features_have_no_nulls(
        self,
        sample_accounts: pl.DataFrame,
        sample_transactions: pl.DataFrame,
    ) -> None:
        builder = GraphBuilder()
        graph = builder.build_from_transactions(sample_transactions, sample_accounts)

        pipeline = FeatureEngineeringPipeline()
        features = pipeline.run(graph, sample_transactions, sample_accounts)

        null_counts = {
            col: features[col].null_count()
            for col in features.columns
            if features[col].null_count() > 0
        }
        assert len(null_counts) == 0, f"Null columns: {null_counts}"

    def test_structural_features_present(
        self,
        sample_accounts: pl.DataFrame,
        sample_transactions: pl.DataFrame,
    ) -> None:
        builder = GraphBuilder()
        graph = builder.build_from_transactions(sample_transactions, sample_accounts)

        pipeline = FeatureEngineeringPipeline()
        features = pipeline.run(graph, sample_transactions, sample_accounts)

        structural = ["in_degree", "out_degree", "pagerank", "betweenness"]
        for feat in structural:
            assert feat in features.columns, f"Missing feature: {feat}"

    def test_behavioral_features_present(
        self,
        sample_accounts: pl.DataFrame,
        sample_transactions: pl.DataFrame,
    ) -> None:
        builder = GraphBuilder()
        graph = builder.build_from_transactions(sample_transactions, sample_accounts)

        pipeline = FeatureEngineeringPipeline()
        features = pipeline.run(graph, sample_transactions, sample_accounts)

        behavioral = ["incoming_volume", "outgoing_volume", "burst_index"]
        for feat in behavioral:
            assert feat in features.columns, f"Missing feature: {feat}"
