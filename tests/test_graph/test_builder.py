from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

from src.graph.builder import GraphBuilder


class TestGraphBuilder:
    def test_build_creates_graph(
        self,
        sample_accounts: pl.DataFrame,
        sample_transactions: pl.DataFrame,
    ) -> None:
        builder = GraphBuilder()
        graph = builder.build_from_transactions(sample_transactions, sample_accounts)

        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

    def test_graph_has_edge_attributes(
        self,
        sample_accounts: pl.DataFrame,
        sample_transactions: pl.DataFrame,
    ) -> None:
        builder = GraphBuilder()
        graph = builder.build_from_transactions(sample_transactions, sample_accounts)

        for _, _, data in list(graph.edges(data=True))[:5]:
            assert "weight" in data
            assert "frequency" in data
            assert data["weight"] > 0
            assert data["frequency"] >= 1

    def test_graph_has_node_attributes(
        self,
        sample_accounts: pl.DataFrame,
        sample_transactions: pl.DataFrame,
    ) -> None:
        builder = GraphBuilder()
        graph = builder.build_from_transactions(sample_transactions, sample_accounts)

        for node in list(graph.nodes())[:5]:
            data = graph.nodes[node]
            if "account_type" in data:
                assert data["account_type"] in {"savings", "business", "wallet", "unknown"}

    def test_incremental_update(
        self,
        sample_accounts: pl.DataFrame,
        sample_transactions: pl.DataFrame,
    ) -> None:
        builder = GraphBuilder()
        half = sample_transactions.height // 2

        first_half = sample_transactions.head(half)
        second_half = sample_transactions.tail(sample_transactions.height - half)

        graph = builder.build_from_transactions(first_half, sample_accounts)
        original_edges = graph.number_of_edges()

        graph = builder.build_incremental(graph, second_half)
        assert graph.number_of_edges() >= original_edges
