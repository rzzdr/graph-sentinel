from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import pytest

if TYPE_CHECKING:
    import polars as pl

from src.graph.builder import GraphBuilder
from src.propagation.engine import RiskPropagationEngine


@pytest.fixture  # type: ignore[misc]
def small_graph(sample_accounts: pl.DataFrame, sample_transactions: pl.DataFrame) -> nx.DiGraph:
    builder = GraphBuilder()
    return builder.build_from_transactions(sample_transactions, sample_accounts)


class TestRiskPropagationEngine:
    def test_propagate_returns_scores(self, small_graph: nx.DiGraph) -> None:
        engine = RiskPropagationEngine(decay_factor=0.5, max_hops=3, max_iterations=10)
        scores = engine.propagate(small_graph)

        assert isinstance(scores, dict)
        assert len(scores) == small_graph.number_of_nodes()

    def test_scores_in_range(self, small_graph: nx.DiGraph) -> None:
        engine = RiskPropagationEngine(decay_factor=0.5, max_hops=3, max_iterations=10)
        scores = engine.propagate(small_graph)

        for node, score in scores.items():
            assert 0.0 <= score <= 1.0, f"Node {node} has score {score}"

    def test_seed_nodes_retain_high_scores(self) -> None:
        graph = nx.DiGraph()
        graph.add_edge("A", "B", weight=100.0, frequency=5)
        graph.add_edge("B", "C", weight=50.0, frequency=3)
        graph.add_edge("C", "D", weight=25.0, frequency=1)

        engine = RiskPropagationEngine(decay_factor=0.5, max_hops=3, max_iterations=20)
        seeds = {"A": 1.0}
        scores = engine.propagate(graph, seed_scores=seeds)

        assert scores["A"] >= 0.9

    def test_risk_decays_with_distance(self) -> None:
        graph = nx.DiGraph()
        graph.add_edge("A", "B", weight=100.0, frequency=5)
        graph.add_edge("B", "C", weight=100.0, frequency=5)
        graph.add_edge("C", "D", weight=100.0, frequency=5)

        engine = RiskPropagationEngine(decay_factor=0.5, max_hops=5, max_iterations=20)
        seeds = {"A": 1.0}
        scores = engine.propagate(graph, seed_scores=seeds)

        assert scores["A"] >= scores["B"]
        assert scores["B"] >= scores["C"]
        assert scores["C"] >= scores["D"]

    def test_empty_graph(self) -> None:
        engine = RiskPropagationEngine()
        scores = engine.propagate(nx.DiGraph())
        assert scores == {}

    def test_no_seeds_no_fraud_nodes(self) -> None:
        graph = nx.DiGraph()
        graph.add_edge("A", "B", weight=1.0, frequency=1)
        engine = RiskPropagationEngine()
        scores = engine.propagate(graph)
        assert all(s == 0.0 for s in scores.values())

    def test_auto_seed_from_fraud_nodes(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("fraud_node", is_fraud=True)
        graph.add_node("clean_node", is_fraud=False)
        graph.add_edge("fraud_node", "clean_node", weight=50.0, frequency=3)

        engine = RiskPropagationEngine(decay_factor=0.5, max_hops=3, max_iterations=20)
        scores = engine.propagate(graph)

        assert scores["fraud_node"] >= 0.9
        assert scores["clean_node"] > 0.0

    def test_convergence(self, small_graph: nx.DiGraph) -> None:
        engine = RiskPropagationEngine(
            decay_factor=0.5, max_hops=3, max_iterations=100, convergence_threshold=1e-6
        )
        scores = engine.propagate(small_graph)
        assert len(scores) > 0
