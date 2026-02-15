from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pytest

if TYPE_CHECKING:
    import polars as pl

from src.graph.builder import GraphBuilder
from src.graph_learning.embeddings import GraphEmbeddingEngine
from src.graph_learning.model import GraphClassifier


@pytest.fixture  # type: ignore[misc]
def small_graph(sample_accounts: pl.DataFrame, sample_transactions: pl.DataFrame) -> nx.DiGraph:
    builder = GraphBuilder()
    return builder.build_from_transactions(sample_transactions, sample_accounts)


class TestGraphEmbeddingEngine:
    def test_compute_embeddings_returns_dict(self, small_graph: nx.DiGraph) -> None:
        engine = GraphEmbeddingEngine(embedding_dim=16, walk_length=10, num_walks=3)
        embeddings = engine.compute_embeddings(small_graph)

        assert isinstance(embeddings, dict)
        assert len(embeddings) == small_graph.number_of_nodes()

    def test_embedding_dimensions(self, small_graph: nx.DiGraph) -> None:
        dim = 16
        engine = GraphEmbeddingEngine(embedding_dim=dim, walk_length=10, num_walks=3)
        embeddings = engine.compute_embeddings(small_graph)

        for node, vec in embeddings.items():
            assert vec.shape == (dim,), f"Node {node} has shape {vec.shape}, expected ({dim},)"

    def test_empty_graph(self) -> None:
        engine = GraphEmbeddingEngine(embedding_dim=16)
        graph = nx.DiGraph()
        embeddings = engine.compute_embeddings(graph)
        assert embeddings == {}

    def test_single_node_graph(self) -> None:
        engine = GraphEmbeddingEngine(embedding_dim=16)
        graph = nx.DiGraph()
        graph.add_node("A")
        embeddings = engine.compute_embeddings(graph)
        assert len(embeddings) == 1
        assert embeddings["A"].shape == (16,)

    def test_embeddings_to_features(self, small_graph: nx.DiGraph) -> None:
        dim = 8
        engine = GraphEmbeddingEngine(embedding_dim=dim, walk_length=10, num_walks=3)
        embeddings = engine.compute_embeddings(small_graph)
        features = engine.embeddings_to_features(embeddings)

        assert "account_id" in features
        assert len(features["account_id"]) == len(embeddings)
        for d in range(dim):
            assert f"gnn_emb_{d}" in features

    def test_embeddings_are_finite(self, small_graph: nx.DiGraph) -> None:
        engine = GraphEmbeddingEngine(embedding_dim=16, walk_length=10, num_walks=3)
        embeddings = engine.compute_embeddings(small_graph)

        for node, vec in embeddings.items():
            assert np.all(np.isfinite(vec)), f"Node {node} has non-finite embedding values"


class TestGraphClassifier:
    def test_train_and_predict(self, small_graph: nx.DiGraph) -> None:
        engine = GraphEmbeddingEngine(embedding_dim=16, walk_length=10, num_walks=3)
        embeddings = engine.compute_embeddings(small_graph)

        nodes = list(embeddings.keys())
        labels = {n: 0 for n in nodes}
        for n in nodes[-5:]:
            labels[n] = 1

        clf = GraphClassifier(embedding_dim=16, n_estimators=10)
        result = clf.train(embeddings, labels)

        assert "cv_pr_auc" in result or "status" in result

        predictions = clf.predict(embeddings)
        assert len(predictions) == len(embeddings)
        for score in predictions.values():
            assert 0.0 <= score <= 1.0

    def test_predict_before_train_returns_zeros(self) -> None:
        clf = GraphClassifier(embedding_dim=8)
        embeddings = {"A": np.zeros(8), "B": np.ones(8)}
        predictions = clf.predict(embeddings)
        assert all(v == 0.0 for v in predictions.values())

    def test_single_class_skips_training(self) -> None:
        clf = GraphClassifier(embedding_dim=8, n_estimators=10)
        embeddings = {"A": np.random.randn(8), "B": np.random.randn(8)}
        labels = {"A": 0, "B": 0}
        result = clf.train(embeddings, labels)
        assert result.get("status") == "skipped"  # type: ignore[comparison-overlap]

    def test_with_node_features(self, small_graph: nx.DiGraph) -> None:
        engine = GraphEmbeddingEngine(embedding_dim=16, walk_length=10, num_walks=3)
        embeddings = engine.compute_embeddings(small_graph)

        nodes = list(embeddings.keys())
        labels = {n: 0 for n in nodes}
        for n in nodes[-5:]:
            labels[n] = 1

        node_features = {n: np.random.randn(4) for n in nodes}

        clf = GraphClassifier(embedding_dim=16, n_estimators=10)
        clf.train(embeddings, labels, node_features=node_features)

        predictions = clf.predict(embeddings, node_features=node_features)
        assert len(predictions) == len(embeddings)
