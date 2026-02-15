from __future__ import annotations

import networkx as nx
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class GraphEmbeddingEngine:
    """Produces node embeddings from NetworkX graphs using spectral methods and
    random-walk-based approaches. These embeddings serve as structured intelligence
    features for downstream models without requiring PyTorch Geometric."""

    def __init__(self, embedding_dim: int = 64, walk_length: int = 40, num_walks: int = 10) -> None:
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks

    def compute_embeddings(self, graph: nx.DiGraph) -> dict[str, np.ndarray]:
        nodes = list(graph.nodes())
        if len(nodes) < 2:
            return {n: np.zeros(self.embedding_dim) for n in nodes}

        structural = self._spectral_embedding(graph, nodes)
        walk_based = self._random_walk_embedding(graph, nodes)

        half = self.embedding_dim // 2
        combined: dict[str, np.ndarray] = {}
        for node in nodes:
            s = structural.get(node, np.zeros(half))[:half]
            w = walk_based.get(node, np.zeros(self.embedding_dim - half))[
                : self.embedding_dim - half
            ]
            combined[node] = np.concatenate([s, w])

        logger.info("graph_embeddings_computed", nodes=len(nodes), dim=self.embedding_dim)
        return combined

    def _spectral_embedding(self, graph: nx.DiGraph, nodes: list[str]) -> dict[str, np.ndarray]:
        undirected = graph.to_undirected()
        half = self.embedding_dim // 2
        k = min(half, len(nodes) - 1)

        if k < 1:
            return {n: np.zeros(half) for n in nodes}

        try:
            laplacian = nx.normalized_laplacian_matrix(undirected, nodelist=nodes)
            from scipy.sparse.linalg import eigsh

            _eigenvalues, eigenvectors = eigsh(laplacian, k=k, which="SM", tol=int(1e-4))

            result: dict[str, np.ndarray] = {}
            for i, node in enumerate(nodes):
                vec = eigenvectors[i]
                if len(vec) < half:
                    vec = np.pad(vec, (0, half - len(vec)))
                result[node] = vec[:half]
            return result
        except Exception:
            logger.warning("spectral_embedding_fallback")
            return {n: np.zeros(half) for n in nodes}

    def _random_walk_embedding(self, graph: nx.DiGraph, nodes: list[str]) -> dict[str, np.ndarray]:
        dim = self.embedding_dim - (self.embedding_dim // 2)
        node_idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)

        cooccurrence = np.zeros((n, n), dtype=np.float32)

        for node in nodes:
            for _ in range(self.num_walks):
                walk = self._do_walk(graph, node)
                idx_walk = [node_idx[w] for w in walk if w in node_idx]
                for i, wi in enumerate(idx_walk):
                    window = idx_walk[max(0, i - 5) : i + 6]
                    for wj in window:
                        if wi != wj:
                            cooccurrence[wi, wj] += 1.0

        cooccurrence = np.log1p(cooccurrence)

        try:
            from sklearn.decomposition import TruncatedSVD

            k = min(dim, n - 1, cooccurrence.shape[1])
            if k < 1:
                return {nd: np.zeros(dim) for nd in nodes}

            svd = TruncatedSVD(n_components=k, random_state=42)
            embeddings_matrix = svd.fit_transform(cooccurrence)

            result: dict[str, np.ndarray] = {}
            for i, node in enumerate(nodes):
                vec = embeddings_matrix[i]
                if len(vec) < dim:
                    vec = np.pad(vec, (0, dim - len(vec)))
                result[node] = vec[:dim]
            return result
        except Exception:
            logger.warning("walk_embedding_fallback")
            return {nd: np.zeros(dim) for nd in nodes}

    def _do_walk(self, graph: nx.DiGraph, start: str) -> list[str]:
        walk = [start]
        current = start
        for _ in range(self.walk_length - 1):
            neighbors = list(graph.successors(current)) + list(graph.predecessors(current))
            if not neighbors:
                break
            current = neighbors[np.random.randint(len(neighbors))]
            walk.append(current)
        return walk

    def embeddings_to_features(
        self, embeddings: dict[str, np.ndarray], prefix: str = "gnn_emb"
    ) -> dict[str, list[str] | list[float]]:
        """Convert node embeddings dict to feature columns for a DataFrame."""
        nodes = sorted(embeddings.keys())
        dim = self.embedding_dim

        account_ids: list[str] = []
        emb_columns: dict[str, list[float]] = {}
        for d in range(dim):
            emb_columns[f"{prefix}_{d}"] = []

        for node in nodes:
            account_ids.append(node)
            vec = embeddings[node]
            for d in range(dim):
                emb_columns[f"{prefix}_{d}"].append(float(vec[d]) if d < len(vec) else 0.0)

        result: dict[str, list[str] | list[float]] = {"account_id": account_ids}
        result.update(emb_columns)
        return result
