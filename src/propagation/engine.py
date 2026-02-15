from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    import networkx as nx

logger = structlog.get_logger(__name__)

DEFAULT_DECAY = 0.5
DEFAULT_MAX_HOPS = 5
DEFAULT_ITERATIONS = 20
DEFAULT_CONVERGENCE_THRESHOLD = 1e-5


class RiskPropagationEngine:
    """Structured graph diffusion mechanism for fraud risk propagation.

    Performs iterative belief propagation from seeded fraud nodes outward,
    applying confidence decay over multi-hop paths. Combines personalized
    PageRank-style diffusion with edge-weight-aware attenuation.

    Key properties:
        - Seeded nodes anchor risk at their initial scores
        - Risk decays exponentially with graph distance
        - Edge weights modulate propagation strength
        - Convergence is guaranteed by damping
    """

    def __init__(
        self,
        decay_factor: float = DEFAULT_DECAY,
        max_hops: int = DEFAULT_MAX_HOPS,
        max_iterations: int = DEFAULT_ITERATIONS,
        convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
    ) -> None:
        self.decay_factor = decay_factor
        self.max_hops = max_hops
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def propagate(
        self,
        graph: nx.DiGraph,
        seed_scores: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Run diffusion from seed nodes through the graph.

        Args:
            graph: Transaction graph with node/edge attributes.
            seed_scores: Initial risk scores for seed nodes (e.g., known fraud = 1.0).
                If None, automatically seeds from nodes with is_fraud=True.

        Returns:
            Propagated risk scores for all nodes in [0, 1].
        """
        nodes = list(graph.nodes())
        if not nodes:
            return {}

        node_idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)

        seeds = seed_scores or self._auto_seed(graph)
        if not seeds:
            return {nd: 0.0 for nd in nodes}

        scores = np.zeros(n, dtype=np.float64)
        seed_mask = np.zeros(n, dtype=bool)

        for node, score in seeds.items():
            if node in node_idx:
                idx = node_idx[node]
                scores[idx] = score
                seed_mask[idx] = True

        transition = self._build_transition_matrix(graph, nodes, node_idx)

        for iteration in range(self.max_iterations):
            prev_scores = scores.copy()

            propagated = transition @ scores
            propagated *= self.decay_factor

            new_scores = np.maximum(propagated, scores)
            new_scores[seed_mask] = np.maximum(
                new_scores[seed_mask],
                np.array([seeds.get(nodes[i], 0.0) for i in range(n)])[seed_mask],
            )

            scores = np.clip(new_scores, 0.0, 1.0)

            delta = np.max(np.abs(scores - prev_scores))
            if delta < self.convergence_threshold:
                logger.debug("propagation_converged", iteration=iteration, delta=delta)
                break

        scores = self._apply_hop_decay(graph, nodes, node_idx, seeds, scores)

        result = {nodes[i]: round(float(scores[i]), 6) for i in range(n)}

        n_affected = sum(1 for s in result.values() if s > 0.01)
        logger.info(
            "risk_propagation_complete",
            total_nodes=n,
            seed_nodes=len(seeds),
            affected_nodes=n_affected,
        )
        return result

    def _auto_seed(self, graph: nx.DiGraph) -> dict[str, float]:
        seeds: dict[str, float] = {}
        for node, data in graph.nodes(data=True):
            if data.get("is_fraud", False):
                seeds[node] = 1.0
        return seeds

    def _build_transition_matrix(
        self,
        graph: nx.DiGraph,
        nodes: list[str],
        node_idx: dict[str, int],
    ) -> np.ndarray:
        """Build a column-normalized transition matrix with edge-weight awareness.

        Uses sparse representation internally for memory efficiency on large graphs,
        converting to dense only for the final matrix-vector products.
        """
        from scipy import sparse

        n = len(nodes)
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for u, v, data in graph.edges(data=True):
            if u in node_idx and v in node_idx:
                weight = data.get("weight", 1.0)
                freq = data.get("frequency", 1)
                edge_strength = np.log1p(weight) * np.log1p(freq)

                i, j = node_idx[v], node_idx[u]
                rows.append(i)
                cols.append(j)
                vals.append(edge_strength)

                ri, rj = node_idx[u], node_idx[v]
                rows.append(ri)
                cols.append(rj)
                vals.append(edge_strength * 0.5)

        W = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()

        # For duplicate entries (same row/col added twice), sum them
        W = W.toarray()

        col_sums = W.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        W = W / col_sums

        return W

    def _apply_hop_decay(
        self,
        graph: nx.DiGraph,
        nodes: list[str],
        node_idx: dict[str, int],
        seeds: dict[str, float],
        scores: np.ndarray,
    ) -> np.ndarray:
        """Apply additional decay based on shortest-path distance from seeds."""
        import networkx as _nx

        undirected = graph.to_undirected()

        seed_nodes = [n for n in seeds if n in node_idx]
        if not seed_nodes:
            return scores

        min_distances = np.full(len(nodes), self.max_hops + 1, dtype=np.float64)

        for seed in seed_nodes:
            try:
                lengths = _nx.single_source_shortest_path_length(
                    undirected, seed, cutoff=self.max_hops
                )
                for target, dist in lengths.items():
                    if target in node_idx:
                        idx = node_idx[target]
                        min_distances[idx] = min(min_distances[idx], dist)
            except Exception:
                continue

        decay_multiplier = np.where(
            min_distances <= self.max_hops,
            self.decay_factor**min_distances,
            0.0,
        )

        seed_indices = {node_idx[s] for s in seed_nodes}
        for i in range(len(nodes)):
            if i not in seed_indices:
                scores[i] = scores[i] * decay_multiplier[i]

        return np.clip(scores, 0.0, 1.0)
