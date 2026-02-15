from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    import networkx as nx

logger = structlog.get_logger(__name__)


class GraphDriftDetector:
    """Detects structural changes in the transaction graph over time.

    Monitors graph-level statistics (density, clustering, degree distribution,
    component structure) and flags significant shifts that may indicate
    emerging fraud patterns or data pipeline changes.
    """

    def __init__(self, significance: float = 0.05) -> None:
        self.significance = significance
        self.baseline_metrics: dict[str, float] | None = None

    def set_baseline(self, graph: nx.DiGraph) -> None:
        self.baseline_metrics = self._compute_graph_metrics(graph)
        logger.info("graph_drift_baseline_set", **self.baseline_metrics)

    def detect(self, current_graph: nx.DiGraph) -> dict[str, Any]:
        if self.baseline_metrics is None:
            raise RuntimeError("Set baseline before detecting graph drift")

        current_metrics = self._compute_graph_metrics(current_graph)

        drift_details: dict[str, Any] = {}
        drifted_count = 0

        for metric, current_val in current_metrics.items():
            baseline_val = self.baseline_metrics.get(metric, 0.0)

            if baseline_val == 0:
                relative_change = abs(current_val) if current_val != 0 else 0.0
            else:
                relative_change = abs(current_val - baseline_val) / abs(baseline_val)

            metric_drifted = relative_change > 0.3

            drift_details[metric] = {
                "baseline": round(baseline_val, 6),
                "current": round(current_val, 6),
                "relative_change": round(relative_change, 4),
                "drifted": metric_drifted,
            }

            if metric_drifted:
                drifted_count += 1

        overall_drifted = drifted_count >= 3

        result: dict[str, Any] = {
            "drifted": overall_drifted,
            "metrics_drifted": drifted_count,
            "total_metrics": len(drift_details),
            "details": drift_details,
        }

        logger.info(
            "graph_drift_detection_complete",
            drifted=overall_drifted,
            metrics_drifted=drifted_count,
        )
        return result

    def _compute_graph_metrics(self, graph: nx.DiGraph) -> dict[str, float]:
        import networkx as nx

        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()

        in_degrees = [d for _, d in graph.in_degree()]
        out_degrees = [d for _, d in graph.out_degree()]

        undirected = graph.to_undirected()

        metrics = {
            "node_count": float(n_nodes),
            "edge_count": float(n_edges),
            "density": nx.density(graph) if n_nodes > 1 else 0.0,
            "avg_in_degree": float(np.mean(in_degrees)) if in_degrees else 0.0,
            "avg_out_degree": float(np.mean(out_degrees)) if out_degrees else 0.0,
            "max_in_degree": float(max(in_degrees)) if in_degrees else 0.0,
            "max_out_degree": float(max(out_degrees)) if out_degrees else 0.0,
            "degree_std": float(np.std(in_degrees + out_degrees)) if in_degrees else 0.0,
        }

        try:
            metrics["avg_clustering"] = nx.average_clustering(undirected)
        except Exception:
            metrics["avg_clustering"] = 0.0

        try:
            sccs = list(nx.strongly_connected_components(graph))
            metrics["n_strongly_connected"] = float(len(sccs))
            metrics["largest_scc_ratio"] = float(max(len(c) for c in sccs) / max(n_nodes, 1))
        except Exception:
            metrics["n_strongly_connected"] = 0.0
            metrics["largest_scc_ratio"] = 0.0

        try:
            wccs = list(nx.weakly_connected_components(graph))
            metrics["n_weakly_connected"] = float(len(wccs))
        except Exception:
            metrics["n_weakly_connected"] = 0.0

        leaf_nodes = sum(1 for n in graph.nodes() if graph.out_degree(n) == 0)
        root_nodes = sum(1 for n in graph.nodes() if graph.in_degree(n) == 0)
        metrics["leaf_ratio"] = float(leaf_nodes / max(n_nodes, 1))
        metrics["root_ratio"] = float(root_nodes / max(n_nodes, 1))

        return metrics
