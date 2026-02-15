from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    import networkx as nx
    import polars as pl

logger = structlog.get_logger(__name__)


class GraphVisualizer:
    def visualize_subgraph(
        self,
        graph: nx.DiGraph,
        center_node: str,
        depth: int = 2,
        output_path: Path | None = None,
        risk_scores: pl.DataFrame | None = None,
    ) -> Path:
        from pyvis.network import Network

        subgraph_nodes = {center_node}
        current_layer = {center_node}

        for _ in range(depth):
            next_layer: set[str] = set()
            for node in current_layer:
                next_layer.update(graph.successors(node))
                next_layer.update(graph.predecessors(node))
            subgraph_nodes.update(next_layer)
            current_layer = next_layer

        subgraph = graph.subgraph(subgraph_nodes)

        risk_map: dict[str, float] = {}
        if risk_scores is not None:
            for row in risk_scores.iter_rows(named=True):
                risk_map[row["account_id"]] = row.get("risk_score", 0.0)

        net = Network(
            height="700px",
            width="100%",
            directed=True,
            notebook=False,
            bgcolor="#1a1a2e",
            font_color="white",
        )

        for node in subgraph.nodes():
            risk = risk_map.get(node, 0.0)
            color = self._risk_to_color(risk)
            is_fraud = graph.nodes[node].get("is_fraud", False)
            label = f"{node[:8]}..."
            title = (
                f"ID: {node}\n"
                f"Risk: {risk:.1f}\n"
                f"Fraud: {is_fraud}\n"
                f"Type: {graph.nodes[node].get('account_type', 'unknown')}"
            )

            size = 15 + risk * 0.3
            if node == center_node:
                size = 40

            net.add_node(
                node,
                label=label,
                title=title,
                color=color,
                size=size,
                borderWidth=3 if node == center_node else 1,
            )

        for u, v, data in subgraph.edges(data=True):
            weight = data.get("weight", 1)
            freq = data.get("frequency", 1)
            net.add_edge(
                u,
                v,
                value=min(10, freq),
                title=f"Amount: {weight:.2f}\nFrequency: {freq}",
                color="#7f8c8d",
            )

        if output_path is None:
            output_path = Path(f"outputs/graph_{center_node[:8]}.html")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        net.write_html(str(output_path))
        logger.info("subgraph_visualized", center=center_node, nodes=len(subgraph_nodes))
        return output_path

    def visualize_fraud_network(
        self,
        graph: nx.DiGraph,
        output_path: Path,
        max_nodes: int = 500,
    ) -> Path:
        from pyvis.network import Network

        fraud_nodes = {n for n, d in graph.nodes(data=True) if d.get("is_fraud", False)}

        display_nodes = set(fraud_nodes)
        for node in fraud_nodes:
            display_nodes.update(list(graph.successors(node))[:5])
            display_nodes.update(list(graph.predecessors(node))[:5])

        if len(display_nodes) > max_nodes:
            display_nodes = set(list(display_nodes)[:max_nodes])

        subgraph = graph.subgraph(display_nodes)

        net = Network(
            height="800px",
            width="100%",
            directed=True,
            notebook=False,
            bgcolor="#1a1a2e",
            font_color="white",
        )

        for node in subgraph.nodes():
            is_fraud = node in fraud_nodes
            color = "#e74c3c" if is_fraud else "#3498db"
            role = graph.nodes[node].get("fraud_role", "normal")

            net.add_node(
                node,
                label=f"{node[:6]}",
                title=f"ID: {node}\nRole: {role}\nFraud: {is_fraud}",
                color=color,
                size=20 if is_fraud else 10,
            )

        for u, v, _ in subgraph.edges(data=True):
            net.add_edge(u, v, color="#e74c3c" if u in fraud_nodes else "#95a5a6")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        net.write_html(str(output_path))
        logger.info("fraud_network_visualized", nodes=subgraph.number_of_nodes())
        return output_path

    def _risk_to_color(self, risk: float) -> str:
        if risk >= 90:
            return "#8e44ad"
        if risk >= 75:
            return "#e74c3c"
        if risk >= 50:
            return "#e67e22"
        if risk >= 25:
            return "#f39c12"
        return "#2ecc71"
