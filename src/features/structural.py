from __future__ import annotations

import networkx as nx
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class StructuralFeatureExtractor:
    def extract(self, graph: nx.DiGraph) -> pl.DataFrame:
        logger.info("extracting_structural_features", nodes=graph.number_of_nodes())

        nodes = list(graph.nodes())
        if not nodes:
            return pl.DataFrame(schema={"account_id": pl.Utf8})

        in_degree = dict(graph.in_degree())
        out_degree = dict(graph.out_degree())

        weighted_in = dict(graph.in_degree(weight="weight"))
        weighted_out = dict(graph.out_degree(weight="weight"))

        # Use a view to avoid copying the full graph (~2-3 GB for 50k/1.6M)
        undirected = graph.to_undirected(as_view=True)
        try:
            betweenness = nx.betweenness_centrality(graph, k=min(500, len(nodes)))
        except Exception:
            betweenness = {n: 0.0 for n in nodes}

        try:
            pagerank = nx.pagerank(graph, max_iter=100, tol=1e-06)
        except Exception:
            pagerank = {n: 1.0 / len(nodes) for n in nodes}

        try:
            clustering = nx.clustering(undirected)
        except Exception:
            clustering = {n: 0.0 for n in nodes}

        try:
            core_numbers = nx.core_number(undirected)
        except Exception:
            core_numbers = {n: 0 for n in nodes}

        scc_map = self._compute_scc_sizes(graph)
        community_map = self._detect_communities(undirected)
        density_map = self._compute_subgraph_density(graph, community_map)

        records: list[dict[str, object]] = []
        for node in nodes:
            ind = in_degree.get(node, 0)
            outd = out_degree.get(node, 0)
            total_degree = ind + outd
            degree_ratio = outd / ind if ind > 0 else float(outd) if outd > 0 else 0.0

            records.append(
                {
                    "account_id": node,
                    "in_degree": ind,
                    "out_degree": outd,
                    "total_degree": total_degree,
                    "weighted_in_degree": weighted_in.get(node, 0.0),
                    "weighted_out_degree": weighted_out.get(node, 0.0),
                    "degree_ratio": round(degree_ratio, 6),
                    "betweenness": round(betweenness.get(node, 0.0), 8),
                    "pagerank": round(pagerank.get(node, 0.0), 8),
                    "clustering_coefficient": round(clustering.get(node, 0.0), 6),
                    "k_core_number": core_numbers.get(node, 0),
                    "scc_size": scc_map.get(node, 1),
                    "community_id": community_map.get(node, -1),
                    "subgraph_density": round(density_map.get(node, 0.0), 6),
                }
            )

        df = pl.DataFrame(records)
        logger.info("structural_features_extracted", features=df.width, rows=df.height)
        return df

    def _compute_scc_sizes(self, graph: nx.DiGraph) -> dict[str, int]:
        scc_map: dict[str, int] = {}
        for component in nx.strongly_connected_components(graph):
            size = len(component)
            for node in component:
                scc_map[node] = size
        return scc_map

    def _detect_communities(self, undirected: nx.Graph) -> dict[str, int]:
        try:
            communities = nx.community.louvain_communities(undirected, seed=42)
            community_map: dict[str, int] = {}
            for idx, comm in enumerate(communities):
                for node in comm:
                    community_map[node] = idx
            return community_map
        except Exception:
            return {n: 0 for n in undirected.nodes()}

    def _compute_subgraph_density(
        self,
        graph: nx.DiGraph,
        community_map: dict[str, int],
    ) -> dict[str, float]:
        communities: dict[int, list[str]] = {}
        for node, cid in community_map.items():
            communities.setdefault(cid, []).append(node)

        density_map: dict[str, float] = {}
        for _cid, members in communities.items():
            subgraph = graph.subgraph(members)
            density = nx.density(subgraph)
            for node in members:
                density_map[node] = density

        return density_map
