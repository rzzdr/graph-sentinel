from __future__ import annotations

import time

import networkx as nx
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class FraudSpecificFeatureExtractor:
    def extract(
        self,
        graph: nx.DiGraph,
        transactions: pl.DataFrame,
        accounts: pl.DataFrame,
    ) -> pl.DataFrame:
        logger.info("extracting_fraud_specific_features")

        nodes = list(graph.nodes())
        if not nodes:
            return pl.DataFrame(schema={"account_id": pl.Utf8})

        t0 = time.perf_counter()
        circularity = self._compute_circularity_scores(graph)
        logger.info("circularity_done", elapsed_s=round(time.perf_counter() - t0, 2))

        t0 = time.perf_counter()
        funnel = self._compute_funnel_scores(graph)
        logger.info("funnel_done", elapsed_s=round(time.perf_counter() - t0, 2))

        t0 = time.perf_counter()
        layering = self._compute_layering_depth(graph)
        logger.info("layering_done", elapsed_s=round(time.perf_counter() - t0, 2))

        t0 = time.perf_counter()
        risk_prop = self._compute_risk_propagation(graph)
        logger.info("risk_propagation_done", elapsed_s=round(time.perf_counter() - t0, 2))

        t0 = time.perf_counter()
        device_cluster = self._compute_device_cluster_scores(accounts)
        logger.info("device_cluster_done", elapsed_s=round(time.perf_counter() - t0, 2))

        t0 = time.perf_counter()
        chain_lengths = self._compute_suspicious_chain_length(graph)
        logger.info("chain_length_done", elapsed_s=round(time.perf_counter() - t0, 2))

        records: list[dict[str, object]] = []
        for node in nodes:
            records.append(
                {
                    "account_id": node,
                    "circularity_score": circularity.get(node, 0.0),
                    "funnel_score": funnel.get(node, 0.0),
                    "layering_depth": layering.get(node, 0),
                    "risk_propagation_score": risk_prop.get(node, 0.0),
                    "device_cluster_score": device_cluster.get(node, 0.0),
                    "suspicious_chain_length": chain_lengths.get(node, 0),
                }
            )

        df = pl.DataFrame(records)
        logger.info("fraud_specific_features_extracted", rows=df.height)
        return df

    def _compute_circularity_scores(
        self, graph: nx.DiGraph, *, max_cycles: int = 50_000, timeout_s: float = 120.0
    ) -> dict[str, float]:
        scores: dict[str, float] = {n: 0.0 for n in graph.nodes()}

        try:
            cycle_participation: dict[str, int] = {}
            deadline = time.perf_counter() + timeout_s
            for count, cycle in enumerate(nx.simple_cycles(graph, length_bound=8), 1):
                for node in cycle:
                    cycle_participation[node] = cycle_participation.get(node, 0) + 1
                if count >= max_cycles:
                    logger.warning("circularity_cycle_cap_reached", max_cycles=max_cycles)
                    break
                if count % 5_000 == 0:
                    if time.perf_counter() > deadline:
                        logger.warning("circularity_timeout", cycles_found=count)
                        break
                    logger.debug("circularity_progress", cycles_found=count)
        except Exception:
            return scores

        if not cycle_participation:
            return scores

        max_part = max(cycle_participation.values())
        for node, cnt in cycle_participation.items():
            scores[node] = cnt / max_part if max_part > 0 else 0.0

        return scores

    def _compute_funnel_scores(self, graph: nx.DiGraph) -> dict[str, float]:
        scores: dict[str, float] = {}

        for node in graph.nodes():
            in_deg = graph.in_degree(node)
            out_deg = graph.out_degree(node)

            if in_deg == 0 and out_deg == 0:
                scores[node] = 0.0
                continue

            in_ratio = in_deg / (in_deg + out_deg) if (in_deg + out_deg) > 0 else 0
            out_weight = sum(d.get("weight", 0) for _, _, d in graph.out_edges(node, data=True))
            in_weight = sum(d.get("weight", 0) for _, _, d in graph.in_edges(node, data=True))

            concentration = 0.0
            if in_weight > 0 and out_weight > 0:
                concentration = min(in_weight, out_weight) / max(in_weight, out_weight)

            funnel_score = in_ratio * concentration * min(1.0, in_deg / 5.0)
            scores[node] = round(funnel_score, 6)

        return scores

    def _compute_layering_depth(
        self, graph: nx.DiGraph, *, timeout_s: float = 120.0
    ) -> dict[str, int]:
        depths: dict[str, int] = {n: 0 for n in graph.nodes()}
        candidates = [
            n for n in graph.nodes() if graph.in_degree(n) > 0 and graph.out_degree(n) > 0
        ]
        deadline = time.perf_counter() + timeout_s

        for processed, node in enumerate(candidates, 1):
            try:
                max_path = 0
                for successor in graph.successors(node):
                    for path in nx.all_simple_paths(graph, node, successor, cutoff=7):
                        max_path = max(max_path, len(path) - 1)
                        if max_path >= 7:
                            break
                    if max_path >= 7:
                        break
                depths[node] = max_path
            except (nx.NetworkXError, StopIteration):
                pass

            if processed % 2_000 == 0:
                if time.perf_counter() > deadline:
                    logger.warning("layering_timeout", processed=processed, total=len(candidates))
                    break
                logger.debug("layering_progress", processed=processed, total=len(candidates))

        return depths

    def _compute_risk_propagation(self, graph: nx.DiGraph) -> dict[str, float]:
        """Propagate risk from known fraud nodes through the network."""
        scores: dict[str, float] = {}
        fraud_nodes = {n for n, d in graph.nodes(data=True) if d.get("is_fraud", False)}

        if not fraud_nodes:
            return {n: 0.0 for n in graph.nodes()}

        for node in graph.nodes():
            if node in fraud_nodes:
                scores[node] = 1.0
                continue

            risk = 0.0
            neighbors = set(graph.predecessors(node)) | set(graph.successors(node))
            fraud_neighbors = neighbors & fraud_nodes

            if neighbors:
                risk = len(fraud_neighbors) / len(neighbors)

                for fn in fraud_neighbors:
                    if graph.has_edge(fn, node):
                        edge_weight = graph[fn][node].get("weight", 0)
                        total_out = sum(
                            d.get("weight", 0) for _, _, d in graph.out_edges(fn, data=True)
                        )
                        if total_out > 0:
                            risk += 0.3 * (edge_weight / total_out)

            scores[node] = round(min(1.0, risk), 6)

        return scores

    def _compute_device_cluster_scores(self, accounts: pl.DataFrame) -> dict[str, float]:
        device_counts = accounts.group_by("device_fingerprint_id").agg(
            pl.len().alias("device_count")
        )

        shared = device_counts.filter(pl.col("device_count") > 1)

        if shared.height == 0:
            return {}

        result = accounts.join(shared, on="device_fingerprint_id", how="inner").select(
            "account_id", "device_count"
        )

        max_count = result["device_count"].max()
        if max_count is None or int(max_count) <= 1:  # pyright: ignore[reportArgumentType]
            return {}

        scores: dict[str, float] = {}
        for row in result.iter_rows(named=True):
            scores[row["account_id"]] = round(row["device_count"] / max_count, 6)

        return scores

    def _compute_suspicious_chain_length(
        self, graph: nx.DiGraph, *, timeout_s: float = 60.0
    ) -> dict[str, int]:
        chain_lengths: dict[str, int] = {n: 0 for n in graph.nodes()}

        sources = [n for n in graph.nodes() if graph.in_degree(n) == 0 and graph.out_degree(n) > 0]
        sinks = [n for n in graph.nodes() if graph.out_degree(n) == 0 and graph.in_degree(n) > 0]

        deadline = time.perf_counter() + timeout_s
        pairs_checked = 0
        for source in sources[:50]:
            for sink in sinks[:50]:
                try:
                    for path in nx.all_simple_paths(graph, source, sink, cutoff=10):
                        path_len = len(path)
                        for node in path:
                            chain_lengths[node] = max(chain_lengths[node], path_len)
                        break
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                pairs_checked += 1
                if pairs_checked % 100 == 0 and time.perf_counter() > deadline:
                    logger.warning("chain_length_timeout", pairs_checked=pairs_checked)
                    return chain_lengths

        return chain_lengths
