from __future__ import annotations

import networkx as nx
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class GraphBuilder:
    def build_from_transactions(
        self,
        transactions: pl.DataFrame,
        accounts: pl.DataFrame | None = None,
    ) -> nx.DiGraph:
        logger.info("building_graph", transactions=transactions.height)

        G = nx.DiGraph()

        if accounts is not None:
            for row in accounts.iter_rows(named=True):
                G.add_node(
                    row["account_id"],
                    account_type=row.get("account_type", "unknown"),
                    kyc_level=row.get("kyc_level", "unknown"),
                    region=row.get("geographic_region", "unknown"),
                    is_fraud=row.get("is_fraud", False),
                    fraud_role=row.get("fraud_role", "normal"),
                )

        edge_agg = transactions.group_by(["sender_id", "receiver_id"]).agg(
            pl.len().alias("frequency"),
            pl.col("amount").sum().alias("total_amount"),
            pl.col("amount").mean().alias("avg_amount"),
            pl.col("amount").min().alias("min_amount"),
            pl.col("amount").max().alias("max_amount"),
            pl.col("timestamp").min().alias("first_txn"),
            pl.col("timestamp").max().alias("last_txn"),
        )

        for row in edge_agg.iter_rows(named=True):
            first_txn = row["first_txn"]
            last_txn = row["last_txn"]
            freq = row["frequency"]

            if freq > 1 and first_txn != last_txn:
                span = (last_txn - first_txn).total_seconds()
                avg_gap = span / (freq - 1) if freq > 1 else 0
            else:
                avg_gap = 0.0

            G.add_edge(
                row["sender_id"],
                row["receiver_id"],
                weight=row["total_amount"],
                frequency=freq,
                avg_amount=row["avg_amount"],
                min_amount=row["min_amount"],
                max_amount=row["max_amount"],
                first_txn=first_txn,
                last_txn=last_txn,
                avg_inter_txn_gap=avg_gap,
            )

        logger.info("graph_built", nodes=G.number_of_nodes(), edges=G.number_of_edges())
        return G

    def build_incremental(
        self,
        existing_graph: nx.DiGraph,
        new_transactions: pl.DataFrame,
    ) -> nx.DiGraph:
        edge_agg = new_transactions.group_by(["sender_id", "receiver_id"]).agg(
            pl.len().alias("frequency"),
            pl.col("amount").sum().alias("total_amount"),
            pl.col("amount").mean().alias("avg_amount"),
            pl.col("timestamp").max().alias("last_txn"),
        )

        for row in edge_agg.iter_rows(named=True):
            s, r = row["sender_id"], row["receiver_id"]

            if existing_graph.has_edge(s, r):
                data = existing_graph[s][r]
                data["weight"] += row["total_amount"]
                data["frequency"] += row["frequency"]
                data["avg_amount"] = data["weight"] / data["frequency"]
                data["last_txn"] = max(data["last_txn"], row["last_txn"])
            else:
                existing_graph.add_edge(
                    s,
                    r,
                    weight=row["total_amount"],
                    frequency=row["frequency"],
                    avg_amount=row["avg_amount"],
                    last_txn=row["last_txn"],
                )

        return existing_graph
