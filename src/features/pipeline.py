from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
import polars as pl
import structlog

from src.features.behavioral import BehavioralFeatureExtractor

if TYPE_CHECKING:
    import networkx as nx
from src.features.fraud_specific import FraudSpecificFeatureExtractor
from src.features.structural import StructuralFeatureExtractor

logger = structlog.get_logger(__name__)


class FeatureEngineeringPipeline:
    def __init__(self) -> None:
        self.structural_extractor = StructuralFeatureExtractor()
        self.behavioral_extractor = BehavioralFeatureExtractor()
        self.fraud_extractor = FraudSpecificFeatureExtractor()

    def run(
        self,
        graph: nx.DiGraph,
        transactions: pl.DataFrame,
        accounts: pl.DataFrame,
        graph_embeddings: dict[str, np.ndarray] | None = None,
        sequence_embeddings: dict[str, np.ndarray] | None = None,
        propagation_scores: dict[str, float] | None = None,
    ) -> pl.DataFrame:
        logger.info("feature_pipeline_start")

        structural = self.structural_extractor.extract(graph)
        behavioral = self.behavioral_extractor.extract(transactions)
        fraud_specific = self.fraud_extractor.extract(graph, transactions, accounts)

        features = structural.join(behavioral, on="account_id", how="left")
        features = features.join(fraud_specific, on="account_id", how="left")

        if graph_embeddings:
            emb_df = self._embeddings_to_df(graph_embeddings, prefix="gnn_emb")
            features = features.join(emb_df, on="account_id", how="left")
            logger.info("graph_embeddings_added", dim=len(emb_df.columns) - 1)

        if sequence_embeddings:
            seq_df = self._embeddings_to_df(sequence_embeddings, prefix="seq_emb")
            features = features.join(seq_df, on="account_id", how="left")
            logger.info("sequence_embeddings_added", dim=len(seq_df.columns) - 1)

        if propagation_scores:
            prop_df = pl.DataFrame(
                {
                    "account_id": list(propagation_scores.keys()),
                    "advanced_propagation_score": list(propagation_scores.values()),
                }
            )
            features = features.join(prop_df, on="account_id", how="left")
            logger.info("propagation_scores_added")

        features = features.fill_null(0.0)
        features = features.fill_nan(0.0)

        logger.info(
            "feature_pipeline_complete",
            total_features=features.width - 1,
            accounts=features.height,
        )
        return features

    def _embeddings_to_df(self, embeddings: dict[str, np.ndarray], prefix: str) -> pl.DataFrame:
        nodes = sorted(embeddings.keys())
        if not nodes:
            return pl.DataFrame(schema={"account_id": pl.Utf8})

        dim = len(next(iter(embeddings.values())))
        data: dict[str, list[object]] = {"account_id": list(nodes)}
        for d in range(dim):
            data[f"{prefix}_{d}"] = [
                float(embeddings[n][d]) if d < len(embeddings[n]) else 0.0 for n in nodes
            ]

        return pl.DataFrame(data)

    def get_feature_names(self) -> list[str]:
        return [
            "in_degree",
            "out_degree",
            "total_degree",
            "weighted_in_degree",
            "weighted_out_degree",
            "degree_ratio",
            "betweenness",
            "pagerank",
            "clustering_coefficient",
            "k_core_number",
            "scc_size",
            "community_id",
            "subgraph_density",
            "incoming_volume",
            "avg_incoming_amount",
            "max_incoming_amount",
            "incoming_txn_count",
            "incoming_counterparty_diversity",
            "outgoing_volume",
            "avg_outgoing_amount",
            "max_outgoing_amount",
            "outgoing_txn_count",
            "outgoing_counterparty_diversity",
            "net_balance_ratio",
            "velocity_score",
            "receive_to_send_delay_hours",
            "burst_index",
            "dormancy_score",
            "repeat_transfer_ratio",
            "circularity_score",
            "funnel_score",
            "layering_depth",
            "risk_propagation_score",
            "device_cluster_score",
            "suspicious_chain_length",
        ]
