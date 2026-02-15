from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import structlog

from src.scoring.aggregator import ScoreAggregator

if TYPE_CHECKING:
    from src.config.settings import ScoringConfig

logger = structlog.get_logger(__name__)

BEHAVIORAL_FEATURES = [
    "incoming_volume",
    "outgoing_volume",
    "net_balance_ratio",
    "velocity_score",
    "receive_to_send_delay_hours",
    "burst_index",
    "dormancy_score",
    "incoming_counterparty_diversity",
    "outgoing_counterparty_diversity",
    "repeat_transfer_ratio",
]

STRUCTURAL_FEATURES = [
    "in_degree",
    "out_degree",
    "degree_ratio",
    "betweenness",
    "pagerank",
    "clustering_coefficient",
    "k_core_number",
    "scc_size",
    "subgraph_density",
]

PROPAGATION_FEATURES = [
    "circularity_score",
    "funnel_score",
    "layering_depth",
    "risk_propagation_score",
    "device_cluster_score",
    "suspicious_chain_length",
]


class RiskScoringEngine:
    def __init__(self, config: ScoringConfig) -> None:
        self.config = config
        self.aggregator = ScoreAggregator()
        self._meta_scorer: Any = None
        self._calibrator: Any = None

    def score(self, features: pl.DataFrame) -> pl.DataFrame:
        """Score accounts using static weighted aggregation (backward compatible).

        For the full learned pipeline, use score_with_signals() instead.
        """
        logger.info("risk_scoring_start", accounts=features.height)

        behavioral_score = self._compute_component_score(features, BEHAVIORAL_FEATURES)
        structural_score = self._compute_component_score(features, STRUCTURAL_FEATURES)
        propagation_score = self._compute_component_score(features, PROPAGATION_FEATURES)

        result = features.select("account_id").with_columns(
            pl.Series("behavioral_risk_score", behavioral_score),
            pl.Series("structural_risk_score", structural_score),
            pl.Series("network_propagation_score", propagation_score),
        )

        final_scores = self.aggregator.aggregate(
            behavioral_scores=behavioral_score,
            structural_scores=structural_score,
            propagation_scores=propagation_score,
            weights=(
                self.config.behavioral_weight,
                self.config.structural_weight,
                self.config.propagation_weight,
            ),
        )

        risk_series = pl.Series("risk_score", final_scores)
        result = result.with_columns(
            risk_series,
            risk_series.map_elements(
                lambda s: self._classify_risk(s),
                return_dtype=pl.Utf8,
            ).alias("risk_level"),
        )

        alert_count = result.filter(pl.col("risk_score") >= self.config.alert_threshold).height

        logger.info(
            "risk_scoring_complete",
            accounts=result.height,
            alerts=alert_count,
            mean_score=round(float(result["risk_score"].mean()), 2),
        )
        return result

    def score_with_signals(
        self,
        features: pl.DataFrame,
        propagation_scores: dict[str, float] | None = None,
        graph_scores: dict[str, float] | None = None,
        sequence_anomaly_scores: dict[str, float] | None = None,
        supervised_probs: np.ndarray | None = None,
        unsupervised_scores: np.ndarray | None = None,
    ) -> pl.DataFrame:
        """Full scoring pipeline using meta-model and calibration when available.

        Assembles all signal sources into a unified signal DataFrame, then:
        1. If meta-model is fitted → uses learned blending
        2. If calibrator is fitted → applies calibration
        3. Otherwise → falls back to static weighted aggregation
        """
        logger.info("multi_signal_scoring_start", accounts=features.height)

        behavioral_score = self._compute_component_score(features, BEHAVIORAL_FEATURES)
        structural_score = self._compute_component_score(features, STRUCTURAL_FEATURES)
        fraud_specific_score = self._compute_component_score(features, PROPAGATION_FEATURES)

        account_ids = features["account_id"].to_list()

        signal_data: dict[str, list[float]] = {
            "account_id": account_ids,
            "sig_behavioral": behavioral_score,
            "sig_structural": structural_score,
            "sig_fraud_specific": fraud_specific_score,
        }

        if propagation_scores:
            signal_data["sig_propagation"] = [
                propagation_scores.get(aid, 0.0) for aid in account_ids
            ]

        if graph_scores:
            signal_data["sig_graph"] = [graph_scores.get(aid, 0.0) for aid in account_ids]

        if sequence_anomaly_scores:
            signal_data["sig_sequence_anomaly"] = [
                sequence_anomaly_scores.get(aid, 0.0) for aid in account_ids
            ]

        if supervised_probs is not None:
            signal_data["sig_supervised"] = supervised_probs.tolist()

        if unsupervised_scores is not None:
            signal_data["sig_unsupervised"] = unsupervised_scores.tolist()

        signals_df = pl.DataFrame(signal_data)

        if self._meta_scorer is not None and self._meta_scorer._fitted:
            raw_probs = self._meta_scorer.predict(signals_df)

            if self._calibrator is not None and self._calibrator._fitted:
                calibrated = self._calibrator.calibrate(raw_probs)
                final_scores_arr = self._calibrator.to_risk_score(calibrated)
            else:
                final_scores_arr = raw_probs * 100.0

            final_scores = np.clip(final_scores_arr, 0, 100)
            final_list = [round(float(s), 2) for s in final_scores]
        else:
            prop_list = signal_data.get("sig_propagation", fraud_specific_score)
            final_list = self.aggregator.aggregate(
                behavioral_scores=behavioral_score,
                structural_scores=structural_score,
                propagation_scores=prop_list,
                weights=(
                    self.config.behavioral_weight,
                    self.config.structural_weight,
                    self.config.propagation_weight,
                ),
            )

        result = features.select("account_id").with_columns(
            pl.Series("behavioral_risk_score", behavioral_score),
            pl.Series("structural_risk_score", structural_score),
            pl.Series("network_propagation_score", fraud_specific_score),
        )

        if propagation_scores:
            result = result.with_columns(
                pl.Series(
                    "advanced_propagation_score",
                    [propagation_scores.get(aid, 0.0) for aid in account_ids],
                )
            )

        risk_series = pl.Series("risk_score", final_list)
        result = result.with_columns(
            risk_series,
            risk_series.map_elements(
                lambda s: self._classify_risk(s),
                return_dtype=pl.Utf8,
            ).alias("risk_level"),
        )

        alert_count = result.filter(pl.col("risk_score") >= self.config.alert_threshold).height
        logger.info(
            "multi_signal_scoring_complete",
            accounts=result.height,
            alerts=alert_count,
            mean_score=round(float(result["risk_score"].mean()), 2),
            meta_model_active=self._meta_scorer is not None and self._meta_scorer._fitted,
            calibration_active=self._calibrator is not None and self._calibrator._fitted,
        )
        return result

    def set_meta_scorer(self, meta_scorer: Any) -> None:
        self._meta_scorer = meta_scorer

    def set_calibrator(self, calibrator: Any) -> None:
        self._calibrator = calibrator

    def _compute_component_score(
        self,
        features: pl.DataFrame,
        feature_names: list[str],
    ) -> list[float]:
        available = [f for f in feature_names if f in features.columns]
        if not available:
            return [0.0] * features.height

        subset = features.select(available).to_numpy()

        from sklearn.preprocessing import RobustScaler

        scaler = RobustScaler()
        scaled = scaler.fit_transform(subset)

        scores = np.mean(scaled, axis=1)

        min_val, max_val = scores.min(), scores.max()
        if max_val - min_val > 1e-8:
            scores = (scores - min_val) / (max_val - min_val) * 100
        else:
            scores = np.zeros_like(scores)

        return [round(float(s), 2) for s in scores]

    def _classify_risk(self, score: float) -> str:
        if score >= self.config.high_risk_threshold:
            return "critical"
        if score >= self.config.alert_threshold:
            return "high"
        if score >= 50.0:
            return "medium"
        if score >= 25.0:
            return "low"
        return "minimal"
