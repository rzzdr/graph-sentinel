from __future__ import annotations

import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class BehavioralFeatureExtractor:
    def extract(self, transactions: pl.DataFrame) -> pl.DataFrame:
        logger.info("extracting_behavioral_features", transactions=transactions.height)

        incoming = self._compute_incoming_features(transactions)
        outgoing = self._compute_outgoing_features(transactions)
        timing = self._compute_timing_features(transactions)

        features = incoming.join(outgoing, on="account_id", how="outer_coalesce")
        features = features.join(timing, on="account_id", how="left")

        features = features.with_columns(
            (
                (pl.col("incoming_volume") - pl.col("outgoing_volume"))
                / (pl.col("incoming_volume") + pl.col("outgoing_volume") + 1e-8)
            ).alias("net_balance_ratio"),
            (pl.col("outgoing_volume") / (pl.col("incoming_volume") + 1e-8)).alias(
                "velocity_score"
            ),
        )

        features = features.fill_null(0.0)
        logger.info("behavioral_features_extracted", features=features.width, rows=features.height)
        return features

    def _compute_incoming_features(self, transactions: pl.DataFrame) -> pl.DataFrame:
        return (
            transactions.group_by("receiver_id")
            .agg(
                pl.col("amount").sum().alias("incoming_volume"),
                pl.col("amount").mean().alias("avg_incoming_amount"),
                pl.col("amount").max().alias("max_incoming_amount"),
                pl.len().alias("incoming_txn_count"),
                pl.col("sender_id").n_unique().alias("incoming_counterparty_diversity"),
            )
            .rename({"receiver_id": "account_id"})
        )

    def _compute_outgoing_features(self, transactions: pl.DataFrame) -> pl.DataFrame:
        return (
            transactions.group_by("sender_id")
            .agg(
                pl.col("amount").sum().alias("outgoing_volume"),
                pl.col("amount").mean().alias("avg_outgoing_amount"),
                pl.col("amount").max().alias("max_outgoing_amount"),
                pl.len().alias("outgoing_txn_count"),
                pl.col("receiver_id").n_unique().alias("outgoing_counterparty_diversity"),
            )
            .rename({"sender_id": "account_id"})
        )

    def _compute_timing_features(self, transactions: pl.DataFrame) -> pl.DataFrame:
        incoming_times = (
            transactions.group_by("receiver_id")
            .agg(
                pl.col("timestamp").min().alias("first_incoming"),
                pl.col("timestamp").max().alias("last_incoming"),
            )
            .rename({"receiver_id": "account_id"})
        )

        outgoing_times = (
            transactions.group_by("sender_id")
            .agg(
                pl.col("timestamp").min().alias("first_outgoing"),
                pl.col("timestamp").max().alias("last_outgoing"),
            )
            .rename({"sender_id": "account_id"})
        )

        timing = incoming_times.join(outgoing_times, on="account_id", how="outer_coalesce")

        timing = timing.with_columns(
            (
                (pl.col("first_outgoing") - pl.col("last_incoming"))
                .dt.total_seconds()
                .clip(lower_bound=0)
                / 3600.0
            ).alias("receive_to_send_delay_hours"),
        )

        all_txns = pl.concat(
            [
                transactions.select(
                    pl.col("sender_id").alias("account_id"),
                    pl.col("timestamp"),
                ),
                transactions.select(
                    pl.col("receiver_id").alias("account_id"),
                    pl.col("timestamp"),
                ),
            ]
        )

        burst_features = (
            all_txns.sort("timestamp")
            .group_by("account_id")
            .agg(
                pl.col("timestamp").diff().dt.total_seconds().std().alias("txn_interval_std"),
                pl.col("timestamp").diff().dt.total_seconds().mean().alias("txn_interval_mean"),
            )
        )

        burst_features = burst_features.with_columns(
            (pl.col("txn_interval_std") / (pl.col("txn_interval_mean") + 1e-8)).alias(
                "burst_index"
            ),
        )

        timing = timing.join(
            burst_features.select("account_id", "burst_index"),
            on="account_id",
            how="left",
        )

        dormancy = self._compute_dormancy(all_txns)
        timing = timing.join(dormancy, on="account_id", how="left")

        repeat_ratio = self._compute_repeat_ratio(transactions)
        timing = timing.join(repeat_ratio, on="account_id", how="left")

        return timing.select(
            "account_id",
            "receive_to_send_delay_hours",
            "burst_index",
            "dormancy_score",
            "repeat_transfer_ratio",
        ).fill_null(0.0)

    def _compute_dormancy(self, all_txns: pl.DataFrame) -> pl.DataFrame:
        return (
            all_txns.sort("timestamp")
            .group_by("account_id")
            .agg(
                pl.col("timestamp").diff().dt.total_seconds().max().alias("max_gap_seconds"),
                (pl.col("timestamp").max() - pl.col("timestamp").min())
                .dt.total_seconds()
                .alias("active_span_seconds"),
            )
            .with_columns(
                (pl.col("max_gap_seconds") / (pl.col("active_span_seconds") + 1e-8)).alias(
                    "dormancy_score"
                ),
            )
            .select("account_id", "dormancy_score")
        )

    def _compute_repeat_ratio(self, transactions: pl.DataFrame) -> pl.DataFrame:
        pair_counts = transactions.group_by(["sender_id", "receiver_id"]).agg(
            pl.len().alias("pair_count")
        )

        repeat_stats = (
            pair_counts.group_by("sender_id")
            .agg(
                pl.col("pair_count").filter(pl.col("pair_count") > 1).sum().alias("repeat_txns"),
                pl.col("pair_count").sum().alias("total_txns"),
            )
            .with_columns(
                (pl.col("repeat_txns") / (pl.col("total_txns") + 1e-8)).alias(
                    "repeat_transfer_ratio"
                ),
            )
            .rename({"sender_id": "account_id"})
            .select("account_id", "repeat_transfer_ratio")
        )

        return repeat_stats
