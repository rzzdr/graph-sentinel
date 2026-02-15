from __future__ import annotations

import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class TemporalSplitter:
    def __init__(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
    ) -> None:
        total = train_ratio + val_ratio + test_ratio
        self.train_ratio = train_ratio / total
        self.val_ratio = val_ratio / total
        self.test_ratio = test_ratio / total

    def split_by_time(
        self,
        transactions: pl.DataFrame,
        features: pl.DataFrame,
        accounts: pl.DataFrame,
    ) -> dict[str, pl.DataFrame]:
        sorted_txns = transactions.sort("timestamp")
        n = sorted_txns.height

        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_cutoff = sorted_txns["timestamp"][train_end]
        val_cutoff = sorted_txns["timestamp"][val_end]

        labels = accounts.select("account_id", "is_fraud")
        labeled_features = features.join(labels, on="account_id", how="left")
        labeled_features = labeled_features.with_columns(
            pl.col("is_fraud").fill_null(False),
        )

        all_txn_accounts = pl.concat(
            [
                transactions.select(pl.col("sender_id").alias("account_id"), "timestamp"),
                transactions.select(pl.col("receiver_id").alias("account_id"), "timestamp"),
            ]
        )

        first_seen = all_txn_accounts.group_by("account_id").agg(
            pl.col("timestamp").min().alias("first_seen")
        )

        labeled_features = labeled_features.join(first_seen, on="account_id", how="left")

        train = labeled_features.filter(pl.col("first_seen") <= train_cutoff)
        val = labeled_features.filter(
            (pl.col("first_seen") > train_cutoff) & (pl.col("first_seen") <= val_cutoff)
        )
        test = labeled_features.filter(pl.col("first_seen") > val_cutoff)

        if val.height == 0:
            val = train.sample(fraction=0.2, seed=42)
        if test.height == 0:
            test = train.sample(fraction=0.2, seed=43)

        logger.info(
            "temporal_split_complete",
            train=train.height,
            val=val.height,
            test=test.height,
            train_fraud=train.filter(pl.col("is_fraud")).height,
            val_fraud=val.filter(pl.col("is_fraud")).height,
            test_fraud=test.filter(pl.col("is_fraud")).height,
        )

        return {
            "train": train.drop("first_seen") if "first_seen" in train.columns else train,
            "val": val.drop("first_seen") if "first_seen" in val.columns else val,
            "test": test.drop("first_seen") if "first_seen" in test.columns else test,
        }
