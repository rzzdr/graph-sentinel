from __future__ import annotations

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

MAX_SEQ_LEN = 200
TRANSACTION_FEATURES = [
    "amount",
    "is_sender",
    "hour_of_day",
    "day_of_week",
    "time_delta_hours",
]


class SequenceBuilder:
    """Converts raw transaction history into fixed-length numerical sequences per account.

    Each transaction is encoded as a feature vector capturing amount (log-scaled),
    direction, temporal position, and inter-transaction gap. Sequences are truncated
    or padded to MAX_SEQ_LEN.
    """

    def __init__(self, max_seq_len: int = MAX_SEQ_LEN) -> None:
        self.max_seq_len = max_seq_len

    def build_sequences(self, transactions: pl.DataFrame) -> dict[str, np.ndarray]:
        sent = transactions.select(
            pl.col("sender_id").alias("account_id"),
            pl.col("amount"),
            pl.col("timestamp"),
            pl.lit(1).alias("is_sender"),
        )
        received = transactions.select(
            pl.col("receiver_id").alias("account_id"),
            pl.col("amount"),
            pl.col("timestamp"),
            pl.lit(0).alias("is_sender"),
        )

        all_txns = pl.concat([sent, received]).sort(["account_id", "timestamp"])
        all_txns = all_txns.with_columns(
            pl.col("timestamp").dt.hour().alias("hour_of_day"),
            (pl.col("timestamp").dt.weekday() - 1).alias("day_of_week"),
        )

        sequences: dict[str, np.ndarray] = {}
        n_features = len(TRANSACTION_FEATURES)

        for account_id, group in all_txns.group_by("account_id"):
            aid = account_id[0] if isinstance(account_id, tuple) else account_id
            df = group.sort("timestamp")
            n = min(df.height, self.max_seq_len)

            seq = np.zeros((self.max_seq_len, n_features), dtype=np.float32)

            amounts = df["amount"].to_numpy()[:n]
            log_amounts = np.log1p(amounts)
            max_log = log_amounts.max()
            if max_log > 0:
                log_amounts = log_amounts / max_log

            seq[:n, 0] = log_amounts
            seq[:n, 1] = df["is_sender"].to_numpy()[:n].astype(np.float32)
            seq[:n, 2] = df["hour_of_day"].to_numpy()[:n].astype(np.float32) / 23.0
            seq[:n, 3] = df["day_of_week"].to_numpy()[:n].astype(np.float32) / 6.0

            timestamps = df["timestamp"].to_numpy()[:n]
            if n > 1:
                deltas = np.diff(timestamps.astype("datetime64[s]").astype(np.float64))
                deltas_hours = deltas / 3600.0
                deltas_hours = np.clip(deltas_hours, 0, 720)
                max_delta = deltas_hours.max()
                if max_delta > 0:
                    deltas_hours = deltas_hours / max_delta
                seq[1:n, 4] = deltas_hours.astype(np.float32)

            sequences[str(aid)] = seq

        logger.info("sequences_built", accounts=len(sequences), max_len=self.max_seq_len)
        return sequences
