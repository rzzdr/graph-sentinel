from __future__ import annotations

import polars as pl


def build_ground_truth(
    accounts: pl.DataFrame,
    transactions: pl.DataFrame,
) -> pl.DataFrame:
    fraud_accounts = accounts.filter(pl.col("is_fraud")).select("account_id", "fraud_role")

    fraud_txn_summary = (
        transactions.filter(pl.col("is_fraud"))
        .group_by("sender_id")
        .agg(
            pl.len().alias("fraud_txn_count"),
            pl.col("amount").sum().alias("fraud_total_amount"),
            pl.col("fraud_pattern").first().alias("primary_pattern"),
        )
        .rename({"sender_id": "account_id"})
    )

    ground_truth = fraud_accounts.join(fraud_txn_summary, on="account_id", how="left").with_columns(
        pl.col("fraud_txn_count").fill_null(0),
        pl.col("fraud_total_amount").fill_null(0.0),
        pl.col("primary_pattern").fill_null("unknown"),
    )

    return ground_truth
