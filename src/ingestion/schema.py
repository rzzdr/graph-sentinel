from __future__ import annotations

from datetime import datetime

import polars as pl
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

TRANSACTION_SCHEMA = {
    "transaction_id": pl.Utf8,
    "sender_id": pl.Utf8,
    "receiver_id": pl.Utf8,
    "amount": pl.Float64,
    "timestamp": pl.Datetime("us"),
    "transaction_type": pl.Utf8,
}

ACCOUNT_SCHEMA = {
    "account_id": pl.Utf8,
    "creation_date": pl.Datetime("us"),
    "account_type": pl.Utf8,
    "kyc_level": pl.Utf8,
    "geographic_region": pl.Utf8,
}


class TransactionSchema(BaseModel):
    transaction_id: str
    sender_id: str
    receiver_id: str
    amount: float = Field(gt=0)
    timestamp: datetime
    transaction_type: str


class AccountSchema(BaseModel):
    account_id: str
    creation_date: datetime
    account_type: str
    kyc_level: str
    geographic_region: str


class SchemaValidator:
    def validate_transactions(self, df: pl.DataFrame) -> tuple[bool, list[str]]:
        errors: list[str] = []

        required_cols = set(TRANSACTION_SCHEMA.keys())
        missing = required_cols - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {missing}")
            return False, errors

        for col, expected_type in TRANSACTION_SCHEMA.items():
            if col in df.columns and df[col].dtype != expected_type:
                try:
                    df = df.with_columns(pl.col(col).cast(expected_type))
                except Exception:
                    errors.append(f"Column '{col}' cannot be cast to {expected_type}")

        null_counts = {
            col: df[col].null_count()
            for col in TRANSACTION_SCHEMA
            if col in df.columns and df[col].null_count() > 0
        }
        if null_counts:
            errors.append(f"Null values found: {null_counts}")

        negative_amounts = df.filter(pl.col("amount") <= 0).height
        if negative_amounts > 0:
            errors.append(f"{negative_amounts} transactions with non-positive amounts")

        self_transfers = df.filter(pl.col("sender_id") == pl.col("receiver_id")).height
        if self_transfers > 0:
            errors.append(f"{self_transfers} self-transfers detected")

        valid = len(errors) == 0
        if valid:
            logger.info("transaction_schema_valid", rows=df.height)
        else:
            logger.warning("transaction_schema_invalid", errors=errors)

        return valid, errors

    def validate_accounts(self, df: pl.DataFrame) -> tuple[bool, list[str]]:
        errors: list[str] = []

        required_cols = set(ACCOUNT_SCHEMA.keys())
        missing = required_cols - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {missing}")
            return False, errors

        n_unique = df["account_id"].n_unique()
        if n_unique != df.height:
            errors.append(f"Duplicate account_ids: {df.height - n_unique} duplicates")

        valid = len(errors) == 0
        return valid, errors

    def check_label_leakage(
        self,
        features: pl.DataFrame,
        forbidden_columns: list[str] | None = None,
    ) -> list[str]:
        forbidden = set(
            forbidden_columns
            or [
                "is_fraud",
                "fraud_role",
                "fraud_pattern",
                "is_fraud_label",
            ]
        )

        leaked = [c for c in features.columns if c in forbidden]
        if leaked:
            logger.warning("label_leakage_detected", columns=leaked)
        return leaked
