from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import polars as pl
import structlog

from src.ingestion.schema import SchemaValidator

if TYPE_CHECKING:
    from src.config.settings import Settings

logger = structlog.get_logger(__name__)


class IngestionPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.validator = SchemaValidator()

    def load_from_parquet(
        self,
        transactions_path: Path | None = None,
        accounts_path: Path | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        data_dir = self.settings.data_dir

        txn_path = transactions_path or data_dir / "transactions.parquet"
        acc_path = accounts_path or data_dir / "accounts.parquet"

        if not txn_path.exists():
            msg = f"Transactions file not found: {txn_path}"
            raise FileNotFoundError(msg)
        if not acc_path.exists():
            msg = f"Accounts file not found: {acc_path}"
            raise FileNotFoundError(msg)

        transactions = pl.read_parquet(txn_path)
        accounts = pl.read_parquet(acc_path)

        txn_valid, txn_errors = self.validator.validate_transactions(transactions)
        acc_valid, acc_errors = self.validator.validate_accounts(accounts)

        if not txn_valid:
            logger.warning("transaction_validation_issues", errors=txn_errors)
        if not acc_valid:
            logger.warning("account_validation_issues", errors=acc_errors)

        logger.info(
            "data_loaded",
            transactions=transactions.height,
            accounts=accounts.height,
        )
        return transactions, accounts

    def load_from_csv(
        self,
        transactions_path: Path,
        accounts_path: Path,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        transactions = pl.read_csv(transactions_path, try_parse_dates=True)
        accounts = pl.read_csv(accounts_path, try_parse_dates=True)

        txn_valid, _ = self.validator.validate_transactions(transactions)
        acc_valid, _ = self.validator.validate_accounts(accounts)

        if not txn_valid or not acc_valid:
            logger.warning("csv_validation_issues")

        return transactions, accounts

    def ingest_streaming_batch(
        self,
        transactions_batch: pl.DataFrame,
    ) -> pl.DataFrame:
        valid, errors = self.validator.validate_transactions(transactions_batch)
        if not valid:
            logger.warning("batch_validation_failed", errors=errors)

        transactions_batch = transactions_batch.filter(pl.col("amount") > 0)
        transactions_batch = transactions_batch.filter(pl.col("sender_id") != pl.col("receiver_id"))

        transactions_batch = transactions_batch.unique(subset=["transaction_id"])

        logger.info("batch_ingested", rows=transactions_batch.height)
        return transactions_batch
