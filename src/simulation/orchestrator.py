from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import structlog

from src.simulation.accounts import AccountGenerator
from src.simulation.transactions import TransactionGenerator
from src.simulation.validator import SimulationValidator

if TYPE_CHECKING:
    from src.config.settings import Settings

logger = structlog.get_logger(__name__)


class SimulationOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.accounts: pl.DataFrame | None = None
        self.transactions: pl.DataFrame | None = None

    def run(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        settings = self.settings
        settings.ensure_dirs()

        account_gen = AccountGenerator(settings.simulation)
        self.accounts = account_gen.generate()

        from src.fraud.engine import FraudInjectionEngine

        fraud_engine = FraudInjectionEngine(settings.simulation, settings.fraud, self.accounts)
        self.accounts, fraud_transactions = fraud_engine.inject_all()

        txn_gen = TransactionGenerator(settings.simulation)
        normal_transactions = txn_gen.generate(self.accounts)

        self.transactions = pl.concat([normal_transactions, fraud_transactions])
        self.transactions = self.transactions.sort("timestamp")

        logger.info(
            "simulation_complete",
            total_accounts=self.accounts.height,
            total_transactions=self.transactions.height,
            fraud_transactions=fraud_transactions.height,
        )
        return self.accounts, self.transactions

    def validate(self) -> dict[str, dict[str, object]]:
        if self.accounts is None or self.transactions is None:
            msg = "Run simulation before validation"
            raise RuntimeError(msg)

        validator = SimulationValidator(self.accounts, self.transactions)
        return validator.validate_all()

    def save(self) -> None:
        if self.accounts is None or self.transactions is None:
            msg = "Run simulation before saving"
            raise RuntimeError(msg)

        data_dir = self.settings.data_dir
        self.accounts.write_parquet(data_dir / "accounts.parquet")
        self.transactions.write_parquet(data_dir / "transactions.parquet")

        logger.info("data_saved", path=str(data_dir))
