from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from src.config.settings import SimulationConfig
from src.simulation.accounts import AccountGenerator
from src.simulation.transactions import TransactionGenerator


class TestTransactionGenerator:
    def test_generates_transactions(self, small_sim_config: SimulationConfig) -> None:
        acc_gen = AccountGenerator(small_sim_config)
        accounts = acc_gen.generate()

        txn_gen = TransactionGenerator(small_sim_config)
        transactions = txn_gen.generate(accounts)

        assert transactions.height > 0
        assert transactions.height <= small_sim_config.total_transactions * 1.1

    def test_has_required_columns(self, small_sim_config: SimulationConfig) -> None:
        acc_gen = AccountGenerator(small_sim_config)
        accounts = acc_gen.generate()

        txn_gen = TransactionGenerator(small_sim_config)
        transactions = txn_gen.generate(accounts)

        required = {
            "transaction_id",
            "sender_id",
            "receiver_id",
            "amount",
            "timestamp",
            "transaction_type",
        }
        assert required.issubset(set(transactions.columns))

    def test_positive_amounts(self, small_sim_config: SimulationConfig) -> None:
        acc_gen = AccountGenerator(small_sim_config)
        accounts = acc_gen.generate()

        txn_gen = TransactionGenerator(small_sim_config)
        transactions = txn_gen.generate(accounts)

        assert transactions.filter(pl.col("amount") <= 0).height == 0

    def test_no_self_transfers(self, small_sim_config: SimulationConfig) -> None:
        acc_gen = AccountGenerator(small_sim_config)
        accounts = acc_gen.generate()

        txn_gen = TransactionGenerator(small_sim_config)
        transactions = txn_gen.generate(accounts)

        self_transfers = transactions.filter(pl.col("sender_id") == pl.col("receiver_id"))
        assert self_transfers.height == 0
