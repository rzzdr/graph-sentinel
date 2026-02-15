from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog
from tqdm import tqdm

from src.simulation.behavioral import BehavioralModel, TransactionProfile

if TYPE_CHECKING:
    from src.config.settings import SimulationConfig

logger = structlog.get_logger(__name__)


class TransactionGenerator:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.behavioral = BehavioralModel(seed=config.seed)

    def generate(self, accounts: pl.DataFrame) -> pl.DataFrame:
        logger.info(
            "generating_transactions",
            target_count=self.config.total_transactions,
            accounts=accounts.height,
        )

        sim_start = datetime(2025, 1, 1)
        account_list = accounts.to_dicts()
        account_ids = [a["account_id"] for a in account_list]

        profiles = {}
        for acc in account_list:
            profiles[acc["account_id"]] = self.behavioral.get_profile(
                acc["account_type"], acc["kyc_level"]
            )

        transactions: list[dict[str, object]] = []
        target = self.config.total_transactions
        days = self.config.simulation_days

        for day_offset in tqdm(range(days), desc="Generating transactions"):
            current_date = sim_start + timedelta(days=day_offset)
            weekday = current_date.weekday()
            day_factor = self.behavioral.apply_day_of_week_pattern(weekday)

            active_accounts = [
                a
                for a in account_list
                if a["creation_date"] <= current_date and a["fraud_role"] == "normal"
            ]

            if not active_accounts:
                continue

            daily_budget = int((target / days) * day_factor)
            daily_budget = max(1, daily_budget + int(self.rng.normal(0, daily_budget * 0.1)))

            generated = 0
            np_indices = self.rng.choice(
                len(active_accounts), size=min(daily_budget, len(active_accounts)), replace=True
            )

            for idx in np_indices:
                if generated >= daily_budget:
                    break

                sender = active_accounts[idx]
                sender_id = sender["account_id"]
                profile = profiles[sender_id]

                n_txns = self.behavioral.sample_daily_txn_count(
                    profile.daily_txn_rate * day_factor / days * 10
                )
                n_txns = min(n_txns, daily_budget - generated, 5)

                for _ in range(max(1, n_txns)):
                    if generated >= daily_budget:
                        break

                    receiver_id = self._pick_receiver(sender_id, account_ids)
                    txn_type = self.behavioral.sample_transaction_type(profile)
                    amount = self._sample_amount_for_type(txn_type, profile)

                    hour = self._sample_hour()
                    minute = int(self.rng.integers(0, 60))
                    second = int(self.rng.integers(0, 60))
                    timestamp = current_date.replace(hour=hour, minute=minute, second=second)

                    transactions.append(
                        {
                            "transaction_id": uuid.uuid4().hex[:16],
                            "sender_id": sender_id,
                            "receiver_id": receiver_id,
                            "amount": round(amount, 2),
                            "timestamp": timestamp,
                            "transaction_type": txn_type,
                            "is_fraud": False,
                            "fraud_pattern": "none",
                        }
                    )
                    generated += 1

            if len(transactions) >= target:
                break

        transactions = transactions[:target]

        df = pl.DataFrame(transactions)
        df = df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("us")),
        )

        logger.info("transactions_generated", count=len(df))
        return df

    def _pick_receiver(self, sender_id: str, account_ids: list[str]) -> str:
        while True:
            idx = int(self.rng.integers(0, len(account_ids)))
            if account_ids[idx] != sender_id:
                return account_ids[idx]

    def _sample_amount_for_type(self, txn_type: str, profile: TransactionProfile) -> float:
        base_mu = profile.amount_mu
        base_sigma = profile.amount_sigma

        type_adjustments = {
            "salary": (base_mu + 1.5, base_sigma * 0.3),
            "bill_payment": (base_mu - 0.5, base_sigma * 0.5),
            "ecommerce": (base_mu - 0.3, base_sigma * 0.8),
            "p2p": (base_mu - 0.8, base_sigma * 1.2),
        }
        mu, sigma = type_adjustments.get(txn_type, (base_mu, base_sigma))
        return self.behavioral.sample_amount(mu, max(0.3, sigma))

    def _sample_hour(self) -> int:
        weights = [self.behavioral.apply_time_of_day_pattern(h) for h in range(24)]
        total = sum(weights)
        probs = [w / total for w in weights]
        return int(self.rng.choice(24, p=probs))
