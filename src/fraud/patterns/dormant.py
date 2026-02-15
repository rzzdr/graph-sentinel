from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from src.fraud.patterns.base import FraudPattern

if TYPE_CHECKING:
    import numpy as np

    from src.config.settings import FraudConfig


class DormantActivationPattern(FraudPattern):
    """Dormant activation: long inactivity followed by sudden burst of transactions."""

    def __init__(self, config: FraudConfig, sim_start: datetime, sim_days: int) -> None:
        self.config = config
        self.sim_start = sim_start
        self.sim_days = sim_days

    def generate(
        self,
        mule_ids: list[str],
        all_account_ids: list[str],
        rng: np.random.Generator,
    ) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
        transactions: list[dict[str, object]] = []
        roles: dict[str, list[str]] = {"mule_dormant": []}

        if not mule_ids:
            return transactions, roles

        roles["mule_dormant"] = list(mule_ids)
        mule_set = set(mule_ids)
        normal_ids = [aid for aid in all_account_ids if aid not in mule_set]
        if not normal_ids:
            return transactions, roles

        for mule_id in mule_ids:
            dormant_high = min(self.sim_days - 10, 150)
            if dormant_high <= 60:
                dormant_sample = min(60, max(1, dormant_high))
            else:
                dormant_sample = int(rng.integers(60, dormant_high))
            dormant_days = max(
                self.config.dormant_inactivity_days,
                dormant_sample,
            )
            activation_day = dormant_days + int(rng.integers(1, 5))

            if activation_day >= self.sim_days:
                activation_day = self.sim_days - 5

            burst_time = self.sim_start + timedelta(days=activation_day)
            burst_window_hours = self.config.burst_window_hours

            n_burst_txns = int(rng.integers(8, 25))
            sources = rng.choice(
                normal_ids,
                size=min(n_burst_txns // 2 + 1, len(normal_ids)),
                replace=False,
            )
            targets = rng.choice(
                normal_ids,
                size=min(n_burst_txns // 2 + 1, len(normal_ids)),
                replace=False,
            )

            for i, source_id in enumerate(sources):
                amount = float(rng.lognormal(6.0, 1.0))
                ts = burst_time + timedelta(
                    hours=float(rng.uniform(0, burst_window_hours / 2)),
                    minutes=float(i * 3),
                )

                transactions.append(
                    {
                        "transaction_id": uuid.uuid4().hex[:16],
                        "sender_id": str(source_id),
                        "receiver_id": mule_id,
                        "amount": round(amount, 2),
                        "timestamp": ts,
                        "transaction_type": "p2p",
                        "is_fraud": True,
                        "fraud_pattern": "dormant_inbound",
                    }
                )

            for i, target_id in enumerate(targets):
                amount = float(rng.lognormal(6.5, 0.8))
                ts = burst_time + timedelta(
                    hours=float(rng.uniform(burst_window_hours / 2, burst_window_hours)),
                    minutes=float(i * 2),
                )

                transactions.append(
                    {
                        "transaction_id": uuid.uuid4().hex[:16],
                        "sender_id": mule_id,
                        "receiver_id": str(target_id),
                        "amount": round(amount, 2),
                        "timestamp": ts,
                        "transaction_type": "p2p",
                        "is_fraud": True,
                        "fraud_pattern": "dormant_outbound",
                    }
                )

        return transactions, roles
