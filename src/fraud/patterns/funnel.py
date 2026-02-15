from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from src.fraud.patterns.base import FraudPattern

if TYPE_CHECKING:
    import numpy as np

    from src.config.settings import FraudConfig


class FunnelPattern(FraudPattern):
    """Funnel accounts: high inbound from many sources, rapid outward to few beneficiaries."""

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
        roles: dict[str, list[str]] = {"mule_funnel": [], "beneficiary": []}

        n_funnels = max(1, len(mule_ids) // 3)
        funnel_ids = mule_ids[:n_funnels]
        beneficiary_ids = mule_ids[n_funnels : n_funnels + max(1, n_funnels // 2)]

        roles["mule_funnel"] = funnel_ids
        roles["beneficiary"] = beneficiary_ids

        normal_ids = [aid for aid in all_account_ids if aid not in mule_ids]

        for funnel_id in funnel_ids:
            n_inbound = max(
                self.config.funnel_min_inbound,
                int(rng.integers(10, 30)),
            )

            day_offset = int(rng.integers(self.sim_days // 4, self.sim_days * 3 // 4))
            base_time = self.sim_start + timedelta(days=day_offset)

            source_ids = rng.choice(normal_ids, size=min(n_inbound, len(normal_ids)), replace=False)
            for i, source_id in enumerate(source_ids):
                amount = float(rng.lognormal(6.0, 0.8))
                ts = base_time + timedelta(hours=float(rng.uniform(0, 12)), minutes=float(i))

                transactions.append(
                    {
                        "transaction_id": uuid.uuid4().hex[:16],
                        "sender_id": str(source_id),
                        "receiver_id": funnel_id,
                        "amount": round(amount, 2),
                        "timestamp": ts,
                        "transaction_type": "p2p",
                        "is_fraud": True,
                        "fraud_pattern": "funnel_inbound",
                    }
                )

            for ben_id in beneficiary_ids:
                outbound_amount = float(rng.lognormal(7.0, 0.5))
                ts = base_time + timedelta(
                    hours=float(rng.uniform(1, self.config.retention_time_hours_max))
                )

                transactions.append(
                    {
                        "transaction_id": uuid.uuid4().hex[:16],
                        "sender_id": funnel_id,
                        "receiver_id": ben_id,
                        "amount": round(outbound_amount, 2),
                        "timestamp": ts,
                        "transaction_type": "p2p",
                        "is_fraud": True,
                        "fraud_pattern": "funnel_outbound",
                    }
                )

        return transactions, roles
