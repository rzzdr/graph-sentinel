from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from src.fraud.patterns.base import FraudPattern

if TYPE_CHECKING:
    import numpy as np

    from src.config.settings import FraudConfig


class CircularFlowPattern(FraudPattern):
    """Circular money flows: strongly connected components with multi-hop cycles."""

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
        roles: dict[str, list[str]] = {"mule_circular": []}

        if len(mule_ids) < self.config.circular_min_cycle:
            return transactions, roles

        cycle_size = int(
            rng.integers(
                self.config.circular_min_cycle,
                min(self.config.circular_max_cycle + 1, len(mule_ids) + 1),
            )
        )

        cycle_ids = list(mule_ids[:cycle_size])
        roles["mule_circular"] = cycle_ids

        n_cycles = max(1, int(rng.integers(2, 6)))

        for cycle_round in range(n_cycles):
            day_offset = int(
                rng.integers(
                    self.sim_days // 4 + cycle_round * 5,
                    min(self.sim_days * 3 // 4, self.sim_days - 1),
                )
            )
            base_time = self.sim_start + timedelta(days=day_offset)
            base_amount = float(rng.lognormal(6.5, 0.6))

            for i in range(cycle_size):
                sender = cycle_ids[i]
                receiver = cycle_ids[(i + 1) % cycle_size]

                amount = base_amount * float(rng.uniform(0.90, 1.05))
                hop_delay = timedelta(
                    minutes=float(rng.uniform(10, 120)),
                    seconds=float(rng.uniform(0, 60)),
                )
                ts = base_time + hop_delay * (i + 1)

                transactions.append(
                    {
                        "transaction_id": uuid.uuid4().hex[:16],
                        "sender_id": sender,
                        "receiver_id": receiver,
                        "amount": round(amount, 2),
                        "timestamp": ts,
                        "transaction_type": "p2p",
                        "is_fraud": True,
                        "fraud_pattern": "circular_flow",
                    }
                )

        return transactions, roles
