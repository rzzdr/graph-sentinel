from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from src.fraud.patterns.base import FraudPattern

if TYPE_CHECKING:
    import numpy as np

    from src.config.settings import FraudConfig


class LayeringPattern(FraudPattern):
    """Layering: multi-hop chains with decreasing amounts and high velocity."""

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
        roles: dict[str, list[str]] = {"originator": [], "mule_layer": [], "beneficiary": []}

        if len(mule_ids) < self.config.layering_min_hops:
            return transactions, roles

        chain_length = int(
            rng.integers(
                self.config.layering_min_hops,
                min(self.config.layering_max_hops + 1, len(mule_ids) + 1),
            )
        )

        normal_ids = [aid for aid in all_account_ids if aid not in mule_ids]
        if not normal_ids:
            return transactions, roles

        originator_id = str(rng.choice(normal_ids))
        chain_ids = list(mule_ids[:chain_length])
        beneficiary_id = chain_ids[-1]

        roles["originator"] = [originator_id]
        roles["mule_layer"] = chain_ids[:-1]
        roles["beneficiary"] = [beneficiary_id]

        n_chains = max(1, int(rng.integers(2, 5)))

        for chain_round in range(n_chains):
            day_offset = int(
                rng.integers(
                    self.sim_days // 3 + chain_round * 7,
                    min(self.sim_days * 3 // 4, self.sim_days - 1),
                )
            )
            base_time = self.sim_start + timedelta(days=day_offset)

            initial_amount = float(rng.lognormal(7.0, 0.7))
            current_amount = initial_amount

            full_chain = [originator_id, *chain_ids]

            for i in range(len(full_chain) - 1):
                sender = full_chain[i]
                receiver = full_chain[i + 1]

                decay = float(rng.uniform(0.85, 0.95))
                current_amount *= decay

                hop_delay = timedelta(
                    minutes=float(rng.uniform(5, 45)),
                    seconds=float(rng.uniform(0, 60)),
                )
                ts = base_time + hop_delay * (i + 1)

                transactions.append(
                    {
                        "transaction_id": uuid.uuid4().hex[:16],
                        "sender_id": sender,
                        "receiver_id": receiver,
                        "amount": round(current_amount, 2),
                        "timestamp": ts,
                        "transaction_type": "p2p",
                        "is_fraud": True,
                        "fraud_pattern": "layering",
                    }
                )

        return transactions, roles
