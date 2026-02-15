from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from src.fraud.patterns.base import FraudPattern

if TYPE_CHECKING:
    import numpy as np

    from src.config.settings import FraudConfig


class DeviceSharingPattern(FraudPattern):
    """Device sharing: multiple mule accounts share device fingerprints with geo inconsistencies."""

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
        roles: dict[str, list[str]] = {"mule_device_cluster": []}

        if len(mule_ids) < 2:
            return transactions, roles

        roles["mule_device_cluster"] = list(mule_ids)

        shared_device_id = uuid.uuid4().hex[:12]
        self._shared_device_ids = {mid: shared_device_id for mid in mule_ids}

        normal_ids = [aid for aid in all_account_ids if aid not in mule_ids]

        for i in range(len(mule_ids) - 1):
            sender = mule_ids[i]
            receiver = mule_ids[i + 1]

            n_txns = int(rng.integers(3, 8))
            for t in range(n_txns):
                day_offset = int(rng.integers(self.sim_days // 4, self.sim_days * 3 // 4))
                ts = self.sim_start + timedelta(
                    days=day_offset,
                    hours=float(rng.uniform(0, 24)),
                    minutes=float(t * 5),
                )
                amount = float(rng.lognormal(5.5, 0.9))

                transactions.append(
                    {
                        "transaction_id": uuid.uuid4().hex[:16],
                        "sender_id": sender,
                        "receiver_id": receiver,
                        "amount": round(amount, 2),
                        "timestamp": ts,
                        "transaction_type": "p2p",
                        "is_fraud": True,
                        "fraud_pattern": "device_sharing",
                    }
                )

        for mule_id in mule_ids:
            if not normal_ids:
                break
            n_external = int(rng.integers(2, 6))
            ext_ids = rng.choice(
                normal_ids,
                size=min(n_external, len(normal_ids)),
                replace=False,
            )
            for ext_id in ext_ids:
                day_offset = int(rng.integers(self.sim_days // 3, self.sim_days * 2 // 3))
                ts = self.sim_start + timedelta(
                    days=day_offset,
                    hours=float(rng.uniform(0, 24)),
                )
                amount = float(rng.lognormal(5.0, 1.0))

                transactions.append(
                    {
                        "transaction_id": uuid.uuid4().hex[:16],
                        "sender_id": mule_id,
                        "receiver_id": str(ext_id),
                        "amount": round(amount, 2),
                        "timestamp": ts,
                        "transaction_type": "p2p",
                        "is_fraud": True,
                        "fraud_pattern": "device_sharing_external",
                    }
                )

        return transactions, roles

    @property
    def shared_device_ids(self) -> dict[str, str]:
        return getattr(self, "_shared_device_ids", {})
