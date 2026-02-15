from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog

from src.fraud.patterns.circular import CircularFlowPattern
from src.fraud.patterns.device_sharing import DeviceSharingPattern
from src.fraud.patterns.dormant import DormantActivationPattern
from src.fraud.patterns.funnel import FunnelPattern
from src.fraud.patterns.layering import LayeringPattern

if TYPE_CHECKING:
    from src.config.settings import FraudConfig, SimulationConfig

logger = structlog.get_logger(__name__)


class FraudInjectionEngine:
    def __init__(
        self,
        sim_config: SimulationConfig,
        fraud_config: FraudConfig,
        accounts: pl.DataFrame,
    ) -> None:
        self.sim_config = sim_config
        self.fraud_config = fraud_config
        self.accounts = accounts
        self.rng = np.random.default_rng(sim_config.seed + 1000)
        self.sim_start = datetime(2025, 1, 1)

    def inject_all(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        n_fraud = int(self.sim_config.total_accounts * self.sim_config.fraud_ratio)
        n_fraud = max(n_fraud, 10)

        all_ids = self.accounts["account_id"].to_list()
        fraud_indices = self.rng.choice(len(all_ids), size=n_fraud, replace=False)
        fraud_ids = [all_ids[i] for i in fraud_indices]

        logger.info("fraud_injection_start", total_fraud_accounts=n_fraud)

        allocations = self._allocate_mules(fraud_ids)
        all_transactions: list[dict[str, object]] = []
        all_roles: dict[str, list[str]] = {}
        device_updates: dict[str, str] = {}

        patterns = [
            (
                "funnel",
                FunnelPattern(self.fraud_config, self.sim_start, self.sim_config.simulation_days),
                allocations["funnel"],
            ),
            (
                "circular",
                CircularFlowPattern(
                    self.fraud_config, self.sim_start, self.sim_config.simulation_days
                ),
                allocations["circular"],
            ),
            (
                "layering",
                LayeringPattern(self.fraud_config, self.sim_start, self.sim_config.simulation_days),
                allocations["layering"],
            ),
            (
                "dormant",
                DormantActivationPattern(
                    self.fraud_config, self.sim_start, self.sim_config.simulation_days
                ),
                allocations["dormant"],
            ),
            (
                "device_sharing",
                DeviceSharingPattern(
                    self.fraud_config, self.sim_start, self.sim_config.simulation_days
                ),
                allocations["device_sharing"],
            ),
        ]

        for name, pattern, mule_ids in patterns:
            if not mule_ids:
                continue

            txns, roles = pattern.generate(mule_ids, all_ids, self.rng)
            all_transactions.extend(txns)

            for role, ids in roles.items():
                all_roles.setdefault(role, []).extend(ids)

            if isinstance(pattern, DeviceSharingPattern):
                device_updates.update(pattern.shared_device_ids)

            logger.info(
                "pattern_injected",
                pattern=name,
                mule_count=len(mule_ids),
                txn_count=len(txns),
            )

        from src.simulation.accounts import AccountGenerator

        account_gen = AccountGenerator(self.sim_config)
        self.accounts = account_gen.assign_fraud_roles(self.accounts, all_roles)

        if device_updates:
            self.accounts = self.accounts.with_columns(
                pl.col("account_id")
                .map_elements(
                    lambda x: device_updates.get(x),
                    return_dtype=pl.Utf8,
                )
                .alias("_shared_device"),
            )
            self.accounts = self.accounts.with_columns(
                pl.when(pl.col("_shared_device").is_not_null())
                .then(pl.col("_shared_device"))
                .otherwise(pl.col("device_fingerprint_id"))
                .alias("device_fingerprint_id")
            ).drop("_shared_device")

        if not all_transactions:
            fraud_df = pl.DataFrame(
                schema={
                    "transaction_id": pl.Utf8,
                    "sender_id": pl.Utf8,
                    "receiver_id": pl.Utf8,
                    "amount": pl.Float64,
                    "timestamp": pl.Datetime("us"),
                    "transaction_type": pl.Utf8,
                    "is_fraud": pl.Boolean,
                    "fraud_pattern": pl.Utf8,
                }
            )
        else:
            fraud_df = pl.DataFrame(all_transactions)
            fraud_df = fraud_df.with_columns(
                pl.col("timestamp").cast(pl.Datetime("us")),
            )

        logger.info(
            "fraud_injection_complete",
            total_fraud_txns=fraud_df.height,
            patterns_used=len([p for _, p, m in patterns if m]),
        )
        return self.accounts, fraud_df

    def _allocate_mules(self, fraud_ids: list[str]) -> dict[str, list[str]]:
        n = len(fraud_ids)
        cfg = self.fraud_config

        sizes = {
            "funnel": max(1, int(n * cfg.funnel_account_ratio)),
            "circular": max(1, int(n * cfg.circular_flow_ratio)),
            "layering": max(1, int(n * cfg.layering_ratio)),
            "dormant": max(1, int(n * cfg.dormant_activation_ratio)),
            "device_sharing": max(1, int(n * cfg.device_sharing_ratio)),
        }

        total_allocated = sum(sizes.values())
        if total_allocated > n:
            scale = n / total_allocated
            sizes = {k: max(1, int(v * scale)) for k, v in sizes.items()}

        allocations: dict[str, list[str]] = {}
        offset = 0
        for pattern, size in sizes.items():
            end = min(offset + size, n)
            allocations[pattern] = fraud_ids[offset:end]
            offset = end

        return allocations
