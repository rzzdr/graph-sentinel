from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog

if TYPE_CHECKING:
    from src.config.settings import SimulationConfig

logger = structlog.get_logger(__name__)

REGIONS = [
    "north_america",
    "europe_west",
    "europe_east",
    "asia_pacific",
    "south_america",
    "middle_east",
    "africa",
    "southeast_asia",
]

KYC_LEVELS = ["basic", "standard", "enhanced", "premium"]
ACCOUNT_TYPES = ["savings", "business", "wallet"]

KYC_WEIGHTS = [0.15, 0.45, 0.30, 0.10]
ACCOUNT_TYPE_WEIGHTS = [0.50, 0.20, 0.30]
REGION_WEIGHTS = [0.25, 0.20, 0.10, 0.15, 0.10, 0.05, 0.05, 0.10]


class AccountGenerator:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def generate(self) -> pl.DataFrame:
        n = self.config.total_accounts
        logger.info("generating_accounts", count=n)

        sim_start = datetime(2025, 1, 1)
        sim_end = sim_start + timedelta(days=self.config.simulation_days)

        creation_offsets = self.rng.exponential(scale=self.config.simulation_days * 0.3, size=n)
        creation_offsets = np.clip(creation_offsets, 0, self.config.simulation_days - 1)

        creation_dates = [sim_start + timedelta(days=float(offset)) for offset in creation_offsets]

        account_ids = [uuid.uuid4().hex[:16] for _ in range(n)]
        device_ids = [uuid.uuid4().hex[:12] for _ in range(n)]

        account_types = self.rng.choice(ACCOUNT_TYPES, size=n, p=ACCOUNT_TYPE_WEIGHTS).tolist()
        kyc_levels = self.rng.choice(KYC_LEVELS, size=n, p=KYC_WEIGHTS).tolist()
        regions = self.rng.choice(REGIONS, size=n, p=REGION_WEIGHTS).tolist()

        risk_profiles = self.rng.beta(2, 8, size=n).tolist()

        df = pl.DataFrame(
            {
                "account_id": account_ids,
                "creation_date": creation_dates,
                "account_type": account_types,
                "kyc_level": kyc_levels,
                "geographic_region": regions,
                "device_fingerprint_id": device_ids,
                "latent_risk_profile": risk_profiles,
                "is_fraud": [False] * n,
                "fraud_role": ["normal"] * n,
            }
        )

        df = df.with_columns(
            pl.col("creation_date").cast(pl.Datetime("us")),
        )

        logger.info(
            "accounts_generated",
            count=len(df),
            sim_start=sim_start.isoformat(),
            sim_end=sim_end.isoformat(),
        )
        return df

    def assign_fraud_roles(
        self,
        accounts: pl.DataFrame,
        fraud_account_ids: dict[str, list[str]],
    ) -> pl.DataFrame:
        fraud_map: dict[str, str] = {}
        all_fraud_ids: set[str] = set()

        for role, ids in fraud_account_ids.items():
            for aid in ids:
                fraud_map[aid] = role
                all_fraud_ids.add(aid)

        accounts = accounts.with_columns(
            pl.when(pl.col("account_id").is_in(list(all_fraud_ids)))
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("is_fraud"),
            pl.col("account_id")
            .map_elements(lambda x: fraud_map.get(x, "normal"), return_dtype=pl.Utf8)
            .alias("fraud_role"),
        )

        fraud_count = accounts.filter(pl.col("is_fraud")).height
        logger.info("fraud_roles_assigned", fraud_accounts=fraud_count, total=accounts.height)
        return accounts
