from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.config.settings import (
    FraudConfig,
    GraphConfig,
    ScoringConfig,
    Settings,
    SimulationConfig,
)


@pytest.fixture  # type: ignore[misc]
def small_sim_config() -> SimulationConfig:
    return SimulationConfig(
        total_accounts=500,
        simulation_days=30,
        total_transactions=5_000,
        fraud_ratio=0.02,
        seed=42,
    )


@pytest.fixture  # type: ignore[misc]
def fraud_config() -> FraudConfig:
    return FraudConfig()


@pytest.fixture  # type: ignore[misc]
def graph_config() -> GraphConfig:
    return GraphConfig(rolling_windows_days=[7, 30])


@pytest.fixture  # type: ignore[misc]
def scoring_config() -> ScoringConfig:
    return ScoringConfig()


@pytest.fixture  # type: ignore[misc]
def settings(
    small_sim_config: SimulationConfig,
    fraud_config: FraudConfig,
    graph_config: GraphConfig,
    scoring_config: ScoringConfig,
    tmp_path: Path,
) -> Settings:
    return Settings(
        simulation=small_sim_config,
        fraud=fraud_config,
        graph=graph_config,
        scoring=scoring_config,
        data_dir=Path(str(tmp_path)) / "data",
        output_dir=Path(str(tmp_path)) / "outputs",
    )


@pytest.fixture  # type: ignore[misc]
def sample_accounts() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "account_id": [f"acc_{i:04d}" for i in range(100)],
            "creation_date": [datetime(2025, 1, 1) for _ in range(100)],
            "account_type": ["savings"] * 50 + ["business"] * 30 + ["wallet"] * 20,
            "kyc_level": ["standard"] * 40 + ["basic"] * 30 + ["enhanced"] * 20 + ["premium"] * 10,
            "geographic_region": ["north_america"] * 25
            + ["europe_west"] * 25
            + ["asia_pacific"] * 25
            + ["south_america"] * 25,
            "device_fingerprint_id": [f"dev_{i:04d}" for i in range(100)],
            "latent_risk_profile": [0.1] * 95 + [0.8] * 5,
            "is_fraud": [False] * 95 + [True] * 5,
            "fraud_role": ["normal"] * 95 + ["mule_funnel"] * 5,
        }
    )


@pytest.fixture  # type: ignore[misc]
def sample_transactions(sample_accounts: pl.DataFrame) -> pl.DataFrame:
    import numpy as np

    rng = np.random.default_rng(42)
    n = 1000
    account_ids = sample_accounts["account_id"].to_list()

    return pl.DataFrame(
        {
            "transaction_id": [f"txn_{i:06d}" for i in range(n)],
            "sender_id": [str(rng.choice(account_ids)) for _ in range(n)],
            "receiver_id": [str(rng.choice(account_ids)) for _ in range(n)],
            "amount": [float(rng.lognormal(4, 1)) for _ in range(n)],
            "timestamp": [
                datetime(2025, 1, 1) + timedelta(hours=int(rng.integers(0, 720))) for _ in range(n)
            ],
            "transaction_type": [
                str(rng.choice(["p2p", "ecommerce", "bill_payment", "salary"])) for _ in range(n)
            ],
            "is_fraud": [False] * 980 + [True] * 20,
            "fraud_pattern": ["none"] * 980 + ["funnel_inbound"] * 20,
        }
    )
