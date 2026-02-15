from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from src.config.settings import SimulationConfig
from src.simulation.accounts import AccountGenerator


class TestAccountGenerator:
    def test_generates_correct_count(self, small_sim_config: SimulationConfig) -> None:
        gen = AccountGenerator(small_sim_config)
        accounts = gen.generate()
        assert accounts.height == small_sim_config.total_accounts

    def test_has_required_columns(self, small_sim_config: SimulationConfig) -> None:
        gen = AccountGenerator(small_sim_config)
        accounts = gen.generate()

        required = {
            "account_id",
            "creation_date",
            "account_type",
            "kyc_level",
            "geographic_region",
            "device_fingerprint_id",
            "latent_risk_profile",
            "is_fraud",
            "fraud_role",
        }
        assert required.issubset(set(accounts.columns))

    def test_unique_account_ids(self, small_sim_config: SimulationConfig) -> None:
        gen = AccountGenerator(small_sim_config)
        accounts = gen.generate()
        assert accounts["account_id"].n_unique() == accounts.height

    def test_valid_account_types(self, small_sim_config: SimulationConfig) -> None:
        gen = AccountGenerator(small_sim_config)
        accounts = gen.generate()
        valid_types = {"savings", "business", "wallet"}
        actual_types = set(accounts["account_type"].unique().to_list())
        assert actual_types.issubset(valid_types)

    def test_initially_no_fraud(self, small_sim_config: SimulationConfig) -> None:
        gen = AccountGenerator(small_sim_config)
        accounts = gen.generate()
        assert accounts.filter(pl.col("is_fraud")).height == 0

    def test_assign_fraud_roles(self, small_sim_config: SimulationConfig) -> None:
        gen = AccountGenerator(small_sim_config)
        accounts = gen.generate()
        ids = accounts["account_id"].to_list()

        fraud_map = {"mule_funnel": ids[:5], "beneficiary": ids[5:8]}
        updated = gen.assign_fraud_roles(accounts, fraud_map)

        assert updated.filter(pl.col("is_fraud")).height == 8
