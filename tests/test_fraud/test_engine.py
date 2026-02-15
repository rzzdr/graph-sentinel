from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from src.config.settings import FraudConfig, SimulationConfig
from src.fraud.engine import FraudInjectionEngine
from src.simulation.accounts import AccountGenerator


class TestFraudInjectionEngine:
    def test_injection_produces_fraud_transactions(
        self,
        small_sim_config: SimulationConfig,
        fraud_config: FraudConfig,
    ) -> None:
        acc_gen = AccountGenerator(small_sim_config)
        accounts = acc_gen.generate()

        engine = FraudInjectionEngine(small_sim_config, fraud_config, accounts)
        updated_accounts, fraud_txns = engine.inject_all()

        assert fraud_txns.height > 0
        assert updated_accounts.filter(pl.col("is_fraud")).height > 0

    def test_fraud_transactions_are_labeled(
        self,
        small_sim_config: SimulationConfig,
        fraud_config: FraudConfig,
    ) -> None:
        acc_gen = AccountGenerator(small_sim_config)
        accounts = acc_gen.generate()

        engine = FraudInjectionEngine(small_sim_config, fraud_config, accounts)
        _, fraud_txns = engine.inject_all()

        if fraud_txns.height > 0:
            assert fraud_txns.filter(pl.col("is_fraud")).height == fraud_txns.height

    def test_fraud_ratio_approximate(
        self,
        small_sim_config: SimulationConfig,
        fraud_config: FraudConfig,
    ) -> None:
        acc_gen = AccountGenerator(small_sim_config)
        accounts = acc_gen.generate()

        engine = FraudInjectionEngine(small_sim_config, fraud_config, accounts)
        updated_accounts, _ = engine.inject_all()

        fraud_count = updated_accounts.filter(pl.col("is_fraud")).height
        ratio = fraud_count / updated_accounts.height

        assert ratio <= 0.05

    def test_fraud_patterns_present(
        self,
        small_sim_config: SimulationConfig,
        fraud_config: FraudConfig,
    ) -> None:
        acc_gen = AccountGenerator(small_sim_config)
        accounts = acc_gen.generate()

        engine = FraudInjectionEngine(small_sim_config, fraud_config, accounts)
        _, fraud_txns = engine.inject_all()

        if fraud_txns.height > 0:
            patterns = fraud_txns["fraud_pattern"].unique().to_list()
            assert len(patterns) >= 1
