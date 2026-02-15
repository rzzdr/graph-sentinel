from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TransactionProfile:
    daily_txn_rate: float
    amount_mu: float
    amount_sigma: float
    p2p_ratio: float
    ecommerce_ratio: float
    bill_ratio: float
    salary_ratio: float


PROFILE_TEMPLATES: dict[str, dict[str, TransactionProfile]] = {
    "savings": {
        "basic": TransactionProfile(1.2, 4.0, 1.2, 0.20, 0.30, 0.30, 0.20),
        "standard": TransactionProfile(2.0, 4.5, 1.0, 0.15, 0.35, 0.25, 0.25),
        "enhanced": TransactionProfile(3.0, 5.0, 0.9, 0.15, 0.35, 0.25, 0.25),
        "premium": TransactionProfile(4.0, 5.5, 0.8, 0.10, 0.40, 0.20, 0.30),
    },
    "business": {
        "basic": TransactionProfile(5.0, 6.0, 1.5, 0.05, 0.10, 0.60, 0.25),
        "standard": TransactionProfile(8.0, 6.5, 1.3, 0.05, 0.10, 0.55, 0.30),
        "enhanced": TransactionProfile(12.0, 7.0, 1.2, 0.05, 0.10, 0.50, 0.35),
        "premium": TransactionProfile(15.0, 7.5, 1.0, 0.05, 0.05, 0.50, 0.40),
    },
    "wallet": {
        "basic": TransactionProfile(2.5, 3.5, 1.3, 0.40, 0.40, 0.10, 0.10),
        "standard": TransactionProfile(3.5, 4.0, 1.1, 0.35, 0.40, 0.15, 0.10),
        "enhanced": TransactionProfile(4.5, 4.5, 1.0, 0.30, 0.40, 0.15, 0.15),
        "premium": TransactionProfile(6.0, 5.0, 0.9, 0.25, 0.40, 0.15, 0.20),
    },
}


class BehavioralModel:
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def get_profile(self, account_type: str, kyc_level: str) -> TransactionProfile:
        templates = PROFILE_TEMPLATES.get(account_type, PROFILE_TEMPLATES["savings"])
        profile = templates.get(kyc_level, templates["standard"])
        noise = self.rng.normal(1.0, 0.1)
        return TransactionProfile(
            daily_txn_rate=max(0.1, profile.daily_txn_rate * noise),
            amount_mu=profile.amount_mu + self.rng.normal(0, 0.2),
            amount_sigma=max(0.3, profile.amount_sigma + self.rng.normal(0, 0.1)),
            p2p_ratio=profile.p2p_ratio,
            ecommerce_ratio=profile.ecommerce_ratio,
            bill_ratio=profile.bill_ratio,
            salary_ratio=profile.salary_ratio,
        )

    def sample_daily_txn_count(self, rate: float) -> int:
        return int(self.rng.poisson(rate))

    def sample_amount(self, mu: float, sigma: float) -> float:
        return float(np.clip(self.rng.lognormal(mu, sigma), 0.01, 1_000_000))

    def sample_transaction_type(self, profile: TransactionProfile) -> str:
        categories = ["p2p", "ecommerce", "bill_payment", "salary"]
        weights = [
            profile.p2p_ratio,
            profile.ecommerce_ratio,
            profile.bill_ratio,
            profile.salary_ratio,
        ]
        total = sum(weights)
        weights = [w / total for w in weights]
        return str(self.rng.choice(categories, p=weights))

    def apply_time_of_day_pattern(self, hour: int) -> float:
        """Probability multiplier based on hour-of-day — peaks at business hours."""
        if 0 <= hour < 6:
            return 0.1
        if 6 <= hour < 9:
            return 0.6
        if 9 <= hour < 17:
            return 1.0
        if 17 <= hour < 21:
            return 0.7
        return 0.3

    def apply_day_of_week_pattern(self, weekday: int) -> float:
        """Probability multiplier based on day of week (0=Monday)."""
        weekend_factor = 0.4 if weekday >= 5 else 1.0
        return weekend_factor
