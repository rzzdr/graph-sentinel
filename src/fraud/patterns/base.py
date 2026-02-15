from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class FraudPattern(abc.ABC):
    @abc.abstractmethod
    def generate(
        self,
        mule_ids: list[str],
        all_account_ids: list[str],
        rng: np.random.Generator,
    ) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
        """Generate fraud transactions and return (transactions, role_mapping).

        Returns:
            transactions: list of transaction dicts
            role_mapping: dict mapping role names to lists of account_ids
        """
        ...
