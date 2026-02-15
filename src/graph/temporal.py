from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl
import structlog

from src.graph.builder import GraphBuilder

if TYPE_CHECKING:
    import networkx as nx

    from src.config.settings import GraphConfig

logger = structlog.get_logger(__name__)


class TemporalGraphManager:
    def __init__(self, config: GraphConfig) -> None:
        self.config = config
        self.builder = GraphBuilder()
        self.windows: dict[str, nx.DiGraph] = {}

    def build_rolling_windows(
        self,
        transactions: pl.DataFrame,
        accounts: pl.DataFrame,
        reference_date: datetime | None = None,
    ) -> dict[str, nx.DiGraph]:
        if reference_date is None:
            max_ts = transactions["timestamp"].max()
            if max_ts is None:
                msg = "No transactions to build windows from"
                raise ValueError(msg)
            reference_date = max_ts

        assert reference_date is not None

        for window_days in self.config.rolling_windows_days:
            window_start = reference_date - timedelta(days=window_days)

            window_txns = transactions.filter(
                (pl.col("timestamp") >= window_start) & (pl.col("timestamp") <= reference_date)
            )

            key = f"window_{window_days}d"
            self.windows[key] = self.builder.build_from_transactions(window_txns, accounts)
            logger.info(
                "rolling_window_built",
                window=key,
                transactions=window_txns.height,
                nodes=self.windows[key].number_of_nodes(),
                edges=self.windows[key].number_of_edges(),
            )

        if self.config.include_lifetime:
            self.windows["lifetime"] = self.builder.build_from_transactions(transactions, accounts)
            logger.info(
                "lifetime_graph_built",
                nodes=self.windows["lifetime"].number_of_nodes(),
                edges=self.windows["lifetime"].number_of_edges(),
            )

        return self.windows

    def get_window(self, key: str) -> nx.DiGraph:
        if key not in self.windows:
            available = list(self.windows.keys())
            msg = f"Window '{key}' not found. Available: {available}"
            raise KeyError(msg)
        return self.windows[key]

    def get_primary_graph(self) -> nx.DiGraph:
        if "lifetime" in self.windows:
            return self.windows["lifetime"]
        if self.windows:
            return next(iter(self.windows.values()))
        msg = "No graphs built yet"
        raise RuntimeError(msg)
