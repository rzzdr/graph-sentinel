from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, MetaData, String, Table, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

if TYPE_CHECKING:
    from src.config.settings import PostgresConfig

logger = structlog.get_logger(__name__)

metadata = MetaData()

accounts_table = Table(
    "accounts",
    metadata,
    Column("account_id", String(32), primary_key=True),
    Column("creation_date", DateTime),
    Column("account_type", String(20)),
    Column("kyc_level", String(20)),
    Column("geographic_region", String(50)),
    Column("device_fingerprint_id", String(24)),
    Column("is_fraud", Boolean, default=False),
    Column("fraud_role", String(30), default="normal"),
)

risk_scores_table = Table(
    "risk_scores",
    metadata,
    Column("account_id", String(32), primary_key=True),
    Column("behavioral_risk_score", Float),
    Column("structural_risk_score", Float),
    Column("network_propagation_score", Float),
    Column("risk_score", Float),
    Column("risk_level", String(20)),
    Column("updated_at", DateTime),
)

alerts_table = Table(
    "alerts",
    metadata,
    Column("alert_id", Integer, primary_key=True, autoincrement=True),
    Column("account_id", String(32), index=True),
    Column("risk_score", Float),
    Column("risk_level", String(20)),
    Column("alert_type", String(50)),
    Column("created_at", DateTime),
    Column("resolved", Boolean, default=False),
)


class PostgresStore:
    def __init__(self, config: PostgresConfig) -> None:
        self.config = config
        self.engine = create_async_engine(
            config.dsn,
            pool_size=10,
            max_overflow=20,
        )

    async def initialize(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
        logger.info("postgres_initialized")

    async def close(self) -> None:
        await self.engine.dispose()

    async def upsert_risk_scores(
        self,
        scores: list[dict[str, Any]],
    ) -> None:
        async with AsyncSession(self.engine) as session:
            for score in scores:
                await session.execute(
                    text("""
                        INSERT INTO risk_scores
                            (account_id, behavioral_risk_score, structural_risk_score,
                             network_propagation_score, risk_score, risk_level, updated_at)
                        VALUES
                            (:account_id, :behavioral_risk_score, :structural_risk_score,
                             :network_propagation_score, :risk_score, :risk_level, NOW())
                        ON CONFLICT (account_id) DO UPDATE SET
                            behavioral_risk_score = EXCLUDED.behavioral_risk_score,
                            structural_risk_score = EXCLUDED.structural_risk_score,
                            network_propagation_score = EXCLUDED.network_propagation_score,
                            risk_score = EXCLUDED.risk_score,
                            risk_level = EXCLUDED.risk_level,
                            updated_at = NOW()
                    """),
                    score,
                )
            await session.commit()

    async def create_alert(self, alert_data: dict[str, Any]) -> None:
        async with AsyncSession(self.engine) as session:
            await session.execute(
                text("""
                    INSERT INTO alerts
                        (account_id, risk_score, risk_level, alert_type, created_at)
                    VALUES
                        (:account_id, :risk_score, :risk_level, :alert_type, NOW())
                """),
                alert_data,
            )
            await session.commit()

    async def get_high_risk_accounts(
        self,
        threshold: float = 75.0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        async with AsyncSession(self.engine) as session:
            result = await session.execute(
                text("""
                    SELECT * FROM risk_scores
                    WHERE risk_score >= :threshold
                    ORDER BY risk_score DESC
                    LIMIT :limit
                """),
                {"threshold": threshold, "limit": limit},
            )
            return [dict(row._mapping) for row in result]

    async def get_alerts(
        self,
        resolved: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        async with AsyncSession(self.engine) as session:
            result = await session.execute(
                text("""
                    SELECT * FROM alerts
                    WHERE resolved = :resolved
                    ORDER BY created_at DESC
                    LIMIT :limit
                """),
                {"resolved": resolved, "limit": limit},
            )
            return [dict(row._mapping) for row in result]
