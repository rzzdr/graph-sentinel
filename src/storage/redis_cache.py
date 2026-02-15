from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore[import-untyped]

    from src.config.settings import RedisConfig

logger = structlog.get_logger(__name__)


class RedisCache:
    def __init__(self, config: RedisConfig) -> None:
        self.config = config
        self._client: AsyncRedis | None = None

    async def connect(self) -> None:
        from redis.asyncio import Redis

        self._client = Redis(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            decode_responses=True,
        )
        await self._client.ping()
        logger.info("redis_connected")

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    @property
    def client(self) -> AsyncRedis:
        if self._client is None:
            msg = "Redis not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._client

    async def cache_risk_score(
        self,
        account_id: str,
        score_data: dict[str, Any],
        ttl: int = 3600,
    ) -> None:
        key = f"risk_score:{account_id}"
        await self.client.setex(key, ttl, json.dumps(score_data))

    async def get_risk_score(self, account_id: str) -> dict[str, Any] | None:
        key = f"risk_score:{account_id}"
        data = await self.client.get(key)
        if data is None:
            return None
        return json.loads(data)  # type: ignore[no-any-return]

    async def cache_features(
        self,
        account_id: str,
        features: dict[str, Any],
        ttl: int = 1800,
    ) -> None:
        key = f"features:{account_id}"
        await self.client.setex(key, ttl, json.dumps(features))

    async def get_features(self, account_id: str) -> dict[str, Any] | None:
        key = f"features:{account_id}"
        data = await self.client.get(key)
        if data is None:
            return None
        return json.loads(data)  # type: ignore[no-any-return]

    async def invalidate_account(self, account_id: str) -> None:
        keys = [f"risk_score:{account_id}", f"features:{account_id}"]
        await self.client.delete(*keys)

    async def get_cache_stats(self) -> dict[str, Any]:
        info = await self.client.info("memory")
        return {
            "used_memory": info.get("used_memory_human", "unknown"),
            "keys": await self.client.dbsize(),
        }
