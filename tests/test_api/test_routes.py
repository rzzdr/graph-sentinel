from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI

from src.api.app import create_app


@pytest.fixture  # type: ignore[misc]
def app() -> FastAPI:
    return create_app()


@pytest.fixture  # type: ignore[misc]
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoints:
    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_health_check(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_readiness(self, client: AsyncClient) -> None:
        response = await client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"


class TestAlertEndpoints:
    @pytest.mark.asyncio  # type: ignore[misc]
    async def test_alert_stats(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/alerts/stats")
        assert response.status_code == 200
