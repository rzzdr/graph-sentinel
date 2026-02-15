from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class RiskScoreResponse(BaseModel):
    account_id: str
    risk_score: float
    risk_level: str
    behavioral_risk_score: float
    structural_risk_score: float
    network_propagation_score: float


class RiskScoreRequest(BaseModel):
    account_ids: list[str] = Field(min_length=1, max_length=1000)


class AlertResponse(BaseModel):
    alert_id: int
    account_id: str
    risk_score: float
    risk_level: str
    alert_type: str
    created_at: datetime
    resolved: bool


class AlertListResponse(BaseModel):
    alerts: list[AlertResponse]
    total: int


class SimulationRequest(BaseModel):
    total_accounts: int = Field(default=10_000, ge=1_000, le=500_000)
    simulation_days: int = Field(default=90, ge=30, le=365)
    total_transactions: int = Field(default=500_000, ge=10_000, le=20_000_000)
    fraud_ratio: float = Field(default=0.015, ge=0.005, le=0.03)


class SimulationResponse(BaseModel):
    status: str
    total_accounts: int
    total_transactions: int
    fraud_accounts: int
    fraud_transactions: int


class HealthResponse(BaseModel):
    status: str
    version: str
    mode: str


class DecompositionResponse(BaseModel):
    account_id: str
    risk_score: float
    risk_level: str
    top_contributing_features: list[dict[str, object]]
    behavioral_risk_score: float
    structural_risk_score: float
    network_propagation_score: float
