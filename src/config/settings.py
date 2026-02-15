from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OperationMode(str, Enum):
    SIMULATION = "simulation"
    PRODUCTION = "production"


class AccountType(str, Enum):
    SAVINGS = "savings"
    BUSINESS = "business"
    WALLET = "wallet"


class SimulationConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GS_SIM_")

    total_accounts: int = Field(default=50_000, ge=100, le=500_000)
    simulation_days: int = Field(default=180, ge=7, le=365)
    total_transactions: int = Field(default=2_000_000, ge=1_000, le=20_000_000)
    fraud_ratio: float = Field(default=0.015, ge=0.005, le=0.03)
    mule_cluster_size_min: int = Field(default=3, ge=3)
    mule_cluster_size_max: int = Field(default=25, le=50)
    time_resolution_seconds: int = Field(default=1, ge=1)
    seed: int = Field(default=42)

    salary_day_range: tuple[int, int] = Field(default=(25, 5))
    salary_amount_mean: float = Field(default=4500.0)
    salary_amount_std: float = Field(default=2000.0)
    p2p_transfer_ratio: float = Field(default=0.15)
    ecommerce_ratio: float = Field(default=0.35)
    bill_payment_ratio: float = Field(default=0.25)

    @field_validator("mule_cluster_size_max")
    @classmethod
    def cluster_max_gte_min(cls, v: int, info: ValidationInfo) -> int:
        min_val = info.data.get("mule_cluster_size_min", 3)
        if v < min_val:
            msg = "mule_cluster_size_max must be >= mule_cluster_size_min"
            raise ValueError(msg)
        return v


class FraudConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GS_FRAUD_")

    funnel_account_ratio: float = Field(default=0.25)
    circular_flow_ratio: float = Field(default=0.20)
    layering_ratio: float = Field(default=0.25)
    dormant_activation_ratio: float = Field(default=0.15)
    device_sharing_ratio: float = Field(default=0.15)

    layering_min_hops: int = Field(default=3, ge=3)
    layering_max_hops: int = Field(default=7, le=10)
    circular_min_cycle: int = Field(default=3, ge=3)
    circular_max_cycle: int = Field(default=8, le=15)
    dormant_inactivity_days: int = Field(default=60, ge=30)
    funnel_min_inbound: int = Field(default=10, ge=5)

    burst_window_hours: int = Field(default=24, ge=1)
    retention_time_hours_max: float = Field(default=4.0, ge=0.5)

    @field_validator("circular_max_cycle")
    @classmethod
    def circular_max_gte_min(cls, v: int, info: ValidationInfo) -> int:
        min_val = info.data.get("circular_min_cycle", 3)
        if v < min_val:
            msg = "circular_max_cycle must be >= circular_min_cycle"
            raise ValueError(msg)
        return v

    @field_validator("layering_max_hops")
    @classmethod
    def layering_max_gte_min(cls, v: int, info: ValidationInfo) -> int:
        min_val = info.data.get("layering_min_hops", 3)
        if v < min_val:
            msg = "layering_max_hops must be >= layering_min_hops"
            raise ValueError(msg)
        return v


class GraphConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GS_GRAPH_")

    rolling_windows_days: list[int] = Field(default=[7, 30, 90])
    include_lifetime: bool = Field(default=True)
    min_edge_weight: int = Field(default=1, ge=1)


class Neo4jConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GS_NEO4J_")

    uri: str = Field(default="bolt://localhost:7687")
    user: str = Field(default="neo4j")
    password: str = Field(default="password")
    database: str = Field(default="neo4j")
    max_connection_pool_size: int = Field(default=50)


class PostgresConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GS_PG_")

    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    user: str = Field(default="graph_sentinel")
    password: str = Field(default="password")
    database: str = Field(default="graph_sentinel")

    @property
    def dsn(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GS_REDIS_")

    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: str | None = Field(default=None)

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class ScoringConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GS_SCORING_")

    behavioral_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    structural_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    propagation_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    alert_threshold: float = Field(default=75.0, ge=0.0, le=100.0)
    high_risk_threshold: float = Field(default=90.0, ge=0.0, le=100.0)

    use_meta_model: bool = Field(default=True)
    use_calibration: bool = Field(default=True)


class PropagationConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GS_PROP_")

    decay_factor: float = Field(default=0.5, ge=0.1, le=0.9)
    max_hops: int = Field(default=5, ge=1, le=10)
    max_iterations: int = Field(default=20, ge=5, le=100)


class GraphLearningConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GS_GL_")

    embedding_dim: int = Field(default=64, ge=8, le=256)
    walk_length: int = Field(default=40, ge=10, le=100)
    num_walks: int = Field(default=10, ge=3, le=50)


class SequenceConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GS_SEQ_")

    max_seq_len: int = Field(default=200, ge=50, le=1000)
    embedding_dim: int = Field(default=32, ge=8, le=128)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GS_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    mode: OperationMode = Field(default=OperationMode.SIMULATION)
    data_dir: Path = Field(default=Path("data"))
    output_dir: Path = Field(default=Path("outputs"))
    models_dir: Path = Field(default=Path("models"))
    log_level: str = Field(default="INFO")

    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    fraud: FraudConfig = Field(default_factory=FraudConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    propagation: PropagationConfig = Field(default_factory=PropagationConfig)
    graph_learning: GraphLearningConfig = Field(default_factory=GraphLearningConfig)
    sequence: SequenceConfig = Field(default_factory=SequenceConfig)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
