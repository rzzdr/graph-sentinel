from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import SimulationRequest, SimulationResponse
from src.config.settings import Settings, SimulationConfig

router = APIRouter()


@router.post("/simulation/run", response_model=SimulationResponse)  # type: ignore[misc]
async def run_simulation(request: SimulationRequest) -> SimulationResponse:
    settings = Settings()
    settings.simulation = SimulationConfig(
        total_accounts=request.total_accounts,
        simulation_days=request.simulation_days,
        total_transactions=request.total_transactions,
        fraud_ratio=request.fraud_ratio,
    )

    from src.simulation.orchestrator import SimulationOrchestrator

    orchestrator = SimulationOrchestrator(settings)
    accounts, transactions = orchestrator.run()
    orchestrator.save()

    import polars as pl

    fraud_accounts = accounts.filter(pl.col("is_fraud")).height
    fraud_txns = transactions.filter(pl.col("is_fraud")).height

    return SimulationResponse(
        status="complete",
        total_accounts=accounts.height,
        total_transactions=transactions.height,
        fraud_accounts=fraud_accounts,
        fraud_transactions=fraud_txns,
    )


@router.get("/simulation/status")  # type: ignore[misc]
async def simulation_status() -> dict[str, object]:
    settings = Settings()
    data_dir = settings.data_dir

    accounts_exists = (data_dir / "accounts.parquet").exists()
    txns_exists = (data_dir / "transactions.parquet").exists()

    if accounts_exists and txns_exists:
        import polars as pl

        accounts = pl.read_parquet(data_dir / "accounts.parquet")
        transactions = pl.read_parquet(data_dir / "transactions.parquet")

        return {
            "status": "available",
            "accounts": accounts.height,
            "transactions": transactions.height,
            "fraud_accounts": accounts.filter(pl.col("is_fraud")).height,
        }

    return {"status": "not_run", "accounts": 0, "transactions": 0}
