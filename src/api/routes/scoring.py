from __future__ import annotations

import polars as pl
from fastapi import APIRouter, HTTPException

from src.api.schemas import DecompositionResponse, RiskScoreRequest, RiskScoreResponse
from src.config.settings import Settings

router = APIRouter()


def _load_risk_scores() -> pl.DataFrame:
    settings = Settings()
    path = settings.output_dir / "risk_scores.parquet"
    if not path.exists():
        raise HTTPException(
            status_code=404, detail="Risk scores not computed yet. Run the pipeline first."
        )
    return pl.read_parquet(path)


def _load_features() -> pl.DataFrame:
    settings = Settings()
    path = settings.output_dir / "features.parquet"
    if not path.exists():
        raise HTTPException(
            status_code=404, detail="Features not computed yet. Run the pipeline first."
        )
    return pl.read_parquet(path)


@router.get("/scores/{account_id}", response_model=RiskScoreResponse)  # type: ignore[misc]
async def get_risk_score(account_id: str) -> RiskScoreResponse:
    scores = _load_risk_scores()
    account = scores.filter(pl.col("account_id") == account_id)

    if account.height == 0:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")

    row = account.row(0, named=True)
    return RiskScoreResponse(**row)


@router.post("/scores/batch", response_model=list[RiskScoreResponse])  # type: ignore[misc]
async def get_batch_risk_scores(request: RiskScoreRequest) -> list[RiskScoreResponse]:
    scores = _load_risk_scores()
    filtered = scores.filter(pl.col("account_id").is_in(request.account_ids))
    return [RiskScoreResponse(**row) for row in filtered.iter_rows(named=True)]


@router.get("/scores", response_model=list[RiskScoreResponse])  # type: ignore[misc]
async def get_top_risk_scores(
    limit: int = 50,
    min_score: float = 0.0,
) -> list[RiskScoreResponse]:
    limit = min(limit, 1000)
    scores = _load_risk_scores()
    filtered = (
        scores.filter(pl.col("risk_score") >= min_score)
        .sort("risk_score", descending=True)
        .head(limit)
    )
    return [RiskScoreResponse(**row) for row in filtered.iter_rows(named=True)]


@router.get("/scores/{account_id}/decomposition", response_model=DecompositionResponse)  # type: ignore[misc]
async def get_score_decomposition(account_id: str) -> DecompositionResponse:
    scores = _load_risk_scores()
    features = _load_features()

    from src.explainability.decomposition import RiskDecomposer

    decomposer = RiskDecomposer()
    result = decomposer.decompose_account(account_id, features, scores)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return DecompositionResponse(**result)
