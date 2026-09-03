"""FastAPI API endpoints."""
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from src.agents.graph import compile_graph
from src.models.schemas import AnalysisRequest, FinalRecommendation

router = APIRouter()


@router.post("/analyze", response_model=FinalRecommendation)
async def analyze_ticker(ticker: str, date_range_days: int = 365):
    request = AnalysisRequest(
        ticker=ticker.upper(),
        date_range_days=date_range_days,
        request_id=str(uuid4()),
    )
    graph = compile_graph()
    result = await graph.ainvoke(
        {"request": request, "errors": [], "audit_log": []},
    )
    if result.get("errors"):
        raise HTTPException(status_code=422, detail=result["errors"])
    if result.get("recommendation") is None:
        raise HTTPException(status_code=500, detail="No recommendation produced")
    return result["recommendation"]


@router.get("/health")
async def health():
    return {"status": "ok"}
