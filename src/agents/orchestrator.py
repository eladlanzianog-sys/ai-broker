"""Orchestrator agent.

Two node functions:
  - orchestrator_entry: Validates the request, checks cache, initializes state.
  - orchestrator_exit: Validates the recommendation, persists results, updates cache.
"""
from src.agents.state import AnalysisState
from src.services.cache import cache_recommendation, get_cached_recommendation
from src.services.persistence import save_analysis_run


async def orchestrator_entry(state: AnalysisState) -> dict:
    request = state["request"]

    if not request.ticker.isalpha() or len(request.ticker) > 5:
        return {
            "errors": [f"Invalid ticker format: {request.ticker}"],
            "audit_log": [
                f"[Orchestrator] REJECTED invalid ticker: {request.ticker}"
            ],
        }

    try:
        cached = await get_cached_recommendation(request.ticker)
        if cached:
            return {
                "recommendation": cached,
                "audit_log": [f"[Orchestrator] Cache HIT for {request.ticker}"],
            }
    except Exception:
        pass

    return {
        "audit_log": [
            f"[Orchestrator] Starting analysis for {request.ticker} "
            f"(request_id={request.request_id})"
        ],
    }


async def orchestrator_exit(state: AnalysisState) -> dict:
    rec = state.get("recommendation")
    if rec is None:
        return {
            "errors": ["Pipeline completed without producing a recommendation"],
            "audit_log": ["[Orchestrator] ERROR: No recommendation produced"],
        }

    try:
        await save_analysis_run(
            request=state["request"],
            market_data=state.get("market_data"),
            technical_report=state.get("technical_report"),
            fundamental_report=state.get("fundamental_report"),
            sentiment_report=state.get("sentiment_report"),
            risk_assessment=state.get("risk_assessment"),
            recommendation=rec,
        )
    except Exception:
        pass

    try:
        await cache_recommendation(rec.ticker, rec, ttl_seconds=900)
    except Exception:
        pass

    return {
        "audit_log": [
            f"[Orchestrator] Analysis complete for {rec.ticker}: "
            f"{rec.action.value} (confidence={rec.confidence:.2f})"
        ],
    }
