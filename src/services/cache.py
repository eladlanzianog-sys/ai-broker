"""Redis cache for analysis results."""
import redis.asyncio as redis

from src.config.settings import Settings
from src.models.schemas import FinalRecommendation

_pool: redis.Redis | None = None


async def get_redis() -> redis.Redis:
    global _pool
    if _pool is None:
        settings = Settings()
        _pool = redis.from_url(settings.redis_url)
    return _pool


async def get_cached_recommendation(ticker: str) -> FinalRecommendation | None:
    r = await get_redis()
    key = f"stock-analysis:{ticker}:recommendation"
    data = await r.get(key)
    if data:
        return FinalRecommendation.model_validate_json(data)
    return None


async def cache_recommendation(
    ticker: str,
    rec: FinalRecommendation,
    ttl_seconds: int = 900,
) -> None:
    r = await get_redis()
    key = f"stock-analysis:{ticker}:recommendation"
    await r.set(key, rec.model_dump_json(), ex=ttl_seconds)
