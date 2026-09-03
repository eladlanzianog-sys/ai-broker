from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/stock_analysis"
    news_api_key: str = ""
    finnhub_api_key: str = ""
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "extra": "ignore"}
