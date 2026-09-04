from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/stock_analysis"
    news_api_key: str = ""
    finnhub_api_key: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_enabled: bool = False
    alert_min_confidence: float = 0.0
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "extra": "ignore"}
