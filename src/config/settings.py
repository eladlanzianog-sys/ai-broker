from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/stock_analysis"
    finnhub_api_key: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_enabled: bool = False
    alert_min_confidence: float = 0.0
    log_level: str = "INFO"

    watchlist: str = "AAPL,MSFT,GOOGL,NVDA,META,AMZN,TSLA,JPM,V,BRK-B"
    scan_hour: int = 8
    scan_minute: int = 0
    scan_only_actionable: bool = True

    model_config = {"env_file": ".env", "extra": "ignore"}

    def get_watchlist(self) -> list[str]:
        return [t.strip().upper() for t in self.watchlist.split(",") if t.strip()]
