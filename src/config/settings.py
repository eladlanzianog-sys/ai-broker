from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = ""
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

    automation_mode: str = "advanced"
    automation_monitor_interval: int = 900
    automation_health_interval: int = 3600
    automation_max_errors_per_day: int = 20
    automation_position_alert_pct: float = -5.0

    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497
    ibkr_client_id: int = 1
    ibkr_trading_enabled: bool = False
    ibkr_risk_per_trade: float = 1.0
    ibkr_max_position_pct: float = 0.20
    ibkr_reward_ratio: float = 2.0
    ibkr_min_confidence: float = 0.5
    ibkr_atr_multiplier: float = 1.5

    model_config = {"env_file": ".env", "extra": "ignore"}

    def get_watchlist(self) -> list[str]:
        return [t.strip().upper() for t in self.watchlist.split(",") if t.strip()]
