from src.models.schemas import Signal, RiskLevel

# --- Analyst Voting Weights (must sum to 1.0) ---
TECHNICAL_WEIGHT = 0.30
FUNDAMENTAL_WEIGHT = 0.40
SENTIMENT_WEIGHT = 0.30

# --- Signal to Numeric Score ---
SIGNAL_TO_SCORE: dict[Signal, float] = {
    Signal.STRONG_BUY: 1.0,
    Signal.BUY: 0.5,
    Signal.HOLD: 0.0,
    Signal.SELL: -0.5,
    Signal.STRONG_SELL: -1.0,
}

# --- Score Thresholds for Final Signal ---
SCORE_THRESHOLDS = {
    "strong_buy": 0.6,
    "buy": 0.2,
    "sell": -0.2,
    "strong_sell": -0.6,
}

# --- Risk Dampening Factors ---
RISK_DAMPENING: dict[RiskLevel, float] = {
    RiskLevel.LOW: 1.0,
    RiskLevel.MODERATE: 0.85,
    RiskLevel.HIGH: 0.60,
    RiskLevel.EXTREME: 0.25,
}

# --- Circuit Breaker Thresholds ---
VOLATILITY_EXTREME_THRESHOLD = 0.95
MAX_DRAWDOWN_SEVERE_THRESHOLD = -0.20
DATA_STALENESS_LIMIT_HOURS = 1.0
MAX_POSITION_SIZE_PCT = 0.10
CIRCUIT_BREAKER_POSITION_CAP = 0.02

# --- Cache ---
RECOMMENDATION_CACHE_TTL_SECONDS = 900
MARKET_DATA_CACHE_TTL_SECONDS = 300
