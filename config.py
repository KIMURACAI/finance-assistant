"""Global configuration with env-based overrides."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_ROOT: Path = Path(__file__).parent
    LOG_DIR: Path = PROJECT_ROOT / "logs"
    DB_PATH: Path = PROJECT_ROOT / "data" / "finance.db"

    # DeepSeek
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_MAX_TOKENS: int = 800
    DEEPSEEK_TEMPERATURE: float = 0.2

    # WeChat Official Account
    WECHAT_APP_ID: str = ""
    WECHAT_APP_SECRET: str = ""
    WECHAT_TOKEN: str = "finance123"

    # ServerChan (optional fallback)
    SERVERCHAN_SENDKEY: str = ""

    # Scheduler
    PUSH_TIME_MORNING: str = "08:30"
    PUSH_TIME_EVENING: str = "15:05"

    # Cache TTLs (seconds)
    MARKET_CACHE_TTL: int = 180
    NEWS_CACHE_TTL: int = 300
    AI_RESPONSE_CACHE_TTL: int = 600

    # Tavily Search
    TAVILY_API_KEY: str = "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"

    # Rate limits
    DEEPSEEK_RPM: int = 10
    WECHAT_MSG_DAILY_LIMIT: int = 200

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
(settings.PROJECT_ROOT / "data").mkdir(exist_ok=True)
