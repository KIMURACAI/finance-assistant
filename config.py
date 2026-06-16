"""全局配置文件 - 所有可配置项集中管理"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ─── 项目路径 ───────────────────────────────────────
    PROJECT_ROOT: Path = Path(__file__).parent
    LOG_DIR: Path = PROJECT_ROOT / "logs"
    DB_PATH: Path = PROJECT_ROOT / "finance_assistant.db"

    # ─── DeepSeek API ───────────────────────────────────
    DEEPSEEK_API_KEY: str = ""           # 填入你的 DeepSeek API Key
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_MAX_TOKENS: int = 2000
    DEEPSEEK_TEMPERATURE: float = 0.7

    # ─── Server酱（微信推送）──────────────────────────
    SERVERCHAN_SENDKEY: str = ""         # SendKey

    # ─── 推送配置 ───────────────────────────────────────
    PUSH_TIME_MORNING: str = "08:30"      # 早间简报推送时间
    PUSH_TIME_EVENING: str = "17:30"      # 收盘简报推送时间
    PUSH_RSSI_LIMIT: int = 5              # 每日最多推送多少条深度分析

    # ─── 数据采集 ───────────────────────────────────────
    FETCH_NEWS_LIMIT: int = 20            # 每次抓取多少条市场新闻
    STOCK_NEWS_DAYS: int = 1              # 回溯抓取最近几天的个股新闻

    # ─── Web 服务 ───────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000  # Railway 会通过环境变量 PORT 覆盖此值

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# 创建目录
settings.LOG_DIR.mkdir(exist_ok=True)
