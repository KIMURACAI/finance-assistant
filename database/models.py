"""SQLAlchemy 数据模型"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, Boolean, JSON, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker

from config import settings

DATABASE_URL = f"sqlite+aiosqlite:///{settings.DB_PATH}"

Base = declarative_base()


# ─── 用户表 ─────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    wecom_user_id = Column(String(128), unique=True, nullable=False, index=True, comment="企业微信用户ID")
    name = Column(String(64), default="", comment="用户昵称")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "wecom_user_id": self.wecom_user_id,
            "name": self.name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ─── 用户持仓表 ─────────────────────────────────────────
class Position(Base):
    """用户持仓股票 / 基金 / 指数"""
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    asset_type = Column(String(16), default="stock", comment="stock / fund / index")
    asset_code = Column(String(32), nullable=False, comment="股票代码 eg. 000001, 600519")
    asset_name = Column(String(64), default="", comment="简称")
    market = Column(String(8), default="A", comment="A / HK / US")
    weight = Column(Float, default=0.0, comment="用户自定义权重 0-100")
    notes = Column(String(256), default="", comment="用户备注")
    added_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "asset_type": self.asset_type,
            "asset_code": self.asset_code,
            "asset_name": self.asset_name,
            "market": self.market,
            "weight": self.weight,
            "notes": self.notes,
        }


# ─── 用户偏好表 ─────────────────────────────────────────
class UserPreference(Base):
    """用户的个性化偏好（由对话不断优化）"""
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True, unique=True)
    focus_keywords = Column(Text, default="", comment="重点关注关键词, 逗号分隔")
    industry_focus = Column(Text, default="", comment="关注行业, 逗号分隔")
    risk_level = Column(String(16), default="medium", comment="low / medium / high")
    push_frequency = Column(String(16), default="daily", comment="daily / realtime / weekly")
    extra_config = Column(JSON, default=dict, comment="其他偏好JSON")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


# ─── 资讯推送日志 ───────────────────────────────────────
class PushLog(Base):
    """每日推送记录"""
    __tablename__ = "push_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    push_type = Column(String(32), default="daily", comment="morning / evening / realtime")
    title = Column(String(256), default="")
    summary = Column(Text, default="")
    related_assets = Column(Text, default="", comment="涉及的持仓代码")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ─── 对话记录 ───────────────────────────────────────────
class ChatHistory(Base):
    """用户对话历史"""
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    role = Column(String(16), nullable=False, comment="user / assistant")
    content = Column(Text, nullable=False)
    msg_type = Column(String(32), default="text", comment="text / command")
    related_assets = Column(Text, default="", comment="关联的资产")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
