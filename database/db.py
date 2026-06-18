"""Optimized async database operations with session reuse."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, async_sessionmaker,
)

from config import settings
from .models import Base, User, Position, UserPreference, PushLog, ChatHistory

DATABASE_URL = f"sqlite+aiosqlite:///{settings.DB_PATH}"

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
    pool_size=5,
    max_overflow=10,
)
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ─── User ─────────────────────────────────────────────
async def get_all_users() -> list[User]:
    """Get all registered users (for scheduled pushes)."""
    async with AsyncSessionLocal() as ses:
        result = await ses.execute(select(User))
        return list(result.scalars().all())


async def get_recent_active_users(hours: int = 48) -> list[User]:
    """Get users who chatted within N hours (WeChat 48h window)."""
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    async with AsyncSessionLocal() as ses:
        result = await ses.execute(
            select(User).join(ChatHistory, ChatHistory.user_id == User.id)
            .where(ChatHistory.created_at >= cutoff)
            .distinct()
        )
        return list(result.scalars().all())


async def get_or_create_user(wecom_user_id: str, name: str = "") -> User:
    async with AsyncSessionLocal() as ses:
        result = await ses.execute(select(User).where(User.wecom_user_id == wecom_user_id))
        user = result.scalar_one_or_none()
        if user:
            if name and user.name != name:
                user.name = name
            # Always bump updated_at so is_new detection works correctly
            user.updated_at = datetime.now(timezone.utc)
            await ses.commit()
            return user

        user = User(wecom_user_id=wecom_user_id, name=name)
        ses.add(user)
        await ses.flush()
        pref = UserPreference(user_id=user.id)
        ses.add(pref)
        await ses.commit()
        return user


# ─── Positions ────────────────────────────────────────
async def add_position(
    user_id: int, asset_code: str, asset_name: str = "",
    asset_type: str = "stock", market: str = "A",
    weight: float = 0.0, notes: str = "",
) -> Position:
    async with AsyncSessionLocal() as ses:
        result = await ses.execute(
            select(Position).where(
                Position.user_id == user_id,
                Position.asset_code == asset_code,
                Position.market == market,
            )
        )
        exist = result.scalar_one_or_none()
        if exist:
            exist.asset_name = asset_name or exist.asset_name
            exist.weight = weight
            await ses.commit()
            return exist
        pos = Position(user_id=user_id, asset_code=asset_code,
                       asset_name=asset_name, asset_type=asset_type,
                       market=market, weight=weight, notes=notes)
        ses.add(pos)
        await ses.commit()
        return pos


async def remove_position(user_id: int, position_id: int) -> bool:
    async with AsyncSessionLocal() as ses:
        result = await ses.execute(
            select(Position).where(Position.id == position_id, Position.user_id == user_id)
        )
        pos = result.scalar_one_or_none()
        if not pos:
            return False
        await ses.delete(pos)
        await ses.commit()
        return True


async def remove_position_by_code(user_id: int, asset_code: str, market: str = "A") -> bool:
    async with AsyncSessionLocal() as ses:
        result = await ses.execute(
            select(Position).where(
                Position.user_id == user_id,
                Position.asset_code == asset_code,
                Position.market == market,
            )
        )
        pos = result.scalar_one_or_none()
        if not pos:
            return False
        await ses.delete(pos)
        await ses.commit()
        return True


async def get_user_positions(user_id: int) -> list[Position]:
    async with AsyncSessionLocal() as ses:
        result = await ses.execute(
            select(Position).where(Position.user_id == user_id)
            .order_by(Position.asset_code)
        )
        return list(result.scalars().all())


async def get_or_create_pref(user_id: int) -> UserPreference:
    async with AsyncSessionLocal() as ses:
        result = await ses.execute(select(UserPreference).where(UserPreference.user_id == user_id))
        pref = result.scalar_one_or_none()
        if pref:
            return pref
        pref = UserPreference(user_id=user_id)
        ses.add(pref)
        await ses.commit()
        return pref


async def update_pref(user_id: int, **kwargs) -> UserPreference:
    async with AsyncSessionLocal() as ses:
        result = await ses.execute(select(UserPreference).where(UserPreference.user_id == user_id))
        pref = result.scalar_one_or_none()
        if not pref:
            pref = UserPreference(user_id=user_id)
            ses.add(pref)
        for k, v in kwargs.items():
            if hasattr(pref, k) and v is not None:
                setattr(pref, k, v)
        pref.updated_at = datetime.now(timezone.utc)
        await ses.commit()
        return pref


# ─── Chat History ─────────────────────────────────────
async def add_chat(user_id: int, role: str, content: str,
                   msg_type: str = "text", related_assets: str = "") -> ChatHistory:
    async with AsyncSessionLocal() as ses:
        ch = ChatHistory(user_id=user_id, role=role, content=content,
                         msg_type=msg_type, related_assets=related_assets)
        ses.add(ch)
        await ses.commit()
        return ch


async def get_recent_chats(user_id: int, limit: int = 10) -> list[ChatHistory]:
    """Get recent N chat messages (compressed)."""
    async with AsyncSessionLocal() as ses:
        result = await ses.execute(
            select(ChatHistory)
            .where(ChatHistory.user_id == user_id)
            .order_by(ChatHistory.created_at.desc())
            .limit(limit)
        )
        return list(reversed(result.scalars().all()))


# ─── Push Log ─────────────────────────────────────────
async def add_push_log(user_id: int, push_type: str, title: str,
                       summary: str, related_assets: str = "") -> PushLog:
    async with AsyncSessionLocal() as ses:
        pl = PushLog(user_id=user_id, push_type=push_type,
                     title=title, summary=summary[:200],
                     related_assets=related_assets)
        ses.add(pl)
        await ses.commit()
        return pl
