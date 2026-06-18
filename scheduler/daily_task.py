"""Daily scheduler — closing summary push to all active users."""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from config import settings
from database.db import get_recent_active_users
from handlers.message_handler import push_daily_briefing

scheduler = AsyncIOScheduler()


async def _safe_run(name: str, push_type: str):
    """Push daily briefing to all recently active users."""
    if not settings.WECHAT_APP_ID and not settings.SERVERCHAN_SENDKEY:
        logger.warning("No push channel configured, skip")
        return

    try:
        users = await get_recent_active_users(hours=48)
        if not users:
            logger.warning(f"{name}: no active users in 48h window, skip")
            return

        logger.info(f"{name}: pushing to {len(users)} active users")
        ok, fail = 0, 0
        for user in users:
            try:
                await push_daily_briefing(user.wecom_user_id, push_type)
                ok += 1
            except Exception as e:
                fail += 1
                logger.error(f"{name} failed for {user.wecom_user_id[:10]}: {e}")

        logger.info(f"{name} done: {ok} ok, {fail} failed")
    except Exception as e:
        logger.error(f"{name} scheduler error: {e}")


def start_scheduler():
    h_m, m_m = settings.PUSH_TIME_MORNING.split(":")
    scheduler.add_job(
        _safe_run,
        CronTrigger(hour=int(h_m), minute=int(m_m)),
        id="morning", replace_existing=True, misfire_grace_time=300,
        kwargs={"name": "Morning", "push_type": "morning"},
    )
    h_e, m_e = settings.PUSH_TIME_EVENING.split(":")
    scheduler.add_job(
        _safe_run,
        CronTrigger(hour=int(h_e), minute=int(m_e)),
        id="closing", replace_existing=True, misfire_grace_time=300,
        kwargs={"name": "Closing", "push_type": "closing"},
    )
    scheduler.start()
    logger.info(f"Scheduler: morning {settings.PUSH_TIME_MORNING} / closing {settings.PUSH_TIME_EVENING}")


def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown(wait=False)
