"""Reliable daily scheduler with error isolation."""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from config import settings
from handlers.message_handler import push_daily_briefing

scheduler = AsyncIOScheduler()


async def _safe_run(name: str, push_type: str):
    """Isolated job execution."""
    if not settings.WECHAT_APP_ID and not settings.SERVERCHAN_SENDKEY:
        logger.warning("No push channel configured, skip")
        return
    try:
        logger.info(f"Running {name}")
        await push_daily_briefing("default", push_type)
        logger.info(f"{name} done")
    except Exception as e:
        logger.error(f"{name} failed: {e}")


def start_scheduler():
    h_m, m_m = settings.PUSH_TIME_MORNING.split(":")
    scheduler.add_job(
        lambda: _safe_run("Morning", "morning"),
        CronTrigger(hour=int(h_m), minute=int(m_m)),
        id="morning", replace_existing=True, misfire_grace_time=300,
    )
    h_e, m_e = settings.PUSH_TIME_EVENING.split(":")
    scheduler.add_job(
        lambda: _safe_run("Evening", "evening"),
        CronTrigger(hour=int(h_e), minute=int(m_e)),
        id="evening", replace_existing=True, misfire_grace_time=300,
    )
    scheduler.start()
    logger.info(f"Scheduler: {settings.PUSH_TIME_MORNING} / {settings.PUSH_TIME_EVENING}")


def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown(wait=False)
