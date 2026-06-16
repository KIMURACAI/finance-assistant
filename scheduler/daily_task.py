"""定时任务调度器 - 每日早晚推送金融简报"""

import asyncio
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from config import settings
from handlers.message_handler import push_daily_briefing


scheduler = AsyncIOScheduler()


async def _morning_job():
    """早间简报推送"""
    if not settings.SERVERCHAN_SENDKEY:
        logger.warning("未配置 Server酱 SendKey，跳过推送")
        return

    logger.info("执行早间简报推送")
    try:
        await push_daily_briefing("default", push_type="morning")
        logger.info("早间简报推送完成")
    except Exception as e:
        logger.error(f"早间简报推送失败: {e}")


async def _evening_job():
    """收盘简报推送"""
    if not settings.SERVERCHAN_SENDKEY:
        logger.warning("未配置 Server酱 SendKey，跳过推送")
        return

    logger.info("执行收盘简报推送")
    try:
        await push_daily_briefing("default", push_type="evening")
        logger.info("收盘简报推送完成")
    except Exception as e:
        logger.error(f"收盘简报推送失败: {e}")


def start_scheduler():
    """启动所有定时任务"""
    hour_m, min_m = settings.PUSH_TIME_MORNING.split(":")
    scheduler.add_job(
        _morning_job,
        CronTrigger(hour=int(hour_m), minute=int(min_m)),
        id="morning_briefing",
        replace_existing=True,
        misfire_grace_time=300,
    )
    logger.info(f"早间简报定时: {settings.PUSH_TIME_MORNING}")

    hour_e, min_e = settings.PUSH_TIME_EVENING.split(":")
    scheduler.add_job(
        _evening_job,
        CronTrigger(hour=int(hour_e), minute=int(min_e)),
        id="evening_briefing",
        replace_existing=True,
        misfire_grace_time=300,
    )
    logger.info(f"收盘简报定时: {settings.PUSH_TIME_EVENING}")

    scheduler.start()
    logger.info("定时任务调度器已启动")


def stop_scheduler():
    """停止调度器"""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("定时任务调度器已停止")
