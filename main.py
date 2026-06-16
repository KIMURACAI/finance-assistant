"""金融资讯助手 - FastAPI 入口
基于 DeepSeek + Server酱 的个性化金融资讯推送服务
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from loguru import logger
import uvicorn

from config import settings
from database.db import init_db
from scheduler.daily_task import start_scheduler, stop_scheduler


def setup_logger():
    """配置日志输出"""
    logger.remove()

    # 文件日志（UTF-8 完整记录）
    logger.add(
        settings.LOG_DIR / "app_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level="INFO",
        encoding="utf-8",
    )

    # 控制台输出（Windows GBK 兼容）
    logger.add(
        lambda msg: sys.stdout.buffer.write(msg.encode("utf-8", errors="replace")),
        level="INFO",
        colorize=False,
        format="<level>{level}</level> | {message}",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    setup_logger()
    logger.info("正在初始化...")

    # 检查配置
    if not settings.DEEPSEEK_API_KEY or settings.DEEPSEEK_API_KEY == "sk-your-deepseek-api-key-here":
        logger.warning("DeepSeek API Key 未配置！请修改 .env 文件")
    if not settings.SERVERCHAN_SENDKEY:
        logger.warning("Server酱 SendKey 未配置！请修改 .env 文件")

    # 初始化数据库
    await init_db()
    logger.info("数据库初始化完成")

    # 启动定时任务
    start_scheduler()

    logger.info("金融资讯助手启动成功！")
    logger.info(f"服务地址: http://{settings.HOST}:{settings.PORT}")
    logger.info(f"早间简报: {settings.PUSH_TIME_MORNING}")
    logger.info(f"收盘简报: {settings.PUSH_TIME_EVENING}")

    yield

    stop_scheduler()
    logger.info("服务已关闭")


app = FastAPI(
    title="金融资讯助手",
    description="基于 DeepSeek + WxPusher 的个性化金融资讯推送",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── WxPusher 消息回调（接收用户回复）─────────────────
@app.api_route("/wxpusher/callback", methods=["GET", "POST"])
async def wxpusher_callback(request: Request):
    """
    WxPusher 回调地址（可选配置）
    用户在你的 WxPusher 应用里发消息时会回调这里
    """
    if request.method == "GET":
        return PlainTextResponse("ok")

    try:
        body = await request.json()
        logger.info(f"WxPusher 回调: {body}")

        # WxPusher 回调格式: { "action": "..." , "data": { "uid": "...", "content": "..." } }
        data = body.get("data", {})
        uid = data.get("uid", "")
        content = data.get("content", "")

        if uid and content:
            import asyncio
            asyncio.ensure_future(handle_user_message(uid, content))
    except Exception as e:
        logger.warning(f"WxPusher 回调处理异常: {e}")

    return PlainTextResponse("ok")


# ─── 手动触发推送（调试用）─────────────────────────────
@app.post("/push/now/{uid}")
async def manual_push(uid: str = ""):
    """手动推送实时简报"""
    from handlers.message_handler import push_daily_briefing
    if not settings.SERVERCHAN_SENDKEY:
        return {"status": "error", "message": "未配置 SendKey"}
    result = await push_daily_briefing(uid or "default", "evening")
    return {"status": "ok"}


@app.post("/push/morning/{uid}")
async def manual_morning(uid: str = ""):
    """手动推送早间简报"""
    from handlers.message_handler import push_daily_briefing
    result = await push_daily_briefing(uid or "default", "morning")
    return {"status": "ok"}


# ─── 健康检查 ──────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "running",
        "deepseek_ok": bool(settings.DEEPSEEK_API_KEY and settings.DEEPSEEK_API_KEY != "sk-your-deepseek-api-key-here"),
        "serverchan_ok": bool(settings.SERVERCHAN_SENDKEY),
        "morning_push": settings.PUSH_TIME_MORNING,
        "evening_push": settings.PUSH_TIME_EVENING,
    }


# ─── 入口 ───────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.PORT))
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=port,
        reload=False,
        log_level="info",
    )
