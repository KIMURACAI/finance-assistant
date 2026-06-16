"""金融资讯助手 - FastAPI 入口
微信公众号测试号 + DeepSeek 的个性化金融资讯推送
"""

import os
import sys
import hashlib
import time
from contextlib import asynccontextmanager
from xml.etree import ElementTree as ET

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from loguru import logger
import uvicorn

from config import settings
from database.db import init_db, get_or_create_user, add_chat
from handlers.message_handler import handle_user_message, push_daily_briefing
from scheduler.daily_task import start_scheduler, stop_scheduler
from pusher.wxpusher_client import send_text as serverchan_send
from wechat.official_account import (
    verify_signature, parse_message, build_text_reply,
    send_customer_message,
)


def setup_logger():
    logger.remove()
    logger.add(
        settings.LOG_DIR / "app_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level="INFO",
        encoding="utf-8",
    )
    logger.add(
        lambda msg: sys.stdout.buffer.write(msg.encode("utf-8", errors="replace")),
        level="INFO",
        colorize=False,
        format="<level>{level}</level> | {message}",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger()
    logger.info("正在初始化...")
    if not settings.DEEPSEEK_API_KEY or settings.DEEPSEEK_API_KEY == "sk-your-deepseek-api-key-here":
        logger.warning("DeepSeek API Key 未配置")
    if not settings.WECHAT_APP_ID:
        logger.warning("微信公众号 appID 未配置")
    await init_db()
    start_scheduler()
    logger.info("金融资讯助手启动成功！")
    logger.info(f"早间简报: {settings.PUSH_TIME_MORNING}")
    logger.info(f"收盘简报: {settings.PUSH_TIME_EVENING}")
    yield
    stop_scheduler()
    logger.info("服务已关闭")


app = FastAPI(
    title="金融资讯助手",
    description="微信公众号 + DeepSeek 个性化金融资讯",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── 微信公众号回调 ──────────────────────────────────
@app.api_route("/wechat/callback", methods=["GET", "POST"])
async def wechat_callback(request: Request):
    """微信公众号消息回调"""
    if request.method == "GET":
        # 验证服务器地址
        params = dict(request.query_params)
        sig = params.get("signature", "")
        ts = params.get("timestamp", "")
        nonce = params.get("nonce", "")
        echostr = params.get("echostr", "")

        if await verify_signature(sig, ts, nonce):
            return PlainTextResponse(echostr)
        return PlainTextResponse("invalid")

    # POST = 用户发消息
    body = await request.body()
    msg = parse_message(body)
    logger.info(f"收到微信消息: {msg}")

    msg_type = msg.get("MsgType", "")
    content = msg.get("Content", "").strip()
    from_user = msg.get("FromUserName", "")

    if msg_type == "text" and content and from_user:
        # 异步处理用户消息
        import asyncio
        asyncio.ensure_future(handle_wechat_message(from_user, content))

    # 微信要求 5 秒内返回
    return PlainTextResponse("")


async def handle_wechat_message(openid: str, content: str):
    """处理微信用户消息并回复"""
    try:
        # 用 openid 作为用户标识
        user = await get_or_create_user(openid, name="微信用户")
        user_id = user.id

        # 保存用户消息
        await add_chat(user_id, "user", content, msg_type="text")

        # 调用 handle_user_message（它内部会调 AI + 回复）
        from handlers.message_handler import handle_user_message as handle_msg
        reply = await handle_msg(openid, content)

        # 通过客服消息回复（直接在公众号对话里显示）
        if reply:
            await send_customer_message(openid, reply)
    except Exception as e:
        logger.error(f"处理微信消息异常: {e}")
        try:
            await send_customer_message(openid, "抱歉，我暂时无法处理，请稍后再试。")
        except:
            pass


# ─── WxPusher 回调（保留）──────────────────────────────
@app.api_route("/wxpusher/callback", methods=["GET", "POST"])
async def wxpusher_callback(request: Request):
    if request.method == "GET":
        return PlainTextResponse("ok")
    try:
        body = await request.json()
        data = body.get("data", {})
        uid = data.get("uid", "")
        content = data.get("content", "")
        if uid and content:
            import asyncio
            asyncio.ensure_future(handle_user_message(uid, content))
    except Exception as e:
        logger.warning(f"WxPusher 回调异常: {e}")
    return PlainTextResponse("ok")


# ─── 手动推送 ──────────────────────────────────────────
@app.post("/push/now/{uid}")
async def manual_push(uid: str = ""):
    from handlers.message_handler import push_daily_briefing
    await push_daily_briefing(uid or "default", "evening")
    return {"status": "ok"}


@app.post("/push/morning/{uid}")
async def manual_morning(uid: str = ""):
    from handlers.message_handler import push_daily_briefing
    await push_daily_briefing(uid or "default", "morning")
    return {"status": "ok"}


# ─── 健康检查 ──────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "running",
        "deepseek_ok": bool(settings.DEEPSEEK_API_KEY),
        "wechat_ok": bool(settings.WECHAT_APP_ID),
        "morning_push": settings.PUSH_TIME_MORNING,
        "evening_push": settings.PUSH_TIME_EVENING,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.PORT))
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=port,
        reload=False,
        log_level="info",
    )
