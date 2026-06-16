"""Finance Assistant - FastAPI entry point."""

import os
import sys
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from loguru import logger
import uvicorn

from config import settings
from database.db import init_db
from core import close_client
from handlers.message_handler import handle_user_message, push_daily_briefing
from scheduler.daily_task import start_scheduler, stop_scheduler
from wechat.official_account import (
    verify_signature, parse_message, build_text_reply, send_customer_message,
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
        format="{message}",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger()
    logger.info("Starting...")
    await init_db()
    start_scheduler()
    logger.info(f"Ready - DeepSeek:{bool(settings.DEEPSEEK_API_KEY)} WeChat:{bool(settings.WECHAT_APP_ID)}")
    yield
    stop_scheduler()
    await close_client()


app = FastAPI(title="Finance Assistant", version="2.0.0", lifespan=lifespan)


@app.api_route("/wechat/callback", methods=["GET", "POST"])
async def wechat_callback(request: Request):
    if request.method == "GET":
        p = dict(request.query_params)
        if await verify_signature(p.get("signature", ""), p.get("timestamp", ""), p.get("nonce", "")):
            return PlainTextResponse(p.get("echostr", ""))
        return PlainTextResponse("invalid")

    p = dict(request.query_params)
    if not await verify_signature(p.get("signature", ""), p.get("timestamp", ""), p.get("nonce", "")):
        return PlainTextResponse("invalid")

    body = await request.body()
    msg = parse_message(body)
    from_user = msg.get("FromUserName", "")
    to_user = msg.get("ToUserName", "")
    content = msg.get("Content", "").strip()

    if msg.get("MsgType") == "text" and content and from_user:
        # Try sync processing first (passive reply within 5s)
        try:
            reply = await asyncio.wait_for(
                handle_user_message(from_user, content),
                timeout=4.0,
            )
            if reply:
                return PlainTextResponse(
                    build_text_reply(from_user, to_user, reply),
                    media_type="application/xml",
                )
        except asyncio.TimeoutError:
            logger.info("Sync processing timed out, using background task")
            asyncio.create_task(_background_reply(from_user, content))
            return PlainTextResponse("")

    return PlainTextResponse("")


async def _background_reply(openid: str, content: str):
    """Fallback: process async and push via customer service."""
    try:
        reply = await handle_user_message(openid, content)
        if reply:
            await send_customer_message(openid, reply)
            logger.info(f"Background reply sent to {openid[:10]}")
    except Exception as e:
        logger.error(f"Background reply error: {e}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "deepseek": bool(settings.DEEPSEEK_API_KEY),
        "wechat": bool(settings.WECHAT_APP_ID),
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.PORT))
    uvicorn.run("main:app", host=settings.HOST, port=port, log_level="info")
