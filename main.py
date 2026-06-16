"""Finance Assistant - FastAPI entry point.
WeChat Official Account + DeepSeek AI + ServerChan push.
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
from core import close_client
from handlers.message_handler import handle_user_message, push_daily_briefing
from scheduler.daily_task import start_scheduler, stop_scheduler
from wechat.official_account import verify_signature, parse_message, send_customer_message


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
    logger.info("Ready - DeepSeek:{} WeChat:{} ServerChan:{}".format(
        bool(settings.DEEPSEEK_API_KEY),
        bool(settings.WECHAT_APP_ID),
        bool(settings.SERVERCHAN_SENDKEY),
    ))
    yield
    stop_scheduler()
    await close_client()
    logger.info("Shutdown.")


app = FastAPI(title="Finance Assistant", version="2.0.0", lifespan=lifespan)

# Background task references
_bg_tasks: set = set()


# ─── WeChat Callback ─────────────────────────────────
@app.api_route("/wechat/callback", methods=["GET", "POST"])
async def wechat_callback(request: Request):
    if request.method == "GET":
        p = dict(request.query_params)
        if await verify_signature(p.get("signature", ""), p.get("timestamp", ""), p.get("nonce", "")):
            return PlainTextResponse(p.get("echostr", ""))
        return PlainTextResponse("invalid")

    # POST: Verify + process
    p = dict(request.query_params)
    if not await verify_signature(p.get("signature", ""), p.get("timestamp", ""), p.get("nonce", "")):
        return PlainTextResponse("invalid")

    body = await request.body()
    msg = parse_message(body)
    from_user = msg.get("FromUserName", "")
    content = msg.get("Content", "").strip()
    msg_type = msg.get("MsgType", "")

    if msg_type == "text" and content and from_user:
        import asyncio
        task = asyncio.create_task(_reply_wechat(from_user, content))
        task.add_done_callback(_bg_tasks.discard)
        _bg_tasks.add(task)

    return PlainTextResponse("")


async def _reply_wechat(openid: str, content: str):
    """Process WeChat message and send reply."""
    try:
        reply = await handle_user_message(openid, content)
        if reply:
            await send_customer_message(openid, reply)
    except Exception as e:
        logger.error(f"WeChat handler error: {e}")
        try:
            await send_customer_message(openid, "系统繁忙，请稍后再试。")
        except Exception:
            pass


# ─── Health ──────────────────────────────────────────
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
