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
from handlers.message_handler import (
    handle_user_message, try_local_command, push_daily_briefing,
)
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

# Background task store
_bg_tasks: set = set()


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
        # Phase 1: Try local processing (fast, no AI needed)
        local_reply = await try_local_command(from_user, content)
        if local_reply:
            return PlainTextResponse(
                build_text_reply(from_user, to_user, local_reply),
                media_type="application/xml",
            )

        # Phase 2: Needs AI → reply "thinking" immediately + background DeepSeek
        thinking = "正在查询，请稍候..."
        task = asyncio.create_task(_ai_reply_async(from_user, content))
        task.add_done_callback(_bg_tasks.discard)
        _bg_tasks.add(task)
        return PlainTextResponse(
            build_text_reply(from_user, to_user, thinking),
            media_type="application/xml",
        )

    return PlainTextResponse("")


async def _ai_reply_async(openid: str, content: str):
    """Background AI processing + push via customer service."""
    try:
        # Longer timeout for AI (up to 25s)
        reply = await asyncio.wait_for(
            handle_user_message(openid, content),
            timeout=25.0,
        )
        if reply:
            await send_customer_message(openid, reply)
            logger.info(f"AI reply sent to {openid[:10]}")
    except asyncio.TimeoutError:
        logger.warning(f"AI timeout for {openid[:10]}")
        await send_customer_message(openid, "查询超时，请简化问题后重试。")
    except Exception as e:
        logger.error(f"AI reply error: {e}")
        try:
            await send_customer_message(openid, "查询出错，请稍后再试。")
        except Exception:
            pass


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "deepseek": bool(settings.DEEPSEEK_API_KEY),
        "wechat": bool(settings.WECHAT_APP_ID),
    }


@app.get("/test/deepseek")
async def test_deepseek():
    """Test DeepSeek from Railway."""
    import time
    from core import get_client
    for url in [
        "https://api.deepseek.com/v1/chat/completions",
        "https://api.deepseek.com/beta/chat/completions",
    ]:
        try:
            t0 = time.time()
            client = get_client()
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}"},
                json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
                timeout=10,
            )
            return {"url": url, "status": resp.status_code, "time": round(time.time() - t0, 2)}
        except Exception as e:
            continue
    return {"status": "error", "message": "All DeepSeek endpoints timed out from US"}



if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.PORT))
    uvicorn.run("main:app", host=settings.HOST, port=port, log_level="info")
