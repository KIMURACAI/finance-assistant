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
    logger.info(
        f"Ready - DeepSeek:{bool(settings.DEEPSEEK_API_KEY)} WeChat:{bool(settings.WECHAT_APP_ID)}"
    )
    yield
    stop_scheduler()
    await close_client()


app = FastAPI(title="Finance Assistant", version="2.0.0", lifespan=lifespan)

# Background task store
_bg_tasks: set = set()

# Request metrics
_req_metrics = {
    "get_count": 0,
    "post_count": 0,
    "post_invalid_sig": 0,
    "post_parsed": 0,
    "post_local_hit": 0,
    "post_ai_triggered": 0,
    "post_other": 0,
    "last_post_at": None,
    "last_post_body": "",
    "last_post_from": "",
    "last_post_content": "",
}


@app.api_route("/wechat/callback", methods=["GET", "POST"])
async def wechat_callback(request: Request):

    # ==============================
    # 微信服务器验证（临时关闭签名验证测试）
    # ==============================
    if request.method == "GET":
        _req_metrics["get_count"] += 1
        p = dict(request.query_params)

        # 直接返回 echostr，不校验 signature
        # 用来测试微信是否能真正访问 Railway
        return PlainTextResponse(
            p.get("echostr", "hello_test")
        )

    # ==============================
    # 处理微信用户消息
    # ==============================
    _req_metrics["post_count"] += 1
    p = dict(request.query_params)

    sig_ok = await verify_signature(
        p.get("signature", ""),
        p.get("timestamp", ""),
        p.get("nonce", "")
    )
    if not sig_ok:
        _req_metrics["post_invalid_sig"] += 1
        logger.warning(f"签名验证失败 signature={p.get('signature','')[:8]}...")
        return PlainTextResponse("invalid")

    body = await request.body()
    _req_metrics["last_post_body"] = body[:200].decode("utf-8", errors="replace")
    logger.info(f"收到微信消息 raw={_req_metrics['last_post_body']}")

    msg = parse_message(body)
    _req_metrics["post_parsed"] += 1
    logger.info(f"解析消息 type={msg.get('MsgType')} from={msg.get('FromUserName','')[:10]} content={msg.get('Content','')[:50]}")

    from_user = msg.get("FromUserName", "")
    to_user = msg.get("ToUserName", "")
    content = msg.get("Content", "").strip()

    if msg.get("MsgType") == "text" and content and from_user:

        # Phase 1: 本地命令优先处理
        _req_metrics["last_post_from"] = from_user
        _req_metrics["last_post_content"] = content
        local_reply = await try_local_command(from_user, content)

        if local_reply:
            _req_metrics["post_local_hit"] += 1
            logger.info(f"本地命令命中: {local_reply[:50]}")
            return PlainTextResponse(
                build_text_reply(from_user, to_user, local_reply),
                media_type="application/xml",
            )

        # Phase 2: AI后台异步处理
        _req_metrics["post_ai_triggered"] += 1
        logger.info(f"启动AI后台处理: {content[:50]}")
        thinking = "正在查询，请稍候..."

        task = asyncio.create_task(
            _ai_reply_async(from_user, content)
        )

        task.add_done_callback(_bg_tasks.discard)
        _bg_tasks.add(task)

        return PlainTextResponse(
            build_text_reply(from_user, to_user, thinking),
            media_type="application/xml",
        )

    logger.info(f"消息未处理 type={msg.get('MsgType')} has_content={bool(content)} has_user={bool(from_user)}")
    _req_metrics["post_other"] += 1
    return PlainTextResponse("")


async def _ai_reply_async(openid: str, content: str):
    """Background AI processing + push via customer service."""
    logger.info(f"AI开始处理 openid={openid[:10]} msg={content[:50]}")
    try:
        reply = await asyncio.wait_for(
            handle_user_message(openid, content),
            timeout=25.0,
        )

        if reply:
            logger.info(f"AI回复生成成功 len={len(reply)} 准备推送...")
            ok = await send_customer_message(openid, reply)
            logger.info(f"客服消息推送结果: {ok} openid={openid[:10]}")
        else:
            logger.warning(f"AI回复为空 openid={openid[:10]}")

    except asyncio.TimeoutError:
        logger.warning(f"AI timeout for {openid[:10]}")
        await send_customer_message(
            openid,
            "查询超时，请简化问题后重试。"
        )

    except Exception as e:
        logger.error(f"AI reply error: {e}")

        try:
            await send_customer_message(
                openid,
                "查询出错，请稍后再试。"
            )
        except Exception:
            pass


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "deepseek": bool(settings.DEEPSEEK_API_KEY),
        "wechat": bool(settings.WECHAT_APP_ID),
    }


@app.get("/debug/metrics")
async def debug_metrics():
    """View request metrics for debugging."""
    return _req_metrics


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
                headers={
                    "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "user",
                            "content": "hi"
                        }
                    ],
                    "max_tokens": 5
                },
                timeout=10,
            )

            return {
                "url": url,
                "status": resp.status_code,
                "time": round(time.time() - t0, 2)
            }

        except Exception:
            continue

    return {
        "status": "error",
        "message": "All DeepSeek endpoints timed out from US"
    }


@app.get("/test/wechat-token")
async def test_wechat_token():
    """Test WeChat access token retrieval."""
    from wechat.official_account import _get_access_token
    t0 = __import__("time").time()
    token = await _get_access_token()
    t = round(__import__("time").time() - t0, 2)
    return {
        "ok": bool(token),
        "token_prefix": token[:10] + "..." if token else "",
        "token_len": len(token),
        "time": t,
    }


@app.get("/test/ai-flow")
async def test_ai_flow(msg: str = "你好"):
    """End-to-end test: local command + AI + WeChat push."""
    import time
    from database.db import get_or_create_user

    t0 = time.time()
    steps = []

    # Step 1: Get or create user
    test_uid = "test_user_flow"
    try:
        user = await get_or_create_user(test_uid)
        steps.append({"step": "user", "ok": True, "user_id": user.id})
    except Exception as e:
        steps.append({"step": "user", "ok": False, "error": str(e)})
        return {"status": "error", "steps": steps, "total_time": round(time.time() - t0, 2)}

    # Step 2: Try local command
    try:
        local = await try_local_command(test_uid, msg)
        steps.append({"step": "local_command", "ok": True, "handled": bool(local), "reply": local})
        if local:
            return {"status": "ok", "note": "本地命令命中，无需AI", "steps": steps, "total_time": round(time.time() - t0, 2)}
    except Exception as e:
        steps.append({"step": "local_command", "ok": False, "error": str(e)})

    # Step 3: Full AI handling
    try:
        ai_reply = await handle_user_message(test_uid, msg)
        steps.append({"step": "ai", "ok": True, "reply_len": len(ai_reply), "reply": ai_reply[:200]})
    except Exception as e:
        steps.append({"step": "ai", "ok": False, "error": str(e)})
        return {"status": "error", "steps": steps, "total_time": round(time.time() - t0, 2)}

    # Step 4: WeChat push
    try:
        ok = await send_customer_message(test_uid, ai_reply[:100])
        steps.append({"step": "wechat_push", "ok": ok})
    except Exception as e:
        steps.append({"step": "wechat_push", "ok": False, "error": str(e)})

    return {"status": "ok", "steps": steps, "total_time": round(time.time() - t0, 2)}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.PORT))

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=port,
        log_level="info"
    )