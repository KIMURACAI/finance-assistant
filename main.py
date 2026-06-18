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

    try:
        msg = parse_message(body)
    except Exception as parse_err:
        logger.error(f"XML解析失败: {parse_err} body={body[:300]}")
        _req_metrics["last_post_body"] = f"PARSE_ERROR: {parse_err}"
        return PlainTextResponse("")

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
    import subprocess
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(settings.PROJECT_ROOT), timeout=2
        ).decode().strip()
    except Exception:
        pass
    return {
        "status": "ok",
        "version": git_hash,
        "deepseek": bool(settings.DEEPSEEK_API_KEY),
        "tavily": bool(settings.TAVILY_API_KEY),
        "wechat": bool(settings.WECHAT_APP_ID),
    }


@app.get("/debug/connectivity")
async def debug_connectivity():
    """Test ALL external API connectivity from this server's location.
    Use this to diagnose Railway vs local discrepancies.
    """
    import time as _time
    from core import get_client as _gc

    client = _gc()
    results = {}
    t0 = _time.time()

    # 1. Sina (market overview, stock quotes)
    try:
        t1 = _time.time()
        r = await client.get("https://hq.sinajs.cn/list=sh000001",
            headers={"Referer": "https://finance.sina.com.cn"}, timeout=8.0)
        results["sina"] = {"ok": r.status_code == 200, "status": r.status_code,
                           "time": round(_time.time() - t1, 2)}
    except Exception as e:
        results["sina"] = {"ok": False, "error": str(e)[:100]}

    # 2. EastMoney (sectors, hot stocks)
    try:
        t1 = _time.time()
        r = await client.get(
            "https://push2.eastmoney.com/api/qt/clist/get",
            params={"cb": "", "pn": 1, "pz": 1, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                    "fid": "f3", "fs": "m:0+t:6", "fields": "f12,f14",
                    "ut": "bd1d9ddb04089700cf9c27f6f7426281"},
            headers={"Referer": "https://quote.eastmoney.com/"}, timeout=8.0)
        results["eastmoney"] = {"ok": r.status_code == 200, "status": r.status_code,
                                "time": round(_time.time() - t1, 2)}
    except Exception as e:
        results["eastmoney"] = {"ok": False, "error": str(e)[:100]}

    # 3. DeepSeek
    try:
        t1 = _time.time()
        r = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}"},
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 5}, timeout=10.0)
        results["deepseek"] = {"ok": r.status_code == 200, "status": r.status_code,
                               "time": round(_time.time() - t1, 2)}
    except Exception as e:
        results["deepseek"] = {"ok": False, "error": str(e)[:100]}

    # 4. Tavily
    if settings.TAVILY_API_KEY:
        try:
            t1 = _time.time()
            r = await client.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {settings.TAVILY_API_KEY}",
                         "Content-Type": "application/json"},
                json={"query": "test", "max_results": 1}, timeout=10.0)
            results["tavily"] = {"ok": r.status_code == 200, "status": r.status_code,
                                 "time": round(_time.time() - t1, 2)}
        except Exception as e:
            results["tavily"] = {"ok": False, "error": str(e)[:100]}
    else:
        results["tavily"] = {"ok": False, "error": "TAVILY_API_KEY not configured"}

    # 5. 同花顺
    try:
        t1 = _time.time()
        r = await client.get(
            "https://d.10jqka.com.cn/v2/realhead/hs_600519/last.js",
            headers={"Referer": "https://stockpage.10jqka.com.cn/",
                     "User-Agent": "Mozilla/5.0"}, timeout=8.0)
        results["10jqka"] = {"ok": r.status_code == 200, "status": r.status_code,
                             "time": round(_time.time() - t1, 2)}
    except Exception as e:
        results["10jqka"] = {"ok": False, "error": str(e)[:100]}

    # 6. WeChat API
    try:
        t1 = _time.time()
        r = await client.get("https://api.weixin.qq.com/cgi-bin/token",
            params={"grant_type": "client_credential",
                    "appid": settings.WECHAT_APP_ID,
                    "secret": settings.WECHAT_APP_SECRET}, timeout=8.0)
        results["wechat_api"] = {"ok": r.status_code == 200, "status": r.status_code,
                                 "time": round(_time.time() - t1, 2)}
    except Exception as e:
        results["wechat_api"] = {"ok": False, "error": str(e)[:100]}

    total_ok = sum(1 for v in results.values() if v.get("ok"))
    results["SUMMARY"] = {"total_ok": total_ok, "total_tested": len(results),
                          "total_time": round(_time.time() - t0, 2)}
    return results


@app.get("/debug/echo")
async def debug_echo(msg: str = "上证指数现在多少点"):
    """Mirror of WeChat user pipeline. Returns EXACTLY what a user would see."""
    import time as _time
    from database.db import get_or_create_user
    from handlers.message_handler import handle_user_message

    t0 = _time.time()
    test_uid = "debug_test_user"

    try:
        user = await get_or_create_user(test_uid)
    except Exception as e:
        return {"error": f"db: {e}"}

    try:
        reply = await handle_user_message(test_uid, msg)
        return {
            "user_message": msg,
            "reply": reply,
            "reply_len": len(reply),
            "time": round(_time.time() - t0, 2),
        }
    except Exception as e:
        return {
            "user_message": msg,
            "error": str(e)[:300],
            "time": round(_time.time() - t0, 2),
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


@app.get("/test/trigger-closing")
async def test_trigger_closing():
    """Manually trigger closing summary push to all active users."""
    from database.db import get_recent_active_users
    from handlers.message_handler import push_daily_briefing

    users = await get_recent_active_users(hours=48)
    if not users:
        return {"status": "ok", "note": "no active users in 48h window", "count": 0}

    results = []
    for user in users:
        try:
            await push_daily_briefing(user.wecom_user_id, "closing")
            results.append({"uid": user.wecom_user_id[:12] + "...", "ok": True})
        except Exception as e:
            results.append({"uid": user.wecom_user_id[:12] + "...", "ok": False, "error": str(e)})

    ok_count = sum(1 for r in results if r["ok"])
    return {
        "status": "ok",
        "total": len(users),
        "ok": ok_count,
        "failed": len(users) - ok_count,
        "results": results,
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


@app.get("/test/tavily")
async def test_tavily(query: str = "今日A股行情"):
    """Debug Tavily search end-to-end: API key check → raw call → formatted result."""
    import time as time_mod
    from services.ai_service import _tavily_search

    t0 = time_mod.time()
    result = {
        "api_key_configured": bool(settings.TAVILY_API_KEY),
        "api_key_prefix": settings.TAVILY_API_KEY[:8] + "..." if settings.TAVILY_API_KEY else "(empty)",
        "query": query,
    }

    if not settings.TAVILY_API_KEY:
        result["status"] = "no_api_key"
        result["note"] = "TAVILY_API_KEY not set in .env — web search is disabled"
        return result

    # Make a raw direct call to verify API connectivity
    from core import get_client
    import httpx as httpx_mod

    client = get_client()
    raw_result = {}

    # Test 1: Direct raw call with verbose logging
    try:
        raw_resp = await client.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {settings.TAVILY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "max_results": 3,
                "search_depth": "basic",
                "include_answer": True,
            },
            timeout=httpx_mod.Timeout(15.0, connect=10.0),
        )
        raw_result["http_status"] = raw_resp.status_code
        raw_result["response_keys"] = list(raw_resp.json().keys()) if raw_resp.status_code == 200 else None
        raw_result["raw_body"] = raw_resp.text[:500]
        raw_result["raw_json"] = raw_resp.json()
    except Exception as e:
        raw_result["error"] = str(e)

    result["raw_api_response"] = raw_result

    # Test 2: Through the normal _tavily_search function
    try:
        formatted = await _tavily_search(query, max_results=5)
        result["search_result_length"] = len(formatted) if formatted else 0
        result["search_result"] = formatted[:1000] if formatted else "(EMPTY)"
    except Exception as e:
        result["search_error"] = str(e)

    result["total_time"] = round(time_mod.time() - t0, 2)
    return result


@app.get("/test/pipeline-trace")
async def test_pipeline_trace(msg: str = "今天科技板块怎么样"):
    """Full pipeline trace: classify → search → model. Prints every step."""
    import time as time_mod
    from services.ai_service import classify_intent, route_and_execute_tools, _tavily_search, _fetch_market_context

    t0 = time_mod.time()
    trace = {"user_message": msg, "steps": []}

    # Step 1: Classify intent
    try:
        intent = await classify_intent(msg)
        trace["steps"].append({"step": "classify_intent", "ok": True, "result": intent})
    except Exception as e:
        trace["steps"].append({"step": "classify_intent", "ok": False, "error": str(e)})
        trace["total_time"] = round(time_mod.time() - t0, 2)
        return trace

    # Step 2: Route and execute tools
    try:
        tools = await route_and_execute_tools(msg, [])
        trace["steps"].append({
            "step": "route_and_execute_tools",
            "ok": True,
            "category": tools["category"],
            "search_ctx_len": len(tools["search_ctx"]),
            "market_ctx_len": len(tools["market_ctx"]),
            "system_note": tools["system_note"],
        })
    except Exception as e:
        trace["steps"].append({"step": "route_and_execute_tools", "ok": False, "error": str(e)})

    # Step 3: Raw Tavily call directly
    if settings.TAVILY_API_KEY:
        try:
            raw_search = await _tavily_search(msg, max_results=3)
            trace["steps"].append({
                "step": "raw_tavily",
                "ok": True,
                "result_len": len(raw_search),
                "result": raw_search[:300] if raw_search else "(EMPTY)",
            })
        except Exception as e:
            trace["steps"].append({"step": "raw_tavily", "ok": False, "error": str(e)})
    else:
        trace["steps"].append({"step": "raw_tavily", "ok": False, "error": "TAVILY_API_KEY not set"})

    # Step 4: Full chat call
    try:
        from services.ai_service import chat
        reply = await chat(msg, [], {}, [])
        trace["steps"].append({
            "step": "chat",
            "ok": True,
            "reply_len": len(reply),
            "reply": reply[:500],
        })
    except Exception as e:
        trace["steps"].append({"step": "chat", "ok": False, "error": str(e)})

    trace["total_time"] = round(time_mod.time() - t0, 2)
    return trace


if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.PORT))

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=port,
        log_level="info"
    )