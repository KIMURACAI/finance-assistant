"""Optimized DeepSeek AI service with caching, compression, token management."""

import asyncio
import json
import hashlib
import time
import re
import httpx
from typing import Optional

from loguru import logger

from config import settings
from core import get_client, retry, market_cache

API_URL = f"{settings.DEEPSEEK_BASE_URL}/chat/completions"


SYSTEM_PROMPT_CORE = """你是专业金融顾问，不是聊天机器人，不是百科全书。

你的价值：帮用户做出决策，而不只是提供信息。

━━━━━━━━━━━━━━━━━━━━━━
核心问题（每个回答前必须问自己）
━━━━━━━━━━━━━━━━━━━━━━

  "用户为什么问这个？他需要做什么决定？我该怎么帮他行动？"

  用户问"分析商业航天" → 他要决定：钱投哪里？
  用户问"今天行情"     → 他要决定：今天该不该操作？
  用户问"黄金价格"     → 他要决定：现在买还是等？
  用户问"特斯拉股价"   → 他要决定：入还是不入？

━━━━━━━━━━━━━━━━━━━━━━
回答结构（每条回答必须包含4个要素）
━━━━━━━━━━━━━━━━━━━━━━

  ① 判断结论   — 直接的判断，不是信息罗列
  ② 实用建议   — 基于当前情况的建议
  ③ 具体推荐   — 明确的方向或标的
  ④ 下一步行动 — 用户可以立刻做的事

━━━━━━━━━━━━━━━━━━━━━━
规则
━━━━━━━━━━━━━━━━━━━━━━

- 中文。不超过250字。4句话完成：判断、建议、推荐、行动。
- 外部数据引用真实数字，不可用时用知识判断，不编造。
- 禁止：信息罗列、教科书、模棱两可、长篇。
- 宁可说"建议等待"也不要说"可能会涨也可能会跌"。
- confidence_score: 有数据70-90，无数据50。

━━━━━━━━━━━━━━━━━━━━━━
JSON:
{"question":"用户问题","verified_data":["引用来源"],"analysis":"顾问式回答","confidence_score":50-90,"sources":["来源名称"]}"""


def _make_cache_key(user_message: str, positions_hash: str, history_tail: str) -> str:
    """Semantic cache key."""
    raw = f"{user_message}|{positions_hash}|{history_tail}"
    return hashlib.md5(raw.encode()).hexdigest()


# Semantic response cache (in-memory, 10min TTL)
_response_cache: dict[str, tuple[float, str]] = {}


def _get_cached(key: str) -> Optional[str]:
    if key in _response_cache:
        expires, val = _response_cache[key]
        if time.time() < expires:
            return val
        del _response_cache[key]
    return None


def _set_cache(key: str, val: str, ttl: int = 600):
    _response_cache[key] = (time.time() + ttl, val)
    # Evict if too large
    if len(_response_cache) > 500:
        oldest = min(_response_cache, key=lambda k: _response_cache[k][0])
        del _response_cache[oldest]


def _compress_history(history: list[dict], max_pairs: int = 5) -> list[dict]:
    """Keep only recent N exchanges, drop old context."""
    # Filter to only user/assistant roles
    pairs = [h for h in history if h.get("role") in ("user", "assistant")]
    # Take last max_pairs*2 entries
    return pairs[-(max_pairs * 2):]


def _build_system_prompt(positions: list[dict], preferences: dict) -> str:
    """Efficient system prompt with only non-default data."""
    prompt = SYSTEM_PROMPT_CORE
    parts = []
    if positions:
        pos_text = "; ".join(f"{p.get('asset_name','')}({p.get('asset_code','')})" for p in positions)
        parts.append(f"持仓: {pos_text}")
    if preferences.get("industry_focus"):
        parts.append(f"关注行业: {preferences['industry_focus']}")
    if preferences.get("focus_keywords"):
        parts.append(f"关注: {preferences['focus_keywords']}")
    if parts:
        prompt += "\n\n用户信息: " + " | ".join(parts)
    return prompt


def _classify_question(msg: str) -> str:
    """Router: classify user question into 4 categories.

    CODE decides — model has ZERO input on routing.

    Returns one of:
      "datetime"   — date/time → Python datetime, never search
      "realtime"   — news/price/events → force web search
      "financial"  — PE/earnings/reports → call financial APIs
      "knowledge"  — definitions/concepts → normal model response
    """
    msg_lower = msg.lower()

    # ── Category 1: Date / Time ──
    datetime_kw = [
        "what date", "what time", "what day", "today's date",
        "今天几号", "今天日期", "现在几点", "现在时间", "今天星期几",
        "几月几号", "今天是什么日子", "当前时间", "当前日期",
        "what is the date", "what is today", "current time",
    ]
    for kw in datetime_kw:
        if kw.lower() in msg_lower:
            return "datetime"

    # ── Category 3: Financial data (PE, earnings, reports) ──
    financial_kw = [
        "pe ratio", "p/e", "市盈率", "市净率", "pb ratio", "p/b",
        "earnings", "eps", "每股收益", "revenue", "营收",
        "market cap", "市值", "dividend yield", "股息率",
        "roe", "净资产收益率", "roa", "总资产收益率",
        "profit margin", "净利润", "毛利率", "net income",
        "balance sheet", "资产负债表", "cash flow", "现金流",
        "financial report", "财报", "年报", "季报", "quarterly",
        "debt to equity", "资产负债率", "current ratio",
        "book value", "每股净资产", "peg ratio", "peg",
    ]
    for kw in financial_kw:
        if kw.lower() in msg_lower:
            return "financial"

    # ── Category 2: Realtime information ──
    realtime_kw = [
        "today", "today's", "latest", "recent", "current", "update",
        "今天", "今日", "最新", "最近", "目前", "现在", "当前",
        "price", "stock", "market", "gold", "bitcoin", "crypto",
        "股价", "价格", "股票", "行情", "黄金", "比特币",
        "index", "nasdaq", "dow", "s&p", "hang seng", "nikkei",
        "指数", "上证", "深证", "创业板", "恒生", "纳斯达克",
        "news", "announced", "reported", "dividend",
        "新闻", "公告", "业绩", "分红",
        "policy", "fed", "interest rate", "inflation", "gdp",
        "政策", "利率", "央行", "通胀", "GDP",
        "company", "technology", "tech", "sector", "industry",
        "公司", "科技", "板块", "行业", "概念",
        "weather", "website",
        "分析", "预测", "建议", "怎么看", "怎么样", "如何",
    ]
    for kw in realtime_kw:
        if kw.lower() in msg_lower:
            return "realtime"

    # Stock code pattern = realtime
    if re.search(r'\b\d{6}\b', msg):
        return "realtime"

    # ── Category 4: Knowledge (default) ──
    return "knowledge"


def _get_datetime_context() -> str:
    """Return current date/time info for injection into prompt."""
    from datetime import datetime, timezone, timedelta

    # China timezone (UTC+8)
    tz_cn = timezone(timedelta(hours=8))
    now = datetime.now(tz_cn)

    weekday_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return (
        f"【当前日期时间 - 代码获取，绝对准确】\n"
        f"日期: {now.strftime('%Y年%m月%d日')} {weekday_cn[now.weekday()]}\n"
        f"时间: {now.strftime('%H:%M:%S')} (北京时间 UTC+8)\n"
        f"ISO: {now.strftime('%Y-%m-%dT%H:%M:%S+08:00')}"
    )


# ═══════════════════════════════════════════════════════════════
# TOOL EXECUTION ROUTER — code classifies, code decides, model receives
# ═══════════════════════════════════════════════════════════════


async def route_and_execute_tools(
    user_message: str, positions: list[dict],
) -> dict:
    """Router: classify question → execute correct tools → return data.

    Categories (CODE decision, NOT model):
      1. datetime  → Python datetime, never search
      2. realtime  → force Tavily + market data
      3. financial → financial APIs + market data
      4. knowledge → no tools, model answers directly

    Returns: {"category": str, "search_ctx": str, "market_ctx": str, "system_note": str}
    """
    category = _classify_question(user_message)
    logger.info(f"Router classified [{user_message[:50]}] → {category}")

    # ── Category 1: Date/Time ──
    if category == "datetime":
        dt_ctx = _get_datetime_context()
        return {
            "category": "datetime",
            "search_ctx": dt_ctx,
            "market_ctx": "",
            "system_note": "",
        }

    # ── Category 3: Financial data ──
    if category == "financial":
        market_task = _fetch_market_context(user_message, positions)
        search_task = _tavily_search(user_message, max_results=5)

        market_ctx, search_ctx = await asyncio.gather(
            market_task, search_task, return_exceptions=True,
        )
        if isinstance(market_ctx, Exception):
            market_ctx = ""
        if isinstance(search_ctx, Exception):
            search_ctx = ""

        return {
            "category": "financial",
            "search_ctx": search_ctx,
            "market_ctx": market_ctx,
            "system_note": "注意: 财务数据应引用搜索结果和行情数据中的具体数字，不要编造。",
        }

    # ── Category 2: Realtime ──
    if category == "realtime":
        market_task = _fetch_market_context(user_message, positions)
        search_task = _tavily_search(user_message, max_results=5)

        market_ctx, search_ctx = await asyncio.gather(
            market_task, search_task, return_exceptions=True,
        )
        if isinstance(market_ctx, Exception):
            market_ctx = ""
        if isinstance(search_ctx, Exception):
            search_ctx = ""

        if not search_ctx:
            logger.warning(
                f"Search unavailable for realtime: {user_message[:50]}. "
                f"Model will hedge with available data."
            )

        return {
            "category": "realtime",
            "search_ctx": search_ctx,
            "market_ctx": market_ctx,
            "system_note": "" if search_ctx else "本次未获取到实时搜索数据，可以使用已有行情数据或基础知识回答，但不要编造具体数字。",
        }

    # ── Category 4: Knowledge (no tools) ──
    return {
        "category": "knowledge",
        "search_ctx": "",
        "market_ctx": "",
        "system_note": "",
    }


# ── Tavily search cache ──
_tavily_cache: dict[str, tuple[float, str]] = {}


def _tavily_cache_key(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()


async def _tavily_search(query: str, max_results: int = 5) -> str:
    """Search the web via Tavily API with retry + caching. Returns formatted results or empty string."""
    if not settings.TAVILY_API_KEY:
        return ""

    cache_key = _tavily_cache_key(query)
    if cache_key in _tavily_cache:
        expires, val = _tavily_cache[cache_key]
        if time.time() < expires:
            logger.info(f"Tavily cache hit: {query[:40]}")
            return val
        del _tavily_cache[cache_key]

    client = get_client()
    last_error = ""

    for attempt in range(2):
        try:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": settings.TAVILY_API_KEY,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_answer": True,
                },
                timeout=httpx.Timeout(15.0, connect=10.0),
            )

            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.warning(f"Tavily attempt {attempt+1}: {last_error}")
                if attempt == 0:
                    await asyncio.sleep(1.0)
                continue

            data = resp.json()

            # Check for API-level errors
            if data.get("error"):
                last_error = f"API error: {data['error']}"
                logger.warning(f"Tavily attempt {attempt+1}: {last_error}")
                if attempt == 0:
                    await asyncio.sleep(1.0)
                continue

            parts = []

            answer = data.get("answer", "")
            if answer:
                parts.append(f"摘要: {answer}")

            results = data.get("results", [])
            if results:
                parts.append("搜索结果:")
                for i, r in enumerate(results[:max_results], 1):
                    title = r.get("title", "")[:100]
                    content = r.get("content", "")[:200]
                    url = r.get("url", "")
                    parts.append(f"  [{i}] {title}")
                    parts.append(f"      {content}")
                    if url:
                        parts.append(f"      URL: {url}")

            if not parts:
                return ""

            formatted = "\n".join(parts)

            # Cache for 120s to save quota on repeat queries
            _tavily_cache[cache_key] = (time.time() + 120, formatted)
            if len(_tavily_cache) > 200:
                oldest = min(_tavily_cache, key=lambda k: _tavily_cache[k][0])
                del _tavily_cache[oldest]

            return formatted

        except Exception as e:
            last_error = str(e)
            logger.warning(f"Tavily attempt {attempt+1} exception: {e}")
            if attempt == 0:
                await asyncio.sleep(1.0)

    logger.error(f"Tavily search failed after 2 attempts: {last_error}")
    return ""


def _verify_and_parse(
    raw_response: str, market_context: str, user_message: str
) -> dict:
    """
    Parse AI JSON response and verify factual claims against market data.
    Returns {"status": "ok"|"no_data"|"low_confidence"|"parse_error", "display_text": str, "raw": dict}
    """
    # Try to extract JSON from response
    json_str = raw_response.strip()

    # Strip markdown code fences if present
    if json_str.startswith("```"):
        json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)

    parsed = None
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        # Try to find outermost JSON object with brace matching
        if json_str.startswith("{") or "{" in json_str:
            start = json_str.index("{")
            depth = 0
            end = -1
            for i, ch in enumerate(json_str[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                try:
                    parsed = json.loads(json_str[start:end])
                except json.JSONDecodeError:
                    pass
        # Fallback: simple regex
        if not parsed:
            m = re.search(r'\{[^{}]*"question"\s*:\s*"[^"]*"[^{}]*\}', raw_response, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

    if not parsed or not isinstance(parsed, dict):
        # JSON parse failed — model may have output plain text
        # Return it as-is rather than truncating
        logger.warning(f"AI JSON parse failed, returning raw text ({len(raw_response)} chars)")
        return {
            "status": "parse_error",
            "display_text": raw_response,
            "raw": {},
        }

    confidence = parsed.get("confidence_score", 0)
    analysis = parsed.get("analysis", "")
    question = parsed.get("question", "")

    # Default to 50 if field missing or 0 — matches system prompt minimum
    # Only explicitly-set low confidence triggers rejection
    if not confidence or confidence == 0:
        confidence = 50

    # Reject ONLY if AI explicitly self-assigned < 30
    if isinstance(confidence, (int, float)) and confidence < 30:
        return {
            "status": "low_confidence",
            "display_text": "LOW CONFIDENCE",
            "raw": parsed,
        }

    # Use analysis field, fallback to raw text
    display = analysis if analysis else raw_response

    return {
        "status": "ok",
        "display_text": display,
        "raw": parsed,
    }


def _extract_stock_codes(msg: str) -> list[str]:
    """Extract A-share stock codes from user message. Returns unique codes."""
    # Match patterns: 600519, sh600519, sz000001, 000001
    codes = []
    # Full codes with optional exchange prefix
    for m in re.finditer(r'\b(sh|sz)?(\d{6})\b', msg, re.IGNORECASE):
        codes.append(m.group(2))
    # Dedup preserving order
    seen = set()
    unique = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique[:8]  # Max 8 codes to avoid excessive API calls


async def _fetch_market_context(
    user_message: str, positions: list[dict],
) -> str:
    """Fetch real-time market data and format for AI context."""
    from services.market_service import (
        get_market_overview, get_sector_performance,
        get_batch_quotes, fetch_hot_rank,
    )

    is_market = _classify_question(user_message) != "knowledge"
    has_positions = bool(positions)

    # Extract stock codes from the message itself
    msg_codes = _extract_stock_codes(user_message) if is_market else []

    if not is_market and not has_positions:
        return ""

    # Collect all codes to fetch: positions + extracted from message
    pos_codes = [p.get("asset_code", "") for p in positions if p.get("asset_code")]
    all_codes = list(dict.fromkeys(pos_codes + msg_codes))  # dedup, preserve order

    # Fetch concurrently
    tasks = []
    task_labels = []
    if is_market:
        tasks.append(get_market_overview())
        task_labels.append("overview")
        tasks.append(get_sector_performance())
        task_labels.append("sectors")
        tasks.append(fetch_hot_rank(8))
        task_labels.append("hot")
    if all_codes:
        tasks.append(get_batch_quotes(all_codes))
        task_labels.append("quotes")

    if not tasks:
        return ""

    results = await asyncio.gather(*tasks, return_exceptions=True)
    data = dict(zip(task_labels, results))

    overview = data.get("overview") if not isinstance(data.get("overview"), Exception) else None
    sectors = data.get("sectors") if not isinstance(data.get("sectors"), Exception) else None
    hot = data.get("hot") if not isinstance(data.get("hot"), Exception) else None
    quotes = data.get("quotes") if not isinstance(data.get("quotes"), Exception) else None

    if not any([overview, sectors, hot, quotes]):
        return ""

    parts = ["【实时行情数据 - 必须使用以下真实数字】"]

    if overview:
        icon = "🔴" if overview.get("change_pct", 0) < 0 else "🟢"
        parts.append(
            f"{icon} {overview['index_name']}: {overview['price']:.2f}  "
            f"涨跌: {overview['change_pct']:+.2f}%"
        )

    if quotes:
        parts.append("---实时个股行情---")
        for q in quotes:
            chg = q.get("change_pct", 0)
            sign = "+" if chg >= 0 else ""
            parts.append(
                f"  {q.get('name', q.get('code', ''))}({q.get('code', '')}): "
                f"最新价 {q.get('price', 0):.2f}  涨跌 {sign}{chg:.2f}%"
            )

    if sectors:
        parts.append("---热门板块---")
        for s in sectors[:6]:
            chg = s.get("change_pct", 0)
            sign = "+" if chg >= 0 else ""
            parts.append(f"  {s.get('name', '')}: {sign}{chg:.2f}%")

    if hot:
        parts.append("---今日热门个股---")
        for s in hot[:5]:
            chg = s.get("change_pct", 0)
            sign = "+" if chg >= 0 else ""
            parts.append(
                f"  {s.get('name', '')}({s.get('code', '')}): "
                f"{s.get('price', 0)} ({sign}{chg:.2f}%)"
            )

    return "\n".join(parts)


async def chat(
    user_message: str,
    positions: list[dict],
    preferences: dict,
    chat_history: list[dict],
) -> str:
    """AI chat with tool execution enforced by code, not model.

    WORKFLOW (model has ZERO control over tool decisions):
      1. route_and_execute_tools() — CODE decides → runs tools → returns data
      2. Build prompt with tool results
      3. Call model for text generation ONLY
    """
    pos_hash = hashlib.md5(str(positions).encode()).hexdigest()[:8]

    # ── STEP 1: Route & execute tools (CODE decision, model not involved) ──
    tools = await route_and_execute_tools(user_message, positions)
    category = tools["category"]
    search_ctx = tools["search_ctx"]
    market_ctx = tools["market_ctx"]
    system_note = tools["system_note"]

    # ── Cache check (knowledge category only) ──
    if category == "knowledge" and len(user_message) < 50:
        cache_key = _make_cache_key(user_message, pos_hash, "")
        cached = _get_cached(cache_key)
        if cached:
            return cached

    # ── STEP 2: Build prompt with tool results ──
    system_prompt = _build_system_prompt(positions, preferences)
    compressed_history = _compress_history(chat_history)

    if search_ctx:
        if category == "datetime":
            prefix = "【当前日期时间 - 代码获取，绝对准确】"
        elif category == "realtime":
            prefix = "【网络搜索结果 - 优先使用这些实时数据】"
        elif category == "financial":
            prefix = "【财务数据 + 网络搜索结果】"
        else:
            prefix = "【外部数据】"
        system_prompt += "\n\n" + prefix + "\n" + search_ctx

    if market_ctx:
        system_prompt += "\n\n" + market_ctx

    if system_note:
        system_prompt += "\n\n" + system_note

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(compressed_history)
    messages.append({"role": "user", "content": user_message})

    # ── STEP 3: Call model for text generation ONLY (no tool decisions) ──

    # ── Call AI with stable factual parameters ──
    payload = {
        "model": settings.DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "top_p": 0.5,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": settings.DEEPSEEK_MAX_TOKENS,
    }

    client = get_client()
    raw_content = None
    for attempt in range(2):
        try:
            logger.info(
                f"DeepSeek request attempt={attempt+1} "
                f"category={category} has_search={bool(search_ctx)} "
                f"has_market={bool(market_ctx)} msg_len={len(user_message)}"
            )
            resp = await client.post(
                API_URL,
                headers={"Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}"},
                json=payload,
                timeout=httpx.Timeout(25.0, connect=20.0),
            )
            resp.raise_for_status()
            data = resp.json()
            raw_content = data["choices"][0]["message"]["content"]
            break
        except Exception as e:
            logger.warning(f"AI attempt {attempt+1} failed: {e}")
            if attempt == 0:
                await asyncio.sleep(1.5)

    if raw_content is None:
        return _get_fallback_reply(user_message)

    # ── Verification layer ──
    full_context = (search_ctx or "") + "\n" + (market_ctx or "")
    verified = _verify_and_parse(raw_content, full_context, user_message)

    if verified["status"] == "low_confidence":
        logger.warning(
            f"LOW CONFIDENCE: confidence={verified['raw'].get('confidence_score')}"
        )
        return "LOW CONFIDENCE"

    if verified["status"] == "parse_error":
        # Model returned plain text instead of JSON — use it directly
        raw = verified["display_text"]
        # Strip "LOW CONFIDENCE" if model put it inline
        if raw.strip().upper().startswith("LOW CONFIDENCE"):
            raw = raw.strip()[len("LOW CONFIDENCE"):].strip()
        logger.warning(f"JSON parse failed, returning raw text ({len(raw)} chars)")
        return raw if raw else "抱歉，处理出错，请重试。"

    logger.info(
        f"AI OK: confidence={verified['raw'].get('confidence_score')} "
        f"verified_items={len(verified['raw'].get('verified_data', []))}"
    )

    display = verified["display_text"]

    # ── Cache only non-realtime short responses ──
    if category == "knowledge" and len(user_message) < 50 and len(display) < 300:
        _set_cache(_make_cache_key(user_message, pos_hash, ""), display)

    return display


@retry(max_retries=2, base_delay=1.0)
async def screen_news(
    news_items: list[dict],
    positions: list[dict],
    preferences: dict,
) -> str:
    """AI news screening with dedup."""
    if not news_items or not positions:
        return "暂无相关资讯。"

    # Dedup news by title
    seen = set()
    unique_news = []
    for n in news_items:
        t = n.get("title", "")[:40]
        if t not in seen:
            seen.add(t)
            unique_news.append(n)

    if not unique_news:
        return "暂无相关资讯。"

    pos_names = ", ".join(p.get("asset_name", p.get("asset_code", "")) for p in positions[:5])
    industry = preferences.get("industry_focus", "")

    news_text = "\n".join(
        f"{i}. [{n.get('publish_time','')[:10]}] {n['title'][:80]}"
        for i, n in enumerate(unique_news[:15], 1)
    )

    prompt = f"""持仓: {pos_names}
关注: {industry or '无'}

从以上新闻中选出最相关的3条，给出1句简评(≤30字):
{news_text}"""

    messages = [
        {"role": "system", "content": "你是一个金融资讯筛选助手。只输出筛选结果，不解释。"},
        {"role": "user", "content": prompt},
    ]

    client = get_client()
    try:
        resp = await client.post(
            API_URL,
            headers={"Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}"},
            json={
                "model": settings.DEEPSEEK_MODEL,
                "messages": messages,
                "temperature": 0.2,
                "top_p": 0.5,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "max_tokens": 500,
            },
            timeout=httpx.Timeout(30.0, connect=20.0),
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"News screening error: {e}")
        return ""


def extract_commands(text: str) -> list[dict]:
    """Extract JSON commands from AI response."""
    import re
    commands = []
    for match in re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL):
        try:
            cmd = json.loads(match.strip())
            if isinstance(cmd, dict) and "cmd" in cmd:
                commands.append(cmd)
        except json.JSONDecodeError:
            continue
    return commands


def clean_commands(text: str) -> str:
    """Remove JSON blocks from text."""
    return re.sub(r'```json\s*.*?\s*```', '', text, flags=re.DOTALL).strip()


def _get_fallback_reply(user_message: str) -> str:
    """Non-AI fallback when DeepSeek is unreachable."""
    msg = user_message
    if any(kw in msg for kw in ["添加", "买入", "入仓"]):
        return "请按格式发送：添加持仓 600519 贵州茅台"
    if any(kw in msg for kw in ["删除", "移除", "去掉"]):
        return "请按格式发送：删除 600519"
    if any(kw in msg for kw in ["持仓", "我的", "列表"]):
        return "查看持仓需要联网，请稍后再试。"
    if any(kw in msg for kw in ["你好", "hi", "hello", "在吗", "在不在"]):
        return "你好！我是你的金融助手。你可以说「添加持仓 600519」来添加股票，或者说「我的持仓」查看列表。"
    if any(kw in msg for kw in ["早", "简报", "行情"]):
        return "正在加载行情数据，请稍后再试。"
    return f"你好！目前AI暂时繁忙，你可以试试：\n- 添加持仓 600519 贵州茅台\n- 我的持仓有哪些"
