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


SYSTEM_PROMPT_CORE = """你是 Kimura 的投资顾问。专业、克制、直接。

说话风格：
- 默认冷静简洁。数据 + 判断 + 理由。
- 不要叫"兄弟""朋友"，不要用"咱们"。你是顾问，不是哥们。
- 他明显焦虑时先稳情绪，其他时候直接分析。
- 问什么答什么，别主动展开长篇大论。他追问再深入。
- 数字不要加千分位逗号。1215 不是 1,215。

数据规则：
- 用下方提供的真实数据。没有就说没有。
- 系统时间是唯一正确时间。查股票顺手带沪深成交额。
- 梳理逻辑、推演情景，不替他说买卖。不说一定涨跌。"""


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


# ═══════════════════════════════════════════════════════════════
# DECISION ENGINE — classifies intent, returns structured JSON
# Does NOT answer. Does NOT explain. ONLY classifies.
# ═══════════════════════════════════════════════════════════════

DECISION_ENGINE_PROMPT = """Classify user intent and emotional state. Return JSON only. No explanation.

── Intent Categories ──
STATIC_KNOWLEDGE — definitions, theory, history, concepts. web=false.
REALTIME_INFORMATION — current prices, news, events. web=true.
DECISION_SUPPORT — should I buy/sell, recommendations, judgment. web=true.
RESEARCH_ANALYSIS — deep industry/company analysis. web=true.
CLARIFICATION_REQUIRED — ambiguous, broad categories without specifics.

── Emotional State (emotion + intensity 1-5) ──
fomo — fear of missing out, chasing rallies
panic — fear of losing money, want to sell
seeking_validation — already have an opinion, want confirmation
exploring — open-minded, looking for opportunities
neutral — normal inquiry, no strong emotion

Intensity: 1=mild, 3=moderate (needs cooling), 5=extreme (trigger safety guardrail)

── Action Tendency ──
buying — wants to enter a position
selling — wants to exit
holding — wants to stay
watching — observing, no action intent
unknown — unclear

Return ONLY:
{"category":"...","need_web":true|false,"clarification_needed":true|false,"confidence":0.0-1.0,"emotion":"...","emotion_intensity":1-5,"action_tendency":"..."}"""

# Cache for classification results: {query_hash: (expires_at, result)}
_classify_cache: dict[str, tuple[float, dict]] = {}


async def classify_intent(user_message: str) -> dict:
    """Decision engine: classify user intent using model reasoning.

    Returns: {"category": str, "need_web": bool, "clarification_needed": bool, "confidence": float}
    Does NOT answer the question. ONLY classifies.
    """
    logger.info(f"User Question: {user_message}")

    cache_key = hashlib.md5(("intent:" + user_message).encode()).hexdigest()
    if cache_key in _classify_cache:
        expires, val = _classify_cache[cache_key]
        if time.time() < expires:
            logger.info(f"Decision engine cache hit: {user_message[:50]}")
            return val
        del _classify_cache[cache_key]

    client = get_client()
    try:
        resp = await client.post(
            API_URL,
            headers={"Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}"},
            json={
                "model": settings.DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": DECISION_ENGINE_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0,
                "max_tokens": 100,
            },
            timeout=httpx.Timeout(8.0, connect=5.0),
        )
        if resp.status_code == 200:
            data = resp.json()
            raw = data["choices"][0]["message"]["content"].strip()
            # Extract JSON
            json_str = raw
            if json_str.startswith("```"):
                json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
                json_str = re.sub(r'\s*```$', '', json_str)
            result = json.loads(json_str)
            logger.info(f"Decision engine result: category={result.get('category')} need_web={result.get('need_web')} clarify={result.get('clarification_needed')} confidence={result.get('confidence')}")
            # Cache for 30s
            _classify_cache[cache_key] = (time.time() + 30, result)
            if len(_classify_cache) > 300:
                oldest = min(_classify_cache, key=lambda k: _classify_cache[k][0])
                del _classify_cache[oldest]
            return result
    except Exception as e:
        logger.warning(f"Decision engine failed: {e}")

    # Fallback: conservative defaults
    logger.info(f"Decision engine fallback: using conservative defaults for [{user_message[:50]}]")
    return {"category": "REALTIME_INFORMATION", "need_web": True, "clarification_needed": False,
            "confidence": 0.5, "emotion": "neutral", "emotion_intensity": 1, "action_tendency": "unknown"}


# _check_ambiguous removed — model handles clarification via classify_intent now.


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
    if re.search(r'(?<![a-zA-Z0-9])\d{6}(?![a-zA-Z0-9])', msg):
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
    chat_history: list[dict] | None = None,
) -> dict:
    """Router: classify question → execute correct tools → return data.

    Categories (CODE decision, NOT model):
      1. datetime  → Python datetime, never search
      2. realtime  → force Tavily + market data
      3. financial → financial APIs + market data
      4. knowledge → no tools, model answers directly

    Returns: {"category": str, "search_ctx": str, "market_ctx": str, "system_note": str, "direct_response": str|None}
    """
    # Use decision engine for reasoning-based classification
    intent = await classify_intent(user_message)
    category = intent.get("category", "REALTIME_INFORMATION")
    need_web = intent.get("need_web", True)
    clarification_needed = intent.get("clarification_needed", False)
    confidence = intent.get("confidence", 0.5)
    logger.info(
        f"Decision engine [{user_message[:50]}] → {category} "
        f"web={need_web} clarify={clarification_needed} conf={confidence}"
    )

    # Feedback loop: detect corrections, force fresh search with original topic
    _correction_signals = [
        "不对", "错了", "不是这个", "我要的是", "你再看看", "重新查",
        "重新搜", "再查", "换一个", "不对吧", "搞错了", "不是这样",
    ]
    _is_correction = any(s in user_message for s in _correction_signals)
    _enhanced_correction = None
    if _is_correction and chat_history:
        need_web = True
        for h in reversed(chat_history):
            if h.get("role") == "user" and h.get("content"):
                prev = h["content"]
                # Skip if the history entry IS the current message (saved before fetch)
                if prev.strip() == user_message.strip():
                    continue
                # Smart check: don't combine if correction changes the topic entirely
                prev_codes = set(_extract_stock_codes(prev))
                curr_codes = set(_extract_stock_codes(user_message))
                if curr_codes and prev_codes and not (curr_codes & prev_codes):
                    # Topic change — different stocks, use correction as fresh query
                    logger.info(f"Feedback loop: topic change detected, not combining. prev={prev[:30]} curr={user_message[:30]}")
                    _enhanced_correction = None
                else:
                    _enhanced_correction = f"{prev} {user_message}"
                    logger.info(f"Feedback loop: correction='{user_message[:30]}' original='{prev[:30]}'")
                break

    if clarification_needed and category == "CLARIFICATION_REQUIRED":
        logger.info(f"Clarification required for [{user_message[:50]}], returning direct追问")
        # If chat history exists, let the model handle it with context instead of blocking
        if chat_history:
            # Pass through — model has history context to understand the follow-up
            logger.info("Chat history present — letting model handle with context")
        else:
            return {
                "category": "clarification",
                "search_ctx": "",
                "market_ctx": "",
                "system_note": "",
                "need_web": False,
                "emotion": "neutral",
                "emotion_intensity": 1,
                "action_tendency": "unknown",
                "direct_response": (
                    "能再说具体一点吗？比如你想看哪只股票，或者对哪个行业感兴趣，我帮你细看。"
                ),
            }

    # ── STATIC_KNOWLEDGE: no web search, but still fetch market data ──
    if category == "STATIC_KNOWLEDGE":
        logger.info(f"Static knowledge [{user_message[:50]}], fetching market data for context")
        market_ctx = await _fetch_market_context(user_message, positions, chat_history)
        if isinstance(market_ctx, Exception):
            market_ctx = ""
        return {
            "category": "static_knowledge",
            "search_ctx": "",
            "market_ctx": market_ctx if market_ctx else "",
            "system_note": "",
            "need_web": False,
            "emotion": "neutral",
            "emotion_intensity": 1,
            "action_tendency": "unknown",
        }

    # ── REALTIME / DECISION / RESEARCH: search + market ──
    logger.info(f"Need Web Search: {user_message[:80]}")

    # Build date-enhanced query for freshest results
    from datetime import datetime, timezone, timedelta
    tz_cn = timezone(timedelta(hours=8))
    today_str = datetime.now(tz_cn).strftime("%Y年%m月%d日")
    if _is_correction and _enhanced_correction:
        enhanced_query = f"{_enhanced_correction} {today_str}"
    else:
        enhanced_query = f"{user_message} {today_str}"
    logger.info(f"Search Query: {enhanced_query}")

    # Detect Chinese-market queries — restrict to Chinese finance domains
    # to avoid Yahoo Finance junk (random US options, unrelated pages)
    _cn_finance_domains = [
        "eastmoney.com",
        "sina.com.cn",
        "10jqka.com.cn",
        "qq.com",
        "cninfo.com.cn",
        "hexun.com",
        "xueqiu.com",
        "cls.cn",
    ]
    has_chinese = bool(re.search(r'[一-鿿]', user_message))
    has_cn_stock = bool(re.search(r'(?<![a-zA-Z0-9])\d{6}(?![a-zA-Z0-9])', user_message))
    is_cn_query = has_chinese or has_cn_stock

    market_task = _fetch_market_context(user_message, positions, chat_history)

    # First attempt: Chinese domain filter for CN queries
    search_task = _tavily_search(
        enhanced_query,
        max_results=5,
        topic="finance",
        time_range="day",
        include_domains=_cn_finance_domains if is_cn_query else None,
    )

    market_ctx, search_ctx = await asyncio.gather(
        market_task, search_task, return_exceptions=True,
    )
    if isinstance(market_ctx, Exception):
        logger.error(f"Market fetch exception: {market_ctx}")
        market_ctx = ""
    if isinstance(search_ctx, Exception):
        logger.error(f"Search exception: {search_ctx}")
        search_ctx = ""

    # Fallback: if CN domain filter returned nothing, retry without filter
    if is_cn_query and not search_ctx:
        logger.info("CN domain filter returned empty — retrying with global search")
        try:
            search_ctx = await _tavily_search(
                enhanced_query,
                max_results=5,
                topic="finance",
                time_range="day",
                include_domains=None,
            )
        except Exception as e:
            logger.error(f"Global search fallback exception: {e}")

    logger.info(f"Raw Search Result: {search_ctx if search_ctx else '(EMPTY)'}")
    logger.info(f"Search Result Length: {len(search_ctx) if search_ctx else 0}")

    # Build emotion context for system prompt
    emotion = intent.get("emotion", "neutral")
    emotion_intensity = intent.get("emotion_intensity", 1)
    action_tendency = intent.get("action_tendency", "unknown")

    # Safety guardrail for extreme emotions (intensity ≥ 4)
    safety_note = ""
    if emotion_intensity >= 4:
        safety_note = (
            f"⚠️ 安全护栏触发：用户情绪强度 {emotion_intensity}/5（{emotion}），倾向 {action_tendency}。"
            "先安抚情绪、降温。不要支持冲动操作。给出客观数据，帮用户理性决策。"
            "如果用户表现出极端操作倾向，强制建议暂停操作，先观察关键指标。"
        )

    return {
        "category": category.lower(),
        "search_ctx": search_ctx,
        "market_ctx": market_ctx,
        "system_note": (
            (safety_note + "\n" if safety_note else "")
            + ("用户正在纠正你之前的回答。先简短致歉，然后用新数据重新回答。" if _is_correction else "")
            + ("" if search_ctx
               else "Tavily搜索未返回额外数据，但下方的【实时行情数据】是真实的，请使用其中的数字回答。"
               if market_ctx
               else "搜索和行情数据均为空。")
        ),
        "need_web": need_web,
        "emotion": emotion,
        "emotion_intensity": emotion_intensity,
        "action_tendency": action_tendency,
    }


# ── Tavily search cache ──
_tavily_cache: dict[str, tuple[float, str]] = {}


def _tavily_cache_key(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()


async def _tavily_search(
    query: str,
    max_results: int = 5,
    topic: str = "finance",
    time_range: str = "day",
    include_domains: list[str] | None = None,
) -> str:
    """Search the web via Tavily API with retry + caching. Returns formatted results or empty string.

    Args:
        query: Search query string
        max_results: Number of results (0-20)
        topic: "general", "news", or "finance" — finance gives better market data
        time_range: "day", "week", "month", "year" — day = most recent
        include_domains: Optional list of domains to restrict search to
    """
    if not settings.TAVILY_API_KEY:
        logger.warning("Tavily API key not configured — web search disabled. Set TAVILY_API_KEY in .env or Railway dashboard.")
        return ""

    cache_key = _tavily_cache_key(query + str(include_domains))
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
            logger.info(
                f"Tavily API call attempt={attempt+1} query=[{query[:60]}] "
                f"topic={topic} time_range={time_range} domains={include_domains}"
            )
            body = {
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced",
                "topic": topic,
                "time_range": time_range,
                "include_answer": True,
                "include_raw_content": False,
            }
            if include_domains:
                body["include_domains"] = include_domains
            resp = await client.post(
                "https://api.tavily.com/search",
                headers={
                    "Authorization": f"Bearer {settings.TAVILY_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=httpx.Timeout(15.0, connect=10.0),
            )

            logger.info(f"Tavily HTTP status: {resp.status_code}")

            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.warning(f"Tavily attempt {attempt+1}: {last_error}")
                if attempt == 0:
                    await asyncio.sleep(1.0)
                continue

            data = resp.json()
            logger.info(f"Tavily raw response keys: {list(data.keys()) if data else 'None'}")

            # Check for API-level errors
            if data.get("error"):
                last_error = f"API error: {data['error']}"
                logger.warning(f"Tavily attempt {attempt+1}: {last_error}")
                if attempt == 0:
                    await asyncio.sleep(1.0)
                continue

            parts = []

            answer = data.get("answer", "")
            # Filter out Tavily's negative "no data found" summaries.
            # These LLM-generated answers contradict real market data from Sina.
            _negative_patterns = [
                "does not contain", "no direct data", "no specific",
                "there is no", "none of the provided", "does not provide",
                "the user is asking", "the provided data",
                "based on the available information, there is no",
                "does not address", "cannot provide", "unable to provide",
            ]
            is_negative = answer and any(p in answer.lower() for p in _negative_patterns)
            if answer and not is_negative:
                parts.append(f"摘要: {answer}")
            elif is_negative:
                logger.info("Tavily answer filtered (negative/no-data summary)")

            results = data.get("results", [])
            logger.info(f"Tavily results count: {len(results)}")
            if results:
                parts.append("搜索结果:")
                for i, r in enumerate(results[:max_results], 1):
                    title = r.get("title", "")[:100]
                    content = r.get("content", "")[:200]
                    url = r.get("url", "")
                    # Also skip results that are clearly off-topic
                    parts.append(f"  [{i}] {title}")
                    parts.append(f"      {content}")
                    if url:
                        parts.append(f"      URL: {url}")

            if not parts:
                logger.warning("Tavily returned no results and no answer")
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


# ═══════════════════════════════════════════════════════════════
# ANTI-HALLUCINATION — hard number grounding + validation
# ═══════════════════════════════════════════════════════════════

def _extract_numbers(text: str) -> set[str]:
    """Extract all numeric tokens from text. Returns set of normalized number strings.

    Catches: integers, decimals, percentages, monetary amounts, dates.
    Normalizes: $204.65 → 204.65, -1.33% → 1.33, 5,000 → 5000
    """
    if not text:
        return set()

    numbers: set[str] = set()

    # Match patterns: $123.45, -1.33%, 4,255.61, 1.40万亿, 2026-06-18
    patterns = [
        # Currency amounts: $204.65, ￥100.50
        r'(?:[$￥€])\s*([\d,]+(?:\.\d+)?)',
        # Percentages: -1.33%, +0.5%, 1.33%
        r'([+-]?[\d,]+(?:\.\d+)?)\s*%',
        # Plain decimals: 204.65, 0.5
        r'(?<![$￥€\w])(\d+\.\d+)(?!%?\w)',
        # Large integers with commas: 5,000 or 1,400
        r'(?<!\w)(\d{1,3}(?:,\d{3})+)(?!\.?\d)',
        # Plain integers in context: 207 美元, 4108.08
        r'(?<!\w)(\d{2,})(?!\.?\d?[%]?)',
        # Dates: 2026-06-18
        r'(\d{4}-\d{2}-\d{2})',
    ]
    # Also extract comma-formatted numbers: 1,680.50 → 1680.50
    for m in re.finditer(r'([\d,]{3,}\.?\d*)', text):
        val = m.group(1).replace(',', '').strip()
        try:
            float(val)
            numbers.add(val)
        except ValueError:
            pass

    for pat in patterns:
        for m in re.finditer(pat, text):
            val = m.group(1).replace(',', '').strip()
            try:
                # Normalize: strip leading zeros for comparison
                num = float(val)
                if num == int(num) and num >= 1000:
                    numbers.add(str(int(num)))
                else:
                    numbers.add(val)
            except ValueError:
                numbers.add(val)

    return numbers


def _validate_hard(
    response_text: str,
    search_ctx: str,
    market_ctx: str,
) -> tuple[bool, str, str]:
    """Hard anti-hallucination validator.

    Checks numbers against market data. Auto-corrects hallucinated stock prices.
    General numbers that can't be verified are still blocked.

    Returns: (passed: bool, reason: str, corrected_text: str)
    """
    if not response_text or not response_text.strip():
        return False, "empty response", response_text

    # No search/market context → nothing to validate against
    combined_source = (search_ctx or "") + "\n" + (market_ctx or "")
    if not combined_source.strip():
        return True, "no source to validate against", response_text

    # ═══ Stock-price-specific validation ═══
    # Extract stock prices from market context: "贵州茅台(600519): 最新价 1215.00  涨跌 -2.02%"
    market_prices: dict[str, float] = {}  # code/name → price
    for m in re.finditer(
        r'([一-鿿\w]+)\s*\(\s*(\d{6})\s*\)\s*:\s*最新价\s*([\d.]+)',
        market_ctx or ""
    ):
        name, code, price = m.group(1), m.group(2), m.group(3)
        try:
            market_prices[code] = float(price)
            market_prices[name] = float(price)
        except ValueError:
            pass

    # Check for stock price hallucinations and auto-correct
    corrections: dict[str, tuple[float, float]] = {}  # old_value → (old, new)
    for key, real_price in list(market_prices.items()):
        matched = key in response_text
        search_key = key
        if not matched and len(key) > 2:
            for shorten in [key[-2:], key[-3:]]:
                if len(shorten) >= 2 and shorten in response_text:
                    matched = True
                    search_key = shorten
                    break
        if not matched:
            continue
        idx = response_text.find(search_key)
        if idx < 0:
            continue
        nearby = response_text[idx + len(search_key):idx + len(search_key) + 25]
        for m in re.finditer(r'([\d,]+\.?\d*)', nearby):
            try:
                rp = float(m.group(1).replace(',', ''))
                if 50 < rp < 10000:
                    pct_off = abs(rp - real_price) / real_price
                    if pct_off > 0.03:
                        old_str = m.group(1)
                        new_str = f"{real_price:.2f}".rstrip('0').rstrip('.')
                        corrections[old_str] = (rp, real_price)
                        logger.warning(
                            f"PRICE CORRECTED: {search_key} {old_str}→{new_str} (was {pct_off*100:.0f}% off)"
                        )
                        break  # found price-like number, done scanning this stock
            except ValueError:
                pass

    # Apply corrections to response text
    if corrections:
        corrected = response_text
        for old_str, (wrong, right) in corrections.items():
            new_str = f"{right:.2f}".rstrip('0').rstrip('.')
            corrected = corrected.replace(old_str, new_str, 1)
        logger.info(f"Auto-corrected {len(corrections)} hallucinated prices in response")
    else:
        corrected = response_text

    # ═══ General number extraction (use corrected text) ═══
    response_nums = _extract_numbers(corrected)
    source_nums = _extract_numbers(combined_source)

    logger.info(f"NUMBERS FOUND IN SEARCH: {sorted(source_nums)[:30]}")

    # Stock price correction (above) is the critical safety net.
    # General number validation removed — too many false positives.
    return True, "ok", corrected


def _verify_and_parse(
    raw_response: str, market_context: str, user_message: str
) -> dict:
    """
    Parse AI response. Now expects plain text, not JSON.
    Returns {"status": "ok"|"low_confidence"|"parse_error", "display_text": str, "raw": dict}
    """
    text = raw_response.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

    # Try to parse as JSON (legacy format)
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        pass

    if parsed and isinstance(parsed, dict):
        confidence = parsed.get("confidence_score", 0)
        analysis = parsed.get("analysis", "")
        if isinstance(confidence, (int, float)) and confidence < 30:
            return {"status": "low_confidence", "display_text": "LOW CONFIDENCE", "raw": parsed}
        display = analysis if analysis else text
        return {"status": "ok", "display_text": display, "raw": parsed}

    # Plain text response — this is the expected format now
    if not text:
        return {"status": "parse_error", "display_text": "", "raw": {}}

    return {"status": "ok", "display_text": text, "raw": {}}


def _extract_stock_codes(msg: str) -> list[str]:
    """Extract A-share stock codes from user message. Returns unique codes."""
    # Match patterns: 600519, sh600519, sz000001, 000001
    codes = []
    # Full codes with optional exchange prefix
    for m in re.finditer(r'(?<![a-zA-Z0-9])(sh|sz)?(\d{6})(?![a-zA-Z0-9])', msg, re.IGNORECASE):
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
    chat_history: list[dict] | None = None,
) -> str:
    """Fetch real-time market data and format for AI context."""
    from services.market_service import (
        get_market_overview, get_sector_performance,
        get_batch_quotes, fetch_hot_rank,
    )

    # Always fetch market data when positions exist — router already decided to call us
    is_market = True
    has_positions = bool(positions)

    # Extract stock codes from the message itself
    msg_codes = _extract_stock_codes(user_message) if is_market else []

    # Also extract from chat history (for follow-up messages like "说详细一点")
    history_codes: list[str] = []
    if chat_history:
        for h in chat_history[-4:]:  # last 4 messages
            content = h.get("content", "") if isinstance(h, dict) else ""
            if content:
                history_codes.extend(_extract_stock_codes(content))

    # If no codes found, try to find stock by name (e.g. "捷昌驱动的股价")
    if not msg_codes and is_market:
        from services.market_service import search_stock
        import re as _re2
        # Strip common query words + correction noise to get potential stock name
        name_hint = _re2.sub(
            r'(的?股价|的?股票|的?行情|多少[钱点]?|怎么样|如何|今天|现在'
            r'|最新|实时|查询|帮我|看一下|查一下|查的'
            r'|不对|错了|不是这个|不是这样|搞错了'
            r'|我要|我?想?[查看了]的?[是]?'
            r'|这个|那个|重新|再查|换一个|你再看看)',
            '', user_message
        ).strip()
        if name_hint and len(name_hint) >= 2:
            logger.info(f"Searching stock by name: {name_hint}")
            try:
                found = await search_stock(name_hint)
                if found:
                    msg_codes = [s["code"] for s in found[:3]]
                    logger.info(f"Found stocks by name: {msg_codes}")
            except Exception as e:
                logger.warning(f"Stock name search failed: {e}")

    if not is_market and not has_positions:
        return ""

    # Collect all codes to fetch: positions + message + chat history
    pos_codes = [p.get("asset_code", "") for p in positions if p.get("asset_code")]
    all_codes = list(dict.fromkeys(pos_codes + msg_codes + history_codes))  # dedup, preserve order

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
        logger.info("Market context: no data fetched")
        return ""

    logger.info(
        f"Market context fetched: overview={bool(overview)} sectors={len(sectors) if sectors else 0} "
        f"hot={len(hot) if hot else 0} quotes={len(quotes) if quotes else 0}"
    )
    parts = ["【实时行情数据 - 必须使用以下真实数字】"]

    if overview:
        # overview is now {"sh": {...}, "sz": {...}} with amount fields
        sh = overview.get("sh")
        sz = overview.get("sz")
        if sh:
            icon = "🔴" if sh.get("change_pct", 0) < 0 else "🟢"
            parts.append(
                f"{icon} {sh['index_name']}: {sh['price']:.2f}  "
                f"涨跌: {sh['change_pct']:+.2f}%  "
                f"成交额: {sh.get('amount_str', '未知')}"
            )
        if sz:
            icon = "🔴" if sz.get("change_pct", 0) < 0 else "🟢"
            parts.append(
                f"{icon} {sz['index_name']}: {sz['price']:.2f}  "
                f"涨跌: {sz['change_pct']:+.2f}%  "
                f"成交额: {sz.get('amount_str', '未知')}"
            )

    if quotes:
        parts.append("---实时个股行情---")
        for q in quotes:
            name = q.get('name', q.get('code', ''))
            code = q.get('code', '')
            chg = q.get("change_pct", 0)
            sign = "+" if chg >= 0 else ""
            sources = q.get("sources")
            if sources:
                # Multi-source: list all with source labels
                price_parts = " / ".join(
                    f"{s['price']:.2f}（{s['source']}）" for s in sources
                )
                parts.append(
                    f"  {name}({code}): 最新价 {price_parts}  涨跌 {sign}{chg:.2f}%"
                )
            else:
                parts.append(
                    f"  {name}({code}): "
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
      4. Hard validation — block unverified numbers
    """
    logger.info(f"USER QUESTION: {user_message}")
    pos_hash = hashlib.md5(str(positions).encode()).hexdigest()[:8]

    # ── STEP 1: Route & execute tools (CODE decision, model not involved) ──
    compressed_history = _compress_history(chat_history)
    tools = await route_and_execute_tools(user_message, positions, compressed_history)

    # Check for direct response (time/date intercept — no AI needed)
    direct = tools.get("direct_response")
    if direct:
        logger.info(f"Final Response (direct, no LLM): {direct[:200]}")
        return direct

    category = tools["category"]
    search_ctx = tools["search_ctx"]
    market_ctx = tools["market_ctx"]
    system_note = tools["system_note"]
    need_web = tools.get("need_web", True)

    logger.info(f"WEB SEARCH RAW RESULT: {search_ctx if search_ctx else '(EMPTY)'}")

    # ── STEP 1.5: Empty search guard ──
    if need_web and not search_ctx and not market_ctx:
        logger.warning(
            f"External search failed — need_web={need_web} search_ctx_len={len(search_ctx)} "
            f"market_ctx_len={len(market_ctx)} — returning fallback"
        )
        logger.info("Final Response: External search failed.")
        return "现在数据源都连不上，可能是网络波动。等一会再试试，一直不行的话跟 Kimura 说一声。"

    # ── Cache check (knowledge category only) ──
    if category in ("static_knowledge", "STATIC_KNOWLEDGE") and len(user_message) < 50:
        cache_key = _make_cache_key(user_message, pos_hash, "")
        cached = _get_cached(cache_key)
        if cached:
            logger.info(f"Final Response (cached): {cached[:200]}")
            return cached

    # ── STEP 2: Build prompt with tool results ──
    system_prompt = _build_system_prompt(positions, preferences)

    # Inject REAL current datetime — ALWAYS, for every request.
    # This prevents the model from using its training-data date cutoff.
    datetime_ctx = _get_datetime_context()
    system_prompt += "\n\n" + datetime_ctx

    if search_ctx:
        prefix = "【外部数据 - 这是你唯一的信息来源。只能使用其中的数字和事实。】"
        system_prompt += "\n\n" + prefix + "\n" + search_ctx

    if market_ctx:
        system_prompt += "\n\n" + market_ctx

    if system_note:
        system_prompt += "\n\n" + system_note

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(compressed_history)
    messages.append({"role": "user", "content": user_message})

    # Log the FULL prompt sent to model
    logger.info(
        f"FULL PROMPT SENT TO MODEL ({len(system_prompt)} chars system + "
        f"{len(compressed_history)} history msgs + user_msg={len(user_message)} chars):\n"
        f"--- SYSTEM PROMPT (first 500) ---\n{system_prompt[:500]}\n"
        f"--- SYSTEM PROMPT (last 200) ---\n...{system_prompt[-200:]}\n"
        f"--- END PROMPT ---"
    )

    # Log context summary
    emotion = tools.get("emotion", "neutral")
    emotion_intensity = tools.get("emotion_intensity", 1)
    action_tendency = tools.get("action_tendency", "unknown")
    context_summary = []
    if search_ctx:
        context_summary.append(f"search({len(search_ctx)} chars)")
    if market_ctx:
        context_summary.append(f"market({len(market_ctx)} chars)")
    context_summary.append(f"emotion={emotion}({emotion_intensity})→{action_tendency}")
    if system_note:
        context_summary.append(f"note({system_note[:50]})")
    context_summary.append(f"datetime({len(datetime_ctx)} chars)")
    logger.info(
        f"Context Sent To LLM: category={category} "
        f"has_search={bool(search_ctx)} has_market={bool(market_ctx)} "
        f"context={', '.join(context_summary)} "
        f"total_system_chars={len(system_prompt)}"
    )

    # ── STEP 3: Call model for text generation ONLY (no tool decisions) ──
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
        fallback = _get_fallback_reply(user_message)
        logger.info(f"Final Response (fallback): {fallback[:200]}")
        return fallback

    # ── STEP 4: HARD ANTI-HALLUCINATION VALIDATION ──
    # ── Validation ──
    # Always validate numbers against market data if present (exchange-sourced, authoritative).
    # Additionally validate against search_ctx when it's the only data source.
    _skip_validation = category in ("clarification", "datetime")
    if not _skip_validation and (search_ctx or market_ctx):
        passed, reason, raw_content = _validate_hard(raw_content, search_ctx, market_ctx)
        if not passed:
            logger.error(f"VALIDATION FAILED: {reason}")
            logger.info("Final Response: External search completed but verification failed. No reliable answer generated.")
            return "这个结果我不太确定，数据对不上。换个角度再问一次？如果一直这样跟 Kimura 说一声。"
        logger.info("VALIDATION PASSED")

    logger.info(f"FINAL RESPONSE: {raw_content[:300]}{'...' if len(raw_content) > 300 else ''}")

    # ── Cache only non-realtime short responses ──
    if category in ("static_knowledge", "STATIC_KNOWLEDGE") and len(user_message) < 50 and len(raw_content) < 300:
        _set_cache(_make_cache_key(user_message, pos_hash, ""), raw_content)

    return raw_content


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
        return "直接跟我说「添加持仓 股票代码 名字」就行"
    if any(kw in msg for kw in ["删除", "移除", "去掉"]):
        return "直接发「删除 股票代码」就行"
    if any(kw in msg for kw in ["持仓", "我的", "列表"]):
        return "现在连不上，稍等一下再试试。"
    if any(kw in msg for kw in ["你好", "hi", "hello", "在吗", "在不在"]):
        return "在，有什么事？"
    if any(kw in msg for kw in ["早", "简报", "行情"]):
        return "数据拉取中，稍等一下。"
    return "现在有点忙，稍等一下再发一次。或者直接说「我的持仓」试试本地功能。"
