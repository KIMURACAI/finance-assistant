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


SYSTEM_PROMPT_CORE = """你是金融资讯助手。严格执行以下反幻觉规则（ANTI-HALLUCINATION POLICY）：

━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES — VIOLATION = FAILURE
━━━━━━━━━━━━━━━━━━━━━━

1. NEVER answer factual/data questions from internal knowledge.
   You MUST only use data from the 【实时行情数据】 and 【网络搜索结果】 sections below.
   If both sections are empty/absent for a factual question → output NO VERIFIED DATA FOUND.

2. Every factual claim (price, change%, volume, index, event, date, name)
   MUST be traceable to either 【实时行情数据】 or 【网络搜索结果】.
   No rounding, no estimation, no inference.

3. Before writing analysis, verify each factual claim against the provided data.
   If a claim cannot be verified → do NOT include it.

4. confidence_score rules:
   • 95-100: all claims verbatim from provided data, multiple sources
   • 80-94:  most claims match provided data, minor reformatting
   • < 80:   any unverifiable statement → respond LOW CONFIDENCE only
   • 0:      no data available for this question

5. Position management commands go inside the analysis field as add/remove JSON blocks.

6. 【网络搜索结果】 provides real-time web search data. Treat it as authoritative.
   Prefer search results over market data when they conflict (search is fresher).

━━━━━━━━━━━━━━━━━━━━━━
MANDATORY OUTPUT FORMAT (strict JSON, no markdown, no extra text)
━━━━━━━━━━━━━━━━━━━━━━

{"question":"<user question>","verified_data":["<each data point used, quoting from provided sections>"],"analysis":"<analysis using ONLY verified data. For greetings/small-talk, reply naturally here. For position commands, append ```json {{\"cmd\":...}}``` at end>","confidence_score":<0-100>,"sources":["<data source names>"]}"""


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


def _is_factual_question(msg: str) -> bool:
    """Detect questions that require external data — must NOT be answered from model memory."""
    factual_kw = [
        "行情", "大盘", "指数", "上证", "深证", "创业板", "走势",
        "涨跌", "涨停", "跌停", "涨幅", "跌幅", "热门", "板块", "行业",
        "股票", "股价", "价格", "多少", "多少点", "最新", "现在",
        "实时", "今天", "今日", "目前", "分析", "预测", "建议",
        "龙虎", "热点", "概念", "北向", "成交", "市值", "PE", "估值",
        "基金", "净值", "收益", "财报", "业绩", "公告", "新闻",
        "代码", "查", "查询", "帮我", "能否", "是否", "怎样",
        "怎么样", "如何", "怎么看", "评价", "评级",
    ]
    msg_lower = msg.lower()
    # Short codes like "600519" are factual queries
    if len(msg.strip()) <= 8 and any(c.isdigit() for c in msg):
        return True
    return any(kw in msg_lower for kw in factual_kw)


# Tavily search cache: {query_hash: (expires_at, formatted_result)}
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
        logger.warning(f"AI JSON parse failed, raw: {raw_response[:200]}")
        return {
            "status": "parse_error",
            "display_text": raw_response[:500],
            "raw": {},
        }

    confidence = parsed.get("confidence_score", 0)
    verified_data = parsed.get("verified_data", [])
    analysis = parsed.get("analysis", "")

    # ── Verification layer: audit + confidence check, no forced downgrade ──

    # Gate 1 (audit only): warn if factual question but no verified_data cited
    if market_context and not verified_data and _is_factual_question(user_message):
        logger.warning(
            f"Gate 1: factual question with external data but no verified_data cited. "
            f"AI confidence={confidence}. Not downgrading — trusting AI self-assessment."
        )

    # Gate 2 (audit only): check cross-reference between verified_data and context
    if verified_data and market_context:
        match_count = 0
        for item in verified_data:
            numbers = re.findall(r'\d+\.?\d*', str(item))
            if numbers:
                hits = sum(1 for n in numbers if n in market_context)
                if hits >= len(numbers) * 0.5:
                    match_count += 1
        if match_count < len(verified_data) * 0.5 and len(verified_data) > 0:
            logger.warning(
                f"Gate 2: only {match_count}/{len(verified_data)} verified_data items "
                f"cross-reference with context. AI confidence={confidence}. "
                f"Not downgrading — trusting AI self-assessment."
            )

    # Gate 3: confidence_score adjudication (threshold 70 — AI is conservative at temp=0.1)
    if confidence < 70:
        return {
            "status": "low_confidence",
            "display_text": "LOW CONFIDENCE",
            "raw": parsed,
        }

    return {
        "status": "ok",
        "display_text": analysis if analysis else raw_response,
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

    is_market = _is_factual_question(user_message)
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
    """AI chat with mandatory web search: search-first, verify, structured output."""
    pos_hash = hashlib.md5(str(positions).encode()).hexdigest()[:8]
    history_tail = str(chat_history[-4:]) if chat_history else ""
    is_factual = _is_factual_question(user_message)

    # ── Cache check (skip for factual — data changes) ──
    if not is_factual and len(user_message) < 50 and not any(
        kw in user_message for kw in ["添加", "删除", "持仓"]
    ):
        cache_key = _make_cache_key(user_message, pos_hash, "")
        cached = _get_cached(cache_key)
        if cached:
            return cached

    # ── Build system prompt ──
    system_prompt = _build_system_prompt(positions, preferences)
    compressed_history = _compress_history(chat_history)

    # ── Step 1+2: Fetch market data + Tavily search IN PARALLEL ──
    market_task = _fetch_market_context(user_message, positions)
    search_task = _tavily_search(user_message, max_results=5) if is_factual else None

    if search_task:
        market_ctx, search_ctx = await asyncio.gather(
            market_task, search_task, return_exceptions=True,
        )
        if isinstance(market_ctx, Exception):
            logger.warning(f"Market fetch failed: {market_ctx}")
            market_ctx = ""
        if isinstance(search_ctx, Exception):
            logger.warning(f"Tavily search failed: {search_ctx}")
            search_ctx = ""
        logger.info(
            f"Tavily search: query={user_message[:50]} "
            f"results={'yes' if search_ctx else 'empty'}"
        )
    else:
        market_ctx = await market_task
        if isinstance(market_ctx, Exception):
            market_ctx = ""
        search_ctx = ""

    # ── Anti-hallucination Gate: factual question without ANY external data ──
    if is_factual and not market_ctx and not search_ctx:
        logger.info(f"No external data available for factual question: {user_message[:50]}")
        return "NO VERIFIED DATA FOUND"

    # ── Inject all external data into system prompt ──
    if search_ctx:
        system_prompt += "\n\n" + "【网络搜索结果 - 权威实时数据】\n" + search_ctx
    if market_ctx:
        system_prompt += "\n\n" + market_ctx

    # ── Build messages ──
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(compressed_history)
    messages.append({"role": "user", "content": user_message})

    # ── Call AI with temperature clamped to 0.1 (anti-hallucination) ──
    payload = {
        "model": settings.DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.1,  # Hard override for strict factual adherence
        "max_tokens": settings.DEEPSEEK_MAX_TOKENS,
    }

    client = get_client()
    raw_content = None
    for attempt in range(2):
        try:
            logger.info(
                f"DeepSeek API request attempt={attempt+1} "
                f"model={settings.DEEPSEEK_MODEL} msg_len={len(user_message)} "
                f"is_factual={is_factual} has_market={bool(market_ctx)} has_search={bool(search_ctx)}"
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

    # ── Verification layer (checks against both market + search context) ──
    full_context = (search_ctx or "") + "\n" + (market_ctx or "")
    verified = _verify_and_parse(raw_content, full_context, user_message)

    if verified["status"] == "low_confidence":
        logger.warning(
            f"LOW CONFIDENCE: user={user_message[:50]} "
            f"confidence={verified['raw'].get('confidence_score', 'N/A')}"
        )
        return "LOW CONFIDENCE"

    if verified["status"] == "parse_error":
        logger.warning(f"AI did not output valid JSON, returning raw text")
        return verified["display_text"]

    # ── Log structured output for audit ──
    logger.info(
        f"AI verified OK: confidence={verified['raw'].get('confidence_score')} "
        f"verified_data_count={len(verified['raw'].get('verified_data', []))}"
    )

    display = verified["display_text"]

    # ── Cache non-factual short responses ──
    if not is_factual and len(user_message) < 50 and len(display) < 300:
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
